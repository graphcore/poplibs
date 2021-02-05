// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popops/GatherStatistics.hpp"
#include "PerformanceEstimation.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

namespace {

struct HistogramOptions {
  bool useFloatArithmetic = false;
};

HistogramOptions parseOptionFlags(const OptionFlags &options) {
  HistogramOptions histogramOpts;
  const poplibs::OptionSpec histogramSpec{
      {"useFloatArithmetic", poplibs::OptionHandler::createWithBool(
                                 histogramOpts.useFloatArithmetic)},
  };
  for (const auto &entry : options) {
    histogramSpec.parse(entry.first, entry.second);
  }
  return histogramOpts;
}

enum class VertexType { SUPERVISOR_BY_DATA, SUPERVISOR_BY_LIMIT, WORKER_2D };

VertexType chooseVertexType(const std::vector<std::vector<Interval>> &intervals,
                            unsigned maxSupervisorElemsByLimit,
                            unsigned maxSupervisorElemsByData, bool isAbsolute,
                            bool isHalf, unsigned numWorkers,
                            unsigned numLimits, unsigned vectorWidth) {
  if (intervals.size() != 1 || intervals[0].size() > 1) {
    return VertexType::WORKER_2D;
  }
  const auto elements = std::accumulate(
      intervals[0].begin(), intervals[0].end(), 0u,
      [](std::size_t total, const Interval &i) { return total + i.size(); });

  // We can use one of the supervisor vertex types, which each have different
  // advantages:
  // SUPERVISOR_BY_LIMIT divides work using numLimits which can leave idle
  //                     workers, but there is no need to have 1 worker combine
  //                     partial results, so the vertex is simpler and faster.
  // SUPERVISOR_BY_DATA  divides work by data but requires each piece to be
  //                     recombined by one worker.  This makes it more complex.

  auto limitsCycles = histogramSupervisorByLimitEstimate(
      elements, numLimits + 1, isAbsolute, isHalf, numWorkers, vectorWidth);
  auto dataCycles = histogramSupervisorByDataEstimate(
      elements, numLimits + 1, isAbsolute, isHalf, numWorkers, vectorWidth);

  if (elements < maxSupervisorElemsByData && dataCycles < limitsCycles) {
    return VertexType::SUPERVISOR_BY_DATA;
  }
  if (elements < maxSupervisorElemsByLimit && dataCycles >= limitsCycles) {
    return VertexType::SUPERVISOR_BY_LIMIT;
  }

  return VertexType::WORKER_2D;
}

poplar::Tensor histogramImpl(poplar::Graph &graph, const poplar::Tensor &input,
                             const poplar::Tensor &levels, bool absoluteOfInput,
                             poplar::program::Sequence &prog,
                             const DebugNameAndId &dnai) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto inType = input.elementType();
  const auto vectorWidth = target.getVectorWidth(inType);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto flattenedInput = input.flatten();

  const auto cs = graph.addComputeSet({dnai, "Histogram"});

  // As work is split by "limits" not data size, the rpt count is a severe
  // limitation on the input size when rptMax is small.  So in that case the
  // vertex overcomes the limitation.  In other cases we must split work by
  // producing more vertices
  const auto rptMax = target.getRptCountMax();
  const auto maxSupervisorElemsByLimit =
      (rptMax < 0xffff) ? UINT32_MAX : rptMax * vectorWidth;
  const auto maxSupervisorElemsByData = rptMax * numWorkers * vectorWidth;

  const auto codeletName2D =
      templateVertex("popops::Histogram2D", inType, absoluteOfInput);
  const auto max2DVectorElems =
      std::min<std::size_t>(graph.getMaxFieldDim(codeletName2D, "data", 1),
                            target.getRptCountMax() * vectorWidth);

  // Gather a vector of results from each vertex created
  std::vector<Tensor> results;
  const auto mapping = graph.getTileMapping(flattenedInput);
  for (unsigned tile = 0; tile < numTiles; tile++) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(flattenedInput, mapping[tile]);

    if (tileContiguousRegions.size() == 0) {
      // No data on this tile
      continue;
    }
    const auto vertexType = chooseVertexType(
        tileContiguousRegions, maxSupervisorElemsByLimit,
        maxSupervisorElemsByData, absoluteOfInput, inType == HALF, numWorkers,
        levels.numElements(), vectorWidth);
    if (vertexType == VertexType::SUPERVISOR_BY_LIMIT ||
        vertexType == VertexType::SUPERVISOR_BY_DATA) {
      // 1 region of suitable size to use a supervisor vertex
      const auto byLimit = (vertexType == VertexType::SUPERVISOR_BY_LIMIT);
      const auto codeletNameSupervisor = templateVertex(
          "popops::HistogramSupervisor", inType, absoluteOfInput, byLimit);
      auto v = graph.addVertex(cs, codeletNameSupervisor);
      graph.setTileMapping(v, tile);

      auto resultSize = levels.numElements() + 1;
      auto histogram = graph.addVariable(
          FLOAT, {byLimit ? resultSize : numWorkers * resultSize}, {dnai});
      results.push_back(histogram.slice(0, resultSize));
      graph.setTileMapping(histogram, tile);

      graph.setInitialValue(v["histogramCount"], results.back().numElements());
      graph.connect(v["limits"], levels);
      graph.connect(v["histogram"], histogram);
      graph.connect(v["data"],
                    flattenedInput.slice(tileContiguousRegions[0][0]));
    } else {
      // > 1 region or not suitable size so use 2D worker vertices
      auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, vectorWidth, 2 * vectorWidth,
          UINT32_MAX, max2DVectorElems);
      for (const auto &regions : vertexRegions) {
        auto v = graph.addVertex(cs, codeletName2D);
        graph.setTileMapping(v, tile);
        results.push_back(
            graph.addVariable(FLOAT, {levels.numElements() + 1}, {dnai}));
        graph.setTileMapping(results.back(), tile);

        graph.setInitialValue(v["histogramCount"],
                              results.back().numElements());
        graph.connect(v["limits"], levels);
        graph.connect(v["histogram"], results.back());
        graph.connect(v["data"], flattenedInput.slices(regions));
      }
    }
  }
  prog.add(Execute(cs, {dnai}));

  return concat(results).reshape({results.size(), results[0].numElements()});
}

} // anonymous namespace

// The above function makes histograms of on-tile data using float format for
// speed. Float can represent exact integers in the range 0-16777216 (Note 1)
// but after that not every value can be represented.  This is not a problem
// on tile, as tile data < 16M * (size of half) = 32MBytes. But the data
// spread over the whole IPU can be more than 32MBytes.
// We need an accurate integer count in our histogram so we may need to use
// int data (Note 2), but at the cost of slower processing so cast only as
// necessary.
//
// Note 1: float can represent integers between +/-16777216 exactly but that
// won't totally solve the problem.  At best we could reduce with a start
// value of -16777216 and remove that bias after casting to int, doubling our
// range before int reduction is required.
//
// Note 2: int, not unsigned int as the reduction library doesn't support
// unsigned ints.  Additionally it doesn't have optimised vertices
// for int (whereas it does for float). int supports
// 2Giga entries (4GBytes of half data)
//
// The option is provided to override the above and do all calculation in
// float arithmetic and produce a float result.
// The functions below do reduction with the appropriate casting and
// data type to combine the vertex results.
//
// In short, if all elements are to be counted into a single histogram entry
// and we want an answer that is exact, we can only use float data for our
// arithmetic if there are <= `maxElementsForFloatReduction` values at any
// point in the calculation (So on 1 tile, float is OK).  Otherwise
// we must use unsigned or int values.
constexpr unsigned maxElementsForFloatReduction = 16777216u;

poplar::Tensor histogram(poplar::Graph &graph, const poplar::Tensor &input,
                         const poplar::Tensor &levels, bool absoluteOfInput,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext,
                         const poplar::OptionFlags &options) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(input, levels, absoluteOfInput, options));

  const auto opts = parseOptionFlags(options);
  auto histogramResult =
      histogramImpl(graph, input, levels, absoluteOfInput, prog, {di});

  if (opts.useFloatArithmetic) {
    // Override all concerns over inaccurate integer representation as float,
    // but tolerate possible inaccurate results
    auto output = reduce(graph, histogramResult, FLOAT, {0},
                         popops::Operation::ADD, prog, {di});
    di.addOutput(output);
    return output;
  }
  // See the above explanation on numeric limits for exact integer
  // representation using float vs unsigned
  if (input.numElements() > maxElementsForFloatReduction) {
    // Reduce as INT, cast to unsigned to return
    histogramResult = cast(graph, histogramResult, INT, prog, {di});
  }
  // Reduce, cast to unsigned to return
  auto output = reduce(graph, histogramResult, histogramResult.elementType(),
                       {0}, popops::Operation::ADD, prog, {di});
  // When casting int to unsigned this appears to generate nothing
  auto output_ = cast(graph, output, UNSIGNED_INT, prog, {di});
  di.addOutput(output_);
  return output_;
}

void histogram(poplar::Graph &graph, const poplar::Tensor &input,
               poplar::Tensor &output, bool updateOutput,
               const poplar::Tensor &levels, bool absoluteOfInput,
               poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(input, output, levels, updateOutput, absoluteOfInput));

  const auto useFloatArithmetic = (output.elementType() == FLOAT);
  auto histogramResult =
      histogramImpl(graph, input, levels, absoluteOfInput, prog, {di});

  if (useFloatArithmetic) {
    // Override all concerns over inaccurate integer representation as float,
    // but tolerate possible inaccurate results
    reduceWithOutput(graph, histogramResult, output, {0},
                     {popops::Operation::ADD, updateOutput}, prog, {di});
    return;
  }
  // See the above explanation on numeric limits for exact integer
  // representation using float vs unsigned
  if (input.numElements() > maxElementsForFloatReduction) {
    // Reduce as INT, cast to unsigned to return
    output = cast(graph, output, INT, prog, {di});
    histogramResult = cast(graph, histogramResult, INT, prog, {di});
    reduceWithOutput(graph, histogramResult, output, {0},
                     {popops::Operation::ADD, updateOutput}, prog, {di});
    output = cast(graph, output, UNSIGNED_INT, prog, {di});
  } else {
    // Reduce as float
    auto result = reduce(graph, histogramResult, FLOAT, {0},
                         popops::Operation::ADD, prog, {di});
    if (updateOutput) {
      // Cast to unsigned and add to the result to return
      result = cast(graph, result, UNSIGNED_INT, prog, {di});
      addInPlace(graph, output, result, prog, {di});
    } else {
      // Cast to unsigned to return
      output = cast(graph, result, UNSIGNED_INT, prog, {di});
    }
  }
}

} // namespace popops
