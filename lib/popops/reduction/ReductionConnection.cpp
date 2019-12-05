#include "ReductionConnection.hpp"

#include <boost/optional.hpp>
#include <boost/range/algorithm/transform.hpp>

#include "poplibs_support/logging.hpp"
#include <poplibs_support/Compiler.hpp>
#include <popops/Reduce.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include "CycleEstimationFunctions.hpp"
#include "ReductionVertex.hpp"
#include <algorithm>
#include <cassert>

using namespace poplibs_support;

namespace popops {

namespace {

// Divide a by b, rounding up.
template <typename T> T udiv(T a, T b) { return ((a + b) - 1) / b; };

// Return the approximate number of operations per cycle for the given
// type and operation. This doesn't account for type conversion, scale or
// update.
double opsPerCycle(unsigned outputSize, const poplar::Target &target,
                   poplar::Type type, popops::Operation operation) {
  if (outputSize == 0)
    return 1;

  // Very roughly, the number of elements we can process per cycle is as
  // follows.
  //
  // In theory we can do better in some cases with complex optimisations.
  //
  // f16 add, square-add, abs-add:
  //  outputSize     == 1: 8     rpt {ld128step; f16v8acc}, f16v4add, f16v2add,
  //                             f16v2add
  //  outputSize     == 2: 8     rpt {ld128step; f16v8acc}, f16v4add, f16v2add
  //  outputSize     == 4: 8     rpt {ld128step; f16v8acc}, f16v4add
  //  outputSize     == 8: 8     rpt {ld128step; f16v8acc}
  //  outputSize % 8 == 0: 8     rpt rpt {ld128step; f16v8acc}
  //  outputSize % 4 == 0: 4     rpt rpt {ld64step; f16v4acc}
  //  outputSize % 2 == 0: 2     rpt rpt {ld32step; f16v2add}
  //            otherwise: 1     rpt rpt {lds16step; f16v2add}
  //
  // f16 other ops:
  //  outputSize     == 1: 4
  //  outputSize     == 2: 4
  //  outputSize     == 4: 4     rpt {ld64step; f16v4mul}
  //  outputSize % 4 == 0: 4
  //  outputSize % 2 == 0: 2
  //            otherwise: 1
  //
  // f32 add, square-add, abs-add:
  //  outputSize     == 1: 4
  //  outputSize     == 2: 4
  //  outputSize     == 4: 4
  //  outputSize % 4 == 0: 4
  //  outputSize % 2 == 0: 2
  //            otherwise: 4/3
  //
  // f32 other ops:
  //  outputSize     == 1: 2     rpt {ld64step; f32v2mul}, f32mul
  //  outputSize     == 2: 2     rpt {ld64step; f32v2mul}
  //  outputSize % 2 == 0: 2
  //            otherwise: 4/3 (we have two do two ld32's instead of one ld64
  //            every other cycle).

  // However this only needs to be rough, so I am ignoring the clever
  // optimisations which means this is simple.
  unsigned vectorWidth = target.getVectorWidth(type);

  // Add, square-add and abs-add can be done twice as fast for floats.
  if ((operation == popops::Operation::ADD ||
       operation == popops::Operation::SQUARE_ADD) // Or ABS_ADD
      && (type == poplar::HALF || type == poplar::FLOAT))
    vectorWidth *= 2;

  // If the output size is 1/2, 1/4 or 1/8th of the vector width we
  // can just treat it as if it were the vector width with a final tiny
  // reduction on the output value.
  for (unsigned w = vectorWidth; w > 0; w /= 2) {
    if (outputSize == w)
      return vectorWidth;
  }

  // Otherwise we can generally process as many numbers as we can load
  // in one cycle, when taking alignment requirements into account.
  for (unsigned w = vectorWidth; w > 0; w /= 2) {
    if (outputSize % w == 0)
      return w;
  }

  // We should never reach here but the answer is probably 1.
  return 1;
}

struct ReductionAssignment {
  // The worker(s) that this reduction has been assigned to. If it is only
  // assigned to one worker it can be done in one stage. If it is assigned to
  // more than one it is done in two stages.
  std::vector<unsigned> workers;
};

std::uint64_t approximateOverheadCyclesForReduction(const unsigned splits) {
  // Simple overhead based on specialisation 3.
  const unsigned vertexOverhead = 30;
  return splits == 1 ? vertexOverhead : 2 * vertexOverhead;
}

std::uint64_t approximateCyclesForReduction(const poplar::Target &target,
                                            popops::Operation operation,
                                            const RegionReduction &reduction) {
  unsigned totalPartialElements = 0;
  for (const auto &p : reduction.partials)
    totalPartialElements += p.numElements();

  double cyclesPerOp =
      1.0 / opsPerCycle(reduction.output.numElements(), target,
                        reduction.output.elementType(), operation);

  // The ceil() is because otherwise some cycle counts come out as 0.5 which is
  // obviously impossible in practice, and it gets rounded to 0 which is even
  // more impossible. This is only rough and there's going to be some overhead
  // which I haven't estimated here anyway.
  return std::ceil(totalPartialElements * cyclesPerOp);
}

// Distribute a set of region reductions between workers on a tile. This never
// splits the individual reductions.
//
// The return value is the worker each reduction is assigned to.
std::vector<unsigned> distributeReductionsBetweenWorkers(
    const poplar::Target &target, popops::Operation operation,
    const std::vector<RegionReduction> &reductions) {

  // If there are more or equal output regions than numWorkers we
  // distribute them as evenly as possible based on the number of partials
  // so each vertex has a similar number of partials.

  std::vector<unsigned> assignments;

  if (reductions.empty())
    return assignments;

  assignments.resize(reductions.size());

  // Calculate (very roughly) the number of cycles needed to calculate
  // each reduction. This doesn't have to be exact.
  //
  // The pair is {reductionNumber, cycleCount}.
  std::vector<std::pair<unsigned, std::uint64_t>> reductionCycles;
  reductionCycles.reserve(reductions.size());

  for (const auto &r : reductions) {
    reductionCycles.emplace_back(
        reductionCycles.size(),
        approximateCyclesForReduction(target, operation, r));
  }

  // Optimisation: There is definitely scope for further optimisation here.
  // Distributing work evenly is almost NP-complete, but there are apparently
  // efficient algorithms. See https://en.wikipedia.org/wiki/Partition_problem
  // and especially Multi-Way Number Partitioning by Richard Korf:
  // http://www.ijcai.org/Proceedings/09/Papers/096.pdf

  // For now, we just sort them from most cycles to least, and greedily
  // assign them to the vertex with the fewest cycles.

  std::sort(
      reductionCycles.begin(), reductionCycles.end(),
      [](std::pair<unsigned, std::uint64_t> a,
         std::pair<unsigned, std::uint64_t> b) { return a.second < b.second; });

  std::vector<std::uint64_t> cyclesPerWorker;
  cyclesPerWorker.reserve(target.getNumWorkerContexts());

  for (auto c : reductionCycles) {
    auto reductionIndex = c.first;
    auto cycles = c.second;

    if (cyclesPerWorker.size() < target.getNumWorkerContexts()) {

      // If we still have spare worker contexts, add another worker.
      assignments[reductionIndex] = cyclesPerWorker.size();
      cyclesPerWorker.push_back(cycles);

    } else {

      // Otherwise add it to the one with the fewest worker cycles so far.
      auto minIt =
          std::min_element(cyclesPerWorker.begin(), cyclesPerWorker.end());
      assignments[reductionIndex] = minIt - cyclesPerWorker.begin();
      *minIt += cycles;
    }
  }
  logging::trace("cycles per worker size {}", cyclesPerWorker.size());

  return assignments;
}

// Get the number of rows, or the reduction factor of a region reduction.
std::size_t reductionFactor(const RegionReduction &reduction) {
  auto outputSize = reduction.output.numElements();

  // Work out the number of rows (reduction ratio) of this reduction.
  std::size_t totalPartialElements = 0;
  for (const auto &partial : reduction.partials)
    totalPartialElements += partial.numElements();
  assert(totalPartialElements % outputSize == 0);

  // Total number of rows for this reduction.
  return totalPartialElements / outputSize;
}

// Given a reduction with a reduction factor of `row` that takes roughly
// `cycles` cycles, what is the highest number of pieces we should split it
// into. We want each piece to have at least 2 rows, and at least
// `minCycles` cycles.
unsigned getMaxSplit(std::size_t rows, std::uint64_t cycles) {

  // Don't split it up so that any piece takes fewer cycles than this.
  // This number was found empirically by trying different numbers on
  // reductions from resnet until it gave the fewest total cycles.
  const std::uint64_t minCycles = 4;

  // Or there are fewer than 2 rows in each piece.
  return std::min(static_cast<std::uint64_t>(rows / 2), cycles / minCycles);
}

// Split the RegionReductions into smaller chunks. This can be for two reasons:
//  - to make more work to utilise all of the workers available
//  - to make the vertex state values fit into their respective types.
// The return value is the number of pieces each reduction is split up into
// (minimum 1) for each reduction.
std::vector<unsigned> splitTwoStageReductionsBetweenWorkers(
    const poplar::Target &target, popops::Operation operation,
    const std::vector<RegionReduction> &reductions,
    const std::size_t vectorListMaxSize) {
  // initialise the number of splits needed for each reduction by how many times
  // it would be needed for it to fit into a DeltaN VectorList.
  std::vector<unsigned> minimumSplits;
  minimumSplits.reserve(reductions.size());

  const auto out = std::back_inserter(minimumSplits);
  boost::transform(reductions, out, [&](const RegionReduction &reduction) {
    return udiv(reduction.partials.size(), vectorListMaxSize);
  });

  std::vector<std::uint64_t> approxCycleCounts(reductions.size());
  for (std::size_t i = 0; i < reductions.size(); ++i) {
    approxCycleCounts[i] =
        approximateOverheadCyclesForReduction(1) +
        approximateCyclesForReduction(target, operation, reductions[i]);
  }

  std::vector<unsigned> splits(minimumSplits.begin(), minimumSplits.end());

  // First work out the maximum number of splits for each worker. It never
  // makes sense to split to less than 2 rows for each piece, and in some
  // cases the limit is more if there aren't many output values.

  std::vector<unsigned> maxSplit(reductions.size(), 1);
  for (unsigned i = 0; i < reductions.size(); ++i) {
    maxSplit[i] =
        getMaxSplit(reductionFactor(reductions[i]), approxCycleCounts[i]);
  }

  unsigned freeWorkers = target.getNumWorkerContexts() - reductions.size();
  // As a baseline to check if an improvement in max cycles per reduction
  // is made by splitting work, record the max cycles prior to splitting.
  const auto maxOneStageCycles =
      *std::max_element(approxCycleCounts.begin(), approxCycleCounts.end());

  while (freeWorkers > 0) {
    boost::optional<unsigned> toSplit;
    // Consider splitting the slowest reduction
    const auto highestCyclesIter =
        std::max_element(approxCycleCounts.begin(), approxCycleCounts.end());
    const auto i = std::distance(approxCycleCounts.begin(), highestCyclesIter);

    // Stop if it doesn't want to be split any more, as we can't improve the
    // maximum
    if (splits[i] + 1 >= maxSplit[i])
      break;

    // Work out the rough number of cycles if it would be split more, accounting
    // for the overhead of needing another stage.
    // TODO: T12961 Use the cycle estimation functions to account for cycles
    // accurately throughout this module, given the decision on which reduction
    // vertex is actually going to be called.
    auto cyclesAfterSplit =
        approximateOverheadCyclesForReduction(splits[i] + 1) +
        approximateCyclesForReduction(target, operation, reductions[i]) /
            (splits[i] + 1);
    if (cyclesAfterSplit < approxCycleCounts[i]) {
      approxCycleCounts[i] = cyclesAfterSplit;
      toSplit = i;
    }

    // Check if we don't want to split any more.
    if (!toSplit)
      break;

    ++splits[toSplit.get()];
    --freeWorkers;
  }
  auto highestCyclesAfterSplits =
      *std::max_element(approxCycleCounts.begin(), approxCycleCounts.end());

  // Is the complexity of splitting to use all workers actually improving
  // the overall maximum?
  // If not then return the minimum possible split, avoiding extra complexity
  // and vertex state for no good reason
  return highestCyclesAfterSplits == maxOneStageCycles ? minimumSplits : splits;
}

static void createPartialsAreInputSizeVertex(
    const std::vector<poplar::Tensor> &partials,
    const std::vector<poplar::Tensor> &outputs, poplar::Graph &graph,
    const unsigned grainSize, const unsigned tile, const ReduceParams &params,
    const poplar::ComputeSet &cs, const bool targetIsCpu,
    const poplar::Type &partialType, const poplar::Type &outputType) {
  auto partialsSize = partials[0].shape()[0] / outputs[0].shape()[0];
  assert(partialsSize > 0);
  const auto outCount = outputs[0].shape()[0] / grainSize;

  const auto name =
      getPartialsEqualSizeReductionVertexName(params, partialType, outputType);
  logging::trace("Creating vertex for reduction: {}, tile {}, "
                 "compute set {}, "
                 "numOutputs {}, partialsSize {}, numPartials {}",
                 name, tile, cs.getId(), outCount, partialsSize,
                 partials.size());
  const auto vertex = graph.addVertex(cs, name);
  graph.setTileMapping(vertex, tile);

  if (partialsSize - 1 > std::numeric_limits<unsigned short>::max() &&
      !targetIsCpu) {
    throw poputil::poplibs_error("Partials size larger than short");
  }

  if (params.useScale) {
    graph.connect(vertex["k"], params.scale.reshape({1}));
  }
  graph.connect(vertex["out"], outputs[0]);
  graph.setInitialValue(vertex["outCount"], outCount);
  graph.connect(vertex["partials"], partials);
  graph.setInitialValue(vertex["partialsSizeM1"], partialsSize - 1);
}

static void createSingleOutputVertex(
    poplar::Graph &graph, const std::vector<RegionReduction> &reductions,
    const poplar::Tensor &allPartials, const bool targetIsCpu,
    const poplar::VertexRef &vertex,
    const ReductionSpecialisation specialisation) {
  // Check that the partials can be received via exchange without
  // requiring a rearrangement due to misalignment. If this is not the
  // case we should be using the generic vertex
  unsigned numBytes = 0;
  for (const auto p : reductions[0].partials) {
    if (numBytes % 4) {
      throw poputil::poplibs_error(
          "Generating a singleIO reduction vertex with misaligned partials");
    }
    numBytes +=
        p.numElements() * graph.getTarget().getTypeSize(p.elementType());
  }
  graph.connect(vertex["out"], reductions[0].output);

  graph.connect(vertex["partials"], allPartials);

  const bool singleOutputRegion =
      specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION;
  const auto numPartials =
      singleOutputRegion
          ? allPartials.numElements() / reductions[0].output.numElements()
          : allPartials.numElements();

  if (numPartials > std::numeric_limits<unsigned short>::max() &&
      !targetIsCpu) {
    throw poputil::poplibs_error("Number of partials larger than short");
  }
  if (singleOutputRegion) {
    // numPartials per output
    graph.setInitialValue(vertex["numPartials"], numPartials);
    graph.setInitialValue(vertex["numOutputs"],
                          reductions[0].output.numElements());
  } else {
    // single output
    graph.setInitialValue(vertex["numPartials"], numPartials);
  }
}

static void
createReductionVertex(poplar::Graph &graph, const unsigned numOutputRegions,
                      const std::vector<RegionReduction> &reductions,
                      const std::string &debugPrefix,
                      const poplar::VertexRef vertex, const bool targetIsCpu) {
  // Work out the total number of partial regions.
  unsigned numPartialRegions = 0;

  // Number of input partial regions for each output region, start with 0.
  std::vector<unsigned short> numPartials;
  numPartials.reserve(numOutputRegions);

  for (const auto &r : reductions) {
    auto sz = r.partials.size();

    if (sz < 1) {
      throw poputil::poplibs_error("output region with no partials");
    }
    if (sz > std::numeric_limits<unsigned short>::max() && !targetIsCpu) {
      // As total memory on Colossus B0 is 2**18, 2**16 * num_workers
      // assuming that the work is split across workers
      // would occupy more memory than we have. If work is not split across
      // workers, then if partials[i].size() < 4 for all reductions
      // could hit this limit.
      // In future we may have to deal with num partials greater than this
      // and create more vertices
      throw poputil::poplibs_error("Number of partials larger than short");
    }
    numPartialRegions += sz;
    numPartials.push_back(static_cast<unsigned short>(sz));
  }

  auto t = graph.addConstant(poplar::UNSIGNED_SHORT, {numPartials.size()},
                             numPartials.data(), debugPrefix + "/numPartials");
  graph.setTileMapping(t, 0);
  graph.connect(vertex["numPartials"], t);

  std::vector<poplar::Tensor> partials;
  partials.reserve(numPartialRegions);
  std::vector<poplar::Tensor> outputs;
  outputs.reserve(numOutputRegions);

  for (const auto &r : reductions) {
    outputs.emplace_back(r.output);

    for (const auto &partial : r.partials) {
      partials.emplace_back(partial);
    }
  }
  graph.connect(vertex["out"], outputs);
  graph.connect(vertex["partials"], partials);
}

static void
createContinuousReductionVertex(poplar::Graph &graph,
                                const std::vector<RegionReduction> &reductions,
                                const poplar::VertexRef vertex) {
  const unsigned numOutputs = reductions.size();
  const unsigned numPartials =
      reductions.front().partials.front().numElements();

  assert(numOutputs != 0);
  graph.setInitialValue(vertex["numOutputs"], numOutputs - 1);
  graph.setInitialValue(vertex["numPartials"], numPartials);
  std::vector<poplar::Tensor> partials;
  std::vector<poplar::Tensor> outputs;
  for (const auto &red : reductions) {
    assert(red.partials.size() == 1);
    partials.push_back(red.partials.front());
    outputs.push_back(red.output);
  }
  auto singlePartials = concat(partials);
  auto singleOutput = concat(outputs);
  graph.connect(vertex["partials"], singlePartials.flatten());
  graph.connect(vertex["out"], singleOutput.flatten());
}

static bool dimensionsMatch(const std::vector<poplar::Tensor> &partials) {
  const unsigned outerPartialDim = partials[0].shape()[0];
  for (unsigned i = 1; i < partials.size(); i++) {
    if (partials[i].shape()[0] != outerPartialDim) {
      return false;
    }
  }
  return true;
}

// Create the reduction vertex most suited to the data.
void createVertex(poplar::Graph &graph,
                  const std::vector<RegionReduction> &reductions,
                  const ReduceParams &params, const poplar::Type &partialType,
                  const poplar::Type &outputType, const poplar::ComputeSet &cs,
                  const unsigned tile, bool reductionUsesInput,
                  const std::string &debugPrefix) {

  const bool targetIsCpu =
      graph.getTarget().getTargetType() == poplar::TargetType::CPU;
  // Number of output regions for this vertex.
  auto numOutputRegions = reductions.size();

  if (numOutputRegions < 1)
    throw poputil::poplibs_error("no output regions in reduction");

  // Check the number of partials in each output region
  for (const auto &r : reductions) {
    auto sz = r.partials.size();
    if (sz < 1) {
      throw poputil::poplibs_error("output region with no partials");
    }
  }
  // logging begin
  if (logging::shouldLog(logging::Level::Trace)) {
    for (const auto &red : reductions) {
      logging::trace("Reduction output size = {}, input vector size = {}",
                     red.output.numElements(), red.partials.size());
      unsigned size = 0;
      unsigned count = 0;
      for (const auto &p : red.partials) {
        if (count == 0 || size == p.numElements()) {
          count++;
          size = p.numElements();
        } else {
          logging::trace("    Partials: {} with size: {}", count, size);
          count = 0;
        }
      }
      if (count != 0) {
        logging::trace("    Partials: {} with size: {}", count, size);
      }
    }
  }
  // Logging end
  std::vector<poplar::Tensor> partials;
  std::vector<poplar::Tensor> outputs;
  outputs.reserve(numOutputRegions);

  for (const auto &r : reductions) {
    outputs.emplace_back(r.output);

    for (const auto &partial : r.partials) {
      partials.emplace_back(partial);
    }
  }

  const auto grainSize = partialType == poplar::HALF ? 8 : 4;

  // If we can use the single output region specialisation with no
  // penalty for gathering the inputs then always do so.

  auto specialisation =
      getReductionVertexSpecialisation(graph, params, reductions, partialType);

  std::vector<poplar::Tensor> flattenedPartials;

  for (const auto p : reductions[0].partials) {
    flattenedPartials.emplace_back(p.flatten());
  }
  const auto allPartials = poplar::concat(flattenedPartials);
  const auto useSingleOutputSpecialisation =
      allPartials.isContiguous() &&
      specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION;

  // Is it possible to use the PartialsEqualSize vertex which is an efficient
  // way to reduce multiple regions of the same size if that size is
  // convenient.
  const bool reducePartialsEqualSizeHasAssembly =
      params.op == Operation::ADD || params.op == Operation::SQUARE_ADD ||
      params.op == Operation::MAX || params.op == Operation::MIN;

  const bool reducePartialsEqualSizeIsPossible =
      reductions.size() == 1 && outputs.size() == 1 &&
      outputs[0].shape().size() == 1 &&
      outputs[0].shape()[0] % grainSize == 0 && outputs[0].shape()[0] != 0 &&
      dimensionsMatch(partials) && reducePartialsEqualSizeHasAssembly;

  // Only use PartialsEqualSize if its use avoids gathering data, which has a
  // copy cost.  This is the case where the input to this reduction involves
  // the actual input (not an intermediate value).  Also avoid the case where
  // the partials are long in comparison to the output.  In this instance the
  // copy cost is relatively small, compared to the speed difference between
  // PartialsEqualSize (slower) and the single output specialisation (faster).
  // The value of 8 was determined based on observed cases.
  if (reducePartialsEqualSizeIsPossible && !useSingleOutputSpecialisation &&
      reductionUsesInput &&
      outputs[0].shape()[0] * 8 > partials[0].shape()[0]) {

    return createPartialsAreInputSizeVertex(partials, outputs, graph, grainSize,
                                            tile, params, cs, targetIsCpu,
                                            partialType, outputType);
  } else {

    const auto name = getReductionVertexName(params, partialType, outputType,
                                             specialisation, params.useScale);
    logging::trace("{}", name);
    const auto vertex = graph.addVertex(cs, name);
    graph.setTileMapping(vertex, tile);
    if (reductionSupportsScaling(specialisation) && params.useScale) {
      graph.connect(vertex["k"], params.scale.reshape({1}));
    }
    if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT ||
        specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION) {
      return createSingleOutputVertex(graph, reductions, allPartials,
                                      targetIsCpu, vertex, specialisation);
    }
    if (specialisation == ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
      return createContinuousReductionVertex(graph, reductions, vertex);
    }
    createReductionVertex(graph, numOutputRegions, reductions, debugPrefix,
                          vertex, targetIsCpu);
  }
}

// Split `rows` into up to N groups with a minimum of 2 rows per group.
// If possible the number of groups is miminised without increasing the
// maximum number of rows in each group. For example with rows=9, N=4
// the output is {3,3,3} rather than {3,2,2,2}.
std::vector<unsigned> splitRowsToWorkers(unsigned rows, unsigned N,
                                         unsigned grainSize) {
  if (rows <= 3 || N <= 1)
    return {rows};
  auto maxRowsPerWorker = udiv(rows, N);

  auto maxWorkers = rows / 2;
  auto numWorkers = std::min(maxWorkers, udiv(rows, maxRowsPerWorker));

  std::vector<unsigned> split(numWorkers,
                              grainSize * (rows / (numWorkers * grainSize)));
  auto plusOnes = rows % (numWorkers * grainSize);
  for (unsigned i = 0; i < plusOnes / grainSize; ++i) {
    split[i] += grainSize;
  }
  // Add the remaining non-grainSize pieces to the last reduction.
  split.back() += plusOnes % grainSize;
  return split;
}

// Every tensor in `partials` is a 1D tensor with a length that is a multiple
// of `outputSize`. Imagine the partials are all concatenated and then wrapped
// to make an N * `outputSize` 2D tensor. This function splits that tensor
// up by the row (according to the row counts in `rows) and returns
// a set of 1D row tensors for each chunk of rows.
//
// The output is not set in the returned values. You have to do that yourself.
std::vector<RegionReduction>
splitPartialsByRows(const RegionReduction &reduction,
                    const std::vector<unsigned> &rows) {

  assert(reduction.partialsDebugInfo.size() == reduction.partials.size());

  std::vector<RegionReduction> out(rows.size());

  const auto outputSize = reduction.output.numElements();

  // Get the number of rows in each partial.
  std::vector<std::size_t> rowsInPartials(reduction.partials.size());
  for (std::size_t i = 0; i < rowsInPartials.size(); ++i)
    rowsInPartials[i] = reduction.partials[i].numElements() / outputSize;

  assert(std::accumulate(rowsInPartials.begin(), rowsInPartials.end(), 0) ==
         std::accumulate(rows.begin(), rows.end(), 0));

  std::size_t currentPartial = 0;
  std::size_t rowBegin = 0;

  for (unsigned i = 0; i < out.size(); ++i) {
    // Get `rows[i]` rows from the reduction and put them in out[i].

    auto rowsNeeded = rows[i];

    while (rowsNeeded > 0) {
      assert(currentPartial < rowsInPartials.size());

      auto rowsInCurrentPartial = rowsInPartials[currentPartial];

      std::size_t rowEnd = rowBegin + rowsNeeded;
      if (rowEnd > rowsInCurrentPartial)
        rowEnd = rowsInCurrentPartial;

      rowsNeeded -= (rowEnd - rowBegin);

      if (rowBegin == 0 && rowEnd == rowsInCurrentPartial) {
        // No need to slice.
        out[i].partials.push_back(reduction.partials[currentPartial]);
        out[i].partialsDebugInfo.push_back(
            reduction.partialsDebugInfo[currentPartial]);
      } else {
        // Take a slice.
        out[i].partials.push_back(
            reduction.partials[currentPartial].flatten().slice(
                rowBegin * outputSize, rowEnd * outputSize));

        // Debug info
        auto di = reduction.partialsDebugInfo[currentPartial];
        if (di.sourceCols.size() == outputSize) {
          di.sourceRows = {di.sourceRows.begin() + rowBegin,
                           di.sourceRows.begin() + rowEnd};
        } else {
          di.sourceRows = {0, 1};
          di.sourceCols = {rowBegin * outputSize, rowEnd * outputSize};
        }
        out[i].partialsDebugInfo.push_back(di);
      }

      rowBegin = rowEnd;
      if (rowBegin >= rowsInCurrentPartial) {
        rowBegin = 0;
        ++currentPartial;
      }
    }
  }
  return out;
}

// Call this function to connect reductions where none of them are two-stage
// reductions and there may be more reductions than worker contexts.
// The `assignments` should be the same size as `reductions` and
// contains the worker context/vertex each reduction should be done by.
void connectSingleStageReductions(
    poplar::Graph &graph, poplar::ComputeSet &cs, ReduceParams params,
    poplar::Type partialType, poplar::Type outputType, unsigned tile,
    const std::vector<RegionReduction> &reductions,
    const std::vector<unsigned> &assignments, bool reductionUsesInput,
    ReductionDebug::TileReduction *tileDebug, const std::string &debugPrefix) {

  assert(reductions.size() == assignments.size());

  logging::trace("Connecting single stage reduction, size: {}",
                 reductions.size());

  // Map from worker context to list of reductions.
  // Optimisation: Copying the RegionReductions around could be avoided.
  std::map<unsigned, std::vector<RegionReduction>> reductionsPerWorker;

  for (std::size_t i = 0; i < assignments.size(); ++i)
    reductionsPerWorker[assignments[i]].emplace_back(reductions[i]);

  assert(reductionsPerWorker.size() <=
         graph.getTarget().getNumWorkerContexts());

  // Connect the single stage reductions.
  for (const auto &it : reductionsPerWorker) {
    const auto &vertexReductions = it.second;

    createVertex(graph, vertexReductions, params, partialType, outputType, cs,
                 tile, reductionUsesInput, debugPrefix);
    if (tileDebug != nullptr) {
      for (const auto &reduction : vertexReductions) {
        ReductionDebug::RegionReduction rr;
        rr.vertex = it.first;
        rr.output = reduction.outputDebugInfo;
        rr.partials = reduction.partialsDebugInfo;
        tileDebug->firstStage.singleStageRegions.push_back(rr);
      }
    }
  }
}

// This is called when the number of reductions is less than or equal to
// the number of workers, and some of them *may* be split. Despite the name
// there may actually be no two-stage reductions, or there may be a mix,
// or there may be no single-stage reductions.
//
// `splits` is the number of pieces to split each reduction into (if 1 it
// means it is a single-stage reduction).
void connectTwoStageReductions(
    poplar::Graph &graph, ComputeSetList &css, const ReduceParams &params,
    poplar::Type partialType, poplar::Type outputType, unsigned tile,
    const std::vector<RegionReduction> &reductions,
    const std::vector<unsigned> &splits, bool reductionUsesInput,
    const std::string &debugPrefix, ReductionDebug::TileReduction *tileDebug) {
  // Triple check...
  assert(splits.size() == reductions.size());
  std::vector<RegionReduction> singleStageReductions;
  std::vector<unsigned> singleStageAssignments;

  for (std::size_t i = 0; i < splits.size(); ++i) {
    if (splits[i] == 1) {
      singleStageReductions.push_back(reductions[i]);
      singleStageAssignments.push_back(singleStageAssignments.size());
    }
  }

  auto firstCs = css.add(graph, debugPrefix + "/Reduce");

  if (!singleStageReductions.empty()) {
    connectSingleStageReductions(graph, firstCs, params, partialType,
                                 outputType, tile, singleStageReductions,
                                 singleStageAssignments, reductionUsesInput,
                                 tileDebug, debugPrefix);
  }

  // If there are no two-stage reductions, that's it!
  if (singleStageReductions.size() == reductions.size())
    return;

  // Map from reduction number to the partial for second stage reductions.
  std::map<unsigned, poplar::Tensor> secondStagePartials;

  unsigned currentVertex = singleStageReductions.size();

  // Now connect the first stage of the two-stage reductions.
  for (std::size_t i = 0; i < splits.size(); ++i) {
    // Ignore the single-stage ones we've already done.
    if (splits[i] <= 1)
      continue;

    // Optimisation: In some cases we'd want to do a two stage reduction anyway
    // e.g. if there are 6 reductions with reduction factors
    //   x1000, x5, x5, x5, x5, x5.

    auto outputSize = reductions[i].output.numElements();

    auto totalRows = reductionFactor(reductions[i]);
    const bool rowSizeOne = reductions[i].output.numElements() == 1;

    auto rowsPerWorker = splitRowsToWorkers(
        totalRows, splits[i],
        rowSizeOne ? graph.getTarget().getVectorWidth(partialType) : 1);

    assert(!rowsPerWorker.empty());
    assert(rowsPerWorker.size() <= splits[i]);
    assert(std::accumulate(rowsPerWorker.begin(), rowsPerWorker.end(), 0u) ==
           totalRows);

    auto partialsPerWorker = splitPartialsByRows(reductions[i], rowsPerWorker);

    assert(partialsPerWorker.size() == rowsPerWorker.size());

    // Create a tensor for all the partial results.
    secondStagePartials[i] =
        graph.addVariable(outputType, {outputSize * partialsPerWorker.size()},
                          debugPrefix + "/secondStagePartials");
    graph.setTileMapping(secondStagePartials[i], tile);

    // Now create the new RegionReductions.
    for (unsigned s = 0; s < rowsPerWorker.size(); ++s) {
      RegionReduction firstStage;
      firstStage.output =
          secondStagePartials[i].slice(s * outputSize, (s + 1) * outputSize);
      firstStage.partials = partialsPerWorker[s].partials;

      // Don't do scale or update in the first stage.
      ReduceParams firstStageParams(params.op);
      createVertex(graph, {firstStage}, firstStageParams, partialType,
                   outputType, firstCs, tile, reductionUsesInput, debugPrefix);

      // Debug info.
      if (tileDebug != nullptr) {
        ReductionDebug::RegionReduction rr;
        rr.vertex = currentVertex++;
        rr.output.outputRegion = reductions[i].outputDebugInfo.outputRegion;
        rr.output.dataRegion = {s * outputSize, (s + 1) * outputSize};
        rr.partials = partialsPerWorker[s].partialsDebugInfo;

        tileDebug->firstStage.firstStageRegions.push_back(rr);
      }
    }
  }

  // And the second stage of the two-stage reductions.

  // Add a second compute set if needed.
  auto secondCs = css.add(graph, debugPrefix + "/Reduce_Second_Stage");

  // Work out which vertex should do each second stage. We just assign
  // in a round-robin manner since there shouldn't really ever be more than
  // the number of worker contexts anyway.

  std::vector<std::vector<RegionReduction>> secondStageReductions;
  secondStageReductions.reserve(graph.getTarget().getNumWorkerContexts());

  // Don't square again in the second stage.
  ReduceParams secondStageParams = params;
  if (params.op == Operation::SQUARE_ADD) {
    secondStageParams.op = Operation::ADD;
  }

  currentVertex = 0;

  std::size_t partialColsHighlightOffset = 0;

  for (const auto &r : secondStagePartials) {
    assert(r.first >= 0 && r.first < reductions.size());

    RegionReduction secondStageReduction;
    secondStageReduction.output = reductions[r.first].output;
    // There's only ever one partial for each second stage reduction.
    secondStageReduction.partials.emplace_back(r.second);

    createVertex(graph, {secondStageReduction}, secondStageParams, outputType,
                 outputType, secondCs, tile, reductionUsesInput, debugPrefix);

    // Debug info
    if (tileDebug != nullptr) {
      ReductionDebug::RegionReduction rr;
      rr.vertex = currentVertex++;
      rr.output = reductions[r.first].outputDebugInfo;

      ReductionDebug::Partial p;
      p.sourceTile = tile;
      p.sourceRows = {0, 1};
      // This is not quite true but it makes the visualisation work.
      // Basically each region reduction gets its own intermediate
      // partial, and they are all drawn next to each other with
      // the same highlight div in the visualisation, so the sourceCols
      // here are relative to the first one.
      p.sourceCols = {partialColsHighlightOffset,
                      partialColsHighlightOffset + r.second.numElements()};
      partialColsHighlightOffset += r.second.numElements();
      rr.partials.push_back(p);

      tileDebug->secondStage.secondStageRegions.push_back(rr);
    }
  }
}

} // anonymous namespace

void connectReductions(poplar::Graph &graph, ComputeSetList &css,
                       ReduceParams params, poplar::Type partialType,
                       poplar::Type outputType, unsigned tile,
                       const std::vector<RegionReduction> &reductions,
                       bool reductionUsesInput, const std::string &debugPrefix,
                       ReductionDebug::TileReduction *tileDebug) {

  const auto &target = graph.getTarget();
  // Optimisation: If there is just one partial for an output we don't need to
  // calculate it, but hooking that up is a bit fiddly.

  // Check that all the inputs and outputs are of the appropriate sizes
  // and types.
  logging::trace("Connecting {} reductions", reductions.size());

  for (const auto &r : reductions) {

    auto outputSize = r.output.numElements();
    if (outputSize == 0)
      throw poputil::poplibs_error("Zero-sized reduction output");

    if (r.output.elementType() != outputType)
      throw poputil::poplibs_error("Reduction output is incorrect type");

    for (const auto &p : r.partials) {
      if (p.numElements() == 0)
        throw poputil::poplibs_error("Zero-sized reduction partial");
      if (p.numElements() % outputSize != 0)
        throw poputil::poplibs_error("Reduction partial size is not a multiple "
                                     "of the output size");

      if (p.elementType() != partialType)
        throw poputil::poplibs_error("Reduction partial is incorrect type");
    }
  }

  const std::size_t vectorListMaxSize = [&] {
    // output region size does not change the field dimension of the vertex and
    // it doesn't matter if the last parameter is true or false. We just set it
    // to true without checking if the size is one or not
    const auto vertex = getReductionVertexName(
        params, partialType, outputType, ReductionSpecialisation::DEFAULT);
    return graph.getMaxFieldDim(vertex, "partials", 0);
  }();

  // reductions contains a list of ReductionRegions, if any of the regions
  // contain more partials than we can fit into a DeltaN list we must split
  // those regions up.
  const auto partialsTooLarge = [&](const RegionReduction &reduction) {
    return reduction.partials.size() >= vectorListMaxSize;
  };

  const bool useTwoStageReduction =
      reductions.size() < target.getNumWorkerContexts() ||
      std::any_of(std::begin(reductions), std::end(reductions),
                  partialsTooLarge);

  // See if there is the possibility of easily splitting reductions
  // into two-level ones.
  if (useTwoStageReduction) {
    // Try to split some into multi-stage reductions.
    auto splits = splitTwoStageReductionsBetweenWorkers(
        target, params.op, reductions, vectorListMaxSize);
    const auto workersUsed = std::accumulate(splits.begin(), splits.end(), 0u);
    logging::trace("Splitting {} reductions between {} vertices on tile {} {}",
                   reductions.size(), workersUsed, tile,
                   reductions.size() == workersUsed
                       ? " "
                       : "(Plus reductions to combine those split by row)");

    connectTwoStageReductions(graph, css, params, partialType, outputType, tile,
                              reductions, splits, reductionUsesInput,
                              debugPrefix, tileDebug);

  } else {
    logging::trace("Using single stage reduction on tile {}", tile);
    // Distribute the reductionTensors among 6 (or whatever) vertices.
    auto reductionAssignments =
        distributeReductionsBetweenWorkers(target, params.op, reductions);

    // We need at least one compute set for the single stage reduction.
    auto cs = css.add(graph, debugPrefix + "/Reduce");

    connectSingleStageReductions(graph, cs, params, partialType, outputType,
                                 tile, reductions, reductionAssignments,
                                 reductionUsesInput, tileDebug, debugPrefix);
  }

  if (tileDebug != nullptr)
    tileDebug->tileIndex = tile;
}

/// reduction with a scalar output and word-aligned inputs can be executed
/// using specialised vertices
/// \param graph  The compute graph
/// \param params The parameters of this reduction
/// \param r      The reductions to be performed
static bool isSingleIOReduction(const poplar::Graph &graph,
                                const ReduceParams &params,
                                const std::vector<RegionReduction> &r) {

  // This must be a reduction of a single region
  if (params.update || r.size() != 1)
    return false;
  const auto &target = graph.getTarget();
  const auto &r0 = r.front();
  // The output must be scalar or 8byte writable
  auto outputElements = r0.output.numElements();
  if (outputElements > 1 &&
      outputElements * target.getTypeSize(r0.output.elementType()) % 8 != 0)
    return false;
  bool nextIsMisaligned = false;
  for (const auto &p : r0.partials) {
    // Each incoming partial region must be for a full set of outputs
    if (p.numElements() % outputElements != 0)
      return false;
    // It must be possible to receive each partial over exchange without
    // requiring a gather.
    if (nextIsMisaligned)
      return false;
    nextIsMisaligned =
        p.numElements() * target.getTypeSize(p.elementType()) % 4;
  }
  return true;
}

static bool allRegionsContinuous(const poplar::Graph &graph,
                                 const std::vector<RegionReduction> &regions,
                                 const ReduceParams &params) {
  boost::optional<unsigned> partialsSize;
  std::vector<poplar::Tensor> outputs;
  std::vector<poplar::Tensor> partials;
  if (regions.empty()) {
    return false;
  }
  for (const auto &red : regions) {
    if (red.output.numElements() != 1) {
      return false;
    }
    // TODO: T12964 We will be able to target this a lot more if we get to use
    // more information about the layout of outputs from multiple reductions.
    // For now, deal with a single one. Also, more cases can be targeted if
    // we use the logic in isSingleIOReduction for "It must be possible to
    // receive each partial over exchange without requiring a gather."

    if (red.partials.size() != 1) {
      return false;
    }
    if (red.partials[0].numElements() == 1) {
      return false;
    }
    if (partialsSize) {
      if (red.partials.front().numElements() != partialsSize.get()) {
        return false;
      }
    } else {
      partialsSize = red.partials.front().numElements();
    }

    outputs.push_back(red.output);
    partials.push_back(red.partials.front());
  }

  const auto singleOut = concat(outputs);
  const auto singlePart = concat(partials);
  return singleOut.isContiguous() && singlePart.isContiguous();
}

ReductionSpecialisation getReductionVertexSpecialisation(
    const poplar::Graph &graph, const ReduceParams &params,
    const std::vector<RegionReduction> &regions, poplar::Type partialType) {

  // We don't have assembler implementations for some specialisations
  // other than for ADD and SQUARE_ADD operations. The code generated by the
  // compiler is very inefficient for these operations.
  bool opIsAddOrSquareAdd =
      params.op == Operation::ADD || params.op == Operation::SQUARE_ADD;

  if (allRegionsContinuous(graph, regions, params) && opIsAddOrSquareAdd) {
    return ReductionSpecialisation::ALL_REGIONS_CONTINUOUS;
  }
  if (isSingleIOReduction(graph, params, regions) && opIsAddOrSquareAdd) {
    const auto &region = regions[0];
    auto scalarOutput = region.output.numElements() == 1;
    if (scalarOutput && !params.useScale) {
      return ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT;
    } else {
      // both input and output must be full width accumulators
      const auto &target = graph.getTarget();
      if (region.output.elementType() == poplar::FLOAT &&
          region.output.numElements() *
                  target.getTypeSize(region.output.elementType()) % 8 ==
              0 &&
          region.output.numElements() *
                  target.getTypeSize(region.partials[0].elementType()) % 8 ==
              0) {
        // output must be whole words
        return ReductionSpecialisation::SINGLE_OUTPUT_REGION;
      }
    }
  }
  // find if all output regions are of size 1
  auto allOutputRegionsOfSizeOne = std::all_of(
      regions.begin(), regions.end(), [](const popops::RegionReduction &r) {
        return r.output.numElements() == 1;
      });
  if (allOutputRegionsOfSizeOne) {
    return ReductionSpecialisation::SCALAR_OUTPUT_REGIONS;
  }
  return ReductionSpecialisation::DEFAULT;
}

} // namespace popops
