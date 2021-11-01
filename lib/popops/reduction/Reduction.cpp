// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "Reduction.hpp"

#include "IntermediatePartials.hpp"
#include "IntermediatePartialsUtil.hpp"
#include "ReductionIntrospection.hpp"
#include "ReductionPlan.hpp"
#include "ReductionStages.hpp"
#include "poplibs_support/Algorithms.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/OptionParsing.hpp"
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/ContiguousRegionsByTile.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_support/print.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <boost/functional/hash.hpp>
#include <boost/icl/separate_interval_set.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/variant.hpp>

#include <tbb/parallel_for.h>

#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

using namespace poplar;
using namespace poplibs;
using namespace poplibs_support;

namespace poputil {
template <> poplar::ProfileValue toProfileValue(const popops::ReduceParams &t) {
  poplar::ProfileValue::Map v;
  v.insert({"op", toProfileValue(t.op)});
  v.insert({"update", toProfileValue(t.update)});
  v.insert({"useScale", toProfileValue(t.useScale)});
  if (t.useScale) {
    v.insert({"scale", toProfileValue(t.scale)});
  }
  return v;
}
template <>
poplar::ProfileValue toProfileValue(const popops::SingleReduceOp &op) {
  poplar::ProfileValue::Map v;
  v.emplace("in", toProfileValue(op.in));
  v.emplace("dims", toProfileValue(op.dims));
  v.emplace("params", toProfileValue(op.params));
  v.emplace("useOutType", toProfileValue(op.useOutType));
  if (op.useOutType)
    v.emplace("outType", toProfileValue(op.outType));
  v.emplace("debugName", toProfileValue(op.debugName));
  return v;
}
} // namespace poputil

namespace popops {

namespace {

// Structure to describe the accumulation types at different levels of the
// reduction.
struct ReductionTypes {
  // The type for temporary stack values used for accumulation within a vertex.
  //
  // If we do multi-stage reductions on a tile, this type is also used for the
  // intermediate values.
  //
  // If this type is not the output type of the reduction then the vertices
  // in the final stage of the reduction may output the accumulator type and
  // then a separate cast operation will convert the output to the final
  // output type.
  Type inVertex;
  // The type for values sent between tiles on the exchange.
  Type interTile;
};

// pick a random start tile for the intermediate to intermediate reductions to
// use. this is an attempt to better balance work across the tiles in a large
// model without having the whole model available at this point.
unsigned getStartTile(const std::vector<std::size_t> &inShape,
                      const std::vector<std::size_t> &outShape,
                      const ReduceParams &params, const unsigned stageNumber,
                      const unsigned reductionId,
                      const unsigned tilesInTarget) {
  // starting seed: 2^32/phi, where phi is the golden ratio.
  std::size_t seed = 0x9e3779b9UL;
  boost::hash_range(seed, std::begin(inShape), std::end(inShape));
  boost::hash_range(seed, std::begin(outShape), std::end(outShape));

  using T = std::underlying_type_t<decltype(params.op)>;
  boost::hash_combine(seed, static_cast<T>(params.op));
  boost::hash_combine(seed, params.update);
  boost::hash_combine(seed, params.useScale);
  boost::hash_combine(seed, reductionId);
  boost::hash_combine(seed, stageNumber);

  return seed % tilesInTarget;
}

// Reduce a 2D tensor in the first dimension. No other tensor shape
// is supported. The tensor must be at least 1x2.
//
// `out` must be a 1D tensor with the right number of elements. Its tile mapping
// need not necessarily be set.
// `reductionResultTensors` is a struct into which this function will push any
// tensor that is written to with a reduction result. These can be written to
// in 2 separate compute sets. If so, they will require a WriteUndef to be
// added to the program before the compute sets in 'css' to prevent them from
// becoming always live.
void reduceFirstDim2D(Graph &graph, const Tensor &in_,
                      boost::optional<Tensor> &out,
                      const std::vector<std::size_t> outputShape,
                      Type outputType, ReduceParams params,
                      const ReductionTypes &reductionTypes,
                      std::vector<ComputeSet> &css,
                      ResultTensors &reductionResultTensors,
                      unsigned reductionId, const DebugNameAndId &dnai) {
  logging::popops::debug("Reducing first dimension");
  // We only accept reductions over 2D tensors.
  if (in_.rank() != 2) {
    throw poputil::poplibs_error("expected rank 2 but got rank " +
                                 std::to_string(in_.rank()));
  }
  const auto columns = in_.dim(1);
  if (out) {
    // Output should be the correct size. It will be flattened when used later
    // so the shape isn't important
    if (columns != out.get().flatten().dim(0)) {
      throw poputil::poplibs_error("expected output size " +
                                   std::to_string(in_.dim(1)) + " but got " +
                                   std::to_string(out.get().flatten().dim(0)));
    }
  }
  Tensor in = in_;
  // Get the tile mapping at the start of this function because it can be
  // slow and we really don't want to call it more often than we need to.
  auto mapping = graph.getTileMapping(in);
  // Introspect to find the groups of columns and their ordering on all the
  // tiles.
  auto [groupedPartials, contiguousRegionsByTile] =
      allTileGroupedPartials(graph, in, mapping, columns);
  // Analyse over all tiles to see if the layout of columns in memory is
  // consistent.  There are 3 possibilities
  // a) Sequential column ordering: 0,1,2... on all tiles
  // b) Inconsistent ordering: 0,1,2 on some tiles, 2,1,0... on others
  // c) Non-sequential ordering: 2,1,0 on all tiles.
  // Below we can rearrange the input and outputs to cope with case b) and c),
  // in case a) there is nothing to do so some of the work can be skipped
  // to save time.
  auto columnsOrder = findCommonColumnOrder(groupedPartials, columns);

  bool withOutput = !(out == boost::none);
  bool groupedPartialsValid = true;
  // Remember the original tensor, as this can be a simpler expression to pass
  // to writeUndef, which will improve compile time
  boost::optional<Tensor> originalOutput = out;

  // Avoid mutating `out` when `columnsOrder` and `withOutput` is true.
  boost::optional<Tensor> outCopy = out;

  if (columnsOrder) {
    if (logging::popops::shouldLog(logging::Level::Debug)) {
      logging::popops::debug("Non incremental column ordering detected,"
                             " rearranging to optimise reduction");
      const unsigned maxDebugColumns = 256;
      if (columnsOrder.get().size() > maxDebugColumns) {
        const std::vector<unsigned> debugColumns(columnsOrder.get().begin(),
                                                 columnsOrder.get().begin() +
                                                     maxDebugColumns);
        logging::popops::debug("First {} columns (of {}), order:{}",
                               maxDebugColumns, columnsOrder.get().size(),
                               debugColumns);

      } else {
        logging::popops::debug("Column order:{}", columnsOrder.get());
      }
    }
    // Build a set of slices that represent the non-sequential column ordering
    // that was detected, grouping those that are consecutive to simplify the
    // resulting representation
    std::vector<Interval> slices;
    const auto columns = columnsOrder.get();
    slices.reserve(columns.size());
    Interval currentSlice = {columns[0], columns[0] + 1};
    for (unsigned i = 1; i < columns.size(); i++) {
      if (columns[i] == currentSlice.end()) {
        currentSlice = {currentSlice.begin(), columns[i] + 1};
      } else {
        slices.push_back(currentSlice);
        currentSlice = {columns[i], columns[i] + 1};
      }
    }
    slices.push_back(currentSlice);
    // re-arrange the input and output to match the input columns memory layout
    if (withOutput) {
      outCopy = concat(out.get().flatten().slices(slices)).reshape(outputShape);
    }
    in = concat(in.slices(slices, 1), 1);
    // We may need to present a revised description of all of this to the later
    // functions, flag that as the input was changed the information is invalid.
    // Mapping is however always needed
    mapping = graph.getTileMapping(in);
    groupedPartialsValid = false;
  }

  // Find the output value whose inputs are spread over the most tiles. In other
  // words find the column of A that is mapped to the most tiles.
  auto maxTileSpread = getMaxTileSpread(mapping, columns);

  // Possible if there are no elements but we shouldn't get to this point
  // if that is true.
  if (maxTileSpread < 1)
    throw poputil::poplibs_error("internal error calculating tile spread for "
                                 "reduction plan");

  ComputeSetList csList(css);

  logging::popops::debug("Num elements to reduce {} -> {}", in.numElements(),
                         columns);

  auto restoreOutputShape = [&](boost::optional<Tensor> &out) {
    if (!withOutput && columnsOrder) {
      std::vector<Interval> outputSlices(out.get().numElements());
      for (unsigned j = 0; j < outputSlices.size(); j++) {
        outputSlices[columnsOrder.get()[j]] = {j, j + 1};
      }
      // Combine individual slices if they are consecutive
      std::vector<Interval> combinedSlices;
      combinedSlices.reserve(outputSlices.size());
      combinedSlices.push_back(outputSlices.front());
      for (unsigned j = 1; j < outputSlices.size(); j++) {
        if (outputSlices[j].begin() == combinedSlices.back().end()) {
          combinedSlices.back() = {combinedSlices.back().begin(),
                                   outputSlices[j].end()};
        } else {
          combinedSlices.push_back(outputSlices[j]);
        }
      }
      out = concat(out.get().flatten().slices(combinedSlices))
                .reshape(outputShape);
      logging::popops::debug("Restoring column ordering");
    }
  };

  if (maxTileSpread == 1) {
    logging::popops::debug("Reduction is completely tile local");
    if (groupedPartialsValid == false) {
      std::tie(groupedPartials, contiguousRegionsByTile) =
          allTileGroupedPartials(graph, in, mapping, columns);
      groupedPartialsValid = true;
    }

    auto numMappedTiles = std::accumulate(
        mapping.begin(), mapping.end(), 0U,
        [](unsigned num, const auto &m) { return num + !m.empty(); });

    // Do the entire reduction on each tile with no exchange at all.

    // Remap output if reduction produces 1 element per tile and would
    // cause a subword write.
    const auto &target = graph.getTarget();
    const auto typeSize = target.getTypeSize(outputType);
    const auto storeGranularityBytes = target.getAtomicStoreGranularity();
    if ((typeSize < storeGranularityBytes) &&
        product(outputShape) == numMappedTiles) {
      if (!withOutput) {
        logging::popops::debug("Remap reduction output {}", dnai.getPathName());
        auto remappedOutput = graph.addVariable(outputType, outputShape,
                                                {dnai, "remappedOutput"});
        poputil::mapTensorLinearly(graph, remappedOutput, 0,
                                   storeGranularityBytes / typeSize);
        outCopy = remappedOutput;
      } else {
        logging::popops::warn(
            "Reduction output layout for {} mapped with "
            "single element per tile that could result in subword-writes",
            dnai.getPathName());
      }
    }

    inputToOutputNoExchange(graph, in, contiguousRegionsByTile, groupedPartials,
                            outCopy, originalOutput, outputShape,
                            reductionTypes.inVertex, outputType, params, csList,
                            reductionResultTensors, {dnai, "ReduceOnTile"});
    restoreOutputShape(outCopy);
  } else {

    IntermediatePartials ip;
    auto reductionStageInputType = in.elementType();

    // Check if we can just convert it without doing anything.
    if (!mappingHasMultipleValuesFromOneColumnOnTheSameTile(mapping, columns)) {
      ip = tensorToIntermediatePartials(in, mapping);
    } else {
      // Reduce as much as possible on each tile and return the intermediate
      // partials. We don't scale or update here.
      logging::popops::debug("Reduce locally with no exchange");
      if (groupedPartialsValid == false) {
        std::tie(groupedPartials, contiguousRegionsByTile) =
            allTileGroupedPartials(graph, in, mapping, columns);
        groupedPartialsValid = true;
      }
      ip = inputToIntermediateNoExchange(
          graph, in, contiguousRegionsByTile, groupedPartials, params.op,
          reductionTypes.inVertex, reductionTypes.interTile, csList,
          reductionResultTensors, {dnai, "ReduceOnTile"});
      reductionStageInputType = reductionTypes.inVertex;
      // If it was a SQUARE_ADD, then at this point we have now done the
      // SQUARE - change it to an ADD.
      if (params.op == Operation::SQUARE_ADD)
        params.op = Operation::ADD;
    }

    // each intermediateToIntermediate stage begins from the same tile. we
    // should be able to improve this by being a bit smarter and distributing
    // the tiles across the stages so that exchange of the partials is less.
    const auto &target = graph.getTarget();

    constexpr unsigned loopExit = ~0u;
    for (unsigned i = 0; i != loopExit; ++i) {
      const auto startTile = getStartTile(in.shape(), outputShape, params, i,
                                          reductionId, target.getNumTiles());
      // At each point, see if it is worth doing another reduction stage or if
      // we should just do the final reduction, and if so should we do
      // it spread over the tiles int he graph or at the destination?
      switch (calculateNextStep(graph.getTarget(), ip)) {
      case INTERMEDIATE_TO_INTERMEDIATE:
        logging::popops::debug("Introducing new intermediate to intermediate "
                               "reduction stage");
        // When splitting up the input we should split it into separate
        // reductions (i.e. split the columns up) as much as possible down to
        // the grain size) and then if necessary split it vertically (chunks
        // of rows) so that we can spread it over the IPU.

        // Don't do the scale or update.
        ip = intermediateToIntermediate(
            graph, ip, params.op, reductionTypes.interTile, csList,
            reductionResultTensors, startTile,
            {dnai, std::string("ReduceStage") + std::to_string(i)});
        // If it was a SQUARE_ADD, then at this point we have now done the
        // SQUARE - change it to an ADD.
        if (params.op == Operation::SQUARE_ADD)
          params.op = Operation::ADD;

        reductionStageInputType = reductionTypes.inVertex;
        break;
      case INTERMEDIATE_TO_OUTPUT:

        logging::popops::debug("Creating final reduction stage");
        intermediateToOutput(graph, ip, outCopy, originalOutput, outputShape,
                             outputType, params, reductionStageInputType,
                             csList, reductionResultTensors, in, reductionId,
                             {dnai, "ReduceFinalStage"});

        restoreOutputShape(outCopy);
        i = loopExit - 1; // exit the loop
        break;
      }
    }
  }

  if (!withOutput) {
    out = outCopy;
  }
}

bool opBenefitsFromHigherIntermediatePrecision(const popops::Operation &op) {
  switch (op) {
  case popops::Operation::ADD:
  case popops::Operation::LOG_ADD:
  case popops::Operation::SQUARE_ADD:
  case popops::Operation::MUL:
    return true;
  default:
    return false;
  }
  POPLIB_UNREACHABLE();
}

static float reductionInitialValue(const popops::Operation &op) {
  switch (op) {
  case popops::Operation::ADD:
  case popops::Operation::SQUARE_ADD:
  case popops::Operation::LOGICAL_OR:
    return 0.0;
  case popops::Operation::MIN:
    return std::numeric_limits<double>::infinity();
  case popops::Operation::MAX:
    return -std::numeric_limits<double>::infinity();
  case popops::Operation::MUL:
  case popops::Operation::LOGICAL_AND:
    return 1.0;
  default:
    throw poputil::poplibs_error("Internal error, unhandled reduction type:" +
                                 std::to_string(static_cast<int>(op)));
  }
}

static void validateReductionParams(ReduceParams const &params) {
  if (params.useScale && !(params.op == popops::Operation::ADD ||
                           params.op == popops::Operation::SQUARE_ADD ||
                           params.op == popops::Operation::LOG_ADD)) {
    throw poputil::poplibs_error("Scale can only be used with ADD, LOG_ADD or "
                                 "SQUARE_ADD");
  }
  if (params.useScale) {
    if (params.scale.elementType() != FLOAT) {
      throw poputil::poplibs_error("Scale must be of type poplar::FLOAT");
    }
  }
  if (params.update && !(params.op == popops::Operation::ADD ||
                         params.op == popops::Operation::SQUARE_ADD ||
                         params.op == popops::Operation::LOG_ADD)) {
    throw poputil::poplibs_error("Update can only be used with ADD, LOG_ADD or "
                                 "SQUARE_ADD");
  }
}

struct ReductionAnalysis {
  // True if the reduction can be replaced with a simpler expression,
  // because the dimensions being reduced in the input tensor are all
  // length 1. For example, reducing [2, 2, 1] along the 3rd dimension,
  // produces just [2, 2] so we can just copy the elements. False otherwise.
  bool canReduceWithMap;
  // The unique set of dimensions that we want to reduce from the input tensor.
  std::set<unsigned> reducedDims;
  // The dimensions of the output tensor; equivalent to the input tensor
  // without the dimensions that have been reduced.
  std::vector<size_t> outputShape;
  // The total number of elements in the output tensor. When the output
  // tensor's shape is empty this is 1. This can be zero if the output
  // shape has a dimension that is zero.
  size_t numOutputElements;
  // The total number of elements in the input tensor. This can be zero
  // if the input shape has a dimension that is zero.
  size_t numInputElements;
};
static ReductionAnalysis analyzeReduction(std::vector<size_t> const &dims,
                                          poplar::Tensor const &in,
                                          boost::optional<Tensor> &out) {
  bool canReduceWithMap = true;

  // Convert the dimensions into a unique set.
  std::set<unsigned> reducedDims(dims.begin(), dims.end());

  // Check that all the dimensions are actual dimensions of the input.
  for (auto dim : reducedDims) {
    if (dim >= in.rank())
      throw poputil::poplibs_error("Invalid dimension " + std::to_string(dim) +
                                   " for tensor rank " +
                                   std::to_string(in.rank()));
    else if (in.dim(dim) >= 2)
      canReduceWithMap = false;
  }

  std::size_t numInputElements = 1;
  std::size_t numOutputElements = 1;
  std::vector<std::size_t> outputShape;
  outputShape.reserve(in.rank() - reducedDims.size());
  for (std::size_t d = 0; d < in.rank(); ++d) {
    std::size_t dim = in.dim(d);
    if (reducedDims.count(d) == 0) {
      outputShape.push_back(dim);
      numOutputElements *= dim;
    }
    numInputElements *= dim;
  }

  // If we have one, check that the output tensor has the right shape.
  if (out && out.get().shape() != outputShape) {
    std::stringstream s;
    s << "Dimension mismatch in output. Input shape: ";
    printContainer(in.shape(), s);
    s << " Output shape: ";
    printContainer(out.get().shape(), s);
    s << " Reduced dimensions: ";
    printContainer(dims, s);
    throw poputil::poplibs_error(s.str());
  }

  return {canReduceWithMap, reducedDims, outputShape, numOutputElements,
          numInputElements};
}

static void convertCssToProg(const std::vector<ComputeSet> &css,
                             program::Sequence &prog,
                             const ResultTensors &reductionResultTensors,
                             DebugNameAndId dnai) {
  // WriteUndef the intermediate tensors used throughout the reduction.
  // Note that if there's only one compute set the liveness analysis can
  // figure it out and there's no need for the WriteUndef(s).
  if (css.size() > 1) {
    if (reductionResultTensors.typeA.size() > 0)
      prog.add(program::WriteUndef(
          concat(reductionResultTensors.typeA),
          {dnai, "reduceMany: type A reduction intermediates"}));
    if (reductionResultTensors.typeB.size() > 0)
      prog.add(program::WriteUndef(
          concat(reductionResultTensors.typeB),
          {dnai, "reduceMany: type B reduction intermediates"}));
  }

  // Prog is not modified until all reduction operations have been created.
  for (auto &cs : css)
    prog.add(program::Execute(cs, dnai));
}

static bool canReduceWithMap(const Tensor &in,
                             const std::vector<std::size_t> &dims) {
  if (in.numElements() == 0)
    return false;
  for (auto dim : dims)
    if (in.dim(dim) >= 2)
      return false;
  return true;
}

static void reduceWithMap(Graph &graph, const Tensor &in,
                          boost::optional<Tensor> &out,
                          const poplar::Type &outputType,
                          const std::vector<std::size_t> &dims,
                          ReduceParams params, program::Sequence &prog,
                          const DebugNameAndId &dnai,
                          const poplar::OptionFlags &options) {
  // If the input only reduces dimensions of size 1, we don't need to
  // do a reduction - just copy the input tensor to the output. A copy is
  // returned (using cast) rather than just returning the original because
  // then the user can be sure that the returned tensor refers to a distinct
  // variable and changing the tile mapping won't affect anything else.
  validateReductionParams(params);
  if (params.op == Operation::LOG_ADD) {
    throw poputil::poplibs_error(
        "Reduction operation LOG_ADD doesn't"
        " support reductions where there is no reduction required");
  }

  auto analysis = analyzeReduction(dims, in, out);
  if (!analysis.canReduceWithMap)
    throw poputil::poplibs_error(
        "reduceWithMap cannot be used with inputs that cannot be reduced "
        "with a map or cast expression.");

  logging::popops::debug("No reduction required");

  if (out) {
    // If the graph mapping isn't complete, set it to the same as the input.
    bool mappingComplete;
    graph.getTileMapping(out.get(), &mappingComplete);
    if (!mappingComplete) {
      graph.setTileMapping(out.get(), graph.getTileMapping(in));
    }
  } else {
    // Create the output Tensor
    out = graph.clone(outputType, in, {dnai}).reshape(analysis.outputShape);
  }

  // If it is a scale or update, or SQUARE_ADD we still need to do that.
  if (params.update || params.useScale || params.op == Operation::SQUARE_ADD) {
    // Calculate the necessary expression. E.g. the most complex case,
    //
    //   x += f * v^2
    //
    // is
    //
    //   Add(_1, Mul(Square(_2), params.scale))
    //
    // And the simplest, x = f * v, is
    //
    //   Mul(_2, params.scale)
    using namespace popops::expr;
    Type arithmeticType = outputType;
    Any expr = _2;
    if (in.elementType() == HALF && params.op == Operation::SQUARE_ADD) {
      // Do square add in higher precision as per normal reductions
      arithmeticType = FLOAT;
      expr = Cast(expr, arithmeticType);
    } else if (in.elementType() != outputType) {
      expr = Cast(expr, arithmeticType);
    }
    if (params.op == Operation::SQUARE_ADD) {
      expr = Square(expr);
    }
    if (params.useScale) {
      if (params.scale.elementType() != arithmeticType) {
        expr = Mul(expr, Cast(_3, arithmeticType));
      } else {
        expr = Mul(expr, _3);
      }
    }
    if (arithmeticType != outputType) {
      expr = Cast(expr, outputType);
    }
    if (params.update) {
      expr = Add(expr, _1);
    }
    if (params.useScale) {
      mapInPlace(graph, expr, {out.get().flatten(), in.flatten(), params.scale},
                 prog, {dnai, "ReduceExpression"});
    } else {
      mapInPlace(graph, expr, {out.get().flatten(), in.flatten()}, prog,
                 {dnai, "ReduceExpression"});
    }

  } else {
    // Cast is used here rather than copy because the type could be different
    // if ADD or MUL are used. If the type is the same cast() will
    // automatically switch to copy.
    auto castProg =
        cast(graph, in.flatten(), out.get().flatten(), {dnai, "ReduceCast"});
    prog.add(castProg);
  }
}

std::map<std::string, poplar::Type> accumTypeMap{{"half", poplar::HALF},
                                                 {"float", poplar::FLOAT}};

// This wangles the tensors into a 2D matrix so that the reduction only
// has to be done on the first dimension. Then it calls reduceFirstDim2D
// to do the reduction.
static void reduceWithOutputCss(
    Graph &graph, const Tensor &in, boost::optional<Tensor> &out,
    const poplar::Type &outputType, const std::vector<std::size_t> &dims,
    ReduceParams params, std::vector<ComputeSet> &css,
    ResultTensors &reductionResultTensors, unsigned reductionId,
    const DebugNameAndId &dnai, const poplar::OptionFlags &options) {

  const auto getShape = [](const Tensor &t) {
    std::stringstream ss;
    printContainer(t.shape(), ss);
    return ss.str();
  };
  const bool withOutput = out != boost::none;
  logging::popops::debug("Reduce{} Op: {} Update:{} Begin DebugStr: {}",
                         withOutput ? "WithOutput" : "", params.op,
                         params.update, dnai.getPathName());
  logging::popops::debug("  in({}){} : {}", in.elementType(), in.shape(),
                         in.getDebugStr());
  if (out) {
    logging::popops::debug("  out({}){}: {}", out.get().elementType(),
                           out.get().shape(), out.get().getDebugStr());

    logging::popops::debug("  {}{} = reduce({}{}), dims={}",
                           out.get().getVarStr(), out.get().shape(),
                           in.getVarStr(), in.shape(), dims);
  } else {
    logging::popops::debug("  {} = reduce({}{}), dims={}", fmap(out, getShape),
                           in.getVarStr(), in.shape(), dims);
  }

  // Decide the reduction types for each stage.
  ReductionTypes reductionTypes;
  auto useFloatAccum = (outputType == poplar::HALF &&
                        opBenefitsFromHigherIntermediatePrecision(params.op));
  auto accumType = useFloatAccum ? poplar::FLOAT : outputType;
  reductionTypes.interTile = accumType;
  reductionTypes.inVertex = accumType;

  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec reductionSpec{
      {"accumType.interTile",
       OptionHandler::createWithEnum(reductionTypes.interTile, accumTypeMap)},
      {"accumType.inVertex",
       OptionHandler::createWithEnum(reductionTypes.inVertex, accumTypeMap)}};

  for (const auto &entry : options) {
    reductionSpec.parse(entry.first, entry.second);
  }

  validateReductionParams(params);

  auto [canReduceWithMap_, reducedDims, outputShape, numOutputElements,
        numInputElements] = analyzeReduction(dims, in, out);
  if (canReduceWithMap_) {
    logging::popops::warn(
        "A reduction could be replaced with a simpler expression if using a "
        "program instead of a compute set");
  }

  // If there are no output elements... this is easy!
  // But we still need to produce an output Tensor if there isn't one.
  if (numOutputElements == 0) {
    logging::popops::debug("Empty output tensor");
    if (!out)
      out = graph.addVariable(outputType, outputShape, {dnai, "emptyOutput"});
    return;
  }

  // If the number of input elements is zero we definitely don't need
  // to do a reduction. In this case there are 0 elements in the input,
  // but some elements in the output. This is possible for example when
  // reducing a 10x10x0 tensor in the third dimension to 10x10. It's a bit
  // weird but it makes sense. This is how Tensorflow works.
  if (numInputElements == 0) {
    if (params.op == Operation::LOG_ADD) {
      throw poputil::poplibs_error(
          "Reduction operation LOG_ADD doesn't"
          " support reductions with zero input elements");
    }

    logging::popops::debug("zero input elements to reduction");

    if (out) {
      // If the output mapping isn't complete, just linearly map it.
      bool mappingComplete;
      graph.getTileMapping(out.get(), &mappingComplete);
      if (!mappingComplete) {
        logging::popops::warn(
            "reduceWithOutput was given an output without a complete "
            "mapping. Mapping it linearly instead.");
        poputil::mapTensorLinearly(graph, out.get());
      }
    } else {
      out = graph.addVariable(outputType, outputShape, {dnai});
      poputil::mapTensorLinearly(graph, out.get());
    }

    // If it's an update and there are no inputs the output won't change.
    if (params.update)
      return;

    float initVal = reductionInitialValue(params.op);

    if (css.empty())
      css.push_back(
          graph.addComputeSet({dnai, "ReductionOnEmptyInputsFillOutputCS"}));
    auto &fillCS = css.front();

    auto outFlat = out.get().flatten();
    graph.reorderToSimplify(&outFlat, {}, false);
    popops::fill(graph, outFlat, graph.getTileMapping(outFlat), fillCS,
                 initVal);
    return;
  }

  // At this point there is definitely some reducing to do. The tensor
  // dimensions are partitioned into 'reduced' and 'other' sets. The tensor
  // is dimshuffled so that the reduced dimensions are at the front, and then
  // it is flattened to 2D - one dimension for reducedDims, and one for
  // the otherDims.
  auto input2D = mangleTo2D(in, reducedDims);
  logging::popops::debug("Get 2D view of tensor for reduction: {}, {}",
                         input2D.dim(0), input2D.dim(1));

  // Do the 2D->1D reduction.
  reduceFirstDim2D(graph, input2D, out, outputShape, outputType, params,
                   reductionTypes, css, reductionResultTensors, reductionId,
                   {dnai});
  if (!withOutput) {
    logging::popops::debug("  out({}){}: {}", out.get().elementType(),
                           out.get().shape(), out.get().getDebugStr());
    logging::popops::debug("  {}{} = reduce({}{}), dims={}",
                           out.get().getVarStr(), out.get().shape(),
                           in.getVarStr(), in.shape(), dims);
  }
  logging::popops::debug("Reduce{} Op:{} End DebugStr: {}",
                         withOutput ? "WithOutput" : "", params.op,
                         dnai.getPathName());
}

// Same as reduceWithOutputCss except it takes a program instead of a compute
// set and it supports additional optimisations that are only possible with
// programs.
static void reduceWithOutputProg(Graph &graph, const Tensor &in,
                                 boost::optional<Tensor> &out,
                                 const poplar::Type &outputType,
                                 const std::vector<std::size_t> &dims,
                                 ReduceParams params, program::Sequence &prog,
                                 const DebugNameAndId &dnai,
                                 const poplar::OptionFlags &options) {
  if (canReduceWithMap(in, dims))
    reduceWithMap(graph, in, out, outputType, dims, params, prog, dnai,
                  options);
  else {
    std::vector<ComputeSet> css;
    ResultTensors reductionResultTensors;
    reduceWithOutputCss(graph, in, out, outputType, dims, params, css,
                        reductionResultTensors, 0, dnai, options);
    convertCssToProg(css, prog, reductionResultTensors, dnai);
  }
}
} // end anonymous namespace

void reduceWithOutput(Graph &graph, const Tensor &in, const Tensor &out_,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      std::vector<ComputeSet> &css,
                      const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(in, out_, dims, params, css, options));
  ResultTensors r;
  boost::optional<Tensor> out = out_;
  reduceWithOutputCss(graph, in, out, out_.elementType(), dims, params, css, r,
                      0, {di}, options);
}

void reduceWithOutput(Graph &graph, const Tensor &in, const Tensor &out_,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      program::Sequence &prog,
                      const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(in, out_, dims, params, options));

  boost::optional<Tensor> out = out_;
  reduceWithOutputProg(graph, in, out, out_.elementType(), dims, params, prog,
                       {di}, options);
}

void reduceMany(poplar::Graph &graph,
                const std::vector<SingleReduceOp> &reductions,
                std::vector<poplar::Tensor> &outputs,
                poplar::program::Sequence &prog,
                const poplar::DebugContext &debugContext,
                const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(reductions, options));

  const bool shouldCreateOutputs = outputs.empty();
  if (shouldCreateOutputs)
    outputs.resize(reductions.size());
  else if (outputs.size() != reductions.size())
    throw poputil::poplibs_error(
        "reduceMany: outputs must be the same size as reductions");

  ResultTensors reductionResultTensors;

  std::vector<ComputeSet> css;
  css.reserve(reductions.size());

  for (size_t i = 0; i < reductions.size(); ++i) {
    const SingleReduceOp &op = reductions[i];

    Type outputType;
    boost::optional<Tensor> optionalOut;
    if (shouldCreateOutputs) {
      if (op.params.update)
        throw poputil::poplibs_error(
            "reduceMany: outputs must be provided to do a "
            "reduction with the update flag set");
      outputType = op.useOutType ? op.outType : op.in.elementType();
    } else {
      outputType = outputs[i].elementType();
      optionalOut = std::move(outputs[i]);
    }

    std::string debugName =
        !op.debugName.empty() ? op.debugName : "reduction-" + std::to_string(i);
    DebugNameAndId dnai = {di, std::move(debugName)};

    if (!canReduceWithMap(op.in, op.dims))
      reduceWithOutputCss(graph, op.in, optionalOut, outputType, op.dims,
                          op.params, css, reductionResultTensors, i,
                          std::move(dnai), options);
    else {
      // Convert all the compute sets we've created so far into programs,
      // so that the simplified program can be added in the right order.
      convertCssToProg(css, prog, reductionResultTensors, di);
      css.clear();
      reductionResultTensors.typeA.clear();
      reductionResultTensors.typeB.clear();

      reduceWithMap(graph, op.in, optionalOut, outputType, op.dims, op.params,
                    prog, std::move(dnai), options);
    }

    outputs[i] = std::move(*optionalOut);
  }
  di.addOutputs(DI_ARGS(outputs));
  convertCssToProg(css, prog, reductionResultTensors, di);
}

Tensor reduce(Graph &graph, const Tensor &in,
              const std::vector<std::size_t> &dims, ReduceParams params,
              program::Sequence &prog, const poplar::DebugContext &debugContext,
              const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(in, dims, params, options));

  auto output =
      reduce(graph, in, in.elementType(), dims, params, prog, {di}, options);
  di.addOutput(output);
  return output;
}

Tensor reduce(Graph &graph, const Tensor &in,
              const std::vector<std::size_t> &dims, ReduceParams params,
              std::vector<ComputeSet> &css,
              const poplar::DebugContext &debugContext,
              const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(in, dims, params, css, options));

  auto output =
      reduce(graph, in, in.elementType(), dims, params, css, {di}, options);
  di.addOutput(output);
  return output;
}

Tensor reduce(Graph &graph, const Tensor &in, const poplar::Type &outType,
              const std::vector<std::size_t> &dims, ReduceParams params,
              std::vector<ComputeSet> &css,
              const poplar::DebugContext &debugContext,
              const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(in, outType, dims, params, css, options));

  if (params.update)
    throw poputil::poplibs_error("Cannot do an update using reduce(); "
                                 "call reduceWithOutput() instead.");
  ResultTensors r;
  boost::optional<Tensor> out;
  reduceWithOutputCss(graph, in, out, outType, dims, params, css, r, 0, {di},
                      options);
  auto output = out.get();
  di.addOutput(output);
  return output;
}

Tensor reduce(Graph &graph, const Tensor &in, const poplar::Type &outType,
              const std::vector<std::size_t> &dims, ReduceParams params,
              program::Sequence &prog, const poplar::DebugContext &debugContext,
              const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(in, outType, dims, params, options));

  if (params.update)
    throw poputil::poplibs_error("Cannot do an update using reduce(); "
                                 "call reduceWithOutput() instead.");
  boost::optional<Tensor> out;
  reduceWithOutputProg(graph, in, out, outType, dims, params, prog, {di},
                       options);
  auto output = out.get();
  di.addOutput(output);
  return output;
}

Tensor mangleTo2D(const Tensor &A, std::set<unsigned> &reducedDims) {

  // The set of dimensions that aren't reduced.
  std::set<unsigned> otherDims;
  for (unsigned i = 0; i < A.rank(); ++i) {
    if (reducedDims.count(i) == 0) {
      otherDims.insert(i);
    }
  }

  // How to permute the dimensions to get the reduced dimensions to the back.
  std::vector<unsigned> permutation;

  // Add the reduced dimensions, then the non-reduced ones.
  permutation.insert(permutation.end(), reducedDims.begin(), reducedDims.end());
  permutation.insert(permutation.end(), otherDims.begin(), otherDims.end());

  // Calculate the number of input elements to each output element.
  std::size_t reductionFactor = 1;
  for (auto i : reducedDims)
    reductionFactor *= A.dim(i);

  // And the number of parallel reductions.
  std::size_t numReductions = A.numElements() / reductionFactor;

  // The dimension of the flattened tensor.
  std::vector<std::size_t> flattenedShape = {reductionFactor, numReductions};

  return A.dimShuffle(permutation).reshape(flattenedShape);
}

} // namespace popops
