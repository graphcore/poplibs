// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "popnn/CTCInference.hpp"

#include "CTCInferencePlan.hpp"
#include "CTCPlanInternal.hpp"

#include <poplar/CSRFunctions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/Tracepoint.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/LogSoftmax.hpp>
#include <popops/Cast.hpp>
#include <poputil/OptionParsing.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <boost/optional.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace popops;
using namespace poputil;

template <unsigned size> using Slice = std::array<std::size_t, size>;
namespace {

void mapDataInputAccordingToPlan(Graph &graph, const Tensor &tensor,
                                 const popnn::ctc::InferencePlan &plan) {
  // Map the data input according to the plan, but the innermost dimension
  // isn't really compatible with the plan, as it is the number of classes
  // whereas we planned for the label length.
  // Choose to split the time dimension as much as possible over the combined
  // time and label partitions.  This avoids splitting the innermost dimension
  // which would result in increased exchange code size.
  const auto batchSize = tensor.dim(1);
  const unsigned timeSize = tensor.dim(2);
  const auto numClasses = tensor.dim(3);

  const auto numNonBatchPartitions = plan.parallel.time * plan.parallel.label;
  const auto remappedTimePartitions = std::min(numNonBatchPartitions, timeSize);
  const auto typeSize = graph.getTarget().getTypeSize(tensor.elementType());

  const auto timePartitionSize = [&]() {
    // Minimum result to map all the time slices onto the tiles within the plan
    // without splitting the innermost dimension
    auto minTimePartitionSize = ceildiv(timeSize, remappedTimePartitions);
    // Ensure that there are always a multiple of 4 bytes per tile to avoid
    // costly exchange.
    // Trialling timePartitionSize+0, +1, +2, +3 must produce a result divisible
    // by 4, as we will hit timePartitionSize+N as a multiple of 4 itself.
    for (unsigned i = 0; i < 4; i++) {
      const auto remainder = (typeSize * numClasses * minTimePartitionSize) % 4;
      if (remainder == 0) {
        break;
      }
      minTimePartitionSize++;
    }
    return minTimePartitionSize;
  }();
  assert((typeSize * timePartitionSize * numClasses) % 4 == 0);

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned time = 0; time < remappedTimePartitions; time++) {
      const auto timeBegin = time * timePartitionSize;
      if (timeBegin < timeSize) {
        auto tile = plan.getTile(batch, time);
        auto b = plan.partitionBatch(batchSize, batch);
        const auto timeEnd = std::min(timeSize, (time + 1) * timePartitionSize);

        graph.setTileMapping(tensor.slice({0, b.begin(), timeBegin, 0},
                                          {1, b.end(), timeEnd, numClasses}),
                             tile);
      }
    }
  }
}

Tensor toInternalShape(const Tensor &data) {
  // We are supplied a data input Tensor with shape
  // [maxTime, batchSize, numClasses].
  // Internally, data is ordered differently, and we will broadcast this data
  // according to the number of partitions made. So internally we use:
  // [partitions, batchSize,maxTime,  numClasses]
  // Here we have not yet broadcast so partitions = 1
  return data.dimShufflePartial({0}, {1}).expand({0});
}

Tensor toExternalShape(const Tensor &data) {
  // Return to the external shape.
  return data.dimShufflePartial({0}, {1});
}

void validateTensorTypes(const poplar::Tensor &data,
                         const poplar::Tensor &dataLengths,
                         const poplar::Type &partialsType,
                         const poplar::Type &outType) {
  if (data.elementType() != poplar::HALF &&
      data.elementType() != poplar::FLOAT) {
    throw poputil::poplibs_error("data tensor must be of type HALF or FLOAT");
  }
  if (dataLengths.elementType() != poplar::UNSIGNED_INT) {
    throw poputil::poplibs_error(
        "dataLengths tensor must be of type UNSIGNED_INT");
  }
  if (partialsType == poplar::HALF && data.elementType() == poplar::FLOAT) {
    throw poputil::poplibs_error(
        "partials type HALF unsupported with input tensor type FLOAT");
  }
  if (outType != poplar::HALF && outType != poplar::FLOAT) {
    throw poputil::poplibs_error("outType must be of type HALF or FLOAT");
  }
}

} // namespace
namespace popnn {
namespace ctc_infer {

poplar::Tensor createDataInput(poplar::Graph &graph, const poplar::Type &type,
                               const std::size_t batchSize,
                               const std::size_t maxTime,
                               const std::size_t numClasses,
                               const ctc::Plan &plan,
                               const poplar::DebugContext &debugContext) {
  const auto &inferPlan = plan.getImpl().getAsInferencePlan();
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(type, batchSize, maxTime, numClasses, plan));

  logging::popnn::debug("Creating data tensor for CTC beam search with Time:{}"
                        " Batches:{} Classes:{}",
                        maxTime, batchSize, numClasses);
  const auto data = graph.addVariable(type, {1, batchSize, maxTime, numClasses},
                                      {di, "data"});
  mapDataInputAccordingToPlan(graph, data, inferPlan);
  di.addOutput(data);
  return toExternalShape(data.squeeze({0}));
}

// beamSearchDecoderLogProbabilitiesImpl output tuple:
// outType  Tensor  labelProbs[batchSize, topPaths]
// unsigned Tensor  labelLengths[batchSize, topPaths]
// unsigned Tensor  decodedLabels[batchSize, topPaths, maxTime]
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
beamSearchDecoderLogProbabilitiesImpl(
    poplar::Graph &graph, const poplar::Type &outType,
    const poplar::Tensor &data, const poplar::Tensor &dataLengths,
    poplar::program::Sequence &prog, const unsigned blankClass,
    const unsigned beamWidth, const unsigned topPaths,
    const ctc::InferencePlan &plan, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options) {

  const auto partialsType = plan.params.partialsType;
  validateTensorTypes(data, dataLengths, partialsType, outType);
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di({debugContext, "CTCBeamSearchDecoder"},
                                 DI_ARGS(outType, data, dataLengths, blankClass,
                                         beamWidth, topPaths, plan, options));

  logging::popnn::debug("Disabled NANOO for CTC beam search decoder operation");
  poplar::FloatingPointBehaviour clear{false, false, false, false,
                                       true}; // Mask out nanoo
  poplar::FloatingPointBehaviour set{false, false, false, false, false};
  auto fpCSRToRestore =
      poplar::getAndModifyFloatingPointBehaviour(graph, prog, clear, set, di);

  logging::popnn::debug("Creating CTC beam search decoder using\n{}", plan);
  const auto maxT = data.dim(0);
  const auto batchSize = data.dim(1);

  // Reshape the input for internal use
  auto internalData = toInternalShape(data);

  // TODO - The actual beam search!

  // Create results with the intended shape and map
  // TODO - map according to plan etc, although not so critical as this is
  // all just used when the result is produced at the end
  auto labelProbs = graph.addVariable(outType, {batchSize, topPaths},
                                      {di, "labelProbabilities"});
  auto labelLengths = graph.addVariable(UNSIGNED_INT, {batchSize, topPaths},
                                        {di, "labelLengths"});
  auto decodedLabels = graph.addVariable(
      UNSIGNED_INT, {batchSize, topPaths, maxT}, {di, "decodedLabels"});
  graph.setTileMapping(labelProbs, 0);
  graph.setTileMapping(labelLengths, 0);
  graph.setTileMapping(decodedLabels, 0);

  auto labelProbsOut = [&]() {
    if (partialsType != outType) {
      poplar::DebugContext castDebug{di, "Cast"};
      auto castCS = graph.addComputeSet(castDebug);
      auto probs = popops::cast(graph, labelProbs, outType, castCS, castDebug);
      prog.add(Execute(castCS, castDebug));
      return probs;
    } else {
      return labelProbs;
    };
  }();

  di.addOutputs({{"labelProbs", poputil::toProfileValue(labelProbsOut)},
                 {"labelLengths", poputil::toProfileValue(labelLengths)},
                 {"decodedLabels", poputil::toProfileValue(labelLengths)}});

  poplar::setFloatingPointBehaviour(graph, prog, fpCSRToRestore, di);
  return {labelProbs, labelLengths, decodedLabels};
}

void printOp(std::string name, const poplar::Type &partialsType,
             const poplar::Type &outType, const poplar::Tensor &data,
             const poplar::Tensor &dataLengths, const unsigned blankClass,
             const unsigned beamWidth, const unsigned topPaths,
             const poplar::DebugContext &debugContext) {
  const auto inType = data.elementType();
  logging::popnn::info("{} data={}, dataLengths={}, "
                       "blankClass={}, beamwidth={}, topPaths={}, inType={}, "
                       "partialsType={}, outType={}, name={}",
                       name, data.shape(), dataLengths.shape(), blankClass,
                       beamWidth, topPaths, inType, partialsType, outType,
                       debugContext.getPathName());
}
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
beamSearchDecoderLogProbabilities(poplar::Graph &graph,
                                  const poplar::Tensor &logProbs,
                                  const poplar::Tensor &dataLengths,
                                  poplar::program::Sequence &prog,
                                  unsigned blankClass, unsigned beamwidth,
                                  unsigned topPaths, const ctc::Plan &plan,
                                  const poplar::DebugContext &debugContext,
                                  const poplar::OptionFlags &options) {

  const auto &inferPlan = plan.getImpl().getAsInferencePlan();
  const auto partialsType = inferPlan.params.partialsType;
  const auto outType = inferPlan.params.outType;
  printOp("CTCBeamSearchDecoderLogProbs", partialsType, outType, logProbs,
          dataLengths, blankClass, beamwidth, topPaths, debugContext);

  return beamSearchDecoderLogProbabilitiesImpl(
      graph, outType, logProbs, dataLengths, prog, blankClass, beamwidth,
      topPaths, inferPlan, {debugContext, "CTCBeamSearchDecoderLogProbs"},
      options);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
beamSearchDecoderLogits(poplar::Graph &graph, const poplar::Tensor &logits,
                        const poplar::Tensor &dataLengths,
                        poplar::program::Sequence &prog, unsigned blankClass,
                        unsigned beamwidth, unsigned topPaths,
                        const ctc::Plan &plan,
                        const poplar::DebugContext &parentDebugContext,
                        const poplar::OptionFlags &options) {

  const auto &inferPlan = plan.getImpl().getAsInferencePlan();
  const auto partialsType = inferPlan.params.partialsType;
  const auto outType = inferPlan.params.outType;
  printOp("CTCBeamSearchDecoderLogits", partialsType, outType, logits,
          dataLengths, blankClass, beamwidth, topPaths, parentDebugContext);
  poplar::DebugContext debugContext{parentDebugContext,
                                    "CTCBeamSearchDecoderLogits"};

  // Ensure we preserve mapping of the result to fit in with the plan
  auto logProbs = graph.clone(logits, debugContext);
  prog.add(Copy(logits, logProbs, false, debugContext));
  logSoftmaxInPlace(graph, logProbs, prog, debugContext);

  return beamSearchDecoderLogProbabilitiesImpl(
      graph, outType, logProbs, dataLengths, prog, blankClass, beamwidth,
      topPaths, inferPlan, debugContext, options);
}

} // namespace ctc_infer

} // end namespace popnn
