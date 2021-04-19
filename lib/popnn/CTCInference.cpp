// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "popnn/CTCInference.hpp"

#include "CTCInferenceConnection.hpp"
#include "CTCInferencePlan.hpp"
#include "CTCPlanInternal.hpp"

#include <poplar/CSRFunctions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <poplibs_support/CTCInferenceDefs.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/Tracepoint.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/LogSoftmax.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <poputil/OptionParsing.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <boost/optional.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace popops;
using namespace popops::expr;
using namespace poputil;

template <unsigned size> using Slice = std::array<std::size_t, size>;
enum class PartitionType { BATCH_ENTRY, COPY, EXTEND, MERGE, OUTPUT };

namespace {

using TempTensors = popnn::ctc_infer::TempTensors;
using BeamTensors = popnn::ctc_infer::BeamTensors;
static constexpr auto voidSymbol = popnn::ctc_infer::voidSymbol;

void mapDataInputAccordingToPlan(Graph &graph, const Tensor &tensor,
                                 const popnn::ctc::InferencePlan &plan) {
  // Map the data input according to the plan, but the innermost dimension
  // isn't really compatible with the plan, as it is the number of classes
  // whereas we planned for batchEntry partitions.
  // Choose to split the time dimension as much as possible over the combined
  // time and batchEntry partitions.  This avoids splitting the innermost
  // dimension which would result in increased exchange code size.
  const auto batchSize = tensor.dim(0);
  const unsigned timeSize = tensor.dim(2);
  const auto numClasses = tensor.dim(3);

  const auto numNonBatchPartitions =
      plan.parallel.time * plan.batchEntryPartitions();
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

        graph.setTileMapping(tensor.slice({b.begin(), 0, timeBegin, 0},
                                          {b.end(), 1, timeEnd, numClasses}),
                             tile);
      }
    }
  }
}

inline poplar::Interval makePartition(unsigned size, unsigned index,
                                      PartitionType partitionType,
                                      const popnn::ctc::InferencePlan &plan) {
  switch (partitionType) {
  case PartitionType::BATCH_ENTRY:
    return plan.partitionBatchEntry(size, index);
  case PartitionType::MERGE:
    return plan.partitionMerge(size, index);
  case PartitionType::EXTEND:
    return plan.partitionExtend(size, index);
  case PartitionType::COPY:
    return plan.partitionCopy(size, index);
  default:
    return plan.partitionOutput(size, index);
  };
}

void mapAccordingToPlanRank3(Graph &graph, const Tensor &tensor,
                             PartitionType partitionType,
                             const popnn::ctc::InferencePlan &plan) {
  // Map any rank 3 tensors used in this process to the correct tiles according
  // to the plan.
  const auto batchSize = tensor.dim(0);
  const auto totalPartitions = tensor.dim(1);
  const auto innermostDimSize = tensor.dim(2);

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned partition = 0; partition < totalPartitions; partition++) {
      auto tile = plan.getTile(batch, 0, partition);
      auto b = plan.partitionBatch(batchSize, batch);
      auto p = makePartition(totalPartitions, partition, partitionType, plan);
      graph.setTileMapping(tensor.slice({b.begin(), p.begin(), 0},
                                        {b.end(), p.end(), innermostDimSize}),
                           tile);
    }
  }
}
void mapAccordingToPlanRank4(Graph &graph, const Tensor &tensor,
                             PartitionType partitionType,
                             const popnn::ctc::InferencePlan &plan) {
  // Map any rank 4 tensors used in this process to the correct tiles according
  // to the plan
  const auto batchSize = tensor.dim(0);
  const auto totalPartitions = tensor.dim(1);
  const auto timeSize = tensor.dim(2);
  const auto innermostDimSize = tensor.dim(3);

  for (unsigned partition = 0; partition < plan.batchEntryPartitions();
       partition++) {
    for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
      for (unsigned time = 0; time < plan.parallel.time; time++) {

        auto tile = plan.getTile(batch, time, partition);
        auto b = plan.partitionBatch(batchSize, batch);
        auto p = makePartition(totalPartitions, partition, partitionType, plan);
        auto t = plan.partitionTime(timeSize, time);
        graph.setTileMapping(
            tensor.slice({b.begin(), p.begin(), t.begin(), 0},
                         {b.end(), p.end(), t.end(), innermostDimSize}),
            tile);
      }
    }
  }
}

void mapAccordingToPlan(Graph &graph, const Tensor &tensor,
                        PartitionType partitionType,
                        const popnn::ctc::InferencePlan &plan) {
  assert(tensor.rank() == 3 || tensor.rank() == 4);
  if (tensor.rank() == 3) {
    mapAccordingToPlanRank3(graph, tensor, partitionType, plan);
  } else if (tensor.rank() == 4) {
    mapAccordingToPlanRank4(graph, tensor, partitionType, plan);
  }
}

TempTensors createAndInitialiseTemporaryTensors(
    Graph &graph, const Tensor &dataLengths,
    const popnn::ctc::InferencePlan &plan, unsigned numClasses,
    unsigned batchSize, unsigned beamwidth, const Type &partialsType,
    Sequence &prog, const poplar::DebugContext &di) {
  TempTensors tempTensors;
  // Make a counter per tile for the vertices to use
  // Note - making these unsigned short would mean the vertex has to do a
  // subword write.  This slows it down, but more importantly when plans put
  // 2 vertices on the same tile will result in poplar running the vertices in
  // series to avoid the subword writes of 2 vertices clashing.  When using
  // unsigned this won't be an issue

  tempTensors.currentTimestep = graph.addVariable(
      UNSIGNED_INT, {batchSize, plan.batchEntryPartitions(), 1},
      {di, "currentTimestep"});
  mapAccordingToPlan(graph, tempTensors.currentTimestep,
                     PartitionType::BATCH_ENTRY, plan);
  // Initialise the count.  The first timestep is 1, with timestep 0 being an
  // initial state
  auto initialiserZero = graph.addConstant<unsigned>(UNSIGNED_INT, {1}, 1u, di);
  graph.setTileMapping(initialiserZero, 0);
  prog.add(Copy(
      initialiserZero.broadcast(tempTensors.currentTimestep.numElements(), 0),
      tempTensors.currentTimestep.flatten(), false, di));

  // Data length tensor broadcast per tile
  tempTensors.dataLengths =
      graph.clone(tempTensors.currentTimestep, {di, "timesteps"});
  auto dataLengthsBroadcast =
      dataLengths.expand({1, 1}).broadcast(plan.batchEntryPartitions(), 1);
  prog.add(Copy(dataLengthsBroadcast, tempTensors.dataLengths));

  // Extend candidates
  const std::vector<std::size_t> extendCandidateShape = {
      batchSize, numClasses - 1, beamwidth};
  tempTensors.extendCandidatesPb = graph.addVariable(
      partialsType, extendCandidateShape, {di, "extendCandidatesPb"});
  tempTensors.extendCandidatesPnb = graph.addVariable(
      partialsType, extendCandidateShape, {di, "extendCandidatesPnb"});
  tempTensors.extendCandidatesPTotal = graph.addVariable(
      partialsType, extendCandidateShape, {di, "extendCandidatesPTotal"});

  tempTensors.extendCandidatesParent = graph.addVariable(
      UNSIGNED_INT, extendCandidateShape, {di, "extendCandidatesParents"});
  tempTensors.extendCandidatesAddend = graph.addVariable(
      UNSIGNED_INT, extendCandidateShape, {di, "extendCandidatesAddends"});

  mapAccordingToPlan(graph, tempTensors.extendCandidatesPb,
                     PartitionType::EXTEND, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesPnb,
                     PartitionType::EXTEND, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesPTotal,
                     PartitionType::EXTEND, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesParent,
                     PartitionType::EXTEND, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesAddend,
                     PartitionType::EXTEND, plan);

  // Copy candidates
  const std::vector<std::size_t> copyCandidateShape = {batchSize, beamwidth, 1};
  tempTensors.copyCandidatesPb = graph.addVariable(
      partialsType, copyCandidateShape, {di, "copyCandidatesPb"});
  tempTensors.copyCandidatesPnb = graph.addVariable(
      partialsType, copyCandidateShape, {di, "copyCandidatesPnb"});
  tempTensors.copyCandidatesPTotal = graph.addVariable(
      partialsType, copyCandidateShape, {di, "copyCandidatesPTotal"});

  tempTensors.copyCandidatesParent = graph.addVariable(
      UNSIGNED_INT, copyCandidateShape, {di, "copyCandidatesParent"});
  tempTensors.copyCandidatesAddend = graph.addVariable(
      UNSIGNED_INT, copyCandidateShape, {di, "copyCandidatesAddend"});

  mapAccordingToPlan(graph, tempTensors.copyCandidatesPb, PartitionType::COPY,
                     plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesPnb, PartitionType::COPY,
                     plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesPTotal,
                     PartitionType::COPY, plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesParent,
                     PartitionType::COPY, plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesAddend,
                     PartitionType::COPY, plan);

  // Merge merge candidates vectors of tensors
  tempTensors.mergeCandidatesPb.resize(beamwidth);
  tempTensors.mergeCandidatesPnb.resize(beamwidth);
  tempTensors.mergeCandidatesPTotal.resize(beamwidth);
  tempTensors.mergeCandidatesParent.resize(beamwidth);
  tempTensors.mergeCandidatesAddend.resize(beamwidth);

  const std::vector<size_t> mergeTensorsShape = {batchSize, plan.parallel.merge,
                                                 1};
  for (unsigned i = 0; i < beamwidth; i++) {

    const auto debugStr = std::to_string(i);
    tempTensors.mergeCandidatesPb[i] = graph.addVariable(
        partialsType, mergeTensorsShape, {di, "mergeCandidatesPb_" + debugStr});
    tempTensors.mergeCandidatesPnb[i] =
        graph.addVariable(partialsType, mergeTensorsShape,
                          {di, "mergeCandidatesPnb_" + debugStr});
    tempTensors.mergeCandidatesPTotal[i] =
        graph.addVariable(partialsType, mergeTensorsShape,
                          {di, "mergeCandidatesPTotal_" + debugStr});

    tempTensors.mergeCandidatesParent[i] =
        graph.addVariable(UNSIGNED_INT, mergeTensorsShape,
                          {di, "mergeCandidatesParent_" + debugStr});
    tempTensors.mergeCandidatesAddend[i] =
        graph.addVariable(UNSIGNED_INT, mergeTensorsShape,
                          {di, "mergeCandidatesAddend_" + debugStr});

    mapAccordingToPlan(graph, tempTensors.mergeCandidatesPb[i],
                       PartitionType::MERGE, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesPnb[i],
                       PartitionType::MERGE, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesPTotal[i],
                       PartitionType::MERGE, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesParent[i],
                       PartitionType::MERGE, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesAddend[i],
                       PartitionType::MERGE, plan);
  }

  return tempTensors;
}

BeamTensors createAndInitialiseBeamTensors(
    Graph &graph, const popnn::ctc::InferencePlan &plan, unsigned batchSize,
    unsigned maxT, unsigned beamwidth, const Type &partialsType, Sequence &prog,
    const poplar::DebugContext &di) {
  BeamTensors beamTensors;

  // Include an additional time step so that we can make a helpful initial
  // state
  const std::vector<std::size_t> beamHistoryShape = {
      batchSize, plan.batchEntryPartitions(), maxT + 1, beamwidth};
  beamTensors.parent =
      graph.addVariable(UNSIGNED_INT, beamHistoryShape, {di, "beamParent"});
  beamTensors.addend =
      graph.addVariable(UNSIGNED_INT, beamHistoryShape, {di, "beamAddend"});

  const std::vector<std::size_t> beamProbsShape = {
      batchSize, plan.batchEntryPartitions(), beamwidth, 1};
  beamTensors.pb =
      graph.addVariable(partialsType, beamProbsShape, {di, "beamPb"});
  beamTensors.pnb =
      graph.addVariable(partialsType, beamProbsShape, {di, "beamPnb"});
  beamTensors.lastOutput =
      graph.addVariable(UNSIGNED_INT, beamProbsShape, {di, "beamLastOutput"});
  beamTensors.previousLastOutput = graph.addVariable(
      UNSIGNED_INT, beamProbsShape, {di, "previousBeamLastOutput"});

  mapAccordingToPlan(graph, beamTensors.parent, PartitionType::BATCH_ENTRY,
                     plan);
  mapAccordingToPlan(graph, beamTensors.addend, PartitionType::BATCH_ENTRY,
                     plan);

  mapAccordingToPlan(graph, beamTensors.pb, PartitionType::BATCH_ENTRY, plan);
  mapAccordingToPlan(graph, beamTensors.pnb, PartitionType::BATCH_ENTRY, plan);
  mapAccordingToPlan(graph, beamTensors.lastOutput, PartitionType::BATCH_ENTRY,
                     plan);
  mapAccordingToPlan(graph, beamTensors.previousLastOutput,
                     PartitionType::BATCH_ENTRY, plan);

  // Initialise the beam probabilities, with only one origin point
  auto initialiserProbZero =
      graph.addConstant<float>(partialsType, {1}, log::probabilityZero, di);
  graph.setTileMapping(initialiserProbZero, 0);
  prog.add(Copy(initialiserProbZero.broadcast(beamTensors.pb.numElements(), 0),
                beamTensors.pb.flatten(), false, di));
  prog.add(Copy(initialiserProbZero.broadcast(beamTensors.pnb.numElements(), 0),
                beamTensors.pnb.flatten(), false, di));

  auto initialiserProbOne =
      graph.addConstant<float>(partialsType, {1}, log::probabilityOne, di);
  graph.setTileMapping(initialiserProbOne, 0);
  auto pnbSlice = beamTensors.pnb.slice(
      {0, 0, 0, 0}, {batchSize, plan.batchEntryPartitions(), 1, 1});
  prog.add(Copy(initialiserProbOne.broadcast(pnbSlice.numElements(), 0),
                pnbSlice.flatten(), false, di));

  // last symbol = voidSymbol initial state
  auto initialiserVoidSymbol =
      graph.addConstant(UNSIGNED_INT, {1}, voidSymbol, di);
  graph.setTileMapping(initialiserVoidSymbol, 0);
  prog.add(Copy(
      initialiserVoidSymbol.broadcast(beamTensors.lastOutput.numElements(), 0),
      beamTensors.lastOutput.flatten(), false, di));

  // Setup the initial beam history with a zero time slice with:
  // beam   parent addend
  // 0      0      voidSymbol      (This is valid, but void)
  // 1      1      voidSymbol-1    (These are invalid symbols and won't merge)
  // 2      2      voidSymbol-2
  // ...
  const auto addendSlice = beamTensors.addend.slice(0, 1, 2).slice(0, 1, 3);
  prog.add(Copy(initialiserVoidSymbol.broadcast(addendSlice.numElements(), 0),
                addendSlice.flatten()));

  const auto numBeamsInstances = batchSize * plan.batchEntryPartitions();
  const auto parentZeroTimeSlice =
      beamTensors.parent.slice(0, 1, 2).reshapePartial(0, 2,
                                                       {numBeamsInstances, 1});
  const auto addendZeroTimeSlice =
      beamTensors.addend.slice(0, 1, 2)
          .slice(1, beamwidth, 3)
          .reshapePartial(0, 2, {numBeamsInstances, 1});
  for (unsigned i = 0; i < numBeamsInstances; i++) {
    iota(graph, parentZeroTimeSlice.slice(i, i + 1, 0).flatten(), 0u, prog, di);
    iota(graph, addendZeroTimeSlice.slice(i, i + 1, 0).flatten(),
         voidSymbol - beamwidth, prog, di);
  }
  return beamTensors;
}

Sequence createLoopBodyProg(Graph &graph, const popnn::ctc::InferencePlan &plan,
                            const Tensor &data, const BeamTensors &beams,
                            const TempTensors &tempTensors, unsigned blankClass,
                            unsigned beamwidth,
                            const poplar::DebugContext &di) {

  const auto maxT = data.dim(2);
  const auto numClasses = data.dim(3);
  const auto numClassesM1 = numClasses - 1;
  Sequence prog;

  // Generate candidates in the 1st compute set
  auto cs1 = graph.addComputeSet(di);
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    // Extend candidates
    for (unsigned c = 0; c < plan.parallel.extend; c++) {
      const unsigned addendClass = c >= blankClass ? c + 1 : c;
      const unsigned tile = plan.getTile(batch, 0, c);
      for (unsigned v = 0; v < plan.parallel.extendVerticesPerPartition; v++) {
        const auto beamPartition = plan.partitionExtendVertices(beamwidth, v);
        generateExtendCandidateVertex(
            graph, data, beams, tempTensors, cs1, batch, {0, maxT}, c,
            blankClass, beamwidth, beamPartition, addendClass, tile);
      }
    }
    // Copy candidates
    for (unsigned c = 0; c < plan.parallel.copy; c++) {
      const unsigned tile = plan.getTile(batch, 0, c);
      generateCopyCandidateVertex(graph, data, beams, tempTensors, cs1, batch,
                                  {0, maxT}, c, blankClass, beamwidth, tile);
    }
  }
  prog.add(Execute(cs1, di));

  // Broadcast the copy candidates into the vectors of merge candidates
  auto transform = [=](const Tensor &in, unsigned i) {
    return in.slice(i, i + 1, 1).expand({1}).broadcast(plan.parallel.merge, 1);
  };

  for (unsigned i = 0; i < beamwidth; i++) {
    prog.add(Copy(transform(tempTensors.copyCandidatesParent, i),
                  tempTensors.mergeCandidatesParent[i]));
    prog.add(Copy(transform(tempTensors.copyCandidatesAddend, i),
                  tempTensors.mergeCandidatesAddend[i]));
    prog.add(Copy(transform(tempTensors.copyCandidatesPb, i),
                  tempTensors.mergeCandidatesPb[i]));
    prog.add(Copy(transform(tempTensors.copyCandidatesPnb, i),
                  tempTensors.mergeCandidatesPnb[i]));
    prog.add(Copy(transform(tempTensors.copyCandidatesPTotal, i),
                  tempTensors.mergeCandidatesPTotal[i]));
  }

  // Merge candidates in the 2nd compute set
  auto cs2 = graph.addComputeSet(di);
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned copy = 0; copy < plan.parallel.copy; copy++) {
      for (unsigned merge = 0; merge < plan.parallel.merge; merge++) {
        const unsigned tile = plan.getTile(batch, 0, merge);
        mergeCandidateVertex(graph, beams, tempTensors, cs2, batch, {0, maxT},
                             merge, copy, blankClass, beamwidth, tile);
      }
    }
  }
  prog.add(Execute(cs2, di));

  // Select the merged copy candidates, and zero the probability of merged
  // extend candidates in the 3rd compute set
  auto cs3 = graph.addComputeSet(di);
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned copy = 0; copy < plan.parallel.preSelectCopy; copy++) {
      const unsigned tile = plan.getTile(batch, 0, copy);
      selectCopyCandidateVertex(graph, tempTensors, cs3, batch, copy, beamwidth,
                                tile);
    }
    for (unsigned extend = 0; extend < plan.parallel.preSelectExtend;
         extend++) {
      const unsigned tile = plan.getTile(batch, 0, extend);
      selectExtendCandidateVertex(graph, tempTensors, cs3, batch, extend,
                                  beamwidth, blankClass, tile);
    }
  }
  prog.add(Execute(cs3, di));

  // Select candidates in the 4th compute set
  // TODO - make some of the sorting work more in parallel
  auto cs4 = graph.addComputeSet(di);
  const unsigned candidatesToCompare =
      plan.parallel.copy + plan.parallel.copy * numClassesM1;
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned select = 0; select < plan.parallel.select; select++) {
      const unsigned tile = plan.getTile(batch, 0, select);
      selectCandidatesVertex(graph, tempTensors, cs4, batch, 0,
                             candidatesToCompare, beamwidth, tile);
    }
  }
  prog.add(Execute(cs4, di));
  // Prepare the copy of the last output for the Update stage
  prog.add(Copy(beams.lastOutput, beams.previousLastOutput));

  // Update beam history and probabilities in the 5th compute set
  auto cs5 = graph.addComputeSet(di);
  const unsigned sortedResultOffset = plan.parallel.copy * (beamwidth - 1);
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned beam = 0; beam < plan.batchEntryPartitions(); beam++) {
      const unsigned tile = plan.getTile(batch, 0, beam);
      updateVertex(graph, beams, tempTensors, cs5, batch, {0, maxT}, beam,
                   sortedResultOffset, beamwidth, tile);
    }
  }
  prog.add(Execute(cs5, di));

  return prog;
}

Tensor toInternalShape(const Tensor &data) {
  // We are supplied a data input Tensor with shape
  // [maxTime, batchSize, numClasses].
  // Internally, data is ordered differently, and we will broadcast this data
  // according to the number of partitions made. So internally we use:
  // [batchSize, batchEntryPartitions, maxTime,  numClasses]
  // Here we have not yet broadcast so batchEntryPartitions = 1
  return data.dimShufflePartial({0}, {1}).expand({1});
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
  const auto data = graph.addVariable(type, {batchSize, 1, maxTime, numClasses},
                                      {di, "data"});
  mapDataInputAccordingToPlan(graph, data, inferPlan);
  di.addOutput(data);
  return toExternalShape(data.squeeze({1}));
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
    const unsigned beamwidth, const unsigned topPaths,
    const ctc::InferencePlan &plan, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options) {

  const auto partialsType = plan.params.partialsType;
  validateTensorTypes(data, dataLengths, partialsType, outType);
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di({debugContext, "CTCBeamSearchDecoder"},
                                 DI_ARGS(outType, data, dataLengths, blankClass,
                                         beamwidth, topPaths, plan, options));

  logging::popnn::debug("Disabled NANOO for CTC beam search decoder operation");
  poplar::FloatingPointBehaviour clear{false, false, false, false,
                                       true}; // Mask out nanoo
  poplar::FloatingPointBehaviour set{false, false, false, false, false};
  auto fpCSRToRestore =
      poplar::getAndModifyFloatingPointBehaviour(graph, prog, clear, set, di);

  logging::popnn::debug("Creating CTC beam search decoder using\n{}", plan);
  const auto maxT = data.dim(0);
  const auto batchSize = data.dim(1);
  const auto numClasses = data.dim(2);

  // Reshape the input for internal use
  auto internalData = toInternalShape(data);
  auto internalDataShape = internalData.shape();
  internalDataShape[1] = plan.batchEntryPartitions();

  // Broadcast the data input and map according to the planned totalPartitions
  // which require a copy of the data while computing
  const auto workingData = [&]() {
    if (plan.batchEntryPartitions() != 1) {
      auto result = graph.addVariable(data.elementType(), internalDataShape,
                                      {di, "broadcastInput"});
      mapAccordingToPlan(graph, result, PartitionType::BATCH_ENTRY, plan);
      auto broadcastData =
          internalData.broadcast(plan.batchEntryPartitions(), 1);
      prog.add(Copy(broadcastData, result, false, di));
      return result;
    } else {
      // No broadcast/copy to do
      return internalData;
    }
  }();

  // Make the beam history tensors, setting only 1 beam to probablity = 1
  // and all outputs = voidSymbol
  auto beams = createAndInitialiseBeamTensors(
      graph, plan, batchSize, maxT, beamwidth, partialsType, prog, di);

  // Make the temporary tensors, initialising only the count
  auto tempTensors = createAndInitialiseTemporaryTensors(
      graph, dataLengths, plan, numClasses, batchSize, beamwidth, partialsType,
      prog, di);

  // Make the loop body and run it
  auto loopBody =
      createLoopBodyProg(graph, plan, workingData, beams, tempTensors,
                         blankClass, beamwidth, {di, "beamSearchDecoderLoop"});
  prog.add(Repeat(maxT, loopBody, di));

  // Create results and map, inserting a 3rd dimension so mapping functions can
  // be used
  // TODO - More flexible mapping functions would remove the need for this
  auto labelProbs = graph.addVariable(outType, {batchSize, topPaths, 1},
                                      {di, "labelProbabilities"});
  auto labelLengths = graph.addVariable(UNSIGNED_INT, {batchSize, topPaths, 1},
                                        {di, "labelLengths"});
  auto decodedLabels = graph.addVariable(
      UNSIGNED_INT, {batchSize, topPaths, maxT}, {di, "decodedLabels"});

  mapAccordingToPlan(graph, labelProbs, PartitionType::OUTPUT, plan);
  mapAccordingToPlan(graph, labelLengths, PartitionType::OUTPUT, plan);
  mapAccordingToPlan(graph, decodedLabels, PartitionType::OUTPUT, plan);
  // Remove the 3rd dimension that was inserted
  labelProbs.squeeze({2});
  labelLengths.squeeze({2});

  auto outputCS = graph.addComputeSet({di, "output"});
  for (unsigned batch = 0; batch < batchSize; batch++) {
    for (unsigned path = 0; path < topPaths; path++) {
      unsigned tile = plan.getTile(batch, 0, path);
      unsigned partition = path;
      generateOutputVertex(graph, beams, tempTensors, decodedLabels,
                           labelLengths, outputCS, batch, beamwidth, partition,
                           path, tile);
    }
  }
  prog.add(Execute(outputCS, di));
  // Combine probabilities for output (Log add of pb,pnb)
  auto pb = beams.pb.slice(0, 1, 1);
  auto pnb = beams.pnb.slice(0, 1, 1);
  // Implement a log add using elementwise operations,
  // when doing a logAdd(a,b) , we use min = min(a,b), max = max (a,b)
  // result =  exp( max + log(1 + exp(min - max)))
  auto plusOne = graph.addConstant<float>(partialsType, {1}, 1.0, di);
  graph.setTileMapping(plusOne, 0);
  auto max = popops::map(graph, Max(_1, _2), {pb, pnb}, prog, di);

  popops::mapInPlace(graph, _3 + Log(_4 + Exp(Min(_1, _2) - _3)),
                     {pb, pnb, max, plusOne}, prog, di);

  labelProbs = pb;

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
  return {labelProbsOut, labelLengths, decodedLabels};
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
