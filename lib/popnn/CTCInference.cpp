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
#include <popops/Reduce.hpp>
#include <poputil/Loop.hpp>
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
enum class PartitionType {
  BATCH,
  BATCH_ENTRY,
  COPY,
  EXTEND,
  MERGE,
  SELECT_COPY,
  SELECT_EXTEND,
  SORT,
  SORT_REDUCE,
  OUTPUT
};

namespace {

using TempTensors = popnn::ctc_infer::TempTensors;
using BeamTensors = popnn::ctc_infer::BeamTensors;
static constexpr auto voidSymbol = popnn::ctc_infer::voidSymbol;
static constexpr auto invalidSymbol = popnn::ctc_infer::invalidSymbol;

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

using PartitionFn = Interval (popnn::ctc::InferencePlan::*)(unsigned,
                                                            unsigned) const;

PartitionFn getPartitionFunction(PartitionType type) {
  switch (type) {
  case PartitionType::BATCH:
    return &popnn::ctc::InferencePlan::partitionBatch;
  case PartitionType::BATCH_ENTRY:
    return &popnn::ctc::InferencePlan::partitionBatchEntry;
  case PartitionType::COPY:
    return &popnn::ctc::InferencePlan::partitionCopy;
  case PartitionType::EXTEND:
    return &popnn::ctc::InferencePlan::partitionExtend;
  case PartitionType::MERGE:
    return &popnn::ctc::InferencePlan::partitionMerge;
  case PartitionType::SELECT_COPY:
    return &popnn::ctc::InferencePlan::partitionSelectCopy;
  case PartitionType::SELECT_EXTEND:
    return &popnn::ctc::InferencePlan::partitionSelectExtend;
  case PartitionType::SORT:
    return &popnn::ctc::InferencePlan::partitionSort;
  case PartitionType::SORT_REDUCE:
    return &popnn::ctc::InferencePlan::partitionSortReduce;
  case PartitionType::OUTPUT:
    return &popnn::ctc::InferencePlan::partitionOutput;
  };
  POPLIB_UNREACHABLE();
}

void mapAccordingToPlan(Graph &graph, const Tensor &tensor,
                        PartitionType partitionType, unsigned totalPartitions,
                        const popnn::ctc::InferencePlan &plan) {
  // Map any rank 3 tensors used in this process to the correct tiles according
  // to the plan.
  assert(tensor.rank() == 3);

  const auto batchSize = tensor.dim(0);
  const auto partitionedDimSize = tensor.dim(1);
  const auto innermostDimSize = tensor.dim(2);
  const auto partitionFunction = getPartitionFunction(partitionType);

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned partition = 0; partition < totalPartitions; partition++) {
      auto tile = plan.getTile(batch, 0, partition);
      auto b = plan.partitionBatch(batchSize, batch);
      auto p = (plan.*partitionFunction)(partitionedDimSize, partition);
      graph.setTileMapping(tensor.slice({b.begin(), p.begin(), 0},
                                        {b.end(), p.end(), innermostDimSize}),
                           tile);
    }
  }
}

void mapAccordingToPlan(Graph &graph, const Tensor &tensor,
                        PartitionType partitionType,
                        const popnn::ctc::InferencePlan &plan) {
  // Map any rank 4 tensors used in this process to the correct tiles according
  // to the plan
  assert(tensor.rank() == 4);

  const auto batchSize = tensor.dim(0);
  const auto totalPartitions = tensor.dim(1);
  const auto timeSize = tensor.dim(2);
  const auto innermostDimSize = tensor.dim(3);
  const auto partitionFunction = getPartitionFunction(partitionType);

  for (unsigned partition = 0; partition < plan.batchEntryPartitions();
       partition++) {
    for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
      for (unsigned time = 0; time < plan.parallel.time; time++) {

        auto tile = plan.getTile(batch, time, partition);
        auto b = plan.partitionBatch(batchSize, batch);
        auto p = (plan.*partitionFunction)(totalPartitions, partition);
        auto t = plan.partitionTime(timeSize, time);
        graph.setTileMapping(
            tensor.slice({b.begin(), p.begin(), t.begin(), 0},
                         {b.end(), p.end(), t.end(), innermostDimSize}),
            tile);
      }
    }
  }
}

TempTensors createAndInitialiseTemporaryTensors(
    Graph &graph, const Tensor &dataLengths,
    const popnn::ctc::InferencePlan &plan, unsigned numClasses,
    unsigned batchSize, unsigned beamwidth, const Type &partialsType,
    Sequence &prog, const poplar::DebugContext &di) {

  TempTensors tempTensors;

  // Find the maximum time for any of the inputs in this batch so that we can
  // stop early if possible
  // Reduce doesn't support unsigned int, only int
  auto dataLengthsInt = popops::cast(graph, dataLengths, INT, prog, di);
  auto maxTimeInBatch = popops::reduce(graph, dataLengthsInt.flatten(), {0},
                                       {popops::Operation::MAX}, prog, di);
  // Cast back and add 1 as we have a dummy 1st timestep and so will count
  // 1,2,3,...maxTimeInBatch   (Loop limit = maxTimeInbatch+1)
  tempTensors.maxTimeInBatch =
      popops::map(graph, Cast(Add(_1, Const(1u)), UNSIGNED_INT),
                  {maxTimeInBatch}, prog, di);
  graph.setTileMapping(tempTensors.maxTimeInBatch, 0);

  // Add a loop variable timestep
  tempTensors.loopTimestep =
      graph.addVariable(UNSIGNED_INT, {}, {di, "loopTimestep"});
  graph.setTileMapping(tempTensors.loopTimestep, 0);

  // A per tile copy of the loop count to copy into at the start of each loop
  // pass
  tempTensors.currentTimestep = graph.addVariable(
      UNSIGNED_INT, {batchSize, plan.batchEntryPartitions(), 1},
      {di, "currentTimestep"});
  mapAccordingToPlan(graph, tempTensors.currentTimestep,
                     PartitionType::BATCH_ENTRY, plan.batchEntryPartitions(),
                     plan);

  // Data length tensor broadcast per tile
  tempTensors.dataLengths = graph.addVariable(
      UNSIGNED_INT, {batchSize, plan.batchEntryPartitions(), 1},
      {di, "dataLengths"});
  mapAccordingToPlan(graph, tempTensors.dataLengths, PartitionType::BATCH_ENTRY,
                     plan.batchEntryPartitions(), plan);
  auto dataLengthsBroadcast =
      dataLengths.expand({1, 1}).broadcast(plan.batchEntryPartitions(), 1);
  prog.add(Copy(dataLengthsBroadcast, tempTensors.dataLengths));

  // A flag to indicate if a batch entry is complete or not.  Initialised to
  // zero, set to one when the update vertex detects that the end is reached
  tempTensors.complete = graph.addVariable(
      UNSIGNED_INT, {batchSize, plan.batchEntryPartitions(), 1},
      {di, "completeFlags"});
  mapAccordingToPlan(graph, tempTensors.complete, PartitionType::BATCH_ENTRY,
                     plan.batchEntryPartitions(), plan);
  auto initialiserZero = graph.addConstant(UNSIGNED_INT, {1}, 0u, di);
  graph.setTileMapping(initialiserZero, 0);
  prog.add(
      Copy(initialiserZero.broadcast(tempTensors.complete.numElements(), 0),
           tempTensors.complete.flatten(), false, di));

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
                     PartitionType::EXTEND, plan.parallel.extend, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesPnb,
                     PartitionType::EXTEND, plan.parallel.extend, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesPTotal,
                     PartitionType::EXTEND, plan.parallel.extend, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesParent,
                     PartitionType::EXTEND, plan.parallel.extend, plan);
  mapAccordingToPlan(graph, tempTensors.extendCandidatesAddend,
                     PartitionType::EXTEND, plan.parallel.extend, plan);

  // Tensors, mapped to match tiles used for the input to the Select step, to
  // hold extend candidates PTotal and addend.  Using these will reduce the copy
  // overhead
  auto makeSelectExtendCandidates = [&](const Type &type,
                                        const std::string &debugName) {
    const std::vector<std::size_t> shape = {batchSize, beamwidth,
                                            numClasses - 1};
    auto result = graph.addVariable(type, shape, {di, debugName});
    mapAccordingToPlan(graph, result, PartitionType::EXTEND,
                       plan.parallel.extend, plan);
    return result.dimShuffle({0, 2, 1});
  };
  tempTensors.selectExtendCandidatesPTotal =
      makeSelectExtendCandidates(partialsType, "selectExtendCandidatesPTotal");
  tempTensors.selectExtendCandidatesAddend =
      makeSelectExtendCandidates(UNSIGNED_INT, "selectExtendCandidatesAddend");

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
                     plan.parallel.copy, plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesPnb, PartitionType::COPY,
                     plan.parallel.copy, plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesPTotal,
                     PartitionType::COPY, plan.parallel.copy, plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesParent,
                     PartitionType::COPY, plan.parallel.copy, plan);
  mapAccordingToPlan(graph, tempTensors.copyCandidatesAddend,
                     PartitionType::COPY, plan.parallel.copy, plan);

  // Merge merge candidates vectors of tensors
  tempTensors.mergeCandidatesPb.resize(beamwidth);
  tempTensors.mergeCandidatesPnb.resize(beamwidth);
  tempTensors.mergeCandidatesPTotal.resize(beamwidth);
  tempTensors.mergeCandidatesParent.resize(beamwidth);
  tempTensors.mergeCandidatesAddend.resize(beamwidth);

  auto mapSortedCandidates = poplibs_support::make_visitor<void>(
      [&](const popnn::ctc::SimpleSortPartitions<unsigned> &sort) {
        // Nothing to map
      },
      [&](const popnn::ctc::RankPartitions<unsigned> &sort) {
        // Sorted result candidates
        const std::vector<std::size_t> sortedCandidateShape = {
            batchSize, sort.rank, beamwidth};
        tempTensors.sortedCandidatesPb = graph.addVariable(
            FLOAT, sortedCandidateShape, {di, "sortedCandidatesPb"});
        tempTensors.sortedCandidatesPnb = graph.addVariable(
            FLOAT, sortedCandidateShape, {di, "sortedCandidatesPnb"});
        tempTensors.sortedCandidatesPTotal = graph.addVariable(
            FLOAT, sortedCandidateShape, {di, "sortedCandidatesTotal"});

        tempTensors.sortedCandidatesParent = graph.addVariable(
            UNSIGNED_INT, sortedCandidateShape, {di, "sortedCandidatesParent"});
        tempTensors.sortedCandidatesAddend = graph.addVariable(
            UNSIGNED_INT, sortedCandidateShape, {di, "sortedCandidatesAddend"});

        mapAccordingToPlan(graph, tempTensors.sortedCandidatesPb,
                           PartitionType::SORT, sort.rank, plan);
        mapAccordingToPlan(graph, tempTensors.sortedCandidatesPnb,
                           PartitionType::SORT, sort.rank, plan);
        mapAccordingToPlan(graph, tempTensors.sortedCandidatesPTotal,
                           PartitionType::SORT, sort.rank, plan);
        mapAccordingToPlan(graph, tempTensors.sortedCandidatesParent,
                           PartitionType::SORT, sort.rank, plan);
        mapAccordingToPlan(graph, tempTensors.sortedCandidatesAddend,
                           PartitionType::SORT, sort.rank, plan);
      });

  const std::vector<size_t> mergeTensorsShape = {batchSize, beamwidth, 1};
  for (unsigned i = 0; i < beamwidth; i++) {

    boost::apply_visitor(mapSortedCandidates, plan.parallel.sort);

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
                       PartitionType::MERGE, plan.parallel.merge, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesPnb[i],
                       PartitionType::MERGE, plan.parallel.merge, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesPTotal[i],
                       PartitionType::MERGE, plan.parallel.merge, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesParent[i],
                       PartitionType::MERGE, plan.parallel.merge, plan);
    mapAccordingToPlan(graph, tempTensors.mergeCandidatesAddend[i],
                       PartitionType::MERGE, plan.parallel.merge, plan);
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
  beamTensors.pTotal =
      graph.addVariable(partialsType, beamProbsShape, {di, "beamPTotal"});
  beamTensors.lastOutput =
      graph.addVariable(UNSIGNED_INT, beamProbsShape, {di, "beamLastOutput"});

  mapAccordingToPlan(graph, beamTensors.parent, PartitionType::BATCH_ENTRY,
                     plan);
  mapAccordingToPlan(graph, beamTensors.addend, PartitionType::BATCH_ENTRY,
                     plan);

  mapAccordingToPlan(graph, beamTensors.pb, PartitionType::BATCH_ENTRY, plan);
  mapAccordingToPlan(graph, beamTensors.pnb, PartitionType::BATCH_ENTRY, plan);
  mapAccordingToPlan(graph, beamTensors.pTotal, PartitionType::BATCH_ENTRY,
                     plan);
  mapAccordingToPlan(graph, beamTensors.lastOutput, PartitionType::BATCH_ENTRY,
                     plan);

  const std::vector<std::size_t> beamLengthShape = {
      batchSize, plan.batchEntryPartitions(), 2 * beamwidth, 1};
  beamTensors.length =
      graph.addVariable(UNSIGNED_INT, beamLengthShape, {di, "beamLength"});
  mapAccordingToPlan(graph, beamTensors.length, PartitionType::BATCH_ENTRY,
                     plan);

  // Initialise the beam probabilities, with only one origin point
  auto initialiserProbZero =
      graph.addConstant<float>(partialsType, {1}, log::probabilityZero, di);
  graph.setTileMapping(initialiserProbZero, 0);
  auto initialiserProbOne =
      graph.addConstant<float>(partialsType, {1}, log::probabilityOne, di);
  graph.setTileMapping(initialiserProbOne, 0);

  auto initialiseBeamProbZero = [&](const Tensor &t) {
    auto tSliceOne = t.slice(0, 1, 2);
    auto tSliceZero = t.slice(1, beamwidth, 2);
    prog.add(Copy(initialiserProbOne.broadcast(tSliceOne.numElements(), 0),
                  tSliceOne.flatten(), false, di));
    prog.add(Copy(initialiserProbZero.broadcast(tSliceZero.numElements(), 0),
                  tSliceZero.flatten(), false, di));
  };
  initialiseBeamProbZero(beamTensors.pnb);
  initialiseBeamProbZero(beamTensors.pTotal);

  // Zero symbols per beam to start with
  auto initialiserZero = graph.addConstant(UNSIGNED_INT, {1}, 0u, di);
  graph.setTileMapping(initialiserZero, 0);
  prog.add(Copy(initialiserZero.broadcast(beamTensors.length.numElements(), 0),
                beamTensors.length.flatten(), false, di));

  // Setup the initial beam history with a zero time slice with:
  // beam   parent addend
  // 0      0      voidSymbol      (This is valid, but void)
  // 1      1      invalidSymbol   (These are invalid symbols and won't merge)
  // 2      1      invalidSymbol
  // 3      1      invalidSymbol
  // ...
  auto initialiserVoidSymbol =
      graph.addConstant(UNSIGNED_INT, {1}, voidSymbol, di);
  graph.setTileMapping(initialiserVoidSymbol, 0);
  auto initialiserInvalidSymbol =
      graph.addConstant(UNSIGNED_INT, {1}, invalidSymbol, di);
  graph.setTileMapping(initialiserInvalidSymbol, 0);

  auto initialiseTimestep0 = [&](const Tensor &t, const Tensor &init0,
                                 const Tensor &initOthers) {
    auto tSliceBeam0 = t.slice(0, 1, 2).slice(0, 1, 3);
    auto tSliceOtherBeams = t.slice(0, 1, 2).slice(1, beamwidth, 3);
    prog.add(Copy(init0.broadcast(tSliceBeam0.numElements(), 0),
                  tSliceBeam0.flatten()));
    prog.add(Copy(initOthers.broadcast(tSliceOtherBeams.numElements(), 0),
                  tSliceOtherBeams.flatten()));
  };

  auto initialiserOne = graph.addConstant(UNSIGNED_INT, {1}, 1u, di);
  graph.setTileMapping(initialiserOne, 0);
  initialiseTimestep0(beamTensors.parent, initialiserZero, initialiserOne);
  initialiseTimestep0(beamTensors.addend, initialiserVoidSymbol,
                      initialiserInvalidSymbol);

  // Last symbol[beam0] = voidSymbol initial state, others = invalidSymbol.
  // Although this is the same as the addend at the end of the beam history and
  // could be optimised out it doesn't cost much - it's stored on each tile,
  // never exchanged and updating it is the same speed path as other update
  // paths in the update vertex.  It speeds up candidate generation marginally
  // by avoiding indexing into the beam history
  const auto lastOutBeam0 = beamTensors.lastOutput.slice(0, 1, 2);
  const auto lastOutOtherBeams = beamTensors.lastOutput.slice(1, beamwidth, 2);

  prog.add(Copy(initialiserVoidSymbol.broadcast(lastOutBeam0.numElements(), 0),
                lastOutBeam0.flatten(), false, di));
  prog.add(Copy(
      initialiserInvalidSymbol.broadcast(lastOutOtherBeams.numElements(), 0),
      lastOutOtherBeams.flatten(), false, di));

  return beamTensors;
}
// A class to manage incrementing a variable and providing the partition that
// variable's value is in.  For example:
// batchSize =  5, 3 partitions with size 2, 2, 1
// [partitionIdx = 0,idx = 0], [0,1], [1,2], [1,3], [2,4]
struct PartitionIdx {
  unsigned partitionIdx;
  unsigned idx;
};
bool operator!=(PartitionIdx a, PartitionIdx b) { return a.idx != b.idx; }

class PartitionCounter {
  Interval partition;
  const popnn::ctc::InferencePlan &plan;
  const unsigned size;
  PartitionFn partitionFunction;
  PartitionIdx indices;

public:
  PartitionCounter(const popnn::ctc::InferencePlan &inPlan, unsigned inSize,
                   PartitionType partitionType)
      : plan(inPlan), size(inSize) {
    indices.partitionIdx = 0;
    indices.idx = 0;
    partitionFunction = getPartitionFunction(partitionType);
    partition = (plan.*partitionFunction)(size, indices.partitionIdx);
  };

  PartitionIdx next(void) {
    indices.idx++;
    if (indices.idx == partition.end()) {
      indices.partitionIdx++;
      partition = (plan.*partitionFunction)(size, indices.partitionIdx);
    }
    return indices;
  }
  PartitionIdx begin(void) {
    indices.partitionIdx = 0;
    indices.idx = 0;
    partition = (plan.*partitionFunction)(size, indices.partitionIdx);
    return indices;
  }

  PartitionIdx end(void) { return PartitionIdx{size, size}; }
};

Sequence createLoopBodyProg(Graph &graph, const popnn::ctc::InferencePlan &plan,
                            const Tensor &data, const BeamTensors &beams,
                            const TempTensors &tempTensors, unsigned blankClass,
                            unsigned beamwidth,
                            const poplar::DebugContext &di) {

  const unsigned batchSize = data.dim(0);
  const auto maxT = data.dim(2);
  const auto numClasses = data.dim(3);
  const auto numClassesM1 = numClasses - 1;
  Sequence prog;

  prog.add(Copy(tempTensors.loopTimestep.flatten().broadcast(
                    tempTensors.currentTimestep.numElements(), 0),
                tempTensors.currentTimestep.flatten()));

  // Generate candidates in the 1st compute set
  auto cs1 = graph.addComputeSet(di);
  PartitionCounter batch(plan, batchSize, PartitionType::BATCH);
  for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
    // Extend candidates
    PartitionCounter extend(plan, numClassesM1, PartitionType::EXTEND);
    for (auto e = extend.begin(); e != extend.end(); e = extend.next()) {
      const unsigned tile = plan.getTile(b.partitionIdx, 0, e.partitionIdx);
      const unsigned addendClass = e.idx >= blankClass ? e.idx + 1 : e.idx;
      for (unsigned v = 0; v < plan.parallel.extendVerticesPerPartition; v++) {
        const auto beamPartition = plan.partitionExtendVertices(beamwidth, v);
        generateExtendCandidateVertex(graph, data, beams, tempTensors, cs1,
                                      b.idx, {0, maxT}, e.idx, e.partitionIdx,
                                      blankClass, beamwidth, beamPartition,
                                      addendClass, tile);
      }
    }
    // Copy candidates
    PartitionCounter copy(plan, beamwidth, PartitionType::COPY);
    for (auto c = copy.begin(); c != copy.end(); c = copy.next()) {
      const unsigned tile = plan.getTile(b.partitionIdx, 0, c.partitionIdx);
      generateCopyCandidateVertex(graph, data, beams, tempTensors, cs1, b.idx,
                                  {0, maxT}, c.idx, c.partitionIdx, blankClass,
                                  beamwidth, tile);
    }
  }
  prog.add(Execute(cs1, di));

  // Broadcast the copy candidates into the vectors of merge candidates
  auto transform = [=](const Tensor &in, unsigned i) {
    return in.slice(i, i + 1, 1).expand({1}).broadcast(beamwidth, 1);
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
  // Copy to provide the extend candidate total probability and addend on the
  // correct tiles for the Select stage.  This data isn't changed in the Merge
  // stage, so we can do the copies all together to allow poplar to optimise
  // the operation.
  prog.add(Copy(tempTensors.extendCandidatesPTotal,
                tempTensors.selectExtendCandidatesPTotal));
  prog.add(Copy(tempTensors.extendCandidatesAddend,
                tempTensors.selectExtendCandidatesAddend));

  // Merge candidates in the 2nd compute set
  auto cs2 = graph.addComputeSet(di);
  for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
    for (unsigned copy = 0; copy < beamwidth; copy++) {
      PartitionCounter merge(plan, beamwidth, PartitionType::MERGE);
      for (auto m = merge.begin(); m != merge.end(); m = merge.next()) {
        const unsigned tile = plan.getTile(b.partitionIdx, 0, m.partitionIdx);
        mergeCandidateVertex(graph, beams, tempTensors, cs2, b.idx, {0, maxT},
                             m.idx, copy, m.partitionIdx, blankClass, beamwidth,
                             numClasses, tile);
      }
    }
  }
  prog.add(Execute(cs2, di));

  // Select the merged copy candidates, and zero the probability of merged
  // extend candidates in the 3rd compute set
  auto cs3 = graph.addComputeSet(di);
  for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
    PartitionCounter copy(plan, beamwidth, PartitionType::SELECT_COPY);
    for (auto c = copy.begin(); c != copy.end(); c = copy.next()) {
      const unsigned tile = plan.getTile(b.partitionIdx, 0, c.partitionIdx);
      selectCopyCandidateVertex(graph, tempTensors, cs3, b.idx, c.idx,
                                c.partitionIdx, beamwidth, tile);
    }
    PartitionCounter extend(plan, beamwidth, PartitionType::SELECT_EXTEND);
    for (auto e = extend.begin(); e != extend.end(); e = extend.next()) {
      const unsigned tile = plan.getTile(b.partitionIdx, 0, e.partitionIdx);
      selectExtendCandidateVertex(graph, tempTensors, cs3, b.idx, e.idx,
                                  e.partitionIdx, beamwidth, blankClass, tile);
    }
  }
  prog.add(Execute(cs3, di));

  // Sort candidates in the 4th compute set
  // There are 2 methods: SIMPLE_SORT and RANK which can have benefits depending
  // on the number of workers available
  const unsigned candidatesToCompare = beamwidth + beamwidth * numClassesM1;
  boost::apply_visitor(
      poplibs_support::make_visitor<void>(
          [&](const popnn::ctc::SimpleSortPartitions<unsigned> &sort) {
            auto cs4 = graph.addComputeSet(di);
            PartitionCounter batch(plan, batchSize, PartitionType::BATCH);
            for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
              for (unsigned simpleSort = 0; simpleSort < sort.simpleSort;
                   simpleSort++) {
                const unsigned tile =
                    plan.getTile(b.partitionIdx, 0, simpleSort);
                simpleSortCandidatesVertex(graph, tempTensors, cs4, b.idx, 0,
                                           candidatesToCompare, beamwidth,
                                           tile);
              }
            }
            prog.add(Execute(cs4, di));
          },
          [&](const popnn::ctc::RankPartitions<unsigned> &sort) {
            // Rank step 1 -  rank each candidate, writing into the partition's
            // result if in the top beamwidth rankings
            auto cs4a = graph.addComputeSet(di);
            PartitionCounter batch(plan, batchSize, PartitionType::BATCH);
            for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
              for (unsigned ranking = 0; ranking < sort.rank; ranking++) {
                const unsigned tile = plan.getTile(b.partitionIdx, 0, ranking);
                const auto perPartition =
                    ceildiv(candidatesToCompare, sort.rank);
                const auto first = perPartition * ranking;
                const auto last =
                    std::min(perPartition * (ranking + 1), candidatesToCompare);

                rankCandidatesVertex(graph, tempTensors, cs4a, b.idx, ranking,
                                     candidatesToCompare, {first, last},
                                     beamwidth, tile);
              }
            }
            prog.add(Execute(cs4a, di));

            // Rank step 2 - reduce all the partition's results down to 1 result
            // (beamwidth values) per batch entry
            auto cs4b = graph.addComputeSet(di);
            for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
              PartitionCounter reduce(plan, beamwidth,
                                      PartitionType::SORT_REDUCE);
              for (auto r = reduce.begin(); r != reduce.end();
                   r = reduce.next()) {
                const auto tile =
                    plan.getTile(b.partitionIdx, 0, r.partitionIdx);
                reduceCandidatesVertex(graph, tempTensors, cs4b, b.idx, r.idx,
                                       r.partitionIdx, sort.rank, tile);
              }
            }
            prog.add(Execute(cs4b, di));
          }),
      plan.parallel.sort);

  // Update beam history and probabilities in the 5th compute set
  auto cs5 = graph.addComputeSet(di);
  const unsigned sortedResultOffset = plan.parallel.copy * (beamwidth - 1);
  for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
    for (unsigned beam = 0; beam < plan.batchEntryPartitions(); beam++) {
      const unsigned tile = plan.getTile(b.partitionIdx, 0, beam);
      updateVertex(graph, beams, tempTensors, cs5, b.idx, {0, maxT}, beam,
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
  // The first timestep is 1, with timestep 0 being an initial state so start
  // counting from 1
  prog.add(countedForLoop(graph, tempTensors.loopTimestep, 1,
                          tempTensors.maxTimeInBatch, 1, loopBody, di));

  // Create results and map, inserting a 3rd dimension so mapping functions can
  // be used
  auto labelLengths = graph.addVariable(UNSIGNED_INT, {batchSize, topPaths},
                                        {di, "labelLengths"});
  auto decodedLabels = graph.addVariable(
      UNSIGNED_INT, {batchSize, topPaths, maxT}, {di, "decodedLabels"});

  mapAccordingToPlan(graph, labelLengths.expand({2}), PartitionType::OUTPUT,
                     plan.parallel.output, plan);
  mapAccordingToPlan(graph, decodedLabels, PartitionType::OUTPUT,
                     plan.parallel.output, plan);

  auto outputCS = graph.addComputeSet({di, "output"});
  PartitionCounter batch(plan, batchSize, PartitionType::BATCH);
  for (auto b = batch.begin(); b != batch.end(); b = batch.next()) {
    PartitionCounter output(plan, topPaths, PartitionType::OUTPUT);
    for (auto o = output.begin(); o != output.end(); o = output.next()) {
      const unsigned tile = plan.getTile(b.partitionIdx, 0, o.partitionIdx);
      generateOutputVertex(graph, beams, tempTensors, decodedLabels,
                           labelLengths, outputCS, b.idx, o.idx, o.partitionIdx,
                           beamwidth, numClasses, tile);
    }
  }
  prog.add(Execute(outputCS, di));
  // Slice the correct output and cast if needed
  auto transform = [&](const Tensor &in) {
    return in.slice(0, 1, 1)
        .slice(0, topPaths, 2)
        .reshape({batchSize, topPaths});
  };
  auto pTotal = transform(beams.pTotal);
  auto labelProbs = [&]() {
    if (partialsType != outType) {
      poplar::DebugContext castDebug{di, "Cast"};
      auto castCS = graph.addComputeSet(castDebug);
      auto probs = popops::cast(graph, pTotal, outType, castCS, castDebug);
      prog.add(Execute(castCS, castDebug));
      return probs;
    } else {
      return pTotal;
    };
  }();

  di.addOutputs({{"labelProbs", poputil::toProfileValue(labelProbs)},
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
