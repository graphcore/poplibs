// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "RnnUtil.hpp"
#include <boost/optional.hpp>
#include <cassert>
#include <cstdint>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/Rnn.hpp>

using namespace poplar;
using namespace poplibs_support;
using namespace poplar::program;
using namespace popops;

namespace poputil {

template <>
poplar::ProfileValue toProfileValue(const popnn::rnn::RnnParams &t) {
  poplar::ProfileValue::Map v;
  v.insert({"dataType", toProfileValue(t.dataType)});
  v.insert({"batchSize", toProfileValue(t.batchSize)});
  v.insert({"timeSteps", toProfileValue(t.timeSteps)});
  v.insert({"layerSizes", toProfileValue(t.layerSizes)});
  return v;
}

} // namespace poputil

namespace popnn {
namespace rnn {

static std::pair<std::size_t, std::size_t> getNumGrains(std::size_t size) {
  auto grainSize = gcd(16UL, size);
  auto numGrains = size / grainSize;
  return {grainSize, numGrains};
}

static std::size_t
calculateTileUsage(bool output, unsigned numTiles, const std::size_t batchSize,
                   const std::vector<std::size_t> &layerSizes) {
  auto size = output ? layerSizes[1] : layerSizes[0];
  auto [grainSize, numGrains] = getNumGrains(size);
  (void)grainSize;
  numGrains *= batchSize;
  auto grainsPerTile = ceildiv(numGrains, numTiles);
  auto usedTiles = ceildiv(numGrains, grainsPerTile);
  return usedTiles;
}

static std::size_t getTilesPerShard(Graph &graph, const RnnParams &params,
                                    unsigned numShards) {
  auto numTiles = graph.getTarget().getNumTiles();
  std::vector<std::size_t> tilesUsed;
  for (unsigned i = 0; i < params.layerSizes.size(); ++i) {
    auto numTilesUsed =
        calculateTileUsage(i, numTiles, params.batchSize, params.layerSizes);
    tilesUsed.push_back(numTilesUsed);
  }
  std::size_t tilesPerShard = std::max(tilesUsed[0], tilesUsed[1]);
  assert(numShards <= numTiles / tilesPerShard);
  assert(numShards <= params.timeSteps);
  return tilesPerShard;
}

static poplar::Tensor createMultiDynamicSliceTensor(
    poplar::Graph &graph, poplar::Type dataType, unsigned numTiles,
    unsigned sequenceLength, unsigned maxShards, bool singleSequencePerShard,
    boost::optional<unsigned> sequenceMultiple,
    boost::optional<unsigned> shardOffset, unsigned numGrains,
    unsigned grainSize, const poplar::DebugNameAndId &dnai) {
  const auto grainsPerTile = ceildiv(numGrains, numTiles);
  const auto numUsedTiles = ceildiv(numGrains, grainsPerTile);
  const auto grainsOnLastTile = numGrains - (numUsedTiles - 1) * grainsPerTile;
  auto sOffset = shardOffset ? *shardOffset : 0;
  auto numTensorShards = shardOffset ? 1 : maxShards;
  unsigned seqLengthExceptLastTile = ceildiv(sequenceLength, maxShards);
  unsigned seqLengthLastTile =
      sequenceLength - seqLengthExceptLastTile * (maxShards - 1);
  if (singleSequencePerShard) {
    seqLengthExceptLastTile = 1;
    seqLengthLastTile = 1;
  }
  if (sequenceMultiple) {
    seqLengthExceptLastTile *= *sequenceMultiple;
    seqLengthLastTile *= *sequenceMultiple;
  }

  std::vector<poplar::Tensor> tExclLast;
  std::vector<poplar::Tensor> tLast;
  for (unsigned i = sOffset; i < numTensorShards + sOffset; ++i) {
    auto seqLength =
        (i < maxShards - 1) ? seqLengthExceptLastTile : seqLengthLastTile;
    tExclLast.push_back(graph.addVariable(
        dataType, {numUsedTiles - 1, seqLength, grainsPerTile, grainSize},
        {dnai, "rnnSlice/" + std::to_string(i)}));
    tLast.push_back(graph.addVariable(dataType,
                                      {seqLength, grainsOnLastTile, grainSize},
                                      {dnai, "rnnSliceLast"}));
  }

  // Tensors for the last sequence
  auto totalNumTiles = graph.getTarget().getNumTiles();
  for (unsigned tileOfShard = 0; tileOfShard != numUsedTiles; ++tileOfShard) {
    for (unsigned shard = sOffset; shard != numTensorShards + sOffset;
         ++shard) {
      // The use of tile 0 could prevent the SubGraphReplicator from replicating
      // the loop counter. For this reason, tiles are mapped beginning from
      // higher numbered tiles.
      auto tile =
          ((tileOfShard + 1) * totalNumTiles / numUsedTiles) - 1 - shard;
      if (tileOfShard == numUsedTiles - 1) {
        graph.setTileMapping(tLast[shard - sOffset], tile);
      } else {
        graph.setTileMapping(tExclLast[shard - sOffset][tileOfShard], tile);
      }
    }
  }

  std::vector<poplar::Tensor> tAll;
  for (unsigned shard = sOffset; shard != numTensorShards + sOffset; ++shard) {
    tAll.push_back(
        poplar::concat(tExclLast[shard - sOffset].dimRoll(0, 1).flatten(1, 3),
                       tLast[shard - sOffset], 1));
  }
  return poplar::concat(tAll, 0);
}

/// Create and map a tensor for a sequence of outputs from a RNN layer.
/// The sequence length is taken from \a sequenceLength parameter, not the
/// \a params structure.
static poplar::Tensor
createShardedTensor(Graph &graph, const RnnParams &params,
                    std::size_t tilesPerShard, std::size_t size,
                    bool singleSequencePerShard,
                    boost::optional<unsigned> sequenceMultiple,
                    boost::optional<unsigned> shardIndex, unsigned numShards,
                    const DebugNameAndId &dnai) {
  // TODO: T12909 Take output grouping from matmul operation.
  auto [grouping, numGroups] = getNumGrains(size);
  Tensor t = createMultiDynamicSliceTensor(
      graph, params.dataType, tilesPerShard, params.timeSteps, numShards,
      singleSequencePerShard, sequenceMultiple, shardIndex,
      numGroups * params.batchSize, grouping, {dnai, "sharded"});
  return t.reshapePartial(1, 2, {numGroups, params.batchSize})
      .dimRoll(1, 2)
      .flatten(2, 4);
}

// Create tensor on shards. The tensor could be sized to a multiple
static Tensor createTensor(Graph &graph, const RnnParams &params,
                           std::size_t tilesPerShard, unsigned size,
                           unsigned multiple, unsigned numShards,
                           const DebugNameAndId &dnai) {
  return createShardedTensor(graph, params, tilesPerShard, size, true, multiple,
                             {}, numShards, {dnai, "singleShard"});
}

static Tensor createTensorShard(Graph &graph, const RnnParams &params,
                                std::size_t tilesPerShard, unsigned size,
                                unsigned multiple, unsigned shardOffset,
                                unsigned numShards,
                                const DebugNameAndId &dnai) {
  return createShardedTensor(graph, params, tilesPerShard, size, true, multiple,
                             shardOffset, numShards, {dnai, "singleShard"});
}

static Tensor createRnnTensor(Graph &graph, const RnnParams &params,
                              std::size_t tilesPerShard, unsigned size,
                              unsigned numShards, const DebugNameAndId &dnai) {
  return createShardedTensor(graph, params, tilesPerShard, size, false, {}, {},
                             numShards, {dnai, "RnnTensor"});
}

static Tensor createRnnTensor(Graph &graph, const RnnParams &params,
                              std::size_t tilesPerShard, unsigned size,
                              unsigned multiple, unsigned numShards,
                              const DebugNameAndId &dnai) {
  return createShardedTensor(graph, params, tilesPerShard, size, false,
                             multiple, {}, numShards, {dnai, "RnnTensor"});
}

static std::vector<Tensor> createState(Graph &graph, const RnnParams &params,
                                       std::size_t tilesPerShard,
                                       const std::vector<Tensor> &init,
                                       unsigned numShards,
                                       const DebugNameAndId &dnai) {
  std::vector<Tensor> state;
  for (unsigned i = 0; i < init.size(); ++i) {
    if (!init[i].valid()) {
      state.push_back(init[i]);
      continue;
    }
    unsigned rank = init[i].rank();
    unsigned multiple = init[i].dim(0);
    unsigned innermostDim = init[i].dim(rank - 1);
    state.push_back(createTensor(graph, params, tilesPerShard, innermostDim,
                                 multiple, numShards,
                                 {dnai, "stateTensor/" + std::to_string(i)}));
  }
  return state;
}

static Tensor createOutputState(Graph &graph, const RnnParams &params,
                                std::size_t tilesPerShard,
                                const Tensor &allOutput, unsigned numShards,
                                const DebugNameAndId &dnai) {
  unsigned rank = allOutput.rank();
  unsigned multiple = allOutput.dim(0) / params.timeSteps;
  unsigned innermostDim = allOutput.dim(rank - 1);
  auto out = createTensor(graph, params, tilesPerShard, innermostDim, multiple,
                          numShards, {dnai, "outputTensor"});
  return out;
}

static std::vector<Tensor> copyStateShard(std::vector<Tensor> &prevStateShard,
                                          std::vector<Tensor> &state,
                                          unsigned index,
                                          program::Sequence &prog,
                                          const DebugNameAndId &dnai) {
  std::vector<Tensor> newShard;
  for (unsigned i = 0; i < state.size(); ++i) {
    if (!state[i].valid()) {
      newShard.push_back(state[i]);
      continue;
    }
    unsigned multiple = prevStateShard[i].dim(0);
    auto shard = state[i].slice(index * multiple, (index + 1) * multiple);
    prog.add(Copy(prevStateShard[i], shard, false, {dnai, "copyState"}));
    newShard.push_back(shard);
  }
  return newShard;
}

class RnnState {
public:
  RnnState(Graph &graph, unsigned seqLen, unsigned numShards, bool reverse,
           poputil::PoplibsOpDebugInfo &dnai);
  const Interval &interval() const { return currInterval; };
  unsigned index() const { return currIndex; };
  unsigned sequenceLength() const { return currInterval.size(); };
  const Tensor &counterTensor() const { return seqIdx; };
  unsigned operator()() const { return count; };
  void next();
  program::Sequence initCounter();
  program::Sequence updateCounter();

private:
  Graph &graph;
  Tensor one;
  Tensor seqIdx;
  Tensor startExclLastShard;
  Tensor startLastShard;
  unsigned seqLengthExclLast;

  unsigned fullSequenceLen;
  unsigned numShards;
  unsigned count;
  bool reverse;
  poputil::PoplibsOpDebugInfo &dnai;

  unsigned currIndex;
  Interval currInterval;
};

RnnParams::RnnParams(poplar::Type dataType, std::size_t batchSize,
                     std::size_t timeSteps, std::vector<std::size_t> layers)
    : dataType(dataType), batchSize(batchSize), timeSteps(timeSteps),
      layerSizes(layers) {}

RnnState::RnnState(Graph &graph, unsigned seqLen, unsigned numShards,
                   bool reverse, poputil::PoplibsOpDebugInfo &dnai)
    : graph(graph), fullSequenceLen(seqLen), numShards(numShards),
      count(numShards), reverse(reverse), dnai(dnai) {

  // loop counter
  seqIdx = graph.addVariable(UNSIGNED_INT, {1}, {dnai, "seqIdx"});
  one = graph.addConstant(UNSIGNED_INT, {1}, 1, {dnai, "one"});
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);

  seqLengthExclLast = ceildiv(seqLen, numShards);
  if (reverse) {
    startExclLastShard = graph.addConstant(
        UNSIGNED_INT, {1}, seqLengthExclLast - 1, {dnai, "start"});
    graph.setTileMapping(startExclLastShard, 0);
    auto seqLengthLast = seqLen - (seqLengthExclLast * (numShards - 1));
    if (seqLengthLast == seqLengthExclLast) {
      startLastShard = startExclLastShard;
    } else {
      startLastShard = graph.addConstant(UNSIGNED_INT, {1}, seqLengthLast - 1,
                                         {dnai, "startLastShard"});
      graph.setTileMapping(startLastShard, 0);
    }
  }

  // initialising
  currIndex = reverse ? (numShards - 1) : 0;
  unsigned intervalBegin = currIndex * seqLengthExclLast;
  unsigned intervalEnd = reverse ? fullSequenceLen : seqLengthExclLast;
  currInterval = Interval{intervalBegin, intervalEnd};
}

void RnnState::next() {
  count--;
  if (count == 0) {
    return;
  }
  currIndex += reverse ? -1 : 1;
  auto nextBegin = currIndex * seqLengthExclLast;
  auto nextEnd = (currIndex < numShards - 1) ? nextBegin + seqLengthExclLast
                                             : fullSequenceLen;
  currInterval = Interval{nextBegin, nextEnd};
}

program::Sequence RnnState::initCounter() {
  auto prog = Sequence({}, {dnai, "initCounter"});
  if (reverse) {
    auto start =
        (currIndex < (numShards - 1)) ? startExclLastShard : startLastShard;
    prog.add(Copy(start, seqIdx, false, {dnai, "initSeqIdxToEnd"}));
  } else {
    popops::zero(graph, seqIdx, prog, {dnai, "initSeqIdxZero"});
  }
  return prog;
}

program::Sequence RnnState::updateCounter() {
  auto prog = Sequence({}, {dnai, "updateCounter"});
  // Update loop counters
  if (reverse) {
    subInPlace(graph, seqIdx, one, prog, {dnai, "seqIdxDecr"});
  } else {
    addInPlace(graph, seqIdx, one, prog, {dnai, "seqIdxIncr"});
  }
  return prog;
}

static std::tuple<program::Sequence, std::vector<Tensor>, Tensor, Tensor,
                  std::vector<Tensor>>
sliceShard(Graph &graph, const RnnParams &params, std::size_t tilesPerShard,
           const RnnState &shard, const std::vector<Tensor> &inputs,
           const Tensor *interimIn, const Tensor *interimOut,
           const std::vector<Tensor> &outputState, unsigned numShards,
           const DebugNameAndId &dnai) {
  auto interval = shard.interval();
  auto counter = shard.counterTensor();
  auto shardIndex = shard.index();
  auto loop = Sequence({}, {dnai, "RnnLoop"});
  std::vector<Tensor> inputSlices(inputs.size());
  for (unsigned i = 0; i < inputs.size(); ++i) {
    auto &input = inputs[i];
    if (input.valid()) {
      auto inputShard = input.slice(interval);
      inputSlices[i] = popops::dynamicSlice(graph, inputShard, counter, {0},
                                            {1}, loop, {dnai, "inputSlice"})[0];
    }
  }

  // Retrieve shard of saved intermediates;
  Tensor interimInSlice;
  if (interimIn) {
    auto interimInShard = interimIn->slice(interval);
    interimInSlice = popops::dynamicSlice(graph, interimInShard, counter, {0},
                                          {1}, loop, {dnai, "inputSlice"})[0];
  }

  // Prepare tensor shard to store intermediates if necessary
  Tensor interimOutSlice;
  if (interimOut) {
    unsigned numInterimIn = interimOut ? interimOut->dim(1) : 0;
    auto outputSize = params.layerSizes[1];
    interimOutSlice = createTensorShard(graph, params, tilesPerShard,
                                        outputSize, numInterimIn, shardIndex,
                                        numShards, {dnai, "interimOutSlice"});
  }
  std::vector<Tensor> outputSlice;
  for (unsigned i = 0; i < outputState.size(); ++i) {
    auto numOutputs = outputState[i].dim(0) / numShards;
    auto sliceBegin = shard.index();
    auto outputShard = outputState[i]
                           .reshapePartial(0, 1, {numShards, numOutputs})
                           .slice(sliceBegin, sliceBegin + 1);
    Tensor out =
        popops::dynamicSlice(graph, outputShard, counter, {0}, {1}, loop,
                             {dnai, "outputSlice/" + std::to_string(i)})[0];
    outputSlice.push_back(out);
  }
  return std::make_tuple(loop, inputSlices, interimInSlice, interimOutSlice,
                         outputSlice);
}

static void updateTensor(Graph &graph, const RnnParams &params, const Tensor &t,
                         const Tensor &s, const RnnState &shard,
                         program::Sequence &prog, const DebugNameAndId &dnai) {
  assert((t.rank() == s.rank()) || (t.rank() == s.rank() + 1));
  auto tRecurrent = t;
  if (t.rank() == s.rank()) {
    auto tensorsPerShard = s.dim(0);
    tRecurrent = t.reshapePartial(0, 1, {params.timeSteps, tensorsPerShard});
  }
  auto interval = shard.interval();
  auto counter = shard.counterTensor();
  auto tSharded = tRecurrent.slice(interval);
  popops::dynamicUpdate(graph, tSharded, s.expand({0}), counter, {0}, {1}, prog,
                        {dnai, "updateTensor"});
}

static program::Sequence
updateShard(Graph &graph, const RnnParams &params, const RnnState &shard,
            Tensor &interimOutSlice, const Tensor *interimOut,
            const Tensor &stateSlice, const Tensor &stateSequence,
            const std::vector<Tensor> &outputSlice,
            const std::vector<Tensor> &outputs, const DebugNameAndId &dnai) {
  auto loop = Sequence({}, {dnai, "updateShard"});
  if (interimOut) {
    updateTensor(graph, params, *interimOut, interimOutSlice, shard, loop,
                 {dnai, "UpdateInterimOutShard"});
  }
  if (stateSequence.valid()) {
    updateTensor(graph, params, stateSequence, stateSlice, shard, loop,
                 {dnai, "updateStateSequenceSlice"});
  }
  for (unsigned i = 0; i < outputs.size(); ++i) {
    updateTensor(graph, params, outputs[i], outputSlice[i], shard, loop,
                 {dnai, "updateOutputSlice/" + std::to_string(i)});
  }
  return loop;
}

Tensor createInitialState(Graph &graph, const RnnParams &params, bool isOutput,
                          unsigned multiple, unsigned numShards,
                          const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params, isOutput, multiple, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params, numShards);
  auto innermostDim = params.layerSizes[isOutput];
  auto state = createTensorShard(graph, params, tilesPerShard, innermostDim,
                                 multiple, 0, numShards, {di, "initState"});
  return state;
}

Tensor createRecurrentTensor(Graph &graph, const RnnParams &params,
                             unsigned size, unsigned numShards,
                             const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, size, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params, numShards);
  return createRnnTensor(graph, params, tilesPerShard, size, numShards,
                         {di, "recurrent"});
}

Tensor createInputTensor(Graph &graph, const RnnParams &params,
                         unsigned numShards,
                         const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params, numShards);
  auto inputSize = params.layerSizes[0];
  return createRnnTensor(graph, params, tilesPerShard, inputSize, numShards,
                         {di});
}

Tensor createOutputTensor(Graph &graph, const RnnParams &params,
                          unsigned numShards,
                          const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params, numShards);
  auto outputSize = params.layerSizes[1];
  return createRnnTensor(graph, params, tilesPerShard, outputSize, numShards,
                         {di});
}

Tensor createOutputTensor(Graph &graph, const RnnParams &params,
                          unsigned multiple, unsigned numShards,
                          const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, multiple, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params, numShards);
  auto outputSize = params.layerSizes[1];
  return createRnnTensor(graph, params, tilesPerShard, outputSize, multiple,
                         numShards, {di});
}

Tensor shiftRnnTensor(Graph &graph, const RnnParams &params,
                      const Tensor &tBase, const Tensor &tSingle,
                      program::Sequence &prog, unsigned numShards,
                      const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params, tBase, tSingle, prog, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params, numShards);
  unsigned rank = tBase.rank();
  unsigned multiple = tBase.dim(0) / params.timeSteps;
  unsigned innermostDim = tBase.dim(rank - 1);
  std::vector<Tensor> sequence;
  Tensor tLast = tSingle;
  for (unsigned shardOffset = 0; shardOffset < numShards; ++shardOffset) {
    // Create a tensor on the current shard
    auto tFirst = createTensorShard(
        graph, params, tilesPerShard, innermostDim, multiple, shardOffset,
        numShards, {di, "shard/" + std::to_string(shardOffset)});
    sequence.push_back(tFirst);

    // Retain all tensors in shard except the last
    unsigned begin = shardOffset * multiple;
    unsigned end =
        (shardOffset < numShards - 1) ? begin + multiple : params.timeSteps;
    if (begin < end - 1) {
      auto tExclLast = tBase.slice(begin, end - 1);
      sequence.push_back(tExclLast);
    }

    // Copy tSingle to the very first tensor. Thereafter copy the last tensor
    // of the previous shard to the first tensor of the current shard
    prog.add(Copy(tLast, tFirst, false,
                  {di, "shiftToRnnShard/" + std::to_string(shardOffset)}));
    tLast = tBase.slice(end - 1, end);
  }
  auto out = concat(sequence);
  return out;
}

std::vector<Tensor>
Rnn(Graph &graph, const RnnParams &params, bool reverse,
    const std::vector<Tensor> &initState, const StateSequence &stateSequence,
    const std::vector<Tensor> &inputs, const Tensor *interimIn,
    Tensor *interimOut, std::vector<Tensor> &outputs,
    program::Sequence &initProg, const LoopBodyType &loopFn, unsigned numShards,
    const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params, reverse, initState, inputs, interimIn,
                            interimOut, outputs, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params, numShards);

  // Create a state tensor in every shard.
  auto state = createState(graph, params, tilesPerShard, initState, numShards,
                           {di, "RnnState"});

  // If state does not include the output, create an output tensor
  // in every shard.
  std::vector<Tensor> outputState;
  for (unsigned i = 0; i < outputs.size(); ++i) {
    outputState.push_back(createOutputState(graph, params, tilesPerShard,
                                            outputs[i], numShards,
                                            {di, "RnnOutputState"}));
  }
  RnnState shard(graph, params.timeSteps, numShards, reverse, di);
  std::vector<Tensor> stateShard;
  std::vector<Tensor> prevStateShard = initState;
  while (shard()) {
    auto shardIndex = shard.index();

    // Copy over state from previous shard
    stateShard = copyStateShard(prevStateShard, state, shardIndex, initProg,
                                {di, "copyStateShard"});

    auto initCounter = shard.initCounter();

    // Dynamically slice from current input shards
    auto [slicer, inputSlices, interimInSlice, interimOutSlice, outputSlice] =
        sliceShard(graph, params, tilesPerShard, shard, inputs, interimIn,
                   interimOut, outputState, numShards, {di, "sliceShard"});

    // Call loop builder
    auto counter = shard.counterTensor();
    auto process =
        loopFn(shardIndex, stateShard, inputSlices, counter, interimInSlice,
               interimOutSlice, outputSlice, initProg,
               {di, "shard/" + std::to_string(shardIndex)});

    Tensor stateSlice, stateSeqOutput;
    if (stateSequence.output.valid()) {
      stateSlice = stateShard[stateSequence.stateIndex];
      stateSeqOutput = stateSequence.output;
    }

    // Dynamically Update slice of current output shards
    auto updater = updateShard(graph, params, shard, interimOutSlice,
                               interimOut, stateSlice, stateSeqOutput,
                               outputSlice, outputs, {di, "updateShard"});

    auto updateCounter = shard.updateCounter();

    initProg.add(initCounter);
    auto loop = Sequence({}, {di});
    loop.add(slicer);
    loop.add(process);
    loop.add(updater);
    loop.add(updateCounter);
    initProg.add(Repeat(shard.sequenceLength(), loop, {di}));

    // Hold on to the updated state.
    prevStateShard = stateShard;
    shard.next();
  }
  return stateShard;
}

} // namespace rnn
} // namespace popnn
