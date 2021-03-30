// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "RnnUtil.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include <boost/optional.hpp>
#include <cassert>
#include <cstdint>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/Rnn.hpp>
#include <poputil/GraphFunction.hpp>

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

struct RnnOpts {
  boost::optional<std::size_t> codeReuse;
};

static RnnOpts parseOptions(const OptionFlags &options) {
  RnnOpts rnnOpts;
  rnnOpts.codeReuse = boost::none;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec rnnSpec{
      {"codeReuse", OptionHandler::createWithInteger(rnnOpts.codeReuse)},
  };
  for (const auto &entry : options) {
    rnnSpec.parse(entry.first, entry.second);
  }
  return rnnOpts;
}

static std::pair<std::size_t, std::size_t> getNumGrains(std::size_t size) {
  auto grainSize = gcd(16UL, size);
  auto numGrains = size / grainSize;
  return {grainSize, numGrains};
}

static std::size_t calculatedBytesPerTile(unsigned numTiles,
                                          std::size_t batchSize,
                                          std::size_t size,
                                          std::size_t typeSize) {
  auto [grainSize, numGrains] = getNumGrains(size);
  numGrains *= batchSize;
  auto grainsPerTile = ceildiv(numGrains, numTiles);
  return grainsPerTile * grainSize * typeSize;
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

static std::size_t getTilesPerShard(const Graph &graph,
                                    const RnnParams &params) {
  auto numTiles = graph.getTarget().getNumTiles();
  std::vector<std::size_t> tilesUsed;
  for (unsigned i = 0; i < params.layerSizes.size(); ++i) {
    auto numTilesUsed =
        calculateTileUsage(i, numTiles, params.batchSize, params.layerSizes);
    tilesUsed.push_back(numTilesUsed);
  }
  std::size_t tilesPerShard = std::max(tilesUsed[0], tilesUsed[1]);
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
  const Tensor &indexTensor() const { return shardSeqIdx[currIndex]; };
  const Tensor &counterTensor() const { return seqIdx; };
  unsigned operator()() const { return count; };
  bool first() { return count == numShards; };
  void next();
  program::Sequence initCounter();
  program::Sequence updateCounter();

private:
  Graph &graph;
  std::vector<Tensor> shardSeqIdx;
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
    : graph(graph), shardSeqIdx(numShards), fullSequenceLen(seqLen),
      numShards(numShards), count(numShards), reverse(reverse), dnai(dnai) {

  // loop counter
  seqIdx = graph.addVariable(UNSIGNED_INT, {1}, {dnai, "seqIdx"});
  one = graph.addConstant(UNSIGNED_INT, {1}, 1, {dnai, "one"});
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);

  seqLengthExclLast = ceildiv(seqLen, numShards);
  for (unsigned i = 0; i < numShards; ++i) {
    shardSeqIdx[i] =
        graph.addConstant(UNSIGNED_INT, {1}, (i * seqLengthExclLast),
                          {dnai, "shardSeqIdx/" + std::to_string(i)});
    graph.setTileMapping(shardSeqIdx[i], 0);
  }

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

std::size_t RnnParams::getMaxShards(const poplar::Graph &graph) const {
  auto numTiles = graph.getTarget().getNumTiles();
  auto tilesPerShard = getTilesPerShard(graph, *this);
  auto maxShards = numTiles / tilesPerShard;
  auto nTimeStepsPerShard = ceildiv(timeSteps, maxShards);
  return ceildiv(timeSteps, nTimeStepsPerShard);
}

std::size_t RnnParams::getInputBytesPerTile(const Graph &graph) const {
  auto target = graph.getTarget();
  auto numTiles = target.getNumTiles();
  auto typeSize = target.getTypeSize(dataType);
  auto bytes =
      calculatedBytesPerTile(numTiles, batchSize, layerSizes[0], typeSize);
  return bytes;
}

std::size_t RnnParams::getOutputBytesPerTile(const Graph &graph) const {
  auto target = graph.getTarget();
  auto numTiles = target.getNumTiles();
  auto typeSize = target.getTypeSize(dataType);
  auto bytes =
      calculatedBytesPerTile(numTiles, batchSize, layerSizes[1], typeSize);
  return bytes;
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
  auto prog = Sequence{{}, {dnai, "initCounter"}};
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
  auto prog = Sequence{{}, {dnai, "updateCounter"}};
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
  auto loop = Sequence{{}, {dnai, "RnnLoop"}};
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

static program::Sequence updateShard(
    Graph &graph, const RnnParams &params, const RnnState &shard,
    Tensor &interimOutSlice, const Tensor *interimOut, const Tensor &stateSlice,
    const Tensor &stateSequence, const std::vector<Tensor> &outputSlice,
    const std::vector<Tensor> &outputs, const std::vector<Tensor> &createdSlice,
    const std::vector<Tensor> &created, const DebugNameAndId &dnai) {
  auto loop = Sequence{{}, {dnai, "updateShard"}};
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
  for (unsigned i = 0; i < created.size(); ++i) {
    updateTensor(graph, params, created[i], createdSlice[i], shard, loop,
                 {dnai, "updateCreatedSlice/" + std::to_string(i)});
  }
  return loop;
}

Tensor createInitialState(Graph &graph, const RnnParams &params, bool isOutput,
                          unsigned multiple, unsigned numShards,
                          const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params, isOutput, multiple, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params);
  auto innermostDim = params.layerSizes[isOutput];
  auto state = createTensorShard(graph, params, tilesPerShard, innermostDim,
                                 multiple, 0, numShards, {di, "initState"});
  return state;
}

Tensor createRecurrentTensor(Graph &graph, const RnnParams &params,
                             unsigned size, unsigned numShards,
                             const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, size, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params);
  return createRnnTensor(graph, params, tilesPerShard, size, numShards,
                         {di, "recurrent"});
}

Tensor createInputTensor(Graph &graph, const RnnParams &params,
                         unsigned numShards,
                         const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params);
  auto inputSize = params.layerSizes[0];
  return createRnnTensor(graph, params, tilesPerShard, inputSize, numShards,
                         {di});
}

Tensor createOutputTensor(Graph &graph, const RnnParams &params,
                          unsigned numShards,
                          const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params);
  auto outputSize = params.layerSizes[1];
  return createRnnTensor(graph, params, tilesPerShard, outputSize, numShards,
                         {di});
}

Tensor createOutputTensor(Graph &graph, const RnnParams &params,
                          unsigned multiple, unsigned numShards,
                          const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, multiple, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params);
  auto outputSize = params.layerSizes[1];
  return createRnnTensor(graph, params, tilesPerShard, outputSize, multiple,
                         numShards, {di});
}

Tensor shiftRnnTensor(Graph &graph, const RnnParams &params,
                      const Tensor &tBase, const Tensor &tSingle,
                      program::Sequence &prog, unsigned numShards,
                      const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params, tBase, tSingle, prog, numShards));
  const auto tilesPerShard = getTilesPerShard(graph, params);
  unsigned rank = tBase.rank();
  unsigned multiple = tBase.dim(0) / params.timeSteps;
  unsigned innermostDim = tBase.dim(rank - 1);
  auto timeStepsPerShard = ceildiv(params.timeSteps, numShards);
  std::vector<Tensor> sequence;
  Tensor tLast = tSingle;
  for (unsigned shardOffset = 0; shardOffset < numShards; ++shardOffset) {
    // Create a tensor on the current shard
    auto tFirst = createTensorShard(
        graph, params, tilesPerShard, innermostDim, multiple, shardOffset,
        numShards, {di, "shard/" + std::to_string(shardOffset)});
    sequence.push_back(tFirst);

    // Retain all tensors in shard except the last
    unsigned begin = shardOffset * multiple * timeStepsPerShard;
    unsigned end = (shardOffset < numShards - 1)
                       ? begin + (multiple * timeStepsPerShard)
                       : params.timeSteps;
    if (begin < end - multiple) {
      auto tExclLast = tBase.slice(begin, end - multiple);
      sequence.push_back(tExclLast);
    }

    // Copy tSingle to the very first tensor. Thereafter copy the last tensor
    // of the previous shard to the first tensor of the current shard
    prog.add(Copy(tLast, tFirst, false,
                  {di, "shiftToRnnShard/" + std::to_string(shardOffset)}));
    tLast = tBase.slice(end - multiple, end);
  }
  auto out = concat(sequence);
  return out;
}

std::vector<Tensor>
Rnn(Graph &graph, const RnnParams &params, bool reverse,
    const std::vector<Tensor> &initState, const StateSequence &stateSequence,
    const std::vector<Tensor> &inputs, const Tensor *interimIn,
    Tensor *interimOut, std::vector<Tensor> &outputs,
    std::vector<Tensor> &created, program::Sequence &initProg,
    const LoopBodyType &loopFn, unsigned numShards,
    poplar::OptionFlags &options, const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params, reverse, initState, inputs, interimIn,
                            interimOut, outputs, numShards, options));

  const auto opt = parseOptions(options);

  // Use VoidFunction for code-reuse if too many shards are used.
  unsigned limitShardingWithoutCodeReuse = 15;
  bool codeReuse = (numShards > limitShardingWithoutCodeReuse) ? true : false;
  if (opt.codeReuse) {
    codeReuse = *opt.codeReuse;
  }
  logging::popnn::debug("'{}': numShards={} code-reuse={} reverse={}",
                        debugContext.getPathName(), numShards, codeReuse,
                        reverse);

  // Create a state tensor in every shard.
  const auto tilesPerShard = getTilesPerShard(graph, params);
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

  // create the forward convolution as a tensor function as we may be able to
  // reuse it for the backwards pass.
  using namespace poputil;
  auto createRnnFunc =
      [&graph, &shard, &initProg, &loopFn](
          const std::vector<poplar::Tensor> &state,
          const std::vector<poplar::Tensor> &inputs,
          const poplar::Tensor &interimIn, const poplar::Tensor &interimOut,
          std::vector<poplar::Tensor> &outputs,
          const std::vector<poplar::Tensor> &created,
          const poplar::DebugNameAndId &dnai) -> graphfn::VoidFunction {
    using graphfn::inout;
    using graphfn::input;
    using graphfn::output;
    const auto recurrence = [&](std::vector<Tensor> &args, Sequence &prog) {
      auto sliceTensors = [](const std::vector<Tensor> &similarTensors,
                             std::vector<Tensor>::iterator &it) {
        std::vector<Tensor> tensors;
        for (auto &s : similarTensors) {
          auto t = s.valid() ? *it++ : s;
          tensors.push_back(t);
        }
        return tensors;
      };
      auto it = args.begin();
      auto index = sliceTensors({shard.indexTensor()}, it)[0];
      auto counter = sliceTensors({shard.counterTensor()}, it)[0];
      auto stateShard = sliceTensors(state, it);
      auto inputSlices = sliceTensors(inputs, it);
      auto interimInSlice = sliceTensors({interimIn}, it)[0];
      auto interimOutSlice = sliceTensors({interimOut}, it)[0];
      auto outputSlices = sliceTensors(outputs, it);
      auto createdSlices = sliceTensors(created, it);
      auto process =
          loopFn(graph, index, counter, stateShard, inputSlices, interimInSlice,
                 interimOutSlice, outputSlices, createdSlices, &initProg, dnai);
      std::copy(createdSlices.begin(), createdSlices.end(),
                args.end() - created.size());
      prog.add(process);
    };

    graphfn::Signature edges;
    auto appendToSignature = [](const std::vector<Tensor> &src,
                                graphfn::ArgType type, graphfn::Signature &sig,
                                const std::string &name) {
      for (auto &s : src) {
        if (s.valid()) {
          auto t = (type == graphfn::ArgType::CreatedArg) ? Tensor() : s;
          sig.emplace_back(graphfn::ArgSig(type, t, name));
        }
      }
    };

    appendToSignature({shard.indexTensor()}, graphfn::InputArg, edges,
                      "shardIndex");
    appendToSignature({shard.counterTensor()}, graphfn::InputArg, edges,
                      "shardCounter");
    appendToSignature(state, graphfn::InOutArg, edges, "state");
    appendToSignature(inputs, graphfn::InputArg, edges, "input");
    appendToSignature({interimIn}, graphfn::InputArg, edges, "interimIn");
    appendToSignature({interimOut}, graphfn::OutputArg, edges, "interimOut");
    appendToSignature(outputs, graphfn::OutputArg, edges, "output");
    appendToSignature(created, graphfn::CreatedArg, edges, "created");
    return {graph, edges, recurrence};
  };

  auto gatherEdges = [&shard](const std::vector<Tensor> &state,
                              const std::vector<Tensor> &inputs,
                              const Tensor &interimIn, const Tensor &interimOut,
                              const std::vector<Tensor> &outputs,
                              std::vector<poplar::Tensor> &created) {
    std::vector<Tensor> edges;
    auto appendTensors = [](const std::vector<Tensor> &src,
                            std::vector<Tensor> &dst, bool created = false) {
      if (created) {
        dst.resize(dst.size() + src.size());
      } else {
        std::copy_if(src.begin(), src.end(), std::back_inserter(dst),
                     [](const auto &t) { return t.valid(); });
      }
    };
    appendTensors({shard.indexTensor()}, edges);
    appendTensors({shard.counterTensor()}, edges);
    appendTensors(state, edges);
    appendTensors(inputs, edges);
    appendTensors({interimIn}, edges);
    appendTensors({interimOut}, edges);
    appendTensors(outputs, edges);
    appendTensors(created, edges, true);
    return edges;
  };

  std::vector<Tensor> stateShard;
  std::vector<Tensor> prevStateShard = initState;

  std::optional<graphfn::VoidFunction> rnnFunc;

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

    initProg.add(initCounter);
    auto loop = Sequence{{}, {di}};
    loop.add(slicer);

    // Call loop builder
    auto counter = shard.counterTensor();
    Sequence process;

    std::vector<poplar::Tensor> createdSlices(created.size());
    if (codeReuse) {
      if (shard.first()) {
        // Create graphfn::VoidFunction only for the first shard. Reuse
        // this function on subsequent shards.
        rnnFunc.emplace(createRnnFunc(stateShard, inputSlices, interimInSlice,
                                      interimOutSlice, outputSlice, created,
                                      {di, "rnnFunction"}));
      }
      auto edges = gatherEdges(stateShard, inputSlices, interimInSlice,
                               interimOutSlice, outputSlice, createdSlices);
      rnnFunc.value()(edges, loop);
      std::copy(edges.end() - createdSlices.size(), edges.end(),
                createdSlices.begin());
    } else {
      // Replicate custom RNN model on every shard.
      auto counter = shard.counterTensor();
      auto index = shard.indexTensor();
      process = loopFn(graph, index, counter, stateShard, inputSlices,
                       interimInSlice, interimOutSlice, outputSlice,
                       createdSlices, (shard.first() ? &initProg : nullptr),
                       {di, "shard/" + std::to_string(shardIndex)});
      loop.add(process);
    }

    Tensor stateSlice, stateSeqOutput;
    if (stateSequence.output.valid()) {
      stateSlice = stateShard[stateSequence.stateIndex];
      stateSeqOutput = stateSequence.output;
    }

    // Dynamically Update slice of current output shards
    auto updater =
        updateShard(graph, params, shard, interimOutSlice, interimOut,
                    stateSlice, stateSeqOutput, outputSlice, outputs,
                    createdSlices, created, {di, "updateShard"});

    auto updateCounter = shard.updateCounter();

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
