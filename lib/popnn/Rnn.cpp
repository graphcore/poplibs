// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "RnnUtil.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poputil/VertexTemplates.hpp"
#include <boost/optional.hpp>
#include <cassert>
#include <cstdint>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/Rnn.hpp>
#include <popops/Cast.hpp>
#include <popops/Encoding.hpp>
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
  v.insert({"maxTimeSteps", toProfileValue(t.maxTimeSteps)});
  if (t.varTimeSteps.valid()) {
    v.insert({"varTimeSteps", toProfileValue(t.varTimeSteps)});
  }
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
    unsigned sequenceLength, unsigned maxShards,
    boost::optional<unsigned> sequencesPerShard,
    boost::optional<unsigned> sequenceMultiple,
    boost::optional<unsigned> shardOffset, unsigned numGrains,
    unsigned grainSize, const poplar::DebugNameAndId &dnai) {
  const auto grainsPerTile = ceildiv(numGrains, numTiles);
  const auto numUsedTiles = ceildiv(numGrains, grainsPerTile);
  const auto grainsOnLastTile = numGrains - (numUsedTiles - 1) * grainsPerTile;
  auto sOffset = shardOffset ? *shardOffset : 0;
  auto numTensorShards = shardOffset ? 1 : maxShards;
  auto seqLengthExceptLastTile =
      sequencesPerShard.value_or(ceildiv(sequenceLength, maxShards));
  auto seqLengthLastTile = sequencesPerShard.value_or(
      sequenceLength - seqLengthExceptLastTile * (maxShards - 1));
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
                    boost::optional<unsigned> sequencesPerShard,
                    boost::optional<unsigned> sequenceMultiple,
                    boost::optional<unsigned> shardIndex, unsigned numShards,
                    const DebugNameAndId &dnai) {
  // TODO: T12909 Take output grouping from matmul operation.
  auto [grouping, numGroups] = getNumGrains(size);
  Tensor t = createMultiDynamicSliceTensor(
      graph, params.dataType, tilesPerShard, params.maxTimeSteps, numShards,
      sequencesPerShard, sequenceMultiple, shardIndex,
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
  return createShardedTensor(graph, params, tilesPerShard, size, 1U, multiple,
                             {}, numShards, {dnai, "singleShard"});
}

static Tensor createTensorShard(Graph &graph, const RnnParams &params,
                                std::size_t tilesPerShard, unsigned size,
                                unsigned multiple, unsigned shardOffset,
                                unsigned stepsPerShard, unsigned numShards,
                                const DebugNameAndId &dnai) {
  return createShardedTensor(graph, params, tilesPerShard, size, stepsPerShard,
                             multiple, shardOffset, numShards,
                             {dnai, "singleShard"});
}

static Tensor createRnnTensor(Graph &graph, const RnnParams &params,
                              std::size_t tilesPerShard, unsigned size,
                              unsigned numShards, const DebugNameAndId &dnai) {
  return createShardedTensor(graph, params, tilesPerShard, size, {}, {}, {},
                             numShards, {dnai, "RnnTensor"});
}

static Tensor createRnnTensor(Graph &graph, const RnnParams &params,
                              std::size_t tilesPerShard, unsigned size,
                              unsigned multiple, unsigned numShards,
                              const DebugNameAndId &dnai) {
  return createShardedTensor(graph, params, tilesPerShard, size, {}, multiple,
                             {}, numShards, {dnai, "RnnTensor"});
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
  unsigned multiple = allOutput.dim(0) / params.maxTimeSteps;
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

struct ShardIndexLimit {
  Tensor exclLastShard;
  Tensor lastShard;
};

static void validateTimeSteps(const Tensor &timeSteps, std::size_t batchSize) {
  // if timeSteps tensor is invalid, use maxTimeSteps for the RNN
  if (!timeSteps.valid()) {
    return;
  }
  if (timeSteps.rank() != 1) {
    throw poputil::poplibs_error("Invalid RNN timeSteps tensor (rank != 1)");
  }
  if (timeSteps.dim(0) != 1 && timeSteps.dim(0) != batchSize) {
    throw poputil::poplibs_error("Invalid RNN timeSteps tensor "
                                 "(size != batchSize) && (size != 1)");
  }
}

class RnnState {
public:
  RnnState(Graph &graph, unsigned seqLen, unsigned numShards, bool reverse,
           boost::optional<unsigned> &stepsPerGather,
           poputil::PoplibsOpDebugInfo &dnai);
  // Step interval used for current shard
  const Interval &interval() const { return currInterval; };

  // Current shard index
  unsigned index() const { return currIndex; };

  // Sequence length of current shard
  unsigned sequenceLength() const { return currInterval.size(); };

  // The number of steps to be accumulated before calling the `Gather` callbck.
  unsigned gatherSize() const;

  // The starting step offset of the current shard
  const Tensor &startIndex() const { return timeStepBegin[currIndex]; };

  // The step offset for the current shard
  const Tensor &counterTensor() const { return timeStepCounter; };

  // If variable time steps is used, get the variable time steps for the
  // current shard
  const Tensor indexTensor() const { return timeSteps[currIndex]; };

  // The step offset within a `Gather` interval
  const Tensor &counterGatherOffset() const { return gatherOffset; };

  // Get required maximum steps interval between calls to `Gather`
  const unsigned gatherCadence() const;

  // Method used to configure variable time steps
  program::Sequence useVariableTimeSteps(const Tensor &numTimeSteps);

  // Initialise step counter
  program::Sequence initCounter();

  // Update Step Counter
  program::Sequence updateCounter();

  // Initialise gather offset counter
  program::Sequence initGatherOffset(bool initialization) const;

  // Update gather offset counter
  program::Sequence updateGatherOffset();

  bool isReverse() const { return reverse; };
  unsigned operator()() const { return counter; };
  bool first() const { return counter == numShards; };
  void next();

private:
  Graph &graph;
  std::vector<Tensor> timeStepBegin;
  Tensor one;

  // Counter which increments every RNN iteration
  Tensor timeStepCounter;

  // The number of time steps for each shard, used in the repeat loop
  Tensor timeSteps;

  Tensor gatherOffset;
  ShardIndexLimit stepLimit;
  Tensor gatherOffsetLimit;
  Tensor gatherOffsetLimitMinus1;

  unsigned seqLengthExclLast;
  unsigned fullSequenceLen;
  unsigned numShards;

  // shard counter
  unsigned counter;

  // Use tensor for the number of time steps
  bool variableTimeSteps;

  bool reverse;
  boost::optional<unsigned> stepsPerGather;
  poputil::PoplibsOpDebugInfo &dnai;

  unsigned currIndex;
  Interval currInterval;
};

RnnParams::RnnParams(poplar::Type dataType, std::size_t batchSize,
                     std::size_t timeSteps, std::vector<std::size_t> layers)
    : dataType(dataType), batchSize(batchSize), maxTimeSteps(timeSteps),
      timeSteps(timeSteps), layerSizes(layers) {}

RnnParams::RnnParams(poplar::Type dataType, std::size_t batchSize,
                     std::size_t maxTimeSteps,
                     const poplar::Tensor &varTimeSteps,
                     std::vector<std::size_t> layers)
    : dataType(dataType), batchSize(batchSize), maxTimeSteps(maxTimeSteps),
      timeSteps(maxTimeSteps), varTimeSteps(varTimeSteps), layerSizes(layers) {
  validateTimeSteps(varTimeSteps, batchSize);
}

RnnState::RnnState(Graph &graph, unsigned seqLen, unsigned numShards,
                   bool reverse, boost::optional<unsigned> &stepsPerGather,
                   poputil::PoplibsOpDebugInfo &dnai)
    : graph(graph), timeStepBegin(numShards), fullSequenceLen(seqLen),
      numShards(numShards), counter(numShards), variableTimeSteps(false),
      reverse(reverse), stepsPerGather(stepsPerGather), dnai(dnai) {

  // loop counter
  timeStepCounter =
      graph.addVariable(UNSIGNED_INT, {1}, {dnai, "timeStepCounter"});
  graph.setTileMapping(timeStepCounter, 0);

  one = graph.addConstant(UNSIGNED_INT, {1}, 1, {dnai, "one"});
  graph.setTileMapping(one, 0);

  seqLengthExclLast = ceildiv(seqLen, numShards);
  for (unsigned i = 0; i < numShards; ++i) {
    timeStepBegin[i] =
        graph.addConstant(UNSIGNED_INT, {1}, (i * seqLengthExclLast),
                          {dnai, "timeStepBegin/" + std::to_string(i)});
    graph.setTileMapping(timeStepBegin[i], 0);
  }

  if (reverse) {
    unsigned seqLengthLast = seqLen - (seqLengthExclLast * (numShards - 1));

    auto createLimit = [&](unsigned exclLast, unsigned last) {
      ShardIndexLimit limit;
      limit.exclLastShard =
          graph.addConstant(UNSIGNED_INT, {1}, exclLast, {dnai, "start"});
      graph.setTileMapping(limit.exclLastShard, 0);
      if (last == exclLast) {
        limit.lastShard = limit.exclLastShard;
      } else {
        limit.lastShard = graph.addConstant(UNSIGNED_INT, {1}, last,
                                            {dnai, "limit.lastShard"});
        graph.setTileMapping(limit.lastShard, 0);
      }
      return limit;
    };

    stepLimit = createLimit(seqLengthExclLast - 1, seqLengthLast - 1);
    if (stepsPerGather) {
      auto gatherCadenceExclLast = std::min(*stepsPerGather, seqLengthExclLast);
      if ((*stepsPerGather > 1) && (*stepsPerGather < seqLen)) {
        gatherOffset = graph.addVariable(UNSIGNED_INT, {1}, {dnai, "step"});
        graph.setTileMapping(gatherOffset, 0);

        // The very first time `gatherOffset` is initialised to
        // `gatherOffsetLimit - 1`. Subsequently gatherOffset must be
        // periodically reinitialised to `gatherOffsetLimit` since it is
        // closely followed by a decrementing of the gatherOffset at tne end of
        // the loop.
        gatherOffsetLimit = graph.addConstant(
            UNSIGNED_INT, {1}, gatherCadenceExclLast, {dnai, "gatherOffset"});
        graph.setTileMapping(gatherOffsetLimit, 0);
        gatherOffsetLimitMinus1 =
            graph.addConstant(UNSIGNED_INT, {1}, gatherCadenceExclLast - 1,
                              {dnai, "gatherOffsetMinus1"});
        graph.setTileMapping(gatherOffsetLimitMinus1, 0);
      }
    }
  }

  currIndex = reverse ? (numShards - 1) : 0;
  unsigned intervalBegin = currIndex * seqLengthExclLast;
  unsigned intervalEnd = reverse ? fullSequenceLen : seqLengthExclLast;
  currInterval = Interval{intervalBegin, intervalEnd};
}

unsigned RnnState::gatherSize() const {
  return stepsPerGather ? std::min(*stepsPerGather, sequenceLength())
                        : interval().size();
}

std::size_t RnnParams::getMaxShards(const poplar::Graph &graph) const {
  auto numTiles = graph.getTarget().getNumTiles();
  auto tilesPerShard = getTilesPerShard(graph, *this);
  auto maxShards = numTiles / tilesPerShard;
  auto nTimeStepsPerShard = ceildiv(maxTimeSteps, maxShards);
  return ceildiv(maxTimeSteps, nTimeStepsPerShard);
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

bool RnnParams::variableTimeSteps() const { return varTimeSteps.valid(); }

bool RnnParams::batchVariableTimeSteps() const {
  return varTimeSteps.valid() && varTimeSteps.numElements() > 1;
}

void RnnState::next() {
  counter--;
  if (counter == 0) {
    return;
  }
  currIndex += reverse ? -1 : 1;
  auto nextBegin = currIndex * seqLengthExclLast;
  auto nextEnd = (currIndex < numShards - 1) ? nextBegin + seqLengthExclLast
                                             : fullSequenceLen;
  currInterval = Interval{nextBegin, nextEnd};
}

const unsigned RnnState::gatherCadence() const {
  return stepsPerGather ? std::min(*stepsPerGather, sequenceLength())
                        : sequenceLength();
}

program::Sequence
RnnState::useVariableTimeSteps(const Tensor &numTimeStepsPerBatch_) {
  auto prog = Sequence{{}, {dnai, "initCounter"}};

  // Allocate tensor for the number of time steps for all shards.
  timeSteps =
      graph.addVariable(UNSIGNED_INT, {numShards}, {dnai, "timeStepCounter"});
  graph.setTileMapping(timeSteps, 0);

  // find the maximum number of time steps for the whole batch
  auto numTimeStepsPerBatch =
      popops::cast(graph, numTimeStepsPerBatch_, INT, prog);
  auto numTimeSteps = popops::reduce(graph, numTimeStepsPerBatch, {0},
                                     {popops::Operation::MAX, false}, prog);

  // Calculate the number of steps for each shard
  using namespace popops::expr;
  iota(graph, timeSteps, 0U, prog);
  mapInPlace(graph,
             Cast(Max(Min(_2 - (Cast(_1, INT) * expr::Const(seqLengthExclLast)),
                          expr::Const(seqLengthExclLast)),
                      expr::Const(0)),
                  UNSIGNED_INT),
             {timeSteps, numTimeSteps}, prog, {dnai, "timeSteps"});

  variableTimeSteps = true;
  return prog;
}

program::Sequence RnnState::initCounter() {
  auto prog = Sequence{{}, {dnai, "initCounter"}};
  if (reverse) {
    auto start = (currIndex < (numShards - 1)) ? stepLimit.exclLastShard
                                               : stepLimit.lastShard;
    prog.add(Copy(start, timeStepCounter, false, {dnai, "counterEnd"}));
  } else {
    popops::zero(graph, timeStepCounter, prog, {dnai, "counterZero"});
  }
  return prog;
}

program::Sequence RnnState::initGatherOffset(bool initialization) const {
  auto prog = Sequence{{}, {dnai, "initGatherOffset"}};
  assert(reverse || !stepsPerGather);
  if (stepsPerGather && (*stepsPerGather > 1) &&
      (*stepsPerGather < sequenceLength())) {
    Tensor limit = initialization ? gatherOffsetLimitMinus1 : gatherOffsetLimit;
    prog.add(Copy(limit, gatherOffset, false, {dnai, "gatherOffsetEnd"}));
  }

  return prog;
}

// Update time step counters
program::Sequence RnnState::updateCounter() {
  auto prog = Sequence{{}, {dnai, "updateCounter"}};
  if (reverse) {
    subInPlace(graph, timeStepCounter, one, prog, {dnai, "counterDecr"});
  } else {
    addInPlace(graph, timeStepCounter, one, prog, {dnai, "counterIncr"});
  }
  return prog;
}

// Update Gather offset counters
program::Sequence RnnState::updateGatherOffset() {
  auto prog = Sequence{{}, {dnai, "updateGatherOffset"}};
  assert(reverse || !stepsPerGather);
  if (stepsPerGather && (*stepsPerGather > 1) &&
      (*stepsPerGather < sequenceLength())) {
    subInPlace(graph, gatherOffset, one, prog, {dnai, "gatherOffsetDecr"});
  }
  return prog;
}

static std::vector<Tensor>
sliceInputs(Graph &graph, const RnnParams &params,
            const boost::optional<Interval> &interval,
            const std::vector<Tensor> &inputs,
            const boost::optional<Tensor> &counter, unsigned numSlices,
            program::Sequence &prog, const DebugNameAndId &dnai) {
  std::vector<Tensor> inputSlices(inputs.size());
  for (unsigned i = 0; i < inputs.size(); ++i) {
    auto &input = inputs[i];
    if (input.valid()) {
      auto inputShard = interval ? input.slice(*interval) : input;
      if (counter) {
        inputSlices[i] =
            popops::dynamicSlice(graph, inputShard, *counter, {0}, {numSlices},
                                 prog, {dnai, "inputSlice"})
                .flatten(0, 2);
      } else {
        inputSlices[i] = inputShard.flatten(0, 2);
      }
    }
  }
  return inputSlices;
}

// Create a tensor of shape `{m * numInterimOut, batchSize, outputSize}` on a
// given shard where `m` is `stepsPerGather` if gathering is used and `1`
// otherwise.
static Tensor createTempBuffer(Graph &graph, const RnnParams &params,
                               std::size_t tilesPerShard, const RnnState &shard,
                               bool forGather, unsigned numInterimOut,
                               unsigned numShards, program::Sequence &prog,
                               const DebugNameAndId &dnai) {
  auto shardIndex = shard.index();
  auto interval = forGather ? shard.gatherSize() : 1;
  auto outputSize = params.layerSizes[1];
  return createTensorShard(graph, params, tilesPerShard, outputSize,
                           numInterimOut, shardIndex, interval, numShards,
                           {dnai, "interimOutSlice"});
}

static std::vector<Tensor>
sliceOutputs(Graph &graph, const RnnParams &params, const RnnState &shard,
             const std::vector<Tensor> &outputState, unsigned numShards,
             const Tensor &counter, program::Sequence &prog,
             const DebugNameAndId &dnai) {
  std::vector<Tensor> outputSlice;
  for (unsigned i = 0; i < outputState.size(); ++i) {
    auto numOutputs = outputState[i].dim(0) / numShards;
    auto sliceBegin = shard.index();
    auto outputShard = outputState[i]
                           .reshapePartial(0, 1, {numShards, numOutputs})
                           .slice(sliceBegin, sliceBegin + 1);
    Tensor out =
        popops::dynamicSlice(graph, outputShard, counter, {0}, {1}, prog,
                             {dnai, "outputSlice/" + std::to_string(i)})[0];
    outputSlice.push_back(out);
  }
  return outputSlice;
}

static RnnSlice
sliceShard(Graph &graph, const RnnParams &params, std::size_t tilesPerShard,
           const RnnState &shard, const std::vector<Tensor> &inputs,
           const Tensor *interimIn, const unsigned numInterimOut,
           const std::vector<Tensor> &outputs, unsigned numShards,
           program::Sequence &prog, const DebugNameAndId &dnai) {
  RnnSlice slices;
  auto counter = shard.counterTensor();
  auto interval = shard.interval();
  if (inputs.size() > 0) {
    slices.inputs =
        sliceInputs(graph, params, interval, inputs, counter, 1, prog, dnai);
  }
  if (interimIn) {
    slices.interimIn = sliceInputs(graph, params, interval, {*interimIn},
                                   counter, 1, prog, dnai)[0];
  }
  if (numInterimOut) {
    slices.interimOut =
        createTempBuffer(graph, params, tilesPerShard, shard, false,
                         numInterimOut, numShards, prog, dnai);
  }
  if (outputs.size() > 0) {
    slices.outputs = sliceOutputs(graph, params, shard, outputs, numShards,
                                  counter, prog, dnai);
  }
  return slices;
}

static RnnSlice
gatherSlices(Graph &graph, const RnnParams &params, const RnnState &shard,
             const std::vector<Tensor> &inputs, const Tensor &tempBuffer,
             const boost::optional<unsigned> &numGatherSteps,
             program::Sequence &prog, const DebugNameAndId &dnai) {
  assert(!numGatherSteps || (*numGatherSteps <= shard.gatherSize()));
  assert(shard.isReverse());
  RnnSlice slices;
  auto interval = shard.interval();
  auto gatherSize = numGatherSteps ? *numGatherSteps : shard.gatherSize();
  boost::optional<Tensor> counter(boost::none);

  // Gather starting from the counter if the gatherSize does not exceed the
  // sharding interval size. Otherwise just gather the entire sharded input.
  if (gatherSize < interval.size()) {
    counter = shard.counterTensor();
  }
  if (inputs.size() > 0) {
    slices.inputs = sliceInputs(graph, params, interval, inputs, counter,
                                gatherSize, prog, dnai);
  }

  // If the user provides `numGatherSteps`, the useful part of the tempBuffer
  // are the last `numGatherSteps` slices of the tensor since these are the
  // slices which will be written to when `reverse=true`. This is possible since
  // gathering is only supported with `reverse=true`.
  if (numGatherSteps && (*numGatherSteps != shard.gatherSize())) {
    auto numUnits = tempBuffer.dim(0) / shard.gatherSize();
    auto offset = (shard.gatherSize() - *numGatherSteps) * numUnits;
    slices.interimIn = tempBuffer.slice(offset, tempBuffer.dim(0));
  } else {
    slices.interimIn = tempBuffer;
  }
  return slices;
}

static void updateTensor(Graph &graph, const RnnParams &params, const Tensor &t,
                         const Tensor &s, const Tensor counter,
                         const boost::optional<Interval> &interval,
                         program::Sequence &prog, const DebugNameAndId &dnai) {
  assert((t.rank() == s.rank()) || (t.rank() == s.rank() + 1));
  auto tRecurrent = t;
  if (t.rank() == s.rank()) {
    auto tensorsPerShard = s.dim(0);
    tRecurrent =
        t.reshapePartial(0, 1, {t.dim(0) / tensorsPerShard, tensorsPerShard});
  }
  Tensor tSharded = interval ? tRecurrent.slice(*interval) : tRecurrent;
  popops::dynamicUpdate(graph, tSharded, s.expand({0}), counter, {0}, {1}, prog,
                        {dnai, "updateTensor"});
}

static void updateTensor(Graph &graph, const RnnParams &params, const Tensor &t,
                         const Tensor &s, const RnnState &shard,
                         program::Sequence &prog, const DebugNameAndId &dnai) {
  auto counter = shard.counterTensor();
  auto interval = shard.interval();
  updateTensor(graph, params, t, s, counter, interval, prog, dnai);
}

static void updateGather(Graph &graph, const RnnParams &params, const Tensor &t,
                         const Tensor &s, const RnnState &shard,
                         program::Sequence &prog, const DebugNameAndId &dnai) {
  auto gatherOffset = (shard.gatherCadence() < shard.sequenceLength())
                          ? shard.counterGatherOffset()
                          : shard.counterTensor();
  updateTensor(graph, params, t, s, gatherOffset, {}, prog, dnai);
}

static program::Sequence
updateShard(Graph &graph, const RnnParams &params, const RnnState &shard,
            const Tensor &interimOutSlice, const Tensor *interimOut,
            const Tensor &stateSlice, const Tensor &stateSequence,
            const std::vector<Tensor> &outputSlice,
            const std::vector<Tensor> &outputs,
            const std::vector<Tensor> &createdSlice,
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

static program::Sequence
zeroInterimOutShard(Graph &graph, const RnnParams &params,
                    const RnnState &shard, const Tensor &interimOutSlice,
                    const Tensor *interimOut, const DebugNameAndId &dnai) {
  Sequence prog;
  if ((interimOut != nullptr) && interimOut->valid()) {
    popops::zero(graph, interimOutSlice, prog, {dnai, "zeroInterimOut"});
    prog.add(updateShard(graph, params, shard, interimOutSlice, interimOut, {},
                         {}, {}, {}, {}, {}, {dnai}));
  }
  return prog;
}

static program::Sequence zeroOutputShard(Graph &graph, const RnnParams &params,
                                         const RnnState &shard,
                                         const Tensor &stateSlice,
                                         const Tensor &stateSequence,
                                         const DebugNameAndId &dnai) {
  Sequence prog;
  if (stateSequence.valid()) {
    auto zeroState = poputil::duplicate(graph, stateSlice, prog, {dnai});
    popops::zero(graph, zeroState, prog, {dnai, "zeroState"});
    Tensor t;
    prog.add(updateShard(graph, params, shard, t, nullptr, zeroState,
                         stateSequence, {}, {}, {}, {}, dnai));
  }
  return prog;
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
                                 multiple, 0, 1, numShards, {di, "initState"});
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
  unsigned multiple = tBase.dim(0) / params.maxTimeSteps;
  unsigned innermostDim = tBase.dim(rank - 1);
  auto timeStepsPerShard = ceildiv(params.maxTimeSteps, numShards);
  std::vector<Tensor> sequence;
  Tensor tLast = tSingle;
  for (unsigned shardOffset = 0; shardOffset < numShards; ++shardOffset) {
    // Create a tensor on the current shard
    auto tFirst = createTensorShard(
        graph, params, tilesPerShard, innermostDim, multiple, shardOffset, 1,
        numShards, {di, "shard/" + std::to_string(shardOffset)});
    auto tCurrent = tFirst;

    // Retain all tensors in shard except the last
    unsigned begin = shardOffset * multiple * timeStepsPerShard;
    unsigned end = (shardOffset < numShards - 1)
                       ? begin + (multiple * timeStepsPerShard)
                       : params.maxTimeSteps;
    if (begin < end - multiple) {
      auto tExclLast = tBase.slice(begin, end - multiple);
      tCurrent = concat({tCurrent, tExclLast});
    }

    sequence.push_back(tCurrent);

    // Copy tSingle to the very first tensor. Thereafter copy the last tensor
    // of the previous shard to the first tensor of the current shard
    prog.add(Copy(tLast, tFirst, false,
                  {di, "shiftToRnnShard/" + std::to_string(shardOffset)}));
    tLast = tBase.slice(end - multiple, end);
  }
  auto out = concat(sequence);
  return out;
}

static void runGather(Graph &graph, const RnnParams &params,
                      const RnnState &shard, const RnnSlice &slice,
                      const Tensor &tempBuffer,
                      const std::vector<Tensor> &gatherInputs,
                      const GatherBodyType &gatherFn, Sequence &initProg,
                      Sequence &prog, Sequence &postProg,
                      const poplar::DebugNameAndId &dnai) {
  if (!gatherFn) {
    popops::zero(graph, slice.interimOut, prog, {dnai, "zeroGradients"});
  }
  if (shard.gatherCadence() > 1) {
    // Update temporary buffer at every step unless gathering is done every
    // step. In the latter case `slice.interimOut` itself is used as temporary
    // buffer.
    updateGather(graph, params, tempBuffer, slice.interimOut, shard, prog,
                 {dnai});
  }
  Sequence gather;
  auto seqLen = shard.sequenceLength();
  auto gatherInterval = shard.gatherCadence();
  if (gatherFn) {
    auto &fnProg = (gatherInterval < seqLen) ? gather : postProg;

    // Slice `stepsPerGather` steps of the inputs every time that the
    // gatherOffset is found to be zero.
    auto gatherSlice =
        gatherSlices(graph, params, shard, gatherInputs, tempBuffer, {}, fnProg,
                     {dnai, "GatherSlice"});
    fnProg.add(gatherFn(graph, gatherSlice, shard.gatherSize(),
                        (shard.first() ? &initProg : nullptr),
                        {dnai, "Gather/" + std::to_string(shard.index())}));

    // The remaining number of ungathered steps at the very end is processed
    // in the following custom code which gets executed after the loop has
    // completed.
    if (seqLen % gatherInterval != 0) {
      auto remainder = seqLen % gatherInterval;
      auto lastGatherSlice =
          gatherSlices(graph, params, shard, gatherInputs, tempBuffer,
                       remainder, postProg, {dnai, "sliceShard"});
      postProg.add(
          gatherFn(graph, lastGatherSlice, remainder, nullptr,
                   {dnai, "lastGather/" + std::to_string(shard.index())}));
    }
  }
  if ((gatherInterval > 1) && (gatherInterval < seqLen)) {
    gather.add(shard.initGatherOffset(false));
    auto continueToGather =
        popops::cast(graph, shard.counterGatherOffset(), BOOL, prog)
            .reshape({});
    prog.add(If(continueToGather, Sequence(), gather));
  } else {
    prog.add(gather);
  }
}

// Greater than comparison of 1-D tensor of batchwise variable step limits with
// the scalar `stepCounter`. The result is a tensor of `dataType`. The `result`
// and `1 - result` are concatenated along an expanded 0th dimension.
static RnnBatchwiseFlags compareGT(Graph &graph, const RnnParams &params,
                                   const Tensor &stepCounter, Sequence &prog,
                                   const poplar::DebugNameAndId &dnai) {
  using namespace popops::expr;
  auto mask = graph.clone(params.dataType, params.varTimeSteps, {dnai});
  auto maskInv = graph.clone(params.dataType, params.varTimeSteps, {dnai});
  std::string vertexName = "popops::BroadcastScalar1DRelationalOpDualOutput";
  auto vertexClass = poputil::templateVertex(
      vertexName, "popops::expr::BinaryOpType::GREATER_THAN",
      stepCounter.elementType(), params.dataType);
  auto cs = graph.addComputeSet({dnai, "compareGT"});
  auto v = graph.addVertex(cs, vertexClass);
  graph.connect(v["B"], stepCounter.squeeze({0}));
  graph.connect(v["data"], params.varTimeSteps);
  graph.connect(v["out"], mask);
  graph.connect(v["outInv"], maskInv);
  graph.setTileMapping(v, 0);
  prog.add(Execute(cs, {dnai}));
  return {mask, maskInv};
}

static void runGather(Graph &graph, const RnnParams &params,
                      const RnnState &shard, const RnnSlice &slice,
                      const Tensor &tempBuffer, Sequence &prog,
                      const poplar::DebugNameAndId &dnai) {
  Sequence initProg, postProg;
  runGather(graph, params, shard, slice, tempBuffer, {}, {}, initProg, prog,
            postProg, dnai);
}

std::vector<Tensor>
Rnn(Graph &graph, const RnnParams &params, bool reverse,
    const std::vector<Tensor> &initState, const StateSequence &stateSequence,
    const std::vector<Tensor> &inputs, const Tensor *interimIn,
    Tensor *interimOut, const unsigned numTemps,
    const std::vector<Tensor> &outputs, const std::vector<Tensor> &created,
    program::Sequence &initProg, const LoopBodyType &loopFn,
    const std::vector<Tensor> &gatherInputs, const GatherBodyType &gatherFn,
    unsigned numShards, boost::optional<unsigned> stepsPerGather,
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
  logging::popnn::debug(
      "'{}': numShards={} code-reuse={} reverse={} stepsPerGather={} "
      "varTimeSteps={}",
      debugContext.getPathName(), numShards, codeReuse, reverse,
      stepsPerGather ? *stepsPerGather : 0, params.varTimeSteps.valid());

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

  RnnState shard(graph, params.maxTimeSteps, numShards, reverse, stepsPerGather,
                 di);

  if (params.varTimeSteps.valid()) {
    initProg.add(shard.useVariableTimeSteps(params.varTimeSteps));
  }

  // create the forward convolution as a tensor function as we may be able to
  // reuse it for the backwards pass.
  using namespace poputil;
  auto createRnnFunc =
      [&graph, &shard, &initProg,
       &loopFn](const std::vector<poplar::Tensor> &state,
                const RnnBatchwiseFlags &flags, const RnnSlice &similarSlice,
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
      auto index = sliceTensors({shard.startIndex()}, it)[0];
      auto counter = sliceTensors({shard.counterTensor()}, it)[0];
      auto mask = sliceTensors({flags.mask}, it)[0];
      auto maskInv = sliceTensors({flags.inverse}, it)[0];
      auto stateShard = sliceTensors(state, it);
      RnnSlice slice;
      slice.inputs = sliceTensors(similarSlice.inputs, it);
      slice.interimIn = sliceTensors({similarSlice.interimIn}, it)[0];
      slice.interimOut = sliceTensors({similarSlice.interimOut}, it)[0];
      slice.outputs = sliceTensors(similarSlice.outputs, it);
      RnnBatchwiseFlags currFlags = {mask, maskInv};
      auto createdSlices = sliceTensors(created, it);
      auto process = loopFn(graph, index, counter, currFlags, stateShard, slice,
                            createdSlices, &initProg, dnai);
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

    appendToSignature({shard.startIndex()}, graphfn::InputArg, edges,
                      "shardIndex");
    appendToSignature({shard.counterTensor()}, graphfn::InputArg, edges,
                      "shardCounter");
    appendToSignature({flags.mask}, graphfn::InputArg, edges, "mask");
    appendToSignature({flags.inverse}, graphfn::InputArg, edges, "maskInv");
    appendToSignature(state, graphfn::InOutArg, edges, "state");
    appendToSignature(similarSlice.inputs, graphfn::InputArg, edges, "input");
    appendToSignature({similarSlice.interimIn}, graphfn::InputArg, edges,
                      "interimIn");
    appendToSignature({similarSlice.interimOut}, graphfn::OutputArg, edges,
                      "interimOut");
    appendToSignature(similarSlice.outputs, graphfn::OutputArg, edges,
                      "output");
    appendToSignature(created, graphfn::CreatedArg, edges, "created");
    return {graph, edges, recurrence};
  };

  auto gatherEdges = [&shard](const std::vector<Tensor> &state,
                              const RnnBatchwiseFlags &flags,
                              const RnnSlice &slice,
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
    appendTensors({shard.startIndex()}, edges);
    appendTensors({shard.counterTensor()}, edges);
    appendTensors({flags.mask}, edges);
    appendTensors({flags.inverse}, edges);
    appendTensors(state, edges);
    appendTensors(slice.inputs, edges);
    appendTensors({slice.interimIn}, edges);
    appendTensors({slice.interimOut}, edges);
    appendTensors(slice.outputs, edges);
    appendTensors(created, edges, true);
    return edges;
  };

  std::vector<Tensor> stateShard;
  std::vector<Tensor> prevStateShard = initState;

  std::optional<graphfn::VoidFunction> rnnFunc;

  while (shard()) {
    auto shardIndex = shard.index();
    std::string shardStr = std::to_string(shardIndex);

    // Copy over state from previous shard
    stateShard = copyStateShard(prevStateShard, state, shardIndex, initProg,
                                {di, "copyStateShard"});

    auto initCounter = shard.initCounter();
    initProg.add(initCounter);
    initProg.add(shard.initGatherOffset(true));

    auto counter = shard.counterTensor();
    auto index = shard.startIndex();
    auto loop = Sequence{{}, {di}};

    // Dynamically slice from current input shards
    auto process = Sequence{{}, {di}};
    auto slice = sliceShard(graph, params, tilesPerShard, shard, inputs,
                            interimIn, numTemps, outputState, numShards,
                            process, {di, "sliceShard"});

    Tensor tempBuffer;
    if (numTemps && stepsPerGather) {
      // if gather is to be called every step just use the slice tensor as
      // temporary buffer.
      tempBuffer = (*stepsPerGather > 1)
                       ? createTempBuffer(graph, params, tilesPerShard, shard,
                                          true, numTemps, numShards, initProg,
                                          {di, "tempBuffer"})
                       : slice.interimOut;
    }

    // Create a mask if the time steps limit is different across batches
    RnnBatchwiseFlags batchFlags;
    if (params.batchVariableTimeSteps()) {
      auto seqIdxAbsolute =
          (numShards > 1)
              ? add(graph, index, counter, process, {di, "sequenceIndex"})
              : counter;
      batchFlags = compareGT(graph, params, seqIdxAbsolute, process,
                             {di, "batchStepsWithinLimit"});
    }

    // Call `loopFn` callback every time step
    std::vector<poplar::Tensor> createdSlices(created.size());
    if (codeReuse) {
      if (shard.first()) {
        // Create graphfn::VoidFunction only for the first shard. Reuse
        // this function on subsequent shards.
        rnnFunc.emplace(createRnnFunc(stateShard, batchFlags, slice, created,
                                      {di, "rnnFunction"}));
      }
      auto edges = gatherEdges(stateShard, batchFlags, slice, createdSlices);
      rnnFunc.value()(edges, process);
      std::copy(edges.end() - createdSlices.size(), edges.end(),
                createdSlices.begin());
    } else {
      // Replicate custom RNN model on every shard.
      auto index = shard.startIndex();
      process.add(loopFn(graph, index, counter, batchFlags, stateShard, slice,
                         createdSlices, (shard.first() ? &initProg : nullptr),
                         {di, "shard/" + std::to_string(shardIndex)}));
    }

    Tensor stateSlice, stateSeqOutput;
    if (stateSequence.output.valid()) {
      stateSlice = stateShard[stateSequence.stateIndex];
      // If the time steps limit varies across batches and the state slice
      // needs to be recorded for every time step as output, zero out
      // the appropriate batch(es) without zeroing out the state itself.
      if (batchFlags.valid()) {
        stateSlice = map(graph, expr::_1 * expr::_2,
                         {stateSlice, batchFlags.mask.expand({1})}, process,
                         {di, "selectStateToStore"});
      }
      stateSeqOutput = stateSequence.output;
    }

    // Dynamically Update slice of current output shards
    auto updater =
        updateShard(graph, params, shard, slice.interimOut, interimOut,
                    stateSlice, stateSeqOutput, slice.outputs, outputs,
                    createdSlices, created, {di, "updateShard"});
    process.add(updater);

    // If the `Gather` callback exists schedule it every `stepsPerGather` steps.
    // Since gathers are only supported with `reverse`=true the `Gather`
    // callback is called when the GatherOffset counter is 0.
    Sequence postProg;
    if (gatherFn) {
      runGather(graph, params, shard, slice, tempBuffer, gatherInputs, gatherFn,
                initProg, process, postProg, {di});
    }

    // If variable time steps is used, the following code handles the
    // truncation of the the time steps up to the maximum time step limits
    // over all the batches. Further truncation must be carried out using
    // masks in the `loopFn` callback function.
    auto progStep = process;
    if (params.variableTimeSteps()) {
      progStep = Sequence{{}, {di, "progStep"}};

      // Don't do anything if time-step limit is 0
      Tensor checkNotZero =
          neq(graph, shard.indexTensor(), 0U, progStep).reshape({});

      // Determine if time-step is within limit
      Tensor predicate =
          lt(graph, counter, shard.indexTensor(), progStep).reshape({});

      // For iterations which exceed the variable maximum time step limit among
      // all the batches ensure that the output and interimOut tensors are
      // updated if required.
      auto resetProg = zeroOutputShard(graph, params, shard, stateSlice,
                                       stateSeqOutput, {di, "zeroOutputShard"});
      resetProg.add(zeroInterimOutShard(graph, params, shard, slice.interimOut,
                                        interimOut, {di}));
      if (gatherFn) {
        runGather(graph, params, shard, slice, tempBuffer, resetProg, {di});
      }
      // T38265: The outer if-condition is functionally redundant. However it
      // serves to inhibit the SubGraphReplicator from replicating the `If`
      // program. Without it the SubgraphReplicator was found to cause the wrong
      // branch to be taken. When the issue is resolved the outer `If` program
      // can be removed.
      progStep.add(
          If(checkNotZero, If(predicate, process, resetProg), resetProg));
    }

    loop.add(progStep);
    loop.add(shard.updateCounter());
    loop.add(shard.updateGatherOffset());
    initProg.add(Repeat(shard.sequenceLength(), loop, {di}));

    // any processing after the time step loop
    initProg.add(postProg);

    // Hold on to the updated state.
    prevStateShard = stateShard;
    shard.next();
  }
  return stateShard;
}

std::vector<Tensor>
Rnn(Graph &graph, const RnnParams &params, bool reverse,
    const std::vector<Tensor> &initState, const StateSequence &stateSequence,
    const std::vector<Tensor> &inputs, const Tensor *interimIn,
    Tensor *interimOut, const std::vector<Tensor> &outputs,
    const std::vector<Tensor> &created, program::Sequence &initProg,
    const LoopBodyType &loopFn, unsigned numShards,
    poplar::OptionFlags &options, const poplar::DebugContext &debugContext) {
  auto numTemps = interimOut ? interimOut->dim(1) : 0U;
  return Rnn(graph, params, reverse, initState, stateSequence, inputs,
             interimIn, interimOut, numTemps, outputs, created, initProg,
             loopFn, {}, {}, numShards, {}, options, debugContext);
}

std::vector<Tensor>
Rnn(Graph &graph, const RnnParams &params, const std::vector<Tensor> &initState,
    const StateSequence &stateSequence, const std::vector<Tensor> &inputs,
    const Tensor &interimIn, const unsigned numTemps,
    program::Sequence &initProg, const LoopBodyType &loopFn,
    const std::vector<Tensor> &gatherInputs, const GatherBodyType &gatherFn,
    unsigned numShards, unsigned stepsPerGather, poplar::OptionFlags &options,
    const poplar::DebugContext &debugContext) {
  return Rnn(graph, params, true, initState, stateSequence, inputs, &interimIn,
             nullptr, numTemps, {}, {}, initProg, loopFn, gatherInputs,
             gatherFn, numShards, stepsPerGather, options, debugContext);
}

} // namespace rnn
} // namespace popnn
