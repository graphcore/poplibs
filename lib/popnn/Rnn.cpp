// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "RnnUtil.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poputil/VertexTemplates.hpp"
#include <boost/optional.hpp>
#include <cassert>
#include <cstdint>
#include <optional>
#include <poplibs_support/logging.hpp>
#include <popnn/Rnn.hpp>
#include <popops/Cast.hpp>
#include <popops/Encoding.hpp>
#include <popops/Loop.hpp>
#include <poputil/GraphFunction.hpp>

#include <gccs/Algorithm.hpp>

#include <boost/functional/hash.hpp>

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
  auto grainSize = std::gcd(16UL, size);
  auto numGrains = size / grainSize;
  return {grainSize, numGrains};
}

static std::size_t calculatedBytesPerTile(unsigned numTiles,
                                          std::size_t batchSize,
                                          std::size_t size,
                                          std::size_t typeSize) {
  auto [grainSize, numGrains] = getNumGrains(size);
  numGrains *= batchSize;
  auto grainsPerTile = gccs::ceildiv(numGrains, numTiles);
  return grainsPerTile * grainSize * typeSize;
}

static std::size_t
calculateTileUsage(bool output, unsigned numTiles, const std::size_t batchSize,
                   const std::vector<std::size_t> &layerSizes) {
  auto size = output ? layerSizes[1] : layerSizes[0];
  auto [grainSize, numGrains] = getNumGrains(size);
  (void)grainSize;
  numGrains *= batchSize;
  auto grainsPerTile = gccs::ceildiv(numGrains, numTiles);
  auto usedTiles = gccs::ceildiv(numGrains, grainsPerTile);
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
    unsigned sequenceLength, unsigned numShards,
    boost::optional<unsigned> sequencesPerShard,
    boost::optional<unsigned> sequenceMultiple,
    boost::optional<unsigned> shardOffset, unsigned numGrains,
    unsigned grainSize, unsigned seed, const poplar::DebugNameAndId &dnai) {
  const auto grainsPerTile = gccs::ceildiv(numGrains, numTiles);
  const auto numUsedTiles = gccs::ceildiv(numGrains, grainsPerTile);
  auto totalNumTiles = graph.getTarget().getNumTiles();
  const auto maxShards = totalNumTiles / numUsedTiles;
  const auto grainsOnLastTile = numGrains - (numUsedTiles - 1) * grainsPerTile;
  auto sOffset = shardOffset ? *shardOffset : 0;
  auto numTensorShards = shardOffset ? 1 : numShards;
  auto seqLengthExceptLastTile =
      sequencesPerShard.value_or(gccs::ceildiv(sequenceLength, numShards));
  auto seqLengthLastTile = sequencesPerShard.value_or(
      sequenceLength - seqLengthExceptLastTile * (numShards - 1));
  if (sequenceMultiple) {
    seqLengthExceptLastTile *= *sequenceMultiple;
    seqLengthLastTile *= *sequenceMultiple;
  }

  std::vector<poplar::Tensor> tExclLast;
  std::vector<poplar::Tensor> tLast;
  for (unsigned i = sOffset; i < numTensorShards + sOffset; ++i) {
    auto seqLength =
        (i < numShards - 1) ? seqLengthExceptLastTile : seqLengthLastTile;
    tExclLast.push_back(graph.addVariable(
        dataType, {numUsedTiles - 1, seqLength, grainsPerTile, grainSize},
        {dnai, "rnnSlice/" + std::to_string(i)}));
    tLast.push_back(graph.addVariable(dataType,
                                      {seqLength, grainsOnLastTile, grainSize},
                                      {dnai, "rnnSliceLast"}));
  }

  // Tensors for the last sequence
  for (unsigned tileOfShard = 0; tileOfShard != numUsedTiles; ++tileOfShard) {
    for (unsigned shard = sOffset; shard != numTensorShards + sOffset;
         ++shard) {
      // The use of tile 0 could prevent the SubGraphReplicator from replicating
      // the loop counter. For this reason, tiles are mapped beginning from
      // higher numbered tiles.
      auto tile = ((tileOfShard + 1) * maxShards) - 1 - shard;

      // Dither tile allocation
      tile = (tile + seed) % totalNumTiles;

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

  // Seed to dither tile allocation
  auto seed = [](const RnnParams &p) {
    std::size_t seed = 0;
    boost::hash_combine(seed, p.maxTimeSteps);
    boost::hash_combine(seed, p.batchSize);
    boost::hash_combine(seed, p.layerSizes[0]);
    boost::hash_combine(seed, p.layerSizes[1]);
    return seed;
  }(params);
  Tensor t = createMultiDynamicSliceTensor(
      graph, params.dataType, tilesPerShard, params.maxTimeSteps, numShards,
      sequencesPerShard, sequenceMultiple, shardIndex,
      numGroups * params.batchSize, grouping, seed, {dnai, "sharded"});
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

struct RnnInterval {
  poplar::Tensor max;
  poplar::Tensor min;
};

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
  const Tensor timeLimit() const { return timeSteps.max[currIndex]; };

  // The step offset within a `Gather` interval
  const Tensor &counterGatherOffset() const { return gatherOffset; };

  // Get required maximum steps interval between calls to `Gather`
  const unsigned gatherCadence() const;

  // Method used to configure variable time steps
  program::Sequence calculateCounterLimits(const Tensor &numTimeSteps);

  // Initialise step counter
  program::Sequence initCounter();

  // Initialise gather offset counter
  program::Sequence initGatherOffset(bool initialization) const;

  // Update gather offset counter
  program::Sequence updateGatherOffset();

  // Program that updates counters and checks if limit has been reached.
  program::Sequence checkLimit() const { return loopLimitCheck[currIndex]; };

  // Boolean tensor that gets set when loop exit conditon has been reached
  const Tensor &continueFlag() const { return loopContinueFlag; };

  const Tensor &variableSequenceFlag() const { return loopVariableSeqFlag; };

  bool isReverse() const { return reverse; };
  unsigned operator()() const { return counter; };
  bool first() const { return counter == numShards; };
  void next();

private:
  Graph &graph;
  std::vector<Tensor> timeStepBegin;
  Tensor one;
  Tensor minusOne;

  // Counter which increments every RNN iteration
  Tensor timeStepCounter;

  // The number of time steps for each shard, used in the repeat loop
  RnnInterval timeSteps;

  Tensor gatherOffset;
  Tensor gatherCadenceLen;
  Tensor gatherCadenceLenInitial;

  unsigned seqLengthExclLast;
  unsigned fullSequenceLen;
  unsigned numShards;

  // shard counter
  unsigned counter;

  Tensor loopContinueFlag;
  Tensor loopVariableSeqFlag;
  std::vector<program::Sequence> loopLimitCheck;

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
      numShards(numShards), counter(numShards), loopLimitCheck(numShards),
      variableTimeSteps(false), reverse(reverse),
      stepsPerGather(stepsPerGather), dnai(dnai) {

  // loop counter
  timeStepCounter =
      graph.addVariable(UNSIGNED_INT, {1}, {dnai, "timeStepCounter"});
  graph.setTileMapping(timeStepCounter, 0);

  one = graph.addConstant(UNSIGNED_INT, {1}, 1, {dnai, "one"});
  graph.setTileMapping(one, 0);

  minusOne = graph.addConstant(UNSIGNED_INT, {1}, -1, {dnai, "minusOne"});
  graph.setTileMapping(minusOne, 0);

  seqLengthExclLast = gccs::ceildiv(seqLen, numShards);
  for (unsigned i = 0; i < numShards; ++i) {
    timeStepBegin[i] =
        graph.addConstant(UNSIGNED_INT, {1}, (i * seqLengthExclLast),
                          {dnai, "timeStepBegin/" + std::to_string(i)});
    graph.setTileMapping(timeStepBegin[i], 0);
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
  auto nTimeStepsPerShard = gccs::ceildiv(maxTimeSteps, maxShards);
  return gccs::ceildiv(maxTimeSteps, nTimeStepsPerShard);
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
RnnState::calculateCounterLimits(const Tensor &numTimeStepsPerBatch_) {
  auto prog = Sequence{{}, {dnai, "initCounter"}};

  using namespace popops::expr;
  if (numTimeStepsPerBatch_.valid()) {
    timeSteps.max =
        graph.addVariable(UNSIGNED_INT, {numShards}, {dnai, "timeStepCounter"});
    graph.setTileMapping(timeSteps.max, 0);

    // find the minimum and maximum time steps over the batch
    RnnInterval seqLenLimitVar;
    auto numTimeStepsPerBatch =
        popops::cast(graph, numTimeStepsPerBatch_, INT, prog);
    seqLenLimitVar.max = popops::reduce(graph, numTimeStepsPerBatch, {0},
                                        {popops::Operation::MAX, false}, prog,
                                        {dnai, "MaxVarSequenceLength"});
    seqLenLimitVar.min = popops::reduce(graph, numTimeStepsPerBatch, {0},
                                        {popops::Operation::MIN, false}, prog,
                                        {dnai, "MinVarSequenceLength"});

    // Calculate the minimum and maximum limits per shard.
    if (numShards > 1) {
      auto shardIndex = graph.clone(INT, timeSteps.max);
      iota(graph, shardIndex, 0, prog);
      mulInPlace(graph, shardIndex, static_cast<int>(seqLengthExclLast), prog,
                 {dnai, "shardedSequenceStart"});
      mapInPlace(graph,
                 Cast(Max(Min(_3 - _2, expr::Const(seqLengthExclLast)),
                          expr::Const(0)),
                      UNSIGNED_INT),
                 {timeSteps.max, shardIndex, seqLenLimitVar.max}, prog,
                 {dnai, "timeSteps.max"});
      timeSteps.min = graph.clone(timeSteps.max);
      mapInPlace(graph,
                 Cast(Max(Min(_3 - _2, expr::Const(seqLengthExclLast)),
                          expr::Const(0)),
                      UNSIGNED_INT),
                 {timeSteps.min, shardIndex, seqLenLimitVar.min}, prog,
                 {dnai, "timeSteps.min"});
    } else {
      timeSteps.max = cast(graph, seqLenLimitVar.max.expand({0}), UNSIGNED_INT,
                           prog, {dnai, "timeSteps.max"});
      timeSteps.min = cast(graph, seqLenLimitVar.min.expand({0}), UNSIGNED_INT,
                           prog, {dnai, "timeSteps.min"});
    }
    variableTimeSteps = true;
  } else {
    // Use constant for the counter upper limit wherever possible to help
    // sub-graph replication.
    std::vector<unsigned> limits(numShards, seqLengthExclLast);
    limits[numShards - 1] =
        fullSequenceLen - (numShards - 1) * seqLengthExclLast;
    timeSteps.max = graph.addConstant(UNSIGNED_INT, {numShards}, limits.data(),
                                      {dnai, "timeSteps.max"});
    graph.setTileMapping(timeSteps.max, 0);
  }

  if (stepsPerGather) {
    unsigned seqLengthLast =
        fullSequenceLen - (seqLengthExclLast * (numShards - 1));
    auto gatherCadenceExclLast = std::min(*stepsPerGather, seqLengthExclLast);
    auto gatherCadenceLast = std::min(*stepsPerGather, seqLengthLast);
    if ((*stepsPerGather > 1) && (*stepsPerGather < fullSequenceLen)) {
      // The very first time `gatherOffset` is initialised to
      // `gatherCadenceLen - 1`. Subsequently gatherOffset must be
      // periodically reinitialised to `gatherCadenceLen` since it is
      // closely followed by a decrementing of the gatherOffset at tne end of
      // the loop.
      std::vector<unsigned> shardGatherLen(numShards, gatherCadenceExclLast);
      shardGatherLen[numShards - 1] = gatherCadenceLast;
      gatherCadenceLen =
          graph.addConstant(UNSIGNED_INT, {numShards}, shardGatherLen.data(),
                            {dnai, "gatherLength"});
      graph.setTileMapping(gatherCadenceLen, 0);

      std::vector<unsigned> shardGatherLenMinus1(numShards,
                                                 gatherCadenceExclLast - 1);
      shardGatherLenMinus1[numShards - 1] = gatherCadenceLast - 1;
      auto gatherCadenceLenMinus1 = graph.addConstant(
          UNSIGNED_INT, {numShards}, shardGatherLenMinus1.data(),
          {dnai, "gatherLength"});
      graph.setTileMapping(gatherCadenceLenMinus1, 0);

      if (numTimeStepsPerBatch_.valid()) {
        std::vector<unsigned> shardFixedRemainder(
            numShards, seqLengthExclLast % *stepsPerGather);
        shardFixedRemainder[numShards - 1] = seqLengthLast % *stepsPerGather;
        auto fixedRemainder = graph.addConstant(UNSIGNED_INT, {numShards},
                                                shardFixedRemainder.data(),
                                                {dnai, "fixedRemainder"});
        graph.setTileMapping(fixedRemainder, 0);
        auto remainder =
            map(graph, Select(_1 + expr::Const(1), (_1 - _2) % _3, Lt(_1, _2)),
                {timeSteps.max, fixedRemainder, gatherCadenceLen}, prog,
                {dnai, "gatherOffsetInitial"});
        gatherCadenceLenInitial =
            map(graph, Select(_1 - expr::Const(1), _2, Cast(_1, BOOL)),
                {remainder, gatherCadenceLenMinus1}, prog,
                {dnai, "gatherOffsetInitial"});
      } else {
        gatherCadenceLenInitial = gatherCadenceLenMinus1;
      }

      gatherOffset = graph.addVariable(UNSIGNED_INT, {1}, {dnai, "step"});
      graph.setTileMapping(gatherOffset, 0);
    }
  }
  return prog;
}

program::Sequence RnnState::initCounter() {
  auto prog = Sequence{{}, {dnai, "initCounter"}};
  if (reverse) {
    prog.add(Copy(timeSteps.max[currIndex], timeStepCounter, false,
                  {dnai, "counterEnd"}));
    loopContinueFlag = addForLoopCounterVertex(graph, timeStepCounter, minusOne,
                                               -1, 0, loopLimitCheck[currIndex],
                                               {dnai, "loopCounter"});
    if (timeSteps.min.valid()) {
      loopVariableSeqFlag =
          gteq(graph, timeStepCounter, timeSteps.min[currIndex],
               loopLimitCheck[currIndex], {dnai, "loopPaddingLimit"})
              .reshape({});
    }
  } else {
    prog.add(Copy(minusOne, timeStepCounter, false, {dnai, "counterEnd"}));
    loopContinueFlag = addForLoopCounterVertex(
        graph, timeStepCounter, timeSteps.max[currIndex], 1, 0,
        loopLimitCheck[currIndex], {dnai, "loopCounter"});
    if (timeSteps.min.valid()) {
      loopVariableSeqFlag =
          gteq(graph, timeStepCounter, timeSteps.min[currIndex],
               loopLimitCheck[currIndex], {dnai, "loopPaddingLimit"})
              .reshape({});
    }
  }
  return prog;
}

program::Sequence RnnState::initGatherOffset(bool initialization) const {
  auto prog = Sequence{{}, {dnai, "initGatherOffset"}};
  assert(reverse || !stepsPerGather);
  if (stepsPerGather && (*stepsPerGather > 1) &&
      (*stepsPerGather < sequenceLength())) {
    Tensor limit = initialization ? gatherCadenceLenInitial[currIndex]
                                  : gatherCadenceLen[currIndex];
    prog.add(Copy(limit, gatherOffset, false, {dnai, "gatherOffsetEnd"}));
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
    if (counter && numGatherSteps && (*numGatherSteps != shard.gatherSize())) {
      // Reset the counter to point to the beginning of the final sequence of
      // gather steps, which should be at offset zero when "reverse=true".
      zero(graph, *counter, prog, {dnai, "zeroGatherOffsetFinal"});
    }
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
  unsigned multiple = tBase.dim(0) / params.maxTimeSteps;
  auto timeStepsPerShard = gccs::ceildiv(params.maxTimeSteps, numShards);
  std::vector<Tensor> sequence;
  Tensor tLast = tSingle;
  for (unsigned shardOffset = 0; shardOffset < numShards; ++shardOffset) {
    unsigned begin = shardOffset * multiple * timeStepsPerShard;
    unsigned end = (shardOffset < numShards - 1)
                       ? begin + (multiple * timeStepsPerShard)
                       : params.maxTimeSteps;
    auto shard = graph.clone(tBase.slice(begin, end));
    prog.add(Copy(tLast, shard.slice(0, multiple), false, {di}));
    prog.add(Copy(tBase.slice(begin, end - multiple),
                  shard.slice(multiple, end - begin), false, {di}));
    sequence.push_back(shard);
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
        gatherSlices(graph, params, shard, gatherInputs, tempBuffer, remainder,
                     postProg, {dnai, "sliceShard"});
    postProg.add(
        gatherFn(graph, lastGatherSlice, remainder, nullptr,
                 {dnai, "lastGather/" + std::to_string(shard.index())}));
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

  initProg.add(shard.calculateCounterLimits(params.varTimeSteps));

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
      auto varSeqFlag = sliceTensors({shard.variableSequenceFlag()}, it)[0];
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
      const TimeStepState timeState = {index, counter, varSeqFlag};
      auto process = loopFn(graph, timeState, currFlags, stateShard, slice,
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
    appendToSignature({shard.variableSequenceFlag()}, graphfn::InputArg, edges,
                      "shardVariableSeqFlag");
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
    appendTensors({shard.variableSequenceFlag()}, edges);
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

    if (!params.variableTimeSteps()) {
      loop.add(shard.checkLimit());
    }

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
      assert(reverse);

      if (*stepsPerGather > 1) {
        // The gathering temp buffer is partially written to at every time step
        // but fully read out periodically by `gatherFn`. Undefine the tensor
        // before the repeat loop.
        initProg.add(WriteUndef(tempBuffer));

        if (params.variableTimeSteps()) {
          // The first shard in the reverse direction may have to run fewer
          // iterations than `stepsPerGather`. The gathered intermediate buffer
          // needs to be zeroed for this case.
          zero(graph, tempBuffer, initProg, {di, "zeroGatherBuffer"});
        }
      }
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
      const TimeStepState timeState = {index, counter,
                                       shard.variableSequenceFlag()};
      process.add(loopFn(graph, timeState, batchFlags, stateShard, slice,
                         createdSlices, (shard.first() ? &initProg : nullptr),
                         {di, "shard/" + std::to_string(shardIndex)}));
    }

    Tensor stateSlice, stateSeqOutput;
    if (stateSequence.output.valid()) {
      stateSlice = stateShard[stateSequence.stateIndex];
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
    loop.add(process);
    loop.add(shard.updateGatherOffset());
    if (params.variableTimeSteps()) {
      initProg.add(RepeatWhileTrue(shard.checkLimit(), shard.continueFlag(),
                                   loop, {di}));
    } else {
      //      loop.add(shard.checkLimit());
      initProg.add(Repeat(shard.sequenceLength(), loop, {di}));
    }

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
