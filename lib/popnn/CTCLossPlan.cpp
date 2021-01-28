// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"

#include "PerformanceEstimation.hpp"

#include <poplibs_support/Memoize.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/CTCLoss.hpp>
#include <popsolver/Model.hpp>
#include <poputil/OptionParsing.hpp>

namespace popnn {
namespace ctc {

struct PartitionVariables {
  ParallelPartition<popsolver::Variable, popsolver::Variable> parallel;
  SerialPartition<popsolver::Variable> serial;
};

struct CtcParams {
  poplar::Type inType;
  poplar::Type outType;
  unsigned batchSize;
  unsigned maxTime;
  unsigned maxLabelLength;
  unsigned numClasses;
};

std::ostream &operator<<(std::ostream &o, const CtcParams &p) {
  o << "CTCLoss params:\n";
  o << "  inType                       " << p.inType << "\n";
  o << "  outType                      " << p.outType << "\n";
  o << "  batchSize                    " << p.batchSize << "\n";
  o << "  maxTime                      " << p.maxTime << "\n";
  o << "  maxLabelLength               " << p.maxLabelLength << "\n";
  o << "  numClasses                   " << p.numClasses << "\n";
  return o;
}

struct CtcOpts {
  boost::optional<double> availableMemoryProportion;
};

static CtcOpts parseOptions(const poplar::OptionFlags &options) {
  CtcOpts opts;
  const poplibs::OptionSpec spec{
      {"availableMemoryProportion", poplibs::OptionHandler::createWithDouble(
                                        opts.availableMemoryProportion)},
  };
  for (const auto &entry : options) {
    spec.parse(entry.first, entry.second);
  }
  return opts;
}

std::ostream &operator<<(std::ostream &o, const CtcOpts &opt) {
  o << "CTCLoss options:\n";
  o << "  availableMemoryProportion    "
    << (opt.availableMemoryProportion
            ? std::to_string(*opt.availableMemoryProportion)
            : "None")
    << "\n";
  return o;
}

struct EstimateCache {
  decltype(poplibs_support::memoize(alphaCycles)) mAlphaCycles;
  decltype(poplibs_support::memoize(betaCycles)) mBetaCycles;
  decltype(
      poplibs_support::memoize(gradGivenAlphaCycles)) mGradGivenAlphaCycles;
  decltype(poplibs_support::memoize(gradGivenBetaCycles)) mGradGivenBetaCycles;
  EstimateCache()
      : mAlphaCycles(alphaCycles), mBetaCycles(betaCycles),
        mGradGivenAlphaCycles(gradGivenAlphaCycles),
        mGradGivenBetaCycles(gradGivenBetaCycles) {}
};

std::ostream &operator<<(std::ostream &o, const Plan::Impl &p) {
  o << "CTCLoss plan:\n";
  o << "  Serial Partition:\n";
  o << "    batch                      " << p.serial.batch << "\n";
  o << "    time                       " << p.serial.time << "\n";
  o << "    label                      " << p.serial.label << "\n";
  o << "  Parallel Partition:\n";
  o << "    batch                      " << p.parallel.batch << "\n";
  o << "    time                       " << p.parallel.time << "\n";
  o << "    label                      " << p.parallel.label << "\n";
  o << "    sliceIntoOutput            "
    << (p.parallel.sliceIntoOutput ? "true" : "false") << "\n";
  o << "    alphabet                   " << p.parallel.alphabet << "\n";
  o << "    sliceFromInput             "
    << (p.parallel.sliceFromInput ? "true" : "false") << "\n";
  o << "  Total:\n";
  o << "    tiles                      " << p.numTiles() << "\n";
  return o;
}
std::ostream &operator<<(std::ostream &o, const CycleEstimate &e) {
  o << "Estimated cycles:\n";
  o << "  Alpha/Beta:\n";
  o << "    compute                    " << e.alphaBetaComputeCycles << "\n";
  o << "    exchange                   " << e.alphaBetaExchangeCost << "\n";
  o << "  Grad given Alpha/Beta:\n";
  o << "    compute                    " << e.gradComputeCycles << "\n";
  o << "    exchange                   " << e.gradExchangeCost << "\n";
  o << "  Total:\n";
  o << "    steps                      " << e.steps << "\n";
  if (e.serialVertexExecutions > 1) {
    o << "    serial vertex executions\n"
         "      per step                 "
      << e.serialVertexExecutions << "\n";
  }
  o << "    cycles                     " << e.total() << "\n";
  return o;
}

/*
  We can illustrate the process of splits in time and label as a diagonal
  wavefront propagating across partitions. This is because for any given
  partition, it requires the dependencies to be executed first. For alpha,
  partitions marked as "B", require the partition marked "A" to have first been
  executed. Considering the following example where we split time and the
  extended label into 4 partitions:

       t
   |0|1|2|3|
  -+-------+
  0|A|B|C|D|
  -+-------+
  1|B|C|D|E|
El-+-------+
  2|C|D|E|F|
  -+-------+
  3|D|E|F|G|
  -+-------+

  Then to complete the operation we have a sequence of steps like the following:

  0: alpha{A}, beta{G}
  1: alpha{B}, beta{F}
  2: alpha{C}, beta{E}
  3: alpha{D} [*]
  4: gradGivenBeta{E}, gradGivenAlpha{D}
  5: gradGivenBeta{F}, gradGivenAlpha{C}
  6: gradGivenBeta{G}, gradGivenAlpha{B}
  7: gradGivenAlpha{A}

  [*] We satisfy the dependencies to compute both alpha and beta at this point,
      however we don't do so as we would need double the temporary memory to
      keep both alpha and beta in memory concurrently when calculating the
      gradient. To instead not increase temporary memory, we arbitrarily pick
      alpha or beta to calculate. In this example we choose alpha, but it is
      just as valid to choose beta. It's worth noting that if we had chosen to
      split El or t into an odd number of partitions, we wouldn't encounter
      this, it's only when they are both even or both odd.
*/
CycleEstimate estimateCycles(const CtcParams &params,
                             const Plan::Impl &partition,
                             const poplar::Target &target,
                             EstimateCache &cache) {

  auto steps = partition.parallel.time + partition.parallel.label - 1;
  if ((partition.parallel.time & 1) == (partition.parallel.label & 1)) {
    // Represents the stall when both alpha and beta are
    // available to be computed at the same step
    steps += 1;
  }
  assert((steps & 1) == 0); // Implicit from above logic

  auto maxBatchPerTile =
      poplibs_support::ceildiv(params.batchSize, partition.parallel.batch);

  // Currently we use 1 worker per batch, noting this is only valid while using
  // worker and not supervisor vertices.
  auto serialVertexExecutionsPerStep =
      poplibs_support::ceildiv(maxBatchPerTile, 6U);

  const auto alphaOrBetaSteps = serialVertexExecutionsPerStep * (steps / 2);
  const auto gradGivenAlphaOrBetaSteps =
      serialVertexExecutionsPerStep * (steps / 2);
  const unsigned exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  auto maxLabelElementsPerTile =
      poplibs_support::ceildiv(params.maxLabelLength, partition.parallel.label);
  auto maxTimeElementsPerTile =
      poplibs_support::ceildiv(params.maxTime, partition.parallel.time);
  // Computing alpha/beta
  uint64_t alphaBetaComputeCycles =
      std::max(
          cache.mAlphaCycles(maxTimeElementsPerTile, maxLabelElementsPerTile),
          cache.mBetaCycles(maxTimeElementsPerTile, maxLabelElementsPerTile)) *
      alphaOrBetaSteps;

  // Computing gradient from alpha/beta
  uint64_t gradComputeCycles =
      std::max(cache.mGradGivenAlphaCycles(maxTimeElementsPerTile,
                                           maxLabelElementsPerTile),
               cache.mGradGivenBetaCycles(maxTimeElementsPerTile,
                                          maxLabelElementsPerTile)) *
      gradGivenAlphaOrBetaSteps;

  // Exchange cost is the same per step per alpha/beta/grad
  const auto elementsPerTilePerStep =
      2 * maxTimeElementsPerTile // If we are splitting by label, we need to
                                 // exchange two rows of t
      + maxLabelElementsPerTile; // If we are splitting by t, we need to
                                 // exchange a column of label
  const auto bytesPerTilePerStep =
      elementsPerTilePerStep * target.getTypeSize(params.inType);

  uint64_t alphaBetaExchangeCost =
      alphaOrBetaSteps * bytesPerTilePerStep / exchangeBytesPerCycle;
  uint64_t gradExchangeCost =
      gradGivenAlphaOrBetaSteps * bytesPerTilePerStep / exchangeBytesPerCycle;

  return {alphaBetaComputeCycles,
          alphaBetaExchangeCost,
          gradComputeCycles,
          gradExchangeCost,
          steps,
          serialVertexExecutionsPerStep};
}

std::ostream &operator<<(std::ostream &o, const MemoryEstimate &e) {
  o << "Estimated max temporary memory per tile (bytes):\n";
  o << "  Breakdown:\n";
  o << "    data                       " << e.data << "\n";
  o << "    labels                     " << e.labels << "\n";
  o << "    gradient                   " << e.gradient << "\n";
  o << "    alpha/beta temp            " << e.alphaBetaTemp << "\n";
  o << "    temp dependencies          " << e.tempDependancies << "\n";
  o << "  Total:\n";
  o << "    bytes                      " << e.total() << "\n";
  return o;
}

MemoryEstimate estimateTempMemory(const CtcParams &params,
                                  const Plan::Impl &partition,
                                  const poplar::Target &target,
                                  EstimateCache &cache) {
  const uint64_t inTypeBytes = target.getTypeSize(params.inType);
  const uint64_t outTypeBytes = target.getTypeSize(params.outType);

  // For estimating max memory cost, we only consider the part where we are
  // calculating gradGivenBeta or gradGivenAlpha, as they use more temporary
  // memory than the part before where we are calculating just alpha or beta.

  // We are also making an assumption that we are spreading out batch partitions
  // across tiles. Meaning partition in any dimension -> different tile
  const uint64_t batchPerPartition =
      poplibs_support::ceildiv(params.batchSize, partition.parallel.batch);
  const uint64_t timePerPartition =
      poplibs_support::ceildiv(params.maxTime, partition.parallel.time);
  const uint64_t maxLabelLengthPerPartition =
      poplibs_support::ceildiv(params.maxLabelLength, partition.parallel.label);
  const auto maxExtendedLabelLength = params.maxLabelLength * 2 + 1;
  const uint64_t maxExtendedLabelLengthPerPartition = poplibs_support::ceildiv(
      maxExtendedLabelLength, partition.parallel.label);
  const uint64_t alphabetPerPartition =
      poplibs_support::ceildiv(params.numClasses, partition.parallel.alphabet);
  assert(partition.parallel.alphabet == 1); // Not yet accounted for

  const uint64_t dataPerTileBytes = [&]() {
    if (partition.parallel.sliceFromInput) {
      // We copy only relevant classes to each tile
      throw poputil::poplibs_error(
          "Plan::parallel::sliceFromInput = true is currently unsupported");
    } else {
      // We copy the entire alphabet to every tile
      return batchPerPartition * timePerPartition * alphabetPerPartition *
             inTypeBytes;
    }
  }();

  // Each partition has batch per partition number of labels stored
  const uint64_t labelsPerTileBytes =
      batchPerPartition * maxLabelLengthPerPartition * inTypeBytes;

  const uint64_t gradientPerTileBytes = [&]() {
    if (partition.parallel.sliceIntoOutput) {
      // We reduce from ExtendedLabel to Alphabet per partition, max temp memory
      // is when we have on tile gradient tensor for each extended label class
      return (batchPerPartition * timePerPartition * alphabetPerPartition) *
             outTypeBytes * maxExtendedLabelLengthPerPartition;
    } else {
      throw poputil::poplibs_error(
          "Plan::parallel::sliceIntoOutput = false is currently unsupported");
    }
  }();

  const uint64_t alphaBetaTempPerTileBytes =
      (batchPerPartition * timePerPartition *
       maxExtendedLabelLengthPerPartition) *
      inTypeBytes;

  const uint64_t tempDependanciesPerTileBytes =
      (maxExtendedLabelLengthPerPartition + // For time splits
       2 * timePerPartition)                // For extended label splits
      * inTypeBytes;

  return {dataPerTileBytes, labelsPerTileBytes, gradientPerTileBytes,
          alphaBetaTempPerTileBytes, tempDependanciesPerTileBytes};
}

// Returns tuple of {Cycle estimate, Max temp memory estimate, Tiles used}
static std::tuple<popsolver::Variable, popsolver::Variable, popsolver::Variable>
constructModel(popsolver::Model &m, const CtcParams &params,
               const CtcOpts &opts, PartitionVariables &vars,
               const poplar::Target &target, EstimateCache &cache) {
  vars.serial.batch = m.addVariable("serialBatch");
  m.equal(vars.serial.batch, m.one()); // Unsupported
  vars.serial.time = m.addVariable("serialTime");
  m.equal(vars.serial.time, m.one()); // Unsupported
  vars.serial.label = m.addVariable("serialLabel");
  m.equal(vars.serial.label, m.one()); // Unsupported

  vars.parallel.batch = m.addVariable("parallelBatch");
  m.lessOrEqual(m.one(), vars.parallel.batch);
  m.lessOrEqual(vars.parallel.batch, m.addConstant(params.batchSize));

  vars.parallel.time = m.addVariable("parallelTime");
  m.lessOrEqual(m.one(), vars.parallel.time);
  m.lessOrEqual(vars.parallel.time, m.addConstant(params.maxTime));

  vars.parallel.label = m.addVariable("parallelLabel");
  m.lessOrEqual(m.one(), vars.parallel.label);
  m.lessOrEqual(vars.parallel.label, m.addConstant(params.maxLabelLength));

  vars.parallel.sliceIntoOutput =
      m.addVariable(false, true, "parallelSliceIntoOutput");
  m.equal(vars.parallel.sliceIntoOutput, m.addConstant(true)); // Unsupported

  vars.parallel.alphabet = m.addConstant(1, "parallelAlphabet");
  m.equal(vars.parallel.alphabet, m.one()); // Unsupported

  vars.parallel.sliceFromInput =
      m.addVariable(false, true, "parallelSliceFromInput");
  m.equal(vars.parallel.sliceFromInput, m.addConstant(false)); // Unsupported

  auto totalParallelSplit =
      m.product({vars.parallel.batch, vars.parallel.time, vars.parallel.label},
                "totalParallelSplit");
  auto totalTiles = m.addConstant(target.getNumTiles(), "totalTiles");
  m.lessOrEqual(totalParallelSplit, totalTiles);

  const std::vector<popsolver::Variable> allVars{vars.serial.batch,
                                                 vars.serial.time,
                                                 vars.serial.label,
                                                 vars.parallel.batch,
                                                 vars.parallel.time,
                                                 vars.parallel.label,
                                                 vars.parallel.sliceIntoOutput,
                                                 vars.parallel.alphabet,
                                                 vars.parallel.sliceFromInput};

  const auto planFromValues =
      [](const std::vector<unsigned> &values) -> Plan::Impl {
    return {{values[0], values[1], values[2]},
            {values[3], values[4], values[5], static_cast<bool>(values[6]),
             values[7], static_cast<bool>(values[8])}};
  };

  auto cycles = m.call<unsigned>(
      allVars,
      [&params, &target, &cache,
       &planFromValues](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const Plan::Impl plan = planFromValues(values);
        return popsolver::DataType{
            estimateCycles(params, plan, target, cache).total()};
      });

  auto maxTileMemory = m.call<unsigned>(
      allVars,
      [&params, &target, &cache,
       &planFromValues](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const Plan::Impl plan = planFromValues(values);
        return popsolver::DataType{
            estimateTempMemory(params, plan, target, cache).total()};
      });

  if (opts.availableMemoryProportion) {
    m.lessOrEqual(maxTileMemory, m.addConstant(*opts.availableMemoryProportion *
                                               target.getBytesPerTile()));
  }

  return {cycles, maxTileMemory, totalTiles};
}

static Plan::Impl planFromSolution(const popsolver::Solution &solution,
                                   const PartitionVariables &vars) {
  Plan::Impl plan;

  plan.serial.batch = solution[vars.serial.batch].getAs<unsigned>();
  plan.serial.time = solution[vars.serial.time].getAs<unsigned>();
  plan.serial.label = solution[vars.serial.label].getAs<unsigned>();
  plan.parallel.batch = solution[vars.parallel.batch].getAs<unsigned>();
  plan.parallel.time = solution[vars.parallel.time].getAs<unsigned>();
  plan.parallel.label = solution[vars.parallel.label].getAs<unsigned>();
  plan.parallel.sliceIntoOutput =
      solution[vars.parallel.sliceIntoOutput].getAs<bool>();
  plan.parallel.alphabet = solution[vars.parallel.alphabet].getAs<unsigned>();
  plan.parallel.sliceFromInput =
      solution[vars.parallel.sliceFromInput].getAs<bool>();

  return plan;
}

Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
          const poplar::Type &outType, const unsigned batchSize,
          const unsigned maxTime, const unsigned maxLabelLength,
          const unsigned numClasses, const poplar::OptionFlags &options) {
  CtcParams params{inType,  outType,        batchSize,
                   maxTime, maxLabelLength, numClasses};
  CtcOpts opts = parseOptions(options);
  popsolver::Model m;
  PartitionVariables vars;
  EstimateCache cache;

  poplibs_support::logging::popnn::debug("Planning CTCLoss with:\n{}\n{}",
                                         params, opts);
  auto [cycles, maxTempMem, tiles] =
      constructModel(m, params, opts, vars, graph.getTarget(), cache);
  auto s = m.minimize({cycles, maxTempMem, tiles});
  if (!s.validSolution()) {
    throw poputil::poplibs_error("No ctc loss plan found");
  }
  Plan::Impl plan = planFromSolution(s, vars);

  poplibs_support::logging::popnn::debug("Found plan\n{}", plan);
  poplibs_support::logging::popnn::trace(
      "Plan cost\n{}\n{}",
      estimateCycles(params, plan, graph.getTarget(), cache),
      estimateTempMemory(params, plan, graph.getTarget(), cache));
  return std::make_unique<Plan::Impl>(std::move(plan));
}

// Complete the definition of the Plan class
Plan::~Plan() = default;
Plan &Plan::operator=(Plan &&) = default;
Plan::Plan(std::unique_ptr<Plan::Impl> impl) : impl(std::move(impl)) {}

std::ostream &operator<<(std::ostream &o, const Plan &p) {
  o << *p.impl;
  return o;
}

} // namespace ctc
} // namespace popnn

namespace poputil {
template <> poplar::ProfileValue toProfileValue(const popnn::ctc::Plan &p) {
  poplar::ProfileValue::Map v;
  v.insert({"serial.batch", toProfileValue(p.impl->serial.batch)});
  v.insert({"serial.time", toProfileValue(p.impl->serial.time)});
  v.insert({"serial.label", toProfileValue(p.impl->serial.label)});
  v.insert({"parallel.batch", toProfileValue(p.impl->parallel.batch)});
  v.insert({"parallel.time", toProfileValue(p.impl->parallel.time)});
  v.insert({"parallel.label", toProfileValue(p.impl->parallel.label)});
  v.insert({"parallel.sliceIntoOutput",
            toProfileValue(p.impl->parallel.sliceIntoOutput)});
  v.insert({"parallel.alphabet", toProfileValue(p.impl->parallel.alphabet)});
  v.insert({"parallel.sliceFromInput",
            toProfileValue(p.impl->parallel.sliceFromInput)});

  return v;
}
} // namespace poputil
