// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"

#include "PerformanceEstimation.hpp"

#include <poplibs_support/Memoize.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/CTCLoss.hpp>
#include <popsolver/Model.hpp>

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
  unsigned maxLabels;
  unsigned numClasses;
};

std::ostream &operator<<(std::ostream &o, const CtcParams &p) {
  o << "CTCLoss params:\n";
  o << "  inType                       " << p.inType << "\n";
  o << "  outType                      " << p.outType << "\n";
  o << "  batchSize                    " << p.batchSize << "\n";
  o << "  maxTime                      " << p.maxTime << "\n";
  o << "  maxLabels                    " << p.maxLabels << "\n";
  o << "  numClasses                   " << p.numClasses << "\n";
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
CycleEstimate estimateCost(const CtcParams &params, const Plan::Impl &partition,
                           const poplar::Target &target, EstimateCache &cache) {

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
      poplibs_support::ceildiv(params.maxLabels, partition.parallel.label);
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

// Returns tuple of {Cycle estimate, Max temp memory estimate, Tiles used}
static std::tuple<popsolver::Variable, popsolver::Variable, popsolver::Variable>
constructModel(popsolver::Model &m, const CtcParams &params,
               PartitionVariables &vars, const poplar::Target &target,
               EstimateCache &cache) {
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
  m.lessOrEqual(vars.parallel.label, m.addConstant(params.maxLabels));

  vars.parallel.sliceIntoOutput =
      m.addVariable(false, true, "parallelSliceIntoOutput");
  m.equal(vars.parallel.sliceIntoOutput, m.addConstant(true)); // Unsupported

  vars.parallel.alphabet = m.addConstant(1, "parallelAlphabet");
  m.equal(vars.parallel.alphabet, m.one()); // Unsupported

  vars.parallel.sliceFromInput =
      m.addVariable(false, true, "parallelSliceFromInput");
  m.equal(vars.parallel.sliceFromInput, m.addConstant(false)); // Unsupported

  m.equal(m.product({vars.parallel.batch, vars.serial.batch}),
          m.addConstant(params.batchSize));

  auto totalParallelSplit =
      m.product({vars.parallel.batch, vars.parallel.time, vars.parallel.label,
                 vars.parallel.alphabet},
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

  auto cycles = m.call<unsigned>(
      allVars,
      [&params, &target, &cache](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const Plan::Impl plan = {{values[0], values[1], values[2]},
                                 {values[3], values[4], values[5],
                                  static_cast<bool>(values[6]), values[7],
                                  static_cast<bool>(values[8])}};
        return popsolver::DataType{
            estimateCost(params, plan, target, cache).total()};
      });

  auto maxTileMemory = m.addConstant(0); // Placeholder

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
          const unsigned maxTime, const unsigned maxLabels,
          const unsigned numClasses) {
  CtcParams params{inType, outType, batchSize, maxTime, maxLabels, numClasses};
  popsolver::Model m;
  PartitionVariables vars;
  EstimateCache cache;

  poplibs_support::logging::popnn::debug("Planning CTCLoss with params:\n{}",
                                         params);
  auto [cycles, maxTempMem, tiles] =
      constructModel(m, params, vars, graph.getTarget(), cache);
  auto s = m.minimize({cycles, maxTempMem, tiles});
  if (!s.validSolution()) {
    throw poputil::poplibs_error("No ctc loss plan found");
  }
  Plan::Impl plan = planFromSolution(s, vars);

  poplibs_support::logging::popnn::trace(
      "Found plan\n{}", estimateCost(params, plan, graph.getTarget(), cache));
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
