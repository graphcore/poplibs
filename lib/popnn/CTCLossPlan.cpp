// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"

#include "PerformanceEstimation.hpp"

#include <poplibs_support/Memoize.hpp>
#include <poplibs_support/PlanConstraints.hpp>
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
  poplibs_support::PlanConstraints planConstraints;
  double availableMemoryProportion = 0.6; // Per tile
};

void validatePlanConstraints(const std::string &path,
                             const boost::property_tree::ptree &t,
                             const std::vector<std::string> &validConstraints) {
  for (const auto &child : t) {
    auto valid = std::find(validConstraints.begin(), validConstraints.end(),
                           child.first) != validConstraints.end();
    if (valid) {
      poplibs_support::validatePlanConstraintsUnsigned(child.first,
                                                       child.second);
    } else {
      throw poputil::poplibs_error("Unrecognised constraint " + path + "." +
                                   child.first);
    }
  }
}

struct ValidateCtcPlanConstraintsOption {
  void operator()(const boost::property_tree::ptree &t) const {
    if (t.empty() && !t.data().empty()) {
      throw poplar::invalid_option("Plan constraints must be an object");
    }

    for (const auto &child : t) {
      if (child.first == "parallel") {
        const std::vector<std::string> validParallelConstraints{
            "batch",           "time",     "timePartitionsPerTile", "label",
            "sliceIntoOutput", "alphabet", "sliceFromInput"};
        validatePlanConstraints(child.first, child.second,
                                validParallelConstraints);
      } else if (child.first == "serial") {
        const std::vector<std::string> validSerialConstraints{"batch", "time",
                                                              "label"};
        validatePlanConstraints(child.first, child.second,
                                validSerialConstraints);
      } else {
        throw poputil::poplibs_error("Unrecognised constraint " + child.first);
      }
    }
  }
};

static CtcOpts parseOptions(const poplar::OptionFlags &options) {
  CtcOpts opts;
  using poplibs_support::makePlanConstraintsOptionHandler;
  const auto makeCtcPlanConstraintsOptionHandler =
      &makePlanConstraintsOptionHandler<ValidateCtcPlanConstraintsOption>;

  const poplibs::OptionSpec spec{
      {"planConstraints",
       makeCtcPlanConstraintsOptionHandler(opts.planConstraints)},
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
  o << "  availableMemoryProportion    " << opt.availableMemoryProportion
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
  o << "    timePartitionsPerTile      " << p.parallel.timePartitionsPerTile
    << "\n";
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
  o << "  Alpha/Beta (total of " << e.steps / 2 << " steps):\n";
  o << "    compute                    " << e.alphaBetaComputeCycles << "\n";
  o << "    exchange                   " << e.alphaBetaExchangeCost << "\n";
  o << "  Grad given Alpha/Beta (total of " << e.steps / 2 << " steps):\n";
  o << "    compute                    " << e.gradComputeCycles << "\n";
  o << "    exchange                   " << e.gradExchangeCost << "\n";
  o << "  Total:\n";
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
    // TODO - If implementing a supervisor vertex, and mapping > 2 timesteps
    // per tile the window in which the stall happens will widen.
    steps += 1;
  }
  assert((steps & 1) == 0); // Implicit from above logic

  auto maxBatchPerTile =
      poplibs_support::ceildiv(params.batchSize, partition.parallel.batch);

  // Currently we use 1 worker per batch, noting this is only valid while using
  // worker and not supervisor vertices.  A "serial vertex execution" accounts
  // for all workers even if only 1 is active, all the rest are burning cycles.
  const auto numWorkers = target.getNumWorkerContexts();
  auto serialVertexExecutionsPerStep =
      poplibs_support::ceildiv(maxBatchPerTile, numWorkers);

  const auto alphaOrBetaSteps = serialVertexExecutionsPerStep * (steps / 2);
  const auto gradGivenAlphaOrBetaSteps =
      serialVertexExecutionsPerStep * (steps / 2);
  const unsigned exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  auto maxLabelElementsPerPartition =
      poplibs_support::ceildiv(params.maxLabelLength, partition.parallel.label);
  auto maxTimeElementsPerPartition =
      poplibs_support::ceildiv(params.maxTime, partition.parallel.time);
  // Computing alpha/beta
  uint64_t alphaBetaComputeCyclesPerStep =
      std::max(cache.mAlphaCycles(maxTimeElementsPerPartition,
                                  maxLabelElementsPerPartition),
               cache.mBetaCycles(maxTimeElementsPerPartition,
                                 maxLabelElementsPerPartition)) *
      numWorkers;
  uint64_t alphaBetaComputeCycles =
      alphaBetaComputeCyclesPerStep * alphaOrBetaSteps;
  // Computing gradient from alpha/beta
  uint64_t gradComputeCyclesPerStep =
      std::max(cache.mGradGivenAlphaCycles(maxTimeElementsPerPartition,
                                           maxLabelElementsPerPartition),
               cache.mGradGivenBetaCycles(maxTimeElementsPerPartition,
                                          maxLabelElementsPerPartition)) *
      numWorkers;
  uint64_t gradComputeCycles =
      gradComputeCyclesPerStep * gradGivenAlphaOrBetaSteps;
  // Exchange cost is the same per step per alpha/beta/grad
  const auto elementsPerTilePerStep =
      2 * maxTimeElementsPerPartition // If we are splitting by label, we need
                                      // to exchange two rows of t
      + maxLabelElementsPerPartition; // If we are splitting by t, we need to
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

MemoryEstimate estimateMaxTileTempMemory(const CtcParams &params,
                                         const Plan::Impl &partition,
                                         const poplar::Target &target,
                                         EstimateCache &cache) {
  const auto partialsType = params.outType;
  const auto labelType = poplar::UNSIGNED_SHORT;
  const uint64_t inTypeBytes = target.getTypeSize(params.inType);
  const uint64_t partialsTypeBytes = target.getTypeSize(partialsType);
  const uint64_t labelTypeBytes = target.getTypeSize(labelType);

  // For estimating max memory cost, we only consider the part where we are
  // calculating gradGivenBeta or gradGivenAlpha, as they use more temporary
  // memory than the part before where we are calculating just alpha or beta.

  const uint64_t batchPerPartition =
      poplibs_support::ceildiv(params.batchSize, partition.parallel.batch);
  const uint64_t timePerPartition =
      poplibs_support::ceildiv(params.maxTime, partition.parallel.time);
  const uint64_t timePerTile =
      timePerPartition * partition.parallel.timePartitionsPerTile;
  const uint64_t maxLabelLengthPerPartition =
      poplibs_support::ceildiv(params.maxLabelLength, partition.parallel.label);
  uint64_t maxExtendedLabelLengthPerPartition = maxLabelLengthPerPartition * 2;
  if (maxLabelLengthPerPartition * partition.parallel.label ==
      params.maxLabelLength) {
    // Divided out labels equally, so last partition has an extra blank
    maxExtendedLabelLengthPerPartition += 1;
  }
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
      return batchPerPartition * timePerTile * alphabetPerPartition *
             inTypeBytes;
    }
  }();

  // Each partition has batch per partition number of labels stored
  const uint64_t labelsPerTileBytes =
      batchPerPartition * maxLabelLengthPerPartition * labelTypeBytes;

  const uint64_t gradientPerTileBytes = [&]() {
    if (partition.parallel.sliceIntoOutput) {
      // We need a working copy of gradient per tile
      return (batchPerPartition * timePerTile * alphabetPerPartition) *
             partialsTypeBytes;
    } else {
      throw poputil::poplibs_error(
          "Plan::parallel::sliceIntoOutput = false is currently unsupported");
    }
  }();

  const uint64_t alphaBetaTempPerTileBytes =
      (batchPerPartition * timePerTile * maxExtendedLabelLengthPerPartition) *
      partialsTypeBytes;

  // For time splits:
  // - 1 El length slice to propagate alpha or beta in the time dimension when
  // calling alpha or beta vertices (currently assumed always live during the
  // operation but may not be the case)
  // - 2 El length slices to propagate alpha or beta the time dimension when
  // calling gradGivenAlpha or gradGivenBeta vertices
  // For extended label splits:
  // - 1 maxT length slice to propagate alpha downwards
  // - 2 maxT length slices to propagate beta upwards
  const uint64_t tempDependanciesPerTileBytes =
      batchPerPartition *
      ((1 + 2) * maxExtendedLabelLengthPerPartition + // For time splits
       (1 + 2) * timePerTile) // For extended label splits
      * partialsTypeBytes;

  return {dataPerTileBytes, labelsPerTileBytes, gradientPerTileBytes,
          alphaBetaTempPerTileBytes, tempDependanciesPerTileBytes};
}

// Explicitly check that no partitions are empty.  If they are there will
// always be another plan that has the same cost. An example is:
// There are 25 timesteps.  We partition by 16, stepsPerPartition=ceildiv(25,16)
// which is 2.  So the 16 partitions contain 2,2,2,2,2,2,2,2,2,2,2,2,1,0,0,0
// timeSteps.  So we don't really get what we thought we had, and a plan with
// time partitioned into 13 will be implemented identically anyhow!
//
// The cost model ought to avoid this but it is not always clear exactly how
// this is going to always be the case.
bool checkForEmptyPartitions(const CtcParams &params,
                             const Plan::Impl &partition) {
  const auto timePartitionSize =
      poplibs_support::ceildiv(params.maxTime, partition.parallel.time);
  const bool lastTimePartitionEmpty =
      params.maxTime <= (timePartitionSize * (partition.parallel.time - 1));

  const auto labelPartitionSize =
      poplibs_support::ceildiv(params.maxLabelLength, partition.parallel.label);
  const bool lastLabelPartitionEmpty =
      params.maxLabelLength <=
      (labelPartitionSize * (partition.parallel.label - 1));
  return lastTimePartitionEmpty || lastLabelPartitionEmpty;
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

  vars.parallel.timePartitionsPerTile =
      m.addVariable("parallelTimePartitionsPerTile");
  m.lessOrEqual(m.one(), vars.parallel.timePartitionsPerTile);
  m.lessOrEqual(vars.parallel.timePartitionsPerTile, vars.parallel.time);

  // Simply constrain this, as coupled with parallel.time the two
  // parameters expand the seach space and things get too slow.
  // In effect parallel.time and parallel.timePartitionsPerTile
  // are complementary so the number of tiles used is much more variable.
  // In addition throttling the number of splits that are possible here
  // means that plans that take a long time to constuct a graph for aren't
  // generated.
  // TODO - make this less arbitrary or find a better way to speed up.
  //        We should consider this when changing graph construction to
  //        use a runtime loop.
  m.lessOrEqual(vars.parallel.timePartitionsPerTile, m.addConstant(1));
  auto tilesForAllTimePartitions =
      m.ceildiv(vars.parallel.time, vars.parallel.timePartitionsPerTile,
                "tilesForAllTimePartitions");

  auto totalTilesUsed = m.product(
      {vars.parallel.batch, tilesForAllTimePartitions, vars.parallel.label},
      "totalTilesUsed");

  auto totalTiles = m.addConstant(target.getNumTiles(), "totalTiles");
  m.lessOrEqual(totalTilesUsed, totalTiles);

  const std::vector<popsolver::Variable> planArray{
      vars.serial.batch,      vars.serial.time,
      vars.serial.label,      vars.parallel.batch,
      vars.parallel.time,     vars.parallel.timePartitionsPerTile,
      vars.parallel.label,    vars.parallel.sliceIntoOutput,
      vars.parallel.alphabet, vars.parallel.sliceFromInput};

  const auto toPlanStruct =
      [](const std::vector<unsigned> &values) -> Plan::Impl {
    return {{values[0], values[1], values[2]},
            {values[3], values[4], values[5], values[6],
             static_cast<bool>(values[7]), values[8],
             static_cast<bool>(values[9])}};
  };

  auto cycles = m.call<unsigned>(
      planArray,
      [&params, &target, &cache,
       &toPlanStruct](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const Plan::Impl plan = toPlanStruct(values);
        return popsolver::DataType{
            estimateCycles(params, plan, target, cache).total()};
      });

  auto maxTileTempMemory = m.call<unsigned>(
      planArray,
      [&params, &target, &cache,
       &toPlanStruct](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const Plan::Impl plan = toPlanStruct(values);
        return popsolver::DataType{
            estimateMaxTileTempMemory(params, plan, target, cache).total()};
      });

  m.lessOrEqual(
      maxTileTempMemory,
      m.addConstant(opts.availableMemoryProportion * target.getBytesPerTile()));

  auto emptyPartitions = m.call<unsigned>(
      planArray,
      [&params, &toPlanStruct](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const Plan::Impl plan = toPlanStruct(values);
        return popsolver::DataType{checkForEmptyPartitions(params, plan)};
      });
  m.equal(emptyPartitions, m.zero());

  return {cycles, maxTileTempMemory, totalTilesUsed};
}

static void
applyPlanConstraints(popsolver::Model &m,
                     const poplibs_support::PlanConstraints &planConstraints,
                     const PartitionVariables &vars) {
  const auto constrainUnsignedVar = [&](const char *name,
                                        popsolver::Variable var) {
    if (auto constraint = planConstraints.get_optional<unsigned>(name)) {
      poplibs_support::logging::popnn::debug("Constraining {} = {}", name,
                                             constraint);
      m.equal(var, popsolver::DataType{*constraint});
    }
  };

  constrainUnsignedVar("parallel.batch", vars.parallel.batch);
  constrainUnsignedVar("parallel.time", vars.parallel.time);
  constrainUnsignedVar("parallel.timePartitionsPerTile",
                       vars.parallel.timePartitionsPerTile);
  constrainUnsignedVar("parallel.label", vars.parallel.label);
  constrainUnsignedVar("parallel.sliceIntoOutput",
                       vars.parallel.sliceIntoOutput);
  constrainUnsignedVar("parallel.alphabet", vars.parallel.alphabet);
  constrainUnsignedVar("parallel.sliceFromInput", vars.parallel.sliceFromInput);

  constrainUnsignedVar("serial.batch", vars.serial.batch);
  constrainUnsignedVar("serial.time", vars.serial.time);
  constrainUnsignedVar("serial.label", vars.serial.label);
}

static Plan::Impl planFromSolution(const popsolver::Solution &solution,
                                   const PartitionVariables &vars) {
  Plan::Impl plan;

  plan.serial.batch = solution[vars.serial.batch].getAs<unsigned>();
  plan.serial.time = solution[vars.serial.time].getAs<unsigned>();
  plan.serial.label = solution[vars.serial.label].getAs<unsigned>();
  plan.parallel.batch = solution[vars.parallel.batch].getAs<unsigned>();
  plan.parallel.time = solution[vars.parallel.time].getAs<unsigned>();
  plan.parallel.timePartitionsPerTile =
      solution[vars.parallel.timePartitionsPerTile].getAs<unsigned>();
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
  applyPlanConstraints(m, opts.planConstraints, vars);

  auto s = m.minimize({cycles, maxTempMem, tiles});
  if (!s.validSolution()) {
    throw poputil::poplibs_error("No ctc loss plan found");
  }
  Plan::Impl plan = planFromSolution(s, vars);

  poplibs_support::logging::popnn::debug("Found plan\n{}", plan);
  poplibs_support::logging::popnn::debug(
      "Plan cost\n{}\n{}",
      estimateCycles(params, plan, graph.getTarget(), cache),
      estimateMaxTileTempMemory(params, plan, graph.getTarget(), cache));
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
  v.insert({"parallel.timePartitionsPerTile",
            toProfileValue(p.impl->parallel.timePartitionsPerTile)});
  v.insert({"parallel.label", toProfileValue(p.impl->parallel.label)});
  v.insert({"parallel.sliceIntoOutput",
            toProfileValue(p.impl->parallel.sliceIntoOutput)});
  v.insert({"parallel.alphabet", toProfileValue(p.impl->parallel.alphabet)});
  v.insert({"parallel.sliceFromInput",
            toProfileValue(p.impl->parallel.sliceFromInput)});

  return v;
}
} // namespace poputil
