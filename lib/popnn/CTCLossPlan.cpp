// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"
#include "CTCPlanInternal.hpp"

#include "PerformanceEstimation.hpp"

#include <poplibs_support/Memoize.hpp>
#include <poplibs_support/PlanConstraints.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/CTCLoss.hpp>
#include <popsolver/Model.hpp>
#include <poputil/OptionParsing.hpp>

namespace popnn {
namespace ctc {

static auto getTupleOfMembers(const CtcLossPlannerParams &p) noexcept {
  return std::tie(p.inType, p.partialsType, p.outType, p.batchSize, p.maxTime,
                  p.maxLabelLength, p.numClasses);
}

bool operator<(const CtcLossPlannerParams &a,
               const CtcLossPlannerParams &b) noexcept {
  return getTupleOfMembers(a) < getTupleOfMembers(b);
}

bool operator==(const CtcLossPlannerParams &a,
                const CtcLossPlannerParams &b) noexcept {
  return getTupleOfMembers(a) == getTupleOfMembers(b);
}

static auto getTupleOfMembers(const LossPlan &p) noexcept {
  return std::tie(p.params, p.serial.batch, p.serial.time, p.serial.label,
                  p.parallel.alphabet, p.parallel.batch, p.parallel.label,
                  p.parallel.sliceFromInput, p.parallel.sliceIntoOutput,
                  p.parallel.lastBlankOnSeparateTile, p.parallel.time);
}
bool operator<(const LossPlan &a, const LossPlan &b) noexcept {
  return getTupleOfMembers(a) < getTupleOfMembers(b);
}
bool operator==(const LossPlan &a, const LossPlan &b) noexcept {
  return getTupleOfMembers(a) == getTupleOfMembers(b);
}

struct PlanVariables {
  ParallelPartition<popsolver::Variable, popsolver::Variable> parallel;
  SerialPartition<popsolver::Variable> serial;
};

std::ostream &operator<<(std::ostream &o, const CtcLossPlannerParams &p) {
  o << "CTCLoss params:\n";
  o << "  inType                       " << p.inType << "\n";
  o << "  partialsType                 " << p.partialsType << "\n";
  o << "  outType                      " << p.outType << "\n";
  o << "  batchSize                    " << p.batchSize << "\n";
  o << "  maxTime                      " << p.maxTime << "\n";
  o << "  maxLabelLength               " << p.maxLabelLength << "\n";
  o << "  numClasses                   " << p.numClasses << "\n";
  o << "  enableReducedClassesInLabel  " << p.enableReducedClassesInLabel
    << "\n";
  return o;
}

struct CtcOpts {
  poplibs_support::PlanConstraints planConstraints;
  poplar::Type partialsType = poplar::FLOAT;
  double availableMemoryProportion = 0.6; // Per tile
  bool enableReducedClassesInLabel = false;
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
            "batch",
            "time",
            "label",
            "sliceIntoOutput",
            "lastBlankOnSeparateTile",
            "alphabet",
            "sliceFromInput"};
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

  std::map<std::string, poplar::Type> partialsTypeMap{{"half", poplar::HALF},
                                                      {"float", poplar::FLOAT}};

  const poplibs::OptionSpec spec{
      {"planConstraints",
       makeCtcPlanConstraintsOptionHandler(opts.planConstraints)},
      {"partialsType", poplibs::OptionHandler::createWithEnum(opts.partialsType,
                                                              partialsTypeMap)},
      {"enableReducedClassesInLabel", poplibs::OptionHandler::createWithBool(
                                          opts.enableReducedClassesInLabel)},
      {"availableMemoryProportion", poplibs::OptionHandler::createWithDouble(
                                        opts.availableMemoryProportion, 0.)},
  };
  for (const auto &entry : options) {
    spec.parse(entry.first, entry.second);
  }
  return opts;
}

std::ostream &operator<<(std::ostream &o, const CtcOpts &opt) {
  o << "CTCLoss options:\n";
  o << "  partialsType                 " << opt.partialsType << "\n";
  o << "  availableMemoryProportion    " << opt.availableMemoryProportion
    << "\n";
  return o;
}

struct EstimateCache {
  decltype(poplibs_support::memoize(alphaCycles)) mAlphaCycles;
  decltype(poplibs_support::memoize(betaCycles)) mBetaCycles;
  decltype(poplibs_support::memoize(
      gradGivenAlphaCycles)) mGradGivenAlphaCycles;
  decltype(poplibs_support::memoize(gradGivenBetaCycles)) mGradGivenBetaCycles;
  EstimateCache()
      : mAlphaCycles(alphaCycles), mBetaCycles(betaCycles),
        mGradGivenAlphaCycles(gradGivenAlphaCycles),
        mGradGivenBetaCycles(gradGivenBetaCycles) {}
};

std::ostream &operator<<(std::ostream &o, const LossPlan &p) {
  o << p.params;
  o << "CTCLoss plan:\n";
  /* Unsupported
  o << "  Serial Partition:\n";
  o << "    batch                      " << p.serial.batch << "\n";
  o << "    time                       " << p.serial.time << "\n";
  o << "    label                      " << p.serial.label << "\n";
  */
  o << "  Parallel Partition:\n";
  o << "    batch                      " << p.parallel.batch << "\n";
  o << "    time                       " << p.parallel.time << "\n";
  o << "    label                      " << p.parallel.label << "\n";
  // o << "    sliceIntoOutput            "
  //   << (p.parallel.sliceIntoOutput ? "true" : "false") << "\n";
  o << "    lastBlankOnSeparateTile    "
    << (p.parallel.lastBlankOnSeparateTile ? "true" : "false") << "\n";
  // o << "    alphabet                   " << p.parallel.alphabet << "\n";
  // o << "    sliceFromInput             "
  //   << (p.parallel.sliceFromInput ? "true" : "false") << "\n";
  o << "  Total:\n";
  o << "    tiles                      " << p.numTiles() << "\n";
  return o;
}
std::ostream &operator<<(std::ostream &o, const CycleEstimate &e) {
  o << "Estimated cycles:\n";
  o << "  Alpha/Beta:\n";
  o << "    compute                    " << e.alphaBetaComputeCycles << "\n";
  o << "    exchange                   " << e.alphaBetaExchangeCycles << "\n";
  o << "    sync                       " << e.alphaBetaSyncCycles << "\n";
  o << "    count                      " << e.alphaBetaSteps << "\n";
  o << "  Grad given Alpha/Beta:\n";
  o << "    compute                    " << e.gradComputeCycles << "\n";
  o << "    exchange                   " << e.gradExchangeCycles << "\n";
  o << "    sync                       " << e.gradSyncCycles << "\n";
  o << "    count                      " << e.gradSteps << "\n";
  o << "  Total:\n";
  if (e.serialVertexExecutions > 1) {
    o << "    serial vertex executions\n"
         "      per step                 "
      << e.serialVertexExecutions << "\n";
  }
  o << "    compute                    " << e.totalCompute() << "\n";
  o << "    exchange                   " << e.totalExchange() << "\n";
  o << "    sync                       " << e.totalSync() << "\n";
  o << "    cycles                     " << e.total() << "\n";
  return o;
}

/*
  We can illustrate splits in extended label by dividing a vertical wavefront
  which is propagating horizontally in time. Implicitly, there is a barrier at
  each timestep, otherwise it would be more like a diagonal wavefront to account
  for the dependency on time. For alpha, partitions marked as "B", require the
  partition marked "A" to have first been executed. Considering the following
  example where we split extended label into 4 partitions:

      t
   |0|1|2|
  -+-----+
  0|A|B|C|
  -+-----+
  1|A|B|C|
El-+-----+
  2|A|B|C|
  -+-----+
  3|A|B|C|
  -+-----+

  Then to complete the operation we have a sequence of steps like the following:

  0: alpha{A}, beta{C}
  1: alpha{B} [*]
  2: gradGivenAlpha{B}
  3: gradGivenAlpha{A}, gradGivenBeta{C}

  [*] We satisfy the dependencies to compute both alpha and beta at this point,
      however we don't do so as we would need double the temporary memory to
      keep both alpha and beta in memory concurrently when calculating the
      gradient. To instead not increase temporary memory, we arbitrarily pick
      alpha or beta to calculate. In this example we choose alpha, but it is
      just as valid to choose beta. It's worth noting that if t was even, we
      wouldn't encounter this, it's only when the length of t is odd.
*/
CycleEstimate estimateCycles(const LossPlan &plan, const poplar::Target &target,
                             EstimateCache &cache) {
  auto timePartitionCount = plan.parallel.time;
  auto maxTimeStepsPerPartition =
      gccs::ceildiv(plan.params.maxTime, plan.parallel.time);
  auto partitionSteps = timePartitionCount;
  if (plan.parallel.time & 1) {
    // When we have an odd number of time partitions, to avoid a clash in the
    // middle for alpha and beta, we add an extra step (where either alpha or
    // beta will wait)
    // TODO - If implementing a supervisor vertex, and mapping > 2 timesteps
    // per tile the window in which the stall happens will widen.
    partitionSteps += 1;
  }

  assert((partitionSteps & 1) == 0); // Implicit from above logic

  auto maxBatchPerTile =
      gccs::ceildiv(plan.params.batchSize, plan.parallel.batch);

  // Currently we use 1 worker per batch, noting this is only valid while using
  // worker and not supervisor vertices.  A "serial vertex execution" accounts
  // for all workers even if only 1 is active, all the rest are burning cycles.
  const auto numWorkers = target.getNumWorkerContexts();
  auto serialVertexExecutionsPerStep =
      gccs::ceildiv(maxBatchPerTile, numWorkers);

  const auto alphaOrBetaSteps = (maxTimeStepsPerPartition * partitionSteps) / 2;
  const auto gradGivenAlphaOrBetaSteps =
      (maxTimeStepsPerPartition * partitionSteps) / 2;

  auto maxLabelElementsPerPartition =
      gccs::ceildiv(plan.params.maxLabelLength, plan.parallel.label);

  auto balancedLabelPartitions =
      maxLabelElementsPerPartition ==
      plan.params.maxLabelLength / plan.parallel.label;
  auto partitionSlowedByExtraBlank =
      (balancedLabelPartitions && !plan.parallel.lastBlankOnSeparateTile);

  // Consider 1 timestep {compute + exchange}, and repeat this for the partition
  const auto timesteps = 1;

  // Computing alpha/beta
  uint64_t alphaBetaComputeCyclesPerTimestep =
      std::max(cache.mAlphaCycles(timesteps, maxLabelElementsPerPartition,
                                  partitionSlowedByExtraBlank),
               cache.mBetaCycles(timesteps, maxLabelElementsPerPartition,
                                 partitionSlowedByExtraBlank)) *
      numWorkers;
  // Computing gradient from alpha/beta
  uint64_t gradComputeCyclesPerTimestep =
      std::max(
          cache.mGradGivenAlphaCycles(timesteps, maxLabelElementsPerPartition,
                                      partitionSlowedByExtraBlank),
          cache.mGradGivenBetaCycles(timesteps, maxLabelElementsPerPartition,
                                     partitionSlowedByExtraBlank)) *
      numWorkers;

  // After each compute set, we do the same exchange (slightly different for
  // alpha/beta and grad phases). Since the exchange is quite small, we model a
  // fixed overhead which usually dominates the exchange cost.
  const auto fixedOverheadExchangeCycles = 100;

  // There's an exchange of a column of El from the previous timestep (either
  // already on tile because it is the same timestep partition, or exchanged
  // across partitions).
  const auto timePartitionExchangeElements =
      plan.parallel.time > 2 ? maxLabelElementsPerPartition : 0;
  // There's overlap of +1 for alpha, +2 for beta in each exchange from
  // neighbouring partitions (of El), we only need consider +2 as the larger
  // exchange.
  const auto labelPartitionExchangeElements = plan.parallel.label > 1 ? 2 : 0;
  // If partitioned in both time and label, we also need to exchange elements
  // from previous timestep and El label partition
  const auto labelAndTimePartitionExchangeElements =
      (plan.parallel.label > 1 && plan.parallel.time > 2) ? 2 : 0;
  const auto alphaBetaExchangeElements = timePartitionExchangeElements +
                                         labelPartitionExchangeElements +
                                         labelAndTimePartitionExchangeElements;

  // For grad calculation we exchange two ping pong buffers of a column of El,
  // one containing previous timeslices, and the other containing alpha/beta of
  // the previous time partition. We also have +2 elements as before from
  // neighbouring partition of El for beta calculation.
  const auto gradExchangeElements = 2 + 2 * (2 * maxLabelElementsPerPartition);

  const unsigned exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  const auto alphaBetaExchangeCost =
      fixedOverheadExchangeCycles +
      (alphaBetaExchangeElements *
       target.getTypeSize(plan.params.partialsType)) /
          exchangeBytesPerCycle;
  const auto gradExchangeCost =
      fixedOverheadExchangeCycles +
      (gradExchangeElements * target.getTypeSize(plan.params.partialsType)) /
          exchangeBytesPerCycle;

  // We also sync prior to each exchange which has approximately the same cost
  // each time
  const auto syncCycles = 90;

  return {alphaBetaComputeCyclesPerTimestep,
          alphaBetaExchangeCost,
          syncCycles,
          alphaOrBetaSteps,

          gradComputeCyclesPerTimestep,
          gradExchangeCost,
          syncCycles,
          gradGivenAlphaOrBetaSteps,

          serialVertexExecutionsPerStep};
}

std::ostream &operator<<(std::ostream &o, const MemoryEstimate &e) {
  o << "Estimated max temporary memory per tile (bytes):\n";
  o << "  Breakdown:\n";
  o << "    data                       " << e.data << "\n";
  o << "    labels                     " << e.labels << "\n";
  o << "    gradient                   " << e.gradient << "\n";
  o << "    temp reduced gradients     " << e.tempReducedGradient << "\n";
  o << "    alpha/beta temp            " << e.alphaBetaTemp << "\n";
  o << "    temp dependencies          " << e.tempDependancies << "\n";
  o << "  Total:\n";
  o << "    bytes                      " << e.total() << "\n";
  return o;
}

MemoryEstimate estimateMaxTileTempMemory(const LossPlan &plan,
                                         const poplar::Target &target,
                                         EstimateCache &cache) {
  const auto labelType = poplar::UNSIGNED_SHORT;
  const uint64_t inTypeBytes = target.getTypeSize(plan.params.inType);
  const uint64_t partialsTypeBytes =
      target.getTypeSize(plan.params.partialsType);
  const uint64_t labelTypeBytes = target.getTypeSize(labelType);

  // For estimating max memory cost, we only consider the part where we are
  // calculating gradGivenBeta or gradGivenAlpha, as they use more temporary
  // memory than the part before where we are calculating just alpha or beta.

  const uint64_t batchPerPartition =
      gccs::ceildiv(plan.params.batchSize, plan.parallel.batch);
  const uint64_t timePerPartition =
      gccs::ceildiv(plan.params.maxTime, plan.parallel.time);
  const uint64_t maxLabelLengthPerPartition =
      gccs::ceildiv(plan.params.maxLabelLength, plan.parallel.label);
  uint64_t maxExtendedLabelLengthPerPartition = maxLabelLengthPerPartition * 2;
  if (maxLabelLengthPerPartition * plan.parallel.label ==
      plan.params.maxLabelLength) {
    // Divided out labels equally, so last partition has an extra blank
    maxExtendedLabelLengthPerPartition += 1;
  }
  const uint64_t alphabetPerPartition =
      gccs::ceildiv(plan.params.numClasses, plan.parallel.alphabet);
  assert(plan.parallel.alphabet == 1); // Not yet accounted for

  const uint64_t dataPerTileBytes = [&]() {
    if (plan.parallel.sliceFromInput) {
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
      batchPerPartition * maxLabelLengthPerPartition * labelTypeBytes;

  const auto gradientBytesPerTile = [&]() {
    if (plan.parallel.sliceIntoOutput) {
      // We need a working copy of gradient per tile
      const auto numPartitionTiles =
          plan.parallel.label + (plan.parallel.lastBlankOnSeparateTile ? 1 : 0);
      const auto gradientBytes =
          (batchPerPartition * timePerPartition * alphabetPerPartition) *
          partialsTypeBytes;
      // we need a copy of the gradients because we can't reduce on-tile in
      // the first stage
      const auto reducedGradientsBytes =
          numPartitionTiles > 1 ? gradientBytes : 0;
      return std::make_pair(gradientBytes, reducedGradientsBytes);
    } else {
      throw poputil::poplibs_error(
          "Plan::parallel::sliceIntoOutput = false is currently unsupported");
    }
  }();

  const uint64_t alphaBetaTempPerTileBytes =
      (batchPerPartition * maxExtendedLabelLengthPerPartition) *
      partialsTypeBytes;

  // We store temporary data to propagate El information each timestep:
  // - 1 El length slice to propagate alpha or beta in the time dimension when
  // calling alpha or beta vertices (currently assumed always live during the
  // operation but may not be the case)
  // - 2 El length slices to propagate alpha or beta the time dimension when
  // calling gradGivenAlpha or gradGivenBeta vertices
  const uint64_t tempDependanciesPerTileBytes =
      batchPerPartition * ((1 + 2) * maxExtendedLabelLengthPerPartition) *
      partialsTypeBytes;

  return {dataPerTileBytes,
          labelsPerTileBytes,
          std::get<0>(gradientBytesPerTile),
          std::get<1>(gradientBytesPerTile),
          alphaBetaTempPerTileBytes,
          tempDependanciesPerTileBytes};
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
bool checkForEmptyPartitions(const LossPlan &plan) {
  const auto timePartitionSize =
      gccs::ceildiv(plan.params.maxTime, plan.parallel.time);
  const bool lastTimePartitionEmpty =
      plan.params.maxTime <= (timePartitionSize * (plan.parallel.time - 1));

  const auto labelPartitionSize =
      gccs::ceildiv(plan.params.maxLabelLength, plan.parallel.label);
  const bool lastLabelPartitionEmpty =
      plan.params.maxLabelLength <=
      (labelPartitionSize * (plan.parallel.label - 1));
  return lastTimePartitionEmpty || lastLabelPartitionEmpty;
}

// Returns tuple of {Cycle estimate, Max temp memory estimate, Tiles used}
static std::tuple<popsolver::Variable, popsolver::Variable, popsolver::Variable>
constructModel(popsolver::Model &m, const CtcLossPlannerParams &params,
               const CtcOpts &opts, PlanVariables &vars,
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

  vars.parallel.lastBlankOnSeparateTile =
      m.addVariable(false, true, "parallelLastBlankOnSeparateTile");

  // If we put the last blank on a separate tile, label partitions increase by 1
  auto labelPartitions =
      m.sum({vars.parallel.label, vars.parallel.lastBlankOnSeparateTile});

  auto totalTilesUsed =
      m.product({vars.parallel.batch, vars.parallel.time, labelPartitions},
                "totalTilesUsed");

  auto totalTiles = m.addConstant(target.getNumTiles(), "totalTiles");
  m.lessOrEqual(totalTilesUsed, totalTiles);

  const std::vector<popsolver::Variable> planVariablesArray{
      vars.serial.batch,
      vars.serial.time,
      vars.serial.label,

      vars.parallel.batch,
      vars.parallel.time,
      vars.parallel.label,
      vars.parallel.sliceIntoOutput,
      vars.parallel.lastBlankOnSeparateTile,
      vars.parallel.alphabet,
      vars.parallel.sliceFromInput};

  const auto toPlanStruct =
      [&params](const std::vector<unsigned> &values) -> LossPlan {
    return {
        params,
        {values[0], values[1], values[2]},
        {values[3], values[4], values[5], static_cast<bool>(values[6]),
         static_cast<bool>(values[7]), values[8], static_cast<bool>(values[9])},
    };
  };

  auto cycles = m.call<unsigned>(
      planVariablesArray,
      [&target, &cache, toPlanStruct](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const LossPlan plan = toPlanStruct(values);
        return popsolver::DataType{estimateCycles(plan, target, cache).total()};
      });

  auto maxTileTempMemory = m.call<unsigned>(
      planVariablesArray,
      [&target, &cache, toPlanStruct](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const LossPlan plan = toPlanStruct(values);
        return popsolver::DataType{
            estimateMaxTileTempMemory(plan, target, cache).total()};
      });

  m.lessOrEqual(
      maxTileTempMemory,
      m.addConstant(opts.availableMemoryProportion * target.getBytesPerTile()));

  auto emptyPartitions = m.call<unsigned>(
      planVariablesArray,
      [toPlanStruct](const std::vector<unsigned> &values)
          -> boost::optional<popsolver::DataType> {
        const LossPlan plan = toPlanStruct(values);
        return popsolver::DataType{checkForEmptyPartitions(plan)};
      });
  m.equal(emptyPartitions, m.zero());

  return {cycles, maxTileTempMemory, totalTilesUsed};
}

static void
applyPlanConstraints(popsolver::Model &m,
                     const poplibs_support::PlanConstraints &planConstraints,
                     const PlanVariables &vars) {
  const auto constrainUnsignedVar = [&](const char *name,
                                        popsolver::Variable var) {
    if (auto constraint = planConstraints.get_optional<unsigned>(name)) {
      poplibs_support::logging::popnn::debug("Constraining {} = {}", name,
                                             *constraint);
      m.equal(var, popsolver::DataType{*constraint});
    }
  };

  constrainUnsignedVar("parallel.batch", vars.parallel.batch);
  constrainUnsignedVar("parallel.time", vars.parallel.time);
  constrainUnsignedVar("parallel.label", vars.parallel.label);
  constrainUnsignedVar("parallel.sliceIntoOutput",
                       vars.parallel.sliceIntoOutput);
  constrainUnsignedVar("parallel.lastBlankOnSeparateTile",
                       vars.parallel.lastBlankOnSeparateTile);
  constrainUnsignedVar("parallel.alphabet", vars.parallel.alphabet);
  constrainUnsignedVar("parallel.sliceFromInput", vars.parallel.sliceFromInput);

  constrainUnsignedVar("serial.batch", vars.serial.batch);
  constrainUnsignedVar("serial.time", vars.serial.time);
  constrainUnsignedVar("serial.label", vars.serial.label);
}

static LossPlan planFromSolution(const CtcLossPlannerParams &params,
                                 const popsolver::Solution &solution,
                                 const PlanVariables &vars) {
  LossPlan plan;

  plan.params = params;
  plan.serial.batch = solution[vars.serial.batch].getAs<unsigned>();
  plan.serial.time = solution[vars.serial.time].getAs<unsigned>();
  plan.serial.label = solution[vars.serial.label].getAs<unsigned>();
  plan.parallel.batch = solution[vars.parallel.batch].getAs<unsigned>();
  plan.parallel.time = solution[vars.parallel.time].getAs<unsigned>();
  plan.parallel.label = solution[vars.parallel.label].getAs<unsigned>();
  plan.parallel.sliceIntoOutput =
      solution[vars.parallel.sliceIntoOutput].getAs<bool>();
  plan.parallel.lastBlankOnSeparateTile =
      solution[vars.parallel.lastBlankOnSeparateTile].getAs<bool>();
  plan.parallel.alphabet = solution[vars.parallel.alphabet].getAs<unsigned>();
  plan.parallel.sliceFromInput =
      solution[vars.parallel.sliceFromInput].getAs<bool>();

  return plan;
}

Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
          const poplar::Type &outType, const unsigned batchSize,
          const unsigned maxTime, const unsigned maxLabelLength,
          const unsigned numClasses, const poplar::OptionFlags &options) {
  CtcOpts opts = parseOptions(options);
  CtcLossPlannerParams params;
  if (opts.enableReducedClassesInLabel) {
    // By using ctcLossPreProcess() we can transform logProbs, the input of
    // CTCLoss, from shape(maxTime, batchSize, numClasses) to
    // shape(maxTime batchSize, maxLabelLength + 1). Which can reduce the data
    // exchange and memory cost during the computation of CTCLoss.
    params = {inType,         opts.partialsType,  outType, batchSize, maxTime,
              maxLabelLength, maxLabelLength + 1, true};
  } else {
    params = {inType,  opts.partialsType, outType,    batchSize,
              maxTime, maxLabelLength,    numClasses, false};
  }
  popsolver::Model m;
  PlanVariables vars;
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
  const auto plan = planFromSolution(params, s, vars);

  poplibs_support::logging::popnn::debug("Found plan\n{}", plan);
  poplibs_support::logging::popnn::debug(
      "Plan cost\n{}\n{}", estimateCycles(plan, graph.getTarget(), cache),
      estimateMaxTileTempMemory(plan, graph.getTarget(), cache));
  return std::make_unique<Plan::Impl>(Plan::Impl{std::move(plan)});
}

} // namespace ctc
} // namespace popnn

namespace poputil {
template <> poplar::ProfileValue toProfileValue(const popnn::ctc::LossPlan &p) {
  poplar::ProfileValue::Map v;
  v.insert({"params.inType", toProfileValue(p.params.inType)});
  v.insert({"params.partialsType", toProfileValue(p.params.partialsType)});
  v.insert({"params.outType", toProfileValue(p.params.outType)});
  v.insert({"params.batchSize", toProfileValue(p.params.batchSize)});
  v.insert({"params.maxTime", toProfileValue(p.params.maxTime)});
  v.insert({"params.maxLabelLength", toProfileValue(p.params.maxLabelLength)});
  v.insert({"params.numClasses", toProfileValue(p.params.numClasses)});
  v.insert({"params.enableReducedClassesInLabel",
            toProfileValue(p.params.enableReducedClassesInLabel)});
  v.insert({"serial.batch", toProfileValue(p.serial.batch)});
  v.insert({"serial.time", toProfileValue(p.serial.time)});
  v.insert({"serial.label", toProfileValue(p.serial.label)});
  v.insert({"parallel.batch", toProfileValue(p.parallel.batch)});
  v.insert({"parallel.time", toProfileValue(p.parallel.time)});
  v.insert({"parallel.label", toProfileValue(p.parallel.label)});
  v.insert(
      {"parallel.sliceIntoOutput", toProfileValue(p.parallel.sliceIntoOutput)});
  v.insert({"parallel.lastBlankOnSeparateTile",
            toProfileValue(p.parallel.lastBlankOnSeparateTile)});
  v.insert({"parallel.alphabet", toProfileValue(p.parallel.alphabet)});
  v.insert(
      {"parallel.sliceFromInput", toProfileValue(p.parallel.sliceFromInput)});

  return v;
}

} // namespace poputil
