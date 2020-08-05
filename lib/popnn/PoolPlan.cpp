// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include "PoolPlan.hpp"
#include "../poplin/ConvPlan.hpp"
#include "PerformanceEstimation.hpp"
#include "PoolVertices.hpp"
#include "poplibs_support/StructHelper.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/print.hpp"
#include "poplin/ConvUtil.hpp"
#include "poputil/VarStructure.hpp"
#include <popsolver/Model.hpp>

#include <boost/range/adaptor/reversed.hpp>

#include <unordered_set>

using namespace poputil;

namespace popnn {
namespace pooling {

// Constraint variables that represent how variables are partitioned per tile
struct PartitionVariables {
  popsolver::Variable batchSplit;
  popsolver::Variable chanGroupsSplit;
  popsolver::Variable chansPerGroup;
  std::vector<popsolver::Variable> fieldSplit;
  PartitionVariables() = default;
};

// Create Partition from the solution to the Pooling Plan solver
static Partition makePartition(const popsolver::Solution &solution,
                               const PartitionVariables &vars) {
  Partition partition;
  partition.chansPerGroup = solution[vars.chansPerGroup].getAs<unsigned>();
  partition.batch = solution[vars.batchSplit].getAs<unsigned>();
  partition.chanGroups = solution[vars.chanGroupsSplit].getAs<unsigned>();
  partition.field.reserve(vars.fieldSplit.size());
  partition.kernel.reserve(vars.fieldSplit.size());
  for (unsigned i = 0; i < vars.fieldSplit.size(); i++) {
    partition.field.push_back(solution[vars.fieldSplit[i]].getAs<unsigned>());

    // currently kernel is not split. set it to 1
    partition.kernel.push_back(1);
  }
  return partition;
}

// Create Partition from component vector
static Partition makePartition(const std::vector<unsigned> &values,
                               const unsigned numFieldDims) {
  Partition partition;
  partition.batch = values[0];
  partition.chanGroups = values[1];
  partition.chansPerGroup = values[2];
  partition.field.insert(partition.field.begin(), values.begin() + 3,
                         values.begin() + 3 + numFieldDims);
  partition.kernel.insert(partition.kernel.begin(),
                          values.begin() + 3 + numFieldDims,
                          values.begin() + 3 + (2 * numFieldDims));
  return partition;
}

// dim is given as an index into an activation shaped tensor
// [N][....F..][C]
static bool canFlattenDim(const poplin::ConvParams &params, unsigned dim) {
  if (dim >= params.getNumFieldDims() + 1) {
    return false;
  }
  if (dim == 0) {
    return true;
  }
  if (params.kernelShape[dim - 1] != 1) {
    return false;
  }
  // With striding in this dimension this doesn't make
  // a lot of sense.
  if (params.outputTransform.stride[dim - 1] != 1 ||
      params.inputTransform.dilation[dim - 1] != 1) {
    return false;
  }
  return true;
}

static Transform getTransform(const poplar::Graph &graph,
                              const poplin::ConvParams &params,
                              const poplar::Tensor &in,
                              std::size_t chanGrainSize) {
  Transform transform;
  // If we don't have enough channels, prefer to take some elements
  // of another dimension to introducing padding.
  const std::size_t numChans = params.getNumInputChans();
  if (numChans < chanGrainSize) {
    std::size_t currentFactor = 1;
    const auto desiredFactor = lcm(numChans, chanGrainSize) / numChans;

    auto groupings = detectDimGroupings(graph, in);

    // Flatten independent spatial dimensions into channels.
    //
    // Try to flatten spatial dimensions with a detectable grouping first.
    std::unordered_set<std::size_t> transformedDims;
    for (const auto &entry : groupings) {
      const auto d = entry.first;
      const std::size_t grouping = entry.second;
      if (canFlattenDim(params, d) && transformedDims.count(d) == 0 &&
          currentFactor < desiredFactor) {
        const auto f =
            gcd(desiredFactor / currentFactor, gcd(in.dim(d), grouping));
        transform.flattenDims.push_back(std::make_pair(d, f));
        transformedDims.emplace(d);
        currentFactor *= f;
      }
    }
    // Flatten any remaining dims possible, innermost spatial dimension first.
    for (int d = params.getNumFieldDims(); d >= 0; --d) {
      if (canFlattenDim(params, d) && transformedDims.count(d) == 0 &&
          currentFactor < desiredFactor) {
        const auto f = gcd(desiredFactor / currentFactor, in.dim(d));
        transform.flattenDims.push_back(std::make_pair(d, f));
        transformedDims.emplace(d);
        currentFactor *= f;
      }
    }
  }
  return transform;
}
// Functions to cache cycle estimates using a map.  Only the Partition
// that is passed will differ on each call so it is simpler/faster to
// compare that instead of the content of all function parameters.
bool operator<(const Partition &lhs, const Partition &rhs) {
  const auto helper = poplibs_support::makeStructHelper(
      &Partition::field, &Partition::kernel, &Partition::batch,
      &Partition::chanGroups, &Partition::chansPerGroup);
  return helper.lt(lhs, rhs);
}

using EstimateCache = std::map<Partition, std::uint64_t>;
std::size_t cachedPoolVertexCycleEstimate(const Partition &tilePartition,
                                          const PoolConfig &poolCfg,
                                          const poplin::ConvParams &params,
                                          const unsigned numContexts,
                                          EstimateCache &cache) {
  auto cached = cache.find(tilePartition);
  if (cached == cache.end()) {
    const auto result =
        poolVertexCycleEstimate(tilePartition, poolCfg, params, numContexts);
    cache.insert(std::make_pair(tilePartition, result));
    return result;
  }
  return cached->second;
}

std::uint64_t getNumberOfOperations(const poplin::ConvParams &params) {
  // The operations used in pooling can be calculated in a similar way to
  // convolution, except that each channel is independent.
  auto convParams = params;
  convParams.inputChannelsPerConvGroup = 1;

  return getNumberOfMACs(convParams);
}

// Build a popsolver model with the appropriate Pooling Planning variables and
// constraints. Return the "cycle" cost as the derived variable which needs to
// be optimized.
static popsolver::Variable constructModel(
    popsolver::Model &m, const poplar::Target &target, PartitionVariables &vars,
    const PoolConfig &poolCfg, const poplin::ConvParams &params,
    const unsigned minGrainsPerChanGroup, const unsigned maxGrainsPerChanGroup,
    const std::size_t chanGrainSize, const std::size_t numChannels,
    const std::size_t detChansPerGroup, const std::size_t minChannelsPerGroup,
    EstimateCache &cache) {

  auto numTiles = target.getNumTiles();
  auto fieldShape = params.getOutputFieldShape();
  auto kernelShape = params.kernelShape;
  auto nChGrain = m.addVariable(minGrainsPerChanGroup,
                                std::min(maxGrainsPerChanGroup, numTiles));
  auto nCh = m.addConstant(numChannels);
  auto chGrainSize = m.addConstant(chanGrainSize);

  // Sweep Channels Per Group
  vars.chansPerGroup = m.product({nChGrain, chGrainSize});

  // Channel-Group split sweep
  auto nChGroupsMax = m.ceildiv(nCh, vars.chansPerGroup);
  auto nTiles = m.addConstant(numTiles);
  vars.chanGroupsSplit = m.addVariable(1, numTiles);
  m.lessOrEqual(vars.chanGroupsSplit, nChGroupsMax);

  // Batch split sweep
  vars.batchSplit = m.addVariable(1, params.batchSize);

  // Sweep across each field dimension
  vars.fieldSplit.reserve(fieldShape.size());
  for (auto dimSize : fieldShape) {
    vars.fieldSplit.push_back(m.addVariable(1, dimSize));
  }

  // Constrain splits to not exceed the given number of tiles
  std::vector<popsolver::Variable> splits = {vars.chanGroupsSplit,
                                             vars.batchSplit};
  splits.insert(splits.end(), vars.fieldSplit.begin(), vars.fieldSplit.end());
  auto usedTiles = m.product(splits);
  m.lessOrEqual(usedTiles, nTiles);

  // Constrain channels to be >= minChannelsPerGroup
  m.lessOrEqual(m.addConstant(minChannelsPerGroup), vars.chansPerGroup);

  // Work out the size of each partition after applying the split
  std::vector<popsolver::Variable> fieldVar;
  std::vector<popsolver::Variable> kernelVar;
  fieldVar.reserve(vars.fieldSplit.size());
  kernelVar.reserve(vars.fieldSplit.size());
  auto nChGroups = m.ceildiv(nChGroupsMax, vars.chanGroupsSplit);
  auto batchSize = m.addConstant(params.batchSize);
  auto nBatch = m.ceildiv(batchSize, vars.batchSplit);

  for (unsigned i = 0; i < vars.fieldSplit.size(); i++) {
    auto fieldDim = m.addConstant(fieldShape[i]);
    fieldVar.push_back(m.ceildiv(fieldDim, vars.fieldSplit[i]));
    kernelVar.push_back(m.addConstant(kernelShape[i]));
  }

  // Compute lower bound on cycles - define some constants
  auto totalOperations = m.addConstant(getNumberOfOperations(params));

  auto vertexCyclesPerInnerLoop = m.addConstant(poolVertexCyclesPerVector(
      poolCfg.type == PoolingType::MAX, poolCfg.pass == PoolPass::POOL_BWD));
  auto vectorWidth = m.addConstant(target.getVectorWidth(params.inputType));
  auto vertexCyclesPerRow = m.addConstant(poolVertexCyclesPerRow());

  // Compute lower bound on cycles, rounding down as it is a lower bound
  auto rowsPerTile = m.floordiv(m.product(fieldVar), fieldVar.back());
  auto rowOverheadPerTile = m.product({rowsPerTile, vertexCyclesPerRow});
  auto operationsPerTile = m.floordiv(totalOperations, usedTiles);
  auto minCyclesPerTile = m.sum(
      {rowOverheadPerTile, m.product({m.ceildiv(operationsPerTile, vectorWidth),
                                      vertexCyclesPerInnerLoop})});

  // Evaluate the cycle-cost for each possible partition
  std::vector<popsolver::Variable> splitVars = {
      nBatch,
      nChGroups,
      vars.chansPerGroup,
  };
  splitVars.insert(splitVars.end(), fieldVar.begin(), fieldVar.end());
  splitVars.insert(splitVars.end(), kernelVar.begin(), kernelVar.end());

  // Passing in fieldSplits, minCyclesPerTile forces them and fieldVar to be
  // evaluated before the main (time consuming) cycles call is made.
  // This avoids evaluating the cycle cost of plans that will be rejected later.
  splitVars.insert(splitVars.end(), vars.fieldSplit.begin(),
                   vars.fieldSplit.end());
  splitVars.push_back(minCyclesPerTile);

  auto cycles = m.call<unsigned>(
      splitVars, [&target, poolCfg, &params, fieldShape, detChansPerGroup,
                  &cache](const std::vector<unsigned> &values) {
        // Note that "perTile" is not a partition per se, but it contains
        // the variables after they were partitioned for each tile:
        //      perTile->batch = batchSize / batchSplit
        //      perTile->chanGroups = numChannels / channelSplit
        //      perTile->field[d] = ConvParams::fieldShape[d] / fieldSplit[d]
        //      perTile->kernel[d] = ConvParams::kernelShape[d] / kernelSplit[d]
        Partition perTile = makePartition(values, fieldShape.size());

        auto computeCost = cachedPoolVertexCycleEstimate(
            perTile, poolCfg, params, target.getNumWorkerContexts(), cache);
        // Where the field was divided it was rounded up.  For plans where
        // that is not exact, the tiles which take the rounded down field
        // pieces can use more cycles.  This could be true for any dimension, so
        // we try them all and allow the planning cost to determine which is
        // best rather than coding assumptions about the planning cost here. It
        // could be that the slowest tile has a combination of rounded down dims
        // to deal with, but that is not necessarily going to happen and adds
        // many more calls to the estimator, so is not implemented.
        for (unsigned i = 0; i < fieldShape.size(); i++) {
          if (fieldShape[i] % perTile.field[i]) {
            auto altPerTile = perTile;
            altPerTile.field[i] -= 1;
            computeCost = std::max(computeCost,
                                   cachedPoolVertexCycleEstimate(
                                       altPerTile, poolCfg, params,
                                       target.getNumWorkerContexts(), cache));
          }
        }
        // But the partitioned field size is the output, we'll be exchanging
        // and rearranging the input - scale to find the input needed to
        // produce the tile's partitioned output
        std::vector<std::size_t> inFieldPerTile;
        for (unsigned i = 0; i < perTile.field.size(); i++) {
          const auto factor =
              params.outputTransform.stride[i] +
              (perTile.kernel[i] - params.outputTransform.stride[i]) -
              params.inputTransform.paddingLower[i];
          inFieldPerTile.push_back(perTile.field[i] * factor);
        }
        std::uint64_t bytesPerTile =
            product(inFieldPerTile) * perTile.batch * perTile.chanGroups *
            perTile.chansPerGroup * target.getTypeSize(params.inputType);
        // exchange cost: assume everything brought onto tile
        const unsigned exchangeBytesPerCycle =
            target.getExchangeBytesPerCycle();
        uint64_t exchangeCost = bytesPerTile / exchangeBytesPerCycle;

        // Penalise for changing from detected group
        std::uint64_t rearrangementCost = 0;
        if (detChansPerGroup != perTile.chansPerGroup) {
          rearrangementCost += bytesPerTile / 4;
        }
        return popsolver::DataType{computeCost + exchangeCost +
                                   rearrangementCost};
      });

  // Constrain so that the min cycles estimate per tile is < cycles
  // which avoids calling the costly cycle estimator in cases where it is clear
  // that the cycles estimate will be large
  m.lessOrEqual(minCyclesPerTile, cycles);

  return cycles;
}

poplin::ConvParams applyTransform(poplin::ConvParams params,
                                  const Transform &transform,
                                  const std::vector<poplar::Tensor *> &as) {
  for (const auto &entry : transform.flattenDims) {
    const auto d = entry.first;
    const auto f = entry.second;
    // Spatial dimension 0 means batch size
    if (d == 0) {
      assert(params.batchSize % f == 0);
      params.inputChannelsPerConvGroup *= f;
      params.outputChannelsPerConvGroup *= f;
      params.batchSize /= f;
    } else {
      assert(params.inputFieldShape[d - 1] % f == 0);
      params.inputChannelsPerConvGroup *= f;
      params.outputChannelsPerConvGroup *= f;
      params.inputFieldShape[d - 1] /= f;
      // N.B. This is only possible if kernel size in this dimension is
      // 1 so we don't touch kernel shape.
      assert(params.kernelShape[d - 1] == 1);
    }
    for (poplar::Tensor *a : as) {
      if (a != nullptr) {
        *a = a->dimRoll(d, a->rank() - 2)
                 .reshapePartial(a->rank() - 2, a->rank(),
                                 {a->dim(d) / f, a->dim(a->rank() - 1) * f})
                 .dimRoll(a->rank() - 2, d);
      }
    }
  }
  return params;
}

void applyTransformInverse(const poplin::ConvParams &params,
                           const Transform &transform,
                           const std::vector<poplar::Tensor *> &as) {
  for (const auto &entry : boost::adaptors::reverse(transform.flattenDims)) {
    const auto d = entry.first;
    const auto f = entry.second;
    for (poplar::Tensor *a : as) {
      if (a != nullptr) {
        *a = a->dimRoll(d, a->rank() - 2)
                 .reshapePartial(a->rank() - 2, a->rank(),
                                 {a->dim(d) * f, a->dim(a->rank() - 1) / f})
                 .dimRoll(a->rank() - 2, d);
      }
    }
  }
}

// Get plan based on compute and exchange cost. As a further improvement, the
// plan could incorporate introspection. For now, keep it simple.
// Fwd and Bwd plans are kept separate as there is possibly no benefit for
// doing a joint one.
PlanResult getPlan(const poplar::Graph &graph, const PoolConfig &poolCfg,
                   const TransformedInput &input,
                   const TransformedInput &inputGrouped) {
  Plan plan;

  // Don't use getTypeSize here because IpuModel will report something
  // different to what it actually uses.
  // We can change this once T6380 is fixed.
  const auto typeSize = (input.params.inputType == poplar::HALF ? 2 : 4);
  const auto chanGrainSize = 8UL / typeSize;

  plan.transform = getTransform(graph, input.params, input.in, chanGrainSize);

  // Apply any transform to the parameters and input then work
  // out partitioning.
  poplar::Tensor in = input.in;
  const auto transformedParams =
      applyTransform(input.params, plan.transform, {&in});
  const auto inShape = in.shape();
  auto chansPerGroupDet = detectInnermostGrouping(graph, in);
  auto numChannels = inShape[inShape.size() - 1];

  // Do not allow a large number of grains as memory cost of exchanging and
  // rearranging is significant
  auto minGrainsPerChanGroup = 1UL;
  auto maxGrainsPerChanGroup =
      std::min((chansPerGroupDet + chanGrainSize - 1) / chanGrainSize,
               (numChannels + chanGrainSize - 1) / chanGrainSize);
  maxGrainsPerChanGroup =
      std::max(std::min(maxGrainsPerChanGroup, 8UL), minGrainsPerChanGroup);

  // Construct model with variables and constraints
  popsolver::Model m;
  PartitionVariables vars;
  EstimateCache cache;

  auto cycles =
      constructModel(m, graph.getTarget(), vars, poolCfg, transformedParams,
                     minGrainsPerChanGroup, maxGrainsPerChanGroup,
                     chanGrainSize, numChannels, chansPerGroupDet, 1, cache);

  // Optimise within constraints
  auto s = m.minimize({cycles});
  assert(s.validSolution());
  plan.partition = makePartition(s, vars);
  auto sResult = *s[cycles];

  // Consider a second plan, constrained to a minimum number of channels, as
  // operations that use the output can benefit from this.  Allow the pooling
  // cycles to degrade by a fairly large percentage in order to achieve the
  // minimum number of channels as pooling is a relatively cheap operation,
  // and the other operations that it may affect are more expensive.

  // Use the grouped input for this plan, as the input may have been transformed
  // so that channels were combined into the spatial dimensions no longer
  // giving enough channels to meet the preferred channel grouping.
  const auto minChannelsPerGroup =
      getPreferredChannelGrouping(inputGrouped.params.inputType);
  auto numChannelsGrouped = inputGrouped.in.shape().back();
  if (plan.partition.chansPerGroup < minChannelsPerGroup &&
      numChannelsGrouped >= minChannelsPerGroup) {
    maxGrainsPerChanGroup =
        (minChannelsPerGroup + chanGrainSize - 1) / chanGrainSize;
    Plan plan;
    plan.transform = getTransform(graph, inputGrouped.params, inputGrouped.in,
                                  chanGrainSize);
    const auto transformedParams =
        applyTransform(inputGrouped.params, plan.transform, {&in});
    auto chansPerGroupDet = detectInnermostGrouping(graph, inputGrouped.in);

    popsolver::Model mConstrained;
    PartitionVariables varsConstrained;
    auto cyclesConstrained =
        constructModel(mConstrained, graph.getTarget(), varsConstrained,
                       poolCfg, transformedParams, minGrainsPerChanGroup,
                       maxGrainsPerChanGroup, chanGrainSize, numChannelsGrouped,
                       chansPerGroupDet, minChannelsPerGroup, cache);

    // Optimise within constraints of minChannelsPerGroup.  There may not be a
    // solution for all targets
    auto sConstrained = mConstrained.minimize({cyclesConstrained});
    if (sConstrained.validSolution()) {
      auto sConstrainedResult = *sConstrained[cyclesConstrained];
      if (sConstrainedResult < (sResult * 4) / 3) {
        plan.partition = makePartition(sConstrained, varsConstrained);
        return {plan, sConstrainedResult, true};
      }
    }
  }
  return {plan, sResult, false};
}

std::ostream &operator<<(std::ostream &os, const Partition &p) {
  os << "Partition:\n";
  os << "        Batch split              " << p.batch << "\n";
  os << "        Channel split            " << p.chanGroups << "\n";
  os << "        Chans per group          " << p.chansPerGroup << "\n";
  os << "        Kernel split             ";
  printContainer(p.kernel, os);
  os << "\n";
  os << "        Field split              ";
  printContainer(p.field, os);
  os << "\n";
  return os;
}

std::ostream &operator<<(std::ostream &o, const Transform &t) {
  o << "Transform:\n"
    << "        Flatten dims ";
  printContainer(t.flattenDims, o);
  o << "\n";
  return o;
}

std::ostream &operator<<(std::ostream &o, const Plan &p) {
  o << p.transform << p.partition;
  return o;
}

} // namespace pooling
} // namespace popnn
