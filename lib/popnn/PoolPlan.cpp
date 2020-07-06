// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include "PoolPlan.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/print.hpp"
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
  partition.chansPerGroup = solution[vars.chansPerGroup];
  partition.batch = solution[vars.batchSplit];
  partition.chanGroups = solution[vars.chanGroupsSplit];
  for (unsigned i = 0; i < vars.fieldSplit.size(); i++) {
    partition.field.push_back(solution[vars.fieldSplit[i]]);

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

// Build a popsolver model with the appropriate Pooling Planning variables and
// constraints. Return the "cycle" cost as the derived variable which needs to
// be optimized.
static popsolver::Variable constructModel(
    popsolver::Model &m, const poplar::Target &target, PartitionVariables &vars,
    const poplin::ConvParams &params, const unsigned minGrainsPerChanGroup,
    const unsigned maxGrainsPerChanGroup, const std::size_t chanGrainSize,
    const std::size_t numChannels, const std::size_t detChansPerGroup) {
  auto numTiles = target.getNumTiles();
  auto fieldShape = params.getOutputFieldShape();
  auto kernelShape = params.kernelShape;
  auto vectorWidth = target.getVectorWidth(params.inputType);
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
  for (auto dimSize : fieldShape) {
    vars.fieldSplit.push_back(m.addVariable(1, dimSize));
  }

  // Constrain splits to not exceed the given number of tiles
  std::vector<popsolver::Variable> splits = {vars.chanGroupsSplit,
                                             vars.batchSplit};
  splits.insert(splits.end(), vars.fieldSplit.begin(), vars.fieldSplit.end());
  auto usedTiles = m.product(splits);
  m.lessOrEqual(usedTiles, nTiles);

  // Work out the size of each partition after applying the split
  std::vector<popsolver::Variable> fieldVar;
  std::vector<popsolver::Variable> kernelVar;
  auto nChGroups = m.ceildiv(nChGroupsMax, vars.chanGroupsSplit);
  auto batchSize = m.addConstant(params.batchSize);
  auto nBatch = m.ceildiv(batchSize, vars.batchSplit);
  for (unsigned i = 0; i < vars.fieldSplit.size(); i++) {
    auto fieldDim = m.addConstant(fieldShape[i]);
    fieldVar.push_back(m.ceildiv(fieldDim, vars.fieldSplit[i]));
    kernelVar.push_back(m.addConstant(kernelShape[i]));
  }

  // Evaluate the cycle-cost for each possible partition
  std::vector<popsolver::Variable> splitVars = {
      nBatch,
      nChGroups,
      vars.chansPerGroup,
  };
  splitVars.insert(splitVars.end(), fieldVar.begin(), fieldVar.end());
  splitVars.insert(splitVars.end(), kernelVar.begin(), kernelVar.end());
  auto cycles = m.call(
      splitVars, [target, params, vectorWidth, fieldShape,
                  detChansPerGroup](const std::vector<unsigned> &values) {
        // Note that "perTile" is not a partition per se, but it contains
        // the variables after they were partitioned for each tile:
        //      perTile->batch = batchSize / batchSplit
        //      perTile->chanGroups = numChannels / channelSplit
        //      perTile->field[d] = ConvParams::fieldShape[d] / fieldSplit[d]
        //      perTile->kernel[d] = ConvParams::kernelShape[d] / kernelSplit[d]
        Partition perTile = makePartition(values, fieldShape.size());
        const auto innerDimElems = perTile.field.back();
        const auto outerDimElems = product(perTile.field) / innerDimElems;
        const auto innerVectorsPerTile =
            (perTile.chansPerGroup + vectorWidth - 1) / vectorWidth;

        // compute cost
        uint64_t computeCost =
            outerDimElems *
            (10 + product(perTile.kernel) *
                      (innerDimElems * (3 + innerVectorsPerTile * 2))) *
            perTile.batch * perTile.chanGroups;

        const unsigned exchangeBytesPerCycle = 4;
        std::uint64_t bytesPerTile =
            product(perTile.field) * perTile.batch * perTile.chanGroups *
            perTile.chansPerGroup * target.getTypeSize(params.inputType);

        // exchange cost: assume everything brought onto tile
        uint64_t exchangeCost = bytesPerTile / exchangeBytesPerCycle;

        // Penalise for changing from detected group
        std::uint64_t rearrangementCost = 0;
        if (detChansPerGroup != perTile.chansPerGroup) {
          rearrangementCost += bytesPerTile / 4;
        }
        return computeCost + exchangeCost + rearrangementCost;
      });
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
                   const poplin::ConvParams &params,
                   const poplar::Tensor &in_) {
  Plan plan;

  // Don't use getTypeSize here because IpuModel will report something
  // different to what it actually uses.
  // We can change this once T6380 is fixed.
  const auto typeSize = (params.inputType == poplar::HALF ? 2 : 4);
  const auto chanGrainSize = 8UL / typeSize;

  plan.transform = getTransform(graph, params, in_, chanGrainSize);

  // Apply any transform to the parameters and input then work
  // out partitioning.
  poplar::Tensor in = in_;
  const auto transformedParams = applyTransform(params, plan.transform, {&in});
  const auto inShape = in.shape();
  const auto chansPerGroupDet = detectInnermostGrouping(graph, in);
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
  auto cycles = constructModel(m, graph.getTarget(), vars, transformedParams,
                               minGrainsPerChanGroup, maxGrainsPerChanGroup,
                               chanGrainSize, numChannels, chansPerGroupDet);

  // Optimise within constraints
  auto s = m.minimize({cycles});
  assert(s.validSolution());
  plan.partition = makePartition(s, vars);

  return {plan, s[cycles]};
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
