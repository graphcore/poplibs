// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include "PoolPlan.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/print.hpp"
#include "poputil/TileMapping.hpp"

#include <boost/range/adaptor/reversed.hpp>

#include <unordered_set>

using namespace poputil;

namespace popnn {
namespace pooling {

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

static uint64_t computeCost(const poplin::ConvParams &params,
                            const Partition &split, std::size_t vectorWidth,
                            std::size_t detChansPerGroup,
                            const poplar::Target &target) {
  const auto perTile = split.getPerTile(params);
  const auto outerDimElems = product(perTile.field) / perTile.field.back();
  const auto innerDimElems = perTile.field.back();
  const auto innerVectorsPerTile =
      (perTile.chansPerGroup + vectorWidth - 1) / vectorWidth;

  // compute cost
  uint64_t computeCost =
      outerDimElems *
      (10 + product(params.kernelShape) *
                (innerDimElems * (3 + innerVectorsPerTile * 2))) *
      perTile.batch * perTile.chanGroups;

  const unsigned exchangeBytesPerCycle = 4;
  std::uint64_t bytesPerTile = product(perTile.field) * perTile.batch *
                               perTile.chanGroups * perTile.chansPerGroup *
                               target.getTypeSize(params.inputType);

  // exchange cost: assume everything brought onto tile
  uint64_t exchangeCost = bytesPerTile / exchangeBytesPerCycle;

  // Penalise for changing from detected group
  std::uint64_t rearrangementCost = 0;
  if (detChansPerGroup != perTile.chansPerGroup) {
    rearrangementCost += bytesPerTile / 4;
  }
  return computeCost + exchangeCost + rearrangementCost;
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
Plan getPlan(const poplar::Graph &graph, const PoolConfig &poolCfg,
             const poplin::ConvParams &params, const poplar::Tensor &in_) {
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

  const auto &target = graph.getTarget();
  auto batchSize = inShape[0];
  auto numChannels = inShape[inShape.size() - 1];

  // Do not allow a large number of grains as memory cost of exchanging and
  // rearranging is significant
  auto minGrainsPerChanGroup = 1UL;
  auto maxGrainsPerChanGroup =
      std::min((chansPerGroupDet + chanGrainSize - 1) / chanGrainSize,
               (numChannels + chanGrainSize - 1) / chanGrainSize);
  maxGrainsPerChanGroup =
      std::max(std::min(maxGrainsPerChanGroup, 8UL), minGrainsPerChanGroup);

  const std::size_t numTiles = graph.getTarget().getNumTiles();
  const auto fieldShape = transformedParams.getOutputFieldShape();
  const auto numFieldDims = fieldShape.size();
  Partition split;
  split.field = std::vector<std::size_t>(numFieldDims, 1);
  const auto vectorWidth =
      graph.getTarget().getVectorWidth(transformedParams.inputType);

  // currently kernel is not split. set it to 1
  split.kernel = std::vector<std::size_t>(numFieldDims, 1);

  uint64_t cost = std::numeric_limits<uint64_t>::max();
  Partition bestSplit;

  std::function<void(std::size_t, std::size_t)> f;
  f = [&](std::size_t tiles, std::size_t fieldIndex) {
    for (std::size_t elem = 1UL;
         elem <= std::max(1UL, std::min(tiles, fieldShape[fieldIndex]));
         ++elem) {
      split.field[fieldIndex] = elem;
      if (fieldIndex + 1 == fieldShape.size()) {
        const auto thisCost = computeCost(transformedParams, split, vectorWidth,
                                          chansPerGroupDet, target);
        if (thisCost < cost) {
          cost = thisCost;
          bestSplit = split;
        }
      } else
        f(tiles / elem, fieldIndex + 1);
    }
  };

  for (std::size_t chanGrain = minGrainsPerChanGroup;
       chanGrain <= std::min(maxGrainsPerChanGroup, numTiles); ++chanGrain) {
    split.chansPerGroup = chanGrain * chanGrainSize;
    auto numChanGroups =
        (numChannels + split.chansPerGroup - 1) / split.chansPerGroup;
    for (std::size_t chanSplit = 1UL;
         chanSplit <= std::max(1UL, std::min(numTiles, numChanGroups));
         ++chanSplit) {
      split.chanGroups = chanSplit;
      for (std::size_t batchSplit = 1;
           batchSplit <=
           std::max(1UL, std::min(numTiles / chanSplit, batchSize));
           ++batchSplit) {
        split.batch = batchSplit;
        f(numTiles / (chanSplit * batchSplit), 0);
      }
    }
  }
  plan.partition = bestSplit;
  return plan;
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
