// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include "PoolPlan.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/print.hpp"

namespace popnn {
namespace pooling {


static uint64_t
computeCost(const poplin::ConvParams &params, const Partition &split,
            std::size_t vectorWidth, std::size_t detChansPerGroup,
            const poplar::Target &target) {
  const auto perTile = split.getPerTile(params);
  const auto outerDimElems = product(perTile.field) / perTile.field.back();
  const auto innerDimElems = perTile.field.back();
  const auto innerVectorsPerTile =
      (perTile.chansPerGroup + vectorWidth - 1) / vectorWidth;

  // compute cost
  uint64_t computeCost =
      outerDimElems * ( 10 + product(params.kernelShape) *
      (innerDimElems * (3 + innerVectorsPerTile * 2)))
      * perTile.batch * perTile.chanGroups;

  const unsigned exchangeBytesPerCycle = 4;
  std::uint64_t bytesPerTile =
      product(perTile.field) * perTile.batch * perTile.chanGroups
      * perTile.chansPerGroup *  target.getTypeSize(params.dType);

  // exchange cost: assume everything brought onto tile
  uint64_t exchangeCost = bytesPerTile / exchangeBytesPerCycle;

  // Penalise for changing from detected group
  std::uint64_t rearrangementCost = 0;
  if (detChansPerGroup != perTile.chansPerGroup) {
    rearrangementCost += bytesPerTile / 4;
  }
  return computeCost + exchangeCost + rearrangementCost;
}

// Get plan based on compute and exchange cost. As a further improvement, the
// plan could incorporate introspection. For now, keep it simple.
// Fwd and Bwd plans are kept separate as there is possibly no benefit for
// doing a joint one.
Partition
getPlan(const poplar::Graph &graph, const poplin::ConvParams &params,
        const std::vector<std::size_t> &inShape, unsigned chansPerGroupDet,
        PoolPass pass) {
  const auto &target = graph.getTarget();
  auto batchSize = inShape[0];
  auto numChannels = inShape[inShape.size() - 1];
  // add a grain size dependent on data type
  const auto chanGrainSize = 8UL / target.getTypeSize(params.dType);

  // Do not allow a large number of grains as memory cost of exchanging and
  // rearranging is significant
  auto minGrainsPerChanGroup = 1UL;
  auto maxGrainsPerChanGroup =
      std::min((chansPerGroupDet + chanGrainSize - 1) / chanGrainSize ,
               (numChannels + chanGrainSize - 1) / chanGrainSize);
  maxGrainsPerChanGroup =
      std::max(std::min(maxGrainsPerChanGroup, 8UL), minGrainsPerChanGroup);

  const std::size_t numTiles = graph.getTarget().getNumTiles();
  const auto fieldShape = params.getOutputFieldShape();
  const auto numFieldDims = fieldShape.size();
  Partition split;
  split.field = std::vector<std::size_t>(numFieldDims, 1);
  const auto vectorWidth = graph.getTarget().getVectorWidth(params.dType);

  // currently kernel is not split. set it to 1
  split.kernel = std::vector<std::size_t>(numFieldDims, 1);

  uint64_t cost = std::numeric_limits<uint64_t>::max();
  Partition bestSplit;

  std::function<void (std::size_t, std::size_t)> f;
  f = [&](std::size_t tiles, std::size_t fieldIndex) {
    for (std::size_t elem = 1UL;
         elem <= std::max(1UL, std::min(tiles, fieldShape[fieldIndex]));
         ++elem) {
      split.field[fieldIndex] = elem;
      if (fieldIndex + 1 == fieldShape.size()) {
        const auto thisCost =
              computeCost(params, split, vectorWidth, chansPerGroupDet, target);
        if (thisCost < cost) {
          cost = thisCost;
          bestSplit = split;
        }
      } else
        f(tiles / elem, fieldIndex + 1);
    }
  };

  for (std::size_t chanGrain = minGrainsPerChanGroup;
       chanGrain <= std::min(maxGrainsPerChanGroup, numTiles);
       ++chanGrain) {
    split.chansPerGroup = chanGrain * chanGrainSize;
    auto numChanGroups =
        (numChannels + split.chansPerGroup - 1) / split.chansPerGroup;
    for (std::size_t chanSplit = 1UL;
       chanSplit <= std::max(1UL, std::min(numTiles, numChanGroups));
       ++chanSplit) {
      split.chanGroups = chanSplit;
      for (std::size_t batchSplit = 1;
         batchSplit <= std::max(1UL, std::min(numTiles / chanSplit, batchSize));
         ++batchSplit) {
        split.batch = batchSplit;
        f(numTiles / (chanSplit * batchSplit), 0);
      }
    }
  }
  return bestSplit;
}

std::ostream& operator<<(std::ostream &os, const Partition &p) {
  os << "Pooling Plan :\n";
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

} // namespace pooling
} // namespace poplibs
