#include "popconv/Convolution.hpp"
#include "popconv/ConvPlan.hpp"
#include <limits>
#include <cassert>
#include <cmath>
#include "popconv/ConvUtil.hpp"
#include "popstd/Pad.hpp"
#include "popstd/ActivationMapping.hpp"
#include "popreduce/Reduce.hpp"
#include "popstd/Regroup.hpp"
#include "popstd/VertexTemplates.hpp"
#include "util/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexOptim.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/Cast.hpp"
#include "popstd/Util.hpp"
#include "Winograd.hpp"
#include "popstd/Zero.hpp"
#include <unordered_set>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace popconv {

static void
applyTensorMapping(
    Graph &graph,
    const Tensor &t,
    const std::vector<
      std::vector<Interval<std::size_t>>
    > &mapping) {
  auto flattened = t.flatten();
  const auto numTiles = mapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &region : mapping[tile]) {
      graph.setTileMapping(flattened.slice(region.begin(), region.end()),
                           tile);
    }
  }
}

struct WeightGradAopTask {
  unsigned kernelY;
  unsigned kernelX;
  unsigned outZGroup;
  unsigned inZGroup;
  WeightGradAopTask(unsigned kernelY, unsigned kernelX,
                    unsigned outZGroup, unsigned inZGroup) :
    kernelY(kernelY), kernelX(kernelX),
    outZGroup(outZGroup), inZGroup(inZGroup) {}

};

Tensor
createWeights(Graph &graph, const Tensor &in,
              unsigned kernelSizeY, unsigned kernelSizeX,
              unsigned outNumChans,
              unsigned strideY, unsigned strideX,
              unsigned paddingY, unsigned paddingX,
              bool isFractional,
              const ConvOptions &options) {
  const auto dType = graph.getTensorElementType(in);
  const auto inNumChans = in.dim(1) * in.dim(4);
  const auto plan = getPlan(graph, dType,
                            in.dim(0), in.dim(2), in.dim(3),
                            inNumChans,
                            {kernelSizeY, kernelSizeX, outNumChans},
                            {strideY, strideX},
                            {paddingY, paddingX},
                            isFractional, options);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto inNumChanGroups = inNumChans / inChansPerGroup;
  auto weights = graph.addTensor(dType, {partialNumChanGroups,
                                         inNumChanGroups,
                                         kernelSizeY,
                                         kernelSizeX,
                                         partialChansPerGroup,
                                         inChansPerGroup},
                                 "weights");
  return weights;
}

poplar::Tensor
createBiases(poplar::Graph &graph, std::string dType,
             unsigned outNumChans) {
  auto biases = graph.addTensor(dType, {outNumChans}, "biases");
  return biases;
}

static unsigned
linearizeTileIndices(unsigned batchGroup, unsigned numBatchGroups,
                     unsigned numTiles,
                     unsigned ky, unsigned izg,
                     unsigned ox, unsigned oy, unsigned ozg,
                     const Plan &plan,
                     bool isMultiIPU) {
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;

  unsigned beginTile;
  if (numBatchGroups <= numTiles) {
    beginTile = (numTiles / numBatchGroups) * batchGroup;
  } else {
    const auto batchGroupsPerTile = (numBatchGroups + numTiles - 1) / numTiles;
    beginTile = batchGroup / batchGroupsPerTile;
  }
  // If this is a multi IPU system then choose an order that avoids splitting
  // partial sums over IPUs
  unsigned tile;
  if (isMultiIPU)
    tile = beginTile +
      (ky + tilesPerKernelYAxis *
        (izg + tilesPerInZGroup *
          (ox + tilesPerX *
            (oy + tilesPerY * ozg))));
  // Use ozg as the innermost dimension to increase the chance that
  // tiles in a supertile both read the same activations. This reduces
  // exchange time when supertile send / receive is used.
  else
    tile = beginTile +
           (ozg + tilesPerZ *
             (ox + tilesPerX *
               (oy + tilesPerY *
                 (ky + tilesPerKernelYAxis *
                   izg))));
  assert(tile < numTiles);
  return tile;
}

static std::pair<unsigned,unsigned>
getOutZGroupRange(unsigned ozgIndex, unsigned partialNumChanGroups,
                  const Plan &plan) {
  const auto tilesPerZAxis = plan.tilesPerZAxis;
  const auto maxZGroupsPerTile = (partialNumChanGroups + tilesPerZAxis - 1) /
                                 tilesPerZAxis;
  const auto outZBegin =
      std::min(ozgIndex * maxZGroupsPerTile, partialNumChanGroups);
  const auto outZEnd =
      std::min((ozgIndex + 1) * maxZGroupsPerTile, partialNumChanGroups);
  return {outZBegin, outZEnd};
}

unsigned getFlattenedIndex(const std::vector<std::size_t> &shape,
                           const std::vector<std::size_t> &indices) {
  const auto rank = shape.size();
  assert(indices.size() == rank);
  unsigned index = 0;
  for (unsigned dim = 0; dim != rank; ++dim) {
    assert(indices[dim] < shape[dim]);
    index *= shape[dim];
    index += indices[dim];
  }
  return index;
}

unsigned getFlattenedIndex(const Tensor &t,
                           const std::vector<std::size_t> &indices) {
  return getFlattenedIndex(t.shape(), indices);
}

static void
addFlattenedRegions(const std::vector<std::size_t> &shape,
                    const std::vector<std::size_t> &begin,
                    const std::vector<std::size_t> &end,
                    std::vector<Interval<std::size_t>> &regions) {
  const auto numDims = shape.size();
  assert(begin.size() == numDims);
  assert(end.size() == numDims);

  for (unsigned dim = 0; dim != numDims; ++dim) {
    if (begin[dim] == end[dim])
      return;
  }

  std::vector<std::size_t> indices = begin;
  bool done = false;
  while (!done) {
    unsigned regionBegin = getFlattenedIndex(shape, indices);
    unsigned regionEnd = regionBegin + (end.back() - begin.back());
    regions.emplace_back(regionBegin, regionEnd);
    done = true;
    for (unsigned dim = 0; dim != numDims - 1; ++dim) {
      if (indices[dim] + 1 == end[dim]) {
        indices[dim] = begin[dim];
      } else {
        ++indices[dim];
        done = false;
        break;
      }
    }
  }
}

static std::vector<std::vector<Interval<std::size_t>>>
calculateActivationMapping(const Graph &graph,
                           const Plan &plan,
                           Tensor acts) {
  const auto numBatchGroups = acts.dim(0);
  const auto isMultiIPU = graph.getDevice().getDeviceInfo().numIPUs > 1;
  const auto inNumChans = acts.dim(1) * acts.dim(4);
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  std::vector<std::vector<Interval<std::size_t>>>
      mapping(deviceInfo.getNumTiles());
  const auto actType = graph.getTensorElementType(acts);
  const auto actTypeSize = actType == "float" ? 4 : 2;
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + actTypeSize - 1) / minBytesPerTile;

  // Map activations such that, for a 1x1 kernel, the activations are spread
  // evenly across the set of tiles that read them. This should also give
  // reasonable results for non 1x1 kernels.
  for (unsigned b = 0; b < numBatchGroups; ++b) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
      for (unsigned y = 0; y != tilesPerY; ++y) {
        const auto yBegin = (y * acts.dim(2)) / tilesPerY;
        const auto yEnd = ((y + 1) * acts.dim(2)) / tilesPerY;
        for (unsigned x = 0; x != tilesPerX; ++x) {
          const auto xBegin = (x * acts.dim(3)) / tilesPerX;
          const auto xEnd = ((x + 1) * acts.dim(3)) / tilesPerX;
          std::vector<Interval<std::size_t>> sharedActivations;
          addFlattenedRegions(acts.shape(),
                              {b, inZGroupBegin, yBegin, xBegin, 0},
                              {b + 1, inZGroupEnd, yEnd, xEnd, acts.dim(4)},
                              sharedActivations);
          mergeAdjacentRegions(sharedActivations);
          std::unordered_set<unsigned> tileSet;
          for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
            for (unsigned ky = 0; ky != tilesPerKernelY; ++ky) {
              const auto tile = linearizeTileIndices(b, numBatchGroups,
                                                     numTiles,
                                                     ky, izg,
                                                     x, y, ozg,
                                                     plan,
                                                     isMultiIPU);
              tileSet.insert(tile);
            }
          }
          std::vector<unsigned> tiles(tileSet.begin(), tileSet.end());
          std::sort(tiles.begin(), tiles.end());
          const auto perTileWeights =
              splitRegions(sharedActivations, inChansPerGroup,
                           tiles.size(), minElementsPerTile);
          for (unsigned i = 0; i != perTileWeights.size(); ++i) {
            mapping[tiles[i]] = perTileWeights[i];
          }
        }
      }
    }
  }
  return mapping;
}

static void mapActivations(Graph &graph, const Plan &plan,
                           const Tensor &acts) {
  auto mapping = calculateActivationMapping(graph, plan, acts);
  graph.setTileMapping(acts, mapping);
}

void mapActivations(poplar::Graph &graph,
                    const poplar::Tensor &in,
                    const poplar::Tensor &w,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX,
                    bool isFractional,
                    const ConvOptions &options) {
  const auto dType = graph.getTensorElementType(in);
  const auto inNumChans = in.dim(1) * in.dim(4);
  const auto plan = getPlan(graph, dType,
                            in.dim(0), in.dim(2), in.dim(3),
                            inNumChans,
                            {w.dim(2), w.dim(3), w.dim(0) * w.dim(4)},
                            {strideY, strideX},
                            {paddingY, paddingX},
                            isFractional, options);
  mapActivations(graph, plan, in);
}

static std::vector<std::vector<Interval<std::size_t>>>
calculateWeightMapping(const std::vector<std::size_t> &wShape,
                       const std::string &dType,
                       const poplar::Graph &graph,
                       const Plan &plan,
                       unsigned batchSize) {
  assert(batchSize % plan.batchesPerGroup == 0);
  const auto numBatchGroups = batchSize / plan.batchesPerGroup;
  const auto partialNumChanGroups = wShape[0];
  const auto numInZGroups = wShape[1];
  const auto kernelDimY = wShape[2];
  const auto kernelDimX = wShape[3];
  const auto partialChansPerGroup = wShape[4];
  const auto inChansPerGroup = wShape[5];

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  std::vector<std::vector<Interval<std::size_t>>>
      mapping(deviceInfo.getNumTiles());

  const auto isMultiIPU = deviceInfo.numIPUs > 1;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto weightTypeSize = dType == "float" ? 4 : 2;
  // Limit the minimum number of weight bytes per tile to reduce the
  // amount of exchange code. Increasing this constant reduces exchange code
  // size and increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto minBytesPerTile = 256;
  const auto minElementsPerTile =
      (minBytesPerTile + weightTypeSize - 1) / weightTypeSize;
  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
    if (inZGroupBegin == inZGroupEnd)
      continue;
    for (unsigned ky = 0; ky != tilesPerKernelY; ++ky) {
      const auto kernelYBegin = (ky * kernelDimY) / tilesPerKernelY;
      const auto kernelYEnd = ((ky + 1) * kernelDimY) / tilesPerKernelY;
      for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
        unsigned outZGroupBegin, outZGroupEnd;
        std::tie(outZGroupBegin, outZGroupEnd) =
            getOutZGroupRange(ozg, partialNumChanGroups, plan);
        if (outZGroupBegin == outZGroupEnd)
          continue;
        // Group weights that are accessed contiguously by tiles within this
        // loop body.
        std::vector<Interval<std::size_t>> sharedWeights;
        addFlattenedRegions(wShape,
                            {outZGroupBegin, inZGroupBegin, kernelYBegin,
                             0, 0, 0},
                            {outZGroupEnd, inZGroupEnd, kernelYEnd,
                             wShape[3], wShape[4], wShape[5]},
                            sharedWeights);
        mergeAdjacentRegions(sharedWeights);
        // Spread groups of weights equally across the tiles that read them.
        unsigned grainSize = partialChansPerGroup *
                             inChansPerGroup;
        if (plan.useConvolutionInstructions) {
          if (kernelDimY == 1 && kernelDimX == 1) {
            grainSize *= numInZGroups;
          }
        } else {
          grainSize *= kernelDimX;
        }
        std::unordered_set<unsigned> tileSet;
        for (unsigned b = 0; b != numBatchGroups; ++b) {
          for (unsigned oy = 0; oy != tilesPerY; ++oy) {
            for (unsigned ox = 0; ox != tilesPerX; ++ox) {
              const auto tile = linearizeTileIndices(b, numBatchGroups,
                                                     numTiles,
                                                     ky, izg,
                                                     ox, oy, ozg,
                                                     plan, isMultiIPU);
              tileSet.insert(tile);
            }
          }
        }
        std::vector<unsigned> tiles(tileSet.begin(), tileSet.end());
        std::sort(tiles.begin(), tiles.end());
        const auto perTileWeights =
            splitRegions(sharedWeights, partialChansPerGroup * inChansPerGroup,
                         tiles.size(), minElementsPerTile);
        for (unsigned i = 0; i != perTileWeights.size(); ++i) {
          mapping[tiles[i]] = perTileWeights[i];
        }
      }
    }
  }
  return mapping;
}

static std::vector<std::vector<Interval<std::size_t>>>
calculateWeightMapping(const Tensor &weights,
                       const poplar::Graph &graph,
                       const Plan &plan,
                       unsigned batchSize) {
  return calculateWeightMapping(weights.shape(),
                                graph.getTensorElementType(weights), graph,
                                plan, batchSize);
}

template <typename Builder>
static void
iterateWeightMapping(Tensor w,
                     const poplar::Graph &graph,
                     const Plan &plan,
                     unsigned batchSize,
                     Builder &&builder) {
  const auto weightMapping =
      calculateWeightMapping(w, graph, plan, batchSize);
  const auto flatWeights = w.flatten();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    auto tileWeights = flatWeights.slice(0, 0);
    for (const auto &region : weightMapping[tile]) {
      const auto weightBegin = region.begin();
      const auto weightEnd = region.end();
      assert(weightBegin != weightEnd);
      tileWeights = concat(tileWeights,
                           flatWeights.slice(weightBegin, weightEnd));
    }
    if (tileWeights.numElements() > 0) {
      builder(tileWeights, tile);
    }
  }
}

static void mapWeights(const Tensor &w, Graph &graph, const Plan &plan,
                       unsigned batchSize) {
  iterateWeightMapping(w, graph, plan, batchSize,
    [&](Tensor tileWeights, unsigned tile) {
    graph.setTileMapping(tileWeights, tile);
  });
}

void
mapWeights(Tensor w, Graph &graph, const Tensor &in,
           unsigned strideY, unsigned strideX,
           unsigned paddingY, unsigned paddingX,
           bool isFractional,
           const ConvOptions &options) {
  const auto dType = graph.getTensorElementType(in);
  const auto inNumChans = in.dim(1) * in.dim(4);
  const auto plan = getPlan(graph, dType,
                            in.dim(0), in.dim(2), in.dim(3),
                            inNumChans,
                            {w.dim(2), w.dim(3), w.dim(0) * w.dim(4)},
                            {strideY, strideX},
                            {paddingY, paddingX},
                            isFractional, options);
  mapWeights(w, graph, plan, in.dim(0));
}

template <typename Builder>
static void iterateBiasMapping(Tensor b, const Graph &graph,
                               const std::vector<std::size_t> actShape,
                               Builder &&builder) {
  const auto batchSize = actShape[0];
  std::vector<std::size_t> actShape1(actShape.begin() + 1, actShape.end());
  const auto dType = graph.getTensorElementType(b);
  const auto activationsMapping = computeActivationsMapping(graph, dType,
                                                            actShape1,
                                                            0,
                                                            batchSize);
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  const auto outNumChans = actShape[1] * actShape[4];
  const auto outNumChanGroups = actShape[1];
  const auto outDimY = actShape[2];
  const auto outDimX = actShape[3];
  std::vector<unsigned> mapping(outNumChanGroups);
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  Tensor biasesByChanGroup =
      b.reshape({outNumChanGroups, outChansPerGroup});
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileActivationsBegin = activationsMapping[tile];
    const auto tileActivationsEnd = activationsMapping[tile + 1];
    assert(tileActivationsBegin % outChansPerGroup == 0);
    assert(tileActivationsEnd % outChansPerGroup == 0);
    const auto tileGroupBegin = tileActivationsBegin / outChansPerGroup;
    const auto tileGroupEnd = tileActivationsEnd / outChansPerGroup;
    const auto tileNumGroups = tileGroupEnd - tileGroupBegin;
    if (tileNumGroups == 0)
      continue;
    const auto minOutChanGroup = tileGroupBegin / (outDimX * outDimY);
    const auto maxOutChanGroup = (tileGroupEnd - 1) / (outDimX * outDimY);
    for (unsigned grp = minOutChanGroup; grp <= maxOutChanGroup; ++grp)
      mapping[grp] = tile;
  }

  unsigned beginGroup = 0;
  unsigned curTile = mapping[0];
  for (unsigned grp = 1; grp < outNumChanGroups; ++grp) {
    if (mapping[grp] != curTile) {
      Tensor biasSlice = biasesByChanGroup.slice(beginGroup, grp);
      builder(biasSlice, curTile);
      curTile = mapping[grp];
      beginGroup = grp;
    }
  }
  Tensor finalSlice = biasesByChanGroup.slice(beginGroup, outNumChanGroups);
  builder(finalSlice, curTile);
}

void mapBiases(Tensor biases, Graph &graph,
               const Tensor &in, const Tensor &w,
               unsigned strideY, unsigned strideX,
               unsigned paddingY, unsigned paddingX,
               bool isFractional,
               const ConvOptions &options) {
  const unsigned kernelSizeY = w.dim(2);
  const unsigned kernelSizeX = w.dim(3);
  std::size_t outDimY;
  std::size_t outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(in.dim(2), in.dim(3),
                                            kernelSizeY, kernelSizeX,
                                            strideY, strideX,
                                            paddingY, paddingX,
                                            isFractional);
  const auto dType = graph.getTensorElementType(in);
  const auto inNumChans = in.dim(1) * in.dim(4);
  const auto outNumChans = w.dim(0) * w.dim(4);
  const auto plan = getPlan(graph, dType,
                            in.dim(0), in.dim(2), in.dim(3),
                            inNumChans,
                            {w.dim(2), w.dim(3), w.dim(0) * w.dim(4)},
                            {strideY, strideX},
                            {paddingY, paddingX},
                            isFractional, options);
  std::vector<std::size_t> actShape{in.dim(0),
                                    outNumChans / plan.partialChansPerGroup,
                                    outDimY, outDimX,
                                    plan.partialChansPerGroup};
  iterateBiasMapping(biases, graph, actShape,
                     [&](Tensor biasSlice, unsigned tile) {
                         graph.setTileMapping(biasSlice, tile);
                     });
}

static void
createConvPartial1x1OutVertex(Graph &graph,
                              unsigned tile,
                              unsigned outXBegin, unsigned outXEnd,
                              unsigned outYBegin, unsigned outYEnd,
                              unsigned ozg,
                              unsigned kernelY,
                              unsigned inZGroupBegin, unsigned inZGroupEnd,
                              const std::vector<unsigned> &stride,
                              const std::vector<unsigned> &padding,
                              ComputeSet fwdCS,
                              const Tensor &in, const Tensor &weights,
                              const Tensor &out) {
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = static_cast<unsigned>(in.dim(3));
  const auto outChansPerGroup = static_cast<unsigned>(out.dim(3));
  const auto dType = graph.getTensorElementType(in);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex = deviceInfo.numWorkerContexts;
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(dType == "float");
  const auto convUnitCoeffLoadBytesPerCycle =
                deviceInfo.convUnitCoeffLoadBytesPerCycle;
  const auto outHeight = outYEnd - outYBegin;
  const auto outWidth = outXEnd - outXBegin;
  const auto partialType = graph.getTensorElementType(out);
  unsigned inYBegin, inYEnd, inXBegin, inXEnd;
  std::tie(inYBegin, inYEnd) =
      getInputRange({outYBegin, outYEnd}, stride[0], kernelSizeY,
                     padding[0], inDimY, kernelY, false);
  std::tie(inXBegin, inXEnd) =
      getInputRange({outXBegin, outXEnd}, stride[1],
                     kernelSizeX, padding[1], inDimX, false);

  std::vector<std::vector<PartialRow>> workerPartition;
  unsigned outputStride = 1;
  workerPartition =
      partitionConvPartialByWorker(outHeight, outWidth,
                                   contextsPerVertex, outputStride);

  std::vector<Tensor> inputEdges;
  std::vector<Tensor> outputEdges;

  unsigned numConvolutions = 0;
  for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
    for (unsigned i = 0; i != contextsPerVertex; ++i) {
      for (const auto &partialRow : workerPartition[i]) {
        const auto workerOutY = outYBegin + partialRow.rowNumber;
        const auto workerOutXBegin = outXBegin + partialRow.begin;
        const auto workerOutXEnd = outXBegin + partialRow.end;
        const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
        const auto workerInY =
            getInputIndex(workerOutY, stride[0], kernelSizeY,
                          padding[0], inDimY, kernelY, false);
        assert(workerInY != ~0U);
        unsigned workerInXBegin, workerInXEnd;
        std::tie(workerInXBegin, workerInXEnd) =
            getInputRange({workerOutXBegin, workerOutXEnd}, stride[1],
                          kernelSizeX, padding[1], inDimX, false);
        const auto workerInWidth = workerInXEnd - workerInXBegin;
        assert(workerInWidth != 0);
        Tensor inWindow =
            in[izg][workerInY].slice(
              {workerInXBegin, 0},
              {workerInXEnd, inChansPerGroup}
            ).reshape({workerInWidth * inChansPerGroup});
        Tensor outWindow =
            out[ozg][workerOutY].slice(
              {workerOutXBegin, 0},
              {workerOutXEnd, outChansPerGroup}
            ).reshape({workerOutWidth * outChansPerGroup});
        inputEdges.push_back(inWindow);
        outputEdges.push_back(outWindow);

        graph.setTileMapping(outWindow, tile);
        ++numConvolutions;
      }
    }
  }
  const auto numEdges = 1 + 2 * numConvolutions;

  // Add the vertex.
  Tensor w =
      weights[ozg].slice(
  {inZGroupBegin, kernelY, 0, 0, 0},
  {inZGroupEnd, kernelY + 1, 1, outChansPerGroup, inChansPerGroup}
        ).flatten();
  auto v = graph.addVertex(
        fwdCS,
        templateVertex("popconv::ConvPartial1x1Out", dType, partialType,
                       useDeltaEdgesForConvPartials(numEdges) ?
                                                    "true" : "false"),
        {{"weights", w}}
        );
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["weightsPerConvUnit"], weightsPerConvUnit);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                        convUnitCoeffLoadBytesPerCycle);
  graph.setFieldSize(v["weightReuseCount"], contextsPerVertex);
  for (unsigned i = 0; i != contextsPerVertex; ++i) {
    graph.setInitialValue(
          v["weightReuseCount"][i],
        static_cast<std::uint32_t>(workerPartition[i].size())
        );
  }
  graph.connect(v["in"], inputEdges);
  graph.connect(v["out"], outputEdges);
  // Map the vertex and output.
  graph.setTileMapping(v, tile);
}

static void
createConvPartialnx1InOutVertex(Graph &graph,
                                unsigned tile,
                                unsigned outXBegin, unsigned outXEnd,
                                unsigned outYBegin, unsigned outYEnd,
                                unsigned outZGroupBegin, unsigned outZGroupEnd,
                                unsigned kernelYBegin, unsigned kernelYEnd,
                                unsigned inZGroupBegin, unsigned inZGroupEnd,
                                const std::vector<unsigned> &stride,
                                const std::vector<unsigned> &padding,
                                ComputeSet fwdCS,
                                const Tensor &in,
                                const Tensor &weights,
                                const Tensor &out,
                                const Tensor &zeros,
                                bool isFractional) {
  if (outXBegin == outXEnd ||
      outYBegin == outYEnd ||
      kernelYBegin == kernelYEnd ||
      inZGroupBegin == inZGroupEnd)
    return;
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = static_cast<unsigned>(in.dim(3));
  const auto outChansPerGroup = static_cast<unsigned>(out.dim(3));
  const auto dType = graph.getTensorElementType(in);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex = deviceInfo.numWorkerContexts;
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(dType == "float");
  assert(weightsPerConvUnit % inChansPerGroup == 0);
  const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
  const auto convUnitCoeffLoadBytesPerCycle
                      = deviceInfo.convUnitCoeffLoadBytesPerCycle;

  const auto partialType = graph.getTensorElementType(out);
  // It is possible that there is no calculation to perform that involves
  // the specified output slice and kernel weight slice. Instead of adding a
  // vertex to the graph upfront add it lazily when we first need it.
  unsigned numWeights = 0;
  unsigned numConvolutions = 0;
  std::vector<Tensor> inputEdges;
  std::vector<Tensor> outputEdges;
  std::vector<Tensor> weightEdges;
  std::vector<unsigned> weightReuseCount;

  for (unsigned wyBegin = kernelYBegin; wyBegin < kernelYEnd;
       wyBegin += convUnitWeightHeight) {
    const auto wyEnd = std::min(static_cast<unsigned>(kernelYEnd),
                                wyBegin + convUnitWeightHeight);
    unsigned convOutYBegin, convOutYEnd;
    std::tie(convOutYBegin, convOutYEnd) =
        getOutputRange({outYBegin, outYEnd}, stride[0], kernelSizeY,
                       padding[0], inDimY, {wyBegin, wyEnd},
                       isFractional);
    const auto convOutHeight = convOutYEnd - convOutYBegin;
    if (convOutHeight == 0)
      continue;
    for (unsigned wx = 0; wx != kernelSizeX; ++wx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRange({outXBegin, outXEnd}, stride[1], kernelSizeX,
                         padding[1], inDimX, wx,
                         isFractional);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;
      // In a fractionally strided pass, if we are handling one row of the
      // kernel at a time, the partitioning of work across the workers can be
      // aware of the stride and only allocate work on the rows that get
      // affected.
      unsigned outputStride =
          (isFractional && convUnitWeightHeight == 1) ? stride[0] : 1;
      std::vector<std::vector<PartialRow>> workerPartition =
          partitionConvPartialByWorker(convOutHeight, convOutWidth,
                                       contextsPerVertex, outputStride);
      assert(workerPartition.size() == contextsPerVertex);
      for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
        for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
          for (unsigned wy = wyBegin; wy != wyBegin + convUnitWeightHeight;
               ++wy) {
            Tensor w;
            if (wy < wyEnd) {
              w = weights[ozg][izg][wy][wx].flatten();
            } else {
              w = zeros.slice(0, inChansPerGroup * outChansPerGroup);
            }
            weightEdges.push_back(w);
          }
          for (unsigned i = 0; i != contextsPerVertex; ++i) {
            weightReuseCount.push_back(
              static_cast<std::uint32_t>(workerPartition[i].size())
            );

            for (const auto &partialRow : workerPartition[i]) {
              const auto workerOutY = convOutYBegin + partialRow.rowNumber;
              unsigned workerOutXBegin, workerOutXEnd;
              std::tie(workerOutXBegin, workerOutXEnd) =
                  getOutputRange({convOutXBegin + partialRow.begin,
                                  convOutXBegin + partialRow.end},
                                  stride[1], kernelSizeX, padding[1], inDimX,
                                  wx,
                                  isFractional);
              const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
              unsigned workerInXBegin, workerInXEnd;
              std::tie(workerInXBegin, workerInXEnd) =
                  getInputRange({workerOutXBegin, workerOutXEnd}, stride[1],
                                kernelSizeX, padding[1], inDimX, wx,
                                isFractional);
              const auto workerInWidth = workerInXEnd - workerInXBegin;
              for (unsigned wy = wyBegin; wy != wyBegin + convUnitWeightHeight;
                   ++wy) {
                const auto workerInY =
                    getInputIndex(workerOutY, stride[0], kernelSizeY,
                                  padding[0], inDimY, wy,
                                  isFractional);
                Tensor inWindow;
                if (workerInY == ~0U) {
                  inWindow = zeros.slice(0, workerInWidth * inChansPerGroup);
                } else {
                  inWindow =
                      in[izg][workerInY].slice(
                        {workerInXBegin, 0},
                        {workerInXEnd, inChansPerGroup}
                      ).reshape({workerInWidth * inChansPerGroup});
                }
                inputEdges.push_back(inWindow);
              }
              Tensor outWindow =
                  out[ozg][workerOutY].slice(
                    {workerOutXBegin, 0},
                    {workerOutXEnd, outChansPerGroup}
                  ).reshape({workerOutWidth * outChansPerGroup});
              // Note the output tensor is mapped in zeroAndMapPartialSums.
              outputEdges.push_back(outWindow);
              ++numConvolutions;
            }
          }
          ++numWeights;
        }
      }
    }
  }
  if (numConvolutions == 0)
    return;

  const auto numEdges = numConvolutions * convUnitWeightHeight
                        + numConvolutions
                        + numWeights * convUnitWeightHeight;

  auto v = graph.addVertex(fwdCS,
                      templateVertex("popconv::ConvPartialnx1InOut",
                                     dType, partialType,
                                     useDeltaEdgesForConvPartials(numEdges) ?
                                                          "true" : "false"));
  graph.setInitialValue(v["inStride"], isFractional ? 1 : stride.back());
  graph.setInitialValue(v["outStride"], isFractional ? stride.back() : 1);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                          convUnitCoeffLoadBytesPerCycle);
  graph.setTileMapping(v, tile);
  graph.connect(v["in"], inputEdges);
  graph.connect(v["out"], outputEdges);
  graph.connect(v["weights"], weightEdges);
  graph.setInitialValue(v["weightReuseCount"], weightReuseCount);
}

struct ConvOutputSlice {
  unsigned outXBegin;
  unsigned outXEnd;
  unsigned outY;
  unsigned outZGroup;
  ConvOutputSlice(unsigned outXBegin, unsigned outXEnd, unsigned outY,
                  unsigned outZGroup) :
    outXBegin(outXBegin), outXEnd(outXEnd),
    outY(outY), outZGroup(outZGroup) {}

};

static void
createConvPartialHorizontalMacVertex(
    Graph &graph,
    unsigned tile,
    const std::vector<ConvOutputSlice> &outRegions,
    unsigned kernelYBegin, unsigned kernelYEnd,
    unsigned inZGroupBegin, unsigned inZGroupEnd,
    const std::vector<unsigned> &stride,
    const std::vector<unsigned> &padding,
    ComputeSet fwdCS,
    const Tensor &in,
    const Tensor &weights,
    const Tensor &out,
    bool isFractional) {
  const auto kernelHeight = weights.dim(2);
  const auto kernelWidth = weights.dim(3);
  const auto dataPathWidth = graph.getDevice().getDeviceInfo().dataPathWidth;
  const auto inHeight = in.dim(1);
  const auto inWidth = in.dim(2);
  const auto dType = graph.getTensorElementType(in);
  const auto partialType = graph.getTensorElementType(out);
  const auto outChansPerGroup = out.dim(3);
  assert(outChansPerGroup == 1);
  (void)outChansPerGroup;
  std::vector<Tensor> inEdges;
  std::vector<Tensor> weightsEdges;
  std::vector<Tensor> outEdges;
  for (const auto &region : outRegions) {
    const auto ozg = region.outZGroup;
    const auto y = region.outY;
    const auto outXBegin = region.outXBegin;
    const auto outXEnd = region.outXEnd;
    for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
      for (unsigned ky = kernelYBegin; ky != kernelYEnd; ++ky) {
        for (unsigned kx = 0; kx != kernelWidth; ++kx) {
          unsigned inY = getInputIndex(y, stride[0], kernelHeight,
                                       padding[0], inHeight, ky,
              isFractional);
          if (inY == ~0U)
            continue;
          auto inRange =
              getInputRange({outXBegin, outXEnd}, stride[1], kernelWidth,
                            padding[1], inWidth, kx, isFractional);
          if (inRange.first == inRange.second)
            continue;
          auto outRange =
              getOutputRange({outXBegin, outXEnd}, stride[1], kernelWidth,
                             padding[1], inWidth, kx, isFractional);
          Tensor inWindow =
              in[izg][inY].slice(inRange.first, inRange.second).flatten();
          Tensor weightsWindow = weights[ozg][izg][ky][kx].flatten();
          Tensor outWindow =
              out[ozg][y].slice(outRange.first, outRange.second).flatten();
          inEdges.emplace_back(std::move(inWindow));
          weightsEdges.emplace_back(std::move(weightsWindow));
          outEdges.emplace_back(std::move(outWindow));
        }
      }
    }
  }
  if (outEdges.empty())
    return;
  auto v = graph.addVertex(fwdCS,
                           templateVertex(
                             "popconv::ConvPartialHorizontalMac", dType,
                             partialType
                           ),
                           {{"in", inEdges},
                            {"weights", weightsEdges},
                            {"out", outEdges},
                           });
  graph.setInitialValue(v["inStride"], isFractional ? 1 : stride.back());
  graph.setInitialValue(v["outStride"], isFractional ? stride.back() : 1);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setTileMapping(v, tile);
}

static void
zeroAndMapPartialSums(Graph &graph,
                      unsigned outXBegin, unsigned outXEnd,
                      unsigned outYBegin, unsigned outYEnd,
                      unsigned tileOutZGroupBegin, unsigned tileOutZGroupEnd,
                      unsigned tile,
                      ComputeSet zeroCS,
                      const Tensor &out) {
  Tensor flatOut = out.flatten();
  std::vector<Interval<std::size_t>> regions;
  for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
    for (unsigned y = outYBegin; y != outYEnd; ++y) {
      const auto regionBegin = out.dim(3) *
                                (outXBegin + out.dim(2) *
                                 (y + out.dim(1) *
                                  ozg));
      const auto regionEnd = regionBegin + out.dim(3) * (outXEnd - outXBegin);
      graph.setTileMapping(flatOut.slice(regionBegin, regionEnd), tile);
      regions.emplace_back(regionBegin, regionEnd);
    }
  }
  mergeAdjacentRegions(regions);
  return zero(graph, out, regions, tile, zeroCS);
}

static bool writtenRangeEqualsOutputRange(
    std::pair<unsigned, unsigned> outRange,
    unsigned stride,
    unsigned padding,
    unsigned kernelSize,
    std::pair<unsigned, unsigned> kernelIndexRange,
    unsigned inDim, bool isFractional) {
  auto writtenYRange =
      getOutputRange(outRange, stride, kernelSize, padding,
                     inDim, kernelIndexRange, isFractional);
  return writtenYRange == outRange;
}

static std::vector<std::vector<ConvOutputSlice>>
partitionConvOutputBetweenWorkers(const Graph &graph,
                                  unsigned outXBegin, unsigned outXEnd,
                                  unsigned outYBegin, unsigned outYEnd,
                                  unsigned outZGroupBegin,
                                  unsigned outZGroupEnd) {
  std::vector<std::vector<ConvOutputSlice>> perWorkerConvOutputSlices;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto outWidth = outXEnd - outXBegin;
  const auto outHeight = outYEnd - outYBegin;
  const auto numRows = outHeight * (outZGroupEnd - outZGroupBegin);
  const auto numWorkers = deviceInfo.numWorkerContexts;
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numRows);
  unsigned numPartRows = numRows * rowSplitFactor;
  for (unsigned worker = 0; worker != numWorkers; ++worker) {
    const auto begin = (worker * numPartRows) / numWorkers;
    const auto end = ((worker + 1) * numPartRows) / numWorkers;
    perWorkerConvOutputSlices.emplace_back();
    for (unsigned partRow = begin; partRow != end; ++partRow) {
      auto row = partRow / rowSplitFactor;
      auto partInRow = partRow % rowSplitFactor;
      const auto ozg = outZGroupBegin + row / outHeight;
      const auto y = outYBegin + row % outHeight;
      const auto workerOutXBegin =
          outXBegin + (partInRow * outWidth) / rowSplitFactor;
      const auto workerOutXEnd =
          outXBegin + ((partInRow + 1) * outWidth) / rowSplitFactor;
      if (!perWorkerConvOutputSlices.back().empty() &&
          ozg == perWorkerConvOutputSlices.back().back().outZGroup &&
          y == perWorkerConvOutputSlices.back().back().outY) {
        perWorkerConvOutputSlices.back().back().outXEnd = workerOutXEnd;
      } else {
        perWorkerConvOutputSlices.back().emplace_back(workerOutXBegin,
                                                      workerOutXEnd, y, ozg);
      }
    }
  }
  return perWorkerConvOutputSlices;
}

static void
calcPartialConvOutput(Graph &graph,
                      const Plan &plan,
                      std::string dType,
                      unsigned tile,
                      unsigned outXBegin, unsigned outXEnd,
                      unsigned outYBegin, unsigned outYEnd,
                      unsigned outZGroupBegin, unsigned outZGroupEnd,
                      unsigned kernelYBegin, unsigned kernelYEnd,
                      unsigned inZGroupBegin, unsigned inZGroupEnd,
                      const std::vector<unsigned> &stride,
                      const std::vector<unsigned> &padding,
                      ComputeSet zeroCS,
                      ComputeSet fwdCS,
                      Tensor in, Tensor weights, Tensor out,
                      bool isFractional) {
  const auto tileKernelHeight = kernelYEnd - kernelYBegin;
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);

  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  Tensor zeros;
  bool useConvPartial1x1OutVertex = false;
  if (plan.useConvolutionInstructions) {
    const auto inDimY = in.dim(1);
    useConvPartial1x1OutVertex =
        kernelSizeX == 1 && tileKernelHeight == 1 &&
        (!isFractional || (stride[1] == 1 && stride[0] == 1)) &&
        writtenRangeEqualsOutputRange({outYBegin, outYEnd}, stride[0],
                                      padding[0], kernelSizeY,
                                      {kernelYBegin, kernelYEnd}, inDimY,
                                      isFractional);
    const auto weightsPerConvUnit =
        deviceInfo.getWeightsPerConvUnit(dType == "float");
    assert(weightsPerConvUnit % inChansPerGroup == 0);
    const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
    if (!useConvPartial1x1OutVertex && convUnitWeightHeight != 1) {
      assert(plan.useConvolutionInstructions);
      const auto inDimX = in.dim(2);
      const auto inputRange = getInputRange({outXBegin, outXEnd}, stride[1],
                                            kernelSizeX, padding[1],
                                            inDimX, isFractional);
      const auto inputRangeSize = inputRange.second - inputRange.first;
      // This isn't split across multiple workers since it can happen in
      // parallel with zeroing the partial sums.
      const auto zeroSize = std::max(inputRangeSize * inChansPerGroup,
                                     inChansPerGroup * outChansPerGroup);
      zeros = graph.addTensor(dType,
                              {zeroSize},
                              "zeros");
      auto v = graph.addVertex(zeroCS, templateVertex("popstd::Zero", dType),
                               {{"out", zeros}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
      graph.setTileMapping(zeros, tile);
    }
  }
  if (useConvPartial1x1OutVertex) {
    for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
      createConvPartial1x1OutVertex(graph, tile,
                                    outXBegin, outXEnd,
                                    outYBegin, outYEnd, ozg,
                                    kernelYBegin, inZGroupBegin, inZGroupEnd,
                                    stride, padding, fwdCS, in, weights, out);
    }
  } else {
    zeroAndMapPartialSums(graph, outXBegin, outXEnd, outYBegin, outYEnd,
                          outZGroupBegin, outZGroupEnd, tile, zeroCS, out);
    if (plan.useConvolutionInstructions) {
      createConvPartialnx1InOutVertex(graph, tile, outXBegin, outXEnd,
                                      outYBegin, outYEnd,
                                      outZGroupBegin, outZGroupEnd,
                                      kernelYBegin, kernelYEnd,
                                      inZGroupBegin, inZGroupEnd,
                                      stride, padding,
                                      fwdCS, in, weights, out,
                                      zeros, isFractional);
    } else {
      auto perWorkerConvOutputSlices =
          partitionConvOutputBetweenWorkers(graph, outXBegin, outXEnd,
                                            outYBegin, outYEnd,
                                            outZGroupBegin, outZGroupEnd);
      for (const auto &workerConvOutputSlices : perWorkerConvOutputSlices) {
        createConvPartialHorizontalMacVertex(graph, tile,
                                             workerConvOutputSlices,
                                             kernelYBegin, kernelYEnd,
                                             inZGroupBegin, inZGroupEnd,
                                             stride, padding,
                                             fwdCS, in, weights, out,
                                             isFractional);
      }
    }
  }
}

// Take an ordered list and return a list of ranges
// representing the contiguous regions in that list.
template <typename It>
static std::vector<std::pair<typename std::iterator_traits<It>::value_type,
                             typename std::iterator_traits<It>::value_type>>
getContiguousRegions(It begin,
                     It end)
{
  using T = typename std::iterator_traits<It>::value_type;
  std::vector<std::pair<T, T>> regions;
  unsigned curBegin = *begin;
  unsigned curEnd = curBegin + 1;
  auto it = begin + 1;
  while (it != end) {
    if (*it == curEnd) {
      ++curEnd;
    } else {
      regions.emplace_back(curBegin, curEnd);
      curBegin = *it;
      curEnd = curBegin + 1;
    }
    ++it;
  }
  regions.emplace_back(curBegin, curEnd);
  return regions;
}

static Program
calcPartialSums(Graph &graph,
                const Plan &plan,
                const std::vector<unsigned> &stride,
                const std::vector<unsigned> &padding,
                unsigned outNumChans,
                std::string dType,
                Tensor in, Tensor weights, Tensor partials,
                const std::string &layerName,
                unsigned outDimX, unsigned outDimY,
                bool isFractional) {
  const auto numBatchGroups = in.dim(0);
  const auto isMultiIPU = graph.getDevice().getDeviceInfo().numIPUs > 1;
  const auto inNumChans = in.dim(1) * in.dim(4);
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;
  const auto kernelHeight = weights.dim(2);
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();

  ComputeSet zeroCS = graph.addComputeSet(layerName +"/Zero");
  ComputeSet convolveCS = graph.addComputeSet(layerName + "/Convolve");
  for (unsigned b = 0; b < numBatchGroups; ++b) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
      for (unsigned ky = 0; ky != tilesPerKernelY; ++ky) {
        const auto kernelYBegin = (ky * kernelHeight) / tilesPerKernelY;
        const auto kernelYEnd = ((ky + 1) * kernelHeight) / tilesPerKernelY;
        for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
          unsigned outZGroupBegin, outZGroupEnd;
          std::tie(outZGroupBegin, outZGroupEnd) =
              getOutZGroupRange(ozg, partialNumChanGroups, plan);
          if (outZGroupBegin == outZGroupEnd)
            continue;
          for (unsigned oy = 0; oy != tilesPerY; ++oy) {
            const auto outYBegin = (oy * outDimY) / tilesPerY;
            const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
            for (unsigned ox = 0; ox != tilesPerX; ++ox) {
              const auto outXBegin = (ox * outDimX) / tilesPerX;
              const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
              const auto tile = linearizeTileIndices(b, numBatchGroups,
                                                     numTiles,
                                                     ky, izg,
                                                     ox, oy, ozg,
                                                     plan,
                                                     isMultiIPU);
              calcPartialConvOutput(graph, plan, dType, tile,
                                    outXBegin, outXEnd, outYBegin, outYEnd,
                                    outZGroupBegin, outZGroupEnd,
                                    kernelYBegin, kernelYEnd,
                                    inZGroupBegin,
                                    inZGroupEnd,
                                    stride, padding, zeroCS,
                                    convolveCS,
                                    in[b], weights,
                                    partials[b][izg][ky],
                                    isFractional);
            }
          }
        }
      }
    }
  }
  Sequence prog;
  if (!graph.getComputeSet(zeroCS).empty()) {
    prog.add(Execute(zeroCS));
  }
  prog.add(Execute(convolveCS));
  return prog;
}

/// Group tiles based on the regions of the reduced output their partial sums
/// contribute to.
static void
groupConvTilesByOutput(
    const poplar::Graph &graph,
    unsigned batchGroup,
    unsigned numBatchGroups,
    const std::vector<std::size_t> &reducedDims,
    const Plan &plan,
    std::vector<std::vector<unsigned>> &tileGroups,
    std::vector<
      std::vector<Interval<std::size_t>>
    > &tileGroupRegions) {
  const auto isMultiIPU = graph.getDevice().getDeviceInfo().numIPUs > 1;
  const auto partialNumChanGroups = reducedDims[0];
  const auto outDimY = reducedDims[1];
  const auto outDimX = reducedDims[2];
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  tileGroups.clear();
  tileGroups.reserve(tilesPerZ * tilesPerY * tilesPerX);
  tileGroupRegions.clear();
  tileGroupRegions.reserve(tilesPerZ * tilesPerY * tilesPerX);
  for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
    unsigned outZGroupBegin, outZGroupEnd;
    std::tie(outZGroupBegin, outZGroupEnd) =
        getOutZGroupRange(ozg, partialNumChanGroups, plan);
    if (outZGroupBegin == outZGroupEnd)
      continue;
    for (unsigned oy = 0; oy != tilesPerY; ++oy) {
      const auto outYBegin = (oy * outDimY) / tilesPerY;
      const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
      for (unsigned ox = 0; ox != tilesPerX; ++ox) {
        tileGroups.emplace_back();
        tileGroupRegions.emplace_back();
        const auto outXBegin = (ox * outDimX) / tilesPerX;
        const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
        addFlattenedRegions(reducedDims,
                            {outZGroupBegin, outYBegin, outXBegin, 0},
                            {outZGroupEnd, outYEnd, outXEnd, reducedDims[3]},
                            tileGroupRegions.back());
        for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
          for (unsigned ky = 0; ky != tilesPerKernelY; ++ky) {
            const auto tile = linearizeTileIndices(batchGroup, numBatchGroups,
                                                   numTiles, ky, izg,
                                                   ox, oy, ozg,
                                                   plan, isMultiIPU);
            tileGroups.back().push_back(tile);
          }
        }
        mergeAdjacentRegions(tileGroupRegions.back());
        std::sort(tileGroups.back().begin(), tileGroups.back().end());
      }
    }
  }
}

static Tensor
partialGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval<std::size_t>>> &
        tileGroupRegions,
    const Tensor &partials,
    unsigned outDepth,
    const std::string &resultType,
    ComputeSet cs) {
  const auto partialsDepth = partials.dim(0);
  assert(partialsDepth >= outDepth);
  auto outDims = partials.shape();
  outDims[0] = outDepth;
  Tensor out = graph.addTensor(resultType,
                               outDims,
                               "partialReduceOut");
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numTileGroups = tileGroupRegions.size();
  const auto grainSize =
      resultType == "float" ? deviceInfo.getFloatVectorWidth() :
                              deviceInfo.getHalfVectorWidth();
  for (unsigned i = 0; i != outDepth; ++i) {
    unsigned begin = (i * partialsDepth) / outDepth;
    unsigned end = ((i + 1) * partialsDepth) / outDepth;
    std::vector<std::vector<Interval<std::size_t>>>
        outSubMapping(numTiles);
    for (unsigned tileGroup = 0; tileGroup != numTileGroups; ++tileGroup) {
      const auto tilesInGroup = tileGroups[tileGroup].size();
      const auto tileBegin = (i * tilesInGroup) / outDepth;
      const auto tileEnd = ((i + 1) * tilesInGroup) / outDepth;
      const auto outSplitRegions =
          splitRegions(tileGroupRegions[tileGroup], grainSize,
                       tileEnd - tileBegin);
      for (unsigned j = 0; j != outSplitRegions.size(); ++j) {
        outSubMapping[tileGroups[tileGroup][j + tileBegin]] =
            outSplitRegions[j];
      }
    }
    applyTensorMapping(graph, out[i], outSubMapping);
    popreduce::reduce(graph, partials.slice(begin, end), out[i],
                      outSubMapping, cs);
  }
  return out;
}

static Tensor
groupedReduce(Graph &graph,
              const std::vector<std::vector<unsigned>> &tileGroups,
              const std::vector<
                std::vector<Interval<std::size_t>>
              > &tileGroupRegions,
              const Tensor &partials,
              const std::string &resultType,
              ComputeSet cs) {
  return partialGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
         1, resultType, cs).reshape(partials[0].shape());
}

/// Return the number of reduce stages to use for a reduction of the specified
/// reduction depth.
static unsigned getNumReduceStages(unsigned partialsDepth) {
  /// Using more reduce stages affects code size as follows.
  /// If the reduction depth is p then a single stage reduction requires each
  /// tile to receive p messages. If instead we break the reduction down into n
  /// stages then each stage involves a reduction of reduce p^(1/n) messages.
  /// The total number of messages is n*p^(1/n). For large p, increase n
  /// will reducing the total number of messages received which is turn likely
  /// to also reduce the exchange code size. The thresholds below have been
  /// chosen based on benchmarking.
  if (partialsDepth >= 125)
    return 3;
  if (partialsDepth >= 16)
    return 2;
  return 1;
}

/// Return a plan for how to split a reduction into multiple stages along with
/// an estimate of the cost of the plan. The first member of the pair is a
/// vector of the depth of each partials tensor in each intermediate stage.
/// If the vector is empty there are no intermediate stages and the reduction
/// is performed in a single step. The second member of the pair is an
/// estimated cost. The cost is an estimate of the average number of messages
/// required per tile.
static std::pair<std::vector<unsigned>, float>
getMultiStageReducePlanAndCost(unsigned partialsDepth, unsigned numStages) {
  if (numStages == 1) {
    return {{}, partialsDepth};
  }
  auto nextDepthRoundDown =
      static_cast<unsigned>(
        std::pow(static_cast<double>(partialsDepth),
                 (numStages - 1.0) / numStages)
      );
  std::vector<unsigned> roundDownPlan, roundUpPlan;
  float roundDownCost, roundUpCost;
  std::tie(roundDownPlan, roundDownCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundDown, numStages - 1);
  roundDownCost += static_cast<float>(partialsDepth) / nextDepthRoundDown;
  auto nextDepthRoundUp = nextDepthRoundDown + 1;
  std::tie(roundUpPlan, roundUpCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundUp, numStages - 1);
  roundUpCost += static_cast<float>(partialsDepth) / nextDepthRoundUp;
  if (roundDownCost < roundUpCost) {
    roundDownPlan.insert(roundDownPlan.begin(), nextDepthRoundDown);
    return {roundDownPlan, roundDownCost};
  }
  roundUpPlan.insert(roundUpPlan.begin(), nextDepthRoundUp);
  return {roundUpPlan, roundUpCost};
}

static std::vector<unsigned>
getMultiStageReducePlan(unsigned partialsDepth) {
  const auto numStages = getNumReduceStages(partialsDepth);
  return getMultiStageReducePlanAndCost(partialsDepth, numStages).first;
}

static Tensor
multiStageGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval<std::size_t>>> &
        tileGroupRegions,
    Tensor partials,
    const std::string &resultType,
    std::vector<ComputeSet> &computeSets,
    const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  auto plan = getMultiStageReducePlan(partialsDepth);
  for (unsigned i = computeSets.size(); i <= plan.size(); ++i) {
    computeSets.push_back(
      graph.addComputeSet(debugPrefix + "/Reduce" +
                             std::to_string(i))
    );
  }
  const auto partialsType = graph.getTensorElementType(partials);
  for (unsigned i = 0; i != plan.size(); ++i) {
    partials = partialGroupedReduce(graph, tileGroups, tileGroupRegions,
                                    partials, plan[i], partialsType,
                                    computeSets[i]);
  }
  auto reduced = groupedReduce(graph, tileGroups, tileGroupRegions, partials,
                               resultType, computeSets[plan.size()]);
  return reduced;
}

static Tensor
convReduceByPartialMapping(Graph &graph, unsigned batchGroup,
                           unsigned numBatchGroups,
                           const Tensor &partials,
                           const std::string &resultType, const Plan &plan,
                           std::vector<ComputeSet> &computeSets,
                           const std::string &debugPrefix) {
  std::vector<
    std::vector<Interval<std::size_t>>
  > tileGroupRegions;
  std::vector<std::vector<unsigned>> tileGroups;
  groupConvTilesByOutput(graph, batchGroup, numBatchGroups, partials[0].shape(),
                         plan, tileGroups, tileGroupRegions);
  return multiStageGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
                                 resultType, computeSets, debugPrefix);
}

static std::pair<Program, Tensor>
convolutionByAmp(Graph &graph, const Plan &plan,
                 const std::vector<unsigned> &stride,
                 const std::vector<unsigned> &padding,
                 const Tensor &in, const Tensor &weights, unsigned outDimY,
                 unsigned outDimX, bool isFractional,
                 const std::string &debugPrefix) {
  if (isFractional) {
    assert(absdiff(outDimY + 2 * padding[0], weights.dim(2)) /
           stride[0] + 1 == in.dim(2));
    assert(absdiff(outDimX + 2 * padding[1], weights.dim(3)) /
           stride[1] + 1 == in.dim(3));
  } else {
    assert(absdiff(in.dim(2) + 2 * padding[0], weights.dim(2)) /
           stride[0] + 1 == outDimY);
    assert(absdiff(in.dim(3) + 2 * padding[1], weights.dim(3)) /
           stride[1] + 1 == outDimX);
  }
  const auto numBatchGroups = in.dim(0);
  Sequence prog;
  const auto dType = graph.getTensorElementType(in);
  const auto outNumChans = weights.dim(0) * weights.dim(4);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;

  const auto partialType = plan.getPartialType();
  const auto batchSize = numBatchGroups * plan.batchesPerGroup;
  mapWeights(weights, graph, plan, batchSize);

  // Calculate a set of partial sums of the convolutions.
  Tensor partials = graph.addTensor(partialType,
                                     {numBatchGroups,
                                      tilesPerInZGroup,
                                      tilesPerKernelY,
                                      partialNumChanGroups,
                                      outDimY,
                                      outDimX,
                                      partialChansPerGroup},
                                    "partials");
  prog.add(calcPartialSums(graph, plan, stride, padding, outNumChans,
                           dType, in, weights, partials, debugPrefix,
                           outDimX, outDimY, isFractional));

  std::vector<ComputeSet> reduceComputeSets;
  // For each element of the batch, we add the reduction vertices to same
  // compute sets so the batch will be executed in parallel.
  Tensor reduced;
  // Perform the reduction of partial sums.
  partials = partials.reshape({numBatchGroups,
                               tilesPerInZGroup * tilesPerKernelY,
                               partialNumChanGroups,
                               outDimY,
                               outDimX,
                               partialChansPerGroup});
  if (partials.dim(1) == 1) {
    if (dType != partialType) {
      reduced = graph.addTensor(dType, partials.shape(), "reduced");
      if (reduceComputeSets.empty()) {
        reduceComputeSets.push_back(graph.addComputeSet(debugPrefix +
                                                           "/Cast"));
      }
      applyTensorMapping(graph, reduced, graph.getTileMapping(partials));
      cast(graph, partials, reduced, reduceComputeSets[0]);
    } else {
      reduced = partials;
    }
    reduced = reduced.reshape({reduced.dim(0),
                               reduced.dim(2),
                               reduced.dim(3),
                               reduced.dim(4),
                               reduced.dim(5)});
  } else {
    auto reducedShape = partials[0][0].shape();
    reducedShape.insert(reducedShape.begin(), 0);
    reduced = graph.addTensor(dType, reducedShape, "reduced");
    for (unsigned b = 0; b < numBatchGroups; ++b) {
      reduced =
          append(reduced,
                 convReduceByPartialMapping(graph, b, numBatchGroups,
                                            partials[b],
                                            dType, plan, reduceComputeSets,
                                            debugPrefix));
    }
  }
  for (const auto &cs : reduceComputeSets) {
    prog.add(Execute(cs));
  }
  return {prog, reduced};
}

static std::string
appendDims(std::string s, const std::vector<unsigned> &dims) {
  bool first = true;
  for (auto dim : dims) {
    if (!first)
      s += 'x';
    s += std::to_string(dim);
    first = false;
  }
  return s;
}

static std::string
convSuffix(const std::vector<unsigned> &kernelSize,
                       const std::vector<unsigned> &stride,
                       bool isFractional) {
  std::string s = appendDims("_", kernelSize);
  s += (isFractional ? "_fractional_stride" : "_stride");
  s = appendDims(std::move(s), stride);
  return s;
}

Tensor
convolution(Graph &graph,
            const std::vector<unsigned> &stride,
            const std::vector<unsigned> &padding,
            unsigned outNumChans,
            Tensor in, Tensor weights,
            const std::string &partialsType, bool isFractional,
            bool transposeAndFlipWeights, Sequence &prog,
            const std::string &debugPrefix,
            const ConvOptions &options) {
  const auto dType = graph.getTensorElementType(in);
  const auto batchSize = in.dim(0);
  if (transposeAndFlipWeights) {
    // Create transposed/flipped weights
    const auto outNumChans = weights.dim(1) * weights.dim(5);
    const auto kernelSizeY = weights.dim(2);
    const auto kernelSizeX = weights.dim(3);
    auto bwdWeights = createWeights(graph, in,
                                    kernelSizeY, kernelSizeX,
                                    outNumChans, stride[0], stride[1],
                                    padding[0], padding[1], isFractional,
                                    options);
    mapWeights(bwdWeights, graph, in, stride[0], stride[1], padding[0],
               padding[1], isFractional, options);
    weightsTransposeChansFlipXY(graph, weights, bwdWeights, prog, debugPrefix);
    weights = bwdWeights;
  }
  const auto plan = getPlan(graph, dType,
                            in.dim(0), in.dim(2), in.dim(3),
                            in.dim(1) * in.dim(4),
                            {weights.dim(2), weights.dim(3),
                             weights.dim(0) * weights.dim(4)},
                            stride, padding,
                            isFractional, options);
  if (in.dim(4) != plan.inChansPerGroup) {
    in = regroup(in, plan.inChansPerGroup);
  }
  const unsigned kernelSizeY = weights.dim(2);
  const unsigned kernelSizeX = weights.dim(3);
  std::size_t outDimY;
  std::size_t outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(in.dim(2), in.dim(3),
                                            kernelSizeY, kernelSizeX,
                                            stride[0], stride[1],
                                            padding[0], padding[1],
                                            isFractional);
  if (plan.useWinograd) {
    auto activations =
        graph.addTensor(dType, {batchSize,
                                outNumChans / plan.partialChansPerGroup,
                                outDimY, outDimX,
                                plan.partialChansPerGroup});
    ::mapActivations(graph, activations);
    prog.add(winogradConvolution(graph, stride[0], stride[1],
                                 padding[0], padding[1],
                                 in, weights, activations,
                                 partialsType,
                                 plan.winogradPatchSize, plan.winogradPatchSize,
                                 debugPrefix));
    return activations;
  }


  assert(plan.getPartialType() == partialsType);

  const auto layerName =
      debugPrefix + "/Conv" + convSuffix({kernelSizeY, kernelSizeX}, stride,
                                         isFractional);
  unsigned partialOutDimY, partialOutDimX;
  assert(batchSize % plan.batchesPerGroup == 0);
  const auto numBatchGroups = batchSize / plan.batchesPerGroup;
  if (plan.flattenXY) {
    partialOutDimY = plan.batchesPerGroup;
    partialOutDimX = outDimX * outDimY;
    const auto inDimY = in.dim(2);
    const auto inDimX = in.dim(3);
    in = in.dimShuffle({1, 0, 2, 3, 4}).reshape(
                          {in.dim(1),
                           numBatchGroups,
                           plan.batchesPerGroup * inDimY,
                           inDimX,
                           in.dim(4)
                          }).dimShuffle({1, 0, 2, 3, 4});

    in = in.reshape({numBatchGroups,
                     in.dim(1),
                     plan.batchesPerGroup,
                     inDimY * inDimX,
                     in.dim(4)});

  } else {
    partialOutDimY = outDimY;
    partialOutDimX = outDimX;
  }

  Program convolveProg;
  Tensor activations;
  std::tie(convolveProg, activations) =
    convolutionByAmp(graph, plan, stride, padding, in, weights,
                     partialOutDimY, partialOutDimX, isFractional, layerName);
  if (plan.flattenXY) {
    activations = activations.dimShuffle({0, 2, 3, 1, 4})
          .reshape({batchSize,
                    outDimY,
                    outDimX,
                    activations.dim(1),
                    activations.dim(4)})
          .dimShuffle({0, 3, 1, 2, 4});
  }
  prog.add(convolveProg);
  return activations;
}

static std::uint64_t getNumberOfMACs(unsigned outDimY, unsigned outDimX,
                                     unsigned outNumChans,
                                     unsigned kernelSizeY, unsigned kernelSizeX,
                                     unsigned strideY, unsigned strideX,
                                     unsigned paddingY, unsigned paddingX,
                                     unsigned inDimY, unsigned inDimX,
                                     unsigned inNumChans) {
  std::uint64_t numMACs = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, strideY, kernelSizeY,
                                               paddingY, inDimY, false);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, strideX, kernelSizeX,
                                                 paddingX, inDimX, false);
      const auto width = inXEnd - inXBegin;
      numMACs += width * height * outNumChans * inNumChans;
    }
  }
  return numMACs;
}


static uint64_t getFlops(unsigned batchSize,
                         unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                         unsigned kernelSizeY, unsigned kernelSizeX,
                         unsigned strideY, unsigned strideX,
                         unsigned paddingY, unsigned paddingX,
                         unsigned outNumChans) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSizeY,
                                            kernelSizeX, strideY, strideX,
                                            paddingY, paddingX, false);
  auto flopsPerItem =
      2 * getNumberOfMACs(outDimY, outDimX, outNumChans,
                          kernelSizeY, kernelSizeX, strideY, strideX,
                          paddingY, paddingX,
                          inDimY, inDimX, inNumChans);
  return batchSize * flopsPerItem;
}


uint64_t getFwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned inNumChans,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     unsigned outNumChans) {
  return getFlops(batchSize, inDimY, inDimX, inNumChans, kernelSizeY,
                  kernelSizeX, strideY, strideX, paddingY, paddingX,
                  outNumChans);
}

uint64_t getBwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned inNumChans,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     unsigned outNumChans) {
  return getFlops(batchSize, inDimY, inDimX, inNumChans, kernelSizeY,
                  kernelSizeX, strideY, strideX, paddingY, paddingX,
                  outNumChans);
}

uint64_t getWuFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned inNumChans,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     unsigned outNumChans) {
  return getFlops(batchSize, inDimY, inDimX, inNumChans, kernelSizeY,
                  kernelSizeX, strideY, strideX, paddingY, paddingX,
                  outNumChans);
}

static double getPerfectCycleCount(const Graph &graph,
                                   std::string dType,
                                   unsigned batchSize,
                                   unsigned inDimY, unsigned inDimX,
                                   unsigned inNumChans,
                                   unsigned kernelSizeY, unsigned kernelSizeX,
                                   unsigned strideY, unsigned strideX,
                                   unsigned paddingY, unsigned paddingX,
                                   unsigned outNumChans) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSizeY,
                                            kernelSizeX, strideY,
                                            strideX, paddingY, paddingX,
                                            false);
  const auto numTiles = deviceInfo.getNumTiles();
  auto numMacs =
      batchSize * getNumberOfMACs(outDimY, outDimX, outNumChans, kernelSizeY,
                                  kernelSizeX, strideY, strideX,
                                  paddingY, paddingX, inDimY, inDimX,
                                  inNumChans);

  if (dType == "float") {
    const auto floatVectorWidth = deviceInfo.getFloatVectorWidth();

    auto macCycles =
       static_cast<double>(numMacs) / (floatVectorWidth * numTiles);

    return macCycles;
  }
  assert(dType == "half");
  const auto convUnitsPerTile =
      std::max(std::max(deviceInfo.fp16InFp16OutConvUnitsPerTile,
                        deviceInfo.fp32InFp32OutConvUnitsPerTile),
               deviceInfo.fp16InFp32OutConvUnitsPerTile);
  const auto halfVectorWidth = deviceInfo.getHalfVectorWidth();
  auto macsPerCycle = convUnitsPerTile * halfVectorWidth;
  auto macCycles = static_cast<double>(numMacs) / (macsPerCycle * numTiles);
  return macCycles;
}

double getFwdPerfectCycleCount(const Graph &graph,
                               std::string dType,
                               unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned inNumChans,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX,
                               unsigned outNumChans) {
  return getPerfectCycleCount(graph, dType, batchSize, inDimY, inDimX,
                              inNumChans, kernelSizeY, kernelSizeX,
                              strideY, strideX, paddingY, paddingX,
                              outNumChans);
}

double getBwdPerfectCycleCount(const Graph &graph,
                               std::string dType,
                               unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned inNumChans,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX,
                               unsigned outNumChans) {
  return getPerfectCycleCount(graph, dType, batchSize, inDimY, inDimX,
                              inNumChans, kernelSizeY, kernelSizeX,
                              strideY, strideX, paddingY, paddingX,
                              outNumChans);
}

double getWuPerfectCycleCount(const Graph &graph,
                              std::string dType,
                              unsigned batchSize,
                              unsigned inDimY, unsigned inDimX,
                              unsigned inNumChans,
                              unsigned kernelSizeY, unsigned kernelSizeX,
                              unsigned strideY, unsigned strideX,
                              unsigned paddingY, unsigned paddingX,
                              unsigned outNumChans) {
  return getPerfectCycleCount(graph, dType, batchSize, inDimY, inDimX,
                              inNumChans, kernelSizeY, kernelSizeX,
                              strideY, strideX, paddingY, paddingX,
                              outNumChans);
}

std::vector<size_t> getElementCoord(size_t element,
                                    const std::vector<size_t> dims) {
  std::vector<size_t> coord(dims.size());
  for (int i = dims.size() - 1; i >= 0; --i) {
    coord[i] = element % dims[i];
    element = element / dims[i];
  }
  return coord;
}

/**
 * Transpose the innermost pair of dimensions of the specified tensor, writing
 * the results to a new tensor.
 */
static Tensor weightsPartialTranspose(Graph &graph, Tensor in, ComputeSet cs) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto rank = in.rank();
  const auto numSrcRows = in.dim(rank - 2);
  const auto numSrcColumns = in.dim(rank - 1);
  const auto dType = graph.getTensorElementType(in);
  auto outShape = in.shape();
  std::swap(outShape[rank - 2], outShape[rank - 1]);
  auto out = graph.addTensor(dType, outShape, "partialTranspose");
  auto inFlat = in.reshape({in.numElements() / (numSrcRows * numSrcColumns),
                            numSrcRows * numSrcColumns});
  auto outFlat = out.reshape(inFlat.shape());
  const auto transpositionMapping =
      graph.getTileMapping(inFlat.slice(0, 1, 1));
  const auto numTiles = transpositionMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerTranspositions =
        splitRegionsBetweenWorkers(deviceInfo, transpositionMapping[tile], 1);
    for (const auto &entry : perWorkerTranspositions) {
      const auto v =
          graph.addVertex(cs, templateVertex("popconv::Transpose2D", dType));
      graph.setInitialValue(v["numSrcColumns"],
                            static_cast<unsigned>(numSrcColumns));
      graph.setTileMapping(v, tile);
      unsigned i = 0;
      for (const auto &interval : entry) {
        for (auto transposition = interval.begin();
             transposition != interval.end(); ++transposition) {
          graph.connect(v["src"][i], inFlat[transposition]);
          graph.connect(v["dst"][i], outFlat[transposition]);
          graph.setTileMapping(outFlat[transposition], tile);
          ++i;
        }
      }
      graph.setFieldSize(v["src"], i);
      graph.setFieldSize(v["dst"], i);
    }
  }
  return out;
}

/** Copy the weights in 'weightsIn' into 'weightsOut' such that
 *  each element of the kernel is transposed w.r.t. the input and output
 *  channels and flip both the X and Y axis of the kernel field.
 */
void weightsTransposeChansFlipXY(Graph &graph,
                                 Tensor weightsIn,
                                 Tensor weightsOut,
                                 Sequence &prog,
                                 const std::string &debugPrefix) {
  // weightsIn = { O/G1, I/G2, KY, KX, G1, G2 }
  // weightsOut = { I/G3, O/G4, KY, KX, G3, G4 }

  const auto dType = graph.getTensorElementType(weightsIn);
  const auto KY = weightsOut.dim(2);
  const auto KX = weightsOut.dim(3);
  const auto I = weightsOut.dim(0) * weightsOut.dim(4);
  const auto O = weightsOut.dim(1) * weightsOut.dim(5);
  const auto G1 = weightsIn.dim(4);
  const auto G2 = weightsIn.dim(5);
  const auto G3 = weightsOut.dim(4);
  const auto G4 = weightsOut.dim(5);

  // Express the rearrangement as a composition of two rearrangements such
  // that the first rearrangement avoids exchange and maximises the size of the
  // block that is rearranged in the second step. This reduces exchange code
  // since the second step involves fewer, larger messages.
  // G5 is the size of the innermost dimension after the partial transposition.
  // To avoid exchange it must divide G1. If G4 divides G1 then set G5 to G4 -
  // this results in the block size of G1 * gcd(G2, G3) elements in the
  // second step. Otherwise set G5 to G1 for a block size of gcd(G1, G4)
  // elements.
  const auto G5 = (G1 % G4 == 0) ? G4 : G1;
  Tensor partiallyTransposed;
  if (G5 == 1) {
    partiallyTransposed = weightsIn.reshape({O/G1, I/G2, KY, KX, G1, G2, 1});
  } else {
    auto cs = graph.addComputeSet(debugPrefix + "/WeightTranspose");
    partiallyTransposed =
        weightsPartialTranspose(
          graph,
          weightsIn.reshape({O/G1, I/G2, KY, KX, G1/G5, G5, G2}),
          cs
        );
    prog.add(Execute(cs));
  }

  auto wFlippedY = graph.addTensor(dType, {O/G1, I/G2, 0, KX, G1/G5, G2, G5});
  for (int wy = KY - 1; wy >= 0; --wy) {
     wFlippedY = concat(wFlippedY,
                        partiallyTransposed.slice(wy, wy + 1, 2), 2);
  }

  auto wFlippedYX= graph.addTensor(dType, {O/G1, I/G2, KY, 0, G1/G5, G2, G5});
  for (int wx = KX - 1; wx >= 0; --wx) {
     wFlippedYX = concat(wFlippedYX,
                         wFlippedY.slice(wx, wx + 1, 3), 3);
  }
  prog.add(Copy(wFlippedYX.dimShuffle({2, 3, 0, 4, 6, 1, 5})
                           .reshape({KY, KX, O/G4, G4, I/G3, G3})
                           .dimShuffle({4, 2, 0, 1, 5, 3}),
                weightsOut));
}

// Let A be a n x m matrix and B be a m x p matrix. Compute C = A x B.
// Let u be the number of convolution units and w be the number of weights
// per convolutional unit. n must be a multiple of u and m must be a multiple
// of w. Elements of A are loaded in to the convolution units.
// The dimensions of A should be split and arranged as follows:
// [n/u][m/w][u][w].
// The dimensions of B should be split and arranged as follows:
// [m/w][p][w].
// The dimensions of return value C are split and arranged as follows:
// [n/u][p][u].
std::pair <Program, Tensor>
matrixMultiplyByConvInstruction(Graph &graph, const Plan &plan,
                                Tensor a, Tensor b,
                                const std::string &debugPrefix) {
  assert(a.rank() == 4);
  assert(b.rank() == 3);
  assert(a.dim(1) == b.dim(0));
  assert(a.dim(3) == b.dim(2));
  const auto w = a.dim(3);
  const auto u = a.dim(2);
  const auto p = b.dim(1);

  // The matrix multiplication is equivalent to a 1d convolutional layer with no
  // padding or striding, with filter size 1 where the number of output
  // channels is equal to the number of rows of A (n) and the number of
  // input channels is equal to the number of row of B (m).
  if (!plan.useConvolutionInstructions ||
      plan.inChansPerGroup != w ||
      plan.partialChansPerGroup != u) {
    std::abort();
  }
  const auto batchSize = 1;
  const auto kernelSize = 1;
  const auto strideY = 1;
  const auto strideX = 1;
  const auto paddingY = 0;
  const auto paddingX = 0;
  const auto inDimY = 1;
  const auto inDimX = p;
  const auto outDimY = inDimY;
  const auto outDimX = inDimX;
  // Insert size one dimensions for the filter height and width.
  const auto weights = a.reshape({a.dim(0),
                                  a.dim(1),
                                  kernelSize,
                                  kernelSize,
                                  a.dim(2),
                                  a.dim(3)});
  // Insert size one dimension for the batch size and field height.
  const auto in = b.reshape({batchSize, b.dim(0), inDimY, b.dim(1), b.dim(2)});
  Program prog;
  Tensor out;
  std::tie(prog, out) =
      convolutionByAmp(graph, plan, {strideY, strideX}, {paddingY, paddingX},
                       in, weights, outDimY, outDimX, false, debugPrefix);
  auto c = out.reshape({out.dim(1), out.dim(3), out.dim(4)});
  return {prog, c};
}

static void
createWeightGradAopVertex(Graph &graph, unsigned tile,
                          unsigned outXBegin, unsigned outXEnd,
                          unsigned outYBegin, unsigned outYEnd,
                          const WeightGradAopTask *taskBegin,
                          const WeightGradAopTask *taskEnd,
                          unsigned kernelSizeY, unsigned kernelSizeX,
                          const std::vector<unsigned> &stride,
                          const std::vector<unsigned> &padding,
                          ComputeSet cs,
                          const Tensor &acts, const Tensor &deltas,
                          const Tensor &weightDeltas) {
  const auto dType = graph.getTensorElementType(acts);
  const auto partialsType = graph.getTensorElementType(weightDeltas);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto outChansPerGroup = static_cast<unsigned>(deltas.dim(3));
  const auto inChansPerGroup = static_cast<unsigned>(acts.dim(3));
  const auto inDimY = acts.dim(1);
  const auto inDimX = acts.dim(2);
  assert(weightDeltas.dim(4) == outChansPerGroup);
  assert(weightDeltas.dim(5) == inChansPerGroup);

  const auto numTasks = taskEnd - taskBegin;

  if (!numTasks)
    return;

  std::vector<Tensor> weightDeltasEdges(numTasks);
  std::vector<unsigned> weightReuseCount(numTasks);
  std::vector<Tensor> actsEdges;
  std::vector<Tensor> deltasEdges;
  unsigned numDeltasEdges = 0;
  for (auto it = taskBegin; it != taskEnd; ++it) {
    const auto &task = *it;
    const auto kernelX = task.kernelX;
    const auto kernelY = task.kernelY;
    const auto izg = task.inZGroup;
    const auto ozg = task.outZGroup;
    const auto weightIndex = it - taskBegin;

    weightDeltasEdges[weightIndex] = weightDeltas[ozg][izg]
                                                 [kernelY][kernelX].flatten();

    unsigned deltaXBegin, deltaXEnd;
    std::tie(deltaXBegin, deltaXEnd) =
        getOutputRange({outXBegin, outXEnd}, stride[1], kernelSizeX, padding[1],
                       inDimX, kernelX, false);
    const auto actXBegin = deltaXBegin * stride[1] + kernelX - padding[1];
    const auto actXEnd = (deltaXEnd - 1) * stride[1] + kernelX - padding[1] + 1;
    unsigned deltaYBegin, deltaYEnd;
    std::tie(deltaYBegin, deltaYEnd) =
        getOutputRange({outYBegin, outYEnd}, stride[0], kernelSizeY, padding[0],
                       inDimY, kernelY, false);

    weightReuseCount[weightIndex] = deltaYEnd - deltaYBegin;

    for (unsigned deltaY = deltaYBegin; deltaY != deltaYEnd;
         ++deltaY, ++numDeltasEdges) {
      const auto actY = deltaY * stride[0] + kernelY - padding[0];
      actsEdges.push_back(acts[izg][actY].slice(actXBegin, actXEnd).flatten());
      deltasEdges.push_back(deltas[ozg][deltaY]
                            .slice(deltaXBegin, deltaXEnd).flatten());
    }
  }

  const auto numEdges = 2 * numDeltasEdges + numTasks;

  auto v = graph.addVertex(
                cs,
                templateVertex("popconv::ConvWeightGradAop",
                               dType, partialsType,
                               useDeltaEdgesForWeightGradAop(numEdges) ?
                                                             "true" : "false"));
  graph.setTileMapping(v, tile);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
  graph.setInitialValue(v["weightReuseCount"], weightReuseCount);
  graph.connect(v["acts"], actsEdges);
  graph.connect(v["deltas"], deltasEdges);
  graph.connect(v["weightDeltas"], weightDeltasEdges);
}

static void
calcPartialWeightGradsAop(Graph &graph,
                          unsigned tile,
                          unsigned outXBegin, unsigned outXEnd,
                          unsigned outYBegin, unsigned outYEnd,
                          unsigned outZGroupBegin, unsigned outZGroupEnd,
                          unsigned kernelYBegin, unsigned kernelYEnd,
                          unsigned inZGroupBegin, unsigned inZGroupEnd,
                          unsigned kernelSizeY, unsigned kernelSizeX,
                          const std::vector<unsigned> &stride,
                          const std::vector<unsigned> &padding,
                          ComputeSet cs,
                          Tensor acts, Tensor deltas, Tensor weightDeltas) {
  std::vector<WeightGradAopTask> tasks;
  const auto inDimY = acts.dim(1);
  const auto inDimX = acts.dim(2);
  for (unsigned kernelY = kernelYBegin; kernelY != kernelYEnd; ++kernelY) {
    for (unsigned kernelX = 0; kernelX != kernelSizeX; ++kernelX) {
      auto xRange =
          getOutputRange({outXBegin, outXEnd}, stride[1], kernelSizeX,
                          padding[1], inDimX, kernelX, false);
      if (xRange.first == xRange.second)
        continue;
      auto yRange =
          getOutputRange({outYBegin, outYEnd}, stride[0], kernelSizeY,
                          padding[0], inDimY, kernelY, false);
      if (yRange.first == yRange.second)
        continue;
      for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
        for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
          tasks.emplace_back(kernelY, kernelX, ozg, izg);
        }
      }
    }
  }
  if (tasks.empty())
    return;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  auto numWorkers = deviceInfo.numWorkerContexts;
  const auto numTasks = tasks.size();
  const auto maxTasksPerVertex = (numTasks + numWorkers - 1) / numWorkers;
  const auto verticesToCreate =
      (numTasks + maxTasksPerVertex - 1) / maxTasksPerVertex;
  for (unsigned i = 0; i != verticesToCreate; ++i) {
    const auto taskBegin = (numTasks * i) / verticesToCreate;
    const auto taskEnd = (numTasks * (i + 1)) / verticesToCreate;
    assert(taskEnd - taskBegin > 0);
    createWeightGradAopVertex(graph, tile, outXBegin, outXEnd,
                              outYBegin, outYEnd, &tasks[0] + taskBegin,
                              &tasks[0] + taskEnd, kernelSizeY, kernelSizeX,
                              stride, padding, cs, acts, deltas, weightDeltas);
  }
}

static void
addWeightDeltaPartialRegions(
    std::vector<Interval<std::size_t>> &regions,
    const std::vector<std::size_t> &partialDims,
    unsigned b, unsigned tileY, unsigned tileX,
    unsigned kernelYBegin, unsigned kernelYEnd,
    unsigned outZGroupBegin, unsigned outZGroupEnd,
    unsigned inZGroupBegin, unsigned inZGroupEnd) {
  addFlattenedRegions(partialDims,
                      {b,
                       tileY,
                       tileX,
                       outZGroupBegin,
                       inZGroupBegin,
                       kernelYBegin,
                       0,
                       0,
                       0},
                      {b + 1,
                       tileY + 1,
                       tileX + 1,
                       outZGroupEnd,
                       inZGroupEnd,
                       kernelYEnd,
                       partialDims[6],
                       partialDims[7],
                       partialDims[8]},
                      regions);
}

std::vector<std::vector<Interval<std::size_t>>>
convertLinearMappingToRegionMapping(const std::vector<unsigned> &mapping) {
  assert(!mapping.empty());
  const auto numTiles = mapping.size() - 1;
  std::vector<std::vector<Interval<std::size_t>>>
      regionMapping(numTiles);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    if (mapping[tile] == mapping[tile + 1])
      continue;
    regionMapping[tile].emplace_back(mapping[tile], mapping[tile + 1]);
  }
  return regionMapping;
}

/// Group tiles based on the regions of the reduced output their partial sums
/// contribute to.
static void
groupWeightUpdateAopTilesByOutput(
    const poplar::Graph &graph,
    unsigned batchSize,
    const std::vector<std::size_t> &reducedDims,
    const Plan &plan,
    std::vector<std::vector<unsigned>> &tileGroups,
    std::vector<
      std::vector<Interval<std::size_t>>
    > &tileGroupRegions) {
  const auto isMultiIPU = graph.getDevice().getDeviceInfo().numIPUs > 1;
  const auto partialNumChanGroups = reducedDims[0];
  const auto inNumChanGroups = reducedDims[1];
  const auto kernelHeight = reducedDims[2];
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  tileGroups.clear();
  tileGroups.reserve(tilesPerZ * tilesPerY * tilesPerX);
  tileGroupRegions.clear();
  tileGroupRegions.reserve(tilesPerZ * tilesPerY * tilesPerX);
  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * inNumChanGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * inNumChanGroups) / tilesPerInZGroup;
    for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
      const auto outZGroupBegin = (ozg * partialNumChanGroups) / tilesPerZ;
      const auto outZGroupEnd =
          ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
      if (outZGroupBegin == outZGroupEnd)
        continue;
      for (unsigned ky = 0; ky != tilesPerKernelY; ++ky) {
        const auto kernelYBegin = (ky * kernelHeight) / tilesPerKernelY;
        const auto kernelYEnd = ((ky + 1) * kernelHeight) / tilesPerKernelY;
        tileGroups.emplace_back();
        tileGroupRegions.emplace_back();
        addFlattenedRegions(reducedDims,
                            {outZGroupBegin, inZGroupBegin, kernelYBegin,
                             0, 0, 0},
                            {outZGroupEnd, inZGroupEnd, kernelYEnd,
                             reducedDims[3], reducedDims[4], reducedDims[5]},
                            tileGroupRegions.back());
        for (unsigned b = 0; b < batchSize; ++b) {
          for (unsigned oy = 0; oy != tilesPerY; ++oy) {
            for (unsigned ox = 0; ox != tilesPerX; ++ox) {
              const auto tile = linearizeTileIndices(b, batchSize,
                                                     numTiles,
                                                     ky, izg,
                                                     ox, oy, ozg,
                                                     plan,
                                                     isMultiIPU);
              tileGroups.back().push_back(tile);
            }
          }
        }
        mergeAdjacentRegions(tileGroupRegions.back());
        std::sort(tileGroups.back().begin(), tileGroups.back().end());
      }
    }
  }
}

static Tensor
weightUpdateAopReduceByPartialMapping(Graph &graph,
                                      const Tensor &partials,
                                      const std::string &resultType,
                                      const Plan &plan,
                                      std::vector<ComputeSet> &computeSets,
                                      const std::string &debugPrefix) {
  std::vector<
    std::vector<Interval<std::size_t>>
  > tileGroupRegions;
  std::vector<std::vector<unsigned>> tileGroups;
  const auto batchSize = partials.dim(0);
  groupWeightUpdateAopTilesByOutput(graph, batchSize, partials[0][0][0].shape(),
                                    plan, tileGroups, tileGroupRegions);
  const auto reductionDepth = partials.dim(0) * partials.dim(1) *
                              partials.dim(2);
  auto flattenedPartialsDims = partials[0][0].shape();
  flattenedPartialsDims[0] = reductionDepth;
  return multiStageGroupedReduce(graph, tileGroups, tileGroupRegions,
                                 partials.reshape(flattenedPartialsDims),
                                 resultType, computeSets, debugPrefix);
}

static Tensor
calculateWeightDeltasAop(Graph &graph, const Plan &plan,
                         const Plan &fwdPlan, Tensor zDeltas,
                         unsigned kernelSizeY,
                         unsigned kernelSizeX,
                         Tensor activations,
                         const std::vector<unsigned> &stride,
                         const std::vector<unsigned> &padding,
                         Sequence &prog,
                         const std::string &debugPrefix) {
  if (activations.dim(4) != fwdPlan.inChansPerGroup) {
    activations = regroup(activations, fwdPlan.inChansPerGroup);
  }
  if (plan.flattenXY) {
    zDeltas = zDeltas.reshape(
        {zDeltas.dim(0), zDeltas.dim(1), 1,
         zDeltas.dim(2) * zDeltas.dim(3), zDeltas.dim(4)}
    );
    activations = activations.reshape(
        {activations.dim(0), activations.dim(1), 1,
         activations.dim(2) * activations.dim(3), activations.dim(4)}
    );
  }
  const auto &partialsType = plan.getPartialType();
  const auto dType = graph.getTensorElementType(zDeltas);
  const auto batchSize = activations.dim(0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numOutChans = zDeltas.dim(1) * zDeltas.dim(4);
  const auto inNumChanGroups = activations.dim(1);
  const auto inChansPerGroup = activations.dim(4);
  assert(plan.inChansPerGroup == inChansPerGroup);
  const auto numInChans = inNumChanGroups * inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(numOutChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = numOutChans / partialChansPerGroup;

  auto outDimY = zDeltas.dim(2), outDimX = zDeltas.dim(3);
  const auto isMultiIPU = deviceInfo.numIPUs > 1;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;
  const auto numTiles = deviceInfo.getNumTiles();

  Tensor partials = graph.addTensor(partialsType, {batchSize,
                                                   tilesPerY, tilesPerX,
                                                   partialNumChanGroups,
                                                   inNumChanGroups,
                                                   kernelSizeY, kernelSizeX,
                                                   partialChansPerGroup,
                                                   inChansPerGroup},
                                    "partialWeightGrads");
  Tensor regroupedDeltas;
  if (zDeltas.dim(1) != partialNumChanGroups) {
    regroupedDeltas = graph.addTensor(dType, {batchSize, partialNumChanGroups,
                                              outDimY, outDimX,
                                              partialChansPerGroup},
                                              "zDeltas'");
    for (unsigned b = 0; b < batchSize; ++b) {
      auto regroupedDeltaMapping =
          computeActivationsMapping(graph, regroupedDeltas[b], b, batchSize);
      popstd::applyTensorMapping(graph, regroupedDeltas[b],
                                 regroupedDeltaMapping);
    }
    prog.add(Copy(regroup(zDeltas, partialChansPerGroup), regroupedDeltas));
  } else {
    regroupedDeltas = zDeltas;
  }
  std::vector<std::vector<Interval<std::size_t>>> partialsMapping(numTiles);
  ComputeSet weightGradCS = graph.addComputeSet(debugPrefix + "/WeightGrad");
  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * inNumChanGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * inNumChanGroups) / tilesPerInZGroup;
      for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
        const auto outZGroupBegin = (ozg * partialNumChanGroups) / tilesPerZ;
        const auto outZGroupEnd =
            ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
        if (outZGroupBegin == outZGroupEnd)
          continue;
        for (unsigned oy = 0; oy != tilesPerY; ++oy) {
          const auto outYBegin = (oy * outDimY) / tilesPerY;
          const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
          for (unsigned ox = 0; ox != tilesPerX; ++ox) {
            const auto outXBegin = (ox * outDimX) / tilesPerX;
            const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
            for (unsigned ky = 0; ky != tilesPerKernelY; ++ky) {
              const auto kernelYBegin = (ky * kernelSizeY) / tilesPerKernelY;
              const auto kernelYEnd =
                  ((ky + 1) * kernelSizeY) / tilesPerKernelY;
              const auto tile = linearizeTileIndices(b, batchSize,
                                                     numTiles,
                                                     ky, izg,
                                                     ox, oy, ozg,
                                                     plan,
                                                     isMultiIPU);
              addWeightDeltaPartialRegions(partialsMapping[tile],
                                           partials.shape(), b, oy, ox,
                                           kernelYBegin, kernelYEnd,
                                           outZGroupBegin, outZGroupEnd,
                                           inZGroupBegin, inZGroupEnd);
              calcPartialWeightGradsAop(graph, tile,
                                        outXBegin, outXEnd, outYBegin, outYEnd,
                                        outZGroupBegin, outZGroupEnd,
                                        kernelYBegin, kernelYEnd,
                                        inZGroupBegin, inZGroupEnd,
                                        kernelSizeY, kernelSizeX,
                                        stride, padding, weightGradCS,
                                        activations[b],
                                        regroupedDeltas[b],
                                        partials[b][oy][ox]);
            }
          }
        }
      }
    }
  }
  mergeAdjacentRegions(partialsMapping);
  applyTensorMapping(graph, partials, partialsMapping);
  ComputeSet zeroCS = graph.addComputeSet(debugPrefix + "/Zero");
  zero(graph, partials, partialsMapping, zeroCS);
  prog.add(Execute(zeroCS));
  prog.add(Execute(weightGradCS));

  const auto fwdPartialChansPerGroup = fwdPlan.partialChansPerGroup;
  const auto fwdInChansPerGroup = fwdPlan.inChansPerGroup;
  const auto weightMapping =
      calculateWeightMapping({numOutChans / fwdPartialChansPerGroup,
                              numInChans / fwdInChansPerGroup,
                              kernelSizeY,
                              kernelSizeX,
                              fwdPartialChansPerGroup,
                              fwdInChansPerGroup}, dType, graph, fwdPlan,
                              batchSize);
  Tensor weightDeltas;
  auto numPartials = batchSize * tilesPerY * tilesPerX;
  if (numPartials == 1 && partialsType == dType) {
    weightDeltas = partials[0][0][0];
  } else if (numPartials == 1) {
    auto reduceCS = graph.addComputeSet(debugPrefix + "/Reduce");
    weightDeltas = graph.addTensor(dType, partials[0][0][0].shape(),
                                   debugPrefix + "/WeightDeltas");
    std::vector<std::vector<Interval<std::size_t>>>
        weightDeltaMapping;
    if (partialChansPerGroup == fwdPartialChansPerGroup) {
      weightDeltaMapping = weightMapping;
    } else {
      weightDeltaMapping =
          convertLinearMappingToRegionMapping(
            computeTensorMapping(graph, weightDeltas)
          );
    }
    applyTensorMapping(graph, weightDeltas, weightDeltaMapping);
    auto flatPartialsDims = partials[0][0][0].shape();
    flatPartialsDims.insert(flatPartialsDims.begin(), numPartials);
    auto flatPartials = partials.reshape(flatPartialsDims);
    popreduce::reduce(graph, flatPartials, weightDeltas, weightMapping,
                      reduceCS);
    prog.add(Execute(reduceCS));
  } else {
    std::vector<ComputeSet> reduceComputeSets;
    weightDeltas =
      weightUpdateAopReduceByPartialMapping(graph, partials, dType, plan,
                                            reduceComputeSets, debugPrefix);
    for (const auto &cs : reduceComputeSets) {
      prog.add(Execute(cs));
    }
  }
  weightDeltas = regroup(weightDeltas, 0, 4, fwdPartialChansPerGroup);
  return weightDeltas;
}

static Tensor
roundUpDimension(Graph &graph, const Tensor &t, unsigned dim,
                 unsigned divisor) {
  const auto size = t.dim(dim);
  const auto roundedSize = ((size + divisor - 1) / divisor) * divisor;
  return pad(graph, t, roundedSize, 0, dim);
}

// Weight deltas can be computed by convolving the activations and the deltas.
// If the kernel is larger than 1x1 a direct computation of weight deltas
// requires a sliding deltas across activations in the main axis of
// accumulation. This sliding stops us using the AMP instruction because,
// as we slide, the vectors of elements we want to load in the inner loop will
// no longer be contiguous / aligned. We fix this by transforming the
// convolution into an equivalent convolution which doesn't require sliding in
// the main axis of accumulation. Given the activation and delta tensors for
// the original convolution (zero padded and reshaped to have a single channel
// dimension) transform them into tensors for the transformed convolution.
static void
convolutionWeightUpdateAmpPreProcess(
    Graph &graph,
    const Plan &plan,
    Tensor &activations,
    std::vector<unsigned> &activationsUpsampleFactor,
    std::vector<unsigned> &activationsPadding,
    Tensor &deltas,
    std::vector<unsigned> &deltasUpsampleFactor,
    std::vector<unsigned> &deltasPadding,
    unsigned kernelSizeY,
    unsigned kernelSizeX) {
  const auto dType = graph.getTensorElementType(activations);
  assert(activationsUpsampleFactor.size() == 2);
  assert(activationsPadding.size() == 2);
  assert(deltasUpsampleFactor.size() == 2);
  assert(deltasPadding.size() == 2);
  assert(activationsUpsampleFactor == std::vector<unsigned>({1, 1}));
  assert(deltasPadding == std::vector<unsigned>({0, 0}));
  // Eliminate the x axis of the kernel by taking the activations that are
  // multiplied by each column of the weights turning them into different input
  // channels.
  auto paddedActivations = pad(graph, activations,
                               {activations.dim(0),
                                activations.dim(1),
                                activations.dim(2) + 2 * activationsPadding[1],
                                activations.dim(3)},
                               {0,
                                0,
                                activationsPadding[1],
                                0});
  activationsPadding[1] = 0;
  auto expandedActivations =
      graph.addTensor(dType, {paddedActivations.dim(0),
                              paddedActivations.dim(1),
                              deltas.dim(2),
                              0});
  for (unsigned wx = 0; wx != kernelSizeX; ++wx) {
    auto usedActivations =
        paddedActivations.slice(wx,
                                absdiff(paddedActivations.dim(2), kernelSizeX) +
                                1 + wx,
                                2);
    auto stridedActivations =
        usedActivations.subSample(deltasUpsampleFactor[1], 2);
    expandedActivations = concat(expandedActivations, stridedActivations, 3);
  }
  deltasUpsampleFactor[1] = 1;
  if (plan.flattenXY) {
    // Eliminate the y axis of the kernel by taking the activations that are
    // multiplied by each row of the weights turning them into different input
    // channels.
    auto yPaddedActivations = pad(graph, expandedActivations,
                                  {expandedActivations.dim(0),
                                   expandedActivations.dim(1) +
                                   2 * activationsPadding[0],
                                   expandedActivations.dim(2),
                                   expandedActivations.dim(3)},
                                  {0,
                                   activationsPadding[0],
                                   0,
                                   0});
    activationsPadding[0] = 0;
    auto yExpandedActivations =
        graph.addTensor(dType, {yPaddedActivations.dim(0),
                                deltas.dim(1),
                                deltas.dim(2),
                                0});
    for (unsigned wy = 0; wy != kernelSizeY; ++wy) {
      auto usedActivations =
          yPaddedActivations.slice(wy,
                                   yPaddedActivations.dim(1) -
                                   kernelSizeY + 1 + wy, 1);
      auto stridedActivations =
          usedActivations.subSample(deltasUpsampleFactor[0], 1);
      yExpandedActivations = concat(yExpandedActivations, stridedActivations,
                                    3);
    }
    expandedActivations = yExpandedActivations;
    deltasUpsampleFactor[0] = 1;
    // Flatten the x and y axes.
    expandedActivations =
        expandedActivations.reshape({expandedActivations.dim(0),
                                     1,
                                     expandedActivations.dim(1) *
                                     expandedActivations.dim(2),
                                     expandedActivations.dim(3)});
    deltas =
        deltas.reshape({deltas.dim(0),
                        1,
                        deltas.dim(1) * deltas.dim(2),
                        deltas.dim(3)});
  }
  // Rearrange the tensors so elements of the batch are treated as part of the
  // x-axis of the field.
  auto flattenedActivations =
      expandedActivations
          .dimShuffle({1, 2, 0, 3})
          .reshape({1,
                    expandedActivations.dim(1),
                    expandedActivations.dim(2) * expandedActivations.dim(0),
                    expandedActivations.dim(3)});
  auto flattenedDeltas =
      deltas.dimShuffle({1, 2, 0, 3})
             .reshape({1,
                       deltas.dim(1),
                       deltas.dim(2) * deltas.dim(0),
                       deltas.dim(3)});
  if (plan.ampWUMethod == Plan::ACTIVATIONS_AS_COEFFICENTS) {
    assert(activationsPadding[1] == 0);
    if (activationsPadding[0] > 0) {
      // Currently we don't support convolutions with a zero padded filter so
      // we must explicitly add padding.
      // TODO extend convolutionByAmp() to support zero padding the filter.
      flattenedActivations = pad(graph, flattenedActivations,
                                 {flattenedActivations.dim(0),
                                  flattenedActivations.dim(1) +
                                  2 * activationsPadding[0],
                                  flattenedActivations.dim(2),
                                  flattenedActivations.dim(3)},
                                 {0,
                                  activationsPadding[0],
                                  0,
                                  0});
      activationsPadding[0] = 0;
    }
    std::swap(flattenedActivations, flattenedDeltas);
    std::swap(activationsUpsampleFactor, deltasUpsampleFactor);
    std::swap(activationsPadding, deltasPadding);
  }
  activations = flattenedActivations;
  deltas = flattenedDeltas;
}

// convolutionWeightUpdateAmpPreProcess() translates a convolution into an
// equivalent convolution that we can use the AMP instruction to compute.
// Given the weight deltas for this transformed convolution (reshaped so neither
// channel dimension is split) transform them into weight deltas for the
// original convolution.
static void
convolutionWeightUpdateAmpPostProcess(const Plan &plan,
                                      Tensor &weightDeltas,
                                      unsigned kernelSizeY,
                                      unsigned kernelSizeX) {
  assert(weightDeltas.dim(1) == 1);
  if (plan.ampWUMethod == Plan::ACTIVATIONS_AS_COEFFICENTS) {
    weightDeltas = weightDeltas.dimShuffle({0, 1, 3, 2});
  }
  if (plan.flattenXY) {
    assert(weightDeltas.dim(0) == 1);
    weightDeltas =
        weightDeltas.reshape({weightDeltas.dim(2),
                              kernelSizeY,
                              kernelSizeX,
                              weightDeltas.dim(3) /
                              (kernelSizeX * kernelSizeY)})
                              .dimShuffle({1, 2, 0, 3});
  } else {
    weightDeltas =
        weightDeltas.reshape({weightDeltas.dim(0),
                              weightDeltas.dim(2),
                              kernelSizeX,
                              weightDeltas.dim(3) /
                                        kernelSizeX})
                              .dimShuffle({0, 2, 1, 3});
  }
}

static Tensor calculateWeightDeltasAmp(Graph &graph, const Plan &plan,
                                       const Plan &fwdPlan, Tensor zDeltas,
                                       unsigned kernelSizeY,
                                       unsigned kernelSizeX,
                                       Tensor activations,
                                       const std::vector<unsigned> &stride,
                                       const std::vector<unsigned> &padding,
                                       Sequence &prog,
                                       const std::string &debugPrefix) {
  // Shuffle dimensions of the activations and the deltas so there is a single
  // channel dimension.
  auto activationsView =
      activations.dimShuffle({0, 2, 3, 1, 4})
                 .reshape({activations.dim(0),
                           activations.dim(2),
                           activations.dim(3),
                           activations.dim(1) * activations.dim(4)});
  auto deltasView =
      zDeltas.dimShuffle({0, 2, 3, 1, 4})
             .reshape({zDeltas.dim(0),
                       zDeltas.dim(2),
                       zDeltas.dim(3),
                       zDeltas.dim(1) * zDeltas.dim(4)});

  // Transform the weight update convolution into an equivalent convolution that
  // can be implemented using the AMP instruction.
  std::vector<unsigned> activationsUpsampleFactor = {1, 1};
  std::vector<unsigned> activationsPadding = padding;
  std::vector<unsigned> deltasUpsampleFactor = stride;
  std::vector<unsigned> deltasPadding = {0, 0};
  convolutionWeightUpdateAmpPreProcess(graph, plan, activationsView,
                                       activationsUpsampleFactor,
                                       activationsPadding, deltasView,
                                       deltasUpsampleFactor, deltasPadding,
                                       kernelSizeY, kernelSizeX);

  const auto dType = graph.getTensorElementType(activations);
  // Reshape so there is no batch dimension.
  assert(activationsView.dim(0) == 1);
  assert(deltasView.dim(0) == 1);
  activationsView = activationsView[0];
  deltasView = deltasView[0];

  assert(activationsView.dim(1) == deltasView.dim(1));

  // Pad the x-axis to a multiple of the input channels per group.
  assert(activationsPadding[1] == 0);
  assert(activationsUpsampleFactor[1] == 1);
  assert(deltasPadding[1] == 0);
  assert(deltasUpsampleFactor[1] == 1);
  const auto inChansPerGroup = plan.inChansPerGroup;
  activationsView = roundUpDimension(graph, activationsView, 1,
                                     inChansPerGroup);
  deltasView = roundUpDimension(graph, deltasView, 1, inChansPerGroup);
  // Pad the output channels to a multiple of the partial channels per group.
  const auto numOutChans = deltasView.dim(2);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  deltasView = roundUpDimension(graph, deltasView, 2, partialChansPerGroup);

  // Transpose the activations.
  auto activationsTransposed =
      graph.addTensor(dType,
                      {1,
                       activationsView.dim(1) / inChansPerGroup,
                       activationsView.dim(0),
                       activationsView.dim(2),
                       inChansPerGroup},
                      "activationsTransposed");
  mapActivations(graph, plan, activationsTransposed);
  prog.add(Copy(activationsView.reshape({activationsView.dim(0),
                                         activationsView.dim(1) /
                                         inChansPerGroup,
                                         inChansPerGroup,
                                         activationsView.dim(2)})
                           .dimShuffle({1, 0, 3, 2})
                           .reshape(activationsTransposed.shape()),
                activationsTransposed));

  // Transpose the deltas.
  auto deltasTransposed =
      graph.addTensor(dType,
                      {deltasView.dim(2) / partialChansPerGroup,
                       deltasView.dim(1) / inChansPerGroup,
                       deltasView.dim(0),
                       1,
                       partialChansPerGroup,
                       inChansPerGroup},
                      "deltasTransposed");
  prog.add(Copy(deltasView.reshape({deltasView.dim(0),
                                    deltasView.dim(1) / inChansPerGroup,
                                    inChansPerGroup,
                                    deltasView.dim(2) / partialChansPerGroup,
                                    partialChansPerGroup})
                           .dimShuffle({3, 1, 0, 4, 2})
                           .reshape(deltasTransposed.shape()),
                deltasTransposed));

  // Perform the convolution.
  const auto outDimY =
      absdiff(activationsTransposed.dim(2) * activationsUpsampleFactor[0] +
              2 * activationsPadding[0],
              deltasTransposed.dim(2)) +
      1;
  Tensor weightDeltasTransposed;
  Program convolveProg;
  auto isNotOne = [](unsigned x) { return x != 1; };
  std::tie(convolveProg, weightDeltasTransposed) =
      convolutionByAmp(graph, plan, activationsUpsampleFactor,
                       activationsPadding, activationsTransposed,
                       deltasTransposed, outDimY, activationsTransposed.dim(3),
                       std::any_of(activationsUpsampleFactor.begin(),
                                   activationsUpsampleFactor.end(),
                                   isNotOne) /* isFractional */,
                       debugPrefix);
  prog.add(convolveProg);

  // Shuffle dimensions so the output channel dimension is not split.
  auto weightDeltas =
      weightDeltasTransposed.dimShuffle({0, 2, 3, 1, 4})
                            .reshape({weightDeltasTransposed.dim(0),
                                      weightDeltasTransposed.dim(2),
                                      weightDeltasTransposed.dim(3),
                                      weightDeltasTransposed.dim(1) *
                                      weightDeltasTransposed.dim(4)
                                     });
  // Ignore output channels added for padding.
  weightDeltas = weightDeltas.slice(0, numOutChans, 3);
  // Reshape so there is no batch dimension.
  assert(weightDeltas.dim(0) == 1);
  weightDeltas = weightDeltas[0];

  // Make the input channel dimension the innermost dimension and add an
  // x-axis.
  weightDeltas =
      weightDeltas.dimShuffle({0, 2, 1})
                  .reshape({weightDeltas.dim(0),
                            1,
                            weightDeltas.dim(2),
                            weightDeltas.dim(1)});

  // Transform the weight deltas back into weight deltas for the original
  // weight update convolution.
  convolutionWeightUpdateAmpPostProcess(plan, weightDeltas, kernelSizeY,
                                        kernelSizeX);
  // Split the input / output channel axes.
  const auto fwdPartialChansPerGroup = fwdPlan.partialChansPerGroup;
  const auto fwdInChansPerGroup = fwdPlan.inChansPerGroup;
  weightDeltas =
      weightDeltas.reshape({weightDeltas.dim(0),
                            weightDeltas.dim(1),
                            weightDeltas.dim(2) /
                            fwdPartialChansPerGroup,
                            fwdPartialChansPerGroup,
                            weightDeltas.dim(3) / fwdInChansPerGroup,
                            fwdInChansPerGroup})
                       .dimShuffle({2, 4, 0, 1, 3, 5});
  return weightDeltas;
}

static Tensor
calculateWeightDeltas(Graph &graph, const Plan &plan,
                      const Plan &fwdPlan, Tensor zDeltas,
                      unsigned kernelSizeY, unsigned kernelSizeX,
                      Tensor activations, const std::vector<unsigned> &stride,
                      const std::vector<unsigned> &padding,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  if (plan.useConvolutionInstructions) {
    return calculateWeightDeltasAmp(graph, plan, fwdPlan, zDeltas,
                                    kernelSizeY, kernelSizeX, activations,
                                    stride, padding, prog, debugPrefix);
  }
  return calculateWeightDeltasAop(graph, plan, fwdPlan, zDeltas,
                                  kernelSizeY, kernelSizeX,
                                  activations, stride, padding, prog,
                                  debugPrefix);
}

Tensor
calculateWeightDeltas(Graph &graph, Tensor zDeltas,
                      unsigned kernelSizeY, unsigned kernelSizeX,
                      Tensor activations,
                      const std::vector<unsigned> &stride,
                      const std::vector<unsigned> &padding,
                      bool isFractional,
                      Sequence &prog,
                      const std::string &debugPrefix,
                      const ConvOptions &options) {
  const auto outNumChans = zDeltas.dim(1) * zDeltas.dim(4);
  const auto dType = graph.getTensorElementType(zDeltas);
  const auto plan =
      getWeightUpdatePlan(graph, activations, zDeltas,
                          {kernelSizeY, kernelSizeX, outNumChans},
                          stride, padding, isFractional, options);
  const auto fwdPlan =
      getPlan(graph, dType,
              activations.dim(0), activations.dim(2), activations.dim(3),
              activations.dim(1) * activations.dim(4),
              {kernelSizeY, kernelSizeX, outNumChans},
              stride, padding,
              isFractional, options);
  return calculateWeightDeltas(graph, plan, fwdPlan, zDeltas, kernelSizeY,
                               kernelSizeX, activations, stride, padding, prog,
                               debugPrefix);
}

void
convolutionWeightUpdate(Graph &graph,
                        Tensor zDeltas, Tensor weights,
                        Tensor activations,
                        const std::vector<unsigned> &stride,
                        const std::vector<unsigned> &padding,
                        bool isFractional, float learningRate,
                        Sequence &prog,
                        const std::string &debugPrefix,
                        const ConvOptions &options) {
  const auto dType = graph.getTensorElementType(zDeltas);
  const auto plan =
      getWeightUpdatePlan(graph, activations, zDeltas,
                          {weights.dim(2), weights.dim(3),
                           weights.dim(0) * weights.dim(4)},
                          stride, padding, isFractional, options);
  const auto fwdPlan =
      getPlan(graph, dType,
              activations.dim(0), activations.dim(2), activations.dim(3),
              activations.dim(1) * activations.dim(4),
              {weights.dim(2), weights.dim(3),
               weights.dim(0) * weights.dim(4)},
              stride, padding,
              isFractional, options);
  const unsigned kernelSizeY = weights.dim(2);
  const unsigned kernelSizeX = weights.dim(3);
  const auto layerName = debugPrefix
                         + "/ConvWeightUpdate"
                         + convSuffix({kernelSizeY, kernelSizeX}, stride,
                                      false)
                         + (plan.useConvolutionInstructions ? "_amp" : "_aop");
  auto weightDeltas = calculateWeightDeltas(graph, plan, fwdPlan, zDeltas,
                                            kernelSizeY, kernelSizeX,
                                            activations, stride, padding,
                                            prog, layerName);


  // Add the weight deltas to the weights.
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto batchSize = activations.dim(0);
  auto addCS = graph.addComputeSet(debugPrefix + "/UpdateWeights");
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  assert(weightDeltas.shape() == weights.shape());
  auto weightsFlattened = weights.flatten();
  auto weightDeltasFlattened = weightDeltas.flatten();
  iterateWeightMapping(weights, graph, fwdPlan, batchSize,
                       [&](const Tensor &tileWeights, unsigned tile) {
    const auto elementIndices = tileWeights.getElementIndices();
    const auto tileNumElements = elementIndices.size();
    const auto workersPerTile = deviceInfo.numWorkerContexts;
    const auto maxElemsPerWorker =
        (tileNumElements + workersPerTile - 1) / workersPerTile;
    const auto verticesToCreate =
        (tileNumElements + maxElemsPerWorker - 1) / maxElemsPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto elemBegin =
          (vertex * tileNumElements) / verticesToCreate;
      const auto elemEnd =
          ((vertex + 1) * tileNumElements) / verticesToCreate;
      if (elemBegin == elemEnd)
        continue;
      auto regions = getContiguousRegions(elementIndices.begin() + elemBegin,
                                          elementIndices.begin() + elemEnd);
      for (unsigned i = 0, numRegions = regions.size(); i != numRegions; ++i) {
        const auto v =
            graph.addVertex(addCS,
                            templateVertex("popconv::ConvWeightUpdate", dType));
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setInitialValue(v["eta"], learningRate);
        graph.connect(v["weights"], weightsFlattened.slice(regions[i].first,
                                                           regions[i].second));
        graph.connect(v["weightDeltas"],
            weightDeltasFlattened.slice(regions[i].first, regions[i].second));
        graph.setTileMapping(v, tile);
      }
    }
  });
  prog.add(Execute(addCS));
}

// Return a program to update the biases tensor with the gradients derived
// from the zDeltas tensor
void
convolutionBiasUpdate(Graph &graph, const Tensor &zDeltas, const Tensor &biases,
                      float learningRate,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  const auto layerName = debugPrefix + "/BiasUpdate";

  auto firstReduceCS = graph.addComputeSet(layerName + "/Reduce1");
  // The bias gradient is the sum of all the deltas.
  // The reduction of these deltas is done in three stages:
  //     The first stage reduces on each tile. It places the partial sum
  //     for each tile in the tensor 'tileReducedBiasDeltas[tile]'.
  //     The second stage reduces across tiles to a set of partial sums
  //     spread across the workers. It takes 'tileReducedBiasDeltas' as input
  //     and outputs to the 'biasPartials' 2-d tensor.
  //     The final stage reduces the 'biasPartials' 2-d tensor to get the
  //     final gradient for each bias, multiplies it by the learning rate and
  //     subtracts from the bias in the 'biases' tensor.
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  auto dType = graph.getTensorElementType(zDeltas);
  auto numTiles = deviceInfo.getNumTiles();
  auto numBiases = biases.numElements();
  auto batchSize = zDeltas.dim(0);
  auto outNumChanGroups = zDeltas.dim(1);
  auto outDimY = zDeltas.dim(2), outDimX = zDeltas.dim(3);
  auto outChansPerGroup = zDeltas.dim(4);
  // Before the cross tile reduction. Reduce biases on each tile.
  auto zDeltasFlat = zDeltas.reshape({batchSize, outNumChanGroups,
                                      outDimY * outDimX, outChansPerGroup});

  // Calculate which bias groups have values to reduce on each tile
  std::vector<std::vector<unsigned>> deltaMappings;
  for (unsigned b = 0; b < batchSize; ++b)
    deltaMappings.push_back(computeActivationsMapping(graph, zDeltas[b], b,
                                                      batchSize));
  std::vector<std::vector<unsigned>> tileBiasGroups(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    std::unordered_set<unsigned> biasGroups;
    for (unsigned b = 0; b < batchSize; ++b) {
      auto begin = deltaMappings[b][tile];
      auto end = deltaMappings[b][tile + 1];
      auto M = outDimY * outDimX * outChansPerGroup;
      auto beginGroup = (begin / M);
      auto endGroup = ((end + M - 1) / M);
      for (unsigned biasGroup = beginGroup; biasGroup < endGroup; ++biasGroup) {
        biasGroups.insert(biasGroup);
      }
    }
    // Set tileBiasGroups[tile] to contain the indices of the bias groups to
    // be reduced on that tile.
    auto &vec = tileBiasGroups[tile];
    vec.insert(vec.end(), biasGroups.begin(), biasGroups.end());
  }

  // On each tile create vertices that reduce the on-tile deltas to a single
  // bias delta value for each bias on each tile stored in the
  // tensor tileReducedBiasDeltas[tile].
  std::vector<Tensor> tileReducedBiasDeltas;
  tileReducedBiasDeltas.reserve(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto tileNumBiasGroups = tileBiasGroups[tile].size();
    Tensor r = graph.addTensor(dType, {tileNumBiasGroups,
                                       outChansPerGroup},
                               "tileReducedBiasDeltas");
    tileReducedBiasDeltas.push_back(r);
    graph.setTileMapping(r, tile);
    for (unsigned i = 0; i < tileBiasGroups[tile].size(); ++i) {
      const auto biasGroup = tileBiasGroups[tile][i];
      auto v = graph.addVertex(firstReduceCS,
                               templateVertex("popconv::ConvBiasReduce1",
                                              dType));
      unsigned numRanges = 0;
      for (unsigned b = 0; b < batchSize; ++b) {
        auto begin = deltaMappings[b][tile];
        auto end = deltaMappings[b][tile + 1];
        auto M = outDimY * outDimX * outChansPerGroup;
        auto beginGroup = (begin / M);
        auto endGroup = ((end + M - 1) / M);
        if (beginGroup > biasGroup || endGroup <= biasGroup)
          continue;
        unsigned fieldBegin;
        if (biasGroup == beginGroup) {
          fieldBegin = (begin % M) / outChansPerGroup;
        } else {
          fieldBegin = 0;
        }
        unsigned fieldEnd;
        if (biasGroup == endGroup - 1) {
          fieldEnd = (end % M) / outChansPerGroup;
          if (fieldEnd == 0)
            fieldEnd = outDimX * outDimY;
        } else {
          fieldEnd = outDimX * outDimY;
        }
        auto in = zDeltasFlat[b][biasGroup].slice({fieldBegin, 0},
                                                  {fieldEnd, outChansPerGroup})
                                           .flatten();
        graph.connect(v["in"][numRanges++], in);
      }
      graph.setFieldSize(v["in"], numRanges);
      graph.connect(v["out"], r[i]);
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }

  /** The number of biases is often small. So the reduction of bias
   *  updates is done in two stages to balance compute.
   */
  auto numWorkers = deviceInfo.numWorkerContexts * deviceInfo.getNumTiles();
  unsigned workersPerBias, usedWorkers, maxBiasPerWorker;
  if (numWorkers > numBiases) {
    workersPerBias = numWorkers / numBiases;
    usedWorkers = workersPerBias * numBiases;
    maxBiasPerWorker = 1;
  } else {
    workersPerBias = 1;
    usedWorkers = numWorkers;
    maxBiasPerWorker = (numBiases + numWorkers - 1) / numWorkers;
  }
  auto biasPartials = graph.addTensor(dType, {usedWorkers, maxBiasPerWorker},
                                      "biasPartials");
  auto secondReduceCS = graph.addComputeSet(
                            layerName + "/Reduce2");
  for (unsigned worker = 0; worker  < usedWorkers; ++worker ) {
    auto tile = worker / deviceInfo.numWorkerContexts;
    graph.setTileMapping(biasPartials[worker].slice(0, maxBiasPerWorker), tile);
    unsigned biasBegin = (worker  * numBiases) / usedWorkers;
    unsigned biasEnd = ((worker  + workersPerBias) * numBiases) / usedWorkers;
    if (biasBegin == biasEnd)
      continue;
    unsigned numWorkerBiases = biasEnd - biasBegin;
    auto toReduce = graph.addTensor(dType, {0});
    for (auto bias = biasBegin; bias != biasEnd; ++bias) {
      auto biasGroup = bias / outChansPerGroup;
      auto biasInGroup = bias % outChansPerGroup;
      auto biasDeltas = graph.addTensor(dType, {0});
      for (unsigned srcTile = 0; srcTile < numTiles; ++srcTile) {
        for (unsigned i = 0; i < tileBiasGroups[srcTile].size(); ++i) {
          if (biasGroup != tileBiasGroups[srcTile][i])
            continue;
          auto srcBias = tileReducedBiasDeltas[srcTile][i][biasInGroup];
          biasDeltas = append(biasDeltas, srcBias);
        }
      }
      const auto numDeltas = biasDeltas.numElements();
      auto deltaBegin =
          ((worker  % workersPerBias) * numDeltas) / workersPerBias;
      unsigned deltaEnd =
          (((worker  % workersPerBias) + 1) * numDeltas) / workersPerBias;
      toReduce = concat(toReduce, biasDeltas.slice(deltaBegin, deltaEnd));
    }
    if (toReduce.numElements() == 0) {
      auto v = graph.addVertex(secondReduceCS,
                             templateVertex("popstd::Zero", dType));
      graph.connect(v["out"], biasPartials[worker].slice(0, maxBiasPerWorker));
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v, tile);
      continue;
    }
    auto v = graph.addVertex(secondReduceCS,
                             templateVertex("popconv::ConvBiasReduce2", dType));
    graph.connect(v["in"], toReduce);
    graph.connect(v["out"], biasPartials[worker].slice(0, numWorkerBiases));
    graph.setTileMapping(v, tile);
  }
  auto updateBiasCS = graph.addComputeSet(layerName + "/FinalUpdate");
  iterateBiasMapping(biases, graph, zDeltas.shape(),
    [&](Tensor biasSlice, unsigned tile){
      for (auto bias : biasSlice.getElementIndices()) {
        auto v = graph.addVertex(updateBiasCS,
                                 templateVertex("popconv::ConvBiasUpdate",
                                                dType));
        unsigned numPartials = 0;
        for (unsigned srcWorker = 0; srcWorker < usedWorkers; ++srcWorker) {
          unsigned biasBegin = (srcWorker * numBiases) / usedWorkers;
          unsigned biasEnd =
              ((srcWorker + workersPerBias) * numBiases) / usedWorkers;
          if (biasBegin > bias || biasEnd <= bias)
            continue;
          graph.connect(v["partials"][numPartials++],
                        biasPartials[srcWorker][bias - biasBegin]);
        }
        graph.setFieldSize(v["partials"], numPartials);
        graph.connect(v["bias"], biases[bias]);
        graph.setInitialValue(v["eta"], learningRate);
        graph.setTileMapping(v, tile);
      }
     });
  prog.add(Execute(firstReduceCS));
  prog.add(Execute(secondReduceCS));
  prog.add(Execute(updateBiasCS));
}


} // namespace conv
