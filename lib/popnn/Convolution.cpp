#include "popnn/Convolution.hpp"
#include <limits>
#include <cassert>
#include "ConvUtil.hpp"
#include "Pad.hpp"
#include "popnn/ActivationMapping.hpp"
#include "Reduce.hpp"
#include "Regroup.hpp"
#include "VertexTemplates.hpp"
#include "gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "popnn/exceptions.hpp"
#include "Cast.hpp"
#include "Util.hpp"
#include "Zero.hpp"
#include <unordered_set>

using namespace poplar;
using namespace poplar::program;
using namespace convutil;

static void
applyTensorMapping(
    Graph &graph,
    const Tensor &t,
    const std::vector<std::vector<std::pair<unsigned, unsigned>>> &mapping) {
  auto flattened = t.flatten();
  const auto numTiles = mapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &region : mapping[tile]) {
      graph.setTileMapping(flattened.slice(region.first, region.second), tile);
    }
  }
}

namespace {
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
}

namespace conv {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX, unsigned strideY,
             unsigned strideX, unsigned paddingY,
             unsigned paddingX) {
  return convutil::getOutputDim(inDimY, inDimX, kernelSizeY, kernelSizeX,
                                 strideY, strideX, paddingY, paddingX);
}

poplar::Tensor
createWeights(poplar::Graph &graph, std::string dType,
             unsigned inNumChans,
             unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned outNumChans,
             const Plan &plan) {
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
linearizeTileIndices(unsigned batchNum, unsigned batchSize,
                     unsigned numTiles,
                     unsigned izg, unsigned ox, unsigned oy,
                     unsigned ozg,
                     const Plan &plan,
                     bool isMultiIPU) {
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto batchElemsPerTile = (batchSize + numTiles - 1) / numTiles;
  const auto numBatchGroups =
      (batchSize + batchElemsPerTile - 1) / batchElemsPerTile;
  const auto tilesPerBatchGroup =
      numTiles / numBatchGroups;
  const auto beginTile = batchNum / batchElemsPerTile * tilesPerBatchGroup;
  // If this is a multi IPU system then choose an order that avoids splitting
  // partial sums over IPUs
  unsigned tile;
  if (isMultiIPU)
    tile = beginTile +
      (izg + tilesPerInZGroup *
        (ox + tilesPerX *
          (oy + tilesPerY * ozg)));
  // Use ozg as the innermost dimension to increase the chance that
  // tiles in a supertile both read the same activations. This reduces
  // exchange time when supertile send / receive is used.
  else
    tile = beginTile +
           (ozg + tilesPerZ *
             (ox + tilesPerX *
               (oy + tilesPerY * izg)));
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

static void
addWeightRegions(std::vector<std::pair<unsigned,unsigned>> &regions,
                 const std::vector<std::size_t> &weightDims,
                 unsigned outZGroupBegin, unsigned outZGroupEnd,
                 unsigned inZGroupBegin, unsigned inZGroupEnd) {
  for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
    const auto regionBegin =
        weightDims[5] * weightDims[4] * weightDims[3] * weightDims[2] *
        (inZGroupBegin + weightDims[1] * ozg);
    const auto regionEnd =
        regionBegin +
        weightDims[5] * weightDims[4] * weightDims[3] * weightDims[2] *
        (inZGroupEnd - inZGroupBegin);
    regions.emplace_back(regionBegin, regionEnd);
  }
}

static std::vector<std::vector<std::pair<unsigned, unsigned>>>
calculateWeightMapping(Tensor w,
                       const poplar::Graph &graph,
                       const Plan &plan,
                       unsigned batchSize) {
  const auto partialNumChanGroups = w.dim(0);
  const auto numInZGroups = w.dim(1);
  const auto kernelSizeY = w.dim(2);
  const auto kernelSizeX = w.dim(3);
  const auto partialChansPerGroup = w.dim(4);
  const auto inChansPerGroup = w.dim(5);

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  std::vector<std::vector<std::pair<unsigned, unsigned>>>
      mapping(deviceInfo.getNumTiles());
  if (batchSize > 1) {
    // For multi-item batches the weights are going to be sent across
    // exchange independently of mapping. So just map weights across the
    // tile array.
    const auto numTiles = deviceInfo.getNumTiles();
    unsigned groupSize = partialChansPerGroup * inChansPerGroup;
    unsigned numGroups = w.numElements() / groupSize;
    for (unsigned tile = 0; tile < numTiles; ++tile) {
      const auto groupBegin = numGroups * tile / numTiles;
      const auto groupEnd = numGroups * (tile + 1) / numTiles;
      if (groupBegin == groupEnd)
        continue;
      mapping[tile].emplace_back(groupBegin * groupSize, groupEnd * groupSize);
    }
    return mapping;
  }

  const auto isMultiIPU = deviceInfo.numIPUs > 1;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto numTiles = deviceInfo.getNumTiles();

  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
    for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
      unsigned outZGroupBegin, outZGroupEnd;
      std::tie(outZGroupBegin, outZGroupEnd) =
          getOutZGroupRange(ozg, partialNumChanGroups, plan);
      // Group weights that are accessed contiguously by tiles within this
      // loop body.
      std::vector<std::pair<unsigned, unsigned>> sharedWeights;
      addWeightRegions(sharedWeights, w.dims(), outZGroupBegin, outZGroupEnd,
                       inZGroupBegin, inZGroupEnd);
      mergeAdjacentRegions(sharedWeights);
      // Spread groups of weights equally across the tiles that read them.
      std::vector<std::vector<std::pair<unsigned, unsigned>>> perTileWeights;
      unsigned grainSize = partialChansPerGroup *
                           inChansPerGroup;
      if (plan.useConvolutionInstructions) {
        if (kernelSizeY == 1 && kernelSizeX == 1) {
          grainSize *= numInZGroups;
        }
      } else {
        grainSize *= kernelSizeX;
      }
      splitRegions(sharedWeights, perTileWeights,
                   partialChansPerGroup * inChansPerGroup,
                   tilesPerY * tilesPerX);
      for (unsigned oy = 0; oy != tilesPerY; ++oy) {
        for (unsigned ox = 0; ox != tilesPerX; ++ox) {
          const auto tile = linearizeTileIndices(0, batchSize, numTiles,
                                                 izg, ox, oy, ozg,
                                                 plan, isMultiIPU);
          const auto iw = ox + tilesPerX * oy;
          if (iw < perTileWeights.size()) {
            mapping[tile] = perTileWeights[iw];
          }
        }
      }
    }
  }
  return mapping;
}

template <typename Builder>
static void
iterateWeightMapping(Tensor w,
                     const poplar::Graph &graph,
                     const Plan &plan,
                     unsigned batchSize,
                     Builder &&builder) {
  const auto weightMapping = calculateWeightMapping(w, graph, plan, batchSize);
  const auto flatWeights = w.flatten();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    auto tileWeights = flatWeights.slice(0, 0);
    for (const auto &region : weightMapping[tile]) {
      const auto weightBegin = region.first;
      const auto weightEnd = region.second;
      assert(weightBegin != weightEnd);
      tileWeights = concat(tileWeights,
                           flatWeights.slice(weightBegin, weightEnd));
    }
    if (tileWeights.numElements() > 0) {
      builder(tileWeights, tile);
    }
  }
}

void
mapWeights(Tensor w, Graph &graph, const Plan &plan,
           unsigned batchSize) {
  iterateWeightMapping(w, graph, plan, batchSize,
    [&](Tensor tileWeights, unsigned tile) {
    graph.setTileMapping(tileWeights, tile);
  });
}

template <typename Builder>
static void iterateBiasMapping(Tensor b, const Graph &graph,
                               Tensor activations,
                               unsigned batchNum,
                               unsigned batchSize,
                               Builder &&builder) {
  const auto activationsMapping = computeActivationsMapping(graph,
                                                            activations,
                                                            batchNum,
                                                            batchSize);
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  const auto outNumChans = activations.dim(0) * activations.dim(3);
  const auto outNumChanGroups = activations.dim(0);
  const auto outDimY = activations.dim(1);
  const auto outDimX = activations.dim(2);
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

void mapBiases(Tensor biases, Graph &graph, Tensor activations) {
  auto batchSize = activations.dim(0);
  for (unsigned b = 0; b < batchSize; ++b) {
    iterateBiasMapping(biases, graph, activations[b], b, batchSize,
                       [&](Tensor biasSlice, unsigned tile) {
                           graph.setTileMapping(biasSlice, tile);
                       });
  }
}

static void
createConvPartial1x1OutVertex(Graph &graph,
                              unsigned tile,
                              unsigned outXBegin, unsigned outXEnd,
                              unsigned outYBegin, unsigned outYEnd,
                              unsigned ozg,
                              unsigned inZGroupBegin, unsigned inZGroupEnd,
                              unsigned strideY,
                              unsigned strideX,
                              unsigned paddingY,
                              unsigned paddingX,
                              ComputeSet fwdCS,
                              const Tensor &in, const Tensor &weights,
                              const Tensor &out) {
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  assert(kernelSizeY == 1 && kernelSizeX == 1);
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
  const auto convUnitCoeffLoadBytesPerCycle =
                deviceInfo.convUnitCoeffLoadBytesPerCycle;
  if (convUnitWeightHeight != 1) {
    throw popnn::popnn_error("Using convolution units for 1x1 convolutions "
                             "where channel grouping is not equal to weights "
                             "stored per convolution unit has not "
                             "been implemented");
  }
  const auto outHeight = outYEnd - outYBegin;
  const auto outWidth = outXEnd - outXBegin;
  const auto partialType = graph.getTensorElementType(out);
  unsigned inYBegin, inYEnd, inXBegin, inXEnd;
  std::tie(inYBegin, inYEnd) =
      getInputRange({outYBegin, outYEnd}, strideY, kernelSizeY,
                     paddingY, inDimY, false);
  std::tie(inXBegin, inXEnd) =
      getInputRange({outXBegin, outXEnd}, strideX,
                     kernelSizeX, paddingX, inDimX, false);

  // Add the vertex.
  Tensor w =
      weights[ozg].slice(
  {inZGroupBegin, 0, 0, 0, 0},
  {inZGroupEnd, 1, 1, outChansPerGroup, inChansPerGroup}
        ).flatten();
  auto v = graph.addVertex(
        fwdCS,
        templateVertex("popnn::ConvPartial1x1Out", dType, partialType),
  {{"weights", w}}
        );
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                        convUnitCoeffLoadBytesPerCycle);
  std::vector<std::vector<PartialRow>> workerPartition;
  unsigned outputStride = 1;
  workerPartition =
      partitionConvPartialByWorker(outHeight, outWidth,
                                   contextsPerVertex, outputStride);
  graph.setFieldSize(v["weightReuseCount"], contextsPerVertex);
  for (unsigned i = 0; i != contextsPerVertex; ++i) {
    graph.setInitialValue(
          v["weightReuseCount"][i],
        static_cast<std::uint32_t>(workerPartition[i].size())
        );
  }
  unsigned numConvolutions = 0;
  for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
    for (unsigned i = 0; i != contextsPerVertex; ++i) {
      for (const auto &partialRow : workerPartition[i]) {
        const auto workerOutY = outYBegin + partialRow.rowNumber;
        const auto workerOutXBegin = outXBegin + partialRow.begin;
        const auto workerOutXEnd = outXBegin + partialRow.end;
        const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
        const auto workerInY =
            getInputIndex(workerOutY, strideY, kernelSizeY,
                          paddingY, inDimY, 0, false);
        assert(workerInY != ~0U);
        unsigned workerInXBegin, workerInXEnd;
        std::tie(workerInXBegin, workerInXEnd) =
            getInputRange({workerOutXBegin, workerOutXEnd}, strideX,
                          kernelSizeX, paddingX, inDimX, false);
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
        graph.setTileMapping(outWindow, tile);
        graph.connect(v["in"][numConvolutions], inWindow);
        graph.connect(v["out"][numConvolutions], outWindow);
        ++numConvolutions;
      }
    }
  }
  graph.setFieldSize(v["in"], numConvolutions);
  graph.setFieldSize(v["out"], numConvolutions);
  // Map the vertex and output.
  graph.setTileMapping(v, tile);
}

static void
createConvPartialnx1InOutVertex(Graph &graph,
                                unsigned tile,
                                unsigned outXBegin, unsigned outXEnd,
                                unsigned outYBegin, unsigned outYEnd,
                                unsigned outZGroup,
                                unsigned inZGroupBegin, unsigned inZGroupEnd,
                                unsigned strideY, unsigned strideX,
                                unsigned paddingY, unsigned paddingX,
                                ComputeSet fwdCS,
                                const Tensor &in,
                                const Tensor &weights,
                                const Tensor &out,
                                const Tensor &zeros,
                                bool isFractional) {
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
  // Add the vertex.
  auto v =
      graph.addVertex(fwdCS,
                      templateVertex("popnn::ConvPartialnx1InOut",
                                     dType, partialType,
                                     isFractional ? "true" : "false"));
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                        convUnitCoeffLoadBytesPerCycle);
  graph.setTileMapping(v, tile);
  unsigned numWeights = 0;
  unsigned numConvolutions = 0;
  for (unsigned wyBegin = 0; wyBegin < kernelSizeY;
       wyBegin += convUnitWeightHeight) {
    const auto wyEnd = std::min(static_cast<unsigned>(kernelSizeY),
                                wyBegin + convUnitWeightHeight);
    unsigned convOutYBegin, convOutYEnd;
    std::tie(convOutYBegin, convOutYEnd) =
        getOutputRange({outYBegin, outYEnd}, strideY, kernelSizeY,
                       paddingY, inDimY, {wyBegin, wyEnd},
                       isFractional);
    const auto convOutHeight = convOutYEnd - convOutYBegin;
    if (convOutHeight == 0)
      continue;
    for (unsigned wx = 0; wx != kernelSizeX; ++wx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRange({outXBegin, outXEnd}, strideX, kernelSizeX,
                         paddingX, inDimX, wx,
                         isFractional);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;

      // In a fractionally strided pass, if we are handling one row of the
      // kernel at a time, the partitioning of work across the workers can be
      // aware of the stride and only allocate work on the rows that get
      // affected.
      unsigned outputStride =
          (isFractional && convUnitWeightHeight == 1) ? strideY : 1;
      std::vector<std::vector<PartialRow>> workerPartition =
          partitionConvPartialByWorker(convOutHeight, convOutWidth,
                                       contextsPerVertex, outputStride);
      assert(workerPartition.size() == contextsPerVertex);
      for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
        for (unsigned wy = wyBegin; wy != wyBegin + convUnitWeightHeight;
             ++wy) {
          Tensor w;
          if (wy < wyEnd) {
            w = weights[outZGroup][izg][wy][wx].flatten();
          } else {
            w = zeros.slice(0, inChansPerGroup * outChansPerGroup);
          }
          const auto weightsIndex =
              numWeights * convUnitWeightHeight + wy - wyBegin;
          graph.connect(v["weights"][weightsIndex], w);
        }
        for (unsigned i = 0; i != contextsPerVertex; ++i) {
          graph.setInitialValue(
            v["weightReuseCount"][numWeights * contextsPerVertex + i],
            static_cast<std::uint32_t>(workerPartition[i].size())
          );
          for (const auto &partialRow : workerPartition[i]) {
            const auto workerOutY = convOutYBegin + partialRow.rowNumber;
            unsigned workerOutXBegin, workerOutXEnd;
            std::tie(workerOutXBegin, workerOutXEnd) =
                getOutputRange({convOutXBegin + partialRow.begin,
                                convOutXBegin + partialRow.end},
                                strideX, kernelSizeX, paddingX, inDimX,
                                wx,
                                isFractional);
            const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
            unsigned workerInXBegin, workerInXEnd;
            std::tie(workerInXBegin, workerInXEnd) =
                getInputRange({workerOutXBegin, workerOutXEnd}, strideX,
                              kernelSizeX, paddingX, inDimX, wx,
                              isFractional);
            const auto workerInWidth = workerInXEnd - workerInXBegin;
            for (unsigned wy = wyBegin; wy != wyBegin + convUnitWeightHeight;
                 ++wy) {
              const auto workerInY =
                  getInputIndex(workerOutY, strideY, kernelSizeY,
                                paddingY, inDimY, wy,
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
              const auto inIndex =
                  numConvolutions * convUnitWeightHeight + wy - wyBegin;
              graph.connect(v["in"][inIndex], inWindow);
            }
            Tensor outWindow =
                out[outZGroup][workerOutY].slice(
                  {workerOutXBegin, 0},
                  {workerOutXEnd, outChansPerGroup}
                ).reshape({workerOutWidth * outChansPerGroup});
            // Note the output tensor is mapped in zeroAndMapPartialSums.
            graph.connect(v["out"][numConvolutions], outWindow);
            ++numConvolutions;
          }
        }
        ++numWeights;
      }
    }
  }
  graph.setFieldSize(v["in"], numConvolutions * convUnitWeightHeight);
  graph.setFieldSize(v["out"], numConvolutions);
  graph.setFieldSize(v["weights"], numWeights * convUnitWeightHeight);
  graph.setFieldSize(v["weightReuseCount"], numWeights * contextsPerVertex);
}

static void
createConvPartialDotProductVertex(Graph &graph,
                                  const Plan &plan,
                                  unsigned tile,
                                  unsigned outXBegin, unsigned outXEnd,
                                  unsigned outYBegin, unsigned outYEnd,
                                  unsigned z,
                                  unsigned inZGroupBegin, unsigned inZGroupEnd,
                                  unsigned strideY, unsigned strideX,
                                  unsigned paddingY, unsigned paddingX,
                                  std::string dType,
                                  ComputeSet fwdCS,
                                  const Tensor &in,
                                  const Tensor &weights,
                                  const Tensor &out) {
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto dataPathWidth = graph.getDevice().getDeviceInfo().dataPathWidth;
  const auto inZGroups = inZGroupEnd - inZGroupBegin;
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto partialType = plan.getPartialType();
  if (outChansPerGroup != 1)
    assert(!"outChansPerGroup must be 1");
  assert(outYEnd - outYBegin == 1);
  const auto y = outYBegin;
  unsigned inYBegin, inYEnd, inXBegin, inXEnd;
  std::tie(inYBegin, inYEnd) =
      getInputRange(y, strideY, kernelSizeY, paddingY, inDimY, false);
  std::tie(inXBegin, inXEnd) =
      getInputRange({outXBegin, outXEnd}, strideX, kernelSizeX,
                    paddingX, inDimX, false);
  // Window into previous layer.
  const auto inWidth = inXEnd - inXBegin;
  const auto inHeight = inYEnd - inYBegin;
  // Weights that match the window.
  unsigned weightYBegin, weightYEnd;
  std::tie(weightYBegin, weightYEnd) =
      getKernelRange(y, strideY, kernelSizeY, paddingY, inDimY, false);
  Tensor inWindow =
      in.slice(
  {inZGroupBegin, inYBegin, inXBegin, 0},
  {inZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
        ).reshape({inHeight * inZGroups,
                   inWidth * inChansPerGroup});
  Tensor w =
      weights[z].slice(
  {inZGroupBegin, weightYBegin, 0, 0, 0},
  {inZGroupEnd, weightYEnd, kernelSizeX, 1, inChansPerGroup}
        ).reshape({inHeight * inZGroups,
                   inChansPerGroup * kernelSizeX});
  Tensor outWindow = out[z][y].slice(outXBegin, outXEnd).flatten();
  // Add the vertex.
  auto v = graph.addVertex(fwdCS,
                           templateVertex("popnn::ConvPartial", dType,
                                          partialType),
  { {"in", inWindow },
    {"weights", w },
    {"out", outWindow },
                           });
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["stride"], strideX);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  const auto inXBeginPaddedField = outXBegin * strideX;
  unsigned vPadding =
      inXBeginPaddedField < paddingX ? paddingX - inXBeginPaddedField : 0;
  graph.setInitialValue(v["padding"], vPadding);
  // Map the vertex and output.
  graph.setTileMapping(v, tile);
  graph.setTileMapping(outWindow, tile);
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
  std::vector<std::pair<unsigned, unsigned>> regions;
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

static void
calcPartialConvOutput(Graph &graph,
                      const Plan &plan,
                      std::string dType,
                      unsigned tile,
                      unsigned outXBegin, unsigned outXEnd,
                      unsigned outYBegin, unsigned outYEnd,
                      unsigned outZGroupBegin, unsigned outZGroupEnd,
                      unsigned inZGroupBegin, unsigned inZGroupEnd,
                      unsigned strideY, unsigned strideX, unsigned paddingY,
                      unsigned paddingX,
                      ComputeSet zeroCS,
                      ComputeSet fwdCS,
                      Tensor in, Tensor weights, Tensor out,
                      bool isFractional) {
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  Tensor zeros;
  bool useConvPartial1x1OutVertex = false;
  if (plan.useConvolutionInstructions) {
    const auto weightsPerConvUnit =
        deviceInfo.getWeightsPerConvUnit(dType == "float");
    assert(weightsPerConvUnit % inChansPerGroup == 0);
    const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
    if (convUnitWeightHeight != 1) {
      assert(plan.useConvolutionInstructions);
      const auto inDimX = in.dim(2);
      const auto inputRange = getInputRange({outXBegin, outXEnd}, strideX,
                                            kernelSizeX, paddingX,
                                            inDimX, isFractional);
      const auto inputRangeSize = inputRange.second - inputRange.first;
      // This isn't split across multiple workers since it can happen in
      // parallel with zeroing the partial sums.
      const auto zeroSize = std::max(inputRangeSize * inChansPerGroup,
                                     inChansPerGroup * outChansPerGroup);
      zeros = graph.addTensor(dType,
                              {zeroSize},
                              "zeros");
      auto v = graph.addVertex(zeroCS, templateVertex("popnn::Zero", dType),
                               {{"out", zeros}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
      graph.setTileMapping(zeros, tile);
    }
    useConvPartial1x1OutVertex = kernelSizeX == 1 && kernelSizeY == 1 &&
                                 (!isFractional ||
                                    (strideX == 1 && strideY == 1));
    if (!useConvPartial1x1OutVertex) {
      zeroAndMapPartialSums(graph, outXBegin, outXEnd, outYBegin, outYEnd,
                            outZGroupBegin, outZGroupEnd, tile, zeroCS, out);
    }
  }
  const auto outHeight = outYEnd - outYBegin;
  const auto verticesPerY = plan.verticesPerTilePerYAxis;
  for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
    for (unsigned vy = 0; vy != verticesPerY; ++vy) {
      const auto vertexOutYBegin =
          outYBegin + (vy * outHeight) / verticesPerY;
      const auto vertexOutYEnd =
          outYBegin + ((vy + 1) * outHeight) / verticesPerY;
      const auto outHeight = vertexOutYEnd - vertexOutYBegin;
      if (outHeight == 0)
        continue;
      if (useConvPartial1x1OutVertex) {
        createConvPartial1x1OutVertex(graph, tile,
                                      outXBegin, outXEnd,
                                      vertexOutYBegin, vertexOutYEnd,
                                      ozg,
                                      inZGroupBegin, inZGroupEnd,
                                      strideY, strideX, paddingY, paddingX,
                                      fwdCS, in, weights, out);
      } else if (plan.useConvolutionInstructions) {
        createConvPartialnx1InOutVertex(graph, tile, outXBegin, outXEnd,
                                        vertexOutYBegin, vertexOutYEnd,
                                        ozg,
                                        inZGroupBegin, inZGroupEnd,
                                        strideY, strideX, paddingY, paddingX,
                                        fwdCS, in, weights, out,
                                        zeros, isFractional);
      } else {
        if (isFractional)
          throw popnn::popnn_error("Non convolution instruction based "
                                   "fractional convolutions are not "
                                   "implemented");
        createConvPartialDotProductVertex(graph, plan, tile,
                                          outXBegin, outXEnd,
                                          vertexOutYBegin, vertexOutYEnd,
                                          ozg,
                                          inZGroupBegin, inZGroupEnd,
                                          strideY, strideX,
                                          paddingY, paddingX,
                                          dType, fwdCS, in, weights, out);
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
                unsigned strideY, unsigned strideX,
                unsigned paddingY, unsigned paddingX,
                unsigned outNumChans,
                std::string dType,
                Tensor in, Tensor weights, Tensor partials,
                const std::string &layerName,
                unsigned outDimX, unsigned outDimY,
                bool isFractional) {
  const auto batchSize = in.dim(0);
  const auto isMultiIPU = graph.getDevice().getDeviceInfo().numIPUs > 1;
  const auto inNumChans = in.dim(1) * in.dim(4);
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();

  ComputeSet zeroCS = graph.createComputeSet(layerName +"/Zero");
  ComputeSet convolveCS = graph.createComputeSet(layerName + "/Convolve");
  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
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
            const auto tile = linearizeTileIndices(b, batchSize, numTiles,
                                                   izg, ox, oy, ozg,
                                                   plan,
                                                   isMultiIPU);
            calcPartialConvOutput(graph, plan, dType,
                                  tile, outXBegin, outXEnd, outYBegin, outYEnd,
                                  outZGroupBegin, outZGroupEnd, inZGroupBegin,
                                  inZGroupEnd,
                                  strideY, strideX,
                                  paddingY, paddingX, zeroCS,
                                  convolveCS,
                                  in[b], weights,
                                  partials[b][izg],
                                  isFractional);
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

static Tensor
reduce(Graph &graph,
       Tensor partials,
       const std::string reducedType,
       const std::vector<
         std::vector<std::pair<unsigned, unsigned>>
       > &reducedMapping,
       ComputeSet reduceCS) {
  const auto partialType = graph.getTensorElementType(partials);
  if (partials.dim(0) == 1 && reducedType == partialType) {
    return partials[0];
  }
  Tensor reduced = graph.addTensor(reducedType,
                                   partials[0].dims(), "reduced");
  applyTensorMapping(graph, reduced, reducedMapping);
  ::reduce(graph, partials, reduced, reducedMapping, reduceCS);
  return reduced;
}

/// Compute a tile mapping for the reduced tensor. The size of each contiguous
/// region mapped to a tile is a multiple of the vector width. Where possible
/// elements of the reduced tensor are mapped to the same tile as the output
/// activations that they are used to compute. If the output channel group size
/// is not a multiple of the vector width then some exchange may be required
/// between the reduce and complete compute sets. No exchange is required if the
/// vector width exactly divides the output channel group size.
static std::vector<std::vector<std::pair<unsigned, unsigned>>>
computeReducedMapping(const poplar::Graph &graph,
                      const std::string &partialType,
                      unsigned partialChansPerGroup,
                      const Tensor &activations,
                      const std::vector<unsigned> &activationsMapping) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto grainSize =
      partialType == "float" ? deviceInfo.getFloatVectorWidth() :
                               deviceInfo.getHalfVectorWidth();
  const auto numTiles = activationsMapping.size() - 1;
  std::vector<std::vector<std::pair<unsigned, unsigned>>>
      reducedMapping(numTiles);
  assert(activations.getDimensionality() == 4);
  const auto dimY = activations.dim(1);
  const auto dimX = activations.dim(2);
  const auto fieldSize = dimY * dimX;
  const auto outChansPerGroup = activations.dim(3);
  const auto numActivations = static_cast<unsigned>(activations.numElements());
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto begin = activationsMapping[tile];
    const auto end = activationsMapping[tile + 1];
    if (begin / outChansPerGroup == end / outChansPerGroup) {
      continue;
    }
    std::vector<unsigned> elements;
    const auto outChanGroupBegin = (begin / outChansPerGroup) / fieldSize;
    const auto outChanGroupEnd = ((end / outChansPerGroup) - 1) / fieldSize + 1;
    for (unsigned outChanGroup = outChanGroupBegin;
         outChanGroup != outChanGroupEnd; ++outChanGroup) {
      unsigned fieldBegin, fieldEnd;
      if (outChanGroup == outChanGroupBegin) {
        fieldBegin = (begin / outChansPerGroup) % fieldSize;
      } else {
        fieldBegin = 0;
      }
      if (outChanGroup + 1 == outChanGroupEnd) {
        fieldEnd = ((end / outChansPerGroup) - 1) % fieldSize + 1;
      } else {
        fieldEnd = fieldSize;
      }
      const auto chanBegin = outChanGroup * outChansPerGroup;
      const auto chanEnd = (outChanGroup + 1) * outChansPerGroup;
      for (unsigned chan = chanBegin; chan != chanEnd; ++chan) {
        const auto partialChanInGroup = chan % partialChansPerGroup;
        const auto partialChanGroup = chan / partialChansPerGroup;
        for (unsigned pos = fieldBegin; pos != fieldEnd; ++pos) {
          elements.push_back(partialChanInGroup +
                               partialChansPerGroup * (pos +
                                 fieldSize * partialChanGroup));
        }
      }
    }
    std::sort(elements.begin(), elements.end());
    const auto contiguousRegions = getContiguousRegions(elements.begin(),
                                                        elements.end());
    for (const auto &region : contiguousRegions) {
      const auto roundedBegin = region.first / grainSize * grainSize;
      auto roundedEnd = std::min(region.second / grainSize * grainSize,
                                       numActivations);

      if (roundedBegin == roundedEnd) {
        /* ensure that any fractional grain size at the end is mapped */
        if (numActivations - roundedEnd < grainSize)
          roundedEnd = numActivations;
        else
          continue;
      }

      if (!reducedMapping[tile].empty() &&
          reducedMapping[tile].back().second == roundedBegin) {
        reducedMapping[tile].back().second = roundedEnd;
      } else {
        reducedMapping[tile].emplace_back(roundedBegin, roundedEnd);
      }
    }
  }
  return reducedMapping;
}

static void
complete(Graph &graph,
         const Plan &plan,
         unsigned outNumChans,
         std::string dType,
         Tensor in, Tensor biases, Tensor activations,
         const std::vector<unsigned> &activationsMapping,
         ComputeSet cs) {
  // Apply the non linearity and write back results in the layout desired by
  // the next layer. Each vertex handles outChansPerGroup output elements.
  // TODO: This step could be merged with the reduction step.
  const auto outDimY = activations.dim(1);
  const auto outDimX = activations.dim(2);
  const auto outNumChanGroups = activations.dim(0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto partialType =  graph.getTensorElementType(in);
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  Tensor biasesByChanGroup =
      biases.reshape({outNumChanGroups, outChansPerGroup});
  const auto numTiles = deviceInfo.getNumTiles();
  const auto workersPerTile = deviceInfo.numWorkerContexts;
  const auto partialChanChunkSize =
      gcd<unsigned>(outChansPerGroup, partialChansPerGroup);

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
    const auto maxGroupsPerWorker =
        (tileNumGroups + workersPerTile - 1) / workersPerTile;
    // Choose the number of vertices such that each vertices is reponsible for
    // at most maxGroupsPerWorker groups.
    const auto verticesToCreate =
        (tileNumGroups + maxGroupsPerWorker - 1) / maxGroupsPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto groupBegin =
          (vertex * tileNumGroups) / verticesToCreate + tileGroupBegin;
      const auto groupEnd =
          ((vertex + 1) * tileNumGroups) / verticesToCreate + tileGroupBegin;
      if (groupBegin == groupEnd)
        continue;
      // Create a vertex for this worker to process a number of output channel
      // groups.
      const auto numGroups = groupEnd - groupBegin;
      auto v = graph.addVertex(cs,
                               templateVertex("popnn::ConvComplete",
                                              partialType,
                                              dType));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);

      // Add the biases and a vector that tells the vertex how many output
      // groups to process for each bias.
      auto minOutChanGroup = groupBegin / (outDimX * outDimY);
      auto maxOutChanGroup = (groupEnd - 1) / (outDimX * outDimY);
      Tensor biasSlice = biasesByChanGroup.slice(minOutChanGroup,
                                                 maxOutChanGroup + 1);
      graph.connect(v["bias"], biasSlice);
      graph.setFieldSize(v["outputChanGroupsPerBias"],
                         maxOutChanGroup - minOutChanGroup + 1);
      for (auto outChanGroup = minOutChanGroup;
           outChanGroup <= maxOutChanGroup;
           ++outChanGroup) {
        auto gBegin = std::max(groupBegin, outChanGroup * outDimY * outDimX);
        auto gEnd = std::min(groupEnd, (outChanGroup+1) * outDimY * outDimX);
        unsigned outputsPerBias = gEnd - gBegin;
        auto i = outChanGroup - minOutChanGroup;
        graph.setInitialValue(v["outputChanGroupsPerBias"][i],
                              outputsPerBias);
      }

      // Connect the output channel groups and inputs from the partial sums.
      graph.setFieldSize(v["out"], numGroups);
      graph.setFieldSize(v["in"],
                         numGroups * outChansPerGroup / partialChanChunkSize);
      unsigned numIn = 0;
      for (auto group = groupBegin; group != groupEnd; ++group) {
        auto outChanGroup = group / (outDimX * outDimY);
        auto y = group % (outDimX * outDimY) / outDimX;
        auto x = group % outDimX;
        auto out = activations[outChanGroup][y][x];
        graph.connect(v["out"][group - groupBegin], out);
        Tensor reducedChans = in.slice(
           {0, y, x, 0},
           {in.dim(0), y + 1, x + 1, partialChansPerGroup}
        ).flatten();
        Tensor reducedByChanGroup =
            reducedChans.reshape({outNumChanGroups,
                                  outChansPerGroup / partialChanChunkSize,
                                  partialChanChunkSize});
        Tensor in = reducedByChanGroup[outChanGroup];
        for (unsigned i = 0; i < in.dim(0); ++i) {
          graph.connect(in[i], v["in"][numIn++]);
        }
      }
    }
  }
}

Program
convolution(Graph &graph,
            const Plan &plan,
            unsigned strideY, unsigned strideX,
            unsigned paddingY, unsigned paddingX,
            Tensor in, Tensor weights, Tensor biases, Tensor activations,
            const std::string &partialsType, bool isFractional,
            bool useWinogradConv, unsigned winogradPatchSize,
            const std::string &debugPrefix) {
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  const auto dType = graph.getTensorElementType(in);
  const auto layerName =
      debugPrefix + "/Conv"
                  + std::to_string(kernelSizeX)
                  + "x" + std::to_string(kernelSizeY)
                  + "_stride" + std::to_string(strideX) + "x"
                  + std::to_string(strideY);
  const auto outDimY = activations.dim(2);
  const auto outDimX = activations.dim(3);
  unsigned partialOutDimY, partialOutDimX;
  const auto batchSize = activations.dim(0);
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
  const auto outNumChans = activations.dim(1) * activations.dim(4);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto partialTypeInPlan = plan.getPartialType();

  mapBiases(biases, graph, activations);

  auto forwardProg = Sequence();

  if (useWinogradConv
      && winogradPatchSize == 4
      && strideY == 1 && strideX == 1
      && kernelSizeY == 3 && kernelSizeX == 3
      && !plan.flattenXY
      && (weights.dim(4) % 4 == 0)
      && (activations.dim(4) % 4 == 0)) {


    // Perform each element of the batch serially
    for (unsigned b = 0; b < batchSize; ++b) {
      forwardProg.add(winogradConvolution(
          graph, strideY, strideX,
          paddingY, paddingX,
          in.dim(3), in.dim(2), outNumChans,
          winogradPatchSize, winogradPatchSize,
          dType, partialsType, in[b],
          weights, biases,
          activations[b],
          debugPrefix));
    }
  } else {

    mapWeights(weights, graph, plan, batchSize);

    // Calculate a set of partial sums of the convolutions.
    Tensor partials = graph.addTensor(partialTypeInPlan,
                                       {numBatchGroups,
                                       tilesPerInZGroup,
                                       partialNumChanGroups,
                                       partialOutDimY,
                                       partialOutDimX,
                                       partialChansPerGroup},
                                      "partials");
    forwardProg.add(calcPartialSums(graph, plan,
                                    strideY, strideX,
                                    paddingY, paddingX, outNumChans,
                                    dType, in, weights, partials, layerName,
                                    partialOutDimX, partialOutDimY,
                                    isFractional));

    if (plan.flattenXY) {
      partials = partials.dimShuffle({1, 2, 0, 3, 4, 5 }).reshape(
        {tilesPerInZGroup, partialNumChanGroups,
         numBatchGroups * plan.batchesPerGroup,
         partialOutDimY / plan.batchesPerGroup,
         partialOutDimX, partialChansPerGroup}).dimShuffle(
            {2, 0, 1, 3, 4, 5});
    }
    ComputeSet reduceCS = graph.createComputeSet(layerName + "/Reduce");

    ComputeSet castCS = graph.createComputeSet(layerName + "/Cast");

    // For each element of the batch, we add the reduction and complete
    // vertices to same compute sets so the batch will be executed in parallel.
    ComputeSet completeCS = graph.createComputeSet(layerName + "/Complete");
    for (unsigned b = 0; b < batchSize; ++b) {
      // Perform the reduction of partial sums.
      auto activationsMapping = computeActivationsMapping(graph,
                                                          activations[b],
                                                          b,
                                                          batchSize);
      auto reducedMapping =
          computeReducedMapping(graph, partialTypeInPlan, partialChansPerGroup,
                                activations[b],
                                activationsMapping);
      Tensor reduced = reduce(graph,
                              partials[b],
                              dType,
                              reducedMapping,
                              reduceCS);

      reduced = reduced.reshape({partialNumChanGroups, outDimY, outDimX,
                                 partialChansPerGroup});

      // Add the bias and rearrange tensor to required output channel grouping.
      complete(graph, plan, outNumChans, dType, reduced, biases,
               activations[b],
               activationsMapping, completeCS);
    }
    if (!graph.getComputeSet(reduceCS).empty())
      forwardProg.add(Execute(reduceCS));
    if (!graph.getComputeSet(castCS).empty())
      forwardProg.add(Execute(castCS));

    forwardProg.add(Execute(completeCS));
  }
  return forwardProg;
}

static std::uint64_t getNumberOfMACs(unsigned outDimY, unsigned outDimX,
                                     unsigned outNumChans,
                                     unsigned kernelSizeY, unsigned kernelSizeX,
                                     unsigned strideY, unsigned strideX,
                                     unsigned paddingY, unsigned paddingX,
                                     unsigned inDimY, unsigned inDimX,
                                     unsigned inNumChans,
                                     bool forwardOnly) {
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
  if (forwardOnly)
    return numMACs;
  else
    return numMACs * 3;
}


uint64_t getFlops(unsigned batchSize,
                  unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSizeY, unsigned kernelSizeX, unsigned strideY,
                  unsigned strideX, unsigned paddingY, unsigned paddingX,
                  unsigned outNumChans, bool forwardOnly) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSizeY,
                                            kernelSizeX, strideY, strideX,
                                            paddingY, paddingX);
  auto flopsPerItem =
      2 * getNumberOfMACs(outDimY, outDimX, outNumChans,
                          kernelSizeY, kernelSizeX, strideY, strideX,
                          paddingY, paddingX,
                          inDimY, inDimX, inNumChans, forwardOnly);
  return batchSize * flopsPerItem;
}

double getPerfectCycleCount(const Graph &graph,
                            std::string dType,
                            unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned inNumChans,
                            unsigned kernelSizeY, unsigned kernelSizeX,
                            unsigned strideY, unsigned strideX,
                            unsigned paddingY, unsigned paddingX,
                            unsigned outNumChans,
                            bool forwardOnly) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSizeY,
                                            kernelSizeX, strideY,
                                            strideX, paddingY, paddingX);
  const auto numTiles = deviceInfo.getNumTiles();
  auto numMacs =
      batchSize * getNumberOfMACs(outDimY, outDimX, outNumChans, kernelSizeY,
                                  kernelSizeX, strideY, strideX,
                                  paddingY, paddingX, inDimY, inDimX,
                                  inNumChans, forwardOnly);

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

std::vector<size_t> getElementCoord(size_t element,
                                    const std::vector<size_t> dims) {
  std::vector<size_t> coord(dims.size());
  for (int i = dims.size() - 1; i >= 0; --i) {
    coord[i] = element % dims[i];
    element = element / dims[i];
  }
  return coord;
}

/** Copy the weights in 'weightsIn' into 'weightsOut' such that
 *  each element of the kernel is transposed w.r.t. the input and output
 *  channels and flip both the X and Y axis of the kernel field.
 */
Program weightsTransposeChansFlipXY(Graph &graph,
                                    Tensor weightsIn,
                                    Tensor weightsOut) {
  // weights = { O/G1, I/G2, KY, KX, G1, G2 }
  // bwdweights = { I/G3, O/G4, KY, KX, G3, G4 }

  const auto dType = graph.getTensorElementType(weightsIn);
  const auto KY = weightsOut.dim(2);
  const auto KX = weightsOut.dim(3);
  const auto I = weightsOut.dim(0) * weightsOut.dim(4);
  const auto O = weightsOut.dim(1) * weightsOut.dim(5);
  const auto G1 = weightsIn.dim(4);
  const auto G2 = weightsIn.dim(5);
  const auto G3 = weightsOut.dim(4);
  const auto G4 = weightsOut.dim(5);

  auto wFlippedY = graph.addTensor(dType, {O/G1, I/G2, 0, KX, G1, G2});
  for (int wy = KY - 1; wy >= 0; --wy) {
     wFlippedY = concat(wFlippedY, weightsIn.slice(wy, wy + 1, 2), 2);
  }

  auto wFlippedYX= graph.addTensor(dType, {O/G1, I/G2, KY, 0, G1, G2});
  for (int wx = KX - 1; wx >= 0; --wx) {
     wFlippedYX = concat(wFlippedYX, wFlippedY.slice(wx, wx + 1, 3), 3);
  }

  return Copy(weightsOut,
              wFlippedYX.dimShuffle({2, 3, 0, 4, 1, 5})
                        .reshape({KY, KX, O/G4, G4, I/G3, G3})
                        .dimShuffle({4, 2, 0, 1, 5, 3}));
}

Program convolutionBackward(Graph &graph,
                            const Plan &plan,
                            Tensor zDeltas, Tensor weights,
                            Tensor deltasOut,
                            unsigned strideY, unsigned strideX,
                            unsigned paddingY, unsigned paddingX,
                            bool isFractional, const std::string &debugPrefix) {
  const auto batchSize = deltasOut.dim(0);
  const auto dType = graph.getTensorElementType(zDeltas);
  const auto outNumChans = deltasOut.dim(1) * deltasOut.dim(4);
  const auto partialType = plan.getPartialType();
  const auto inNumChans = zDeltas.dim(1) * zDeltas.dim(4);

  auto prog = Sequence();

  // Create transpose/flipped weights
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  auto bwdWeights = createWeights(graph, dType, inNumChans, kernelSizeY,
                                  kernelSizeX, outNumChans, plan);
  mapWeights(bwdWeights, graph, plan, batchSize);
  prog.add(weightsTransposeChansFlipXY(graph, weights, bwdWeights));

  // Create zero biases
  auto zeros = graph.addConstantTensor(dType, {outNumChans}, 0);
  auto biases = graph.addTensor(dType, {outNumChans}, "zeroBiases");
  mapBiases(biases, graph, deltasOut);
  prog.add(Copy(biases, zeros));

  // Perform a fractional convolution
  prog.add(convolution(graph, plan, strideY, strideX, paddingY, paddingX,
                       zDeltas, bwdWeights, biases, deltasOut, partialType,
                       isFractional, false, 4, debugPrefix));
  return prog;
}

// Let A be a n x m matrix and B be a m x p matrix. Compute C = A x B.
// Let u be the number of convolution units and w be the number of weights
// per convolutional unit. n must be a multiple of u and m must be a multiple
// of w. Elements of A are loaded in to the convolution units.
// The dimensions of A should be split and arranged as follows:
// [n/u][m/w][u][w].
// The dimensions of B should be split and arranged as follows:
// [m/w][p][w].
// The dimensions of the C should be split and arranged as follows:
// [n/u][p][u].
Program
matrixMultiplyByConvInstruction(Graph &graph, const Plan &plan,
                                Tensor a, Tensor b, Tensor c,
                                const std::vector<unsigned> &cTileMapping,
                                const std::string &debugPrefix) {
  const auto dType = graph.getTensorElementType(a);
  assert(a.getDimensionality() == 4);
  assert(b.getDimensionality() == 3);
  assert(c.getDimensionality() == 3);
  assert(a.dim(0) == c.dim(0));
  assert(a.dim(1) == b.dim(0));
  assert(a.dim(2) == c.dim(2));
  assert(a.dim(3) == b.dim(2));
  assert(b.dim(1) == c.dim(1));
  const auto w = a.dim(3);
  const auto u = a.dim(2);
  const auto n = a.dim(0) * u;
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
  const auto stride = 1;
  const auto padding = 0;
  const auto outDimY = 1;
  const auto outDimX = p;
  const auto outNumChans = n;
  const auto outChansPerGroup = u;
  // Insert size one dimensions for the filter height and width.
  const auto weights = a.reshape({a.dim(0),
                                  a.dim(1),
                                  kernelSize,
                                  kernelSize,
                                  a.dim(2),
                                  a.dim(3)});
  // Insert size one dimension for the batch size and field height.
  const auto in = b.reshape({batchSize, b.dim(0), outDimY, b.dim(1), b.dim(2)});
  const auto out = c.reshape({batchSize, c.dim(0), outDimY, c.dim(1),
                              c.dim(2)});

  auto prog = Sequence();
  Tensor partials;
  const auto partialType = plan.getPartialType();
  const auto outputType = graph.getTensorElementType(out);
  const bool reductionOrCastRequired = partialType != outputType
                                       || plan.tilesPerInZGroupAxis != 1;
  if (!reductionOrCastRequired) {
    // No reduction required.
    partials = out.reshape({batchSize,
                            plan.tilesPerInZGroupAxis,
                            outNumChans / outChansPerGroup,
                            outDimY,
                            outDimX,
                            outChansPerGroup});
  } else {
    partials = graph.addTensor(partialType,
                                {batchSize,
                                 plan.tilesPerInZGroupAxis,
                                 outNumChans / outChansPerGroup,
                                 outDimY,
                                 outDimX,
                                 outChansPerGroup},
                                 "partials");
  }

  // Calculate a set of partial sums of the convolutions.
  prog.add(calcPartialSums(graph, plan, stride, stride,
                           padding, padding, outNumChans,
                           dType, in, weights, partials,
                           debugPrefix + "/MatrixMul",
                           outDimX, outDimY, false));

  if ( plan.tilesPerInZGroupAxis > 1) {
    // Perform the reduction of partial sums.
    auto reduceCS = graph.createComputeSet(debugPrefix + "/Reduce");
    auto reducedMapping = computeReducedMapping(graph, partialType,
                                                outChansPerGroup,
                                                out[0], cTileMapping);
    ::reduce(graph, partials[0], out[0], reducedMapping, reduceCS);

    prog.add(Execute(reduceCS));
  } else if (partialType != outputType) {
    /* If no reduction is required where all input channel groups are allocated
     * to a tile, partials must be cast to the output type
     */
    prog.add(cast(graph, cTileMapping, partials[0], out[0], debugPrefix));
  }
  return prog;
}

// Return a program to update the biases tensor with the gradients derived
// from the zDeltas tensor assuming that the compute set firstReduceCS is
// executed first.
static Program
convolutionBiasUpdate(Graph &graph, const Tensor &zDeltas, const Tensor &biases,
                      float learningRate, ComputeSet firstReduceCS,
                      const std::string &debugPrefix) {
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
                               templateVertex("popnn::ConvBiasReduce1",
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
  auto secondReduceCS = graph.createComputeSet(
                            debugPrefix + "/ReduceBias2");
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
                             templateVertex("popnn::Zero", dType));
      graph.connect(v["out"], biasPartials[worker].slice(0, maxBiasPerWorker));
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v, tile);
      continue;
    }
    auto v = graph.addVertex(secondReduceCS,
                             templateVertex("popnn::ConvBiasReduce2", dType));
    graph.connect(v["in"], toReduce);
    graph.connect(v["out"], biasPartials[worker].slice(0, numWorkerBiases));
    graph.setTileMapping(v, tile);
  }
  auto updateBiasCS = graph.createComputeSet(debugPrefix + "/UpdateBias");
  iterateBiasMapping(biases, graph, zDeltas[0], 0, 1,
    [&](Tensor biasSlice, unsigned tile){
      for (auto bias : biasSlice.getElementIndices()) {
        auto v = graph.addVertex(updateBiasCS,
                                 templateVertex("popnn::ConvBiasUpdate",
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
  return Sequence(Execute(secondReduceCS), Execute(updateBiasCS));
}

static unsigned weightUpdateNumActivationReducedViews(unsigned kernelSizeY,
                                                      unsigned kernelSizeX,
                                                      unsigned strideY) {
  return std::min(strideY, kernelSizeY) * kernelSizeX;
}


static Tensor weightUpdateActivationReducedViews(Graph &graph,
                                                 const Plan &plan,
                                                 Tensor zDeltas,
                                                 Tensor activations,
                                                 unsigned kernelSizeY,
                                                 unsigned kernelSizeX,
                                                 unsigned strideY,
                                                 unsigned strideX,
                                                 unsigned paddingY,
                                                 unsigned paddingX,
                                                 Sequence &prog) {
  const auto dType = graph.getTensorElementType(activations);
  const auto batchSize = zDeltas.dim(0);
  auto height = zDeltas.dim(2);
  auto width = zDeltas.dim(3);
  auto activationsNumChanGroups = activations.dim(1);
  auto activationsChansPerGroup = activations.dim(4);
  auto activationsChans = activationsNumChanGroups * activationsChansPerGroup;
  const auto fieldGroupSize = plan.inChansPerGroup;

  // Pad the field so the size is a multiple of the number of weights in the
  // convolutional unit.
  const auto batchFieldSize = height * width * batchSize;
  const auto batchPaddedFieldSize =
      ((batchFieldSize + fieldGroupSize - 1) / fieldGroupSize) * fieldGroupSize;

  const auto outputGroupSize = plan.partialChansPerGroup;

  const auto unstridedHeight = ((kernelSizeY -1 + strideY - 1)/strideY + height)
                                * strideY;

  const auto unstridedWidth = activations[0].dim(2) + 2 * paddingX;

  const auto stridedHeight = unstridedHeight / strideY;

  const auto numViews = weightUpdateNumActivationReducedViews(kernelSizeY,
                                                              kernelSizeX,
                                                              strideY);
  const auto subViews = numViews / kernelSizeX;

  auto activationViews =
      graph.addTensor(dType, {subViews,
                              kernelSizeX,
                              activationsNumChanGroups,
                              stridedHeight,
                              0,
                              activationsChansPerGroup});

  for (unsigned b = 0; b != batchSize; ++b) {

    /* seed tensor to build tensor of subviews */
    auto subElemActivationViews =
        graph.addTensor(dType, {0,
                                kernelSizeX,
                                activationsNumChanGroups,
                                stridedHeight,
                                width,
                                activationsChansPerGroup});

    for (auto phase = 0U; phase < subViews; ++phase) {
      auto elemActivations = activations[b];
      auto elemActivationViews =
          graph.addTensor(dType, {0,
                                  activationsNumChanGroups,
                                  stridedHeight,
                                  width,
                                  activationsChansPerGroup});
      for (auto wx = 0U; wx != kernelSizeX; ++wx) {
        auto paddedActivations = pad(graph, elemActivations,
                                     {elemActivations.dim(0),
                                      unstridedHeight,
                                      unstridedWidth,
                                      elemActivations.dim(3)},
                                     {0, paddingY, paddingX, 0});

        auto usedActivations =
            paddedActivations.slice({0, phase, wx, 0},
                                    {activationsNumChanGroups,
                                     phase + (stridedHeight - 1) * strideY + 1,
                                     wx + (width - 1) * strideX + 1,
                                     activationsChansPerGroup});


        auto stridedActivations =
            usedActivations.subSample(strideY, 1).subSample(strideX, 2);

        assert(stridedActivations.dim(2) == width);

        elemActivationViews = append(elemActivationViews,
                                     stridedActivations);
      }
      subElemActivationViews = append(subElemActivationViews,
                                      elemActivationViews);
    }
    activationViews = concat(activationViews, subElemActivationViews, 4);
  }

  const auto actSubviewSize = kernelSizeX * activationsChans;
  const auto actViewSize = kernelSizeY * actSubviewSize;
  const auto paddedActViewSize =
     ((actViewSize + outputGroupSize - 1) / outputGroupSize) * outputGroupSize;

  auto copiedActivations =
      graph.addTensor(dType,
                      {subViews,
                       actSubviewSize,
                       stridedHeight * width * batchSize},
                       "activationsToCopy");

  auto copiedActivationsMapping =
      computeTensorMapping(graph, copiedActivations);
  applyTensorMapping(graph, copiedActivations,
                     copiedActivationsMapping);
  activationViews =
      activationViews.dimShuffle({0, 1, 2, 5, 3, 4})
                     .reshape({
                               subViews,
                               actSubviewSize,
                               stridedHeight * width * batchSize});

  prog.add(Copy(copiedActivations, activationViews));

  auto activationsTransposed = graph.addTensor(dType,
                                               {0,
                                                actSubviewSize,
                                                batchFieldSize});

  for (auto wy = 0U; wy < kernelSizeY; ++wy) {
    const auto yBegin = wy / subViews;
    const auto fieldPosBegin = yBegin * width * batchSize;
    const auto fieldPosEnd = fieldPosBegin + batchFieldSize;

    auto slice = copiedActivations[wy % subViews].slice(
                      {0, fieldPosBegin},
                      {actSubviewSize, fieldPosEnd});
    activationsTransposed = append(activationsTransposed, slice);
  }

  activationsTransposed = activationsTransposed.reshape({actViewSize,
                                                         batchFieldSize});

  auto matMulActivationsIn =
      pad(graph,
          activationsTransposed,
          {paddedActViewSize, batchPaddedFieldSize},
          {0, 0}).reshape({paddedActViewSize / outputGroupSize,
                           outputGroupSize,
                           batchPaddedFieldSize / fieldGroupSize,
                           fieldGroupSize}).dimShuffle({0, 2, 1, 3});

  mapWeights(
    matMulActivationsIn.reshape({matMulActivationsIn.dim(0),
                                 matMulActivationsIn.dim(1),
                                 1, 1,
                                 matMulActivationsIn.dim(2),
                                 matMulActivationsIn.dim(3)}),
    graph, plan, 1);

  return matMulActivationsIn;
}

static unsigned weightUpdateNumActivationFullViews(unsigned kernelSizeY,
                                                   unsigned kernelSizeX) {
  return kernelSizeX * kernelSizeY;
}


static Tensor weightUpdateActivationFullViews(Graph &graph,
                                              const Plan &plan,
                                              Tensor zDeltas,
                                              Tensor activations,
                                              unsigned kernelSizeY,
                                              unsigned kernelSizeX,
                                              unsigned strideY,
                                              unsigned strideX,
                                              unsigned paddingY,
                                              unsigned paddingX,
                                              Sequence &prog) {
  const auto dType = graph.getTensorElementType(activations);
  const auto batchSize = zDeltas.dim(0);
  const auto height = zDeltas.dim(2);
  const auto width = zDeltas.dim(3);
  const auto fieldSize = height * width;
  const auto activationsNumChanGroups = activations.dim(1);
  const auto activationsChansPerGroup = activations.dim(4);
  auto activationsChans = activationsNumChanGroups * activationsChansPerGroup;
  const auto fieldGroupSize = plan.inChansPerGroup;
  // Pad the field so the size is a multiple of the number of weights in the
  // convolutional unit.
  const auto batchFieldSize = fieldSize * batchSize;
  const auto batchPaddedFieldSize =
      ((batchFieldSize + fieldGroupSize - 1) / fieldGroupSize) * fieldGroupSize;
  const auto outputGroupSize = plan.partialChansPerGroup;

  const auto numViews = weightUpdateNumActivationFullViews(kernelSizeY,
                                                           kernelSizeX);

  // The activationViews tensor contains the view on the activations for
  // each element of the kernel.
  auto activationViews =
      graph.addTensor(dType, {numViews,
                              activationsNumChanGroups,
                              height,
                              0,
                              activationsChansPerGroup});
  for (unsigned b = 0; b != batchSize; ++b) {
    auto elemActivations = activations[b];
    auto elemActivationViews =
        graph.addTensor(dType, {0,
                                activationsNumChanGroups,
                                height,
                                width,
                                activationsChansPerGroup});
    for (unsigned wy = 0; wy != kernelSizeY; ++wy) {
      for (unsigned wx = 0; wx != kernelSizeX; ++wx) {
        auto paddedActivations = pad(graph, elemActivations,
                                     {elemActivations.dim(0),
                                      elemActivations.dim(1) + 2 * paddingY,
                                      elemActivations.dim(2) + 2 * paddingX,
                                      elemActivations.dim(3)},
                                     {0, paddingY, paddingX, 0});
        auto usedActivations =
            paddedActivations.slice({0, wy, wx, 0},
                                    {activationsNumChanGroups,
                                     wy + (height - 1) * strideY + 1,
                                     wx + (width - 1) * strideX + 1,
                                     activationsChansPerGroup});
        auto activationsStrided = usedActivations.subSample(strideY, 1)
                                                 .subSample(strideX, 2);
        assert(activationsStrided.dim(1) == height);
        assert(activationsStrided.dim(2) == width);

        // Pad the activations so the field size is a multiple of the number of
        // weights in the convolutional unit.
        elemActivationViews = append(elemActivationViews,
                                     activationsStrided);
      }
    }
    activationViews = concat(activationViews, elemActivationViews, 3);
  }

  /* flatten after concatentation */
  auto flattenedActivationViews =
      activationViews.reshape({numViews,
                               activationsNumChanGroups,
                               batchFieldSize,
                               activationsChansPerGroup});

  flattenedActivationViews = pad(graph,
                                 flattenedActivationViews,
                                 {numViews,
                                  activationsNumChanGroups,
                                  batchPaddedFieldSize,
                                  activationsChansPerGroup},
                                 {0, 0, 0,0});

  const auto actViewSize = kernelSizeY * kernelSizeX * activationsChans;
  const auto paddedActViewSize =
     ((actViewSize + outputGroupSize - 1) / outputGroupSize) * outputGroupSize;

  auto activationsTransposed =
      graph.addTensor(dType, {paddedActViewSize / outputGroupSize,
                              batchPaddedFieldSize / fieldGroupSize,
                              outputGroupSize,
                              fieldGroupSize},
                      "activationsTransposed");

  mapWeights(
    activationsTransposed.reshape({activationsTransposed.dim(0),
                                   activationsTransposed.dim(1),
                                   1, 1,
                                   activationsTransposed.dim(2),
                                   activationsTransposed.dim(3)}),
    graph, plan, 1);

  flattenedActivationViews =
      flattenedActivationViews.dimShuffle({0, 1, 3, 2})
                              .reshape({actViewSize,
                                        batchPaddedFieldSize / fieldGroupSize,
                                        fieldGroupSize});
  flattenedActivationViews = pad(graph, flattenedActivationViews,
                                 {paddedActViewSize,
                                  batchPaddedFieldSize / fieldGroupSize,
                                  fieldGroupSize},
                                 {0, 0, 0});

  auto activationsTransposedIn =
     flattenedActivationViews.reshape({paddedActViewSize / outputGroupSize,
                                       outputGroupSize,
                                       batchPaddedFieldSize / fieldGroupSize,
                                       fieldGroupSize})
                             .dimShuffle({0, 2, 1, 3});


  prog.add(Copy(activationsTransposed, activationsTransposedIn));
  return activationsTransposed;
}

static void
createWeightGradAopVertex(Graph &graph, unsigned tile,
                          unsigned outXBegin, unsigned outXEnd,
                          unsigned outYBegin, unsigned outYEnd,
                          const WeightGradAopTask *taskBegin,
                          const WeightGradAopTask *taskEnd,
                          unsigned kernelSizeY, unsigned kernelSizeX,
                          unsigned strideY, unsigned strideX,
                          unsigned paddingY, unsigned paddingX,
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
  auto v = graph.addVertex(cs, templateVertex("popnn::ConvWeightGradAop",
                                              dType, partialsType));
  graph.setTileMapping(v, tile);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
  graph.setFieldSize(v["weightDeltas"], taskEnd - taskBegin);
  graph.setFieldSize(v["weightReuseCount"], taskEnd - taskBegin);
  unsigned deltasIndex = 0;
  for (auto it = taskBegin; it != taskEnd; ++it) {
    const auto &task = *it;
    const auto weightIndex = it - taskBegin;
    const auto kernelX = task.kernelX;
    const auto kernelY = task.kernelY;
    const auto izg = task.inZGroup;
    const auto ozg = task.outZGroup;
    graph.connect(v["weightDeltas"][weightIndex],
                  weightDeltas[ozg][izg][kernelY][kernelX].flatten());
    unsigned deltaXBegin, deltaXEnd;
    std::tie(deltaXBegin, deltaXEnd) =
        getOutputRange({outXBegin, outXEnd}, strideX, kernelSizeX, paddingX,
                       inDimX, kernelX, false);
    const auto actXBegin = deltaXBegin * strideX + kernelX - paddingX;
    const auto actXEnd = (deltaXEnd - 1) * strideX + kernelX - paddingX + 1;
    unsigned deltaYBegin, deltaYEnd;
    std::tie(deltaYBegin, deltaYEnd) =
        getOutputRange({outYBegin, outYEnd}, strideY, kernelSizeY, paddingY,
                       inDimY, kernelY, false);
    graph.setInitialValue(
          v["weightReuseCount"][weightIndex],
        deltaYEnd - deltaYBegin);
    for (unsigned deltaY = deltaYBegin; deltaY != deltaYEnd;
         ++deltaY, ++deltasIndex) {
      const auto actY = deltaY * strideY + kernelY - paddingY;
      graph.connect(v["acts"][deltasIndex],
                    acts[izg][actY].slice(actXBegin, actXEnd).flatten());
      graph.connect(v["deltas"][deltasIndex],
                    deltas[ozg][deltaY]
                        .slice(deltaXBegin, deltaXEnd).flatten());
    }
  }
  graph.setFieldSize(v["acts"], deltasIndex);
  graph.setFieldSize(v["deltas"], deltasIndex);
}

static void
calcPartialWeightGradsAop(Graph &graph,
                          unsigned tile,
                          unsigned outXBegin, unsigned outXEnd,
                          unsigned outYBegin, unsigned outYEnd,
                          unsigned outZGroupBegin, unsigned outZGroupEnd,
                          unsigned inZGroupBegin, unsigned inZGroupEnd,
                          unsigned kernelSizeY, unsigned kernelSizeX,
                          unsigned strideY, unsigned strideX,
                          unsigned paddingY, unsigned paddingX,
                          ComputeSet cs,
                          Tensor acts, Tensor deltas, Tensor weightDeltas) {
  std::vector<WeightGradAopTask> tasks;
  const auto inDimY = acts.dim(1);
  const auto inDimX = acts.dim(2);
  for (unsigned kernelY = 0; kernelY != kernelSizeY; ++kernelY) {
    for (unsigned kernelX = 0; kernelX != kernelSizeX; ++kernelX) {
      auto xRange =
          getOutputRange({outXBegin, outXEnd}, strideX, kernelSizeX, paddingX,
                         inDimX, kernelX, false);
      if (xRange.first == xRange.second)
        continue;
      auto yRange =
          getOutputRange({outYBegin, outYEnd}, strideY, kernelSizeY, paddingY,
                         inDimY, kernelY, false);
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
                              strideY, strideX, paddingY, paddingX, cs, acts,
                              deltas, weightDeltas);
  }
}

static void
addWeightDeltaPartialRegions(std::vector<std::pair<unsigned,unsigned>> &regions,
                             const std::vector<std::size_t> &partialDims,
                             unsigned b, unsigned tileY, unsigned tileX,
                             unsigned outZGroupBegin, unsigned outZGroupEnd,
                             unsigned inZGroupBegin, unsigned inZGroupEnd) {
  for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
    const auto regionBegin =
        partialDims[8] * partialDims[7] * partialDims[6] * partialDims[5] *
        (inZGroupBegin + partialDims[4] *
         (ozg + partialDims[3] *
          (tileX + partialDims[2] *
           (tileY + partialDims[1] * b))));
    const auto regionEnd =
        regionBegin +
        partialDims[8] * partialDims[7] * partialDims[6] *partialDims[5] *
        (inZGroupEnd - inZGroupBegin);
    regions.emplace_back(regionBegin, regionEnd);
  }
}

std::vector<std::vector<std::pair<unsigned, unsigned>>>
convertLinearMappingToRegionMapping(const std::vector<unsigned> &mapping) {
  assert(!mapping.empty());
  const auto numTiles = mapping.size() - 1;
  std::vector<std::vector<std::pair<unsigned, unsigned>>>
      regionMapping(numTiles);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    if (mapping[tile] == mapping[tile + 1])
      continue;
    regionMapping[tile].emplace_back(mapping[tile], mapping[tile + 1]);
  }
  return regionMapping;
}

Program
convolutionWeightUpdateAop(Graph &graph,
                           const Plan &plan,
                           const Plan &fwdPlan,
                           Tensor zDeltas, Tensor weights, Tensor biases,
                           Tensor activations,
                           unsigned strideY, unsigned strideX,
                           unsigned paddingY, unsigned paddingX,
                           float learningRate,
                           const std::string &debugPrefix = "") {
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
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  const auto batchSize = activations.dim(0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto layerName = debugPrefix +
      "/Conv" + std::to_string(kernelSizeX) + "x" + std::to_string(kernelSizeY)
             + "WeightUpdateAop";
  const auto outNumChanGroups = weights.dim(0);
  const auto inNumChanGroups = weights.dim(1);
  const auto outChansPerGroup = weights.dim(4);
  const auto inChansPerGroup = weights.dim(5);
  assert(activations.dim(1) == inNumChanGroups);
  assert(activations.dim(4) == inChansPerGroup);
  assert(plan.inChansPerGroup == inChansPerGroup);
  assert(plan.partialChansPerGroup == outChansPerGroup);

  auto prog = Sequence();
  auto outDimY = zDeltas.dim(2), outDimX = zDeltas.dim(3);
  const auto isMultiIPU = deviceInfo.numIPUs > 1;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto numTiles = deviceInfo.getNumTiles();

  Tensor partials = graph.addTensor(partialsType, {batchSize,
                                                   tilesPerY, tilesPerX,
                                                   outNumChanGroups,
                                                   inNumChanGroups,
                                                   kernelSizeY, kernelSizeX,
                                                   outChansPerGroup,
                                                   inChansPerGroup},
                                    "partialWeightGrads");
  // TODO consider regrouping the weight deltas instead.
  Tensor regroupedDeltas;
  if (zDeltas.dim(1) != outNumChanGroups) {
    regroupedDeltas = graph.addTensor(dType, {batchSize, outNumChanGroups,
                                              outDimY, outDimX,
                                              outChansPerGroup},
                                              "zDeltas'");
    for (unsigned b = 0; b < batchSize; ++b) {
      auto regroupedDeltaMapping =
          computeActivationsMapping(graph, regroupedDeltas[b], b, batchSize);
      applyTensorMapping(graph, regroupedDeltas[b], regroupedDeltaMapping);
    }
    prog.add(Copy(regroupedDeltas, regroup(zDeltas, outChansPerGroup)));
  } else {
    regroupedDeltas = zDeltas;
  }
  std::vector<std::vector<std::pair<unsigned,unsigned>>>
      partialsMapping(numTiles);
  ComputeSet weightGradCS = graph.createComputeSet(layerName + "/WeightGrad");
  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * inNumChanGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * inNumChanGroups) / tilesPerInZGroup;
      for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
        const auto outZGroupBegin = (ozg * outNumChanGroups) / tilesPerZ;
        const auto outZGroupEnd =
            ((ozg + 1) * outNumChanGroups) / tilesPerZ;
        for (unsigned oy = 0; oy != tilesPerY; ++oy) {
          const auto outYBegin = (oy * outDimY) / tilesPerY;
          const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
          for (unsigned ox = 0; ox != tilesPerX; ++ox) {
            const auto outXBegin = (ox * outDimX) / tilesPerX;
            const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
            const auto tile = linearizeTileIndices(b, batchSize,
                                                   numTiles,
                                                   izg, ox, oy, ozg,
                                                   plan,
                                                   isMultiIPU);
            addWeightDeltaPartialRegions(partialsMapping[tile], partials.dims(),
                                         b, oy, ox,
                                         outZGroupBegin, outZGroupEnd,
                                         inZGroupBegin, inZGroupEnd);
            calcPartialWeightGradsAop(graph, tile,
                                      outXBegin, outXEnd, outYBegin, outYEnd,
                                      outZGroupBegin, outZGroupEnd,
                                      inZGroupBegin, inZGroupEnd,
                                      kernelSizeY, kernelSizeX,
                                      strideY, strideX, paddingY, paddingX,
                                      weightGradCS, activations[b],
                                      regroupedDeltas[b],
                                      partials[b][oy][ox]);
          }
        }
      }
    }
  }
  mergeAdjacentRegions(partialsMapping);
  applyTensorMapping(graph, partials, partialsMapping);
  ComputeSet zeroCS = graph.createComputeSet(layerName + "/Zero");
  zero(graph, partials, partialsMapping, zeroCS);
  prog.add(Execute(zeroCS));
  prog.add(Execute(weightGradCS));

  Tensor weightDeltas;
  auto numPartials = batchSize * tilesPerY * tilesPerX;
  if (numPartials == 1 && partialsType == dType) {
    weightDeltas = partials[0][0][0];
  } else {
    std::vector<std::vector<std::pair<unsigned,unsigned>>> weightDeltaMapping;
    auto reduceCS = graph.createComputeSet(layerName + "/Reduce");
    weightDeltas = graph.addTensor(dType, weights.dims(),
                                   layerName + "/WeightDeltas");
    if (numPartials == 1) {
      weightDeltaMapping = calculateWeightMapping(weights, graph, plan,
                                                  batchSize);
    } else {
      /** The reduction of weights is not performed where the weights are
       *  stored in the weight tensor. This causes some output exchange
       *  after the reduction but allows balancing of compute.
       */
      weightDeltaMapping =
          convertLinearMappingToRegionMapping(
            computeTensorMapping(graph, weightDeltas)
          );
    }
    applyTensorMapping(graph, weightDeltas, weightDeltaMapping);
    auto flatPartialsDims = weightDeltas.dims();
    flatPartialsDims.insert(flatPartialsDims.begin(), numPartials);
    auto flatPartials = partials.reshape(flatPartialsDims);
    ::reduce(graph, flatPartials, weightDeltas, weightDeltaMapping, reduceCS);
    prog.add(Execute(reduceCS));
  }

  // Add the weight deltas to the weights.
  auto addCS = graph.createComputeSet(layerName + "/UpdateWeights");
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto weightsFlattened = weights.flatten();
  auto weightDeltasFlattened = weightDeltas.flatten();
  iterateWeightMapping(weights, graph, fwdPlan, 1,
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
        const auto v = graph.addVertex(addCS,
                                       templateVertex("popnn::ConvWeightUpdate",
                                                      dType));
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

  auto updateBiasProg =
      convolutionBiasUpdate(graph, zDeltas, biases, learningRate, addCS,
                            layerName);
  prog.add(Execute(addCS));
  prog.add(updateBiasProg);
  return prog;
}

Program
convolutionWeightUpdateAmp(Graph &graph,
                           const Plan &plan,
                           const Plan &fwdPlan,
                           Tensor zDeltas, Tensor weights, Tensor biases,
                           Tensor activations,
                           unsigned strideY, unsigned strideX,
                           unsigned paddingY, unsigned paddingX,
                           float learningRate,
                           const std::string &debugPrefix = "") {
  const auto kernelSizeY = weights.dim(2);
  const auto kernelSizeX = weights.dim(3);
  const auto layerName =
      debugPrefix
              + "/Conv"
              + std::to_string(kernelSizeX) + "x" + std::to_string(kernelSizeY)
              + "_stride"
              + std::to_string(strideX) + "x" + std::to_string(strideY)
              + "/WeightUpdate";
  // We can calculate weight deltas using the AMP instruction where we
  // accumulate over the field in contrast to forward pass where we accumulate
  // over input channels. Let w be the number of weights per convolutional unit.
  // The activations for each channel are flattened, grouped into groups of w
  // and loaded into convolutional units. Different convolutional units are
  // loaded with activations for different input channels. The activations
  // loaded into the convolutional unit are convolved with the deltas that
  // correspond to the activations across the different output channels.
  //
  // The following table compares the role of the field, the input channels and
  // the output channels in the forward and weight update passes.
  // ----------------------------------------------------------------------
  // | Role                            | Forward pass    | Weight update   |
  // |=================================|=================|=================|
  // | Accumulate over...              | Input channels  | Field           |
  // | Iterate over...                 | Field           | Output channels |
  // | Use conv units for different... | Output channels | Input channels  |
  // -----------------------------------------------------------------------
  const auto dType = graph.getTensorElementType(activations);
  const auto batchSize = zDeltas.dim(0);
  auto deltasNumChanGroups = zDeltas.dim(1);
  auto height = zDeltas.dim(2);
  auto width = zDeltas.dim(3);
  auto deltasChansPerGroup = zDeltas.dim(4);
  auto deltasChans = deltasNumChanGroups * deltasChansPerGroup;
  auto activationsNumChanGroups = activations.dim(1);
  auto activationsChansPerGroup = activations.dim(4);
  auto activationsChans = activationsNumChanGroups * activationsChansPerGroup;
  const auto fieldGroupSize = plan.inChansPerGroup;

  // Pad the field so the size is a multiple of the number of weights in the
  // convolutional unit.
  const auto batchFieldSize = height * width * batchSize;
  const auto batchPaddedFieldSize =
      ((batchFieldSize + fieldGroupSize - 1) / fieldGroupSize) * fieldGroupSize;
  const auto outputGroupSize = plan.partialChansPerGroup;

  auto prog = Sequence();

  /* Heuristic to keep reduced or full views.
   *
   * Reduced views require less copying of activation tensor than full views.
   * Views are expanded to a full set after copying and then padded to
   * make an integer multiple of the output and input group size to fit the
   * convolution unit constraints. Padding results in an increase in the
   * message memory and in copy pointers. This is expected to reduce with
   * enhancements to poplar.
   */
  const auto numFullViews = weightUpdateNumActivationFullViews(kernelSizeY,
                                                               kernelSizeX);

  const auto numReducedViews = weightUpdateNumActivationReducedViews(
                                    kernelSizeY,
                                    kernelSizeX,
                                    strideY);

  Tensor matMulActivationsIn = numFullViews > 2 * numReducedViews ?
      weightUpdateActivationReducedViews(graph, plan,
                                         zDeltas,
                                         activations,
                                         kernelSizeY, kernelSizeX,
                                         strideY, strideX,
                                         paddingY, paddingX,
                                         prog) :
      weightUpdateActivationFullViews(graph, plan,
                                      zDeltas,
                                      activations,
                                      kernelSizeY, kernelSizeX,
                                      strideY, strideX,
                                      paddingY, paddingX,
                                      prog);


  const auto actViewSize = kernelSizeY * kernelSizeX * activationsChans;
  const auto paddedActViewSize =
     ((actViewSize + outputGroupSize - 1) / outputGroupSize) * outputGroupSize;

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();

  // Pad the field so the size is a multiple of the number of weights in the
  // convolutional unit.
  auto batchZDeltas = graph.addTensor(dType,
                                     {deltasNumChanGroups,
                                      height,
                                      0,
                                      deltasChansPerGroup});
  for (unsigned b = 0; b != batchSize; ++b) {
    auto zDeltasFlattened = zDeltas[b].reshape({deltasNumChanGroups, height,
                                                width,
                                                deltasChansPerGroup});

    batchZDeltas = concat(batchZDeltas, zDeltasFlattened, 2);
  }

  batchZDeltas = batchZDeltas.reshape({deltasNumChanGroups,
                                       batchFieldSize,
                                       deltasChansPerGroup});

  batchZDeltas = pad(graph, batchZDeltas,
                     {deltasNumChanGroups,
                      batchPaddedFieldSize,
                      deltasChansPerGroup},
                     {0, 0, 0});

  // Transpose the deltas.
  auto zDeltasTransposed =
      graph.addTensor(dType,
                      {batchPaddedFieldSize / fieldGroupSize,
                       deltasChans,
                       fieldGroupSize},
                      "zDeltasTransposed");
  mapActivations(graph, zDeltasTransposed.reshape({1, zDeltasTransposed.dim(0),
                                                   1, zDeltasTransposed.dim(1),
                                                   zDeltasTransposed.dim(2)}));
  prog.add(Copy(zDeltasTransposed,
                batchZDeltas.reshape({deltasNumChanGroups,
                                       batchPaddedFieldSize / fieldGroupSize,
                                       fieldGroupSize,
                                       deltasChansPerGroup})
                             .dimShuffle({1, 0, 3, 2})));

  const auto weightDeltasType = dType;

  auto weightDeltasTransposed =
      graph.addTensor(
          weightDeltasType,
          {paddedActViewSize / outputGroupSize,
           deltasChans,
           outputGroupSize},
          "weightDeltas");
  auto weightDeltasTransposedMapping =
    computeActivationsMapping(
      graph,
      weightDeltasTransposed.reshape({paddedActViewSize / outputGroupSize,
                                       1, deltasChans, outputGroupSize}),
      0, 1);
  applyTensorMapping(graph, weightDeltasTransposed,
                     weightDeltasTransposedMapping);
  // Perform the matrix multiplication.
  prog.add(matrixMultiplyByConvInstruction(
             graph, plan, matMulActivationsIn, zDeltasTransposed,
             weightDeltasTransposed, weightDeltasTransposedMapping,
             layerName
           )
  );
  // Transpose the weight deltas.
  auto fwdInChansPerGroup = fwdPlan.inChansPerGroup;
  auto fwdPartialChansPerGroup = fwdPlan.partialChansPerGroup;
  auto weightDeltas =
      graph.addTensor(weightDeltasType,
                      {deltasChans / fwdPartialChansPerGroup,
                       activationsChans / fwdInChansPerGroup,
                       kernelSizeY,
                       kernelSizeX,
                       fwdPartialChansPerGroup,
                       fwdInChansPerGroup},
                      "weightDeltas");
  auto weightDeltasMapping = computeTensorMapping(graph, weightDeltas);
  applyTensorMapping(graph, weightDeltas, weightDeltasMapping);
  prog.add(Copy(weightDeltas,
                weightDeltasTransposed
                  .dimShuffle({1, 0, 2})
                  .reshape({deltasChans, paddedActViewSize})
                  .slice(0, actViewSize, 1)
                  .reshape({deltasChans / fwdPartialChansPerGroup,
                            fwdPartialChansPerGroup,
                            kernelSizeY, kernelSizeX,
                            activationsChans / fwdInChansPerGroup,
                            fwdInChansPerGroup})
                  .dimShuffle({0, 4, 2, 3, 1, 5})));
  // Add the weight deltas to the weights.
  auto addCS = graph.createComputeSet(layerName + "/UpdateWeights");
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto weightsFlattened = weights.flatten();
  auto weightDeltasFlattened = weightDeltas.flatten();
  iterateWeightMapping(weights, graph, fwdPlan, 1,
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
        const auto v = graph.addVertex(addCS,
                                       templateVertex("popnn::ConvWeightUpdate",
                                                      dType));
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
  auto updateBiasProg =
      convolutionBiasUpdate(graph, zDeltas, biases, learningRate, addCS,
                            layerName);
  prog.add(Execute(addCS));
  prog.add(updateBiasProg);
  return prog;
}

Program
convolutionWeightUpdate(Graph &graph,
                        const Plan &plan, const Plan &fwdPlan,
                        Tensor zDeltas, Tensor weights, Tensor biases,
                        Tensor activations,
                        unsigned strideY, unsigned strideX,
                        unsigned paddingY, unsigned paddingX,
                        float learningRate, const std::string &debugPrefix) {
  if (plan.useConvolutionInstructions) {
    return convolutionWeightUpdateAmp(graph, plan, fwdPlan, zDeltas,
                                      weights, biases, activations,
                                      strideY, strideX,
                                      paddingY, paddingX, learningRate,
                                      debugPrefix);
  }
  return convolutionWeightUpdateAop(graph, plan, fwdPlan, zDeltas, weights,
                                    biases, activations, strideY, strideX,
                                    paddingY, paddingX, learningRate,
                                    debugPrefix);
}

} // namespace conv
