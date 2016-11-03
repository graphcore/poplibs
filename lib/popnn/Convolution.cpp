#include "popnn/Convolution.hpp"
#include <limits>
#include <cassert>
#include "ConvUtil.hpp"
#include "Pad.hpp"
#include "popnn/ActivationMapping.hpp"
#include "Regroup.hpp"
#include "VertexTemplates.hpp"
#include "gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "popnn/exceptions.hpp"
#include "Cast.hpp"

using namespace poplar;
using namespace poplar::program;

namespace conv {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX, unsigned strideY,
             unsigned strideX, unsigned paddingY,
             unsigned paddingX) {
  unsigned outDimX = (inDimX + (paddingX * 2) - kernelSizeX) / strideX + 1;
  unsigned outDimY = (inDimY + (paddingY * 2) - kernelSizeY) / strideY + 1;
  return {outDimY, outDimX};
}

poplar::Tensor
createWeights(poplar::Graph &graph, std::string dType,
             unsigned inNumChans,
             unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned outNumChans,
             const ConvPlan &plan) {
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto inChansPerGroup = plan.fwdPartition.inChansPerGroup;
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
                     const Partition &partition,
                     bool isMultiIPU) {
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto batchElemsPerTile = (batchSize + numTiles - 1) / numTiles;
  const auto numBatchGroups =
      (batchSize + batchElemsPerTile - 1) / batchElemsPerTile;
  const auto tilesPerBatchGroup =
      numTiles / numBatchGroups;
  const auto beginTile = batchNum / batchElemsPerTile * tilesPerBatchGroup;
  // If this is a multi IPU system then choose an order that avoids splitting
  // partial sums over IPUs
  if (isMultiIPU)
    return beginTile +
      (izg + tilesPerInZGroup *
        (ox + tilesPerX *
          (oy + tilesPerY * ozg)));
  // Use ozg as the innermost dimension to increase the chance that
  // tiles in a supertile both read the same activations. This reduces
  // exchange time when supertile send / receive is used.
  return beginTile +
           (ozg + tilesPerZ *
             (ox + tilesPerX *
               (oy + tilesPerY * izg)));
}

static std::pair<unsigned,unsigned>
getOutZGroupRange(unsigned ozgIndex, unsigned partialNumChanGroups,
                  const Partition &partition) {
  const auto tilesPerZAxis = partition.tilesPerZAxis;
  const auto maxZGroupsPerTile = (partialNumChanGroups + tilesPerZAxis - 1) /
                                 tilesPerZAxis;
  const auto outZBegin =
      std::min(ozgIndex * maxZGroupsPerTile, partialNumChanGroups);
  const auto outZEnd =
      std::min((ozgIndex + 1) * maxZGroupsPerTile, partialNumChanGroups);
  return {outZBegin, outZEnd};
}

template <typename Builder>
static void
iterateWeightMapping(Tensor w,
                     const poplar::Graph &graph,
                     const Partition &partition,
                     unsigned batchSize,
                     Builder &&builder) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto partialChansPerGroup = partition.partialChansPerGroup;
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
      const auto tileWeights = w.reshape({numGroups, groupSize})
                                .slice(groupBegin, groupEnd);
      builder(tileWeights, tile);
    }
    return;
  }

  const auto isMultiIPU = deviceInfo.numIPUs > 1;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto inNumChans = w.dim(1) * w.dim(5);
  const auto outNumChans = w.dim(0) * w.dim(4);
  const auto kernelSizeY = w.dim(2);
  const auto kernelSizeX = w.dim(3);
  const auto numInZGroups = inNumChans / inChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;


  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
    const auto numInZGroups = inZGroupEnd - inZGroupBegin;
    for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
      unsigned outZGroupBegin, outZGroupEnd;
      std::tie(outZGroupBegin, outZGroupEnd) =
          getOutZGroupRange(ozg, partialNumChanGroups, partition);
      const auto numOutZGroups = outZGroupEnd - outZGroupBegin;
      // Group weights that are accessed contiguously by tiles within this
      // loop body.
      Tensor sharedWeights;
      if (partition.useConvolutionInstructions) {
        if (kernelSizeY == 1 && kernelSizeX == 1) {
          sharedWeights =
              w.slice(
          {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
          {outZGroupEnd, inZGroupEnd, kernelSizeY, kernelSizeX,
           partialChansPerGroup, inChansPerGroup}
                ).reshape({numOutZGroups,
                           numInZGroups * partialChansPerGroup *
                           inChansPerGroup});
        } else {
          sharedWeights =
              w.slice(
          {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
          {outZGroupEnd, inZGroupEnd, kernelSizeY, kernelSizeX,
           partialChansPerGroup, inChansPerGroup}
                ).reshape({numOutZGroups * numInZGroups * kernelSizeY *
                           kernelSizeX,
                           partialChansPerGroup * inChansPerGroup});
        }
      } else {
        sharedWeights =
            w.slice(
        {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
        {outZGroupEnd, inZGroupEnd, kernelSizeY, kernelSizeX,
         1, inChansPerGroup}
              ).reshape({numInZGroups * numOutZGroups * kernelSizeY,
                         kernelSizeX * inChansPerGroup});
      }
      const auto numSharedWeightGroups = sharedWeights.dim(0);
      // Spread groups of weights equally across the tiles that read them.
      for (unsigned oy = 0; oy != tilesPerY; ++oy) {
        for (unsigned ox = 0; ox != tilesPerX; ++ox) {
          const auto iw = ox + tilesPerX * oy;
          const auto sharedWeightGroupBegin =
              (iw * numSharedWeightGroups) / (tilesPerY * tilesPerX);
          const auto sharedWeightGroupEnd =
              ((iw + 1) * numSharedWeightGroups) / (tilesPerY * tilesPerX);
          if (sharedWeightGroupBegin == sharedWeightGroupEnd)
            continue;
          const auto tileWeights =
              sharedWeights.slice(sharedWeightGroupBegin,
                                  sharedWeightGroupEnd);
          const auto tile = linearizeTileIndices(0, batchSize, numTiles,
                                                 izg, ox, oy, ozg,
                                                 partition, isMultiIPU);
          builder(tileWeights, tile);
        }
      }
    }
  }
}


static void
mapWeights(Tensor w, Graph &graph, const Partition &partition,
           unsigned batchSize) {
  iterateWeightMapping(w, graph, partition, batchSize,
    [&](Tensor tileWeights, unsigned tile) {
    graph.setTileMapping(tileWeights, tile);
  });
}

void mapWeights(Tensor w, Graph &graph, const ConvPlan &plan,
                unsigned batchSize) {
  mapWeights(w, graph, plan.fwdPartition, batchSize);
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
                              unsigned kernelSizeY,
                              unsigned kernelSizeX,
                              unsigned strideY,
                              unsigned strideX,
                              unsigned paddingY,
                              unsigned paddingX,
                              ComputeSet fwdCS,
                              const Tensor &in, const Tensor &weights,
                              const Tensor &out, bool forward) {
  assert(kernelSizeY == 1 && kernelSizeX == 1);
  assert(forward || (strideY == 1 && strideX == 1));
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = static_cast<unsigned>(in.dim(3));
  const auto outChansPerGroup = static_cast<unsigned>(out.dim(3));
  const auto dType = graph.getTensorElementType(in);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex =
      deviceInfo.sharedConvWeights ? deviceInfo.numWorkerContexts : 1;
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(dType == "float");
  assert(weightsPerConvUnit % inChansPerGroup == 0);
  const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
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
                     paddingY, inDimY, forward);
  std::tie(inXBegin, inXEnd) =
      getInputRange({outXBegin, outXEnd}, strideX,
                     kernelSizeX, paddingX, inDimX, forward);

  // Add the vertex.
  const char *baseClass =
      deviceInfo.sharedConvWeights ? "poplar::SupervisorVertex" :
                                     "poplar::Vertex";
  Tensor w =
      weights[ozg].slice(
  {inZGroupBegin, 0, 0, 0, 0},
  {inZGroupEnd, 1, 1, outChansPerGroup, inChansPerGroup}
        ).flatten();
  auto v = graph.addVertex(
        fwdCS,
        templateVertex("popnn::ConvPartial1x1Out", baseClass, dType,
                       partialType),
  {{"weights", w}}
        );
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
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
                          paddingY, inDimY, 0, forward);
        assert(workerInY != ~0U);
        unsigned workerInXBegin, workerInXEnd;
        std::tie(workerInXBegin, workerInXEnd) =
            getInputRange({workerOutXBegin, workerOutXEnd}, strideX,
                          kernelSizeX, paddingX, inDimX, forward);
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
                                unsigned kernelSizeY,
                                unsigned kernelSizeX,
                                unsigned strideY, unsigned strideX,
                                unsigned paddingY, unsigned paddingX,
                                ComputeSet fwdCS,
                                const Tensor &in,
                                const Tensor &weights,
                                const Tensor &out,
                                const Tensor &zeros,
                                bool forward) {
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = static_cast<unsigned>(in.dim(3));
  const auto outChansPerGroup = static_cast<unsigned>(out.dim(3));
  const auto dType = graph.getTensorElementType(in);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex =
      deviceInfo.sharedConvWeights ? deviceInfo.numWorkerContexts : 1;
  const char *baseClass =
      deviceInfo.sharedConvWeights ? "poplar::SupervisorVertex" :
                                     "poplar::Vertex";
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(dType == "float");
  assert(weightsPerConvUnit % inChansPerGroup == 0);
  const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
  const auto partialType = graph.getTensorElementType(out);
  // Add the vertex.
  auto v =
      graph.addVertex(fwdCS,
                      templateVertex("popnn::ConvPartialnx1InOut", baseClass,
                                     dType, partialType,
                                     forward ? "true" : "false"));
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setTileMapping(v, tile);
  unsigned numWeights = 0;
  unsigned numConvolutions = 0;
  for (unsigned wyBegin = 0; wyBegin < kernelSizeY;
       wyBegin += convUnitWeightHeight) {
    const auto wyEnd = std::min(kernelSizeY, wyBegin + convUnitWeightHeight);
    unsigned convOutYBegin, convOutYEnd;
    std::tie(convOutYBegin, convOutYEnd) =
        getOutputRange({outYBegin, outYEnd}, strideY, kernelSizeY,
                       paddingY, inDimY, {wyBegin, wyEnd}, forward);
    const auto convOutHeight = convOutYEnd - convOutYBegin;
    if (convOutHeight == 0)
      continue;
    for (unsigned wx = 0; wx != kernelSizeX; ++wx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRange({outXBegin, outXEnd}, strideX, kernelSizeX,
                         paddingX, inDimX, wx, forward);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;

      // In the backwards pass, if we are handling one row of the kernel at
      // a time, the partitioning of work across the workers can be aware of
      // the stride and only allocate work on the rows that get affected.
      unsigned outputStride =
          (!forward && convUnitWeightHeight == 1) ? strideY : 1;
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
                                strideX, kernelSizeX, paddingX, inDimX, wx,
                               forward);
            const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
            unsigned workerInXBegin, workerInXEnd;
            std::tie(workerInXBegin, workerInXEnd) =
                getInputRange({workerOutXBegin, workerOutXEnd}, strideX,
                              kernelSizeX, paddingX, inDimX, wx, forward);
            const auto workerInWidth = workerInXEnd - workerInXBegin;
            for (unsigned wy = wyBegin; wy != wyBegin + convUnitWeightHeight;
                 ++wy) {
              const auto workerInY =
                  getInputIndex(workerOutY, strideY, kernelSizeY,
                                paddingY, inDimY, wy, forward);
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
            // Note the output tensor is mapped in zeroPartialSums.
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
                                  const Partition &partition,
                                  unsigned tile,
                                  unsigned outXBegin, unsigned outXEnd,
                                  unsigned outYBegin, unsigned outYEnd,
                                  unsigned z,
                                  unsigned inZGroupBegin, unsigned inZGroupEnd,
                                  unsigned kernelSizeY, unsigned kernelSizeX,
                                  unsigned strideY, unsigned strideX,
                                  unsigned paddingY, unsigned paddingX,
                                  std::string dType,
                                  ComputeSet fwdCS,
                                  const Tensor &in,
                                  const Tensor &weights,
                                  const Tensor &out) {
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto dataPathWidth = graph.getDevice().getDeviceInfo().dataPathWidth;
  const auto inZGroups = inZGroupEnd - inZGroupBegin;
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto partialType = partition.getPartialType();
  if (outChansPerGroup != 1)
    assert(!"outChansPerGroup must be 1");
  assert(outYEnd - outYBegin == 1);
  const auto y = outYBegin;
  unsigned inYBegin, inYEnd, inXBegin, inXEnd;
  std::tie(inYBegin, inYEnd) =
      getInputRange(y, strideY, kernelSizeY, paddingY, inDimY, true);
  std::tie(inXBegin, inXEnd) =
      getInputRange({outXBegin, outXEnd}, strideX, kernelSizeX,
                    paddingX, inDimX, true);
  // Window into previous layer.
  const auto inWidth = inXEnd - inXBegin;
  const auto inHeight = inYEnd - inYBegin;
  // Weights that match the window.
  unsigned weightYBegin, weightYEnd;
  std::tie(weightYBegin, weightYEnd) =
      getKernelRange(y, strideY, kernelSizeY, paddingY, inDimY, true);
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
  unsigned vPadding = inXBegin < paddingX ? paddingX - inXBegin : 0;
  graph.setInitialValue(v["padding"], vPadding);
  // Map the vertex and output.
  graph.setTileMapping(v, tile);
  graph.setTileMapping(outWindow, tile);
}

static void
zeroPartialSums(Graph &graph,
                const Partition &partition,
                unsigned outXBegin, unsigned outXEnd,
                unsigned outYBegin, unsigned outYEnd,
                unsigned tileOutZGroupBegin, unsigned tileOutZGroupEnd,
                unsigned tile,
                ComputeSet zeroCS,
                Tensor &out) {
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto outZGroups = tileOutZGroupEnd - tileOutZGroupBegin;
  const auto outHeight = outYEnd - outYBegin;
  const auto outWidth = outXEnd - outXBegin;
  const auto partialType = partition.getPartialType();

  Tensor toZero =
      out.slice({tileOutZGroupBegin, outYBegin, outXBegin, 0},
                {tileOutZGroupEnd, outYEnd, outXEnd, outChansPerGroup})
         .reshape({outZGroups * outHeight,
                   outWidth * outChansPerGroup});
  graph.setTileMapping(toZero, tile);
  const auto workersPerTile = deviceInfo.numWorkerContexts;
  const auto tileOutRows = toZero.dim(0);
  if (tileOutRows == 0)
    return;
  const auto maxRowsPerWorker =
      (tileOutRows + workersPerTile - 1) / workersPerTile;
  // Choose the number of vertices such that each vertices is reponsible for
  // at most maxRowsPerWorker groups.
  const auto verticesToCreate =
      (tileOutRows + maxRowsPerWorker - 1) / maxRowsPerWorker;
  for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
    const auto beginRow = (vertex * tileOutRows) / verticesToCreate;
    const auto endRow = ((vertex + 1) * tileOutRows) / verticesToCreate;
    if (beginRow == endRow)
      continue;
    auto zv = graph.addVertex(
      zeroCS, templateVertex("popnn::Zero2D", partialType),
      {{"out", toZero.slice(beginRow, endRow)}}
    );
    graph.setInitialValue(zv["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(zv, tile);
  }
}

static void
calcPartialConvOutput(Graph &graph,
                      const Partition &partition,
                      std::string dType,
                      unsigned tile,
                      unsigned outXBegin, unsigned outXEnd,
                      unsigned outYBegin, unsigned outYEnd,
                      unsigned outZGroupBegin, unsigned outZGroupEnd,
                      unsigned inZGroupBegin, unsigned inZGroupEnd,
                      unsigned kernelSizeY, unsigned kernelSizeX,
                      unsigned strideY, unsigned strideX, unsigned paddingY,
                      unsigned paddingX,
                      ComputeSet zeroCS,
                      ComputeSet fwdCS,
                      Tensor in, Tensor weights, Tensor out, bool forward) {
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  Tensor zeros;
  bool useConvPartial1x1OutVertex = false;
  if (partition.useConvolutionInstructions) {
    const auto weightsPerConvUnit =
        deviceInfo.getWeightsPerConvUnit(dType == "float");
    assert(weightsPerConvUnit % inChansPerGroup == 0);
    const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
    if (convUnitWeightHeight != 1) {
      assert(partition.useConvolutionInstructions);
      const auto inDimX = in.dim(2);
      const auto inputRange = getInputRange({outXBegin, outXEnd}, strideX,
                                            kernelSizeX, paddingX,
                                            inDimX, forward);
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
                                 (forward || (strideX == 1 && strideY == 1));
    if (!useConvPartial1x1OutVertex) {
      zeroPartialSums(graph, partition,
                      outXBegin, outXEnd, outYBegin, outYEnd,
                      outZGroupBegin, outZGroupEnd,
                      tile, zeroCS, out);
    }
  }
  const auto outHeight = outYEnd - outYBegin;
  const auto verticesPerY = partition.verticesPerTilePerYAxis;
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
                                      kernelSizeY, kernelSizeX, strideY,
                                      strideX, paddingY, paddingX,
                                      fwdCS, in, weights, out,
                                      forward);
      } else if (partition.useConvolutionInstructions) {
        createConvPartialnx1InOutVertex(graph, tile, outXBegin, outXEnd,
                                        vertexOutYBegin, vertexOutYEnd,
                                        ozg,
                                        inZGroupBegin, inZGroupEnd,
                                        kernelSizeY, kernelSizeX, strideY,
                                        strideX, paddingY, paddingX,
                                        fwdCS, in, weights, out,
                                        zeros, forward);
      } else {
        if (!forward)
          assert(0 && "Non convolution instruction backward "
                      "pass not yet implemented");
        createConvPartialDotProductVertex(graph, partition, tile,
                                          outXBegin, outXEnd,
                                          vertexOutYBegin, vertexOutYEnd,
                                          ozg,
                                          inZGroupBegin, inZGroupEnd,
                                          kernelSizeY, kernelSizeX, strideY,
                                          strideX, paddingY, paddingX,
                                          dType, fwdCS, in, weights, out);
      }
    }
  }
}


// Take an index range from a grouped tensor of dimensions
// {N/grouping, M, grouping} and return the corresponding indices
// of the ungrouped tensor of size {M, N}.
static std::vector<unsigned> getUngroupedIndices(unsigned start,
                                                 unsigned end,
                                                 unsigned N,
                                                 unsigned M,
                                                 unsigned grouping) {
  assert(N % grouping == 0);
  std::vector<unsigned> elems;
  for (unsigned i = start; i < end; ++i) {
    auto g = i / grouping;
    auto ii = (g % M) * N + (g / M) * grouping + i % grouping;
    elems.push_back(ii);
  }
  return elems;
}

// Take the indices from a ungrouped tensor of dimensions
// {M, N} and return the corresponding indices
// of the grouped tensor of size {N/grouping, M, grouping}.
static std::vector<unsigned>
getGroupedIndices(const std::vector<unsigned> &in,
                  unsigned N,
                  unsigned M,
                  unsigned grouping) {
  assert(N % grouping == 0);
  std::vector<unsigned> elems;
  for (auto i : in) {
    auto m = i / N;
    auto n = i % N;
    auto ii = (n / grouping) * M * grouping + m * grouping + n % grouping;
    elems.push_back(ii);
  }
  return elems;
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
                const Partition &partition,
                unsigned kernelSizeY, unsigned kernelSizeX, unsigned strideY,
                unsigned strideX, unsigned paddingY, unsigned paddingX,
                unsigned outNumChans,
                std::string dType,
                Tensor in, Tensor weights, Tensor partials,
                const std::string &layerName,
                unsigned outDimX, unsigned outDimY,
                bool forward) {
  const auto batchSize = in.dim(0);
  const auto isMultiIPU = graph.getDevice().getDeviceInfo().numIPUs > 1;
  const auto inNumChans = in.dim(1) * in.dim(4);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();

  ComputeSet zeroCS = graph.createComputeSet(layerName +".zero");
  ComputeSet convolveCS = graph.createComputeSet(layerName + ".convolve");
  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
      for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
        unsigned outZGroupBegin, outZGroupEnd;
        std::tie(outZGroupBegin, outZGroupEnd) =
            getOutZGroupRange(ozg, partialNumChanGroups, partition);
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
                                                   partition,
                                                   isMultiIPU);
            calcPartialConvOutput(graph, partition, dType,
                                  tile, outXBegin, outXEnd, outYBegin, outYEnd,
                                  outZGroupBegin, outZGroupEnd, inZGroupBegin,
                                  inZGroupEnd,
                                  kernelSizeY, kernelSizeX, strideY, strideX,
                                  paddingY, paddingX, zeroCS,
                                  convolveCS,
                                  in[b], weights,
                                  partials[b][izg],
                                  forward);
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

static void
reduce(Graph &graph,
       const Partition &partition,
       unsigned outNumChans, unsigned outNumChanGroups,
       Tensor partials,
       Tensor reduced,
       const std::vector<unsigned> &reducedMapping,
       ComputeSet reduceCS) {
  const auto partialType = partition.getPartialType();
  const auto outDimY = partials.dim(2);
  const auto outDimX = partials.dim(3);
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  // Accumulate the partial sums.
  const auto numTiles = deviceInfo.getNumTiles();
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto activationsBegin = reducedMapping[tile];
    const auto activationsEnd = reducedMapping[tile + 1];
    if (activationsBegin == activationsEnd)
      continue;
    auto elems = getUngroupedIndices(activationsBegin,
                                     activationsEnd,
                                     outNumChans,
                                     outDimY * outDimX,
                                     outChansPerGroup);
    elems = getGroupedIndices(elems, outNumChans, outDimY * outDimX,
                              partialChansPerGroup);
    std::sort(elems.begin(), elems.end());
    auto flatPartials = partials.reshape({tilesPerInZGroup,
                                          outDimX * outDimY * outNumChans});
    auto flatReduced = reduced.flatten();
    const auto workersPerTile = deviceInfo.numWorkerContexts;
    const auto maxElemsPerWorker =
      (elems.size() + workersPerTile - 1) / workersPerTile;
    const auto verticesToCreate =
      (elems.size() + maxElemsPerWorker - 1) / maxElemsPerWorker;
    for (unsigned vertex = 0; vertex < verticesToCreate; ++vertex) {
      unsigned elemBegin = (vertex * elems.size()) / verticesToCreate;
      unsigned elemEnd = ((vertex + 1) * elems.size()) / verticesToCreate;
      if (elemBegin == elemEnd)
        continue;
      auto regions = getContiguousRegions(elems.begin() + elemBegin,
                                          elems.begin() + elemEnd);
      const auto v = graph.addVertex(reduceCS,
                                     templateVertex("popnn::ConvReduce",
                                                    partialType));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setFieldSize(v["partials"], tilesPerInZGroup * regions.size());
      graph.setFieldSize(v["out"], regions.size());
      for (unsigned i = 0; i < regions.size(); ++i) {
        auto out = flatReduced.slice(regions[i].first, regions[i].second);
        graph.connect(v["out"][i], out);
        for (unsigned j = 0; j < tilesPerInZGroup; ++j) {
          graph.connect(v["partials"][i * tilesPerInZGroup + j],
                        flatPartials[j].slice(regions[i].first,
                                              regions[i].second));
        }
      }
      graph.setTileMapping(v, tile);
    }
  }
}

static Tensor
reduce(Graph &graph,
       const Partition &partition,
       unsigned outNumChans, unsigned outNumChanGroups,
       Tensor partials,
       const std::vector<unsigned> &activationsMapping,
       ComputeSet reduceCS) {
  if (partials.dim(0) == 1) {
    return partials[0];
  }

  const auto partialType = partition.getPartialType();
  const auto partialNumChanGroups = partials.dim(1);
  const auto outDimY = partials.dim(2);
  const auto outDimX = partials.dim(3);
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  Tensor reduced = graph.addTensor(partialType,
                                   {partialNumChanGroups, outDimY, outDimX,
                                    partialChansPerGroup}, "reduced");
  applyTensorMapping(graph, reduced, activationsMapping);
  reduce(graph, partition, outNumChans, outNumChanGroups,
         partials, reduced, activationsMapping, reduceCS);
  return reduced;
}

static void
complete(Graph &graph,
         const ConvPlan &plan,
         unsigned outNumChans,
         std::string dType,
         Tensor in, Tensor biases, Tensor activations,
         const std::vector<unsigned> &activationsMapping,
         const std::string &layerName,
         ComputeSet cs) {
  // Apply the non linearity and write back results in the layout desired by
  // the next layer. Each vertex handles outChansPerGroup output elements.
  // TODO: This step could be merged with the reduction step.
  const auto outDimY = activations.dim(1);
  const auto outDimX = activations.dim(2);
  const auto outNumChanGroups = activations.dim(0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto partialType = plan.fwdPartition.getPartialType();
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
            const ConvPlan &plan,
            unsigned kernelSizeY, unsigned kernelSizeX, unsigned strideY,
            unsigned strideX, unsigned paddingY, unsigned paddingX,
            unsigned outNumChans,
            Tensor in, Tensor weights, Tensor biases, Tensor activations,
            bool useWinogradConv, unsigned winogradPatchSize) {
  const auto dType = graph.getTensorElementType(in);
  const auto layerName =
      "Conv" + std::to_string(kernelSizeX) + "x" + std::to_string(kernelSizeY)
          + ".fwd";
  const auto outDimY = activations.dim(2);
  const auto outDimX = activations.dim(3);
  unsigned partialOutDimY, partialOutDimX;
  if (plan.flattenXY) {
    partialOutDimY = plan.fwdPartition.batchesPerGroup;
    partialOutDimX = outDimX * outDimY;
    const auto inDimY = in.dim(2);
    const auto inDimX = in.dim(3);
    in = in.dimShuffle({1, 0, 2, 3, 4}).reshape(
                          {in.dim(1),
                           plan.fwdPartition.numBatchGroups,
                           plan.fwdPartition.batchesPerGroup * inDimY,
                           inDimX,
                           in.dim(4)
                          }).dimShuffle({1, 0, 2, 3, 4});

    in = in.reshape({plan.fwdPartition.numBatchGroups,
                     in.dim(1),
                     plan.fwdPartition.batchesPerGroup,
                     inDimY * inDimX,
                     in.dim(4)});

  } else {
    partialOutDimY = outDimY;
    partialOutDimX = outDimX;
  }
  const auto batchSize = activations.dim(0);
  const auto outNumChanGroups = activations.dim(1);
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerInZGroup = plan.fwdPartition.tilesPerInZGroupAxis;
  const auto partialType = plan.fwdPartition.getPartialType();

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
          graph, kernelSizeY, kernelSizeX, strideY, strideX,
          paddingY, paddingX,
          in.dim(3), in.dim(2), outNumChans,
          winogradPatchSize, winogradPatchSize,
          dType, in[b],
          weights, biases,
          activations[b]));
    }
  } else {

    mapWeights(weights, graph, plan, batchSize);

    // Calculate a set of partial sums of the convolutions.
    Tensor partials = graph.addTensor(partialType,
                                       {plan.fwdPartition.numBatchGroups,
                                       tilesPerInZGroup,
                                       partialNumChanGroups,
                                       partialOutDimY,
                                       partialOutDimX,
                                       partialChansPerGroup},
                                      "partials");
    forwardProg.add(calcPartialSums(graph, plan.fwdPartition,
                                    kernelSizeY, kernelSizeX, strideY, strideX,
                                    paddingY, paddingX, outNumChans,
                                    dType, in, weights, partials, layerName,
                                    partialOutDimX, partialOutDimY,
                                    true));

    if (plan.flattenXY) {
      partials = partials.dimShuffle({1, 2, 0, 3, 4, 5 }).reshape(
        {tilesPerInZGroup, partialNumChanGroups,
         plan.fwdPartition.numBatchGroups * plan.fwdPartition.batchesPerGroup,
         partialOutDimY / plan.fwdPartition.batchesPerGroup,
         partialOutDimX, partialChansPerGroup}).dimShuffle(
            {2, 0, 1, 3, 4, 5});
    }

    // Before the reduction step we add any copying of the residual into the
    // reduce compute set
    ComputeSet reduceCS = graph.createComputeSet(layerName + ".reduce");

    // For each element of the batch, we add the reduction and complete
    // vertices to same compute sets so the batch will be executed in parallel.
    ComputeSet completeCS = graph.createComputeSet(layerName + ".complete");
    for (unsigned b = 0; b < batchSize; ++b) {
      // Perform the reduction of partial sums
      auto reducedMapping =
          computeActivationsMapping(graph, partials[b][0], b, batchSize);
      Tensor reduced = reduce(graph, plan.fwdPartition, outNumChans,
                              outNumChanGroups, partials[b],
                              reducedMapping, reduceCS);

      reduced = reduced.reshape({partialNumChanGroups, outDimY, outDimX,
                                 partialChansPerGroup});

      auto activationsMapping = computeActivationsMapping(graph,
                                                          activations[b],
                                                          b,
                                                          batchSize);
      // Add the residual (if any), apply the non-linearity and rearrange tensor
      // to required output channel grouping.
      complete(graph, plan, outNumChans, dType, reduced, biases,
               activations[b],
               activationsMapping, layerName, completeCS);
    }
    if (!graph.getComputeSet(reduceCS).empty())
      forwardProg.add(Execute(reduceCS));
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
                                               paddingY, inDimY, true);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, strideX, kernelSizeX,
                                                 paddingX, inDimX, true);
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
      std::max(deviceInfo.fp16AccumConvUnitsPerTile,
               deviceInfo.fp32AccumConvUnitsPerTile);
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

/** Copy the weights in 'weights' into 'bwdWeights' such that
 *  each element of the kernel is transposed w.r.t. the input and output
 *  channels. Note that the ordering of the kernel elements is not changed, just
 *  the ordering within each element.
 */
Program transformWeights(Graph &graph,
                         const std::string &layerName,
                         const ConvPlan &plan,
                         const std::string &dType,
                         Tensor weights,
                         Tensor bwdWeights) {
  const Partition &bwdPartition = plan.bwdPartition;
  const Partition &fwdPartition = plan.fwdPartition;
  const auto inChansPerGroup = bwdPartition.inChansPerGroup;
  const auto partialChansPerGroup = bwdPartition.partialChansPerGroup;
  auto cs = graph.createComputeSet(layerName + ".transformWeights");

  // TODO: is the weight mapping still the best given this transformation?

  assert(bwdPartition.useConvolutionInstructions &&
         "Backward pass for non dot product style conv not implemented");

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  iterateWeightMapping(bwdWeights, graph, bwdPartition, 1,
    [&](const Tensor &tileWeights, unsigned tile) {
    const auto flatTileWeights = tileWeights.flatten();
    const auto tileNumElements = flatTileWeights.numElements();
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
      const auto vertexWeights = flatTileWeights.slice(elemBegin, elemEnd);
      const auto elements = vertexWeights.getElementIndices();
      auto numElements = elements.size();
      auto regions = getContiguousRegions(elements.begin(),
                                          elements.end());
      auto v = graph.addVertex(cs, templateVertex("popnn::ConvTransformWeights",
                                                  dType));
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setFieldSize(v["out"], regions.size());
      for (unsigned i = 0; i < regions.size(); i++) {
        auto &region = regions[i];
        graph.connect(v["out"][i],
                      bwdWeights.flatten().slice(region.first,
                                                 region.second));
      };
      graph.setFieldSize(v["in"], numElements);
      for (unsigned i = 0; i < numElements; ++i) {
        const auto &elem = elements[i];
        auto coord = getElementCoord(elem, bwdWeights.dims());
        auto wx = coord[2]; auto wy = coord[3];
        auto outChan = coord[0] * partialChansPerGroup + coord[4];
        auto inChan = coord[1] * inChansPerGroup + coord[5];
        auto fwdPartialChansPerGroup = fwdPartition.partialChansPerGroup;
        auto fwdInChansPerGroup = fwdPartition.inChansPerGroup;
        auto w = weights[inChan / fwdPartialChansPerGroup]
            [outChan / fwdInChansPerGroup]
            [wx][wy]
            [inChan % fwdPartialChansPerGroup]
            [outChan % fwdInChansPerGroup];
        graph.connect(v["in"][i], w);
      }
      graph.setTileMapping(v, tile);
    }
  });
  return Execute(cs);
}

Program convolutionBackward(Graph &graph,
                            const ConvPlan &plan,
                            Tensor zDeltas, Tensor weights,
                            Tensor deltasOut,
                            unsigned kernelSizeY, unsigned kernelSizeX,
                            unsigned strideY, unsigned strideX,
                            unsigned paddingY, unsigned paddingX) {
  const auto dType = graph.getTensorElementType(zDeltas);
  const auto layerName =
      "Conv" + std::to_string(kernelSizeX) + "x" + std::to_string(kernelSizeY)
             + ".bwd";
  const auto batchSize = deltasOut.dim(0);
  const auto outDimY = deltasOut.dim(2);
  const auto outDimX = deltasOut.dim(3);
  unsigned partialOutDimY, partialOutDimX;
  if (plan.flattenXY) {
    partialOutDimY = plan.bwdPartition.batchesPerGroup;
    partialOutDimX = outDimX * outDimY;
    const auto inDimY = zDeltas.dim(2);
    const auto inDimX = zDeltas.dim(3);

    zDeltas = zDeltas.dimShuffle({1, 0, 2, 3, 4}).reshape(
           {zDeltas.dim(1),
            plan.bwdPartition.numBatchGroups,
            plan.bwdPartition.batchesPerGroup * inDimY,
            inDimX,
            zDeltas.dim(4)
           }).dimShuffle({1, 0, 2, 3, 4});

    zDeltas = zDeltas.reshape({plan.bwdPartition.numBatchGroups,
                               zDeltas.dim(1),
                               plan.bwdPartition.batchesPerGroup,
                               inDimY * inDimX,
                               zDeltas.dim(4)});

  } else {
    partialOutDimY = outDimY;
    partialOutDimX = outDimX;
  }
  const auto outNumChans = deltasOut.dim(1) * deltasOut.dim(4);
  const auto outNumChanGroups = deltasOut.dim(1);
  const auto partialChansPerGroup = plan.bwdPartition.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerInZGroup = plan.bwdPartition.tilesPerInZGroupAxis;
  const auto partialType = plan.bwdPartition.getPartialType();
  const auto inNumChanGroups = zDeltas.dim(1), inChansPerGroup = zDeltas.dim(4);
  assert(inChansPerGroup == plan.bwdPartition.inChansPerGroup);

  auto bwdProg = Sequence();
  auto bwdWeights = graph.addTensor(dType, {partialNumChanGroups,
                                            inNumChanGroups,
                                            kernelSizeY,
                                            kernelSizeX,
                                            partialChansPerGroup,
                                            inChansPerGroup},
                                            "bwdWeights");

  assert(bwdWeights.numElements() == weights.numElements());
  mapWeights(bwdWeights, graph, plan.bwdPartition, batchSize);

  bwdProg.add(transformWeights(graph, layerName, plan, dType, weights,
                               bwdWeights));

  // Calculate a set of partial sums of the convolutions.
  Tensor partials = graph.addTensor(partialType,
                                    {plan.bwdPartition.numBatchGroups,
                                     tilesPerInZGroup,
                                     partialNumChanGroups,
                                     partialOutDimY,
                                     partialOutDimX,
                                     partialChansPerGroup},
                                    "partials");
  bwdProg.add(calcPartialSums(graph, plan.bwdPartition,
                              kernelSizeY, kernelSizeX, strideY, strideX,
                              paddingY, paddingX, outNumChans, dType,
                              zDeltas, bwdWeights, partials, layerName,
                              partialOutDimX, partialOutDimY, false));


  if (plan.flattenXY) {
    partials = partials.dimShuffle({1, 2, 0, 3, 4, 5 }).reshape(
      {tilesPerInZGroup,
       partialNumChanGroups,
       plan.bwdPartition.numBatchGroups * plan.bwdPartition.batchesPerGroup,
       partialOutDimY / plan.bwdPartition.batchesPerGroup,
       partialOutDimX,
       partialChansPerGroup
      }).dimShuffle({2, 0, 1, 3, 4, 5});
  }

  // Perform the reduction of partial sums

  // For each element of the batch, we add the reduction and regroup
  // vertices to same compute sets so the batch will be executed in parallel.
  ComputeSet reduceCS = graph.createComputeSet(layerName + ".reduce");
  auto regroups = Sequence();
  const auto outChansPerGroup = deltasOut.dim(4);
  for (unsigned b = 0; b < batchSize; ++b) {
    auto reducedMapping =
        computeActivationsMapping(graph, partials[b][0], b, batchSize);
    auto reduced = reduce(graph, plan.bwdPartition, outNumChans,
                          outNumChanGroups,
                          partials[b],
                          reducedMapping, reduceCS);
    // Rearrange tensor to required output channel grouping.
    // TODO: the next layer's non-linearity derivative could be merged
    // into this.
    reduced = reduced.reshape({partialNumChanGroups, outDimY, outDimX,
                             partialChansPerGroup});
    auto activationsMapping =
        computeActivationsMapping(graph, deltasOut[b], b, batchSize);
    regroups.add(cast(graph, activationsMapping,
                      regroup(reduced, outChansPerGroup),
                      deltasOut[b]));
  }

  if (!graph.getComputeSet(reduceCS).empty())
    bwdProg.add(Execute(reduceCS));
  bwdProg.add(regroups);


  return bwdProg;
}

static void
createWeightGradVertex(Graph &graph,
                       const Partition &partition,
                       unsigned tile, const std::string &dType,
                       unsigned outXBegin, unsigned outXEnd,
                       unsigned outYBegin, unsigned outYEnd,
                       unsigned outZGroupBegin, unsigned outZGroup,
                       unsigned inZGroupBegin, unsigned inZGroupEnd,
                       unsigned kernelSizeY, unsigned kernelSizeX,
                       unsigned strideY, unsigned strideX,
                       unsigned paddingY, unsigned paddingX,
                       ComputeSet cs,
                       const Tensor &in, const Tensor &deltas,
                       const Tensor &weights) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  assert(outChansPerGroup == deltas.dim(3));
  const auto outHeight = outYEnd - outYBegin;
  const auto outWidth = outXEnd - outXBegin;
  for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
    auto v = graph.addVertex(cs, templateVertex("popnn::ConvWeightGradCalc",
                                                dType));
    graph.setTileMapping(v, tile);
    graph.setInitialValue(v["kernelSizeY"], kernelSizeY);
    graph.setInitialValue(v["kernelSizeX"], kernelSizeX);
    graph.setInitialValue(v["strideY"], strideY);
    graph.setInitialValue(v["strideX"], strideX);
    graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
    graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
    graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) =
        getInputRange({outYBegin, outYEnd}, strideY,
                      kernelSizeY, paddingY, inDimY, true);
    const auto inHeight = inYEnd - inYBegin;
    assert (inHeight != 0);
    unsigned inXBegin, inXEnd;
    std::tie(inXBegin, inXEnd) =
        getInputRange({outXBegin, outXEnd}, strideX,
                      kernelSizeX, paddingX, inDimX, true);
    graph.setInitialValue(v["ypadding"],
                          inYBegin < paddingY ? paddingY - inYBegin : 0);
    graph.setInitialValue(v["xpadding"],
                          inXBegin < paddingX ? paddingX - inXBegin : 0);
    const auto convInWidth = inXEnd - inXBegin;
    Tensor acts =
        in[izg].slice(
          {inYBegin, inXBegin, 0},
          {inYEnd, inXEnd, inChansPerGroup}
        ).reshape({inHeight, convInWidth * inChansPerGroup});
    Tensor ds =
        deltas[outZGroup].slice(
          {outYBegin, outXBegin, 0},
          {outYEnd, outXEnd, outChansPerGroup}
        ).reshape({outHeight, outWidth * outChansPerGroup});
    graph.connect(v["acts"], acts);
    graph.connect(v["deltas"], ds);
    auto w = weights[outZGroup][izg].flatten();
    graph.connect(v["weights"], w);
    graph.setTileMapping(w, tile);
  }
}

static void
calcPartialWeightGrads(Graph &graph,
                       const Partition &partition,
                       std::string dType,
                       unsigned tile,
                       unsigned outXBegin, unsigned outXEnd,
                       unsigned outYBegin, unsigned outYEnd,
                       unsigned outZGroupBegin, unsigned outZGroupEnd,
                       unsigned inZGroupBegin, unsigned inZGroupEnd,
                       unsigned kernelSizeY, unsigned kernelSizeX,
                       unsigned strideY, unsigned strideX,
                       unsigned paddingY, unsigned paddingX,
                       ComputeSet cs,
                       Tensor in, Tensor deltas, Tensor weights) {
  for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
    createWeightGradVertex(graph, partition, tile,
                           dType,
                           outXBegin, outXEnd, outYBegin,
                           outYEnd, outZGroupBegin, ozg, inZGroupBegin,
                           inZGroupEnd, kernelSizeY, kernelSizeX, strideY,
                           strideX, paddingY, paddingX,
                           cs, in, deltas, weights);
  }
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
matrixMultiplyByConvInstruction(Graph &graph, Partition partition,
                                Tensor a, Tensor b, Tensor c,
                                const std::vector<unsigned> &cTileMapping) {
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
  if (!partition.useConvolutionInstructions ||
      partition.inChansPerGroup != w ||
      partition.partialChansPerGroup != u) {
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
  if (partition.tilesPerInZGroupAxis > 1) {
    partials = graph.addTensor(partition.getPartialType(),
                              {batchSize,
                               partition.tilesPerInZGroupAxis,
                               outNumChans / outChansPerGroup,
                               outDimY,
                               outDimX,
                               outChansPerGroup},
                               "partials");
  } else {
    // No reduction required.
    partials = out.reshape({batchSize,
                            partition.tilesPerInZGroupAxis,
                            outNumChans / outChansPerGroup,
                            outDimY,
                            outDimX,
                            outChansPerGroup});
  }
  // Calculate a set of partial sums of the convolutions.
  prog.add(calcPartialSums(graph, partition,
                           kernelSize, kernelSize, stride, stride,
                           padding, padding, outNumChans,
                           dType, in, weights, partials, "matrixMul",
                           outDimX, outDimY,
                           true));
  // Perform the reduction of partial sums.
  if (partition.tilesPerInZGroupAxis > 1) {
    auto reduceCS = graph.createComputeSet("reduce");
    reduce(graph, partition, outNumChans, outNumChans / outChansPerGroup,
           partials[0], out[0], cTileMapping, reduceCS);
    prog.add(Execute(reduceCS));
  }
  return prog;
}

static void
convolutionBiasUpdate(Graph &graph, const Tensor &zDeltas, const Tensor &biases,
                      float learningRate, ComputeSet reduceCS,
                      ComputeSet updateBiasCS) {
  /** The number of biases is often small. So the reduction of bias
   *  updates is done in two stages to balance compute.
   *  TODO: This can probably be improved by reducing the biases on
   *  each tile according to the mapping of the deltas.
   */
  const auto batchSize = zDeltas.dim(0);
  const auto dType = graph.getTensorElementType(zDeltas);
  auto outDimY = zDeltas.dim(2), outDimX = zDeltas.dim(3);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  auto numWorkers = deviceInfo.numWorkerContexts * deviceInfo.getNumTiles();
  auto numBiases = biases.numElements();
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
  auto elemsPerBias = outDimY * outDimX;
  auto biasPartials = graph.addTensor(dType, {usedWorkers, maxBiasPerWorker},
                                      "biasPartials");
  auto zDeltasFlat = zDeltas.dimShuffle({1, 4, 0, 2, 3})
                            .reshape({numBiases,
                                      batchSize,
                                      outDimY * outDimX});
  for (unsigned worker = 0; worker  < usedWorkers; ++worker ) {
    auto tile = worker / deviceInfo.numWorkerContexts;
    graph.setTileMapping(biasPartials[worker].slice(0, maxBiasPerWorker), tile);
    unsigned biasBegin = (worker  * numBiases) / usedWorkers;
    unsigned biasEnd = ((worker  + workersPerBias) * numBiases) / usedWorkers;
    if (biasBegin == biasEnd)
      continue;
    unsigned elemBegin =
        ((worker  % workersPerBias) * elemsPerBias) / workersPerBias;
    unsigned elemEnd =
        (((worker  % workersPerBias) + 1) * elemsPerBias) / workersPerBias;
    if (elemBegin == elemEnd)
      continue;
    unsigned numWorkerBiases = biasEnd - biasBegin;
    auto v = graph.addVertex(reduceCS,
                             templateVertex("popnn::ConvBiasReduce", dType));
    graph.connect(v["deltas"], zDeltasFlat.slice({biasBegin, 0, elemBegin},
                                                 {biasEnd, batchSize, elemEnd})
                                          .flatten());
    graph.connect(v["biases"], biasPartials[worker].slice(0, numWorkerBiases));
    graph.setTileMapping(v, tile);
  }
  iterateBiasMapping(biases, graph, zDeltas[0], 0, 1,
    [&](Tensor biasSlice, unsigned tile){
      for (auto bias : biasSlice.getElementIndices()) {
        auto v = graph.addVertex(updateBiasCS,
                                 templateVertex("popnn::ConvBiasUpdate",
                                                dType));
        unsigned numDeltas = 0;
        for (unsigned srcWorker = 0; srcWorker < usedWorkers; ++srcWorker) {
          unsigned biasBegin = (srcWorker * numBiases) / usedWorkers;
          unsigned biasEnd =
              ((srcWorker + workersPerBias) * numBiases) / usedWorkers;
          if (biasBegin > bias || biasEnd <= bias)
            continue;
          graph.connect(v["deltas"][numDeltas++],
                        biasPartials[srcWorker][bias - biasBegin]);
        }
        graph.setFieldSize(v["deltas"], numDeltas);
        graph.connect(v["bias"], biases[bias]);
        graph.setInitialValue(v["eta"], learningRate);
        graph.setTileMapping(v, tile);
      }
     });
}

Program
convolutionWeightUpdateConvInst(Graph &graph,
                                const ConvPlan &plan,
                                Tensor zDeltas, Tensor weights, Tensor biases,
                                Tensor activations,
                                unsigned kernelSizeY, unsigned kernelSizeX,
                                unsigned strideY, unsigned strideX,
                                unsigned paddingY, unsigned paddingX,
                                float learningRate) {
  const auto layerName =
      "Conv" + std::to_string(kernelSizeX) + "x" + std::to_string(kernelSizeY)
             + ".weight_update";
  const auto &partition = plan.wuPartition;
  // We can calculate weight deltas using the convolution instruction where we
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
  auto fieldSize = height * width;
  auto deltasChansPerGroup = zDeltas.dim(4);
  auto deltasChans = deltasNumChanGroups * deltasChansPerGroup;
  auto activationsNumChanGroups = activations.dim(1);
  auto activationsChansPerGroup = activations.dim(4);
  auto activationsChans = activationsNumChanGroups * activationsChansPerGroup;
  const auto fieldGroupSize = partition.inChansPerGroup;
  // Pad the field so the size is a multiple of the number of weights in the
  // convolutional unit.
  const auto batchFieldSize = height * width * batchSize;
  const auto batchPaddedFieldSize =
      ((batchFieldSize + fieldGroupSize - 1) / fieldGroupSize) * fieldGroupSize;
  const auto outputGroupSize = partition.partialChansPerGroup;
  // The activationViews tensor contains the view on the activations for
  // each element of the kernel.
  auto activationViews =
      graph.addTensor(dType, {kernelSizeY * kernelSizeX,
                              activationsNumChanGroups,
                              0,
                              activationsChansPerGroup});
  auto prog = Sequence();
  for (unsigned b = 0; b != batchSize; ++b) {
    auto elemActivations = activations[b];
    auto elemActivationViews =
        graph.addTensor(dType, {0,
                                activationsNumChanGroups,
                                fieldSize,
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
        auto activationsFlattened =
            activationsStrided.reshape({activationsNumChanGroups,
                                        fieldSize,
                                        activationsChansPerGroup});
        // Pad the activations so the field size is a multiple of the number of
        // weights in the convolutional unit.
        elemActivationViews = append(elemActivationViews,
                                     activationsFlattened);
      }
    }
    activationViews = concat(activationViews, elemActivationViews, 2);
  }
  activationViews = pad(graph, activationViews,
                        {kernelSizeY * kernelSizeX,
                         activationsNumChanGroups,
                         batchPaddedFieldSize,
                         activationsChansPerGroup},
                        {0, 0, 0,0});
  const auto actViewSize = kernelSizeY * kernelSizeX * activationsChans;
  const auto paddedActViewSize =
     ((actViewSize + outputGroupSize - 1) / outputGroupSize) * outputGroupSize;
  auto activationsTransposed =
      graph.addTensor(dType,
          {paddedActViewSize / outputGroupSize,
           batchPaddedFieldSize / fieldGroupSize,
           outputGroupSize,
           fieldGroupSize},
          "activationsTransposed");
  auto activationsTransposedMapping =
      computeTensorMapping(graph, activationsTransposed);
  applyTensorMapping(graph, activationsTransposed,
                     activationsTransposedMapping);
  activationViews =
      activationViews.dimShuffle({0, 1, 3, 2})
                     .reshape({actViewSize,
                               batchPaddedFieldSize / fieldGroupSize,
                               fieldGroupSize});
  activationViews = pad(graph, activationViews,
                             {paddedActViewSize,
                              batchPaddedFieldSize / fieldGroupSize,
                              fieldGroupSize}, {0, 0, 0});
  auto activationsTransposedIn =
     activationViews.reshape({paddedActViewSize / outputGroupSize,
                              outputGroupSize,
                              batchPaddedFieldSize / fieldGroupSize,
                              fieldGroupSize})
                         .dimShuffle({0, 2, 1, 3});
  prog.add(Copy(activationsTransposed, activationsTransposedIn));
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  // Pad the field so the size is a multiple of the number of weights in the
  // convolutional unit.
  auto batchZDeltas = graph.addTensor(dType,
                                     {deltasNumChanGroups,
                                      0,
                                      deltasChansPerGroup});
  for (unsigned b = 0; b != batchSize; ++b) {
    // Flatten x and y into a single dimension.
    auto zDeltasFlattened = zDeltas[b].reshape({deltasNumChanGroups, fieldSize,
                                                deltasChansPerGroup});
    batchZDeltas = concat(batchZDeltas, zDeltasFlattened, 1);
  }
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
  auto zDeltasTransposedMapping =
      computeTensorMapping(graph, zDeltasTransposed);
  applyTensorMapping(graph, zDeltasTransposed, zDeltasTransposedMapping);
  prog.add(Copy(zDeltasTransposed,
                batchZDeltas.reshape({deltasNumChanGroups,
                                       batchPaddedFieldSize / fieldGroupSize,
                                       fieldGroupSize,
                                       deltasChansPerGroup})
                             .dimShuffle({1, 0, 3, 2})));
  const auto weightDeltasType = partition.floatPartials ? "float" : "half";
  auto weightDeltasTransposed =
      graph.addTensor(
          weightDeltasType,
          {paddedActViewSize / outputGroupSize,
           deltasChans,
           outputGroupSize},
          "weightDeltas");
  auto weightDeltasTransposedMapping =
      computeTensorMapping(graph, weightDeltasTransposed);
  applyTensorMapping(graph, weightDeltasTransposed,
                     weightDeltasTransposedMapping);
  // Perform the matrix multiplication.
  prog.add(matrixMultiplyByConvInstruction(
             graph, partition, activationsTransposed, zDeltasTransposed,
             weightDeltasTransposed, weightDeltasTransposedMapping
           )
  );
  // Transpose the weight deltas.
  auto fwdInChansPerGroup = plan.fwdPartition.inChansPerGroup;
  auto fwdPartialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
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
  auto addCS = graph.createComputeSet(layerName + ".update_weights");
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto weightsFlattened = weights.flatten();
  auto weightDeltasFlattened = weightDeltas.flatten();
  iterateWeightMapping(weights, graph, plan.fwdPartition, 1,
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
                                                      dType, weightDeltasType));
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setInitialValue(v["eta"], learningRate);
        graph.connect(v["weights"], weightsFlattened.slice(regions[i].first,
                                                           regions[i].second));
        graph.setFieldSize(v["partials"], 1);
        graph.connect(v["partials"][0],
            weightDeltasFlattened.slice(regions[i].first, regions[i].second));
        graph.setTileMapping(v, tile);
      }
    }
  });
  prog.add(Execute(addCS));
  auto reduceBiasCS = graph.createComputeSet(layerName + ".reduce_bias");
  auto updateBiasCS = graph.createComputeSet(layerName + ".update_bias");
  convolutionBiasUpdate(graph, zDeltas, biases, learningRate, reduceBiasCS,
                        updateBiasCS);
  prog.add(Execute(reduceBiasCS));
  prog.add(Execute(updateBiasCS));
  return prog;
}

Program
convolutionWeightUpdate(Graph &graph,
                        const ConvPlan &plan,
                        Tensor zDeltas, Tensor weights, Tensor biases,
                        Tensor activations,
                        unsigned kernelSizeY, unsigned kernelSizeX,
                        unsigned strideY, unsigned strideX,
                        unsigned paddingY, unsigned paddingX,
                        float learningRate) {
  const auto dType = graph.getTensorElementType(zDeltas);
  if (plan.wuPartition.useConvolutionInstructions) {
    return convolutionWeightUpdateConvInst(graph, plan, zDeltas, weights,
                                           biases, activations, kernelSizeY,
                                           kernelSizeX, strideY,
                                           strideX, paddingY,
                                           paddingX, learningRate);
  }
  const auto batchSize = activations.dim(0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto layerName =
      "Conv" + std::to_string(kernelSizeX) + "x" + std::to_string(kernelSizeY)
             + ".weight_update";
  assert(activations.dim(1) == weights.dim(1));
  auto outChansPerGroup = weights.dim(4);
  auto outNumChanGroups = weights.dim(0);
  auto outNumChans = outChansPerGroup * outNumChanGroups;
  auto outDimY = zDeltas.dim(2), outDimX = zDeltas.dim(3);
  auto prog = Sequence();
  const auto &partition = plan.wuPartition;
  const auto isMultiIPU = deviceInfo.numIPUs > 1;
  const auto inNumChans = activations.dim(1) * activations.dim(4);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto inNumChanGroups = activations.dim(1);
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;
  const auto numTiles = deviceInfo.getNumTiles();

  Tensor partials = graph.addTensor(dType, {batchSize,
                                            tilesPerY, tilesPerX,
                                            outNumChanGroups,
                                            inNumChanGroups,
                                            kernelSizeY, kernelSizeX,
                                            outChansPerGroup,
                                            inChansPerGroup},
                                    "partialWeightGrads");

  ComputeSet weightGradCS = graph.createComputeSet(layerName + ".weightGrad");
  for (unsigned b = 0; b < batchSize; ++b) {
    Tensor regroupedDeltas = graph.addTensor(dType, {outNumChanGroups,
                                                     outDimY, outDimX,
                                                     outChansPerGroup},
                                             "zDeltas'");
    prog.add(Copy(regroupedDeltas, regroup(zDeltas[b], outChansPerGroup)));
    auto regroupedDeltaMapping =
        computeActivationsMapping(graph, regroupedDeltas, b, batchSize);
    applyTensorMapping(graph, regroupedDeltas, regroupedDeltaMapping);
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
      for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
        const auto outZGroupBegin = (ozg * partialNumChanGroups) / tilesPerZ;
        const auto outZGroupEnd =
            ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
        for (unsigned oy = 0; oy != tilesPerY; ++oy) {
          const auto outYBegin = (oy * outDimY) / tilesPerY;
          const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
          for (unsigned ox = 0; ox != tilesPerX; ++ox) {
            const auto outXBegin = (ox * outDimX) / tilesPerX;
            const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
            const auto tile = linearizeTileIndices(b, batchSize, numTiles,
                                                   izg, ox, oy, ozg,
                                                   partition,
                                                   isMultiIPU);
            calcPartialWeightGrads(graph, partition, dType,
                                   tile, outXBegin, outXEnd, outYBegin, outYEnd,
                                   outZGroupBegin, outZGroupEnd, inZGroupBegin,
                                   inZGroupEnd, kernelSizeY, kernelSizeX,
                                   strideY, strideX, paddingY, paddingX,
                                   weightGradCS, activations[b],
                                   regroupedDeltas,
                                   partials[b][oy][ox]);
          }
        }
      }
    }
  }
  prog.add(Execute(weightGradCS));


  auto reduceCS = graph.createComputeSet(layerName + ".reduce");
  auto numPartials = batchSize * tilesPerY * tilesPerX;
  auto flatPartials = partials.reshape({numPartials,
                                        weights.numElements()});
  /** The reduction of weights is not performed where the weights are
   *  stored in the weight tensor. This causes some output exchange
   *  after the reduction but allows balancing of compute.
   */
  auto numWorkers = deviceInfo.numWorkerContexts * deviceInfo.getNumTiles();
  for (unsigned worker = 0; worker < numWorkers; ++worker) {
    auto beginElem = (worker * weights.numElements()) / numWorkers;
    auto endElem = ((worker + 1) * weights.numElements()) / numWorkers;
    if (beginElem == endElem)
      continue;
    auto numElems = endElem - beginElem;

    auto p = flatPartials.slice({0, beginElem},
                                {numPartials, endElem})
                         .reshape({numPartials,
                                   numElems});
    auto w = weights.flatten().slice(beginElem, endElem);
    auto v = graph.addVertex(reduceCS,
                             templateVertex("popnn::ConvWeightUpdate", dType,
                                            dType),
                             {{"weights", w}, {"partials", p}});
    graph.setInitialValue(v["eta"], learningRate);
    graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    graph.setTileMapping(v, worker / deviceInfo.numWorkerContexts);
  }
  auto updateBiasCS = graph.createComputeSet(layerName + ".update_bias");
  convolutionBiasUpdate(graph, zDeltas, biases, learningRate, reduceCS,
                        updateBiasCS);
  prog.add(Execute(reduceCS));
  prog.add(Execute(updateBiasCS));
  return prog;
}

} // namespace conv
