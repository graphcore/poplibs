#include "Convolution.hpp"
#include <limits>
#include <cassert>
#include "ConvUtil.hpp"
#include "ActivationMapping.hpp"
#include "VertexTemplates.hpp"
#include "neural_net_common.h"
#include "gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "exceptions.hpp"

using namespace poplar;
using namespace poplar::program;

namespace conv {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSize,
             unsigned stride, unsigned padding) {
  unsigned outDimX = (inDimX + (padding * 2) - kernelSize) / stride + 1;
  unsigned outDimY = (inDimY + (padding * 2) - kernelSize) / stride + 1;
  return {outDimY, outDimX};
}

poplar::Tensor
createWeights(poplar::Graph &graph, std::string dType,
             unsigned inNumChans,
             unsigned kernelSize,
             unsigned outNumChans,
             const ConvPlan &plan) {
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto inChansPerGroup = plan.fwdPartition.inChansPerGroup;
  const auto inNumChanGroups = inNumChans / inChansPerGroup;
  auto weights = graph.addTensor(dType, {partialNumChanGroups,
                                         inNumChanGroups,
                                         kernelSize,
                                         kernelSize,
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
linearizeTileIndices(unsigned izg, unsigned ox, unsigned oy,
                     unsigned ozg,
                     const Partition &partition,
                     bool isMultiIPU) {
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;

  // If this is a multi IPU system then choose an order that avoids splitting
  // partial sums over IPUs
  if (isMultiIPU)
    return izg + tilesPerInZGroup *
             (ox + tilesPerX *
               (oy + tilesPerY *
                 ozg));
  // For single IPU systems this order appears to give the best results.
  // TODO understand why this is. Intuitively I'd expect the an ordering
  // that matches the input tensor, i.e. (izg, iy, ix, iz) to result in
  // less exchange.
  return ox + tilesPerX *
           (oy + tilesPerY *
             (ozg + tilesPerZ *
               izg));
}

template <typename Builder>
static void
iterateWeightMapping(Tensor w,
                     const DeviceInfo &deviceInfo,
                     const Partition &partition,
                     Builder &&builder) {
  const auto isMultiIPU = deviceInfo.getNumIPUs() > 1;
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto inNumChans = w.dim(1) * w.dim(5);
  const auto outNumChans = w.dim(0) * w.dim(4);
  const auto kernelSize = w.dim(2);
  const auto numInZGroups = inNumChans / inChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;

  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
    const auto numInZGroups = inZGroupEnd - inZGroupBegin;
    for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
      const auto outZGroupBegin =
          (ozg * partialNumChanGroups) / tilesPerZ;
      const auto outZGroupEnd =
          ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
      const auto numOutZGroups = outZGroupEnd - outZGroupBegin;
      // Group weights that are accessed contiguously by tiles within this
      // loop body.
      Tensor sharedWeights;
      if (partition.useConvolutionInstructions) {
        if (kernelSize == 1) {
          sharedWeights =
              w.slice(
          {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
          {outZGroupEnd, inZGroupEnd, kernelSize, kernelSize,
           partialChansPerGroup, inChansPerGroup}
                ).reshape({numOutZGroups,
                           numInZGroups * partialChansPerGroup *
                           inChansPerGroup});
        } else {
          sharedWeights =
              w.slice(
          {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
          {outZGroupEnd, inZGroupEnd, kernelSize, kernelSize,
           partialChansPerGroup, inChansPerGroup}
                ).reshape({numOutZGroups * numInZGroups * kernelSize *
                           kernelSize,
                           partialChansPerGroup * inChansPerGroup});
        }
      } else {
        sharedWeights =
            w.slice(
        {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
        {outZGroupEnd, inZGroupEnd, kernelSize, kernelSize,
         1, inChansPerGroup}
              ).reshape({numInZGroups * numOutZGroups * kernelSize,
                         kernelSize * inChansPerGroup});
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
          const auto tile = linearizeTileIndices(izg, ox, oy, ozg, partition,
                                                 isMultiIPU);
          builder(tileWeights, tile);
        }
      }
    }
  }
}


static void
mapWeights(Tensor w, IPUModelEngineBuilder::TileMapping &mapping,
           const DeviceInfo &deviceInfo, const Partition &partition) {
  iterateWeightMapping(w, deviceInfo, partition,
    [&](Tensor tileWeights, unsigned tile) {
    mapping.setMapping(tileWeights, tile);
  });
}

void mapWeights(Tensor w, IPUModelEngineBuilder::TileMapping &mapping,
                const DeviceInfo &deviceInfo, const ConvPlan &plan) {
  mapWeights(w, mapping, deviceInfo, plan.fwdPartition);
}

template <typename Builder>
static void iterateBiasMapping(Tensor b, const DeviceInfo &deviceInfo,
                               Tensor activations,
                               Builder &&builder) {

  const auto activationsMapping = computeActivationsMapping(activations,
                                                            deviceInfo);
  const auto numTiles = deviceInfo.getNumTiles();
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

void mapBiases(Tensor b,
               IPUModelEngineBuilder::TileMapping &mapping,
               const DeviceInfo &deviceInfo,
               Tensor activations) {
  iterateBiasMapping(b, deviceInfo, activations,
    [&](Tensor biasSlice, unsigned tile) {
      mapping.setMapping(biasSlice, tile);
    });
}

static void
createConvPartial1x1OutVertex(Graph &graph,
                              IPUModelEngineBuilder::TileMapping &mapping,
                              const DeviceInfo &deviceInfo,
                              const Partition &partition,
                              const std::string &dType,
                              unsigned tile,
                              unsigned outXBegin, unsigned outXEnd,
                              unsigned outYBegin, unsigned outYEnd,
                              unsigned ozg,
                              unsigned inZGroupBegin, unsigned inZGroupEnd,
                              unsigned kernelSize, unsigned stride,
                              unsigned padding,
                              ComputeSet fwdCS,
                              const Tensor &in, const Tensor &weights,
                              const Tensor &out, bool forward) {
  assert(kernelSize == 1);
  assert(forward || stride == 1);
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex =
      deviceInfo.sharedConvWeights ? deviceInfo.getNumWorkerContexts() : 1;
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(dType == "float");
  assert(weightsPerConvUnit % inChansPerGroup == 0);
  const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
  if (convUnitWeightHeight != 1) {
    // Unimplemented.
    std::abort();
  }
  const auto outHeight = outYEnd - outYBegin;
  const auto outWidth = outXEnd - outXBegin;
  const auto partialType = partition.getPartialType();
  unsigned inYBegin, inYEnd, inXBegin, inXEnd;
  std::tie(inYBegin, inYEnd) =
      getInputRange({outYBegin, outYEnd}, stride, kernelSize,
                     padding, inDimY, forward);
  std::tie(inXBegin, inXEnd) =
      getInputRange({outXBegin, outXEnd}, stride,
                     kernelSize, padding, inDimX, forward);

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
        templateVertex("ConvPartial1x1Out", baseClass, dType, partialType),
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
            getInputIndex(workerOutY, stride, kernelSize,
                          padding, inDimY, 0, forward);
        assert(workerInY != ~0U);
        unsigned workerInXBegin, workerInXEnd;
        std::tie(workerInXBegin, workerInXEnd) =
            getInputRange({workerOutXBegin, workerOutXEnd}, stride,
                          kernelSize, padding, inDimX, forward);
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
        mapping.setMapping(outWindow, tile);
        graph.connect(v["in"][numConvolutions], inWindow);
        graph.connect(v["out"][numConvolutions], outWindow);
        ++numConvolutions;
      }
    }
  }
  graph.setFieldSize(v["in"], numConvolutions);
  graph.setFieldSize(v["out"], numConvolutions);
  // Map the vertex and output.
  mapping.setMapping(v, tile);
}

static void
createConvPartialnx1InOutVertex(Graph &graph,
                                IPUModelEngineBuilder::TileMapping &mapping,
                                const DeviceInfo &deviceInfo,
                                const Partition &partition,
                                const std::string &dType,
                                unsigned tile,
                                unsigned outXBegin, unsigned outXEnd,
                                unsigned outYBegin, unsigned outYEnd,
                                unsigned outZGroup,
                                unsigned inZGroupBegin, unsigned inZGroupEnd,
                                unsigned kernelSize, unsigned stride,
                                unsigned padding,
                                ComputeSet fwdCS,
                                const Tensor &in,
                                const Tensor &weights,
                                const Tensor &out,
                                const Tensor &zeros,
                                bool forward) {
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex =
      deviceInfo.sharedConvWeights ? deviceInfo.getNumWorkerContexts() : 1;
  const char *baseClass =
      deviceInfo.sharedConvWeights ? "poplar::SupervisorVertex" :
                                  "poplar::Vertex";
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(dType == "float");
  assert(weightsPerConvUnit % inChansPerGroup == 0);
  const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
  // Add the vertex.
  auto v =
      graph.addVertex(fwdCS,
                      templateVertex("ConvPartialnx1InOut", baseClass,
                                     dType, partition.getPartialType(),
                                     forward ? "true" : "false"));
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  mapping.setMapping(v, tile);
  unsigned numWeights = 0;
  unsigned numConvolutions = 0;
  for (unsigned wyBegin = 0; wyBegin < kernelSize;
       wyBegin += convUnitWeightHeight) {
    const auto wyEnd = std::min(kernelSize, wyBegin + convUnitWeightHeight);
    unsigned convOutYBegin, convOutYEnd;
    std::tie(convOutYBegin, convOutYEnd) =
        getOutputRange({outYBegin, outYEnd}, stride, kernelSize,
                       padding, inDimY, {wyBegin, wyEnd}, forward);
    const auto convOutHeight = convOutYEnd - convOutYBegin;
    if (convOutHeight == 0)
      continue;
    for (unsigned wx = 0; wx != kernelSize; ++wx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRange({outXBegin, outXEnd}, stride, kernelSize,
                         padding, inDimX, wx, forward);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;
      unsigned outputStride = forward ? 1 : stride;
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
            w = zeros.slice(0, inChansPerGroup);
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
            const auto workerOutXBegin = convOutXBegin + partialRow.begin;
            const auto workerOutXEnd = convOutXBegin + partialRow.end;
            const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
            unsigned workerInXBegin, workerInXEnd;
            std::tie(workerInXBegin, workerInXEnd) =
                getInputRange({workerOutXBegin, workerOutXEnd}, stride,
                              kernelSize, padding, inDimX, wx, forward);
            const auto workerInWidth = workerInXEnd - workerInXBegin;
            for (unsigned wy = wyBegin; wy != wyBegin + convUnitWeightHeight;
                 ++wy) {
              const auto workerInY =
                  getInputIndex(workerOutY, stride, kernelSize,
                                padding, inDimY, wy, forward);
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
                                  IPUModelEngineBuilder::TileMapping &mapping,
                                  const DeviceInfo &deviceInfo,
                                  const Partition &partition,
                                  unsigned tile,
                                  unsigned outXBegin, unsigned outXEnd,
                                  unsigned outYBegin, unsigned outYEnd,
                                  unsigned z,
                                  unsigned inZGroupBegin, unsigned inZGroupEnd,
                                  unsigned kernelSize, unsigned stride,
                                  unsigned padding,
                                  std::string dType,
                                  ComputeSet fwdCS,
                                  const Tensor &in,
                                  const Tensor &weights,
                                  const Tensor &out) {
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto inZGroups = inZGroupEnd - inZGroupBegin;
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto partialType = partition.getPartialType();
  assert(outChansPerGroup == 1);
  assert(outYEnd - outYBegin == 1);
  const auto y = outYBegin;
  unsigned inYBegin, inYEnd, inXBegin, inXEnd;
  std::tie(inYBegin, inYEnd) =
      getInputRange(y, stride, kernelSize, padding, inDimY, true);
  std::tie(inXBegin, inXEnd) =
      getInputRange({outXBegin, outXEnd}, stride, kernelSize,
                    padding, inDimX, true);
  // Window into previous layer.
  const auto inWidth = inXEnd - inXBegin;
  const auto inHeight = inYEnd - inYBegin;
  // Weights that match the window.
  unsigned weightYBegin, weightYEnd;
  std::tie(weightYBegin, weightYEnd) =
      getKernelRange(y, stride, kernelSize, padding, inDimY, true);
  Tensor inWindow =
      in.slice(
  {inZGroupBegin, inYBegin, inXBegin, 0},
  {inZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
        ).reshape({inHeight * inZGroups,
                   inWidth * inChansPerGroup});
  Tensor w =
      weights[z].slice(
  {inZGroupBegin, weightYBegin, 0, 0, 0},
  {inZGroupEnd, weightYEnd, kernelSize, 1, inChansPerGroup}
        ).reshape({inHeight * inZGroups,
                   inChansPerGroup * kernelSize});
  Tensor outWindow = out[z][y].slice(outXBegin, outXEnd).flatten();
  // Add the vertex.
  auto v = graph.addVertex(fwdCS,
                           templateVertex("ConvPartial", dType,
                                          partialType),
  { {"in", inWindow },
    {"weights", w },
    {"out", outWindow },
                           });
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["stride"], stride);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  unsigned vPadding = inXBegin < padding ? padding - inXBegin : 0;
  graph.setInitialValue(v["padding"], vPadding);
  // Map the vertex and output.
  mapping.setMapping(v, tile);
  mapping.setMapping(outWindow, tile);
}

static void
zeroPartialSums(Graph &graph,
                IPUModelEngineBuilder::TileMapping &mapping,
                const DeviceInfo &deviceInfo,
                const Partition &partition,
                unsigned outXBegin, unsigned outXEnd,
                unsigned outYBegin, unsigned outYEnd,
                unsigned tileOutZGroupBegin, unsigned tileOutZGroupEnd,
                unsigned tile,
                ComputeSet zeroCS,
                Tensor &out) {
  const auto outChansPerGroup = partition.partialChansPerGroup;
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
  mapping.setMapping(toZero, tile);
  const auto workersPerTile = deviceInfo.getNumWorkerContexts();
  const auto tileOutRows = toZero.dim(0);
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
      zeroCS, templateVertex("Zero2D", partialType),
      {{"out", toZero.slice(beginRow, endRow)}}
    );
    graph.setInitialValue(zv["dataPathWidth"], dataPathWidth);
    mapping.setMapping(zv, tile);
  }
}

static void
calcPartialConvOutput(Graph &graph,
                      IPUModelEngineBuilder::TileMapping &mapping,
                      const DeviceInfo &deviceInfo,
                      const Partition &partition,
                      std::string dType,
                      unsigned tile,
                      unsigned outXBegin, unsigned outXEnd,
                      unsigned outYBegin, unsigned outYEnd,
                      unsigned outZGroupBegin, unsigned outZGroupEnd,
                      unsigned inZGroupBegin, unsigned inZGroupEnd,
                      unsigned kernelSize, unsigned stride, unsigned padding,
                      ComputeSet zeroCS,
                      ComputeSet fwdCS,
                      Tensor in, Tensor weights, Tensor out, bool forward) {
  const auto inChansPerGroup = partition.inChansPerGroup;
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
      const auto inputRange = getInputRange({outXBegin, outXEnd}, stride,
                                            kernelSize, padding,
                                            inDimX, forward);
      const auto inputRangeSize = inputRange.second - inputRange.first;
      // This isn't split across multiple workers since it can happen in
      // parallel with zeroing the partial sums.
      zeros = graph.addTensor(dType, {inputRangeSize * inChansPerGroup},
                              "zeros");
      auto v = graph.addVertex(zeroCS, templateVertex("Zero", dType),
                               {{"out", zeros}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      mapping.setMapping(v, tile);
      mapping.setMapping(zeros, tile);
    }
    useConvPartial1x1OutVertex = kernelSize == 1 && (forward || stride == 1);
    if (!useConvPartial1x1OutVertex) {
      zeroPartialSums(graph, mapping, deviceInfo, partition,
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
        createConvPartial1x1OutVertex(graph, mapping, deviceInfo,
                                      partition, dType, tile,
                                      outXBegin, outXEnd,
                                      vertexOutYBegin, vertexOutYEnd,
                                      ozg,
                                      inZGroupBegin, inZGroupEnd,
                                      kernelSize, stride, padding,
                                      fwdCS, in, weights, out,
                                      forward);
      } else if (partition.useConvolutionInstructions) {
        createConvPartialnx1InOutVertex(graph, mapping, deviceInfo,
                                        partition, dType, tile,
                                        outXBegin, outXEnd,
                                        vertexOutYBegin, vertexOutYEnd,
                                        ozg,
                                        inZGroupBegin, inZGroupEnd,
                                        kernelSize, stride, padding,
                                        fwdCS, in, weights, out,
                                        zeros, forward);
      } else {
        if (!forward)
          assert(0 && "Non convolution instruction backward "
                      "pass not yet implemented");
        createConvPartialDotProductVertex(graph, mapping, deviceInfo,
                                          partition, tile,
                                          outXBegin, outXEnd,
                                          vertexOutYBegin, vertexOutYEnd,
                                          ozg,
                                          inZGroupBegin, inZGroupEnd,
                                          kernelSize, stride, padding,
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

static std::pair<unsigned, Tensor>
addResidualCalc(Graph &graph,
                IPUModelEngineBuilder::TileMapping &mapping,
                const DeviceInfo &deviceInfo,
                Tensor resIn, ComputeSet cs,
                unsigned outDimY, unsigned outDimX,
                unsigned outNumChans, unsigned outNumChanGroups,
                std::string dType,
                ResidualMethod resMethod) {
  auto resDimY = resIn.dim(1);
  auto resDimX = resIn.dim(2);
  if (resDimX < outDimX || resDimY < outDimY) {
    throw net_creation_error("Residual layers must use previous layers "
                             "with X and Y dimensions that are larger"
                             "than the current layer's output.");
  }
  unsigned resStride = resDimX / outDimX;
  if (resDimY / outDimY != resStride) {
    throw net_creation_error("Only residual layers with the same X/Y stride"
                             "are supported");
  }
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto resNumChanGroups = resIn.dim(0);
  auto resChansPerGroup = resIn.dim(3);
  auto resNumChans = resNumChanGroups * resChansPerGroup;
  if (resMethod != RESIDUAL_WEIGHTED_CONV &&
      resNumChans == outNumChans &&
      resNumChanGroups == outNumChanGroups) {
    // We can directly add the output of the previous layer to this
    // layer's output.
    return {resStride, resIn};
  }
  Tensor residual;
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  size_t resOutNumChanGroups =
      (resNumChans + outChansPerGroup - 1) / outChansPerGroup;
  residual = graph.addTensor(dType, {resOutNumChanGroups,
                                     outDimY, outDimX,
                                     outChansPerGroup},
                             "residual");
  mapTensor(residual, mapping, deviceInfo);
  switch (resMethod) {
  case RESIDUAL_PAD:
    for (unsigned outChanGroup = 0;
         outChanGroup < resOutNumChanGroups;
         ++outChanGroup) {
      for (unsigned y = 0; y < outDimY; ++y) {
        for (unsigned x = 0; x < outDimX; ++x) {
          auto chansPerVertex = dType == "float" ? 1 : 2;
          assert(outChansPerGroup % chansPerVertex == 0);
          assert(resChansPerGroup % chansPerVertex == 0);
          for (unsigned outChanGroupElement = 0;
               outChanGroupElement < outChansPerGroup;
               outChanGroupElement += chansPerVertex) {
            Tensor out = residual[outChanGroup][y][x]
              .slice(outChanGroupElement,
                     outChanGroupElement + chansPerVertex);
            auto outChan = outChanGroup * outChansPerGroup +
              outChanGroupElement;
            if (outChan >= resNumChans) {
              auto v = graph.addVertex(cs, templateVertex("Zero", dType),
                                       {{"out",out}});
              graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
              continue;
            }
            auto resChanGroup = outChan / resChansPerGroup;
            auto resChanGroupElement = outChan % resChansPerGroup;
            assert(resChanGroup < resNumChanGroups);
            assert(resChanGroupElement < resChansPerGroup);
            assert(y * resStride < resIn.dim(1));
            assert(x * resStride < resIn.dim(2));
            Tensor in = resIn[resChanGroup][y * resStride][x * resStride]
              .slice(resChanGroupElement,
                     resChanGroupElement + chansPerVertex);
            auto v = graph.addVertex(cs,
                                     templateVertex("CopyResidual", dType),
                                     {{"in", in}, {"out",out}});
            graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
          }
        }
      }
    }
    break;
  case RESIDUAL_WEIGHTED_CONV:
  case RESIDUAL_WEIGHTED_CONV_IF_SIZES_DIFFER:
    assert(0 && "Weighted calculation of residual input not implemented");
    break;
  default:
    assert(0 && "Unknown residual calculation method");
  }
  // This compute set may have more added with a specific mapping later. Here,
  // we map the current vertices of the compute set evenly over all tiles.
  auto vs = graph.getComputeSet(cs);
  std::uint64_t size = vs.size();
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned i = 0; i < numTiles; ++i) {
    const auto begin = (size * i) / numTiles;
    const auto end = (size * (i + 1)) / numTiles;
    if (begin == end)
      continue;
    for (unsigned j = begin; j != end; ++j) {
      mapping.setMapping(vs[j], i);
    }
  }
  return {1, residual};
}

static Program
calcPartialSums(Graph &graph,
                IPUModelEngineBuilder::TileMapping &mapping,
                const DeviceInfo &deviceInfo,
                const Partition &partition,
                unsigned kernelSize, unsigned stride, unsigned padding,
                unsigned outNumChans,
                std::string dType,
                Tensor in, Tensor weights, Tensor partials,
                const std::string &layerName,
                unsigned outDimX, unsigned outDimY,
                bool forward) {
  const auto isMultiIPU = deviceInfo.getNumIPUs() > 1;
  const auto inNumChans = in.dim(0) * in.dim(3);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;

  ComputeSet zeroCS = graph.createComputeSet(layerName +".zero");
  ComputeSet convolveCS = graph.createComputeSet(layerName + ".convolve");
  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
    for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
      const auto outZGroupBegin = (ozg * partialNumChanGroups) / tilesPerZ;
      const auto outZGroupEnd = ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
      for (unsigned oy = 0; oy != tilesPerY; ++oy) {
        const auto outYBegin = (oy * outDimY) / tilesPerY;
        const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
        for (unsigned ox = 0; ox != tilesPerX; ++ox) {
          const auto outXBegin = (ox * outDimX) / tilesPerX;
          const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
          const auto tile = linearizeTileIndices(izg, ox, oy, ozg,
                                                 partition,
                                                 isMultiIPU);
          calcPartialConvOutput(graph, mapping, deviceInfo, partition, dType,
                      tile, outXBegin, outXEnd, outYBegin, outYEnd,
                      outZGroupBegin, outZGroupEnd, inZGroupBegin, inZGroupEnd,
                      kernelSize, stride, padding,zeroCS, convolveCS,
                      in, weights, partials[izg], forward);
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

static std::pair<Tensor, Program>
reduce(Graph &graph,
       IPUModelEngineBuilder::TileMapping &mapping,
       const DeviceInfo &deviceInfo,
       const Partition &partition,
       unsigned outNumChans, unsigned outNumChanGroups,
       Tensor partials,
       const std::vector<unsigned> &activationsMapping,
       ComputeSet reduceCS) {
  if (partials.dim(0) == 1) {
    // Since only one partial sum was created, there is no need to
    // do a reduction.
    if (graph.getComputeSet(reduceCS).empty())
      return {partials[0], Sequence()};
    else
      return {partials[0], Execute(reduceCS)};
  }

  const auto partialType = partition.getPartialType();
  const auto partialNumChanGroups = partials.dim(1);
  const auto outDimY = partials.dim(2);
  const auto outDimX = partials.dim(3);
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto dataPathWidth = deviceInfo.dataPathWidth;

  Tensor reduced = graph.addTensor(partialType,
                                   {partialNumChanGroups, outDimY, outDimX,
                                    partialChansPerGroup}, "reduced");
  // Accumulate the partial sums.
  const auto numTiles = deviceInfo.getNumTiles();
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto activationsBegin = activationsMapping[tile];
    const auto activationsEnd = activationsMapping[tile + 1];
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
    const auto workersPerTile = deviceInfo.getNumWorkerContexts();
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
                                     templateVertex("ConvReduce",
                                                    partialType));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setFieldSize(v["partials"], tilesPerInZGroup * regions.size());
      graph.setFieldSize(v["out"], regions.size());
      for (unsigned i = 0; i < regions.size(); ++i) {
        auto out = flatReduced.slice(regions[i].first, regions[i].second);
        graph.connect(v["out"][i], out);
        mapping.setMapping(out, tile);
        for (unsigned j = 0; j < tilesPerInZGroup; ++j) {
          graph.connect(v["partials"][i * tilesPerInZGroup + j],
                        flatPartials[j].slice(regions[i].first,
                                              regions[i].second));
        }
      }
      mapping.setMapping(v, tile);
    }
  }
  return {reduced, Execute(reduceCS)};
}

static Program
complete(Graph &graph,
         IPUModelEngineBuilder::TileMapping &mapping,
         const DeviceInfo &deviceInfo,
         const ConvPlan &plan,
         unsigned outNumChans, NonLinearityType nonLinearityType,
         std::string dType,
         Tensor in, Tensor biases, Tensor activations,
         bool doResidual, Tensor residual, unsigned resStride,
         const std::vector<unsigned> &activationsMapping,
         const std::string &layerName) {
  // Apply the non linearity and write back results in the layout desired by
  // the next layer. Each vertex handles outChansPerGroup output elements.
  // TODO: This step could be merged with the reduction step.
  const auto outDimY = activations.dim(1);
  const auto outDimX = activations.dim(2);
  const auto outNumChanGroups = activations.dim(0);
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto partialType = plan.fwdPartition.getPartialType();
  ComputeSet cs = graph.createComputeSet(layerName + ".complete");
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  Tensor biasesByChanGroup =
      biases.reshape({outNumChanGroups, outChansPerGroup});
  const auto numTiles = deviceInfo.getNumTiles();
  const auto workersPerTile = deviceInfo.getNumWorkerContexts();
  const auto partialChanChunkSize =
      gcd<unsigned>(outChansPerGroup, partialChansPerGroup);
  const auto resOutChanGroups = doResidual ? residual.dim(0) : 0;

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
                               templateVertex("ConvComplete",
                                              partialType,
                                              dType));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      mapping.setMapping(v, tile);

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
      unsigned numResUsed = 0;
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
        if (doResidual && outChanGroup < resOutChanGroups) {
          // If the residual is taken directly from the previous layer (
          // as opposed to being zero-padded or converted), then striding over
          // the X,Y plane may still be needed (in this case resStride will not
          // be 1).
          Tensor res = residual[outChanGroup][y * resStride][x * resStride];
          graph.connect(res, v["res"][numResUsed++]);
        }
      }
      graph.setFieldSize(v["res"], numResUsed);
    }
  }
  return Execute(cs);
}

static Program
regroup(Graph &graph,
        IPUModelEngineBuilder::TileMapping &mapping,
        const DeviceInfo &deviceInfo,
        const std::string &layerName,
        const std::string &inType, const std::string &outType,
        const std::vector<unsigned> &outTileMapping,
        Tensor in, Tensor out) {
  const auto outDimY = out.dim(1);
  const auto outDimX = out.dim(2);
  const auto outNumChanGroups = out.dim(0);
  const auto outNumChans = out.dim(0) * out.dim(3);
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto inChansPerGroup = in.dim(3);
  ComputeSet cs = graph.createComputeSet(layerName + ".regroup");
  unsigned outChansPerGroup = outNumChans / outNumChanGroups;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto workersPerTile = deviceInfo.getNumWorkerContexts();
  const auto inChanChunkSize = gcd<unsigned>(outChansPerGroup, inChansPerGroup);

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileActivationsBegin = outTileMapping[tile];
    const auto tileActivationsEnd = outTileMapping[tile + 1];
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
                               templateVertex("RegroupChans",
                                              inType, outType));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      mapping.setMapping(v, tile);
      graph.setInitialValue(v["outChans"], outChansPerGroup);
      // Connect the output channel groups and inputs from the partial sums.
      graph.setFieldSize(v["out"], numGroups);
      graph.setFieldSize(v["in"],
                         numGroups * outChansPerGroup / inChanChunkSize);
      unsigned numIn = 0;
      for (auto group = groupBegin; group != groupEnd; ++group) {
        auto outChanGroup = group / (outDimX * outDimY);
        auto y = group % (outDimX * outDimY) / outDimX;
        auto x = group % outDimX;
        auto groupOut = out[outChanGroup][y][x];
        graph.connect(v["out"][group - groupBegin], groupOut);
        Tensor inChans = in.slice(
           {0, y, x, 0},
           {in.dim(0), y + 1, x + 1, inChansPerGroup}
        ).flatten();
        Tensor inByChanGroup =
            inChans.reshape({outNumChanGroups,
                             outChansPerGroup / inChanChunkSize,
                             inChanChunkSize});
        Tensor in = inByChanGroup[outChanGroup];
        for (unsigned i = 0; i < in.dim(0); ++i) {
          graph.connect(in[i], v["in"][numIn++]);
        }
      }
    }
  }
  return Execute(cs);
}

Program
convolution(Graph &graph,
            IPUModelEngineBuilder::TileMapping &mapping,
            const DeviceInfo &deviceInfo,
            const ConvPlan &plan,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned outNumChans, NonLinearityType nonLinearityType,
            std::string dType,
            Tensor in, Tensor weights, Tensor biases, Tensor activations,
            ResidualMethod resMethod, Tensor resIn) {
  const auto layerName =
      "Conv" + std::to_string(kernelSize) + "x" + std::to_string(kernelSize)
          + ".fwd";
  const auto outDimY = activations.dim(1);
  const auto outDimX = activations.dim(2);
  unsigned partialOutDimY, partialOutDimX;
  if (plan.flattenXY) {
    partialOutDimY = 1;
    partialOutDimX = outDimX * outDimY;
    in = in.reshape({in.dim(0), 1, in.dim(1) * in.dim(2), in.dim(3)});
  } else {
    partialOutDimY = outDimY;
    partialOutDimX = outDimX;
  }
  const auto outNumChanGroups = activations.dim(0);
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerInZGroup = plan.fwdPartition.tilesPerInZGroupAxis;
  const auto partialType = plan.fwdPartition.getPartialType();

  mapBiases(biases, mapping, deviceInfo, activations);
  mapWeights(weights, mapping, deviceInfo, plan);

  auto forwardProg = Sequence();

  // Calculate a set of partial sums of the convolutions.
  Tensor partials = graph.addTensor(partialType,
                                    {tilesPerInZGroup,
                                     partialNumChanGroups,
                                     partialOutDimY,
                                     partialOutDimX,
                                     partialChansPerGroup}, 
                                    "partials");
  forwardProg.add(calcPartialSums(graph, mapping, deviceInfo, plan.fwdPartition,
                                  kernelSize, stride, padding, outNumChans,
                                  dType, in, weights, partials, layerName,
                                  partialOutDimX, partialOutDimY,
                                  true));


  // Before the reduction step we add any copying of the residual into the
  // reduce compute set
  ComputeSet reduceCS = graph.createComputeSet(layerName + ".reduce");
  unsigned resStride; Tensor residual;
  bool doResidual = resMethod != RESIDUAL_NONE;
  if (doResidual) {
    std::tie(resStride, residual) =
        addResidualCalc(graph, mapping, deviceInfo, resIn, reduceCS,
                        outDimY, outDimX, outNumChans, outNumChanGroups, dType,
                        resMethod);
  }

  // Perform the reduction of partial sums
  auto activationsMapping = computeActivationsMapping(activations, deviceInfo);
  auto r = reduce(graph, mapping, deviceInfo, plan.fwdPartition, outNumChans,
                  outNumChanGroups, partials, activationsMapping, reduceCS);
  Tensor reduced = r.first;
  Program &reduceProg = r.second;
  forwardProg.add(reduceProg);
  reduced = reduced.reshape({partialNumChanGroups, outDimY, outDimX,
                             partialChansPerGroup});

  // Add the residual (if any), apply the non-linearity and rearrange tensor
  // to required output channel grouping.
  forwardProg.add(complete(graph, mapping, deviceInfo, plan, outNumChans,
                           nonLinearityType, dType, reduced, biases,
                           activations, doResidual, residual, resStride,
                           activationsMapping, layerName));

  return forwardProg;
}

static std::uint64_t getNumberOfMACs(unsigned outDimY, unsigned outDimX,
                                     unsigned outNumChans,
                                     unsigned kernelSize, unsigned stride,
                                     unsigned padding,
                                     unsigned inDimY, unsigned inDimX,
                                     unsigned inNumChans,
                                     bool forwardOnly) {
  std::uint64_t numMACs = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                               padding, inDimY, true);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 padding, inDimX, true);
      const auto width = inXEnd - inXBegin;
      numMACs += width * height * outNumChans * inNumChans;
    }
  }
  if (forwardOnly)
    return numMACs;
  else
    return numMACs * 3;
}

static std::uint64_t getNumberOfAdds(unsigned outDimY, unsigned outDimX,
                                     unsigned outNumChans, bool doResidual,
                                     bool forwardOnly) {
  if (!doResidual)
    return 0;

  // An addition is required to add in the residual information
  // TODO: backward residual operations
  return outNumChans * outDimX * outDimY;
}

uint64_t getFlops(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSize, unsigned stride, unsigned padding,
                  unsigned outNumChans, bool doResidual, bool forwardOnly) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  return 2 * getNumberOfMACs(outDimY, outDimX, outNumChans,
                             kernelSize, stride, padding,
                             inDimY, inDimX, inNumChans, forwardOnly) +
         getNumberOfAdds(outDimY, outDimX, outNumChans, doResidual,
                         forwardOnly);
}

double getPerfectCycleCount(const DeviceInfo &deviceInfo,
                            std::string dType,
                            unsigned inDimY, unsigned inDimX,
                            unsigned inNumChans,
                            unsigned kernelSize, unsigned stride,
                            unsigned padding,
                            unsigned outNumChans, bool doResidual,
                            bool forwardOnly) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  const auto numTiles = deviceInfo.getNumTiles();
  auto numMacs = getNumberOfMACs(outDimY, outDimX, outNumChans, kernelSize,
                                 stride, padding, inDimY, inDimX,
                                 inNumChans, forwardOnly);
  auto numAdds = getNumberOfAdds(outDimY, outDimX, outNumChans, doResidual,
                                 forwardOnly);
  if (dType == "float") {
    const auto floatVectorWidth = deviceInfo.getFloatVectorWidth();

    auto macCycles =
       static_cast<double>(numMacs) / (floatVectorWidth * numTiles);
    auto addCycles =
       static_cast<double>(numAdds) / (floatVectorWidth * numTiles);
    return macCycles + addCycles;
  }
  assert(dType == "half");
  const auto convUnitsPerTile =
      std::max(deviceInfo.fp16AccumConvUnitsPerTile,
               deviceInfo.fp32AccumConvUnitsPerTile);
  const auto halfVectorWidth = deviceInfo.getHalfVectorWidth();
  auto macsPerCycle = convUnitsPerTile * halfVectorWidth;
  auto macCycles = static_cast<double>(numMacs) / (macsPerCycle * numTiles);
  auto addCycles = static_cast<double>(numAdds) / (halfVectorWidth * numTiles);
  return macCycles + addCycles;
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

Program transformWeights(Graph &graph,
                         IPUModelEngineBuilder::TileMapping &mapping,
                         const DeviceInfo &deviceInfo,
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

  iterateWeightMapping(bwdWeights, deviceInfo, bwdPartition,
    [&](const Tensor &tileWeights, unsigned tile) {
    const auto flatTileWeights = tileWeights.flatten();
    const auto tileNumElements = flatTileWeights.numElements();
    const auto workersPerTile = deviceInfo.getNumWorkerContexts();
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
      auto v = graph.addVertex(cs, templateVertex("ConvTransformWeights",
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
      mapping.setMapping(v, tile);
    }
  });
  return Execute(cs);
}

Program convolutionBackward(Graph &graph,
                            IPUModelEngineBuilder::TileMapping &mapping,
                            const DeviceInfo &deviceInfo,
                            const ConvPlan &plan,
                            std::string dType,
                            Tensor zDeltas, Tensor weights,
                            Tensor deltasOut,
                            unsigned kernelSize, unsigned stride,
                            unsigned padding) {
  const auto layerName =
      "Conv" + std::to_string(kernelSize) + "x" + std::to_string(kernelSize)
             + ".bwd";
  const auto outDimY = deltasOut.dim(1);
  const auto outDimX = deltasOut.dim(2);
  unsigned partialOutDimY, partialOutDimX;
  if (plan.flattenXY) {
    partialOutDimY = 1;
    partialOutDimX = outDimX * outDimY;
    zDeltas = zDeltas.reshape({zDeltas.dim(0), 1,
                               zDeltas.dim(1) * zDeltas.dim(2),
                               zDeltas.dim(3)});
  } else {
    partialOutDimY = outDimY;
    partialOutDimX = outDimX;
  }
  const auto outNumChans = deltasOut.dim(0) * deltasOut.dim(3);
  const auto outNumChanGroups = deltasOut.dim(0);
  const auto partialChansPerGroup = plan.bwdPartition.partialChansPerGroup;
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerInZGroup = plan.bwdPartition.tilesPerInZGroupAxis;
  const auto partialType = plan.bwdPartition.getPartialType();
  const auto inNumChanGroups = zDeltas.dim(0), inChansPerGroup = zDeltas.dim(3);
  assert(inChansPerGroup == plan.bwdPartition.inChansPerGroup);

  auto bwdProg = Sequence();
  auto bwdWeights = graph.addTensor(dType, {partialNumChanGroups,
                                            inNumChanGroups,
                                            kernelSize,
                                            kernelSize,
                                            partialChansPerGroup,
                                            inChansPerGroup},
                                            "bwdWeights");

  assert(bwdWeights.numElements() == weights.numElements());
  mapWeights(bwdWeights, mapping, deviceInfo, plan.bwdPartition);

  bwdProg.add(transformWeights(graph, mapping, deviceInfo, layerName,
                               plan, dType, weights, bwdWeights));

  // Calculate a set of partial sums of the convolutions.
  Tensor partials = graph.addTensor(partialType,
                                    {tilesPerInZGroup,
                                     partialNumChanGroups,
                                     partialOutDimY,
                                     partialOutDimX,
                                     partialChansPerGroup},
                                    "partials");
  bwdProg.add(calcPartialSums(graph, mapping, deviceInfo, plan.bwdPartition,
                              kernelSize, stride, padding, outNumChans, dType,
                              zDeltas, bwdWeights, partials, layerName,
                              partialOutDimX, partialOutDimY, false));

  // TODO - residuals

  // Perform the reduction of partial sums
  ComputeSet reduceCS = graph.createComputeSet(layerName + ".reduce");
  auto activationsMapping = computeActivationsMapping(deltasOut, deviceInfo);
  auto r = reduce(graph, mapping, deviceInfo, plan.bwdPartition, outNumChans,
                  outNumChanGroups, partials, activationsMapping, reduceCS);
  Tensor reduced = r.first;
  Program &reduceProg = r.second;
  bwdProg.add(reduceProg);

  // Rearrange tensor to required output channel grouping.
  // TODO: the next layer's non-linearity derivative could be merged
  // into this.
  reduced = reduced.reshape({partialNumChanGroups, outDimY, outDimX,
                             partialChansPerGroup});
  bwdProg.add(regroup(graph, mapping, deviceInfo, layerName, partialType,
                      dType, activationsMapping, reduced, deltasOut));

  return bwdProg;
}

static void
createWeightGradVertex(Graph &graph,
                       IPUModelEngineBuilder::TileMapping &mapping,
                       const DeviceInfo &deviceInfo,
                       const Partition &partition,
                       unsigned tile, const std::string &dType,
                       unsigned outXBegin, unsigned outXEnd,
                       unsigned outYBegin, unsigned outYEnd,
                       unsigned outZGroupBegin, unsigned outZGroup,
                       unsigned inZGroupBegin, unsigned inZGroupEnd,
                       unsigned kernelSize, unsigned stride,
                       unsigned padding,
                       ComputeSet cs,
                       const Tensor &in, const Tensor &deltas,
                       const Tensor &weights) {
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto outHeight = outYEnd - outYBegin;
  const auto outWidth = outXEnd - outXBegin;
  for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
    auto v = graph.addVertex(cs, templateVertex("ConvWeightGradCalc",
                                                dType));
    mapping.setMapping(v, tile);
    graph.setInitialValue(v["kernelSize"], kernelSize);
    graph.setInitialValue(v["stride"], stride);
    graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
    graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
    graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) =
        getInputRange({outYBegin, outYEnd}, stride,
                      kernelSize, padding, inDimY, true);
    const auto inHeight = inYEnd - inYBegin;
    assert (inHeight != 0);
    unsigned inXBegin, inXEnd;
    std::tie(inXBegin, inXEnd) =
        getInputRange({outXBegin, outXEnd}, stride,
                      kernelSize, padding, inDimX, true);
    graph.setInitialValue(v["ypadding"],
                          inYBegin < padding ? padding - inYBegin : 0);
    graph.setInitialValue(v["xpadding"],
                          inXBegin < padding ? padding - inXBegin : 0);
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
  }
}

static void
calcPartialWeightGrads(Graph &graph,
                       IPUModelEngineBuilder::TileMapping &mapping,
                       const DeviceInfo &deviceInfo,
                       const Partition &partition,
                       std::string dType,
                       unsigned tile,
                       unsigned outXBegin, unsigned outXEnd,
                       unsigned outYBegin, unsigned outYEnd,
                       unsigned outZGroupBegin, unsigned outZGroupEnd,
                       unsigned inZGroupBegin, unsigned inZGroupEnd,
                       unsigned kernelSize, unsigned stride, unsigned padding,
                       ComputeSet cs,
                       Tensor in, Tensor deltas, Tensor weights) {
  for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
    createWeightGradVertex(graph, mapping, deviceInfo, partition, tile,
                           dType,
                           outXBegin, outXEnd, outYBegin,
                           outYEnd, outZGroupBegin, ozg, inZGroupBegin,
                           inZGroupEnd, kernelSize, stride,
                           padding,
                           cs, in, deltas, weights);
  }
}

Program
convolutionWeightUpdate(Graph &graph,
                        IPUModelEngineBuilder::TileMapping &mapping,
                        const DeviceInfo &deviceInfo,
                        const ConvPlan &plan,
                        std::string dType,
                        Tensor zDeltas, Tensor weights, Tensor biases,
                        Tensor activations,
                        unsigned kernelSize, unsigned stride,
                        unsigned padding, float learningRate) {
  const auto layerName =
      "Conv" + std::to_string(kernelSize) + "x" + std::to_string(kernelSize)
             + ".weight_update";
  assert(activations.dim(0) == weights.dim(1));
  auto outChansPerGroup = weights.dim(4);
  auto outNumChanGroups = weights.dim(0);
  auto outNumChans = outChansPerGroup * outNumChanGroups;
  auto outDimY = zDeltas.dim(1), outDimX = zDeltas.dim(2);
  Tensor regroupedDeltas = graph.addTensor(dType, {outNumChanGroups,
                                                   outDimY, outDimX,
                                                   outChansPerGroup},
                                           "zDeltas'");
  mapActivations(regroupedDeltas, mapping, deviceInfo);
  auto regroupedDeltaMapping = computeActivationsMapping(regroupedDeltas,
                                                         deviceInfo);
  auto prog = Sequence();
  prog.add(regroup(graph, mapping, deviceInfo, layerName, dType,
                   dType, regroupedDeltaMapping, zDeltas, regroupedDeltas));

  const auto &partition = plan.wuPartition;
  const auto isMultiIPU = deviceInfo.getNumIPUs() > 1;
  const auto inNumChans = activations.dim(0) * activations.dim(3);
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto inNumChanGroups = activations.dim(0);
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;

  Tensor partials = graph.addTensor(dType, {tilesPerY, tilesPerX,
                                            outNumChanGroups,
                                            inNumChanGroups,
                                            kernelSize, kernelSize,
                                            outChansPerGroup,
                                            inChansPerGroup},
                                    "partialWeightGrads");



  ComputeSet weightGradCS = graph.createComputeSet(layerName + ".weightGrad");
  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
    for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
      const auto outZGroupBegin = (ozg * partialNumChanGroups) / tilesPerZ;
      const auto outZGroupEnd = ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
      for (unsigned oy = 0; oy != tilesPerY; ++oy) {
        const auto outYBegin = (oy * outDimY) / tilesPerY;
        const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
        for (unsigned ox = 0; ox != tilesPerX; ++ox) {
          const auto outXBegin = (ox * outDimX) / tilesPerX;
          const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
          const auto tile = linearizeTileIndices(izg, ox, oy, ozg,
                                                 partition,
                                                 isMultiIPU);
          calcPartialWeightGrads(graph, mapping, deviceInfo, partition, dType,
                                 tile, outXBegin, outXEnd, outYBegin, outYEnd,
                                 outZGroupBegin, outZGroupEnd, inZGroupBegin,
                                 inZGroupEnd, kernelSize, stride, padding,
                                 weightGradCS, activations, regroupedDeltas,
                                 partials[oy][ox]);
        }
      }
    }
  }
  prog.add(Execute(weightGradCS));


  auto reduceCS = graph.createComputeSet(layerName + ".reduce");

  /** The reduction of weights is not performed where the weights are
   *  stored in the weight tensor. This causes some output exchange
   *  after the reduction but allows balancing of compute.
   */
  auto numWorkers =
      deviceInfo.getNumWorkerContexts() * deviceInfo.getNumTiles();
  for (unsigned worker = 0; worker < numWorkers; ++worker) {
    auto beginElem = (worker * weights.numElements()) / numWorkers;
    auto endElem = ((worker + 1) * weights.numElements()) / numWorkers;
    if (beginElem == endElem)
      continue;
    auto numElems = endElem - beginElem;
    auto flatPartials = partials.reshape({tilesPerY * tilesPerX,
                                          weights.numElements()});
    auto p = flatPartials.slice({0, beginElem},
                                {tilesPerY * tilesPerX, endElem})
                         .reshape({tilesPerY * tilesPerX,
                                   numElems});
    auto w = weights.flatten().slice(beginElem, endElem);
    auto v = graph.addVertex(reduceCS,
                             templateVertex("ConvWeightUpdate", dType),
                             {{"weights", w}, {"partials", p}});
    graph.setInitialValue(v["eta"], learningRate);
    graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    mapping.setMapping(v, worker / deviceInfo.getNumWorkerContexts());
  }
  /** The number of biases is often small. So the reduction of bias
   *  updates is done in two stages to balance compute.
   *  TODO: This can probably be improved by reducing the biases on
   *  each tile according to the mapping of the deltas.
   */
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
  auto biasPartials = graph.addTensor(dType, {numWorkers, maxBiasPerWorker},
                                      "biasPartials");

  for (unsigned worker = 0; worker  < usedWorkers; ++worker ) {
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
    auto numElems = elemEnd - elemBegin;
    unsigned numBiases = biasEnd - biasBegin;
    auto v = graph.addVertex(reduceCS,
                             templateVertex("ConvBiasReduce", dType));
    graph.setFieldSize(v["deltas"], numElems * numBiases);
    auto zDeltasFlat = zDeltas.reshape({zDeltas.dim(0),
                                        outDimY * outDimX,
                                        zDeltas.dim(3)});
    for (unsigned i = 0; i < numBiases; ++i) {
      unsigned deltaGroup = (biasBegin + i) / zDeltas.dim(3);
      unsigned deltaInGroup = (biasEnd + i) % zDeltas.dim(3);
      auto in = zDeltasFlat.slice({deltaGroup, elemBegin, deltaInGroup},
                                  {deltaGroup + 1, elemEnd, deltaInGroup + 1})
                           .flatten();
      for (unsigned j = 0; j < numElems; ++j) {
        graph.connect(v["deltas"][i * numElems + j], in[j]);
      }
    }
    graph.connect(v["biases"], biasPartials[worker].slice(0, numBiases));
    auto tile =  worker / deviceInfo.getNumWorkerContexts();
    mapping.setMapping(v, tile);
    mapping.setMapping(biasPartials[worker].slice(0, numBiases), tile);
  }

  auto updateBiasCS = graph.createComputeSet(layerName + ".update_bias");
  iterateBiasMapping(biases, deviceInfo, zDeltas,
    [&](Tensor biasSlice, unsigned tile){
      for (auto bias : biasSlice.getElementIndices()) {
        auto v = graph.addVertex(updateBiasCS,
                                 templateVertex("ConvBiasUpdate", dType));
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
        mapping.setMapping(v, tile);
      }
     });

  prog.add(Execute(reduceCS));
  prog.add(Execute(updateBiasCS));
  return prog;
}




} // namespace conv
