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


std::pair<poplar::Tensor, poplar::Tensor>
createParams(poplar::Graph &graph, std::string dType,
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
  auto biases = graph.addTensor(dType, {outNumChans}, "biases");
  return {weights, biases};
}

static unsigned
linearizeTileIndices(unsigned izg, unsigned ox, unsigned oy,
                     unsigned ozg,
                     const ConvPlan &plan,
                     bool isMultiIPU) {
  const auto tilesPerX = plan.fwdPartition.tilesPerXAxis;
  const auto tilesPerY = plan.fwdPartition.tilesPerYAxis;
  const auto tilesPerZ = plan.fwdPartition.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.fwdPartition.tilesPerInZGroupAxis;

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

void mapWeights(Tensor w, IPUModelEngineBuilder::TileMapping &mapping,
                const DeviceInfo &deviceInfo, const ConvPlan &plan) {
  const auto isMultiIPU = deviceInfo.getNumIPUs() > 1;
  const auto inChansPerGroup = plan.fwdPartition.inChansPerGroup;
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto tilesPerX = plan.fwdPartition.tilesPerXAxis;
  const auto tilesPerY = plan.fwdPartition.tilesPerYAxis;
  const auto tilesPerZ = plan.fwdPartition.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.fwdPartition.tilesPerInZGroupAxis;
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
      if (plan.useConvolutionInstruction) {
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
          const auto tileWeights =
              sharedWeights.slice(sharedWeightGroupBegin,
                                  sharedWeightGroupEnd);
          const auto tile = linearizeTileIndices(izg, ox, oy, ozg, plan,
                                                 isMultiIPU);
          mapping.setMapping(tileWeights, tile);
        }
      }
    }
  }
}

void mapBiases(Tensor b,
               IPUModelEngineBuilder::TileMapping &mapping,
               const DeviceInfo &deviceInfo,
               Tensor activations) {
  const auto activationsMapping = computeActivationsMapping(activations,
                                                            deviceInfo);
  const auto numTiles = deviceInfo.getNumTiles();
  const auto outNumChans = activations.dim(0) * activations.dim(3);
  const auto outNumChanGroups = activations.dim(0);
  const auto outDimY = activations.dim(1);
  const auto outDimX = activations.dim(2);
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
    Tensor biasSlice = biasesByChanGroup.slice(minOutChanGroup,
                                               maxOutChanGroup + 1);
    mapping.setMapping(biasSlice, tile);
  }
}


void
createConvPartial1x1InOutVertex(Graph &graph,
                                IPUModelEngineBuilder::TileMapping &mapping,
                                const DeviceInfo &deviceInfo,
                                const ConvPlan &plan,
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
                                const Tensor &out) {
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto inChansPerGroup = plan.fwdPartition.inChansPerGroup;
  const auto outChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex =
      deviceInfo.sharedConvWeights ? deviceInfo.getNumWorkerContexts() : 1;
  const char *baseClass =
      deviceInfo.sharedConvWeights ? "poplar::SupervisorVertex" :
                                  "poplar::Vertex";

  // Add the vertex.
  auto v =
      graph.addVertex(fwdCS,
                      templateVertex("ConvPartial1x1InOut", baseClass,
                                     plan.fwdPartition.getPartialType()));
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  mapping.setMapping(v, tile);
  unsigned numWeights = 0;
  unsigned numConvolutions = 0;
  for (unsigned wy = 0; wy != kernelSize; ++wy) {
    unsigned convOutYBegin, convOutYEnd;
    std::tie(convOutYBegin, convOutYEnd) =
        getOutputRange({outYBegin, outYEnd}, stride, kernelSize,
                       padding, inDimY, wy);
    const auto convOutHeight = convOutYEnd - convOutYBegin;
    if (convOutHeight == 0)
      continue;
    for (unsigned wx = 0; wx != kernelSize; ++wx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRange({outXBegin, outXEnd}, stride, kernelSize,
                         padding, inDimX, wx);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;
      std::vector<std::vector<PartialRow>> workerPartition =
          partitionConvPartialByWorker(convOutHeight, convOutWidth,
                                       contextsPerVertex);
      assert(workerPartition.size() == contextsPerVertex);
      for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
        Tensor w =
            weights[outZGroup][izg][wy][wx].flatten();
        graph.connect(v["weights"][numWeights], w);
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
            const auto workerInY =
              getInputIndex(workerOutY, stride, kernelSize,
                            padding, inDimY, wy);
            assert(workerInY != ~0U);
            unsigned workerInXBegin, workerInXEnd;
            std::tie(workerInXBegin, workerInXEnd) =
                getInputRange({workerOutXBegin, workerOutXEnd}, stride,
                              kernelSize, padding, inDimX, wx);
            const auto workerInWidth = workerInXEnd - workerInXBegin;
            Tensor inWindow =
                in[izg][workerInY].slice(
                  {workerInXBegin, 0},
                  {workerInXEnd, inChansPerGroup}
                ).reshape({workerInWidth * inChansPerGroup});
            Tensor outWindow =
                out[outZGroup][workerOutY].slice(
                  {workerOutXBegin, 0},
                  {workerOutXEnd, outChansPerGroup}
                ).reshape({workerOutWidth * outChansPerGroup});
            mapping.setMapping(outWindow, tile);
            graph.connect(v["in"][numConvolutions], inWindow);
            graph.connect(v["out"][numConvolutions], outWindow);
            ++numConvolutions;
          }
        }
        ++numWeights;
      }
    }
  }
  graph.setFieldSize(v["in"], numConvolutions);
  graph.setFieldSize(v["out"], numConvolutions);
  graph.setFieldSize(v["weights"], numWeights);
  graph.setFieldSize(v["weightReuseCount"], numWeights * contextsPerVertex);
}

void
forwardTile(Graph &graph,
            IPUModelEngineBuilder::TileMapping &mapping,
            const DeviceInfo &deviceInfo,
            const ConvPlan &plan,
            std::string dType,
            unsigned tile,
            unsigned tileOutXBegin, unsigned tileOutXEnd,
            unsigned tileOutYBegin, unsigned tileOutYEnd,
            unsigned tileOutZGroupBegin, unsigned tileOutZGroupEnd,
            unsigned tileInZGroupBegin, unsigned tileInZGroupEnd,
            unsigned kernelSize, unsigned stride, unsigned padding,
            ComputeSet zeroCS,
            ComputeSet fwdCS,
            Tensor in, Tensor weights, Tensor out) {
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  const auto outDimY = out.dim(1);
  const auto outDimX = out.dim(2);
  const auto inChansPerGroup = plan.fwdPartition.inChansPerGroup;
  const auto outChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto tileOutHeight = tileOutYEnd - tileOutYBegin;
  const auto tileOutWidth = tileOutXEnd - tileOutXBegin;
  const auto verticesPerY = plan.fwdPartition.verticesPerTilePerYAxis;
  const auto partialType = plan.fwdPartition.getPartialType();

  if (plan.useConvolutionInstruction && kernelSize == 1) {
    const auto inZGroups = tileInZGroupEnd - tileInZGroupBegin;
    const auto contextsPerVertex =
        deviceInfo.sharedConvWeights ? deviceInfo.getNumWorkerContexts() : 1;
    for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
      for (unsigned vy = 0; vy != verticesPerY; ++vy) {
        const auto outYBegin =
            tileOutYBegin + (vy * tileOutHeight) / verticesPerY;
        const auto outYEnd =
            tileOutYBegin + ((vy + 1) * tileOutHeight) / verticesPerY;
        const auto outHeight = outYEnd - outYBegin;
        if (outHeight == 0)
          continue;
        // Add the vertex.
        const char *baseClass =
            deviceInfo.sharedConvWeights ? "poplar::SupervisorVertex" :
                                           "poplar::Vertex";
        Tensor w =
            weights[ozg].slice(
              {tileInZGroupBegin, 0, 0, 0, 0},
              {tileInZGroupEnd, 1, 1, outChansPerGroup, inChansPerGroup}
            ).flatten();
        auto v = graph.addVertex(
          fwdCS,
          templateVertex("ConvPartial1x1Out", baseClass, partialType),
          {{"weights", w}}
        );
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
        graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
        std::vector<std::vector<PartialRow>> workerPartition;
        bool mergeRows =
            stride == 1 && tileOutWidth == outDimX && tileOutWidth == outDimX;
        if (mergeRows) {
          workerPartition =
              partitionConvPartialByWorker(1, outHeight * tileOutWidth,
                                           contextsPerVertex);
        } else {
          workerPartition =
              partitionConvPartialByWorker(outHeight, tileOutWidth,
                                           contextsPerVertex);
        }
        graph.setFieldSize(v["weightReuseCount"], contextsPerVertex);
        for (unsigned i = 0; i != contextsPerVertex; ++i) {
          graph.setInitialValue(
            v["weightReuseCount"][i],
            static_cast<std::uint32_t>(workerPartition[i].size())
          );
        }
        unsigned numConvolutions = 0;
        for (unsigned izg = tileInZGroupBegin; izg != tileInZGroupEnd; ++izg) {
          for (unsigned i = 0; i != contextsPerVertex; ++i) {
            for (const auto &partialRow : workerPartition[i]) {
              if (mergeRows) {
                assert(tileOutXBegin == 0);
                const auto workerOutBegin = partialRow.begin;
                const auto workerOutEnd = partialRow.end;
                assert(inDimX == outDimX && inDimY == outDimY);
                Tensor inWindow =
                    in[izg].slice({tileOutYBegin, 0, 0},
                                  {tileOutYEnd, inDimX, inChansPerGroup})
                           .reshape({tileOutHeight * inDimX, inChansPerGroup})
                           .slice({workerOutBegin, 0},
                                  {workerOutEnd, inChansPerGroup})
                           .flatten();
                Tensor outWindow =
                    out[ozg].slice({tileOutYBegin, 0, 0},
                                   {tileOutYEnd, outDimX, outChansPerGroup})
                            .reshape({tileOutHeight * outDimX,
                                      outChansPerGroup})
                            .slice({workerOutBegin, 0},
                                   {workerOutEnd, outChansPerGroup})
                            .flatten();
                mapping.setMapping(outWindow, tile);
                graph.connect(v["in"][numConvolutions], inWindow);
                graph.connect(v["out"][numConvolutions], outWindow);
                ++numConvolutions;
              } else {
                const auto workerOutY = outYBegin + partialRow.rowNumber;
                const auto workerOutXBegin = tileOutXBegin + partialRow.begin;
                const auto workerOutXEnd = tileOutXBegin + partialRow.end;
                const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
                const auto workerInY =
                  getInputIndex(workerOutY, stride, kernelSize,
                                padding, inDimY, 0);
                assert(workerInY != ~0U);
                unsigned workerInXBegin, workerInXEnd;
                std::tie(workerInXBegin, workerInXEnd) =
                    getInputRange({workerOutXBegin, workerOutXEnd}, stride,
                                  kernelSize, padding, inDimX);
                const auto workerInWidth = workerInXEnd - workerInXBegin;
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
        }
        graph.setFieldSize(v["in"], numConvolutions);
        graph.setFieldSize(v["out"], numConvolutions);
        // Map the vertex and output.
        mapping.setMapping(v, tile);
      }
    }
  } else if (plan.useConvolutionInstruction) {
    // Zero the partial sums.
    Tensor tileOut =
        out.slice(
          {tileOutZGroupBegin, tileOutYBegin, tileOutXBegin, 0},
          {tileOutZGroupEnd, tileOutYEnd, tileOutXEnd, outChansPerGroup}
        );
    const auto outZGroups = tileOutZGroupEnd - tileOutZGroupBegin;
    Tensor tileOutFlattened =
        tileOut.reshape({outZGroups * tileOutHeight,
                         tileOutWidth * outChansPerGroup});
    const auto workersPerTile = deviceInfo.getNumWorkerContexts();
    const auto tileOutRows = tileOutFlattened.dim(0);
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
        {{"out", tileOutFlattened.slice(beginRow, endRow)}}
      );
      graph.setInitialValue(zv["dataPathWidth"], dataPathWidth);
      mapping.setMapping(zv, tile);
    }
    for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
      for (unsigned vy = 0; vy != verticesPerY; ++vy) {
        const auto outYBegin =
            tileOutYBegin + (vy * tileOutHeight) / verticesPerY;
        const auto outYEnd =
            tileOutYBegin + ((vy + 1) * tileOutHeight) / verticesPerY;
        const auto outHeight = outYEnd - outYBegin;
        if (outHeight == 0)
          continue;
        createConvPartial1x1InOutVertex(graph, mapping, deviceInfo,
                                        plan, tile,
                                        tileOutXBegin, tileOutXEnd,
                                        outYBegin, outYEnd,
                                        ozg,
                                        tileInZGroupBegin, tileInZGroupEnd,
                                        kernelSize, stride, padding,
                                        fwdCS, in, weights, out);
      }
    }
  } else {
    const auto inZGroups = tileInZGroupEnd - tileInZGroupBegin;
    for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
      assert(outChansPerGroup == 1);
      const auto z = ozg;
      for (unsigned vy = 0; vy != verticesPerY; ++vy) {
        const auto outYBegin =
            tileOutYBegin + (vy * tileOutHeight) / verticesPerY;
        const auto outYEnd =
            tileOutYBegin + ((vy + 1) * tileOutHeight) / verticesPerY;
        if (outYBegin == outYEnd)
          continue;
        assert(outYEnd - outYBegin == 1);
        const auto y = outYBegin;
        unsigned inYBegin, inYEnd, inXBegin, inXEnd;
        std::tie(inYBegin, inYEnd) =
          getInputRange(y, stride, kernelSize, padding, inDimY);
        std::tie(inXBegin, inXEnd) =
            getInputRange({tileOutXBegin, tileOutXEnd}, stride, kernelSize,
                          padding, inDimX);
        // Window into previous layer.
        const auto inWidth = inXEnd - inXBegin;
        const auto inHeight = inYEnd - inYBegin;
        // Weights that match the window.
        unsigned weightYBegin, weightYEnd;
        std::tie(weightYBegin, weightYEnd) =
          getKernelRange(y, stride, kernelSize, padding, inDimY);
        Tensor inWindow =
            in.slice(
              {tileInZGroupBegin, inYBegin, inXBegin, 0},
              {tileInZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
            ).reshape({inHeight * inZGroups,
                       inWidth * inChansPerGroup});
        Tensor w =
            weights[z].slice(
              {tileInZGroupBegin, weightYBegin, 0, 0, 0},
              {tileInZGroupEnd, weightYEnd, kernelSize, 1, inChansPerGroup}
            ).reshape({inHeight * inZGroups,
                       inChansPerGroup * kernelSize});
        Tensor outWindow = out[z][y].slice(tileOutXBegin, tileOutXEnd).flatten();
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
        graph.setInitialValue(v["padding"], padding);
        // Map the vertex and output.
        mapping.setMapping(v, tile);
        mapping.setMapping(outWindow, tile);
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
static std::vector<std::pair< unsigned, unsigned >>
getContiguousRegions(std::vector<unsigned>::iterator begin,
                     std::vector<unsigned>::iterator end)
{
  std::vector<std::pair<unsigned, unsigned>> regions;
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
  size_t resOutNumChans = resOutNumChanGroups * outChansPerGroup;
  residual = graph.addTensor(dType, {resOutNumChanGroups,
                                     outDimY, outDimX,
                                     outChansPerGroup},
                             "residual");
  //mapActivations(residual, mapping, deviceInfo);
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


Program
convolution(Graph &graph,
            IPUModelEngineBuilder::TileMapping &mapping,
            const DeviceInfo &deviceInfo,
            const ConvPlan &plan,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned outNumChans, NonLinearityType nonLinearityType,
            std::string dType,
            Tensor in, Tensor weights, Tensor biases,
            Tensor z, Tensor activations,
            ResidualMethod resMethod, Tensor resIn) {
  bool doResidual = resMethod != RESIDUAL_NONE;
  mapBiases(biases, mapping, deviceInfo, activations);
  const auto layerName = "Conv" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);
  const auto inDimY = in.dim(1);
  const auto inDimX = in.dim(2);
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  assert(outDimY = activations.dim(1));
  assert(outDimX = activations.dim(2));
  const auto outNumChanGroups = activations.dim(0);
  const auto isMultiIPU = deviceInfo.getNumIPUs() > 1;

  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto inNumChans = in.dim(0) * in.dim(3);
  const auto inChansPerGroup = plan.fwdPartition.inChansPerGroup;
  const auto partialChansPerGroup = plan.fwdPartition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;

  const auto tilesPerX = plan.fwdPartition.tilesPerXAxis;
  const auto tilesPerY = plan.fwdPartition.tilesPerYAxis;
  const auto tilesPerZ = plan.fwdPartition.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.fwdPartition.tilesPerInZGroupAxis;
  const auto partialType = plan.fwdPartition.getPartialType();

  assert(inNumChans % inChansPerGroup == 0);

  auto forwardProg = Sequence();

  const auto numInZGroups = inNumChans / inChansPerGroup;
  Tensor partials = graph.addTensor(partialType,
                                    {tilesPerInZGroup,
                                     partialNumChanGroups,
                                     outDimY,
                                     outDimX,
                                     partialChansPerGroup}, 
                                    "partials");
  ComputeSet zeroCS;
  if (plan.useConvolutionInstruction && kernelSize != 1) {
    zeroCS = graph.createComputeSet(layerName + ".zero");
    forwardProg.add(Execute(zeroCS));
  }
  ComputeSet fwdCS = graph.createComputeSet(layerName + ".fwd");
  forwardProg.add(Execute(fwdCS));
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
          const auto tile = linearizeTileIndices(izg, ox, oy, ozg, plan,
                                                 isMultiIPU);
          forwardTile(graph, mapping, deviceInfo, plan, dType,
                      tile, outXBegin, outXEnd, outYBegin, outYEnd,
                      outZGroupBegin, outZGroupEnd, inZGroupBegin, inZGroupEnd,
                      kernelSize, stride, padding,zeroCS, fwdCS,
                      in, weights, partials[izg]);
        }
      }
    }
  }
  mapWeights(weights, mapping, deviceInfo, plan);
  Tensor reduced;
  ComputeSet reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
  bool executeReduceCS = false;
  unsigned resStride; Tensor residual;
  if (doResidual) {
    std::tie(resStride, residual) =
        addResidualCalc(graph, mapping, deviceInfo, resIn, reduceCS, outDimY,
                        outDimX, outNumChans, outNumChanGroups, dType,
                        resMethod);
    executeReduceCS = true;
  }
  auto activationsMapping = computeActivationsMapping(activations, deviceInfo);
  if (tilesPerInZGroup == 1) {
    reduced = partials[0];
  } else {
    // Accumulate the partial sums.
    const auto numTiles = deviceInfo.getNumTiles();
    reduced = graph.addTensor(partialType,
                              {partialNumChanGroups, outDimY, outDimX,
                               partialChansPerGroup}, 
                              "reduced");
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
        unsigned numWorkerElems = elemEnd - elemBegin;
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
    executeReduceCS = true;
  }
  if (executeReduceCS) {
    forwardProg.add(Execute(reduceCS));
  }

  // Apply the non linearity and write back results in the layout desired by
  // the next layer. Each vertex handles outChansPerGroup output elements.
  // TODO: This step could be merged with the reduction step above.
  ComputeSet completionCS =
     graph.createComputeSet(layerName + ".fwd.complete");
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
      auto v = graph.addVertex(completionCS,
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
      graph.setFieldSize(v["z"], numGroups);
      graph.setFieldSize(v["in"],
                         numGroups * outChansPerGroup / partialChanChunkSize);
      unsigned numIn = 0;
      unsigned numResUsed = 0;
      for (auto group = groupBegin; group != groupEnd; ++group) {
        auto outChanGroup = group / (outDimX * outDimY);
        auto y = group % (outDimX * outDimY) / outDimX;
        auto x = group % outDimX;
        auto out = activations[outChanGroup][y][x];
        auto zz = z[outChanGroup][y][x];
        graph.connect(v["out"][group - groupBegin], out);
        graph.connect(v["z"][group - groupBegin], zz);
        Tensor reducedChans = reduced.slice(
           {0, y, x, 0},
           {partialNumChanGroups, y + 1, x + 1, partialChansPerGroup}
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
  forwardProg.add(Execute(completionCS));
  return forwardProg;
}

static std::uint64_t getNumberOfMACs(unsigned outDimY, unsigned outDimX,
                                     unsigned outNumChans,
                                     unsigned kernelSize, unsigned stride,
                                     unsigned padding,
                                     unsigned inDimY, unsigned inDimX,
                                     unsigned inNumChans) {
  std::uint64_t numMACs = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                               padding, inDimY);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 padding, inDimX);
      const auto width = inXEnd - inXBegin;
      numMACs += width * height * outNumChans * inNumChans;
    }
  }
  return numMACs;
}

static std::uint64_t getNumberOfAdds(unsigned outDimY, unsigned outDimX,
                                     unsigned outNumChans, bool doResidual) {
  if (!doResidual)
    return 0;

  // An addition is required to add in the residual information
  return outNumChans * outDimX * outDimY;
}

uint64_t getFlops(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSize, unsigned stride, unsigned padding,
                  unsigned outNumChans, bool doResidual) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  return 2 * getNumberOfMACs(outDimY, outDimX, outNumChans,
                             kernelSize, stride, padding,
                             inDimY, inDimX, inNumChans) +
         getNumberOfAdds(outDimY, outDimX, outNumChans, doResidual);
}

double getPerfectCycleCount(const DeviceInfo &deviceInfo,
                            std::string dType,
                            unsigned inDimY, unsigned inDimX,
                            unsigned inNumChans,
                            unsigned kernelSize, unsigned stride,
                            unsigned padding,
                            unsigned outNumChans, bool doResidual) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  const auto numTiles = deviceInfo.getNumTiles();
  auto numMacs = getNumberOfMACs(outDimY, outDimX, outNumChans, kernelSize,
                                 stride, padding, inDimY, inDimX,
                                 inNumChans);
  auto numAdds = getNumberOfAdds(outDimY, outDimX, outNumChans, doResidual);
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
  bool canUseConvolutions = true;
  if (stride >= (1 << 4))
    canUseConvolutions = false;
  if (inNumChans < deviceInfo.getInputChannelsPerConvUnit())
     canUseConvolutions = false;
  if (outNumChans % deviceInfo.fp16AccumConvUnitsPerTile != 0)
    canUseConvolutions = false;
  if (outNumChans % deviceInfo.fp32AccumConvUnitsPerTile != 0)
    canUseConvolutions = false;
  auto macsPerCycle =
      canUseConvolutions ? convUnitsPerTile * halfVectorWidth :
                           halfVectorWidth;
  auto macCycles = static_cast<double>(numMacs) / (macsPerCycle * numTiles);
  auto addCycles = static_cast<double>(numAdds) / (halfVectorWidth * numTiles);
  return macCycles + addCycles;
}

Program convolutionBwdNonLinearity(Graph &graph,
                                   IPUModelEngineBuilder::TileMapping &mapping,
                                   const DeviceInfo &deviceInfo,
                                   std::string dType,
                                   Tensor deltasIn, Tensor z, Tensor zDeltas,
                                   NonLinearityType nonLinearityType) {
  auto bwdNonLinearityCS = graph.createComputeSet("conv.bwd.nonLinearity");
  auto v = graph.addVertex(bwdNonLinearityCS,
                           templateVertex("NonLinearityBwd", dType),
                           {{"deltasIn", deltasIn.flatten()},
                            {"z", z.flatten()},
                            {"deltasOut", zDeltas.flatten()},
                           });
  graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
  return Execute(bwdNonLinearityCS);
}

Program convolutionBackward(Graph &graph,
                            IPUModelEngineBuilder::TileMapping &mapping,
                            const DeviceInfo &deviceInfo,
                            std::string dType,
                            Tensor zDeltas, Tensor weights,
                            Tensor deltasOut,
                            unsigned kernelSize, unsigned stride,
                            unsigned padding) {
  const auto partialType = "float";
  const auto layerName = "Conv" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);
  const auto inNumChanGroups = deltasOut.dim(0);
  const auto inDimY = deltasOut.dim(1);
  const auto inDimX = deltasOut.dim(2);
  const auto inChansPerGroup = deltasOut.dim(3);
  const auto partialChansPerGroup = weights.dim(4);
  const auto inNumChans = inNumChanGroups * inChansPerGroup;
  const auto outNumChanGroups = zDeltas.dim(0);
  const auto outDimY = zDeltas.dim(1);
  const auto outDimX = zDeltas.dim(2);
  const auto outChansPerGroup = zDeltas.dim(3);
  auto partials = graph.addTensor(partialType,
                                  {outNumChanGroups,
                                   inNumChans,
                                   kernelSize,
                                   kernelSize,
                                   inDimY, inDimX});
  auto zeroCS = graph.createComputeSet(layerName + ".bwd.zero");
  graph.addVertex(zeroCS, templateVertex("Zero", partialType),
                  {{"out",partials.flatten()}});
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  for (unsigned outGroup = 0; outGroup < outNumChanGroups; ++outGroup) {
    for (unsigned inChan = 0; inChan < inNumChans; ++inChan) {
      for (unsigned wy = 0; wy < kernelSize; ++wy) {
        for (unsigned wx = 0; wx < kernelSize; ++wx) {
          unsigned convOutYBegin, convOutYEnd;
          std::tie(convOutYBegin, convOutYEnd) =
              getOutputRange({0, outDimY}, stride, kernelSize,
                             padding, inDimY, wy);
          const auto convOutHeight = convOutYEnd - convOutYBegin;
          if (convOutHeight == 0) {
            std::abort();
            continue;
          }
          unsigned convOutXBegin, convOutXEnd;
          std::tie(convOutXBegin, convOutXEnd) =
              getOutputRange({0, outDimX}, stride, kernelSize,
                             padding, inDimX, wx);
          const auto convOutWidth = convOutXEnd - convOutXBegin;
          if (convOutWidth == 0)
            continue;
          unsigned convInYBegin, convInYEnd;
          std::tie(convInYBegin, convInYEnd) =
              getInputRange({0, outDimY}, stride, kernelSize,
                            padding, inDimY,
                            wy);
          const auto convInHeight = convInYEnd - convInYBegin;
          if (convInHeight == 0)
            continue;
          unsigned convInXBegin, convInXEnd;
          std::tie(convInXBegin, convInXEnd) =
              getInputRange({0, outDimX}, stride, kernelSize, padding,
                                inDimX, wx);
          const auto convInWidth = convInXEnd - convInXBegin;
          if (convInWidth == 0)
            continue;
          auto out =
              partials[outGroup][inChan][wy][wx]
                  .slice({convInYBegin, convInXBegin},
                         {convInYEnd, convInXEnd})
                  .reshape({convInHeight, convInWidth});
          auto in = zDeltas[outGroup]
                         .slice({convOutYBegin, convOutXBegin, 0},
                                {convOutYEnd, convOutXEnd, outChansPerGroup})
                         .reshape({convOutHeight,
                                   convOutWidth * outChansPerGroup});
          auto v = graph.addVertex(bwdCS,
                                   templateVertex("ConvBwd", dType,
                                                  partialType),
                                   {{"in", in},
                                    {"out", out}});
          graph.setFieldSize(v["weights"], outChansPerGroup);
          for (unsigned i = 0; i < outChansPerGroup; ++i) {
            auto outChan = outGroup * outChansPerGroup + i;
            Tensor w;
            w = weights[outChan / partialChansPerGroup]
                       [inChan / inChansPerGroup]
                       [wy]
                       [wx]
                       [outChan % partialChansPerGroup]
                       [inChan % inChansPerGroup];
            graph.connect(v["weights"][i], w);
          }
        }
      }
    }
  }
  auto reduced = graph.addTensor(partialType,
                                 {inNumChans, inDimY, inDimX});
  auto reduceCS = graph.createComputeSet(layerName + ".bwd.reduce");
  for (unsigned inChan = 0; inChan < inNumChans; ++inChan) {
    auto p = partials.slice({0, inChan, 0, 0, 0, 0},
                            {outNumChanGroups, inChan + 1, kernelSize,
                             kernelSize, inDimX, inDimY})
                     .reshape({outNumChanGroups * kernelSize * kernelSize,
                               inDimX * inDimY});
    graph.addVertex(reduceCS, templateVertex("ConvReduceBwd", partialType),
                    {{"out", reduced[inChan].flatten()},
                     {"partials", p}});
  }

  auto completeCS = graph.createComputeSet(layerName + ".bwd.complete");
  for (unsigned inChanGroup = 0; inChanGroup < inNumChanGroups; ++inChanGroup) {
    for (unsigned y = 0; y < inDimY; ++y) {
      for (unsigned x = 0; x < inDimX; ++x) {
        auto inChanBegin = inChanGroup * inChansPerGroup;
        auto inChanEnd = (inChanGroup + 1) * inChansPerGroup;
        auto in = reduced.slice({inChanBegin, y, x},
                                {inChanEnd, y+1, x+1})
                         .flatten();
        graph.addVertex(completeCS,
                        templateVertex("ConvCompleteBwd", partialType, dType),
                        {{"out", deltasOut[inChanGroup][y][x].flatten()},
                         {"in", in}});
      }
    }
  }

  return Sequence(Execute(zeroCS), Execute(bwdCS),
                  Execute(reduceCS), Execute(completeCS));
}

Program
convolutionWeightUpdate(Graph &graph,
                        IPUModelEngineBuilder::TileMapping &mapping,
                        const DeviceInfo &deviceInfo,
                        std::string dType,
                        Tensor zDeltas, Tensor weights, Tensor biases,
                        Tensor activations,
                        unsigned kernelSize, unsigned stride,
                        unsigned padding, float learningRate) {
  const auto inNumChanGroups = activations.dim(0);
  const auto inDimY = activations.dim(1);
  const auto inDimX = activations.dim(2);
  const auto inChansPerGroup = activations.dim(3);
  const auto partialChansPerGroup = weights.dim(4);
  const auto inNumChans = inNumChanGroups * inChansPerGroup;
  const auto outNumChanGroups = zDeltas.dim(0);
  const auto outDimY = zDeltas.dim(1);
  const auto outDimX = zDeltas.dim(2);
  const auto outChansPerGroup = zDeltas.dim(3);
  const auto outNumChans = outChansPerGroup * outNumChanGroups;
  const auto layerName = "Conv" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);

  auto wPartials = graph.addTensor(dType,
                                   {outNumChans, outDimY, outDimX,
                                    kernelSize, kernelSize, inNumChans});
  auto zeroCS = graph.createComputeSet(layerName + ".weight_update.zero");
  graph.addVertex(zeroCS, templateVertex("Zero", dType),
                  {{"out",wPartials.flatten()}});
  auto partialCS = graph.createComputeSet(layerName + ".weight_update.partial");
  for (unsigned outChanGroup = 0; outChanGroup < outNumChanGroups; ++outChanGroup) {
    for (unsigned y = 0; y < outDimY; ++y) {
      for (unsigned x = 0; x < outDimX; ++x) {
        for (unsigned outChanInGroup = 0; outChanInGroup < outChansPerGroup; ++outChanInGroup) {
          for (unsigned wy = 0; wy < kernelSize; ++wy) {
            for (unsigned wx = 0; wx < kernelSize; ++wx) {
              auto inX = getInputIndex(x, stride, kernelSize,
                                       padding, inDimX, wx);
              if (inX == ~0U)
                continue;
              auto inY = getInputIndex(y, stride, kernelSize,
                                       padding, inDimY, wy);
              if (inY == ~0U)
                continue;
              auto outChan = outChanGroup * outChansPerGroup + outChanInGroup;
              auto w = wPartials[outChan][y][x][wy][wx].flatten();
              auto d = zDeltas[outChanGroup][y][x][outChanInGroup];
              auto ii = activations.slice({0, inY, inX, 0},
                                          {inNumChanGroups, inY + 1,
                                           inX + 1, inChansPerGroup})
                                   .flatten();
              auto v = graph.addVertex(partialCS,
                                       templateVertex("ConvPartialWeightUpdate",
                                                      dType),
                                       {{"d", d},
                                        {"in", ii},
                                        {"weightUpdates", w}});
            }
          }
        }
      }
    }
  }
  auto reduceCS = graph.createComputeSet(layerName + ".weight_update.reduce");

  for (unsigned inChan = 0; inChan < inNumChans; ++inChan) {
    for (unsigned outChan = 0; outChan < outNumChans; ++outChan) {
      for (unsigned wy = 0; wy < kernelSize; ++wy) {
        for (unsigned wx = 0; wx < kernelSize; ++wx) {
          auto w = weights[outChan / partialChansPerGroup]
                          [inChan / inChansPerGroup]
                          [wy][wx]
                          [outChan % partialChansPerGroup]
                          [inChan % inChansPerGroup];
          auto in =
              wPartials[outChan].slice({0, 0, wy, wx, inChan},
                                       {outDimY, outDimX,
                                        wy + 1, wx + 1,
                                        inChan + 1})
                                .flatten();
          auto v = graph.addVertex(reduceCS,
                                   templateVertex("ConvWeightUpdateReduce",
                                                  dType),
                                   {{"weight", w}, {"partials", in}});
          graph.setInitialValue(v["eta"], learningRate);
        }
      }
    }
  }

  for (unsigned outChan = 0; outChan < outNumChans; ++outChan) {
    const auto outChanGroup = outChan / outChansPerGroup;
    const auto outChanInGroup = outChan % outChansPerGroup;
    auto in = zDeltas.slice({outChanGroup, 0, 0, outChanInGroup},
                            {outChanGroup + 1, outDimY, outDimX,
                             outChanInGroup + 1}).flatten();
    auto v = graph.addVertex(reduceCS,
                             templateVertex("ConvBiasUpdate", dType),
                             {{"bias", biases[outChan]}, {"deltas", in}});
    graph.setInitialValue(v["eta"],
                          learningRate);
  }


  return Sequence(Execute(zeroCS), Execute(partialCS), Execute(reduceCS));
}




} // namespace conv
