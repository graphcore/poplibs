#include "FullyConnected.hpp"
#include "FullyConnectedPlan.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexTemplates.hpp"
#include "ActivationMapping.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;

namespace fc {

std::pair<Tensor, Tensor>
createParams(Graph &graph, std::string dType, unsigned inSize,
             unsigned outSize) {
  auto weights = graph.addTensor(dType, {outSize, inSize}, "weights");
  auto biases = graph.addTensor(dType, {outSize}, "biases");
  return {weights, biases};
}

Program
fullyConnected(Graph &graph,
               IPUModelEngineBuilder::TileMapping &mapping,
               DeviceInfo &deviceInfo,
               unsigned size, NonLinearityType nonLinearityType,
               std::string dType,
               Tensor in0, Tensor weights,
               Tensor biases,
               Tensor z, Tensor activations,
               const Plan &plan) {
  const auto layerName = "FullyConnected" + std::to_string(size);
  Tensor in = in0.flatten();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto prevSize = in.numElements();

  const auto numRows = size;
  const auto numCols = prevSize;
  const auto numIPUs = deviceInfo.getNumIPUs();
  const auto tilesPerIPU = deviceInfo.getTilesPerIPU();
  const auto &activationsOutMapping = plan.outputMapping;
  bool isFloat = dType == "float";
  assert(isFloat || dType == "half");

  const auto &ipuPartition = plan.ipuPartition;
  Tensor partials = graph.addTensor("float", {numRows,
                                              ipuPartition.tilesPerRow},
                                    "partials");
  auto prog = Sequence();

  ComputeSet dotProductCS = graph.createComputeSet(layerName + ".fwd");
  prog.add(Execute(dotProductCS));

  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    const auto ipuBeginRow = activationsOutMapping[ipu * tilesPerIPU];
    const auto ipuEndRow = activationsOutMapping[(ipu + 1) * tilesPerIPU];
    const auto ipuRows = ipuEndRow - ipuBeginRow;
    for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
      const auto tileRowBegin = ipuBeginRow + (tileY * ipuRows) /
                                ipuPartition.tilesPerColumn;
      const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
                              ipuPartition.tilesPerColumn;
      if (tileRowBegin == tileRowEnd)
        continue;
      for (unsigned tileX = 0; tileX != ipuPartition.tilesPerRow; ++tileX) {
        const auto tile = ipu * tilesPerIPU +
                          tileY * ipuPartition.tilesPerRow +
                          tileX;
        const auto j = tileX;
        const auto beginElement =
            (numCols * j) / ipuPartition.tilesPerRow;
        const auto endElement =
            (numCols * (j + 1)) / ipuPartition.tilesPerRow;
        if (beginElement == endElement)
          continue;
        for (unsigned i = tileRowBegin; i != tileRowEnd; ++i) {
            Tensor partialIn = in.slice(beginElement, endElement);
            Tensor partialWeights = weights[i].slice(beginElement, endElement);
            auto v =
                graph.addVertex(dotProductCS,
                                templateVertex("FullyConnectedPartial", dType),
                                {{"in", partialIn},
                                 {"weights", partialWeights},
                                 {"out", partials[i][j]}});
            graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
            mapping.setMapping(partialWeights, tile);
            mapping.setMapping(partials[i][j], tile);
            mapping.setMapping(v, tile);
        }
      }
    }
  }
  ComputeSet reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
  prog.add(Execute(reduceCS));
  const auto numTiles = numIPUs * tilesPerIPU;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto activationsBegin = activationsOutMapping[tile];
    const auto activationsEnd = activationsOutMapping[tile + 1];
    for (unsigned i = activationsBegin; i != activationsEnd; ++i) {
      // Sum the partial sums.
      auto v =
          graph.addVertex(reduceCS,
                          templateVertex("FullyConnectedReduce",
                                         dType),
                          {{"partials", partials[i]},
                           {"bias", biases[i]},
                           {"zOut", z[i]},
                           {"activationOut", activations[i]}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      mapping.setMapping(v, tile);
      mapping.setMapping(biases[i], tile);
    }
  }
  return prog;
}

uint64_t getNumFlops(unsigned inSize, unsigned outSize) {
  return 2 * inSize * outSize;
}

double getPerfectCycleCount(const DeviceInfo &deviceInfo,
                            unsigned inSize, unsigned outSize,
                            std::string dType) {
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getNumFlops(inSize, outSize);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (2 * vectorWidth * numTiles);
}

/// Given a mapping of data to tiles, use the specified builder function to
/// create vertices that operate on that data. Each vertex operates on data
/// that is local to the tile it runs on. The number of vertices per tile is
/// decided based on the number of worker contexts.
template <class Builder>
void buildTransform(const std::vector<unsigned> &tileMapping,
                    DeviceInfo &deviceInfo, Builder &&builder) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto workersPerTile = deviceInfo.getNumWorkerContexts();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileElementBegin = tileMapping[tile];
    const auto tileElementEnd = tileMapping[tile + 1];
    const auto tileNumElements = tileElementEnd - tileElementBegin;
    if (tileNumElements == 0)
      continue;
    const auto maxElementsPerWorker =
      (tileNumElements + workersPerTile - 1) / workersPerTile;
    const auto verticesToCreate =
      (tileNumElements + maxElementsPerWorker - 1) / maxElementsPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto elementBegin =
          tileElementBegin +
          (vertex * tileNumElements) / verticesToCreate;
      const auto elementEnd =
          tileElementBegin +
          ((vertex + 1) * tileNumElements) / verticesToCreate;
      builder(elementBegin, elementEnd, tile);
    }
  }
}

Program
fullyConnectedBwdNonLinearity(Graph &graph,
                              IPUModelEngineBuilder::TileMapping &mapping,
                              DeviceInfo &deviceInfo,
                              std::string dType,
                              Tensor z, Tensor deltasIn,
                              Tensor zDeltas,
                              NonLinearityType nonLinearityType,
                              const Plan &) {
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto deltasInMapping = computeActivationsMapping(z, deviceInfo);
  deltasIn = deltasIn.flatten();
  const auto size = deltasIn.numElements();
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto bwdNonLinearityCS =
      graph.createComputeSet(layerName + ".bwd.nonLinearity");

  buildTransform(deltasInMapping, deviceInfo, [&](unsigned deltaBegin,
                                                  unsigned deltaEnd,
                                                  unsigned tile) {
    auto v =
        graph.addVertex(bwdNonLinearityCS,
                        templateVertex("NonLinearityBwd", dType),
                        {{"deltasIn", deltasIn.slice(deltaBegin, deltaEnd)},
                         {"z", z.slice(deltaBegin, deltaEnd)},
                         {"deltasOut", zDeltas.slice(deltaBegin, deltaEnd)},
                        });
    graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    mapping.setMapping(v, tile);
  });
  return Execute(bwdNonLinearityCS);
}

Program fullyConnectedBackward(Graph &graph,
                               IPUModelEngineBuilder::TileMapping &mapping,
                               DeviceInfo &deviceInfo,
                               std::string dType,
                               Tensor zDeltas,
                               Tensor weights, Tensor deltasOut,
                               const Plan &plan) {
  const auto size = static_cast<unsigned>(zDeltas.numElements());
  const auto prevSize = static_cast<unsigned>(weights.dim(1));
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto prog = Sequence();
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  prog.add(Execute(bwdCS));
  const auto &ipuPartition = plan.ipuPartition;
  const auto numIPUs = deviceInfo.getNumIPUs();
  const auto tilesPerIPU = deviceInfo.getTilesPerIPU();
  const auto numCols = prevSize;
   Tensor partials =
       graph.addTensor("float", {numCols, numIPUs, ipuPartition.tilesPerColumn},
                       "partials");
  const auto &activationsOutMapping = plan.outputMapping;
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    const auto ipuBeginRow = activationsOutMapping[ipu * tilesPerIPU];
    const auto ipuEndRow = activationsOutMapping[(ipu + 1) * tilesPerIPU];
    const auto ipuRows = ipuEndRow - ipuBeginRow;
    for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
      const auto tileRowBegin = ipuBeginRow + (tileY * ipuRows) /
                                ipuPartition.tilesPerColumn;
      const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
                              ipuPartition.tilesPerColumn;
      if (tileRowBegin == tileRowEnd)
        continue;
      for (unsigned tileX = 0; tileX != ipuPartition.tilesPerRow; ++tileX) {
        const auto tile = ipu * tilesPerIPU +
                          tileY * ipuPartition.tilesPerRow +
                          tileX;
        const auto j = tileY;
        const auto beginElement =
            (numCols * tileX) / ipuPartition.tilesPerRow;
        const auto endElement =
            (numCols * (tileX + 1)) / ipuPartition.tilesPerRow;
        if (beginElement == endElement)
          continue;
        const auto vectorWidth =
            dType == "float" ? deviceInfo.getFloatVectorWidth() :
                               deviceInfo.getHalfVectorWidth();
        for (unsigned i = beginElement; i < endElement; i += vectorWidth) {
          const auto vectorNumElements = std::min(endElement - i, vectorWidth);
          auto w = weights.slice({tileRowBegin, i},
                                 {tileRowEnd, i + vectorNumElements});
          Tensor inWindow = zDeltas.slice(tileRowBegin, tileRowEnd);
          Tensor outWindow =
              partials.slice({i, ipu, j},
                             {i + vectorNumElements, ipu + 1, j + 1}).flatten();
          auto v = graph.addVertex(bwdCS,
                                   templateVertex("FullyConnectedBwd",
                                                  dType),
                                   {{"in", inWindow},
                                    {"weights", w},
                                    {"out", outWindow},
                                   });
          mapping.setMapping(v, tile);
          mapping.setMapping(outWindow, tile);
        }
      }
    }
  }
  const auto deltasOutMapping =
      computeActivationsMapping(deltasOut, deviceInfo);
  deltasOut = deltasOut.flatten();
  ComputeSet intraIPUReduce = graph.createComputeSet(layerName + ".bwd.reduce");
  prog.add(Execute(intraIPUReduce));
  Tensor intraIPUPartialSums;
  if (numIPUs > 1) {
    Tensor intraIPUPartialSums =
        graph.addTensor("float", {size, numIPUs}, "ipu_partials");
  } else {
    intraIPUPartialSums = deltasOut.reshape({deltasOut.numElements(), 1});
  }
  // Sum the partial sums on each IPU.
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    for (unsigned resultTile = 0; resultTile != numTiles; ++resultTile) {
      const auto deltasBegin = deltasOutMapping[resultTile];
      const auto deltasEnd = deltasOutMapping[resultTile + 1];
      if (deltasBegin == deltasEnd)
        continue;
      const auto tile = resultTile % tilesPerIPU;
      for (unsigned i = deltasBegin; i != deltasEnd; ++i) {
        const char *outType = numIPUs > 1 ? "float" : dType.c_str();
        auto v =
            graph.addVertex(intraIPUReduce,
                            templateVertex("FullyConnectedBwdReduce", "float",
                                                                      outType),
                            {{"partials", partials[i][ipu]},
                             {"out", intraIPUPartialSums[i][ipu]}});
        mapping.setMapping(v, tile);
        graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
        if (numIPUs > 1) {
          mapping.setMapping(intraIPUPartialSums[i][ipu], tile);
        }
      }
    }
  }
  // Sum the partial sums from different IPUs.
  if (numIPUs > 1) {
    ComputeSet interIPUReduce =
        graph.createComputeSet(layerName + ".bwd.reduce2");
    prog.add(Execute(interIPUReduce));
    const auto numTiles = deviceInfo.getNumTiles();
    for (unsigned tile = 0; tile != numTiles; ++tile) {
      const auto deltasBegin = deltasOutMapping[tile];
      const auto deltasEnd = deltasOutMapping[tile + 1];
      if (deltasBegin == deltasEnd)
        continue;
      for (unsigned i = deltasBegin; i != deltasEnd; ++i) {
        auto v =
            graph.addVertex(intraIPUReduce,
                            templateVertex("FullyConnectedBwdReduce", "float",
                                           dType),
                            {{"partials", intraIPUPartialSums[i]},
                             {"out", deltasOut[i]}});
        mapping.setMapping(v, tile);
        graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      }
    }
  }
  return prog;
}

Program
fullyConnectedWeightUpdate(Graph &graph,
                           IPUModelEngineBuilder::TileMapping &mapping,
                           DeviceInfo &deviceInfo,
                           std::string dType,
                           Tensor zDeltas,
                           Tensor activations,
                           Tensor weights, Tensor biases,
                           float learningRate,
                           const Plan &plan) {
  const auto &activationsOutMapping = plan.outputMapping;
  activations = activations.flatten();
  const auto size = zDeltas.numElements();
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto cs = graph.createComputeSet(layerName + ".weight_update");

  const auto numCols = activations.numElements();
  const auto numIPUs = deviceInfo.getNumIPUs();
  const auto tilesPerIPU = deviceInfo.getTilesPerIPU();
  const auto &ipuPartition = plan.ipuPartition;
  // Update the weights.
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    const auto ipuBeginRow = activationsOutMapping[ipu * tilesPerIPU];
    const auto ipuEndRow = activationsOutMapping[(ipu + 1) * tilesPerIPU];
    const auto ipuRows = ipuEndRow - ipuBeginRow;
    for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
      const auto tileRowBegin = ipuBeginRow + (tileY * ipuRows) /
                                ipuPartition.tilesPerColumn;
      const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
                              ipuPartition.tilesPerColumn;
      if (tileRowBegin == tileRowEnd)
        continue;
      for (unsigned tileX = 0; tileX != ipuPartition.tilesPerRow; ++tileX) {
        const auto tile = ipu * tilesPerIPU +
                          tileY * ipuPartition.tilesPerRow +
                          tileX;
        const auto j = tileX;
        const auto beginElement =
            (numCols * j) / ipuPartition.tilesPerRow;
        const auto endElement =
            (numCols * (j + 1)) / ipuPartition.tilesPerRow;
        if (beginElement == endElement)
          continue;
        for (unsigned i = tileRowBegin; i != tileRowEnd; ++i) {
          auto w = weights[i].slice(beginElement, endElement);
          auto actWindow = activations.slice(beginElement, endElement);
          auto v = graph.addVertex(cs,
                                   templateVertex("FullyConnectedWeightUpdate",
                                                  dType),
                                   {{"d", zDeltas[i]},
                                    {"weights", w},
                                    {"in", actWindow}});
          graph.setInitialValue(v["eta"],
                                learningRate);
          mapping.setMapping(v, tile);
        }
      }
    }
  }

  // Update the biases.
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto activationsBegin = activationsOutMapping[tile];
    const auto activationsEnd = activationsOutMapping[tile + 1];
    for (unsigned i = activationsBegin; i != activationsEnd; ++i) {
      // Sum the partial sums.
      auto v =
          graph.addVertex(cs,
                          templateVertex("FullyConnectedBiasUpdate",
                                         dType),
                          {{"d", zDeltas[i]},
                           {"bias", biases[i]}});
      graph.setInitialValue(v["eta"],
                            learningRate);
      mapping.setMapping(v, tile);
    }
  }
  return Execute(cs);
}


}

