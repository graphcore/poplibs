#include "popnn/FullyConnected.hpp"
#include "popnn/FullyConnectedPlan.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexTemplates.hpp"
#include "popnn/ActivationMapping.hpp"
#include "popnn/exceptions.hpp"
#include "Util.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;

namespace fc {

std::pair<Tensor, Tensor>
createParams(Graph &graph, std::string dType, unsigned inSize,
             unsigned outSize) {
  auto weights = graph.addTensor(dType, {outSize, inSize}, "fcWeights");
  auto biases = graph.addTensor(dType, {outSize}, "fcBiases");
  return {weights, biases};
}

Program
fullyConnected(Graph &graph,
               unsigned size, NonLinearityType nonLinearityType,
               Tensor in0, Tensor weights,
               Tensor biases, Tensor activations,
               const Plan &plan,
               const std::string &debugPrefix) {
  const auto dType = graph.getTensorElementType(in0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto layerName = debugPrefix
                         + "/FullyConnected" + std::to_string(size);
  const auto batchSize = in0.dim(0);
  Tensor in = in0.reshape({batchSize, in0.numElements() / batchSize});
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto prevSize = in[0].numElements();

  const auto numRows = size;
  const auto numCols = prevSize;
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
  assert(dType == "float" || dType == "half");
  const auto &ipuPartition = plan.ipuPartition;
  auto prog = Sequence();
  ComputeSet dotProductCS = graph.addComputeSet(layerName + "/Fwd/DotProd");
  prog.add(Execute(dotProductCS));
  ComputeSet reduceCS = graph.addComputeSet(layerName + "/Fwd/Reduce");
  prog.add(Execute(reduceCS));
  // Iterate through the batch add to the same compute set
  // (i.e. execute the batch in parallel).
  for (unsigned b = 0; b < batchSize; ++b) {
    const auto &activationsOutMapping =
      computeActivationsMapping(graph, activations[b], b, batchSize);
    Tensor partials = graph.addTensor("float", {numRows,
                                                ipuPartition.tilesPerRow},
                                      "partials");
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
            Tensor partialIn = in[b].slice(beginElement, endElement);
            Tensor partialWeights = weights[i].slice(beginElement, endElement);
            auto v =
                graph.addVertex(dotProductCS,
                                templateVertex("popnn::FullyConnectedPartial",
                                               dType),
                                {{"in", partialIn},
                                 {"weights", partialWeights},
                                 {"out", partials[i][j]}});
            graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
            graph.setTileMapping(partialWeights, tile);
            graph.setTileMapping(partials[i][j], tile);
            graph.setTileMapping(v, tile);
          }
        }
      }
    }
    const auto numTiles = numIPUs * tilesPerIPU;
    for (unsigned tile = 0; tile != numTiles; ++tile) {
      const auto activationsBegin = activationsOutMapping[tile];
      const auto activationsEnd = activationsOutMapping[tile + 1];
      for (unsigned i = activationsBegin; i != activationsEnd; ++i) {
        // Sum the partial sums.
        auto v =
            graph.addVertex(reduceCS,
                            templateVertex("popnn::FullyConnectedReduce",
                                           dType),
        {{"partials", partials[i]},
         {"bias", biases[i]},
         {"activationOut", activations[b][i]}});
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setTileMapping(v, tile);
        graph.setTileMapping(biases[i], tile);
      }
    }
  }
  return prog;
}

uint64_t getNumFlops(unsigned batchSize, unsigned inSize, unsigned outSize) {
  return batchSize * (2 * inSize * outSize);
}

uint64_t getFwdFlops(unsigned batchSize, unsigned inSize, unsigned outSize) {
  return getNumFlops(batchSize, inSize, outSize);
}

uint64_t getBwdFlops(unsigned batchSize, unsigned inSize, unsigned outSize) {
  return getNumFlops(batchSize, inSize, outSize);
}

uint64_t getWuFlops(unsigned batchSize, unsigned inSize, unsigned outSize) {
  return getNumFlops(batchSize, inSize, outSize);
}

static double getPerfectCycleCount(const Graph &graph,
                                   unsigned batchSize,
                                   unsigned inSize, unsigned outSize,
                                   std::string dType) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getNumFlops(batchSize, inSize, outSize);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (2 * vectorWidth * numTiles);
}

double getFwdPerfectCycleCount(const Graph &graph,
                               unsigned batchSize,
                               unsigned inSize, unsigned outSize,
                               std::string dType) {
  return getPerfectCycleCount(graph, batchSize, inSize, outSize, dType);
}

double getBwdPerfectCycleCount(const Graph &graph,
                               unsigned batchSize,
                               unsigned inSize, unsigned outSize,
                               std::string dType) {
  return getPerfectCycleCount(graph, batchSize, inSize, outSize, dType);
}

double getWuPerfectCycleCount(const Graph &graph,
                              unsigned batchSize,
                              unsigned inSize, unsigned outSize,
                              std::string dType) {
  return getPerfectCycleCount(graph, batchSize, inSize, outSize, dType);
}

Program fullyConnectedBackward(Graph &graph,
                               Tensor zDeltas,
                               Tensor weights, Tensor deltasOut,
                               const Plan &plan,
                               const std::string &debugPrefix) {
  const auto dType = graph.getTensorElementType(zDeltas);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto batchSize = zDeltas.dim(0);
  const auto size = static_cast<unsigned>(zDeltas[0].numElements());
  const auto prevSize = static_cast<unsigned>(weights.dim(1));
  const auto layerName = debugPrefix + "/FullyConnected" + std::to_string(size);
  const auto &ipuPartition = plan.ipuPartition;
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
  const auto numCols = prevSize;
  auto prog = Sequence();
  auto bwdCS = graph.addComputeSet(layerName + "/Bwd/DotProd");
  prog.add(Execute(bwdCS));
  ComputeSet intraIPUReduce = graph.addComputeSet(layerName + "/Bwd/Reduce");
  prog.add(Execute(intraIPUReduce));
  ComputeSet interIPUReduce;

  if (numIPUs > 1) {
    interIPUReduce =
        graph.addComputeSet(layerName + "/Bwd/Reduce2");
    prog.add(Execute(interIPUReduce));
  }
  for (unsigned b = 0; b < batchSize; ++b) {
    Tensor partials =
        graph.addTensor("float", {numCols, numIPUs,
                                  ipuPartition.tilesPerColumn},
                        "partials");

    auto activationsOutMapping = computeActivationsMapping(graph,
                                                           zDeltas[b],
                                                           b, batchSize);
    for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
      const auto ipuBeginRow = activationsOutMapping[ipu * tilesPerIPU];
      const auto ipuEndRow = activationsOutMapping[(ipu + 1) * tilesPerIPU];
      const auto ipuRows = ipuEndRow - ipuBeginRow;

      for (unsigned tileY = 0; tileY != ipuPartition.tilesPerColumn; ++tileY) {
        const auto tileRowBegin = ipuBeginRow + (tileY * ipuRows) /
                                  ipuPartition.tilesPerColumn;
        const auto tileRowEnd = ipuBeginRow + ((tileY + 1) * ipuRows) /
                                ipuPartition.tilesPerColumn;

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
            const auto vectorNumElements = std::min(endElement - i,
                                                    vectorWidth);

            Tensor outWindow =
                partials.slice({i, ipu, j},
                               {i + vectorNumElements, ipu + 1, j + 1})
                        .flatten();

            if (tileRowBegin == tileRowEnd) {
              auto vZ = graph.addVertex(bwdCS,
                                        templateVertex("popnn::Zero", "float"));
              graph.setInitialValue(vZ["dataPathWidth"],
                                    deviceInfo.dataPathWidth);

              graph.connect(vZ["out"], outWindow);
              graph.setTileMapping(vZ, tile);
            } else {
              auto w = weights.slice({ tileRowBegin, i },
                                     {tileRowEnd, i + vectorNumElements});
              Tensor inWindow = zDeltas[b].slice(tileRowBegin, tileRowEnd);


              auto v = graph.addVertex(
                            bwdCS,
                            templateVertex("popnn::FullyConnectedBwd",
                                           dType),
                            {{"in", inWindow},
                             {"weights", w},
                             {"out", outWindow},
                             });
              graph.setTileMapping(v, tile);
            }
            graph.setTileMapping(outWindow, tile);
          }
        }
      }
    }
    const auto deltasOutMapping =
        computeActivationsMapping(graph, deltasOut[b], b, batchSize);
    const auto deltasOutFlat = deltasOut[b].flatten();
    Tensor intraIPUPartialSums;
    if (numIPUs > 1) {
      intraIPUPartialSums =
          graph.addTensor("float",
                          {deltasOutFlat.numElements(), numIPUs},
                          "ipu_partials");
    } else {
      intraIPUPartialSums = deltasOutFlat.reshape({deltasOutFlat.numElements(),
                                                   1});
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
                              templateVertex("popnn::FullyConnectedBwdReduce",
                                             "float",
                                             outType),
                              {{"partials", partials[i][ipu]},
                               {"out", intraIPUPartialSums[i][ipu]}});
          graph.setTileMapping(v, tile);
          graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
          if (numIPUs > 1) {
            graph.setTileMapping(intraIPUPartialSums[i][ipu], tile);
          }
        }
      }
    }
    // Sum the partial sums from different IPUs.
    if (numIPUs > 1) {
      const auto numTiles = deviceInfo.getNumTiles();
      for (unsigned tile = 0; tile != numTiles; ++tile) {
        const auto deltasBegin = deltasOutMapping[tile];
        const auto deltasEnd = deltasOutMapping[tile + 1];
        if (deltasBegin == deltasEnd)
          continue;
        for (unsigned i = deltasBegin; i != deltasEnd; ++i) {
          auto v =
              graph.addVertex(interIPUReduce,
                              templateVertex("popnn::FullyConnectedBwdReduce",
                                             "float",
                                             dType),
                              {{"partials", intraIPUPartialSums[i]},
                               {"out", deltasOutFlat[i]}});
          graph.setTileMapping(v, tile);
          graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
        }
      }
    }
  }

  return prog;
}

Program
fullyConnectedWeightUpdate(Graph &graph,
                           Tensor zDeltas,
                           Tensor activations,
                           Tensor weights, Tensor biases,
                           float learningRate,
                           const Plan &plan,
                           const std::string &debugPrefix) {
  const auto dType = graph.getTensorElementType(activations);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto &activationsOutMapping = plan.outputMapping;
  const auto batchSize = zDeltas.dim(0);
  const auto size = zDeltas[0].numElements();
  const auto layerName = debugPrefix + "/FullyConnected" + std::to_string(size);
  auto cs = graph.addComputeSet(layerName + "/WeightUpdate");

  const auto numCols = activations[0].numElements();
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
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
          auto activationsFlat = activations.reshape({batchSize, numCols});
          auto actWindow = activationsFlat.slice({0, beginElement},
                                                 {batchSize, endElement});
          auto deltasWindow = zDeltas.slice({0, i},
                                            {batchSize, i + 1})
                                     .flatten();
          auto vertexType = templateVertex("popnn::FullyConnectedWeightUpdate",
                                           dType);
          auto v = graph.addVertex(cs, vertexType,
                                   {{"d", deltasWindow},
                                    {"weights", w},
                                    {"in", actWindow}});
          graph.setInitialValue(v["eta"],
                                learningRate);
          graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
          graph.setTileMapping(v, tile);
        }
      }
    }
  }

  // Update the biases.
  buildTransform(activationsOutMapping, graph,
                 [&](unsigned activationBegin, unsigned activationEnd,
                 unsigned tile) {
    auto deltasWindow =  zDeltas.slice({0, activationBegin},
                                       {batchSize, activationEnd});
    auto v =
        graph.addVertex(cs,
                        templateVertex("popnn::FullyConnectedBiasUpdate",
                                       dType),
                        {{"d", deltasWindow},
                         {"bias", biases.slice(activationBegin,
                                               activationEnd)}});
    graph.setInitialValue(v["eta"], learningRate);
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  });
  return Execute(cs);
}


}
