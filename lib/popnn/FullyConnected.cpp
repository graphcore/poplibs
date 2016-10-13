#include "popnn/FullyConnected.hpp"
#include "popnn/FullyConnectedPlan.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexTemplates.hpp"
#include "popnn/ActivationMapping.hpp"
#include "popnn/exceptions.hpp"
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
               const Plan &plan) {
  const auto dType = graph.getTensorElementType(in0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto layerName = "FullyConnected" + std::to_string(size);
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
  // Iterate through the batch creating new compute sets to add to the
  // program (i.e. execute the batch in sequence).
  for (unsigned b = 0; b < batchSize; ++b) {
    const auto &activationsOutMapping =
      computeActivationsMapping(graph, activations[b], b, batchSize);
    ComputeSet dotProductCS = graph.createComputeSet(layerName + ".fwd");
    prog.add(Execute(dotProductCS));
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
                                templateVertex("FullyConnectedPartial", dType),
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
         {"activationOut", activations[b][i]}});
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setTileMapping(v, tile);
        graph.setTileMapping(biases[i], tile);
      }
    }
  }
  return prog;
}

uint64_t getNumFlops(unsigned batchSize,
                     unsigned inSize, unsigned outSize, bool forwardOnly) {
  if (forwardOnly)
    return batchSize * (2 * inSize * outSize);
  else
    return batchSize * 3 * (2 * inSize * outSize);
}

double getPerfectCycleCount(const Graph &graph,
                            unsigned batchSize,
                            unsigned inSize, unsigned outSize,
                            std::string dType, bool forwardOnly) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getNumFlops(batchSize, inSize, outSize, forwardOnly);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (2 * vectorWidth * numTiles);
}

Program fullyConnectedBackward(Graph &graph,
                               Tensor zDeltas,
                               Tensor weights, Tensor deltasOut,
                               const Plan &plan) {
  if (zDeltas.dim(0) != 1) {
    throw popnn::popnn_error("Batch size != 1 not implemented for backwards "
                             "pass");
  }
  auto zDeltas0 = zDeltas[0];
  auto deltasOut0 = deltasOut[0];
  const auto dType = graph.getTensorElementType(zDeltas0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto size = static_cast<unsigned>(zDeltas0.numElements());
  const auto prevSize = static_cast<unsigned>(weights.dim(1));
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto prog = Sequence();
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  prog.add(Execute(bwdCS));
  const auto &ipuPartition = plan.ipuPartition;
  const auto numIPUs = deviceInfo.numIPUs;
  const auto tilesPerIPU = deviceInfo.tilesPerIPU;
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
          Tensor inWindow = zDeltas0.slice(tileRowBegin, tileRowEnd);
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
          graph.setTileMapping(v, tile);
          graph.setTileMapping(outWindow, tile);
        }
      }
    }
  }
  const auto deltasOutMapping =
      computeActivationsMapping(graph, deltasOut0, 0, 1);
  deltasOut0 = deltasOut0.flatten();
  ComputeSet intraIPUReduce = graph.createComputeSet(layerName + ".bwd.reduce");
  prog.add(Execute(intraIPUReduce));
  Tensor intraIPUPartialSums;
  if (numIPUs > 1) {
    Tensor intraIPUPartialSums =
        graph.addTensor("float", {size, numIPUs}, "ipu_partials");
  } else {
    intraIPUPartialSums = deltasOut0.reshape({deltasOut0.numElements(), 1});
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
                             {"out", deltasOut0[i]}});
        graph.setTileMapping(v, tile);
        graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
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
                           const Plan &plan) {
  if (zDeltas.dim(0) != 1) {
    throw popnn::popnn_error("Batch size != 1 not implemented for backwards "
                             "pass");
  }
  auto zDeltas0 = zDeltas[0];
  auto activations0 = activations[0];
  const auto dType = graph.getTensorElementType(activations0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto &activationsOutMapping = plan.outputMapping;
  activations0 = activations0.flatten();
  const auto size = zDeltas0.numElements();
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto cs = graph.createComputeSet(layerName + ".weight_update");

  const auto numCols = activations0.numElements();
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
          auto actWindow = activations0.slice(beginElement, endElement);
          auto v = graph.addVertex(cs,
                                   templateVertex("FullyConnectedWeightUpdate",
                                                  dType),
                                   {{"d", zDeltas0[i]},
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
    auto v =
        graph.addVertex(cs,
                        templateVertex("FullyConnectedBiasUpdate",
                                       dType),
                        {{"d", zDeltas0.slice(activationBegin,
                                             activationEnd)},
                         {"bias", biases.slice(activationBegin,
                                               activationEnd)}});
    graph.setInitialValue(v["eta"], learningRate);
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  });
  return Execute(cs);
}


}
