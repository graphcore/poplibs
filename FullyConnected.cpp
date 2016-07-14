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
  auto activationsMapping = computeActivationsMapping(activations, deviceInfo);
  bool isFloat = dType == "float";
  assert(isFloat || dType == "half");

  const auto &ipuPartition = plan.ipuPartition;
  Tensor partials = graph.addTensor("float", {numRows,
                                              ipuPartition.tilesPerRow},
                                    "partials");
  auto forwardProg = Sequence();

  ComputeSet dotProductCS = graph.createComputeSet(layerName + ".fwd");
  forwardProg.add(Execute(dotProductCS));
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    const auto ipuBeginRow = activationsMapping[ipu * tilesPerIPU];
    const auto ipuEndRow = activationsMapping[(ipu + 1) * tilesPerIPU];
    const auto ipuRows = ipuEndRow - ipuBeginRow;
    for (unsigned i = ipuBeginRow; i != ipuEndRow; ++i) {
      const auto tileY = ((i - ipuBeginRow) * ipuPartition.tilesPerColumn) /
                         ipuRows;
      for (unsigned j = 0; j != ipuPartition.tilesPerRow; ++j) {
        const auto tileX = j;
        const auto tile = ipu * tilesPerIPU +
                          tileY * ipuPartition.tilesPerRow +
                          tileX;
        const auto beginElement =
            (numCols * j) / ipuPartition.tilesPerRow;
        const auto endElement =
            (numCols * (j + 1)) / ipuPartition.tilesPerRow;
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
  ComputeSet reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
  forwardProg.add(Execute(reduceCS));
  const auto numTiles = numIPUs * tilesPerIPU;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto activationsBegin = activationsMapping[tile];
    const auto activationsEnd = activationsMapping[tile + 1];
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
  return forwardProg;
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

Program
fullyConnectedBwdNonLinearity(Graph &graph,
                              IPUModelEngineBuilder::TileMapping &mapping,
                              DeviceInfo &deviceInfo,
                              std::string dType,
                              Tensor z, Tensor deltasIn,
                              Tensor zDeltas,
                              NonLinearityType nonLinearityType,
                              const Plan &plan) {
  const auto size = deltasIn.numElements();
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto bwdNonLinearityCS =
      graph.createComputeSet(layerName + ".bwd.nonLinearity");
  auto v = graph.addVertex(bwdNonLinearityCS,
                           templateVertex("NonLinearityBwd", dType),
                           {{"deltasIn", deltasIn.flatten() },
                            {"z", z},
                            {"deltasOut", zDeltas},
                           });
  graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
  return Execute(bwdNonLinearityCS);
}

Program fullyConnectedBackward(Graph &graph,
                               IPUModelEngineBuilder::TileMapping &mapping,
                               DeviceInfo &deviceInfo,
                               std::string dType,
                               Tensor zDeltas,
                               Tensor weights, Tensor deltasOut,
                               const Plan &plan) {
  const auto size = zDeltas.numElements();
  const auto prevSize = weights.dim(1);
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  auto flatDeltas = deltasOut.flatten();
  for (unsigned i = 0; i < prevSize; ++i) {
    auto w = weights.slice({0, i}, {size, i+1}).flatten();
    auto v = graph.addVertex(bwdCS,
                             templateVertex("FullyConnectedBwd", dType),
                             {{"in", zDeltas.flatten()},
                              {"weights", w},
                              {"out", flatDeltas[i]},
                             });
  }
  return Execute(bwdCS);
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
  const auto size = zDeltas.numElements();
  const auto layerName = "FullyConnected" + std::to_string(size);
  auto cs = graph.createComputeSet(layerName + ".weight_update");
  for (unsigned i = 0; i < size; ++i) {
    auto prev = activations.flatten();
    auto v = graph.addVertex(cs,
                             templateVertex("FullyConnectedWeightUpdate",
                                            dType),
                             {{"d", zDeltas[i]},
                              {"weights", weights[i]},
                              {"in", prev},
                              {"bias", biases[i]}});
    graph.setInitialValue(v["eta"],
                          learningRate);
  }

  return Execute(cs);
}


}

