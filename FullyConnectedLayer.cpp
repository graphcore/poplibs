#include "FullyConnectedLayer.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexTemplates.hpp"

namespace {
  struct PartitionShape {
    unsigned tilesPerColumn;
    unsigned tilesPerRow;
    PartitionShape(unsigned tilesPerColumn, unsigned tilesPerRow) :
      tilesPerColumn(tilesPerColumn), tilesPerRow(tilesPerRow) {}
  };
}

static unsigned
estimatePartitionCost(IPUModelEngineBuilder &engineBuilder, bool isFloat,
                      unsigned numRows, unsigned numCols, unsigned tilesPerRow,
                      unsigned tilesPerColumn,
                      const IPUMachineInfo &machineInfo) {
  auto numTiles = tilesPerRow * tilesPerColumn;
  auto numVertices = numRows * tilesPerRow;
  auto numWorkerContexts = engineBuilder.getNumWorkerContexts();
  auto vertexElements = (numCols + tilesPerRow - 1) / tilesPerRow;
  auto partialSumsPerTile = (numRows + tilesPerColumn - 1) / tilesPerColumn;
  auto vertexRuntime =
      getFullyConnectedPartialCycleEstimate(isFloat, vertexElements,
                                            machineInfo.dataPathWidth);
  auto verticesPerWorker = (numVertices + numTiles * numWorkerContexts - 1) /
                           (numTiles * numWorkerContexts);
  auto computeCycles = vertexRuntime * verticesPerWorker;
  auto exchangeBytesPerCycle = engineBuilder.getIPUExchangeBandwidth();
  auto inputBytes = vertexElements * (isFloat ? 4 : 2);
  auto partialSumBytes = partialSumsPerTile * 4;
  auto exchangeCycles =
      (inputBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;
  return computeCycles + exchangeCycles;
}

static PartitionShape
choosePartition(IPUModelEngineBuilder &engineBuilder,
                bool isFloat, unsigned numRows,
                unsigned numCols, unsigned numTiles,
                const IPUMachineInfo &machineInfo) {
  unsigned lowestCost = std::numeric_limits<unsigned>::max();
  unsigned bestTilesPerColumn, bestTilesPerRow;
  for (unsigned tilesPerRow = 1; tilesPerRow <= numTiles; ++tilesPerRow) {
    unsigned tilesPerColumn = numTiles / tilesPerRow;
    const auto cost = estimatePartitionCost(engineBuilder, isFloat, numRows,
                                            numCols, tilesPerRow,
                                            tilesPerColumn, machineInfo);
    if (cost < lowestCost) {
      lowestCost = cost;
      bestTilesPerColumn = tilesPerColumn;
      bestTilesPerRow = tilesPerRow;
    }
  }
  return PartitionShape(bestTilesPerColumn, bestTilesPerRow);
}

void FullyConnectedLayerImpl::describe(std::ostream &out) {
  std::cout << "   -- Fully connected layer:\n"
            << "        Input: "  << prevSize << "\n"
            << "        Output: " << size << "\n"
            << "        Params: " << size * (prevSize + 1) << "\n"
            << "        FLOPs: " << getNumberOfFlops() << "\n";
}

std::uint64_t FullyConnectedLayerImpl::getNumberOfFlops() {
  auto numRows = size;
  auto numCols = prevSize;
  return 2 * numRows * numCols;
}

double FullyConnectedLayerImpl::getPerfectCycleCount() {
  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  const auto &machineInfo = getNetOptions().ipuMachineInfo;
  const auto numFLOPs = getNumberOfFlops();
  const auto vectorWidth = machineInfo.dataPathWidth / (8 * getDTypeSize());
  return static_cast<double>(numFLOPs) / (2 * vectorWidth * numTiles);
}

void FullyConnectedLayerImpl::
init(Graph &graph, std::mt19937 &randomEngine,
     IPUModelEngineBuilder::TileMapping &mapping) {
  const auto dType = getDType();
  Layer *prev = getPrevLayer();
  prevSize = prev->getFwdActivations().numElements();

  weights = graph.addTensor(dType, {size, prevSize}, makeLayerName("weights"));
  biases = graph.addTensor(dType, {size}, makeLayerName("biases"));
  z = graph.addTensor(dType, {size}, makeLayerName("z"));
  activations = graph.addTensor(dType, {size}, makeLayerName("activations"));
  mapTensor(biases, mapping);
  mapTensor(z, mapping);
  mapTensor(activations, mapping);
  // weights mapped in forward()
  if (getNetType() == TrainingNet) {
    zDeltas = graph.addTensor(dType, z.dims(), makeLayerName("zDeltas"));
    deltas = graph.addTensor(dType, prev->getFwdActivations().dims(),
                             makeLayerName("deltas"));
    bwdWeights = graph.addTensor(dType, {prevSize + 1, size}, 
                                 makeLayerName("bwdWeights"));
    mapTensor(zDeltas, mapping);
    mapTensor(deltas, mapping);
    mapTensor(bwdWeights, mapping);
  }
  // Initialize weights using "xavier" weight filler that scales
  // variance based on number of inputs to a neuron.
  hWeights = createRandomWeightInitializers(weights, 0, 1.0 / prevSize,
                                            randomEngine);
  hBiases = createRandomWeightInitializers(biases, 0, 1.0 / prevSize,
                                           randomEngine);
}

Program FullyConnectedLayerImpl::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping &mapping) {
  Layer *prev = getPrevLayer();
  Tensor in = prev->getFwdActivations().flatten();
  const auto dType = getDType();
  const auto dataPathWidth = getNetOptions().ipuMachineInfo.dataPathWidth;

  const auto numRows = size;
  const auto numCols = prevSize;
  // In theory a 2D tiling of the matrix across IPUs could decrease the
  // amount of communication. Unfortunately it introduces a new causal layer.
  // It turns out that, at least up to 16 IPUs, it is better to always keep
  // all row elements on the same IPU to avoid the need for an extra sync.
  const auto numIPUs = getNumIPUs();
  const auto maxRowsPerTile = (numRows + numIPUs - 1) / numIPUs;

  bool isFloat = dType == "float";
  assert(isFloat || dType == "half");
  const auto tilesPerIPU = getTilesPerIPU();
  auto ipuPartition =
      choosePartition(getIPUModelEngineBuilder(), isFloat, maxRowsPerTile,
                      numCols, tilesPerIPU, getNetOptions().ipuMachineInfo);

  ComputeSet dotProductCS = graph.createComputeSet(layerName + ".fwd");
  ComputeSet reduceCS;
  Tensor partials;
  if (ipuPartition.tilesPerRow > 1) {
     reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
     partials = graph.addTensor("float", {numRows, ipuPartition.tilesPerRow}, 
                                makeLayerName("partials"));
  }

  for (unsigned i = 0; i != numRows; ++i) {
    const auto ipu = (i * numIPUs) / numRows;
    const auto ipuBeginRow = (numRows * ipu) / numIPUs;
    const auto ipuEndRow = (numRows * (ipu + 1)) / numIPUs;
    const auto ipuRows = ipuEndRow - ipuBeginRow;
    if (ipuPartition.tilesPerRow > 1) {
      const auto tileY = ((i - ipuBeginRow) * ipuPartition.tilesPerColumn) /
                         ipuRows;
      for (unsigned j = 0; j != ipuPartition.tilesPerRow; ++j) {
        const auto tileX = j;
        const auto tile = ipu * tilesPerIPU +
                          tileY * ipuPartition.tilesPerRow +
                          tileX;
        if (ipuPartition.tilesPerRow > 1) {
          const auto beginElement =
              (numCols * j) / ipuPartition.tilesPerRow;
          const auto endElement =
              (numCols * (j + 1)) / ipuPartition.tilesPerRow;
          Tensor partialIn = in.slice(beginElement, endElement);
          Tensor partialWeights = weights[i].slice(beginElement, endElement);
          auto v =
              graph.addVertex(dotProductCS,
                              templateVertex("FullyConnectedPartial", getDType()),
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
    const auto resultTile = (i * getNumIPUs() * getTilesPerIPU()) /
                            numRows;
    mapping.setMapping(biases[i], resultTile);
    mapping.setMapping(z[i], resultTile);
    mapping.setMapping(activations[i], resultTile);
    if (ipuPartition.tilesPerRow > 1) {
      // Sum the partial sums.
      auto v =
          graph.addVertex(reduceCS,
                          templateVertex("FullyConnectedReduce",
                                         getDType()),
                          {{"partials", partials[i]},
                           {"bias", biases[i]},
                           {"zOut", z[i]},
                           {"activationOut", activations[i]}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      mapping.setMapping(v, resultTile);
    } else {
      auto v =
          graph.addVertex(dotProductCS,
                          templateVertex("FullyConnected", getDType()),
                          {{"activationIn", in},
                           {"weights", weights[i]},
                           {"bias", biases[i]},
                           {"zOut", z[i]},
                           {"activationOut", activations[i]}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      mapping.setMapping(v, resultTile);
      mapping.setMapping(weights[i], resultTile);
    }
  }
  if (ipuPartition.tilesPerRow > 1) {
    return Sequence(Execute(dotProductCS), Execute(reduceCS));
  }
  return Sequence(Execute(dotProductCS));
}

Program FullyConnectedLayerImpl::
backward(Graph &graph, IPUModelEngineBuilder::TileMapping &mapping) {
  auto bwdNonLinearityCS =
      graph.createComputeSet(layerName + ".bwd.nonLinearity");
  auto deltasIn = getNextLayer()->getBwdDeltas().flatten();

  auto v = graph.addVertex(bwdNonLinearityCS,
                           templateVertex("NonLinearityBwd",
                                          getDType()),
                           {{"deltasIn", deltasIn },
                            {"z", z},
                            {"deltasOut", zDeltas},
                           });
  graph.setInitialValue(v["nonLinearityType"], nonLinearityType);

  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  auto flatDeltas = deltas.flatten();
  for (unsigned i = 0; i < prevSize; ++i) {
    auto w = weights.slice({0, i}, {size, i+1}).flatten();
    auto in = getNextLayer()->getBwdDeltas().flatten();
    auto v = graph.addVertex(bwdCS,
                             templateVertex("FullyConnectedBwd",
                                            getDType()),
                             {{"in", zDeltas},
                              {"weights", w},
                              {"out", flatDeltas[i]},
                             });
  }
  return Sequence(Execute(bwdNonLinearityCS), Execute(bwdCS));
}

Program FullyConnectedLayerImpl::
weightUpdate(Graph &graph, IPUModelEngineBuilder::TileMapping &mapping) {
  auto cs = graph.createComputeSet(layerName + ".weight_update");
  for (unsigned i = 0; i < size; ++i) {
    auto prev = getPrevLayer()->getFwdActivations().flatten();
    auto v = graph.addVertex(cs,
                             templateVertex("FullyConnectedWeightUpdate",
                                            getDType()),
                             {{"d", zDeltas[i]},
                              {"weights", weights[i]},
                              {"in", prev},
                              {"bias", biases[i]}});
    graph.setInitialValue(v["eta"],
                          getLearningRate());
  }

  return Execute(cs);
}
