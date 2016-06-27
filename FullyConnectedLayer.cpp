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
estimatePartitionCost(unsigned numWorkerContexts, bool isFloat,
                      unsigned numRows, unsigned numCols, unsigned tilesPerRow,
                      unsigned tilesPerColumn) {
  auto numTiles = tilesPerRow * tilesPerColumn;
  auto numVertices = numRows * tilesPerRow;
  auto vertexElements = (numCols + tilesPerRow - 1) / tilesPerRow;
  auto partialSumsPerTile = (numRows + tilesPerColumn - 1) / tilesPerColumn;
  auto vertexRuntime =
      getFullyConnectedPartialCycleEstimate(isFloat, vertexElements);
  auto verticesPerWorker = (numVertices + numTiles * numWorkerContexts - 1) /
                           (numTiles * numWorkerContexts);
  auto computeCycles = vertexRuntime * verticesPerWorker;
  auto exchangeElementsPerCycle = isFloat ? 1 : 2;
  auto exchangeCycles =
    (vertexElements + exchangeElementsPerCycle - 1) / exchangeElementsPerCycle +
    partialSumsPerTile;
  return computeCycles + exchangeCycles;
}

static PartitionShape
choosePartition(unsigned numWorkerContexts, bool isFloat, unsigned numRows,
                unsigned numCols, unsigned numTiles) {
  unsigned lowestCost = std::numeric_limits<unsigned>::max();
  unsigned bestTilesPerColumn, bestTilesPerRow;
  for (unsigned tilesPerRow = 1; tilesPerRow <= numTiles; ++tilesPerRow) {
    unsigned tilesPerColumn = numTiles / tilesPerRow;
    const auto cost = estimatePartitionCost(numWorkerContexts, isFloat,
                                            numRows, numCols, tilesPerRow,
                                            tilesPerColumn);
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
  const auto numFLOPs = getNumberOfFlops();
  const auto vectorWidth = 64 / (8 * getDTypeSize());
  return static_cast<double>(numFLOPs) / (2 * vectorWidth * numTiles);
}

void FullyConnectedLayerImpl::
init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
  const auto dType = getDType();
  Layer *prev = getPrevLayer();
  prevSize = prev->getFwdActivations().numElements();

  weights = graph.addTensor(dType, {size, prevSize});
  biases = graph.addTensor(dType, {size});
  z = graph.addTensor(dType, {size});
  activations = graph.addTensor(dType, {size});
  mapTensor(biases, mapping);
  mapTensor(z, mapping);
  mapTensor(activations, mapping);
  // weights mapped in forward()
  if (getNetType() == TrainingNet) {
    const auto batchSize = getBatchSize();
    errors = graph.addTensor(dType, {prevSize});
    activationRecord = graph.addTensor(dType, {prevSize, batchSize});
    actRecordIndex = graph.addTensor("unsigned", {1});
    errorRecord = graph.addTensor(dType, {size, batchSize});
    errorRecordIndex = graph.addTensor("unsigned", {1});
    bwdWeights = graph.addTensor(dType, {prevSize + 1, size});
    mapTensor(errors, mapping);
    mapTensor(activationRecord, mapping);
    mapTensor(actRecordIndex, mapping);
    mapTensor(errorRecord, mapping);
    mapTensor(errorRecordIndex, mapping);
    mapTensor(bwdWeights, mapping);
  }
  hWeights = std::unique_ptr<float[]>(new float[prevSize * size]);
  hBiases = std::unique_ptr<float[]>(new float[size]);
  unsigned seed = time(0);
  boost::variate_generator< boost::mt19937, boost::normal_distribution<> >
    generator(boost::mt19937(seed), boost::normal_distribution<>(0, 1));
  for (unsigned i = 0; i < prevSize*size; ++i)
    hWeights[i] = generator();
  for (unsigned i = 0; i < size; ++i)
    hBiases[i] = generator();
}

Program FullyConnectedLayerImpl::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
  Layer *prev = getPrevLayer();
  Tensor in = prev->getFwdActivations().flatten();
  const auto dType = getDType();

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
  auto ipuPartition = choosePartition(getWorkerContextsPerTile(),
                                      isFloat, maxRowsPerTile, numCols,
                                      tilesPerIPU);

  ComputeSet dotProductCS = graph.createComputeSet(layerName + ".fwd");
  ComputeSet reduceCS;
  Tensor partials;
  if (ipuPartition.tilesPerRow > 1) {
     reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
     partials = graph.addTensor("float", {numRows, ipuPartition.tilesPerRow});
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
          if (mapping) {
            mapping->setMapping(partialWeights, tile);
            mapping->setMapping(partials[i][j], tile);
            mapping->setMapping(v, tile);
          }
        }
      }
    }
    const auto resultTile = (i * getNumIPUs() * getTilesPerIPU()) /
                            numRows;
    if (mapping) {
      mapping->setMapping(biases[i], resultTile);
      mapping->setMapping(z[i], resultTile);
      mapping->setMapping(activations[i], resultTile);
    }
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
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      if (mapping) {
        mapping->setMapping(v, resultTile);
      }
    } else {
      auto v =
          graph.addVertex(dotProductCS,
                          templateVertex("FullyConnected", getDType()),
                          {{"activationIn", in},
                           {"weights", weights[i]},
                           {"bias", biases[i]},
                           {"zOut", z[i]},
                           {"activationOut", activations[i]}});
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      if (mapping) {
        mapping->setMapping(v, resultTile);
        mapping->setMapping(weights[i], resultTile);
      }
    }
  }
  if (ipuPartition.tilesPerRow > 1) {
    return Sequence(Execute(dotProductCS), Execute(reduceCS));
  }
  return Sequence(Execute(dotProductCS));
}

Program FullyConnectedLayerImpl::backward(Graph &graph) {
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  for (unsigned i = 0; i < prevSize; ++i) {
    auto w = weights.slice({0, i}, {size, i+1}).flatten();
    auto in = getNextLayer()->getBwdErrors().flatten();
    auto v = graph.addVertex(bwdCS,
                             templateVertex("FullyConnectedBwd",
                                            getDType()),
                             {{"in", in},
                              {"z", z},
                              {"weights", w},
                              {"out", errors[i]},
                             });
    graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
  }
  return Execute(bwdCS);
}

Program FullyConnectedLayerImpl::weightUpdate(Graph &graph) {
  auto cs = graph.createComputeSet(layerName + ".weight_update");
  for (unsigned i = 0; i < size; ++i) {
    auto errorIn = getNextLayer()->getBwdErrors().flatten();
    auto prev = getPrevLayer()->getFwdActivations().flatten();
    auto v = graph.addVertex(cs,
                             templateVertex("FullyConnectedWeightUpdate",
                                            getDType()),
                             {{"error", errorIn[i]},
                              {"weights", weights[i]},
                              {"in", prev},
                              {"z", z[i]},
                              {"bias", biases[i]}});
    graph.setInitialValue(v["eta"],
                          getLearningRate());
    graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
  }

  return Execute(cs);
}
