#ifndef _fully_connected_layer_hpp_
#define _fully_connected_layer_hpp_
#include "Net.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>

static uint64_t estimateVertexCycles(bool isFloat, unsigned size) {
    return (size + 3) / 4 + 2 + 5;
  return (size + 1) / 2 + 2 + 5;
}

struct PartitionShape {
  unsigned tilesPerColumn;
  unsigned tilesPerRow;
  PartitionShape(unsigned tilesPerColumn, unsigned tilesPerRow) :
    tilesPerColumn(tilesPerColumn), tilesPerRow(tilesPerRow) {}
};

// TODO Instead of hardcoding this we should querying it somehow.
static unsigned numWorkerContexts = 6;

static unsigned estimatePartitionCost(bool isFloat, unsigned numRows,
                                 unsigned numCols, unsigned tilesPerRow,
                                 unsigned tilesPerColumn) {
  auto numTiles = tilesPerRow * tilesPerColumn;
  auto numVertices = numRows * tilesPerRow;
  auto vertexElements = (numCols + tilesPerRow - 1) / tilesPerRow;
  auto partialSumsPerTile = (numRows + tilesPerColumn - 1) / tilesPerColumn;
  auto vertexRuntime = estimateVertexCycles(isFloat, vertexElements);
  auto verticesPerWorker = (numVertices + numTiles * numWorkerContexts - 1) /
                           (numTiles * numWorkerContexts);
  auto computeCycles = vertexRuntime * verticesPerWorker;
  auto exchangeElementsPerCycle = isFloat ? 1 : 2;
  auto exchangeCycles =
    (vertexElements + exchangeElementsPerCycle - 1) / exchangeElementsPerCycle +
    partialSumsPerTile;
  return computeCycles + exchangeCycles;
}

PartitionShape
choosePartition(bool isFloat, unsigned numRows, unsigned numCols,
                unsigned numTiles) {
  unsigned lowestCost = std::numeric_limits<unsigned>::max();
  unsigned bestTilesPerColumn, bestTilesPerRow;
  for (unsigned tilesPerRow = 1; tilesPerRow <= numTiles; ++tilesPerRow) {
    unsigned tilesPerColumn = numTiles / tilesPerRow;
    const auto cost = estimatePartitionCost(isFloat, numRows,
                                            numCols, tilesPerRow,
                                            tilesPerColumn);
    if (cost < lowestCost) {
      lowestCost = cost;
      bestTilesPerColumn = tilesPerColumn;
      bestTilesPerRow = tilesPerRow;
    }
  }
  return PartitionShape(bestTilesPerColumn, bestTilesPerRow);
}

class FullyConnectedLayerImpl : public Layer {
public:
  std::size_t size, prevSize;
  NonLinearityType nonLinearityType;

  Tensor weights, biases, bwdWeights, z,
    activations, errors, activationRecord, errorRecord,
    actRecordIndex, errorRecordIndex;

  std::unique_ptr<float []> hWeights;
  std::string layerName;

  FullyConnectedLayerImpl(Net &net, int index,
                          unsigned size,
                          NonLinearityType nonLinearityType) :
    Layer(net, index),
    size(size),
    nonLinearityType(nonLinearityType) {
    layerName = "FullyConnected" + std::to_string(size);
  }

  Tensor getFwdActivations() const {
    return activations;
  }

  Tensor getFwdZs() const {
    return z;
  }

  Tensor getBwdErrors() const {
    return errors;
  }

  NonLinearityType getNonLinearityType() const {
    return nonLinearityType;
  }

  void describe(std::ostream &out) {
    std::cout << "   -- Fully connected layer:\n"
              << "        Input: "  << prevSize << "\n"
              << "        Output: " << size << "\n"
              << "        Params: " << size * (prevSize + 1) << "\n"
              << "        FLOPs: " << getNumberOfFlops() << "\n";
  }

  std::uint64_t getNumberOfFlops() {
    auto numRows = size;
    auto numCols = prevSize;
    return 2 * numRows * numCols;
  }

  virtual double getPerfectCycleCount() {
    // Can execute 4 f16 MACs of 2 f32 MACs per cycle.
    return static_cast<double>(getNumberOfFlops()) / (2 * getDTypeSize());
  }

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
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
    hWeights = std::unique_ptr<float[]>(new float[(prevSize+1) * size]);
    unsigned seed = time(0);
    boost::variate_generator< boost::mt19937, boost::normal_distribution<> >
      generator(boost::mt19937(seed), boost::normal_distribution<>(0, 1));
    for (unsigned i = 0; i < (prevSize+1)*size; ++i)
      hWeights[i] = generator();
  }

  Program initParams(Graph &graph) {
    return Sequence();
  }

  Program startBatch(Graph &graph) {
    return Sequence();
  }

  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
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
    assert(isFloat || dType == "short");
    const auto tilesPerIPU = getTilesPerIPU();
    auto ipuPartition = choosePartition(isFloat, maxRowsPerTile, numCols,
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
      const auto tileY = ((i - ipuBeginRow) * ipuPartition.tilesPerColumn) /
                         ipuRows;
      for (unsigned j = 0; j != ipuPartition.tilesPerRow; ++j) {
        const auto tileX = j;
        const auto tile = ipu * tilesPerIPU +
                          tileY * ipuPartition.tilesPerRow +
                          tileX;
        VertexRef v;
        if (ipuPartition.tilesPerRow > 1) {
          const auto beginElement =
              (numCols * j) / ipuPartition.tilesPerRow;
          const auto endElement =
              (numCols * (j + 1)) / ipuPartition.tilesPerRow;
          Tensor partialIn = in.slice(beginElement, endElement);
          Tensor partialWeights = weights[i].slice(beginElement, endElement);
          v = graph.addVertex(dotProductCS, "FullyConnectedPartial",
                              {{"in", partialIn},
                               {"weights", partialWeights},
                               {"out", partials[i][j]}});
          if (mapping) {
            mapping->setMapping(partialWeights, tile);
            mapping->setMapping(v, tile);
          }
        } else {
          v = graph.addVertex(dotProductCS, "FullyConnected",
                              {{"activationIn", in},
                               {"weights", weights[i]},
                               {"bias", biases[i]},
                               {"zOut", z[i]},
                               {"activationOut", activations[i]}});
          graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
          if (mapping) {
            mapping->setMapping(v, tile);
          }
        }
      }
      if (ipuPartition.tilesPerRow > 1) {
        // Sum the partial sums.
        auto v = graph.addVertex(reduceCS, "FullyConnectedReduce",
                                 {{"partials", partials[i]},
                                  {"bias", biases[i]},
                                  {"zOut", z[i]},
                                  {"activationOut", activations[i]}});
        graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
        if (mapping) {
          const auto resultTile = (i * getNumIPUs() * getTilesPerIPU()) /
                                  numRows;
          mapping->setMapping(partials[i], resultTile);
          mapping->setMapping(v, resultTile);
        }
      }
    }
    if (ipuPartition.tilesPerRow > 1) {
      return Sequence(Execute(dotProductCS), Execute(reduceCS));
    }
    return Sequence(Execute(dotProductCS));
  }

  Program backward(Graph &graph) {
    // TODO
    return Sequence();
  }

  Program weightSync(Graph &graph) {
    // TODO
    return Sequence();
  }
};

class FullyConnectedLayer : public LayerSpec {
  unsigned size;
  NonLinearityType nonLinearityType;
public:
  FullyConnectedLayer(unsigned size,
                      NonLinearityType nonLinearityType) :
    size(size), nonLinearityType(nonLinearityType) {}
  std::unique_ptr<Layer>
  makeLayer(Net &net, int index) {
    return std::unique_ptr<Layer>(
       new FullyConnectedLayerImpl(net, index, size, nonLinearityType));
  }
};


#endif // _fully_connected_layer_hpp_
