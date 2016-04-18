#ifndef _fully_connected_layer_hpp_
#define _fully_connected_layer_hpp_
#include "Net.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>

#define USE_PARTIAL_SUMS 1

class FullyConnectedLayerImpl : public Layer {
public:
  std::size_t size, prevSize;
  NonLinearityType nonLinearityType;

  Tensor weights, biases, bwdWeights, z,
    activations, errors, activationRecord, errorRecord,
    actRecordIndex, errorRecordIndex;

  std::unique_ptr<float []> hWeights;
  std::string layerName;
  size_t verticesPerRow;

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
              << "        Params: " << size * (prevSize + 1) << "\n";
  }

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    const auto dType = getDType();
    Layer *prev = getPrevLayer();
    prevSize = prev->getFwdActivations().numElements();
    if (USE_PARTIAL_SUMS) {
      const auto numTiles = getNumIPUs() * getTilesPerIPU();
      const auto numRows = size;
      const auto numCols = prevSize;
      // The cost of partial sum elements relative to input vector elements.
      unsigned partialSumCost;
      if (dType == "float") {
        partialSumCost = 1;
      } else {
        assert(dType == "short");
        partialSumCost = 2;
      }
      verticesPerRow =
        static_cast<unsigned>(
          std::floor(std::sqrt((numCols * numTiles) /
                               (static_cast<float>(numRows * partialSumCost))))
        );
      verticesPerRow = std::max(verticesPerRow, 1UL);
    } else {
      verticesPerRow = 1;
    }
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

    if (verticesPerRow > 1) {
      const auto numTiles = getNumIPUs() * getTilesPerIPU();
      const auto numRows = size;
      const auto numCols = prevSize;

      Tensor partials = graph.addTensor("float", {numRows, verticesPerRow});
      ComputeSet fwd1 = graph.createComputeSet(layerName + ".fwd"),
                 fwd2 = graph.createComputeSet(layerName + ".fwd.reduce");

      for (unsigned i = 0; i != numRows; ++i) {
        const auto resultTile = (i * numTiles) / numRows;
        for (unsigned j = 0; j != verticesPerRow; ++j) {
          const auto beginElement = (numCols * j) / verticesPerRow;
          const auto endElement = (numCols * (j + 1)) / verticesPerRow;
          Tensor partialIn = in.slice(beginElement, endElement);
          Tensor partialWeights = weights[i].slice(beginElement, endElement);
          auto v = graph.addVertex(fwd1, "FullyConnectedPartial",
                                   {{"in", partialIn},
                                    {"weights", partialWeights},
                                    {"out", partials[i][j]}});
          const auto tile = j +
                            verticesPerRow *
                            ((i * (numTiles / verticesPerRow)) / numRows);
          if (mapping) {
            mapping->setMapping(v, tile);
            mapping->setMapping(partialWeights, tile);
          }
        }
        auto v = graph.addVertex(fwd2, "FullyConnectedReduce",
                                 {{"partials", partials[i]},
                                  {"bias", biases[i]},
                                  {"zOut", z[i]},
                                  {"activationOut", activations[i]}});
        graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
        if (mapping) {
          mapping->setMapping(partials[i], resultTile);
          mapping->setMapping(v, resultTile);
        }
      }
      return Sequence(Execute(fwd1), Execute(fwd2));
    } else {
      ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
      for (unsigned i = 0; i < size; ++i) {
        auto v = graph.addVertex(fwd, "FullyConnected",
                                 {{"activationIn", in},
                                  {"weights", weights[i]},
                                  {"bias", biases[i]},
                                  {"zOut", z[i]},
                                  {"activationOut", activations[i]}});
        graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      }
      mapComputeSet(graph, fwd, mapping);
      mapTensor(weights, mapping);
      return Sequence(Execute(fwd));
    }
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
