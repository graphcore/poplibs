#ifndef _layer_hpp_
#define _layer_hpp_

#include <cassert>
#include <poplar/Graph.hpp>
#include <poplar/CPUEngine.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <string>
#include "neural_net_common.h"

using namespace poplar;
using namespace poplar::program;

class Net;

/* A data set full of test and training data along with its dimensions */
class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
  std::vector<std::size_t> dim;
};

enum NetType {
  TrainingNet,
  TestOnlyNet
};

/* The layer class represents a single layer in the net.
 */
class Layer {
  const Net &net;
  int index;
protected:
  Layer(const Net &net, int index) : net(net), index(index) {}
  unsigned getWorkerContextsPerTile() const;
  unsigned getNumIPUs() const;
  unsigned getTilesPerIPU() const;
  const std::string &getDType() const;
  unsigned getDTypeSize() const;
  enum NetType getNetType() const;
  unsigned getBatchSize() const;
  void mapTensor(Tensor t, IPUModelEngineBuilder::TileMapping *mapping);
  void mapComputeSet(const Graph &graph, ComputeSet c,
                     IPUModelEngineBuilder::TileMapping *mapping);
public:
  Layer *getNextLayer() const;
  Layer *getPrevLayer() const;
  virtual void init(Graph &graph,
                    IPUModelEngineBuilder::TileMapping *mapping) = 0;
  virtual Program initParams(Graph &graph) = 0;
  virtual Program startBatch(Graph &graph) = 0;
  virtual Program forward(Graph &graph,
                          IPUModelEngineBuilder::TileMapping *mapping) = 0;
  virtual Program backward(Graph &graph) = 0;
  virtual Program weightSync(Graph &graph) = 0;
  virtual void describe(std::ostream &out) = 0;
  /// Return the number of FLOPs required for a naive implementation of the
  /// forward pass. A FLOP is a basic arithmetic operation such as multiply,
  /// add, subtract. A fused multiply accumulate counts as 2 FLOPs.
  /// The count of the number of FLOPs does not include operations used to
  /// compute the non-linearity at the end of the layer.
  virtual std::uint64_t getNumberOfFlops() = 0;
  /// Return the number of cycles you would expect the forward pass to require
  /// based only on amount of compute required, ignoring any overheads. The
  /// return value may be fractional in cases where the number of operations
  /// required is not an exact multiple of the ideal number of operations per
  /// cycle.
  virtual double getPerfectCycleCount() = 0;
  virtual Tensor getFwdActivations() const = 0;
  virtual Tensor getFwdZs() const = 0;
  virtual NonLinearityType getNonLinearityType() const {
    return NON_LINEARITY_NONE;
  };
  virtual Tensor getBwdErrors() const = 0;

  // Called if the previous layer provides a 3D volume as output.
  // A layer can request that the previous layer provides the z-axis
  // (or channel-axis) to be split into a number of groups. So the provided
  // tensor from the previous layer will be a 4D tensor of dimension
  // {numGroups, x, y, z / numGroups}.
  //
  // If this function returns 0 then it indicates that it does not care how
  // the previous output is grouped.
  virtual size_t getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                                       size_t zPrev) const {
    return 0;
  }
};

class LayerSpec {
public:
  virtual std::unique_ptr<Layer>
  makeLayer(Net &net, int index) = 0;
};


class InputLayer : public Layer {
  DataSet &data;
  Tensor out, z, isTraining;
public:
  InputLayer(const Net &net, int index, DataSet &data, Tensor isTraining) :
    Layer(net, index), data(data), isTraining(isTraining) {}

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    const auto dType = getDType();
    Layer *next = getNextLayer();
    // Re-arrange so that the channels are the major
    auto numGroups = next->getNumChannelGroupsIn(data.dim[0], data.dim[1],
                                               data.dim[2]);
    if (!numGroups)
      numGroups = 1;
    const auto dim = std::vector<size_t>({numGroups, data.dim[0], data.dim[1],
                                          data.dim[2]/numGroups});
    out = graph.addTensor(dType, dim);
    z = graph.addTensor(dType, dim);
    mapTensor(out, mapping);
    mapTensor(z, mapping);
  }

  Program initParams(Graph &graph) { return Sequence(); }
  Program startBatch(Graph &graph) { return Sequence(); }
  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    size_t trainingDataSize = data.numTraining * data.dataSize;
    size_t testDataSize = data.numTest * data.dataSize;
    return Sequence();
    #if 0
    return IfProg(
             isTraining,
             Copy(out,
                  &data.trainingData[0],
                  &data.trainingData[trainingDataSize]),
             Copy(out,
                  &data.testData[0],
                  &data.testData[testDataSize]));
    #endif
  }

  Program backward(Graph &graph) { return Sequence(); }
  Program weightSync(Graph &graph) { return Sequence(); }
  void describe(std::ostream &out) {}
  std::uint64_t getNumberOfFlops() { return 0; }
  virtual double getPerfectCycleCount() { return 0.0; }
  Tensor getFwdActivations() const { return out; }
  Tensor getFwdZs() const { return out; }
  Tensor getBwdErrors() const { return {}; }
};

class LossLayer : public Layer {
  DataSet &data;
  LossType lossType;
  Tensor errors, expected, lossTypeTensor, loss, numCorrect, isTraining;
  unsigned hNumCorrect;
  ComputeSet fwd;
public:
  LossLayer(const Net &net, int index,
            DataSet &data, LossType lossType, Tensor isTraining) :
    Layer(net, index), data(data), lossType(lossType), isTraining(isTraining) {}

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    const auto dType = getDType();
    Layer *prev = getPrevLayer();
    assert(prev);
    errors = graph.addTensor(dType, {prev->getFwdActivations().numElements()});
    expected = graph.addTensor("unsigned", {1});
    lossTypeTensor = graph.addTensor("LossType", {1});
    graph.setInitialValue(lossTypeTensor[0], lossType);
    loss = graph.addTensor(dType, {1});
    numCorrect = graph.addTensor("unsigned", {1});
    mapTensor(errors, mapping);
    mapTensor(expected, mapping);
    mapTensor(lossTypeTensor, mapping);
    mapTensor(loss, mapping);
    mapTensor(numCorrect, mapping);
    fwd = graph.createComputeSet("LossLayer");
  }

  void resetNumCorrect() {
    hNumCorrect = 0;
  }

  unsigned getNumCorrect() {
    return hNumCorrect;
  }

  Program initParams(Graph &graph) { return Sequence(); }
  Program startBatch(Graph &graph) {
    return Assign(numCorrect[0], 0);
  }
  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    Layer *prev = getPrevLayer();
    auto v = graph.addVertex(fwd, "CalcLoss",
                             {{"zIn", prev->getFwdZs().flatten()},
                              {"errorOut", errors},
                              {"label", expected[0]},
                              {"lossType", lossTypeTensor[0]},
                              {"loss", loss[0]},
                              {"numCorrect", numCorrect[0]}});
    graph.setFieldSize(v["probs"], prev->getFwdActivations().numElements());
    graph.setInitialValue(v["nonLinearityType"], prev->getNonLinearityType());
    mapComputeSet(graph, fwd, mapping);
    #if 0
    Program copyLabelsProg =
      Ifprog(isTraining,
             Copy(expected,
                  &data.trainingLabels[0],
                  &data.trainingLabels[data.numTraining]),
             Copy(expected,
                  &data.testLabels[0],
                  &data.testLabels[data.numTest]));
    #endif
    Program copyLabelsProg =
      Copy(expected,
           &data.testLabels[0],
           &data.testLabels[data.numTest]);
    return Sequence(Copy(numCorrect, &hNumCorrect),
                    copyLabelsProg,
                    Execute(fwd),
                    Copy(&hNumCorrect, numCorrect));
  }
  Program backward(Graph &graph) { return Sequence(); }
  Program weightSync(Graph &graph) { return Sequence(); }
  void describe(std::ostream &out) {}
  std::uint64_t getNumberOfFlops() { return 0; }
  virtual double getPerfectCycleCount() { return 0.0; }
  Tensor getFwdActivations() const { return {}; }
  Tensor getFwdZs() const { return {}; }
  Tensor getBwdErrors() const { return errors; }
};

/* This utility function wraps a vector of normal pointers as unique_ptrs.
   It allows the hidden layer array to be initializes with an
   initializer list. */
static std::vector<std::unique_ptr<LayerSpec>>
makeLayers(std::vector<LayerSpec *> vs)
{
  std::vector<std::unique_ptr<LayerSpec>> xs;
  for (auto p: vs)
    xs.push_back(std::unique_ptr<LayerSpec>(p));
  return xs;
}

#endif // _layer_hpp_
