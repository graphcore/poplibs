#ifndef _net_hpp_
#define _net_hpp_
#include <poplar/Graph.hpp>
#include <poplar/CPUEngine.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include "neural_net_common.h"
#include <map>
#include <stdexcept>
#include <cassert>

using namespace poplar;
using namespace poplar::program;

struct net_creation_error : std::logic_error {
  std::string type;
  explicit net_creation_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit net_creation_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};


enum DType {
  FP16,
  FP32
};


enum NetType {
  TrainingNet,
  TestOnlyNet
};

class NetOptions {
public:
  bool useIPUModel = false;
  bool doComputation = true;
  bool singleBatchProfile = false;
  unsigned numIPUs = 1;
  unsigned numBatchesBetweenTest = 2500;
};

/* A data set full of test and training data along with its dimensions */
class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
  std::vector<std::size_t> dim;
};

class Net;

/* The layer class represents a single layer in the net.
 */
class Layer {
  const Net &net;
  int index;
protected:
  Layer(const Net &net, int index) : net(net), index(index) {}
  unsigned getNumIPUs() const;
  unsigned getTilesPerIPU() const;
  const std::string &getDType() const;
  enum NetType getNetType() const;
  unsigned getBatchSize() const;
  void mapTensor(Tensor t, IPUModelEngineBuilder::TileMapping *mapping);
  void mapComputeSet(const Graph &graph, ComputeSet c,
                     IPUModelEngineBuilder::TileMapping *mapping);
  Layer *getNextLayer() const;
  Layer *getPrevLayer() const;
public:
  virtual void init(Graph &graph,
                    IPUModelEngineBuilder::TileMapping *mapping) = 0;
  virtual Program initParams(Graph &graph) = 0;
  virtual Program startBatch(Graph &graph) = 0;
  virtual Program forward(Graph &graph,
                          IPUModelEngineBuilder::TileMapping *mapping) = 0;
  virtual Program backward(Graph &graph) = 0;
  virtual Program weightSync(Graph &graph) = 0;
  virtual void describe(std::ostream &out) = 0;
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

static std::string getDTypeString(DType dType) {
  switch (dType) {
  case FP32:
    return "float";
  case FP16:
    return "short";
  }
}

/* This class represent the entire network. */
class Net {
public:
  NetType netType;
  NetOptions options;

  unsigned batchSize;
  float eta;
  std::vector<std::unique_ptr<LayerSpec>> hiddenLayerSpecs;
  std::vector<std::unique_ptr<Layer>> hiddenLayers;

  /* Poplar program creation state. */
  std::unique_ptr<GraphProgEnv> env;
  std::unique_ptr<Graph> graph;
  std::unique_ptr<InputLayer> inputLayer;
  std::unique_ptr<LossLayer> lossLayer;
  std::unique_ptr<EngineBuilder> engineBuilder;
  std::unique_ptr<Engine> engine;

  unsigned hIsTraining;
  unsigned numTestBatches;

  std::string dType;
  unsigned numIPUs, tilesPerIPU;

  unsigned getNumIPUs() const { return numIPUs; }
  unsigned getTilesPerIPU() const { return tilesPerIPU; }

  Layer *getPrevLayer(int index) const {
    assert(index >= 0);
    if (index == 0)
      return inputLayer.get();
    return hiddenLayers[index - 1].get();
  }

  Layer *getNextLayer(int index) const {
    if (index == -1)
      return hiddenLayers[0].get();
    assert(index < hiddenLayers.size());
    if (index == hiddenLayers.size() - 1)
      return lossLayer.get();
    return hiddenLayers[index + 1].get();
  }

  const std::string &getDType() const { return dType; }

  unsigned getBatchSize() const { return batchSize; }

  enum NetType getNetType() const { return netType; }

  void initialize(DataSet &data, LossType lossType) {
    unsigned inputSize = data.dataSize;
    numTestBatches = data.numTest / batchSize;
    std::string obj;
    if (dType == "float") {
      obj = "obj/neural_net_graph32.ppo";
    } else {
      obj = "obj/neural_net_graph16.ppo";
    }
    env = std::unique_ptr<GraphProgEnv>(
      new GraphProgEnv(obj, GraphProgFileType::Object));

    graph = std::unique_ptr<Graph>(new Graph(*env));
    std::unique_ptr<IPUModelEngineBuilder::TileMapping> mapping;
    if (options.useIPUModel) {
      mapping.reset(new IPUModelEngineBuilder::TileMapping(*graph));
      IPUModelEngineBuilder *ipuEB = new IPUModelEngineBuilder(*env);
      engineBuilder = std::unique_ptr<EngineBuilder>(ipuEB);
      numIPUs = options.numIPUs;
      tilesPerIPU = ipuEB->getTilesPerIPU();
    } else {
      engineBuilder =
        std::unique_ptr<EngineBuilder>(new CPUEngineBuilder(*env));
      numIPUs = 1;
      tilesPerIPU = 1;
    }
    EngineBuilder &eb = *engineBuilder;

    std::cerr << "Constructing program\n";
    Tensor isTraining = graph->addTensor("unsigned", {1});
    if (mapping) {
      mapping->setMapping(isTraining, 0);
    }
    inputLayer = std::unique_ptr<InputLayer>(
      new InputLayer(*this, -1, data, isTraining));
    lossLayer = std::unique_ptr<LossLayer>(
      new LossLayer(*this, hiddenLayerSpecs.size(), data, lossType,
                    isTraining));

    for (size_t i = 0; i < hiddenLayerSpecs.size(); ++i) {
      hiddenLayers.push_back(hiddenLayerSpecs[i]->makeLayer(*this, i));
    }

    auto initParamsProg = Sequence();
    auto startBatchProg = Sequence();
    auto fwdProg = Sequence();
    auto bwdProg = Sequence();
    auto weightSyncProg = Sequence();

    inputLayer->init(*graph, mapping.get());
    startBatchProg.add(inputLayer->startBatch(*graph));
    fwdProg.add(inputLayer->forward(*graph, mapping.get()));

    initParamsProg.add(inputLayer->initParams(*graph));

    for (unsigned i = 0; i < hiddenLayers.size(); ++i) {
      hiddenLayers[i]->init(*graph, mapping.get());
      startBatchProg.add(hiddenLayers[i]->startBatch(*graph));
      fwdProg.add(hiddenLayers[i]->forward(*graph, mapping.get()));
      initParamsProg.add(hiddenLayers[i]->initParams(*graph));
      std::cout << "-- Layer " << i << "\n";
      hiddenLayers[i]->describe(std::cout);
    }

    lossLayer->init(*graph, mapping.get());
    startBatchProg.add(lossLayer->startBatch(*graph));
    fwdProg.add(lossLayer->forward(*graph, mapping.get()));
    initParamsProg.add(lossLayer->initParams(*graph));

    if (netType == TrainingNet) {
      bwdProg.add(lossLayer->backward(*graph));
      weightSyncProg.add(lossLayer->weightSync(*graph));
      for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
        bwdProg.add(hiddenLayers[i]->backward(*graph));
        weightSyncProg.add(hiddenLayers[i]->weightSync(*graph));
      }
      bwdProg.add(inputLayer->backward(*graph));
      weightSyncProg.add(inputLayer->weightSync(*graph));
    }

    if (options.useIPUModel) {
      IPUModelEngineBuilder *ipuEB =
        static_cast<IPUModelEngineBuilder *>(&eb);
      ipuEB->setNumIPUs(options.numIPUs);
      unsigned numTiles = ipuEB->getTilesPerIPU() * ipuEB->getNumIPUs();
      ipuEB->setIPUExchangeImplementation(IPUModelEngineBuilder::BARE_NAKED_WITH_MULTICAST);
      ipuEB->setGlobalSyncCycles(500);
      std::vector <Tensor> tensors = graph->getTensors();
      std::vector <ComputeSet> computeSets = graph->getComputeSets();

      IPUModelEngineBuilder::UserTilePartitioner p(*mapping);
      ipuEB->setTilePartitioner(p);
      switch (ipuEB->getNumIPUs()) {
      case 1:
        break;
      case 2:
        ipuEB->setGlobalExchangeConstraints({
            IPUModelEngineBuilder::GlobalExchangeConstraint(140*1024*1024*1024LL,
              {IPUModelEngineBuilder::GlobalExchangeFlow(0,1)}),
            IPUModelEngineBuilder::GlobalExchangeConstraint(140*1024*1024*1024LL,
              {IPUModelEngineBuilder::GlobalExchangeFlow(1,0)}),
             });
        break;
      default:
        std::cerr << "IPU modeling does not support > 2 IPUs\n";
        std::abort();
      }
    }

    hIsTraining = (netType == TrainingNet);
    std::cerr << "Creating engine\n";
    auto prog = Sequence();
    prog.add(Copy(isTraining, &hIsTraining));
    prog.add(startBatchProg);
    auto doBatchProg = Sequence();
    doBatchProg.add(fwdProg);
    if (netType == TrainingNet) {
      doBatchProg.add(bwdProg);
      #if 0
      doBatchProg->add(ifprog(isTraining,
                               *bwdProg,
                               *Sequence()));
      #endif
    }
    unsigned repeatSize = options.singleBatchProfile ? 1 : batchSize;
    prog.add(Repeat(repeatSize, doBatchProg));
    if (netType == TrainingNet) {
      #if 0
      prog.add(ifprog(isTraining,*weightSyncProg,*Sequence()));
      #endif
    }
    engine = eb.makeEngine(*graph, {&initParamsProg, &prog});
  }

  /* When a Net object is constructed the corrensponding poplar graph is
     made */
  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<LayerSpec>> &hiddenLayerSpecs,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions()) : netType(netType), options(options),
                                           batchSize(batchSize),
                         hiddenLayerSpecs(std::move(hiddenLayerSpecs)),
                         eta(learningRate),
                         dType(getDTypeString(dType))
      {
    initialize(data, lossType);
  }

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<LayerSpec>> &&hiddenLayerSpecs,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions()) : netType(netType), options(options),
                         batchSize(batchSize),
                         hiddenLayerSpecs(std::move(hiddenLayerSpecs)),
                         eta(learningRate),
                         dType(getDTypeString(dType))
      {
    initialize(data, lossType);
  }

  void run(unsigned numBatches) {
    /* All this method needs to do is set the relevant parameters and
       run the control program. */
    std::cerr << "Running program\n";
    if (options.doComputation) {
      if (netType == TrainingNet) {
        engine->run(0); // initialize params
        for (unsigned i = 0; i < numBatches; i++) {
          if (!options.singleBatchProfile &&
              i % options.numBatchesBetweenTest == 0) {
            hIsTraining = 0;
            lossLayer->resetNumCorrect();
            for (unsigned j = 0; j < numTestBatches; j++) {
              engine->run(1);
            }
            float numCorrect = lossLayer->getNumCorrect();
            unsigned numTests = (numTestBatches * batchSize);
            float percentCorrect = 100 * numCorrect / numTests;
            std::cout << "--- Accuracy after " << i << " batches = "
                      << percentCorrect << "%\n";
          }
          hIsTraining = 1;
          engine->run(1);
        }
      } else {
        hIsTraining = 0;
        engine->run(0);
        lossLayer->resetNumCorrect();
        for (unsigned i = 0; i < numBatches; i++) {
          engine->run(1);
        }
        float numCorrect = lossLayer->getNumCorrect();
        unsigned numTests = (numTestBatches * batchSize);
        float percentCorrect = 100 * numCorrect / numTests;
        std::cout << "--- Accuracy = " << percentCorrect << "%\n";
      }
    }
    if (options.useIPUModel) {
      IPUModelEngine *ipuEngine = static_cast<IPUModelEngine *>(&*engine);
      ipuEngine->report(std::cout, true);
    }
  }

};

#endif //_net_hpp_
