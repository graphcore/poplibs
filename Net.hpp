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


typedef enum DType {
  FP16,
  FP32
} DType;


typedef enum NetType {
  TrainingNet,
  TestOnlyNet
} NetType;

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

/* The layer class represents a single layer in the net.
 */
class Layer {
protected:
  unsigned numIPUs;
  unsigned tilesPerIPU;
  void init(unsigned numIPUs_, unsigned tilesPerIPU_) {
    numIPUs = numIPUs_;
    tilesPerIPU = tilesPerIPU_;
  }
  void mapTensor(Tensor t, IPUModelEngineBuilder::TileMapping *mapping) {
    if (!mapping)
      return;
    std::uint64_t size = t.numElements();
    const auto numTiles = tilesPerIPU * numIPUs;
    for (unsigned i = 0; i < numTiles; ++i) {
      const auto begin = (size * i) / numTiles;
      const auto end = (size * (i + 1)) / numTiles;
      if (begin == end)
        continue;
      mapping->setMapping(t.flatten().slice(begin, end), i);
    }
  }
  void mapComputeSet(const Graph &graph, ComputeSet c,
                     IPUModelEngineBuilder::TileMapping *mapping) {
    if (!mapping)
      return;
    auto cs = graph.getComputeSet(c);
    std::uint64_t size = cs.size();
    const auto numTiles = tilesPerIPU * numIPUs;
    for (unsigned i = 0; i < numTiles; ++i) {
      const auto begin = (size * i) / numTiles;
      const auto end = (size * (i + 1)) / numTiles;
      if (begin == end)
        continue;
      for (unsigned j = begin; j != end; ++j) {
        mapping->setMapping(cs[j], i);
      }
    }
  }
public:
  virtual void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping,
                    Layer *prev, Layer *next,
                    NetType netType, float eta, unsigned batchSize,
                    unsigned numIPUs, unsigned tilesPerIPU,
                    const std::string &dType) = 0;
  virtual Program initParams(Graph &graph) = 0;
  virtual Program startBatch(Graph &graph) = 0;
  virtual Program forward(Graph &graph,
                          IPUModelEngineBuilder::TileMapping *mapping,
                          Layer *prev) = 0;
  virtual Program backward(Graph &graph, Layer *prev, Layer *next) = 0;
  virtual Program weightSync(Graph &graph) = 0;
  virtual void describe(std::ostream &out) = 0;
  virtual Tensor getFwdActivations() const = 0;
  virtual Tensor getFwdZs() const = 0;
  virtual NonLinearityType getNonLinearityType() const {
    return NON_LINEARITY_NONE;
  };
  virtual Tensor getBwdErrors() const = 0;
};

class InputLayer : public Layer {
  DataSet &data;
  Tensor out, z, isTraining;
public:
  InputLayer(DataSet &data, Tensor isTraining) :
    data(data), isTraining(isTraining) {}

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping,
            Layer *prev, Layer *next, NetType netType,
            float eta, unsigned batchSize,
            unsigned numIPUs, unsigned tilesPerIPU,
            const std::string &dType) {
    Layer::init(numIPUs, tilesPerIPU);
    out = graph.addTensor(dType, data.dim);
    z = graph.addTensor(dType, data.dim);
    mapTensor(out, mapping);
    mapTensor(z, mapping);
  }
  Program initParams(Graph &graph) { return Sequence(); }
  Program startBatch(Graph &graph) { return Sequence(); }
  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping,
                  Layer *prev) {
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

  Program backward(Graph &graph, Layer *prev, Layer *next) { return Sequence(); }
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
  std::string dType;
  ComputeSet fwd;
public:
  LossLayer(DataSet &data, LossType lossType, Tensor isTraining) :
    data(data), lossType(lossType), isTraining(isTraining) {}

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping,
            Layer *prev, Layer *next, NetType netType,
            float eta, unsigned batchSize,
            unsigned numIPUs, unsigned tilesPerIPU,
            const std::string &dType) {
    Layer::init(numIPUs, tilesPerIPU);
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
    this->dType = dType;
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
  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping,
                  Layer *prev) {
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
  Program backward(Graph &graph, Layer *prev, Layer *next) { return Sequence(); }
  Program weightSync(Graph &graph) { return Sequence(); }
  void describe(std::ostream &out) {}
  Tensor getFwdActivations() const { return {}; }
  Tensor getFwdZs() const { return {}; }
  Tensor getBwdErrors() const { return errors; }
};

/* This utility function wraps a vector of normal pointers as unique_ptrs.
   It allows the hidden layer array to be initializes with an
   initializer list. */
static std::vector<std::unique_ptr<Layer>>
makeLayers(std::vector<Layer *> vs)
{
  std::vector<std::unique_ptr<Layer>> xs;
  for (auto p: vs)
    xs.push_back(std::unique_ptr<Layer>(p));
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
    unsigned numIPUs, tilesPerIPU;
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
      new InputLayer(data, isTraining));
    lossLayer = std::unique_ptr<LossLayer>(
      new LossLayer(data, lossType, isTraining));
    auto initParamsProg = Sequence();
    auto startBatchProg = Sequence();
    auto fwdProg = Sequence();
    auto bwdProg = Sequence();
    auto weightSyncProg = Sequence();

    Layer *first = &**hiddenLayers.begin();
    inputLayer->init(*graph, mapping.get(), 0, first, netType, eta, batchSize,
                     numIPUs, tilesPerIPU, dType);
    startBatchProg.add(inputLayer->startBatch(*graph));
    fwdProg.add(inputLayer->forward(*graph, mapping.get(), 0));

    initParamsProg.add(inputLayer->initParams(*graph));

    for (unsigned i = 0; i < hiddenLayers.size(); ++i) {
      Layer *prev = (i == 0) ? &*inputLayer : &*hiddenLayers[i-1];
      Layer *next = (i == hiddenLayers.size() - 1) ?
                          &*lossLayer :
                          &*hiddenLayers[i+1];
      hiddenLayers[i]->init(*graph, mapping.get(), prev, next, netType, eta,
                            batchSize, numIPUs, tilesPerIPU, dType);
      startBatchProg.add(hiddenLayers[i]->startBatch(*graph));
      fwdProg.add(hiddenLayers[i]->forward(*graph, mapping.get(), prev));
      initParamsProg.add(hiddenLayers[i]->initParams(*graph));
      std::cout << "-- Layer " << i << "\n";
      hiddenLayers[i]->describe(std::cout);
    }

    Layer *last = &**(hiddenLayers.end() - 1);
    lossLayer->init(*graph, mapping.get(), last, 0, netType, eta, batchSize,
                    numIPUs, tilesPerIPU, dType);
    startBatchProg.add(lossLayer->startBatch(*graph));
    fwdProg.add(lossLayer->forward(*graph, mapping.get(), last));
    initParamsProg.add(lossLayer->initParams(*graph));

    if (netType == TrainingNet) {
      bwdProg.add(lossLayer->backward(*graph, last, 0));
      weightSyncProg.add(lossLayer->weightSync(*graph));
      for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
        Layer *prev = (i == 0) ? &*inputLayer : &*hiddenLayers[i-1];
        Layer *next = (i == hiddenLayers.size() - 1) ?
                          &*lossLayer :
                          &*hiddenLayers[i+1];
        bwdProg.add(hiddenLayers[i]->backward(*graph, prev, next));
        weightSyncProg.add(hiddenLayers[i]->weightSync(*graph));
      }
      bwdProg.add(inputLayer->backward(*graph, 0, first));
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
      std::vector<std::unique_ptr<Layer>> &hiddenLayers,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions()) : netType(netType), options(options),
                                           batchSize(batchSize),
                         hiddenLayers(std::move(hiddenLayers)),
                         eta(learningRate),
                         dType(getDTypeString(dType))
      {
    initialize(data, lossType);
  }

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<Layer>> &&hiddenLayers,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions()) : netType(netType), options(options),
                         batchSize(batchSize),
                         hiddenLayers(std::move(hiddenLayers)),
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
