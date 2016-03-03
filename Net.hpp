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

using namespace poplar;
using namespace poplar::program;

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
  bool useSuperTiles = false;
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
public:
  virtual void init(Graph &graph, Layer *prev, Layer *next,
                    NetType netType, float eta, unsigned batchSize,
                    const std::string &dType) = 0;
  virtual Program initParams(Graph &graph) = 0;
  virtual Program startBatch(Graph &graph) = 0;
  virtual Program forward(Graph &graph, Layer *prev) = 0;
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

  void init(Graph &graph, Layer *prev, Layer *next, NetType netType,
            float eta, unsigned batchSize, const std::string &dType) {
    out = graph.addTensor(dType, data.dim);
    z = graph.addTensor(dType, data.dim);
  }
  Program initParams(Graph &graph) { return Sequence(); }
  Program startBatch(Graph &graph) { return Sequence(); }
  Program forward(Graph &graph, Layer *prev) {
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
  Tensor errors, expected, loss, numCorrect, isTraining;
  unsigned hNumCorrect;
  std::string dType;
  ComputeSet fwd;
public:
  LossLayer(DataSet &data, LossType lossType, Tensor isTraining) :
    data(data), lossType(lossType), isTraining(isTraining) {}

  void init(Graph &graph, Layer *prev, Layer *next, NetType netType,
            float eta, unsigned batchSize, const std::string &dType) {
    errors = graph.addTensor(dType, {prev->getFwdActivations().numElements()});
    expected = graph.addTensor("unsigned", {1});
    loss = graph.addTensor(dType, {1});
    numCorrect = graph.addTensor("unsigned", {1});
    this->dType = dType;
    fwd = graph.createComputeSet();
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
  Program forward(Graph &graph, Layer *prev) {
    auto v = graph.addVertex(fwd, "CalcLoss",
                             {{"zIn", prev->getFwdZs().flatten()},
                              {"errorOut", errors},
                              {"label", expected[0]},
                              {"lossType", lossType},
                              {"loss", loss[0]},
                              {"numCorrect", numCorrect[0]}});
    graph.setFieldSize(v["probs"], prev->getFwdActivations().numElements());
    graph.setInitialValue(v["nonLinearityType"], prev->getNonLinearityType());
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
    std::cerr << "Constructing program\n";
    Tensor isTraining = graph->addTensor("unsigned", {1});
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
    inputLayer->init(*graph, 0, first, netType, eta, batchSize, dType);
    startBatchProg.add(inputLayer->startBatch(*graph));
    fwdProg.add(inputLayer->forward(*graph, 0));

    initParamsProg.add(inputLayer->initParams(*graph));

    for (unsigned i = 0; i < hiddenLayers.size(); ++i) {
      Layer *prev = (i == 0) ? &*inputLayer : &*hiddenLayers[i-1];
      Layer *next = (i == hiddenLayers.size() - 1) ?
                          &*lossLayer :
                          &*hiddenLayers[i+1];
      hiddenLayers[i]->init(*graph, prev, next, netType, eta, batchSize, dType);
      startBatchProg.add(hiddenLayers[i]->startBatch(*graph));
      fwdProg.add(hiddenLayers[i]->forward(*graph, prev));
      initParamsProg.add(hiddenLayers[i]->initParams(*graph));
      std::cout << "-- Layer " << i << "\n";
      hiddenLayers[i]->describe(std::cout);
    }

    Layer *last = &**(hiddenLayers.end() - 1);
    lossLayer->init(*graph, last, 0, netType, eta, batchSize, dType);
    startBatchProg.add(lossLayer->startBatch(*graph));
    fwdProg.add(lossLayer->forward(*graph, last));
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

    /* Now that the program is constructed, a poplar engine is created. */
    if (options.useIPUModel) {
      engineBuilder =
        std::unique_ptr<EngineBuilder>(new IPUModelEngineBuilder(*env));
    } else {
      engineBuilder =
        std::unique_ptr<EngineBuilder>(new CPUEngineBuilder(*env));
    }

    EngineBuilder &eb = *engineBuilder;

    if (options.useIPUModel) {
      IPUModelEngineBuilder *ipuEB =
        static_cast<IPUModelEngineBuilder *>(&eb);
      ipuEB->setNumIPUs(options.numIPUs);
      unsigned superTileDiv = options.useSuperTiles ? 4 : 1;
      ipuEB->setTilesPerIPU(1152/superTileDiv);
      ipuEB->setNumBytesPerTile(256*superTileDiv*1024);
      unsigned numTiles = ipuEB->getTilesPerIPU() * ipuEB->getNumIPUs();
      ipuEB->setIPUExchangeImplementation(IPUModelEngineBuilder::BARE_NAKED_WITH_MULTICAST);
      ipuEB->setGlobalSyncCycles(500);
      IPUModelEngineBuilder::TileMapping mapping(*graph);
      std::vector <Tensor> tensors = graph->getTensors();
      std::vector <ComputeSet> computeSets = graph->getComputeSets();

      
      for (Tensor t : tensors) {
        std::uint64_t size = t.numElements();
        for (unsigned j = 0; j < numTiles; ++j) {
          const auto begin = (size * j) / numTiles;
          const auto end = (size * (j + 1)) / numTiles;
          if (begin == end)
            continue;
          mapping->setMapping(t.flatten().slice(begin, end),
                              j);
        }
      }

      for (ComputeSet c : computeSets) {
        auto cs = graph->getComputeSet(c);
        std::uint64_t size = cs.size();
        for (unsigned j = 0; j < numTiles; ++j) {
          const auto begin = (size * j) / numTiles;
          const auto end = (size * (j + 1)) / numTiles;
          if (begin == end)
            continue;
          for (unsigned i = begin; i != end; ++i) {
            mapping->setMapping(cs[i], j);
          }
        }
      }
      IPUModelEngineBuilder::UserTilePartitioner p(mapping);
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
      ipuEngine->report(std::cout);
    }
  }

};

#endif //_net_hpp_
