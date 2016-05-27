#ifndef _net_hpp_
#define _net_hpp_
#include <poplar/Graph.hpp>
#include <poplar/CPUEngine.hpp>
#include <poplar/IPUModelEngine.hpp>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>
#include "neural_net_common.h"
#include <map>
#include <stdexcept>
#include <cassert>
#include "Layer.hpp"

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

class NetOptions {
public:
  bool useIPUModel = false;
  bool doComputation = true;
  bool singleBatchProfile = false;
  bool sharedConvWeights = true;
  unsigned numIPUs = 1;
  unsigned numBatchesBetweenTest = 2500;
  bool reuseLayerImplGraphs = true;
};

bool parseCommandLine(int argc, char **argv, NetOptions &options);

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
  unsigned workerContextsPerTile, numIPUs, tilesPerIPU;

  unsigned getWorkerContextsPerTile() const { return workerContextsPerTile; }
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

  void initialize(DataSet &data, LossType lossType);

  /* When a Net object is constructed the corrensponding poplar graph is
     made */
  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<LayerSpec>> &hiddenLayerSpecs,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions());

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<LayerSpec>> &&hiddenLayerSpecs,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions());

  void run(unsigned numBatches);
};

#endif //_net_hpp_
