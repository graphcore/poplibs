#ifndef _net_hpp_
#define _net_hpp_
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <random>
#include <array>
#include "popnn/exceptions.hpp"
#include "popnn/FullyConnectedPlan.hpp"
#include "popnn/ConvPlan.hpp"
#include "popnn/NonLinearityDef.hpp"
#include "popnn/ResidualDef.hpp"
#include "popnn/NetDef.hpp"

class Layer { public: virtual ~Layer() {};};

class ConvLayer : public Layer {
public:
  unsigned kernelSizeY;
  unsigned kernelSizeX;
  unsigned strideY;
  unsigned strideX;
  unsigned paddingY;
  unsigned paddingX;
  unsigned numChannels;
  NonLinearityType nonLinearityType;
  ConvLayer(unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned numChannels,
            NonLinearityType nonLinearityType) :
  kernelSizeY(kernelSize),
  kernelSizeX(kernelSize),
  strideY(stride),
  strideX(stride),
  paddingY(padding),
  paddingX(padding),
  numChannels(numChannels),
  nonLinearityType(nonLinearityType) {}

  ConvLayer(std::array<unsigned, 2> kernelSize,
            std::array<unsigned, 2> stride,
            std::array<unsigned, 2> padding,
            unsigned numChannels,
            NonLinearityType nonLinearityType) :
  kernelSizeY(kernelSize[0]),
  kernelSizeX(kernelSize[1]),
  strideY(stride[0]),
  strideX(stride[1]),
  paddingY(padding[0]),
  paddingX(padding[1]),
  numChannels(numChannels),
  nonLinearityType(nonLinearityType) {}
};

class ResidualLayer : public Layer {
public:
  // resIndex is a list of offsets to the layers that input to this layer; 1
  // is the immediately preceding layer. Exactly two offsets must be supplied
  std::vector<unsigned> resIndex;
  NonLinearityType nonLinearityType;
  enum ResidualMethod resMethod;
  ResidualLayer(std::vector<unsigned> resIndex,
                NonLinearityType nonLinearityType,
                enum ResidualMethod resMethod) :
    resIndex(resIndex),
    nonLinearityType(nonLinearityType),
    resMethod(resMethod) {}
};

class MaxPoolLayer : public Layer {
public:
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  MaxPoolLayer(unsigned kernelSize,
               unsigned stride,
               unsigned padding=0) :
  kernelSize(kernelSize),
  stride(stride),
  padding(padding) {}
};

class FullyConnectedLayer : public Layer {
public:
  unsigned size;
  NonLinearityType nonLinearityType;
  FullyConnectedLayer(unsigned size,
                      NonLinearityType nonLinearityType) :
    size(size), nonLinearityType(nonLinearityType) {}
};

/* This utility function wraps a vector of normal pointers as unique_ptrs.
   It allows the hidden layer array to be initializes with an
   initializer list. */
inline std::vector<std::unique_ptr<Layer>>
makeLayers(std::vector<Layer *> vs)
{
  std::vector<std::unique_ptr<Layer>> xs;
  for (auto p: vs)
    xs.push_back(std::unique_ptr<Layer>(p));
  return xs;
}


enum NetType {
  TrainingNet,
  TestOnlyNet
};

/* A data set full of test and training data along with its dimensions */
class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
  std::vector<std::size_t> dim;
};

enum DType {
  FP16,
  FP32
};

class NetOptions {
public:
  bool useIPUModel = false;
  bool doComputation = true;
  bool doTestsDuringTraining = true;
  unsigned numIPUs = 1;
  unsigned tilesPerIPU = 1216;
  unsigned ipuExchangeBandwidth = 4;
  unsigned memoryBytesPerTile = 1024 * 256;
  unsigned numBatchesBetweenTest = 2500;
  bool reuseLayerImplGraphs = true;
  bool ignoreData = false;
  unsigned dataPathWidth = 64;
  unsigned convUnitPipelineDepth = 4;
  unsigned fp16InFp16OutConvUnitsPerTile = 8;
  unsigned fp16InFp32OutConvUnitsPerTile = 8;
  unsigned fp32InFp32OutConvUnitsPerTile = 4;
  unsigned convUnitCoeffLoadBytesPerCycle = 16;
  bool useWinogradConv = false;
  unsigned winogradPatchSize = 4;
  unsigned batchSize = 1;
};

bool parseCommandLine(int argc, char **argv, NetOptions &options,
                      bool &doTraining);

/* This class represent the entire network. */
class Net {
  NetType netType;
  NetOptions options;

  unsigned batchSize;
  float eta;
  std::vector<std::unique_ptr<Layer>> layers;

  /* Poplar program creation state. */
  std::unique_ptr<poplar::GraphProgEnv> env;
  std::unique_ptr<poplar::Graph> graph;
  std::unique_ptr<poplar::Engine> engine;
  std::unique_ptr<char[]> hAct;
  std::vector<std::unique_ptr<float[]>> hParams;
  std::mt19937 randomEngine;
  unsigned hIsTraining;
  unsigned numTestBatches;
  unsigned hNumCorrect;
  std::string dType;
  std::string partialsType;

  std::map<unsigned, fc::Plan> fullyConnectedPlan;
  std::vector<poplar::Tensor> acts, deltas;
  std::vector<std::pair<unsigned, unsigned>> residualDeltaIdxs;
  std::vector<std::vector<poplar::Tensor>> params;
  std::map<unsigned, conv::Plan> fwdConvPlans, bwdConvPlans, wuConvPlans;
  std::uint64_t numFlops;
  std::uint64_t numParams;
  double perfectCycleTime;

  conv::Planner planner;
  conv::Plan getFwdConvPlan(unsigned i, unsigned prevDimY, unsigned prevDimX,
                            unsigned prevNumChans);
  conv::Plan getBwdConvPlan(unsigned i, unsigned prevDimY, unsigned prevDimX,
                            unsigned prevNumChans);
  conv::Plan getWuConvPlan(unsigned i, unsigned prevDimY, unsigned prevDimX,
                            unsigned prevNumChans);
  unsigned
  getRequiredChansPerGroupFwd(unsigned i, unsigned prevDimY, unsigned prevDimX,
                              unsigned prevNumChans);

  unsigned getRequiredChansPerGroupBwd(int i);

  struct ConvOp;
  poplar::program::Program
  createConvLayerFwd(unsigned i, unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX, unsigned numChannels,
                     poplar::program::Sequence &initParamsProg,
                     ConvOp &op, const std::string &debugPrefix = "");

  struct ConvBwdWeightsOp; struct ConvWuOp;
  poplar::program::Program
  createConvLayerBwd(unsigned i, unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     NonLinearityType nonLinearityType,
                     bool backwardPassRequired,
                     ConvBwdWeightsOp &convBwdWeightsOp,
                     ConvOp &convOp,
                     ConvWuOp &wuOp,
                     const std::string &debugPrefix = "");

  poplar::program::Program
  createResidualLayerFwd(unsigned i,
                         const ResidualLayer &resLayer,
                         const std::string &debugPrefix = "");

  poplar::program::Program
  createResidualLayerBwd(unsigned i, const std::string &debugPrefix = "");

  void outputConvDescription(unsigned inDimY, unsigned inDimX,
                             unsigned inNumChans,
                             unsigned kernelSizeY, unsigned kernelSizeX,
                             unsigned strideY, unsigned strideX,
                             unsigned paddingY, unsigned paddingX,
                             unsigned outNumChans, bool forwardOnly);

  void outputDescription(const Layer *layer, unsigned i, poplar::Tensor in,
                         bool forwardOnly);

  void initialize(DataSet &dataSet, LossType lossType);

public:
  /* When a Net object is constructed the corrensponding poplar graph is
     made */
  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<Layer>> &layers,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions());

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<Layer>> &&layers,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      NetOptions options = NetOptions());

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<Layer>> &layers,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      DType partialsType,
      NetOptions options = NetOptions());

  Net(DataSet &data, unsigned batchSize,
      std::vector<std::unique_ptr<Layer>> &&layers,
      LossType lossType,
      float learningRate,
      NetType netType,
      DType dType,
      DType partialsType,
      NetOptions options = NetOptions());

  void run(unsigned numBatches);
};


namespace popnn {
  std::string findGraphProg();
}

#endif //_net_hpp_
