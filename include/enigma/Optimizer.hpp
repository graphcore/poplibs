#ifndef _enigma_Optimizer_hpp_
#define _enigma_Optimizer_hpp_
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <random>
#include <array>
#include <set>
#include "poplin/MatMul.hpp"
#include "popnn/NonLinearity.hpp"
#include "popnn/Residual.hpp"
#include "popnn/Loss.hpp"
#include "popstd/GraphFunction.hpp"
#include "popconv/Convolution.hpp"

namespace enigma {


/* A data set full of test and training data along with its dimensions */
class DataSet {
public:
  std::unique_ptr<float[]> testData, trainingData;
  std::unique_ptr<unsigned[]> testLabels, trainingLabels;
  unsigned dataSize, numTest, numTraining;
  std::vector<std::size_t> dim;
};

// A context holds information about an expression tree to be optimized.
// All expressions will hold references into the context so its lifetime must
// be longer than any expressions using it.
class ContextImpl;
class Context {
public:
  std::unique_ptr<ContextImpl> impl;
  Context();
  ~Context();
};

// An expression to be optimized.
class ExpImpl;
class Exp {
public:
  ExpImpl *impl;
  Exp() = default;
  Exp(ExpImpl *impl) : impl(impl) {} ;
};

// A rectangle, used to provide information about rectangular parameters (e.g
// convolution stride).
struct Rect {
  unsigned height;
  unsigned width;
  Rect(unsigned height, unsigned width) : height(height), width(width) {}
};

// These functions allow you to build up an expression to be optimized.
Exp feed(const DataSet &, Context &context);
Exp conv2d(unsigned kernelSize, unsigned stride, unsigned padding,
           unsigned channels, Exp in);
Exp conv2d(const Rect &kernelSize, const Rect &stride, const Rect &padding,
           unsigned channels, Exp in);
Exp relu(Exp in);
Exp sigmoid(Exp in);
Exp maxPool(unsigned windowSize, unsigned stride, Exp in);
Exp maxPool(unsigned windowSize, unsigned stride, unsigned padding,
            Exp in);
Exp maxPool(const Rect &windowSize, const Rect &stride, const Rect &padding,
            Exp in);
Exp fullyconnected(unsigned channels, Exp in);
Exp residualAdd(Exp a, Exp b,
                popnn::ResidualMethod method =
                   popnn::ResidualMethod::RESIDUAL_PAD);
Exp softMaxCrossEntropyLoss(Exp in, Exp out);
Exp sumSquaredLoss(Exp in, Exp out);

enum DType {
  FP16,
  FP32
};

class OptimizerOptions {
public:
  OptimizerOptions();

  // Options to override defaults in DeviceInfo (initialized in default ctor).
  unsigned numIPUs;
  unsigned tilesPerIPU;
  unsigned ipuExchangeBandwidth;
  unsigned memoryBytesPerTile;
  unsigned dataPathWidth;
  unsigned convUnitPipelineDepth;
  unsigned fp16InFp16OutConvUnitsPerTile;
  unsigned fp16InFp32OutConvUnitsPerTile;
  unsigned fp32InFp32OutConvUnitsPerTile;
  unsigned convUnitCoeffLoadBytesPerCycle;
  bool supportsSuperTileSendReceive;

  // Other options.
  bool training = true;
  bool useIPUModel = false;
  bool doComputation = true;
  bool doTestsDuringTraining = true;
  unsigned numBatchesBetweenTest = 2500;
  bool reuseLayerImplGraphs = true;
  bool ignoreData = false;
  bool useWinogradConv = false;
  unsigned winogradPatchSize = 4;
  unsigned batchSize = 1;
  bool showPlanInfo = false;
  bool skipFwd = false;
  bool skipBwd = false;
  bool skipWU = false;
  float learningRate = 0.9;
  bool inPlaceParamUpdate = true;

  DType dataType = FP16, partialsType = FP32;

  popconv::ConvOptions convOptions;
};

bool parseCommandLine(int argc, char **argv, OptimizerOptions &options);

using TensorSig = std::pair<std::string, std::vector<std::size_t>>;

// The optimizer class which will try and optimize the loss of an expression
// over the data provided by any feeds in the expression.
class Optimizer {
public:
  Optimizer(const Exp &exp, OptimizerOptions options);
  ~Optimizer();
  void run(unsigned numIterations);
private:
  OptimizerOptions options;

  /* Poplar program creation state. */
  std::unique_ptr<poplar::Graph> graph;
  std::unique_ptr<poplar::Engine> engine;
  std::map<const ExpImpl *, std::vector<std::unique_ptr<float[]>>> hParams;
  std::mt19937 randomEngine;
  unsigned numTestBatches;
  unsigned hNumCorrect;
  std::string dType, partialsType;

  std::vector<const ExpImpl *> schedule;
  std::map<const ExpImpl *, std::set<const ExpImpl *>> uses;
  poplin::PlanningCache poplinCache;
  std::map<const ExpImpl *, poplar::Tensor> out;
  std::map<const ExpImpl *, std::vector<poplar::Tensor>> inGradient;
  std::map<const ExpImpl *, std::vector<poplar::Tensor>> params;
  std::uint64_t fwdFlops, bwdFlops, wuFlops;
  std::uint64_t numParams;
  double fwdPerfectCycleTime, bwdPerfectCycleTime, wuPerfectCycleTime;
  popconv::PlanningCache convCache;
  const DataSet *dataSet;
  poplar::Tensor expected, feedIn;

  double getPerfectCycleTime(unsigned flops, const std::string &dType,
                             bool useVectors, bool useAmp);
  void outputConvDescription(const ExpImpl *exp,
                             unsigned inDimY, unsigned inDimX,
                             unsigned inNumChans,
                             unsigned kernelSizeY, unsigned kernelSizeX,
                             unsigned strideY, unsigned strideX,
                             unsigned paddingY, unsigned paddingX,
                             unsigned outNumChans);
  void outputDescription(const ExpImpl *exp);
  void createSchedule(const Exp &exp);
  using ConvKey =
    std::tuple<std::vector<unsigned>, std::vector<unsigned>,
               TensorSig, TensorSig, unsigned,
               std::string, bool, bool>;
  std::map<ConvKey, popstd::graphfn::TensorFunction> convGraphCache;
  using BwdWeightKey = std::vector<TensorSig>;
  std::map<BwdWeightKey, popstd::graphfn::VoidFunction> bwdWeightGraphCache;
  using WUKey =
    std::tuple<TensorSig, TensorSig, TensorSig, unsigned, unsigned,
               unsigned, unsigned, float>;
  std::map<WUKey, popstd::graphfn::VoidFunction> wuGraphCache;

  void
  createBwdWeights(poplar::Graph &graph,
                   unsigned prevNumChans,
                   poplar::Tensor weights,
                   poplar::Tensor deltasIn,
                   poplar::Tensor bwdWeights,
                   unsigned strideY, unsigned strideX,
                   unsigned paddingY, unsigned paddingX,
                   bool isFractional,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix);

  void
  doConvolutionWeightUpdate(poplar::Graph &graph,
                            poplar::Tensor zDeltas, poplar::Tensor weights,
                            poplar::Tensor activations,
                            unsigned strideY, unsigned strideX,
                            unsigned paddingY, unsigned paddingX,
                            float learningRate,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix);

  poplar::Tensor
  doConvolution(poplar::Graph &graph,
                const std::vector<unsigned> &stride,
                const std::vector<unsigned> &padding, unsigned outNumChans,
                poplar::Tensor in, poplar::Tensor weights,
                const std::string &partialsType,
                bool isFractional, bool transposeAndFlipWeights,
                poplar::program::Sequence &prog,
                const std::string &debugPrefix = "");

  poplar::Tensor
  createConvLayerFwd(const ExpImpl *exp,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     unsigned numChannels,
                     poplar::program::Sequence &initParamsProg,
                     poplar::program::Sequence &prog,
                     const std::string &debugPrefix);

  poplar::program::Program
  createResidualLayerFwd(const ExpImpl *exp, unsigned layerIndex,
                         const std::string &debugPrefix);
  void genFwd(poplar::program::Sequence &fwdProg,
              poplar::program::Sequence &initParamsProg);

  void
  createConvLayerBwd(const ExpImpl *exp, poplar::Tensor outGradient,
                     unsigned layerIndex,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     bool backwardPassRequired,
                     poplar::program::Sequence &actualBwdprog,
                     const std::string &debugPrefix);
  void genBwd(poplar::program::Sequence &bwdProg);
  void reportTotals();
  void createEngine(poplar::program::Program initParamsProg,
                    poplar::program::Program fwdProg,
                    poplar::program::Program bwdProg);
};

} // end namespace enigma

#endif //_enigma_Optimizer_hpp_
