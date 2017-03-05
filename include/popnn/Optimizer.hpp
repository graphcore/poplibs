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
#include <set>
#include "popnn/exceptions.hpp"
#include "popnn/FullyConnectedPlan.hpp"
#include "popnn/ConvPlan.hpp"
#include "popnn/NonLinearityDef.hpp"
#include "popnn/ResidualDef.hpp"
#include "popnn/NetDef.hpp"

namespace popnn {
namespace optimizer {

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
Exp residualAdd(Exp a, Exp b, ResidualMethod method = RESIDUAL_PAD);
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

  DType dataType = FP16, partialsType = FP32;

  /* Perform memory optimisation if cycles performance is
   * within percentage excess of optimum cycles performance
   *
   * i.e. if C_opt is the optimium cycles performance bound,
   *  allow memory optimisations in if cycles cost is
   *    < C_opt * (100 + percentageCyclesExcessForMemOptim)/100
   */
  unsigned percentageCyclesExcessForMemOptim = 0;
  conv::PlanControl convPlanControl;
};

bool parseCommandLine(int argc, char **argv, OptimizerOptions &options);

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
  std::map<const ExpImpl *, fc::Plan> fullyConnectedPlan;
  std::map<const ExpImpl *, poplar::Tensor> out;
  std::map<const ExpImpl *, std::vector<poplar::Tensor>> inGradient;
  std::map<const ExpImpl *, std::vector<poplar::Tensor>> params;
  std::map<const ExpImpl *, conv::Plan> fwdConvPlans, bwdConvPlans, wuConvPlans;
  std::uint64_t fwdFlops, bwdFlops, wuFlops;
  std::uint64_t numParams;
  double fwdPerfectCycleTime, bwdPerfectCycleTime, wuPerfectCycleTime;
  conv::Planner planner;
  const DataSet *dataSet;
  poplar::Tensor expected, feedIn;

  conv::Plan getBwdConvPlan(const ExpImpl *exp, unsigned prevDimY,
                            unsigned prevDimX, unsigned prevNumChans);
  conv::Plan getFwdConvPlan(const ExpImpl *exp, unsigned inDimY,
                            unsigned inDimX, unsigned inNumChans);
  conv::Plan
  getWuConvPlan(const ExpImpl *exp, unsigned prevDimY, unsigned prevDimX,
                unsigned prevNumChans, unsigned actsChansPerGroup,
                unsigned deltasChanPerGroup, unsigned weightOutChansPerGroup);
  unsigned getRequiredChansPerGroupBwd(const ExpImpl *exp);
  unsigned getRequiredChansPerGroupFwd(const ExpImpl *exp, unsigned inDimY,
                                       unsigned inDimX, unsigned inNumChans);
  void outputConvDescription(const ExpImpl *exp,
                             unsigned inDimY, unsigned inDimX,
                             unsigned inNumChans,
                             unsigned kernelSizeY, unsigned kernelSizeX,
                             unsigned strideY, unsigned strideX,
                             unsigned paddingY, unsigned paddingX,
                             unsigned outNumChans);
  void outputDescription(const ExpImpl *exp);
  void createSchedule(const Exp &exp);
  struct ConvOp;
  struct ConvBwdWeightsOp; struct ConvWuOp;
  poplar::program::Program
  createConvLayerFwd(const ExpImpl *exp,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     unsigned numChannels,
                     poplar::program::Sequence &initParamsProg,
                     ConvOp &doConv,
                     const std::string &debugPrefix);
  poplar::program::Program
  createResidualLayerFwd(const ExpImpl *exp, unsigned layerIndex,
                         const std::string &debugPrefix);
  void genFwd(poplar::program::Sequence &fwdProg,
              poplar::program::Sequence &initParamsProg,
              struct ConvOp &convOp);
  void createInGradients(const ExpImpl *exp, unsigned index);
  poplar::program::Program
  createConvLayerBwd(const ExpImpl *exp, poplar::Tensor outGradient,
                     unsigned layerIndex,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     bool backwardPassRequired,
                     ConvBwdWeightsOp &convBwdWeights,
                     ConvOp &doConv, ConvWuOp &convWU,
                     const std::string &debugPrefix);
  void genBwd(poplar::program::Sequence &bwdProg,
              ConvOp &convOp,
              ConvWuOp &convWuOp,
              ConvBwdWeightsOp &convBwdWeightsOp);
  void reportTotals();
  void createEngine(poplar::program::Program initParamsProg,
                    poplar::program::Program fwdProg,
                    poplar::program::Program bwdProg);
};

}} // end namespace popnn::optimizer

#endif //_net_hpp_
