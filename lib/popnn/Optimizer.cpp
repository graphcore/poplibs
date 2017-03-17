#include "popnn/Optimizer.hpp"
#include <boost/program_options.hpp>
#include <poplar/HalfFloat.hpp>
#include "popnn/codelets.hpp"
#include "popnn/Convolution.hpp"
#include "popnn/Loss.hpp"
#include "popnn/MaxPool.hpp"
#include "popnn/FullyConnected.hpp"
#include "popnn/FullyConnectedPlan.hpp"
#include "popnn/ActivationMapping.hpp"
#include "Residual.hpp"
#include "VertexTemplates.hpp"
#include "popnn/NonLinearity.hpp"
#include "popnn/Compiler.hpp"
#include "TensorOp.hpp"
#include <iomanip>
#include <array>
#include <deque>
#include <queue>
#include <unordered_set>

using namespace poplar;
using namespace poplar::program;
namespace popnn {
namespace optimizer {

OptimizerOptions::OptimizerOptions() {
  DeviceInfo defaultDevice;
  numIPUs = defaultDevice.numIPUs;
  tilesPerIPU = defaultDevice.tilesPerIPU;
  ipuExchangeBandwidth = defaultDevice.exchangeBytesPerCycle;
  memoryBytesPerTile = defaultDevice.memoryBytesPerTile;
  dataPathWidth = defaultDevice.dataPathWidth;
  convUnitPipelineDepth = defaultDevice.convUnitPipelineDepth;
  fp16InFp16OutConvUnitsPerTile = defaultDevice.fp16InFp16OutConvUnitsPerTile;
  fp16InFp32OutConvUnitsPerTile = defaultDevice.fp16InFp32OutConvUnitsPerTile;
  fp32InFp32OutConvUnitsPerTile = defaultDevice.fp32InFp32OutConvUnitsPerTile;
  convUnitCoeffLoadBytesPerCycle = defaultDevice.convUnitCoeffLoadBytesPerCycle;
  supportsSuperTileSendReceive = defaultDevice.supportsSuperTileSendReceive;
}

bool parseCommandLine(int argc, char **argv, OptimizerOptions &options) {
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("ipus",
     po::value<unsigned>(&options.numIPUs)->default_value(options.numIPUs),
     "Number of IPUs")
    ("tiles-per-ipu",
     po::value<unsigned>(
       &options.tilesPerIPU
     )->default_value(options.tilesPerIPU),
     "Number of tiles per IPU")
    ("bytes-per-tile",
     po::value<unsigned>(&options.memoryBytesPerTile)
         ->default_value(options.memoryBytesPerTile),
     "Amount of memory per tile in bytes")
    ("ipu-exchange-bandwidth",
     po::value<unsigned>(&options.ipuExchangeBandwidth)
         ->default_value(options.ipuExchangeBandwidth),
     "IPU exchange bandwidth per tile in bytes")
    ("data-path-width",
     po::value<unsigned>(
       &options.dataPathWidth
     )->default_value(options.dataPathWidth),
     "Width of the data path in bits")
    ("num-fp16-in-fp16-out-conv-units",
     po::value<unsigned>(
       &options.fp16InFp16OutConvUnitsPerTile
     )->default_value(options.fp16InFp16OutConvUnitsPerTile),
     "Number of convolutional units per tile with fp16 input and fp16 output")
    ("num-fp16-in-fp32-out-conv-units",
     po::value<unsigned>(
         &options.fp16InFp32OutConvUnitsPerTile
     )->default_value(options.fp16InFp32OutConvUnitsPerTile),
     "Number of convolutional units per tile with fp16 input and fp32 output")
    ("num-fp32-in-fp32-out-conv-units",
     po::value<unsigned>(
         &options.fp32InFp32OutConvUnitsPerTile
     )->default_value(options.fp32InFp32OutConvUnitsPerTile),
     "Number of convolutional units per tile with fp32 input and fp32 output")
    ("conv-coeff-load-bytes-per-cycle",
     po::value<unsigned>(
         &options.convUnitCoeffLoadBytesPerCycle
     )->default_value(options.convUnitCoeffLoadBytesPerCycle),
     "Number of bytes of coefficients loaded in the convolutional"
     " unit per cycle")
    ("supertile-exchange",
     po::value<bool>(
       &options.supportsSuperTileSendReceive
     )->default_value(options.supportsSuperTileSendReceive),
      "Supertiles can combine to give 64bit exchange")
    ("graph-reuse",
     po::value<bool>(
       &options.reuseLayerImplGraphs
     )->default_value(options.reuseLayerImplGraphs),
     "Re-use graph structure for similar layers")
    ("train",
     po::value<bool>(
       &options.training
     )->default_value(false),
     "Do training (forward, backward and weight update pass)")
    ("use-winograd-conv",
     po::value<bool>(
       &options.convPlanControl.useWinograd
     )->default_value(options.convPlanControl.useWinograd),
     "Use winograd for convolution layers")
    ("winograd-patch-size",
     po::value<unsigned>(
       &options.convPlanControl.winogradPatchSize
     )->default_value(options.convPlanControl.winogradPatchSize),
     "Patch size for winograd convolution")
    ("use-new-amp-wu",
     po::value<bool>(
       &options.convPlanControl.useNewAMPWU
     )->default_value(options.convPlanControl.useNewAMPWU),
     "Use new AMP weight update method")
    ("batch-size",
     po::value<unsigned>(
       &options.batchSize
     )->default_value(options.batchSize),
     "Batch size")
    ("show-plan-info",
     po::value<bool>(
       &options.showPlanInfo
     )->default_value(options.showPlanInfo),
     "Display result of planning decision for conv layers")
    ("percent-cyc-excess-for-mem-optim",
     po::value<unsigned>(
       &options.percentageCyclesExcessForMemOptim
     )->default_value(options.percentageCyclesExcessForMemOptim),
     "Percentage cycles excess to use for memory optimisation. "
     "if 0, no memory optimisation is performed")
    ("weight-update-method",
     po::value<conv::WeightUpdateMethod>(
         &options.convPlanControl.weightUpdateMethod
     )->default_value(options.convPlanControl.weightUpdateMethod),
     "Weight update method: amp | aop | auto")
    ("skip-fwd",
     po::value<bool>(
       &options.skipFwd
    )->default_value(options.skipFwd),
    "Skip forward pass calculation")
    ("skip-bwd",
     po::value<bool>(
       &options.skipBwd
    )->default_value(options.skipBwd),
    "Skip backward pass calculation")
    ("skip-wu",
    po::value<bool>(
       &options.skipWU
    )->default_value(options.skipWU),
    "Skip weight update pass calculation")
  ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return false;
    }
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return false;
  }
  return true;
}

class ExpImpl {
public:
  Context &context;
  ExpImpl(Context &context) : context(context) {}
  virtual std::vector<const ExpImpl *> deps() const = 0;
  virtual ~ExpImpl() {}
};

// The context provides the memory ownership for expressions.
// Expressions are pointers into the list of expression implementation
// structures held in the context.
class ContextImpl {
public:
  std::deque<std::unique_ptr<ExpImpl>> exps;
  Exp add(ExpImpl *e) {
    exps.emplace_back(e);
    return Exp(exps.back().get());
  }
};

Context::Context() { impl.reset(new ContextImpl()); }
Context::~Context() = default;

class Feed : public ExpImpl {
public:
  const DataSet &dataset;
  Feed(Context &context, const DataSet &dataset) :
     ExpImpl(context), dataset(dataset) {}
  std::vector<const ExpImpl *> deps() const override { return {}; }
  ~Feed() {}
};

Exp feed(const DataSet &dataset, Context &context) {
  return context.impl->add(new Feed(context, dataset));
}

class Conv2d : public ExpImpl {
public:
  unsigned kernelSizeY, kernelSizeX;
  unsigned strideY, strideX;
  unsigned paddingY, paddingX;
  unsigned numChannels;
  ExpImpl *in;
  Conv2d(Context &context, unsigned kernelSize, unsigned stride,
         unsigned padding, unsigned channels, ExpImpl *in) :
    ExpImpl(context), kernelSizeY(kernelSize), kernelSizeX(kernelSize),
    strideY(stride), strideX(stride), paddingY(padding), paddingX(padding),
    numChannels(channels), in(in) {}
  Conv2d(Context &context, const Rect &kernelSize, const Rect &stride,
         const Rect &padding, unsigned channels, ExpImpl *in) :
    ExpImpl(context), kernelSizeY(kernelSize.height),
    kernelSizeX(kernelSize.width), strideY(stride.height),
    strideX(stride.width), paddingY(padding.height), paddingX(padding.width),
    numChannels(channels), in(in) {}
  std::vector<const ExpImpl *> deps() const override { return {in}; }
};

Exp conv2d(unsigned kernelSize, unsigned stride, unsigned padding,
           unsigned channels, Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new Conv2d(context, kernelSize, stride, padding,
                                      channels, in.impl));
}

Exp conv2d(const Rect &kernelSize, const Rect &stride, const Rect &padding,
           unsigned channels, Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new Conv2d(context, kernelSize, stride, padding,
                                      channels, in.impl));
}

class NonLinearity : public ExpImpl {
public:
  NonLinearityType type;
  ExpImpl *in;
  NonLinearity(Context &context, NonLinearityType type, ExpImpl *in) :
    ExpImpl(context), type(type), in(in) {}
  std::vector<const ExpImpl *> deps() const override { return {in}; }
};

Exp relu(Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new NonLinearity(context, NON_LINEARITY_RELU,
                                            in.impl));
}

Exp sigmoid(Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new NonLinearity(context, NON_LINEARITY_SIGMOID,
                                            in.impl));
}

class MaxPool : public ExpImpl {
public:
  unsigned kernelSizeY, kernelSizeX;
  unsigned strideY, strideX;
  unsigned paddingY, paddingX;
  ExpImpl *in;
  MaxPool(Context &context, unsigned windowSize, unsigned stride,
          unsigned padding, ExpImpl *in) :
    ExpImpl(context), kernelSizeY(windowSize), kernelSizeX(windowSize),
    strideY(stride), strideX(stride), paddingY(padding), paddingX(padding),
    in(in) {}
  MaxPool(Context &context, const Rect &windowSize, const Rect &stride,
          const Rect &padding, ExpImpl *in) :
    ExpImpl(context), kernelSizeY(windowSize.height),
    kernelSizeX(windowSize.width), strideY(stride.height),
    strideX(stride.width), paddingY(padding.height), paddingX(padding.width),
    in(in) {}
  std::vector<const ExpImpl *> deps() const override { return {in}; }
};

Exp maxPool(unsigned windowSize, unsigned stride, unsigned padding,
            Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new MaxPool(context, windowSize, stride, padding,
                                       in.impl));
}

Exp maxPool(const Rect &windowSize, const Rect &stride, const Rect &padding,
            Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new MaxPool(context, windowSize, stride, padding,
                                       in.impl));
}

Exp maxPool(unsigned windowSize, unsigned stride, Exp in) {
  return maxPool(windowSize, stride, 0, in);
}

class FullyConnected : public ExpImpl {
public:
  unsigned channels;
  ExpImpl *in;
  FullyConnected(Context &context, unsigned channels, ExpImpl *in) :
    ExpImpl(context), channels(channels), in(in) {}
  std::vector<const ExpImpl *> deps() const override { return {in}; }
};

Exp fullyconnected(unsigned channels, Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new FullyConnected(context, channels, in.impl));
}

class ResidualAdd : public ExpImpl {
public:
  ExpImpl *a, *b;
  ResidualMethod method;
  ResidualAdd(Context &context, ExpImpl *a, ExpImpl *b,
              ResidualMethod method) :
    ExpImpl(context), a(a), b(b), method(method) {}
  std::vector<const ExpImpl *> deps() const override { return {a, b}; }
};

Exp residualAdd(Exp a, Exp b, ResidualMethod method) {
  auto &context = a.impl->context;
  return context.impl->add(new ResidualAdd(context, a.impl, b.impl, method));
}

class Loss : public ExpImpl {
public:
  LossType type;
  ExpImpl *in, *out;
  Loss(Context &context, LossType type, ExpImpl *in, ExpImpl *out) :
    ExpImpl(context), type(type), in(in), out(out) {}
  std::vector<const ExpImpl *> deps() const override { return {out, in}; }
};

Exp softMaxCrossEntropyLoss(Exp in, Exp out) {
  auto &context = in.impl->context;
  return context.impl->add(new Loss(context, SOFTMAX_CROSS_ENTROPY_LOSS,
                                    in.impl, out.impl));
}

Exp sumSquaredLoss(Exp in, Exp out) {
  auto &context = in.impl->context;
  return context.impl->add(new Loss(context, SUM_SQUARED_LOSS, in.impl,
                                    out.impl));
}

static std::string getDTypeString(DType dType) {
  switch (dType) {
  case FP32:
    return "float";
  case FP16:
    return "half";
  default:
    throw popnn::popnn_error("dType must be FP16 or FP32");
  }
}

// Wrapper functions to map tensors and then execute convolution
static Program
convolution(Graph &graph,
     const conv::Plan &plan,
     unsigned strideY, unsigned strideX, unsigned paddingY, unsigned paddingX,
     Tensor in, Tensor weights, Tensor biases, Tensor activations,
     const std::string &partialsType, bool isFractional,
     const std::string &debugPrefix) {
  const auto batchSize = activations.dim(0);
  mapActivations(graph, in);
  conv::mapWeights(weights, graph, plan, batchSize);
  conv::mapBiases(biases, graph, activations);
  mapActivations(graph, activations);
  return conv::convolution(graph, plan, strideY, strideX, paddingY, paddingX,
                           in, weights, biases, activations, partialsType,
                           isFractional, debugPrefix);
}

static Program
createBwdWeightsAndBiases(Graph &graph, const conv::Plan &bwdPlan,
                          const conv::Plan &fwdPlan,
                          Tensor weights, Tensor deltasOut,
                          Tensor bwdWeights,
                          Tensor bwdBiases,
                          const std::string &debugPrefix) {
  const auto batchSize = deltasOut.dim(0);
  const auto outNumChans = deltasOut.dim(1) * deltasOut.dim(4);
  const auto dType = graph.getTensorElementType(weights);
  auto prog = Sequence();
  conv::mapWeights(weights, graph, fwdPlan, batchSize);
  conv::mapWeights(bwdWeights, graph, bwdPlan, batchSize);
  prog.add(conv::weightsTransposeChansFlipXY(graph, weights, bwdWeights,
                                             debugPrefix));
  auto zeros = graph.addConstantTensor(dType, {outNumChans}, 0);
  conv::mapBiases(bwdBiases, graph, deltasOut);
  prog.add(Copy(zeros, bwdBiases));
  return prog;
}

static Program
convolutionWeightUpdate(poplar::Graph &graph,
                        const conv::Plan &wuPlan, const conv::Plan &fwdPlan,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        unsigned strideY, unsigned strideX, unsigned paddingY,
                        unsigned paddingX, float learningRate,
                        const std::string &debugPrefix) {
  const auto batchSize = zDeltas.dim(0);
  mapActivations(graph, zDeltas);
  conv::mapWeights(weights, graph, fwdPlan, batchSize);
  conv::mapBiases(biases, graph, zDeltas);
  mapActivations(graph, activations);
  return conv::convolutionWeightUpdate(graph, wuPlan, fwdPlan, zDeltas,
                                       weights, biases, activations,
                                       strideY, strideX, paddingY, paddingX,
                                       learningRate, debugPrefix);
}

Optimizer::~Optimizer() = default;

void Optimizer::createSchedule(const Exp &exp) {
  // First find all expression values that need to be calculated
  std::unordered_set<const ExpImpl *> exps;
  std::unordered_set<const ExpImpl *> wl;
  wl.insert(exp.impl);
  while (!wl.empty()) {
    auto e = *wl.begin();
    wl.erase(e);
    auto res = exps.insert(e);
    if (res.second) {
      for (const auto &d : e->deps()) {
        wl.insert(d);
        uses[d].insert(e);
      }
    }
  }
  // Now schedule them in a topological order
  std::queue<const ExpImpl *> toSchedule;
  std::unordered_set<const ExpImpl *> seen;
  for (const auto &exp : exps) {
    if (exp->deps().empty()) {
      toSchedule.push(exp);
      seen.insert(exp);
    }
  }
  while (!toSchedule.empty()) {
    auto e = toSchedule.front();
    toSchedule.pop();
    schedule.push_back(e);
    for (const auto next : uses[e]) {
      if (seen.count(next))
        continue;
      bool ready = true;
      for (const auto dep : next->deps()) {
        if (!seen.count(dep)) {
          ready = false;
          break;
        }
      }
      if (!ready)
        continue;
      toSchedule.push(next);
      seen.insert(next);
    }
  }
}

conv::Plan
Optimizer::getBwdConvPlan(const ExpImpl *exp, unsigned prevDimY,
                          unsigned prevDimX, unsigned prevNumChans) {
  auto it = bwdConvPlans.find(exp);
  if (it != bwdConvPlans.end())
    return it->second;
  const auto *c = dynamic_cast<const Conv2d *>(exp);
  assert(c);
  conv::Plan plan;
  unsigned inDimY, inDimX;
  std::tie(inDimY, inDimX) =
      conv::getOutputDim(prevDimY, prevDimX, c->kernelSizeY, c->kernelSizeX,
                           c->strideY, c->strideX, c->paddingY, c->paddingX);
  auto paddingX = c->paddingX, paddingY = c->paddingY;
  bool isFractional = c->strideX != 1 || c->strideY != 1;
  assert(paddingX < c->kernelSizeX);
  assert(paddingY < c->kernelSizeY);
  if (!isFractional) {
    paddingX = c->kernelSizeX - 1 - paddingX;
    paddingY = c->kernelSizeY - 1 - paddingY;
  }
  plan = planner.createPlan(inDimY, inDimX, c->numChannels,
                            c->kernelSizeY, c->kernelSizeX,
                            c->strideY, c->strideX,
                            paddingY, paddingX,
                            prevNumChans, options.batchSize, dType,
                            partialsType, isFractional, *graph,
                            options.convPlanControl);
  bwdConvPlans.emplace(exp, plan);
  return plan;
}

conv::Plan
Optimizer::getFwdConvPlan(const ExpImpl *exp, unsigned inDimY, unsigned inDimX,
                          unsigned inNumChans) {
  auto it = fwdConvPlans.find(exp);
  if (it != fwdConvPlans.end())
    return it->second;
  const auto *c = dynamic_cast<const Conv2d *>(exp);
  assert(c);
  conv::Plan plan =
      planner.createPlan(inDimY, inDimX, inNumChans,
                         c->kernelSizeY, c->kernelSizeX,
                         c->strideY, c->strideX, c->paddingY,
                         c->paddingX,
                         c->numChannels, options.batchSize, dType,
                         partialsType, false, *graph,
                         options.convPlanControl);

  fwdConvPlans.emplace(exp, plan);
  return plan;
}

unsigned
Optimizer::getRequiredChansPerGroupBwd(const ExpImpl *exp) {
  if (exp->deps().empty())
    return 0;
  const auto prev = exp->deps().front();
  if (dynamic_cast<const Feed *>(prev)) {
    return 0;
  } else if (dynamic_cast<const FullyConnected *>(prev)) {
    return 0;
  } else if (dynamic_cast<const Conv2d *>(prev)) {
    const auto prevprev = prev->deps().front();
    if (dynamic_cast<const Feed *>(prevprev)) {
      // There is no need to calculate the gradient of the activations for the
      // first layer. TODO pick a sensible channel grouping in this case.
      return out[prev].dim(4);
    }
    const auto prevOut = out[prevprev];
    auto prevDimY = prevOut.dim(2);
    auto prevDimX = prevOut.dim(3);
    auto prevNumChans = prevOut.dim(1) * prevOut.dim(4);
    auto plan = getBwdConvPlan(prev, prevDimY, prevDimX, prevNumChans);
    return plan.inChansPerGroup;
  } else if (dynamic_cast<const MaxPool *>(prev) ||
             dynamic_cast<const ResidualAdd *>(prev) ||
             dynamic_cast<const NonLinearity *>(prev)) {
    return getRequiredChansPerGroupBwd(prev);
  } else {
    throw popnn::popnn_error("Unrecognized layer type");
  }
}

unsigned
Optimizer:: getRequiredChansPerGroupFwd(const ExpImpl *exp, unsigned inDimY,
                                       unsigned inDimX, unsigned inNumChans) {
  if (uses[exp].empty())
    return 0;
  // If the current expression is used in multiple places, just choose one
  // of them.
  const ExpImpl *next = nullptr;
  for (const auto use : uses[exp]) {
    if (dynamic_cast<const Conv2d *>(use)) {
      next = use;
    } else if (!next || dynamic_cast<const Loss *>(next)) {
      next = use;
    }
  }
  if (dynamic_cast<const FullyConnected *>(next) ||
      dynamic_cast<const Loss *>(next)) {
    // A fully connected layer wants the channel grouping to be
    // the same forwards and backwards.
    if (options.training)
      return getRequiredChansPerGroupBwd(next);
    else
      return 0;
  } else if (dynamic_cast<const Conv2d *>(next)) {
    auto plan = getFwdConvPlan(next, inDimY, inDimX, inNumChans);
    return plan.inChansPerGroup;
  } else if (dynamic_cast<const ResidualAdd *>(next) ||
             dynamic_cast<const NonLinearity *>(next)) {
    // Use grouping of the use following the add
    return getRequiredChansPerGroupFwd(next, inDimY, inDimX, inNumChans);
  } else if (const auto *m = dynamic_cast<const MaxPool *>(next)) {
    unsigned outDimY, outDimX;
    std::tie(outDimY, outDimX) = maxpool::getOutputDim(inDimY, inDimX,
                                                       m->kernelSizeY,
                                                       m->kernelSizeX,
                                                       m->strideY,
                                                       m->strideX,
                                                       m->paddingY,
                                                       m->paddingX);
    return getRequiredChansPerGroupFwd(next, outDimY, outDimX, inNumChans);
  } else {
    throw popnn::popnn_error("Unrecognized expression type");
  }
}

void
Optimizer::outputConvDescription(const ExpImpl *exp,
                                 unsigned inDimY, unsigned inDimX,
                                 unsigned inNumChans,
                                 unsigned kernelSizeY, unsigned kernelSizeX,
                                 unsigned strideY, unsigned strideX,
                                 unsigned paddingY, unsigned paddingX,
                                 unsigned outNumChans) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = conv::getOutputDim(inDimY,
                                                  inDimX,
                                                  kernelSizeY,
                                                  kernelSizeX,
                                                  strideY,
                                                  strideX,
                                                  paddingY,
                                                  paddingX);
  const auto numParams =
      kernelSizeY * kernelSizeX * inNumChans * outNumChans + outNumChans;
  const auto batchSize = options.batchSize;
  auto fwdFlops = conv::getFwdFlops(batchSize, inDimY, inDimX, inNumChans,
                                    kernelSizeY, kernelSizeX, strideY,
                                    strideX, paddingY, paddingX, outNumChans);
  auto bwdFlops = conv::getBwdFlops(batchSize, inDimY, inDimX, inNumChans,
                                    kernelSizeY, kernelSizeX, strideY,
                                    strideX, paddingY, paddingX, outNumChans);
  auto wuFlops = conv::getWuFlops(batchSize, inDimY, inDimX, inNumChans,
                                  kernelSizeY, kernelSizeX, strideY,
                                  strideX, paddingY, paddingX, outNumChans);
  std::cout << "   -- Convolutional layer:\n"
            << "        Size: " << kernelSizeX << "x" << kernelSizeY << "\n"
            << "        Stride: " << strideX << "x" << strideY << "\n"
            << "        Padding: " << paddingX << "x" << paddingY << "\n"
            << "        Input:  " << inDimY << "x" << inDimX
            <<   "x" << inNumChans << "\n"
            << "        Output: " << outDimY << "x" << outDimX
            <<   "x" << outNumChans << "\n"
            << "        Params: " << numParams << "\n"
            << "        Forward FLOPs:  " << fwdFlops << "\n";
  if (options.training) {
    std::cout << "        Backward FLOPs:  " << bwdFlops << "\n"
              << "        Weight Update FLOPs:  " << wuFlops << "\n";
  }
  if (options.showPlanInfo) {
    std::cout << fwdConvPlans[exp];
  }
}

void Optimizer::outputDescription(const ExpImpl *exp) {
  if (dynamic_cast<const Feed *>(exp)) {
    std::cout << "   -- Input layer\n";
  } else if (dynamic_cast<const Loss *>(exp)) {
    std::cout << "   -- Loss layer\n";
  } else if (const auto *fc = dynamic_cast<const FullyConnected *>(exp)) {
    const auto prev = exp->deps().front();
    const auto in = out[prev];
    const auto prevSize = in.numElements();
    const auto size = fc->channels;
    const auto fwdFlops = fc::getFwdFlops(options.batchSize, prevSize, size);
    const auto bwdFlops = fc::getBwdFlops(options.batchSize, prevSize, size);
    const auto wuFlops = fc::getWuFlops(options.batchSize, prevSize, size);
    std::cout << "   -- Fully connected layer:\n"
              << "        Input:  "  << prevSize << "\n"
              << "        Output: " << size << "\n"
              << "        Params: " << size * (prevSize + 1) << "\n"
              << "        Forward FLOPs:  " << fwdFlops << "\n";
    if (options.training) {
      std::cout << "        Backward FLOPs:  " << bwdFlops << "\n"
                << "        Weight Update FLOPs:  " << wuFlops << "\n";
    }
  } else if (const auto *c = dynamic_cast<const Conv2d *>(exp)) {
    const auto prev = exp->deps().front();
    const auto in = out[prev];
    outputConvDescription(exp, in.dim(2), in.dim(3), in.dim(1) * in.dim(4),
                          c->kernelSizeY, c->kernelSizeX, c->strideY,
                          c->strideX, c->paddingY, c->paddingX,
                          c->numChannels);
  } else if (dynamic_cast<const ResidualAdd *>(exp)) {
    std::cout << "   -- Residual layer:\n";
    const auto a = out[exp->deps()[0]];
    const auto b = out[exp->deps()[1]];
    std::cout << "        Input 1:  "
              << a.dim(2) << "x" << a.dim(3) <<"x" << a.dim(1) * a.dim(4)
              << "\n";
    std::cout << "        Input 2:  "
              << b.dim(2) << "x" << b.dim(3) <<"x" << b.dim(1) * b.dim(4)
              << "\n";
    const auto c = out[exp->deps()[0]];
    std::cout << "        Output:  "
              << c.dim(2) << "x" << c.dim(3) <<"x" << c.dim(1) * c.dim(4)
              << "\n";
  } else if (const auto *nl = dynamic_cast<const NonLinearity *>(exp)) {
    switch (nl->type) {
    case NON_LINEARITY_RELU:
      std::cout << "   -- ReLU layer\n";
      break;
    case NON_LINEARITY_SIGMOID:
      std::cout << "   -- Sigmoid layer\n";
      break;
    }
  } else if (const auto *m = dynamic_cast<const MaxPool *>(exp)) {
    const auto prev = exp->deps().front();
    const auto in = out[prev];
    unsigned outDimY, outDimX;
    std::tie(outDimY, outDimX) = maxpool::getOutputDim(in.dim(2),
                                                       in.dim(3),
                                                       m->kernelSizeY,
                                                       m->kernelSizeX,
                                                       m->strideY,
                                                       m->strideX,
                                                       m->paddingY,
                                                       m->paddingX);
    const auto numChannels = in.dim(1) * in.dim(4);
    const auto fwdFlops = maxpool::getFwdFlops(options.batchSize,
                                               in.dim(2),
                                               in.dim(3),
                                               numChannels,
                                               m->kernelSizeY,
                                               m->kernelSizeX,
                                               m->strideY,
                                               m->strideX,
                                               m->paddingY,
                                               m->paddingX);
    const auto bwdFlops = maxpool::getBwdFlops(options.batchSize,
                                               in.dim(2),
                                               in.dim(3),
                                               numChannels,
                                               m->kernelSizeY,
                                               m->kernelSizeX,
                                               m->strideY,
                                               m->strideX,
                                               m->paddingY,
                                               m->paddingX);
    std::cout << "   -- Max pooling layer:\n"
              << "        Size: " << m->kernelSizeX << "x"
              << m->kernelSizeY << "\n"
              << "        Stride: " << m->strideX << "x" << m->strideY << "\n"
              << "        Padding: " << m->paddingX << "x" << m->paddingY
              <<   "\n"
              << "        Input:  " << in.dim(2) << "x" << in.dim(3)
              <<   "x" << numChannels << "\n"
              << "        Output: " << outDimY << "x" << outDimX
              <<   "x" << numChannels << "\n"
              << "        Forward FLOPs:  " << fwdFlops << "\n";
    if (options.training) {
      std::cout << "        Backward FLOPs:  " << bwdFlops << "\n";
    }
  } else {
    assert(0 && "Unrecognized layer type");
  }
}

// Define structures containing tensor ops to pass between functions/methods.
struct Optimizer::ConvOp {
  POPNN_TENSOR_OP_TYPE(convolution, 1, 12) op;
  ConvOp(POPNN_TENSOR_OP_TYPE(convolution, 1, 12) op) :
    op(std::move(op)) {}
  template<typename ...Args>
  Program operator()(Args&&... args) {
    return op(std::forward<Args>(args)...);
  };
};
struct Optimizer::ConvBwdWeightsOp {
  POPNN_TENSOR_OP_TYPE(createBwdWeightsAndBiases, 1, 2, 7) op;
  ConvBwdWeightsOp(
    POPNN_TENSOR_OP_TYPE(createBwdWeightsAndBiases, 1, 2, 7) op
  ) :  op(std::move(op)) {}
  template<typename ...Args>
  Program operator()(Args&&... args) {
    return op(std::forward<Args>(args)...);
  };
};
struct Optimizer::ConvWuOp {
  POPNN_TENSOR_OP_TYPE(convolutionWeightUpdate, 1, 2, 12) op;
  ConvWuOp(POPNN_TENSOR_OP_TYPE(convolutionWeightUpdate, 1, 2, 12) op) :
    op(std::move(op)) {}
  template<typename ...Args>
  Program operator()(Args&&... args) {
    return op(std::forward<Args>(args)...);
  };
};

static std::unique_ptr<float[]>
createRandomWeightInitializers(Tensor t, float mean, float stdDev,
                               std::mt19937 &randomEngine) {
  const auto numWeights = t.numElements();
  auto inits = std::unique_ptr<float[]>(new float[numWeights]);

  std::normal_distribution<> dist(mean, stdDev);
  for (unsigned i = 0; i < numWeights; ++i)
    inits[i] = dist(randomEngine);

  return inits;
}

Program
Optimizer::createConvLayerFwd(const ExpImpl *exp,
                              unsigned kernelSizeY, unsigned kernelSizeX,
                              unsigned strideY, unsigned strideX,
                              unsigned paddingY, unsigned paddingX,
                              unsigned numChannels,
                              Sequence &initParamsProg,
                              ConvOp &doConv,
                              const std::string &debugPrefix) {
  auto prev = exp->deps().front();
  auto &in = out[prev];
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) =
      conv::getOutputDim(in.dim(2), in.dim(3), kernelSizeY, kernelSizeX,
                         strideY, strideX, paddingY, paddingX);
  auto outChansPerGroup = getRequiredChansPerGroupFwd(exp,
                                                      outDimY, outDimX,
                                                      numChannels);
  if (outChansPerGroup == 0) {
    // The next layer has no preference on channel grouping. Set the
    // output channel group size to match the channel grouping of the partial
    // sums. This is likely to be more efficient as it avoids regrouping of data
    // after the reduction of partial sums.
    const auto &deviceInfo = graph->getDevice().getDeviceInfo();
    if (dType == "float") {
      outChansPerGroup = deviceInfo.fp32InFp32OutConvUnitsPerTile;
    } else if (partialsType == "float") {
      outChansPerGroup = deviceInfo.fp16InFp32OutConvUnitsPerTile;
    } else {
      outChansPerGroup = deviceInfo.fp16InFp16OutConvUnitsPerTile;
    }
  }
  assert(numChannels % outChansPerGroup == 0);
  const auto outNumChanGroups = numChannels / outChansPerGroup;
  const auto batchSize = options.batchSize;
  out[exp] = graph->addTensor(dType,
                              {batchSize,
                               outNumChanGroups,
                               outDimY, outDimX,
                               outChansPerGroup},
                              "convOut");
  mapActivations(*graph, out[exp]);

  unsigned inNumChanGroups = in.dim(1);
  unsigned inNumChans = inNumChanGroups * in.dim(4);
  unsigned inDimY = in.dim(2), inDimX = in.dim(3);
  auto plan = getFwdConvPlan(exp, inDimY, inDimX, inNumChans);
  Tensor weights = conv::createWeights(*graph, dType, inNumChans,
                                       kernelSizeY, kernelSizeX,
                                       numChannels, plan);
  Tensor biases = conv::createBiases(*graph, dType, numChannels);
  params[exp].push_back(weights);
  params[exp].push_back(biases);

  conv::mapWeights(weights, *graph, plan, batchSize);
  conv::mapBiases(biases, *graph, out[exp]);
  if (dType == "float") {
    auto hWeights =
        createRandomWeightInitializers(weights, 0, 1.0 / kernelSizeY,
                                       randomEngine);
    auto hBiases =
        createRandomWeightInitializers(weights, 0, 1.0 / kernelSizeY,
                                       randomEngine);
    initParamsProg.add(Copy(hWeights.get(), weights));
    initParamsProg.add(Copy(hBiases.get(), biases));
    hParams[exp].push_back(std::move(hWeights));
    hParams[exp].push_back(std::move(hBiases));
  }

  numParams += weights.numElements() + biases.numElements();

  if (options.skipFwd)
    return Sequence();

  fwdFlops += conv::getFwdFlops(batchSize,
                                inDimY, inDimX, inNumChans, kernelSizeY,
                                kernelSizeX, strideY,
                                strideX, paddingY, paddingX, numChannels);
  fwdPerfectCycleTime +=
      conv::getFwdPerfectCycleCount(*graph, dType, batchSize, inDimY, inDimX,
                                    inNumChans, kernelSizeY, kernelSizeX,
                                    strideY, strideX, paddingY, paddingX,
                                    numChannels);

  return doConv(*graph, plan, strideY, strideX, paddingY, paddingX, in, weights,
                biases, out[exp], partialsType, false, debugPrefix);
}

Program
Optimizer::createResidualLayerFwd(const ExpImpl *exp, unsigned layerIndex,
                                  const std::string &debugPrefix) {
  const auto residual = dynamic_cast<const ResidualAdd *>(exp);
  assert(residual);
  unsigned numChannels, outDimY, outDimX;
  //The output will be the same batch/y/x dimensions as the first input with
  //channel grouping chosen to match the following layer
  const auto prev0 = exp->deps()[0];
  const auto prev1 = exp->deps()[1];
  const auto &in0Shape = out[prev0].shape();
  numChannels = in0Shape[1] * in0Shape[4];
  outDimY = in0Shape[2];
  outDimX = in0Shape[3];
  auto outChansPerGroup = getRequiredChansPerGroupFwd(exp, outDimY, outDimX,
                                                      numChannels);
  if (outChansPerGroup == 0) {
    outChansPerGroup = in0Shape[4];
  }

  out[exp] = graph->addTensor(dType,
                              {in0Shape[0], numChannels / outChansPerGroup,
                               outDimY, outDimX, outChansPerGroup},
                              "activations." + std::to_string(layerIndex));
  mapActivations(*graph, out[exp]);

  if (options.skipFwd)
    return Sequence();

  Tensor in0 =
    residual::arrangeResidualInput(*graph,
                                   out[prev0],
                                   out[exp].shape(), dType,
                                   residual->method);
  Tensor in1 =
    residual::arrangeResidualInput(*graph,
                                   out[prev1],
                                   out[exp].shape(), dType,
                                   residual->method);
  switch (residual->method) {
  case RESIDUAL_PAD:
    {
      Program fwdProg =
        residual::joinResidual(*graph,
                               in0,
                               in1,
                               out[exp],
                               debugPrefix);
      auto outShape = out[exp].shape();
      fwdFlops += outShape[0] *
        residual::getNumberOfAdds(outShape[2], outShape[3],
                                  outShape[1] * outShape[4]);
      fwdPerfectCycleTime +=
        residual::getPerfectCycleCount(*graph, dType,
                                       outShape[0], outShape[2], outShape[3],
                                       outShape[1] * outShape[4]);
      return fwdProg;
    }
  default:
    throw popnn::popnn_error("This residual type not supported yet");
  }
  POPNN_UNREACHABLE();
}

void Optimizer::genFwd(Sequence &fwdProg,
                       Sequence &initParamsProg,
                       struct ConvOp &convOp) {
  for (unsigned layerIndex = 0; layerIndex < schedule.size(); ++layerIndex) {
    const auto &exp = schedule[layerIndex];
    std::cout << "-- Layer " << layerIndex << "\n";
    const std::string layerPrefix =
        "Layer" + std::to_string(layerIndex) + "/Fwd";
    outputDescription(exp);
    if (const auto *feed = dynamic_cast<const Feed *>(exp)) {
      dataSet = &feed->dataset;
      auto chansPerGroup =
          getRequiredChansPerGroupFwd(exp, dataSet->dim[0], dataSet->dim[1],
                                      dataSet->dim[2]);
      if (chansPerGroup == 0)
        chansPerGroup = dataSet->dim[2];
      assert(dataSet->dim[2] % chansPerGroup == 0);
      const auto numChanGroups = dataSet->dim[2] / chansPerGroup;
      const auto dim = std::vector<size_t>({options.batchSize,
                                            numChanGroups,
                                            dataSet->dim[0], dataSet->dim[1],
                                            chansPerGroup});
      out[exp] = graph->addTensor(dType, dim, "input");
      feedIn = out[exp];
      mapActivations(*graph, out[exp]);
    } else if (const auto *c = dynamic_cast<const Conv2d *>(exp)) {
      fwdProg.add(createConvLayerFwd(exp, c->kernelSizeY, c->kernelSizeX,
                                     c->strideY, c->strideX, c->paddingY,
                                     c->paddingX,
                                     c->numChannels,
                                     initParamsProg, convOp,
                                     layerPrefix));
    } else if (const auto *nl = dynamic_cast<const NonLinearity *>(exp)) {
      auto prev = exp->deps().front();
      auto in = out[prev];
      out[exp] = graph->addTensor(dType, in.shape(), "out");
      mapActivations(*graph, out[exp]);
      fwdProg.add(Copy(in, out[exp]));
      nonLinearity(*graph, nl->type, out[exp], fwdProg, layerPrefix);
    } else if (const auto *m = dynamic_cast<const MaxPool *>(exp)) {
      const auto &in = out[exp->deps().front()];
      const auto batchSize = options.batchSize;
      if (!options.skipFwd) {
        out[exp] = maxpool::maxPool(*graph,
                                    m->kernelSizeY, m->kernelSizeX,
                                    m->strideY, m->strideX,
                                    m->paddingY, m->paddingX,
                                    in, fwdProg, layerPrefix);
        fwdFlops += maxpool::getFwdFlops(batchSize,
                                       in.dim(2), in.dim(3),
                                       in.dim(1) * in.dim(4),
                                       m->kernelSizeY, m->kernelSizeX,
                                       m->strideY, m->strideX,
                                       m->paddingY, m->paddingX);
        fwdPerfectCycleTime +=
            maxpool::getFwdPerfectCycleCount(*graph, dType, batchSize,
                                             in.dim(2), in.dim(3),
                                             in.dim(1) * in.dim(4),
                                             m->kernelSizeY, m->kernelSizeX,
                                             m->strideY, m->strideX,
                                             m->paddingY, m->paddingX);
      } else {
        // If the forward pass is skipped, an output tensor still needs
        // to be created.
        unsigned outDimY, outDimX;
        std::tie(outDimY, outDimX) = maxpool::getOutputDim(in.dim(2),
                                                           in.dim(3),
                                                           m->kernelSizeY,
                                                           m->kernelSizeX,
                                                           m->strideY,
                                                           m->strideX,
                                                           m->paddingY,
                                                           m->paddingX);
        out[exp] = graph->addTensor(dType,
                                    {batchSize, in.dim(1), outDimY, outDimX,
                                     in.dim(4)},
                                    "maxPoolOut");
        mapActivations(*graph, out[exp]);
      }
    } else if (const auto *fc = dynamic_cast<const FullyConnected *>(exp)) {
      const auto prev = exp->deps().front();
      const auto in = out[prev];
      const auto prevSize = in[0].numElements();
      const auto size = fc->channels;
      const auto batchSize = options.batchSize;
      out[exp] = graph->addTensor(dType, {batchSize, size},
                                 "activations." + std::to_string(layerIndex));
      mapActivations(*graph, out[exp]);
      auto activationsMapping =
          computeActivationsMapping(*graph, out[exp][0], 0, batchSize);
      bool first = dynamic_cast<const Feed *>(prev);
      bool forwardOnly = first || !options.training;
      const auto &plan =
          fullyConnectedPlan.emplace(
            exp,
            fc::createPlan(*graph, dType, partialsType, prevSize,
                           activationsMapping,
                           forwardOnly)
         ).first->second;
      Tensor weights, biases;
      std::tie(weights, biases) = fc::createParams(*graph, dType,
                                                   prevSize, size);
      params[exp] = {weights, biases};
      if (dType == "float") {
         auto hWeights =
             createRandomWeightInitializers(weights, 0, 1.0 / prevSize,
                                            randomEngine);
         auto hBiases =
             createRandomWeightInitializers(weights, 0, 1.0 / prevSize,
                                            randomEngine);
         initParamsProg.add(Copy(hWeights.get(), weights));
         initParamsProg.add(Copy(hBiases.get(), biases));
         hParams[exp].push_back(std::move(hWeights));
         hParams[exp].push_back(std::move(hBiases));
      }
      fc::mapWeights(*graph, weights, activationsMapping, plan);
      fc::mapBiases(*graph, biases, activationsMapping);
      if (!options.skipFwd) {
        fwdProg.add(fc::fullyConnected(*graph, size,
                                       in, weights, biases,
                                       out[exp], plan,
                                       layerPrefix));
        fwdFlops += fc::getFwdFlops(batchSize, prevSize, size);
        fwdPerfectCycleTime +=
            fc::getFwdPerfectCycleCount(*graph, batchSize, prevSize,
                                        size, dType);
      }
      numParams += weights.numElements() + biases.numElements();
    } else if (dynamic_cast<const ResidualAdd *>(exp)) {
      fwdProg.add(createResidualLayerFwd(exp, layerIndex, layerPrefix));
    } else if (const auto *loss = dynamic_cast<const Loss *>(exp)) {
      const auto prev = exp->deps()[0];
      auto in = out[prev];
      const auto batchSize = options.batchSize;
      expected = graph->addTensor("unsigned", {batchSize}, "expected");
      graph->setTileMapping(expected, 0);
      Tensor numCorrect = graph->addTensor("unsigned", {1}, "numCorrect");
      graph->setTileMapping(numCorrect, 0);
      out[exp] = graph->addTensor(dType, {batchSize}, "loss");
      graph->setTileMapping(out[exp], 0);
      inGradient[exp].emplace_back();
      auto &inGrad = inGradient[exp].back();
      inGrad = graph->addTensor(dType, in.shape(), "deltas");
      mapActivations(*graph, inGrad);
      auto inFlat =
          in.reshape({batchSize, in.numElements() / batchSize});
      auto inGradFlat =
          inGrad.reshape({batchSize, in.numElements() / batchSize});
      auto calcLossProg = calcLoss(*graph,
                                   inFlat,
                                   expected,
                                   out[exp],
                                   inGradFlat,
                                   numCorrect,
                                   dType, "unsigned int",
                                   loss->type);
      fwdProg.add(Sequence(Copy(&hNumCorrect, numCorrect),
                           calcLossProg,
                           Copy(numCorrect, &hNumCorrect)));
    } else {
      throw popnn::popnn_error("Unknown expresssion type");
    }
  }
}

conv::Plan Optimizer::
getWuConvPlan(const ExpImpl *exp, unsigned prevDimY, unsigned prevDimX,
              unsigned prevNumChans, unsigned actsChansPerGroup,
              unsigned deltasChanPerGroup, unsigned weightOutChansPerGroup) {
  auto it = wuConvPlans.find(exp);
  if (it != wuConvPlans.end())
    return it->second;
  const auto batchSize = options.batchSize;
  const auto *c = dynamic_cast<const Conv2d *>(exp);
  assert(c);
  conv::Plan plan =
      planner.createWeightUpdatePlan(prevDimY, prevDimX, prevNumChans,
                                     actsChansPerGroup, deltasChanPerGroup,
                                     weightOutChansPerGroup, c->kernelSizeY,
                                     c->kernelSizeX, c->strideY, c->strideX,
                                     c->paddingY, c->paddingX, c->numChannels,
                                     batchSize, dType, partialsType, false,
                                     *graph, options.convPlanControl);
  wuConvPlans.emplace(exp, plan);
  return plan;
}

Program Optimizer::
createConvLayerBwd(const ExpImpl *exp, Tensor outGradient, unsigned layerIndex,
                   unsigned kernelSizeY, unsigned kernelSizeX,
                   unsigned strideY, unsigned strideX,
                   unsigned paddingY, unsigned paddingX,
                   bool backwardPassRequired,
                   ConvBwdWeightsOp &convBwdWeights,
                   ConvOp &doConv, ConvWuOp &convWU,
                   const std::string &debugPrefix) {
  const auto prev = exp->deps()[0];
  const auto in = out[prev];
  auto prog = Sequence();
  auto prevDimY = in.dim(2);
  auto prevDimX = in.dim(3);
  auto prevNumChans = in.dim(1) * in.dim(4);
  auto nextNumChans = out[exp].dim(1) * out[exp].dim(4);
  auto fwdPlan = getFwdConvPlan(exp, prevDimY, prevDimX, prevNumChans);
  auto bwdPlan = getBwdConvPlan(exp, prevDimY, prevDimX, prevNumChans);
  auto weights = params[exp][0];
  auto biases = params[exp][1];
  const auto batchSize = options.batchSize;
  if (backwardPassRequired) {
    bool isFractional = strideX != 1 || strideY != 1;
    assert(paddingX < kernelSizeX);
    assert(paddingY < kernelSizeY);

    auto bwdPaddingX = paddingX;
    auto bwdPaddingY = paddingY;
    if (!isFractional) {
      bwdPaddingX = kernelSizeX - 1 - paddingX;
      bwdPaddingY = kernelSizeY - 1 - paddingY;
    }
    createInGradients(exp, layerIndex);
    auto &inGrad = inGradient[exp][0];
    // Create transpose/flipped weights
    auto bwdWeights =
        conv::createWeights(*graph, dType, nextNumChans, kernelSizeY,
                            kernelSizeX, prevNumChans, bwdPlan);
    conv::mapWeights(bwdWeights, *graph, bwdPlan, batchSize);
    auto biases = graph->addTensor(dType, {prevNumChans}, "zeroBiases");
    conv::mapBiases(biases, *graph, inGrad);

    if (!options.skipBwd) {
      bwdFlops += conv::getBwdFlops(batchSize,
                                    prevDimY, prevDimX, prevNumChans,
                                    kernelSizeY, kernelSizeX, strideY,
                                    strideX, paddingY, paddingX, nextNumChans);
      bwdPerfectCycleTime +=
          conv::getBwdPerfectCycleCount(*graph, dType, batchSize, prevDimY,
                                        prevDimX, prevNumChans, kernelSizeY,
                                        kernelSizeX, strideY, strideX, paddingY,
                                        paddingX, nextNumChans);

      prog.add(convBwdWeights(*graph, bwdPlan, fwdPlan, weights,
                              inGrad, bwdWeights, biases, debugPrefix));
      // Perform convolution
      prog.add(doConv(*graph, bwdPlan, strideY, strideX,
                      bwdPaddingY, bwdPaddingX, outGradient, bwdWeights,
                      biases, inGrad, bwdPlan.getPartialType(),
                      isFractional, debugPrefix));
    }
  }
  if (!options.skipWU) {
    // TODO move before backward pass to reduce live range of the deltas.
    auto wuPlan = getWuConvPlan(exp, prevDimY, prevDimX, prevNumChans,
                                in.dim(4), outGradient.dim(4), weights.dim(4));
    prog.add(convWU(*graph, wuPlan, fwdPlan, outGradient, weights, biases,
                    in, strideY, strideX, paddingY, paddingX,
                    options.learningRate, debugPrefix));
    wuFlops += conv::getWuFlops(batchSize,
                                prevDimY, prevDimX, prevNumChans, kernelSizeY,
                                kernelSizeX, strideY,
                                strideX, paddingY, paddingX, nextNumChans);
    wuPerfectCycleTime +=
        conv::getWuPerfectCycleCount(*graph, dType, batchSize, prevDimY,
                                     prevDimX, prevNumChans, kernelSizeY,
                                     kernelSizeX, strideY, strideX, paddingY,
                                     paddingX, nextNumChans);
  }
  return prog;
}

void Optimizer::createInGradients(const ExpImpl *exp, unsigned index) {
  for (const auto &prev : exp->deps()) {
    const auto in = out[prev];
    inGradient[exp].emplace_back();
    auto &inGrad = inGradient[exp].back();
    if (in.rank() == 5) {
      auto numChannels = in.dim(1) * in.dim(4);
      auto chansPerGroup = getRequiredChansPerGroupBwd(exp);
      if (chansPerGroup == 0)
        chansPerGroup = numChannels;
      assert(numChannels % chansPerGroup == 0);
      const auto numChanGroups = numChannels / chansPerGroup;

      inGrad = graph->addTensor(dType, {options.batchSize,
                                        numChanGroups,
                                        in.dim(2),
                                        in.dim(3),
                                        chansPerGroup},
                                "inGradient." + std::to_string(index));
    } else {
      assert(in.rank() == 2);
      inGrad = graph->addTensor(dType, in.shape(),
                                "inGradient." + std::to_string(index));
    }
    mapActivations(*graph, inGrad);
  }
}

void Optimizer::genBwd(Sequence &bwdProg,
                       ConvOp &convOp,
                       ConvWuOp &convWuOp,
                       ConvBwdWeightsOp &convBwdWeightsOp) {
  const auto eta = options.learningRate;
  for (int layerIndex = schedule.size() - 1; layerIndex > 0; --layerIndex) {
    const std::string layerPrefix =
        "Layer" + std::to_string(layerIndex) + "/Bwd";
    const auto &exp = schedule[layerIndex];

    if (dynamic_cast<const Loss *>(exp) ||
        dynamic_cast<const Feed *>(exp)) {
      // Loss layer gradients are computed in the forward pass and
      // feed layers do not need a backwards pass.
      continue;
    }
    bool backwardPassRequired = true;
    if (exp->deps().empty()) {
      backwardPassRequired = false;
    }
    const auto prev = exp->deps()[0];
    if (dynamic_cast<const Feed *>(prev)) {
      backwardPassRequired = false;
    }
    const auto in = out[prev];
    const auto batchSize = options.batchSize;
    Tensor outGradient;
    // Calculated the gradient of the output of the operation.
    if (uses[exp].size() == 1) {
      outGradient = inGradient[*uses[exp].begin()][0];
    } else if (uses[exp].size() == 2) {
      const ExpImpl *res = nullptr;
      const ExpImpl *other = nullptr;
      for (const auto &use : uses[exp]) {
        if (dynamic_cast<const ResidualAdd *>(use)) {
          res = use;
        } else {
          other = use;
        }
      }
      if (!other || !res) {
        throw popnn::popnn_error("Backpropagation of values consumed twice "
                                 "only implemented between a residual add"
                                 "and a non residual add");
      }
      if (exp != res->deps()[1]) {
        throw popnn::popnn_error("Backpropagation of values consumed twice "
                                 "only implemented between a residual add"
                                 "where residual is second parameter");
      }
      auto resGradient = inGradient[res][1];
      outGradient = inGradient[other][0];
      bwdProg.add(residual::joinDeltas(*graph, outGradient, resGradient,
                                       layerPrefix));
    } else {
      throw popnn::popnn_error("Backpropagation of values consumed by more "
                               "than 2 expressions not implemented");
    }

    if (const auto *fc = dynamic_cast<const FullyConnected *>(exp)) {
      auto weights = params[exp][0];
      auto biases = params[exp][1];
      const auto &plan = fullyConnectedPlan.find(exp)->second;
      const auto prevSize = in[0].numElements();
      const auto size = fc->channels;
      if (backwardPassRequired  && !options.skipBwd) {
        bwdFlops += fc::getBwdFlops(batchSize, prevSize, size);
        bwdPerfectCycleTime +=
            fc::getBwdPerfectCycleCount(*graph, batchSize, prevSize,
                                        size, dType);
        createInGradients(exp, layerIndex);
        auto &inGrad = inGradient[exp][0];
        bwdProg.add(fc::fullyConnectedBackward(*graph, outGradient, weights,
                                               inGrad, plan, layerPrefix));
      }
      if (!options.skipWU) {
        bwdProg.add(fc::fullyConnectedWeightUpdate(*graph, outGradient,
                                                   in, weights, biases, eta,
                                                   plan, layerPrefix));
        wuFlops += fc::getWuFlops(batchSize, prevSize, size);
        wuPerfectCycleTime +=
            fc::getWuPerfectCycleCount(*graph, batchSize, prevSize,
                                       size, dType);
      }
    } else if (const auto *c = dynamic_cast<const Conv2d *>(exp)) {
      bwdProg.add(createConvLayerBwd(exp, outGradient, layerIndex,
                                     c->kernelSizeY, c->kernelSizeX,
                                     c->strideY, c->strideX, c->paddingY,
                                     c->paddingX,
                                     backwardPassRequired,
                                     convBwdWeightsOp, convOp, convWuOp,
                                     layerPrefix));
    } else if (const auto *m = dynamic_cast<const MaxPool *>(exp)) {
      if (backwardPassRequired && !options.skipBwd) {
        inGradient[exp].push_back(
          maxpool::maxPoolInputGradient(*graph,
                                        m->kernelSizeY, m->kernelSizeX,
                                        m->strideY, m->strideX,
                                        m->paddingY, m->paddingX,
                                        in, out[exp], outGradient, bwdProg,
                                        layerPrefix)
        );
        bwdFlops += maxpool::getBwdFlops(batchSize,
                                         in.dim(2), in.dim(3),
                                         in.dim(1) * in.dim(4),
                                         m->kernelSizeY, m->kernelSizeX,
                                         m->strideY, m->strideX,
                                         m->paddingY, m->paddingX);
        bwdPerfectCycleTime +=
            maxpool::getBwdPerfectCycleCount(*graph, dType, batchSize,
                                             in.dim(2), in.dim(3),
                                             in.dim(1) * in.dim(4),
                                             m->kernelSizeY, m->kernelSizeX,
                                             m->strideY, m->strideX,
                                             m->paddingY, m->paddingX);
      } else {
        // Create the correct shaped tensor even if we skip the backwards
        // pass.
        createInGradients(exp, layerIndex);
      }
    } else if (dynamic_cast<const ResidualAdd *>(exp)) {
      // Set the input gradient to both inputs to be the same, even
      // though the they may have a different shape. The difference is handled
      // by the consumer of the gradient.
      inGradient[exp].push_back(outGradient);
      inGradient[exp].push_back(outGradient);
    } else if (const auto *nl = dynamic_cast<const NonLinearity *>(exp)) {
      inGradient[exp].push_back(
        nonLinearityInputGradient(*graph, nl->type, out[exp], outGradient,
                                  bwdProg, layerPrefix)
      );
    } else {
      throw popnn::popnn_error("Unrecognized layer type");
    }
  }
}

void Optimizer::reportTotals() {
  const auto numFlops = fwdFlops + bwdFlops + wuFlops;
  std::cout << "Total number of Forward FLOPs:          "
            << std::right << std::setw(12) << fwdFlops << "\n";
  std::cout << "Total number of Backward FLOPs          "
            << std::right << std::setw(12) << bwdFlops << "\n";
  std::cout << "Total number of WU FLOPs:               "
            << std::right << std::setw(12) << wuFlops << "\n";
  std::cout << "Total number of FLOPs:                  "
            << std::right << std::setw(12) << numFlops << "\n";
  std::cout << "Total number of inputs and activations: "
            << std::setw(12)
            << std::accumulate(schedule.begin(), schedule.end(), 0,
                               [&](unsigned sum, const ExpImpl *exp) {
                                 return sum + out[exp].numElements();
                               })
            << "\n";
  std::cout << "Total number of Params:                 "
            << std::setw(12) << numParams << "\n";
  const auto perfectCycleTime =
      fwdPerfectCycleTime + bwdPerfectCycleTime + wuPerfectCycleTime;
  std::cout << "Fwd Perfect cycle time:                 ";
  std::cout << std::setw(12) << static_cast<std::uint64_t>(perfectCycleTime)
            << "\n";
  std::cout << "Bwd Perfect cycle time:                 ";
  std::cout << std::setw(12) << static_cast<std::uint64_t>(perfectCycleTime)
            << "\n";
  std::cout << "WU Perfect cycle time:                  ";
  std::cout << std::setw(12) << static_cast<std::uint64_t>(perfectCycleTime)
            << "\n";
  std::cout << "Perfect cycle time:                     ";
  std::cout << std::setw(12) << static_cast<std::uint64_t>(perfectCycleTime)
            << "\n";
}

enum {
  INIT_PARAMS_PROG,
  TRAIN_PROG,
  TEST_PROG,
  NUM_PROGS
};

void Optimizer::createEngine(Program initParamsProg, Program fwdProg,
                             Program bwdProg) {
  std::cerr << "Creating engine\n";
  auto trainProg = Sequence();
  if (options.training) {
    if (!options.ignoreData) {
      size_t trainingDataSize = dataSet->numTraining * dataSet->dataSize;
      trainProg.add(Copy(&dataSet->trainingData[0],
                    &dataSet->trainingData[trainingDataSize],
                    feedIn));
      trainProg.add(Copy(&dataSet->trainingLabels[0],
                    &dataSet->trainingLabels[dataSet->numTraining],
                    expected));
    }
    trainProg.add(fwdProg);
    trainProg.add(bwdProg);
  }
  auto testProg = Sequence();
  if (!options.ignoreData) {
    size_t testDataSize = dataSet->numTest * dataSet->dataSize;
    testProg.add(Copy(&dataSet->testData[0], &dataSet->testData[testDataSize],
                 feedIn));
    testProg.add(Copy(&dataSet->testLabels[0],
                 &dataSet->testLabels[dataSet->numTest],
                 expected));
  }
  testProg.add(fwdProg);
  std::vector<Program> progs(NUM_PROGS);
  progs[INIT_PARAMS_PROG] = std::move(initParamsProg);
  progs[TRAIN_PROG] = std::move(trainProg);
  progs[TEST_PROG] = std::move(testProg);
  engine = std::unique_ptr<Engine>(new Engine(*graph, progs));
}

Optimizer::Optimizer(const Exp &exp, OptimizerOptions options) :
  options(options) {
  if (options.useIPUModel) {
    DeviceInfo info;
    info.memcpyBytesPerCycle = options.dataPathWidth / 8;
    info.numIPUs = options.numIPUs;
    info.tilesPerIPU = options.tilesPerIPU;
    info.memoryBytesPerTile = options.memoryBytesPerTile;
    info.exchangeBytesPerCycle = options.ipuExchangeBandwidth;
    info.IPUExchangeType =
        DeviceInfo::ExchangeType::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST;
    info.globalSyncCycles = 500;
    info.dataPathWidth = options.dataPathWidth;
    info.convUnitPipelineDepth = options.convUnitPipelineDepth;
    info.fp16InFp16OutConvUnitsPerTile
         = options.fp16InFp16OutConvUnitsPerTile;
    info.fp16InFp32OutConvUnitsPerTile
         = options.fp16InFp32OutConvUnitsPerTile;
    info.fp32InFp32OutConvUnitsPerTile
         = options.fp32InFp32OutConvUnitsPerTile;
    info.convUnitCoeffLoadBytesPerCycle
         = options.convUnitCoeffLoadBytesPerCycle;
    info.supportsSuperTileSendReceive
         = options.supportsSuperTileSendReceive;

    const double syncLatencyPerHop = 15e-9;
    unsigned numHops = 0;

    switch (info.numIPUs) {
    case 1:
      break;
    case 2:
      {
        /* Assume all 6 links of 128Gbps are used when only 2 IPUs
         * are configured (i.e all links are intra-card)
         */
        info.globalExchangeConstraints = {
            GlobalExchangeConstraint(6 * 128 * 1024 * 1024 * 1024LL,
              {GlobalExchangeFlow(0,1)}),
            GlobalExchangeConstraint(6 * 128 * 1024 * 1024 * 1024LL,
              {GlobalExchangeFlow(1,0)}),
             };

        /* Assume for a 2 IPU system the intra card hop delay is 1 */
        numHops = 1;
      }
      break;
    default:
      throw popnn::popnn_error("IPU modeling does not support > 2 IPUs");
    }

    info.globalSyncCycles =
        std::ceil(syncLatencyPerHop
                  * static_cast<double>(info.frequencyInHz * numHops * 2));

    graph = std::unique_ptr<Graph>(new Graph(createIPUModelDevice(info)));
  } else {
    graph = std::unique_ptr<Graph>(new Graph(createCPUDevice()));
  }
  popnn::addCodelets(*graph);
  std::cerr << "Constructing program\n";
  ConvOp convOp =
      createTensorOp<1, 12>(
        *graph, convolution, "conv",
        {{TensorOpParamType::InputTensor},
         {TensorOpParamType::InputTensor},
         {TensorOpParamType::InputTensor},
         {TensorOpParamType::OutputTensor}});
  ConvBwdWeightsOp convBwdWeightsOp =
      createTensorOp<1, 2, 7>(
         *graph, createBwdWeightsAndBiases, "createBwdWeights",
         {{TensorOpParamType::InputTensor,
           TensorOpParamType::NotParamTensor,
           TensorOpParamType::OutputTensor,
           TensorOpParamType::OutputTensor}});
  ConvWuOp convWuOp =
      createTensorOp<1, 2, 12>(
        *graph, convolutionWeightUpdate, "convWeightUpdate",
        {{TensorOpParamType::InputTensor},
         {TensorOpParamType::InOutTensor},
         {TensorOpParamType::InOutTensor},
         {TensorOpParamType::InputTensor}});
  fwdFlops = bwdFlops = wuFlops = 0;
  numParams = 0;
  fwdPerfectCycleTime = bwdPerfectCycleTime = wuPerfectCycleTime = 0;
  dType = getDTypeString(options.dataType);
  partialsType = getDTypeString(options.partialsType);
  auto initParamsProg = Sequence();
  auto fwdProg = Sequence();
  auto bwdProg = Sequence();
  createSchedule(exp);
  genFwd(fwdProg, initParamsProg, convOp);
  if (options.training)
    genBwd(bwdProg, convOp, convWuOp, convBwdWeightsOp);
  reportTotals();
  numTestBatches = dataSet->numTest / options.batchSize;
  createEngine(initParamsProg, fwdProg, bwdProg);
}

void Optimizer::run(unsigned numIterations) {
  /* All this method needs to do is set the relevant parameters and
     run the control program. */
  std::cerr << "Running program\n";
  if (options.doComputation) {
    if (options.training) {
      engine->run(INIT_PARAMS_PROG); // initialize params
      for (unsigned i = 0; i < numIterations; i++) {
        if (options.doTestsDuringTraining &&
            i % options.numBatchesBetweenTest == 0) {
          hNumCorrect = 0;
          for (unsigned j = 0; j < numTestBatches; j++) {
            engine->run(TEST_PROG);
          }
          unsigned numTests = (numTestBatches * options.batchSize);
          float percentCorrect = float(100 * hNumCorrect) / numTests;
          std::cout << "--- Accuracy after " << i << " batches = "
                    << percentCorrect << "%\n";
        }
        engine->run(TRAIN_PROG);
      }
    } else {
      engine->run(INIT_PARAMS_PROG);
      hNumCorrect = 0;
      for (unsigned i = 0; i < numIterations; i++) {
        engine->run(TEST_PROG);
      }
      unsigned numTests = (numTestBatches * options.batchSize);
      float percentCorrect = float(100 * hNumCorrect) / numTests;
      std::cout << "--- Accuracy = " << percentCorrect << "%\n";
    }
  }
  if (options.useIPUModel) {
    Engine::ReportOptions opt;
    opt.doLayerWiseProfile = true;
    engine->report(std::cout, opt);
  }
}

}} // end namespace poplar::optimizer
