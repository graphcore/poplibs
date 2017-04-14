#include "enigma/Optimizer.hpp"

#include <boost/program_options.hpp>
#include "enigma/exceptions.hpp"
#include <poplar/HalfFloat.hpp>
#include "popstd/Add.hpp"
#include "popstd/codelets.hpp"
#include "popconv/codelets.hpp"
#include "poplin/codelets.hpp"
#include "popreduce/codelets.hpp"
#include "popnn/codelets.hpp"
#include "popconv/Convolution.hpp"
#include "popnn/Loss.hpp"
#include "popnn/MaxPool.hpp"
#include "poplin/MatMul.hpp"
#include "popstd/ActivationMapping.hpp"
#include "popreduce/Reduce.hpp"
#include "popnn/Residual.hpp"
#include "popnn/NonLinearity.hpp"
#include "util/Compiler.hpp"
#include <iomanip>
#include <array>
#include <deque>
#include <queue>
#include <unordered_set>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace popstd;
using namespace popreduce;
using namespace popnn;

namespace enigma {

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
       &options.convOptions.useWinograd
     )->default_value(options.convOptions.useWinograd),
     "Use winograd for convolution layers")
    ("winograd-patch-size",
     po::value<unsigned>(
       &options.convOptions.winogradPatchSize
     )->default_value(options.convOptions.winogradPatchSize),
     "Patch size for winograd convolution")
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
       &options.convOptions.percentageCyclesExcessForMemOptim
     )->default_value(options.convOptions.percentageCyclesExcessForMemOptim),
     "Percentage cycles excess to use for memory optimisation. "
     "if 0, no memory optimisation is performed")
    ("weight-update-method",
     po::value<popconv::WeightUpdateMethod>(
         &options.convOptions.weightUpdateMethod
     )->default_value(options.convOptions.weightUpdateMethod),
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
    ("in-place-update",
    po::value<bool>(
       &options.inPlaceParamUpdate
    )->default_value(options.inPlaceParamUpdate),
    "Perform parameter update in place")
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

Exp tanh(Exp in) {
  auto &context = in.impl->context;
  return context.impl->add(new NonLinearity(context, NON_LINEARITY_TANH,
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
    throw enigma::enigma_error("dType must be FP16 or FP32");
  }
}


static TensorSig sig(const Graph &graph, const Tensor &t) {
  return {graph.getTensorElementType(t), t.shape()};
}

void Optimizer::
createBwdWeights(Graph &graph,
                 unsigned prevNumChans,
                 Tensor weights,
                 Tensor deltasIn,
                 Tensor bwdWeights,
                 unsigned strideY, unsigned strideX,
                 unsigned paddingY, unsigned paddingX,
                 bool isFractional, Sequence &prog,
                 const std::string &debugPrefix) {
  popconv::mapWeights(bwdWeights, graph, deltasIn, strideY, strideX,
                      paddingY, paddingX, isFractional, options.convOptions);

  const auto dType = graph.getTensorElementType(weights);
  auto key = std::vector<TensorSig>{sig(graph, weights),
                                    sig(graph, bwdWeights)};
  std::vector<Tensor> args = {weights, bwdWeights};
  auto it = bwdWeightGraphCache.find(key);
  if (it != bwdWeightGraphCache.end()) {
    auto &f = it->second;
    f(args, prog);
    return;
  }
  using namespace popstd::graphfn;
  auto f = VoidFunction(
    graph,
    {input(weights, "weights"), output(bwdWeights, "bwdWeights")},
    [&](std::vector<poplar::Tensor> &args, Sequence &prog) {
      popconv::weightsTransposeChansFlipXY(graph, args[0], args[1], prog,
                                           debugPrefix);
      return prog;
    });
  bwdWeightGraphCache.emplace(key, f);
  f(args, prog);
}

void Optimizer::
doConvolutionWeightUpdate(poplar::Graph &graph,
                          poplar::Tensor zDeltas, poplar::Tensor weights,
                          poplar::Tensor activations,
                          unsigned strideY, unsigned strideX, unsigned paddingY,
                          unsigned paddingX, float learningRate,
                          Sequence &prog,
                          const std::string &debugPrefix) {
  auto key = std::make_tuple(sig(graph, zDeltas), sig(graph, weights),
                             sig(graph, activations),
                             strideY, strideX, paddingY, paddingX,
                             learningRate);
  std::vector<Tensor> args = {zDeltas, weights, activations};
  auto it = wuGraphCache.find(key);
  if (it != wuGraphCache.end()) {
    auto &f = it->second;
    return f(args, prog);
  }
  using namespace popstd::graphfn;
  auto f = VoidFunction(
    graph,
    {input(zDeltas, "zDeltas"), inout(weights, "weights"),
     input(activations, "activations")},
    [&](std::vector<poplar::Tensor> &args, Sequence &prog) {
       popconv::convolutionWeightUpdate(graph, args[0],
                                        args[1], args[2],
                                        strideY, strideX, paddingY,
                                        paddingX, false,
                                        learningRate, prog, debugPrefix,
                                        options.convOptions);
    });
  wuGraphCache.emplace(key, f);
  return f(args, prog);
}

Optimizer::~Optimizer() = default;

double
Optimizer::getPerfectCycleTime(unsigned flops, const std::string &dType,
                               bool useVectors, bool useAmp) {
  const auto &deviceInfo = graph->getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  if (useAmp) {
    const auto convUnitsPerTile =
        std::max(std::max(deviceInfo.fp16InFp16OutConvUnitsPerTile,
                          deviceInfo.fp32InFp32OutConvUnitsPerTile),
                 deviceInfo.fp16InFp32OutConvUnitsPerTile);
    const auto halfVectorWidth = deviceInfo.getHalfVectorWidth();
    auto macsPerCycle = convUnitsPerTile * halfVectorWidth;
    double numMacs = static_cast<double>(flops) / 2;
    auto macCycles = static_cast<double>(numMacs) / (macsPerCycle * numTiles);
    return macCycles;
  } else if (useVectors) {
    unsigned dTypeSize = dType == "float" ? 4 : 2;
    const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
    return static_cast<double>(flops) / (2 * vectorWidth * numTiles);
  } else {
    return flops;
  }
}

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

void
Optimizer::outputConvDescription(const ExpImpl *exp,
                                 unsigned inDimY, unsigned inDimX,
                                 unsigned inNumChans,
                                 unsigned kernelSizeY, unsigned kernelSizeX,
                                 unsigned strideY, unsigned strideX,
                                 unsigned paddingY, unsigned paddingX,
                                 unsigned outNumChans) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = popconv::getOutputDim(inDimY,
                                                     inDimX,
                                                     kernelSizeY,
                                                     kernelSizeX,
                                                     strideY,
                                                     strideX,
                                                     paddingY,
                                                     paddingX,
                                                     false);
  const auto numParams =
      kernelSizeY * kernelSizeX * inNumChans * outNumChans + outNumChans;
  const auto batchSize = options.batchSize;
  auto fwdFlops = popconv::getFwdFlops(batchSize, inDimY, inDimX, inNumChans,
                                       kernelSizeY, kernelSizeX, strideY,
                                       strideX, paddingY, paddingX,
                                       outNumChans);
  auto bwdFlops = popconv::getBwdFlops(batchSize, inDimY, inDimX, inNumChans,
                                       kernelSizeY, kernelSizeX, strideY,
                                       strideX, paddingY, paddingX,
                                       outNumChans);
  auto wuFlops = popconv::getWuFlops(batchSize, inDimY, inDimX, inNumChans,
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
    popconv::reportPlanInfo(std::cout, *graph, dType, batchSize,
                            inDimY, inDimX, inNumChans,
                            {kernelSizeY, kernelSizeX, outNumChans},
                            {strideY, strideX}, {paddingY, paddingX},
                            false, options.convOptions);
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
    // FLOPS = number of macs in matrix multiply * 2
    const auto fwdFlops = prevSize * options.batchSize * size * 2;
    const auto bwdFlops = prevSize * options.batchSize * size * 2;
    const auto wuFlops = prevSize * options.batchSize * size * 2;
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
    case NON_LINEARITY_TANH:
      std::cout << "   -- Tanh layer\n";
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

Tensor Optimizer::
doConvolution(Graph &graph,
              const std::vector<unsigned> &stride,
              const std::vector<unsigned> &padding, unsigned outNumChans,
              Tensor in, Tensor weights,
              const std::string &partialsType,
              bool isFractional, bool transposeAndFlipWeights,
              Sequence &prog,
              const std::string &debugPrefix) {
  auto key = std::make_tuple(stride, padding, sig(graph, in),
                             sig(graph, weights),
                             outNumChans,
                             partialsType, isFractional,
                             transposeAndFlipWeights);
  std::vector<Tensor> args = {in, weights};
  auto it = convGraphCache.find(key);
  if (it != convGraphCache.end()) {
    auto &f = it->second;
    return f(args, prog);
  }
  using namespace popstd::graphfn;
  auto f = TensorFunction(
    graph,
    {input(in, "in"), input(weights, "weights")},
    [&](std::vector<poplar::Tensor> &args, Sequence &prog) {
      return convolution(graph, stride, padding, outNumChans,
                         args[0], args[1],
                         partialsType,
                         isFractional, transposeAndFlipWeights,
                         prog, debugPrefix, options.convOptions);
    });
  convGraphCache.emplace(key, f);
  return f(args, prog);
}

Tensor
Optimizer::createConvLayerFwd(const ExpImpl *exp,
                              unsigned kernelSizeY, unsigned kernelSizeX,
                              unsigned strideY, unsigned strideX,
                              unsigned paddingY, unsigned paddingX,
                              unsigned numChannels,
                              Sequence &initParamsProg,
                              Sequence &prog,
                              const std::string &debugPrefix) {
  auto prev = exp->deps().front();
  auto &in = out[prev];
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) =
      popconv::getOutputDim(in.dim(2), in.dim(3), kernelSizeY, kernelSizeX,
                            strideY, strideX, paddingY, paddingX, false);
  const auto batchSize = options.batchSize;
  unsigned inNumChanGroups = in.dim(1);
  unsigned inNumChans = inNumChanGroups * in.dim(4);
  unsigned inDimY = in.dim(2), inDimX = in.dim(3);
  Tensor weights = popconv::createWeights(*graph, in,
                                          kernelSizeY, kernelSizeX, numChannels,
                                          strideY, strideX,
                                          paddingY, paddingX,
                                          false, options.convOptions);
  params[exp].push_back(weights);

  popconv::mapWeights(weights, *graph, in, strideY, strideX,
                      paddingY, paddingX, false, options.convOptions);
  if (dType == "float") {
    auto hWeights =
        createRandomWeightInitializers(weights, 0, 1.0 / kernelSizeY,
                                       randomEngine);
    initParamsProg.add(Copy(hWeights.get(), weights));
    hParams[exp].push_back(std::move(hWeights));
  }

  numParams += weights.numElements();

  if (!options.skipFwd) {
    fwdFlops += popconv::getFwdFlops(batchSize,
                                     inDimY, inDimX, inNumChans, kernelSizeY,
                                     kernelSizeX, strideY,
                                     strideX, paddingY, paddingX, numChannels);
    fwdPerfectCycleTime +=
        popconv::getFwdPerfectCycleCount(*graph, dType, batchSize, inDimY,
                                         inDimX, inNumChans, kernelSizeY,
                                         kernelSizeX, strideY, strideX,
                                         paddingY, paddingX,
                                         numChannels);
  }
  return doConvolution(*graph, {strideY, strideX}, {paddingY, paddingX},
                       numChannels, in, weights, partialsType, false,
                       false, prog, debugPrefix);
}

Program
Optimizer::createResidualLayerFwd(const ExpImpl *exp, unsigned layerIndex,
                                  const std::string &debugPrefix) {
  const auto residual = dynamic_cast<const ResidualAdd *>(exp);
  assert(residual);
  unsigned numChannels, outDimY, outDimX;
  //The output will be the same batch/y/x dimensions and channel grouping
  //as the first input
  const auto prev0 = exp->deps()[0];
  const auto prev1 = exp->deps()[1];
  const auto &in0Shape = out[prev0].shape();
  numChannels = in0Shape[1] * in0Shape[4];
  outDimY = in0Shape[2];
  outDimX = in0Shape[3];
  auto outChansPerGroup = in0Shape[4];

  out[exp] = graph->addTensor(dType,
                              {in0Shape[0], numChannels / outChansPerGroup,
                               outDimY, outDimX, outChansPerGroup},
                              "activations." + std::to_string(layerIndex));
  mapActivations(*graph, out[exp]);

  Tensor in0 =
    popnn::arrangeResidualInput(*graph,
                                out[prev0],
                                out[exp].shape(), dType,
                                residual->method);
  Tensor in1 =
    popnn::arrangeResidualInput(*graph,
                                out[prev1],
                                out[exp].shape(), dType,
                                residual->method);
  switch (residual->method) {
  case RESIDUAL_PAD:
    {
      Program fwdProg =
        popnn::joinResidual(*graph,
                            in0,
                            in1,
                            out[exp],
                            debugPrefix);
      auto outShape = out[exp].shape();
      if (!options.skipFwd) {
        fwdFlops += outShape[0] *
            popnn::getNumberOfAdds(outShape[2], outShape[3],
                                   outShape[1] * outShape[4]);
        fwdPerfectCycleTime +=
            popnn::getPerfectCycleCount(*graph, dType,
                                        outShape[0], outShape[2], outShape[3],
                                        outShape[1] * outShape[4]);
      }
      return fwdProg;
    }
  default:
    throw enigma::enigma_error("This residual type not supported yet");
  }
  POPLIB_UNREACHABLE();
}

void Optimizer::genFwd(Sequence &actualFwdProg,
                       Sequence &initParamsProg) {
  Sequence dummyProg;
  Sequence &fwdProg = options.skipFwd ? dummyProg : actualFwdProg;
  for (unsigned layerIndex = 0; layerIndex < schedule.size(); ++layerIndex) {
    const auto &exp = schedule[layerIndex];
    std::cout << "-- Layer " << layerIndex << "\n";
    const std::string layerPrefix =
        "Layer" + std::to_string(layerIndex) + "/Fwd";
    outputDescription(exp);
    if (const auto *feed = dynamic_cast<const Feed *>(exp)) {
      dataSet = &feed->dataset;
      const Conv2d *conv = nullptr;
      for (const auto &use : uses[exp]) {
        if (const auto *c = dynamic_cast<const Conv2d *>(use)) {
          conv = c;
          break;
        }
      }
      if (conv) {
        out[exp] = popconv::createInput(*graph, dType, options.batchSize,
                                        dataSet->dim[0], dataSet->dim[1],
                                        dataSet->dim[2], conv->kernelSizeY,
                                        conv->kernelSizeX, conv->numChannels,
                                        conv->strideY, conv->strideX,
                                        conv->paddingY, conv->paddingX,
                                        false, "input", options.convOptions);
      } else {
        const auto dim = std::vector<size_t>({options.batchSize,
                                              dataSet->dim[2],
                                              dataSet->dim[0], dataSet->dim[1],
                                              1});
        out[exp] = graph->addTensor(dType, dim, "input");
        mapActivations(*graph, out[exp]);
      }
      feedIn = out[exp];
    } else if (const auto *c = dynamic_cast<const Conv2d *>(exp)) {
      out[exp] = createConvLayerFwd(exp, c->kernelSizeY, c->kernelSizeX,
                                    c->strideY, c->strideX, c->paddingY,
                                    c->paddingX,
                                    c->numChannels,
                                    initParamsProg, fwdProg, layerPrefix);

      Tensor biases = popconv::createBiases(*graph, dType, c->numChannels);
      params[exp].push_back(biases);
      const auto &weights = params[exp][0];
      popconv::mapBiases(biases, *graph, out[exp], weights, c->strideY,
                         c->strideX, c->paddingY, c->paddingX, false,
                         options.convOptions);

      if (dType == "float") {
        auto hBiases =
            createRandomWeightInitializers(biases, 0, 1.0 / c->numChannels,
                                           randomEngine);
        initParamsProg.add(Copy(hBiases.get(), biases));
        hParams[exp].push_back(std::move(hBiases));
      }

      numParams += biases.numElements();

      if (!options.skipFwd) {
        const auto dimY = out[exp].dim(2);
        const auto dimX = out[exp].dim(3);
        const auto numGroups = out[exp].dim(1);
        const auto numChansPerGroup = out[exp].dim(4);
        auto bBiases =
            biases.broadcast(options.batchSize * dimY * dimX, 0)
                  .reshape({options.batchSize, dimY, dimX, numGroups,
                            numChansPerGroup}).dimShuffle({0, 3, 1, 2, 4});
        addTo(*graph, out[exp], bBiases, 1.0, fwdProg, layerPrefix);
      }
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
    } else if (const auto *fc = dynamic_cast<const FullyConnected *>(exp)) {
      const auto prev = exp->deps().front();
      auto in = out[prev];
      const auto batchSize = options.batchSize;
      assert(in.numElements() % batchSize == 0);
      in = in.reshape({batchSize, in.numElements() / batchSize});
      const auto prevSize = in[0].numElements();
      const auto size = fc->channels;
      bool first = dynamic_cast<const Feed *>(prev);
      MatMulOptions mmOpt;
      mmOpt.leftHandArgUsedInTranspose = options.training && !first;
      mmOpt.partialsType = options.partialsType;
      mmOpt.cache = &poplinCache;
      auto weights = createMatMulInputA(*graph, dType,
                                        {size, prevSize},
                                        in.transpose(),
                                        "weights." + std::to_string(layerIndex),
                                        mmOpt);
      auto biases = graph->addTensor(dType, {size},
                                     "biases." + std::to_string(layerIndex));
      mapTensor(*graph, biases);
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

      out[exp] = matMul(*graph, in, weights.transpose(), fwdProg,
                        layerPrefix, mmOpt);
      auto bBiases = biases.broadcast(batchSize, 0)
                           .reshape({batchSize, size});
      addTo(*graph, out[exp], bBiases, 1.0, fwdProg, layerPrefix);
      const auto flops = prevSize * size * batchSize * 2;
      if (!options.skipFwd) {
        fwdFlops += flops;
        fwdPerfectCycleTime += getPerfectCycleTime(flops, dType, true, false);
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
      throw enigma::enigma_error("Unknown expresssion type");
    }
  }
}

void Optimizer::
createConvLayerBwd(const ExpImpl *exp, Tensor outGradient, unsigned layerIndex,
                   unsigned kernelSizeY, unsigned kernelSizeX,
                   unsigned strideY, unsigned strideX,
                   unsigned paddingY, unsigned paddingX,
                   bool backwardPassRequired, Sequence &actualBwdProg,
                   const std::string &debugPrefix) {
  Sequence dummyProg;
  Sequence &bwdProg = options.skipBwd ? dummyProg : actualBwdProg;
  Sequence &wuProg = options.skipWU ? dummyProg : actualBwdProg;
  const auto prev = exp->deps()[0];
  const auto in = out[prev];
  auto prevDimY = in.dim(2);
  auto prevDimX = in.dim(3);
  auto prevNumChans = in.dim(1) * in.dim(4);
  auto nextNumChans = out[exp].dim(1) * out[exp].dim(4);
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
    // Create transpose/flipped weights
    auto bwdWeights =
    popconv::createWeights(*graph, outGradient,
                           kernelSizeY, kernelSizeX, prevNumChans,
                           strideY, strideX, bwdPaddingY, bwdPaddingX,
                           isFractional, options.convOptions);
    popconv::mapWeights(bwdWeights, *graph, outGradient,
                        strideY, strideX, bwdPaddingY, bwdPaddingX,
                        isFractional, options.convOptions);
    if (!options.skipBwd) {
      bwdFlops += popconv::getBwdFlops(batchSize,
                                       prevDimY, prevDimX, prevNumChans,
                                       kernelSizeY, kernelSizeX, strideY,
                                       strideX, paddingY, paddingX,
                                       nextNumChans);
      bwdPerfectCycleTime +=
          popconv::getBwdPerfectCycleCount(*graph, dType, batchSize, prevDimY,
                                           prevDimX, prevNumChans, kernelSizeY,
                                           kernelSizeX, strideY, strideX,
                                           paddingY,
                                           paddingX, nextNumChans);
    }
    createBwdWeights(*graph, prevNumChans, weights, outGradient,
                     bwdWeights, strideY, strideX,
                     paddingY, paddingX, isFractional, bwdProg,
                     debugPrefix);
    // Perform convolution
    auto inGrad = doConvolution(*graph, {strideY, strideX},
                                {bwdPaddingY, bwdPaddingX}, prevNumChans,
                                outGradient, bwdWeights,
                                partialsType,
                                isFractional, false, bwdProg, debugPrefix);
    inGradient[exp].push_back(inGrad);
  }
  // TODO move before backward pass to reduce live range of the deltas.
  doConvolutionWeightUpdate(*graph, outGradient, weights,
                            in, strideY, strideX, paddingY,
                            paddingX, options.learningRate, wuProg,
                            debugPrefix);
  popconv::convolutionBiasUpdate(*graph, outGradient, biases,
                                 options.learningRate, wuProg,
                                 debugPrefix);

  if (!options.skipWU) {
    wuFlops += popconv::getWuFlops(batchSize,
                                   prevDimY, prevDimX, prevNumChans,
                                   kernelSizeY, kernelSizeX, strideY,
                                   strideX, paddingY, paddingX, nextNumChans);
    wuPerfectCycleTime +=
        popconv::getWuPerfectCycleCount(*graph, dType, batchSize, prevDimY,
                                        prevDimX, prevNumChans, kernelSizeY,
                                        kernelSizeX, strideY, strideX, paddingY,
                                        paddingX, nextNumChans);
  }
}


void Optimizer::genBwd(Sequence &actualBwdProg) {
  Sequence dummyProg;
  Sequence &bwdProg = options.skipBwd ? dummyProg : actualBwdProg;
  Sequence &wuProg = options.skipWU ? dummyProg : actualBwdProg;
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
    auto in = out[prev];
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
        throw enigma::enigma_error("Backpropagation of values consumed twice "
                                   "only implemented between a residual add"
                                   "and a non residual add");
      }
      if (exp != res->deps()[1]) {
        throw enigma::enigma_error("Backpropagation of values consumed twice "
                                   "only implemented between a residual add"
                                   "where residual is second parameter");
      }
      auto resGradient = inGradient[res][1];
      outGradient = inGradient[other][0];
      bwdProg.add(popnn::joinDeltas(*graph, outGradient, resGradient,
                                    layerPrefix));
    } else {
      throw enigma::enigma_error("Backpropagation of values consumed by more "
                                 "than 2 expressions not implemented");
    }

    if (const auto *fc = dynamic_cast<const FullyConnected *>(exp)) {
      auto weights = params[exp][0];
      auto biases = params[exp][1];
      const auto prevSize = in[0].numElements();
      const auto size = fc->channels;
      const auto prevShape = in.shape();
      in = in.reshape({in.dim(0), in.numElements() / in.dim(0)});
      MatMulOptions mmOpt;
      mmOpt.leftHandArgUsedInTranspose = backwardPassRequired;
      mmOpt.partialsType = options.partialsType;
      mmOpt.cache = &poplinCache;
      if (backwardPassRequired) {
        if (!options.skipBwd) {
          const auto flops = batchSize * prevSize * size * 2;
          bwdFlops += flops;
          bwdPerfectCycleTime += getPerfectCycleTime(flops, dType, true, false);
        }
        auto inGrad = matMul(*graph, outGradient, weights, bwdProg,
                             layerPrefix, mmOpt);
        inGrad = inGrad.reshape(prevShape);
        inGradient[exp].push_back(inGrad);
      }
      if (!options.skipWU) {
        const auto flops = batchSize * prevSize * size * 2;
        wuFlops += flops;
        wuPerfectCycleTime += getPerfectCycleTime(flops, dType, true, false);
      }
      if (options.inPlaceParamUpdate) {
        matMulAcc(*graph, weights, -eta, outGradient.transpose(), in,
                  wuProg, layerPrefix, mmOpt);
        reduceAcc(*graph, biases, -eta, outGradient, wuProg, layerPrefix);
      } else {
        auto weightDeltas = matMul(*graph, outGradient.transpose(), in,
                                   wuProg, layerPrefix, mmOpt);
        addTo(*graph, weights, weightDeltas, -eta, wuProg, layerPrefix);
        auto biasDeltas = reduce(*graph, outGradient, wuProg, layerPrefix);
        addTo(*graph, biases, biasDeltas, -eta, wuProg, layerPrefix);
      }
    } else if (const auto *c = dynamic_cast<const Conv2d *>(exp)) {
      createConvLayerBwd(exp, outGradient, layerIndex,
                         c->kernelSizeY, c->kernelSizeX,
                         c->strideY, c->strideX, c->paddingY,
                         c->paddingX,
                         backwardPassRequired, actualBwdProg,
                         layerPrefix);
    } else if (const auto *m = dynamic_cast<const MaxPool *>(exp)) {
      if (backwardPassRequired) {
        inGradient[exp].push_back(
          maxpool::maxPoolInputGradient(*graph,
                                        m->kernelSizeY, m->kernelSizeX,
                                        m->strideY, m->strideX,
                                        m->paddingY, m->paddingX,
                                        in, out[exp], outGradient, bwdProg,
                                        layerPrefix)
        );
        if (!options.skipBwd) {
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
        }
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
      throw enigma::enigma_error("Unrecognized layer type");
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
  this->options.convOptions.cache = &convCache;
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
      throw enigma::enigma_error("IPU modeling does not support > 2 IPUs");
    }

    info.globalSyncCycles =
        std::ceil(syncLatencyPerHop
                  * static_cast<double>(info.frequencyInHz * numHops * 2));

    graph = std::unique_ptr<Graph>(new Graph(createIPUModelDevice(info)));
  } else {
    graph = std::unique_ptr<Graph>(new Graph(createCPUDevice()));
  }
  popstd::addCodelets(*graph);
  popreduce::addCodelets(*graph);
  poplin::addCodelets(*graph);
  popconv::addCodelets(*graph);
  popnn::addCodelets(*graph);
  std::cerr << "Constructing program\n";
  fwdFlops = bwdFlops = wuFlops = 0;
  numParams = 0;
  fwdPerfectCycleTime = bwdPerfectCycleTime = wuPerfectCycleTime = 0;
  dType = getDTypeString(options.dataType);
  partialsType = getDTypeString(options.partialsType);
  auto initParamsProg = Sequence();
  auto fwdProg = Sequence();
  auto bwdProg = Sequence();
  createSchedule(exp);
  genFwd(fwdProg, initParamsProg);
  if (options.training)
    genBwd(bwdProg);
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

} // end namespace enigma
