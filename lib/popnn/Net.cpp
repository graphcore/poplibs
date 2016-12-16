#include "popnn/Net.hpp"
#include <boost/program_options.hpp>
#include <poplar/HalfFloat.hpp>
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
#include <fstream>
#include <iomanip>
#include <array>

using namespace poplar;
using namespace poplar::program;

bool parseCommandLine(int argc, char **argv, NetOptions &options,
                      bool &doTraining) {
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("ipus", po::value<unsigned>(&options.numIPUs)->default_value(1),
             "Number of IPUs")
    ("tiles-per-ipu",
     po::value<unsigned>(&options.tilesPerIPU)->default_value(1216),
     "Number of tiles per IPU")
    ("bytes-per-tile",
     po::value<unsigned>(&options.memoryBytesPerTile)
         ->default_value(1024 * 256),
     "Amount of memory per tile in bytes")
    ("ipu-exchange-bandwidth",
     po::value<unsigned>(&options.ipuExchangeBandwidth)->default_value(4),
     "IPU exchange bandwidth per tile in bytes")
    ("graph-reuse",
     po::value<bool>(&options.reuseLayerImplGraphs)->default_value(true),
     "Re-use graph structure for similar layers")
    ("data-path-width",
     po::value<unsigned>(
       &options.dataPathWidth
     )->default_value(64),
     "Width of the data path in bits")
    ("num-fp16-in-fp16-out-conv-units",
     po::value<unsigned>(
       &options.fp16InFp16OutConvUnitsPerTile
     )->default_value(8),
     "Number of convolutional units per tile with fp16 input and fp16 output")
    ("num-fp16-in-fp32-out-conv-units",
     po::value<unsigned>(
         &options.fp16InFp32OutConvUnitsPerTile
     )->default_value(8),
     "Number of convolutional units per tile with fp16 input and fp32 output")
    ("num-fp32-in-fp32-out-conv-units",
     po::value<unsigned>(
         &options.fp32InFp32OutConvUnitsPerTile
     )->default_value(4),
     "Number of convolutional units per tile with fp32 input and fp32 output")
    ("conv-coeff-load-bytes-per-cycle",
     po::value<unsigned>(
         &options.convUnitCoeffLoadBytesPerCycle
     )->default_value(16),
     "Number of bytes of coefficients loaded in the convolutional"
     " unit per cycle")
    ("train",
     po::value<bool>(
       &doTraining
     )->default_value(false),
     "Do training (forward, backward and weight update pass)")
    ("use-winograd-conv",
     po::value<bool>(
       &options.useWinogradConv
     )->default_value(false),
     "Use winograd for convolution layers")
    ("winograd-patch-size",
     po::value<unsigned>(
       &options.winogradPatchSize
     )->default_value(4),
     "Patch size for winograd convolution")
    ("batch-size",
     po::value<unsigned>(
       &options.batchSize
     )->default_value(1),
     "Batch size")
      ("show-plan-info",
     po::value<bool>(
       &options.showPlanInfo
     )->default_value(false),
     "Display result of planning decision for conv layers")
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
     bool useWinogradConv, unsigned winogradPatchSize,
     const std::string &debugPrefix) {
  const auto batchSize = activations.dim(0);
  mapActivations(graph, in);
  conv::mapWeights(weights, graph, plan, batchSize);
  conv::mapBiases(biases, graph, activations);
  mapActivations(graph, activations);
  return conv::convolution(graph, plan, strideY, strideX, paddingY, paddingX,
                           in, weights, biases, activations, partialsType,
                           isFractional, useWinogradConv, winogradPatchSize,
                           debugPrefix);
}

static Program
createBwdWeightsAndBiases(Graph &graph, const conv::Plan &bwdPlan,
                          const conv::Plan &fwdPlan,
                          Tensor weights, Tensor deltasOut,
                          Tensor bwdWeights,
                          Tensor bwdBiases) {
  const auto batchSize = deltasOut.dim(0);
  const auto outNumChans = deltasOut.dim(1) * deltasOut.dim(4);
  const auto dType = graph.getTensorElementType(weights);
  auto prog = Sequence();
  conv::mapWeights(weights, graph, fwdPlan, batchSize);
  conv::mapWeights(bwdWeights, graph, bwdPlan, batchSize);
  prog.add(conv::weightsTransposeChansFlipXY(graph, weights, bwdWeights));
  auto zeros = graph.addConstantTensor(dType, {outNumChans}, 0);
  conv::mapBiases(bwdBiases, graph, deltasOut);
  prog.add(Copy(bwdBiases, zeros));
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

// Define structures containing tensor ops to pass between functions/methods.
struct Net::ConvOp {
  POPNN_TENSOR_OP_TYPE(convolution, conv::Plan) op;
  ConvOp(POPNN_TENSOR_OP_TYPE(convolution, conv::Plan) op) :
    op(std::move(op)) {}
  template<typename ...Args>
  Program operator()(Args&&... args) {
    return op(std::forward<Args>(args)...);
  };
};
struct Net::ConvBwdWeightsOp {
  POPNN_TENSOR_OP_TYPE(createBwdWeightsAndBiases, conv::Plan) op;
  ConvBwdWeightsOp(
    POPNN_TENSOR_OP_TYPE(createBwdWeightsAndBiases, conv::Plan) op
  ) :  op(std::move(op)) {}
  template<typename ...Args>
  Program operator()(Args&&... args) {
    return op(std::forward<Args>(args)...);
  };
};
struct Net::ConvWuOp {
  POPNN_TENSOR_OP_TYPE(convolutionWeightUpdate, conv::Plan) op;
  ConvWuOp(POPNN_TENSOR_OP_TYPE(convolutionWeightUpdate, conv::Plan) op) :
    op(std::move(op)) {}
  template<typename ...Args>
  Program operator()(Args&&... args) {
    return op(std::forward<Args>(args)...);
  };
};

/* When a Net object is constructed the corrensponding poplar graph is
   made */
Net::Net(DataSet &data, unsigned batchSize,
         std::vector<std::unique_ptr<Layer>> &layers,
         LossType lossType,
         float learningRate,
         NetType netType,
         DType dType,
         NetOptions options) :
  netType(netType), options(options),
  batchSize(batchSize),
  eta(learningRate),
  layers(std::move(layers)),
  dType(getDTypeString(dType)),
  partialsType(getDTypeString(FP32))
{
  initialize(data, lossType);
}

Net::Net(DataSet &data, unsigned batchSize,
         std::vector<std::unique_ptr<Layer>> &&layers,
         LossType lossType,
         float learningRate,
         NetType netType,
         DType dType,
         NetOptions options) :
  netType(netType), options(options),
  batchSize(batchSize),
  eta(learningRate),
  layers(std::move(layers)),
  dType(getDTypeString(dType)),
  partialsType(getDTypeString(FP32))
{
  initialize(data, lossType);
}

Net::Net(DataSet &data, unsigned batchSize,
         std::vector<std::unique_ptr<Layer>> &layers,
         LossType lossType,
         float learningRate,
         NetType netType,
         DType dType,
         DType partialsType,
         NetOptions options) :
  netType(netType), options(options),
  batchSize(batchSize),
  eta(learningRate),
  layers(std::move(layers)),
  dType(getDTypeString(dType)),
  partialsType(getDTypeString(partialsType))
{
  initialize(data, lossType);
}

Net::Net(DataSet &data, unsigned batchSize,
         std::vector<std::unique_ptr<Layer>> &&layers,
         LossType lossType,
         float learningRate,
         NetType netType,
         DType dType,
         DType partialsType,
         NetOptions options) :
  netType(netType), options(options),
  batchSize(batchSize),
  eta(learningRate),
  layers(std::move(layers)),
  dType(getDTypeString(dType)),
  partialsType(getDTypeString(partialsType))
{
  initialize(data, lossType);
}

conv::Plan
Net::getFwdConvPlan(unsigned i, unsigned inDimY, unsigned inDimX,
                    unsigned inNumChans) {
  auto it = fwdConvPlans.find(i);
  if (it != fwdConvPlans.end())
    return it->second;
  const auto *layer = layers[i].get();
  const auto *c = dynamic_cast<const ConvLayer *>(layer);
  assert(c);
  conv::Plan plan =
      planner.createPlan(inDimY, inDimX, inNumChans,
                         c->kernelSizeY, c->kernelSizeX,
                         c->strideY, c->strideX, c->paddingY,
                         c->paddingX,
                         c->numChannels, batchSize, dType,
                         partialsType, false, *graph);

  fwdConvPlans.emplace(i, plan);
  return plan;
}

conv::Plan
Net::getBwdConvPlan(unsigned i, unsigned prevDimY, unsigned prevDimX,
                    unsigned prevNumChans) {
  auto it = bwdConvPlans.find(i);
  if (it != bwdConvPlans.end())
    return it->second;
  const auto *layer = layers[i].get();
  const auto *c = dynamic_cast<const ConvLayer *>(layer);
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
                            prevNumChans, batchSize, dType,
                            partialsType, isFractional, *graph);
  bwdConvPlans.emplace(i, plan);
  return plan;
}

conv::Plan
Net::getWuConvPlan(unsigned i, unsigned prevDimY, unsigned prevDimX,
                   unsigned prevNumChans, unsigned actsChansPerGroup,
                   unsigned deltasChanPerGroup,
                   unsigned weightOutChansPerGroup) {
  auto it = wuConvPlans.find(i);
  if (it != wuConvPlans.end())
    return it->second;
  const auto *layer = layers[i].get();
  const auto *c = dynamic_cast<const ConvLayer *>(layer);
  assert(c);
  conv::Plan plan =
      planner.createWeightUpdatePlan(prevDimY, prevDimX, prevNumChans,
                                     actsChansPerGroup, deltasChanPerGroup,
                                     weightOutChansPerGroup, c->kernelSizeY,
                                     c->kernelSizeX, c->strideY, c->strideX,
                                     c->paddingY, c->paddingX, c->numChannels,
                                     batchSize, dType, partialsType, false,
                                     *graph);
  wuConvPlans.emplace(i, plan);
  return plan;
}


unsigned
Net::getRequiredChansPerGroupBwd(int i) {
  if (i < 0)
    return 0;
  const auto *layer = layers[i].get();
  if (dynamic_cast<const FullyConnectedLayer *>(layer)) {
    return 0;
  } else if (dynamic_cast<const ConvLayer *>(layer)) {
    // There is no need to calculate the gradient of the activations for the
    // first layer. TODO pick a sensible channel grouping in this case.
    if (i == 0)
      return acts[i + 1].dim(4);
    auto prevDimY = acts[i].dim(2);
    auto prevDimX = acts[i].dim(3);
    auto prevNumChans = acts[i].dim(1) * acts[i].dim(4);
    auto plan = getBwdConvPlan(i, prevDimY, prevDimX, prevNumChans);
    return plan.inChansPerGroup;
  } else if (dynamic_cast<const MaxPoolLayer *>(layer)) {
    return getRequiredChansPerGroupBwd(i - 1);
  } else if (const auto *r = dynamic_cast<const ResidualLayer *>(layer)) {
    return getRequiredChansPerGroupBwd(i - r->resIndex[0]);
  } else {
    throw popnn::popnn_error("Unrecognized layer type");
  }
}

unsigned
Net::getRequiredChansPerGroupFwd(unsigned i, unsigned inDimY,
                                 unsigned inDimX,
                                 unsigned inNumChans) {
  if (i >= layers.size())
    return 0;
  const auto *layer = layers[i].get();
  if (dynamic_cast<const FullyConnectedLayer *>(layer)) {
    // A fully connected layer wants the channel grouping to be
    // the same forwards and backwards.
    if (netType == TrainingNet)
      return getRequiredChansPerGroupBwd(i - 1);
    else
      return 0;
  } else if (dynamic_cast<const ConvLayer *>(layer)) {
    auto plan = getFwdConvPlan(i, inDimY, inDimX, inNumChans);
    return plan.inChansPerGroup;
  } else if (dynamic_cast<const ResidualLayer *>(layer)) {
    // Use grouping of the following layer
    // For Inception type networks this cannot rely on (i+1), but must search
    return getRequiredChansPerGroupFwd(i + 1, inDimY, inDimX, inNumChans);
  } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
    unsigned outDimY, outDimX;
    std::tie(outDimY, outDimX) = maxpool::getOutputDim(inDimY, inDimX,
                                                       m->kernelSize,
                                                       m->stride,
                                                       m->padding);
    return getRequiredChansPerGroupFwd(i + 1, outDimY, outDimX, inNumChans);
  } else {
    throw popnn::popnn_error("Unrecognized layer type");
  }
}

enum {
  INIT_PARAMS_PROG,
  TRAIN_PROG,
  TEST_PROG,
  NUM_PROGS
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

static void
outputResidualDescription(const ResidualLayer &rLayer,
                          unsigned thisLayer,
                          const std::vector<poplar::Tensor> &acts,
                          const Tensor &in,
                          bool forwardOnly)
{
  std::cout << "   -- Residual layer:\n";
  for (auto inputOffset : rLayer.resIndex) {
    auto idx = thisLayer - inputOffset + 1;
    assert(acts[idx].getDimensionality() == 5);
    const auto &dims = acts[idx].dims();
    std::cout << "        Input(-"<<inputOffset<<"):  "
              << dims[2] << "x" << dims[3] <<"x" << dims[1] * dims[4] << "\n";
  }

  const auto &dims = acts[thisLayer - rLayer.resIndex[0] + 1].dims();
  auto flops = residual::getNumberOfAdds(dims[2], dims[3],
                                         dims[1] * dims[4],
                                         forwardOnly);
  std::cout << "        Output:     "
            << dims[2] << "x" << dims[3] <<"x" << dims[1] * dims[4] << "\n"
            << "        FLOPs:      " << flops << "\n";
}

void
Net::outputConvDescription(unsigned layerIdx,
                           unsigned inDimY, unsigned inDimX,
                           unsigned inNumChans,
                           unsigned kernelSizeY, unsigned kernelSizeX,
                           unsigned strideY, unsigned strideX,
                           unsigned paddingY, unsigned paddingX,
                           unsigned outNumChans, bool forwardOnly) {
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
  auto flops = conv::getFlops(batchSize, inDimY, inDimX, inNumChans,
                              kernelSizeY, kernelSizeX, strideY,
                              strideX, paddingY, paddingX, outNumChans,
                              forwardOnly);
  std::cout << "   -- Convolutional layer:\n"
            << "        Size: " << kernelSizeX << "x" << kernelSizeY << "\n"
            << "        Stride: " << strideX << "x" << strideY << "\n"
            << "        Padding: " << paddingX << "x" << paddingY << "\n"
            << "        Input:  " << inDimY << "x" << inDimX
            <<   "x" << inNumChans << "\n"
            << "        Output: " << outDimY << "x" << outDimX
            <<   "x" << outNumChans << "\n"
            << "        Params: " << numParams << "\n"
            << "        FLOPs:  " << flops << "\n";

  if (options.showPlanInfo) {
    std::cout << fwdConvPlans[layerIdx];
  }
}

void Net::outputDescription(const Layer *layer, unsigned i, Tensor in,
                            bool forwardOnly) {
  if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
    const auto prevSize = in[0].numElements();
    const auto size = fc->size;
    const auto flops = fc::getNumFlops(batchSize, prevSize, size, forwardOnly);
    std::cout << "   -- Fully connected layer:\n"
              << "        Input:  "  << prevSize << "\n"
              << "        Output: " << size << "\n"
              << "        Params: " << size * (prevSize + 1) << "\n"
              << "        FLOPs:  " << flops << "\n";
  } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
    outputConvDescription(i, in.dim(2), in.dim(3), in.dim(1) * in.dim(4),
                          c->kernelSizeY, c->kernelSizeX, c->strideY,
                          c->strideX, c->paddingY, c->paddingX,
                          c->numChannels, forwardOnly);
  } else if (const auto *r = dynamic_cast<const ResidualLayer *>(layer)) {
    outputResidualDescription(*r, i, acts, in, forwardOnly);
  } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
    unsigned outDimY, outDimX;
    std::tie(outDimY, outDimX) = maxpool::getOutputDim(in.dim(2),
                                                       in.dim(3),
                                                       m->kernelSize,
                                                       m->stride,
                                                       m->padding);
    const auto numChannels = in.dim(1) * in.dim(4);
    const auto flops = maxpool::getNumFlops(batchSize,
                                            in.dim(2),
                                            in.dim(3),
                                            numChannels,
                                            m->kernelSize,
                                            m->stride,
                                            m->padding);
    std::cout << "   -- Max pooling layer:\n"
              << "        Size: " << m->kernelSize << "x"
              << m->kernelSize << "\n"
              << "        Stride: " << m->stride << "\n"
              << "        Input:  " << in.dim(2) << "x" << in.dim(3)
              <<   "x" << numChannels << "\n"
              << "        Output: " << outDimY << "x" << outDimX
              <<   "x" << numChannels << "\n"
              << "        FLOPs:  " << flops << "\n";
  } else {
    assert(0 && "Unrecognized layer type");
  }
}

Program
Net::createResidualLayerFwd(unsigned i,
                            const ResidualLayer &residualLayer,
                            const std::string &debugPrefix) {
  if (residualLayer.resIndex.size() != 2) {
    throw popnn::popnn_error("A residual layer must have exactly two inputs");
  }
  if (residualLayer.resIndex[0] == 0 || residualLayer.resIndex[1] == 0) {
    throw popnn::popnn_error(
        "Residual layer inputs must be from earlier layers");
  }
  if (residualLayer.resIndex[0] > i + 1|| residualLayer.resIndex[1] > i + 1) {
    throw popnn::popnn_error("Residual offset points beyond start of network");
  }
  unsigned numChannels, outDimY, outDimX;
  //The output will be the same batch/y/x dimensions as the first input with
  //channel grouping chosen to match the following layer
  const auto &in0Dims = acts[i - residualLayer.resIndex[0] + 1].dims();
  numChannels = in0Dims[1] * in0Dims[4];
  outDimY = in0Dims[2];
  outDimX = in0Dims[3];
  auto outChansPerGroup = getRequiredChansPerGroupFwd(i + 1, outDimY, outDimX,
                                                      numChannels);
  if (outChansPerGroup == 0) {
    outChansPerGroup = in0Dims[4];
  }

  acts[i + 1] = graph->addTensor(dType,
                                 {in0Dims[0], numChannels / outChansPerGroup,
                                   outDimY, outDimX, outChansPerGroup},
                                 "activations." + std::to_string(i));
  mapActivations(*graph, acts[i + 1]);
  Tensor in0 =
    residual::arrangeResidualInput(*graph,
                                   acts[i - residualLayer.resIndex[0] + 1],
                                   acts[i + 1].dims(), dType,
                                   residualLayer.resMethod);
  Tensor in1 =
    residual::arrangeResidualInput(*graph,
                                   acts[i - residualLayer.resIndex[1] + 1],
                                   acts[i + 1].dims(), dType,
                                   residualLayer.resMethod);
  switch (residualLayer.resMethod) {
  case RESIDUAL_PAD:
    {
      Program fwdProg =
        residual::joinResidual(*graph,
                               in0,
                               in1,
                               acts[i + 1],
                               debugPrefix);
      auto outDims = acts[i].dims();
      numFlops += outDims[0] *
        residual::getNumberOfAdds(outDims[2], outDims[3],
                                  outDims[1] * outDims[4],
                                  netType == TestOnlyNet);
      perfectCycleTime +=
        residual::getPerfectCycleCount(*graph, dType,
                                       outDims[0], outDims[2], outDims[3],
                                       outDims[1] * outDims[4],
                                       netType == TestOnlyNet);
      return fwdProg;
    }
  default:
    throw popnn::popnn_error("This residual type not supported yet");
  }
  POPNN_UNREACHABLE();
}

// Combined deltas at the branch towards a residual layer
// \a i index of the layer that will take the combined deltas
Program
Net::createResidualLayerBwd(unsigned i, const std::string &debugPrefix) {
  // Add the residual deltas to the existing deltas[i]

  assert(residualDeltaIdxs[i].first == i + 1);
  // The residual path must come from a later layer
  assert(residualDeltaIdxs[i].second > i + 1);
  Tensor &outIn0 = deltas[residualDeltaIdxs[i].first];
  Tensor &in1 = deltas[residualDeltaIdxs[i].second];
  const auto &in0Dims = outIn0.dims();

  switch (RESIDUAL_PAD) {
  case RESIDUAL_PAD:
    {
      Program fwdProg =
        residual::joinDeltas(*graph,
                             outIn0,
                             in1,
                             debugPrefix);
      return fwdProg;
    }
  case RESIDUAL_CONCATENATE:
    throw popnn::popnn_error("this residual type not supported yet");
  }
  POPNN_UNREACHABLE();
}


Program
Net::createConvLayerFwd(unsigned i,
                        unsigned kernelSizeY, unsigned kernelSizeX,
                        unsigned strideY, unsigned strideX,
                        unsigned paddingY, unsigned paddingX,
                        unsigned numChannels,
                        Sequence &initParamsProg,
                        ConvOp &doConv,
                        const std::string &debugPrefix) {
  auto &in = acts[i];
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) =
      conv::getOutputDim(in.dim(2), in.dim(3), kernelSizeY, kernelSizeX,
                         strideY, strideX, paddingY, paddingX);
  auto outChansPerGroup = getRequiredChansPerGroupFwd(i + 1,
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
  acts[i + 1] = graph->addTensor(dType,
                                 {batchSize,
                                  outNumChanGroups,
                                  outDimY, outDimX,
                                  outChansPerGroup},
                                 "activations." + std::to_string(i));
  mapActivations(*graph, acts[i + 1]);

  unsigned inNumChanGroups = in.dim(1);
  unsigned inNumChans = inNumChanGroups * in.dim(4);
  unsigned inDimY = in.dim(2), inDimX = in.dim(3);
  auto plan = getFwdConvPlan(i, inDimY, inDimX, inNumChans);
  Tensor weights = conv::createWeights(*graph, dType, inNumChans,
                                       kernelSizeY, kernelSizeX,
                                       numChannels, plan);
  Tensor biases = conv::createBiases(*graph, dType, numChannels);
  params[i].push_back(weights);
  params[i].push_back(biases);

  conv::mapWeights(weights, *graph, plan, batchSize);
  conv::mapBiases(biases, *graph, acts[i + 1]);
  if (dType == "float") {
    auto hWeights =
        createRandomWeightInitializers(weights, 0, 1.0 / kernelSizeY,
                                       randomEngine);
    auto hBiases =
        createRandomWeightInitializers(weights, 0, 1.0 / kernelSizeY,
                                       randomEngine);
    initParamsProg.add(Copy(weights, hWeights.get()));
    initParamsProg.add(Copy(biases, hBiases.get()));
    hParams.push_back(std::move(hWeights));
    hParams.push_back(std::move(hBiases));
  }

  numFlops += conv::getFlops(batchSize,
                             inDimY, inDimX, inNumChans, kernelSizeY,
                             kernelSizeX, strideY,
                             strideX, paddingY, paddingX, numChannels,
                             netType == TestOnlyNet || i == 0);
  numParams += weights.numElements() + biases.numElements();
  perfectCycleTime +=
      conv::getPerfectCycleCount(*graph, dType, batchSize, inDimY, inDimX,
                                 inNumChans, kernelSizeY, kernelSizeX,
                                 strideY, strideX, paddingY, paddingX,
                                 numChannels, netType == TestOnlyNet || i == 0);
  /* use empty string to ensure that layer graph can be reused */
  return doConv(*graph, plan, strideY, strideX, paddingY, paddingX, in, weights,
                biases, acts[i + 1], partialsType, false, false, 4, "");
}

Program Net::createConvLayerBwd(unsigned i,
                                unsigned kernelSizeY, unsigned kernelSizeX,
                                unsigned strideY, unsigned strideX,
                                unsigned paddingY, unsigned paddingX,
                                NonLinearityType nonLinearityType,
                                bool backwardPassRequired,
                                ConvBwdWeightsOp &convBwdWeights,
                                ConvOp &doConv, ConvWuOp &convWU,
                                const std::string &debugPrefix) {
  auto prog = Sequence();
  auto prevDimY = acts[i].dim(2);
  auto prevDimX = acts[i].dim(3);
  auto prevNumChans = acts[i].dim(1) * acts[i].dim(4);
  auto nextNumChans = acts[i + 1].dim(1) * acts[i + 1].dim(4);
  auto fwdPlan = getFwdConvPlan(i, prevDimY, prevDimX, prevNumChans);
  auto bwdPlan = getBwdConvPlan(i, prevDimY, prevDimX, prevNumChans);
  auto weights = params[i][0];
  auto biases = params[i][1];
  Tensor zDeltas = deltas[i + 1];
  if (nonLinearityType != NON_LINEARITY_NONE) {
    zDeltas = graph->addTensor(dType, deltas[i + 1].dims(),
                                      "zDeltas");
    mapActivations(*graph, zDeltas);
    prog.add(bwdNonLinearity(*graph, acts[i + 1], deltas[i + 1], zDeltas,
                             nonLinearityType, debugPrefix));
  }

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
        conv::createWeights(*graph, dType, nextNumChans, kernelSizeY,
                            kernelSizeX, prevNumChans, bwdPlan);
    conv::mapWeights(bwdWeights, *graph, bwdPlan, batchSize);
    auto biases = graph->addTensor(dType, {prevNumChans}, "zeroBiases");
    conv::mapBiases(biases, *graph, deltas[i]);

    prog.add(convBwdWeights(*graph, bwdPlan, fwdPlan, weights, deltas[i],
                            bwdWeights, biases));
    // Perform convolution
    /* use empty string to ensure that layer graph can be reused */
    prog.add(doConv(*graph, bwdPlan, strideY, strideX,
                    bwdPaddingY, bwdPaddingX, zDeltas, bwdWeights,
                    biases, deltas[i], bwdPlan.getPartialType(),
                    isFractional, false, 4, ""));
  }
  // TODO move before backward pass to reduce live range of the deltas.
  auto wuPlan = getWuConvPlan(i, prevDimY, prevDimX, prevNumChans,
                              acts[i].dim(4), zDeltas.dim(4), weights.dim(4));
  /* use empty string to ensure that layer graph can be reused */
  prog.add(convWU(*graph, wuPlan, fwdPlan, zDeltas, weights, biases, acts[i],
                  strideY, strideX, paddingY, paddingX, eta, ""));
  return prog;
}

namespace popnn {
std::string findGraphProg() {
  // TODO: This needs to be replaced with a proper object search mechanism
  // in poplar.
  std::string path = "lib/popnn/popnn.gp";
  if (std::ifstream(path).good())
    return path;
  path = "../" + path;
  return path;
}
}

using namespace popnn;

void Net::initialize(DataSet &dataSet, LossType lossType) {
  numTestBatches = dataSet.numTest / batchSize;
  env = std::unique_ptr<GraphProgEnv>(
      new GraphProgEnv(popnn::findGraphProg(), GraphProgFileType::Object));
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

    graph = std::unique_ptr<Graph>(new Graph(*env, createIPUModelDevice(info)));
  } else {
    graph = std::unique_ptr<Graph>(new Graph(*env, createCPUDevice()));
  }
  std::cerr << "Constructing program\n";
  ConvOp convOp =
      createTensorOp<conv::Plan>(
        *graph, convolution, "conv",
        {{TensorOpParamType::InputTensor},
         {TensorOpParamType::InputTensor},
         {TensorOpParamType::InputTensor},
         {TensorOpParamType::OutputTensor}});
  ConvBwdWeightsOp convBwdWeightsOp =
      createTensorOp<conv::Plan>(
         *graph, createBwdWeightsAndBiases, "createBwdWeights",
         {{TensorOpParamType::InputTensor,
           TensorOpParamType::NotParamTensor,
           TensorOpParamType::OutputTensor,
           TensorOpParamType::OutputTensor}});
  ConvWuOp convWuOp =
      createTensorOp<conv::Plan>(
        *graph, convolutionWeightUpdate, "convWeightUpdate",
        {{TensorOpParamType::InputTensor},
         {TensorOpParamType::InOutTensor},
         {TensorOpParamType::InOutTensor},
         {TensorOpParamType::InputTensor}});
  numFlops = 0;
  numParams = 0;
  perfectCycleTime = 0;
  auto initParamsProg = Sequence();
  auto fwdProg = Sequence();
  auto bwdProg = Sequence();
  auto chansPerGroup =
      getRequiredChansPerGroupFwd(0, dataSet.dim[0], dataSet.dim[1],
                                  dataSet.dim[2]);
  if (chansPerGroup == 0)
    chansPerGroup = dataSet.dim[2];
  assert(dataSet.dim[2] % chansPerGroup == 0);
  const auto numChanGroups = dataSet.dim[2] / chansPerGroup;
  const auto dim = std::vector<size_t>({batchSize,
                                        numChanGroups,
                                        dataSet.dim[0], dataSet.dim[1],
                                        chansPerGroup});
  acts.resize(layers.size() + 1);
  deltas.resize(layers.size() + 1);
  residualDeltaIdxs.resize(layers.size() + 1);// last 2 entries always empty
  params.resize(layers.size());
  acts[0] = graph->addTensor(dType, dim, "input");
  mapActivations(*graph, acts[0]);
  for (unsigned i = 0; i < layers.size(); ++i) {
    const auto *layer = layers[i].get();
    std::cout << "-- Layer " << i << "\n";
    const std::string layerPrefix = "Layer:" + std::to_string(i);
    outputDescription(layer, i, acts[i], netType == TestOnlyNet || i == 0);
    if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
      const auto prevSize = acts[i][0].numElements();
      const auto size = fc->size;
      acts[i + 1] = graph->addTensor(dType, {batchSize, size},
                                 "activations." + std::to_string(i));
      mapActivations(*graph, acts[i + 1]);
      auto activationsMapping =
          computeActivationsMapping(*graph, acts[i + 1][0], 0, batchSize);
      bool forwardOnly = i == 0 || netType == TestOnlyNet;
      const auto &plan =
          fullyConnectedPlan.emplace(
            i,
            fc::createPlan(*graph, dType, partialsType, prevSize,
                           std::move(activationsMapping),
                           forwardOnly)
         ).first->second;
      Tensor weights, biases;
      std::tie(weights, biases) = fc::createParams(*graph, dType,
                                                   prevSize, size);
      params[i] = {weights, biases};
      if (dType == "float") {
         auto hWeights =
             createRandomWeightInitializers(weights, 0, 1.0 / prevSize,
                                            randomEngine);
         auto hBiases =
             createRandomWeightInitializers(weights, 0, 1.0 / prevSize,
                                            randomEngine);
         initParamsProg.add(Copy(weights, hWeights.get()));
         initParamsProg.add(Copy(biases, hBiases.get()));
         hParams.push_back(std::move(hWeights));
         hParams.push_back(std::move(hBiases));
      }
      fwdProg.add(fc::fullyConnected(*graph, size, fc->nonLinearityType,
                                     acts[i], weights, biases,
                                     acts[i + 1], plan,
                                     layerPrefix));
      numFlops += fc::getNumFlops(batchSize, prevSize, size,
                                  netType == TestOnlyNet || i == 0);
      fwdProg.add(fwdNonLinearity(*graph, acts[i + 1],
                                  fc->nonLinearityType,
                                  layerPrefix));
      numParams += weights.numElements() + biases.numElements();
      perfectCycleTime +=
          fc::getPerfectCycleCount(*graph, batchSize, prevSize,
                                   size, dType,
                                   netType == TestOnlyNet || i == 0);
    } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
      fwdProg.add(createConvLayerFwd(i, c->kernelSizeY, c->kernelSizeX,
                                     c->strideY, c->strideX, c->paddingY,
                                     c->paddingX,
                                     c->numChannels,
                                     initParamsProg, convOp,
                                     layerPrefix));
      fwdProg.add(fwdNonLinearity(*graph, acts[i + 1],
                                  c->nonLinearityType,
                                  layerPrefix));
    } else if (const auto *r = dynamic_cast<const ResidualLayer *>(layer)) {
      fwdProg.add(createResidualLayerFwd(i, *r, layerPrefix));
      if (r->nonLinearityType != NON_LINEARITY_NONE) {
        fwdProg.add(fwdNonLinearity(*graph, acts[i + 1],
                                    r->nonLinearityType,
                                    layerPrefix));
      }

    } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
      const auto &in = acts[i];
      unsigned outDimY, outDimX;
      std::tie(outDimY, outDimX) = maxpool::getOutputDim(in.dim(2),
                                                         in.dim(3),
                                                         m->kernelSize,
                                                         m->stride,
                                                         m->padding);
      acts[i + 1] = graph->addTensor(dType,
                                     {batchSize,
                                      in.dim(1),
                                      outDimY, outDimX,
                                      in.dim(4)},
                                     "activations." + std::to_string(i));
      mapActivations(*graph, acts[i + 1]);
      fwdProg.add(maxpool::maxPool(*graph,
                                   m->kernelSize, m->stride, m->padding,
                                   acts[i], acts[i + 1],
                                   layerPrefix));
      numFlops += maxpool::getNumFlops(batchSize,
                                       in.dim(2), in.dim(3),
                                       in.dim(1) * in.dim(4),
                                       m->kernelSize,
                                       m->stride, m->padding);
      perfectCycleTime +=
          maxpool::getPerfectCycleCount(*graph, dType, batchSize,
                                        in.dim(2), in.dim(3),
                                        in.dim(1) * in.dim(4),
                                        m->kernelSize,
                                        m->stride, m->padding);
    } else {
      assert(0 && "Unrecognized layer type");
    }
  }
  auto lastAct = *(acts.end() - 1);
  Tensor expected = graph->addTensor("unsigned", {batchSize}, "expected");
  graph->setTileMapping(expected, 0);
  Tensor numCorrect = graph->addTensor("unsigned", {1}, "numCorrect");
  graph->setTileMapping(numCorrect, 0);
  Tensor loss = graph->addTensor(dType, {batchSize}, "loss");
  graph->setTileMapping(loss, 0);
  deltas[layers.size()] = graph->addTensor(dType, lastAct.dims(), "deltas");
  mapActivations(*graph, deltas[layers.size()]);
  lastAct = lastAct.reshape({batchSize, lastAct.numElements() / batchSize});
  auto firstDeltas = deltas[layers.size()];
  firstDeltas = firstDeltas.reshape(lastAct.dims());
  auto calcLossProg = calcLoss(*graph,
                               lastAct,
                               expected,
                               loss,
                               firstDeltas,
                               numCorrect,
                               dType, "unsigned int",
                               lossType);
  fwdProg.add(Sequence(Copy(numCorrect, &hNumCorrect),
                       calcLossProg,
                       Copy(&hNumCorrect, numCorrect)));
  if (netType == TrainingNet) {
    for (int i = layers.size() - 1; i >= 0; --i) {
      bool backwardPassRequired = (i != 0);

      const std::string layerPrefix = "LayerBwd:" + std::to_string(i);

      if (residualDeltaIdxs[i].first != 0) {
        // A residual path was taken from this layer
        bwdProg.add(createResidualLayerBwd(i, layerPrefix));
      }

      if (backwardPassRequired) {
        if (acts[i].dims().size() == 5) {
          auto numChannels = acts[i].dim(1) * acts[i].dim(4);
          auto chansPerGroup = getRequiredChansPerGroupBwd(i - 1);
          if (chansPerGroup == 0)
            chansPerGroup = numChannels;
          assert(numChannels % chansPerGroup == 0);
          const auto numChanGroups = numChannels / chansPerGroup;
          deltas[i] = graph->addTensor(dType, {batchSize,
                                               numChanGroups,
                                               acts[i].dim(2),
                                               acts[i].dim(3),
                                               chansPerGroup} , "deltas");
         } else {
           assert(acts[i].dims().size() == 2);
           deltas[i] = graph->addTensor(dType, acts[i].dims(), "deltas");
        }
        mapActivations(*graph, deltas[i]);
      }

      const auto *layer = layers[i].get();
      if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
        Tensor zDeltas = graph->addTensor(dType, deltas[i + 1].dims(),
                                          "zDeltas");
        mapActivations(*graph, zDeltas);
        auto weights = params[i][0];
        auto biases = params[i][1];
        const auto &plan = fullyConnectedPlan.find(i)->second;
        bwdProg.add(bwdNonLinearity(*graph, acts[i + 1], deltas[i + 1],
                                    zDeltas,
                                    fc->nonLinearityType,
                                    layerPrefix));

        if (backwardPassRequired)
          bwdProg.add(fc::fullyConnectedBackward(*graph, zDeltas, weights,
                                                 deltas[i], plan, layerPrefix));
        bwdProg.add(fc::fullyConnectedWeightUpdate(*graph, zDeltas, acts[i],
                                                   weights, biases, eta, plan,
                                                   layerPrefix));
      } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
        bwdProg.add(createConvLayerBwd(i, c->kernelSizeY, c->kernelSizeX,
                                       c->strideY, c->strideX, c->paddingY,
                                       c->paddingX,
                                       c->nonLinearityType,
                                       backwardPassRequired,
                                       convBwdWeightsOp, convOp, convWuOp,
                                       layerPrefix));
      } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
        if (backwardPassRequired)
          bwdProg.add(maxpool::maxPoolBackward(*graph, m->kernelSize, m->stride,
                                               m->padding, acts[i],
                                               acts[i + 1],
                                               deltas[i + 1], deltas[i],
                                               layerPrefix));
      } else if (const auto *r = dynamic_cast<const ResidualLayer *>(layer)) {
        if (r->nonLinearityType == NON_LINEARITY_NONE) {
          // Pass deltas directly to previous layer
          deltas[i]=deltas[i+1];
        } else {

          bwdProg.add(bwdNonLinearity(*graph, acts[i + 1], deltas[i + 1],
                                      deltas[i],
                                      r->nonLinearityType,
                                      layerPrefix));
        }

        // record index for earlier layer
        if (residualDeltaIdxs[i - r->resIndex[1]].second != 0)
          throw popnn::popnn_error(
              "Attempting to connect residual bwd pass to layer that is already"
              "connected to a residual");
        residualDeltaIdxs[i - r->resIndex[1]] = {i - r->resIndex[1] + 1,
                                                 i - r->resIndex[0] + 1};
      } else {
        throw popnn::popnn_error("Unrecognized layer type");
      }
    }
  }

  std::cout << "Total number of FLOPs:  "
            << std::right << std::setw(12) << numFlops << "\n";
  std::cout << "Total number of Params: " << std::setw(12) << numParams << "\n";
  std::cout << "Perfect cycle time:     ";
  std::cout << std::setw(12) << static_cast<std::uint64_t>(perfectCycleTime)
            << "\n";
  std::cerr << "Creating engine\n";
  auto trainProg = Sequence();
  if (netType == TrainingNet) {
    if (!options.ignoreData) {
      size_t trainingDataSize = dataSet.numTraining * dataSet.dataSize;
      trainProg.add(Copy(acts[0], &dataSet.trainingData[0],
                         &dataSet.trainingData[trainingDataSize]));
      trainProg.add(Copy(expected, &dataSet.trainingLabels[0],
                         &dataSet.trainingLabels[dataSet.numTraining]));
    }
    trainProg.add(fwdProg);
    trainProg.add(bwdProg);
  }
  auto testProg = Sequence();
  if (!options.ignoreData) {
    size_t testDataSize = dataSet.numTest * dataSet.dataSize;
    testProg.add(Copy(acts[0], &dataSet.testData[0],
                       &dataSet.testData[testDataSize]));
    testProg.add(Copy(expected, &dataSet.testLabels[0],
                      &dataSet.testLabels[dataSet.numTest]));
  }
  testProg.add(fwdProg);
  std::vector<Program> progs(NUM_PROGS);
  progs[INIT_PARAMS_PROG] = std::move(initParamsProg);
  progs[TRAIN_PROG] = std::move(trainProg);
  progs[TEST_PROG] = std::move(testProg);
  engine = std::unique_ptr<Engine>(new Engine(*graph, progs));
}

void Net::run(unsigned numBatches) {
  /* All this method needs to do is set the relevant parameters and
     run the control program. */
  std::cerr << "Running program\n";
  if (options.doComputation) {
    if (netType == TrainingNet) {
      engine->run(INIT_PARAMS_PROG); // initialize params
      for (unsigned i = 0; i < numBatches; i++) {
        if (options.doTestsDuringTraining &&
            i % options.numBatchesBetweenTest == 0) {
          hNumCorrect = 0;
          for (unsigned j = 0; j < numTestBatches; j++) {
            engine->run(TEST_PROG);
          }
          unsigned numTests = (numTestBatches * batchSize);
          float percentCorrect = float(100 * hNumCorrect) / numTests;
          std::cout << "--- Accuracy after " << i << " batches = "
                    << percentCorrect << "%\n";
        }
        engine->run(TRAIN_PROG);
      }
    } else {
      engine->run(INIT_PARAMS_PROG);
      hNumCorrect = 0;
      for (unsigned i = 0; i < numBatches; i++) {
        engine->run(TEST_PROG);
      }
      unsigned numTests = (numTestBatches * batchSize);
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
