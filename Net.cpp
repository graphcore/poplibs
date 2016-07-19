#include "Net.hpp"
#include <boost/program_options.hpp>
#include <poplar/HalfFloat.hpp>
#include "Convolution.hpp"
#include "MaxPool.hpp"
#include "FullyConnected.hpp"
#include "FullyConnectedPlan.hpp"
#include "DeviceInfo.hpp"
#include "ActivationMapping.hpp"
#include "VertexTemplates.hpp"

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
    ("shared-conv-weights",
     po::value<bool>(
       &options.sharedConvWeights
     )->default_value(true),
     "Use of shared weights for convolution instructions")
    ("data-path-width",
     po::value<unsigned>(
       &options.dataPathWidth
     )->default_value(64),
     "Width of the data path in bits")
    ("num-fp16-accum-conv-units",
     po::value<unsigned>(
       &options.fp16AccumConvUnitsPerTile
     )->default_value(8),
     "Number of convolutional units per tile with fp16 accumulation")
    ("num-fp32-accum-conv-units",
     po::value<unsigned>(
       &options.fp32AccumConvUnitsPerTile
     )->default_value(4),
     "Number of convolutional units per tile with fp32 accumulation")
    ("retain-activations",
     po::value<bool>(
       &options.retainActivations
     )->default_value(false),
     "Make sure all activations are retained in memory during the foward pass")
    ("train",
     po::value<bool>(
       &doTraining
     )->default_value(false),
     "Do training (forward, backward and weight update pass)")
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
  }
}


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
  layers(std::move(layers)),
  eta(learningRate),
  dType(getDTypeString(dType))
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
  layers(std::move(layers)),
  eta(learningRate),
  dType(getDTypeString(dType))
{
  initialize(data, lossType);
}

conv::ConvPlan
Net::getConvPlan(unsigned i, unsigned inDimY, unsigned inDimX,
                 unsigned inNumChans) {
  auto it = convPlans.find(i);
  if (it != convPlans.end())
    return it->second;
  const auto *layer = layers[i].get();
  conv::ConvPlan plan;
  if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
    plan = conv::createPlan(inDimY, inDimX, inNumChans,
                            c->kernelSize, c->stride, c->padding,
                            c->numChannels, dType, *deviceInfo);
  } else if (const auto *c = dynamic_cast<const ConvResLayer *>(layer)) {
    plan = conv::createPlan(inDimY, inDimX, inNumChans,
                            c->kernelSize, c->stride, c->padding,
                            c->numChannels, dType, *deviceInfo);
  } else {
    assert(0);
  }
  convPlans.emplace(i, plan);
  return plan;
}


unsigned
Net::getRequiredNumChanGroupsBwd(int i, unsigned inNumChans) {
  if (i < 0)
    return 0;
  const auto *layer = layers[i].get();
  if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
    return 0;
  } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
    auto it = convPlans.find(i);
    assert(it != convPlans.end());
    if (inNumChans == ~0U) {
      inNumChans = acts[i + 1].dim(0) * acts[i + 1].dim(3);
    }
    return inNumChans / it->second.bwdPartition.inChansPerGroup;
  } else if (const auto *c = dynamic_cast<const ConvResLayer *>(layer)) {
    auto it = convPlans.find(i);
    assert(it != convPlans.end());
    if (inNumChans == ~0U) {
      inNumChans = acts[i + 1].dim(0) * acts[i + 1].dim(3);
    }
    return inNumChans / it->second.bwdPartition.inChansPerGroup;
  } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
    return getRequiredNumChanGroupsBwd(i - 1, inNumChans);
  } else {
    assert(0 && "Unrecognized layer type");
  }
}

unsigned
Net::getRequiredNumChanGroupsFwd(unsigned i, unsigned inDimY, unsigned inDimX,
                                 unsigned inNumChans) {
  if (i >= layers.size())
    return 0;
  const auto *layer = layers[i].get();
  if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
    // A fully connected layer wants the channel grouping to be
    // the same forwards and backwards.
    return getRequiredNumChanGroupsBwd(i - 1, inNumChans);
  } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
    auto plan = getConvPlan(i, inDimY, inDimX, inNumChans);
    return inNumChans / plan.fwdPartition.inChansPerGroup;
  } else if (const auto *c = dynamic_cast<const ConvResLayer *>(layer)) {
    auto plan = getConvPlan(i, inDimY, inDimX, inNumChans);
    return inNumChans / plan.fwdPartition.inChansPerGroup;
  } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
    unsigned outDimY, outDimX;
    std::tie(outDimY, outDimX) = maxpool::getOutputDim(inDimY, inDimX,
                                                       m->kernelSize,
                                                       m->stride,
                                                       m->padding);
    return getRequiredNumChanGroupsFwd(i + 1, outDimY, outDimX,
                                                inNumChans);
  } else {
    assert(0 && "Unrecognized layer type");
  }
}

enum {
  INIT_PARAMS_PROG,
  TRAIN_PROG,
  TEST_PROG,
  NUM_PROGS
};

static std::unique_ptr<float[]>
createRandomWeightInitializers(Tensor t, float mean, float variance,
                               std::mt19937 &randomEngine) {
  const auto numWeights = t.numElements();
  auto inits = std::unique_ptr<float[]>(new float[numWeights]);

  std::normal_distribution<> dist(mean, variance);
  for (unsigned i = 0; i < numWeights; ++i)
    inits[i] = dist(randomEngine);

  return inits;
}

void
Net::outputConvDescription(unsigned inDimY, unsigned inDimX,
                           unsigned inNumChans,
                           unsigned kernelSize, unsigned stride,
                           unsigned padding, unsigned outNumChans,
                           bool doResidual) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = conv::getOutputDim(inDimY,
                                                  inDimX,
                                                  kernelSize,
                                                  stride,
                                                  padding);
  const auto numParams =
      kernelSize * kernelSize * inNumChans * outNumChans + outNumChans;
  auto flops = conv::getFlops(inDimY, inDimX, inNumChans,
                              kernelSize, stride, padding, outNumChans,
                              doResidual, netType == TestOnlyNet);
  if (doResidual) {
    std::cout << "   -- Convolutional layer (residual):\n";
  } else {
    std::cout << "   -- Convolutional layer:\n";
  }
  std::cout << "        Size: " << kernelSize << "x" << kernelSize << "\n"
            << "        Stride: " << stride << "\n"
            << "        Padding: " << padding << "\n"
            << "        Input: " << inDimY << "x" << inDimX
            <<   "x" << inNumChans << "\n"
            << "        Output: " << outDimY << "x" << outDimX
            <<   "x" << outNumChans << "\n"
            << "        Params: " << numParams << "\n"
            << "        FLOPs: " << flops << "\n";
}

void Net::outputDescription(const Layer *layer, Tensor in) {
  if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
    const auto prevSize = in.numElements();
    const auto size = fc->size;
    std::cout << "   -- Fully connected layer:\n"
              << "        Input: "  << prevSize << "\n"
              << "        Output: " << size << "\n"
              << "        Params: " << size * (prevSize + 1) << "\n"
              << "        FLOPs: " << fc::getNumFlops(prevSize, size) << "\n";
  } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
    outputConvDescription(in.dim(1), in.dim(2), in.dim(0) * in.dim(3),
                          c->kernelSize, c->stride, c->padding,
                          c->numChannels, false);
  } else if (const auto *c = dynamic_cast<const ConvResLayer *>(layer)) {
    outputConvDescription(in.dim(1), in.dim(2), in.dim(0) * in.dim(3),
                          c->kernelSize, c->stride, c->padding,
                          c->numChannels, true);
  } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
    unsigned outDimY, outDimX;
    std::tie(outDimY, outDimX) = maxpool::getOutputDim(in.dim(1),
                                                       in.dim(2),
                                                       m->kernelSize,
                                                       m->stride,
                                                       m->padding);
    const auto numChannels = in.dim(0) * in.dim(3);
    const auto flops = maxpool::getNumFlops(in.dim(1),
                                            in.dim(2),
                                            numChannels,
                                            m->kernelSize,
                                            m->stride,
                                            m->padding);
    std::cout << "   -- Max pooling layer:\n"
              << "        Size: " << m->kernelSize << "x"
              << m->kernelSize << "\n"
              << "        Stride: " << m->stride << "\n"
              << "        Input: " << in.dim(1) << "x" << in.dim(2)
              <<   "x" << numChannels << "\n"
              << "        Output: " << outDimY << "x" << outDimX
              <<   "x" << numChannels << "\n"
              << "        FLOPs: " << flops << "\n";
  } else {
    assert(0 && "Unrecognized layer type");
  }
}


static ReusableLayer
getOrCreateConvImpl(Graph &graph,
                    IPUModelEngineBuilder::TileMapping &mapping,
                    const DeviceInfo &deviceInfo,
                    const conv::ConvPlan &plan,
                    std::string dType,
                    std::map<ConvImplSpec, ReusableLayer> &impls,
                    ConvImplSpec impl,
                    bool reuseLayers) {
  if (reuseLayers) {
    auto it = impls.find(impl);
    if (it != impls.end()) {
      return it->second;
    }
  }
  auto inChansPerGroups = impl.inNumChans / impl.inNumChanGroups;
  auto in = graph.addTensor(dType, {impl.inNumChanGroups, impl.inDimY,
                                    impl.inDimX, inChansPerGroups},
                            "activations");
  mapActivations(in, mapping, deviceInfo);
  Tensor weights, biases;
  std::tie(weights, biases) =
      conv::createParams(graph, dType, impl.inNumChans,
                         impl.kernelSize, impl.outNumChans, plan);
  auto z = graph.addTensor(dType,
                           {impl.outNumChanGroups,
                            impl.outDimY, impl.outDimX,
                            impl.outNumChans / impl.outNumChanGroups},
                           "z");
  mapActivations(z, mapping, deviceInfo);
  auto out = graph.addTensor(dType,
                           {impl.outNumChanGroups,
                            impl.outDimY, impl.outDimX,
                            impl.outNumChans / impl.outNumChanGroups},
                           "z");
  mapActivations(out, mapping, deviceInfo);
  Tensor residual;
  if (impl.resMethod != RESIDUAL_NONE) {
    residual = graph.addTensor(dType,
                               {impl.resNumChanGroups,
                                impl.resDimY, impl.resDimX,
                                impl.resNumChans / impl.resNumChanGroups});
    mapActivations(residual, mapping, deviceInfo);
  }
  auto prog = conv::convolution(graph, mapping, deviceInfo, plan,
                                impl.kernelSize, impl.stride, impl.padding,
                                impl.outNumChans, impl.nonLinearityType,
                                dType, in, weights, biases, z, out,
                                impl.resMethod, residual);
  auto reusableLayer = ReusableLayer(prog,
                                     {in, weights, biases, residual},
                                     {z, out});
  if (reuseLayers) {
    impls.emplace(impl, reusableLayer);
  }
  return reusableLayer;
}

Program
Net::createConvLayerFwd(unsigned i,
                        unsigned kernelSize, unsigned stride,
                        unsigned padding, unsigned numChannels,
                        NonLinearityType nonLinearityType,
                        unsigned resIndex, ResidualMethod resMethod,
                        Sequence &initParamsProg) {
  auto &in = acts[i];
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) =
      conv::getOutputDim(in.dim(1), in.dim(2), kernelSize, stride, padding);
  auto outNumChanGroups = getRequiredNumChanGroupsFwd(i + 1,
                                                      outDimY, outDimX,
                                                      numChannels);
  if (!outNumChanGroups) {
    auto chansPerGroup = (dType == "float") ? 1 : 2;
    outNumChanGroups = numChannels / chansPerGroup;
  }
  acts[i + 1] = graph->addTensor(dType,
                                 {outNumChanGroups,
                                  outDimY, outDimX,
                                  numChannels / outNumChanGroups},
                                 "activations." + std::to_string(i));
  mapActivations(acts[i + 1], *mapping, *deviceInfo);
  z[i + 1] = graph->addTensor(dType,
                              {outNumChanGroups,
                               outDimY, outDimX,
                               numChannels / outNumChanGroups},
                               "z." + std::to_string(i));
  mapActivations(z[i + 1], *mapping, *deviceInfo);
  unsigned inNumChans = in.dim(0) * in.dim(3);
  unsigned inNumChanGroups = in.dim(0);
  unsigned inDimY = in.dim(1), inDimX = in.dim(2);
  auto plan = getConvPlan(i, inDimY, inDimX, inNumChans);
  Tensor weights, biases;
  std::tie(weights, biases) =
      conv::createParams(*graph, dType, inNumChans,
                         kernelSize, numChannels, plan);
  params[i].push_back(weights);
  params[i].push_back(biases);
  Tensor residual;
  unsigned resDimY = 0, resDimX = 0, resNumChans = 0, resNumChanGroups = 0;
  if (resMethod != RESIDUAL_NONE) {
    residual = acts[i - resIndex + 1];
    resDimY = residual.dim(1);
    resDimX = residual.dim(2);
    resNumChans = residual.dim(0) * residual.dim(3);
    resNumChanGroups = residual.dim(0);
  }
  ReusableLayer reusableLayer =
      getOrCreateConvImpl(*graph, *mapping, *deviceInfo, plan, dType,
                          convImpls,
                          {inNumChans, inNumChanGroups, inDimX, inDimY,
                           numChannels, outNumChanGroups, outDimX,
                           outDimY,
                           resNumChans, resNumChanGroups, resDimX, resDimY,
                           kernelSize, stride, padding,
                           nonLinearityType, resMethod},
                          options.reuseLayerImplGraphs);
 conv::mapWeights(weights, *mapping, *deviceInfo, plan);
 conv::mapBiases(biases, *mapping, *deviceInfo, acts[i + 1]);
 if (dType == "float") {
    auto hWeights =
        createRandomWeightInitializers(weights, 0, 1.0 / kernelSize,
                                       randomEngine);
    auto hBiases =
        createRandomWeightInitializers(weights, 0, 1.0 / kernelSize,
                                       randomEngine);
    initParamsProg.add(Copy(weights, hWeights.get()));
    initParamsProg.add(Copy(biases, hBiases.get()));
    hParams.push_back(std::move(hWeights));
    hParams.push_back(std::move(hBiases));
 }
 numFlops += conv::getFlops(inDimY, inDimX, inNumChans, kernelSize,
                            stride, padding, numChannels,
                            resMethod != RESIDUAL_NONE,
                            netType == TestOnlyNet && i != 0);
 perfectCycleTime += conv::getPerfectCycleCount(*deviceInfo, dType, inDimY,
                                                inDimX, inNumChans, kernelSize,
                                                stride, padding, numChannels,
                                                resMethod != RESIDUAL_NONE,
                                                netType == TestOnlyNet && i != 0);
 auto prog = Sequence();
 prog.add(Copy(reusableLayer.inputs[0], in));
 prog.add(Copy(reusableLayer.inputs[1], weights));
 prog.add(Copy(reusableLayer.inputs[2], biases));
 if (resMethod != RESIDUAL_NONE)
   prog.add(Copy(reusableLayer.inputs[3], residual));
 prog.add(reusableLayer.prog);
 prog.add(Copy(z[i + 1], reusableLayer.outputs[0]));
 prog.add(Copy(acts[i + 1], reusableLayer.outputs[1]));
 return prog;
}

void Net::initialize(DataSet &dataSet, LossType lossType) {
  assert(batchSize == 1 && "Only batch size of 1 is supported");
  unsigned inputSize = dataSet.dataSize;
  numTestBatches = dataSet.numTest / batchSize;
  env = std::unique_ptr<GraphProgEnv>(
    new GraphProgEnv("obj/neural_net_graph.gp", GraphProgFileType::Object));

  graph = std::unique_ptr<Graph>(new Graph(*env));
  mapping = std::unique_ptr<IPUModelEngineBuilder::TileMapping>(
    new IPUModelEngineBuilder::TileMapping(*graph)
  );
  IPUModelEngineBuilder ipuEB(*env);
  if (options.useIPUModel) {
    ipuEB.setMemcpyBytesPerCycle(options.dataPathWidth / 8);
    ipuEB.setNumIPUs(options.numIPUs);
    ipuEB.setTilesPerIPU(options.tilesPerIPU);
    ipuEB.setNumBytesPerTile(options.memoryBytesPerTile);
    ipuEB.setIPUExchangeBandwidth(options.ipuExchangeBandwidth);
    ipuEB.setIPUExchangeImplementation(
        IPUModelEngineBuilder::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST
    );
    ipuEB.setGlobalSyncCycles(500);
    switch (ipuEB.getNumIPUs()) {
    case 1:
      break;
    case 2:
      ipuEB.setGlobalExchangeConstraints({
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
  } else {
    ipuEB.setNumIPUs(1);
    ipuEB.setTilesPerIPU(1);
    ipuEB.setNumWorkerContexts(1);
  }
  deviceInfo = std::unique_ptr<DeviceInfo>(
        new DeviceInfo(ipuEB, options.dataPathWidth,
                       options.convUnitPipelineDepth,
                       options.fp16AccumConvUnitsPerTile,
                       options.fp32AccumConvUnitsPerTile));
  std::cerr << "Constructing program\n";
  numFlops = 0;
  perfectCycleTime = 0;
  auto initParamsProg = Sequence();
  auto fwdProg = Sequence();
  auto bwdProg = Sequence();
  auto numChanGroups =
      getRequiredNumChanGroupsFwd(0, dataSet.dim[0], dataSet.dim[1],
                                  dataSet.dim[2]);
  if (numChanGroups == 0)
    numChanGroups = 1;
  const auto dim = std::vector<size_t>({numChanGroups,
                                        dataSet.dim[0], dataSet.dim[1],
                                        dataSet.dim[2] / numChanGroups});
  acts.resize(layers.size() + 1);
  z.resize(layers.size() + 1);
  deltas.resize(layers.size() + 1);
  params.resize(layers.size());
  acts[0] = graph->addTensor(dType, dim, "input");
  mapActivations(acts[0], *mapping, *deviceInfo);
  for (unsigned i = 0; i < layers.size(); ++i) {
    const auto *layer = layers[i].get();
    std::cout << "-- Layer " << i << "\n";
    outputDescription(layer, acts[i]);
    if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
      const auto prevSize = acts[i].numElements();
      const auto size = fc->size;
      acts[i + 1] = graph->addTensor(dType, {size},
                                 "activations." + std::to_string(i));
      mapActivations(acts[i + 1], *mapping, *deviceInfo);
      z[i + 1] = graph->addTensor(dType, {size}, "z." + std::to_string(i));
      mapActivations(z[i + 1], *mapping, *deviceInfo);
      auto activationsMapping =
          computeActivationsMapping(z[i + 1], *deviceInfo);
      bool forwardOnly = i == 0 || netType == TestOnlyNet;
      const auto &plan =
          fullyConnectedPlan.emplace(
            i,
            fc::createPlan(*deviceInfo, dType, prevSize,
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
      fwdProg.add(fc::fullyConnected(*graph, *mapping, *deviceInfo,
                                     size, fc->nonLinearityType, dType,
                                     acts[i], weights, biases, z[i + 1],
                                     acts[i + 1], plan));
      numFlops += fc::getNumFlops(prevSize, size);
      perfectCycleTime += fc::getPerfectCycleCount(*deviceInfo, prevSize,
                                                   size, dType);
    } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
      fwdProg.add(createConvLayerFwd(i, c->kernelSize, c->stride, c->padding,
                                     c->numChannels, c->nonLinearityType,
                                     0, RESIDUAL_NONE, initParamsProg));
    } else if (const auto *c = dynamic_cast<const ConvResLayer *>(layer)) {
      fwdProg.add(createConvLayerFwd(i, c->kernelSize, c->stride, c->padding,
                                     c->numChannels, c->nonLinearityType,
                                     c->resIndex, c->resMethod,
                                     initParamsProg));
    } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
      const auto &in = acts[i];
      unsigned outDimY, outDimX;
      std::tie(outDimY, outDimX) = maxpool::getOutputDim(in.dim(1),
                                                         in.dim(2),
                                                         m->kernelSize,
                                                         m->stride,
                                                         m->padding);
      acts[i + 1] = graph->addTensor(dType,
                                     {in.dim(0),
                                      outDimY, outDimX,
                                      in.dim(3)},
                                     "activations." + std::to_string(i));
      mapActivations(acts[i], *mapping, *deviceInfo);
      fwdProg.add(maxpool::maxPool(*graph, *mapping, *deviceInfo,
                                   m->kernelSize, m->stride, m->padding,
                                   dType, acts[i], acts[i + 1]));
      numFlops += maxpool::getNumFlops(in.dim(1), in.dim(2),
                                       in.dim(0) * in.dim(3), m->kernelSize,
                                       m->stride, m->padding);
      perfectCycleTime +=
          maxpool::getPerfectCycleCount(*deviceInfo, dType,
                                        in.dim(1), in.dim(2),
                                        in.dim(0) * in.dim(3), m->kernelSize,
                                        m->stride, m->padding);
    } else {
      assert(0 && "Unrecognized layer type");
    }
  }
  auto lossCS = graph->createComputeSet("LossLayer");
  auto lastAct = *(acts.end() - 1);
  Tensor expected = graph->addTensor("unsigned", {1});
  Tensor numCorrect = graph->addTensor("unsigned", {1});
  Tensor loss = graph->addTensor(dType, {1});
  deltas[layers.size()] = graph->addTensor(dType, lastAct.dims());
  mapActivations(deltas[layers.size()], *mapping, *deviceInfo);
  auto v = graph->addVertex(lossCS, templateVertex("CalcLoss", dType),
                           {{"in", lastAct.flatten()},
                            {"deltaOut", deltas[layers.size()].flatten()},
                            {"label", expected[0]},
                            {"loss", loss[0]},
                            {"numCorrect", numCorrect[0]}});
  mapping->setMapping(v, 0);
  graph->setFieldSize(v["probs"], lastAct.numElements());
  graph->setInitialValue(v["lossType"], lossType);;
  fwdProg.add(Sequence(Copy(numCorrect, &hNumCorrect),
                       Execute(lossCS),
                       Copy(&hNumCorrect, numCorrect)));
  if (options.retainActivations) {
    size_t maxActSize = 0;
    size_t maxElemSize = std::max(sizeof(float), sizeof(half));
    for (const auto &act : acts) {
      maxActSize = std::max(maxActSize, act.numElements());
    }
    hAct = std::unique_ptr<char[]>(new char[maxActSize * maxElemSize]);
    for (const auto &act : acts) {
      fwdProg.add(Copy(&hAct[0], act));
    }
  }
  if (netType == TrainingNet) {
    for (int i = layers.size() - 1; i >= 0; --i) {
      bool backwardPassRequired = (i != 0);
      if (backwardPassRequired) {
        if (acts[i].dims().size() == 4) {
          auto numChannels = acts[i].dim(0) * acts[i].dim(3);
          auto numChanGroups = getRequiredNumChanGroupsBwd(i - 1);
          if (numChanGroups == 0)
            numChanGroups = 1;
          auto chansPerGroup = numChannels / numChanGroups;
          deltas[i] = graph->addTensor(dType, {numChanGroups,
                                               acts[i].dim(1),
                                               acts[i].dim(2),
                                               chansPerGroup} , "deltas");
         } else {
           assert(acts[i].dims().size() == 1);
           deltas[i] = graph->addTensor(dType, acts[i].dims(), "deltas");
        }
        mapActivations(deltas[i], *mapping, *deviceInfo);
      }
      const auto *layer = layers[i].get();
      if (const auto *fc = dynamic_cast<const FullyConnectedLayer *>(layer)) {
        Tensor zDeltas = graph->addTensor(dType, deltas[i + 1].dims(),
                                          "zDeltas");

        mapActivations(zDeltas, *mapping, *deviceInfo);
        auto weights = params[i][0];
        auto biases = params[i][1];
        const auto &plan = fullyConnectedPlan.find(i)->second;
        bwdProg.add(fc::fullyConnectedBwdNonLinearity(*graph, *mapping,
                                                      *deviceInfo, dType,
                                                      z[i + 1], deltas[i + 1],
                                                      zDeltas,
                                                      fc->nonLinearityType,
                                                      plan));

        if (backwardPassRequired)
          bwdProg.add(fc::fullyConnectedBackward(*graph, *mapping, *deviceInfo,
                                                 dType, zDeltas,
                                                 weights, deltas[i],
                                                 plan));
        bwdProg.add(fc::fullyConnectedWeightUpdate(*graph, *mapping,
                                                   *deviceInfo, dType, zDeltas,
                                                   acts[i], weights, biases,
                                                   eta, plan));
      } else if (const auto *c = dynamic_cast<const ConvLayer *>(layer)) {
        auto it = convPlans.find(i);
        assert(it != convPlans.end());
        auto &plan = it->second;
        Tensor zDeltas = graph->addTensor(dType, deltas[i + 1].dims(),
                                          "zDeltas");
        mapActivations(zDeltas, *mapping, *deviceInfo);
        auto weights = params[i][0];
        auto biases = params[i][1];
        bwdProg.add(conv::convolutionBwdNonLinearity(*graph, *mapping,
                                                     *deviceInfo, dType,
                                                     deltas[i + 1], z[i + 1],
                                                     zDeltas,
                                                     c->nonLinearityType));

        if (backwardPassRequired)
          bwdProg.add(conv::convolutionBackward(*graph, *mapping, *deviceInfo,
                                                plan, dType, zDeltas, weights,
                                                deltas[i], c->kernelSize,
                                                c->stride, c->padding));
        bwdProg.add(
              conv::convolutionWeightUpdate(*graph, *mapping, *deviceInfo,
                                            dType, zDeltas, weights, biases,
                                            acts[i], c->kernelSize,
                                            c->stride, c->padding, eta));
      } else if (const auto *c = dynamic_cast<const ConvResLayer *>(layer)) {
        // TODO residuals
        auto it = convPlans.find(i);
        assert(it != convPlans.end());
        auto &plan = it->second;
        Tensor zDeltas = graph->addTensor(dType, deltas[i + 1].dims(),
                                          "zDeltas");
        mapActivations(zDeltas, *mapping, *deviceInfo);
        auto weights = params[i][0];
        auto biases = params[i][1];
        bwdProg.add(conv::convolutionBwdNonLinearity(*graph, *mapping,
                                                     *deviceInfo, dType,
                                                     deltas[i + 1], z[i + 1],
                                                     zDeltas,
                                                     c->nonLinearityType));
        if (backwardPassRequired)
          bwdProg.add(conv::convolutionBackward(*graph, *mapping, *deviceInfo,
                                                plan, dType, zDeltas, weights,
                                                deltas[i], c->kernelSize,
                                                c->stride, c->padding));
        bwdProg.add(
              conv::convolutionWeightUpdate(*graph, *mapping, *deviceInfo,
                                            dType, zDeltas, weights, biases,
                                            acts[i], c->kernelSize,
                                            c->stride, c->padding, eta));
      } else if (const auto *m = dynamic_cast<const MaxPoolLayer *>(layer)) {
        if (backwardPassRequired)
          bwdProg.add(maxpool::maxPoolBackward(*graph, *mapping, *deviceInfo,
                                               m->kernelSize, m->stride,
                                               m->padding, dType,
                                               acts[i], acts[i + 1],
                                               deltas[i + 1], deltas[i]));
      } else {
        assert(0 && "Unrecognized layer type");
      }
    }
  }
  std::cout << "Total number of FLOPs: " << numFlops << "\n";
  std::cout << "Perfect cycle time: ";
  std::cout << static_cast<std::uint64_t>(perfectCycleTime) << "\n";
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
  std::vector<const Program *> progs(NUM_PROGS);
  progs[INIT_PARAMS_PROG] = &initParamsProg;
  progs[TRAIN_PROG] = &trainProg;
  progs[TEST_PROG] = &testProg;
  if (options.useIPUModel) {
    IPUModelEngineBuilder::UserTilePartitioner p(*mapping);
    ipuEB.setTilePartitioner(p);
    engine = ipuEB.makeEngine(*graph, progs);
  } else {
    CPUEngineBuilder cpuEB(*env);
    engine = cpuEB.makeEngine(*graph, progs);
  }
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
    IPUModelEngine *ipuEngine = static_cast<IPUModelEngine *>(&*engine);
    IPUModelEngine::ReportOptions opt;
    opt.doLayerWiseProfile = true;
    ipuEngine->report(std::cout, opt);
  }
}
