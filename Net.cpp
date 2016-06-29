#include "Net.hpp"
#include <boost/program_options.hpp>
#include <poplar/HalfFloat.hpp>

bool parseCommandLine(int argc, char **argv, NetOptions &options) {
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
       &options.ipuMachineInfo.sharedConvWeights
     )->default_value(true),
     "Use of shared weights for convolution instructions")
    ("data-path-width",
     po::value<unsigned>(
       &options.ipuMachineInfo.dataPathWidth
     )->default_value(64),
     "Width of the data path in bits")
    ("num-fp16-accum-conv-units",
     po::value<unsigned>(
       &options.ipuMachineInfo.fp16AccumConvUnitsPerTile
     )->default_value(8),
     "Number of convolutional units per tile with fp16 accumulation")
    ("num-fp32-accum-conv-units",
     po::value<unsigned>(
       &options.ipuMachineInfo.fp32AccumConvUnitsPerTile
     )->default_value(4),
     "Number of convolutional units per tile with fp32 accumulation")
    ("retain-activations",
     po::value<bool>(
       &options.retainActivations
     )->default_value(false),
     "Make sure all activations are retained in memory during the foward pass")
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
         std::vector<std::unique_ptr<LayerSpec>> &hiddenLayerSpecs,
         LossType lossType,
         float learningRate,
         NetType netType,
         DType dType,
         NetOptions options) :
  netType(netType), options(options),
  batchSize(batchSize),
  hiddenLayerSpecs(std::move(hiddenLayerSpecs)),
  eta(learningRate),
  dType(getDTypeString(dType))
{
  initialize(data, lossType);
}

Net::Net(DataSet &data, unsigned batchSize,
         std::vector<std::unique_ptr<LayerSpec>> &&hiddenLayerSpecs,
         LossType lossType,
         float learningRate,
         NetType netType,
         DType dType,
         NetOptions options) :
  netType(netType), options(options),
  batchSize(batchSize),
  hiddenLayerSpecs(std::move(hiddenLayerSpecs)),
  eta(learningRate),
  dType(getDTypeString(dType))
{
  initialize(data, lossType);
}

enum {
  INIT_PARAMS_PROG,
  TRAIN_PROG,
  TEST_PROG,
  NUM_PROGS
};

void Net::initialize(DataSet &data, LossType lossType) {
  assert(batchSize == 1 && "Only batch size of 1 is supported");
  unsigned inputSize = data.dataSize;
  numTestBatches = data.numTest / batchSize;
  env = std::unique_ptr<GraphProgEnv>(
    new GraphProgEnv("obj/neural_net_graph.ppo", GraphProgFileType::Object));

  graph = std::unique_ptr<Graph>(new Graph(*env));
  std::unique_ptr<IPUModelEngineBuilder::TileMapping> mapping;
  if (options.useIPUModel) {
    mapping.reset(new IPUModelEngineBuilder::TileMapping(*graph));
    IPUModelEngineBuilder *ipuEB = new IPUModelEngineBuilder(*env);
    engineBuilder = std::unique_ptr<EngineBuilder>(ipuEB);
    ipuEB->setMemcpyBytesPerCycle(options.ipuMachineInfo.dataPathWidth / 8);
    ipuEB->setNumIPUs(options.numIPUs);
    ipuEB->setTilesPerIPU(options.tilesPerIPU);
    ipuEB->setNumBytesPerTile(options.memoryBytesPerTile);
    ipuEB->setIPUExchangeBandwidth(options.ipuExchangeBandwidth);
    ipuEB->setIPUExchangeImplementation(
      IPUModelEngineBuilder::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST
    );
    ipuEB->setGlobalSyncCycles(500);
  } else {
    engineBuilder =
        std::unique_ptr<EngineBuilder>(new CPUEngineBuilder(*env));
    dummyIpuEngineBuilder =
        std::unique_ptr<IPUModelEngineBuilder>(new IPUModelEngineBuilder(*env));
    dummyIpuEngineBuilder->setNumIPUs(1);
    dummyIpuEngineBuilder->setTilesPerIPU(1);
    dummyIpuEngineBuilder->setNumWorkerContexts(1);
  }
  EngineBuilder &eb = *engineBuilder;

  std::cerr << "Constructing program\n";
  inputLayer = std::unique_ptr<InputLayer>(
    new InputLayer(*this, -1, data));
  lossLayer = std::unique_ptr<LossLayer>(
    new LossLayer(*this, hiddenLayerSpecs.size(), data, lossType));

  for (size_t i = 0; i < hiddenLayerSpecs.size(); ++i) {
    hiddenLayers.push_back(hiddenLayerSpecs[i]->makeLayer(*this, i));
  }

  auto initParamsProg = Sequence();
  auto fwdProg = Sequence();
  auto bwdProg = Sequence();
  auto weightUpdateProg = Sequence();
  std::vector<Tensor> acts;

  std::mt19937 randomEngine;
  inputLayer->init(*graph, randomEngine, mapping.get());

  fwdProg.add(inputLayer->forward(*graph, mapping.get()));
  acts.push_back(inputLayer->getFwdActivations());

  initParamsProg.add(inputLayer->initParams(*graph));

  std::uint64_t numFlops = 0;
  double perfectCycleTime = 0.0;

  for (unsigned i = 0; i < hiddenLayers.size(); ++i) {
    hiddenLayers[i]->init(*graph, randomEngine, mapping.get());
    fwdProg.add(hiddenLayers[i]->forward(*graph, mapping.get()));
    acts.push_back(hiddenLayers[i]->getFwdActivations());
    initParamsProg.add(hiddenLayers[i]->initParams(*graph));
    std::cout << "-- Layer " << i << "\n";
    hiddenLayers[i]->describe(std::cout);
    numFlops += hiddenLayers[i]->getNumberOfFlops();
    perfectCycleTime += hiddenLayers[i]->getPerfectCycleCount();
  }
  std::cout << "Total number of FLOPs: " << numFlops << "\n";
  std::cout << "Perfect cycle time: ";
  std::cout << static_cast<std::uint64_t>(perfectCycleTime) << "\n";

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

  lossLayer->init(*graph, randomEngine, mapping.get());
  fwdProg.add(lossLayer->forward(*graph, mapping.get()));
  initParamsProg.add(lossLayer->initParams(*graph));

  if (netType == TrainingNet) {
    bwdProg.add(lossLayer->backward(*graph));
    weightUpdateProg.add(lossLayer->weightUpdate(*graph));
    for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
      bwdProg.add(hiddenLayers[i]->backward(*graph));
      weightUpdateProg.add(hiddenLayers[i]->weightUpdate(*graph));
    }
    bwdProg.add(inputLayer->backward(*graph));
    weightUpdateProg.add(inputLayer->weightUpdate(*graph));
  }

  if (options.useIPUModel) {
    IPUModelEngineBuilder *ipuEB =
      static_cast<IPUModelEngineBuilder *>(&eb);
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

  std::cerr << "Creating engine\n";
  auto trainProg = Sequence();
  if (netType == TrainingNet) {
    if (!options.ignoreData) {
      trainProg.add(inputLayer->loadData(*graph, true));
      trainProg.add(lossLayer->loadLabels(*graph, true));
    }
    trainProg.add(fwdProg);
    trainProg.add(bwdProg);
    trainProg.add(weightUpdateProg);
  }
  auto testProg = Sequence();
  if (!options.ignoreData) {
    testProg.add(inputLayer->loadData(*graph, false));
    testProg.add(lossLayer->loadLabels(*graph, false));
  }
  testProg.add(fwdProg);
  std::vector<const Program *> progs(NUM_PROGS);
  progs[INIT_PARAMS_PROG] = &initParamsProg;
  progs[TRAIN_PROG] = &trainProg;
  progs[TEST_PROG] = &testProg;
  engine = eb.makeEngine(*graph, progs);
}

void Net::run(unsigned numBatches) {
  /* All this method needs to do is set the relevant parameters and
     run the control program. */
  std::cerr << "Running program\n";
  if (options.doComputation) {
    if (netType == TrainingNet) {
      engine->run(INIT_PARAMS_PROG); // initialize params
      for (unsigned i = 0; i < numBatches; i++) {
        if (!options.singleBatchProfile &&
            i % options.numBatchesBetweenTest == 0) {
          lossLayer->resetNumCorrect();
          for (unsigned j = 0; j < numTestBatches; j++) {
            engine->run(TEST_PROG);
          }
          float numCorrect = lossLayer->getNumCorrect();
          unsigned numTests = (numTestBatches * batchSize);
          float percentCorrect = 100 * numCorrect / numTests;
          std::cout << "--- Accuracy after " << i << " batches = "
                    << percentCorrect << "%\n";
        }
        engine->run(TRAIN_PROG);
      }
    } else {
      engine->run(INIT_PARAMS_PROG);
      lossLayer->resetNumCorrect();
      for (unsigned i = 0; i < numBatches; i++) {
        engine->run(TEST_PROG);
      }
      float numCorrect = lossLayer->getNumCorrect();
      unsigned numTests = (numTestBatches * batchSize);
      float percentCorrect = 100 * numCorrect / numTests;
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
