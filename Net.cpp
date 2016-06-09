#include "Net.hpp"
#include <boost/program_options.hpp>

bool parseCommandLine(int argc, char **argv, NetOptions &options) {
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("ipus", po::value<unsigned>(&options.numIPUs)->default_value(1),
             "Number of IPUs")
    ("graph-reuse",
     po::value<bool>(&options.reuseLayerImplGraphs)->default_value(true),
     "Re-use graph structure for similar layers")
    ("shared-conv-weights",
     po::value<bool>(
       &options.ipuMachineInfo.sharedConvWeights
     )->default_value(true),
     "Use of shared weights for convolution instructions")
    ("num-conv-units",
     po::value<unsigned>(
       &options.ipuMachineInfo.numConvUnitsPerTile
     )->default_value(4),
     "Number of convolutional units per tile")
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

void Net::initialize(DataSet &data, LossType lossType) {
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
    ipuEB->setNumIPUs(options.numIPUs);
    ipuEB->setIPUExchangeImplementation(
      IPUModelEngineBuilder::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST
    );
    ipuEB->setGlobalSyncCycles(500);
  } else {
    engineBuilder =
      std::unique_ptr<EngineBuilder>(new CPUEngineBuilder(*env));
  }
  EngineBuilder &eb = *engineBuilder;

  std::cerr << "Constructing program\n";
  Tensor isTraining = graph->addTensor("unsigned", {1});
  if (mapping) {
    mapping->setMapping(isTraining, 0);
  }
  inputLayer = std::unique_ptr<InputLayer>(
    new InputLayer(*this, -1, data, isTraining));
  lossLayer = std::unique_ptr<LossLayer>(
    new LossLayer(*this, hiddenLayerSpecs.size(), data, lossType,
                  isTraining));

  for (size_t i = 0; i < hiddenLayerSpecs.size(); ++i) {
    hiddenLayers.push_back(hiddenLayerSpecs[i]->makeLayer(*this, i));
  }

  auto initParamsProg = Sequence();
  auto startBatchProg = Sequence();
  auto fwdProg = Sequence();
  auto bwdProg = Sequence();
  auto weightSyncProg = Sequence();

  inputLayer->init(*graph, mapping.get());
  startBatchProg.add(inputLayer->startBatch(*graph));
  fwdProg.add(inputLayer->forward(*graph, mapping.get()));

  initParamsProg.add(inputLayer->initParams(*graph));

  std::uint64_t numFlops = 0;
  double perfectCycleTime = 0.0;
  for (unsigned i = 0; i < hiddenLayers.size(); ++i) {
    hiddenLayers[i]->init(*graph, mapping.get());
    startBatchProg.add(hiddenLayers[i]->startBatch(*graph));
    fwdProg.add(hiddenLayers[i]->forward(*graph, mapping.get()));
    initParamsProg.add(hiddenLayers[i]->initParams(*graph));
    std::cout << "-- Layer " << i << "\n";
    hiddenLayers[i]->describe(std::cout);
    numFlops += hiddenLayers[i]->getNumberOfFlops();
    perfectCycleTime += hiddenLayers[i]->getPerfectCycleCount();
  }
  std::cout << "Total number of FLOPs: " << numFlops << "\n";
  std::cout << "Perfect cycle time: ";
  std::cout << static_cast<std::uint64_t>(perfectCycleTime) << "\n";

  lossLayer->init(*graph, mapping.get());
  startBatchProg.add(lossLayer->startBatch(*graph));
  fwdProg.add(lossLayer->forward(*graph, mapping.get()));
  initParamsProg.add(lossLayer->initParams(*graph));

  if (netType == TrainingNet) {
    bwdProg.add(lossLayer->backward(*graph));
    weightSyncProg.add(lossLayer->weightSync(*graph));
    for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
      bwdProg.add(hiddenLayers[i]->backward(*graph));
      weightSyncProg.add(hiddenLayers[i]->weightSync(*graph));
    }
    bwdProg.add(inputLayer->backward(*graph));
    weightSyncProg.add(inputLayer->weightSync(*graph));
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

void Net::run(unsigned numBatches) {
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
    IPUModelEngine::ReportOptions opt;
    opt.doLayerWiseProfile = true;
    ipuEngine->report(std::cout, opt);
  }
}
