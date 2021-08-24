// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <fstream>
#include <istream>
#include <ostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Lstm.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/Lstm.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/codelets.hpp>
#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popnn;
using namespace poplibs_support;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.1
#define HALF_REL_TOL 0.3
#define FLOAT_ABS_TOL 1e-5
#define HALF_ABS_TOL 7e-2

const OptionFlags defaultEngineOptions;

std::ostream &operator<<(std::ostream &os, const BasicLstmCellUnit u) {
  switch (u) {
  case BASIC_LSTM_CELL_FORGET_GATE:
    return os << "forget";
  case BASIC_LSTM_CELL_INPUT_GATE:
    return os << "input";
  case BASIC_LSTM_CELL_CANDIDATE:
    return os << "cell";
  case BASIC_LSTM_CELL_OUTPUT_GATE:
    return os << "output";
  case BASIC_LSTM_CELL_NUM_UNITS:
    break;
  }

  throw poputil::poplibs_error("Invalid unit");
}

std::istream &operator>>(std::istream &is, BasicLstmCellUnit &u) {
  std::string token;
  is >> token;

  if (token == "forget") {
    u = BASIC_LSTM_CELL_FORGET_GATE;
  } else if (token == "input") {
    u = BASIC_LSTM_CELL_INPUT_GATE;
  } else if (token == "cell") {
    u = BASIC_LSTM_CELL_CANDIDATE;
  } else if (token == "output") {
    u = BASIC_LSTM_CELL_OUTPUT_GATE;
  } else {
    throw poputil::poplibs_error("Invalid token for unit: " + token);
  }

  return is;
}

std::vector<BasicLstmCellUnit>
getCellOrder(const std::vector<std::string> &in) {
  std::vector<BasicLstmCellUnit> cellOrder;
  for (const auto &x : in) {
    cellOrder.emplace_back();

    std::stringstream ss(x);
    ss >> cellOrder.back();
  }

  return cellOrder;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  DeviceType deviceType = DeviceType::IpuModel2;

  unsigned sequenceSize, inputSize, outputSize;
  unsigned batchSize = 1;
  unsigned stepsPerWU = 1;
  double weightUpdateMemoryProportion = 0.0;
  bool disableWUPartialInterleaving = false;
  unsigned numShards = 1;
  bool codeReuse = false;

  Type dataType;
  Type partialsType;
  Type accumulatorsType;
  double relativeTolerance;
  double absoluteTolerance;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  bool outputAllSequence = true;
  bool preweightInput = false;
  bool ignoreFinalState = false;
  poplibs_test::Pass pass = poplibs_test::Pass::FWD;
  std::string recompMode;
  unsigned runs = 1;
  std::string profileDir = ".";
  double availableMemoryProportion;
  ShapeOption<std::size_t> variableTimeStepsOption;
  ShapeOption<std::string> cellOrder;
  popnn::NonLinearityType activation = popnn::NonLinearityType::TANH;
  popnn::NonLinearityType recurrentActivation =
      popnn::NonLinearityType::SIGMOID;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       deviceTypeHelp)
    ("profile", "Output profiling report")
    ("profile-dir",
      po::value<std::string>(&profileDir)->default_value(profileDir),
      "The directory to output profiling report")
    ("sequence-size", po::value<unsigned>(&sequenceSize)->required(),
     "Sequence size in the RNN")
    ("variable-time-steps",
     po::value<ShapeOption<std::size_t>>(&variableTimeStepsOption),
     "Variable time steps could either be a scalar value or tensor of "
     "batch-size length")
    ("shards", po::value<unsigned>(&numShards),
     "The number of shards")
    ("code-reuse",
     po::value<bool>(&codeReuse),
     "Force LSTM code reuse")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs in each element in the sequence")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of outputs in each element in the sequence")
    ("data-type",
      po::value<Type>(&dataType)->default_value(HALF),
      "Input and output data type")
    ("batch-size", po::value<unsigned>(&batchSize)->default_value(batchSize),
      "Batch size")
    ("steps-per-wu", po::value<unsigned>(&stepsPerWU),
      "Steps per Weight Update")
    ("wu-memory-proportion", po::value<double>(&weightUpdateMemoryProportion),
     "What percentage of memory is available to the operation for LSTM "
     "temporary use")
    ("disable-wu-partial-interleaving", po::value<bool>(&disableWUPartialInterleaving),
      "Limit WU interleaving to either full or none")
    ("partials-type",
     po::value<Type>(&partialsType),
     "Type of the partials")
    ("accumulators-type",
     po::value<Type>(&accumulatorsType),
     "Type of the partials")
    ("rel-tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("abs-tolerance",po::value<double>(&absoluteTolerance),
     "Absolute tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&numIPUs)->default_value(numIPUs),
     "Number of IPUs")
    ("output-all-sequence",
       po::value<bool>(&outputAllSequence)->default_value(outputAllSequence),
     "output the data from all cells (1 / 0)")
    ("ignore-final-state",
       po::value<bool>(&ignoreFinalState)->default_value(ignoreFinalState),
     "use new lstmFwd() API that ignores final cell state")
    ("pre-weight-input",
       po::value<bool>(&preweightInput)->default_value(preweightInput),
     "Pre-weight whole sequence before recursive part is computed (0 / 1)")
      ("phase",
     po::value<poplibs_test::Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("recomputation-mode",
     po::value<std::string>(&recompMode),
     "Recomputation mode none | cellAndTanh")

    // This option can be used to exercise certain alternate Bwd-pass APIs
    ("ignore-input-gradient",
     "Don't provide a tensor for input gradients to bwd pass")
    ("ignore-data",
     "Don't perform host-to-device or vice versa transfers (no validation)")
    ("runs", po::value<unsigned>(&runs)->default_value(runs),
     "Number of calls to Engine::run")
    ("available-memory-proportion",
     po::value<double>(&availableMemoryProportion),
     "What percentage of memory is available to the operation for temporary "
     "use")
    ("cell-order",
     po::value<ShapeOption<std::string>>(&cellOrder)->default_value(cellOrder),
     "The order that the gates are stored in the weights and bias tensors")
    ("activation",
     po::value<popnn::NonLinearityType>(&activation)->default_value(activation),
     "Activation function for LSTM")
    ("recurrent-activation",
     po::value<popnn::NonLinearityType>(&recurrentActivation)->default_value(recurrentActivation),
     "Recurrent activation function for LSTM")
  ;
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (vm["rel-tolerance"].empty()) {
    if (dataType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }

  if (vm["abs-tolerance"].empty()) {
    if (dataType == FLOAT) {
      absoluteTolerance = FLOAT_ABS_TOL;
    } else {
      absoluteTolerance = HALF_ABS_TOL;
    }
  }

  bool ignoreInputGradient = vm.count("ignore-input-gradient");

  bool ignoreData = vm.count("ignore-data");

  auto device = tilesPerIPU
                    ? createTestDevice(deviceType, numIPUs, *tilesPerIPU)
                    : createTestDeviceFullSize(deviceType, numIPUs);

  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);
  // Bwd pass is always run if WU is run. This may change is tensors input to
  //  WU are created on host
  bool doBwdPass = pass == poplibs_test::Pass::ALL ||
                   pass == poplibs_test::Pass::BWD ||
                   pass == poplibs_test::Pass::WU;
  bool doWuPass =
      pass == poplibs_test::Pass::ALL || pass == poplibs_test::Pass::WU;
  bool fwdOnly = !doBwdPass && !doWuPass;

  auto &varTimeSteps = variableTimeStepsOption.val;
  if ((varTimeSteps.size() > 1) && (varTimeSteps.size() != batchSize)) {
    throw poputil::poplibs_error("timeSteps must either be a scalar or a "
                                 "tensor of batch-size length");
  }
  Tensor timeSteps;
  if (varTimeSteps.size()) {
    timeSteps = graph.addVariable(UNSIGNED_INT, {varTimeSteps.size()},
                                  "var-time-steps");
    mapTensorLinearly(graph, timeSteps);
    if (!ignoreFinalState) {
      // Final cell state is not calculated for variable time steps.
      std::cout << "Forcing `ignoreFinalState=true` since variable time steps "
                   "options is set!\n";
      ignoreFinalState = true;
    }
  }

  poplin::matmul::PlanningCache cache;
  lstm::LstmParams params(dataType, batchSize, sequenceSize, timeSteps,
                          {inputSize, outputSize}, activation,
                          recurrentActivation);
  params.outputFullSequence = outputAllSequence;
  if (!cellOrder.val.empty()) {
    params.cellOrder = getCellOrder(cellOrder.val);
  }
  if (ignoreInputGradient) {
    params.calcInputGradients = false;
  }
  poplar::OptionFlags fwdOptions = {
      {"inferenceOnly", fwdOnly ? "true" : "false"},
  };
  if (!vm["shards"].empty()) {
    fwdOptions.set("numShards", std::to_string(numShards));
  }
  if (!vm["code-reuse"].empty()) {
    fwdOptions.set("rnnCodeReuse", codeReuse ? "true" : "false");
  }
  if (!vm["available-memory-proportion"].empty()) {
    fwdOptions.set("availableMemoryProportion",
                   std::to_string(availableMemoryProportion));
  }
  if (!vm["partials-type"].empty()) {
    fwdOptions.set("partialsType", partialsType.toString());
  }
  if (!vm["accumulators-type"].empty()) {
    fwdOptions.set("weightAccumulatorsType", accumulatorsType.toString());
  }
  if (!vm["recomputation-mode"].empty()) {
    fwdOptions.set("recomputationMode", recompMode);
  }
  if (preweightInput) {
    fwdOptions.set({{"preCalcWeights", "true"}});
  }

  auto bwdOptions = fwdOptions;
  if (!vm["steps-per-wu"].empty()) {
    bwdOptions.set("rnnStepsPerWU", std::to_string(stepsPerWU));
  }
  if (!vm["wu-memory-proportion"].empty()) {
    bwdOptions.set("weightUpdateMemoryProportion",
                   std::to_string(weightUpdateMemoryProportion));
  }
  if (!vm["disable-wu-partial-interleaving"].empty()) {
    bwdOptions.set("disableWUPartialInterleaving",
                   disableWUPartialInterleaving ? "true" : "false");
  }

  auto input = lstm::createInput(graph, params, "input", fwdOptions, &cache);

  auto prog = Sequence();
  auto fwdStateInit =
      lstm::createInitialState(graph, params, "fwdState", fwdOptions, &cache);
  auto outputInit = fwdStateInit.output;
  auto cellStateInit = fwdStateInit.cellState;
  auto weights =
      lstm::createWeights(graph, params, "weights", fwdOptions, &cache);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  Tensor fwdOutputSeq, lastCellState, fwdIntermediates;
  Tensor *fwdIntermediatesPtr =
      (doBwdPass || doWuPass) ? &fwdIntermediates : nullptr;
  if (ignoreFinalState) {
    fwdOutputSeq =
        popnn::lstm::lstmFwd(graph, params, prog, fwdStateInit, weights, input,
                             fwdIntermediatesPtr, "fwd", fwdOptions, &cache);
  } else {
    // Test deprecated API
    std::tie(fwdOutputSeq, lastCellState) = popnn::lstm::lstmFwd(
        graph, params, fwdStateInit, input, weights, fwdIntermediatesPtr, prog,
        "fwd", fwdOptions, &cache);
  }
  auto nextLayerGrads = graph.addVariable(
      dataType, {sequenceSize, batchSize, outputSize}, "nextLayerGrads");
  mapTensorLinearly(graph, nextLayerGrads);

  Tensor prevLayerGrads;
  Tensor *inputGrad = ignoreInputGradient ? nullptr : &prevLayerGrads;
  lstm::LstmWeights weightGrads;
  Tensor lastGradLayerOut;
  Tensor lastGradCellState;
  if (doBwdPass || doWuPass) {
    const Tensor *lastCellStateGradPtr = nullptr;
    const Tensor nextGrad =
        params.outputFullSequence ? nextLayerGrads : nextLayerGrads[0];
    lstm::LstmState lastGradLayer;
    if (doWuPass) {
      lastGradLayer = lstm::lstmBwdWithWU(
          graph, params, prog, fwdStateInit, fwdIntermediates, weights, input,
          fwdOutputSeq, nextGrad, lastCellStateGradPtr, inputGrad, weightGrads,
          "bwd", bwdOptions, &cache);
    } else {
      lastGradLayer = lstm::lstmBwd(
          graph, params, prog, fwdStateInit, fwdIntermediates, weights, input,
          fwdOutputSeq, nextGrad, lastCellStateGradPtr, inputGrad, nullptr,
          "bwd", bwdOptions, &cache);
    }
    lastGradLayerOut = lastGradLayer.output;
    lastGradCellState = lastGradLayer.cellState;
  }

  std::unique_ptr<char[]> rawTimeSteps;
  std::unique_ptr<char[]> rawHostWeightsInput;
  std::unique_ptr<char[]> rawHostWeightsOutput;
  std::unique_ptr<char[]> rawHostPrevLayerAct;
  std::unique_ptr<char[]> rawHostBiases;
  std::unique_ptr<char[]> rawHostOutputInit;
  std::unique_ptr<char[]> rawHostCellStateInit;
  std::unique_ptr<char[]> rawHostNextLayerGrads;
  std::unique_ptr<char[]> rawHostPrevLayerGrads;
  std::unique_ptr<char[]> rawHostWeightsInputDeltas;
  std::unique_ptr<char[]> rawHostWeightsOutputDeltas;
  std::unique_ptr<char[]> rawHostBiasDeltas;

  std::unique_ptr<char[]> rawHostNextAct;
  std::unique_ptr<char[]> rawLastCellState;
  std::unique_ptr<char[]> rawGradPrevLayerOut;
  std::unique_ptr<char[]> rawGradPrevCellState;

  if (varTimeSteps.size()) {
    rawTimeSteps = allocateHostMemoryForTensor(timeSteps, "timeSteps", graph,
                                               uploadProg, downloadProg, tmap);
  }
  if (!ignoreData) {
    rawHostWeightsInput =
        allocateHostMemoryForTensor(weights.inputWeights, "weightsInput", graph,
                                    uploadProg, downloadProg, tmap);
    rawHostWeightsOutput =
        allocateHostMemoryForTensor(weights.outputWeights, "weightsOutput",
                                    graph, uploadProg, downloadProg, tmap);
    rawHostPrevLayerAct = allocateHostMemoryForTensor(
        input, "prevLayerAct", graph, uploadProg, downloadProg, tmap);
    rawHostBiases = allocateHostMemoryForTensor(weights.biases, "biases", graph,
                                                uploadProg, downloadProg, tmap);
    rawHostOutputInit = allocateHostMemoryForTensor(
        outputInit, "outputInit", graph, uploadProg, downloadProg, tmap);
    rawHostCellStateInit = allocateHostMemoryForTensor(
        cellStateInit, "cellStateInit", graph, uploadProg, downloadProg, tmap);

    if (doBwdPass) {
      rawHostNextLayerGrads =
          allocateHostMemoryForTensor(nextLayerGrads, "nextLayerGrads", graph,
                                      uploadProg, downloadProg, tmap);
      if (!ignoreInputGradient) {
        rawHostPrevLayerGrads =
            allocateHostMemoryForTensor(prevLayerGrads, "prevLayerGrads", graph,
                                        uploadProg, downloadProg, tmap);
      }
    }
    if (doWuPass) {
      rawHostWeightsInputDeltas = allocateHostMemoryForTensor(
          weightGrads.inputWeights, "weightsInputDeltas", graph, uploadProg,
          downloadProg, tmap);

      rawHostWeightsOutputDeltas = allocateHostMemoryForTensor(
          weightGrads.outputWeights, "weightsOutputDeltas", graph, uploadProg,
          downloadProg, tmap);
      rawHostBiasDeltas =
          allocateHostMemoryForTensor(weightGrads.biases, "biasDeltas", graph,
                                      uploadProg, downloadProg, tmap);
    }

    rawHostNextAct = allocateHostMemoryForTensor(
        fwdOutputSeq, "nextAct", graph, uploadProg, downloadProg, tmap);
    if (!ignoreFinalState) {
      rawLastCellState =
          allocateHostMemoryForTensor(lastCellState, "lastCellState", graph,
                                      uploadProg, downloadProg, tmap);
    }
    if (doBwdPass || doWuPass) {
      rawGradPrevLayerOut =
          allocateHostMemoryForTensor(lastGradLayerOut, "lastGradLayerOut",
                                      graph, uploadProg, downloadProg, tmap);
      rawGradPrevCellState =
          allocateHostMemoryForTensor(lastGradCellState, "lastGradCellState",
                                      graph, uploadProg, downloadProg, tmap);
    }
  }

  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile") || vm.count("profile-dir")) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (vm.count("profile-dir")) {
      engineOptions.set("autoReport.all", "true");
      engineOptions.set("autoReport.directory", profileDir);
    }
  }
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);

  if (vm.count("compile-only"))
    return 0;

  attachStreams(engine, tmap);

  boost::multi_array<unsigned, 1> hostTimeSteps(
      boost::extents[varTimeSteps.size()]);
  boost::multi_array<double, 3> hostPrevLayerAct(
      boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3> hostWeightsOutput(
      boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsInput(
      boost::extents[BASIC_LSTM_CELL_NUM_UNITS][inputSize][outputSize]);
  boost::multi_array<double, 2> hostBiases(
      boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2> hostCellStateInit(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 3> hostNextAct(
      boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 2> modelCellState(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2> modelLastCellState(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2> modelGradPrevLayerOut(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2> modelGradPrevCellState(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2> modelLastOutput(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2> hostOutputInit(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 4> modelFwdState(
      boost::extents[LSTM_NUM_FWD_STATES][sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3> hostNextLayerGrads(
      boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3> hostPrevLayerGrads(
      boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3> modelPrevLayerGrads(
      boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 4> modelBwdState(
      boost::extents[LSTM_NUM_BWD_STATES][sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsOutputDeltas(
      boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsInputDeltas(
      boost::extents[BASIC_LSTM_CELL_NUM_UNITS][inputSize][outputSize]);
  boost::multi_array<double, 2> hostBiasesDeltas(
      boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);

  if (varTimeSteps.size()) {
    copy(varTimeSteps.begin(), varTimeSteps.end(), hostTimeSteps.data());
    copy(target, hostTimeSteps, UNSIGNED_INT, rawTimeSteps.get());
  }

  std::mt19937 randomEngine;

  if (!ignoreData) {
    writeRandomValues(target, dataType, hostPrevLayerAct, -4.0, 4.0,
                      randomEngine);
    writeRandomValues(target, dataType, hostOutputInit, -3.0, 3.0,
                      randomEngine);
    writeRandomValues(target, dataType, hostCellStateInit, -3.0, 3.0,
                      randomEngine);
    writeRandomValues(target, dataType, hostWeightsInput, -1.0, 1.0,
                      randomEngine);
    writeRandomValues(target, dataType, hostWeightsOutput, -1.0, 1.0,
                      randomEngine);
    writeRandomValues(target, dataType, hostBiases, -1.0, 1.0, randomEngine);

    if (params.outputFullSequence) {
      writeRandomValues(target, dataType, hostNextAct, -1.0, 1.0, randomEngine);
    }

    if (!ignoreInputGradient && doBwdPass) {
      writeRandomValues(target, dataType, hostPrevLayerGrads, -1.0, 1.0,
                        randomEngine);
    }

    if (doBwdPass) {
      writeRandomValues(target, dataType, hostNextLayerGrads, -2.0, 2.0,
                        randomEngine);
    }

    modelCellState = hostCellStateInit;

    copy(target, hostPrevLayerAct, dataType, rawHostPrevLayerAct.get());
    copy(target, hostCellStateInit, dataType, rawHostCellStateInit.get());
    copy(target, hostOutputInit, dataType, rawHostOutputInit.get());
    copy(target, hostBiases, dataType, rawHostBiases.get());
    copy(target, hostWeightsInput, dataType, rawHostWeightsInput.get());
    copy(target, hostWeightsOutput, dataType, rawHostWeightsOutput.get());
    if (doBwdPass) {
      copy(target, hostNextLayerGrads, dataType, rawHostNextLayerGrads.get());
    }

    // Prefill output buffer with random data
    if (params.outputFullSequence) {
      copy(target, hostNextAct, dataType, rawHostNextAct.get());
    }
    if (!ignoreInputGradient && doBwdPass) {
      copy(target, hostPrevLayerGrads, dataType, rawHostPrevLayerGrads.get());
    }
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    // Can do multiple calls to run to check
    // nothing is accumulating between runs
    for (unsigned i = 0; i < runs; i++) {
      engine.run(0);
    }
  });

  if (deviceType != DeviceType::Cpu) {
    if (vm.count("profile")) {
      engine.printProfileSummary(std::cout,
                                 OptionFlags{
                                     // { "showExecutionSteps", "true" }
                                 });
    }
  }

  bool matchesModel = true;
  if (!ignoreData) {
    boost::optional<boost::multi_array_ref<unsigned, 1>> hostTimeStepsOpt;
    if (varTimeSteps.size()) {
      hostTimeStepsOpt = hostTimeSteps;
    }
    poplibs_test::lstm::basicLstmCellForwardPass(
        hostPrevLayerAct, hostBiases, hostOutputInit, hostWeightsInput,
        hostWeightsOutput, hostTimeStepsOpt, modelCellState, modelFwdState,
        modelLastOutput, modelLastCellState, params.cellOrder, activation,
        recurrentActivation);

    if (doBwdPass) {
      poplibs_test::lstm::basicLstmCellBackwardPass(
          params.outputFullSequence, hostWeightsInput, hostWeightsOutput,
          hostNextLayerGrads, hostCellStateInit, modelFwdState,
          hostTimeStepsOpt, modelBwdState, modelPrevLayerGrads,
          modelGradPrevLayerOut, modelGradPrevCellState, params.cellOrder,
          activation, recurrentActivation);
    }

    boost::multi_array<double, 3> matImpl(
        boost::extents[params.outputFullSequence ? sequenceSize : 1][batchSize]
                      [outputSize]);
    copy(target, dataType, rawHostNextAct.get(), matImpl);
    if (params.outputFullSequence) {
      for (auto s = 0U; s != sequenceSize; ++s) {
        boost::multi_array<double, 2> subMatImpl = matImpl[s];
        boost::multi_array<double, 2> subMatRef =
            modelFwdState[LSTM_FWD_STATE_ACTS_IDX][s];
        matchesModel &= checkIsClose("nextLayerAct", subMatImpl, subMatRef,
                                     relativeTolerance, absoluteTolerance);
      }
    } else if (!ignoreFinalState) {
      const boost::multi_array<double, 2> subMatImpl = matImpl[0];
      matchesModel &= checkIsClose("nextLayerAct", subMatImpl, modelLastOutput,
                                   relativeTolerance, absoluteTolerance);
    }

    if (!ignoreFinalState) {
      boost::multi_array<double, 2> hostLastCellState(
          boost::extents[batchSize][outputSize]);
      copy(target, dataType, rawLastCellState.get(), hostLastCellState);
      matchesModel &=
          checkIsClose("lastCellState", hostLastCellState, modelLastCellState,
                       relativeTolerance, absoluteTolerance);
    }
    if (doBwdPass) {
      if (!ignoreInputGradient) {
        copy(target, dataType, rawHostPrevLayerGrads.get(), hostPrevLayerGrads);
        matchesModel &= checkIsClose("prevLayerGrads", hostPrevLayerGrads,
                                     modelPrevLayerGrads, relativeTolerance,
                                     absoluteTolerance);
      }

      boost::multi_array<double, 2> hostGradPrevLayerOut(
          boost::extents[batchSize][outputSize]);
      copy(target, dataType, rawGradPrevLayerOut.get(), hostGradPrevLayerOut);
      matchesModel &= checkIsClose("lastGradLayerOut", hostGradPrevLayerOut,
                                   modelGradPrevLayerOut, relativeTolerance,
                                   absoluteTolerance);

      boost::multi_array<double, 2> hostGradPrevCellState(
          boost::extents[batchSize][outputSize]);
      copy(target, dataType, rawGradPrevCellState.get(), hostGradPrevCellState);
      matchesModel &= checkIsClose("lastGradCellState", hostGradPrevCellState,
                                   modelGradPrevCellState, relativeTolerance,
                                   absoluteTolerance);
    }

    if (doWuPass) {
      copy(target, weightGrads.inputWeights.elementType(),
           rawHostWeightsInputDeltas.get(), hostWeightsInputDeltas);
      copy(target, weightGrads.outputWeights.elementType(),
           rawHostWeightsOutputDeltas.get(), hostWeightsOutputDeltas);
      copy(target, weightGrads.biases.elementType(), rawHostBiasDeltas.get(),
           hostBiasesDeltas);
      boost::multi_array<double, 3> modelWeightsOutputDeltas(
          boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize][outputSize]);
      boost::multi_array<double, 3> modelWeightsInputDeltas(
          boost::extents[BASIC_LSTM_CELL_NUM_UNITS][inputSize][outputSize]);
      boost::multi_array<double, 2> modelBiasesDeltas(
          boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);
      poplibs_test::lstm::basicLstmCellParamUpdate(
          hostPrevLayerAct, modelFwdState, hostOutputInit, modelBwdState,
          modelWeightsInputDeltas, modelWeightsOutputDeltas, modelBiasesDeltas,
          params.cellOrder);
      matchesModel &= checkIsClose("weightsInputDeltas", hostWeightsInputDeltas,
                                   modelWeightsInputDeltas, relativeTolerance,
                                   absoluteTolerance);
      matchesModel &= checkIsClose(
          "weightsOutputDeltas", hostWeightsOutputDeltas,
          modelWeightsOutputDeltas, relativeTolerance, absoluteTolerance);
      matchesModel &=
          checkIsClose("biasDeltas", hostBiasesDeltas, modelBiasesDeltas,
                       relativeTolerance, absoluteTolerance);
    }
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
