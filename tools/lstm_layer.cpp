#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <fstream>
#include <istream>
#include <ostream>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/Lstm.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popops/Zero.hpp>
#include <popnn/codelets.hpp>
#include "TestDevice.hpp"
#include <poplibs_test/Lstm.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_support/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popnn;

// Default tolerances used in tests
#define FLOAT_REL_TOL  0.1
#define HALF_REL_TOL   0.3
#define FLOAT_ABS_TOL  1e-5
#define HALF_ABS_TOL   7e-2

const OptionFlags defaultEngineOptions {
  {"target.workerStackSizeInBytes", "0x200"}
};

void savePoplarReport(poplar::Engine &engine, std::string &dir) {
  // Graph Report
  poplar::ProfileValue graphProfile = engine.getGraphProfile();
  std::ofstream graphReport;
  graphReport.open(dir + "/graph.json");
  poplar::serializeToJSON(graphReport, graphProfile);
  graphReport.close();

  // Execution Report
  poplar::ProfileValue execProfile = engine.getExecutionProfile();
  std::ofstream execReport;
  execReport.open(dir + "/execution.json");
  poplar::serializeToJSON(execReport, execProfile);
  execReport.close();
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  DeviceType deviceType = DeviceType::IpuModel;

  unsigned sequenceSize, inputSize, outputSize;
  unsigned batchSize = 1;

  Type dataType;
  Type partialsType;
  double relativeTolerance;
  double absoluteTolerance;
  IPUModel ipuModel;
  bool preweightInput = false;
  poplibs_test::Pass pass = poplibs_test::Pass::FWD;
  std::string recompMode = "none";
  unsigned runs = 1;
  std::string profileDir = ".";
  double availableMemoryProportion = 0.0;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Hw | IpuModel")
    ("profile", "Output profiling report")
    ("profile-dir",
      po::value<std::string>(&profileDir)->default_value(profileDir),
      "The directory to output profiling report")
    ("sequence-size", po::value<unsigned>(&sequenceSize)->required(),
     "Sequence size in the RNN")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs in each element in the sequence")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of outputs in each element in the sequence")
    ("data-type",
      po::value<Type>(&dataType)->default_value(HALF),
      "Input and output data type")
    ("batch-size", po::value<unsigned>(&batchSize)->default_value(batchSize),
      "Batch size")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(FLOAT),
     "Type of the partials")
    ("rel-tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("abs-tolerance",po::value<double>(&absoluteTolerance),
     "Absolute tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&ipuModel.tilesPerIPU)->
                           default_value(ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&ipuModel.numIPUs)->default_value(ipuModel.numIPUs),
     "Number of IPUs")
    ("pre-weight-input",
       po::value<bool>(&preweightInput)->default_value(preweightInput),
     "Pre-weight whole sequence before recursive part is computed (0 / 1)")
      ("phase",
     po::value<poplibs_test::Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("recomputation-mode",
     po::value<std::string>(&recompMode
    )->default_value(recompMode),
     "Recomputation mode none | cellAndTanh")
    ("ignore-data",
     "Don't perform host-to-device or vice versa transfers (no validation)")
    ("runs", po::value<unsigned>(&runs)->default_value(runs),
     "Number of calls to Engine::run")
    ("available-memory-proportion",
     po::value<double>(&availableMemoryProportion),
     "What percentage of memory is available to the operation for temporary "
     "use")
  ;

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception& e) {
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

  bool ignoreData = vm.count("ignore-data");

  auto device = createTestDevice(deviceType, ipuModel.numIPUs,
                                  ipuModel.tilesPerIPU);

  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  // Bwd pass is always run if WU is run. This may change is tensors input to
  //  WU are created on host
  bool doBwdPass = pass == poplibs_test::Pass::ALL
                   || pass == poplibs_test::Pass::BWD
                   || pass == poplibs_test::Pass::WU;
  bool doWuPass = pass == poplibs_test::Pass::ALL
                  || pass == poplibs_test::Pass::WU;
  bool fwdOnly = !doBwdPass && !doWuPass;

  poplin::matmul::PlanningCache cache;
  lstm::LstmParams params(dataType, batchSize, sequenceSize,
                          {inputSize, outputSize});
  poplar::OptionFlags options({{"inferenceOnly", fwdOnly ? "true" : "false"},
                               {"partialsType", partialsType.toString()},
                               {"recomputationMode", recompMode}});
  if (!vm["available-memory-proportion"].empty()) {
    options.set("availableMemoryProportion",
                std::to_string(availableMemoryProportion));
  }
  if (preweightInput) {
    options.set({{"preCalcWeights", "true"}});
  }

  auto input = lstm::createInput(graph, params, "input", options, &cache);

  auto prog = Sequence();
  auto fwdStateInit = lstm::createInitialState(graph, params, "fwdState",
                                               options, &cache);
  auto outputInit = fwdStateInit.output;
  auto cellStateInit = fwdStateInit.cellState;
  auto weights = lstm::createWeights(graph, params, "weights", options,
                                     &cache);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  Tensor fwdOutputSeq, lastCellState, fwdIntermediates;
  Tensor *fwdIntermediatesPtr = (doBwdPass || doWuPass) ? &fwdIntermediates :
                                                          nullptr;
  std::tie(fwdOutputSeq, lastCellState) =
      popnn::lstm::lstmFwd(graph, params, fwdStateInit,
                           input, weights, fwdIntermediatesPtr, prog, "fwd",
                           options, &cache);
  auto nextLayerGrads =
     graph.addVariable(dataType, {sequenceSize, batchSize, outputSize},
                       "nextLayerGrads");
  mapTensorLinearly(graph, nextLayerGrads);

  Tensor prevLayerGrads;
  lstm::LstmWeights weightGrads;
  if (doBwdPass || doWuPass) {
    const Tensor *lastCellStateGradPtr = nullptr;
    if (doWuPass) {
      lstm::lstmBwdWithWU(graph, params, prog, fwdStateInit, fwdIntermediates,
                          weights, input, fwdOutputSeq, nextLayerGrads,
                          lastCellStateGradPtr, &prevLayerGrads,
                          weightGrads, "bwd", options, &cache);
    } else {
      lstm::lstmBwd(graph, params, prog, fwdStateInit, fwdIntermediates,
                    weights, input, fwdOutputSeq, nextLayerGrads,
                    lastCellStateGradPtr, &prevLayerGrads,
                    nullptr, "bwd", options, &cache);
    }
  }

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

  std::vector<std::unique_ptr<char[]>> rawHostNextAct;

  if (!ignoreData) {
    rawHostWeightsInput =
      allocateHostMemoryForTensor(weights.inputWeights, "weightsInput", graph,
                                  uploadProg, downloadProg, tmap);
    rawHostWeightsOutput =
      allocateHostMemoryForTensor(weights.outputWeights, "weightsOutput", graph,
                                  uploadProg, downloadProg, tmap);
    rawHostPrevLayerAct =
      allocateHostMemoryForTensor(input, "prevLayerAct", graph, uploadProg,
                                  downloadProg, tmap);
    rawHostBiases =
      allocateHostMemoryForTensor(weights.biases, "biases", graph, uploadProg,
                                  downloadProg, tmap);
    rawHostOutputInit =
      allocateHostMemoryForTensor(outputInit, "outputInit", graph, uploadProg,
                                  downloadProg, tmap);
    rawHostCellStateInit =
      allocateHostMemoryForTensor(cellStateInit, "cellStateInit", graph,
                                  uploadProg, downloadProg, tmap);

    if (doBwdPass) {
      rawHostNextLayerGrads =
        allocateHostMemoryForTensor(nextLayerGrads, "nextLayerGrads", graph,
                                    uploadProg, downloadProg, tmap);
      rawHostPrevLayerGrads =
        allocateHostMemoryForTensor(prevLayerGrads, "prevLayerGrads", graph,
                                    uploadProg, downloadProg, tmap);
    }
    if (doWuPass) {
      rawHostWeightsInputDeltas =
        allocateHostMemoryForTensor(weightGrads.inputWeights,
                                    "weightsInputDeltas",
                                    graph, uploadProg, downloadProg, tmap);
      rawHostWeightsOutputDeltas =
        allocateHostMemoryForTensor(weightGrads.outputWeights,
                                    "weightsOutputDeltas",
                                    graph, uploadProg, downloadProg, tmap);
      rawHostBiasDeltas =
        allocateHostMemoryForTensor(weightGrads.biases, "biasDeltas", graph,
                                    uploadProg, downloadProg, tmap);
    }

    for (auto s = 0U; s != sequenceSize; ++s) {
      auto nextAct = fwdOutputSeq[s];
      rawHostNextAct.push_back(allocateHostMemoryForTensor(nextAct,
                                                           "nextAct" +
                                                             std::to_string(s),
                                                           graph, uploadProg,
                                                           downloadProg, tmap));
    }
  }

  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);
  attachStreams(engine, tmap);

  boost::multi_array<double, 3>
      hostPrevLayerAct(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3>
      hostWeightsOutput(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                [outputSize][outputSize]);
  boost::multi_array<double, 3>
      hostWeightsInput(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                [inputSize][outputSize]);
  boost::multi_array<double, 2>
      hostBiases(boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2>
      hostCellStateInit(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2>
      modelCellState(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2>
      hostOutputInit(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 4>
      modelFwdState(boost::extents[LSTM_NUM_FWD_STATES][sequenceSize]
                                  [batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostNextLayerGrads(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostPrevLayerGrads(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3>
      modelPrevLayerGrads(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 4>
      modelBwdState(boost::extents[LSTM_NUM_BWD_STATES][sequenceSize]
                                  [batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostWeightsOutputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                            [outputSize][outputSize]);
  boost::multi_array<double, 3>
      hostWeightsInputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                           [inputSize][outputSize]);
  boost::multi_array<double, 2>
      hostBiasesDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);

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
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    // Can do multiple calls to run to check
    // nothing is accumulating between runs
    for (unsigned  i = 0; i < runs; i++) {
      engine.run(0);
    }
  });

  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    engine.printProfileSummary(std::cout, OptionFlags{
     // { "showExecutionSteps", "true" }
    });

    if (vm.count("profile-dir"))
      savePoplarReport(engine, profileDir);
  }

  bool matchesModel = true;
  if (!ignoreData) {
    poplibs_test::lstm::basicLstmCellForwardPass(
        hostPrevLayerAct, hostBiases, hostOutputInit,
        hostWeightsInput, hostWeightsOutput, modelCellState,
        modelFwdState);

    if (doBwdPass) {
      poplibs_test::lstm::basicLstmCellBackwardPass(
          hostWeightsInput, hostWeightsOutput,
          hostNextLayerGrads, hostCellStateInit,
          modelFwdState,  modelBwdState, modelPrevLayerGrads);
    }

    for (auto s = 0U; s != rawHostNextAct.size(); ++s) {
      boost::multi_array<double, 2> subMatImp(boost::extents[batchSize]
                                                            [outputSize]);
      copy(target, dataType, rawHostNextAct[s].get(), subMatImp);
      boost::multi_array<double, 2> subMatRef =
          modelFwdState[LSTM_FWD_STATE_ACTS_IDX][s];
      matchesModel &= checkIsClose("nextLayerAct", subMatImp, subMatRef,
                                   relativeTolerance, absoluteTolerance);
    }

    if (doBwdPass) {
      copy(target, dataType, rawHostPrevLayerGrads.get(), hostPrevLayerGrads);

      matchesModel &=
        checkIsClose("prevLayerGrads", hostPrevLayerGrads, modelPrevLayerGrads,
                     relativeTolerance, absoluteTolerance);
    }

    if (doWuPass) {
      copy(target, dataType, rawHostWeightsInputDeltas.get(),
           hostWeightsInputDeltas);
      copy(target, dataType, rawHostWeightsOutputDeltas.get(),
           hostWeightsOutputDeltas);
      copy(target, dataType, rawHostBiasDeltas.get(), hostBiasesDeltas);
      boost::multi_array<double, 3>
          modelWeightsOutputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                                [outputSize][outputSize]);
      boost::multi_array<double, 3>
          modelWeightsInputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                               [inputSize][outputSize]);
      boost::multi_array<double, 2>
          modelBiasesDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                          [outputSize]);
      poplibs_test::lstm::basicLstmCellParamUpdate(
                              hostPrevLayerAct, modelFwdState, hostOutputInit,
                              modelBwdState, modelWeightsInputDeltas,
                              modelWeightsOutputDeltas, modelBiasesDeltas);
      matchesModel &=
          checkIsClose("weightsInputDeltas", hostWeightsInputDeltas,
                       modelWeightsInputDeltas, relativeTolerance,
                       absoluteTolerance);
      matchesModel &=
        checkIsClose("weightsOutputDeltas", hostWeightsOutputDeltas,
                     modelWeightsOutputDeltas, relativeTolerance,
                     absoluteTolerance);
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
