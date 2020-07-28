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
#include <poplibs_test/Gru.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/Gru.hpp>
#include <popnn/codelets.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popnn;
using namespace poplibs_support;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.10
#define HALF_REL_TOL 0.3
#define FLOAT_ABS_TOL 1e-5
#define HALF_ABS_TOL 7e-2

const OptionFlags defaultEngineOptions;

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
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  bool outputAllSequence = true;
  poplibs_test::Pass pass = poplibs_test::Pass::FWD;
  unsigned runs = 1;
  std::string profileDir = ".";
  double availableMemoryProportion;
  boost::optional<std::string> jsonProfileOut;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Sim2 | Hw | IpuModel | IpuModel2")
    ("profile", "Output profiling report")
    ("profile-dir",
      po::value<std::string>(&profileDir)->default_value(profileDir),
      "The directory to output profiling report")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("use-unstable-format", "Use the unstable profile format")
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
     po::value<Type>(&partialsType),
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
    ("phase",
     po::value<poplibs_test::Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("ignore-data",
     "Don't perform host-to-device or vice versa transfers (no validation)")
    ("runs", po::value<unsigned>(&runs)->default_value(runs),
     "Number of calls to Engine::run")
    ("available-memory-proportion",
     po::value<double>(&availableMemoryProportion),
     "What percentage of memory is available to the operation for temporary "
     "use")
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

  bool ignoreData = vm.count("ignore-data");
  const bool useUnstableFormat = vm.count("use-unstable-format");

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

  poplin::matmul::PlanningCache cache;
  gru::GruParams params(dataType, batchSize, sequenceSize,
                        {inputSize, outputSize});
  params.outputFullSequence = outputAllSequence;

  poplar::OptionFlags options = {
      {"inferenceOnly", fwdOnly ? "true" : "false"},
  };
  if (!vm["available-memory-proportion"].empty()) {
    options.set("availableMemoryProportion",
                std::to_string(availableMemoryProportion));
  }
  if (!vm["partials-type"].empty()) {
    options.set("partialsType", partialsType.toString());
  }

  auto input = gru::createInput(graph, params, "input", options, &cache);

  auto prog = Sequence();
  auto outputInit =
      gru::createInitialState(graph, params, "fwdState", options, &cache);

  auto weights = gru::createWeights(graph, params, "weights", options, &cache);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  Tensor fwdOutputSeq, fwdIntermediates;
  Tensor *fwdIntermediatesPtr =
      (doBwdPass || doWuPass) ? &fwdIntermediates : nullptr;
  fwdOutputSeq =
      popnn::gru::gruFwd(graph, params, outputInit, input, weights,
                         fwdIntermediatesPtr, prog, "fwd", options, &cache);

  auto nextLayerGrads = graph.addVariable(
      dataType, {sequenceSize, batchSize, outputSize}, "nextLayerGrads");
  mapTensorLinearly(graph, nextLayerGrads);

  Tensor prevLayerGrads;
  gru::GruWeights weightGrads =
      gru::createWeights(graph, params, "weightGrad", options, &cache);
  if (doBwdPass || doWuPass) {
    if (doWuPass) {
      if (params.outputFullSequence)
        gru::gruBwdWithWU(graph, params, prog, outputInit, fwdIntermediates,
                          weights, input, fwdOutputSeq, nextLayerGrads,
                          &prevLayerGrads, weightGrads, "bwd", options, &cache);
      else
        // If only output the last cell, nextLayerGrads only contains the
        // gradient for the last cell.
        gru::gruBwdWithWU(graph, params, prog, outputInit, fwdIntermediates,
                          weights, input, fwdOutputSeq, nextLayerGrads[0],
                          &prevLayerGrads, weightGrads, "bwd", options, &cache);
    } else {
      if (params.outputFullSequence)
        gru::gruBwd(graph, params, prog, outputInit, fwdIntermediates, weights,
                    input, fwdOutputSeq, nextLayerGrads, &prevLayerGrads,
                    nullptr, "bwd", options, &cache);
      else
        gru::gruBwd(graph, params, prog, outputInit, fwdIntermediates, weights,
                    input, fwdOutputSeq, nextLayerGrads[0], &prevLayerGrads,
                    nullptr, "bwd", options, &cache);
    }
  }

  std::unique_ptr<char[]> rawHostWeightsInput;
  std::unique_ptr<char[]> rawHostWeightsOutput;
  std::unique_ptr<char[]> rawHostPrevLayerAct;
  std::unique_ptr<char[]> rawHostBiases;
  std::unique_ptr<char[]> rawHostOutputInit;
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
        allocateHostMemoryForTensor(weights.outputWeights, "weightsOutput",
                                    graph, uploadProg, downloadProg, tmap);
    rawHostPrevLayerAct = allocateHostMemoryForTensor(
        input, "prevLayerAct", graph, uploadProg, downloadProg, tmap);
    rawHostBiases = allocateHostMemoryForTensor(weights.biases, "biases", graph,
                                                uploadProg, downloadProg, tmap);
    rawHostOutputInit = allocateHostMemoryForTensor(
        outputInit, "outputInit", graph, uploadProg, downloadProg, tmap);
    if (doBwdPass) {
      rawHostNextLayerGrads =
          allocateHostMemoryForTensor(nextLayerGrads, "nextLayerGrads", graph,
                                      uploadProg, downloadProg, tmap);
      rawHostPrevLayerGrads =
          allocateHostMemoryForTensor(prevLayerGrads, "prevLayerGrads", graph,
                                      uploadProg, downloadProg, tmap);
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

    if (params.outputFullSequence) {
      for (auto s = 0U; s != sequenceSize; ++s) {
        auto nextAct = fwdOutputSeq[s];
        rawHostNextAct.push_back(
            allocateHostMemoryForTensor(nextAct, "nextAct" + std::to_string(s),
                                        graph, uploadProg, downloadProg, tmap));
      }
    } else {
      rawHostNextAct.push_back(allocateHostMemoryForTensor(
          fwdOutputSeq, "nextAct", graph, uploadProg, downloadProg, tmap));
    }
  }

  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile") || jsonProfileOut) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (useUnstableFormat) {
      engineOptions.set("profiler.useUnstableFormat", "true");
    }
  }
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);
  attachStreams(engine, tmap);

  boost::multi_array<double, 3> hostPrevLayerAct(
      boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3> hostWeightsOutput(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsInput(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][inputSize][outputSize]);
  boost::multi_array<double, 2> hostBiases(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2> hostOutputInit(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 4> modelFwdState(
      boost::extents[GRU_NUM_FWD_STATES][sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3> hostNextLayerGrads(
      boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3> hostPrevLayerGrads(
      boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3> modelPrevLayerGrads(
      boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 4> modelBwdState(
      boost::extents[GRU_NUM_BWD_STATES][sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsOutputDeltas(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsInputDeltas(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][inputSize][outputSize]);
  boost::multi_array<double, 2> hostBiasesDeltas(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);

  std::mt19937 randomEngine;
  randomEngine.seed(1002);

  if (!ignoreData) {
    writeRandomValues(target, dataType, hostPrevLayerAct, -4.0, 4.0,
                      randomEngine);
    writeRandomValues(target, dataType, hostOutputInit, -3.0, 3.0,
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

    copy(target, hostPrevLayerAct, dataType, rawHostPrevLayerAct.get());
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
    for (unsigned i = 0; i < runs; i++) {
      engine.run(0);
    }
  });

  if (deviceType != DeviceType::Cpu) {
    if (jsonProfileOut) {
      const auto pr = engine.getProfile();
      std::ofstream os(*jsonProfileOut);
      poplar::serializeToJSON(os, pr);
    }
    if (vm.count("profile")) {
      engine.printProfileSummary(std::cout,
                                 OptionFlags{
                                     //{ "showExecutionSteps", "true" }
                                     //{ "showVarStorage",     "true" }
                                 });
      if (vm.count("profile-dir"))
        savePoplarReport(engine, profileDir);
    }
  }

  bool matchesModel = true;
  if (!ignoreData) {
    poplibs_test::gru::basicGruCellForwardPass(
        hostPrevLayerAct, hostBiases, hostOutputInit, hostWeightsInput,
        hostWeightsOutput, modelFwdState);

    if (doBwdPass) {
      poplibs_test::gru::basicGruCellBackwardPass(
          params.outputFullSequence, hostWeightsInput, hostWeightsOutput,
          hostNextLayerGrads, modelFwdState, hostOutputInit, modelBwdState,
          modelPrevLayerGrads);
    }

    if (params.outputFullSequence) {
      for (auto s = 0U; s != rawHostNextAct.size(); ++s) {
        boost::multi_array<double, 2> subMatImp(
            boost::extents[batchSize][outputSize]);
        copy(target, dataType, rawHostNextAct[s].get(), subMatImp);
        boost::multi_array<double, 2> subMatRef =
            modelFwdState[GRU_FWD_STATE_ACTS_IDX][s];
        bool ret = checkIsClose("nextLayerAct", subMatImp, subMatRef,
                                relativeTolerance, absoluteTolerance);
        if (!ret)
          printf("step = %d\n", s);
        matchesModel &= ret;
      }
    } else {
      boost::multi_array<double, 2> subMatImp(
          boost::extents[batchSize][outputSize]);
      copy(target, dataType, rawHostNextAct[0].get(), subMatImp);
      boost::multi_array<double, 2> subMatRef =
          modelFwdState[GRU_FWD_STATE_ACTS_IDX][sequenceSize - 1];
      matchesModel &= checkIsClose("nextLayerAct", subMatImp, subMatRef,
                                   relativeTolerance, absoluteTolerance);
    }

    if (doBwdPass) {
      copy(target, dataType, rawHostPrevLayerGrads.get(), hostPrevLayerGrads);

      matchesModel &= checkIsClose("prevLayerGrads", hostPrevLayerGrads,
                                   modelPrevLayerGrads, relativeTolerance,
                                   absoluteTolerance);
    }

    if (doWuPass) {
      copy(target, dataType, rawHostWeightsInputDeltas.get(),
           hostWeightsInputDeltas);
      copy(target, dataType, rawHostWeightsOutputDeltas.get(),
           hostWeightsOutputDeltas);
      copy(target, dataType, rawHostBiasDeltas.get(), hostBiasesDeltas);
      boost::multi_array<double, 3> modelWeightsOutputDeltas(
          boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize][outputSize]);
      boost::multi_array<double, 3> modelWeightsInputDeltas(
          boost::extents[BASIC_GRU_CELL_NUM_UNITS][inputSize][outputSize]);
      boost::multi_array<double, 2> modelBiasesDeltas(
          boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
      poplibs_test::gru::basicGruCellParamUpdate(
          hostPrevLayerAct, modelFwdState, hostOutputInit, modelBwdState,
          modelWeightsInputDeltas, modelWeightsOutputDeltas, modelBiasesDeltas);
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
