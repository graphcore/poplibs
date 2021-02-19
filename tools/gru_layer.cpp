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
#define FLOAT_REL_TOL 0.10
#define HALF_REL_TOL 0.3
#define FLOAT_ABS_TOL 1e-5
#define HALF_ABS_TOL 7e-2

const OptionFlags defaultEngineOptions;

std::ostream &operator<<(std::ostream &os, const BasicGruCellUnit u) {
  switch (u) {
  case BASIC_GRU_CELL_RESET_GATE:
    return os << "reset";
  case BASIC_GRU_CELL_UPDATE_GATE:
    return os << "update";
  case BASIC_GRU_CELL_CANDIDATE:
    return os << "cell";
  case BASIC_GRU_CELL_NUM_UNITS:
    break;
  }

  throw poputil::poplibs_error("Invalid unit");
}

std::istream &operator>>(std::istream &is, BasicGruCellUnit &u) {
  std::string token;
  is >> token;

  if (token == "reset") {
    u = BASIC_GRU_CELL_RESET_GATE;
  } else if (token == "update") {
    u = BASIC_GRU_CELL_UPDATE_GATE;
  } else if (token == "cell") {
    u = BASIC_GRU_CELL_CANDIDATE;
  } else {
    throw poputil::poplibs_error("Invalid token for unit: " + token);
  }

  return is;
}

std::vector<BasicGruCellUnit> getCellOrder(const std::vector<std::string> &in) {
  std::vector<BasicGruCellUnit> cellOrder;
  for (const auto &x : in) {
    cellOrder.emplace_back();

    std::stringstream ss(x);
    ss >> cellOrder.back();
  }

  return cellOrder;
}

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
  DeviceType deviceType = DeviceType::IpuModel2;

  unsigned sequenceSize, inputSize, outputSize;
  unsigned batchSize = 1;
  unsigned numShards = 1;

  Type dataType;
  Type partialsType;
  double relativeTolerance;
  double absoluteTolerance;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  bool outputAllSequence = true;
  bool withAttention = false;
  bool withRealTimeSteps = false;
  poplibs_test::Pass pass = poplibs_test::Pass::FWD;
  unsigned runs = 1;
  std::string profileDir = ".";
  double availableMemoryProportion;
  ShapeOption<std::string> cellOrder;
  boost::optional<std::string> jsonProfileOut;
  boost::optional<std::string> profileFormat;
  bool resetAfter = false;

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
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("use-unstable-format", "Deprecated: use \"--profile-format experimental\"")
    ("profile-format",
     po::value<decltype(profileFormat)>(&profileFormat)
      ->default_value(boost::none),
     "Profile formats: v1 | experimental | unstable")
    ("sequence-size", po::value<unsigned>(&sequenceSize)->required(),
     "Sequence size in the RNN")
    ("shards", po::value<unsigned>(&numShards)->required(),
     "The number of shards")
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
    ("with-attention",
     po::value<bool>(&withAttention)->default_value(withAttention),
     "true with attention, otherwise without")
    ("with-real-time-steps",
     po::value<bool>(&withRealTimeSteps)->default_value(withRealTimeSteps),
     "true with realTimeSteps, otherwise without")
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
    ("cell-order",
     po::value<ShapeOption<std::string>>(&cellOrder)->default_value(cellOrder),
     "The order that the gates are stored in the weights and bias tensors")
     ("reset-after",
     po::value<bool>(&resetAfter)->default_value(resetAfter),
      "Apply reset gate after matrix multiplication (1 / 0)")
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
  if (vm.count("use-unstable-format")) {
    throw poputil::poplibs_error("\"--use-unstable-format\" is deprecated. Use "
                                 "\"--profile-format experimental\" instead");
  }

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
  if (!cellOrder.val.empty()) {
    params.cellOrder = getCellOrder(cellOrder.val);
  }
  params.resetAfter = resetAfter;

  poplar::OptionFlags options = {
      {"inferenceOnly", fwdOnly ? "true" : "false"},
      {"numShards", std::to_string(numShards)},
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

  Tensor realTimeSteps;
  if (withRealTimeSteps) {
    realTimeSteps =
        graph.addVariable(INT, {params.rnn.batchSize}, "realTimeSteps");
    graph.setTileMapping(realTimeSteps, 0);
  }

  Tensor attScores;
  if (withAttention) {
    attScores = gru::createAttention(graph, params, "attScores", options);
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  Tensor fwdOutputSeq, fwdIntermediates;
  Tensor *fwdIntermediatesPtr =
      (doBwdPass || doWuPass) ? &fwdIntermediates : nullptr;

  if ((!withAttention) && (!withRealTimeSteps)) {
    fwdOutputSeq =
        popnn::gru::gruFwd(graph, params, outputInit, input, weights,
                           fwdIntermediatesPtr, prog, "fwd", options, &cache);
  } else if ((!withAttention) && (withRealTimeSteps)) {
    fwdOutputSeq = popnn::gru::gruFwd(
        graph, params, outputInit, input, realTimeSteps, weights,
        fwdIntermediatesPtr, prog, "fwd", options, &cache);
  } else if (withAttention && (!withRealTimeSteps)) {
    fwdOutputSeq = popnn::gru::auGruFwd(graph, params, outputInit, input,
                                        weights, fwdIntermediatesPtr, attScores,
                                        prog, "fwd", options, &cache);
  } else if (withAttention && withRealTimeSteps) {
    fwdOutputSeq = popnn::gru::auGruFwd(
        graph, params, outputInit, input, realTimeSteps, weights,
        fwdIntermediatesPtr, attScores, prog, "fwd", options, &cache);
  }

  auto nextLayerGrads = graph.addVariable(
      dataType, {sequenceSize, batchSize, outputSize}, "nextLayerGrads");
  mapTensorLinearly(graph, nextLayerGrads);

  Tensor attScoresGrads;
  Tensor prevLayerGrads;
  gru::GruWeights weightGrads =
      gru::createWeights(graph, params, "weightGrad", options, &cache);
  if (doBwdPass || doWuPass) {
    if (doWuPass) {
      if (params.outputFullSequence) {
        if ((!withAttention) && (!withRealTimeSteps)) {
          gru::gruBwdWithWU(graph, params, prog, outputInit, fwdIntermediates,
                            weights, input, fwdOutputSeq, nextLayerGrads,
                            &prevLayerGrads, weightGrads, "bwd", options,
                            &cache);
        } else if ((!withAttention) && withRealTimeSteps) {
          gru::gruBwdWithWU(graph, params, prog, outputInit, fwdIntermediates,
                            weights, input, realTimeSteps, fwdOutputSeq,
                            nextLayerGrads, &prevLayerGrads, weightGrads, "bwd",
                            options, &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          gru::auGruBwdWithWU(graph, params, prog, outputInit, fwdIntermediates,
                              weights, input, fwdOutputSeq, nextLayerGrads,
                              &prevLayerGrads, weightGrads, attScores,
                              &attScoresGrads, "bwd", options, &cache);
        } else if (withAttention && withRealTimeSteps) {
          gru::auGruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads, &prevLayerGrads,
              weightGrads, attScores, &attScoresGrads, "bwd", options, &cache);
        }
      } else {
        // If only output the last cell, nextLayerGrads only contains the
        // gradient for the last cell.
        if ((!withAttention) && (!withRealTimeSteps)) {
          gru::gruBwdWithWU(graph, params, prog, outputInit, fwdIntermediates,
                            weights, input, fwdOutputSeq, nextLayerGrads[0],
                            &prevLayerGrads, weightGrads, "bwd", options,
                            &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          gru::auGruBwdWithWU(graph, params, prog, outputInit, fwdIntermediates,
                              weights, input, fwdOutputSeq, nextLayerGrads[0],
                              &prevLayerGrads, weightGrads, attScores,
                              &attScoresGrads, "bwd", options, &cache);
        }
      }
    } else {
      if (params.outputFullSequence) {
        if ((!withAttention) && (!withRealTimeSteps)) {
          gru::gruBwd(graph, params, prog, outputInit, fwdIntermediates,
                      weights, input, fwdOutputSeq, nextLayerGrads,
                      &prevLayerGrads, nullptr, "bwd", options, &cache);
        } else if ((!withAttention) && withRealTimeSteps) {
          gru::gruBwd(graph, params, prog, outputInit, fwdIntermediates,
                      weights, input, realTimeSteps, fwdOutputSeq,
                      nextLayerGrads, &prevLayerGrads, nullptr, "bwd", options,
                      &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          gru::auGruBwd(graph, params, prog, outputInit, fwdIntermediates,
                        weights, input, fwdOutputSeq, nextLayerGrads,
                        &prevLayerGrads, nullptr, attScores, &attScoresGrads,
                        "bwd", options, &cache);
        } else if (withAttention && withRealTimeSteps) {
          gru::auGruBwd(graph, params, prog, outputInit, fwdIntermediates,
                        weights, input, realTimeSteps, fwdOutputSeq,
                        nextLayerGrads, &prevLayerGrads, nullptr, attScores,
                        &attScoresGrads, "bwd", options, &cache);
        }
      }

      else {
        if ((!withAttention) && (!withRealTimeSteps)) {
          gru::gruBwd(graph, params, prog, outputInit, fwdIntermediates,
                      weights, input, fwdOutputSeq, nextLayerGrads[0],
                      &prevLayerGrads, nullptr, "bwd", options, &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          gru::auGruBwd(graph, params, prog, outputInit, fwdIntermediates,
                        weights, input, fwdOutputSeq, nextLayerGrads[0],
                        &prevLayerGrads, nullptr, attScores, &attScoresGrads,
                        "bwd", options, &cache);
        }
      }
    }
  }

  std::unique_ptr<char[]> rawHostWeightsInput;
  std::unique_ptr<char[]> rawHostWeightsOutput;
  std::unique_ptr<char[]> rawHostPrevLayerAct;
  std::unique_ptr<char[]> rawHostBiases;
  std::unique_ptr<char[]> rawHostRecurrantBiases;
  std::unique_ptr<char[]> rawHostOutputInit;

  std::unique_ptr<char[]> rawHostRealTimeSteps;
  std::unique_ptr<char[]> rawHostAttScores;
  std::unique_ptr<char[]> rawHostAttGrads;

  std::unique_ptr<char[]> rawHostNextLayerGrads;
  std::unique_ptr<char[]> rawHostPrevLayerGrads;
  std::unique_ptr<char[]> rawHostWeightsInputDeltas;
  std::unique_ptr<char[]> rawHostWeightsOutputDeltas;
  std::unique_ptr<char[]> rawHostBiasDeltas;
  std::unique_ptr<char[]> rawHostRecurrantBiasDeltas;

  std::unique_ptr<char[]> rawHostNextAct;

  if (!ignoreData) {
    rawHostWeightsInput =
        allocateHostMemoryForTensor(weights.inputWeights, "weightsInput", graph,
                                    uploadProg, downloadProg, tmap);
    rawHostWeightsOutput =
        allocateHostMemoryForTensor(weights.outputWeights, "weightsOutput",
                                    graph, uploadProg, downloadProg, tmap);
    rawHostPrevLayerAct = allocateHostMemoryForTensor(
        input, "prevLayerAct", graph, uploadProg, downloadProg, tmap);
    rawHostOutputInit = allocateHostMemoryForTensor(
        outputInit, "outputInit", graph, uploadProg, downloadProg, tmap);

    if (withRealTimeSteps) {
      rawHostRealTimeSteps =
          allocateHostMemoryForTensor(realTimeSteps, "realTimeSteps", graph,
                                      uploadProg, downloadProg, tmap);
    }
    if (withAttention) {
      rawHostAttScores = allocateHostMemoryForTensor(
          attScores, "attScores", graph, uploadProg, downloadProg, tmap);
    }

    if (doBwdPass) {
      rawHostNextLayerGrads =
          allocateHostMemoryForTensor(nextLayerGrads, "nextLayerGrads", graph,
                                      uploadProg, downloadProg, tmap);
      rawHostPrevLayerGrads =
          allocateHostMemoryForTensor(prevLayerGrads, "prevLayerGrads", graph,
                                      uploadProg, downloadProg, tmap);

      if (withAttention) {
        rawHostAttGrads =
            allocateHostMemoryForTensor(attScoresGrads, "attScoresGrads", graph,
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
    }
    if (params.resetAfter) {
      auto biases = weights.biases.slice(0, 1, 1).squeeze({1});
      auto recurrantBiases = weights.biases.slice(1, 2, 1).squeeze({1});
      auto biasesDeltas = weightGrads.biases.slice(0, 1, 1).squeeze({1});
      auto recurrantBiasesDeltas =
          weightGrads.biases.slice(1, 2, 1).squeeze({1});
      rawHostBiases = allocateHostMemoryForTensor(
          biases, "biases", graph, uploadProg, downloadProg, tmap);
      rawHostRecurrantBiases =
          allocateHostMemoryForTensor(recurrantBiases, "recurrantBiases", graph,
                                      uploadProg, downloadProg, tmap);
      rawHostBiasDeltas = allocateHostMemoryForTensor(
          biasesDeltas, "biasDeltas", graph, uploadProg, downloadProg, tmap);
      rawHostRecurrantBiasDeltas = allocateHostMemoryForTensor(
          recurrantBiasesDeltas, "recurrantBiasDeltas", graph, uploadProg,
          downloadProg, tmap);
    } else {
      rawHostBiases = allocateHostMemoryForTensor(
          weights.biases, "biases", graph, uploadProg, downloadProg, tmap);
      rawHostBiasDeltas =
          allocateHostMemoryForTensor(weightGrads.biases, "biasDeltas", graph,
                                      uploadProg, downloadProg, tmap);
    }

    rawHostNextAct = allocateHostMemoryForTensor(
        fwdOutputSeq, "nextAct", graph, uploadProg, downloadProg, tmap);
  }

  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile") || jsonProfileOut) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (profileFormat) {
      engineOptions.set("profiler.format", *profileFormat);
    }
  }
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);

  if (vm.count("compile-only"))
    return 0;

  attachStreams(engine, tmap);

  boost::multi_array<double, 3> hostPrevLayerAct(
      boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3> hostWeightsOutput(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsInput(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][inputSize][outputSize]);
  boost::multi_array<double, 2> hostBiases(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2> hostRecurrantBiases(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2> hostOutputInit(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 4> modelFwdState(
      boost::extents[GRU_NUM_FWD_STATES][sequenceSize][batchSize][outputSize]);

  boost::multi_array<int, 1> hostRealTimeSteps(boost::extents[batchSize]);
  boost::multi_array<double, 2> hostAttScores(
      boost::extents[batchSize][sequenceSize]);
  boost::multi_array<double, 2> hostAttScoresGrads(
      boost::extents[batchSize][sequenceSize]);
  boost::multi_array<double, 2> modelAttScoresGrads(
      boost::extents[batchSize][sequenceSize]);

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
  boost::multi_array<double, 2> hostRecurrantBiasesDeltas(
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

    if (withRealTimeSteps) {
      writeRandomValues(target, poplar::INT, hostRealTimeSteps, 1,
                        int(sequenceSize), randomEngine);
    }

    if (withAttention) {
      writeRandomValues(target, dataType, hostAttScores, 0.0, 1.0,
                        randomEngine);
    }

    if (doBwdPass) {
      writeRandomValues(target, dataType, hostNextLayerGrads, -2.0, 2.0,
                        randomEngine);
    }
    if (resetAfter) {
      writeRandomValues(target, dataType, hostRecurrantBiases, -1.0, 1.0,
                        randomEngine);
    }

    copy(target, hostPrevLayerAct, dataType, rawHostPrevLayerAct.get());
    copy(target, hostOutputInit, dataType, rawHostOutputInit.get());
    copy(target, hostBiases, dataType, rawHostBiases.get());
    copy(target, hostWeightsInput, dataType, rawHostWeightsInput.get());
    copy(target, hostWeightsOutput, dataType, rawHostWeightsOutput.get());

    if (withRealTimeSteps) {
      copy(target, hostRealTimeSteps, poplar::INT, rawHostRealTimeSteps.get());
    }

    if (withAttention) {
      copy(target, hostAttScores, dataType, rawHostAttScores.get());
    }

    if (doBwdPass) {
      copy(target, hostNextLayerGrads, dataType, rawHostNextLayerGrads.get());
    }
    if (resetAfter) {
      copy(target, hostRecurrantBiases, dataType, rawHostRecurrantBiases.get());
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

  boost::optional<boost::multi_array_ref<double, 2>> hostAttScoresOpt;
  boost::optional<boost::multi_array_ref<int, 1>> hostRealTimeStepsOpt;
  boost::optional<boost::multi_array_ref<double, 2>> hostAttScoresGradsOpt;

  bool matchesModel = true;
  if (!ignoreData) {
    if ((!withAttention) && (!withRealTimeSteps)) {
      ;
    } else if ((!withAttention) && withRealTimeSteps) {
      hostRealTimeStepsOpt = hostRealTimeSteps;
    } else if (withAttention && (!withRealTimeSteps)) {
      hostAttScoresOpt = hostAttScores;
      hostAttScoresGradsOpt = modelAttScoresGrads;
    } else if (withAttention && withRealTimeSteps) {
      hostAttScoresOpt = hostAttScores;
      hostRealTimeStepsOpt = hostRealTimeSteps;
      hostAttScoresGradsOpt = modelAttScoresGrads;
    }

    if (params.resetAfter) {
      poplibs_test::gru::basicGruCellForwardPass(
          params.outputFullSequence, hostPrevLayerAct, hostBiases,
          hostOutputInit, hostWeightsInput, hostWeightsOutput, hostAttScoresOpt,
          hostRealTimeStepsOpt, modelFwdState, params.cellOrder, true,
          hostRecurrantBiases);
    } else {
      poplibs_test::gru::basicGruCellForwardPass(
          params.outputFullSequence, hostPrevLayerAct, hostBiases,
          hostOutputInit, hostWeightsInput, hostWeightsOutput, hostAttScoresOpt,
          hostRealTimeStepsOpt, modelFwdState, params.cellOrder, false);
    }

    if (doBwdPass) {
      if (params.resetAfter) {
        poplibs_test::gru::basicGruCellBackwardPass(
            params.outputFullSequence, hostWeightsInput, hostWeightsOutput,
            hostNextLayerGrads, modelFwdState, hostOutputInit,
            hostRealTimeStepsOpt, hostAttScoresOpt, hostAttScoresGradsOpt,
            modelBwdState, modelPrevLayerGrads, params.cellOrder, true,
            hostRecurrantBiases);
      } else {
        poplibs_test::gru::basicGruCellBackwardPass(
            params.outputFullSequence, hostWeightsInput, hostWeightsOutput,
            hostNextLayerGrads, modelFwdState, hostOutputInit,
            hostRealTimeStepsOpt, hostAttScoresOpt, hostAttScoresGradsOpt,
            modelBwdState, modelPrevLayerGrads, params.cellOrder, false);
      }
    }

    boost::multi_array<double, 3> matImpl(
        boost::extents[sequenceSize][batchSize][outputSize]);
    copy(target, dataType, rawHostNextAct.get(), matImpl);
    if (params.outputFullSequence) {
      for (auto s = 0U; s != sequenceSize; ++s) {
        const boost::multi_array<double, 2> subMatImpl = matImpl[s];
        const boost::multi_array<double, 2> subMatRef =
            modelFwdState[GRU_FWD_STATE_ACTS_IDX][s];
        bool ret = checkIsClose("nextLayerAct", subMatImpl, subMatRef,
                                relativeTolerance, absoluteTolerance);
        if (!ret)
          printf("step = %d\n", s);
        matchesModel &= ret;
      }
    } else {
      const boost::multi_array<double, 2> subMatImpl = matImpl[0];
      const boost::multi_array<double, 2> subMatRef =
          modelFwdState[GRU_FWD_STATE_ACTS_IDX][sequenceSize - 1];
      matchesModel &= checkIsClose("nextLayerAct", subMatImpl, subMatRef,
                                   relativeTolerance, absoluteTolerance);
    }

    if (doBwdPass) {
      copy(target, dataType, rawHostPrevLayerGrads.get(), hostPrevLayerGrads);

      matchesModel &= checkIsClose("prevLayerGrads", hostPrevLayerGrads,
                                   modelPrevLayerGrads, relativeTolerance,
                                   absoluteTolerance);
      if (withAttention) {
        copy(target, dataType, rawHostAttGrads.get(), hostAttScoresGrads);
        matchesModel &= checkIsClose("attScoresGrads", hostAttScoresGrads,
                                     modelAttScoresGrads, relativeTolerance,
                                     absoluteTolerance);
      }
    }

    if (doWuPass) {
      copy(target, dataType, rawHostWeightsInputDeltas.get(),
           hostWeightsInputDeltas);
      copy(target, dataType, rawHostWeightsOutputDeltas.get(),
           hostWeightsOutputDeltas);
      copy(target, dataType, rawHostBiasDeltas.get(), hostBiasesDeltas);
      if (resetAfter) {
        copy(target, dataType, rawHostRecurrantBiasDeltas.get(),
             hostRecurrantBiasesDeltas);
      }
      boost::multi_array<double, 3> modelWeightsOutputDeltas(
          boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize][outputSize]);
      boost::multi_array<double, 3> modelWeightsInputDeltas(
          boost::extents[BASIC_GRU_CELL_NUM_UNITS][inputSize][outputSize]);
      boost::multi_array<double, 2> modelBiasesDeltas(
          boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
      boost::multi_array<double, 2> modelRecurrantBiasesDeltas(
          boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);

      if (params.resetAfter) {
        poplibs_test::gru::basicGruCellParamUpdate(
            hostPrevLayerAct, modelFwdState, hostOutputInit, modelBwdState,
            modelWeightsInputDeltas, modelWeightsOutputDeltas,
            modelBiasesDeltas, params.cellOrder, true,
            modelRecurrantBiasesDeltas);
      } else {
        poplibs_test::gru::basicGruCellParamUpdate(
            hostPrevLayerAct, modelFwdState, hostOutputInit, modelBwdState,
            modelWeightsInputDeltas, modelWeightsOutputDeltas,
            modelBiasesDeltas, params.cellOrder, false);
      }

      matchesModel &= checkIsClose("weightsInputDeltas", hostWeightsInputDeltas,
                                   modelWeightsInputDeltas, relativeTolerance,
                                   absoluteTolerance);
      matchesModel &= checkIsClose(
          "weightsOutputDeltas", hostWeightsOutputDeltas,
          modelWeightsOutputDeltas, relativeTolerance, absoluteTolerance);
      matchesModel &=
          checkIsClose("biasDeltas", hostBiasesDeltas, modelBiasesDeltas,
                       relativeTolerance, absoluteTolerance);
      if (params.resetAfter) {
        matchesModel &= checkIsClose(
            "recurrantBiasDeltas", hostRecurrantBiasesDeltas,
            modelRecurrantBiasesDeltas, relativeTolerance, absoluteTolerance);
      }
    }
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
