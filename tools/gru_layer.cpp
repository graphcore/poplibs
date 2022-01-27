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
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/Gru.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/codelets.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
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

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  DeviceType deviceType = DeviceType::IpuModel2;

  unsigned sequenceSize, inputSize, outputSize;
  unsigned batchSize = 1;
  unsigned numShards = 1;
  bool codeReuse = false;

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
  boost::optional<std::string> profileDir;
  double availableMemoryProportion;
  ShapeOption<std::size_t> variableTimeStepsOption;
  ShapeOption<std::string> cellOrder;
  bool resetAfter = false;
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
      po::value<decltype(profileDir)>(&profileDir)->default_value(boost::none),
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
     "Force GRU code reuse")
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
     ("reset-after",
     po::value<bool>(&resetAfter)->default_value(resetAfter),
      "Apply reset gate after matrix multiplication (1 / 0)")
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
  if (varTimeSteps.size() && withRealTimeSteps) {
    throw poputil::poplibs_error("variable-time-steps cannot be used together "
                                 " with with-real-time-steps");
  }
  if ((varTimeSteps.size() > 1) && (varTimeSteps.size() != batchSize)) {
    throw poputil::poplibs_error("timeSteps must either be a scalar or a "
                                 "tensor of batch-size length");
  }
  Tensor timeSteps;
  if (varTimeSteps.size()) {
    timeSteps = graph.addVariable(UNSIGNED_INT, {varTimeSteps.size()},
                                  "var-time-steps");
    mapTensorLinearly(graph, timeSteps);
  }

  poplin::matmul::PlanningCache cache;
  gru::GruParams params(dataType, batchSize, sequenceSize, timeSteps,
                        {inputSize, outputSize}, activation,
                        recurrentActivation);
  params.outputFullSequence = outputAllSequence;
  if (!cellOrder.val.empty()) {
    params.cellOrder = getCellOrder(cellOrder.val);
  }
  params.resetAfter = resetAfter;
  if (ignoreInputGradient) {
    params.calcInputGradients = false;
  }

  poplar::OptionFlags options = {
      {"inferenceOnly", fwdOnly ? "true" : "false"},
  };
  if (!vm["shards"].empty()) {
    options.set("numShards", std::to_string(numShards));
  }
  if (!vm["code-reuse"].empty()) {
    options.set("rnnCodeReuse", codeReuse ? "true" : "false");
  }
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
    realTimeSteps = graph.addVariable(UNSIGNED_INT, {params.rnn.batchSize},
                                      "realTimeSteps");
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
  Tensor lastGradLayerOut;
  Tensor *inputGrad = ignoreInputGradient ? nullptr : &prevLayerGrads;
  gru::GruWeights weightGrads =
      gru::createWeights(graph, params, "weightGrad", options, &cache);
  if (doBwdPass || doWuPass) {
    if (doWuPass) {
      if (params.outputFullSequence) {
        if ((!withAttention) && (!withRealTimeSteps)) {
          lastGradLayerOut = gru::gruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              fwdOutputSeq, nextLayerGrads, inputGrad, weightGrads, "bwd",
              options, &cache);
        } else if ((!withAttention) && withRealTimeSteps) {
          lastGradLayerOut = gru::gruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads, inputGrad,
              weightGrads, "bwd", options, &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          lastGradLayerOut = gru::auGruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              fwdOutputSeq, nextLayerGrads, inputGrad, weightGrads, attScores,
              &attScoresGrads, "bwd", options, &cache);
        } else if (withAttention && withRealTimeSteps) {
          lastGradLayerOut = gru::auGruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads, inputGrad,
              weightGrads, attScores, &attScoresGrads, "bwd", options, &cache);
        }
      } else {
        // If only output the last cell, nextLayerGrads only contains the
        // gradient for the last cell.
        if ((!withAttention) && (!withRealTimeSteps)) {
          lastGradLayerOut = gru::gruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              fwdOutputSeq, nextLayerGrads[0], inputGrad, weightGrads, "bwd",
              options, &cache);
        } else if ((!withAttention) && withRealTimeSteps) {
          lastGradLayerOut = gru::gruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads[0], inputGrad,
              weightGrads, "bwd", options, &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          lastGradLayerOut = gru::auGruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              fwdOutputSeq, nextLayerGrads[0], inputGrad, weightGrads,
              attScores, &attScoresGrads, "bwd", options, &cache);
        } else if (withAttention && withRealTimeSteps) {
          lastGradLayerOut = gru::auGruBwdWithWU(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads[0], inputGrad,
              weightGrads, attScores, &attScoresGrads, "bwd", options, &cache);
        }
      }
    } else {
      if (params.outputFullSequence) {
        if ((!withAttention) && (!withRealTimeSteps)) {
          lastGradLayerOut =
              gru::gruBwd(graph, params, prog, outputInit, fwdIntermediates,
                          weights, input, fwdOutputSeq, nextLayerGrads,
                          inputGrad, nullptr, "bwd", options, &cache);
        } else if ((!withAttention) && withRealTimeSteps) {
          lastGradLayerOut = gru::gruBwd(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads, inputGrad, nullptr,
              "bwd", options, &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          lastGradLayerOut = gru::auGruBwd(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              fwdOutputSeq, nextLayerGrads, inputGrad, nullptr, attScores,
              &attScoresGrads, "bwd", options, &cache);
        } else if (withAttention && withRealTimeSteps) {
          lastGradLayerOut = gru::auGruBwd(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads, inputGrad, nullptr,
              attScores, &attScoresGrads, "bwd", options, &cache);
        }
      } else {
        if ((!withAttention) && (!withRealTimeSteps)) {
          lastGradLayerOut =
              gru::gruBwd(graph, params, prog, outputInit, fwdIntermediates,
                          weights, input, fwdOutputSeq, nextLayerGrads[0],
                          inputGrad, nullptr, "bwd", options, &cache);
        } else if ((!withAttention) && withRealTimeSteps) {
          lastGradLayerOut = gru::gruBwd(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads[0], inputGrad,
              nullptr, "bwd", options, &cache);
        } else if (withAttention && (!withRealTimeSteps)) {
          lastGradLayerOut = gru::auGruBwd(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              fwdOutputSeq, nextLayerGrads[0], inputGrad, nullptr, attScores,
              &attScoresGrads, "bwd", options, &cache);
        } else if (withAttention && withRealTimeSteps) {
          lastGradLayerOut = gru::auGruBwd(
              graph, params, prog, outputInit, fwdIntermediates, weights, input,
              realTimeSteps, fwdOutputSeq, nextLayerGrads[0], inputGrad,
              nullptr, attScores, &attScoresGrads, "bwd", options, &cache);
        }
      }
    }
  }

  std::unique_ptr<char[]> rawTimeSteps;
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
  std::unique_ptr<char[]> rawGradPrevLayerOut;

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
      if (!ignoreInputGradient) {
        rawHostPrevLayerGrads =
            allocateHostMemoryForTensor(prevLayerGrads, "prevLayerGrads", graph,
                                        uploadProg, downloadProg, tmap);
      }
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
    if (doBwdPass || doWuPass) {
      rawGradPrevLayerOut =
          allocateHostMemoryForTensor(lastGradLayerOut, "lastGradLayerOut",
                                      graph, uploadProg, downloadProg, tmap);
    }
  }

  std::optional<TempDir> tempDir;
  OptionFlags engineOptions = defaultEngineOptions;
  if (vm.count("profile") || profileDir) {
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    if (profileDir) {
      engineOptions.set("autoReport.directory", *profileDir);
    } else {
      tempDir.emplace(TempDir::create());
      engineOptions.set("autoReport.directory", tempDir->getPath());
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
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize][outputSize]);
  boost::multi_array<double, 3> hostWeightsInput(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][inputSize][outputSize]);
  boost::multi_array<double, 2> hostBiases(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2> hostRecurrantBiases(
      boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2> hostOutputInit(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 3> hostNextAct(
      boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 2> modelLastOutput(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2> modelGradPrevLayerOut(
      boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 4> modelFwdState(
      boost::extents[GRU_NUM_FWD_STATES][sequenceSize][batchSize][outputSize]);

  boost::multi_array<unsigned, 1> hostRealTimeSteps(boost::extents[batchSize]);
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

  if (varTimeSteps.size()) {
    copy(varTimeSteps.begin(), varTimeSteps.end(), hostTimeSteps.data());
    copy(target, hostTimeSteps, UNSIGNED_INT, rawTimeSteps.get());
  }

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

    if (params.outputFullSequence) {
      writeRandomValues(target, dataType, hostNextAct, -1.0, 1.0, randomEngine);
    }

    if (!ignoreInputGradient && doBwdPass) {
      writeRandomValues(target, dataType, hostPrevLayerGrads, -1.0, 1.0,
                        randomEngine);
    }

    if (withRealTimeSteps) {
      writeRandomValues(target, UNSIGNED_INT, hostRealTimeSteps, 1U,
                        sequenceSize, randomEngine);
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
      copy(target, hostRealTimeSteps, UNSIGNED_INT, rawHostRealTimeSteps.get());
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
                                     //{ "showExecutionSteps", "true" }
                                     //{ "showVarStorage",     "true" }
                                 });
    }
  }

  boost::optional<boost::multi_array_ref<double, 2>> hostAttScoresOpt;
  boost::optional<boost::multi_array_ref<unsigned, 1>> hostRealTimeStepsOpt;
  boost::optional<boost::multi_array_ref<double, 2>> hostAttScoresGradsOpt;
  bool matchesModel = true;
  if (!ignoreData) {
    if (withRealTimeSteps) {
      hostRealTimeStepsOpt = hostRealTimeSteps;
    } else if (varTimeSteps.size()) {
      hostRealTimeStepsOpt = hostTimeSteps;
    }
    if (withAttention) {
      hostAttScoresOpt = hostAttScores;
      hostAttScoresGradsOpt = modelAttScoresGrads;
    }

    if (params.resetAfter) {
      poplibs_test::gru::basicGruCellForwardPass(
          hostPrevLayerAct, hostBiases, hostOutputInit, hostWeightsInput,
          hostWeightsOutput, hostAttScoresOpt, hostRealTimeStepsOpt,
          modelFwdState, modelLastOutput, params.cellOrder, true,
          hostRecurrantBiases, activation, recurrentActivation);
    } else {
      poplibs_test::gru::basicGruCellForwardPass(
          hostPrevLayerAct, hostBiases, hostOutputInit, hostWeightsInput,
          hostWeightsOutput, hostAttScoresOpt, hostRealTimeStepsOpt,
          modelFwdState, modelLastOutput, params.cellOrder, false, boost::none,
          activation, recurrentActivation);
    }

    if (doBwdPass) {
      if (params.resetAfter) {
        poplibs_test::gru::basicGruCellBackwardPass(
            params.outputFullSequence, hostWeightsInput, hostWeightsOutput,
            hostNextLayerGrads, modelFwdState, hostOutputInit,
            hostRealTimeStepsOpt, hostAttScoresOpt, hostAttScoresGradsOpt,
            modelBwdState, modelPrevLayerGrads, modelGradPrevLayerOut,
            params.cellOrder, true, hostRecurrantBiases, activation,
            recurrentActivation);
      } else {
        poplibs_test::gru::basicGruCellBackwardPass(
            params.outputFullSequence, hostWeightsInput, hostWeightsOutput,
            hostNextLayerGrads, modelFwdState, hostOutputInit,
            hostRealTimeStepsOpt, hostAttScoresOpt, hostAttScoresGradsOpt,
            modelBwdState, modelPrevLayerGrads, modelGradPrevLayerOut,
            params.cellOrder, false, boost::none, activation,
            recurrentActivation);
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
      matchesModel &= checkIsClose("nextLayerAct", subMatImpl, modelLastOutput,
                                   relativeTolerance, absoluteTolerance);
    }

    if (doBwdPass) {
      if (!ignoreInputGradient) {
        copy(target, dataType, rawHostPrevLayerGrads.get(), hostPrevLayerGrads);
        matchesModel &= checkIsClose("prevLayerGrads", hostPrevLayerGrads,
                                     modelPrevLayerGrads, relativeTolerance,
                                     absoluteTolerance);
      }
      if (withAttention) {
        copy(target, dataType, rawHostAttGrads.get(), hostAttScoresGrads);
        matchesModel &= checkIsClose("attScoresGrads", hostAttScoresGrads,
                                     modelAttScoresGrads, relativeTolerance,
                                     absoluteTolerance);
      }
      boost::multi_array<double, 2> hostGradPrevLayerOut(
          boost::extents[batchSize][outputSize]);
      copy(target, dataType, rawGradPrevLayerOut.get(), hostGradPrevLayerOut);
      matchesModel &= checkIsClose("lastGradLayerOut", hostGradPrevLayerOut,
                                   modelGradPrevLayerOut, relativeTolerance,
                                   absoluteTolerance);
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
