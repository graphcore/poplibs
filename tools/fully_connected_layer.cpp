// Copyright (c) 2016 Graphcore Ltd, All rights reserved.
#include "TestDevice.hpp"
#include "popops/Cast.hpp"
#include "poputil/exceptions.hpp"
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
#include <poplibs_test/FullyConnected.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popops;
using poplibs_test::Pass;

const OptionFlags defaultEngineOptions{
    {"target.workerStackSizeInBytes", "0x200"},
};

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  unsigned numGroups;
  unsigned inputSize;
  unsigned outputSize;
  unsigned batchSize;
  bool bias;
  bool inPlaceUpdate = true;
  bool reportPlan;
  Type inputType;
  Type outputType;
  double relativeTolerance, absoluteTolerance;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  Pass pass = Pass::ALL;
  bool reportVarStorage = false;
  std::string matmulOptionsString;
  boost::optional<std::string> jsonProfileOut;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type: Cpu | Sim | Hw | IpuModel")
    ("profile", "Output profiling report")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of output channels")
    ("bias", po::value<bool>(&bias)->default_value(true),
     "Add a bias to each output")
    ("data-type",
     po::value<Type>(&inputType)->default_value(HALF),
     "Type of the input and output data")
    ("input-type",
     po::value<Type>(&inputType),
     "Type of the input data")
    ("output-type",
     po::value<Type>(&outputType),
     "Type of the output data")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&numIPUs),
     "Number of IPUs")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("num-groups",
     po::value<unsigned>(&numGroups)->default_value(1),
     "Number of groups in grouped matrix multiplication")
    ("in-place-update",
     po::value<bool>(&inPlaceUpdate)->default_value(true),
     "Perform param update in place")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("report-var-storage",
     po::value<bool>(
       &reportVarStorage
     )->default_value(reportVarStorage),
     "Report tensor storage information "
    )
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
    ("matmul-options", po::value<std::string>(&matmulOptionsString),
     "Options to use for the matrix multiplication, specified as a JSON "
     "string, e.g. {\"key\":\"value\"}")
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

  bool ignoreData = vm.count("ignore-data");
  bool inferenceOnly = vm.count("inference-only");
  if (inferenceOnly && pass != Pass::ALL && pass != Pass::FWD) {
    std::cerr << "pass=" << pass << " specified with --inference-only\n";
    return 1;
  }

  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::BWD);
  bool doWuPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::WU);

  if ((vm["output-type"].empty() != vm["input-type"].empty()) ||
      (!vm["data-type"].defaulted() && !vm["output-type"].empty())) {
    throw poputil::poplibs_error("Please specify either --data-type OR "
                                 "(--input-type AND --output-type), not both.");
  }
  if (vm["output-type"].empty()) {
    outputType = inputType;
  }

  if (vm["tolerance"].empty()) {
    if (outputType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }
  if (outputType == FLOAT) {
    absoluteTolerance = FLOAT_ABS_TOL;
  } else {
    absoluteTolerance = HALF_ABS_TOL;
  }

  const auto learningRate = 0.5;
  auto device = tilesPerIPU.has_value()
                    ? createTestDevice(deviceType, numIPUs, *tilesPerIPU)
                    : createTestDeviceFullSize(deviceType, numIPUs);
  const auto &target = device.getTarget();

  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  OptionFlags fwdOptions;
  if (!matmulOptionsString.empty()) {
    poplar::readJSON(matmulOptionsString, fwdOptions);
  }
  fwdOptions.set("fullyConnectedPass",
                 inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");

  matmul::PlanningCache cache;
  Tensor prevAct = createMatMulGroupedInputLHS(
      graph, inputType, outputType, {numGroups, batchSize, inputSize},
      {numGroups, inputSize, outputSize}, "prevAct", fwdOptions, &cache);
  Tensor weights = createMatMulGroupedInputRHS(
      graph, inputType, outputType, {numGroups, batchSize, inputSize},
      {numGroups, inputSize, outputSize}, "weights", fwdOptions, &cache);
  Tensor biases;
  if (bias) {
    biases = graph.addVariable(outputType, {numGroups, outputSize}, "biases");
    mapTensorLinearly(graph, biases);
  }

  auto bwdOptions = fwdOptions;
  bwdOptions.set("fullyConnectedPass", "TRAINING_BWD");

  auto fwdProg = Sequence();
  auto bwdProg = Sequence();

  Tensor nextAct;
  if (doFwdPass) {
    nextAct = poplin::matMulGrouped(graph, prevAct, weights, fwdProg,
                                    outputType, "Fwd", fwdOptions, &cache);
    if (reportPlan) {
      std::cout << "Forward plan:\n";
      poplin::matMulGroupedReportPlan(std::cout, graph, inputType, outputType,
                                      prevAct.shape(), weights.shape(),
                                      fwdOptions, &cache);
    }
    if (bias) {
      auto bBiases =
          biases.reshape({numGroups, 1, outputSize}).broadcast(batchSize, 1);
      addInPlace(graph, nextAct, bBiases, fwdProg);
    }
  } else {
    nextAct = graph.addVariable(inputType, {numGroups, batchSize, outputSize},
                                "nextAct");
    mapTensorLinearly(graph, nextAct);
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct = allocateHostMemoryForTensor(
      prevAct, "prevAct", graph, uploadProg, downloadProg, tmap);
  auto rawHostWeights = allocateHostMemoryForTensor(
      weights, "weights", graph, uploadProg, downloadProg, tmap);
  std::unique_ptr<char[]> rawHostBiases;
  if (bias) {
    rawHostBiases = allocateHostMemoryForTensor(biases, "biases", graph,
                                                uploadProg, downloadProg, tmap);
  }
  auto rawHostNextAct = allocateHostMemoryForTensor(
      nextAct, "nextAct", graph, uploadProg, downloadProg, tmap);

  Tensor zDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas = poplin::createMatMulGroupedInputLHS(
        graph, inputType, outputType, {numGroups, batchSize, outputSize},
        {numGroups, outputSize, inputSize}, "zDeltas", bwdOptions, &cache);
    rawHostZDeltas = allocateHostMemoryForTensor(
        zDeltas, "zDeltas", graph, uploadProg, downloadProg, tmap);
  }
  Tensor prevDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass) {
    auto weightsTransposed = poplin::transposeGroupedMatrix(weights);
    prevDeltas =
        poplin::matMulGrouped(graph, zDeltas, weightsTransposed, bwdProg,
                              outputType, "Bwd", bwdOptions, &cache);
    if (reportPlan) {
      std::cout << "Backward plan:\n";
      poplin::matMulGroupedReportPlan(
          std::cout, graph, inputType, outputType, zDeltas.shape(),
          weightsTransposed.shape(), bwdOptions, &cache);
    }
    rawHostPrevDeltas = allocateHostMemoryForTensor(
        prevDeltas, "prevDeltas", graph, uploadProg, downloadProg, tmap);
  }
  if (doWuPass) {
    auto wuOptions = fwdOptions;
    wuOptions.set("fullyConnectedPass", "TRAINING_WU");
    auto prevActTransposed = poplin::transposeGroupedMatrix(prevAct);
    auto scale = graph.addConstant(weights.elementType(), {}, -learningRate);
    // the check on groups is done to exercise both grouped and ungrouped
    // variants of matmul
    if (numGroups == 1) {
      poplin::matMulAcc(graph, weights.squeeze({0}), scale,
                        prevActTransposed.squeeze({0}), zDeltas.squeeze({0}),
                        bwdProg, "Wu", wuOptions, &cache);
      weights.expand({0});
    } else {
      poplin::matMulGroupedAcc(graph, weights, scale, prevActTransposed,
                               zDeltas, bwdProg, "Wu", wuOptions, &cache);
    }
    if (reportPlan) {
      std::cout << "WU plan:\n";
      poplin::matMulGroupedReportPlan(std::cout, graph, inputType, outputType,
                                      prevActTransposed.shape(),
                                      zDeltas.shape(), wuOptions, &cache);
    }
    if (bias) {
      auto biasDeltas =
          reduce(graph, zDeltas, {1}, popops::Operation::ADD, bwdProg);
      const auto biasDeltasOut = cast(graph, biasDeltas, outputType, bwdProg);
      scaledAddTo(graph, biases, biasDeltasOut, -learningRate, bwdProg);
    }
  }

  std::vector<Program> programs;
  const auto fwdProgIndex = programs.size();
  programs.push_back(std::move(fwdProg));
  const auto bwdProgIndex = programs.size();
  programs.push_back(std::move(bwdProg));
  const auto uploadProgIndex = programs.size();
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size();
  programs.push_back(std::move(downloadProg));
  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Engine engine(graph, std::move(programs), engineOptions);
  attachStreams(engine, tmap);

  boost::multi_array<double, 3> hostPrevAct(
      boost::extents[numGroups][batchSize][inputSize]);
  boost::multi_array<double, 3> hostWeights(
      boost::extents[numGroups][inputSize][outputSize]);
  boost::multi_array<double, 2> hostBiases(
      boost::extents[numGroups][outputSize]);
  boost::multi_array<double, 3> hostNextAct(
      boost::extents[numGroups][batchSize][outputSize]);
  std::mt19937 randomEngine;
  writeRandomValues(target, inputType, hostPrevAct, -4.0, 4.0, randomEngine);
  writeRandomValues(target, inputType, hostWeights, -3.0, 3.0, randomEngine);
  if (bias) {
    writeRandomValues(target, outputType, hostBiases, -4.0, 4.0, randomEngine);
  } else {
    std::fill(hostBiases.data(), hostBiases.data() + hostBiases.num_elements(),
              0.0);
  }
  copy(target, hostPrevAct, inputType, rawHostPrevAct.get());
  copy(target, hostWeights, inputType, rawHostWeights.get());
  if (bias) {
    copy(target, hostBiases, outputType, rawHostBiases.get());
  }
  // Run the forward pass.
  device.bind([&](const Device &d) {
    engine.load(d);
    if (!ignoreData) {
      engine.run(uploadProgIndex);
    }
    engine.run(fwdProgIndex); // Run.
    if (!ignoreData) {
      engine.run(downloadProgIndex);
    }
  });
  copy(target, outputType, rawHostNextAct.get(), hostNextAct);

  // Validate against a reference model.
  bool matchesModel = true;
  if (!ignoreData) {
    if (doFwdPass) {
      boost::multi_array<double, 3> modelNextAct(
          boost::extents[numGroups][batchSize][outputSize]);
      poplibs_test::fc::fullyConnected(hostPrevAct, hostWeights, hostBiases,
                                       modelNextAct);
      matchesModel &= checkIsClose("fwd", hostNextAct, modelNextAct,
                                   relativeTolerance, absoluteTolerance);
    }
  }
  if (doBwdPass || doWuPass) {
    boost::multi_array<double, 3> hostZDeltas(
        boost::extents[numGroups][batchSize][outputSize]);
    boost::multi_array<double, 3> hostPrevDeltas(
        boost::extents[numGroups][batchSize][inputSize]);
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards pass.
    writeRandomValues(target, inputType, hostZDeltas, -5.0, 5.0, randomEngine);
    copy(target, hostZDeltas, inputType, rawHostZDeltas.get());
    device.bind([&](const Device &d) {
      engine.load(d);
      if (!ignoreData) {
        engine.run(uploadProgIndex);
      }
      engine.run(bwdProgIndex); // Run.
      if (!ignoreData) {
        engine.run(downloadProgIndex);
      }
    });

    if (!ignoreData) {
      // Validate against a reference model.
      if (doBwdPass) {
        copy(target, outputType, rawHostPrevDeltas.get(), hostPrevDeltas);
        boost::multi_array<double, 3> modelPrevDeltas(
            boost::extents[numGroups][batchSize][inputSize]);
        poplibs_test::fc::fullyConnectedBackward(hostZDeltas, modelWeights,
                                                 modelPrevDeltas);
        matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                     relativeTolerance, absoluteTolerance);
      }
      if (doWuPass) {
        copy(target, inputType, rawHostWeights.get(), hostWeights);
        if (bias) {
          copy(target, outputType, rawHostBiases.get(), hostBiases);
        }
        poplibs_test::fc::fullyConnectedWeightUpdate(
            learningRate, hostPrevAct, hostZDeltas, modelWeights, modelBiases);
        matchesModel &= checkIsClose("weights", hostWeights, modelWeights,
                                     relativeTolerance, absoluteTolerance);
        if (bias) {
          matchesModel &= checkIsClose("biases", hostBiases, modelBiases,
                                       relativeTolerance, absoluteTolerance);
        }
      }
    }
  }

  if (jsonProfileOut) {
    const auto pr = engine.getProfile();

    std::ofstream os(*jsonProfileOut);
    poplar::serializeToJSON(os, pr);
  }

  if (vm.count("profile")) {
    OptionFlags opt = {{"showExecutionSteps", "true"}};
    if (reportVarStorage) {
      opt.set("showVarStorage", "true");
    }

    engine.printProfileSummary(std::cout, opt);
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
