// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"
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
#include <poplibs_support/MultiArray.hpp>
#include <poplibs_test/Pooling.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <random>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplibs_support;
using namespace poputil;

using popnn::PoolingType;

namespace popnn {
std::ostream &operator<<(std::ostream &os, const PoolingType &pType) {
  return os << popnn::pooling::asString(pType);
}

std::istream &operator>>(std::istream &is, PoolingType &pType) {
  std::string token;
  is >> token;
  if (token == "max")
    pType = PoolingType::MAX;
  else if (token == "avg")
    pType = PoolingType::AVG;
  else if (token == "sum") {
    pType = PoolingType::SUM;
  } else
    throw poputil::poplibs_error("Unknown pooling type<" + token + ">");
  return is;
}
} // namespace popnn

// For max pool. the gradient is scaled depending on number of activations
// which have the same value. This guarantees that the difference between
// any two activations is either 0 or greater than the minimum half precision.
// This also increases the probability of acts having the same values
static void adjustActivations(MultiArray<double> &acts, unsigned maxValue) {
  double scale = 64.0 / maxValue;
  forEachIndex(acts.shape(), [&](const MultiArrayShapeRange indices) {
    double act = std::floor(acts[indices] * scale) / 64.0 * maxValue;
    acts[indices] = act;
  });
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned chans;
  unsigned fwdChansPerGroup;
  unsigned bwdChansPerGroup;
  unsigned batchSize;
  Type dataType;
  double relativeTolerance, absoluteTolerance;

  ShapeOption<std::size_t> inputFieldSizeOption;
  ShapeOption<std::size_t> kernelSizeOption;
  ShapeOption<unsigned> strideOption;
  ShapeOption<int> paddingLowerOption;
  ShapeOption<int> paddingUpperOption;

  DeviceType deviceType = DeviceType::IpuModel;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  PoolingType poolingType = PoolingType::MAX;

  OptionFlags engineOptions;
  OptionFlags poolingOptions;
  bool useIntrospectiveMapping;
  bool scaledGradientForMaxPool;
  bool optimizeForSpeed;

  boost::optional<std::string> jsonProfileOut;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type: Cpu | Sim | Hw | IpuModel",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling report")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("channels", po::value<unsigned>(&chans)->required(),
     "Number of channels")
    ("field",
     po::value<ShapeOption<std::size_t>>(&inputFieldSizeOption)->required(),
     "Field size")
    ("kernel-size",
     po::value<ShapeOption<std::size_t>>(&kernelSizeOption)->default_value(1),
     "kernel size")
    ("padding-upper",
     po::value<ShapeOption<int>>(&paddingUpperOption)->default_value(0),
     "Amount of zero padding to add at the end of each dimension")
    ("padding-lower",
     po::value<ShapeOption<int>>(&paddingLowerOption)->default_value(0),
     "Amount of zero padding to add at the start of each dimension")
    ("stride",
     po::value<ShapeOption<unsigned>>(&strideOption)->default_value(1),
     "Stride")
    ("data-type",
     po::value<Type>(&dataType)->default_value(HALF),
     "Type of the data and the parameters")
    ("fwd-chans-per-group",
     po::value<unsigned>(&fwdChansPerGroup),
     "The number of channels per group of the activations written in the "
     "forward pass")
    ("bwd-chans-per-group",
     po::value<unsigned>(&bwdChansPerGroup),
     "The number of channels per group of the deltas written in the backwards "
     "pass")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value(&tilesPerIPU),
     "Number of tiles per IPU")
     ("ipus",
     po::value<unsigned>(&numIPUs)->default_value(numIPUs),
     "Number of IPUs")
    ("pooling-type",
     po::value<PoolingType>(
         &poolingType
     )->default_value(poolingType),
     "Pooling Type (max | avg | sum)")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("use-scaled-grad",
       po::value<bool>(&scaledGradientForMaxPool)->default_value(false),
       "Whether or not to use scaled gradient for max pool")
    ("use-introspection",
     po::value<bool>(&useIntrospectiveMapping)->default_value(true),
     "Whether or not to use introspection when performaing tile mapping")
    ("optimize-for-speed",
      po::value<bool>(&optimizeForSpeed)->default_value(false),
      "Allow optimisations for speed at the cost of memory allocation"
      " constraints")
  ;
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      std::cout << "A multi-dimensional shape can be specified using a brace "
                   "enclosed comma\n"
                   "separated list, for example --stride={1,2}. You may also "
                   "specify a single\n"
                   "number without braces in which case that value is used for "
                   "each dimension,\n"
                   "for example --stride=2\n";
      return 1;
    }

    if (vm.count("profile") || jsonProfileOut) {
      engineOptions.set("debug.instrumentCompute", "true");
    }
    if (isSimulator(deviceType) && numIPUs > 1) {
      engineOptions.set("debug.globalExchangeViaDebug", "true");
    }

    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  poolingOptions.set("poolUseIntrospectiveMapping",
                     useIntrospectiveMapping ? "true" : "false");

  poolingOptions.set("optimizeForSpeed", optimizeForSpeed ? "true" : "false");

  auto &inputFieldSize = inputFieldSizeOption.val;
  const auto numFieldDims = inputFieldSize.size();

  kernelSizeOption.broadcast(numFieldDims);
  auto &kernelSize = kernelSizeOption.val;

  strideOption.broadcast(numFieldDims);
  auto &stride = strideOption.val;

  paddingLowerOption.broadcast(numFieldDims);
  auto &paddingLower = paddingLowerOption.val;

  paddingUpperOption.broadcast(numFieldDims);
  auto &paddingUpper = paddingUpperOption.val;

  const bool inferenceOnly = vm.count("inference-only");
  const bool ignoreData = vm.count("ignore-data");

  auto device = tilesPerIPU
                    ? createTestDevice(deviceType, numIPUs, *tilesPerIPU)
                    : createTestDeviceFullSize(deviceType, numIPUs);

  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  // If the output grouping is unspecified, assume the output uses the same
  // grouping as the input unless that is impossible.
  if (!vm.count("fwd-chans-per-group")) {
    if (chans % 16 == 0)
      fwdChansPerGroup = 16;
    else
      fwdChansPerGroup = 1;
  }
  if (!inferenceOnly && !vm.count("bwd-chans-per-group")) {
    if (chans % 16 == 0)
      bwdChansPerGroup = 16;
    else
      bwdChansPerGroup = 1;
  }

  const auto poolParams = popnn::pooling::PoolParams(
      poolingType, inputFieldSize, kernelSize, stride, paddingLower,
      paddingUpper, chans, batchSize, dataType);

  const auto outDims = poolParams.getOutputFieldShape();

  // Create tensors.
  Tensor prevAct = [&] {
    // start with channels in the outer most dimension so that the batches
    // get distributed across the tiles when the tensor is mapped.
    std::vector<std::size_t> prevActShape = {chans / fwdChansPerGroup,
                                             batchSize};
    prevActShape.insert(prevActShape.end(), inputFieldSize.begin(),
                        inputFieldSize.end());
    prevActShape.push_back(fwdChansPerGroup);
    Tensor prevAct = graph.addVariable(dataType, prevActShape, "prevAct");
    mapTensorLinearly(graph, prevAct);
    // squash channels and groups into the same dimension.
    return prevAct.dimShufflePartial({0, prevAct.rank() - 1}, {1, 2})
        .reshapePartial(1, 3, {chans});
  }();

  Tensor zDeltas = [&] {
    if (!inferenceOnly) {
      // start with channels in the outer most dimension so that the batches
      // get distributed across the tiles when the tensor is mapped.
      std::vector<std::size_t> zDeltasShape = {chans / bwdChansPerGroup,
                                               batchSize};
      zDeltasShape.insert(zDeltasShape.end(), outDims.begin(), outDims.end());
      zDeltasShape.push_back(bwdChansPerGroup);

      Tensor zDeltas = graph.addVariable(dataType, zDeltasShape, "zDeltas");
      mapTensorLinearly(graph, zDeltas);
      return zDeltas.dimShufflePartial({0, zDeltas.rank() - 1}, {1, 2})
          .reshapePartial(1, 3, {chans});
    } else {
      return Tensor{};
    }
  }();

  // create shapes for the model pooling.
  MultiArrayShape prevActShape = {batchSize, chans};
  prevActShape.insert(std::end(prevActShape), std::begin(inputFieldSize),
                      std::end(inputFieldSize));

  MultiArrayShape zDeltasShape = {batchSize, chans};
  zDeltasShape.insert(std::end(zDeltasShape), std::begin(outDims),
                      std::end(outDims));

  auto fwdProg = Sequence();
  auto nextAct = popnn::pooling::pool(graph, poolParams, prevAct, fwdProg,
                                      "TestFwd", poolingOptions);

  auto bwdProg = Sequence();
  Tensor prevDeltas;
  if (!inferenceOnly) {
    if (poolingType == PoolingType::MAX) {
      prevDeltas = popnn::pooling::poolInputGradient(
          graph, poolParams, prevAct, nextAct, zDeltas,
          scaledGradientForMaxPool, bwdProg, "TestBwdMax", poolingOptions);
    } else {
      prevDeltas = popnn::pooling::poolInputGradient(
          graph, poolParams, fwdChansPerGroup, zDeltas, bwdProg, "TestBwdSum",
          poolingOptions);
    }
  }
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct = allocateHostMemoryForTensor(
      prevAct, "prevAct", graph, uploadProg, downloadProg, tmap);
  auto rawHostNextAct = allocateHostMemoryForTensor(
      nextAct, "nextAct", graph, uploadProg, downloadProg, tmap);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (!inferenceOnly) {
    rawHostZDeltas = allocateHostMemoryForTensor(
        zDeltas, "zDeltas", graph, uploadProg, downloadProg, tmap);
    rawHostPrevDeltas = allocateHostMemoryForTensor(
        prevDeltas, "prevDeltas", graph, uploadProg, downloadProg, tmap);
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

  Engine engine(graph, std::move(programs), engineOptions);
  attachStreams(engine, tmap);

  MultiArray<double> hostPrevAct{prevActShape};
  MultiArray<double> hostNextAct{zDeltasShape};
  std::mt19937 randomEngine;
  unsigned maxValue = 4;

  writeRandomValues(target, dataType, hostPrevAct,
                    -static_cast<double>(maxValue),
                    static_cast<double>(maxValue), randomEngine);
  // Guarantee that differences in input activations are well above the minimum
  // half value
  adjustActivations(hostPrevAct, maxValue);

  copy(target, hostPrevAct, dataType, rawHostPrevAct.get());
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

  // Validate against a reference model.
  if (vm["tolerance"].empty()) {
    if (dataType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }
  if (dataType == FLOAT) {
    absoluteTolerance = FLOAT_ABS_TOL;
  } else {
    absoluteTolerance = HALF_ABS_TOL;
  }

  bool matchesModel = true;
  copy(target, dataType, rawHostNextAct.get(), hostNextAct);
  MultiArray<double> modelNextAct{zDeltasShape};
  std::fill_n(modelNextAct.data(), modelNextAct.numElements(), 37.2);
  if (!ignoreData) {
    poplibs_test::pooling::pooling(poolingType, stride, kernelSize,
                                   paddingLower, paddingUpper, hostPrevAct,
                                   modelNextAct);
    matchesModel = checkIsClose("fwd", hostNextAct, modelNextAct,
                                relativeTolerance, absoluteTolerance);
  }

  if (!inferenceOnly) {
    MultiArray<double> hostZDeltas{zDeltasShape};
    MultiArray<double> hostPrevDeltas{prevActShape};

    // Run the backwards pass.
    writeRandomValues(target, dataType, hostZDeltas, -5.0, 5.0, randomEngine);
    copy(target, hostZDeltas, dataType, rawHostZDeltas.get());
    copy(target, modelNextAct, dataType, rawHostNextAct.get());
    copy(target, hostPrevAct, dataType, rawHostPrevAct.get());
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
    copy(target, dataType, rawHostZDeltas.get(), hostZDeltas);
    copy(target, dataType, rawHostPrevDeltas.get(), hostPrevDeltas);

    // Validate against a reference model.
    if (!ignoreData) {
      MultiArray<double> modelPrevDeltas{prevActShape};
      poplibs_test::pooling::poolingBackward(
          poolingType, scaledGradientForMaxPool, stride, kernelSize,
          paddingLower, paddingUpper, hostPrevAct, modelNextAct, hostZDeltas,
          modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance, absoluteTolerance);
    }
  }

  if (jsonProfileOut) {
    const auto pr = engine.getProfile();

    std::ofstream os(*jsonProfileOut);
    poplar::serializeToJSON(os, pr);
  }

  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
