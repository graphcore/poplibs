// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PoolingCodeletTest
#include "CreatePoolingVertex.hpp"
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
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Pooling.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <random>

// Tolerances used in tests
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
  unsigned channelGroups = 1;
  unsigned batchSize;
  Type dataType;
  bool debugOutput = false;

  ShapeOption<std::size_t> inputFieldSizeOption;
  ShapeOption<std::size_t> kernelSizeOption;
  ShapeOption<unsigned> strideOption;

  DeviceType deviceType = DeviceType::IpuModel2;
  PoolingType poolingType = PoolingType::MAX;

  OptionFlags engineOptions;
  OptionFlags poolingOptions;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("profile", "Output profiling report")
    ("debug", "Output debug information - in, out, vertex state")
    ("channels", po::value<unsigned>(&chans)->required(),
     "Number of channels")
    ("channel-groups", po::value<unsigned>(&channelGroups)->default_value(channelGroups),
     "Number of channel groups")
    ("field",
     po::value<ShapeOption<std::size_t>>(&inputFieldSizeOption)->required(),
     "Field size")
    ("kernel-size",
     po::value<ShapeOption<std::size_t>>(&kernelSizeOption)->default_value(1),
     "kernel size")
    ("stride",
     po::value<ShapeOption<unsigned>>(&strideOption)->default_value(1),
     "Stride")
    ("data-type",
     po::value<Type>(&dataType)->default_value(HALF),
     "Type of the data")
    ("pooling-type",
     po::value<PoolingType>(&poolingType)->default_value(poolingType),
     "Pooling Type (max | avg | sum)")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
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

    if (vm.count("profile")) {
      engineOptions.set("debug.instrumentCompute", "true");
    }
    if (vm.count("debug")) {
      debugOutput = true;
    }

    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (chans % channelGroups) {
    throw poputil::poplibs_error(
        "Channels must be divisible by channel groups");
  }
  const auto chansPerGroup = chans / channelGroups;
  const auto vertexVectorWidth = dataType == HALF ? 4 : 2;
  if (chansPerGroup % vertexVectorWidth) {
    throw poputil::poplibs_error(
        "Channels per group must be divisible by 4 (half) or 2 (float)");
  }

  auto &inputFieldSize = inputFieldSizeOption.val;
  const auto numFieldDims = inputFieldSize.size();

  kernelSizeOption.broadcast(numFieldDims);
  auto &kernelSize = kernelSizeOption.val;

  strideOption.broadcast(numFieldDims);
  auto &stride = strideOption.val;

  std::vector<int> paddingLower(numFieldDims, 0);
  std::vector<int> paddingUpper(numFieldDims, 0);

  auto device = createTestDevice(deviceType, 1, 1);

  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);

  const auto poolParams = popnn::pooling::PoolParams(
      poolingType, inputFieldSize, kernelSize, stride, paddingLower,
      paddingUpper, chans, batchSize, dataType);

  const auto outDims = poolParams.getOutputFieldShape();

  // Create input tensor.
  Tensor prevAct = [&] {
    std::vector<std::size_t> prevActShape = {channelGroups, batchSize,
                                             chansPerGroup};
    // Create an input Tensor
    // [ChannelGroups, batchSize, x,y... , channelsPerGroup]
    prevActShape.insert(prevActShape.begin() + 2, inputFieldSize.begin(),
                        inputFieldSize.end());

    Tensor prevAct = graph.addVariable(dataType, prevActShape, "prevAct");
    graph.setTileMapping(prevAct, 0);
    return prevAct;
  }();

  // create shapes for the model pooling.
  MultiArrayShape prevActShape = {batchSize, chans};
  prevActShape.insert(std::end(prevActShape), std::begin(inputFieldSize),
                      std::end(inputFieldSize));

  MultiArrayShape zDeltasShape = {batchSize, chans};
  zDeltasShape.insert(std::end(zDeltasShape), std::begin(outDims),
                      std::end(outDims));

  auto prog = Sequence();

  auto outShape = prevAct.shape();
  for (unsigned dim = 2; dim < outShape.size() - 1; dim++) {
    outShape[dim] = outShape[dim] - kernelSize[dim - 2] + 1;
    outShape[dim] = (outShape[dim] + stride[0] - 1) / stride[0];
  }

  auto nextAct = graph.addVariable(dataType, outShape, "nextAct");

  // Call the test function to create the vertex.
  // TODO - Support all vertices.  At present all but MaxPoolingGrad(half,float)
  // and MaxPoolingGradientScale(half,float) are supported.
  // MaxPoolingGradientScale doesn't have an assembler implementation so a
  // codelet test is probably not necessary.
  const poplar::DebugNameAndId dnai;
  popnn::pooling::createPoolingVertex(graph, poolParams, prevAct, nextAct, prog,
                                      {dnai});

  // Roll the channel to before the spatial dimensions as
  // Pooling vertices work with [batches, spatial dims, channels]
  // Data comparision works with [batches, channels, spatial dims]
  prevAct = prevAct.dimRoll(prevAct.rank() - 1, 2);
  nextAct = nextAct.dimRoll(nextAct.rank() - 1, 2);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct = allocateHostMemoryForTensor(
      prevAct, "prevAct", graph, uploadProg, downloadProg, tmap);
  auto rawHostNextAct = allocateHostMemoryForTensor(
      nextAct, "nextAct", graph, uploadProg, downloadProg, tmap);

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);
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
    engine.run(0);
  });

  // Validate against a reference model.
  const double relativeTolerance =
      (dataType == FLOAT) ? FLOAT_REL_TOL : HALF_REL_TOL;
  const double absoluteTolerance =
      (dataType == FLOAT) ? FLOAT_ABS_TOL : HALF_ABS_TOL;

  bool matchesModel = true;
  copy(target, dataType, rawHostNextAct.get(), hostNextAct);
  MultiArray<double> modelNextAct{zDeltasShape};
  std::fill_n(modelNextAct.data(), modelNextAct.numElements(), 37.2);

  poplibs_test::pooling::pooling(poolingType, stride, kernelSize, paddingLower,
                                 paddingUpper, hostPrevAct, modelNextAct);
  matchesModel = checkIsClose("fwd", hostNextAct, modelNextAct,
                              relativeTolerance, absoluteTolerance);
  if (debugOutput) {
    std::cout << "In:\n";
    forEachIndex(hostPrevAct.shape(), [&](const MultiArrayShapeRange indices) {
      std::cout << hostPrevAct[indices] << ",";
    });

    std::cout << "\n\nOut:\n";
    forEachIndex(hostNextAct.shape(), [&](const MultiArrayShapeRange indices) {
      std::cout << hostNextAct[indices] << ",";
    });
    std::cout << "\n";
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
