#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <istream>
#include <ostream>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <poplibs_test/Pooling.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poputil/exceptions.hpp>
#include "TestDevice.hpp"
#include <random>

// Default tolerances used in tests
#define FLOAT_REL_TOL  0.1
#define HALF_REL_TOL   0.3
#define FLOAT_ABS_TOL  1e-5
#define HALF_ABS_TOL   7e-2

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;

using popnn::PoolingType;

namespace popnn {
  std::ostream &
  operator<<(std::ostream &os, const PoolingType &pType) {
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
      throw poputil::poplibs_error(
        "Unknown pooling type<" + token + ">");
    return is;
  }
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
  IPUModel ipuModel;
  PoolingType poolingType = PoolingType::MAX;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type: Cpu | Sim | Hw | IpuModel",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling report")
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
     po::value<unsigned>(&ipuModel.tilesPerIPU)->
                           default_value(ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
     ("ipus",
     po::value<unsigned>(&ipuModel.numIPUs)->default_value(ipuModel.numIPUs),
     "Number of IPUs")
    ("pooling-type",
     po::value<PoolingType>(
         &poolingType
     )->default_value(poolingType),
     "Pooling Type (max | avg | sum)")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
  ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      std::cout <<
"A multi-dimensional shape can be specified using a brace enclosed comma\n"
"separated list, for example --stride={1,2}. You may also specify a single\n"
"number without braces in which case that value is used for each dimension,\n"
"for example --stride=2\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  auto &inputFieldSize = inputFieldSizeOption.val;
  const auto numFieldDims = inputFieldSize.size();

  if (numFieldDims != 2) {
      throw poputil::poplibs_error("Only 2D pooling is currently supported");
  }

  kernelSizeOption.broadcast(numFieldDims);
  auto &kernelSize = kernelSizeOption.val;

  strideOption.broadcast(numFieldDims);
  auto &stride = strideOption.val;

  paddingLowerOption.broadcast(numFieldDims);
  auto &paddingLower = paddingLowerOption.val;

  paddingUpperOption.broadcast(numFieldDims);
  auto &paddingUpper = paddingUpperOption.val;

  bool inferenceOnly = vm.count("inference-only");
  auto device = createTestDevice(deviceType, ipuModel.numIPUs,
                                   ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);

  // If the output grouping is unspecified, assume the output uses the same
  // grouping as the input unless that is impossible.
  if (!vm.count("fwd-chans-per-group")) {
    if (chans % 16 == 0)
      fwdChansPerGroup = 16;
    else
      fwdChansPerGroup = 1;
  }
  if (!inferenceOnly &&
      !vm.count("bwd-chans-per-group")) {
    if (chans % 16 == 0)
      bwdChansPerGroup = 16;
    else
      bwdChansPerGroup = 1;
  }

  const auto poolParams =
      popnn::pooling::PoolParams(poolingType,
                                 inputFieldSize,
                                 kernelSize,
                                 stride,
                                 paddingLower,
                                 paddingUpper,
                                 chans,
                                 batchSize,
                                 dataType);

  const auto outDims = poolParams.getOutputFieldShape();
  const auto height = inputFieldSize[0];
  const auto width = inputFieldSize[1];
  const auto outHeight = outDims[0];
  const auto outWidth = outDims[1];
  const auto strideHeight = stride[0];
  const auto strideWidth = stride[1];
  const auto kernelHeight = kernelSize[0];
  const auto kernelWidth = kernelSize[1];
  const auto paddingHeightL = paddingLower[0];
  const auto paddingWidthL = paddingLower[1];
  const auto paddingHeightU = paddingUpper[0];
  const auto paddingWidthU = paddingUpper[1];

  // Create tensors.
  std::vector<std::size_t> prevActShape = {chans / fwdChansPerGroup,
                                           batchSize};
  prevActShape.insert(prevActShape.end(),
                      inputFieldSize.begin(),
                      inputFieldSize.end());
  prevActShape.push_back(fwdChansPerGroup);
  Tensor prevAct = graph.addVariable(dataType, prevActShape, "prevAct");
  mapTensorLinearly(graph, prevAct);
  prevAct = prevAct.dimShufflePartial({0, prevAct.rank() - 1}, {1, 2})
                   .reshapePartial(1, 3, {chans});

  Tensor zDeltas;
  if (!inferenceOnly) {
    std::vector<std::size_t> zDeltasShape = {chans / bwdChansPerGroup,
                                             batchSize};
    zDeltasShape.insert(zDeltasShape.end(), outDims.begin(), outDims.end());
    zDeltasShape.push_back(bwdChansPerGroup);
    zDeltas = graph.addVariable(dataType, zDeltasShape, "zDeltas");
    mapTensorLinearly(graph, zDeltas);
    zDeltas = zDeltas.dimShufflePartial({0, zDeltas.rank() - 1}, {1, 2})
                     .reshapePartial(1, 3, {chans});
  }

  auto fwdProg = Sequence();
  auto nextAct = popnn::pooling::pool(graph, poolParams, prevAct, fwdProg);

  auto bwdProg = Sequence();
  Tensor prevDeltas;
  if (!inferenceOnly) {
    prevDeltas =
        popnn::pooling::poolInputGradient(graph, poolParams, prevAct, nextAct,
                                          zDeltas, bwdProg);
  }
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct = allocateHostMemoryForTensor(prevAct, "prevAct",
                                                    graph, uploadProg,
                                                    downloadProg, tmap);
  auto rawHostNextAct = allocateHostMemoryForTensor(nextAct, "nextAct",
                                                    graph, uploadProg,
                                                    downloadProg, tmap);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (!inferenceOnly) {
    rawHostZDeltas = allocateHostMemoryForTensor(zDeltas, "zDeltas",
                                                 graph, uploadProg,
                                                 downloadProg, tmap);
    rawHostPrevDeltas = allocateHostMemoryForTensor(prevDeltas, "prevDeltas",
                                                    graph, uploadProg,
                                                    downloadProg, tmap);
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
  OptionFlags engineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.executionProfile", "compute_sets");
  }
  Engine engine(graph, std::move(programs), engineOptions);
  attachStreams(engine, tmap);

  boost::multi_array<double, 4>
      hostPrevAct(boost::extents[batchSize][chans][height][width]);
  boost::multi_array<double, 4>
      hostNextAct(boost::extents[batchSize][chans][outHeight][outWidth]);
  std::mt19937 randomEngine;
  writeRandomValues(target, dataType, hostPrevAct, -4.0, 4.0, randomEngine);
  copy<4>(target, hostPrevAct, dataType, rawHostPrevAct.get());
  // Run the forward pass.
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(uploadProgIndex);
    engine.run(fwdProgIndex); // Run.
    engine.run(downloadProgIndex);
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
  copy<4>(target, dataType, rawHostNextAct.get(), hostNextAct);
  boost::multi_array<double, 4>
      modelNextAct(boost::extents[batchSize][chans][outHeight][outWidth]);
  poplibs_test::pooling::pooling(poolingType, strideHeight, strideWidth,
                                kernelHeight, kernelWidth,
                                paddingHeightL, paddingWidthL,
                                paddingHeightU, paddingWidthU,
                                hostPrevAct, modelNextAct);
  bool matchesModel = checkIsClose("fwd", hostNextAct, modelNextAct,
                                   relativeTolerance, absoluteTolerance);

  if (!inferenceOnly) {
    boost::multi_array<double, 4> hostZDeltas(
      boost::extents[batchSize][chans][outHeight][outWidth]
    );
    boost::multi_array<double, 4> hostPrevDeltas(
      boost::extents[batchSize][chans][height][width]
    );
    // Run the backwards pass.
    writeRandomValues(target, dataType, hostZDeltas, -5.0, 5.0, randomEngine);
    copy<4>(target, hostZDeltas, dataType, rawHostZDeltas.get());
    copy<4>(target, modelNextAct, dataType, rawHostNextAct.get());
    copy<4>(target, hostPrevAct, dataType, rawHostPrevAct.get());
    device.bind([&](const Device &d) {
      engine.load(d);
      engine.run(uploadProgIndex);
      engine.run(bwdProgIndex); // Run.
      engine.run(downloadProgIndex);
    });
    copy<4>(target, dataType, rawHostZDeltas.get(), hostZDeltas);
    copy<4>(target, dataType, rawHostPrevDeltas.get(), hostPrevDeltas);

    // Validate against a reference model.
    boost::multi_array<double, 4>
        modelPrevDeltas(boost::extents[batchSize][chans][height][width]);
    poplibs_test::pooling::poolingBackward(poolingType,
                                          strideHeight, strideWidth,
                                          kernelHeight, kernelWidth,
                                          paddingHeightL, paddingWidthL,
                                          paddingHeightU, paddingWidthU,
                                          hostPrevAct, modelNextAct,
                                          hostZDeltas, modelPrevDeltas);
    matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                 relativeTolerance, absoluteTolerance);
  }

  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    engine.printSummary(std::cout, OptionFlags{
      { "doLayerWiseBreakdown", "true" }
    });
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
