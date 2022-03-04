// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvPartial1xNSLIC
#include "ConvVertices.hpp"
#include "popops/Cast.hpp"
#include "popops/codelets.hpp"

#include <poplibs_support/TestDevice.hpp>

#include <assert.h>
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_support/print.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>

#include <poplin/codelets.hpp>

#include <poplar/CSRFunctions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poputil;
using namespace poplibs_support;

unsigned stridedDimSize(unsigned inputSize, unsigned kernelSize,
                        unsigned stride, bool applyStride = true) {
  const auto outputSize = inputSize - kernelSize + 1;
  return applyStride ? (outputSize + stride - 1) / stride : outputSize;
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  unsigned convGroupsPerGroup = 1;
  unsigned chansPerGroup = 4;

  unsigned convGroupGroups = 1;
  unsigned inChanGroups = 1;
  unsigned outChanGroups = 1;

  unsigned convChainsRequired = 2;

  ShapeOption<std::size_t> inputFieldSizeOption;
  ShapeOption<std::size_t> kernelSizeOption;
  ShapeOption<unsigned> outputPaddingLower(0);
  ShapeOption<unsigned> outputPaddingUpper(0);
  ShapeOption<unsigned> outputTruncationLower(0);
  ShapeOption<unsigned> outputTruncationUpper(0);
  ShapeOption<unsigned> outputStride(1);
  unsigned batchSize = 1;

  Type inputType = HALF;
  Type partialsType = FLOAT;

  int weightFp8Scale = 0;
  int inputFp8Scale = 0;
  Fp8Format weightFp8Format = Fp8Format::QUART143;
  Fp8Format inputFp8Format = Fp8Format::QUART143;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling information for the program")
    ("ignore-data", "Don't validate outputs, don't add streams etc. Useful for profiling")
    ("show-execution-steps", "If profiling, show execution steps in the summary")
    ("show-var-storage", "If profiling, show variable liveness information in the summary")
    ("conv-groups-per-group",
     po::value<unsigned>(&convGroupsPerGroup)->default_value(convGroupsPerGroup),
     "Number of conv groups per group (1, 2, or 4)")
    ("chans-per-group",
     po::value<unsigned>(&chansPerGroup)->default_value(chansPerGroup),
     "Number of input/output channels per group (1, 2, 4 or 8)")
    ("conv-group-groups",
     po::value<unsigned>(&convGroupGroups)->default_value(convGroupGroups),
     "Number of groups of conv groups")
    ("out-chan-groups",
     po::value<unsigned>(&outChanGroups)->default_value(outChanGroups),
     "Number of output channel groups")
    ("in-chan-groups",
     po::value<unsigned>(&inChanGroups)->default_value(inChanGroups),
     "Number of input channel groups")
    ("in-field-size",
     po::value<ShapeOption<std::size_t>>(&inputFieldSizeOption)->required(),
     "Input field size")
    ("kernel-size",
     po::value<ShapeOption<std::size_t>>(&kernelSizeOption)->required(),
     "Kernel size")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(batchSize),
     "Batch size")
    ("input-type",
     po::value<Type>(&inputType)->default_value(inputType),
     "Input type")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(partialsType),
     "Partials type")
    ("output-padding-lower",
     po::value<ShapeOption<unsigned>>(&outputPaddingLower),
     "Output padding lower")
    ("output-padding-upper",
     po::value<ShapeOption<unsigned>>(&outputPaddingUpper),
     "Output padding upper")
    ("output-stride",
     po::value<ShapeOption<unsigned>>(&outputStride),
     "Output stride")
    ("output-truncation-lower",
     po::value<ShapeOption<unsigned>>(&outputTruncationLower),
     "Output truncation lower")
    ("output-truncation-upper",
     po::value<ShapeOption<unsigned>>(&outputTruncationUpper),
     "Output truncation upper")
    ("conv-chains",
     po::value<unsigned>(&convChainsRequired)->default_value(convChainsRequired),
     "Conv chains to use.  If partials are float(=2), if half (=2 or 4),"
     " if quarter(=4)")
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

  bool profile = vm.count("profile");
  bool ignoreData = vm.count("ignore-data");
  bool showExecutionSteps = vm.count("show-execution-steps");
  bool showVarStorage = vm.count("show-var-storage");

  if (inputType != HALF && inputType != QUARTER) {
    throw poputil::poplibs_error("Only inputTypes of HALF or QUARTER are "
                                 " currently supported");
  }

  if (inChanGroups != 1) {
    throw poputil::poplibs_error(
        "ConvPartial1xNSLIC vertex only handles 1 input channel group");
  }

  if (outChanGroups != 1) {
    throw poputil::poplibs_error(
        "ConvPartial1xNSLIC vertex only handles 1 output channel group");
  }

  const unsigned convGroups = convGroupGroups * convGroupsPerGroup;
  const unsigned inChans = inChanGroups * chansPerGroup;
  const unsigned outChans = outChanGroups * chansPerGroup;

  const auto &inputFieldSize = inputFieldSizeOption.val;
  const auto numFieldDims = inputFieldSize.size();

  kernelSizeOption.broadcast(numFieldDims);
  const auto &kernelSize = kernelSizeOption.val;
  if (kernelSize.back() % 4u) {
    throw poputil::poplibs_error("kernelSize.back() must be divisible by 4");
  }
  outputPaddingLower.broadcast(numFieldDims);
  outputPaddingUpper.broadcast(numFieldDims);
  outputTruncationLower.broadcast(numFieldDims);
  outputTruncationUpper.broadcast(numFieldDims);
  outputStride.broadcast(numFieldDims);

  std::vector<std::size_t> outputFieldSize(inputFieldSize.size());
  std::vector<std::size_t> stridedPaddedOutputFieldSize(inputFieldSize.size());

  for (unsigned d = 0; d < numFieldDims; ++d) {
    outputFieldSize[d] = inputFieldSize[d] - kernelSize[d] + 1;

    const auto unpaddedDimSize =
        stridedDimSize(inputFieldSize[d], kernelSize[d], outputStride[d]);
    stridedPaddedOutputFieldSize[d] =
        outputPaddingLower[d] + unpaddedDimSize + outputPaddingUpper[d] -
        outputTruncationLower[d] - outputTruncationUpper[d];
  }
  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  const bool isFp8 = inputType == QUARTER;
  const auto hostInputType = isFp8 ? HALF : inputType;

  // Create input, weights, output
  constexpr std::size_t overreadConvGroups = 1;
  std::vector<std::size_t> inShape = {convGroupGroups + overreadConvGroups,
                                      inChanGroups, batchSize};
  inShape.insert(inShape.end(), inputFieldSize.begin(), inputFieldSize.end());
  inShape.insert(inShape.end(), {convGroupsPerGroup, chansPerGroup});
  std::vector<std::size_t> weightsShape = {convGroupGroups, outChanGroups,
                                           inChanGroups};
  weightsShape.insert(weightsShape.end(), kernelSize.begin(), kernelSize.end());
  weightsShape.insert(weightsShape.end(),
                      {convGroupsPerGroup, chansPerGroup, chansPerGroup});
  std::vector<std::size_t> outputShape = {convGroupGroups, outChanGroups,
                                          batchSize};
  outputShape.insert(outputShape.end(), stridedPaddedOutputFieldSize.begin(),
                     stridedPaddedOutputFieldSize.end());

  outputShape.insert(outputShape.end(), {convGroupsPerGroup, chansPerGroup});

  const auto inGroupedWithOverread =
      graph.addVariable(hostInputType, inShape, "in");
  const auto inGrouped = inGroupedWithOverread.slice(0, convGroupGroups, 0);
  const auto inOverreadMemory = inGroupedWithOverread.slice(
      convGroupGroups, convGroupGroups + overreadConvGroups, 0);
  const auto weightsGrouped =
      graph.addVariable(hostInputType, weightsShape, "weights");
  const auto outGrouped = graph.addVariable(partialsType, outputShape, "out");

  graph.setTileMapping(inGroupedWithOverread, 0);
  graph.setTileMapping(weightsGrouped, 0);
  graph.setTileMapping(outGrouped, 0);

  unsigned windowWidth = 4;
  poplin::ConvParams params{inputType, batchSize, inputFieldSize, kernelSize,
                            inChans,   outChans,  convGroups};
  params.outputTransform.paddingLower = outputPaddingLower;
  params.outputTransform.paddingUpper = outputPaddingUpper;
  params.outputTransform.truncationLower = outputTruncationLower;
  params.outputTransform.truncationUpper = outputTruncationUpper;
  params.outputTransform.stride = outputStride;

  Sequence prog;
  if (!ignoreData) {
    bool exceptOnInv = true;
    bool exceptOnDiv0 = true;
    bool exceptOnOflo = true;
    bool enableStochasticRounding = false;
    bool nanOnOverflow = true;
    setFloatingPointBehaviour(graph, prog,
                              {exceptOnInv, exceptOnDiv0, exceptOnOflo,
                               enableStochasticRounding, nanOnOverflow},
                              "enableAllFpExceptions");
  }

  std::map<Type, Tensor> copyWritten{
      {inputType, graph.addVariable(inputType, {0})}};

  // fill the output and input overread detection space with (signalling) NaNs
  auto fillWithNaNs = [&](const Tensor &t) {
    const auto nan =
        graph.addConstant(t.elementType(), t.shape(),
                          std::numeric_limits<float>::signaling_NaN());
    graph.setTileMapping(nan, 0);
    prog.add(Copy(nan, t));
  };
  fillWithNaNs(outGrouped);
  fillWithNaNs(inOverreadMemory);

  Tensor inGroupedFp8, weightsGroupedFp8;
  Sequence castProg;
  if (isFp8) {
    auto weightsMetadata =
        createFp8MetadataTensor(graph, weightFp8Format, weightFp8Scale);
    auto inMetadata =
        createFp8MetadataTensor(graph, inputFp8Format, inputFp8Scale);

    // TODO - T57103 won't need an on-IPU cast once we can copy data to the IPU
    inGroupedFp8 =
        popops::cast(graph, inGrouped, QUARTER, inMetadata, castProg, "CastIn");
    weightsGroupedFp8 = popops::cast(graph, weightsGrouped, QUARTER,
                                     weightsMetadata, castProg, "CastWeights");
  }

  // create the vertex
  auto fwdCS = graph.addComputeSet("fwdCS");

  std::vector<Copy> transformPre;
  std::map<poplar::Type,
           std::pair<std::vector<poplar::Tensor>, std::vector<poplar::Tensor>>>
      postProg;
  auto vertexIn = isFp8 ? inGroupedFp8 : inGrouped;
  auto vertexWeights = isFp8 ? weightsGroupedFp8 : weightsGrouped;
  createConvPartialSlicVertex(graph, windowWidth, convGroupsPerGroup,
                              chansPerGroup, convChainsRequired, 0, params,
                              transformPre, copyWritten, fwdCS, postProg,
                              vertexIn, vertexWeights, outGrouped, "vertex");
  if (isFp8) {
    prog.add(castProg);
  }
  for (const auto &copy : transformPre) {
    prog.add(copy);
  }
  prog.add(Execute(fwdCS));

  for (auto &p : postProg) {
    prog.add(Copy(concat(p.second.first), concat(p.second.second)));
  }

  // Get ordinary view of input/weights/output without grouping for reference
  // etc.
  const auto in = inGrouped.dimRoll(inGrouped.rank() - 2, 1)
                      .flatten(0, 2) // Flatten conv group groups
                      .dimRoll(inGrouped.rank() - 2, 2)
                      .flatten(1, 3)  // Flatten input channel groups
                      .dimRoll(2, 0); // Batch size to the front
  const auto weights = weightsGrouped.dimRoll(weightsGrouped.rank() - 3, 1)
                           .flatten(0, 2) // Flatten conv group groups
                           .dimRoll(weightsGrouped.rank() - 3, 2)
                           .flatten(1, 3) // Flatten output channel groups
                           .dimRoll(weightsGrouped.rank() - 3, 3)
                           .flatten(2, 4); // Flatten input channel groups
  const auto out = outGrouped.dimRoll(outGrouped.rank() - 2, 1)
                       .flatten(0, 2) // Flatten conv group groups
                       .dimRoll(outGrouped.rank() - 2, 2)
                       .flatten(1, 3)  // Flatten output channels groups
                       .dimRoll(2, 0); // Batch size to the front

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostIn, rawHostOut, rawHostWeights;
  if (!ignoreData) {
    rawHostIn = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                            downloadProg, tmap);
    rawHostWeights = allocateHostMemoryForTensor(
        weights, "weights", graph, uploadProg, downloadProg, tmap);
    rawHostOut = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                             downloadProg, tmap);
  }

  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (profile) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);

  attachStreams(engine, tmap);

  boost::multi_array<double, 3> hostIn(
      boost::extents[batchSize][convGroups * inChans][product(inputFieldSize)]);
  boost::multi_array<double, 4> hostWeights(
      boost::extents[convGroups][outChans][inChans][product(kernelSize)]);

  boost::multi_array<double, 3> hostOut(
      boost::extents[batchSize][convGroups * outChans]
                    [product(stridedPaddedOutputFieldSize)]);

  // 0 biases just for the reference convolution implementation.
  boost::multi_array<double, 1> hostBiases(
      boost::extents[convGroups * outChans]);

  if (!ignoreData) {
    std::mt19937 randomEngine;
    writeRandomValues(target, inputType, hostIn, -1.0, +5.0, randomEngine);
    writeRandomValues(target, inputType, hostWeights, -1.0, +7.0, randomEngine);
    copy(target, hostIn, hostInputType, rawHostIn.get());
    copy(target, hostWeights, hostInputType, rawHostWeights.get());
  }
  device.bind([&](const Device &d) { engine.loadAndRun(d); });

  if (!ignoreData) {
    boost::multi_array<double, 3> modelOut(
        boost::extents[batchSize][convGroups * outChans]
                      [product(stridedPaddedOutputFieldSize)]);
    std::vector<unsigned> modelOutputPaddingLower = outputPaddingLower;
    std::vector<unsigned> modelOutputPaddingUpper = outputPaddingUpper;
    std::vector<unsigned> modelOutputTruncationLower = outputTruncationLower;
    std::vector<unsigned> modelOutputTruncationUpper = outputTruncationUpper;

    std::vector<unsigned> modelStrides = outputStride;

    poplibs_test::conv::convolution(
        // Input:
        vectorConvert<unsigned>(inputFieldSize),
        std::vector<unsigned>(numFieldDims, 0), // truncationLower
        std::vector<unsigned>(numFieldDims, 0), // truncationUpper
        std::vector<unsigned>(numFieldDims, 1), // dilation
        std::vector<unsigned>(numFieldDims, 0), // paddingLower
        std::vector<unsigned>(numFieldDims, 0), // paddingUpper
        std::vector<bool>(numFieldDims, false), // flip
        // Kernel:
        vectorConvert<unsigned>(kernelSize),
        std::vector<unsigned>(numFieldDims, 0), // truncationLower
        std::vector<unsigned>(numFieldDims, 0), // truncationUpper
        std::vector<unsigned>(numFieldDims, 1), // dilation
        std::vector<unsigned>(numFieldDims, 0), // paddingLower
        std::vector<unsigned>(numFieldDims, 0), // paddingUpper
        std::vector<bool>(numFieldDims, false), // flip
        // Output:
        modelOutputTruncationLower, // truncationLower
        modelOutputTruncationUpper, // truncationUpper
        modelStrides,               // stride
        modelOutputPaddingLower,    // paddingLower
        modelOutputPaddingUpper,    // paddingUpper
        // Buffers:
        hostIn, hostWeights, hostBiases, modelOut);

    copy(target, partialsType, rawHostOut.get(), hostOut);
    bool matchesModel =
        checkIsClose("fwd", hostOut, modelOut, HALF_REL_TOL, HALF_ABS_TOL);
    if (!matchesModel) {
      std::cerr << "Validation failed\n";
      return 1;
    }
  }

  if (profile) {
    engine.printProfileSummary(
        std::cerr,
        {{"showExecutionSteps", (showExecutionSteps ? "true" : "false")},
         {"showVarStorage", (showVarStorage ? "true" : "false")}});
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
