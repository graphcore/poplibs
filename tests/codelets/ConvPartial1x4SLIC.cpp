// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"

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
#include <poplibs_test/Util.hpp>

#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>

#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <popops/Zero.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace poplibs_support;

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  unsigned convGroupsPerGroup = 1;
  unsigned chansPerGroup = 4;

  unsigned convGroupGroups = 1;
  unsigned inChanGroups = 1;
  unsigned outChanGroups = 1;

  ShapeOption<std::size_t> inputFieldSizeOption;
  ShapeOption<std::size_t> kernelSizeOption;
  ShapeOption<unsigned> outputPaddingLower(0);
  ShapeOption<unsigned> outputPaddingUpper(0);
  unsigned batchSize = 1;

  Type inputType = HALF;
  Type partialsType = FLOAT;

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
     "Number of input/output channels per group (1, 2, or 4)")
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
    ("allow-cpp-codelet", "Allow fallback to C++ codelet rather than erroring")
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
  bool allowCppCodelet = vm.count("allow-cpp-codelet");

  if (inputType != HALF) {
    throw poputil::poplibs_error("Only inputType=HALF is currently supported");
  }

  if (partialsType != FLOAT) {
    throw poputil::poplibs_error(
        "Only partialsType=FLOAT is currently supported");
  }

  unsigned char mode;
  switch (convGroupsPerGroup) {
  case 4u:
    if (chansPerGroup == 1u) {
      mode = 0;
      break;
    }
    // fallthrough
  case 2u:
    if (chansPerGroup == 2u) {
      mode = 1;
      break;
    }
    // fallthrough
  case 1u:
    if (chansPerGroup == 4u) {
      mode = 2;
      break;
    }
    // fallthrough
  default:
    throw poputil::poplibs_error(
        "Unsupported combination of channel and conv group groupings! "
        "convGroupsPerGroup=" +
        std::to_string(convGroupsPerGroup) +
        ", chansPerGroup=" + std::to_string(chansPerGroup));
    break;
  }

  if (inChanGroups != 1) {
    throw poputil::poplibs_error(
        "ConvPartial1x4SLIC vertex only handles 1 input channel group");
  }

  if (outChanGroups != 1) {
    throw poputil::poplibs_error(
        "ConvPartial1x4SLIC vertex only handles 1 output channel group");
  }

  const unsigned convGroups = convGroupGroups * convGroupsPerGroup;
  const unsigned inChans = inChanGroups * chansPerGroup;
  const unsigned outChans = outChanGroups * chansPerGroup;

  const auto &inputFieldSize = inputFieldSizeOption.val;
  const auto numFieldDims = inputFieldSize.size();

  kernelSizeOption.broadcast(numFieldDims);
  const auto &kernelSize = kernelSizeOption.val;

  outputPaddingLower.broadcast(numFieldDims);
  outputPaddingUpper.broadcast(numFieldDims);

  std::vector<std::size_t> outputFieldSize(inputFieldSize.size());
  std::vector<std::size_t> paddedOutputFieldSize(inputFieldSize.size());
  for (unsigned d = 0; d < numFieldDims; ++d) {
    outputFieldSize[d] = inputFieldSize[d] - kernelSize[d] + 1;
    paddedOutputFieldSize[d] =
        outputPaddingLower[d] + outputFieldSize[d] + outputPaddingUpper[d];
  }
  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  // Create input, weights, output
  std::vector<std::size_t> inShape = {convGroupGroups, inChanGroups, batchSize};
  inShape.insert(inShape.end(), inputFieldSize.begin(), inputFieldSize.end());
  inShape.insert(inShape.end(), {convGroupsPerGroup, chansPerGroup});
  std::vector<std::size_t> weightsShape = {convGroupGroups, outChanGroups,
                                           inChanGroups};
  weightsShape.insert(weightsShape.end(), kernelSize.begin(), kernelSize.end());
  weightsShape.insert(weightsShape.end(),
                      {convGroupsPerGroup, chansPerGroup, chansPerGroup});
  std::vector<std::size_t> outputShape = {convGroupGroups, outChanGroups,
                                          batchSize};
  outputShape.insert(outputShape.end(), paddedOutputFieldSize.begin(),
                     paddedOutputFieldSize.end());
  outputShape.insert(outputShape.end(), {convGroupsPerGroup, chansPerGroup});

  const auto inGrouped = graph.addVariable(inputType, inShape, "in");
  const auto weightsGrouped =
      graph.addVariable(inputType, weightsShape, "weights");
  const auto outGrouped = graph.addVariable(partialsType, outputShape, "out");

  graph.setTileMapping(inGrouped, 0);
  graph.setTileMapping(weightsGrouped, 0);
  graph.setTileMapping(outGrouped, 0);

  std::cout << "inGrouped.shape()=";
  printContainer(inShape, std::cout);
  std::cout << "\n";
  std::cout << "weightsGrouped.shape()=";
  printContainer(weightsShape, std::cout);
  std::cout << "\n";
  std::cout << "outputPaddingLower=";
  printContainer(outputPaddingLower, std::cout);
  std::cout << "\n";
  std::cout << "outputPaddingUpper=";
  printContainer(outputPaddingUpper, std::cout);
  std::cout << "\n";
  std::cout << "outputGrouped.shape()=";
  printContainer(outputShape, std::cout);
  std::cout << "\n";

  auto kernelSizeGroups = kernelSize;
  if (kernelSizeGroups.back() % 4u) {
    throw poputil::poplibs_error("kernelSize.back() must be divisible by 4");
  }
  assert(kernelSizeGroups.back() % 4u == 0);
  kernelSizeGroups.back() /= 4u;

  const auto numSubKernels = product(kernelSizeGroups);
  const auto inputSpatialSize = [&] {
    auto r = inputFieldSize;
    r.insert(r.begin(), batchSize);
    return r;
  }();
  const auto outputSpatialSize = [&] {
    auto r = outputFieldSize;
    r.insert(r.begin(), batchSize);
    return r;
  }();
  const auto paddedOutputSpatialSize = [&] {
    auto r = paddedOutputFieldSize;
    r.insert(r.begin(), batchSize);
    return r;
  }();
  const auto numWorkers = target.getNumWorkerContexts();
  const unsigned numFieldElems = product(outputSpatialSize);
  const auto fieldElemsPerWorker = ceildiv(numFieldElems, numWorkers);
  std::cout << "numFieldElems=" << numFieldElems << "\n";
  std::cout << "numSubKernels=" << numSubKernels << "\n";
  std::cout << "numWorkers=" << numWorkers << "\n";
  std::cout << "fieldElemsPerWorker=" << fieldElemsPerWorker << "\n";

  bool useShortTypes = true;
  const auto shortTypesVertexClass =
      templateVertex("poplin::ConvPartial1x4SLIC", inputType, partialsType,
                     /* useShortTypes */ true);
  const unsigned maxShortWorklistValue =
      graph.getMaxVertexFieldValue(shortTypesVertexClass, "worklists");
  const unsigned maxRptCount = target.getRptCountMax();
  std::vector<std::vector<unsigned short>> worklists(numWorkers *
                                                     numSubKernels);
  const unsigned fieldElemsPerRow = outputFieldSize.back();
  unsigned remainingFieldElems = numFieldElems;

  for (unsigned context = 0; context < numWorkers; ++context) {
    const unsigned fieldElemsThisContext =
        std::min(remainingFieldElems, fieldElemsPerWorker);
    for (unsigned kg = 0; kg < numSubKernels; ++kg) {
      unsigned outOffset = (numFieldElems - remainingFieldElems);
      auto kernelPosition = unflattenIndex(kernelSizeGroups, kg);
      kernelPosition.back() *= 4;

      const auto calcInOffset = [&](const unsigned unpaddedOutOffset) {
        auto inputOffsets =
            unflattenIndex(outputSpatialSize, unpaddedOutOffset);
        for (unsigned d = 0; d < numFieldDims; ++d) {
          // +1 because outer-most dimension is batch size.
          inputOffsets[d + 1] += kernelPosition[d];
        }
        return flattenIndex(inputSpatialSize, inputOffsets);
      };

      // transform outOffset into paddedOutOffset
      const auto calcPaddedOffset = [&](const unsigned unpaddedOutOffset) {
        auto paddedOutputOffsets =
            unflattenIndex(outputSpatialSize, unpaddedOutOffset);
        for (unsigned d = 0; d < numFieldDims; ++d) {
          // +1 because outer-most dimension is batch size.
          paddedOutputOffsets[d + 1] += outputPaddingLower[d];
        }
        return flattenIndex(paddedOutputSpatialSize, paddedOutputOffsets);
      };

      unsigned remainingElemsThisContext = fieldElemsThisContext;
      while (remainingElemsThisContext != 0) {
        const auto elemsThisRow =
            std::min((fieldElemsPerRow - (outOffset % fieldElemsPerRow)),
                     remainingElemsThisContext);
        const auto inOffset = calcInOffset(outOffset);
        const auto paddedOutOffset = calcPaddedOffset(outOffset);
        worklists[kg * numWorkers + context].push_back(inOffset);
        worklists[kg * numWorkers + context].push_back(paddedOutOffset);
        worklists[kg * numWorkers + context].push_back(elemsThisRow);
        useShortTypes &= (inOffset <= maxShortWorklistValue &&
                          paddedOutOffset <= maxShortWorklistValue &&
                          elemsThisRow <= maxShortWorklistValue);
        useShortTypes &= (std::max(elemsThisRow, 5u) - 5 <= maxRptCount);
        outOffset += elemsThisRow;
        remainingElemsThisContext -= elemsThisRow;
      }
    }
    remainingFieldElems -= fieldElemsThisContext;
  }
  assert(remainingFieldElems == 0);

  // Determine whether or not we can use the assembly implementation
  // with short types.
  if (numSubKernels - 1 >
      graph.getMaxVertexFieldValue(shortTypesVertexClass, "numSubKernelsM1")) {
    useShortTypes = false;
  }
  if (convGroupGroups - 1 >
      graph.getMaxVertexFieldValue(shortTypesVertexClass,
                                   "numConvGroupGroupsM1")) {
    useShortTypes = false;
  }

  std::vector<Tensor> inWindow, weightsWindow, outWindow;
  for (unsigned cg = 0; cg < convGroupGroups; ++cg) {
    for (unsigned kg = 0; kg < numSubKernels; ++kg) {
      auto kernelStart = unflattenIndex(kernelSizeGroups, kg);
      auto kernelEnd = kernelStart;
      for (auto &s : kernelEnd) {
        s += 1;
      }
      kernelStart.back() *= 4;
      kernelEnd.back() *= 4;
      for (unsigned ig = 0; ig < inChanGroups; ++ig) {
        for (unsigned og = 0; og < outChanGroups; ++og) {
          const auto window = weightsGrouped[cg][og][ig]
                                  .slice(kernelStart, kernelEnd)
                                  .flatten();
          weightsWindow.push_back(window);
        }
      }
    }
  }

  for (unsigned cg = 0; cg < convGroupGroups; ++cg) {
    for (unsigned og = 0; og < outChanGroups; ++og) {
      const auto window = outGrouped[cg][og].flatten();
      outWindow.push_back(window);
    }

    for (unsigned ig = 0; ig < inChanGroups; ++ig) {
      const auto window = inGrouped[cg][ig].flatten();
      inWindow.push_back(window);
    }
  }

  // We also need an extra buffer for our vertex with size equal the
  // number of output elements per conv group group, plus 8 bytes to
  // enforce (&out[i][0] - &outFieldBuffer[0]) % 16 == 8 so that we
  // can use ld2xst64pace in the codelet even when out and outFieldBuffer
  // reside in the same bank.
  //
  // Additionally, we need 192 bytes (maximum), to store rearranged
  // weights, plus 4 bytes to store a pointer.
  // This isn't actually true for the mode which doesn't
  // use the weight storage (1cgx4ocx4ic) but for now we'll keep it
  // simple and uniform.
  constexpr unsigned extraBytes = 200u;
  assert(extraBytes % 16 == 8);
  assert(extraBytes % target.getTypeSize(partialsType) == 0);
  const auto extraOutputElems = extraBytes / target.getTypeSize(partialsType);
  const auto outFieldBuffer = graph.addVariable(
      partialsType,
      {extraOutputElems + numFieldElems * convGroupsPerGroup * chansPerGroup},
      "outFieldBuffer");
  graph.setTileMapping(outFieldBuffer, 0);

  if (!useShortTypes && !allowCppCodelet) {
    throw poplibs_error("Trying to fall back to c++ codelet, some bounds "
                        "were exceeded. Use --allow-cpp-codelet to continue");
  }

  Sequence prog;

  // fill the output with NaNs
  const auto outGroupedFlattened = outGrouped.flatten();
  const auto nan =
      graph.addConstant(partialsType, outGroupedFlattened.shape(), NAN);
  graph.setTileMapping(nan, 0);
  prog.add(Copy(nan, outGroupedFlattened));

  // Now create a vertex on our 1 tile to perform this convolution.
  const auto cs = graph.addComputeSet("Convolve");
  const auto vertexClass = templateVertex(
      "poplin::ConvPartial1x4SLIC", inputType, partialsType, useShortTypes);
  auto v = graph.addVertex(cs, vertexClass);
  graph.setTileMapping(v, 0);

  graph.connect(v["in"], inWindow);
  graph.connect(v["weights"], weightsWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["outFieldBuffer"], outFieldBuffer);
  graph.setFieldSize(v["worklists"], worklists.size());
  for (unsigned i = 0; i < worklists.size(); ++i) {
    const auto t = graph.addConstant(UNSIGNED_SHORT, {worklists[i].size()},
                                     worklists[i].data(), "worklists");
    graph.setTileMapping(t, 0);
    graph.connect(v["worklists"][i], t);
  }
  graph.setInitialValue(v["mode"], mode);
  graph.setInitialValue(v["outPtrLoadOffset"], (numSubKernels % 2) ? 0 : 4);
  graph.setInitialValue(v["numSubKernelsM1"], numSubKernels - 1);
  graph.setInitialValue(v["numConvGroupGroupsM1"], convGroupGroups - 1);

  prog.add(Execute(cs));

  // explicitly zero all of the output padding.
  const auto zeroCs = graph.addComputeSet("zeroCs");
  {
    // dims before the spatial dims are G1, OC1 and B.
    unsigned spatialDimOffset = 3;

    auto outGroupedPartial = outGrouped;
    for (unsigned d = 0; d < numFieldDims; ++d) {
      const unsigned spatialDim = spatialDimOffset + d;

      const auto &shape = outGroupedPartial.shape();
      const unsigned N = shape[spatialDim];

      // add zeros to the padding.
      popops::zero(
          graph, outGroupedPartial.slice(0, outputPaddingLower[d], spatialDim),
          0, zeroCs);
      popops::zero(
          graph,
          outGroupedPartial.slice(N - outputPaddingUpper[d], N, spatialDim), 0,
          zeroCs);

      // prune the padding off of the tensor so that we don't repad elements
      // when we pad the next dimension.
      outGroupedPartial = outGroupedPartial.slice(
          outputPaddingLower[d], N - outputPaddingUpper[d], spatialDim);
    }
  }
  prog.add(Execute(zeroCs));

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
  std::cout << "in.shape()=";
  printContainer(in.shape(), std::cout);
  std::cout << "\n";
  std::cout << "weights.shape()=";
  printContainer(weights.shape(), std::cout);
  std::cout << "\n";
  std::cout << "output.shape()=";
  printContainer(out.shape(), std::cout);
  std::cout << "\n";
  std::cout << "worklists=";
  printContainer(worklists, std::cout);
  std::cout << "\n";

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostIn, rawHostOut, rawHostWeights;
  if (!ignoreData) {
    rawHostIn = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                            downloadProg, tmap);
    rawHostOut = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                             downloadProg, tmap);
    rawHostWeights = allocateHostMemoryForTensor(
        weights, "weights", graph, uploadProg, downloadProg, tmap);
  }

  OptionFlags engineOptions{{"target.supervisorStackSizeInBytes", "0x200"}};
  Sequence debugSeqIn, debugSeqOut;
  if (deviceType == DeviceType::IpuModel) {
    debugSeqIn.add(PrintTensor(in));
    debugSeqIn.add(PrintTensor(weights));
    debugSeqOut.add(PrintTensor(out));
  }
  Engine engine(
      graph, Sequence(uploadProg, debugSeqIn, prog, debugSeqOut, downloadProg),
      engineOptions);

  attachStreams(engine, tmap);

  boost::multi_array<double, 3> hostIn(
      boost::extents[batchSize][convGroups * inChans][product(inputFieldSize)]);
  boost::multi_array<double, 4> hostWeights(
      boost::extents[convGroups][outChans][inChans][product(kernelSize)]);
  boost::multi_array<double, 3> hostOut(
      boost::extents[batchSize][convGroups * outChans]
                    [product(paddedOutputFieldSize)]);
  // 0 biases just for the reference convolution implementation.
  boost::multi_array<double, 1> hostBiases(
      boost::extents[convGroups * outChans]);
  auto printMultiArrayContents = [](std::ostream &o, const auto &arr) {
    o << "{";
    bool first = true;
    for (std::size_t i = 0; i < arr.num_elements(); ++i) {
      if (!first) {
        o << ",";
      }
      o << arr.data()[i];
      first = false;
    }
    o << "}";
  };

  std::mt19937 randomEngine;
  writeRandomValues(target, inputType, hostIn, -1.0, +5.0, randomEngine);
  writeRandomValues(target, inputType, hostWeights, -1.0, +7.0, randomEngine);

  std::cout << "Input: ";
  printMultiArrayContents(std::cout, hostIn);
  std::cout << "\nWeights: ";
  printMultiArrayContents(std::cout, hostWeights);
  std::cout << "\n";
  copy(target, hostIn, inputType, rawHostIn.get());
  copy(target, hostWeights, inputType, rawHostWeights.get());

  device.bind([&](const Device &d) { engine.loadAndRun(d); });

  if (!ignoreData) {
    boost::multi_array<double, 3> modelOut(
        boost::extents[batchSize][convGroups * outChans]
                      [product(paddedOutputFieldSize)]);
    std::vector<unsigned> modelOutputPaddingLower = outputPaddingLower;
    std::vector<unsigned> modelOutputPaddingUpper = outputPaddingUpper;
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
        std::vector<unsigned>(numFieldDims, 0), // truncationLower
        std::vector<unsigned>(numFieldDims, 0), // truncationUpper
        std::vector<unsigned>(numFieldDims, 1), // stride
        modelOutputPaddingLower,                // paddingLower
        modelOutputPaddingUpper,                // paddingUpper
        // Buffers:
        hostIn, hostWeights, hostBiases, modelOut);

    copy(target, partialsType, rawHostOut.get(), hostOut);

    std::cout << "\nOutputs[" << hostOut.num_elements() << "]: ";
    printMultiArrayContents(std::cout, hostOut);
    std::cout << "\n";

    std::cout << "\nModel[" << modelOut.num_elements() << "]: ";
    printMultiArrayContents(std::cout, modelOut);
    std::cout << "\n";

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
