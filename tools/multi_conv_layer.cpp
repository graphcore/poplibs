// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"
#include "poplin/MultiConvolution.hpp"
#include <boost/assign/list_of.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <boost/version.hpp>
#include <iostream>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <sstream>

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  boost::optional<unsigned> tilesPerIPU;
  std::vector<std::string> convs;
  bool transposeAndFlipWeights;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help,h", "produce help message")
    ("device-type",
      po::value<DeviceType>(&deviceType)->default_value(DeviceType::IpuModel),
      "Device type: Cpu | Sim | Sim2 | Hw | IpuModel | IpuModel2")
    ("profile", "Output profiling report to standard output")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("tiles-per-ipu", po::value(&tilesPerIPU), "Number of tiles per IPU")
    ("conv", po::value<std::vector<std::string>>(&convs)
      ->default_value(boost::assign::list_of(""), "")->composing(),
     "parameters for a convolution used in the multiconv")
    ("transpose-flip-weights",
     po::value<bool>(&transposeAndFlipWeights)->default_value(false),
     "whether or not to transpose and flip the weights before each convolution")
    ;
  // clang-format on

  po::variables_map vm;
  try {
    const po::positional_options_description p;
    po::store(
        po::command_line_parser(argc, argv).options(desc).positional(p).run(),
        vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const boost::program_options::error &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  const bool profile = deviceType != DeviceType::Cpu && vm.count("profile");
  const bool ignoreData = vm.count("ignore-data");

  std::cout << "got " << convs.size() << " convs" << std::endl;

  const unsigned numIPUs = 1;
  const bool compileIPUCode = true;
  auto device =
      tilesPerIPU
          ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, compileIPUCode)
          : createTestDeviceFullSize(deviceType, numIPUs, compileIPUCode);

  const auto &target = device.getTarget();
  poplar::Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  poplin::PlanningCache cache;
  poplar::program::Sequence uploadProg, prog, downloadProg;

  const poplar::OptionFlags multiConvOptions;
  const std::vector<poplar::OptionFlags> options(convs.size());
  std::vector<poplin::ConvParams> params;
  for (const auto &conv : convs) {
    std::stringstream ss;
    ss << conv;
    poplin::ConvParams p;
    ss >> p;

    params.push_back(std::move(p));
  }

  std::vector<poplin::multiconv::CreateTensorArgs> createInputArgs;
  std::vector<poplin::multiconv::CreateTensorArgs> createWeightsArgs;
  for (unsigned i = 0; i < params.size(); ++i) {
    createInputArgs.push_back(
        {params[i], options[i], "convInput_" + std::to_string(i)});
    createWeightsArgs.push_back(
        {params[i], options[i], "convWeights_" + std::to_string(i)});
  }

  std::vector<poplin::multiconv::ConvolutionArgs> convolutionArgs;
  for (unsigned i = 0; i < params.size(); ++i) {
    auto input = poplin::multiconv::createInput(graph, createInputArgs, i,
                                                multiConvOptions, &cache);
    auto weights = poplin::multiconv::createWeights(graph, createWeightsArgs, i,
                                                    multiConvOptions, &cache);

    convolutionArgs.push_back(
        {std::move(input), std::move(weights), params[i], options[i]});
  }

  const auto outs = poplin::multiconv::convolution(
      graph, convolutionArgs, transposeAndFlipWeights, prog, "multiConv",
      multiConvOptions);

  std::vector<std::pair<std::string, char *>> tmap;
  std::vector<std::unique_ptr<char[]>> rawHostInputs, rawHostWeights,
      rawHostOutputs;
  if (!ignoreData) {
    for (unsigned i = 0; i < convolutionArgs.size(); ++i) {
      auto rawHostInput = poplibs_test::util::allocateHostMemoryForTensor(
          convolutionArgs[i].inputs, createInputArgs[i].name, graph, uploadProg,
          boost::none, tmap);
      rawHostInputs.push_back(std::move(rawHostInput));

      auto rawHostWeight = poplibs_test::util::allocateHostMemoryForTensor(
          convolutionArgs[i].weights, createWeightsArgs[i].name, graph,
          uploadProg, boost::none, tmap);
      rawHostWeights.push_back(std::move(rawHostWeight));

      auto rawHostOutput = poplibs_test::util::allocateHostMemoryForTensor(
          outs[i], "output_" + std::to_string(i), graph, boost::none,
          downloadProg, tmap);
      rawHostOutputs.push_back(std::move(rawHostOutput));
    }
  }

  const poplar::OptionFlags engineOptions;
  poplar::Engine engine(graph, {uploadProg, prog, downloadProg}, engineOptions);

  std::vector<boost::multi_array<double, 3>> hostInputs;
  std::vector<boost::multi_array<double, 4>> hostWeights;
  std::vector<boost::multi_array<double, 3>> modelOutputs;
  if (!ignoreData) {
    poplibs_test::util::attachStreams(engine, tmap);

    std::mt19937 randomEngine;
    for (unsigned i = 0; i < convolutionArgs.size(); ++i) {
      const auto &p = createInputArgs[i].params;
      const auto inChannels = p.inputChannelsPerConvGroup * p.numConvGroups;
      const auto outChannels = p.outputChannelsPerConvGroup * p.numConvGroups;

      hostInputs.emplace_back(
          boost::extents[p.batchSize][inChannels][product(p.inputFieldShape)]);
      poplibs_test::util::writeRandomBinaryValues(
          target, p.inputType, hostInputs.back(), -1.0, 1.0, randomEngine);
      poplibs_test::util::copy(target, hostInputs.back(), p.inputType,
                               rawHostInputs[i].get());

      hostWeights.emplace_back(
          boost::extents[p.numConvGroups][p.outputChannelsPerConvGroup]
                        [p.inputChannelsPerConvGroup][product(p.kernelShape)]);
      poplibs_test::util::writeRandomBinaryValues(
          target, p.inputType, hostWeights.back(), -1.0, 1.0, randomEngine);
      poplibs_test::util::copy(target, hostWeights.back(), p.inputType,
                               rawHostWeights[i].get());

      // build a reference model to validate against
      boost::multi_array<double, 1> biases(boost::extents[outChannels]);
      std::fill(biases.data(), biases.data() + biases.num_elements(), 0.0);

      const auto outFieldShape = p.getOutputFieldShape();
      modelOutputs.emplace_back(
          boost::extents[p.batchSize][outChannels][product(outFieldShape)]);

      poplibs_test::conv::convolution(
          vectorConvert<unsigned>(p.inputFieldShape),
          p.inputTransform.truncationLower, p.inputTransform.truncationUpper,
          p.inputTransform.dilation, p.inputTransform.paddingLower,
          p.inputTransform.paddingUpper, p.inputTransform.flip,
          vectorConvert<unsigned>(p.kernelShape),
          p.kernelTransform.truncationLower, p.kernelTransform.truncationUpper,
          p.kernelTransform.dilation, p.kernelTransform.paddingLower,
          p.kernelTransform.paddingUpper, p.kernelTransform.flip,
          p.outputTransform.truncationLower, p.outputTransform.truncationUpper,
          p.outputTransform.stride, p.outputTransform.paddingLower,
          p.outputTransform.paddingUpper, hostInputs.back(), hostWeights.back(),
          biases, modelOutputs.back());
    }
  }

  device.bind([&](const poplar::Device &d) {
    engine.load(d);
    if (!ignoreData) {
      // upload
      engine.run(0);
    }

    // convolve
    engine.run(1);

    if (!ignoreData) {
      // download
      engine.run(2);
    }
  });

  bool matchesModel = true;
  if (!ignoreData) {
    for (unsigned i = 0; i < convolutionArgs.size(); ++i) {
      const auto &p = convolutionArgs[i].params;
      const auto outFieldShape = p.getOutputFieldShape();
      const auto outChannels = p.outputChannelsPerConvGroup * p.numConvGroups;

      boost::multi_array<double, 3> hostOutput(
          boost::extents[p.batchSize][outChannels][product(outFieldShape)]);
      poplibs_test::util::copy(target, p.outputType, rawHostOutputs[i].get(),
                               hostOutput);

      const auto tolerance = 0.0;
      matchesModel &= poplibs_test::util::checkIsClose(
          "conv_" + std::to_string(i), hostOutput, modelOutputs[i], tolerance,
          tolerance);
    }
  }

  if (profile) {
    engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
}
