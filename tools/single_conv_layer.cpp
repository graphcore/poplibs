// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/print.hpp"
#include <poplibs_support/TestDevice.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/Collectives.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/GraphFunction.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <istream>
#include <ostream>
#include <random>

// Default tolerances used in tests with uniform distributions of random data
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace poplibs_support;
using poplibs_test::Pass;

const OptionFlags defaultEngineOptions;

static void overloadConstraintsFromFile(const std::string &path,
                                        std::string &s) {
  if (!path.empty()) {
    std::ifstream is(path, std::ios_base::in);
    if (!is.good()) {
      throw poputil::poplibs_error("Constraints file " + path +
                                   " could not be opened");
    }
    is.seekg(0, std::ios::end);
    const auto bytes = is.tellg();
    s = std::string(bytes, '\0');
    is.seekg(0);
    is.read(&s[0], bytes);
  }
}

static Tensor createGenericConvInput(Graph &graph,
                                     const poplin::ConvParams &params,
                                     const std::string &name = "") {
  return poplibs_test::util::createGenericConvInput(
      graph, params.inputType, params.getBatchSize(), params.getNumConvGroups(),
      params.getNumInputChansPerConvGroup(), params.getInputFieldShape(), name);
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  unsigned fwdInChansPerConvGroup;
  unsigned fwdOutChansPerConvGroup;
  ShapeOption<std::size_t> inputFieldSizeOption;
  ShapeOption<std::size_t> kernelSizeOption;
  unsigned numConvGroups = 1;
  ShapeOption<unsigned> truncationLowerOption, truncationUpperOption,
      truncationOption;
  ShapeOption<unsigned> inDilationOption;
  ShapeOption<unsigned> paddingLowerOption, paddingUpperOption, paddingOption;
  ShapeOption<bool> flipInputOption;
  ShapeOption<unsigned> kernelTruncationLowerOption,
      kernelTruncationUpperOption, kernelTruncationOption;
  ShapeOption<unsigned> kernelDilationOption;
  ShapeOption<unsigned> kernelPaddingLowerOption, kernelPaddingUpperOption,
      kernelPaddingOption;
  ShapeOption<bool> flipKernelOption;
  ShapeOption<unsigned> outputTruncationOption, outputTruncationLowerOption,
      outputTruncationUpperOption;
  ShapeOption<unsigned> strideOption;
  ShapeOption<unsigned> outputPaddingOption, outputPaddingLowerOption,
      outputPaddingUpperOption;
  unsigned batchSize;
  bool bias;
  Type inputType;
  Type outputType;
  double absoluteTolerance, relativeTolerance;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  bool reportPlan;
  bool reportVarStorage;
  unsigned replicationFactor;
  unsigned numDeterminismChecks;
  bool enableConvolutionReuse;
  bool remapOutputTensor;
  bool useCreateInput;
  bool preplan;

  Pass pass = Pass::ALL;
  std::string fwdPlanConstraints, fwdPlanConstraintsFile, bwdPlanConstraints,
      bwdPlanConstraintsFile, wuPlanConstraints, wuPlanConstraintsFile,
      convOptionsString;
  poplin::PlanningCache cache;

  boost::optional<std::string> jsonProfileOut;
  boost::optional<std::string> profileFormat;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling report")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("use-unstable-format", "Deprecated: use \"--profile-format experimental\"")
    ("profile-format",
     po::value<decltype(profileFormat)>(&profileFormat)
      ->default_value(boost::none),
     "Profile formats: v1 | experimental | unstable")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("input-channels", po::value<unsigned>(&fwdInChansPerConvGroup)->required(),
     "Number of input channels per grouped convolution")
    ("output-channels",
     po::value<unsigned>(&fwdOutChansPerConvGroup)->required(),
     "Number of output channels per grouped convolution")
    ("field",
     po::value<ShapeOption<std::size_t>>(&inputFieldSizeOption)->required(),
      "Field size")
    ("kernel-size",
      po::value<ShapeOption<std::size_t>>(&kernelSizeOption)->default_value(1),
     "Size of square kernel. If set, it is an error to also set either "
     "kernel-height and/or kernel-width")
    ("bias", po::value<bool>(&bias)->default_value(true),
     "Add a bias to each channel")
    ("data-type",
     po::value<Type>(&inputType)->default_value(HALF),
     "Type of the input and output data")
    ("input-type",
     po::value<Type>(&inputType),
     "Type of the input data")
    ("output-type",
     po::value<Type>(&outputType),
     "Type of the output data and the parameters")
    ("truncation",
     po::value<ShapeOption<unsigned>>(&truncationOption)->default_value(0),
     "Amount to truncate the start and end of each dimension of the input")
    ("truncation-upper",
     po::value<ShapeOption<unsigned>>(&truncationUpperOption)->default_value(0),
     "Amount to truncate the end of each dimension of the input")
    ("truncation-lower",
     po::value<ShapeOption<unsigned>>(&truncationLowerOption)->default_value(0),
     "Amount to truncate the start of each dimension of the input")
    ("in-dilation",
     po::value<ShapeOption<unsigned>>(&inDilationOption)->default_value(1),
     "Input dilation")
    ("padding",
     po::value<ShapeOption<unsigned>>(&paddingOption)->default_value(0),
     "Amount of zero padding to add to the start and end of each dimension")
    ("padding-upper",
     po::value<ShapeOption<unsigned>>(&paddingUpperOption)->default_value(0),
     "Amount of zero padding to add at the end of each dimension")
    ("padding-lower",
     po::value<ShapeOption<unsigned>>(&paddingLowerOption)->default_value(0),
     "Amount of zero padding to add at the start of each dimension")
    ("flip-input",
     po::value<ShapeOption<bool>>(&flipInputOption)->default_value(false),
     "Whether to flip each input spatial field")
    ("kernel-truncation",
     po::value<ShapeOption<unsigned>>(&kernelTruncationOption)
         ->default_value(0),
     "Amount to truncate the start and end of each dimension of the kernel")
    ("kernel-truncation-upper",
     po::value<ShapeOption<unsigned>>(&kernelTruncationUpperOption)
         ->default_value(0),
     "Amount to truncate the end of each dimension of the kernel")
    ("kernel-truncation-lower",
     po::value<ShapeOption<unsigned>>(&kernelTruncationLowerOption)
         ->default_value(0),
     "Amount to truncate the start of each dimension of the kernel")
    ("kernel-dilation",
     po::value<ShapeOption<unsigned>>(&kernelDilationOption)
         ->default_value(1),
     "Kernel dilation")
    ("kernel-padding",
     po::value<ShapeOption<unsigned>>(&kernelPaddingOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the start and end of each"
     "dimension")
    ("kernel-padding-upper",
     po::value<ShapeOption<unsigned>>(&kernelPaddingUpperOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the start of each dimension")
    ("kernel-padding-lower",
     po::value<ShapeOption<unsigned>>(&kernelPaddingLowerOption)
         ->default_value(0),
     "Amount of zero kernel padding to add at the end of each dimension")
    ("flip-kernel",
     po::value<ShapeOption<bool>>(&flipKernelOption)->default_value(false),
     "Whether to flip each kernel spatial field")
    ("output-truncation",
     po::value<ShapeOption<unsigned>>(&outputTruncationOption)
         ->default_value(0),
     "Number of output elements to truncate")
    ("output-truncation-upper",
     po::value<ShapeOption<unsigned>>(&outputTruncationUpperOption)
         ->default_value(0),
     "Number of output elements to truncate at the end of each dimension")
    ("output-truncation-lower",
     po::value<ShapeOption<unsigned>>(&outputTruncationLowerOption)
         ->default_value(0),
     "Number of output elements to truncate at the start of each dimension")
    ("stride",
     po::value<ShapeOption<unsigned>>(&strideOption)->default_value(1),
     "Stride")
    ("output-padding",
     po::value<ShapeOption<unsigned>>(&outputPaddingOption)->default_value(0),
     "Number of output elements to truncate")
    ("output-padding-upper",
     po::value<ShapeOption<unsigned>>(&outputPaddingUpperOption)
         ->default_value(0),
     "Number of output elements to truncate at the end of each dimension")
    ("output-padding-lower",
     po::value<ShapeOption<unsigned>>(&outputPaddingLowerOption)
         ->default_value(0),
     "Number of output elements to truncate at the start of each dimension")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("plan-only", "Only plan the requested passes, don't build or run a graph")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model. Upon failure, the error code relates to what multiple of this tolerance failed. "
     "Error code 11 -> 1-2x; 12 -> 2-3x; 13 -> 3-4x, e.g. tolerance of 2 and failure of 5, returns error code 12.")
    ("absolute-tolerance",
     po::value<double>(&absoluteTolerance),
     "Absolute tolerance to use when validating results against the reference "
     "model. Upon failure, the error code relates to what multiple of this tolerance failed. "
     "Error code 11 -> 1-2x; 12 -> 2-3x; 13 -> 3-4x, e.g. tolerance of 2 and failure of 5, returns error code 12.")
    ("ipus",
     po::value<unsigned>(&numIPUs)->default_value(numIPUs),
     "Number of IPUs")
    ("tiles-per-ipu", po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("workers-per-tile",
     po::value<unsigned>(),
     "Number of worker contexts per tile")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("conv-groups",
     po::value<unsigned>(&numConvGroups)->default_value(1),
     "Number of convolution groups in grouped convolution")
    ("fwd-plan-constraints",
     po::value<std::string>(&fwdPlanConstraints)
        ->default_value(fwdPlanConstraints),
     "Constraints on the chosen convolution plan for the forward pass "
     "as a JSON string")
    ("fwd-plan-constraints-file",
     po::value<std::string>(&fwdPlanConstraintsFile)
        ->default_value(fwdPlanConstraintsFile),
     "Constraints on the chosen convolution plan for the forward pass "
     "as a path to a JSON file")
    ("bwd-plan-constraints",
     po::value<std::string>(&bwdPlanConstraints)
        ->default_value(bwdPlanConstraints),
     "Constraints on the chosen convolution plan for the backward pass "
     "as a JSON string")
    ("bwd-plan-constraints-file",
     po::value<std::string>(&bwdPlanConstraintsFile)
        ->default_value(bwdPlanConstraintsFile),
     "Constraints on the chosen convolution plan for the backward pass "
     "as a path to a JSON file")
    ("wu-plan-constraints",
     po::value<std::string>(&wuPlanConstraints)
        ->default_value(wuPlanConstraints),
     "Constraints on the chosen convolution plan for the weight update pass "
     "as a JSON string")
    ("wu-plan-constraints-file",
     po::value<std::string>(&wuPlanConstraintsFile)
        ->default_value(wuPlanConstraintsFile),
     "Constraints on the chosen convolution plan for the weight update pass "
     "as a path to a JSON file")
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
    ("report-var-storage",
     po::value<bool>(&reportVarStorage)->default_value(false),
     "Report variable storage information")
    ("replication-factor",
     po::value<unsigned>(&replicationFactor)->default_value(1),
     "Number of parallel copies of the graph to run. Each copy of the graph "
     "shares the same parameters but reads different input samples. The "
     "effective batch size is the batch size of the graph multiplied by the "
     "replication factor")
    ("remap-output-tensor",
     po::value<bool>(&remapOutputTensor)->default_value(false),
     "Remap output tensor if layout is detected to be poor")
    ("enable-convolution-reuse",
     po::value<bool>(&enableConvolutionReuse)->default_value(true),
     "Apply optimization to reuse the forward convolution in the backward pass")
    ("convolution-options", po::value<std::string>(&convOptionsString),
    "Options to use for the convolution, specified as a JSON string, "
    "e.g. {\"key\":\"value\"}")
    ("use-create-input",
     po::value<bool>(&useCreateInput)->default_value(true),
     "Use the input allocation function to create an input tensor with a "
     "layout optimised for the plan. If set to false use a generic layout that "
     "is independent of the plan and representative of a typical layout in a "
     "neural network")
    ("num-determinism-checks",
     po::value<unsigned>(&numDeterminismChecks)->default_value(0),
     "The amount of additional identical executions (results are compared to check determinism)."
     "This option is required to be 0 if ignore-data is set or single-phase is not 'all' or device-type is not Hw.")
    ("preplan",
     po::value<bool>(&preplan)->default_value(true),
     "Whether or not to preplan the convolutions")
  ;
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      std::cout << "A multi-dimensional shape can be specified using a brace "
                   "enclosed comma\n"
                   "separated list, for example --stride={1,2}. You may also "
                   "specify a single\n"
                   "number without braces in which case that value is used for "
                   "each dimension,\n"
                   "for example --stride=2\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  auto &inputFieldSize = inputFieldSizeOption.val;
  const auto numFieldDims = inputFieldSize.size();

  kernelSizeOption.broadcast(numFieldDims);
  auto &kernelSize = kernelSizeOption.val;

  struct UpperLowerOption {
    ShapeOption<unsigned> &lowerOption;
    ShapeOption<unsigned> &upperOption;
    std::string name;
  } upperLowerOptionTriples[] = {
      {paddingLowerOption, paddingUpperOption, "padding"},
      {truncationLowerOption, truncationUpperOption, "truncation"},
      {kernelTruncationLowerOption, kernelTruncationUpperOption,
       "kernel-truncation"},
      {kernelPaddingLowerOption, kernelPaddingUpperOption, "kernel-padding"},
      {outputTruncationLowerOption, outputTruncationUpperOption,
       "output-truncation"},
      {outputPaddingLowerOption, outputPaddingUpperOption, "output-padding"}};
  for (const auto &entry : upperLowerOptionTriples) {
    if (!vm[entry.name].defaulted()) {
      std::string conflictingOptions[] = {entry.name + "-lower",
                                          entry.name + "-upper"};
      for (auto option : conflictingOptions) {
        if (!vm[option].defaulted()) {
          std::cerr << "--" << entry.name << " as well as --";
          std::cerr << option << " set\n";
          return 1;
        }
      }
      entry.lowerOption = vm[entry.name].as<ShapeOption<unsigned>>();
      entry.upperOption = vm[entry.name].as<ShapeOption<unsigned>>();
    }
    entry.lowerOption.broadcast(numFieldDims);
    entry.upperOption.broadcast(numFieldDims);
  }
  auto &truncationLower = truncationLowerOption.val;
  auto &truncationUpper = truncationUpperOption.val;
  auto &paddingLower = paddingLowerOption.val;
  auto &paddingUpper = paddingUpperOption.val;

  auto &outputTruncationLower = outputTruncationLowerOption.val;
  auto &outputTruncationUpper = outputTruncationUpperOption.val;
  auto &outputPaddingLower = outputPaddingLowerOption.val;
  auto &outputPaddingUpper = outputPaddingUpperOption.val;

  auto &kernelTruncationLower = kernelTruncationLowerOption.val;
  auto &kernelTruncationUpper = kernelTruncationUpperOption.val;
  auto &kernelPaddingLower = kernelPaddingLowerOption.val;
  auto &kernelPaddingUpper = kernelPaddingUpperOption.val;

  inDilationOption.broadcast(numFieldDims);
  auto &inDilation = inDilationOption.val;
  flipInputOption.broadcast(numFieldDims);
  auto &flipInput = flipInputOption.val;

  kernelDilationOption.broadcast(numFieldDims);
  auto &kernelDilation = kernelDilationOption.val;
  flipKernelOption.broadcast(numFieldDims);
  auto &flipKernel = flipKernelOption.val;

  strideOption.broadcast(numFieldDims);
  auto &stride = strideOption.val;
  const auto fwdInChans = fwdInChansPerConvGroup * numConvGroups;
  const auto fwdOutChans = fwdOutChansPerConvGroup * numConvGroups;

  const bool planOnly = vm.count("plan-only");
  const bool inferenceOnly = vm.count("inference-only");
  const bool ignoreData = vm.count("ignore-data");
  if (vm.count("use-unstable-format")) {
    throw poputil::poplibs_error("\"--use-unstable-format\" is deprecated. Use "
                                 "\"--profile-format experimental\" instead");
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
  if (vm["absolute-tolerance"].empty()) {
    if (outputType == FLOAT) {
      absoluteTolerance = FLOAT_ABS_TOL;
    } else {
      absoluteTolerance = HALF_ABS_TOL;
    }
  }

  auto dev = [&]() -> TestDevice {
    if (isIpuModel(deviceType)) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      IPUModel ipuModel(deviceTypeToIPUName(deviceType));
      ipuModel.numIPUs = numIPUs;
      if (vm.count("profile") || jsonProfileOut) {
        ipuModel.compileIPUCode = true;
      }
      if (vm.count("workers-per-tile"))
        ipuModel.numWorkerContexts = vm["workers-per-tile"].as<unsigned>();
      if (tilesPerIPU)
        ipuModel.tilesPerIPU = *tilesPerIPU;
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      if (tilesPerIPU)
        return createTestDevice(deviceType, numIPUs, *tilesPerIPU);
      else
        return createTestDeviceFullSize(deviceType, numIPUs);
    }
  }();

  if (numDeterminismChecks && (ignoreData || !doWuPass)) {
    throw poputil::poplibs_error(
        "Determinism checks cannot ignore data or avoid weight upload pass.");
  }
  if (numDeterminismChecks && deviceType != DeviceType::Hw) {
    throw poputil::poplibs_error(
        "Determinism checks only work on Hardware device");
  }

  Graph parentGraph(dev.getTarget());
  popops::addCodelets(parentGraph);
  poplin::addCodelets(parentGraph);
  auto graph = parentGraph.createReplicatedGraph(replicationFactor);

  const poplin::ConvParams::InputTransform inputTransform{
      truncationLower, truncationUpper, inDilation,
      paddingLower,    paddingUpper,    flipInput};
  const poplin::ConvParams::InputTransform kernelTransform{
      kernelTruncationLower, kernelTruncationUpper, kernelDilation,
      kernelPaddingLower,    kernelPaddingUpper,    flipKernel,
  };
  const poplin::ConvParams::OutputTransform outputTransform{
      outputTruncationLower, outputTruncationUpper, stride, outputPaddingLower,
      outputPaddingUpper};
  const auto params = poplin::ConvParams{inputType,
                                         outputType,
                                         batchSize,
                                         inputFieldSize,
                                         kernelSize,
                                         fwdInChansPerConvGroup,
                                         fwdOutChansPerConvGroup,
                                         numConvGroups,
                                         inputTransform,
                                         kernelTransform,
                                         outputTransform};

  const auto outFieldSize = params.getOutputFieldShape();
  const auto bwdParams = getGradientParams(params);

  OptionFlags convOptions;
  convOptions.set(
      {{"remapOutputTensor", remapOutputTensor ? "true" : "false"}});

  if (!convOptionsString.empty()) {
    poplar::readJSON(convOptionsString, convOptions);
  }

  auto fwdOptions = convOptions;
  fwdOptions.set("pass", inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  overloadConstraintsFromFile(fwdPlanConstraintsFile, fwdPlanConstraints);
  fwdOptions.set("planConstraints", fwdPlanConstraints);
  auto bwdOptions = convOptions;
  bwdOptions.set("pass", "TRAINING_BWD");
  bwdOptions.set("planConstraints", bwdPlanConstraints);
  overloadConstraintsFromFile(bwdPlanConstraintsFile, bwdPlanConstraints);
  auto wuOptions = convOptions;
  wuOptions.set("pass", "TRAINING_WU");
  overloadConstraintsFromFile(wuPlanConstraintsFile, wuPlanConstraints);
  wuOptions.set("planConstraints", wuPlanConstraints);

  if (preplan) {
    const auto &replicatedTarget = graph.getTarget();
    std::set<poplin::ConvPlanParams> convs;

    if (doFwdPass) {
      convs.insert(std::make_tuple(&replicatedTarget, params, &fwdOptions));
    }

    if (doBwdPass) {
      convs.insert(std::make_tuple(&replicatedTarget, bwdParams, &bwdOptions));
    }

    if (doWuPass) {
      auto wuParams = getWeightUpdateParams(params);
      convs.insert(
          std::make_tuple(&replicatedTarget, std::move(wuParams), &wuOptions));
    }

    poplin::preplanConvolutions(graph, convs, cache);
  }

  if (reportPlan) {
    std::cout << "Convolution parameters:\n"
                 " Batch size: "
              << params.batchSize
              << "\n"
                 " Kernel:"
              << params.kernelShape
              << "\n"
                 " Stride:"
              << params.outputTransform.stride
              << "\n"
                 " Padding Lower: "
              << params.inputTransform.paddingLower
              << "\n"
                 " Padding Upper: "
              << params.inputTransform.paddingUpper
              << "\n"
                 " Group size: "
              << params.numConvGroups
              << "\n"
                 " Input: "
              << params.inputChannelsPerConvGroup << "x"
              << params.inputFieldShape
              << "\n"
                 " Output: "
              << params.outputChannelsPerConvGroup << "x" << outFieldSize
              << "\n";

    if (doFwdPass) {
      std::cout << "Forward plan:\n";
      poplin::reportPlanInfo(std::cout, graph, params, fwdOptions, &cache);
      std::cout << "Forward FLOPs: " << getFwdFlops(params) << "\n";
    }

    if (doBwdPass) {
      std::cout << "Backward plan:\n";
      poplin::reportPlanInfo(std::cout, graph, bwdParams, bwdOptions, &cache);
      std::cout << "Backward FLOPs: " << getBwdFlops(bwdParams) << "\n";
    }

    if (doWuPass) {
      std::cout << "WU plan:\n";
      poplin::reportWeightUpdatePlanInfo(std::cout, graph, params, wuOptions,
                                         &cache);
      std::cout << "WU FLOPs: " << getWuFlops(params) << "\n";
    }

    if (planOnly) {
      return 0;
    }
  }

  const auto &target = parentGraph.getTarget();
  std::size_t maxAccsPerOutputElement = 0;
  if (doFwdPass) {
    const std::size_t numOutElems = product(params.getOutputFieldShape()) *
                                    params.getNumOutputChans() *
                                    params.getBatchSize();
    if (numOutElems) {
      const std::size_t fwdMaxAccsPerOutElem =
          (poplin::getFwdFlops(params) / 2) / numOutElems;
      maxAccsPerOutputElement =
          std::max(maxAccsPerOutputElement, fwdMaxAccsPerOutElem);
    }
  }
  if (doBwdPass) {
    const std::size_t numOutElems = product(params.getInputFieldShape()) *
                                    params.getNumInputChans() *
                                    params.getBatchSize();
    if (numOutElems) {
      const std::size_t bwdMaxAccsPerOutElem =
          (poplin::getBwdFlops(params) / 2) / numOutElems;
      maxAccsPerOutputElement =
          std::max(maxAccsPerOutputElement, bwdMaxAccsPerOutElem);
    }
  }
  if (doWuPass) {
    const std::size_t numOutElems = product(params.getKernelShape()) *
                                    params.getNumInputChans() *
                                    params.getNumOutputChans();
    if (numOutElems) {
      const std::size_t wuMaxAccsPerOutElem =
          (poplin::getWuFlops(params) / 2) / numOutElems;
      maxAccsPerOutputElement =
          std::max(maxAccsPerOutputElement, wuMaxAccsPerOutElem);
    }
  }

  // To avoid destructive addition (a + b) + c != a + (b + c), which is
  // particularly poor with halves, we look to only using values which we can
  // represent exactly, such that (a + b) + c == a + (b + c).
  // Accumulating random binary distribution of {-1, 1}, with a mean of 0
  // provides us this most of the time. Such that accumulating many items from
  // this distribution is unlikely to be not exactly representable as it should
  // generally be a small number. It's less likely the larger the convolution
  // however.

  // We also disable this for determinism checks, which is testing stochastic
  // rounding specifically.
  const bool useUniformRandomData =
      numDeterminismChecks ||
      isLikelyToHaveNumericalErrorsUsingBernoulli(maxAccsPerOutputElement,
                                                  inputType, outputType);

  if (useUniformRandomData) {
    std::cout << "Using uniform random data\n";
  } else {
    std::cout << "Using random binary {-1,1} data with no error tolerance\n";
    if (vm["tolerance"].empty()) {
      relativeTolerance = 0;
    }
    if (vm["absolute-tolerance"].empty()) {
      absoluteTolerance = 0;
    }
  }

  // Create tensors.
  Tensor prevAct;
  if (useCreateInput) {
    prevAct = poplin::createInput(graph, params, "prevAct", fwdOptions, &cache);
  } else {
    prevAct = createGenericConvInput(graph, params, "prevAct");
  }
  Tensor weights =
      poplin::createWeights(graph, params, "weights", fwdOptions, &cache);

  Tensor prevDeltas, zDeltas;
  if (doBwdPass || doWuPass) {
    if (useCreateInput) {
      zDeltas =
          poplin::createInput(graph, bwdParams, "zDeltas", bwdOptions, &cache);
    } else {
      zDeltas = createGenericConvInput(graph, bwdParams, "zDeltas");
    }
  }

  // Always generate the fwd program as it maps the weights and biases. Only
  // actually create the engined if the fwd pass is to be run
  auto fwdProg = Sequence();

  // create the forward convolution as a tensor function as we may be able to
  // reuse it for the backwards pass.
  auto fwdConv = [&]() -> graphfn::TensorFunction {
    using graphfn::input;

    const auto conv = [&](std::vector<Tensor> &args, Sequence &prog) {
      return poplin::convolution(graph, args[0], args[1], params, false, prog,
                                 "fwd", fwdOptions, &cache);
    };

    return {graph, {input(prevAct, "in"), input(weights, "weights")}, conv};
  }();

  std::vector<Tensor> fwdArgs{prevAct, weights};
  Tensor nextAct = fwdConv(fwdArgs, fwdProg);

  Tensor biases;
  if (bias) {
    biases = poplin::createBiases(graph, nextAct);
    poplin::addBias(graph, nextAct, biases, fwdProg, "bias");
  }
  if (!doFwdPass) {
    fwdProg = Sequence();
  }

  auto revProg = Sequence();
  const auto learningRate = [&]() {
    if (useUniformRandomData) {
      return 0.05;
    } else {
      return 1.0;
    }
  }();

  if (doBwdPass) {
    // we may be able to reuse the forward pass convolution if the convoltution
    // is symmetrical.
    if (enableConvolutionReuse &&
        params.canonicalize() == bwdParams.canonicalize()) {
      // transform the weights prior to the convolution so we can reuse the
      // existing sub-graph.
      auto bwdWeights = poplin::createWeights(graph, bwdParams, "bwdWeights",
                                              fwdOptions, &cache);
      poplin::weightsTransposeChansFlipXY(graph, weights, bwdWeights, revProg,
                                          "bwd");

      std::vector<Tensor> bwdArgs{zDeltas, bwdWeights};
      prevDeltas = fwdConv(bwdArgs, revProg);
    } else {
      prevDeltas = poplin::convolution(graph, zDeltas, weights, bwdParams, true,
                                       revProg, "bwd", bwdOptions, &cache);
    }
  }
  if (doWuPass) {
    if (replicationFactor == 1) {
      auto scale = graph.addConstant(weights.elementType(), {}, -learningRate);
      graph.setTileMapping(scale, 0);
      poplin::convolutionWeightUpdate(graph, zDeltas, weights, prevAct, params,
                                      scale, revProg, "wu", wuOptions, &cache);

    } else {
      auto weightDeltas = poplin::calculateWeightDeltas(
          graph, zDeltas, prevAct, params, revProg, "wu", wuOptions, &cache);
      auto weightDeltasReduced =
          popops::replicatedAllReduce(graph, parentGraph, weightDeltas,
                                      popops::CollectiveOperator::ADD, revProg);
      popops::scaledAddTo(graph, weights, weightDeltasReduced, -learningRate,
                          revProg, "wu/UpdateWeights");
    }
    if (bias) {
      if (replicationFactor == 1) {
        auto scale = graph.addConstant(FLOAT, {}, -learningRate);
        graph.setTileMapping(scale, 0);
        poplin::convolutionBiasUpdate(graph, zDeltas, biases, scale,
                                      convOptions, revProg);
      } else {
        std::vector<std::size_t> reduceDims(zDeltas.rank() - 1);
        std::iota(std::next(reduceDims.begin()), reduceDims.end(), 2);
        auto biasDeltas = graph.clone(biases, "biasDeltas");
        popops::reduceWithOutput(graph, zDeltas, biasDeltas, reduceDims,
                                 popops::Operation::ADD, revProg,
                                 "wu/CalcBiasDeltas");
        auto biasDeltasReduced = popops::replicatedAllReduce(
            graph, parentGraph, biasDeltas, popops::CollectiveOperator::ADD,
            revProg);
        popops::scaledAddTo(graph, biases, biasDeltasReduced, -learningRate,
                            revProg, "wu/UpdateBiases");
      }
    }
  }
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto parentPrevAct = parentGraph.getNonReplicatedTensor(prevAct);
  auto rawHostPrevAct = allocateHostMemoryForTensor(
      parentPrevAct, "prevAct", parentGraph, uploadProg, downloadProg, tmap);
  auto parentWeights = parentGraph.getNonReplicatedTensor(weights);
  auto rawHostWeights = allocateHostMemoryForTensor(
      parentWeights, "weights", parentGraph, uploadProg, downloadProg, tmap);
  Tensor parentBiases;
  std::unique_ptr<char[]> rawHostBiases;
  if (bias) {
    parentBiases = parentGraph.getNonReplicatedTensor(biases);
    rawHostBiases = allocateHostMemoryForTensor(
        parentBiases, "biases", parentGraph, uploadProg, downloadProg, tmap);
  }
  auto parentNextAct = parentGraph.getNonReplicatedTensor(nextAct);
  auto rawHostNextAct = allocateHostMemoryForTensor(
      parentNextAct, "nextAct", parentGraph, uploadProg, downloadProg, tmap);
  Tensor parentZDeltas, parentPrevDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass || doWuPass) {
    parentZDeltas = parentGraph.getNonReplicatedTensor(zDeltas);
    rawHostZDeltas = allocateHostMemoryForTensor(
        parentZDeltas, "zDeltas", parentGraph, uploadProg, downloadProg, tmap);
  }
  if (doBwdPass) {
    parentPrevDeltas = parentGraph.getNonReplicatedTensor(prevDeltas);
    rawHostPrevDeltas =
        allocateHostMemoryForTensor(parentPrevDeltas, "prevDeltas", parentGraph,
                                    uploadProg, downloadProg, tmap);
  }
  std::vector<Program> programs;
  const auto fwdProgIndex = programs.size(); // 0
  programs.push_back(std::move(fwdProg));
  const auto revProgIndex = programs.size(); // 1
  programs.push_back(std::move(revProg));
  const auto uploadProgIndex = programs.size(); // 2
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size(); // 3
  programs.push_back(std::move(downloadProg));

  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile") || jsonProfileOut) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (profileFormat) {
      engineOptions.set("profiler.format", *profileFormat);
    }
  }

  Engine engine(parentGraph, std::move(programs), engineOptions);

  if (vm.count("compile-only"))
    return 0;

  attachStreams(engine, tmap);
  boost::multi_array<double, 3> hostPrevAct(
      boost::extents[batchSize * replicationFactor][fwdInChans]
                    [product(inputFieldSize)]);
  boost::multi_array<double, 4> hostWeights(
      boost::extents[numConvGroups][fwdOutChansPerConvGroup]
                    [fwdInChansPerConvGroup][product(kernelSize)]);
  boost::multi_array<double, 1> hostBiases(boost::extents[fwdOutChans]);
  boost::multi_array<double, 3> hostNextAct(
      boost::extents[batchSize * replicationFactor][fwdOutChans]
                    [product(outFieldSize)]);
  std::mt19937 randomEngine;
  if (useUniformRandomData) {
    writeRandomValues(target, inputType, hostPrevAct, -2.0, 2.0, randomEngine);
    writeRandomValues(target, inputType, hostWeights, -1.0, +1.0, randomEngine);
  } else {
    writeRandomBinaryValues(target, inputType, hostPrevAct, -1.0, 1.0,
                            randomEngine);
    writeRandomBinaryValues(target, inputType, hostWeights, -1.0, 1.0,
                            randomEngine);
  }
  if (bias) {
    if (useUniformRandomData) {
      writeRandomValues(target, outputType, hostBiases, -2.0, +6.0,
                        randomEngine);
    } else {
      writeRandomBinaryValues(target, outputType, hostBiases, -1.0, 1.0,
                              randomEngine);
    }
  } else {
    std::fill(hostBiases.data(), hostBiases.data() + hostBiases.num_elements(),
              0.0);
  }
  copy(target, hostPrevAct, inputType, rawHostPrevAct.get());

  boost::multi_array<double, 5> duplicatedHostWeights(
      boost::extents[replicationFactor][numConvGroups][fwdOutChansPerConvGroup]
                    [fwdInChansPerConvGroup][product(kernelSize)]);
  boost::multi_array<double, 2> duplicatedHostBiases(
      boost::extents[replicationFactor][fwdOutChans]);

  // Used for determinism checking
  boost::multi_array<double, 5> prevExecutionWeights(
      boost::extents[replicationFactor][numConvGroups][fwdOutChansPerConvGroup]
                    [fwdInChansPerConvGroup][product(kernelSize)]);
  boost::multi_array<double, 2> prevExecutionBiases(
      boost::extents[replicationFactor][fwdOutChans]);

  boost::multi_array<double, 3> hostZDeltas(
      boost::extents[batchSize * replicationFactor]
                    [bwdParams.getNumInputChans()][product(outFieldSize)]);
  if (useUniformRandomData) {
    writeRandomValues(target, inputType, hostZDeltas, -3.0, 3.0, randomEngine);
  } else {
    writeRandomBinaryValues(target, inputType, hostZDeltas, -1.0, 1.0,
                            randomEngine);
  }

  const auto fwdModel = [&](const auto &hostPrevAct, const auto &hostWeights,
                            const auto &hostBiases) {
    boost::multi_array<double, 3> modelNextAct(
        boost::extents[batchSize * replicationFactor][fwdOutChans]
                      [product(outFieldSize)]);
    poplibs_test::conv::convolution(
        vectorConvert<unsigned>(inputFieldSize), truncationLower,
        truncationUpper, inDilation, paddingLower, paddingUpper, flipInput,
        vectorConvert<unsigned>(kernelSize), kernelTruncationLower,
        kernelTruncationUpper, kernelDilation, kernelPaddingLower,
        kernelPaddingUpper, flipKernel, outputTruncationLower,
        outputTruncationUpper, stride, outputPaddingLower, outputPaddingUpper,
        hostPrevAct, hostWeights, hostBiases, modelNextAct);
    return modelNextAct;
  };

  const auto bwdModel = [&](const auto &hostZDeltas, const auto &modelWeights) {
    boost::multi_array<double, 3> modelPrevDeltas(
        boost::extents[batchSize * replicationFactor][fwdInChans]
                      [product(inputFieldSize)]);
    poplibs_test::conv::convolutionBackward(
        vectorConvert<unsigned>(inputFieldSize), truncationLower,
        truncationUpper, inDilation, paddingLower, paddingUpper, flipInput,
        vectorConvert<unsigned>(kernelSize), kernelTruncationLower,
        kernelTruncationUpper, kernelDilation, kernelPaddingLower,
        kernelPaddingUpper, flipKernel, outputTruncationLower,
        outputTruncationUpper, stride, outputPaddingLower, outputPaddingUpper,
        hostZDeltas, modelWeights, modelPrevDeltas);
    return modelPrevDeltas;
  };

  const auto wuModel = [&](const auto &hostPrevAct, const auto &hostZDeltas,
                           const auto &hostWeights, const auto &hostBiases) {
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    poplibs_test::conv::weightUpdate(
        vectorConvert<unsigned>(inputFieldSize), truncationLower,
        truncationUpper, inDilation, paddingLower, paddingUpper, flipInput,
        vectorConvert<unsigned>(kernelSize), kernelTruncationLower,
        kernelTruncationUpper, kernelDilation, kernelPaddingLower,
        kernelPaddingUpper, flipKernel, outputTruncationLower,
        outputTruncationUpper, stride, outputPaddingLower, outputPaddingUpper,
        learningRate, hostPrevAct, hostZDeltas, modelWeights, modelBiases);
    return std::make_pair(modelWeights, modelBiases);
  };

  enum class DataValidation { AgainstModel, AgainstPreviousRuns };
  const auto validationMethod = [&]() -> boost::optional<DataValidation> {
    if (ignoreData) {
      return boost::none;
    } else {
      if (numDeterminismChecks > 0) {
        return DataValidation::AgainstPreviousRuns;
      } else {
        return DataValidation::AgainstModel;
      }
    }
  }();

  int rc = EXIT_SUCCESS;
  std::stringstream errs;
  for (unsigned determinismCheckIdx = 0;
       determinismCheckIdx < numDeterminismChecks + 1; ++determinismCheckIdx) {

    for (unsigned i = 0; i != replicationFactor; ++i) {
      duplicatedHostWeights[i] = hostWeights;
      if (bias) {
        duplicatedHostBiases[i] = hostBiases;
      }
    }

    copy(target, duplicatedHostWeights, inputType, rawHostWeights.get());
    if (bias) {
      copy(target, duplicatedHostBiases, outputType, rawHostBiases.get());
    }

    if (doBwdPass || doWuPass) {
      copy(target, hostZDeltas, inputType, rawHostZDeltas.get());
    }

    dev.bind([&](const Device &d) {
      engine.load(d);
      if (validationMethod) {
        engine.run(uploadProgIndex);
      }
      // Run the forward pass.
      engine.run(fwdProgIndex);
      if (doBwdPass || doWuPass) {
        // Run the backwards and/or weight update passes.
        engine.run(revProgIndex);
      }
      if (validationMethod) {
        engine.run(downloadProgIndex);
      }
    });

    bool fwdFailed = false;
    bool bwdFailed = false;
    bool weightsFailed = false;
    bool biasesFailed = false;

    if (doFwdPass) {
      copy(target, outputType, rawHostNextAct.get(), hostNextAct);
      if (validationMethod == DataValidation::AgainstModel) {
        fwdFailed = !checkIsClose(
            "fwd", hostNextAct, fwdModel(hostPrevAct, hostWeights, hostBiases),
            relativeTolerance, absoluteTolerance);
      }
    }

    if (doBwdPass || doWuPass) {
      boost::multi_array<double, 3> hostPrevDeltas(
          boost::extents[batchSize * replicationFactor]
                        [params.getNumInputChans()][product(inputFieldSize)]);

      if (doBwdPass) {
        copy(target, outputType, rawHostPrevDeltas.get(), hostPrevDeltas);
      }

      if (doBwdPass) {
        if (validationMethod == DataValidation::AgainstModel) {
          bwdFailed = !checkIsClose("bwd", hostPrevDeltas,
                                    bwdModel(hostZDeltas, hostWeights),
                                    relativeTolerance, absoluteTolerance);
        }
      }
      if (doWuPass) {
        copy(target, inputType, rawHostWeights.get(), duplicatedHostWeights);
        if (bias) {
          copy(target, outputType, rawHostBiases.get(), duplicatedHostBiases);
        }

        if (validationMethod == DataValidation::AgainstModel) {
          // Take dimensions and shape from host tensors.
          auto modelWeights = hostWeights;
          auto modelBiases = hostBiases;
          std::tie(modelWeights, modelBiases) =
              wuModel(hostPrevAct, hostZDeltas, hostWeights, hostBiases);

          for (unsigned i = 0; i != replicationFactor; ++i) {
            std::string suffix;
            if (replicationFactor > 1)
              suffix = "_ipu" + std::to_string(i);
            boost::multi_array<double, 4> hostWeights =
                duplicatedHostWeights[i];
            auto failed =
                !checkIsClose("weights" + suffix, hostWeights, modelWeights,
                              relativeTolerance, absoluteTolerance);
            if (failed) {
              weightsFailed = true;
            }

            if (bias) {
              boost::multi_array<double, 1> hostBiases =
                  duplicatedHostBiases[i];
              failed = !checkIsClose("biases" + suffix, hostBiases, modelBiases,
                                     relativeTolerance, absoluteTolerance);
              if (failed) {
                biasesFailed = true;
              }
            }
          }
        }

        if (validationMethod == DataValidation::AgainstPreviousRuns) {
          prevExecutionWeights = duplicatedHostWeights;
          if (bias) {
            prevExecutionBiases = duplicatedHostBiases;
          }

          if (determinismCheckIdx > 0) {
            if (duplicatedHostWeights != prevExecutionWeights) {
              weightsFailed = true;
            }
            if (bias) {
              if (duplicatedHostBiases != prevExecutionBiases) {
                biasesFailed = true;
              }
            }
          }
        }
      }
    }

    if (validationMethod) {
      const std::vector<std::pair<std::string, int>> results{
          {"fwd", fwdFailed},
          {"bwd", bwdFailed},
          {"weights", weightsFailed},
          {"biases", biasesFailed},
      };
      for (const auto &result : results) { // Report all failures
        if (result.second) {
          errs << result.first << " validation failed\n";
        }
      }
      for (const auto &result : results) { // Abort if any failed
        if (result.second) {
          rc = EXIT_FAILURE;
        }
      }
    }

  } // for num_determinism_checks

  if (jsonProfileOut) {
    const auto pr = engine.getProfile();
    std::ofstream os(*jsonProfileOut);
    poplar::serializeToJSON(os, pr);
  }

  if (vm.count("profile")) {
    auto reportOptions = OptionFlags{{"showExecutionSteps", "true"}};
    if (reportVarStorage) {
      reportOptions.set("showVarStorage", "true");
    }
    engine.printProfileSummary(std::cout, reportOptions);
  }

  std::cerr << errs.str();
  return rc;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
