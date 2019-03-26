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
#include <poputil/TileMapping.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/ConvUtil.hpp>
#include <poputil/exceptions.hpp>
#include <popops/codelets.hpp>
#include <popops/Collectives.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <poplibs_test/Convolution.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/Util.hpp>
#include <poplibs_support/Compiler.hpp>
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/print.hpp"
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
using poplibs_test::Pass;

const OptionFlags defaultEngineOptions {
  {"target.workerStackSizeInBytes", "0x200"},
  {"target.supervisorStackSizeInBytes", "0x80"}
};

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
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
                        kernelTruncationUpperOption,
                        kernelTruncationOption;
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
  Type dataType;
  double absoluteTolerance, relativeTolerance;
  IPUModel ipuModel;
  bool reportPlan;
  bool reportVarStorage;
  unsigned startTileMultiplier;
  unsigned replicationFactor;

  Pass pass = Pass::ALL;
  Type partialsType = FLOAT;
  Type interTilePartialsType = FLOAT;
  Type interIpuPartialsType = FLOAT;
  std::string useWinograd = "false";
  std::string winogradPatchSize = "4";
  std::string tempMemoryBudget = "0";
  std::string use128BitConvUnitLoad = "false";
  std::string weightUpdateMethod = "AUTO";
  poplin::PlanningCache cache;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling report")
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
     po::value<Type>(&dataType)->default_value(HALF),
     "Type of the data and the parameters")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(partialsType),
     "Type of partials")
    ("tile-partials-type",
     po::value<Type>(&interTilePartialsType)
         ->default_value(interTilePartialsType),
     "Type of inter-tile partials")
    ("ipu-partials-type",
     po::value<Type>(&interIpuPartialsType)
         ->default_value(interIpuPartialsType),
     "Type of inter-IPU partials")
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
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("absolute-tolerance",
     po::value<double>(&absoluteTolerance),
     "Absolute tolerance to use when validating results against the reference "
     "model")
    ("ipus",
     po::value<unsigned>(&ipuModel.numIPUs)->default_value(ipuModel.numIPUs),
     "Number of IPUs")
    ("tiles-per-ipu",
     po::value<unsigned>(&ipuModel.tilesPerIPU)->
                           default_value(ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("workers-per-tile",
     po::value<unsigned>(&ipuModel.numWorkerContexts)->
                           default_value(ipuModel.numWorkerContexts),
     "Number of worker contexts per tile")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("conv-groups",
     po::value<unsigned>(&numConvGroups)->default_value(1),
     "Number of convolution groups in grouped convolution")
    ("use-winograd-conv",
     po::value<std::string>(&useWinograd)->default_value(useWinograd),
     "Use winograd convolution")
    ("winograd-patch-size",
     po::value<std::string>(&winogradPatchSize)
         ->default_value(winogradPatchSize),
     "Square patch size to use in winograd convolution")
      ("use-128bit-conv-load",
       po::value<std::string>(&use128BitConvUnitLoad)
          ->default_value(use128BitConvUnitLoad),
       "Use 128-bit loading of convolution unit")
    ("tempMemoryBudget",
     po::value<std::string>(&tempMemoryBudget)
         ->default_value(tempMemoryBudget),
     "Constrain the planner to limit the expected memory use. "
     "If 0, memory usage is unconstrained")
    ("weight-update-method",
     po::value<std::string>(&weightUpdateMethod)
         ->default_value(weightUpdateMethod),
     "Weight update method: amp | auto")
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
    ("report-var-storage",
     po::value<bool>(&reportVarStorage)->default_value(false),
     "Report variable storage information")
    ("start-tile-multiplier",
     po::value<unsigned>(&startTileMultiplier)->default_value(2),
     "Multiplier used to distribute convolutions across an IPU.")
    ("replication-factor",
     po::value<unsigned>(&replicationFactor)->default_value(1),
     "Number of parallel copies of the graph to run. Each copy of the graph "
     "shares the same parameters but reads different input samples. The "
     "effective batch size is the batch size of the graph multiplied by the "
     "replication factor")
  ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
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
    {outputPaddingLowerOption, outputPaddingUpperOption, "output-padding"}
  };
  for (const auto &entry : upperLowerOptionTriples) {
    if (!vm[entry.name].defaulted()) {
      std::string conflictingOptions[] = {
        entry.name + "-lower",
        entry.name + "-upper"
      };
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

  bool inferenceOnly = vm.count("inference-only");
  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::BWD);
  bool doWuPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::WU);

  if (vm["tolerance"].empty()) {
    if (dataType == FLOAT) {
      relativeTolerance = FLOAT_REL_TOL;
    } else {
      relativeTolerance = HALF_REL_TOL;
    }
  }
  if (vm["absolute-tolerance"].empty()) {
    if (dataType == FLOAT) {
      absoluteTolerance = FLOAT_ABS_TOL;
    } else {
      absoluteTolerance = HALF_ABS_TOL;
    }
  }

  auto dev = [&]() -> TestDevice {
    if (deviceType == DeviceType::IpuModel) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      return createTestDevice(deviceType,
                              ipuModel.numIPUs,
                              ipuModel.tilesPerIPU);
    }
  }();

  Graph parentGraph(dev.getTarget());
  popops::addCodelets(parentGraph);
  poplin::addCodelets(parentGraph);
  auto graph = parentGraph.createReplicatedGraph(replicationFactor);

  const auto params =
      poplin::ConvParams(dataType,
                          batchSize,
                          inputFieldSize,
                          kernelSize,
                          fwdInChansPerConvGroup,
                          fwdOutChansPerConvGroup,
                          numConvGroups,
                          truncationLower,
                          truncationUpper,
                          inDilation,
                          paddingLower,
                          paddingUpper,
                          flipInput,
                          kernelTruncationLower,
                          kernelTruncationUpper,
                          kernelDilation,
                          kernelPaddingLower,
                          kernelPaddingUpper,
                          flipKernel,
                          outputTruncationLower,
                          outputTruncationUpper,
                          stride,
                          outputPaddingLower,
                          outputPaddingUpper);


  const auto outFieldSize = params.getOutputFieldShape();
  const auto bwdParams = getGradientParams(params);
  OptionFlags convOptions{
    { "partialsType", partialsType.toString() },
    { "partialsType.interTile", interTilePartialsType.toString() },
    { "partialsType.interIPU", interIpuPartialsType.toString() },
    { "useWinograd", useWinograd },
    { "winogradPatchSize", winogradPatchSize },
    { "tempMemoryBudget", tempMemoryBudget },
    { "weightUpdateMethod", weightUpdateMethod },
    { "use128BitConvUnitLoad", use128BitConvUnitLoad },
    { "startTileMultiplier", std::to_string(startTileMultiplier) }
  };
  auto fwdOptions = convOptions;
  fwdOptions.set("pass", inferenceOnly ? "INFERENCE_FWD" :
                                         "TRAINING_FWD");
  auto bwdOptions = convOptions;
  bwdOptions.set("pass", "TRAINING_BWD");
  auto wuOptions = convOptions;
  wuOptions.set("pass", "TRAINING_WU");

  if (reportPlan) {
    std::cout
        << "Convolution parameters:\n"
           " Batch size: " << params.batchSize << "\n"
           " Kernel:" << params.kernelShape << "\n"
           " Stride:" << params.outputTransform.stride << "\n"
           " Padding Lower: " << params.inputTransform.paddingLower << "\n"
           " Padding Upper: " << params.inputTransform.paddingUpper << "\n"
           " Group size: " << params.numConvGroups << "\n"
           " Input: " << params.inputChannels << "x" <<
               params.inputFieldShape << "\n"
           " Output: " << params.outputChannels << "x" << outFieldSize << "\n";
  }

  // Create tensors.
  Tensor prevAct =
      poplin::createInput(graph, params, "prevAct", fwdOptions, &cache);
  Tensor weights =
      poplin::createWeights(graph, params, "weights", fwdOptions, &cache);

  Tensor prevDeltas, zDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas = poplin::createInput(graph, bwdParams, "zDeltas",
                                   bwdOptions, &cache);
  }

  auto fwdProg = Sequence();
  // Always generate the fwd program as it maps the weights and biases. Only
  // actually create the engined if the fwd pass is to be run
  Tensor nextAct = poplin::convolution(graph, prevAct, weights, params, false,
                                        fwdProg, "fwd/", fwdOptions, &cache);
  if (reportPlan) {
    std::cout << "Forward plan:\n";
    poplin::reportPlanInfo(std::cout, graph, params, fwdOptions, &cache);
  }
  Tensor biases;
  if (bias) {
    biases = poplin::createBiases(graph, nextAct);
    poplin::addBias(graph, nextAct, biases, fwdProg, "");
  }
  if (!doFwdPass)
    fwdProg = Sequence();

  auto revProg = Sequence();
  const auto learningRate = 0.05;

  if (doBwdPass) {
    prevDeltas = poplin::convolution(graph, zDeltas, weights, bwdParams,
                                      true, revProg, "bwd",
                                      bwdOptions, &cache);
    if (reportPlan) {
      std::cout << "Backward plan:\n";
      poplin::reportPlanInfo(std::cout, graph, bwdParams, bwdOptions, &cache);
    }
  }
  if (doWuPass) {
    if (replicationFactor == 1) {
      poplin::convolutionWeightUpdate(graph, zDeltas, weights, prevAct, params,
                                      learningRate, revProg, "wu", wuOptions,
                                      &cache);

    } else {
      auto weightDeltas =
          poplin::calculateWeightDeltas(graph, zDeltas, prevAct, params,
                                        revProg, "wu", wuOptions, &cache);
      auto weightDeltasReduced =
          popops::replicatedAllReduce(graph, parentGraph, weightDeltas,
                                      popops::Operation::ADD, revProg);
      popops::scaledAddTo(graph, weights, weightDeltasReduced, -learningRate,
                          revProg, "wu/UpdateWeights");
    }
    if (bias) {
      if (replicationFactor == 1) {
        poplin::convolutionBiasUpdate(graph, zDeltas, biases, learningRate,
                                       partialsType, revProg);
      } else {
        std::vector<std::size_t> reduceDims(zDeltas.rank() - 1);
        std::iota(std::next(reduceDims.begin()), reduceDims.end(), 2);
        auto biasDeltas = graph.clone(biases, "biasDeltas");
        popops::reduceWithOutput(graph, zDeltas, biasDeltas, reduceDims,
                                 popops::Operation::ADD, revProg,
                                 "wu/CalcBiasDeltas");
        auto biasDeltasReduced =
            popops::replicatedAllReduce(graph, parentGraph, biasDeltas,
                                        popops::Operation::ADD, revProg);
        popops::scaledAddTo(graph, biases, biasDeltasReduced, -learningRate,
                            revProg, "wu/UpdateBiases");
      }
    }
    if (reportPlan) {
      std::cout << "WU plan:\n";
      poplin::reportWeightUpdatePlanInfo(std::cout, graph, params,
                                          wuOptions, &cache);
    }
  }
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto parentPrevAct = parentGraph.getNonReplicatedTensor(prevAct);
  auto rawHostPrevAct = allocateHostMemoryForTensor(parentPrevAct, "prevAct",
                                                    parentGraph, uploadProg,
                                                    downloadProg, tmap);
  auto parentWeights = parentGraph.getNonReplicatedTensor(weights);
  auto rawHostWeights = allocateHostMemoryForTensor(parentWeights,
                                                    "weights", parentGraph,
                                                    uploadProg, downloadProg,
                                                    tmap);
  Tensor parentBiases;
  std::unique_ptr<char []> rawHostBiases;
  if (bias) {
    parentBiases = parentGraph.getNonReplicatedTensor(biases);
    rawHostBiases = allocateHostMemoryForTensor(parentBiases, "biases",
                                                parentGraph, uploadProg,
                                                downloadProg, tmap);
  }
  auto parentNextAct = parentGraph.getNonReplicatedTensor(nextAct);
  auto rawHostNextAct = allocateHostMemoryForTensor(parentNextAct, "nextAct",
                                                    parentGraph, uploadProg,
                                                    downloadProg, tmap);
  Tensor parentZDeltas, parentPrevDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass || doWuPass) {
    parentZDeltas = parentGraph.getNonReplicatedTensor(zDeltas);
    rawHostZDeltas = allocateHostMemoryForTensor(parentZDeltas, "zDeltas",
                                                 parentGraph, uploadProg,
                                                 downloadProg, tmap);
  }
  if (doBwdPass) {
    parentPrevDeltas = parentGraph.getNonReplicatedTensor(prevDeltas);
    rawHostPrevDeltas = allocateHostMemoryForTensor(parentPrevDeltas,
                                                    "prevDeltas", parentGraph,
                                                    uploadProg, downloadProg,
                                                    tmap);
  }
  std::vector<Program> programs;
  const auto fwdProgIndex = programs.size();
  programs.push_back(std::move(fwdProg));
  const auto revProgIndex = programs.size();
  programs.push_back(std::move(revProg));
  const auto uploadProgIndex = programs.size();
  programs.push_back(std::move(uploadProg));
  const auto downloadProgIndex = programs.size();
  programs.push_back(std::move(downloadProg));
  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.executionProfile", "compute_sets");
  }
  Engine engine(parentGraph, std::move(programs), engineOptions);
  attachStreams(engine, tmap);
  boost::multi_array<double, 3>
      hostPrevAct(boost::extents[batchSize * replicationFactor][fwdInChans]
                                [product(inputFieldSize)]);
  boost::multi_array<double, 4>
      hostWeights(boost::extents[numConvGroups]
                                [fwdOutChansPerConvGroup]
                                [fwdInChansPerConvGroup]
                                [product(kernelSize)]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[fwdOutChans]);
  boost::multi_array<double, 3>
      hostNextAct(boost::extents[batchSize * replicationFactor][fwdOutChans]
                                [product(outFieldSize)]);
  std::mt19937 randomEngine;
  auto target = parentGraph.getTarget();
  writeRandomValues(target, dataType, hostPrevAct, -1.0, +5.0, randomEngine);
  writeRandomValues(target, dataType, hostWeights, -1.0, +7.0, randomEngine);
  if (bias) {
    writeRandomValues(target, dataType, hostBiases, -2.0, +6.0, randomEngine);
  } else {
    std::fill(hostBiases.data(), hostBiases.data() + hostBiases.num_elements(),
              0.0);
  }
  copy(target, hostPrevAct, dataType, rawHostPrevAct.get());

  boost::multi_array<double, 5>
      duplicatedHostWeights(boost::extents[replicationFactor]
                                          [numConvGroups]
                                          [fwdOutChansPerConvGroup]
                                          [fwdInChansPerConvGroup]
                                          [product(kernelSize)]);
  boost::multi_array<double, 2>
      duplicatedHostBiases(boost::extents[replicationFactor][fwdOutChans]);
  for (unsigned i = 0; i != replicationFactor; ++i) {
    duplicatedHostWeights[i] = hostWeights;
    if (bias) {
      duplicatedHostBiases[i] = hostBiases;
    }
  }
  copy(target, duplicatedHostWeights, dataType,
       rawHostWeights.get());
  if (bias) {
    copy(target, duplicatedHostBiases, dataType,
         rawHostBiases.get());
  }

  // Run the forward pass.
  dev.bind([&](const Device &d) {
    engine.load(d);
    engine.run(uploadProgIndex);
    engine.run(fwdProgIndex); // Run.
    engine.run(downloadProgIndex);
  });

  // Validate against a reference model.
  bool matchesModel = true;
  boost::multi_array<double, 3>
      modelNextAct(boost::extents[batchSize * replicationFactor][fwdOutChans]
                                 [product(outFieldSize)]);
  poplibs_test::conv::convolution(vectorConvert<unsigned>(inputFieldSize),
                                 truncationLower,
                                 truncationUpper,
                                 inDilation,
                                 paddingLower,
                                 paddingUpper,
                                 flipInput,
                                 vectorConvert<unsigned>(kernelSize),
                                 kernelTruncationLower,
                                 kernelTruncationUpper,
                                 kernelDilation,
                                 kernelPaddingLower,
                                 kernelPaddingUpper,
                                 flipKernel,
                                 outputTruncationLower,
                                 outputTruncationUpper,
                                 stride,
                                 outputPaddingLower,
                                 outputPaddingUpper,
                                 hostPrevAct,
                                 hostWeights, hostBiases, modelNextAct);
  if (doFwdPass) {
    copy(target, dataType, rawHostNextAct.get(), hostNextAct);
    matchesModel &= checkIsClose("fwd", hostNextAct, modelNextAct,
                                 relativeTolerance, absoluteTolerance);
  }

  if (doBwdPass || doWuPass) {
    boost::multi_array<double, 3> hostZDeltas(
      boost::extents[batchSize * replicationFactor]
                    [bwdParams.getNumInputChans()][product(outFieldSize)]
    );
    boost::multi_array<double, 3> hostPrevDeltas(
      boost::extents[batchSize * replicationFactor]
                    [params.getNumInputChans()][product(inputFieldSize)]
    );
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards and/or weight update passes.
    writeRandomValues(target, dataType, hostZDeltas, -3.0, 7.0, randomEngine);
    copy(target, hostZDeltas, dataType, rawHostZDeltas.get());
    dev.bind([&](const Device &d) {
      engine.load(d);
      engine.run(uploadProgIndex);
      engine.run(revProgIndex);
      engine.run(downloadProgIndex);
    });

    copy(target, dataType, rawHostZDeltas.get(), hostZDeltas);
    if (doBwdPass) {
      copy(target, dataType, rawHostPrevDeltas.get(), hostPrevDeltas);
    }

    // Validate against a reference model.
    if (doBwdPass) {
      boost::multi_array<double, 3>
          modelPrevDeltas(boost::extents[batchSize * replicationFactor]
                                        [fwdInChans]
                                        [product(inputFieldSize)]);
      poplibs_test::conv::convolutionBackward(
              vectorConvert<unsigned>(inputFieldSize),
              truncationLower,
              truncationUpper,
              inDilation,
              paddingLower,
              paddingUpper,
              flipInput,
              vectorConvert<unsigned>(kernelSize),
              kernelTruncationLower,
              kernelTruncationUpper,
              kernelDilation,
              kernelPaddingLower,
              kernelPaddingUpper,
              flipKernel,
              outputTruncationLower,
              outputTruncationUpper,
              stride,
              outputPaddingLower,
              outputPaddingUpper,
              hostZDeltas,
              modelWeights,
              modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance, absoluteTolerance);
    }
    if (doWuPass) {
      poplibs_test::conv::weightUpdate(vectorConvert<unsigned>(inputFieldSize),
                                      truncationLower,
                                      truncationUpper,
                                      inDilation,
                                      paddingLower,
                                      paddingUpper,
                                      flipInput,
                                      vectorConvert<unsigned>(kernelSize),
                                      kernelTruncationLower,
                                      kernelTruncationUpper,
                                      kernelDilation,
                                      kernelPaddingLower,
                                      kernelPaddingUpper,
                                      flipKernel,
                                      outputTruncationLower,
                                      outputTruncationUpper,
                                      stride,
                                      outputPaddingLower,
                                      outputPaddingUpper,
                                      learningRate, hostPrevAct,
                                      hostZDeltas, modelWeights, modelBiases);
      copy(target, dataType, rawHostWeights.get(), duplicatedHostWeights);
      if (bias) {
        copy(target, dataType, rawHostBiases.get(), duplicatedHostBiases);
      }
      for (unsigned i = 0; i != replicationFactor; ++i) {
        std::string suffix;
        if (replicationFactor > 1)
          suffix = "_ipu" + std::to_string(i);
        hostWeights = duplicatedHostWeights[i];
        matchesModel &= checkIsClose("weights" + suffix, hostWeights,
                                     modelWeights, relativeTolerance,
                                     absoluteTolerance);
        if (bias) {
          hostBiases = duplicatedHostBiases[i];
          matchesModel &= checkIsClose("biases" + suffix,
                                       hostBiases, modelBiases,
                                       relativeTolerance,
                                       absoluteTolerance);
        }
      }
    }
  }

  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    // Rerun the program to get cycles excluding host copies.
    engine.resetExecutionProfile();

    dev.bind([&](const Device &d) {
      engine.load(d);
      if (doFwdPass)
        engine.run(fwdProgIndex);
      if (doBwdPass || doWuPass)
        engine.run(revProgIndex);
    });
    auto reportOptions = OptionFlags{
      { "showExecutionSteps", "true" }
    };
    if (reportVarStorage) {
      reportOptions.set("includeVarStorageReport", "true");
    }
    engine.printProfileSummary(std::cout, reportOptions);
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
