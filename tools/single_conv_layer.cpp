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
#include <popstd/TileMapping.hpp>
#include <popconv/Convolution.hpp>
#include <popconv/ConvUtil.hpp>
#include <popstd/exceptions.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popstd/Add.hpp>
#include <popreduce/codelets.hpp>
#include <popconv/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <poplib_test/Convolution.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <poplib_test/Pass.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>
#include "util/VectorUtils.hpp"
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace popstd;
using poplib_test::Pass;

static void addGlobalExchangeConstraints(IPUModel &ipuModel) {
  const auto numIPUs = ipuModel.numIPUs;
  if (numIPUs == 1)
    return;
  // The amount of data each IPU sends or receives cannot exceed the available
  // bandwidth. This constraint provides an optimistic lower bound on the
  // amount of time required. This is equivalent to assuming all-to-all
  // connectivity between IPUs limited only by the total off chip bandwidth.
  // TODO Derive a more precise set of constraints based on the network
  // topology.
  const auto linkBandwidth = 128.0 * 1024 * 1024 * 1024;
  const auto linkEfficiency = 0.85;
  const auto numLinks = 10;
  const auto ipuExternalBandwidth = linkBandwidth * linkEfficiency * numLinks;
  for (unsigned i = 0; i != numIPUs; ++i) {
    std::vector<GlobalExchangeFlow> inFlows;
    std::vector<GlobalExchangeFlow> outFlows;
    for (unsigned j = 0; j != numIPUs; ++j) {
      if (j == i)
        continue;
      inFlows.emplace_back(j, i);
      outFlows.emplace_back(i, j);
    }
    ipuModel.globalExchangeConstraints.emplace_back(ipuExternalBandwidth,
                                                    std::move(inFlows));
    ipuModel.globalExchangeConstraints.emplace_back(ipuExternalBandwidth,
                                                    std::move(outFlows));
  }
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  bool useCpuModel;
  unsigned fwdInChansPerConvGroup;
  unsigned fwdOutChansPerConvGroup;
  ShapeOption<std::size_t> inputFieldSizeOption;
  ShapeOption<std::size_t> kernelSizeOption;
  unsigned numConvGroups = 1;
  ShapeOption<int> paddingLowerOption, paddingUpperOption, paddingOption;
  ShapeOption<unsigned> inDilationOption;
  ShapeOption<bool> flipInputOption;
  ShapeOption<int> kernelPaddingLowerOption, kernelPaddingUpperOption,
                   kernelPaddingOption;
  ShapeOption<unsigned> kernelDilationOption;
  ShapeOption<bool> flipKernelOption;
  ShapeOption<unsigned> strideOption;
  unsigned batchSize;
  bool bias;
  Type dataType;
  Type partialsType;
  double absoluteTolerance, relativeTolerance;
  IPUModel ipuModel;
  ipuModel.IPUExchangeType =
      IPUModel::ExchangeType::AGGRESSIVE_MULTICAST;
  bool reportPlan;
  bool reportVarStorage;

  Pass pass = Pass::ALL;
  popconv::ConvOptions convOptions;
  popconv::PlanningCache cache;
  convOptions.cache = &cache;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("use-cpu", po::value<bool>(&useCpuModel)->default_value(false),
     "When true, use a CPU model of the device. Otherwise use the IPU model")
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
     po::value<Type>(&partialsType)->default_value(FLOAT),
     "Type of partials")
    ("padding", po::value<ShapeOption<int>>(&paddingOption)->default_value(0),
     "Amount of zero padding to add to the start and end of each dimension")
    ("padding-upper",
     po::value<ShapeOption<int>>(&paddingUpperOption)->default_value(0),
     "Amount of zero padding to add at the end of each dimension")
    ("padding-lower",
     po::value<ShapeOption<int>>(&paddingLowerOption)->default_value(0),
     "Amount of zero padding to add at the start of each dimension")
    ("in-dilation",
     po::value<ShapeOption<unsigned>>(&inDilationOption)->default_value(1),
     "Input dilation")
    ("flip-input",
     po::value<ShapeOption<bool>>(&flipInputOption)->default_value(false),
     "Whether to flip each input spatial field")
    ("kernel-padding",
     po::value<ShapeOption<int>>(&kernelPaddingOption)->default_value(0),
     "Amount of zero kernel padding to add at the start and end of each"
     "dimension")
    ("kernel-padding-upper",
     po::value<ShapeOption<int>>(&kernelPaddingUpperOption)->default_value(0),
     "Amount of zero kernel padding to add at the start of each dimension")
    ("kernel-padding-lower",
     po::value<ShapeOption<int>>(&kernelPaddingLowerOption)->default_value(0),
     "Amount of zero kernel padding to add at the end of each dimension")
    ("kernel-dilation",
     po::value<ShapeOption<unsigned>>(&kernelDilationOption)
         ->default_value(1),
     "Kernel dilation")
    ("flip-kernel",
     po::value<ShapeOption<bool>>(&flipKernelOption)->default_value(false),
     "Whether to flip each kernel spatial field")
    ("stride",
     po::value<ShapeOption<unsigned>>(&strideOption)->default_value(1),
     "Stride")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("absolute-tolerance",
     po::value<double>(&absoluteTolerance)->default_value(0.00001),
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
     po::value<bool>(&convOptions.useWinograd)->default_value(0),
     "Use winograd convolution")
    ("winograd-patch-size",
      po::value<unsigned>(&convOptions.winogradPatchSize)->default_value(4),
     "Square patch size to use in winograd convolution")
    ("percent-cyc-excess-for-mem-optim",
     po::value<unsigned>(
       &convOptions.percentageCyclesExcessForMemOptim
     )->default_value(0),
     "Percentage cycles excess to use for memory optimisation. "
     "if 0, no memory optimisation is performed")
    ("weight-update-method",
     po::value<popconv::WeightUpdateMethod>(
         &convOptions.weightUpdateMethod
     )->default_value(convOptions.weightUpdateMethod),
     "Weight update method: amp | auto")
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
    ("report-var-storage",
     po::value<bool>(&reportVarStorage)->default_value(false),
     "Report variable storage information")
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

  if (!vm["padding"].defaulted()) {
    const char *conflictingOptions[] = {
      "padding-lower",
      "padding-upper"
    };
    for (auto option : conflictingOptions) {
      if (!vm[option].defaulted()) {
        std::cerr << "--padding as well as --" << option << " set\n";
        return 1;
      }
    }
    paddingLowerOption = paddingOption;
    paddingUpperOption = paddingOption;
  }

  paddingLowerOption.broadcast(numFieldDims);
  auto &paddingLower = paddingLowerOption.val;
  paddingUpperOption.broadcast(numFieldDims);
  auto &paddingUpper = paddingUpperOption.val;

  inDilationOption.broadcast(numFieldDims);
  auto &inDilation = inDilationOption.val;
  flipInputOption.broadcast(numFieldDims);
  auto &flipInput = flipInputOption.val;

  if (!vm["kernel-padding"].defaulted()) {
    const char *conflictingOptions[] = {
      "kernel-padding-lower",
      "kernel-padding-upper",
    };
    for (auto option : conflictingOptions) {
      if (!vm[option].defaulted()) {
        std::cerr << "--kernel-padding as well as --" << option << " set\n";
        return 1;
      }
    }
    kernelPaddingLowerOption = kernelPaddingOption;
    kernelPaddingUpperOption = kernelPaddingOption;
  }

  kernelPaddingLowerOption.broadcast(numFieldDims);
  auto &kernelPaddingLower = kernelPaddingLowerOption.val;
  kernelPaddingUpperOption.broadcast(numFieldDims);
  auto &kernelPaddingUpper = kernelPaddingUpperOption.val;

  kernelDilationOption.broadcast(numFieldDims);
  auto &kernelDilation = kernelDilationOption.val;
  flipKernelOption.broadcast(numFieldDims);
  auto &flipKernel = flipKernelOption.val;
  strideOption.broadcast(numFieldDims);
  auto &stride = strideOption.val;
  const auto fwdInChans = fwdInChansPerConvGroup * numConvGroups;
  const auto fwdOutChans = fwdOutChansPerConvGroup * numConvGroups;

  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = pass == Pass::ALL || pass == Pass::BWD;
  bool doWuPass = pass == Pass::ALL || pass == Pass::WU;

  addGlobalExchangeConstraints(ipuModel);
  Device dev = useCpuModel ? Device::createCPUDevice() :
                             ipuModel.createDevice();
  Graph graph(dev);
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  popconv::addCodelets(graph);

  const auto params =
      popconv::ConvParams(dataType,
                          batchSize,
                          inputFieldSize,
                          kernelSize,
                          fwdInChansPerConvGroup,
                          fwdOutChansPerConvGroup,
                          stride,
                          paddingLower,
                          paddingUpper,
                          inDilation,
                          flipInput,
                          kernelPaddingLower,
                          kernelPaddingUpper,
                          kernelDilation,
                          flipKernel,
                          numConvGroups);


  const auto outFieldSize = params.getOutputFieldShape();
  const auto bwdParams = getGradientParams(params);

  // Create tensors.
  Tensor prevAct =
      popconv::createInput(graph, params, "prevAct", convOptions);
  Tensor weights =
      popconv::createWeights(graph, params, "weights", convOptions);

  Tensor prevDeltas, zDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas = popconv::createInput(graph, bwdParams, "zDeltas", convOptions);
  }

  auto fwdProg = Sequence();
  // Always generate the fwd program as it maps the weights and biases. Only
  // actually create the engined if the fwd pass is to be run
  Tensor nextAct = popconv::convolution(graph, prevAct, weights, params, false,
                                        fwdProg, "", convOptions);
  if (reportPlan) {
    std::cout << "Forward plan:\n";
    popconv::reportPlanInfo(std::cout, graph, params, convOptions);
  }
  Tensor biases;
  if (bias) {
    biases = popconv::createBiases(graph, nextAct);
    popconv::addBias(graph, nextAct, biases, fwdProg, "");
  }
  if (!doFwdPass)
    fwdProg = Sequence();

  auto revProg = Sequence();
  const auto learningRate = 0.5;

  if (doBwdPass) {
    prevDeltas = popconv::convolution(graph, zDeltas, weights, bwdParams,
                                      true, revProg, "",
                                      convOptions);
    if (reportPlan) {
      std::cout << "Backward plan:\n";
      popconv::reportPlanInfo(std::cout, graph, bwdParams, convOptions);
    }
  }
  if (doWuPass) {
    popconv::convolutionWeightUpdate(graph, zDeltas, weights, prevAct,
                                     params, learningRate,
                                     revProg, "", convOptions);
    if (bias) {
      popconv::convolutionBiasUpdate(graph, zDeltas, biases, learningRate,
                                     partialsType, revProg);
    }
    if (reportPlan) {
      std::cout << "WU plan:\n";
      popconv::reportWeightUpdatePlanInfo(std::cout, graph, zDeltas, prevAct,
                                          params, convOptions);
    }
  }
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct = allocateHostMemoryForTensor(prevAct, "prevAct", graph,
                                                    tmap);
  auto rawHostWeights = allocateHostMemoryForTensor(weights, "weights", graph,
                                                    tmap);
  std::unique_ptr<char []> rawHostBiases;
  if (bias) {
    rawHostBiases = allocateHostMemoryForTensor(biases, "biases", graph,
                                                tmap);
  }
  auto rawHostNextAct = allocateHostMemoryForTensor(nextAct, "nextAct", graph,
                                                    tmap);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass || doWuPass) {
    rawHostZDeltas = allocateHostMemoryForTensor(zDeltas, "zDeltas", graph,
                                                 tmap);
  }
  if (doBwdPass) {
    rawHostPrevDeltas = allocateHostMemoryForTensor(prevDeltas, "prevDeltas",
                                                    graph, tmap);
  }
  Engine engine(dev, graph, {std::move(fwdProg), std::move(revProg)});

  boost::multi_array<double, 3>
      hostPrevAct(boost::extents[batchSize][fwdInChans]
                                [product(inputFieldSize)]);
  boost::multi_array<double, 4>
      hostWeights(boost::extents[numConvGroups]
                                [fwdOutChansPerConvGroup]
                                [fwdInChansPerConvGroup]
                                [product(kernelSize)]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[fwdOutChans]);
  boost::multi_array<double, 3>
      hostNextAct(boost::extents[batchSize][fwdOutChans]
                                [product(outFieldSize)]);
  std::mt19937 randomEngine;
  writeRandomValues(hostPrevAct, -1.0, +5.0, randomEngine);
  writeRandomValues(hostWeights, -1.0, +7.0, randomEngine);
  if (bias) {
    writeRandomValues(hostBiases, -2.0, +6.0, randomEngine);
  } else {
    std::fill(hostBiases.data(), hostBiases.data() + hostBiases.num_elements(),
              0.0);
  }
  copy(hostPrevAct, dataType, rawHostPrevAct.get());
  copy(hostWeights, dataType, rawHostWeights.get());
  if (bias) {
    copy(hostBiases, dataType, rawHostBiases.get());
  }

  // Run the forward pass.
  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap);

  // Validate against a reference model.
  bool matchesModel = true;
  copy(dataType, rawHostNextAct.get(), hostNextAct);
  boost::multi_array<double, 3>
      modelNextAct(boost::extents[batchSize][fwdOutChans]
                                 [product(outFieldSize)]);
  poplib_test::conv::convolution(vectorConvert<unsigned>(inputFieldSize),
                                 inDilation,
                                 paddingLower,
                                 paddingUpper,
                                 flipInput,
                                 vectorConvert<unsigned>(kernelSize),
                                 kernelDilation,
                                 kernelPaddingLower,
                                 kernelPaddingUpper,
                                 flipKernel,
                                 stride,
                                 hostPrevAct,
                                 hostWeights, hostBiases, modelNextAct);
  if (doFwdPass) {
    matchesModel &= checkIsClose("fwd", hostNextAct, modelNextAct,
                                 relativeTolerance, absoluteTolerance);
  }

  if (doBwdPass || doWuPass) {
    boost::multi_array<double, 3> hostZDeltas(
      boost::extents[batchSize][bwdParams.getNumInputChans()]
                    [product(outFieldSize)]
    );
    boost::multi_array<double, 3> hostPrevDeltas(
      boost::extents[batchSize][params.getNumInputChans()]
                    [product(inputFieldSize)]
    );
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards and/or weight update passes.
    writeRandomValues(hostZDeltas, -3.0, 7.0, randomEngine);
    copy(hostZDeltas, dataType, rawHostZDeltas.get());
    upload(engine, tmap);
    engine.run(1); // Run.
    download(engine, tmap);

    copy(dataType, rawHostZDeltas.get(), hostZDeltas);
    if (doBwdPass) {
      copy(dataType, rawHostPrevDeltas.get(), hostPrevDeltas);
    }
    copy(dataType, rawHostWeights.get(), hostWeights);
    if (bias) {
      copy(dataType, rawHostBiases.get(), hostBiases);
    }

    // Validate against a reference model.
    if (doBwdPass) {
      boost::multi_array<double, 3>
          modelPrevDeltas(boost::extents[batchSize][fwdInChans]
                                        [product(inputFieldSize)]);
      poplib_test::conv::convolutionBackward(
              vectorConvert<unsigned>(inputFieldSize),
              inDilation,
              paddingLower,
              paddingUpper,
              flipInput,
              vectorConvert<unsigned>(kernelSize),
              kernelDilation,
              kernelPaddingLower,
              kernelPaddingUpper,
              flipKernel,
              stride,
              hostZDeltas,
              modelWeights,
              modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance, absoluteTolerance);
    }
    if (doWuPass) {
      poplib_test::conv::weightUpdate(vectorConvert<unsigned>(inputFieldSize),
                                      inDilation,
                                      paddingLower,
                                      paddingUpper,
                                      flipInput,
                                      vectorConvert<unsigned>(kernelSize),
                                      kernelDilation,
                                      kernelPaddingLower,
                                      kernelPaddingUpper,
                                      flipKernel,
                                      stride,
                                      learningRate, hostPrevAct,
                                      hostZDeltas, modelWeights, modelBiases);
      matchesModel &= checkIsClose("weights",
                                  hostWeights, modelWeights, relativeTolerance,
                                  absoluteTolerance);
      if (bias) {
        matchesModel &= checkIsClose("biases",
                                     hostBiases, modelBiases,
                                     relativeTolerance,
                                     absoluteTolerance);
      }
    }
  }

  if (!useCpuModel) {
    Engine::ReportOptions opt;
    opt.doLayerWiseProfile = true;
    if (reportVarStorage) {
      opt.showVariableStorage = true;
    }
    engine.report(std::cout, opt);
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
