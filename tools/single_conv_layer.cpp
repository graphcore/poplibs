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
#include <popstd/ActivationMapping.hpp>
#include <popconv/Convolution.hpp>
#include <popconv/ConvPlan.hpp>
#include <popstd/exceptions.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <popconv/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <poplib_test/Convolution.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <poplib_test/Pass.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace popstd;
using poplib_test::Pass;

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned fwdInChans;
  unsigned fwdOutChans;
  unsigned width;
  unsigned height;
  unsigned kernelHeight;
  unsigned kernelWidth;
  unsigned paddingHeight;
  unsigned paddingWidth;
  unsigned strideH;
  unsigned strideW;
  unsigned fwdOutChansPerGroup;
  unsigned bwdOutChansPerGroup;
  unsigned batchSize;
  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;
  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST;

  /* these are used when the same value is shared across both height and width*/
  unsigned kernelSize;
  unsigned padding;
  unsigned stride;
  unsigned percentageCyclesExcessForMemOptim;
  Pass pass = Pass::ALL;
  popconv::PlanControl convPlanControl;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("input-channels", po::value<unsigned>(&fwdInChans)->required(),
     "Number of input channels")
    ("output-channels", po::value<unsigned>(&fwdOutChans)->required(),
     "Number of output channels")
    ("width", po::value<unsigned>(&width)->required(), "Field width")
    ("height", po::value<unsigned>(&height)->required(), "Field height")

    ("kernel-size",
      po::value<unsigned>(&kernelSize)->default_value(1),
     "Size of square kernel. If set, it is an error to also set either "
     "kernel-height and/or kernel-width")
    ("kernel-height",
      po::value<unsigned>(&kernelHeight)->default_value(1),
     "Size of kernel height")
    ("kernel-width",
      po::value<unsigned>(&kernelWidth)->default_value(1),
     "Size of kernel width")
    ("data-type",
     po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
     "Type of the data and the parameters")
    ("partials-type",
     po::value<FPDataType>(&partialsType)->default_value(FPDataType::FLOAT),
     "Type of partials")
    ("padding", po::value<unsigned>(&padding)->default_value(0),
     "Amount of zero padding for height and width. If set, it is an "
     "error to also set either padding-height and/or padding-width")
    ("padding-height", po::value<unsigned>(&paddingHeight)->default_value(0),
     "Amount of zero padding in the height dimension")
    ("padding-width", po::value<unsigned>(&paddingWidth)->default_value(0),
     "Amount of zero padding in the width dimension")

    ("stride", po::value<unsigned>(&stride)->default_value(1),
     "Kernel stride for both height and width. If set, it is an error "
     "to also set either stride-height and/or stride-width")
    ("stride-height", po::value<unsigned>(&strideH)->default_value(1),
     "Kernel stride in the height dimension")
    ("stride-width", po::value<unsigned>(&strideW)->default_value(1),
     "Kernel stride in the width dimension")

    ("fwd-out-chans-per-group",
     po::value<unsigned>(&fwdOutChansPerGroup),
     "The number of channels per group of the activations written in the "
     "forward pass")
    ("bwd-out-chans-per-group",
     po::value<unsigned>(&bwdOutChansPerGroup),
     "The number of channels per group of the deltas written in the backwards "
     "pass")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("use-winograd-conv",
     po::value<bool>(&convPlanControl.useWinograd)->default_value(0),
     "Use winograd convolution")
    ("winograd-patch-size",
      po::value<unsigned>(&convPlanControl.winogradPatchSize)->default_value(4),
     "Square patch size to use in winograd convolution")
    ("percent-cyc-excess-for-mem-optim",
     po::value<unsigned>(
       &percentageCyclesExcessForMemOptim
     )->default_value(0),
     "Percentage cycles excess to use for memory optimisation. "
     "if 0, no memory optimisation is performed")
    ("weight-update-method",
     po::value<popconv::WeightUpdateMethod>(
         &convPlanControl.weightUpdateMethod
     )->default_value(convPlanControl.weightUpdateMethod),
     "Weight update method: amp | aop | auto")
  ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (!vm["kernel-size"].defaulted()) {
    if (!vm["kernel-height"].defaulted()) {
      std::cerr << "--kernel as well as --kernel-height set\n";
      return 1;
    }
    if (!vm["kernel-width"].defaulted()) {
      std::cerr << "--kernel as well as --kernel-width set\n";
      return 1;
    }
    kernelHeight = kernelSize;
    kernelWidth = kernelSize;
  }

  if (!vm["padding"].defaulted()) {
    if (!vm["padding-height"].defaulted()) {
      std::cerr << "--padding as well as --padding-height set\n";
      return 1;
    }
    if (!vm["padding-width"].defaulted()) {
      std::cerr << "--padding as well as --padding-width set\n";
      return 1;
    }
    paddingHeight = padding;
    paddingWidth = padding;
  }

  if (!vm["stride"].defaulted()) {
    if (!vm["stride-height"].defaulted()) {
      std::cerr << "--stride as well as --stride-height set\n";
      return 1;
    }
    if (!vm["stride-width"].defaulted()) {
      std::cerr << "--stride as well as --stride-width set\n";
      return 1;
    }
    strideH = stride;
    strideW = stride;
  }

  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = pass == Pass::ALL || pass == Pass::BWD;
  bool doWuPass = pass == Pass::ALL || pass == Pass::WU;

  Graph graph(createIPUModelDevice(info));
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  popconv::addCodelets(graph);

  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));

  const auto outDims =
      popconv::getOutputDim(height, width, kernelHeight, kernelWidth,
                            strideH, strideW,
                            paddingHeight, paddingWidth);
  const auto outHeight = outDims.first;
  const auto outWidth = outDims.second;
  // TODO support residual connections.
  popconv::Planner planner(percentageCyclesExcessForMemOptim);
  //Forward plan is always required as bwd/wu passes may use fwd groupings
  auto fwdPlan = planner.createPlan(height, width, fwdInChans,
                                    kernelHeight, kernelWidth,
                                    strideH, strideW,
                                    paddingHeight, paddingWidth,
                                    fwdOutChans, batchSize,
                                    dataTypeStr, partialsTypeStr,
                                    false, graph, convPlanControl);
  bool bwdIsFractional = strideH != 1 || strideW != 1;
  if (paddingHeight >= kernelHeight || paddingWidth >= kernelWidth) {
    throw popstd::poplib_error("Backwards convolution pass does not support "
                             "padding that is greater than or equal to the "
                             "kernel size");
  }
  auto bwdPaddingHeight = paddingHeight, bwdPaddingWidth = paddingWidth;
  if (!bwdIsFractional) {
    bwdPaddingWidth = kernelWidth - 1 - paddingWidth;
    bwdPaddingHeight = kernelHeight - 1 - paddingHeight;
  }
  popconv::Plan bwdPlan;
  // Backward plan is also needed for WU
  if (doBwdPass || doWuPass)
    bwdPlan = planner.createPlan(outHeight, outWidth, fwdOutChans,
                                 kernelHeight, kernelWidth,
                                 strideH, strideW,
                                 bwdPaddingHeight, bwdPaddingWidth,
                                 fwdInChans, batchSize,
                                 dataTypeStr, partialsTypeStr,
                                 bwdIsFractional, graph, convPlanControl);
  auto fwdInChansPerGroup = fwdPlan.inChansPerGroup;
  // If the output grouping is unspecified, assume the output uses the same
  // grouping as the input unless that is impossible.
  if (!vm.count("fwd-out-chans-per-group")) {
    fwdOutChansPerGroup = (fwdOutChans % fwdInChansPerGroup == 0) ?
                          fwdInChansPerGroup :
                          fwdPlan.partialChansPerGroup;
  }
  const auto bwdInChans = fwdOutChans;
  const auto bwdOutChans = fwdInChans;
  unsigned bwdInChansPerGroup;
  if (doBwdPass || doWuPass) {
    bwdInChansPerGroup = bwdPlan.inChansPerGroup;
    if (!vm.count("bwd-out-chans-per-group")) {
      bwdOutChansPerGroup = (bwdOutChans % bwdInChansPerGroup == 0) ?
                            bwdInChansPerGroup :
                            bwdPlan.partialChansPerGroup;
    }
  }
  popconv::Plan wuPlan;
  if (doWuPass)
    wuPlan = planner.createWeightUpdatePlan(height, width, fwdInChans,
                                            fwdInChansPerGroup,
                                            bwdInChansPerGroup,
                                            fwdPlan.partialChansPerGroup,
                                            kernelHeight, kernelWidth,
                                            strideH, strideW,
                                            paddingHeight, paddingWidth,
                                            fwdOutChans, batchSize,
                                            dataTypeStr, partialsTypeStr,
                                             false, graph, convPlanControl);

  // Create tensors.
  Tensor prevAct = graph.addTensor(dataTypeStr,
                                   {batchSize,
                                    fwdInChans / fwdInChansPerGroup,
                                    height,
                                    width,
                                    fwdInChansPerGroup}, "prevAct");
  mapActivations(graph, prevAct);
  Tensor weights = popconv::createWeights(graph, dataTypeStr, fwdInChans,
                                          kernelHeight, kernelWidth,
                                          fwdOutChans, fwdPlan);
  Tensor biases = popconv::createBiases(graph, dataTypeStr, fwdOutChans);
  Tensor nextAct =
      graph.addTensor(dataTypeStr, {batchSize,
                                    fwdOutChans / fwdOutChansPerGroup,
                                    outHeight,
                                    outWidth, fwdOutChansPerGroup},
                      "nextAct");
  mapActivations(graph, nextAct);
  Tensor prevDeltas, zDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas =
        graph.addTensor(dataTypeStr, {batchSize,
                                      bwdInChans / bwdInChansPerGroup,
                                      outHeight,
                                      outWidth, bwdInChansPerGroup},
                        "zDeltas");
    mapActivations(graph, zDeltas);
  }
  if (doBwdPass) {
    prevDeltas =
          graph.addTensor(dataTypeStr, {batchSize,
                                        bwdOutChans / bwdOutChansPerGroup,
                                        height,
                                        width, bwdOutChansPerGroup},
                          "prevDeltas");
      mapActivations(graph, prevDeltas);
  }


  auto upload = Sequence();
  auto download = Sequence();
  auto rawHostPrevAct = allocateHostMemoryForTensor(graph, prevAct, upload,
                                                    download);
  auto rawHostWeights = allocateHostMemoryForTensor(graph, weights, upload,
                                                    download);
  auto rawHostBiases = allocateHostMemoryForTensor(graph, biases, upload,
                                                   download);
  auto rawHostNextAct = allocateHostMemoryForTensor(graph, nextAct, upload,
                                                    download);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass || doWuPass) {
    rawHostZDeltas = allocateHostMemoryForTensor(graph, zDeltas, upload,
                                                 download);
  }
  if (doBwdPass) {
    rawHostPrevDeltas = allocateHostMemoryForTensor(graph, prevDeltas, upload,
                                                      download);
  }

  auto fwdProg = Sequence();
  // Always generate the fwd program as it maps the weights and biases. Only
  // actually create the engined if the fwd pass is to be run
  {
    Program program = popconv::convolution(graph, fwdPlan,
                                           strideH, strideW,
                                           paddingHeight, paddingWidth,
                                           prevAct, weights, biases, nextAct,
                                           partialsTypeStr, false, false);
    if (doFwdPass)
      fwdProg.add(program);
  }

  auto revProg = Sequence();
  const auto learningRate = 0.5;

  if (doBwdPass) {
    auto zeros = graph.addConstantTensor(dataTypeStr, {fwdInChans}, 0);
    auto zeroBiases = graph.addTensor(dataTypeStr, {fwdInChans}, "zeroBiases");
    popconv::mapBiases(zeroBiases, graph, prevDeltas);
    revProg.add(Copy(zeros, zeroBiases));
    revProg.add(
      popconv::convolution(graph, bwdPlan,
                           strideH, strideW,
                           bwdPaddingHeight, bwdPaddingWidth,
                           zDeltas, weights, zeroBiases, prevDeltas,
                           partialsTypeStr, bwdIsFractional, true)
    );
  }
  if (doWuPass) {
    revProg.add(
      popconv::convolutionWeightUpdate(graph, wuPlan, fwdPlan,
                                       zDeltas, weights, biases, prevAct,
                                       strideH, strideW, paddingHeight,
                                       paddingWidth, learningRate)
    );
  }
  Engine engine(graph, {std::move(upload), std::move(download),
                        std::move(fwdProg), std::move(revProg)});

  boost::multi_array<double, 4>
      hostPrevAct(boost::extents[batchSize][fwdInChans][height][width]);
  boost::multi_array<double, 4>
      hostWeights(boost::extents[fwdOutChans][fwdInChans][kernelHeight]
                                 [kernelWidth]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[fwdOutChans]);
  boost::multi_array<double, 4>
      hostNextAct(boost::extents[batchSize][fwdOutChans][outHeight][outWidth]);
  std::mt19937 randomEngine;
  writeRandomValues(hostPrevAct, -4.0, +4.0, randomEngine);
  writeRandomValues(hostWeights, -3.0, +3.0, randomEngine);
  writeRandomValues(hostBiases, -4.0, +4.0, randomEngine);
  groupActivations(hostPrevAct, dataTypeStr, prevAct.shape(),
                   rawHostPrevAct.get());
  groupWeights(hostWeights, dataTypeStr, weights.shape(), rawHostWeights.get());
  copy(hostBiases, dataTypeStr, rawHostBiases.get());

  // Run the forward pass.
  engine.run(0); // Upload.
  engine.run(2); // Run.
  engine.run(1); // Download.

  // Validate against a reference model.
  bool matchesModel = true;
  ungroupActivations(dataTypeStr, nextAct.shape(), rawHostNextAct.get(),
                     hostNextAct);
  boost::multi_array<double, 4>
      modelNextAct(boost::extents[batchSize][fwdOutChans][outHeight][outWidth]);
  poplib_test::conv::convolution(strideH, strideW, paddingHeight, paddingWidth,
                                 hostPrevAct,
                                 hostWeights, hostBiases, modelNextAct);
  if (doFwdPass) {
    matchesModel &= checkIsClose("fwd", hostNextAct, modelNextAct,
                                 relativeTolerance);
  }

  if (doBwdPass || doWuPass) {
    boost::multi_array<double, 4> hostZDeltas(
      boost::extents[batchSize][bwdInChans][outHeight][outWidth]
    );
    boost::multi_array<double, 4> hostPrevDeltas(
      boost::extents[batchSize][bwdOutChans][height][width]
    );
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards and/or weight update passes.
    writeRandomValues(hostZDeltas, -5.0, 5.0, randomEngine);
    groupActivations(hostZDeltas, dataTypeStr, zDeltas.shape(),
                     rawHostZDeltas.get());
    engine.run(0); // Upload.
    engine.run(3); // Run.
    engine.run(1); // Download.

    ungroupActivations(dataTypeStr, zDeltas.shape(), rawHostZDeltas.get(),
                       hostZDeltas);
    if (doBwdPass) {
      ungroupActivations(dataTypeStr, prevDeltas.shape(),
                         rawHostPrevDeltas.get(), hostPrevDeltas);
    }
    ungroupWeights(dataTypeStr, weights.shape(), rawHostWeights.get(),
                   hostWeights);
    copy(dataTypeStr, rawHostBiases.get(), hostBiases);

    // Validate against a reference model.
    if (doBwdPass) {
      boost::multi_array<double, 4>
          modelPrevDeltas(boost::extents[batchSize][fwdInChans][height][width]);
      poplib_test::conv::convolutionBackward(strideH, strideW, paddingHeight,
                                             paddingWidth, hostZDeltas,
                                             modelWeights,
                                             modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance);
    }
    if (doWuPass) {
      poplib_test::conv::weightUpdate(strideH, strideW, paddingHeight,
                                      paddingWidth,
                                      learningRate, hostPrevAct,
                                      hostZDeltas, modelWeights, modelBiases);
      matchesModel &= checkIsClose("weights",
                                  hostWeights, modelWeights, relativeTolerance);
      matchesModel &= checkIsClose("biases",
                                   hostBiases, modelBiases, relativeTolerance);
    }
  }

  Engine::ReportOptions opt;
  opt.doLayerWiseProfile = true;
  engine.report(std::cout, opt);
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
