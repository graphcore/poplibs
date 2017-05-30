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
  unsigned paddingHeightLower;
  unsigned paddingWidthLower;
  unsigned paddingHeightUpper;
  unsigned paddingWidthUpper;
  unsigned strideH;
  unsigned strideW;
  unsigned batchSize;
  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;
  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::AGGRESSIVE_MULTICAST;

  /* these are used when the same value is shared across both height and width*/
  unsigned kernelSize;
  unsigned padding;
  unsigned paddingHeight;
  unsigned paddingWidth;
  unsigned stride;
  Pass pass = Pass::ALL;
  popconv::ConvOptions convOptions;
  popconv::PlanningCache cache;
  convOptions.cache = &cache;

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
     "error to also set any other padding value")
    ("padding-height", po::value<unsigned>(&paddingHeight)->default_value(0),
     "Amount of zero padding in the height dimension, upper and lower")
    ("padding-width", po::value<unsigned>(&paddingWidth)->default_value(0),
     "Amount of zero padding in the width dimension, upper and lower")
    ("padding-height-lower",
     po::value<unsigned>(&paddingHeightLower)->default_value(0),
     "Amount of zero padding in the height dimension, lower edge")
    ("padding-width-lower",
     po::value<unsigned>(&paddingWidthLower)->default_value(0),
     "Amount of zero padding in the width dimension, lower edge")
    ("padding-height-upper",
     po::value<unsigned>(&paddingHeightUpper)->default_value(0),
     "Amount of zero padding in the height dimension, upper edge")
    ("padding-width-upper",
     po::value<unsigned>(&paddingWidthUpper)->default_value(0),
     "Amount of zero padding in the width dimension, upper edge")

    ("stride", po::value<unsigned>(&stride)->default_value(1),
     "Kernel stride for both height and width. If set, it is an error "
     "to also set either stride-height and/or stride-width")
    ("stride-height", po::value<unsigned>(&strideH)->default_value(1),
     "Kernel stride in the height dimension")
    ("stride-width", po::value<unsigned>(&strideW)->default_value(1),
     "Kernel stride in the width dimension")
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
    if (!vm["padding-height-lower"].defaulted()) {
      std::cerr << "--padding as well as --padding-height-lower set\n";
      return 1;
    }
    if (!vm["padding-width-lower"].defaulted()) {
      std::cerr << "--padding as well as --padding-width-lower set\n";
      return 1;
    }
    if (!vm["padding-height-upper"].defaulted()) {
      std::cerr << "--padding as well as --padding-height-upper set\n";
      return 1;
    }
    if (!vm["padding-width-upper"].defaulted()) {
      std::cerr << "--padding as well as --padding-width-upper set\n";
      return 1;
    }
    paddingHeightLower = padding;
    paddingHeightUpper = padding;
    paddingWidthLower = padding;
    paddingWidthUpper = padding;
  }

  if (!vm["padding-height"].defaulted()) {
    if (!vm["padding-height-lower"].defaulted()) {
      std::cerr << "--padding-height as well as --padding-height-lower set\n";
      return 1;
    }
    if (!vm["padding-height-upper"].defaulted()) {
      std::cerr << "--padding-height as well as --padding-height-upper set\n";
      return 1;
    }
    paddingHeightLower = paddingHeight;
    paddingHeightUpper = paddingHeight;
  }

  if (!vm["padding-width"].defaulted()) {
    if (!vm["padding-width-lower"].defaulted()) {
      std::cerr << "--padding-width as well as --padding-width-lower set\n";
      return 1;
    }
    if (!vm["padding-width-upper"].defaulted()) {
      std::cerr << "--padding-width as well as --padding-width-upper set\n";
      return 1;
    }
    paddingWidthLower = paddingWidth;
    paddingWidthUpper = paddingWidth;
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

  const auto params =
      popconv::ConvParams(dataTypeStr,
                          {batchSize, height, width, fwdInChans},
                          {kernelHeight, kernelWidth, fwdOutChans,
                           fwdInChans},
                          {strideH, strideW},
                          {paddingHeightLower, paddingWidthLower},
                          {paddingHeightUpper, paddingWidthUpper},
                          false);
  const auto outHeight = params.getOutputHeight();
  const auto outWidth = params.getOutputWidth();
  bool bwdIsFractional = strideH != 1 || strideW != 1;
  if (paddingHeightLower >= kernelHeight ||
      paddingHeightUpper >= kernelHeight ||
      paddingWidthLower >= kernelWidth ||
      paddingWidthUpper >= kernelWidth) {
    throw popstd::poplib_error("Backwards convolution pass does not support "
                             "padding that is greater than or equal to the "
                             "kernel size");
  }
  auto bwdPaddingHeightLower = paddingHeightLower;
  auto bwdPaddingWidthLower = paddingWidthLower;
  auto bwdPaddingHeightUpper = paddingHeightUpper;
  auto bwdPaddingWidthUpper = paddingWidthUpper;
  if (!bwdIsFractional) {
    bwdPaddingWidthLower = kernelWidth - 1 - paddingWidthLower;
    bwdPaddingHeightLower = kernelHeight - 1 - paddingHeightLower;
    bwdPaddingWidthUpper = kernelWidth - 1 - paddingWidthUpper;
    bwdPaddingHeightUpper = kernelHeight - 1 - paddingHeightUpper;
  }
  const auto bwdInChans = fwdOutChans;
  const auto bwdOutChans = fwdInChans;

  const auto bwdParams =
      popconv::ConvParams(dataTypeStr,
                          {batchSize, outHeight, outWidth, fwdOutChans},
                          {kernelHeight, kernelWidth, fwdInChans, fwdOutChans},
                          {strideH, strideW},
                          {bwdPaddingHeightLower, bwdPaddingWidthLower},
                          {bwdPaddingHeightUpper, bwdPaddingWidthUpper},
                          bwdIsFractional);
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
  Tensor biases = popconv::createBiases(graph, nextAct);
  popconv::addBias(graph, nextAct, biases, fwdProg, "");
  if (!doFwdPass)
    fwdProg = Sequence();

  auto revProg = Sequence();
  const auto learningRate = 0.5;

  if (doBwdPass) {
    auto zeros = graph.addConstantTensor(dataTypeStr, {fwdInChans}, 0);
    auto zeroBiases = graph.addTensor(dataTypeStr, {fwdInChans}, "zeroBiases");
    popstd::mapTensor(graph, zeroBiases);
    revProg.add(Copy(zeros, zeroBiases));
    prevDeltas = popconv::convolution(graph, zDeltas, weights, bwdParams,
                                      true, revProg, "",
                                      convOptions);
  }
  if (doWuPass) {
    popconv::convolutionWeightUpdate(graph, zDeltas, weights, prevAct,
                                     params, learningRate,
                                     revProg, "", convOptions);
    popconv::convolutionBiasUpdate(graph, zDeltas, biases, learningRate,
                                   revProg);
  }
  auto upload = Sequence();
  auto download = Sequence();
  auto rawHostPrevAct = allocateHostMemoryForTensor(prevAct, upload, download);
  auto rawHostWeights = allocateHostMemoryForTensor(weights, upload, download);
  auto rawHostBiases = allocateHostMemoryForTensor(biases, upload, download);
  auto rawHostNextAct = allocateHostMemoryForTensor(nextAct, upload, download);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass || doWuPass) {
    rawHostZDeltas = allocateHostMemoryForTensor(zDeltas, upload, download);
  }
  if (doBwdPass) {
    rawHostPrevDeltas = allocateHostMemoryForTensor(prevDeltas, upload,
                                                    download);
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
  writeRandomValues(hostPrevAct, -1.0, +5.0, randomEngine);
  writeRandomValues(hostWeights, -1.0, +7.0, randomEngine);
  writeRandomValues(hostBiases, -2.0, +6.0, randomEngine);
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
  poplib_test::conv::convolution({strideH, strideW},
                                 {paddingHeightLower, paddingWidthLower},
                                 {paddingHeightUpper, paddingWidthUpper},
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
    writeRandomValues(hostZDeltas, -3.0, 7.0, randomEngine);
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
      poplib_test::conv::convolutionBackward(
              {strideH, strideW},
              {paddingHeightLower, paddingWidthLower},
              {paddingHeightUpper, paddingWidthUpper},
              hostZDeltas,
              modelWeights,
              modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance);
    }
    if (doWuPass) {
      poplib_test::conv::weightUpdate({strideH, strideW},
                                      {paddingHeightLower, paddingWidthLower},
                                      {paddingHeightUpper, paddingWidthUpper},
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
