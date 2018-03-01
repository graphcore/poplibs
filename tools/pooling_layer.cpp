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
      throw poputil::poplib_error(
        "Unknown pooling type<" + token + ">");
    return is;
  }
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned chans;
  unsigned width;
  unsigned height;
  unsigned kernelHeight;
  unsigned kernelWidth;
  int paddingHeightL;
  int paddingHeightU;
  int paddingWidthL;
  int paddingWidthU;
  unsigned strideHeight;
  unsigned strideWidth;
  unsigned fwdChansPerGroup;
  unsigned bwdChansPerGroup;
  unsigned batchSize;
  Type dataType;
  double relativeTolerance, absoluteTolerance;
  IPUModel ipuModel;
  ipuModel.IPUExchangeType =
      IPUModel::ExchangeType::AGGRESSIVE_MULTICAST;
  PoolingType poolingType = PoolingType::MAX;

  /* these are used when the same value is shared across both height and width*/
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned paddingHeight;
  unsigned paddingWidth;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("channels", po::value<unsigned>(&chans)->required(),
     "Number of channels")
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
     po::value<Type>(&dataType)->default_value(HALF),
     "Type of the data and the parameters")

    ("padding", po::value<unsigned>(&padding)->default_value(0),
     "Amount of zero padding for height and width. If set, it is an "
     "error to also set either padding-height and/or padding-width")
    ("padding-height", po::value<unsigned>(&paddingHeight)->default_value(0),
     "Amount of zero padding in the height dimension")
    ("padding-width", po::value<unsigned>(&paddingWidth)->default_value(0),
     "Amount of zero padding in the width dimension")
    ("padding-height-upper",
     po::value<int>(&paddingHeightU)->default_value(0),
     "Amount of zero padding in the height dimension")
    ("padding-height-lower",
     po::value<int>(&paddingHeightL)->default_value(0),
     "Amount of zero padding in the height dimension")
    ("padding-width-upper",
     po::value<int>(&paddingWidthU)->default_value(0),
     "Amount of zero padding in the width dimension")
    ("padding-width-lower",
     po::value<int>(&paddingWidthL)->default_value(0),
     "Amount of zero padding in the width dimension")

    ("stride", po::value<unsigned>(&stride)->default_value(1),
     "Kernel stride for both height and width. If set, it is an error "
     "to also set either stride-height and/or stride-width")
    ("stride-height", po::value<unsigned>(&strideHeight)->default_value(1),
     "Kernel stride in the height dimension")
    ("stride-width", po::value<unsigned>(&strideWidth)->default_value(1),
     "Kernel stride in the width dimension")

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
    if (!vm["padding-height-lower"].defaulted()) {
      std::cerr << "--padding as well as --padding-height-lower set\n";
      return 1;
    }
    if (!vm["padding-height-upper"].defaulted()) {
      std::cerr << "--padding as well as --padding-height-upper set\n";
      return 1;
    }
    if (!vm["padding-width"].defaulted()) {
      std::cerr << "--padding as well as --padding-width set\n";
      return 1;
    }
    if (!vm["padding-width-lower"].defaulted()) {
      std::cerr << "--padding as well as --padding-width-lower set\n";
      return 1;
    }
    if (!vm["padding-width-upper"].defaulted()) {
      std::cerr << "--padding as well as --padding-width-upper set\n";
      return 1;
    }
    paddingHeightL = padding;
    paddingWidthL = padding;
    paddingHeightU = padding;
    paddingWidthU = padding;
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
    paddingHeightL = paddingHeight;
    paddingHeightU = paddingHeight;
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
    paddingWidthL = paddingWidth;
    paddingWidthU = paddingWidth;
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
    strideHeight = stride;
    strideWidth = stride;
  }

  bool inferenceOnly = vm.count("inference-only");
  auto device = ipuModel.createDevice();
  const auto &target = device.getTarget();
  Graph graph(device);
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

  const auto outDims =
      popnn::pooling::getOutputFieldShape({height, width},
                                          {kernelHeight, kernelWidth},
                                          {strideHeight, strideWidth},
                                          {paddingHeightL, paddingWidthL},
                                          {paddingHeightU, paddingWidthU});
  const auto outHeight = outDims[0];
  const auto outWidth = outDims[1];
  // Create tensors.
  Tensor prevAct = graph.addVariable(dataType,
                                     {chans / fwdChansPerGroup,
                                     batchSize,
                                     height,
                                     width,
                                     fwdChansPerGroup}, "prevAct");
  mapTensorLinearly(graph, prevAct);
  prevAct = prevAct.dimShufflePartial({0, 4}, {1, 2})
                   .reshapePartial(1, 3, {chans});

  Tensor zDeltas;
  if (!inferenceOnly) {
    zDeltas =
        graph.addVariable(dataType, {chans / bwdChansPerGroup,
                                     batchSize,
                                     outHeight,
                                     outWidth,
                                     bwdChansPerGroup},
                          "zDeltas");
    mapTensorLinearly(graph, zDeltas);
    zDeltas = zDeltas.dimShufflePartial({0, 4}, {1, 2})
                     .reshapePartial(1, 3, {chans});
  }

  auto fwdProg = Sequence();
  auto nextAct = popnn::pooling::pool(graph,
                                      poolingType,
                                      {kernelHeight, kernelWidth},
                                      {strideHeight, strideWidth},
                                      {paddingHeightL, paddingWidthL},
                                      {paddingHeightU, paddingWidthU},
                                      prevAct, fwdProg);

  auto bwdProg = Sequence();
  Tensor prevDeltas;
  if (!inferenceOnly) {
    prevDeltas =
        popnn::pooling::poolInputGradient(graph,
                                          poolingType,
                                          {kernelHeight, kernelWidth},
                                          {strideHeight, strideWidth},
                                          {paddingHeightL, paddingWidthL},
                                          {paddingHeightU, paddingWidthU},
                                          prevAct, nextAct, zDeltas,
                                          bwdProg);
  }
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrevAct = allocateHostMemoryForTensor(prevAct, "prevAct",
                                                    graph, tmap);
  auto rawHostNextAct = allocateHostMemoryForTensor(nextAct, "nextAct",
                                                    graph, tmap);
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (!inferenceOnly) {
    rawHostZDeltas = allocateHostMemoryForTensor(zDeltas, "zDeltas",
                                                 graph, tmap);
    rawHostPrevDeltas = allocateHostMemoryForTensor(prevDeltas, "prevDeltas",
                                                    graph, tmap);
  }
  EngineOptions engOpt;
  engOpt.cpuMultiThreadExecution = false;
  Engine engine(device, graph, {std::move(fwdProg), std::move(bwdProg)},
                engOpt);

  boost::multi_array<double, 4>
      hostPrevAct(boost::extents[batchSize][chans][height][width]);
  boost::multi_array<double, 4>
      hostNextAct(boost::extents[batchSize][chans][outHeight][outWidth]);
  std::mt19937 randomEngine;
  writeRandomValues(target, dataType, hostPrevAct, -4.0, 4.0, randomEngine);
  copy<4>(target, hostPrevAct, dataType, rawHostPrevAct.get());
  // Run the forward pass.
  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap);

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
    upload(engine, tmap);
    engine.run(1); // Run.
    download(engine, tmap);
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
  Engine::ReportOptions opt;
  opt.doLayerWiseProfile = true;
  engine.reportStatic(std::cout, opt);
  engine.reportDynamic(std::cout, opt);
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
