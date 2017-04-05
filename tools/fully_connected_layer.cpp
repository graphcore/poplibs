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
#include <popconv/codelets.hpp>
#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popreduce/Reduce.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <poplin/codelets.hpp>
#include <poplib_test/FullyConnected.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <poplib_test/Pass.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace poplin;
using namespace popstd;
using namespace popreduce;
using poplib_test::Pass;

template <class T>
static void
groupFullyConnectedPrevAct(boost::const_multi_array_ref<double, 2> src,
                           boost::multi_array_ref<T, 6> dst) {
  unsigned batchSize = src.shape()[0];
  unsigned inputSize = src.shape()[1];
  assert(dst.shape()[2] == 1);
  assert(dst.shape()[3] == 1);
  unsigned outChansPerGroup = dst.shape()[4];
  unsigned inChansPerGroup = dst.shape()[5];
  assert(dst.shape()[0] * outChansPerGroup == batchSize);
  assert(dst.shape()[1] * inChansPerGroup == inputSize);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned x = 0; x != inputSize; ++x) {
      dst[b / outChansPerGroup][x / inChansPerGroup][0][0]
         [b % outChansPerGroup][x % inChansPerGroup] = src[b][x];
    }
  }
}

static void
groupFullyConnectedPrevAct(boost::const_multi_array_ref<double, 2> src,
                           const std::string &dstType,
                           const std::vector<std::size_t> &dstDims,
                           void *dst) {
  assert(dstDims.size() == 6);
  const auto &multiArrayDims =
    boost::extents[dstDims[0]][dstDims[1]][dstDims[2]][dstDims[3]][dstDims[4]]
                  [dstDims[5]];
  if (dstType == "float") {
    groupFullyConnectedPrevAct(
      src,
      boost::multi_array_ref<float, 6>(reinterpret_cast<float*>(dst),
                                       multiArrayDims)
    );
  } else {
    assert(dstType == "half");
    groupFullyConnectedPrevAct(
      src,
      boost::multi_array_ref<poplar::half, 6>(
        reinterpret_cast<poplar::half*>(dst),
        multiArrayDims
      )
    );
  }
}

template <class T>
static void
groupFullyConnectedZDeltas(boost::const_multi_array_ref<double, 2> src,
                           boost::multi_array_ref<T, 5> dst) {
  unsigned batchSize = src.shape()[0];
  unsigned inputSize = src.shape()[1];
  assert(dst.shape()[0] == 1);
  assert(dst.shape()[2] == 1);
  unsigned chansPerGroup = dst.shape()[4];
  assert(dst.shape()[1] * chansPerGroup == batchSize);
  assert(dst.shape()[3] == inputSize);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned x = 0; x != inputSize; ++x) {
      dst[0][b / chansPerGroup][0][x][b % chansPerGroup] = src[b][x];
    }
  }
}

static void
groupFullyConnectedZDeltas(boost::const_multi_array_ref<double, 2> src,
                           const std::string &dstType,
                           const std::vector<std::size_t> &dstDims,
                           void *dst) {
  assert(dstDims.size() == 5);
  const auto &multiArrayDims =
    boost::extents[dstDims[0]][dstDims[1]][dstDims[2]][dstDims[3]][dstDims[4]];
  if (dstType == "float") {
    groupFullyConnectedZDeltas(
      src,
      boost::multi_array_ref<float, 5>(reinterpret_cast<float*>(dst),
                                       multiArrayDims)
    );
  } else {
    assert(dstType == "half");
    groupFullyConnectedZDeltas(
      src,
      boost::multi_array_ref<poplar::half, 5>(
        reinterpret_cast<poplar::half*>(dst),
        multiArrayDims
      )
    );
  }
}

template <class T>
static void
ungroupFullyConnectedPrevDeltas(boost::const_multi_array_ref<T, 6> src,
                                boost::multi_array_ref<double, 2> dst) {
  unsigned batchSize = dst.shape()[0];
  unsigned inputSize = dst.shape()[1];
  assert(src.shape()[2] == 1);
  assert(src.shape()[3] == 1);
  unsigned outChansPerGroup = src.shape()[4];
  unsigned inChansPerGroup = src.shape()[5];
  assert(src.shape()[0] * outChansPerGroup == batchSize);
  assert(src.shape()[1] * inChansPerGroup == inputSize);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned x = 0; x != inputSize; ++x) {
       dst[b][x] = src[b / outChansPerGroup][x / inChansPerGroup][0][0]
                      [b % outChansPerGroup][x % inChansPerGroup];
    }
  }
}

static void
ungroupFullyConnectedPrevDeltas(const std::string &srcType,
                                const std::vector<std::size_t> &srcDims,
                                const void *src,
                                boost::multi_array_ref<double, 2> dst) {
  assert(srcDims.size() == 6);
  const auto &multiArrayDims =
    boost::extents[srcDims[0]][srcDims[1]][srcDims[2]][srcDims[3]][srcDims[4]]
                  [srcDims[5]];
  if (srcType == "float") {
   ungroupFullyConnectedPrevDeltas(
      boost::const_multi_array_ref<float, 6>(
        reinterpret_cast<const float*>(src), multiArrayDims
      ),
      dst
    );
  } else {
    assert(srcType == "half");
    ungroupFullyConnectedPrevDeltas(
       boost::const_multi_array_ref<poplar::half, 6>(
         reinterpret_cast<const poplar::half*>(src),
         multiArrayDims
       ),
       dst
     );
  }
}

template <class T>
static void
groupFullyConnectedWeights(boost::const_multi_array_ref<double, 2> src,
                           boost::multi_array_ref<T, 5> dst) {
  unsigned outputSize = src.shape()[0];
  unsigned inputSize = src.shape()[1];
  assert(dst.shape()[0] == 1);
  assert(dst.shape()[2] == 1);
  assert(dst.shape()[3] == outputSize);
  unsigned chansPerGroup = dst.shape()[4];
  assert(chansPerGroup * dst.shape()[1] == inputSize);
  for (unsigned o = 0; o != outputSize; ++o) {
    for (unsigned i = 0; i != inputSize; ++i) {
      dst[0][i / chansPerGroup][0][o][i % chansPerGroup] = src[o][i];
    }
  }
}

static void
groupFullyConnectedWeights(boost::const_multi_array_ref<double, 2> src,
                           const std::string &dstType,
                           const std::vector<std::size_t> &dstDims,
                           void *dst) {
  assert(dstDims.size() == 5);
  const auto &multiArrayDims =
    boost::extents[dstDims[0]][dstDims[1]][dstDims[2]][dstDims[3]][dstDims[4]];
  if (dstType == "float") {
    groupFullyConnectedWeights(
      src,
      boost::multi_array_ref<float, 5>(reinterpret_cast<float*>(dst),
                                       multiArrayDims)
    );
  } else {
    assert(dstType == "half");
    groupFullyConnectedWeights(
      src,
      boost::multi_array_ref<poplar::half, 5>(
        reinterpret_cast<poplar::half*>(dst),
        multiArrayDims
      )
    );
  }
}

template <class T>
static void
ungroupFullyConnectedWeights(boost::const_multi_array_ref<T, 5> src,
                             boost::multi_array_ref<double, 2> dst) {
  unsigned outputSize = dst.shape()[0];
  unsigned inputSize = dst.shape()[1];
  assert(src.shape()[0] == 1);
  assert(src.shape()[2] == 1);
  assert(src.shape()[3] == outputSize);
  unsigned chansPerGroup = src.shape()[4];
  assert(chansPerGroup * src.shape()[1] == inputSize);
  for (unsigned o = 0; o != outputSize; ++o) {
    for (unsigned i = 0; i != inputSize; ++i) {
      dst[o][i] = src[0][i / chansPerGroup][0][o][i % chansPerGroup];
    }
  }
}

static void
ungroupFullyConnectedWeights(const std::string &srcType,
                             const std::vector<std::size_t> &srcDims,
                             void *src,
                             boost::multi_array_ref<double, 2> dst) {
  assert(srcDims.size() == 5);
  const auto &multiArrayDims =
    boost::extents[srcDims[0]][srcDims[1]][srcDims[2]][srcDims[3]][srcDims[4]];
  if (srcType == "float") {
    ungroupFullyConnectedWeights(
      boost::const_multi_array_ref<float, 5>(
        reinterpret_cast<const float*>(src), multiArrayDims
      ),
      dst
    );
  } else {
    assert(srcType == "half");
    ungroupFullyConnectedWeights(
      boost::const_multi_array_ref<poplar::half, 5>(
        reinterpret_cast<const poplar::half*>(src), multiArrayDims
      ),
      dst
    );
  }
}

template <class T>
static void
ungroupFullyConnectedOutput(boost::const_multi_array_ref<T, 5> src,
                            boost::multi_array_ref<double, 2> dst) {
  assert(src.shape()[0] == 1);
  assert(src.shape()[2] == 1);
  unsigned batchSize = dst.shape()[0];
  unsigned outputSize = dst.shape()[1];
  unsigned batchGroupSize = src.shape()[4];
  assert(src.shape()[1] * batchGroupSize == batchSize);
  assert(src.shape()[3] == outputSize);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned x = 0; x != outputSize; ++x) {
      dst[b][x] = src[0][b / batchGroupSize][0][x][b % batchGroupSize];
    }
  }
}

static void
ungroupFullyConnectedOutput(const std::string &srcType,
                            const std::vector<std::size_t> &srcDims,
                            const void *src,
                            boost::multi_array_ref<double, 2> dst) {
  assert(srcDims.size() == 5);
  const auto &multiArrayDims =
    boost::extents[srcDims[0]][srcDims[1]][srcDims[2]][srcDims[3]][srcDims[4]];
  if (srcType == "float") {
    ungroupFullyConnectedOutput(
      boost::const_multi_array_ref<float, 5>(
        reinterpret_cast<const float*>(src),
        multiArrayDims
      ),
      dst
    );
  } else {
    assert(srcType == "half");
    ungroupFullyConnectedOutput(
      boost::const_multi_array_ref<half, 5>(
        reinterpret_cast<const half*>(src),
        multiArrayDims
      ),
      dst
    );
  }
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned inputSize;
  unsigned outputSize;
  unsigned batchSize;
  bool inPlaceUpdate = true;
  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;
  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST;
  Pass pass = Pass::ALL;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of output channels")
    ("data-type",
     po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
     "Type of the data and the parameters")
    ("partials-type",
     po::value<FPDataType>(&partialsType)->default_value(FPDataType::FLOAT),
     "Type of the partials")
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&info.numIPUs)->default_value(info.numIPUs),
     "Number of IPUs")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("in-place-update",
     po::value<bool>(&inPlaceUpdate)->default_value(true),
     "Perform param update in place")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
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

  bool inferenceOnly = vm.count("inference-only");
  if (inferenceOnly && pass != Pass::ALL && pass != Pass::FWD) {
    std::cerr << "pass=" << pass << " specified with --inference-only\n";
    return 1;
  }

  bool doFwdPass = pass == Pass::ALL || pass == Pass::FWD;
  bool doBwdPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::BWD);
  bool doWuPass = !inferenceOnly && (pass == Pass::ALL || pass == Pass::WU);

  const auto learningRate = 0.5;
  Graph graph(createIPUModelDevice(info));
  popconv::addCodelets(graph);
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  poplin::addCodelets(graph);

  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));

  popconv::Planner planner;
  // A fully connected fwd pass is equivalent to a convolution with
  // input channels = inputSize
  // width = outputSize
  // height = 1
  // output channels = batchSize.
  popconv::PlanControl convPlanControl;
  auto fwdPlan = planner.createPlan(1 /*height*/, outputSize, inputSize,
                                    1 /*kernelHeight*/, 1 /*kernelWidth*/,
                                    1 /*strideH*/, 1 /*strideW*/,
                                    0 /*paddingHeight*/, 0 /*paddingWidth*/,
                                    batchSize, 1,
                                    dataTypeStr, partialsTypeStr,
                                    false, graph, convPlanControl);

  // Create tensors.
  Tensor weights = graph.addTensor(dataTypeStr,
                                   {1,
                                    inputSize / fwdPlan.inChansPerGroup,
                                    1 /*height*/,
                                    outputSize,
                                    fwdPlan.inChansPerGroup}, "weights");
  popconv::mapActivations(graph, fwdPlan, weights);
  Tensor prevAct = popconv::createWeights(graph, dataTypeStr, inputSize,
                                          1 /*kernelHeight*/, 1 /*kernelWidth*/,
                                          batchSize, fwdPlan);
  Tensor nextAct =
      graph.addTensor(dataTypeStr, {1 /*batchSize*/,
                                    batchSize / fwdPlan.partialChansPerGroup,
                                    1 /* outHeight */,
                                    outputSize, fwdPlan.partialChansPerGroup},
                      "nextAct");
  mapActivations(graph, nextAct);

  auto biases = graph.addTensor(dataTypeStr, {outputSize}, "biases");
  mapTensor(graph, biases);

  // A fully connected bwd pass is equivalent to a weight update pass for a
  // convolutional layer with
  // input channels = input size
  // width = outputSize
  // height = 1
  // output channels = batchSize.
  // Use the forward plan to avoid rearrangement of weight deltas.
  // TODO produce a joint plan for the forward and backward passes.
  auto bwdPlan = fwdPlan;
  bwdPlan.useConvolutionInstructions = false;
  auto upload = Sequence();
  auto download = Sequence();
  Tensor zDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  if (doBwdPass || doWuPass) {
    zDeltas = graph.addTensor(dataTypeStr,
                              {1 /*batchSize*/,
                               batchSize / bwdPlan.partialChansPerGroup,
                               1 /* outHeight */,
                               outputSize, bwdPlan.partialChansPerGroup},
                              "zDeltas");
    mapActivations(graph, zDeltas);
    rawHostZDeltas = allocateHostMemoryForTensor(graph, zDeltas, upload,
                                                 download);
  }

  auto rawHostPrevAct = allocateHostMemoryForTensor(graph, prevAct, upload,
                                                    download);
  auto rawHostWeights = allocateHostMemoryForTensor(graph, weights, upload,
                                                    download);
  auto rawHostBiases = allocateHostMemoryForTensor(graph, biases, upload,
                                                   download);
  auto rawHostNextAct = allocateHostMemoryForTensor(graph, nextAct, upload,
                                                    download);
  auto fwdProg = Sequence();
  auto bwdProg = Sequence();

  if (doFwdPass) {
    auto zeroBiases = graph.addConstantTensor(dataTypeStr, {batchSize}, 0);
    fwdProg.add(popconv::convolution(graph, fwdPlan, {1, 1}, {0, 0}, weights,
                                     prevAct, zeroBiases, nextAct,
                                     partialsTypeStr, false));
    auto bBiases = biases.broadcast(batchSize, 0)
                         .reshape({batchSize / fwdPlan.partialChansPerGroup,
                                   fwdPlan.partialChansPerGroup, outputSize})
                         .dimShuffle({0, 2, 1});
    addTo(graph, nextAct, bBiases, 1, fwdProg);
  } else {
    popconv::mapActivations(graph, fwdPlan, weights);
    popconv::mapWeights(prevAct, graph, fwdPlan, 1);
  }

  Tensor prevDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (doBwdPass) {
    prevDeltas = popconv::calculateWeightDeltas(graph, bwdPlan, fwdPlan,
                                                zDeltas, 1 /*kernelSizeY*/,
                                                1 /*kernelSizeX*/, weights,
                                                {1, 1}, {0, 0}, bwdProg);
  } else {
    prevDeltas = graph.addTensor(dataTypeStr,
                                 {batchSize / fwdPlan.partialChansPerGroup,
                                  inputSize / fwdPlan.inChansPerGroup,
                                  1,
                                  1,
                                  fwdPlan.partialChansPerGroup,
                                  fwdPlan.inChansPerGroup}, "prevDeltas");
    popconv::mapWeights(prevDeltas, graph, fwdPlan, 1);
  }
  rawHostPrevDeltas = allocateHostMemoryForTensor(graph, prevDeltas, upload,
                                                  download);
  if (doWuPass) {
    // Implement the weight update using matMul. This is inefficient as it
    // take advantage of batching and a costly rearrangement of the weight
    // deltas is required to match the layout of the weights.
    // TODO optimize this.
    auto zDeltasRearrangedView = zDeltas.dimShuffle({0, 1, 4, 2, 3})
                                        .reshape({batchSize, outputSize});
    auto zDeltasRearranged = graph.addTensor(dataTypeStr,
                                             zDeltasRearrangedView.shape());
    bwdProg.add(Copy(zDeltasRearrangedView, zDeltasRearranged));
    graph.setTileMapping(zDeltasRearranged,
                         graph.getTileMapping(zDeltasRearrangedView));
    auto prevActRearrangedView = prevAct.dimShuffle({0, 4, 1, 5, 2, 3})
                                        .reshape({batchSize, inputSize});
    auto prevActRearranged = graph.addTensor(dataTypeStr,
                                             prevActRearrangedView.shape());
    bwdProg.add(Copy(prevActRearrangedView, prevActRearranged));
    graph.setTileMapping(prevActRearranged,
                         graph.getTileMapping(prevActRearrangedView));
    MatMulOptions mmOpt;
    mmOpt.partialsType = partialsTypeStr;
    auto weightDeltas = matMul(graph, zDeltasRearranged.transpose(),
                               prevActRearranged, bwdProg, "fc", mmOpt);
    auto weightDeltasRearrangedView =
        weightDeltas.reshape({outputSize, inputSize / fwdPlan.inChansPerGroup,
                              fwdPlan.inChansPerGroup})
                    .dimShuffle({1, 0, 2})
                    .reshape(weights.shape());
    addTo(graph, weights, weightDeltasRearrangedView, -learningRate, bwdProg);
    auto biasDeltas = reduce(graph, zDeltasRearranged, bwdProg);
    addTo(graph, biases, biasDeltas, -learningRate, bwdProg);
  }

  Engine engine(graph, {std::move(upload), std::move(download),
                        std::move(fwdProg), std::move(bwdProg)});

  boost::multi_array<double, 2>
      hostPrevAct(boost::extents[batchSize][inputSize]);
  boost::multi_array<double, 2>
      hostWeights(boost::extents[outputSize][inputSize]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[outputSize]);
  boost::multi_array<double, 2>
      hostNextAct(boost::extents[batchSize][outputSize]);
  std::mt19937 randomEngine;
  writeRandomValues(hostPrevAct, -4.0, 4.0, randomEngine);
  writeRandomValues(hostWeights, -3.0, 3.0, randomEngine);
  writeRandomValues(hostBiases, -4.0, 4.0, randomEngine);
  groupFullyConnectedPrevAct(hostPrevAct, dataTypeStr, prevAct.shape(),
                             rawHostPrevAct.get());
  groupFullyConnectedWeights(hostWeights, dataTypeStr, weights.shape(),
                             rawHostWeights.get());
  copy(hostBiases, dataTypeStr, rawHostBiases.get());
  // Run the forward pass.
  engine.run(0); // Upload.
  engine.run(2); // Run.
  engine.run(1); // Download.
  ungroupFullyConnectedOutput(dataTypeStr, nextAct.shape(),
                              rawHostNextAct.get(), hostNextAct);

  // Validate against a reference model.
  bool matchesModel = true;
  if (doFwdPass) {
    boost::multi_array<double, 2>
        modelNextAct(boost::extents[batchSize][outputSize]);
    poplib_test::fc::fullyConnected(hostPrevAct, hostWeights, hostBiases,
                                    modelNextAct);
    matchesModel &= checkIsClose("fwd", hostNextAct, modelNextAct,
                                 relativeTolerance);

  }
  if (doBwdPass || doWuPass) {
    boost::multi_array<double, 2> hostZDeltas(
      boost::extents[batchSize][outputSize]
    );
    boost::multi_array<double, 2> hostPrevDeltas(
      boost::extents[batchSize][inputSize]
    );
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards pass.
    if (doBwdPass) {
      writeRandomValues(hostZDeltas, -5.0, 5.0, randomEngine);
      groupFullyConnectedZDeltas(hostZDeltas, dataTypeStr, zDeltas.shape(),
                                 rawHostZDeltas.get());
    } else {
      writeRandomValues(hostPrevDeltas, -5.0, 5.0, randomEngine);
      groupFullyConnectedPrevAct(hostPrevDeltas, dataTypeStr,
                                 prevDeltas.shape(),
                                 rawHostPrevDeltas.get());
    }
    engine.run(0); // Upload.
    engine.run(3); // Run.
    engine.run(1); // Download.

    // Validate against a reference model.
    if (doBwdPass) {
      ungroupFullyConnectedPrevDeltas(dataTypeStr, prevDeltas.shape(),
                                      rawHostPrevDeltas.get(), hostPrevDeltas);
      boost::multi_array<double, 2>
          modelPrevDeltas(boost::extents[batchSize][inputSize]);
      poplib_test::fc::fullyConnectedBackward(hostZDeltas, modelWeights,
                                              modelPrevDeltas);
      matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                   relativeTolerance);
    }
    if (doWuPass) {
      ungroupFullyConnectedWeights(dataTypeStr, weights.shape(),
                                   rawHostWeights.get(), hostWeights);
      copy(dataTypeStr, rawHostBiases.get(), hostBiases);
      poplib_test::fc::fullyConnectedWeightUpdate(learningRate, hostPrevAct,
                                                  hostZDeltas, modelWeights,
                                                  modelBiases);
      matchesModel &= checkIsClose("weights",
                                   hostWeights, modelWeights,
                                   relativeTolerance);
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
