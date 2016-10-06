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
#include <popnn/ActivationMapping.hpp>
#include <popnn/Convolution.hpp>
#include <popnn/ConvPlan.hpp>
#include <poplar/HalfFloat.hpp>
#include <popnn/Net.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn_ref/Convolution.hpp>
#include <popnn_ref/NonLinearity.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;

static std::unique_ptr<char []>
allocateHostMemoryForTensor(Graph &graph, const Tensor &t) {
  const auto dType = graph.getTensorElementType(t);
  std::unique_ptr<char []> p;
  if (dType == "float") {
    p.reset(new char[t.numElements() * sizeof(float)]);
  } else {
    assert(dType == "half");
    p.reset(new char[t.numElements() * sizeof(poplar::half)]);
  }
  return p;
}

static std::unique_ptr<char []>
allocateHostMemoryForTensor(Graph &graph, const Tensor &t,
                            Sequence &upload, Sequence &download) {
  std::unique_ptr<char []> p = allocateHostMemoryForTensor(graph, t);
  upload.add(Copy(t, p.get()));
  download.add(Copy(p.get(), t));
  return p;
}

void
writeRandomValues(double *begin, double *end, double mean,
                  double stdDev, std::mt19937 &randomEngine) {
  std::normal_distribution<> dist(mean, stdDev);
  for (auto it = begin; it != end; ++it) {
    *it = dist(randomEngine);
  }
}

template <class T, std::size_t N> void
writeRandomValues(boost::multi_array<T, N> &a, double mean,
                  double stdDev, std::mt19937 &randomEngine) {
  return writeRandomValues(a.data(), a.data() + a.num_elements(),
                           mean, stdDev, randomEngine);
}

template <class T>
static void
groupActivations(boost::const_multi_array_ref<double, 4> src,
                 boost::multi_array_ref<T, 5> dst) {
  unsigned batchSize = src.shape()[0];
  unsigned channels = src.shape()[1];
  unsigned height = src.shape()[2];
  unsigned width = src.shape()[3];
  unsigned channelsPerGroup = dst.shape()[4];
  assert(dst.shape()[0] == batchSize);
  assert(dst.shape()[1] * channelsPerGroup == channels);
  for (unsigned b = 0; b != batchSize; ++b) {
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != height; ++y) {
        for (unsigned x = 0; x != width; ++x) {
          dst[b][c  /channelsPerGroup][y][x][c % channelsPerGroup] =
              src[b][c][y][x];
         }
      }
    }
  }
}

static void
groupActivations(boost::const_multi_array_ref<double, 4> src,
                 const std::string &dstType,
                 const std::vector<std::size_t> &dstDims,
                 void *dst) {
  assert(dstDims.size() == 5);
  const auto &multiArrayDims =
    boost::extents[dstDims[0]][dstDims[1]][dstDims[2]][dstDims[3]][dstDims[4]];
  if (dstType == "float") {
    groupActivations(
      src,
      boost::multi_array_ref<float, 5>(reinterpret_cast<float*>(dst),
                                       multiArrayDims)
    );
  } else {
    assert(dstType == "half");
    groupActivations(
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
groupWeights(boost::const_multi_array_ref<double, 4> src,
             boost::multi_array_ref<T, 6> dst) {
  unsigned outputChannels = src.shape()[0];
  unsigned inputChannels = src.shape()[1];
  unsigned kernelHeight = src.shape()[2];
  unsigned kernelWidth = src.shape()[3];

  unsigned outputChansPerGroup = dst.shape()[4];
  unsigned inputChansPerGroup = dst.shape()[5];
  assert(dst.shape()[0] * outputChansPerGroup == outputChannels);
  assert(dst.shape()[1] * inputChansPerGroup == inputChannels);
  assert(dst.shape()[2] == kernelHeight);
  assert(dst.shape()[3] == kernelWidth);

  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    for (unsigned ic = 0; ic != inputChannels; ++ic) {
      for (unsigned y = 0; y != kernelHeight; ++y) {
        for (unsigned x = 0; x != kernelWidth; ++x) {
          dst[oc / outputChansPerGroup]
             [ic / inputChansPerGroup]
             [y]
             [x]
             [oc % outputChansPerGroup]
             [ic % inputChansPerGroup] =
              src[oc][ic][y][x];
        }
      }
    }
  }
}

static void
groupWeights(boost::const_multi_array_ref<double, 4> src,
             const std::string &dstType,
             const std::vector<std::size_t> &dstDims,
             void *dst) {
  assert(dstDims.size() == 6);
  const auto &multiArrayDims =
      boost::extents[dstDims[0]][dstDims[1]][dstDims[2]][dstDims[3]]
                    [dstDims[4]][dstDims[5]];
  if (dstType == "float") {
    groupWeights(
      src,
      boost::multi_array_ref<float, 6>(reinterpret_cast<float*>(dst),
                                       multiArrayDims)
    );
  } else {
    assert(dstType == "half");
    groupWeights(
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
ungroupWeights(boost::const_multi_array_ref<T, 6> src,
               boost::multi_array_ref<double, 4> dst) {
  unsigned outputChannels = dst.shape()[0];
  unsigned inputChannels = dst.shape()[1];
  unsigned kernelHeight = dst.shape()[2];
  unsigned kernelWidth = dst.shape()[3];

  unsigned outputChansPerGroup = src.shape()[4];
  unsigned inputChansPerGroup = src.shape()[5];
  assert(src.shape()[0] * outputChansPerGroup == outputChannels);
  assert(src.shape()[1] * inputChansPerGroup == inputChannels);
  assert(src.shape()[2] == kernelHeight);
  assert(src.shape()[3] == kernelWidth);

  for (unsigned oc = 0; oc != outputChannels; ++oc) {
    for (unsigned ic = 0; ic != inputChannels; ++ic) {
      for (unsigned y = 0; y != kernelHeight; ++y) {
        for (unsigned x = 0; x != kernelWidth; ++x) {
          dst[oc][ic][y][x] =
            src[oc / outputChansPerGroup]
               [ic / inputChansPerGroup]
               [y]
               [x]
               [oc % outputChansPerGroup]
               [ic % inputChansPerGroup];
        }
      }
    }
  }
}

static void
ungroupWeights(const std::string &srcType,
               const std::vector<std::size_t> &srcDims,
               const void *src,
               boost::multi_array_ref<double, 4> dst) {
  assert(srcDims.size() == 6);
  const auto &multiArrayDims =
      boost::extents[srcDims[0]][srcDims[1]][srcDims[2]][srcDims[3]]
                    [srcDims[4]][srcDims[5]];
  if (srcType == "float") {
    ungroupWeights(
      boost::const_multi_array_ref<float, 6>(
        reinterpret_cast<const float*>(src),
        multiArrayDims
      ),
      dst
    );
  } else {
    assert(srcType == "half");
    ungroupWeights(
      boost::const_multi_array_ref<half, 6>(
        reinterpret_cast<const half*>(src),
        multiArrayDims
      ),
      dst
    );
  }
}

static void
copy(boost::const_multi_array_ref<double, 1> src,
     const std::string &dstType,
     void *dst) {
  if (dstType == "float") {
    std::copy(src.begin(), src.end(), reinterpret_cast<float*>(dst));
  } else {
    assert(dstType == "half");
    std::copy(src.begin(), src.end(), reinterpret_cast<half*>(dst));
  }
}

static void
copy(const std::string &srcType,
     void *src,
     boost::multi_array_ref<double, 1> dst) {
  if (srcType == "float") {
    std::copy(reinterpret_cast<float*>(src),
              reinterpret_cast<float*>(src) + dst.size(),
              dst.begin());
  } else {
    assert(srcType == "half");
    std::copy(reinterpret_cast<half*>(src),
              reinterpret_cast<half*>(src) + dst.size(),
              dst.begin());
  }
}

template <class T>
static void
ungroupActivations(boost::const_multi_array_ref<T, 5> src,
                   boost::multi_array_ref<double, 4> dst) {
  unsigned batchSize = dst.shape()[0];
  unsigned channels = dst.shape()[1];
  unsigned height = dst.shape()[2];
  unsigned width = dst.shape()[3];
  unsigned channelsPerGroup = src.shape()[4];
  assert(src.shape()[0] == batchSize);
  assert(src.shape()[1] * channelsPerGroup == channels);
  for (unsigned b = 0; b < batchSize; ++b) {
    for (unsigned c = 0; c != channels; ++c) {
      for (unsigned y = 0; y != height; ++y) {
        for (unsigned x = 0; x != width; ++x) {
          dst[b][c][y][x] =
              src[b][c / channelsPerGroup][y][x][c % channelsPerGroup];
        }
      }
    }
  }
}

static void
ungroupActivations(const std::string &srcType,
                   const std::vector<std::size_t> &srcDims,
                   const void *src,
                   boost::multi_array_ref<double, 4> dst) {
  assert(srcDims.size() == 5);
  const auto &multiArrayDims =
    boost::extents[srcDims[0]][srcDims[1]][srcDims[2]][srcDims[3]][srcDims[4]];
  if (srcType == "float") {
    ungroupActivations(
      boost::const_multi_array_ref<float, 5>(
        reinterpret_cast<const float*>(src),
        multiArrayDims
      ),
      dst
    );
  } else {
    assert(srcType == "half");
    ungroupActivations(
      boost::const_multi_array_ref<half, 5>(
        reinterpret_cast<const half*>(src),
        multiArrayDims
      ),
      dst
    );
  }
}

static bool checkIsClose(double a, double b, double relativeTolerance) {
  return boost::math::fpc::close_at_tolerance<double>(relativeTolerance)(a, b);
}

std::string prettyCoord(const std::string &name, std::size_t index,
                        const std::vector<std::size_t> &dims) {
  std::string str = name + "[";
  auto N = std::accumulate(dims.begin(), dims.end(), std::size_t(1),
                           std::multiplies<size_t>());
  for (unsigned i = 0; i != dims.size(); ++i) {
    N = N / dims[i];
    if (i != 0)
        str = str += ",";
    str = str += std::to_string(index / N);
    index = index % N;
  }
  str += "]";
  return str;
}

template <std::size_t N>
static bool checkIsClose(const std::string &name,
                         const boost::multi_array<double, N> &actual,
                         const boost::multi_array<double, N> &expected,
                         double relativeTolerance) {
  std::vector<std::size_t> dims;
  for (unsigned i = 0; i != N; ++i)
    dims.push_back(actual.shape()[i]);
  if (actual.num_elements() != expected.num_elements()) {
    std::cerr << "mismatched number of elements [" + name + "]:";
    std::cerr << " expected=" << expected.num_elements();
    std::cerr << " actual=" << actual.num_elements() << '\n';
    return false;
  }
  auto it = actual.data();
  auto end = it + actual.num_elements();
  auto expectedIt = expected.data();
  bool isClose = true;
  for (; it != end; ++it, ++expectedIt) {
    if (!checkIsClose(*it, *expectedIt, relativeTolerance)) {
      const auto n = it - actual.data();
      std::cerr << "mismatch on element " << prettyCoord(name, n, dims) << ':';
      std::cerr << " expected=" << *expectedIt;
      std::cerr << " actual=" << *it << '\n';
      isClose = false;
    }
  }
  return isClose;
}

enum class FPDataType {
  HALF,
  FLOAT
};

const char *asString(const FPDataType &type) {
  switch (type) {
  case FPDataType::HALF: return "half";
  case FPDataType::FLOAT: return "float";
  }
}

inline std::ostream &operator<<(std::ostream &os, const FPDataType &type) {
  return os << asString(type);
}

inline std::istream &operator>>(std::istream &in, FPDataType &type) {
  std::string token;
  in >> token;
  if (token == "half")
    type = FPDataType::HALF;
  else if (token == "float")
    type = FPDataType::FLOAT;
  else
    throw std::runtime_error("Invalid data type name");
  return in;
}

const char *asString(const NonLinearityType &type) {
  switch (type) {
  case NON_LINEARITY_NONE: return "none";
  case NON_LINEARITY_RELU: return "relu";
  case NON_LINEARITY_SIGMOID: return "sigmoid";
  }
}

inline std::ostream &operator<<(std::ostream &os,
                                const NonLinearityType &type) {
  return os << asString(type);
}

inline std::istream &operator>>(std::istream &in, NonLinearityType &type) {
  std::string token;
  in >> token;
  if (token == "none")
    type = NON_LINEARITY_NONE;
  else if (token == "relu")
    type = NON_LINEARITY_RELU;
  else if (token == "sigmoid")
    type = NON_LINEARITY_SIGMOID;
  else
    throw std::runtime_error("Invalid non-linearity type");
  return in;
}

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
  NonLinearityType nonLinearityType;
  FPDataType dataType;
  double relativeTolerance;
  DeviceInfo info;
  bool useWinogradConv;
  unsigned winogradPatchSize;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("input-channels", po::value<unsigned>(&fwdInChans)->required(),
     "Number of input channels")
    ("output-channels", po::value<unsigned>(&fwdOutChans)->required(),
     "Number of output channels")
    ("width", po::value<unsigned>(&width)->required(), "Field width")
    ("height", po::value<unsigned>(&height)->required(), "Field height")
    ("kernel-height",
      po::value<unsigned>(&kernelHeight)->default_value(1),
     "Size of kernel height")
    ("kernel-width",
      po::value<unsigned>(&kernelWidth)->default_value(1),
     "Size of kernel width")
    ("non-linearity",
     po::value<NonLinearityType>(&nonLinearityType)
         ->default_value(NON_LINEARITY_RELU),
     "Non-linearity type")
    ("data-type",
     po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
     "Type of the data and the parameters")
    ("padding-height", po::value<unsigned>(&paddingHeight)->default_value(0),
     "Amount of zero padding in the height dimension")
    ("padding-width", po::value<unsigned>(&paddingWidth)->default_value(0),
     "Amount of zero padding in the width dimension")
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
    ("inference-only", "Benchmark inference only")
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("use-winograd-conv", po::value<bool>(&useWinogradConv)->default_value(0),
     "Use winograd convolution")
    ("winograd-patch-size",
      po::value<unsigned>(&winogradPatchSize)->default_value(4),
     "Square patch size to use in winograd convolution")
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
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  Graph graph(env, createIPUModelDevice(info));

  std::string dataTypeStr(asString(dataType));
  // TODO support residual connections.
  auto plan = conv::createPlan(height, width, fwdInChans,
                               kernelHeight, kernelWidth, strideH,
                               strideW, paddingHeight,
                               paddingWidth,
                               fwdOutChans, batchSize,
                               dataTypeStr, graph,
                               inferenceOnly);
  auto fwdInChansPerGroup = plan.fwdPartition.inChansPerGroup;
  // If the output grouping is unspecified, assume the output uses the same
  // grouping as the input unless that is impossible.
  if (!vm.count("forward-output-chans-per-group")) {
    fwdOutChansPerGroup = (fwdOutChans % fwdInChansPerGroup == 0) ?
                          fwdInChansPerGroup :
                          plan.fwdPartition.partialChansPerGroup;
  }
  const auto bwdInChans = fwdOutChans;
  const auto bwdOutChans = fwdInChans;
  auto bwdInChansPerGroup = plan.bwdPartition.inChansPerGroup;
  if (!inferenceOnly &&
      !vm.count("backward-output-chans-per-group")) {
    bwdOutChansPerGroup = (bwdOutChans % bwdInChansPerGroup == 0) ?
                          bwdInChansPerGroup :
                          plan.bwdPartition.partialChansPerGroup;
  }
  const auto outDims =
      conv::getOutputDim(height, width, kernelHeight, kernelWidth,
                         strideH, strideW,
                         paddingHeight, paddingWidth);
  const auto outHeight = outDims.first;
  const auto outWidth = outDims.second;
  // Create tensors.
  Tensor prevAct = graph.addTensor(dataTypeStr,
                                   {batchSize,
                                    fwdInChans / fwdInChansPerGroup,
                                    height,
                                    width,
                                    fwdInChansPerGroup}, "prevAct");
  mapActivations(graph, prevAct);
  Tensor weights = conv::createWeights(graph, dataTypeStr, fwdInChans,
                                       kernelHeight, kernelWidth,
                                       fwdOutChans, plan);
  Tensor biases = conv::createBiases(graph, dataTypeStr, fwdOutChans);
  Tensor nextAct =
      graph.addTensor(dataTypeStr, {batchSize,
                                    fwdOutChans / fwdOutChansPerGroup,
                                    outHeight,
                                    outWidth, fwdOutChansPerGroup},
                      "nextAct");
  mapActivations(graph, nextAct);
  Tensor prevDeltas, zDeltas, nextDeltas;
  if (!inferenceOnly) {
    nextDeltas =
        graph.addTensor(dataTypeStr, {batchSize,
                                      bwdInChans / bwdInChansPerGroup,
                                      outHeight,
                                      outWidth, bwdInChansPerGroup},
                        "nextDeltas");
    mapActivations(graph, nextDeltas);
    zDeltas =
        graph.addTensor(dataTypeStr, nextDeltas.dims(),
                        "zDeltas");
    mapActivations(graph, zDeltas);
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
  std::unique_ptr<char[]> rawHostNextDeltas;
  std::unique_ptr<char[]> rawHostZDeltas;
  std::unique_ptr<char[]> rawHostPrevDeltas;
  if (!inferenceOnly) {
    rawHostNextDeltas = allocateHostMemoryForTensor(graph, nextDeltas, upload,
                                                    download);
    rawHostZDeltas = allocateHostMemoryForTensor(graph, zDeltas, upload,
                                                 download);
    rawHostPrevDeltas = allocateHostMemoryForTensor(graph, prevDeltas, upload,
                                                    download);
  }

  auto fwdProg =
    conv::convolution(graph, plan,
                      kernelHeight, kernelWidth, strideH,
                      strideW, paddingHeight,
                      paddingWidth, fwdOutChans,
                      nonLinearityType, prevAct, weights, biases, nextAct,
                      RESIDUAL_NONE, {}, useWinogradConv, winogradPatchSize);

  auto bwdProg = Sequence();
  const auto learningRate = 0.5;
  if (!inferenceOnly) {
    bwdProg.add(
      bwdNonLinearity(graph, nextAct, nextDeltas, zDeltas, nonLinearityType)
    );
    bwdProg.add(
      conv::convolutionBackward(graph, plan, zDeltas, weights, prevDeltas,
                                kernelHeight, kernelWidth, strideH,
                                strideW,
                                paddingHeight, paddingWidth)
    );
    bwdProg.add(
      conv::convolutionWeightUpdate(graph, plan, zDeltas, weights, biases,
                                    prevAct, kernelHeight, kernelWidth,
                                    strideH, strideW, paddingHeight,
                                    paddingWidth, learningRate)
    );
  }
  Engine engine(graph, {&upload, &download, &fwdProg, &bwdProg});


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
  writeRandomValues(hostPrevAct, 0.0, 1.0, randomEngine);
  writeRandomValues(hostWeights, 0.0, 1.0, randomEngine);
  writeRandomValues(hostBiases, 0.0, 1.0, randomEngine);
  groupActivations(hostPrevAct, dataTypeStr, prevAct.dims(),
                   rawHostPrevAct.get());
  groupWeights(hostWeights, dataTypeStr, weights.dims(), rawHostWeights.get());
  copy(hostBiases, dataTypeStr, rawHostBiases.get());

  // Run the forward pass.
  engine.run(0); // Upload.
  engine.run(2); // Run.
  engine.run(1); // Download.

  // Validate against a reference model.
  ungroupActivations(dataTypeStr, nextAct.dims(), rawHostNextAct.get(),
                     hostNextAct);
  boost::multi_array<double, 4>
      modelNextAct(boost::extents[batchSize][fwdOutChans][outHeight][outWidth]);
  ref::conv::convolution(strideH, strideW, paddingHeight, paddingWidth,
                         nonLinearityType, hostPrevAct,
                         hostWeights, hostBiases, modelNextAct);
  bool matchesModel = checkIsClose("fwd", hostNextAct, modelNextAct,
                                   relativeTolerance);

  if (!inferenceOnly) {
    boost::multi_array<double, 4> hostNextDeltas(
      boost::extents[batchSize][bwdInChans][outHeight][outWidth]
    );
    boost::multi_array<double, 4> hostZDeltas(
      boost::extents[batchSize][bwdInChans][outHeight][outWidth]
    );
    boost::multi_array<double, 4> hostPrevDeltas(
      boost::extents[batchSize][bwdOutChans][height][width]
    );
    auto modelWeights = hostWeights;
    auto modelBiases = hostBiases;
    // Run the backwards pass.
    writeRandomValues(hostNextDeltas, 0.0, 1.0, randomEngine);
    groupActivations(hostNextDeltas, dataTypeStr, nextDeltas.dims(),
                     rawHostNextDeltas.get());
    engine.run(0); // Upload.
    engine.run(3); // Run.
    engine.run(1); // Download.
    ungroupActivations(dataTypeStr, zDeltas.dims(), rawHostZDeltas.get(),
                       hostZDeltas);
    ungroupActivations(dataTypeStr, prevDeltas.dims(), rawHostPrevDeltas.get(),
                       hostPrevDeltas);
    ungroupWeights(dataTypeStr, weights.dims(), rawHostWeights.get(),
                   hostWeights);
    copy(dataTypeStr, rawHostBiases.get(), hostBiases);

    // Validate against a reference model.
    auto modelZDeltas = hostNextDeltas;
    ref::bwdNonLinearity(nonLinearityType, hostNextAct, modelZDeltas);
    matchesModel &= checkIsClose("zdeltas",
                                 hostZDeltas, modelZDeltas, relativeTolerance);
    boost::multi_array<double, 4>
        modelPrevDeltas(boost::extents[batchSize][fwdInChans][height][width]);
    ref::conv::convolutionBackward(strideH, strideW, paddingHeight,
                                   paddingWidth, hostZDeltas, modelWeights,
                                   modelPrevDeltas);
    matchesModel &= checkIsClose("bwd", hostPrevDeltas, modelPrevDeltas,
                                 relativeTolerance);
    ref::conv::weightUpdate(strideH, strideW, paddingHeight, paddingWidth,
                            learningRate, hostPrevAct,
                            hostZDeltas, modelWeights, modelBiases);
    matchesModel &= checkIsClose("weights",
                                 hostWeights, modelWeights, relativeTolerance);
    matchesModel &= checkIsClose("biases",
                                 hostBiases, modelBiases, relativeTolerance);
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
