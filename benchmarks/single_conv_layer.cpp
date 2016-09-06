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
#include <popnn_ref/Convolution.hpp>
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

static std::unique_ptr<char []>
addTensorUpload(Graph &graph, const Tensor &t, Sequence &prog) {
  std::unique_ptr<char []> p = allocateHostMemoryForTensor(graph, t);
  prog.add(Copy(t, p.get()));
  return p;
}

static std::unique_ptr<char []>
addTensorDownload(Graph &graph, const Tensor &t, Sequence &prog) {
  std::unique_ptr<char []> p = allocateHostMemoryForTensor(graph, t);
  prog.add(Copy(p.get(), t));
  return p;
}

template <class T>
static void
groupActivations(boost::const_multi_array_ref<double, 3> src,
                 boost::multi_array_ref<T, 4> dst) {
  unsigned channels = src.shape()[0];
  unsigned height = src.shape()[1];
  unsigned width = src.shape()[2];
  unsigned channelsPerGroup = dst.shape()[3];
  assert(dst.shape()[0] * channelsPerGroup == channels);
  for (unsigned c = 0; c != channels; ++c) {
    for (unsigned y = 0; y != height; ++y) {
      for (unsigned x = 0; x != width; ++x) {
        dst[c  /channelsPerGroup][y][x][c % channelsPerGroup] =
            src[c][y][x];
      }
    }
  }
}

static void
groupActivations(boost::const_multi_array_ref<double, 3> src,
                 const std::string &dstType,
                 const std::vector<std::size_t> &dstDims,
                 void *dst) {
  assert(dstDims.size() == 4);
  const auto &multiArrayDims =
      boost::extents[dstDims[0]][dstDims[1]][dstDims[2]][dstDims[3]];
  if (dstType == "float") {
    groupActivations(
      src,
      boost::multi_array_ref<float, 4>(reinterpret_cast<float*>(dst),
                                       multiArrayDims)
    );
  } else {
    assert(dstType == "half");
    groupActivations(
      src,
      boost::multi_array_ref<poplar::half, 4>(
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

template <class T>
static void
ungroupActivations(boost::const_multi_array_ref<T, 4> src,
                   boost::multi_array_ref<double, 3> dst) {
  unsigned channels = dst.shape()[0];
  unsigned height = dst.shape()[1];
  unsigned width = dst.shape()[2];
  unsigned channelsPerGroup = src.shape()[3];
  assert(src.shape()[0] * channelsPerGroup == channels);
  for (unsigned c = 0; c != channels; ++c) {
    for (unsigned y = 0; y != height; ++y) {
      for (unsigned x = 0; x != width; ++x) {
        dst[c][y][x] =
            src[c / channelsPerGroup][y][x][c % channelsPerGroup];
      }
    }
  }
}

static void
ungroupActivations(const std::string &srcType,
                   const std::vector<std::size_t> &srcDims,
                   const void *src,
                   boost::multi_array_ref<double, 3> dst) {
  assert(srcDims.size() == 4);
  const auto &multiArrayDims =
      boost::extents[srcDims[0]][srcDims[1]][srcDims[2]][srcDims[3]];
  if (srcType == "float") {
    ungroupActivations(
      boost::const_multi_array_ref<float, 4>(
        reinterpret_cast<const float*>(src),
        multiArrayDims
      ),
      dst
    );
  } else {
    assert(srcType == "half");
    ungroupActivations(
      boost::const_multi_array_ref<half, 4>(
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

template <std::size_t N>
static bool checkIsClose(const boost::multi_array<double, N> &actual,
                         const boost::multi_array<double, N> &expected,
                         double relativeTolerance) {
  if (actual.num_elements() != expected.num_elements()) {
    std::cerr << "mismatched number of elements: ";
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
      std::cerr << "mismatch on element " << n << ':';
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

  unsigned inputChannels;
  unsigned outputChannels;
  unsigned width;
  unsigned height;
  unsigned kernelSize;
  unsigned padding;
  unsigned stride;
  unsigned outChansPerGroup;
  NonLinearityType nonLinearityType;
  FPDataType dataType;
  double relativeTolerance;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("input-channels", po::value<unsigned>(&inputChannels)->required(),
     "Number of input channels")
    ("output-channels", po::value<unsigned>(&outputChannels)->required(),
     "Number of output channels")
    ("width", po::value<unsigned>(&width)->required(), "Field width")
    ("height", po::value<unsigned>(&height)->required(), "Field height")
    ("kernel-size", po::value<unsigned>(&kernelSize)->default_value(1),
     "Size of kernel")
    ("non-linearity",
     po::value<NonLinearityType>(&nonLinearityType)
         ->default_value(NON_LINEARITY_RELU),
     "Non-linearity type")
    ("data-type",
     po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
     "Type of the data and the parameters")
    ("padding", po::value<unsigned>(&padding)->default_value(0),
     "Amount of zero padding")
    ("stride", po::value<unsigned>(&stride)->default_value(1),
     "Kernel stride")
    ("out-chans-per-group",
     po::value<unsigned>(&outChansPerGroup),
     "The number of channels per group of the activations written in the "
     "forward pass")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
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
  GraphProgEnv env(popnn::findGraphProg(), GraphProgFileType::Object);
  DeviceInfo info;
  Graph graph(env, createIPUModelDevice(info));

  std::string dataTypeStr(asString(dataType));
  // TODO support backwards pass.
  bool forwardOnly = true;
  // TODO support residual connections.
  auto plan = conv::createPlan(height, width, inputChannels,
                               kernelSize, stride, padding,
                               outputChannels, dataTypeStr, graph, forwardOnly);
  auto inChansPerGroup = plan.fwdPartition.inChansPerGroup;
  if (!vm.count("output-chans-per-group")) {
    // If the output grouping is unspecified, assume the output uses the same
    // grouping as the input unless that is impossible.
    outChansPerGroup = (outputChannels % inChansPerGroup == 0) ?
                        inChansPerGroup :
                        plan.fwdPartition.partialChansPerGroup;
  }
  const auto outDims =
      conv::getOutputDim(height, width, kernelSize, stride, padding);
  const auto outHeight = outDims.first;
  const auto outWidth = outDims.second;
  // Create tensors.
  Tensor in = graph.addTensor(dataTypeStr,
                              {inputChannels / inChansPerGroup,
                               height,
                               width,
                               inChansPerGroup}, "in");
  mapActivations(graph, in);
  Tensor weights = conv::createWeights(graph, dataTypeStr, inputChannels,
                                       kernelSize, outputChannels, plan);
  Tensor biases = conv::createBiases(graph, dataTypeStr, outputChannels);
  Tensor out = graph.addTensor(dataTypeStr,
                               {outputChannels / outChansPerGroup, outHeight,
                                outWidth, outChansPerGroup}, "out");
  mapActivations(graph, out);
  auto upload = Sequence();
  auto download = Sequence();
  auto convProg =
    conv::convolution(graph, plan,
                      kernelSize, stride, padding, outputChannels,
                      nonLinearityType, in, weights, biases, out);
  auto rawHostIn = addTensorUpload(graph, in, upload);
  auto rawHostWeights = addTensorUpload(graph, weights, upload);
  auto rawhostBiases = addTensorUpload(graph, biases, upload);
  auto rawHostOut = addTensorDownload(graph, out, download);
  Engine engine(graph, {&upload, &convProg, &download});

  boost::multi_array<double, 3>
      hostIn(boost::extents[inputChannels][height][width]);
  boost::multi_array<double, 4>
      hostWeights(boost::extents[outputChannels][inputChannels][kernelSize]
                                 [kernelSize]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[outputChannels]);
  boost::multi_array<double, 3>
      hostOut(boost::extents[outputChannels][outHeight][outWidth]);
  std::mt19937 randomEngine;
  writeRandomValues(hostIn, 0.0, 1.0, randomEngine);
  writeRandomValues(hostWeights, 0.0, 1.0, randomEngine);
  writeRandomValues(hostBiases, 0.0, 1.0, randomEngine);
  groupActivations(hostIn, dataTypeStr, in.dims(), rawHostIn.get());
  groupWeights(hostWeights, dataTypeStr, weights.dims(), rawHostWeights.get());
  copy(hostBiases, dataTypeStr, rawhostBiases.get());
  engine.run(0); // Upload.
  engine.run(1); // Run.
  engine.run(2); // Download.
  ungroupActivations(dataTypeStr, out.dims(), rawHostOut.get(), hostOut);

  // Validate the results against a reference model.
  boost::multi_array<double, 3>
      modelOut(boost::extents[outputChannels][outHeight][outWidth]);
  ref::conv::convolution(stride, padding, nonLinearityType, hostIn,
                         hostWeights, hostBiases, modelOut);
  bool matchesModel = checkIsClose(hostOut, modelOut, relativeTolerance);

  Engine::ReportOptions opt;
  opt.doLayerWiseProfile = true;
  engine.report(std::cout, opt);
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
