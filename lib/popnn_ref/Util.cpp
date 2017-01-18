#include <popnn_ref/Util.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <popnn/Compiler.hpp>
#include <popnn/Net.hpp>

using namespace poplar;
using namespace poplar::program;

namespace ref {
namespace util {

std::unique_ptr<char []>
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

std::unique_ptr<char []>
allocateHostMemoryForTensor(Graph &graph, const Tensor &t,
                            Sequence &upload, Sequence &download) {
  std::unique_ptr<char []> p = allocateHostMemoryForTensor(graph, t);
  upload.add(Copy(t, p.get()));
  download.add(Copy(p.get(), t));
  return p;
}

void
writeRandomValues(double *begin, double *end, double min, double max,
                  std::mt19937 &randomEngine) {
  std::uniform_real_distribution<> dist(min, max);
  for (auto it = begin; it != end; ++it) {
    *it = dist(randomEngine);
  }
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
          dst[b][c / channelsPerGroup][y][x][c % channelsPerGroup] =
              src[b][c][y][x];
         }
      }
    }
  }
}

void
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

void
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

void
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

void
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

bool checkIsClose(double a, double b, double relativeTolerance) {
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

bool checkIsClose(const std::string &name, const double *actual,
                  const std::vector<std::size_t> &dims,
                  const double *expected, std::size_t N,
                  double relativeTolerance) {
  auto it = actual;
  auto end = it + N;
  bool isClose = true;
  for (; it != end; ++it, ++expected) {
    if (!checkIsClose(*it, *expected, relativeTolerance)) {
      const auto n = it - actual;
      std::cerr << "mismatch on element " << prettyCoord(name, n, dims) << ':';
      std::cerr << " expected=" << *expected;
      std::cerr << " actual=" << *it << '\n';
      isClose = false;
    }
  }
  return isClose;
}

const char *asString(const FPDataType &type) {
  switch (type) {
  case FPDataType::HALF: return "half";
  case FPDataType::FLOAT: return "float";
  }
  POPNN_UNREACHABLE();
}

std::ostream &operator<<(std::ostream &os, const FPDataType &type) {
  return os << asString(type);
}

std::istream &operator>>(std::istream &in, FPDataType &type) {
  std::string token;
  in >> token;
  if (token == "half")
    type = FPDataType::HALF;
  else if (token == "float")
    type = FPDataType::FLOAT;
  else
    throw popnn::popnn_error(
      "Invalid data-type <" + token + ">; must be half or float");
  return in;
}

} // end namespace util
} // end namespace ref
