#include <poplibs_test/Util.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poputil/exceptions.hpp>
#include <cmath>
#include <boost/random.hpp>

using namespace poplar;
using namespace poplar::program;

namespace poplibs_test {
namespace util {

std::unique_ptr<char []>
allocateHostMemoryForTensor(const Target &target, const Tensor &t) {
  const auto dType = t.elementType();
  std::unique_ptr<char []> p;
  if (dType == FLOAT) {
    p.reset(new char[t.numElements() * sizeof(float)]);
    std::fill(&p[0], &p[t.numElements() * sizeof(float)], 0);
  } else if (dType == HALF){
    p.reset(new char[t.numElements() * target.getTypeSize(HALF)]);
    std::fill(&p[0], &p[t.numElements() * target.getTypeSize(HALF)], 0);
  } else if (dType == UNSIGNED_INT) {
    p.reset(new char[t.numElements() * sizeof(unsigned int)]);
    std::fill(&p[0], &p[t.numElements() * sizeof(unsigned int)], 0);
  } else if (dType == INT) {
    p.reset(new char[t.numElements() * sizeof(int)]);
    std::fill(&p[0], &p[t.numElements() * sizeof(int)], 0);
  } else {
    assert(dType == BOOL);
    p.reset(new char[t.numElements() * sizeof(bool)]);
    std::fill(&p[0], &p[t.numElements() * sizeof(bool)], 0);
  }
  return p;
}

std::unique_ptr<char []>
allocateHostMemoryForTensor(const Tensor &t,  const std::string &name,
                            Graph &graph, Sequence &uploadProg,
                            Sequence &downloadProg,
                            std::vector<std::pair<std::string, char *>> &map) {
  std::unique_ptr<char []> p = allocateHostMemoryForTensor(graph.getTarget(),
                                                           t);
  auto downloadId =
      graph.addDeviceToHostFIFO(name + "_download", t.elementType(),
                                t.numElements());
  downloadProg.add(Copy(t, downloadId));
  auto uploadId =
      graph.addHostToDeviceFIFO(name + "_upload", t.elementType(),
                                t.numElements());
  uploadProg.add(Copy(uploadId, t));
  map.emplace_back(name, p.get());
  return p;
}

void attachStreams(Engine &e,
                   const std::vector<std::pair<std::string, char *>> &map) {
  for (const auto &p : map) {
    e.connectStream(p.first + "_upload", p.second);
    e.connectStream(p.first + "_download", p.second);
  }
}

void
writeRandomValues(const Target &target,
                  const Type &type,
                  double *begin, double *end, double min, double max,
                  std::mt19937 &randomEngine) {
  boost::random::uniform_real_distribution<> dist(min, max);
  for (auto it = begin; it != end; ++it) {
    *it = dist(randomEngine);
  }
  // Round floating point values to nearest representable value on device.
  if (type == poplar::FLOAT) {
    for (auto it = begin; it != end; ++it) {
      *it = static_cast<float>(*it);
    }
  } else if (type == poplar::HALF) {
    auto N = end - begin;
    std::vector<char> buf(N * target.getTypeSize(type));
    poplar::copyDoubleToDeviceHalf(target, begin, buf.data(), N);
    poplar::copyDeviceHalfToDouble(target, buf.data(), begin, N);
  }
}

template <typename FPType>
bool checkIsClose(FPType a, FPType b, double relativeTolerance) {
  // These checks are necessary because close_at_tolerance doesn't handle
  // NaN, infinity or comparing against zero.
  if (a == b) {
    return true;
  }
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  boost::math::fpc::small_with_tolerance<FPType> nearZero(relativeTolerance);
  if (nearZero(a) && nearZero(b)) {
    return true;
  }
  if (!boost::math::fpc::close_at_tolerance<FPType>(relativeTolerance)(a, b)) {
    return false;
  }

  return true;
}

template bool checkIsClose<bool>(bool, bool, double);
template bool checkIsClose<int>(int, int, double);
template bool checkIsClose<float>(float, float, double);
template bool checkIsClose<double>(double, double, double);

std::string prettyCoord(const std::string &name, std::size_t index,
                        const std::vector<std::size_t> &shape) {
  std::string str = name + "[";
  auto N = std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                           std::multiplies<size_t>());
  for (unsigned i = 0; i != shape.size(); ++i) {
    N = N / shape[i];
    if (i != 0)
        str = str += ",";
    str = str += std::to_string(index / N);
    index = index % N;
  }
  str += "]";
  return str;
}

template <typename intType>
bool checkEqual(const std::string &name, const intType *actual,
                const std::vector<std::size_t> &shape,
                const intType *expected, std::size_t N) {
  auto it = actual;
  auto end = it + N;
  bool equal = true;
  for (; it != end; ++it, ++expected) {
    if (*it != *expected) {
      std::cerr << "mismatch on element ";
      equal = false;
      const auto n = it - actual;
      std::cerr << prettyCoord(name, n, shape) << ':';
      std::cerr << " expected=" << *expected;
      std::cerr << " actual=" << *it << '\n';
    }
  }
  return equal;
}

template bool checkEqual<unsigned>(
    const std::string &, const unsigned *,
    const std::vector<std::size_t> &,
    const unsigned *, std::size_t);

template bool checkEqual<std::uint64_t>(
    const std::string &, const std::uint64_t *,
    const std::vector<std::size_t> &,
    const std::uint64_t *, std::size_t);

template <typename FPType>
bool checkIsClose(const std::string &name, const FPType *actual,
                  const std::vector<std::size_t> &shape,
                  const FPType *expected, std::size_t N,
                  double relativeTolerance,
                  double absoluteTolerance) {
  auto it = actual;
  auto end = it + N;
  bool isClose = true;
  for (; it != end; ++it, ++expected) {
    if (!checkIsClose(*it, *expected, relativeTolerance)) {
      if (std::fabs(*expected) < 0.01 && checkIsClose(*it, *expected,
                                                      5 * relativeTolerance)) {
        std::cerr << "close to mismatch on element ";
        // values close to zero have 5x the tolerance
      } else if   (std::fabs(*expected - *it) < absoluteTolerance) {
        std::cerr << "within absolute tolerance bounds on element ";
      } else {
        std::cerr << "mismatch on element ";
        isClose = false;
      }
      const auto n = it - actual;
      std::cerr << prettyCoord(name, n, shape) << ':';
      std::cerr << " expected=" << *expected;
      std::cerr << " actual=" << *it << '\n';
    }
  }
  return isClose;
}

template bool checkIsClose<float>(const std::string &, const float *,
                                  const std::vector<std::size_t> &,
                                  const float *, std::size_t, double, double);

template bool checkIsClose<double>(const std::string &, const double *,
                                   const std::vector<std::size_t> &,
                                   const double *, std::size_t, double, double);

} // end namespace util
} // end namespace poplibs_test

namespace std {
std::istream &operator>>(std::istream &in, poplar::Type &type) {
  std::string token;
  in >> token;
  if (token == "half")
    type = poplar::HALF;
  else if (token == "float")
    type = poplar::FLOAT;
  else if (token == "unsigned")
    type = poplar::UNSIGNED_INT;
  else if (token == "int")
    type = poplar::INT;
  else if (token == "bool")
    type = poplar::BOOL;
  else
    throw poputil::poplib_error(
      "Invalid data-type <" + token + ">; must be half, float, int or bool");
  return in;
}
}
