#include <poplib_test/Util.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <util/Compiler.hpp>
#include <popstd/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;

namespace poplib_test {
namespace util {

std::unique_ptr<char []>
allocateHostMemoryForTensor(const Tensor &t) {
  const auto dType = t.elementType();
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
allocateHostMemoryForTensor(const Tensor &t,
                            Sequence &upload, Sequence &download) {
  std::unique_ptr<char []> p = allocateHostMemoryForTensor(t);
  upload.add(Copy(p.get(), t));
  download.add(Copy(t, p.get()));
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

bool checkIsClose(double a, double b, double relativeTolerance) {
  return boost::math::fpc::close_at_tolerance<double>(relativeTolerance)(a, b);
}

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

bool checkIsClose(const std::string &name, const double *actual,
                  const std::vector<std::size_t> &shape,
                  const double *expected, std::size_t N,
                  double relativeTolerance) {
  auto it = actual;
  auto end = it + N;
  bool isClose = true;
  for (; it != end; ++it, ++expected) {
    if (!checkIsClose(*it, *expected, relativeTolerance)) {
      if (fabs(*expected) < 0.01 && checkIsClose(*it, *expected,
                                                 5 * relativeTolerance)) {
        std::cerr << "close to mismatch on element ";
        // values close to zero have 5x the tolerance
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

const char *asString(const FPDataType &type) {
  switch (type) {
  case FPDataType::HALF: return "half";
  case FPDataType::FLOAT: return "float";
  }
  POPLIB_UNREACHABLE();
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
    throw popstd::poplib_error(
      "Invalid data-type <" + token + ">; must be half or float");
  return in;
}

} // end namespace util
} // end namespace poplib_test
