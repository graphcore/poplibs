#ifndef _poplib_test_Util_hpp_
#define _poplib_test_Util_hpp_
#include <memory>
#include <random>
#include <boost/multi_array.hpp>
#include <poplar/Graph.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/Program.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>

namespace poplib_test {
namespace util {
std::unique_ptr<char []>
allocateHostMemoryForTensor(poplar::Graph &graph, const poplar::Tensor &t);

std::unique_ptr<char []>
allocateHostMemoryForTensor(poplar::Graph &graph, const poplar::Tensor &t,
                            poplar::program::Sequence &upload,
                            poplar::program::Sequence &download);

/// Fill a vector with values in the interval [min:max)
/// The specific values returned seem the same on ubuntu/gcc and
/// osx/clang
void
writeRandomValues(double *begin, double *end, double min, double max,
                  std::mt19937 &randomEngine);

template <class T, std::size_t N>
void inline
writeRandomValues(boost::multi_array<T, N> &a, double min,
                  double max, std::mt19937 &randomEngine) {
  return writeRandomValues(a.data(), a.data() + a.num_elements(),
                           min, max, randomEngine);
}

void groupActivations(boost::const_multi_array_ref<double, 4> src,
                      const std::string &dstType,
                      const std::vector<std::size_t> &dstDims,
                      void *dst);

void groupWeights(boost::const_multi_array_ref<double, 4> src,
                  const std::string &dstType,
                  const std::vector<std::size_t> &dstDims,
                  void *dst);

void groupWeights(boost::const_multi_array_ref<double, 4> src,
                  const std::string &dstType,
                  const std::vector<std::size_t> &dstDims,
                  void *dst);

void
ungroupWeights(const std::string &srcType,
               const std::vector<std::size_t> &srcDims,
               const void *src,
               boost::multi_array_ref<double, 4> dst);

template <unsigned long N>
inline void
copy(boost::multi_array_ref<double, N> src,
     const std::string &dstType,
     void *dst) {
  assert(src.storage_order() == boost::c_storage_order());
  if (dstType == "float") {
    std::copy(src.data(), src.data() + src.num_elements(),
              reinterpret_cast<float*>(dst));
  } else {
    assert(dstType == "half");
    std::copy(src.data(), src.data() + src.num_elements(),
              reinterpret_cast<poplar::half*>(dst));
  }
}

template <unsigned long N>
inline void
copy(const std::string &srcType,
     void *src,
     boost::multi_array_ref<double, N> dst) {
  assert(dst.storage_order() == boost::c_storage_order());
  if (srcType == "float") {
    std::copy(reinterpret_cast<float*>(src),
              reinterpret_cast<float*>(src) + dst.num_elements(),
              dst.data());
  } else {
    assert(srcType == "half");
    std::copy(reinterpret_cast<poplar::half*>(src),
              reinterpret_cast<poplar::half*>(src) + dst.num_elements(),
              dst.data());
  }
}

void ungroupActivations(const std::string &srcType,
                        const std::vector<std::size_t> &srcDims,
                        const void *src,
                        boost::multi_array_ref<double, 4> dst);

bool checkIsClose(const std::string &name, const double *actual,
                  const std::vector<std::size_t> &dims,
                  const double *expected, std::size_t N,
                  double relativeTolerance);

template <std::size_t N>
inline bool checkIsClose(const std::string &name,
                         const boost::multi_array<double, N> &actual,
                         const boost::multi_array<double, N> &expected,
                         double relativeTolerance) {
  assert(actual.storage_order() == boost::c_storage_order());
  assert(expected.storage_order() == boost::c_storage_order());
  if (actual.num_elements() != expected.num_elements()) {
    std::cerr << "mismatched number of elements [" + name + "]:";
    std::cerr << " expected=" << expected.num_elements();
    std::cerr << " actual=" << actual.num_elements() << '\n';
    return false;
  }
  std::vector<std::size_t> dims;
  for (unsigned i = 0; i != N; ++i)
    dims.push_back(actual.shape()[i]);
  return checkIsClose(name, actual.data(), dims,
                      expected.data(), actual.num_elements(),
                      relativeTolerance);
}

enum class FPDataType {
  HALF,
  FLOAT
};

const char *asString(const FPDataType &type);
std::ostream &operator<<(std::ostream &os, const FPDataType &type);
std::istream &operator>>(std::istream &in, FPDataType &type);

} // End namespace poplib_test
} // End namespace ref.


#endif  // _poplib_test_Util_hpp_
