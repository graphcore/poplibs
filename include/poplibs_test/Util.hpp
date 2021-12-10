// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_Util_hpp
#define poplibs_test_Util_hpp

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <poplar/Target.hpp>
#include <poplar/Type.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/MultiArray.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace poplibs_test {
namespace util {

namespace detail {

template <typename T>
void inline copyToDevice(const poplar::Target &target, const T *src, void *dst,
                         std::size_t n) {}

template <>
void inline copyToDevice(const poplar::Target &target, const float *src,
                         void *dst, std::size_t n) {
  poplar::copyFloatToDeviceHalf(target, src, dst, n);
}

template <>
void inline copyToDevice(const poplar::Target &target, const double *src,
                         void *dst, std::size_t n) {
  poplar::copyDoubleToDeviceHalf(target, src, dst, n);
}

template <typename T>
void inline copyFromDevice(const poplar::Target &target, const void *src,
                           T *dst, std::size_t n) {}

template <>
void inline copyFromDevice(const poplar::Target &target, const void *src,
                           float *dst, std::size_t n) {
  poplar::copyDeviceHalfToFloat(target, src, dst, n);
}

template <>
void inline copyFromDevice(const poplar::Target &target, const void *src,
                           double *dst, std::size_t n) {
  poplar::copyDeviceHalfToDouble(target, src, dst, n);
}

} // namespace detail

std::unique_ptr<char[]>
allocateHostMemoryForTensor(const poplar::Target &target,
                            const poplar::Tensor &t, unsigned replicationFactor,
                            std::size_t &allocatedSizeInBytes);

std::unique_ptr<char[]>
allocateHostMemoryForTensor(const poplar::Target &target,
                            const poplar::Tensor &t,
                            unsigned replicationFactor);

std::unique_ptr<char[]> allocateHostMemoryForTensor(
    const poplar::Tensor &t, const std::string &name, poplar::Graph &graph,
    boost::optional<poplar::program::Sequence &> uploadProg,
    boost::optional<poplar::program::Sequence &> downloadProg,
    std::vector<std::pair<std::string, char *>> &map);

void attachStreams(poplar::Engine &e,
                   const std::vector<std::pair<std::string, char *>> &map);

template <typename T>
void writeRandomBinaryValues(const poplar::Target &target,
                             const poplar::Type &type, T *begin, T *end, T a,
                             T b, std::mt19937 &randomEngine);

template <class T, std::size_t N>
void inline writeRandomBinaryValues(const poplar::Target &target,
                                    const poplar::Type &type,
                                    boost::multi_array<T, N> &x, T a, T b,
                                    std::mt19937 &randomEngine) {
  return writeRandomBinaryValues(
      target, type, x.data(), x.data() + x.num_elements(), a, b, randomEngine);
}

template <class T>
void inline writeRandomBinaryValues(const poplar::Target &target,
                                    const poplar::Type &type,
                                    poplibs_support::MultiArray<T> &x, T a, T b,
                                    std::mt19937 &randomEngine) {
  return writeRandomBinaryValues(
      target, type, x.data(), x.data() + x.numElements(), a, b, randomEngine);
}

/// Fill a vector with values in the interval [min:max)
/// The specific values returned seem the same on ubuntu/gcc and
/// osx/clang
template <typename T>
void writeRandomValues(const poplar::Target &target, const poplar::Type &type,
                       T *begin, T *end, T min, T max,
                       std::mt19937 &randomEngine);

template <typename T>
void inline writeRandomValues(const poplar::Target &target,
                              const poplar::Type &type, std::vector<T> &a,
                              T min, T max, std::mt19937 &randomEngine) {
  return writeRandomValues(target, type, a.data(), a.data() + a.size(), min,
                           max, randomEngine);
}

template <class T, std::size_t N>
void inline writeRandomValues(const poplar::Target &target,
                              const poplar::Type &type,
                              boost::multi_array<T, N> &a, T min, T max,
                              std::mt19937 &randomEngine) {
  return writeRandomValues(target, type, a.data(), a.data() + a.num_elements(),
                           min, max, randomEngine);
}

template <class T>
void inline writeRandomValues(const poplar::Target &target,
                              const poplar::Type &type,
                              poplibs_support::MultiArray<T> &a, T min, T max,
                              std::mt19937 &randomEngine) {
  return writeRandomValues(target, type, a.data(), a.data() + a.numElements(),
                           min, max, randomEngine);
}

template <typename T, std::size_t N>
void inline writeRandomValues(const poplar::Target &target,
                              const poplar::Type type,
                              std::vector<boost::multi_array<T, N>> &a,
                              const T min, const T max,
                              std::mt19937 &randomEngine) {
  for (unsigned i = 0; i < a.size(); ++i) {
    writeRandomValues(target, type, a[i], min, max, randomEngine);
  }
}

size_t maxContiguousInteger(const poplar::Type &t);

size_t maxContiguousIntegerFromBinaryOp(const poplar::Type &inputType,
                                        const poplar::Type &outputType);

// Return true if the number of multiply-accumulates per output element is
// likely to exceed the contiguous range of exactly representable integer
// values for either `inputType` or `outputType`, assuming the values being
// accumulated are randomly chosen from {-1, 1}.
// Note that the distribution of {-1, 1} is not strictly Bernoulli, as
// Bernoulli uses the values {0, 1}.
bool isLikelyToHaveNumericalErrorsUsingBernoulli(
    size_t macsPerOutputElement, const poplar::Type &inputType,
    const poplar::Type &outputType);

template <typename T>
void copy(const poplar::Target &target, const T *src, std::size_t n,
          const poplar::Type &dstType, void *dst) {

  // For HALF we have to convert ...
  if (dstType == poplar::HALF) {
    detail::copyToDevice<T>(target, src, dst, n);
    return;
  }
  // ... for all other types we just do a std::copy. We just need to cast to
  // the appropriate pointer type
#define STD_COPY(DEV_TYPE, HOST_TYPE)                                          \
  if (dstType == poplar::DEV_TYPE) {                                           \
    std::copy(src, src + n, reinterpret_cast<HOST_TYPE *>(dst));               \
    return;                                                                    \
  }
  STD_COPY(FLOAT, float)
  STD_COPY(INT, int)
  STD_COPY(UNSIGNED_INT, unsigned int)
  STD_COPY(SHORT, short)
  STD_COPY(UNSIGNED_SHORT, unsigned short)
  STD_COPY(CHAR, char)
  STD_COPY(SIGNED_CHAR, signed char)
  STD_COPY(UNSIGNED_CHAR, unsigned char)
  STD_COPY(BOOL, bool)
  STD_COPY(UNSIGNED_LONGLONG, unsigned long long)
  STD_COPY(LONGLONG, long long)
#undef STD_COPY
  throw std::runtime_error("destination type " + dstType.toString() +
                           " not supported in poplibs_test::util::copy()");
}

template <typename T, unsigned long N>
inline void copy(const poplar::Target &target, boost::multi_array_ref<T, N> src,
                 const poplar::Type &dstType, void *dst) {
  assert(src.storage_order() == boost::c_storage_order());
  copy(target, src.data(), src.num_elements(), dstType, dst);
}

template <typename T>
inline void copy(const poplar::Target &target, const std::vector<T> &src,
                 const poplar::Type &dstType, void *dst) {
  copy(target, src.data(), src.size(), dstType, dst);
}

inline void copy(const poplar::Target &target,
                 const poplibs_support::MultiArray<double> &src,
                 const poplar::Type &dstType, void *dst) {
  copy(target, src.data(), src.numElements(), dstType, dst);
}

template <typename T>
void copy(const poplar::Target &target, const poplar::Type &srcType, void *src,
          T *dst, size_t n) {

  // For HALF we have to convert ...
  if (srcType == poplar::HALF) {
    detail::copyFromDevice<T>(target, src, dst, n);
    return;
  }

  // ... for all other types we just do a std::copy. We just need to cast to
  // the appropriate pointer types
#define STD_COPY(DEV_TYPE, HOST_TYPE)                                          \
  if (srcType == poplar::DEV_TYPE) {                                           \
    std::copy(reinterpret_cast<HOST_TYPE *>(src),                              \
              reinterpret_cast<HOST_TYPE *>(src) + n, dst);                    \
    return;                                                                    \
  }
  STD_COPY(FLOAT, float)
  STD_COPY(INT, int)
  STD_COPY(UNSIGNED_INT, unsigned int)
  STD_COPY(SHORT, short)
  STD_COPY(UNSIGNED_SHORT, unsigned short)
  STD_COPY(CHAR, char)
  STD_COPY(SIGNED_CHAR, signed char)
  STD_COPY(UNSIGNED_CHAR, unsigned char)
  STD_COPY(BOOL, bool)
  STD_COPY(UNSIGNED_LONGLONG, unsigned long long)
  STD_COPY(LONGLONG, long long)

#undef STD_COPY
  throw std::runtime_error("source type " + srcType.toString() +
                           " not supported in poplibs_test::util::copy()");
}

template <typename T>
inline void copy(const poplar::Target &target, const poplar::Type &srcType,
                 void *src, std::vector<T> &dst) {
  copy(target, srcType, src, dst.data(), dst.size());
}

template <typename T, unsigned long N>
inline void copy(const poplar::Target &target, const poplar::Type &srcType,
                 void *src, boost::multi_array_ref<T, N> dst) {
  assert(dst.storage_order() == boost::c_storage_order());
  copy(target, srcType, src, dst.data(), dst.num_elements());
}

inline void copy(const poplar::Target &target, const poplar::Type &srcType,
                 void *src, poplibs_support::MultiArray<double> &dst) {
  copy(target, srcType, src, dst.data(), dst.numElements());
}

template <typename T, unsigned N>
inline void copy(const poplar::Target &target, const poplar::Type &type,
                 const std::vector<boost::multi_array<T, N>> &src,
                 const std::vector<std::unique_ptr<char[]>> &dst) {
  for (unsigned i = 0; i < src.size(); ++i) {
    copy(target, src[i], type, dst[i].get());
  }
}

template <typename T, unsigned N>
inline void copy(const poplar::Target &target, const poplar::Type &type,
                 const std::vector<std::unique_ptr<char[]>> &src,
                 const std::vector<boost::multi_array<T, N>> &dst) {
  for (unsigned i = 0; i < src.size(); ++i) {
    copy(target, type, src[i].get(), dst[i]);
  }
}

template <typename intType>
bool checkEqual(const std::string &name, const intType *actual,
                const std::vector<std::size_t> &shape, const intType *expected,
                std::size_t N);

template <typename FPType>
bool checkIsClose(FPType a, FPType b, double relativeTolerance);

template <typename FPType>
bool checkIsClose(const std::string &name, const FPType *actual,
                  const std::vector<std::size_t> &shape, const FPType *expected,
                  std::size_t N, double relativeTolerance,
                  double absoluteTolerance = 0);

inline bool
checkIsClose(const std::string &name, const std::size_t *const shape_,
             const std::size_t rank, const double *const actual,
             const std::size_t numActualElements, const double *const expected,
             const std::size_t numExpectedElements,
             const double relativeTolerance, const double absoluteTolerance) {
  if (numActualElements != numExpectedElements) {
    std::cerr << "mismatched number of elements [" + name + "]:";
    std::cerr << " expected=" << numExpectedElements;
    std::cerr << " actual=" << numActualElements << '\n';
    return false;
  }
  std::vector<std::size_t> shape;
  for (unsigned i = 0; i != rank; ++i) {
    shape.push_back(shape_[i]);
  }

  return checkIsClose(name, actual, shape, expected, numActualElements,
                      relativeTolerance, absoluteTolerance);
}

template <std::size_t N>
inline bool checkIsClose(const std::string &name,
                         const boost::multi_array<double, N> &actual,
                         const boost::multi_array<double, N> &expected,
                         double relativeTolerance,
                         double absoluteTolerance = 0) {
  assert(actual.storage_order() == boost::c_storage_order());
  assert(expected.storage_order() == boost::c_storage_order());
  return checkIsClose(name, actual.shape(), N, actual.data(),
                      actual.num_elements(), expected.data(),
                      expected.num_elements(), relativeTolerance,
                      absoluteTolerance);
}

inline bool checkIsClose(const std::string &name,
                         const poplibs_support::MultiArray<double> &actual,
                         const poplibs_support::MultiArray<double> &expected,
                         double relativeTolerance,
                         double absoluteTolerance = 0) {
  const auto shape = actual.shape();
  assert(shape.size() > 0);
  return checkIsClose(name, &shape[0], actual.numDimensions(), actual.data(),
                      actual.numElements(), expected.data(),
                      expected.numElements(), relativeTolerance,
                      absoluteTolerance);
}

template <class T> struct VectorOption {
  std::vector<T> val;

  VectorOption() = default;
  VectorOption(const T &x) { val.push_back(x); }

  operator const std::vector<T> &() const { return val; }
  const std::vector<T> *operator->() const { return &val; }
  const T &operator[](std::size_t i) const { return val[i]; }
  typename std::vector<T>::const_iterator begin() const { return val.begin(); }
  typename std::vector<T>::const_iterator end() const { return val.end(); }
};

template <class T> struct ShapeOption : public VectorOption<T> {
  bool canBeBroadcast = false;

  ShapeOption() = default;
  ShapeOption(const T &x) : VectorOption<T>(x), canBeBroadcast(true) {}

  void broadcast(unsigned numDims) {
    if (!canBeBroadcast)
      return;
    assert(this->val.size() == 1);
    this->val.resize(numDims, this->val.back());
    canBeBroadcast = false;
  }
};

template <class T>
std::ostream &operator<<(std::ostream &os, const VectorOption<T> &s);
template <class T>
std::istream &operator>>(std::istream &in, VectorOption<T> &s);
template <class T>
std::ostream &operator<<(std::ostream &os, const ShapeOption<T> &s);
template <class T>
std::istream &operator>>(std::istream &in, ShapeOption<T> &s);

inline void skipSpaces(std::istream &in) {
  while (std::isspace(in.peek()))
    in.ignore();
}

#ifdef __clang__
// -Wrange-loop-analysis does not work here because this is called with
// containers that iterate over values, *and* containers that iterate over
// references, so the range loop analysis always complains about one or the
// other.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#endif

template <class T>
std::ostream &operator<<(std::ostream &os, const VectorOption<T> &s) {
  os << '{';
  bool needComma = false;
  for (const auto &x : s.val) {
    if (needComma)
      os << ", ";
    os << x;
    needComma = true;
  }
  return os << '}';
}
template <class T>
std::ostream &operator<<(std::ostream &os, const ShapeOption<T> &s) {
  if (s.canBeBroadcast) {
    assert(s.val.size() == 1);
    return os << s.val.front();
  } else {
    return os << static_cast<const VectorOption<T> &>(s);
  }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace detail {

template <class T> inline T readValue(std::istream &in) {
  std::string number;
  auto c = in.peek();
  if (!std::isdigit(c) && c != '-' && c != '.')
    throw std::runtime_error("Invalid value; expected digit or `.'");
  do {
    number += in.get();
    c = in.peek();
  } while (std::isdigit(c) || c == '.');
  return boost::lexical_cast<T>(number);
}

template <> inline std::string readValue<std::string>(std::istream &in) {
  std::string value;
  auto c = in.peek();
  if (!std::isalpha(c))
    throw std::runtime_error("Invalid value; expected character");
  do {
    value += in.get();
  } while (std::isalpha(in.peek()));
  return value;
}

} // namespace detail

template <class T>
std::istream &operator>>(std::istream &in, VectorOption<T> &s) {
  auto c = in.peek();
  if (c == '{') {
    in.ignore();
    skipSpaces(in);
    auto c = in.peek();
    if (c == '}') {
      in.ignore();
    } else {
      while (true) {
        s.val.push_back(detail::readValue<T>(in));
        skipSpaces(in);
        c = in.get();
        if (c == '}') {
          break;
        } else if (c != ',') {
          throw std::runtime_error("Invalid vector; expected `,' or `}'");
        }
        skipSpaces(in);
      }
    }
  } else {
    throw std::runtime_error("Invalid vector; expected `{'");
  }
  return in;
}

template <class T>
std::istream &operator>>(std::istream &in, ShapeOption<T> &s) {
  auto c = in.peek();
  if (c == '{') {
    return in >> static_cast<VectorOption<T> &>(s);
  } else {
    if (!std::isdigit(c) && c != '-' && c != '.') {
      throw std::runtime_error("Invalid shape; expected `{', digit or `.'");
    }
    s.canBeBroadcast = true;
    s.val.push_back(detail::readValue<T>(in));
  }
  return in;
}

// Add a default set of global exchange constraints based on the number of IPUs
// in the model. The number of IPUs should be set before calling this function.
void addGlobalExchangeConstraints(poplar::IPUModel &ipuModel);

// Add a default global sync latency based on the number of IPUs in the model.
// The number of IPUs should be set before calling this function.
void setGlobalSyncLatency(poplar::IPUModel &ipuModel);

// Create an input tensor for a convolution with a generic layout that
// is representative of a typical layout in a neural network.
poplar::Tensor createGenericConvInput(
    poplar::Graph &graph, const poplar::Type &type, std::size_t batchSize,
    std::size_t numConvGroups, std::size_t chansPerConvGroup,
    const std::vector<std::size_t> &fieldShape, const std::string &name = "");

// Create an input tensor for a fully connected layer with a generic
// layout that is representative of a typical layout in a neural network.
poplar::Tensor createGenericFullyConnectedInput(
    poplar::Graph &graph, const poplar::Type &type, std::size_t numGroups,
    std::size_t batchSize, std::size_t inputSize, const std::string &name = "");

} // namespace util
} // namespace poplibs_test

namespace std {
std::istream &operator>>(std::istream &in, poplar::Type &type);

inline std::istream &operator>>(std::istream &in, poputil::Fp8Format &format) {
  std::string token;
  in >> token;
  if (token == "quart152") {
    format = poputil::Fp8Format::QUART152;
  } else if (token == "quart143") {
    format = poputil::Fp8Format::QUART143;
  } else {
    throw poputil::poplibs_error("Invalid fp8 format <" + token +
                                 " Must be quart143 or quart 152");
  }
  return in;
}

inline std::ostream &operator<<(std::ostream &os,
                                const poputil::Fp8Format &format) {
  if (format == poputil::Fp8Format::QUART152) {
    os << "quart152";
  } else if (format == poputil::Fp8Format::QUART143) {
    os << "quart143";
  }
  return os;
}

} // namespace std

#endif // poplibs_test_Util_hpp
