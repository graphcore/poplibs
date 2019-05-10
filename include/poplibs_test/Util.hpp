// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Util_hpp
#define poplibs_test_Util_hpp

#include <cassert>
#include <memory>
#include <random>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Target.hpp>
#include <poplibs_test/Util.hpp>
#include <stdexcept>
#include <poplibs_support/Compiler.hpp>
#include <iostream>

namespace poplibs_test {
namespace util {

namespace detail {

template <typename T>
void copyToDevice(const poplar::Target &target, const T *src, void *dst,
                  std::size_t n) {}

template <>
void copyToDevice(const poplar::Target &target, const float *src, void *dst,
                  std::size_t n) {
  poplar::copyFloatToDeviceHalf(target, src, dst, n);
}

template <>
void copyToDevice(const poplar::Target &target, const double *src, void *dst,
                  std::size_t n) {
  poplar::copyDoubleToDeviceHalf(target, src, dst, n);
}

template <typename T>
void copyFromDevice(const poplar::Target &target, const void *src, T *dst,
                    std::size_t n) {}

template <>
void copyFromDevice(const poplar::Target &target, const void *src, float *dst,
                    std::size_t n) {
  poplar::copyDeviceHalfToFloat(target, src, dst, n);
}

template <>
void copyFromDevice(const poplar::Target &target, const void *src, double *dst,
                    std::size_t n) {
  poplar::copyDeviceHalfToDouble(target, src, dst, n);
}

} // namespace detail

std::unique_ptr<char []>
allocateHostMemoryForTensor(const poplar::Target &target,
                            const poplar::Tensor &t,
                            std::size_t &allocatedSizeInBytes);

std::unique_ptr<char []>
allocateHostMemoryForTensor(const poplar::Target &target,
                            const poplar::Tensor &t);

std::unique_ptr<char []>
allocateHostMemoryForTensor(const poplar::Tensor &t,  const std::string &name,
                            poplar::Graph &graph,
                            poplar::program::Sequence &uploadProg,
                            poplar::program::Sequence &downloadProg,
                            std::vector<std::pair<std::string, char *>> &map);

void attachStreams(poplar::Engine &e,
                   const std::vector<std::pair<std::string, char *>> &map);

/// Fill a vector with values in the interval [min:max)
/// The specific values returned seem the same on ubuntu/gcc and
/// osx/clang
void
writeRandomValues(const poplar::Target &target,
                  const poplar::Type &type,
                  double *begin, double *end, double min, double max,
                  std::mt19937 &randomEngine);

template <class T, std::size_t N>
void inline
writeRandomValues(const poplar::Target &target,
                  const poplar::Type &type,
                  boost::multi_array<T, N> &a, double min,
                  double max, std::mt19937 &randomEngine) {
  return writeRandomValues(target, type, a.data(), a.data() + a.num_elements(),
                           min, max, randomEngine);
}

template <typename T>
void
copy(const poplar::Target &target,
     const T *src,
     std::size_t n,
     const poplar::Type &dstType,
     void *dst) {
  if (dstType == poplar::FLOAT) {
    std::copy(src, src + n, reinterpret_cast<float*>(dst));
  } else if (dstType == poplar::HALF) {
    detail::copyToDevice<T>(target, src, dst, n);
  } else if (dstType == poplar::UNSIGNED_INT) {
    std::copy(src, src + n, reinterpret_cast<unsigned*>(dst));
  } else if (dstType == poplar::INT) {
    std::copy(src, src + n, reinterpret_cast<int*>(dst));
  } else {
    assert(dstType == poplar::BOOL);
    std::copy(src, src + n, reinterpret_cast<bool*>(dst));
  }
}

template <unsigned long N>
inline void
copy(const poplar::Target &target,
     boost::multi_array_ref<double, N> src,
     const poplar::Type &dstType,
     void *dst) {
  assert(src.storage_order() == boost::c_storage_order());
  copy(target, src.data(), src.num_elements(), dstType, dst);
}

template <typename T>
void
copy(const poplar::Target &target,
     const poplar::Type &srcType,
     void *src,
     T *dst,
     size_t n) {
  if (srcType == poplar::FLOAT) {
    std::copy(reinterpret_cast<float*>(src),
              reinterpret_cast<float*>(src) + n,
              dst);
  } else if (srcType == poplar::HALF) {
    detail::copyFromDevice<T>(target, src, dst, n);
  } else if (srcType == poplar::UNSIGNED_INT) {
    std::copy(reinterpret_cast<unsigned*>(src),
              reinterpret_cast<unsigned*>(src) + n,
              dst);
  } else if (srcType == poplar::INT) {
    std::copy(reinterpret_cast<int*>(src),
              reinterpret_cast<int*>(src) + n,
              dst);
  } else {
    assert(srcType == poplar::BOOL);
    std::copy(reinterpret_cast<bool*>(src),
              reinterpret_cast<bool*>(src) + n,
              dst);
  }
}

template <unsigned long N>
inline void
copy(const poplar::Target &target,
     const poplar::Type &srcType,
     void *src,
     boost::multi_array_ref<double, N> dst) {
  assert(dst.storage_order() == boost::c_storage_order());
  copy(target, srcType, src, dst.data(), dst.num_elements());
}

template <typename intType>
bool checkEqual(const std::string &name, const intType *actual,
                const std::vector<std::size_t> &shape,
                const intType *expected, std::size_t N);

template <typename FPType>
bool checkIsClose(FPType a, FPType b, double relativeTolerance);

template <typename FPType>
bool checkIsClose(const std::string &name, const FPType *actual,
                  const std::vector<std::size_t> &shape,
                  const FPType *expected, std::size_t N,
                  double relativeTolerance,
                  double absoluteTolerance = 0);

template <std::size_t N>
inline bool checkIsClose(const std::string &name,
                         const boost::multi_array<double, N> &actual,
                         const boost::multi_array<double, N> &expected,
                         double relativeTolerance,
                         double absoluteTolerance = 0) {
  assert(actual.storage_order() == boost::c_storage_order());
  assert(expected.storage_order() == boost::c_storage_order());
  if (actual.num_elements() != expected.num_elements()) {
    std::cerr << "mismatched number of elements [" + name + "]:";
    std::cerr << " expected=" << expected.num_elements();
    std::cerr << " actual=" << actual.num_elements() << '\n';
    return false;
  }
  std::vector<std::size_t> shape;
  for (unsigned i = 0; i != N; ++i)
    shape.push_back(actual.shape()[i]);
  return checkIsClose(name, actual.data(), shape,
                      expected.data(), actual.num_elements(),
                      relativeTolerance,
                      absoluteTolerance);
}

template <class T>
struct ShapeOption {
  bool canBeBroadcast = false;
  std::vector<T> val;

  ShapeOption() = default;
  ShapeOption(const T &x) : canBeBroadcast(true) {
    val.push_back(x);
  }

  void broadcast(unsigned numDims) {
    if (!canBeBroadcast)
      return;
    assert(val.size() == 1);
    val.resize(numDims, val.back());
    canBeBroadcast = false;
  }
};

template <class T>
std::ostream &operator<<(std::ostream &os, const ShapeOption<T> &s);
template <class T>
std::istream &operator>>(std::istream &in, ShapeOption<T> &s);

inline void skipSpaces(std::istream &in) {
  while (std::isspace(in.peek()))
    in.ignore();
}

template <class T>
std::ostream &operator<<(std::ostream &os, const ShapeOption<T> &s) {
  if (s.canBeBroadcast) {
    assert(s.val.size() == 1);
    return os << s.val.front();
  }
  os << '{';
  bool needComma = false;
  for (const auto x : s.val) {
    if (needComma)
      os << ", ";
    os << x;
    needComma = true;
  }
  return os << '}';
}

template <class T>
inline T readInteger(std::istream &in) {
  std::string number;
  auto c = in.peek();
  if (!std::isdigit(c) && c != '-')
    throw std::runtime_error("Invalid shape; expected digit");
  do {
    number += in.get();
  } while (std::isdigit(in.peek()));
  return boost::lexical_cast<T>(number);
}

template <class T>
std::istream &operator>>(std::istream &in, ShapeOption<T> &s) {
  auto c = in.peek();
  if (c == '{') {
    in.ignore();
    skipSpaces(in);
    auto c = in.peek();
    if (c == '}') {
      in.ignore();
    } else {
      while (true) {
        s.val.push_back(readInteger<T>(in));
        skipSpaces(in);
        c = in.get();
        if (c == '}') {
          break;
        } else if (c != ',') {
          throw std::runtime_error("Invalid shape; expected `,' or `}'");
        }
        skipSpaces(in);
      }
    }
  } else {
    if (!std::isdigit(c) && c != '-') {
      throw std::runtime_error("Invalid shape; expected `{' or digit");
    }
    s.canBeBroadcast = true;
    s.val.push_back(readInteger<T>(in));
  }
  return in;
}

// Add a default set of global exchange constraints based on the number of IPUs
// in the model. The number of IPUs should be set before calling this function.
void addGlobalExchangeConstraints(poplar::IPUModel &ipuModel);

// Add a default global sync latency based on the number of IPUs in the model.
// The number of IPUs should be set before calling this function.
void setGlobalSyncLatency(poplar::IPUModel &ipuModel);

}} // End namespace poplibs_test::util

namespace std {
std::istream &operator>>(std::istream &in, poplar::Type &type);
}

#endif // poplibs_test_Util_hpp
