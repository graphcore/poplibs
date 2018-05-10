// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_Util_hpp
#define poplibs_test_Util_hpp
#include <cassert>
#include <memory>
#include <random>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Target.hpp>
#include <poplibs_test/Util.hpp>
#include <stdexcept>
#include <poplibs_support/Compiler.hpp>
#include <iostream>

namespace poplibs_test {
namespace util {
std::unique_ptr<char []>
allocateHostMemoryForTensor(const poplar::Target &target,
                            const poplar::Tensor &t);

std::unique_ptr<char []>
allocateHostMemoryForTensor(const poplar::Tensor &t,  const std::string &name,
                            poplar::Graph &graph,
                            std::vector<std::pair<std::string, char *>> &map);

void upload(poplar::Engine &e,
            std::vector<std::pair<std::string, char *>> &map);

void download(poplar::Engine &e,
              std::vector<std::pair<std::string, char *>> &map);

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

inline void
copy(const poplar::Target &target,
     const double *src,
     std::size_t n,
     const poplar::Type &dstType,
     void *dst) {
  if (dstType == poplar::FLOAT) {
    std::copy(src, src + n, reinterpret_cast<float*>(dst));
  } else if (dstType == poplar::HALF) {
    poplar::copyDoubleToDeviceHalf(target, src, dst, n);
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

inline void
copy(const poplar::Target &target,
     const poplar::Type &srcType,
     void *src,
     double *dst,
     size_t n) {
  if (srcType == poplar::FLOAT) {
    std::copy(reinterpret_cast<float*>(src),
              reinterpret_cast<float*>(src) + n,
              dst);
  } else if (srcType == poplar::HALF) {
    poplar::copyDeviceHalfToDouble(target, src, dst, n);
  } else if (srcType == poplar::INT) {
    std::copy(reinterpret_cast<int*>(src),
              reinterpret_cast<int*>(src) + n,
              dst);
  } else {
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

bool checkIsClose(const std::string &name, const double *actual,
                  const std::vector<std::size_t> &shape,
                  const double *expected, std::size_t N,
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

} // End namespace poplibs_test
} // End namespace ref.

namespace std {
std::istream &operator>>(std::istream &in, poplar::Type &type);
}

#endif // poplibs_test_Util_hpp
