// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_VectorUtils_hpp
#define poplibs_support_VectorUtils_hpp

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

template <class T> inline T product(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), T(1), std::multiplies<T>());
}
template <class T> inline T sum(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), T(0), std::plus<T>());
}

template <class T>
inline std::vector<T> inversePermutation(const std::vector<T> &v) {
  static_assert(
      std::is_unsigned<T>(),
      "Data type must be unsigned integer as data represents indices");
  std::vector<T> result(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[v[i]] = i;
  }
  return result;
}

template <class To, class From>
std::vector<To> vectorConvert(const std::vector<From> &in) {
  std::vector<To> out;
  out.reserve(in.size());
  for (const auto &x : in) {
    out.emplace_back(x);
  }
  return out;
}

template <class T>
std::vector<T> removeSingletonDimensions(const std::vector<T> &v) {
  static_assert(std::is_integral<T>::value, "Integral required.");
  std::vector<T> out;
  for (auto e : v) {
    if (e != 1)
      out.emplace_back(e);
  }
  return out;
}

#endif // poplibs_support_VectorUtils_hpp
