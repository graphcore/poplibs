// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_support_VectorUtils_hpp
#define poplibs_support_VectorUtils_hpp

#include <functional>
#include <numeric>
#include <vector>

template <class T>
T product(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), T(1), std::multiplies<T>());
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

#endif // poplibs_support_VectorUtils_hpp
