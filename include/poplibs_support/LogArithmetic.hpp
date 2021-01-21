// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_log_arithmetic_hpp
#define poplibs_support_log_arithmetic_hpp

// Conditionally avoid including functions that use multi_array
#ifndef __POPC__
#include <boost/multi_array.hpp>
#endif

namespace poplibs_support {
namespace log {

inline constexpr auto min = -10000;

// Given log values, perform an equivalent `linear add` operation
template <typename FPType> FPType add(const FPType a_, const FPType b_) {
  FPType a = a_ < b_ ? b_ : a_;
  FPType b = a_ < b_ ? a_ : b_;
  return a + std::log(1 + std::exp(b - a));
}

// Given log values, perform an equivalent `linear sub` operation
template <typename FPType> FPType sub(const FPType a_, const FPType b_) {
  FPType a = a_ < b_ ? b_ : a_;
  FPType b = a_ < b_ ? a_ : b_;
  return a + std::log(1 - std::exp(b - a));
}

// Given log values, perform an equivalent `linear mul` operation
template <typename FPType> FPType mul(const FPType a, const FPType b) {
  return a + b;
}
// Given log values, perform an equivalent `linear divide` operation
template <typename FPType> FPType div(const FPType a, const FPType b) {
  return a - b;
}

#ifndef __POPC__
// Simply to save having to carefully enter values, use a softmax to
// convert them into probabilities
template <typename FPType>
boost::multi_array<FPType, 2> softMax(const boost::multi_array<FPType, 2> &in) {
  boost::multi_array<FPType, 2> out(boost::extents[in.size()][in[0].size()]);
  for (unsigned i = 0; i < in[0].size(); i++) {
    FPType sum = 0;
    for (unsigned j = 0; j < in.size(); j++) {
      sum += std::exp(in[j][i]);
    }
    for (unsigned j = 0; j < in.size(); j++) {
      out[j][i] = std::exp(in[j][i]) / sum;
    }
  }
  return out;
}
// Converted each individual element to natural log.
// Add a small constant to prevent numeric errors
template <typename FPType>
boost::multi_array<FPType, 2> log(const boost::multi_array<FPType, 2> &in) {
  boost::multi_array<FPType, 2> out(boost::extents[in.size()][in[0].size()]);
  for (unsigned i = 0; i < in.size(); i++) {
    for (unsigned j = 0; j < in[i].size(); j++) {
      out[i][j] = std::log(in[i][j] + 1e-50);
    }
  }
  return out;
}
// Find exp of each individual element
template <typename FPType>
boost::multi_array<FPType, 2> exp(const boost::multi_array<FPType, 2> &in) {
  boost::multi_array<FPType, 2> out(boost::extents[in.size()][in[0].size()]);
  for (unsigned i = 0; i < in.size(); i++) {
    for (unsigned j = 0; j < in[i].size(); j++) {
      out[i][j] = std::exp(in[i][j]);
    }
  }
  return out;
}
#endif // ifndef __POPC__
} // namespace log
} // namespace poplibs_support

#endif // poplibs_support_log_arithmetic_hpp
