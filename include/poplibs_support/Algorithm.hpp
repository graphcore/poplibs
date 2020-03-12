// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef poplibs_support_Algorithm_hpp
#define poplibs_support_Algorithm_hpp

#include <exception>
#include <type_traits>

#include <utility>

namespace poplibs_support {

// Unsigned integer version of log2 rounded up
// Single-line constexpr form added to allow compile-time calculation.
// Could be nicer if using multi-line constexpr function (needs C++14).
constexpr static unsigned ceilLog2Aux(unsigned n) {
  return (n ? 1 + ceilLog2Aux(n >> 1) : 0);
}
// Check if power of 2 and then call to count up to most significant bit
constexpr static unsigned ceilLog2(unsigned n) {
  return ((n & (n - 1)) ? 1 : 0) + ceilLog2Aux(n >> 1);
}

template <typename T, typename U> constexpr static auto ceildiv(T x, U y) {
  static_assert(std::is_unsigned<T>::value && std::is_unsigned<U>::value,
                "Only valid for unsigned integral types");
  return (x + y - 1) / y;
}

template <typename T, typename U> constexpr static auto floordiv(T x, U y) {
  static_assert(std::is_unsigned<T>::value && std::is_unsigned<U>::value,
                "Only valid for unsigned integral types");
  return x / y;
}

template <typename T, typename U> constexpr static auto roundUp(T x, U y) {
  return ceildiv(x, y) * y;
}

template <typename T, typename U> constexpr static auto roundDown(T x, U y) {
  return floordiv(x, y) * y;
}

// Calculate how to split `n` elements into `d` partitions
// using partitions of ceil(n/d) and floor(n/d) elements.
//
// Result is the number of partitions with ceil(n/d) elements
// and the number of partitions with floor(n/d) elements.
//
// i.e.
//
//   result.first * ceil(n/d) + result.second * floor(n/d) == n
//
// and
//
//   result.first + result.second == d
//
// If there is only one non-zero partition (i.e. n is evenly divisible
// by d) then this will always be in result.first
//
template <typename T>
constexpr static std::pair<T, T> balancedPartition(T n, T d) {
  static_assert(std::is_unsigned<T>::value,
                "Only valid for unsigned integral types");
  if (d == 0) {
    // We know we need 0 of either size of partition in this case
    // and n/d is inf and will signal if we try and calculate it
    // so just return the answer.
    return std::make_pair<T, T>(0, 0);
  }

  const auto x = ceildiv(n, d);

  // We calculate b first as if ceil(n/d) == floor(n/d)
  // (i.e. n is evenly divisible by d) we want a to be non-zero
  // rather than b.
  const auto b = x * d - n;
  const auto a = d - b;

  return std::make_pair(a, b);
}

} // end namespace poplibs_support

#endif // poplibs_support_Algorithm_hpp
