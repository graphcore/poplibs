// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_gcd_hpp
#define poplibs_support_gcd_hpp

// Greatest common divisor
template <typename T> static T gcd(T a, T b) {
  while (b != 0) {
    T tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

// Least common multiple
template <typename T> static T lcm(T a, T b) { return (a * b) / gcd(a, b); }

#endif // poplibs_support_gcd_hpp
