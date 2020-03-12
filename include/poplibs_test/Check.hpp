// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_Check_hpp
#define poplibs_test_Check_hpp

#include <poplibs_support/MultiArray.hpp>

#include <iosfwd>
#include <vector>

template <typename T>
std::ostream &print(std::ostream &os, const T &arr, size_t len) {
  os << "{";
  for (size_t i = 0; i < len - 1; i++) {
    os << arr[i] << ", ";
  }
  os << arr[len - 1];
  os << "}";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  return print(os, vec, vec.size());
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const poplibs_support::MultiArray<T> &vec) {
  return print(os, vec, vec.numElements());
}

#define CHECK_IF(result, cond)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "Condition failed: " << #cond << '\n';                      \
      result = false;                                                          \
    }                                                                          \
  } while (false)

#define CHECK_ELEMWISE_EQ(result, lhs, rhs, len)                               \
  do {                                                                         \
    for (unsigned i = 0; i < len; ++i) {                                       \
      if (lhs[i] != rhs[i]) {                                                  \
        result = false;                                                        \
      }                                                                        \
    }                                                                          \
    if (!result) {                                                             \
      std::cerr << "Condition failed: " << #lhs << " == " #rhs << '\n';        \
      std::cerr << #lhs << lhs;                                                \
      std::cerr << #rhs << rhs;                                                \
    }                                                                          \
  } while (false)

#endif // poplibs_test_Check_hpp
