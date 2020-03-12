// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/EncodingConstants.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popops {
namespace {
template <typename T>
inline T getParam(const T *params, unsigned index, unsigned start, unsigned end,
                  unsigned width) {
  static_assert(std::is_same<T, float>() || std::is_same<T, half>(),
                "T must be a either float or half");
  if (width <= index && index != MASKED_LABEL_CODE) {
    return static_cast<T>(__builtin_nanf(""));
  }
  if (index < start || end <= index || index == MASKED_LABEL_CODE) {
    return static_cast<T>(0.f);
  }
  return params[index - start];
}

// The type half does not have the -- operator.
template <typename T>
inline void decrementParams(T *params, unsigned index, unsigned startCol,
                            unsigned endCol, unsigned paramsWidth) {
  if (__builtin_expect(index < paramsWidth, 1)) {
    if (__builtin_expect(startCol <= index && index < endCol, 0)) {
      params[index - startCol] = params[index - startCol] - static_cast<T>(1.f);
    }
  } else {
    if (index != MASKED_LABEL_CODE) {
      for (unsigned col = startCol; col != endCol; ++col) {
        params[col - startCol] = static_cast<T>(__builtin_nanf(""));
      }
    }
  }
}

} // namespace
} // namespace popops
