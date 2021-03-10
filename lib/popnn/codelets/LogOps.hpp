// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cmath>

namespace popnn {
// Given log values, perform an equivalent `linear mul` operation
template <typename FPType>
inline FPType logMul(const FPType a, const FPType b) {
  return a + b;
}
#ifdef __IPU__
// Given log values, perform an equivalent `linear add` operation
template <typename FPType>
inline FPType logAdd(const FPType a_, const FPType b_) {
  // Casting required as exp<half>() undefined
  const auto a = static_cast<float>(a_);
  const auto b = static_cast<float>(b_);
  float max, min;
  // Compiled code doesn't produce optimal f32max, f32min instructions
  asm(" f32max %[asm_max], %[asm_a], %[asm_b]"
      : [asm_max] "=r"(max)
      : [asm_a] "r"(a), [asm_b] "r"(b));
  asm(" f32min %[asm_min], %[asm_a], %[asm_b]"
      : [asm_min] "=r"(min)
      : [asm_a] "r"(a), [asm_b] "r"(b));
  return static_cast<FPType>(max + std::log(1 + std::exp(min - max)));
}
#else
template <typename FPType>
inline FPType logAdd(const FPType a_, const FPType b_) {
  FPType max = a_ < b_ ? b_ : a_;
  FPType min = a_ < b_ ? a_ : b_;
  // Casting required as exp<half>() undefined
  return static_cast<FPType>(
      static_cast<float>(max) +
      std::log(1 + std::exp(static_cast<float>(min - max))));
}
#endif

} // namespace popnn
