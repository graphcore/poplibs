// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cmath>

#include "CheckAccuracyWhenCast.hpp"

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#ifdef __IPU__
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#endif // __IPU__

using namespace poplar;

namespace popops {

// Do the actual casting and accuracy check. This must be done in a separate
// function while T12725 is not solved to prevent LLVM schedulers from
// reordering the corresponding instructions before the UPUT.
template <>
bool __attribute__((noinline))
castAndCheck<float, half>(float input, float tolerance) {
#ifdef __IPU__
  // Cast to half and back to float, decision is based on relative error
  const auto castInput = static_cast<half>(input);
  return (ipu::fabs(input) * tolerance) >
         ipu::fabs(static_cast<float>(castInput) - input);

#else
  const auto castInput = static_cast<half>(input);
  // As the CPU doesn't deal with halves correctly, then exclude out of
  // range numbers (as half) from being considered accurate.
  return std::fabs(input) > 65504
             ? false
             : (std::fabs(input) * tolerance) >
                   std::fabs(static_cast<float>(castInput) - input);
#endif
}

bool checkAccuracyWhenCastFloatV2ToHalf(float x0, float x1, float tolerance) {
  if (std::fabs(x0) > 65504 || std::fabs(x1) > 65504)
    return false;
#ifdef __IPU__
  unsigned save_fp_ctl =
      __builtin_ipu_uget(CSR_W_FP_CTL__INDEX & CSR_UPPER_MASK);
  __builtin_ipu_uput(0x00000000, CSR_W_FP_CTL__INDEX & CSR_UPPER_MASK);
#endif
  float maxErr0 = std::fabs(tolerance * x0);
  float maxErr1 = std::fabs(tolerance * x1);
  half x0half = static_cast<half>(x0);
  half x1half = static_cast<half>(x1);
  float diff0 = static_cast<float>(x0half) - x0;
  float diff1 = static_cast<float>(x1half) - x1;
#ifdef __IPU__
  __builtin_ipu_uput(save_fp_ctl, CSR_W_FP_CTL__INDEX & CSR_UPPER_MASK);
#endif
  return std::fabs(diff0) < maxErr0 && std::fabs(diff1) < maxErr1;
}

template <typename InputType, typename OutputType>
class CheckAccuracyWhenCast : public Vertex {
public:
  const float tolerance;
  Input<InputType> input;
  Output<bool> output;

  CheckAccuracyWhenCast();

  bool compute() {
    *output = checkAccuracyWhenCastComputeImpl<InputType, OutputType>(
        *input, tolerance);
    return true;
  }
};

template class CheckAccuracyWhenCast<float, half>;

} // namespace popops
