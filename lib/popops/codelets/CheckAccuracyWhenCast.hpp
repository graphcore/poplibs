// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef popops_codelets_CheckAccuracyWhenCast_hpp
#define popops_codelets_CheckAccuracyWhenCast_hpp

#include <cmath>

#ifdef __IPU__
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#include <poplar/TileConstants.hpp>
#endif // __IPU__

namespace popops {

// Do the actual casting and accuracy check. This must be done in a separate
// function while T12725 is not solved to prevent LLVM schedulers from
// reordering the corresponding instructions before the UPUT.
template <typename InputType, typename OutputType>
bool __attribute__((noinline)) castAndCheck(InputType input, float tolerance) {
  const auto castInput = static_cast<OutputType>(input);
  const auto relativeError = static_cast<InputType>(
      (static_cast<float>(std::fabs(input)) * tolerance));
  return relativeError > std::abs(static_cast<InputType>(castInput) - input);
}

template <typename InputType, typename OutputType>
bool checkAccuracyWhenCastComputeImpl(InputType input, float tolerance) {
#ifdef __IPU__
  // Disable exceptions as the following can create numbers that are out of
  // range in half precision.  We need to store / restore the FP_CTL as
  // the worker will continue to run the actual scaledAdd code - done
  // outside this function
  __builtin_ipu_uput(0x00000000,
                     CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
#endif
  return castAndCheck<InputType, OutputType>(input, tolerance);
}

// Check if two values x0, x1 are "accurate enough" when cast to HALF.
// For each value 'x' we compute a maximum acceptable error which is
// x*tolerance and then we check if (x-half(x)) is (strictly) smaller than
// that error.
// Return 'true' if both values pass the check.
bool checkAccuracyWhenCastFloatV2ToHalf(float x0, float x1, float tolerance);

} // namespace popops

#endif // popops_codelets_CheckAccuracyWhenCast_hpp
