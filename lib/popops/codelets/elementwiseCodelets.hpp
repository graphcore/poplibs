// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef elementwise_codeletes_hpp
#define elementwise_codeletes_hpp
/** \file
 *
 * Codelets for element-wise operations.
 *
 */

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

#ifdef __IPU__
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>

// helper templates to differentiate between scalar and vector types
template <class T> struct isVectorType { static const bool value = false; };
template <> struct isVectorType<float2> { static const bool value = true; };
template <> struct isVectorType<half2> { static const bool value = true; };
template <> struct isVectorType<half4> { static const bool value = true; };

// helper templates for float type
template <class T> struct isFloatType { static const bool value = false; };
template <> struct isFloatType<float2> { static const bool value = true; };
template <> struct isFloatType<float> { static const bool value = true; };

static __attribute__((always_inline)) unsigned getWsr(void) {
  return __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
}

// Use attributes to ensure that the maskForRepeat is inline just prior to the
// repeat instruction and therefore picked up by the compiler pass which
// optimises for repeat instructions.
// See T9902.
static __attribute__((always_inline)) unsigned maskForRepeat(unsigned input) {
  return input & CSR_W_REPEAT_COUNT__VALUE__MASK;
}

// Called by each of the workers of a MultiVertex Unary or
// Binary/Broadcast vertex to find out how many elements it must process.
// Each worker will be assigned a whole number of vectors (1 vector =
// 2/4 elements).
//
// Example:   size = 283 elements,
//          vector = 4 elements (vectorWidthShifts=2)
//
// There are floor(283/4) = 70 vectors to process (with remainder=3 elements)
//
// This function will return to each workers the following number of vectors to
// process:
//
//          worker id:      0     1     2     3     4     5
// vectors to process:     12     12   12    12    11    11
//
// The remainder 3 elements (not assigned by this function) will be processed
// by worker 5, which will normally be assigned one less vector to process, as
// in the above example [unless floor(size)/vectorSize is a multiple of 6].
//
// \param size        The total number of elements that ALL the workers started
//                    by the MultiVertex will process.
//
// \vectorWidthShifts log2(vector size). i.e.:
//                         vectorWidthShifts=1  =>  vector has 2 elements
//                         vectorWidthShifts=2  =>  vector has 4 elements
//
// \worker            The id (0..5) of the worker
//
// \return number     The number of *vectors* that 'worker' has to process
//
static __attribute__((always_inline)) unsigned
divideWork(const unsigned size, const unsigned vectorWidthShifts,
           const unsigned worker) {
  // Multiply by 0xaaab and shift by 18 is just a multiplication by 1/6 in
  // fixed point (Q14.18)
  return (((size >> vectorWidthShifts) + 5 - worker) * 0xaaab) >> 18;
}
#endif

namespace {

// Macros to instantiate a template class for an operator and a number
// of types.
#define INSTANTIATE_OP_1(v, op, t) template class v<op, t>;
#define INSTANTIATE_OP_2(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_1(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_3(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_2(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_4(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_3(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_5(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_4(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_6(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_5(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_7(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_6(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_8(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_7(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_9(v, op, t, ...)                                        \
  template class v<op, t>;                                                     \
  INSTANTIATE_OP_8(v, op, __VA_ARGS__)

#define SELECT_VARGS(_1, _2, _3, _4, _5, _6, _7, _8, _9, NAME, ...)            \
  INSTANTIATE_OP##NAME
#define INSTANTIATE_OP(v, op, ...)                                             \
  SELECT_VARGS(__VA_ARGS__, _9, _8, _7, _6, _5, _4, _3, _2, _1)                \
  (v, op, __VA_ARGS__)

// Helper function to explicitly and specifically cast half to float
// to indicate this is intentional floating point promotion, while
// allowing other types to pass through unchanged.
template <typename T> struct PromoteHalfsToFloatsHelper {
  using ReturnType =
      typename std::conditional<std::is_same<T, half>::value, float, T>::type;
  ReturnType operator()(const T &x) { return static_cast<ReturnType>(x); }
};

template <typename T>
typename PromoteHalfsToFloatsHelper<T>::ReturnType PromoteHalfsToFloats(T x) {
  return PromoteHalfsToFloatsHelper<T>()(x);
}

template <class T1, class T2, class T3 = T1>
using enable_same_size_t = std::enable_if_t<sizeof(T1) == sizeof(T2), T3>;

template <class RetTy, class ArgTy>
enable_same_size_t<RetTy, ArgTy> copy_cast(ArgTy const &val) {
  RetTy ret;
  memcpy(&ret, &val, sizeof(RetTy));
  return ret;
}

#ifdef __IPU__
char4 tochar4(short4 val) {
  return char4{static_cast<char>(val[0]), static_cast<char>(val[1]),
               static_cast<char>(val[2]), static_cast<char>(val[3])};
}

char4 tochar4(int2 lo, int2 hi) {
  short4 tmp{static_cast<short>(lo[0]), static_cast<short>(lo[1]),
             static_cast<short>(hi[0]), static_cast<short>(hi[1])};
  return tochar4(tmp);
}

char4 tochar4(long2 lo, long2 hi) {
  short4 tmp{static_cast<short>(lo[0]), static_cast<short>(lo[1]),
             static_cast<short>(hi[0]), static_cast<short>(hi[1])};
  return tochar4(tmp);
}
#endif

} // namespace

namespace architecture {
// Use tag dispatching in place of #ifdef
struct generic;
struct ipu;
#ifdef __IPU__
using active = ipu;
#else
using active = generic;
#endif
} // namespace architecture

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;

#endif // elementwise_codeletes_hpp
