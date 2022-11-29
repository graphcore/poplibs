// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#ifdef __IPU__
#include <ipu_builtins.h>
#endif

#include <cassert>
#include <cmath>
#include <cstring>
#include <tuple>

#include "elementwiseCodelets.hpp"
#include "popops/ExprOp.hpp"
#include "util.hpp"

#ifdef __IPU__
#include "inlineAssembler.hpp"
#include "inlineAssemblerUnaryOps.hpp"
#endif

using namespace poplar;

namespace popops {

using UnaryOpType = popops::expr::UnaryOpType;

namespace {

// certain operators need to call a different function depending on the input
// type. this deals with dispatching unary ops to their correct functions.
template <expr::UnaryOpType Op> struct UnaryLibCall {};

template <> struct UnaryLibCall<expr::UnaryOpType::ABSOLUTE> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x) const {
    return ipu::fabs(x);
  }
#endif

  int operator()(int x) const { return std::abs(x); }
  long long operator()(long long x) const { return std::abs(x); }
};

template <> struct UnaryLibCall<expr::UnaryOpType::SQRT> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x) const {
    return ipu::sqrt(x);
  }
#endif

  int operator()(int x) const { return std::sqrt(x); }
};

template <> struct UnaryLibCall<expr::UnaryOpType::CBRT> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x) const {
    const auto xFloat = PromoteHalfsToFloats(x);
    const auto third = PromoteHalfsToFloats(Const<FPType>(1.f / 3.f));
    return ipu::copysign(ipu::exp(ipu::log(ipu::fabs(xFloat)) * third), xFloat);
  }
#endif
};

template <> struct UnaryLibCall<expr::UnaryOpType::ERF> {
#ifdef __IPU__
  float poly(float x) const {
    constexpr unsigned numCoeffs = 3;
    constexpr float coeffs[numCoeffs] = {7.181612491607666015625f,
                                         -0.4331750571727752685546875f,
                                         0.7397372722625732421875f};

    float y = coeffs[0];
    for (unsigned i = 1; i != numCoeffs; ++i) {
      y = y * x + coeffs[i];
    }
    return y * x;
  }

  // Approximation of error function
  // Cecil Hastings Jr : Approximations for Digital Computers Pg 169.
  // The max absolute error in fp32 is <2.5e-5 with max error at 0.067. As the
  // approximation is only used for half type, the max absolute error of the
  // final half result is 2.65e-4 with max relative error exceeding 1e-3
  // occuring at ranges < 1e-5.
  // Do all computations in fp32 as error introduced due to the polynomial
  // computation is expected to be significant.
  float compute(float xAbs) const {
    const float eta = 1.0f / (2.1255340576171875f + xAbs);
    const auto y = (1.0f - poly(eta) * ipu::exp(-xAbs * xAbs));
    return y;
  }

  template <typename FPType> FPType operator()(FPType x) const {
    if constexpr (isFloatType<FPType>::value) {
      return ipu::erf(x);
    } else {
      const auto xAbs = ipu::fabs(x);
      FPType y;
      if constexpr (isVectorType<FPType>::value) {
        unsigned n = sizeof(x) / sizeof(x[0]);
        for (unsigned i = 0; i != n; ++i) {
          y[i] = compute(static_cast<float>(xAbs[i]));
        }
      } else {
        y = compute(static_cast<float>(xAbs));
      }
      return ipu::copysign(y, x);
    }
  }
#endif
};

template <> struct UnaryLibCall<expr::UnaryOpType::GELU_ERF> {
#ifdef __IPU__
  float poly(float x) const {
    constexpr unsigned numCoeffs = 3;
    constexpr float coeffs[numCoeffs] = {20.31266740574137, -0.8663501338893118,
                                         1.046146442804559};
    float y = coeffs[0];
    for (unsigned i = 1; i != numCoeffs; ++i) {
      y = y * x + coeffs[i];
    }
    return y * x;
  }

  // Approximation of error function based on with erf computed on
  // input / sqrt(2).
  // Cecil Hastings Jr : Approximations for Digital Computers Pg 169.
  // In double precision, error is <1.5e-5. The error is similar to a 5th
  // order polynomial where the erf differs for small values of x. As the
  // approximation is used only for fp16, the max relative error is around 1e03.
  // The (1 + erf(x/sqrt(2)))  done in fp16 has similar results because for
  // small x, 1 + erf(x/sqrt(2)) ~= 1 with the product with x pushing the result
  // toward 0.
  float compute(float xAbs) const {
    const float eta = 1.0f / (3.005959067258476f + xAbs);
    const auto y = (1.0f - poly(eta) * ipu::exp(-xAbs * xAbs / 2.0f));
    return y;
  }

  template <typename FPType> FPType operator()(FPType x) const {
    // Use approximation only for half types as we guarantee the polynomial
    // expansion gives the same error as the one using ipu::erf
    if constexpr (isFloatType<FPType>::value) {
      return (Const<FPType>(1.0f) +
              ipu::erf(Const<FPType>(0.7071067811865475f) * x)) *
             Const<FPType>(0.5f) * x;
    } else {
      const auto xAbs = ipu::fabs(x);
      FPType y;
      if constexpr (isVectorType<FPType>::value) {
        unsigned n = sizeof(x) / sizeof(x[0]);
        for (unsigned i = 0; i != n; ++i) {
          y[i] = compute(static_cast<float>(xAbs[i]));
        }
      } else {
        y = compute(static_cast<float>(xAbs));
      }
      return (ipu::copysign(y, x) + Const<FPType>(1.0f)) * Const<FPType>(0.5f) *
             x;
    }
  }
#endif
};

template <> struct UnaryLibCall<expr::UnaryOpType::LOGARITHM_ONE_PLUS> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x) const {
    const auto mOne = Const<FPType>(-1.0f);
    // clang-format off
#pragma fast-math push
#pragma fast-math off
    // clang-format on
    FPType a = x - mOne;
    FPType b = a + mOne;
    // clang-format off
#pragma fast-math pop
    // clang-format on
    return ipu::log(a) + (x - b) / a;
  }
#endif
};

#ifdef __IPU__

// domain below which sine(theta) = theta for half precision
constexpr float smallAngleApproxThresh = 3.90625e-03f;

// Compute Tau function based on Legendre polynomial of the 5th order.
// (https://mae.ufl.edu/~uhk/IEEETrigpaper8.pdf)
// The algorithm used here deviates from the one in the paper in that the input
// range is extended over  (-pi:pi) which simplifies the calculation without
// having to check ranges that should enable this to be vectorised. Note that
// the approximation in the range [-pi/4:pi/4] is even better than the native
// implementation used by the compiler.
// The relative error for sine is less than 1ulp except at ~339 * 2 * pi where
// it is 1.6ulp
// The relative error for cosine is less than 1ulp except at the following
// points
// +/-54304 (8.4ulp error and value of function is 1.86373627898000e-04)
// +/-42944 (1.1ulp)
// +/-887.5 (2.6ulp)
// +/-532.5 (1.64ulp)
// +/-177.5 (1.06ulp)
// We could use a common sine function by using the identity
// cos(x) = sin(x + pi/2) but the addition would need to be in float precision.

// Return (x % (2*pi) - pi)/2
// Get to within 2*pi and and then subtract pi to get within [-pi,pi].
// Further scale by 0.5 to get within [-pi/2,pi/2] because we actually
// compute angle/2 to avoid checking for quadrants.
static float rangeReduceAndHalve(float x) {
  // Split the numerator into two parts such that the product of the first part
  // with xNorm always remains within the mantissa of FP32. After the first
  // subtraction x * piDen - xNorm * piNum, there is no loss of information but
  // with range substantially reduced if the number is close to 2*pi. The second
  // part is then subtracted.
  constexpr float piNum1 = 6.28125f;
  constexpr float piNum2 = 0.00193530716933310031890869140625f;
  constexpr float invTwoPi = 0.15915493667125701904296875f;
  const auto xNorm = ipu::floor(x * invTwoPi) + 0.5f;
  return (x - xNorm * piNum1 - xNorm * piNum2) * 0.5f;
}
#endif

template <> struct UnaryLibCall<expr::UnaryOpType::COS> {
#ifdef __IPU__
  float compute(float x) const {
    const auto y = rangeReduceAndHalve(x);
    const auto y2 = y * y;
    constexpr float c0 = 945.0f;
    constexpr float c1 = -105.0f;
    constexpr float c2 = -420.000f;
    constexpr float c3 = 15.0f;
    const auto num = y * (c0 + y2 * (c1 + y2));
    const auto den = (c0 + y2 * (c2 + c3 * y2));
    const auto num2 = num * num;
    const auto den2 = den * den;
    return (num2 - den2) / (num2 + den2);
  }

  template <typename FPType> FPType operator()(FPType x) const {
    if constexpr (isFloatType<FPType>::value) {
      return ipu::cos(x);
    } else {
      if constexpr (isVectorType<FPType>::value) {
        unsigned n = sizeof(x) / sizeof(x[0]);
        FPType y;
        for (unsigned i = 0; i != n; ++i) {
          y[i] = compute(static_cast<float>(x[i]));
        }
        return y;
      } else {
        return compute(static_cast<float>(x));
      }
    }
  }
#endif
};

template <> struct UnaryLibCall<expr::UnaryOpType::SIN> {
#ifdef __IPU__
  float compute(float x) const {
    const auto y = rangeReduceAndHalve(x);
    const auto y2 = y * y;
    constexpr float c0 = 945.0f;
    constexpr float c1 = -105.00625f;
    constexpr float c2 = -420.0007f;
    constexpr float c3 = 14.99822f;
    const auto num = y * (c0 + y2 * (c1 + y2));
    const auto den = (c0 + y2 * (c2 + c3 * y2));
    return -2.0f * num * den / (num * num + den * den);
  }

  template <typename FPType> FPType operator()(FPType x) const {
    if constexpr (isFloatType<FPType>::value) {
      return ipu::sin(x);
    } else {
      if constexpr (isVectorType<FPType>::value) {
        unsigned n = sizeof(x) / sizeof(x[0]);
        FPType y;
        for (unsigned i = 0; i != n; ++i) {
          y[i] = compute(static_cast<float>(x[i]));
        }
        // use small angle approximation (sin(x) = x)
        auto mask = ipu::fabs(x) < Const<FPType>(smallAngleApproxThresh);
        auto *xPtr = reinterpret_cast<decltype(mask) *>(&x);
        auto *yPtr = reinterpret_cast<decltype(mask) *>(&y);
        *yPtr = (*xPtr & mask) | (*yPtr & ~mask);
        return y;
      } else {
        FPType y = compute(static_cast<float>(x));
        if (ipu::fabs(x) < Const<FPType>(smallAngleApproxThresh)) {
          y = x;
        }
        return y;
      }
    }
  }
#endif
};

// Structure with template specialization to define the output type
// of a unary operation
template <expr::UnaryOpType op, typename T> struct UnaryOpOutputType {
  using type = T;
};

#ifndef __IPU__
#define DEFINE_UNARY_OUTPUT_TYPE_BOOL(op)                                      \
  template <typename T> struct UnaryOpOutputType<op, T> {                      \
    using type = bool;                                                         \
  };
#else
#define DEFINE_UNARY_OUTPUT_TYPE_BOOL(op)                                      \
  template <typename T> struct UnaryOpOutputType<op, T> {                      \
    using type = bool;                                                         \
  };                                                                           \
  template <> struct UnaryOpOutputType<op, float2> {                           \
    using type = int2;                                                         \
  };                                                                           \
  template <> struct UnaryOpOutputType<op, half4> {                            \
    using type = short4;                                                       \
  };
#endif

DEFINE_UNARY_OUTPUT_TYPE_BOOL(expr::UnaryOpType::IS_FINITE)
DEFINE_UNARY_OUTPUT_TYPE_BOOL(expr::UnaryOpType::IS_INF)
DEFINE_UNARY_OUTPUT_TYPE_BOOL(expr::UnaryOpType::IS_NAN)

// Structure with template specialization to define the function
// that performes that operation on one element
template <expr::UnaryOpType op, typename T, typename A> struct UnaryOpFn {};

template <expr::UnaryOpType op, typename T>
using UnaryOpOutputType_t = typename UnaryOpOutputType<op, T>::type;

template <expr::UnaryOpType op, typename InT, typename OutT>
constexpr bool hasInlineAssemblerInnerLoopImpl() {
#ifdef __IPU__
  constexpr bool typeMatch =
      std::is_same<InT, OutT>::value &&
      (std::is_same<InT, float>::value || std::is_same<InT, half>::value);

  constexpr bool opMatch =
      (op == expr::UnaryOpType::ABSOLUTE) ||
      (op == expr::UnaryOpType::EXPONENT) ||
      (op == expr::UnaryOpType::EXPONENT2) ||
      (op == expr::UnaryOpType::INVERSE) ||
      (op == expr::UnaryOpType::LOGARITHM) ||
      (op == expr::UnaryOpType::NEGATE) || (op == expr::UnaryOpType::SQRT) ||
      (op == expr::UnaryOpType::SQUARE) || (op == expr::UnaryOpType::RSQRT);

  return typeMatch && opMatch;
#else
  return false;
#endif
}

template <expr::UnaryOpType op, typename InT, typename OutT>
static void inline computeUnaryInnerLoop(const InT *src, OutT *dst,
                                         unsigned numSamples,
                                         unsigned stride = 1) {

  for (unsigned i = 0; i != numSamples; i += stride) {
    dst[i] = UnaryOpFn<op, InT, architecture::generic>::fn(src[i]);
  }
}

#define DEFINE_UNARY_OP_FN_GEN(op, body)                                       \
  template <typename T, typename A> struct UnaryOpFn<op, T, A> {               \
    using arch = architecture::generic;                                        \
    static UnaryOpOutputType_t<op, T> fn(T x) { body }                         \
  };

#ifdef __IPU__
#define DEFINE_UNARY_OP_FN_IPU(op, body)                                       \
  template <typename T> struct UnaryOpFn<op, T, architecture::ipu> {           \
    using arch = architecture::ipu;                                            \
    static UnaryOpOutputType_t<op, T> fn(T x) { body }                         \
  };

#define DEFINE_UNARY_OP_FN_1(op, body)                                         \
  DEFINE_UNARY_OP_FN_GEN(op, body)                                             \
  DEFINE_UNARY_OP_FN_IPU(op, body)

#define DEFINE_UNARY_OP_FN_2(op, body, ipubody)                                \
  DEFINE_UNARY_OP_FN_GEN(op, body)                                             \
  DEFINE_UNARY_OP_FN_IPU(op, ipubody)

#else
#define DEFINE_UNARY_OP_FN_1(op, body) DEFINE_UNARY_OP_FN_GEN(op, body)
#define DEFINE_UNARY_OP_FN_2(op, body, _) DEFINE_UNARY_OP_FN_GEN(op, body)
#endif

#define UNARY_VARGS(_1, _2, _3, N, ...) DEFINE_UNARY_OP_FN##N
#define DEFINE_UNARY_OP_FN(...) UNARY_VARGS(__VA_ARGS__, _2, _1, _)(__VA_ARGS__)

// helper macro for the common case of just a different namespace
#define DEFINE_UNARY_OP_FN_STD(op, fn)                                         \
  DEFINE_UNARY_OP_FN(op, return std::fn(PromoteHalfsToFloats(x));              \
                     , return ipu::fn(x);)

DEFINE_UNARY_OP_FN(
    expr::UnaryOpType::ABSOLUTE,
    if constexpr (std::is_integral<T>::value) { return std::abs(x); } else {
      // This path is avoided for half specialisation, but we
      // promote half to float to avoid promotion error.
      return std::fabs(PromoteHalfsToFloats(x));
    },
    return UnaryLibCall<expr::UnaryOpType::ABSOLUTE>{}(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::BITWISE_NOT, return ~x;)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::CBRT,
                   return std::cbrt(PromoteHalfsToFloats(x));
                   , return UnaryLibCall<expr::UnaryOpType::CBRT>{}(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::ERF,
                   return std::erf(PromoteHalfsToFloats(x));
                   , return UnaryLibCall<expr::UnaryOpType::ERF>{}(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::GELU_ERF,
                   return (1.0f + std::erf(7.071067811865475e-01f *
                                           PromoteHalfsToFloats(x))) *
                          0.5f * PromoteHalfsToFloats(x);
                   , return UnaryLibCall<expr::UnaryOpType::GELU_ERF>{}(x);)
DEFINE_UNARY_OP_FN(
    expr::UnaryOpType::LOGARITHM_ONE_PLUS,
    return std::log1p(PromoteHalfsToFloats(x));
    , return UnaryLibCall<expr::UnaryOpType::LOGARITHM_ONE_PLUS>{}(x);)

DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::CEIL, ceil)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::COS,
                   return std::cos(PromoteHalfsToFloats(x));
                   , return UnaryLibCall<expr::UnaryOpType::COS>{}(x);)

DEFINE_UNARY_OP_FN(expr::UnaryOpType::COUNT_LEADING_ZEROS,
                   return x ? __builtin_clz(x) : 32;)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::EXPONENT, exp)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::EXPONENT_MINUS_ONE, expm1)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::EXPONENT2, exp2)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::FLOOR, floor)

DEFINE_UNARY_OP_FN(expr::UnaryOpType::INVERSE, return 1 / x;)

DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_INF,
                   return std::isinf(PromoteHalfsToFloats(x));
                   , return __builtin_ipu_isinf(PromoteHalfsToFloats(x));)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_FINITE,
                   return std::isfinite(PromoteHalfsToFloats(x));
                   , return __builtin_ipu_isfinite(PromoteHalfsToFloats(x));)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_NAN,
                   return std::isnan(PromoteHalfsToFloats(x));
                   , return __builtin_ipu_isnan(PromoteHalfsToFloats(x));)

DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::LOGARITHM, log)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::LOGICAL_NOT, return !x;)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::NEGATE, return -x;)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::POPCOUNT, return __builtin_popcount(x);)
#ifdef __IPU__
// comparison functions that return 1/0 in the arg type rather than true/false
template <typename T>
typename std::enable_if<!isVectorType<T>::value, T>::type compareLT(T a, T b) {
  return a < b;
}
template <typename T>
typename std::enable_if<isVectorType<T>::value, T>::type compareLT(T a, T b) {
  T r;
  unsigned n = sizeof(a) / sizeof(a[0]);
  for (unsigned i = 0u; i < n; ++i)
    r[i] = a[i] < b[i];
  return r;
}
#endif
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIGNUM, return (0 < x) - (x < 0);
                   , return compareLT(decltype(x){0}, x) -
                            compareLT(x, decltype(x){0});)

DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIN,
                   return std::sin(PromoteHalfsToFloats(x));
                   , return UnaryLibCall<expr::UnaryOpType::SIN>{}(x);)

DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::ASIN, asin)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::TAN,
                   return std::tan(PromoteHalfsToFloats(x));
                   , return ipu::sin(x) / ipu::cos(x);)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::TANH, tanh)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::RELU,
                   return (x > decltype(x){0}) ? x : decltype(x){0};)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::ROUND, round)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::TRUNC, trunc)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQRT,
                   return std::sqrt(PromoteHalfsToFloats(x));
                   , return UnaryLibCall<expr::UnaryOpType::SQRT>{}(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQUARE, return (x * x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIGMOID,
                   using Ty = decltype(PromoteHalfsToFloats(x));
                   return Ty(1) / (Ty(1) + std::exp(-Ty(x)));
                   , return ipu::sigmoid(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::RSQRT,
                   using Ty = decltype(PromoteHalfsToFloats(x));
                   return Ty(1) / std::sqrt(Ty(x));, return ipu::rsqrt(x);)

} // namespace

template <expr::UnaryOpType op, typename inT, typename outT, typename A>
struct UnaryOpDispatch {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) inT *in,
                      __attribute__((align_value(8))) outT *out) {
    computeUnaryInnerLoop<op, inT, outT>(in, out, size);
  }
};

#ifdef __IPU__

/// Performs the bulk of a unary 'op' that has bool as output type:
///    T op T => BOOL (LOGICAL_NOT, IS_FINITE, etc).
/// This processes 4 elements of type T in each cycle of the loop, and writes
/// the 4 aggregated boolean results (1 full word) at a time, avoiding calls
/// to __st8/__st16.
/// This is run both by the '2D' and '1D' vertices .
/// Implemented as as a struct with a static function instead of a templated
/// function because you cannot partially specialise a templated function
/// (also, doesn't use operator() because it cannot be static)
///
///  \tparam     op        Operation to perform
///
///  \tparam     T         Type of the operands
///
///  \tparam     stride    Stride (in units of 4 elements) to advance
///                        in/out at each cycle of the loop.
///
///  \param[in]  loopCount How many groups of 4 elements to process.
///
///  \param[in]  in        Pointer to an array of T with input operand.
///
///  \param[out] out       Pointer to an array of boolean (1 byte each) that
///                        will be populated with the results
///
template <UnaryOpType op, typename T, unsigned stride> struct unaryBoolOpBulk {
  static void compute(unsigned loopCount, const T *in, int *out) {
    for (unsigned i = 0; i < loopCount; i++) {
      unsigned result4 = 0;
      // Accumulate in 'result4' the 4 bytes for this loop
      for (unsigned j = 0, shifts = 0; j != 4; ++j, shifts += 8) {
        bool res = UnaryOpFn<op, T, architecture::active>::fn(in[j]);
        result4 |= res << shifts;
      }
      in += 4 * stride;
      *out = result4;
      out += stride;
    }
  }
};

// Optimisations for LOGICAL_NOT processing 4 boolean stored in a single word.
// This can be done by a 'vectorized' operation on the whole word.
template <unsigned stride>
struct unaryBoolOpBulk<UnaryOpType::LOGICAL_NOT, bool, stride> {
  static void compute(unsigned loopCount, const bool *in, int *out) {
    const unsigned *b4In = reinterpret_cast<const unsigned *>(in);
    for (unsigned i = 0; i < loopCount; i++) {
      unsigned word = *b4In;
      //  LOGICAL_NOT is NOT of the two words, plus masking with 01010101
      *out = ~word & 0x01010101;
      b4In += stride;
      out += stride;
    }
  }
};

/// Unary 'op' for bool output type ( T op T => BOOL) for the
/// trailing 0..3 elements of the input data (used in conjunction with
/// 'unaryBoolOpBulk')
/// This is run both by the '2D' and '1D' vertices
///
///  \tparam     op        Operation to perform. One of the comparison ops
///                        (EQUAL, LESS_THAN, ...)
///
///  \tparam     T         data type (float or half)
///
///  \param[in]  size      Total number of data elements
///
///  \param[in]  in        Pointer to an array of T with first operand.
///
///  \param[out] out       Pointer to an array of boolean (1 byte each) that
///                        is/will be populated with the results
///
template <UnaryOpType op, typename T>
void unaryBoolOpRemainder(unsigned size, const T *in,
                          __attribute__((align_value(4))) bool *out) {
  unsigned remainder = size & 3;
  if (remainder) {
    unsigned offs = size - remainder;
    in = &in[offs]; // make it point to the 'remainder'
    // Read the word of the output in memory that will contain the 1-3 bytes
    // of the remainder, so that we can write back the byte(s) we will not be
    // updating.
    unsigned *out4 = reinterpret_cast<unsigned *>(&out[offs]);
    unsigned result4 = *out4;
    // Accumulate in 'result4' the 1-3 bytes of the remainder
    unsigned mask = 0xff;
    for (unsigned j = 0, shifts = 0; j != remainder; ++j, shifts += 8) {
      bool res = UnaryOpFn<op, T, architecture::active>::fn(in[j]);
      result4 &= ~(mask << shifts);
      result4 |= res << shifts;
    }
    // Do a single word write back to memory
    *out4 = result4;
  }
}

template <UnaryOpType op, typename T>
struct UnaryOpDispatch<op, T, bool, architecture::ipu> {
  static void compute(unsigned size, const T *in,
                      __attribute__((align_value(4))) bool *out) {
    if (size >= 4) {
      const rptsize_t loopCount = size / 4u;
      unaryBoolOpBulk<op, T, 1>::compute(loopCount, in,
                                         reinterpret_cast<int *>(out));
    }
    unaryBoolOpRemainder<op, T>(size, in, out);
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatch<op, half, bool, architecture::ipu> {

  template <class T> using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const half4 *h4In = reinterpret_cast<const half4 *>(in);
      int *iOut = reinterpret_cast<int *>(out);
      const rptsize_t loopCount = size / 4u;
      for (unsigned i = 0; i < loopCount; ++i) {
        short4 calc = static_cast<short4>(FuncTy<half4>::fn(*h4In++));
        char4 result = tochar4(calc);
        *iOut++ = copy_cast<int>(result) & 0x01010101;
      }
      in = reinterpret_cast<const half *>(h4In);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      *out++ = FuncTy<half>::fn(*in++);
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatch<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const float2 *f2In = reinterpret_cast<const float2 *>(in);
      int *iOut = reinterpret_cast<int *>(out);

      const rptsize_t loopCount = size / 4u;
      for (unsigned i = 0; i < loopCount; ++i) {
        int2 calc_lo = static_cast<int2>(FuncTy<float2>::fn(*f2In++));
        int2 calc_hi = static_cast<int2>(FuncTy<float2>::fn(*f2In++));

        char4 result = tochar4(calc_lo, calc_hi);
        *iOut++ = copy_cast<int>(result) & 0x01010101;
      }
      in = reinterpret_cast<const float *>(f2In);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      *out++ = FuncTy<float>::fn(*in++);
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatch<op, half, half, architecture::ipu> {
  static_assert(
      std::is_same<half, typename UnaryOpOutputType<op, half>::type>::value,
      "");
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) half *out) {
    using arch = architecture::ipu;

    const half4 *h4In = reinterpret_cast<const half4 *>(in);
    half4 *h4Out = reinterpret_cast<half4 *>(out);
    if constexpr (hasInlineAssemblerInnerLoopImpl<op, half, half>()) {
      std::tie(h4In, h4Out) =
          inlineAssemblerUnaryOp<op, half, 1>::loopBody(size / 4, h4In, h4Out);
    } else {
      if (size >= 4) {
        // Partially unroll the loop
        // Can this be done automatically? See T59317
        half4 load = *h4In++;
        const rptsize_t loopCount = (size / 4u) - 1u;
        for (unsigned i = 0; i < loopCount; ++i) {
          half4 calc = UnaryOpFn<op, half4, arch>::fn(load);
          load = ipu::load_postinc(&h4In, 1);
          *h4Out++ = calc;
        }
        *h4Out++ = UnaryOpFn<op, half4, arch>::fn(load);
      }
    }
    in = reinterpret_cast<const half *>(h4In);
    half *tmp = reinterpret_cast<half *>(h4Out);
    size -= (tmp - out);
    out = tmp;

    const half2 *h2In = reinterpret_cast<const half2 *>(in);
    half2 *h2Out = reinterpret_cast<half2 *>(out);

    if (size >= 2) {
      *h2Out++ = UnaryOpFn<op, half2, arch>::fn(*h2In++);
      size -= 2;
    }

    if (size == 1) {
      write16Aligned32(UnaryOpFn<op, half, arch>::fn((*h2In)[0]), h2Out);
    }
  }
};

template <expr::UnaryOpType op>
class UnaryOpDispatch<op, float, float, architecture::ipu> {
public:
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) float *out) {
    const float2 *f2In = reinterpret_cast<const float2 *>(in);
    float2 *f2Out = reinterpret_cast<float2 *>(out);
    if constexpr (hasInlineAssemblerInnerLoopImpl<op, float, float>()) {
      inlineAssemblerUnaryOp<op, float, 1>::loopBody(size / 2, f2In, f2Out);
    } else {
      if (size >= 2) {
        const rptsize_t loopCount = (size / 2u) - 1;

        // Partially unroll the loop
        // Can this be done automatically? See T59317
        float2 load = *f2In++;
        for (unsigned j = 0; j < loopCount; j++) {
          float2 calc = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
          load = ipu::load_postinc(&f2In, 1);
          *f2Out++ = calc;
        }
        *f2Out++ = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
      }
    }
    if (size & 1) {
      out[size - 1] = UnaryOpFn<op, float, architecture::ipu>::fn(in[size - 1]);
    }
  }
};

/// Performs the bulk of any operation that can be performed on a 'short2'
/// vector (starting from an array of 16 bit short integers).
///
///  \tparam     op        Operation to  perform
///
///  \tparam     stride    Stride (in units of a vector) to advance in/out at
///                        each cycle of the loop.
///
///  \param[in]  loopCount How many vectors to process.
///
///  \param[in]  in        Pointer to an array with the input data.
///
///  \param[out] out       Pointer to an array that will be populated with the
///                        results
///
template <UnaryOpType op, typename T, unsigned stride>
static void unaryShort2Bulk(unsigned loopCount,
                            const __attribute__((align_value(4))) T *in,
                            __attribute__((align_value(4))) T *out) {
  const short2 *s2In = reinterpret_cast<const short2 *>(in);
  short2 *s2Out = reinterpret_cast<short2 *>(out);
  for (unsigned j = 0; j < loopCount; j++) {
    *s2Out = UnaryOpFn<op, short2, architecture::active>::fn(*s2In);
    s2In += stride;
    s2Out += stride;
  }
}

/// Processes the 'trailing' 1-3 elements (if present) for any operation that
/// can be performed on a 'short4'/'short2' vector (starting from an array
/// of 16 bit short integers)
///
/// \tparam op            Operation to perform
///
/// \tparam vectorWidthShifts  1 or 2 to indicate if we have a vector of 2
///                            or 4 elements, i.e.: log2(vectorWidth)
///
/// \param[in]  size      Total number of elements contained in 'in'
///
/// \param[in]  in        Pointer to an array of input data
///
/// \param[out] out       Pointer to an array that will be populated with the
///                       results
///
template <UnaryOpType op, typename T, unsigned vectorWidthShifts>
static void unaryShort2Short4Remainder(unsigned size,
                                       const __attribute__((align_value(8)))
                                       T *in,
                                       __attribute__((align_value(8))) T *out) {
  constexpr unsigned mask = (1 << vectorWidthShifts) - 1;
  const unsigned rem = size & mask;
  const short2 *s2In = reinterpret_cast<const short2 *>(&in[size - rem]);
  short2 *s2Out = reinterpret_cast<short2 *>(&out[size - rem]);
  if constexpr (mask == 0x3) {
    if (size & 0x3) {
      *s2Out = UnaryOpFn<op, short2, architecture::active>::fn(*s2In);
      s2In++;
      s2Out++;
    }
  }
  if (size & 1) {
    short2 res = {
        UnaryOpFn<UnaryOpType::BITWISE_NOT, short, architecture::active>::fn(
            (*s2In)[0]),
        (*s2Out)[1],
    };
    *s2Out = res;
  }
}

template <UnaryOpType op, typename T>
static void unaryShort2_MultiVertex(unsigned size, unsigned worker,
                                    const __attribute__((align_value(4))) T *in,
                                    __attribute__((align_value(4))) T *out) {
  const rptsize_t loopCount = divideWork(size, 1, worker);

  unaryShort2Bulk<op, T, CTXT_WORKERS>(loopCount, in + 2 * worker,
                                       out + 2 * worker);

  // To process the trailing elements (if any) use the last worker as it is
  // most likely to have less to do than the others.
  if (worker == (CTXT_WORKERS - 1)) {
    unaryShort2Short4Remainder<op, T, 1>(size, in, out);
  }
}

template <UnaryOpType op, typename T>
static void unaryShort2_2D(unsigned size,
                           const __attribute__((align_value(4))) T *in,
                           __attribute__((align_value(4))) T *out) {
  const rptsize_t loopCount = size / 2u;
  unaryShort2Bulk<op, T, 1>(loopCount, in, out);
  unaryShort2Short4Remainder<op, T, 1>(size, in, out);
}

template <UnaryOpType op>
class UnaryOpDispatch<op, short, short, architecture::ipu> {
public:
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) short *in,
                      __attribute__((align_value(8))) short *out) {
    unaryShort2_2D<op, short>(size, in, out);
  }
};

template <UnaryOpType op>
class UnaryOpDispatch<op, unsigned short, unsigned short, architecture::ipu> {
public:
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) unsigned short *in,
                      __attribute__((align_value(8))) unsigned short *out) {
    unaryShort2_2D<op, unsigned short>(size, in, out);
  }
};

#endif

template <UnaryOpType op, typename T> constexpr static bool isExternal() {
  const bool isExternalSIGNUM =
      op == UnaryOpType::SIGNUM && std::is_same<T, half>::value;
  const bool isExternalBITWISE =
      op == UnaryOpType::BITWISE_NOT &&
      (std::is_same<T, short>::value || std::is_same<T, unsigned short>::value);
  const bool isExternalNonLinearity =
      (op == UnaryOpType::TANH || op == UnaryOpType::SIGMOID ||
       op == UnaryOpType::RELU) &&
      (std::is_same<T, half>::value || std::is_same<T, float>::value);
  return isExternalSIGNUM || isExternalBITWISE || isExternalNonLinearity;
}

template <expr::UnaryOpType op, typename T> class UnaryOp2D : public Vertex {
  typedef typename UnaryOpOutputType<op, T>::type outputType;

public:
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in;
  Vector<Output<Vector<outputType, SPAN, 8>>> out;
  IS_EXTERNAL_CODELET((isExternal<op, T>()));

  void compute() {
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    unsigned limI = out.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::UnaryOpDispatch<op, T, outputType, arch>::compute(
          out[i].size(), &in[i][0], &out[i][0]);
    }
  }
};

template <expr::UnaryOpType op, typename T>
class UnaryOp2DInPlace : public Vertex {
  typedef typename UnaryOpOutputType<op, T>::type outputType;
  static_assert(std::is_same<T, outputType>::value,
                "In, Out types must match for in place operations");

public:
  Vector<InOut<Vector<T, SPAN, 8>>> inOut;
  IS_EXTERNAL_CODELET((isExternal<op, T>()));

  void compute() {
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    unsigned limI = inOut.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::UnaryOpDispatch<op, T, outputType, arch>::compute(
          inOut[i].size(), &inOut[i][0], &inOut[i][0]);
    }
  }
};

// The *in-place* vertices for the 3 non-linearity operators TANH, SIGMOID, RELU
// have their vertex state fields defined differently from the rest of the
// operators, so require separate templates.
#define DEFINE_UNARY_OP_NL_2D(op, LAYOUT)                                      \
  template <typename T> class UnaryOp2DInPlace<op, T> : public Vertex {        \
    typedef typename UnaryOpOutputType<op, T>::type outputType;                \
    static_assert(std::is_same<T, outputType>::value,                          \
                  "In, Out types must match for in place operations");         \
                                                                               \
  public:                                                                      \
    InOut<VectorList<T, poplar::VectorListLayout::LAYOUT>> inOut;              \
    IS_EXTERNAL_CODELET(true);                                                 \
    void compute() {                                                           \
      using arch =                                                             \
          typename popops::UnaryOpFn<op, T, architecture::active>::arch;       \
      unsigned limI = inOut.size();                                            \
      for (unsigned i = 0; i != limI; ++i) {                                   \
        popops::UnaryOpDispatch<op, T, outputType, arch>::compute(             \
            inOut[i].size(), &inOut[i][0], &inOut[i][0]);                      \
      }                                                                        \
    }                                                                          \
  };

#if defined(VECTORLIST_AVAIL_DELTAN)
DEFINE_UNARY_OP_NL_2D(expr::UnaryOpType::TANH, DELTAN)
DEFINE_UNARY_OP_NL_2D(expr::UnaryOpType::SIGMOID, DELTAN)
DEFINE_UNARY_OP_NL_2D(expr::UnaryOpType::RELU, DELTAN)
#else
DEFINE_UNARY_OP_NL_2D(expr::UnaryOpType::TANH, DELTANELEMENTS)
DEFINE_UNARY_OP_NL_2D(expr::UnaryOpType::SIGMOID, DELTANELEMENTS)
DEFINE_UNARY_OP_NL_2D(expr::UnaryOpType::RELU, DELTANELEMENTS)
#endif

//******************************************************************************
// Dispatch for use with Unary Operation MultiVertex
//******************************************************************************
template <expr::UnaryOpType op, typename inT, typename outT, typename A>
struct UnaryOpDispatchMultiVertex {
public:
  static void compute(unsigned size, unsigned worker, const inT *in,
                      outT *out) {
    // No vectorisation for int, unsigned int, but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = UnaryOpFn<op, inT, A>::fn(in[j]);
  }
};

#ifdef __IPU__

/// Processing for operators that return a bool (LOGICAL_NOT) for MultiVertex.
template <UnaryOpType op, typename T, typename A>
class UnaryOpDispatchMultiVertex<op, T, bool, A> {
public:
  static void compute(unsigned size, unsigned worker, const T *in,
                      __attribute__((align_value(4))) bool *out) {
    const rptsize_t loopCount = divideWork(size, 2, worker);
    unaryBoolOpBulk<op, T, CTXT_WORKERS>::compute(
        loopCount, in + 4 * worker, reinterpret_cast<int *>(out) + worker);

    // To process the trailing elements (if any) use the last worker as it is
    // most likely to have less to do than the others.
    if (worker == (CTXT_WORKERS - 1)) {
      unaryBoolOpRemainder<op, T>(size, in, out);
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatchMultiVertex<op, half, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) bool *out) {

    const half4 *h4In = reinterpret_cast<const half4 *>(in) + worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;
    const rptsize_t loopCount = divideWork(size, 2, worker);

    for (unsigned j = 0; j < loopCount; j++) {
      half4 load = ipu::load_postinc(&h4In, CTXT_WORKERS);
      short4 calc = static_cast<short4>(FuncTy<half4>::fn(load));
      char4 result = tochar4(calc);
      *iOut = copy_cast<int>(result) & 0x01010101;
      iOut += CTXT_WORKERS;
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if (worker == (CTXT_WORKERS - 1) && remainder) {
      in = &in[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        *out++ = FuncTy<half>::fn(*in++);
      }
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatchMultiVertex<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) bool *out) {

    const float2 *f2In = reinterpret_cast<const float2 *>(in) + 2 * worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;

    const rptsize_t loopCount = divideWork(size, 2, worker);
    for (unsigned j = 0; j < loopCount; j++) {
      float2 load = ipu::load_postinc(&f2In, 1);
      int2 calc_lo = static_cast<int2>(FuncTy<float2>::fn(load));
      load = ipu::load_postinc(&f2In, 2 * CTXT_WORKERS - 1);
      int2 calc_hi = static_cast<int2>(FuncTy<float2>::fn(load));

      char4 result = tochar4(calc_lo, calc_hi);
      *iOut = copy_cast<int>(result) & 0x01010101;
      iOut += CTXT_WORKERS;
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if (worker == (CTXT_WORKERS - 1) && remainder) {
      in = &in[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        *out++ = FuncTy<float>::fn(*in++);
      }
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatchMultiVertex<op, half, half, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, const half *in,
                      typename UnaryOpOutputType<op, half>::type *out) {

    const half4 *h4In = reinterpret_cast<const half4 *>(in) + worker;
    half4 *h4Out = reinterpret_cast<half4 *>(out) + worker;
    const auto remainder = size & 3;
    const rptsize_t loopCount = divideWork(size, 2, worker);

    if constexpr (hasInlineAssemblerInnerLoopImpl<op, half, half>()) {
      std::tie(h4In, h4Out) =
          inlineAssemblerUnaryOp<op, half, CTXT_WORKERS>::loopBody(loopCount,
                                                                   h4In, h4Out);
    } else {
      for (unsigned i = 0; i < loopCount; i++) {
        *h4Out = UnaryOpFn<op, half4, architecture::ipu>::fn(*h4In);
        h4In += CTXT_WORKERS;
        h4Out += CTXT_WORKERS;
      }
    }
    if (remainder) {
      const half2 *h2In = reinterpret_cast<const half2 *>(h4In);
      half2 *h2Out = reinterpret_cast<half2 *>(h4Out);
      if (size & 2) {
        if (h4Out == (half4 *)&out[size & (~3)]) {
          *h2Out++ = UnaryOpFn<op, half2, architecture::ipu>::fn(*h2In++);
        }
      }
      assert(size != 0);
      if (h2Out == (half2 *)&out[size - 1]) {
        write16Aligned32(UnaryOpFn<op, half, architecture::ipu>::fn((*h2In)[0]),
                         h2Out);
      }
    }
  }
};

template <expr::UnaryOpType op>
class UnaryOpDispatchMultiVertex<op, float, float, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, const float *in,
                      typename UnaryOpOutputType<op, float>::type *out) {

    const float2 *f2In = reinterpret_cast<const float2 *>(in) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;

    const rptsize_t loopCount = divideWork(size, 1, worker);

    if constexpr (hasInlineAssemblerInnerLoopImpl<op, float, float>()) {
      inlineAssemblerUnaryOp<op, float, CTXT_WORKERS>::loopBody(loopCount, f2In,
                                                                f2Out);
    } else {
      if (loopCount) {
        // Partially unroll the loop, carefully avoiding overreads (which will
        // be strided).
        // Can this be done automatically? See T59317
        auto load = *f2In;
        f2In += CTXT_WORKERS;
        for (unsigned j = 0; j < rptsize_t(loopCount - 1); j++) {
          auto calc = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
          load = *f2In;
          f2In += CTXT_WORKERS;
          *f2Out = calc;
          f2Out += CTXT_WORKERS;
        }
        *f2Out = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
      }
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if (worker == (CTXT_WORKERS - 1) && (size & 1)) {
      out[size - 1] = UnaryOpFn<op, float, architecture::ipu>::fn(in[size - 1]);
    }
  }
};

template <UnaryOpType op>
class UnaryOpDispatchMultiVertex<op, short, short, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(4))) short *in,
                      __attribute__((align_value(4))) short *out) {
    unaryShort2_MultiVertex<op, short>(size, worker, in, out);
  }
};

template <UnaryOpType op>
class UnaryOpDispatchMultiVertex<op, unsigned short, unsigned short,
                                 architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(4))) unsigned short *in,
                      __attribute__((align_value(4))) unsigned short *out) {
    unaryShort2_MultiVertex<op, unsigned short>(size, worker, in, out);
  }
};

#endif

template <expr::UnaryOpType op, typename T>
class UnaryOp1D : public MultiVertex {
  typedef typename UnaryOpOutputType<op, T>::type outputType;

  static const bool needsAlignWorkers = false;

public:
  Input<Vector<T, ONE_PTR, 8>> in;
  Output<Vector<outputType, SPAN, 8>> out;

  IS_EXTERNAL_CODELET((isExternal<op, T>()));

  void compute(unsigned wid) {
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    popops::UnaryOpDispatchMultiVertex<op, T, outputType, arch>::compute(
        out.size(), wid, &in[0], &out[0]);
  }
};

template <expr::UnaryOpType op, typename T>
class UnaryOp1DInPlace : public MultiVertex {
  typedef typename UnaryOpOutputType<op, T>::type outputType;
  static_assert(std::is_same<T, outputType>::value,
                "In, Out types must match for in place operations");

  static const bool needsAlignWorkers = false;

public:
  InOut<Vector<T, SPAN, 8>> inOut;

  IS_EXTERNAL_CODELET((isExternal<op, T>()));

  void compute(unsigned wid) {
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    popops::UnaryOpDispatchMultiVertex<op, T, outputType, arch>::compute(
        inOut.size(), wid, &inOut[0], &inOut[0]);
  }
};

// The *in-place* vertices for the 3 non-linearity operators TANH, SIGMOID, RELU
// have their vertex state fields defined differently from the rest of the
// operators, so require separate templates.
#define DEFINE_UNARY_OP_NL_SV(op, PTR_TYPE)                                    \
  template <typename T> class UnaryOp1DInPlace<op, T> : public MultiVertex {   \
    typedef typename UnaryOpOutputType<op, T>::type outputType;                \
    static_assert(std::is_same<T, outputType>::value,                          \
                  "In, Out types must match for in place operations");         \
                                                                               \
    static const bool needsAlignWorkers = false;                               \
                                                                               \
  public:                                                                      \
    UnaryOp1DInPlace();                                                        \
    InOut<Vector<T, PTR_TYPE, 4>> inOut;                                       \
    const unsigned short n;                                                    \
    IS_EXTERNAL_CODELET(true);                                                 \
    void compute(unsigned wid) {                                               \
      using arch =                                                             \
          typename popops::UnaryOpFn<op, T, architecture::active>::arch;       \
      popops::UnaryOpDispatchMultiVertex<op, T, outputType, arch>::compute(    \
          n, wid, &inOut[0], &inOut[0]);                                       \
    }                                                                          \
  };

#ifdef VECTOR_AVAIL_SCALED_PTR32
DEFINE_UNARY_OP_NL_SV(expr::UnaryOpType::TANH, SCALED_PTR32)
DEFINE_UNARY_OP_NL_SV(expr::UnaryOpType::SIGMOID, SCALED_PTR32)
DEFINE_UNARY_OP_NL_SV(expr::UnaryOpType::RELU, SCALED_PTR32)
#else
DEFINE_UNARY_OP_NL_SV(expr::UnaryOpType::TANH, ONE_PTR)
DEFINE_UNARY_OP_NL_SV(expr::UnaryOpType::SIGMOID, ONE_PTR)
DEFINE_UNARY_OP_NL_SV(expr::UnaryOpType::RELU, ONE_PTR)
#endif

INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ABSOLUTE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ASIN, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::BITWISE_NOT, int, unsigned, short,
               unsigned short, long long, unsigned long long)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::CBRT, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::COUNT_LEADING_ZEROS, int, unsigned)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ERF, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::GELU_ERF, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::EXPONENT2, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::IS_INF, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::IS_NAN, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::NEGATE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::TAN, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::RELU, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::TRUNC, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SQUARE, float, half, int, unsigned)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::RSQRT, float, half)

INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ABSOLUTE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ASIN, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::BITWISE_NOT, int, unsigned, short,
               unsigned short, long long, unsigned long long)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::CBRT, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::COUNT_LEADING_ZEROS, int, unsigned)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ERF, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::GELU_ERF, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::EXPONENT2, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::IS_INF, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::IS_NAN, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::NEGATE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::TAN, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::RELU, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::TRUNC, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SQUARE, float, half, int, unsigned)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::RSQRT, float, half)

INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ABSOLUTE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ASIN, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::BITWISE_NOT, int, unsigned,
               short, unsigned short, long long, unsigned long long)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::CBRT, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::COUNT_LEADING_ZEROS, int,
               unsigned)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ERF, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::GELU_ERF, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::EXPONENT_MINUS_ONE, float,
               half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::EXPONENT2, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float,
               half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::NEGATE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::TAN, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::RELU, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::TRUNC, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SQUARE, float, half, int,
               unsigned)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::RSQRT, float, half)

INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ABSOLUTE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ASIN, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::BITWISE_NOT, int, unsigned,
               short, unsigned short, long long, unsigned long long)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::CBRT, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::COUNT_LEADING_ZEROS, int,
               unsigned)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ERF, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::GELU_ERF, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::EXPONENT_MINUS_ONE, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::EXPONENT2, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::NEGATE, float, half, int,
               long long)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::TAN, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::RELU, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::TRUNC, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SQUARE, float, half, int,
               unsigned)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::RSQRT, float, half)
} // namespace popops
