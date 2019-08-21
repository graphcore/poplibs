// Copyright (c) 2019, Graphcore Ltd, All rights reserved.
#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

#include <cassert>
#include <cstring>
#include <cmath>

#include "util.hpp"
#include "popops/ExprOp.hpp"
#include "popops/elementwiseCodelets.hpp"

namespace popops {

namespace {

  // certain operators need to call a different function depending on the input
  // type. this deals with dispatching unary ops to their correct functions.
  template <expr::UnaryOpType Op>
  struct UnaryLibCall {};

  template <>
  struct UnaryLibCall<expr::UnaryOpType::ABSOLUTE> {
#ifdef __IPU__
    template <typename FPType>
    FPType operator()(FPType x) const {
      return ipu::fabs(x);
    }
#endif

    int operator()(int x) const {
      return std::abs(x);
    }
  };

  template <>
  struct UnaryLibCall<expr::UnaryOpType::SQRT> {
#ifdef __IPU__
    template <typename FPType>
    FPType operator()(FPType x) const {
      return ipu::sqrt(x);
    }
#endif

    int operator()(int x) const {
      return std::sqrt(x);
    }
  };

  // Structure with template specialization to define the output type
  // of a unary operation
  template <expr::UnaryOpType op, typename T>
  struct UnaryOpOutputType { using type = T; };

#ifndef __IPU__
#define DEFINE_UNARY_OUTPUT_TYPE_BOOL(op)                                      \
  template <typename T>                                                        \
  struct UnaryOpOutputType<op, T> {                                            \
    using type = bool;                                                         \
  };
#else
#define DEFINE_UNARY_OUTPUT_TYPE_BOOL(op)                                      \
  template <typename T>                                                        \
  struct UnaryOpOutputType<op, T> {                                            \
    using type = bool;                                                         \
  };                                                                           \
  template <>                                                                  \
  struct UnaryOpOutputType<op, float2> {                                       \
    using type = long2;                                                        \
  };                                                                           \
  template <>                                                                  \
  struct UnaryOpOutputType<op, half4> {                                        \
    using type = short4;                                                       \
  };
#endif

DEFINE_UNARY_OUTPUT_TYPE_BOOL(expr::UnaryOpType::IS_FINITE)
DEFINE_UNARY_OUTPUT_TYPE_BOOL(expr::UnaryOpType::IS_INF)
DEFINE_UNARY_OUTPUT_TYPE_BOOL(expr::UnaryOpType::IS_NAN)

  // Structure with template specialization to define the function
  // that performes that operation on one element
  template <expr::UnaryOpType op, typename T, typename A>
  struct UnaryOpFn {};

  template <expr::UnaryOpType op, typename T>
  using UnaryOpOutputType_t = typename UnaryOpOutputType<op, T>::type;

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

#define UNARY_VARGS(_1, _2, _3, N, ...) DEFINE_UNARY_OP_FN ## N
#define DEFINE_UNARY_OP_FN(...) \
  UNARY_VARGS(__VA_ARGS__, _2, _1, _)(__VA_ARGS__)

// helper macro for the common case of just a different namespace
#define DEFINE_UNARY_OP_FN_STD(op, fn)                                         \
  DEFINE_UNARY_OP_FN(op, return std::fn(PromoteHalfsToFloats(x));,             \
                         return ipu::fn(x);)

DEFINE_UNARY_OP_FN(expr::UnaryOpType::ABSOLUTE,
                   if (std::is_integral<T>::value) {
                     // This path is avoided for half specialisation, but we
                     // promote half to float to avoid promotion error.
                     return std::abs(PromoteHalfsToFloats(x));
                   } else {
                     return std::fabs(PromoteHalfsToFloats(x));
                   },
                   return UnaryLibCall<expr::UnaryOpType::ABSOLUTE>{}(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::BITWISE_NOT, return ~x;)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::CEIL, ceil)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::COS, cos)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::COUNT_LEADING_ZEROS,
                   return x ? __builtin_clz(x) : 32;)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::EXPONENT, exp)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::EXPONENT_MINUS_ONE, expm1)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::FLOOR, floor)

DEFINE_UNARY_OP_FN(expr::UnaryOpType::INVERSE, return 1 / x;)

#ifdef __IPU__
template <typename T>
auto isinf(T a) -> decltype(a == INFINITY || a == -INFINITY) {
  return a == INFINITY || a == -INFINITY;
}
template <typename T>
auto isninf(T a) -> decltype(a != INFINITY && a != -INFINITY) {
  return a != INFINITY && a != -INFINITY;
}
#endif
DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_INF,
                  return std::isinf(PromoteHalfsToFloats(x));,
                  return isinf(PromoteHalfsToFloats(x));)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_FINITE,
                  return std::isfinite(PromoteHalfsToFloats(x));,
                  return x == x && isninf(PromoteHalfsToFloats(x));)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_NAN, return x != x;)

DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::LOGARITHM, log)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::LOGARITHM_ONE_PLUS, log1p)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::LOGICAL_NOT, return !x;)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::NEGATE, return -x;)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::POPCOUNT,
                   return __builtin_popcount(x);)
#ifdef __IPU__
// comparison functions that return 1/0 in the arg type rather than true/false
template<typename T>
typename std::enable_if<!isVectorType<T>::value, T>::type
compareLT(T a, T b) {
  return a < b;
}
template<typename T>
typename std::enable_if<isVectorType<T>::value, T>::type
compareLT(T a, T b) {
  T r;
  unsigned n = sizeof(a) / sizeof(a[0]);
  for (unsigned i = 0u; i < n; ++i)
    r[i] = a[i] < b[i];
  return r;
}

#endif
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIGNUM,
                   return (0 < x) - (x < 0);,
return compareLT(decltype(x){0}, x) - compareLT(x, decltype(x){0});)

DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::SIN, sin)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::TANH, tanh)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::ROUND, round)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQRT,
                   return std::sqrt(PromoteHalfsToFloats(x));,
                   return UnaryLibCall<expr::UnaryOpType::SQRT>{}(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQUARE,
                   return (x * x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIGMOID,
                   using Ty = decltype(PromoteHalfsToFloats(x));
                   return Ty(1) / (Ty(1) + std::exp(-Ty(x)));,
                   return ipu::sigmoid(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::RSQRT,
                   using Ty = decltype(PromoteHalfsToFloats(x));
                   return Ty(1) / std::sqrt(Ty(x));,
                   return ipu::rsqrt(x);)

} // namespace

template <expr::UnaryOpType op, typename inT, typename outT, typename A>
struct UnaryOpDispatch {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) inT *in,
                      __attribute__((align_value(8))) outT *out) {

    for (unsigned j = 0; j != size; ++j) {
      out[j] = UnaryOpFn<op, inT, A>::fn(in[j]);
    }
  }
};

#ifdef __IPU__

template <expr::UnaryOpType op>
struct UnaryOpDispatch<op, half, bool, architecture::ipu> {

  template<class T>
  using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const half4 *h4In = reinterpret_cast<const half4 *>(in);
      int *iOut = reinterpret_cast<int *>(out);
      const unsigned loopCount = maskForRepeat(size / 4u);
      for (unsigned i = 0; i < loopCount; ++i) {
        half4 load = ipu::load_postinc(&h4In, 1);
        short4 calc = static_cast<short4>(FuncTy<half4>::fn(load));
        char4 result = tochar4(calc);
        int ires = copy_cast<int>(result) & 0x01010101;
        ipu::store_postinc(&iOut, ires, 1);
      }
      in = reinterpret_cast<const half *>(h4In);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      half load = ipu::load_postinc(&in, 1);
      *out++ = FuncTy<half>::fn(load);
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatch<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template<class T>
  using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const float2 *f2In = reinterpret_cast<const float2 *>(in);
      int *iOut = reinterpret_cast<int *>(out);

      const unsigned loopCount = maskForRepeat(size / 4u);
      for (unsigned i = 0; i < loopCount; ++i) {
        float2 load = ipu::load_postinc(&f2In, 1);
        long2 calc_lo = static_cast<long2>(FuncTy<float2>::fn(load));

        load = ipu::load_postinc(&f2In, 1);
        long2 calc_hi = static_cast<long2>(FuncTy<float2>::fn(load));

        char4 result = tochar4(calc_lo, calc_hi);
        int ires = copy_cast<int>(result) & 0x01010101;
        ipu::store_postinc(&iOut, ires, 1);
      }
      in = reinterpret_cast<const float *>(f2In);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      float load = ipu::load_postinc(&in, 1);
      *out++ = FuncTy<float>::fn(load);
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

    if (size >= 4) {
      const half4 *h4In = reinterpret_cast<const half4 *>(in);
      half4 *h4Out = reinterpret_cast<half4 *>(out);

      // LLVM currently chooses to rotate the loop in a way that is not optimal
      // for our hardware. The inline asm blocks this. The loop is pipelined
      // sufficiently to overlap load with calculation. This was used a it seems
      // a reasonable compromise over zero overlap and unrolling far enough to
      // overlap the store with calculation.

      half4 load = ipu::load_postinc(&h4In, 1);
      const unsigned loopCount = maskForRepeat((size / 4u) - 1u);
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < loopCount; ++i) {
        half4 calc = UnaryOpFn<op, half4, arch>::fn(load);
        load = ipu::load_postinc(&h4In, 1);
        ipu::store_postinc(&h4Out, calc, 1);
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&h4Out, UnaryOpFn<op, half4, arch>::fn(load), 1);

      in = reinterpret_cast<const half *>(h4In);
      half *tmp = reinterpret_cast<half *>(h4Out);
      size -= (tmp - out);
      out = tmp;
    }

    const half2 *h2In = reinterpret_cast<const half2 *>(in);
    half2 *h2Out = reinterpret_cast<half2 *>(out);

    if (size >= 2) {
      ipu::store_postinc(
          &h2Out, UnaryOpFn<op, half2, arch>::fn(ipu::load_postinc(&h2In, 1)),
          1);
      size -= 2;
    }

    if (size == 1) {
      half2 res = (half2){
          UnaryOpFn<op, half, arch>::fn((*h2In)[0]),
          (*h2Out)[1],
      };
      *h2Out = res;
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
    if (size >= 2) {
      const unsigned loopCount = maskForRepeat((size / 2u) - 1);

      float2 load = *f2In++;
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for(unsigned j = 0; j < loopCount ; j ++) {
        float2 calc = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
        load = *f2In++;
        *f2Out++ = calc;
       }
       asm volatile("# Thwart loop rotation (end)" ::: "memory");
       *f2Out++ = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
    }
    if(size & 1) {
      out[size-1] = UnaryOpFn<op,float,architecture::ipu>::fn(in[size-1]);
    }
  }
};
#endif

template <expr::UnaryOpType op, typename T>
class
UnaryOp2D : public Vertex {
typedef typename UnaryOpOutputType<op, T>::type outputType;
public:
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in;
  Vector<Output<Vector<outputType, SPAN, 8>>> out;

  bool compute() {
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    unsigned limI = out.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::UnaryOpDispatch<op, T, outputType, arch>
                          ::compute(out[i].size(), &in[i][0], &out[i][0]);
    }
    return true;
  }
};

template <expr::UnaryOpType op, typename T>
class
UnaryOp2DInPlace : public Vertex {
typedef typename UnaryOpOutputType<op, T>::type outputType;
static_assert(std::is_same<T, outputType>::value,
        "In, Out types must match for in place operations");

public:
  Vector<InOut<Vector<T, SPAN, 8>>> inOut;

  bool compute() {
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    unsigned limI = inOut.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::UnaryOpDispatch<op, T, outputType, arch>
                      ::compute(inOut[i].size(), &inOut[i][0], &inOut[i][0]);
    }
    return true;
  }
};
//******************************************************************************
// Dispatch for use with Unary Operation supervisor vertices
//******************************************************************************
template <expr::UnaryOpType op, typename inT, typename outT, typename A>
struct UnaryOpDispatchSupervisor {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      inT *in,
                      outT *out) {
    // No vectorisation for int, unsigned int, but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = UnaryOpFn<op, inT, architecture::generic>::fn(in[j]);
  }
};

#ifdef __IPU__

template <expr::UnaryOpType op>
struct UnaryOpDispatchSupervisor<op, half, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template<class T>
  using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      unsigned worker,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8))) bool *out) {

    const half4 *h4In = reinterpret_cast<const half4 *>(in) + worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;
    const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
    for(unsigned j = 0; j < loopCount; j ++) {
      half4 load = ipu::load_postinc(&h4In, CTXT_WORKERS);
      short4 calc = static_cast<short4>(FuncTy<half4>::fn(load));
      char4 result = tochar4(calc);
      int ires = copy_cast<int>(result) & 0x01010101;
      ipu::store_postinc(&iOut, ires, CTXT_WORKERS);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if(worker == (CTXT_WORKERS - 1)  && remainder ) {
      in = &in[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        half load = ipu::load_postinc(&in, 1);
        *out++ = FuncTy<half>::fn(load);
      }
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatchSupervisor<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template<class T>
  using FuncTy = UnaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      unsigned worker,
                      const __attribute__((align_value(8))) float *in,
                      __attribute__((align_value(8))) bool *out) {

    const float2 *f2In = reinterpret_cast<const float2 *>(in) + 2 * worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;

    const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
    for(unsigned j = 0; j < loopCount; j++) {
      float2 load = ipu::load_postinc(&f2In, 1);
      long2 calc_lo = static_cast<long2>(FuncTy<float2>::fn(load));

      load = ipu::load_postinc(&f2In, 2 * CTXT_WORKERS - 1);
      long2 calc_hi = static_cast<long2>(FuncTy<float2>::fn(load));

      char4 result = tochar4(calc_lo, calc_hi);
      int ires = copy_cast<int>(result) & 0x01010101;
      ipu::store_postinc(&iOut, ires, CTXT_WORKERS);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if(worker == (CTXT_WORKERS - 1)  && remainder ) {
      in = &in[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        float load = ipu::load_postinc(&in, 1);
        *out++ = FuncTy<float>::fn(load);
      }
    }
  }
};

template <expr::UnaryOpType op>
struct UnaryOpDispatchSupervisor<op, half, half, architecture::ipu> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      half *in,
                       typename UnaryOpOutputType<op, half>::type *out) {

    const half4 *h4In = reinterpret_cast<const half4 *>(in) + worker;
    half4 *h4Out = reinterpret_cast<half4 *>(out) + worker;

    const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
    asm volatile ("# Thwart loop rotation (start)" ::: "memory");
    for (unsigned i = 0; i < loopCount; i++) {
      half4 load = ipu::load_postinc(&h4In, CTXT_WORKERS);
      half4 calc = UnaryOpFn<op, half4,architecture::ipu>::fn(load);
      ipu::store_postinc(&h4Out, calc, CTXT_WORKERS);
    }
    asm volatile ("# Thwart loop rotation (end)" ::: "memory");
    if(size & 3) {
      const half2 *h2In = reinterpret_cast<const half2*>(h4In);
      half2 *h2Out = reinterpret_cast<half2 *>(h4Out);
      if(size & 2) {
        if(h4Out == (half4*)&out[size & (~3)]) {
          ipu::store_postinc(&h2Out,
                              UnaryOpFn<op, half2, architecture::ipu>::fn(
                              ipu::load_postinc(&h2In, 1)),
                              1);
        }
      }
      assert(size != 0);
      if(h2Out == (half2*)&out[size-1]) {
        half2 res = (half2)
        {
          UnaryOpFn<op, half, architecture::ipu>::fn((*h2In)[0]), (*h2Out)[1],
        };
        *h2Out = res;
      }
    }
  }
};

template <expr::UnaryOpType op>
class UnaryOpDispatchSupervisor<op, float, float, architecture::ipu> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      float *in,
                      typename UnaryOpOutputType<op, float>::type *out) {

    const float2 *f2In = reinterpret_cast<const float2 *>(in) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;

    const unsigned loopCount = maskForRepeat(divideWork(size, 1, worker));
    for(unsigned j = 0; j < loopCount; j ++) {
      float2 load = ipu::load_postinc(&f2In, CTXT_WORKERS);
      float2 calc = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
      ipu::store_postinc(&f2Out, calc, CTXT_WORKERS);
     }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if(worker == (CTXT_WORKERS - 1)  && (size & 1)) {
      out[size-1] = UnaryOpFn<op,float,architecture::ipu>::fn(in[size-1]);
    }
  }
};
#endif

template <expr::UnaryOpType op, typename T>
class
UnaryOp1DSupervisor : public SupervisorVertex {
typedef typename UnaryOpOutputType<op, T>::type outputType;

public:
  Input<Vector<T, ONE_PTR, 8>> in;
  Output<Vector<outputType, SPAN, 8>> out;

  IS_EXTERNAL_CODELET(!(std::is_same<T, bool>::value));
  bool compute() {
    for (unsigned j = 0; j != out.size(); ++j) {
      out[j] = UnaryOpFn<op, T,  architecture::generic>::fn(in[j]);
    }
    return true;
  }
};
template <expr::UnaryOpType op, typename T>
class
UnaryOp1DInPlaceSupervisor : public SupervisorVertex {
typedef typename UnaryOpOutputType<op, T>::type outputType;
static_assert(std::is_same<T, outputType>::value,
        "In, Out types must match for in place operations");

public:
  InOut<Vector<T, SPAN, 8>> inOut;

  IS_EXTERNAL_CODELET(!(std::is_same<outputType, bool>::value));
  bool compute() {
    for (unsigned j = 0; j != inOut.size(); ++j) {
      inOut[j] = UnaryOpFn<op, T, architecture::generic>::fn(inOut[j]);
    }
    return true;
  }
};

//******************************************************************************
// Worker vertex to actually do the work of the operation for the
// UnaryOp1DSupervisor vertex when it is an external codelet
//******************************************************************************
template <expr::UnaryOpType op, typename T>
class
UnaryOp1D : public Vertex {
typedef typename UnaryOpOutputType<op, T>::type outputType;
public:
  Input<Vector<T, ONE_PTR, 8>> in;
  Output<Vector<outputType, SPAN, 8>> out;

  bool compute() {
#ifdef __IPU__
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    popops::UnaryOpDispatchSupervisor<op, T, outputType, arch>
                                      ::compute(out.size(),
      getWsr(), &in[0], &out[0]);
#endif
    return true;
  }
};

//******************************************************************************
// Worker vertex to actually do the work of the operation for the
// UnaryOp1DInPlaceSupervisor vertex when it is an external codelet
//******************************************************************************
template <expr::UnaryOpType op, typename T>
class
UnaryOp1DInPlace : public SupervisorVertex {
typedef typename UnaryOpOutputType<op, T>::type outputType;
static_assert(std::is_same<T, outputType>::value,
        "In, Out types must match for in place operations");

public:
  InOut<Vector<T, SPAN, 8>> inOut;

  bool compute() {
#ifdef __IPU__
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    popops::UnaryOpDispatchSupervisor<op, T, outputType, arch>
                                      ::compute(inOut.size(),
      getWsr(), &inOut[0], &inOut[0]);
#endif
    return true;
  }
};

INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::BITWISE_NOT, int, unsigned)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::COUNT_LEADING_ZEROS, int, unsigned)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::IS_INF, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::IS_NAN, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SQUARE, float, half, int, unsigned)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::RSQRT, float, half)

// UnaryOp1DSupervisor - supervisor stubs for all types except bool.  If bool
// they will generate single worker code. See T4642 - a task to add
// these.
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::ABSOLUTE, float, half,
               int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::BITWISE_NOT, int,
                                                                      unsigned)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::COUNT_LEADING_ZEROS, int,
                                                                      unsigned)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::EXPONENT_MINUS_ONE,
               float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::IS_INF, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::IS_NAN, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::LOGARITHM_ONE_PLUS,
               float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::POPCOUNT, int,
                                                                  unsigned)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SQUARE, float, half, int,
                                                                    unsigned)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::RSQRT, float, half)

// UnaryOp1D - worker vertex for all types except bool.
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::BITWISE_NOT, int, unsigned)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::COUNT_LEADING_ZEROS, int, unsigned)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::IS_INF, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::IS_NAN, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SQUARE, float, half, int, unsigned)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::RSQRT, float, half)



INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::BITWISE_NOT, int, unsigned)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::COUNT_LEADING_ZEROS, int,
                                                                    unsigned)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::EXPONENT_MINUS_ONE, float,
                                                                        half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float,
                                                                        half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SQUARE, float, half, int,
                                                                    unsigned)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::RSQRT, float, half)

// UnaryOp1DInPlaceSupervisor - supervisor stubs for all types except bool.
// If bool they will generate single worker code. See T4642 - a task to add
// these.

INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::ABSOLUTE, float,
               half, int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::BITWISE_NOT, int,
                                                              unsigned)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor,
               expr::UnaryOpType::COUNT_LEADING_ZEROS, int, unsigned)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::EXPONENT, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor,
               expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor,
               expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor,
               expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::LOGARITHM, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor,
               expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::NEGATE, float,
               half, int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::POPCOUNT, int,
                                                                unsigned)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::SIGNUM, float,
               half, int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::TANH, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::ROUND, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::SQRT, float, half,
               int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::SQUARE, float,
               half, int, unsigned)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::SIGMOID, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::RSQRT, float,
               half)

// UnaryOp1DInPlace - worker vertex for all types except bool.

INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::BITWISE_NOT, int, unsigned)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::COUNT_LEADING_ZEROS, int,
                                                                    unsigned)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::EXPONENT_MINUS_ONE, float,
                                                                        half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::INVERSE, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float,
                                                                        half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::POPCOUNT, int, unsigned)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SQUARE, float, half, int,
                                                                unsigned)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIGMOID, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::RSQRT, float, half)

}
