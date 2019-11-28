// Copyright (c) 2019, Graphcore Ltd, All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <cstring>

#include "popops/ExprOp.hpp"
#include "popops/elementwiseCodelets.hpp"
#include "util.hpp"

namespace popops {

namespace {

// certain operators need to call a different function depending on the input
// type. this deals with dispatching binary ops to their correct functions.
template <expr::BinaryOpType Op> struct BinaryLibCall {};

template <> struct BinaryLibCall<expr::BinaryOpType::MAXIMUM> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x, FPType y) const {
    return ipu::fmax(x, y);
  }
#endif

  int operator()(int x, int y) const { return max(x, y); }
  unsigned operator()(unsigned x, unsigned y) const { return max(x, y); }
};

template <> struct BinaryLibCall<expr::BinaryOpType::MINIMUM> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x, FPType y) const {
    return ipu::fmin(x, y);
  }
#endif

  int operator()(int x, int y) const { return min(x, y); }
  unsigned operator()(unsigned x, unsigned y) const { return min(x, y); }
};

template <> struct BinaryLibCall<expr::BinaryOpType::REMAINDER> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x, FPType y) const {
    return ipu::fmod(x, y);
  }
#endif
  int operator()(int x, int y) const {
    int r = x / y;
    return x - r * y;
  }
  unsigned operator()(unsigned x, unsigned y) const {
    unsigned r = x / y;
    return x - r * y;
  }
  float operator()(float x, float y) const { return fmod(x, y); }
};

// Structure with template specialization to define the output type
// of a binary operation
template <expr::BinaryOpType op, typename T> struct BinaryOpOutputType {
  using type = T;
};
#ifdef __IPU__
template <>
struct BinaryOpOutputType<expr::BinaryOpType::GREATER_THAN, float2> {
  using type = long2;
};
template <> struct BinaryOpOutputType<expr::BinaryOpType::GREATER_THAN, half4> {
  using type = short4;
};
template <>
struct BinaryOpOutputType<expr::BinaryOpType::GREATER_THAN_EQUAL, float2> {
  using type = long2;
};
template <>
struct BinaryOpOutputType<expr::BinaryOpType::GREATER_THAN_EQUAL, half4> {
  using type = short4;
};
template <> struct BinaryOpOutputType<expr::BinaryOpType::LESS_THAN, float2> {
  using type = long2;
};
template <> struct BinaryOpOutputType<expr::BinaryOpType::LESS_THAN, half4> {
  using type = short4;
};
template <>
struct BinaryOpOutputType<expr::BinaryOpType::LESS_THAN_EQUAL, float2> {
  using type = long2;
};
template <>
struct BinaryOpOutputType<expr::BinaryOpType::LESS_THAN_EQUAL, half4> {
  using type = short4;
};
template <> struct BinaryOpOutputType<expr::BinaryOpType::EQUAL, float2> {
  using type = long2;
};
template <> struct BinaryOpOutputType<expr::BinaryOpType::EQUAL, half4> {
  using type = short4;
};
template <> struct BinaryOpOutputType<expr::BinaryOpType::NOT_EQUAL, float2> {
  using type = long2;
};
template <> struct BinaryOpOutputType<expr::BinaryOpType::NOT_EQUAL, half4> {
  using type = short4;
};
#endif

template <typename T>
struct BinaryOpOutputType<expr::BinaryOpType::GREATER_THAN, T> {
  using type = bool;
};

template <typename T>
struct BinaryOpOutputType<expr::BinaryOpType::GREATER_THAN_EQUAL, T> {
  using type = bool;
};

template <typename T>
struct BinaryOpOutputType<expr::BinaryOpType::LESS_THAN, T> {
  using type = bool;
};

template <typename T>
struct BinaryOpOutputType<expr::BinaryOpType::LESS_THAN_EQUAL, T> {
  using type = bool;
};
template <typename T> struct BinaryOpOutputType<expr::BinaryOpType::EQUAL, T> {
  using type = bool;
};
template <typename T>
struct BinaryOpOutputType<expr::BinaryOpType::NOT_EQUAL, T> {
  using type = bool;
};

// Structure with template specialization to define the function
// that performes that operation on scalar elements
template <expr::BinaryOpType op, typename T, typename A> struct BinaryOpFn {};

template <expr::BinaryOpType op, typename T>
using BinaryOpOutputType_t = typename BinaryOpOutputType<op, T>::type;

#define DEFINE_BINARY_OP_FN_GEN(op, body)                                      \
  template <typename T, typename A> struct BinaryOpFn<op, T, A> {              \
    using arch = architecture::generic;                                        \
    static BinaryOpOutputType_t<op, T> fn(T x, T y) { body }                   \
  };

#ifdef __IPU__
#define DEFINE_BINARY_OP_FN_IPU(op, body)                                      \
  template <typename T> struct BinaryOpFn<op, T, architecture::ipu> {          \
    using arch = architecture::ipu;                                            \
    static BinaryOpOutputType_t<op, T> fn(T x, T y) { body }                   \
  };

#define DEFINE_BINARY_OP_FN_1(op, body)                                        \
  DEFINE_BINARY_OP_FN_GEN(op, body)                                            \
  DEFINE_BINARY_OP_FN_IPU(op, body)

#define DEFINE_BINARY_OP_FN_2(op, body, ipubody)                               \
  DEFINE_BINARY_OP_FN_GEN(op, body)                                            \
  DEFINE_BINARY_OP_FN_IPU(op, ipubody)

#else
#define DEFINE_BINARY_OP_FN_1(op, body) DEFINE_BINARY_OP_FN_GEN(op, body)
#define DEFINE_BINARY_OP_FN_2(op, body, _) DEFINE_BINARY_OP_FN_GEN(op, body)
#endif

#define BINARY_VARGS(_1, _2, _3, N, ...) DEFINE_BINARY_OP_FN##N
#define DEFINE_BINARY_OP_FN(...)                                               \
  BINARY_VARGS(__VA_ARGS__, _2, _1, _)(__VA_ARGS__)

// helper macro for the common case of just a different namespace
#define DEFINE_BINARY_OP_FN_STD(op, fn)                                        \
  DEFINE_BINARY_OP_FN(                                                         \
      op, return std::fn(PromoteHalfsToFloats(x), PromoteHalfsToFloats(y));    \
      , return ipu::fn(x, y);)

DEFINE_BINARY_OP_FN(expr::BinaryOpType::ADD, return x + y;)
DEFINE_BINARY_OP_FN_STD(expr::BinaryOpType::ATAN2, atan2)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_AND, return x & y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_OR, return x | y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_XOR, return x ^ y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_XNOR, return ~(x ^ y);)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::DIVIDE, return x / y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::EQUAL, return x == y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::GREATER_THAN_EQUAL, return x >= y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::GREATER_THAN, return x > y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::LESS_THAN_EQUAL, return x <= y;)
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::LOGICAL_AND, return x && y;)
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::LOGICAL_OR, return x || y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::LESS_THAN, return x < y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::MAXIMUM, return max(x, y);
                    ,
                    return BinaryLibCall<expr::BinaryOpType::MAXIMUM>{}(x, y);)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::MINIMUM, return min(x, y);
                    ,
                    return BinaryLibCall<expr::BinaryOpType::MINIMUM>{}(x, y);)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::MULTIPLY, return x * y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::NOT_EQUAL, return x != y;)
DEFINE_BINARY_OP_FN_STD(expr::BinaryOpType::POWER, pow);
DEFINE_BINARY_OP_FN(
    expr::BinaryOpType::REMAINDER,
    if (std::is_integral<T>::value) {
      auto r = x / y;
      return x - r * y;
    } else { return std::fmod(float(x), float(y)); },
    return BinaryLibCall<expr::BinaryOpType::REMAINDER>{}(x, y);)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SHIFT_LEFT, return x << y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SHIFT_RIGHT, return (unsigned)x >> y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, return x >> y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SUBTRACT, return x - y;)
} // namespace

template <expr::BinaryOpType op, typename inT, typename outT, typename A>
struct BinaryOpDispatch {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) inT *in1,
                      const __attribute__((align_value(8))) inT *in2,
                      __attribute__((align_value(8))) outT *out) {

    for (unsigned j = 0; j != size; ++j) {
      out[j] = BinaryOpFn<op, inT, A>::fn(in1[j], in2[j]);
    }
  }
};

#ifdef __IPU__

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in1,
                      const __attribute__((align_value(8))) float *in2,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const float2 *f2In1 = reinterpret_cast<const float2 *>(in1);
      const float2 *f2In2 = reinterpret_cast<const float2 *>(in2);
      int *iOut = reinterpret_cast<int *>(out);
      const unsigned loopCount = maskForRepeat(size / 4u);
      for (unsigned i = 0; i < loopCount; ++i) {
        float2 load1 = ipu::load_postinc(&f2In1, 1);
        float2 load2 = ipu::load_postinc(&f2In2, 1);
        long2 calc_lo = static_cast<long2>(FuncTy<float2>::fn(load1, load2));

        load1 = ipu::load_postinc(&f2In1, 1);
        load2 = ipu::load_postinc(&f2In2, 1);
        long2 calc_hi = static_cast<long2>(FuncTy<float2>::fn(load1, load2));

        char4 result = tochar4(calc_lo, calc_hi);
        int ires = copy_cast<int>(result) & 0x01010101;
        ipu::store_postinc(&iOut, ires, 1);
      }
      in1 = reinterpret_cast<const float *>(f2In1);
      in2 = reinterpret_cast<const float *>(f2In2);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      float load1 = ipu::load_postinc(&in1, 1);
      float load2 = ipu::load_postinc(&in2, 1);
      *out++ = FuncTy<float>::fn(load1, load2);
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, half, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in1,
                      const __attribute__((align_value(8))) half *in2,
                      __attribute__((align_value(8))) bool *out) {

    if (size >= 4) {
      const half4 *h4In1 = reinterpret_cast<const half4 *>(in1);
      const half4 *h4In2 = reinterpret_cast<const half4 *>(in2);
      int *iOut = reinterpret_cast<int *>(out);

      const unsigned loopCount = maskForRepeat(size / 4u);
      for (unsigned i = 0; i < loopCount; ++i) {
        half4 load1 = ipu::load_postinc(&h4In1, 1);
        half4 load2 = ipu::load_postinc(&h4In2, 1);
        short4 calc = static_cast<short4>(FuncTy<half4>::fn(load1, load2));
        char4 result = tochar4(calc);
        int ires = copy_cast<int>(result) & 0x01010101;
        ipu::store_postinc(&iOut, ires, 1);
      }
      in1 = reinterpret_cast<const half *>(h4In1);
      in2 = reinterpret_cast<const half *>(h4In2);
      out = reinterpret_cast<bool *>(iOut);
    }
    // process any remainder, up to 3 of
    size = size & 3;
    for (unsigned j = 0; j != size; ++j) {
      half load1 = ipu::load_postinc(&in1, 1);
      half load2 = ipu::load_postinc(&in2, 1);
      *out++ = FuncTy<half>::fn(load1, load2);
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, half, half, architecture::ipu> {
  static_assert(
      std::is_same<half, typename BinaryOpOutputType<op, half>::type>::value,
      "");
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in1,
                      const __attribute__((align_value(8))) half *in2,
                      __attribute__((align_value(8)))
                      typename BinaryOpOutputType<op, half>::type *out) {
    using arch = architecture::ipu;

    if (size >= 4) {
      const half4 *h4In1 = reinterpret_cast<const half4 *>(in1);
      const half4 *h4In2 = reinterpret_cast<const half4 *>(in2);
      half4 *h4Out = reinterpret_cast<half4 *>(out);

      // LLVM currently chooses to rotate the loop in a way that is not optimal
      // for our hardware. The inline asm blocks this. The loop is pipelined
      // sufficiently to overlap load with calculation. This was used a it seems
      // a reasonable compromise over zero overlap and unrolling far enough to
      // overlap the store with calculation.

      half4 load1 = ipu::load_postinc(&h4In1, 1);
      half4 load2 = ipu::load_postinc(&h4In2, 1);
      const unsigned loopCount = maskForRepeat((size / 4u) - 1u);
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < loopCount; ++i) {
        half4 calc = BinaryOpFn<op, half4, arch>::fn(load1, load2);
        load1 = ipu::load_postinc(&h4In1, 1);
        load2 = ipu::load_postinc(&h4In2, 1);
        ipu::store_postinc(&h4Out, calc, 1);
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&h4Out, BinaryOpFn<op, half4, arch>::fn(load1, load2),
                         1);

      in1 = reinterpret_cast<const half *>(h4In1);
      in2 = reinterpret_cast<const half *>(h4In2);
      half *tmp = reinterpret_cast<half *>(h4Out);
      size -= (tmp - out);
      out = tmp;
    }

    const half2 *h2In1 = reinterpret_cast<const half2 *>(in1);
    const half2 *h2In2 = reinterpret_cast<const half2 *>(in2);
    half2 *h2Out = reinterpret_cast<half2 *>(out);

    if (size >= 2) {
      ipu::store_postinc(
          &h2Out,
          BinaryOpFn<op, half2, arch>::fn(ipu::load_postinc(&h2In1, 1),
                                          ipu::load_postinc(&h2In2, 1)),
          1);
      size -= 2;
    }

    if (size == 1) {
      half2 res = (half2){
          BinaryOpFn<op, half, arch>::fn((*h2In1)[0], (*h2In2)[0]),
          (*h2Out)[1],
      };
      *h2Out = res;
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, float, float, architecture::ipu> {
  static_assert(
      std::is_same<float, typename BinaryOpOutputType<op, float>::type>::value,
      "");

  static void compute(unsigned size,
                      const __attribute__((align_value(8))) float *in1,
                      const __attribute__((align_value(8))) float *in2,
                      __attribute__((align_value(8)))
                      typename BinaryOpOutputType<op, float>::type *out) {
    using arch = architecture::ipu;

    if (size >= 2) {
      const float2 *f2In1 = reinterpret_cast<const float2 *>(in1);
      const float2 *f2In2 = reinterpret_cast<const float2 *>(in2);
      float2 *f2Out = reinterpret_cast<float2 *>(out);

      float2 load1 = ipu::load_postinc(&f2In1, 1);
      float2 load2 = ipu::load_postinc(&f2In2, 1);
      unsigned loopCount = maskForRepeat((size / 2u) - 1u);
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < loopCount; ++i) {
        float2 calc = BinaryOpFn<op, float2, arch>::fn(load1, load2);
        load1 = ipu::load_postinc(&f2In1, 1);
        load2 = ipu::load_postinc(&f2In2, 1);
        ipu::store_postinc(&f2Out, calc, 1);
      }
      asm volatile("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&f2Out, BinaryOpFn<op, float2, arch>::fn(load1, load2),
                         1);

      in1 = reinterpret_cast<const float *>(f2In1);
      in2 = reinterpret_cast<const float *>(f2In2);
      float *tmp = reinterpret_cast<float *>(f2Out);
      size -= (tmp - out);
      out = tmp;
    }

    if (size == 1) {
      *out = BinaryOpFn<op, float, arch>::fn(*in1, *in2);
    }
  }
};

#endif

template <expr::BinaryOpType op, typename T> class BinaryOp2D : public Vertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;

public:
  constexpr static bool isExternal() {
    return (std::is_same<outputType, float>::value ||
            std::is_same<outputType, half>::value) &&
           (op == expr::BinaryOpType::ADD ||
            op == expr::BinaryOpType::SUBTRACT ||
            op == expr::BinaryOpType::MULTIPLY);
  }
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in1;
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in2;
  Vector<Output<Vector<outputType, SPAN, 8>>> out;
  IS_EXTERNAL_CODELET(isExternal());

  bool compute() {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;
    const unsigned limI = out.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::BinaryOpDispatch<op, T, outputType, arch>::compute(
          out[i].size(), &in1[i][0], &in2[i][0], &out[i][0]);
    }
    return true;
  }
};

template <expr::BinaryOpType op, typename T>
class BinaryOp2DInPlace : public Vertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;
  static_assert(std::is_same<T, outputType>::value,
                "In, Out types must match for in place operations");

public:
  constexpr static bool isExternal() {
    return (std::is_same<outputType, float>::value ||
            std::is_same<outputType, half>::value) &&
           (op == expr::BinaryOpType::ADD ||
            op == expr::BinaryOpType::SUBTRACT ||
            op == expr::BinaryOpType::MULTIPLY);
  }
  Vector<InOut<Vector<outputType, SPAN, 8>>> in1Out;
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in2;
  IS_EXTERNAL_CODELET(isExternal());

  bool compute() {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;
    const unsigned limI = in1Out.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::BinaryOpDispatch<op, T, outputType, arch>::compute(
          in1Out[i].size(), &in1Out[i][0], &in2[i][0], &in1Out[i][0]);
    }
    return true;
  }
};

//******************************************************************************
// Dispatch for use with Binary Operation supervisor vertices
//******************************************************************************
template <expr::BinaryOpType op, typename inT, typename outT, typename A>
struct BinaryOpDispatchSupervisor {
public:
  static void compute(unsigned size, unsigned worker, inT *in1, inT *in2,
                      outT *out) {
    // No vectorisation for int, unsigned int, but still split over workers
    // However cannot use this when writing bool
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BinaryOpFn<op, inT, architecture::generic>::fn(in1[j], in2[j]);
  }
};

#ifdef __IPU__

template <expr::BinaryOpType op>
struct BinaryOpDispatchSupervisor<op, half, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) half *in1,
                      const __attribute__((align_value(8))) half *in2,
                      __attribute__((align_value(8))) bool *out) {

    const half4 *h4In1 = reinterpret_cast<const half4 *>(in1) + worker;
    const half4 *h4In2 = reinterpret_cast<const half4 *>(in2) + worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;

    const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
    for (unsigned j = 0; j < loopCount; j++) {
      half4 load1 = ipu::load_postinc(&h4In1, CTXT_WORKERS);
      half4 load2 = ipu::load_postinc(&h4In2, CTXT_WORKERS);
      short4 calc = static_cast<short4>(FuncTy<half4>::fn(load1, load2));
      char4 result = tochar4(calc);
      int ires = copy_cast<int>(result) & 0x01010101;
      ipu::store_postinc(&iOut, ires, CTXT_WORKERS);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if (worker == (CTXT_WORKERS - 1) && remainder) {
      in1 = &in1[size - remainder];
      in2 = &in2[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        half load1 = ipu::load_postinc(&in1, 1);
        half load2 = ipu::load_postinc(&in2, 1);
        *out++ = FuncTy<half>::fn(load1, load2);
      }
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatchSupervisor<op, float, bool, architecture::ipu> {

  static_assert(sizeof(int) == sizeof(char4), "");
  static_assert(sizeof(bool) == sizeof(char), "");

  template <class T> using FuncTy = BinaryOpFn<op, T, architecture::ipu>;

  static void compute(unsigned size, unsigned worker,
                      const __attribute__((align_value(8))) float *in1,
                      const __attribute__((align_value(8))) float *in2,
                      __attribute__((align_value(8))) bool *out) {

    const float2 *f2In1 = reinterpret_cast<const float2 *>(in1) + 2 * worker;
    const float2 *f2In2 = reinterpret_cast<const float2 *>(in2) + 2 * worker;
    int *iOut = reinterpret_cast<int *>(out) + worker;

    const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
    for (unsigned j = 0; j < loopCount; j++) {
      float2 load1 = ipu::load_postinc(&f2In1, 1);
      float2 load2 = ipu::load_postinc(&f2In2, 1);
      long2 calc_lo = static_cast<long2>(FuncTy<float2>::fn(load1, load2));

      load1 = ipu::load_postinc(&f2In1, 2 * CTXT_WORKERS - 1);
      load2 = ipu::load_postinc(&f2In2, 2 * CTXT_WORKERS - 1);
      long2 calc_hi = static_cast<long2>(FuncTy<float2>::fn(load1, load2));

      char4 result = tochar4(calc_lo, calc_hi);
      int ires = copy_cast<int>(result) & 0x01010101;
      ipu::store_postinc(&iOut, ires, CTXT_WORKERS);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    // As we are writing bools it's dangerous to share this between workers
    unsigned remainder = size & 3;
    if (worker == (CTXT_WORKERS - 1) && remainder) {
      in1 = &in1[size - remainder];
      in2 = &in2[size - remainder];
      out = &out[size - remainder];
      for (unsigned j = 0; j != remainder; ++j) {
        float load1 = ipu::load_postinc(&in1, 1);
        float load2 = ipu::load_postinc(&in2, 1);
        *out++ = FuncTy<float>::fn(load1, load2);
      }
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatchSupervisor<op, half, half, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, half *in1, half *in2,
                      typename BinaryOpOutputType<op, half>::type *out) {

    const half4 *h4In1 = reinterpret_cast<const half4 *>(in1) + worker;
    const half4 *h4In2 = reinterpret_cast<const half4 *>(in2) + worker;
    half4 *h4Out = reinterpret_cast<half4 *>(out) + worker;

    asm volatile("# Thwart loop rotation (start)" ::: "memory");
    const unsigned loopCount = maskForRepeat(divideWork(size, 2, worker));
    for (unsigned i = 0; i < loopCount; i++) {
      half4 load1 = ipu::load_postinc(&h4In1, CTXT_WORKERS);
      half4 load2 = ipu::load_postinc(&h4In2, CTXT_WORKERS);
      half4 calc = BinaryOpFn<op, half4, architecture::ipu>::fn(load1, load2);
      ipu::store_postinc(&h4Out, calc, CTXT_WORKERS);
    }
    asm volatile("# Thwart loop rotation (end)" ::: "memory");
    if (size & 3) {
      const half2 *h2In1 = reinterpret_cast<const half2 *>(h4In1);
      const half2 *h2In2 = reinterpret_cast<const half2 *>(h4In2);
      half2 *h2Out = reinterpret_cast<half2 *>(h4Out);
      if (size & 2) {
        if (h4Out == (half4 *)&out[size & (~3)]) {
          ipu::store_postinc(
              &h2Out,
              BinaryOpFn<op, half2, architecture::ipu>::fn(
                  ipu::load_postinc(&h2In1, 1), ipu::load_postinc(&h2In2, 1)),
              1);
        }
      }
      assert(size != 0);
      if (h2Out == (half2 *)&out[size - 1]) {
        half2 res = (half2){
            BinaryOpFn<op, half, architecture::ipu>::fn((*h2In1)[0],
                                                        (*h2In2)[0]),
            (*h2Out)[1],
        };
        *h2Out = res;
      }
    }
  }
};

template <expr::BinaryOpType op>
class BinaryOpDispatchSupervisor<op, float, float, architecture::ipu> {
public:
  static void compute(unsigned size, unsigned worker, float *in1, float *in2,
                      typename BinaryOpOutputType<op, float>::type *out) {

    const float2 *f2In1 = reinterpret_cast<const float2 *>(in1) + worker;
    const float2 *f2In2 = reinterpret_cast<const float2 *>(in2) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;

    const unsigned loopCount = maskForRepeat(divideWork(size, 1, worker));
    for (unsigned j = 0; j < loopCount; j++) {
      float2 load1 = ipu::load_postinc(&f2In1, CTXT_WORKERS);
      float2 load2 = ipu::load_postinc(&f2In2, CTXT_WORKERS);
      float2 calc = BinaryOpFn<op, float2, architecture::ipu>::fn(load1, load2);
      ipu::store_postinc(&f2Out, calc, CTXT_WORKERS);
    }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if (worker == (CTXT_WORKERS - 1) && (size & 1)) {
      out[size - 1] = BinaryOpFn<op, float, architecture::ipu>::fn(
          in1[size - 1], in2[size - 1]);
    }
  }
};
#endif

template <expr::BinaryOpType op, typename T>
constexpr bool binaryOp1DIsSupervisor() {
  using OutType = typename BinaryOpOutputType<op, T>::type;

  // Exclude those that output a bool and aren't vectorised as multiple
  // workers writing < 32 bits at once is dangerous
  const bool boolOut = std::is_same<OutType, bool>::value;
  const bool intIn = std::is_same<T, int>::value ||
                     std::is_same<T, bool>::value ||
                     std::is_same<T, unsigned>::value;
  return !(boolOut && intIn);
}

template <expr::BinaryOpType op, typename T>
class BinaryOp1DSupervisor
    : public SupervisorVertexIf<binaryOp1DIsSupervisor<op, T>() &&
                                ASM_CODELETS_ENABLED> {
  typedef typename BinaryOpOutputType<op, T>::type OutputType;

public:
  Input<Vector<T, ONE_PTR, 8>> in1;
  Input<Vector<T, ONE_PTR, 8>> in2;
  Output<Vector<OutputType, SPAN, 8>> out;

  IS_EXTERNAL_CODELET((binaryOp1DIsSupervisor<op, T>()));

  bool compute() {
    for (unsigned j = 0; j != out.size(); ++j) {
      out[j] = BinaryOpFn<op, T, architecture::generic>::fn(in1[j], in2[j]);
    }
    return true;
  }
};

template <expr::BinaryOpType op, typename T>
constexpr bool BinaryOp1DInPlaceIsSupervisor() {
  using OutType = typename BinaryOpOutputType<op, T>::type;
  return !std::is_same<OutType, bool>::value;
}

template <expr::BinaryOpType op, typename T>
class BinaryOp1DInPlaceSupervisor
    : public SupervisorVertexIf<BinaryOp1DInPlaceIsSupervisor<op, T>() &&
                                ASM_CODELETS_ENABLED> {
  typedef typename BinaryOpOutputType<op, T>::type OutputType;
  static_assert(std::is_same<T, OutputType>::value,
                "In, Out types must match for in place operations");

public:
  InOut<Vector<OutputType, SPAN, 8>> in1Out;
  Input<Vector<T, ONE_PTR, 8>> in2;

  IS_EXTERNAL_CODELET((BinaryOp1DInPlaceIsSupervisor<op, T>()));
  bool compute() {
    for (unsigned j = 0; j != in1Out.size(); ++j) {
      in1Out[j] =
          BinaryOpFn<op, T, architecture::generic>::fn(in1Out[j], in2[j]);
    }
    return true;
  }
};

//******************************************************************************
// Worker vertex to actually do the work of the operation for the
// BinaryOp1DSupervisor vertex when it is an external codelet
//******************************************************************************

template <expr::BinaryOpType op, typename T> class BinaryOp1D : public Vertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;

public:
  constexpr static bool isExternal() {
    return (std::is_same<T, float>::value || std::is_same<T, half>::value) &&
           (op == expr::BinaryOpType::ADD ||
            op == expr::BinaryOpType::SUBTRACT ||
            op == expr::BinaryOpType::MULTIPLY);
  }
  Input<Vector<T, ONE_PTR, 8>> in1;
  Input<Vector<T, ONE_PTR, 8>> in2;
  Output<Vector<outputType, SPAN, 8>> out;

  IS_EXTERNAL_CODELET(isExternal());
  bool compute() {
#ifdef __IPU__
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;

    popops::BinaryOpDispatchSupervisor<op, T, outputType, arch>::compute(
        out.size(), getWsr(), &in1[0], &in2[0], &out[0]);
#endif
    return true;
  }
};

//******************************************************************************
// Worker vertex to actually do the work of the operation for the
// BinaryOp1DInPlaceSupervisor vertex when it is an external codelet
//******************************************************************************
template <expr::BinaryOpType op, typename T>
class BinaryOp1DInPlace : public Vertex {
  typedef typename BinaryOpOutputType<op, T>::type outputType;
  static_assert(std::is_same<T, outputType>::value,
                "In, Out types must match for in place operations");

public:
  constexpr static bool isExternal() {
    return (std::is_same<T, float>::value || std::is_same<T, half>::value) &&
           (op == expr::BinaryOpType::ADD ||
            op == expr::BinaryOpType::SUBTRACT ||
            op == expr::BinaryOpType::MULTIPLY);
  }
  InOut<Vector<outputType, SPAN, 8>> in1Out;
  Input<Vector<T, ONE_PTR, 8>> in2;

  IS_EXTERNAL_CODELET(isExternal());
  bool compute() {
#ifdef __IPU__
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;

    popops::BinaryOpDispatchSupervisor<op, T, outputType, arch>::compute(
        in1Out.size(), getWsr(), &in1Out[0], &in2[0], &in1Out[0]);
#endif
    return true;
  }
};

INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::ADD, float, half, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_AND, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_OR, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_XOR, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_XNOR, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::EQUAL, float, half, bool, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::GREATER_THAN_EQUAL, float, half,
               int, unsigned, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::GREATER_THAN, float, half, int,
               unsigned, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LESS_THAN_EQUAL, float, half,
               int, unsigned, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LESS_THAN, float, half, int,
               unsigned, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MULTIPLY, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::NOT_EQUAL, float, half, int,
               unsigned, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::REMAINDER, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_LEFT, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_RIGHT, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SUBTRACT, float, half, int,
               unsigned)

// BinaryOp1DSupervisor - supervisor stubs for all types except bool.  If bool
// they will generate single worker code. See T4642 - a task to add
// these.
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::ADD, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::BITWISE_AND, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::BITWISE_OR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::BITWISE_XOR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::BITWISE_XNOR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::DIVIDE, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::EQUAL, float, half,
               bool, int, unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::GREATER_THAN_EQUAL,
               float, half, int, unsigned, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::GREATER_THAN, float,
               half, int, unsigned, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LESS_THAN_EQUAL, float,
               half, int, unsigned, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LESS_THAN, float, half,
               int, unsigned, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::MAXIMUM, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::MINIMUM, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::MULTIPLY, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::NOT_EQUAL, float, half,
               int, unsigned, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::REMAINDER, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::SHIFT_LEFT, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::SHIFT_RIGHT, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor,
               expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::SUBTRACT, float, half,
               int, unsigned)

// BinaryOp1D  - Worker code for all types except bool
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::ADD, float, half, int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_AND, int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_OR, int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_XOR, int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_XNOR, int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::EQUAL, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::GREATER_THAN_EQUAL, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::GREATER_THAN, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LESS_THAN_EQUAL, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LESS_THAN, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MULTIPLY, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::NOT_EQUAL, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::REMAINDER, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_LEFT, int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_RIGHT, int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SUBTRACT, float, half, int,
               unsigned)

INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::ADD, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_AND, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_OR, int, unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_XOR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_XNOR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::GREATER_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::GREATER_THAN, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LESS_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LESS_THAN, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MULTIPLY, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::NOT_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::REMAINDER, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_LEFT, int, unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_RIGHT, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SUBTRACT, float, half,
               int, unsigned)

// Supervisor vertices, creating stubs in the IPU build
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::ADD, float,
               half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::ATAN2, float,
               half)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::BITWISE_AND,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::BITWISE_OR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::BITWISE_XOR,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::BITWISE_XNOR,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::DIVIDE, float,
               half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor,
               expr::BinaryOpType::GREATER_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::GREATER_THAN,
               bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::LESS_THAN_EQUAL,
               bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::LOGICAL_AND,
               bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::LOGICAL_OR,
               bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::LESS_THAN, bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::MAXIMUM, float,
               half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::MINIMUM, float,
               half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::MULTIPLY, float,
               half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::NOT_EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::POWER, float,
               half)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::REMAINDER,
               float, half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::SHIFT_LEFT, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::SHIFT_RIGHT,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor,
               expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::SUBTRACT, float,
               half, int, unsigned)

// Worker vertices, for the IPU build
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::ADD, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_AND, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_OR, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_XOR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_XNOR, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::DIVIDE, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MAXIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MINIMUM, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MULTIPLY, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::REMAINDER, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_LEFT, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_RIGHT, int,
               unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SUBTRACT, float, half,
               int, unsigned)

} // namespace popops
