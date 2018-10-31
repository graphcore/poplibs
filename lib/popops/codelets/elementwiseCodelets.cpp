#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

#include <cmath>

#include "util.hpp"
#include "popops/ExprOp.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <tileimplconsts.h>

#ifdef __IPU__
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#include <tilearch.h>

inline unsigned getWsr(void) {
  return __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
}
#endif

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

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;

namespace popops {

// Macros to instatiate a template class for an operator and a number
// of types.
#define INSTANTIATE_OP_1(v, op, t) \
  template class v<op, t>;
#define INSTANTIATE_OP_2(v, op, t, ...) \
  template class v<op, t>; INSTANTIATE_OP_1(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_3(v, op, t, ...) \
  template class v<op, t>; INSTANTIATE_OP_2(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_4(v, op, t, ...) \
  template class v<op, t>; INSTANTIATE_OP_3(v, op, __VA_ARGS__)
#define INSTANTIATE_OP_5(v, op, t, ...) \
  template class v<op, t>; INSTANTIATE_OP_4(v, op, __VA_ARGS__)

#define SELECT_VARGS(_1,_2,_3,_4,_5,NAME,...) INSTANTIATE_OP ## NAME
#define INSTANTIATE_OP(v, op, ...) \
  SELECT_VARGS(__VA_ARGS__,_5,_4,_3,_2,_1)(v, op, __VA_ARGS__)

namespace {
  // Helper function to explicitly and specifically cast half to float
  // to indicate this is intentional floating point promotion, while
  // allowing other types to pass through unchanged.
  template <typename T>
  struct PromoteHalfsToFloatsHelper {
    using ReturnType =
      typename std::conditional<std::is_same<T, half>::value, float, T>::type;
    ReturnType operator()(const T &x) { return static_cast<ReturnType>(x); }
  };

  template <typename T>
  typename PromoteHalfsToFloatsHelper<T>::ReturnType
  PromoteHalfsToFloats(T x) {
    return PromoteHalfsToFloatsHelper<T>()(x);
  }

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

  template <typename T>
  struct UnaryOpOutputType<expr::UnaryOpType::IS_FINITE, T> {
    using type = bool;
  };

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
// TODO: T4609 enable fast patch for codelets that return bools.
DEFINE_UNARY_OP_FN_GEN(expr::UnaryOpType::IS_FINITE,
                       return (x == x) &&
                              (std::abs(PromoteHalfsToFloats(x)) != INFINITY);)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::LOGARITHM, log)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::LOGARITHM_ONE_PLUS, log1p)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::LOGICAL_NOT, return !x;)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::NEGATE, return -x;)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::POPCOUNT,
                   return __builtin_popcount(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIGNUM,
                   return (0 < x) - (x < 0);)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::SIN, sin)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::TANH, tanh)
DEFINE_UNARY_OP_FN_STD(expr::UnaryOpType::ROUND, round)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQRT,
                   return std::sqrt(PromoteHalfsToFloats(x));,
                   return UnaryLibCall<expr::UnaryOpType::SQRT>{}(x);)
DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQUARE,
                   return (x * x);)

} // namespace

template <expr::UnaryOpType op, typename T, typename A>
struct UnaryOpDispatch {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) T *in,
                      __attribute__((align_value(8)))
                      typename UnaryOpOutputType<op, T>::type *out) {

    for (unsigned j = 0; j != size; ++j) {
      out[j] = UnaryOpFn<op, T, A>::fn(in[j]);
    }
  }
};

#ifdef __IPU__
template <expr::UnaryOpType op>
struct UnaryOpDispatch<op, half, architecture::ipu> {
  static_assert(
      std::is_same<half, typename UnaryOpOutputType<op, half>::type>::value,
      "");
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) half *in,
                      __attribute__((align_value(8)))
                      typename UnaryOpOutputType<op, half>::type *out) {
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
      asm volatile("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < (size / 4u) - 1u; i++) {
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
      popops::UnaryOpDispatch<op, T, arch>::compute(out[i].size(), &in[i][0],
                                                    &out[i][0]);
    }
    return true;
  }
};

template <expr::UnaryOpType op, typename T>
class
UnaryOp2DInPlace : public Vertex {
public:
  Vector<InOut<Vector<T, SPAN, 8>>> inOut;

  bool compute() {
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    unsigned limI = inOut.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::UnaryOpDispatch<op, T, arch>::compute(inOut[i].size(),
                                                    &inOut[i][0], &inOut[i][0]);
    }
    return true;
  }
};
//******************************************************************************
// Dispatch for use with Unary Operation supervisor vertices
//******************************************************************************
template <expr::UnaryOpType op, typename T, typename A>
struct UnaryOpDispatchSupervisor {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      T *in,
                      typename UnaryOpOutputType<op, T>::type *out) {
    // No vectorisation for int, unsigned int, but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = UnaryOpFn<op, T, architecture::generic>::fn(in[j]);
  }
};

#ifdef __IPU__

template <expr::UnaryOpType op>
struct UnaryOpDispatchSupervisor<op, half, architecture::ipu> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      half *in,
                       typename UnaryOpOutputType<op, half>::type *out) {

    const half4 *h4In = reinterpret_cast<const half4 *>(in) + worker;
    half4 *h4Out = reinterpret_cast<half4 *>(out) + worker;
    half4 load = ipu::load_postinc(&h4In, CTXT_WORKERS);

    asm volatile ("# Thwart loop rotation (start)" ::: "memory");
    for (unsigned i = worker; i < size>>2; i+=CTXT_WORKERS) {
      half4 calc = UnaryOpFn<op, half4,architecture::ipu>::fn(load);
      load = ipu::load_postinc(&h4In, CTXT_WORKERS);
      ipu::store_postinc(&h4Out, calc, CTXT_WORKERS);
    }
    asm volatile ("# Thwart loop rotation (end)" ::: "memory");
    if(size & 3) {
      const half2 *h2In = reinterpret_cast<const half2*>(h4In - CTXT_WORKERS);
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
class UnaryOpDispatchSupervisor<op, float, architecture::ipu> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      float *in,
                      typename UnaryOpOutputType<op, float>::type *out) {

    const float2 *f2In = reinterpret_cast<const float2 *>(in) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;
    float2 load = ipu::load_postinc(&f2In, CTXT_WORKERS);

    for(unsigned j = worker; j < (size>>1) ; j += CTXT_WORKERS) {
      float2 calc = UnaryOpFn<op, float2, architecture::ipu>::fn(load);
      load = ipu::load_postinc(&f2In, CTXT_WORKERS);
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

  IS_EXTERNAL_CODELET(!(std::is_same<outputType, bool>::value));
  bool compute() {
    for (unsigned j = 0; j != out.size(); ++j) {
      out[j] = UnaryOpFn<op, T, architecture::generic>::fn(in[j]);
    }
    return true;
  }
};
template <expr::UnaryOpType op, typename T>
class
UnaryOp1DInPlaceSupervisor : public SupervisorVertex {
public:
  InOut<Vector<T, SPAN, 8>> inOut;
  IS_EXTERNAL_CODELET(!(std::is_same<T, bool>::value));
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
    popops::UnaryOpDispatchSupervisor<op, T, arch>::compute(out.size(),
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
public:
  InOut<Vector<T, SPAN, 8>> inOut;

  bool compute() {
#ifdef __IPU__
    using arch = typename popops::UnaryOpFn<op, T, architecture::active>::arch;
    popops::UnaryOpDispatchSupervisor<op, T, arch>::compute(inOut.size(),
      getWsr(), &inOut[0], &inOut[0]);
#endif
    return true;
  }
};

INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp2D, expr::UnaryOpType::SQUARE, float, half)

// UnaryOp1DSupervisor - supervisor stubs for all types except bool.  If bool
// they will generate single worker code. See T4642 - a task to add
// these.
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::ABSOLUTE, float, half,
               int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::EXPONENT_MINUS_ONE,
               float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::LOGARITHM_ONE_PLUS,
               float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1DSupervisor, expr::UnaryOpType::SQUARE, float, half)

// UnaryOp1D - worker vertex for all types except bool.
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1D, expr::UnaryOpType::SQUARE, float, half)



INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp2DInPlace, expr::UnaryOpType::SQUARE, float, half)

// UnaryOp1DInPlaceSupervisor - supervisor stubs for all types except bool.
// If bool they will generate single worker code. See T4642 - a task to add
// these.

INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::ABSOLUTE, float,
               half, int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor,
               expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::EXPONENT, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor,
               expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::LOGARITHM, float,
               half)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::NEGATE, float,
               half, int)
INSTANTIATE_OP(UnaryOp1DInPlaceSupervisor, expr::UnaryOpType::POPCOUNT, int)
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
               half)

// UnaryOp1DInPlace - worker vertex for all types except bool.

INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::LOGARITHM, float,half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp1DInPlace, expr::UnaryOpType::SQUARE, float, half)

namespace {
  // certain operators need to call a different function depending on the input
  // type. this deals with dispatching binary ops to their correct functions.
  template <expr::BinaryOpType Op>
  struct BinaryLibCall {};

  template <>
  struct BinaryLibCall<expr::BinaryOpType::MAXIMUM> {
#ifdef __IPU__
    template <typename FPType>
    FPType operator()(FPType x, FPType y) const {
      return ipu::fmax(x, y);
    }
#endif

    int operator()(int x, int y) const {
      return max(x, y);
    }
  };

  template <>
  struct BinaryLibCall<expr::BinaryOpType::MINIMUM> {
#ifdef __IPU__
    template <typename FPType>
    FPType operator()(FPType x, FPType y) const {
      return ipu::fmin(x, y);
    }
#endif

    int operator()(int x, int y) const {
      return min(x, y);
    }
  };

  // Structure with template specialization to define the output type
  // of a binary operation
  template <expr::BinaryOpType op, typename T>
  struct BinaryOpOutputType { using type = T; };
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
  template <typename T>
  struct BinaryOpOutputType<expr::BinaryOpType::EQUAL, T> {
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

#define BINARY_VARGS(_1, _2, _3, N, ...) DEFINE_BINARY_OP_FN ## N
#define DEFINE_BINARY_OP_FN(...) \
  BINARY_VARGS(__VA_ARGS__, _2, _1, _)(__VA_ARGS__)


// helper macro for the common case of just a different namespace
#define DEFINE_BINARY_OP_FN_STD(op, fn)                                       \
  DEFINE_BINARY_OP_FN(op, return std::fn(PromoteHalfsToFloats(x),             \
                                         PromoteHalfsToFloats(y));,           \
                          return ipu::fn(x, y);)

DEFINE_BINARY_OP_FN(expr::BinaryOpType::ADD, return x + y;)
DEFINE_BINARY_OP_FN_STD(expr::BinaryOpType::ATAN2, atan2)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_AND, return x & y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_OR, return x | y; )
DEFINE_BINARY_OP_FN(expr::BinaryOpType::DIVIDE, return x / y; )
// TODO: T4609 enable fast path for codelets that return bools.
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::EQUAL, return x == y; )
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::GREATER_THAN_EQUAL, return x >= y; )
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::GREATER_THAN, return x > y; )
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::LESS_THAN_EQUAL, return x <= y; )
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::LOGICAL_AND, return x && y; )
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::LOGICAL_OR, return x || y; )
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::LESS_THAN, return x < y; )
DEFINE_BINARY_OP_FN(expr::BinaryOpType::MAXIMUM,
                    return max(x, y);,
                    return BinaryLibCall<expr::BinaryOpType::MAXIMUM>{}(x, y);)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::MINIMUM,
                    return min(x, y);,
                    return BinaryLibCall<expr::BinaryOpType::MINIMUM>{}(x, y);)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::MULTIPLY, return x * y; )
// TODO: T4609 enable fast path for codelets that return bools.
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::NOT_EQUAL, return x != y; )
DEFINE_BINARY_OP_FN_STD(expr::BinaryOpType::POWER, pow);
// TODO: T4609 enable fast path for codelets that return a different type.
DEFINE_BINARY_OP_FN_GEN(expr::BinaryOpType::REMAINDER,
                        if (std::is_same<T, int>::value) {
                          int r = x / y;
                          return x - r * y;
                        } else {
                          return std::fmod(float(x), float(y));
                        })
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SHIFT_LEFT, return x << y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SHIFT_RIGHT,
                    return (unsigned)x >> y; )
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
                    return x >> y;)
DEFINE_BINARY_OP_FN(expr::BinaryOpType::SUBTRACT, return x - y; )
} // namespace

template <expr::BinaryOpType op, typename T, typename A>
struct BinaryOpDispatch {
  static void compute(unsigned size,
                      const __attribute__((align_value(8))) T *in1,
                      const __attribute__((align_value(8))) T *in2,
                      __attribute__((align_value(8)))
                      typename BinaryOpOutputType<op, T>::type *out) {

    for (unsigned j = 0; j != size; ++j) {
      out[j] = BinaryOpFn<op, T, A>::fn(in1[j], in2[j]);
    }
  }
};

#ifdef __IPU__

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, half, architecture::ipu> {
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
      asm volatile ("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < (size/4u)-1u; i++) {
        half4 calc = BinaryOpFn<op, half4, arch>::fn(load1, load2);
        load1 = ipu::load_postinc(&h4In1, 1);
        load2 = ipu::load_postinc(&h4In2, 1);
        ipu::store_postinc(&h4Out, calc, 1);
      }
      asm volatile ("# Thwart loop rotation (end)" ::: "memory");
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
      half2 res = (half2)
        {
          BinaryOpFn<op, half, arch>::fn((*h2In1)[0], (*h2In2)[0]),
          (*h2Out)[1],
        };
      *h2Out = res;
    }
  }
};

template <expr::BinaryOpType op>
struct BinaryOpDispatch<op, float, architecture::ipu> {
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
      asm volatile ("# Thwart loop rotation (start)" ::: "memory");
      for (unsigned i = 0; i < (size/2u)-1u; i++) {
        float2 calc = BinaryOpFn<op, float2, arch>::fn(load1, load2);
        load1 = ipu::load_postinc(&f2In1, 1);
        load2 = ipu::load_postinc(&f2In2, 1);
        ipu::store_postinc(&f2Out, calc, 1);
      }
      asm volatile ("# Thwart loop rotation (end)" ::: "memory");
      ipu::store_postinc(&f2Out, BinaryOpFn<op, float2, arch>::fn(load1, load2),
                         1);

      in1 = reinterpret_cast<const float *>(f2In1);
      in2 = reinterpret_cast<const float *>(f2In2);
      float *tmp = reinterpret_cast<float *>(f2Out);
      size -= (tmp - out);
      out = tmp;
    }

    if (size == 1) {
      *out = BinaryOpFn<op,float,arch>::fn(*in1,*in2);
    }
  }
};

#endif

template <expr::BinaryOpType op, typename T>
class
BinaryOp2D : public Vertex {
typedef typename BinaryOpOutputType<op, T>::type outputType;
public:
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in1;
  Vector<Input<Vector<T, ONE_PTR, 8>>, ONE_PTR> in2;
  Vector<Output<Vector<outputType, SPAN, 8>>> out;

  bool compute() {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;
    const unsigned limI = out.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::BinaryOpDispatch<op, T, arch>::compute(out[i].size(), &in1[i][0],
                                                     &in2[i][0], &out[i][0]);
    }
    return true;
  }
};


template <expr::BinaryOpType op, typename T>
class
BinaryOp2DInPlace : public Vertex {
typedef typename BinaryOpOutputType<op, T>::type outputType;
public:
  Vector<InOut<Vector<outputType, SPAN, 8>>> in1Out;
  Vector<Input<Vector<T, ONE_PTR, 8>>> in2;

  bool compute() {
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;
    const unsigned limI = in1Out.size();
    for (unsigned i = 0; i != limI; ++i) {
      popops::BinaryOpDispatch<op, T, arch>::compute(
          in1Out[i].size(), &in1Out[i][0], &in2[i][0], &in1Out[i][0]);
    }
    return true;
  }
};

//******************************************************************************
// Dispatch for use with Binary Operation supervisor vertices
//******************************************************************************
template <expr::BinaryOpType op, typename T, typename A>
struct BinaryOpDispatchSupervisor {
public:
  static void compute(unsigned size,
                      unsigned worker,
                       T *in1,
                       T *in2,
                      typename BinaryOpOutputType<op, T>::type *out) {
    // No vectorisation for int, unsigned int, but still split over workers
    for (unsigned j = worker; j < size; j += CTXT_WORKERS)
      out[j] = BinaryOpFn<op, T, architecture::generic>::fn(in1[j], in2[j]);
  }
};

#ifdef __IPU__

template <expr::BinaryOpType op>
struct BinaryOpDispatchSupervisor<op, half, architecture::ipu> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      half *in1,
                      half *in2,
                      typename BinaryOpOutputType<op, half>::type *out) {

    const half4 *h4In1 = reinterpret_cast<const half4 *>(in1) + worker;
    const half4 *h4In2 = reinterpret_cast<const half4 *>(in2) + worker;
    half4 *h4Out = reinterpret_cast<half4 *>(out) + worker;
    half4 load1 = ipu::load_postinc(&h4In1, CTXT_WORKERS);
    half4 load2 = ipu::load_postinc(&h4In2, CTXT_WORKERS);

    asm volatile ("# Thwart loop rotation (start)" ::: "memory");
    for (unsigned i = worker; i < size>>2; i+=CTXT_WORKERS) {
      half4 calc = BinaryOpFn<op, half4,architecture::ipu>::fn(load1, load2);
      load1 = ipu::load_postinc(&h4In1, CTXT_WORKERS);
      load2 = ipu::load_postinc(&h4In2, CTXT_WORKERS);
      ipu::store_postinc(&h4Out, calc, CTXT_WORKERS);
    }
    asm volatile ("# Thwart loop rotation (end)" ::: "memory");
    if(size & 3) {
      const half2 *h2In1 = reinterpret_cast<const half2*>(h4In1 - CTXT_WORKERS);
      const half2 *h2In2 = reinterpret_cast<const half2*>(h4In2 - CTXT_WORKERS);
      half2 *h2Out = reinterpret_cast<half2 *>(h4Out);
      if(size & 2) {
        if(h4Out == (half4*)&out[size & (~3)]) {
          ipu::store_postinc(&h2Out,
                              BinaryOpFn<op, half2, architecture::ipu>::fn(
                              ipu::load_postinc(&h2In1, 1),
                              ipu::load_postinc(&h2In2, 1)),
                              1);
        }
      }
      assert(size != 0);
      if(h2Out == (half2*)&out[size-1]) {
        half2 res = (half2)
        {
          BinaryOpFn<op, half, architecture::ipu>::fn((*h2In1)[0],
                    (*h2In2)[0]), (*h2Out)[1],
        };
        *h2Out = res;
      }
    }
  }
};

template <expr::BinaryOpType op>
class BinaryOpDispatchSupervisor<op, float, architecture::ipu> {
public:
  static void compute(unsigned size,
                      unsigned worker,
                      float *in1,
                      float *in2,
                      typename BinaryOpOutputType<op, float>::type *out) {

    const float2 *f2In1 = reinterpret_cast<const float2 *>(in1) + worker;
    const float2 *f2In2 = reinterpret_cast<const float2 *>(in2) + worker;
    float2 *f2Out = reinterpret_cast<float2 *>(out) + worker;
    float2 load1 = ipu::load_postinc(&f2In1, CTXT_WORKERS);
    float2 load2 = ipu::load_postinc(&f2In2, CTXT_WORKERS);

    for(unsigned j = worker; j < (size>>1) ; j += CTXT_WORKERS) {
      float2 calc = BinaryOpFn<op, float2,architecture::ipu>::fn( load1, load2);
      load1 = ipu::load_postinc(&f2In1, CTXT_WORKERS);
      load2 = ipu::load_postinc(&f2In2, CTXT_WORKERS);
      ipu::store_postinc(&f2Out, calc, CTXT_WORKERS);
     }
    // The higher number worker is likely to have the least work in the
    // loop so allow it to process the remainder
    if(worker == (CTXT_WORKERS - 1)  && (size & 1)) {
      out[size-1] = BinaryOpFn<op,float,architecture::ipu>::fn(in1[size-1],
                                                               in2[size-1]);
    }
  }
};
#endif

template <expr::BinaryOpType op, typename T>
class
BinaryOp1DSupervisor : public SupervisorVertex {
typedef typename BinaryOpOutputType<op, T>::type outputType;
public:
  Input<Vector<T, ONE_PTR, 8>> in1;
  Input<Vector<T, ONE_PTR, 8>> in2;
  Output<Vector<outputType, SPAN, 8>> out;

  IS_EXTERNAL_CODELET(!(std::is_same<outputType, bool>::value));
  bool compute() {
    for (unsigned j = 0; j != out.size(); ++j) {
      out[j] = BinaryOpFn<op, T, architecture::generic>::fn(in1[j], in2[j]);
    }
    return true;
  }
};

template <expr::BinaryOpType op, typename T>
class
BinaryOp1DInPlaceSupervisor : public SupervisorVertex {
typedef typename BinaryOpOutputType<op, T>::type outputType;
public:
  InOut<Vector<outputType, SPAN, 8>> in1Out;
  Input<Vector<T, ONE_PTR, 8>> in2;
 IS_EXTERNAL_CODELET(!(std::is_same<outputType, bool>::value));
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

template <expr::BinaryOpType op, typename T>
class
BinaryOp1D : public Vertex {
typedef typename BinaryOpOutputType<op, T>::type outputType;
public:
  Input<Vector<T, ONE_PTR, 8>> in1;
  Input<Vector<T, ONE_PTR, 8>> in2;
  Output<Vector<outputType, SPAN, 8>> out;

  bool compute() {
#ifdef __IPU__
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;

    popops::BinaryOpDispatchSupervisor<op, T, arch>::compute(out.size(),
      getWsr(), &in1[0], &in2[0], &out[0]);

#endif
    return true;
  }
};

//******************************************************************************
// Worker vertex to actually do the work of the operation for the
// BinaryOp1DInPlaceSupervisor vertex when it is an external codelet
//******************************************************************************
template <expr::BinaryOpType op, typename T>
class
BinaryOp1DInPlace : public Vertex {
typedef typename BinaryOpOutputType<op, T>::type outputType;
public:
  InOut<Vector<outputType, SPAN, 8>> in1Out;
  Input<Vector<T, ONE_PTR, 8>> in2;

  bool compute() {
#ifdef __IPU__
    using arch = typename popops::BinaryOpFn<op, T, architecture::active>::arch;

    popops::BinaryOpDispatchSupervisor<op, T, arch>::compute(in1Out.size(),
      getWsr(), &in1Out[0], &in2[0], &in1Out[0]);
#endif
    return true;
  }
};

INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::ADD, float, half, int, unsigned)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_AND, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::BITWISE_OR, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::DIVIDE, float, half, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::EQUAL, float, half, bool, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::GREATER_THAN_EQUAL,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::GREATER_THAN,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LESS_THAN_EQUAL,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::LESS_THAN,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MAXIMUM, float, half, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MINIMUM, float, half, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::MULTIPLY, float, half, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::NOT_EQUAL, float, half, int,
               bool)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::REMAINDER, float, half, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_RIGHT, int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int)
INSTANTIATE_OP(BinaryOp2D, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)

// BinaryOp1DSupervisor - supervisor stubs for all types except bool.  If bool
// they will generate single worker code. See T4642 - a task to add
// these.
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::ADD, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::BITWISE_AND, int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::BITWISE_OR, int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::DIVIDE, float, half,
               int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::EQUAL, float, half,
               bool, int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::GREATER_THAN_EQUAL,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::GREATER_THAN,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LESS_THAN_EQUAL,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::LESS_THAN,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::MAXIMUM, float, half,
               int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::MINIMUM, float, half,
               int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::MULTIPLY, float, half,
               int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::NOT_EQUAL, float, half,
               int, bool)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::REMAINDER, float, half,
               int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::SHIFT_RIGHT, int)
INSTANTIATE_OP(BinaryOp1DSupervisor,
               expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp1DSupervisor, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)

// BinaryOp1D  - Worker code for all types except bool
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::ADD, float, half,
               int, unsigned)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_AND, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::BITWISE_OR, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::DIVIDE, float, half,
               int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::EQUAL, float, half,
               int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::GREATER_THAN_EQUAL,
               float, half, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::GREATER_THAN,
               float, half, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LESS_THAN_EQUAL,
               float, half, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::LESS_THAN,
               float, half, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MAXIMUM, float, half,
               int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MINIMUM, float, half,
               int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::MULTIPLY, float, half,
               int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::NOT_EQUAL, float, half,
               int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::REMAINDER, float, half,
               int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SHIFT_RIGHT, int)
INSTANTIATE_OP(BinaryOp1D,
               expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp1D, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)


INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::ADD, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_AND, int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::BITWISE_OR, int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::DIVIDE, float, half, int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::GREATER_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::GREATER_THAN, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LESS_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::LESS_THAN, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MAXIMUM, float, half,
               int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MINIMUM, float, half,
               int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::MULTIPLY, float, half,
               int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::NOT_EQUAL, bool)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::REMAINDER, float, half,
               int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_RIGHT, int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int)
INSTANTIATE_OP(BinaryOp2DInPlace, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)

// Supervisor vertices, creating stubs in the IPU build
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::ADD, float,
               half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::ATAN2, float,
               half)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::BITWISE_AND,
               int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::BITWISE_OR,
               int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::DIVIDE, float,
               half, int)
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
               half, int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::MINIMUM, float,
               half, int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::MULTIPLY, float,
               half, int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::NOT_EQUAL, bool)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::POWER, float,
               half)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::REMAINDER,
               float, half, int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::SHIFT_RIGHT,
               int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor,
               expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp1DInPlaceSupervisor, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)

// Worker vertices, for the IPU build
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::ADD, float,
               half, int, unsigned)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::ATAN2, float,
               half)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_AND,
               int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::BITWISE_OR,
               int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::DIVIDE, float,
               half, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MAXIMUM, float,
               half, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MINIMUM, float,
               half, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::MULTIPLY, float,
               half, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::POWER, float,
               half)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::REMAINDER,
               float, half, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SHIFT_RIGHT,
               int)
INSTANTIATE_OP(BinaryOp1DInPlace,
               expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, int)
INSTANTIATE_OP(BinaryOp1DInPlace, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)


template <typename InType>
class
[[poplar::constraint("elem(*data) != elem(*deltas)")]]
ScaledAddSupervisor : public SupervisorVertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<InType, SPAN, minAlign()>> data;
  Input<Vector<InType, ONE_PTR, minAlign()>> deltas;
  InType K;

  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; ++i) {
      data[i] += K * deltas[i];
    }
    return true;
  }
};

template class ScaledAddSupervisor<float>;
template class ScaledAddSupervisor<half>;
template class ScaledAddSupervisor<int>;
template class ScaledAddSupervisor<unsigned>;

template <typename InType>
class
[[poplar::constraint("elem(**data) != elem(**deltas)")]]
ScaledAdd2D : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  Vector<InOut<Vector<InType, SPAN, 8>>> data;
  Vector<Input<Vector<InType, ONE_PTR, 8>>, ONE_PTR> deltas;
  InType K;

  bool compute() {
    unsigned limI = data.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = data[i].size();
      auto const &refIn = deltas[i];
      auto &refOut = data[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] += K * refIn[j];
      }
    }
    return true;
  }
};

template class ScaledAdd2D<float>;
template class ScaledAdd2D<half>;
template class ScaledAdd2D<int>;
template class ScaledAdd2D<unsigned>;


template <typename FPType>
class
[[poplar::constraint("elem(**A) != elem(**B)")]]
HadamardProd : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> A;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> B;

  bool compute() {
    const unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      const unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] *= refIn[j];
      }
    }
    return true;
  }
};

template class HadamardProd<float>;
template class HadamardProd<half>;



template <typename InType>
class Zero : public Vertex {
public:
  Output<Vector<InType>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (auto &x : out) {
      x = 0;
    }
    return true;
  }
};

template class Zero<float>;
template class Zero<half>;
template class Zero<int>;
template class Zero<unsigned>;

template <typename FPType>
class Zero2d : public Vertex {
public:
  Vector<Output<Vector<FPType>>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (auto &row : out) {
      for (auto &x : row) {
        x = 0;
      }
    }
    return true;
  }
};

template class Zero2d<float>;
template class Zero2d<half>;

template <typename SrcType, typename DstType>
class
[[poplar::constraint("elem(*src) != elem(*dst)")]]
Cast : public Vertex {
public:

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf = std::is_same<SrcType,float>::value
            && std::is_same<DstType,half>::value;
  static const bool halfFloat = std::is_same<SrcType,half>::value
            && std::is_same<DstType,float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : 1;
  static const unsigned inAlign = ext ? 8 : 1;

  Input<Vector<SrcType, ONE_PTR, inAlign>> src;
  Output<Vector<DstType, SPAN, outAlign>> dst;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    const unsigned limI = dst.size();
    for (unsigned i = 0; i < limI; ++i) {
      dst[i] = static_cast<DstType>(src[i]);
    }
    return true;
  }
};

template class Cast<float, float>;
template class Cast<float, half>;
template class Cast<float, int>;
template class Cast<float, bool>;

template class Cast<half, float>;
template class Cast<half, half>;
template class Cast<half, int>;
template class Cast<half, bool>;

template class Cast<int,float>;
template class Cast<int,half>;
template class Cast<int,int>;
template class Cast<int,bool>;

template class Cast<bool,float>;
template class Cast<bool,half>;
template class Cast<bool,int>;
template class Cast<bool,bool>;

template <typename SrcType, typename DstType>
class
[[poplar::constraint("elem(**src) != elem(**dst)")]]
Cast2d : public Vertex {
public:

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf = std::is_same<SrcType,float>::value
            && std::is_same<DstType,half>::value;
  static const bool halfFloat = std::is_same<SrcType,half>::value
            && std::is_same<DstType,float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : 1;
  static const unsigned inAlign = ext ? 8 : 1;

  Vector<Input<Vector<SrcType, ONE_PTR, inAlign>>, ONE_PTR> src;
  Vector<Output<Vector<DstType, SPAN, outAlign>>> dst;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    const unsigned limI = dst.size();
    for (unsigned i = 0; i != limI; ++i) {
      const unsigned limJ = dst[i].size();
      auto const &refSrc = src[i];
      auto &refDst = dst[i];
      for (unsigned j = 0; j != limJ; ++j) {
        refDst[j] = static_cast<DstType>(refSrc[j]);
      }
    }
    return true;
  }
};

template class Cast2d<float, float>;
template class Cast2d<float, half>;
template class Cast2d<float, int>;
template class Cast2d<float, bool>;

template class Cast2d<half, float>;
template class Cast2d<half, half>;
template class Cast2d<half, int>;
template class Cast2d<half, bool>;

template class Cast2d<int,float>;
template class Cast2d<int,half>;
template class Cast2d<int,int>;
template class Cast2d<int,bool>;

template class Cast2d<bool,float>;
template class Cast2d<bool,half>;
template class Cast2d<bool,int>;
template class Cast2d<bool,bool>;

template <typename InType>
class Clamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;  // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3;  // upper bound
  Vector<Output<Vector<InType>>> out;

  static const bool ext = std::is_same<InType,float>::value
            || std::is_same<InType,half>::value;
  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {

      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in1[i][j];
        if (out[i][j] < in2[i][j]) {
          out[i][j] = in2[i][j];
        }
        if (out[i][j] > in3[i][j]) {
          out[i][j] = in3[i][j];
        }
      }
    }
    return true;
  }
};

template class Clamp<float>;
template class Clamp<half>;
template class Clamp<int>;

template <typename InType>
class Select : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Vector<Input<Vector<bool,   ONE_PTR>>, ONE_PTR> in3;
  Vector<Output<Vector<InType, SPAN, 4>>> out;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1[i][j] : in2[i][j];
      }
    }
    return true;
  }
};

template class Select<float>;
template class Select<half>;
template class Select<int>;
template class Select<bool>;


template <typename InType>
class ClampInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;  // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3;  // upper bound

  bool compute() {
    for (unsigned i = 0; i != in1Out.size(); ++i) {
      for (unsigned j = 0; j != in1Out[i].size(); ++j) {
        if (in1Out[i][j] < in2[i][j]) {
          in1Out[i][j] = in2[i][j];
        }
        if (in1Out[i][j] > in3[i][j]) {
          in1Out[i][j] = in3[i][j];
        }
      }
    }
    return true;
  }
};

template class ClampInPlace<float>;
template class ClampInPlace<half>;
template class ClampInPlace<int>;

template <typename InType>
class SelectInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Vector<Input<Vector<bool, ONE_PTR>>, ONE_PTR> in3;

  bool compute() {
    for (unsigned i = 0; i != in1Out.size(); ++i) {
      for (unsigned j = 0; j != in1Out[i].size(); ++j) {
        in1Out[i][j] = in3[i][j] ? in1Out[i][j] : in2[i][j];
      }
    }
    return true;
  }
};

template class SelectInPlace<float>;
template class SelectInPlace<half>;
template class SelectInPlace<int>;
template class SelectInPlace<bool>;

}
