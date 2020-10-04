/// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef POPLIBS_LIB_POPOPS_CODELETS_ELEMWISEBINARYOPS_HPP_
#define POPLIBS_LIB_POPOPS_CODELETS_ELEMWISEBINARYOPS_HPP_

// Definitions and declarations for generic binary and broadcast binary
// operations codelets (implemented in C++).
//
// Here we define the various operations that can be used in binary and
// broadcast operations, as a structure:
//
//    BinaryOpFn<Op, dataType, architecture>
//
// which has a method "fn()" that performs the operation "Op" (ADD, SUBTRACT,
// etc) between two *single* elements "x" and "y":
//
//    OutputType fn(dataType x, dataType y)
//
// NOTES:
//
// 1. The output type might be different from the data type (for instance for
//    relational operators (EQUAL, GREATER_THAN, etc). This is supported  with
//    the use of the struct "BinaryOpOutputType"
//
// 2. The 'single' elements "x" and "y" could be vector elements, as supported
//    by popc (float2, half4, etc). This is to implement efficient C++ code that
//    uses the hardware vector operations.
//
#include "popops/ExprOp.hpp"
#include "popops/elementwiseCodelets.hpp"

#include "util.hpp"

using namespace poplar;

namespace popops {

namespace {

// Certain operators need to call a different function depending on the input
// type. This deals with dispatching binary ops to their correct functions.
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

template <> struct BinaryLibCall<expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV> {
#ifdef __IPU__
  template <typename FPType> FPType operator()(FPType x, FPType y) const {
    return ipu::rsqrt(x + y);
  }
#else
  template <typename FPType> FPType operator()(FPType x, FPType y) const {
    return 1 / (std::sqrt(x + y));
  }
#endif
};

// This structure uses template specialization to define the output type
// of a binary operation based on the operation itself and the data type.
// In the default case, the output type is the same as the data type (T),
// but for relational (comparison) operators the output type is a boolean,
// or a "long2"/"short4" for vector data types.
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
// that performs that operation on scalar elements
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
DEFINE_BINARY_OP_FN(expr::BinaryOpType::INV_STD_DEV_TO_VARIANCE,
                    return (1 / (x * x)) - y;)

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
DEFINE_BINARY_OP_FN(
    expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV,
    return BinaryLibCall<expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV>{}(x, y);)

// The Binary and BroadcastScalar '1DSupervisor' vertices that output a bool and
// are not vectorised will be created as plain single Worker vertex instead of
// a Supervisor one (despite the name).
// This is because we don't want multiple workers, started by the supervisor,
// each writing < 32 bits (i.e. calling __st8/__st16), which will potentially
// overwrite each others' results.

// Not-in-place vertices
template <expr::BinaryOpType op, typename T>
constexpr bool binaryOp1DIsSupervisor() {
  using OutType = typename BinaryOpOutputType<op, T>::type;

  const bool boolOut = std::is_same<OutType, bool>::value;
  const bool intIn = std::is_same<T, int>::value ||
                     std::is_same<T, bool>::value ||
                     std::is_same<T, unsigned>::value;
  return !(boolOut && intIn);
}

// In-place vertices
template <expr::BinaryOpType op, typename T>
constexpr bool binaryOp1DInPlaceIsSupervisor() {
  using OutType = typename BinaryOpOutputType<op, T>::type;
  return !std::is_same<OutType, bool>::value;
}

} // unnamed namespace

} // namespace popops

#endif /* POPLIBS_LIB_POPOPS_CODELETS_ELEMWISEBINARYOPS_HPP_ */
