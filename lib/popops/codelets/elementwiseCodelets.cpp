#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath>

#include "util.hpp"
#include "popops/ExprOp.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto TWO_PTR = poplar::VectorLayout::TWO_PTR;

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
  template <expr::UnaryOpType op, typename T>
  struct UnaryOpFn {};

#define DEFINE_UNARY_OP_FN(op, body) \
  template <typename T> struct UnaryOpFn<op, T> { \
    static typename UnaryOpOutputType<op, T>::type fn(T x) { body } \
  };

  DEFINE_UNARY_OP_FN(expr::UnaryOpType::ABSOLUTE,
                     if (std::is_integral<T>::value) {
                       return std::abs(x);
                     } else {
                       return std::fabs(x);
                     })
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::BITWISE_NOT, return ~x;)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::CEIL, return std::ceil(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::COS, return std::cos(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::COUNT_LEADING_ZEROS,
                     return x ? __builtin_clz(x) : 32;)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::EXPONENT, return std::exp(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::EXPONENT_MINUS_ONE,
                     return std::expm1(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::FLOOR, return std::floor(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_FINITE,
                     return (x == x) && (std::abs(x) != INFINITY);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::LOGARITHM, return std::log(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::LOGARITHM_ONE_PLUS,
                     return std::log1p(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::LOGICAL_NOT, return !x;)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::NEGATE, return -x;)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::POPCOUNT, return __builtin_popcount(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIGNUM,
                     return (0 < x) - (x < 0);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::SIN,
                     return std::sin(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::TANH,
                     return std::tanh(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::ROUND,
                     return std::round(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQRT,
                     return std::sqrt(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::SQUARE,
                     return (x * x);)
}

template <expr::UnaryOpType op, typename T>
class
[[poplar::constraint("elem(**in) != elem(**out)")]]
UnaryOp : public Vertex {
public:
  Vector<Input<Vector<T, ONE_PTR>>, ONE_PTR> in;
  Vector<Output<Vector<typename UnaryOpOutputType<op, T>::type>>> out;

  bool compute() {
    unsigned limI = out.size();
    for (unsigned i = 0; i != limI; ++i) {
      unsigned limJ = out[i].size();
      auto const &refIn = in[i];
      auto &refOut = out[i];
      for (unsigned j = 0; j != limJ; ++j) {
        refOut[j] = UnaryOpFn<op, T>::fn(refIn[j]);
      }
    }
    return true;
  }
};


template <expr::UnaryOpType op, typename T>
class
UnaryOpInPlace : public Vertex {
public:
  Vector<InOut<Vector<T>>> inOut;

  bool compute() {
    for (auto &row : inOut) {
      const unsigned limJ = row.size();
      for (unsigned j = 0; j != limJ; ++j) {
        row[j] = UnaryOpFn<op, T>::fn(row[j]);
      }
    }
    return true;
  }
};

INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::EXPONENT_MINUS_ONE, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::LOGARITHM_ONE_PLUS, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SQUARE, float, half)

INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::ABSOLUTE, float, half, int)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::BITWISE_NOT, int)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::CEIL, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::COS, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::COUNT_LEADING_ZEROS, int)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::EXPONENT, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOpInPlace, expr::UnaryOpType::SQUARE, float, half)

namespace {
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
  template <expr::BinaryOpType op, typename T>
  struct BinaryOpFn {};

#define DEFINE_BINARY_OP_FN(op, body) \
  template <typename T> struct BinaryOpFn<op, T> { \
    static typename BinaryOpOutputType<op, T>::type fn(T x, T y) { body } \
  };

  DEFINE_BINARY_OP_FN(expr::BinaryOpType::ADD, return x + y;)
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::ATAN2, return std::atan2(x, y);)
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_AND, return x & y;)
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::BITWISE_OR, return x | y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::DIVIDE, return x / y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::EQUAL, return x == y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::GREATER_THAN_EQUAL, return x >= y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::GREATER_THAN, return x > y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::LESS_THAN_EQUAL, return x <= y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::LOGICAL_AND, return x && y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::LOGICAL_OR, return x || y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::LESS_THAN, return x < y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::MAXIMUM, return max(x, y); )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::MINIMUM, return min(x, y); )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::MULTIPLY, return x * y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::NOT_EQUAL, return x != y; )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::POWER, return std::pow(x, y); )
  DEFINE_BINARY_OP_FN(expr::BinaryOpType::REMAINDER,
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
}


template <expr::BinaryOpType op, typename T>
class
[[poplar::constraint("elem(**in1) != elem(**in2)",
                     "elem(**in2) != elem(**out)",
                     "elem(**in1) != elem(**out)",
                     "upper(**in1) || upper(**in2)")]]
BinaryOp : public Vertex {
public:
  Vector<Input<Vector<T, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<T, ONE_PTR>>, ONE_PTR> in2;
  Vector<Output<Vector<typename BinaryOpOutputType<op, T>::type>>> out;

  bool compute() {
    const unsigned limI = out.size();
    for (unsigned i = 0; i != limI; ++i) {
      const unsigned limJ = out[i].size();
      auto const &refIn1 = in1[i];
      auto const &refIn2 = in2[i];
      auto &refOut = out[i];
      for (unsigned j = 0; j != limJ; ++j) {
        refOut[j] = BinaryOpFn<op, T>::fn(refIn1[j], refIn2[j]);
      }
    }
    return true;
  }
};

template <expr::BinaryOpType op, typename T>
class
[[poplar::constraint("elem(**in2) != elem(**in1Out)")]]
BinaryOpInPlace : public Vertex {
public:
  Vector<InOut<Vector<typename BinaryOpOutputType<op, T>::type, TWO_PTR, 1,
         true>>> in1Out;
  Vector<Input<Vector<T, ONE_PTR>>, ONE_PTR> in2;

  bool compute() {
    const unsigned limI = in1Out.size();
    for (unsigned i = 0; i != limI; ++i) {
      const unsigned limJ = in1Out[i].size();
      auto const &refIn2 = in2[i];
      auto &refIn1Out = in1Out[i];
      for (unsigned j = 0; j != limJ; ++j) {
        refIn1Out[j] = BinaryOpFn<op, T>::fn(refIn1Out[j], refIn2[j]);
      }
    }
    return true;
  }
};

INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::ADD, float, half, int, unsigned)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::BITWISE_AND, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::BITWISE_OR, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::DIVIDE, float, half, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::EQUAL, float, half, bool, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::GREATER_THAN_EQUAL,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::GREATER_THAN,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::LESS_THAN_EQUAL,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::LESS_THAN,
               float, half, int, bool)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::MAXIMUM, float, half, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::MINIMUM, float, half, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::MULTIPLY, float, half, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::NOT_EQUAL, float, half, int, bool)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::REMAINDER, float, half, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::SHIFT_RIGHT, int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int)
INSTANTIATE_OP(BinaryOp, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)


INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::ADD, float, half, int,
               unsigned)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::ATAN2, float, half)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::BITWISE_AND, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::BITWISE_OR, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::DIVIDE, float, half, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::EQUAL, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::GREATER_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::GREATER_THAN, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::LESS_THAN_EQUAL, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::LOGICAL_AND, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::LOGICAL_OR, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::LESS_THAN, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::MAXIMUM, float, half, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::MINIMUM, float, half, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::MULTIPLY, float, half, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::NOT_EQUAL, bool)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::POWER, float, half)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::REMAINDER, float, half, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::SHIFT_LEFT, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::SHIFT_RIGHT, int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND,
               int)
INSTANTIATE_OP(BinaryOpInPlace, expr::BinaryOpType::SUBTRACT,
               float, half, int, unsigned)



template <typename InType>
class
[[poplar::constraint("elem(*data) != elem(*deltas)")]]
ScaledAddSupervisor : public SupervisorVertex {
public:
  InOut<Vector<InType>> data;
  Input<Vector<InType, ONE_PTR>> deltas;
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
  Vector<InOut<Vector<InType>>> data;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> deltas;
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
  Input<Vector<SrcType, ONE_PTR>> src;
  Output<Vector<DstType>> dst;

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
  Vector<Input<Vector<SrcType, ONE_PTR>>, ONE_PTR> src;
  Vector<Output<Vector<DstType>>> dst;

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
  Vector<Input<Vector<bool>>> in3;
  Vector<Output<Vector<InType>>> out;

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
