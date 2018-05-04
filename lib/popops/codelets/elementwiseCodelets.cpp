#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath>

#include "util.hpp"
#include "popops/ExprOp.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

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
                     if (x >= 0) {
                       return x;
                     } else {
                       return -x;
                     })
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::BITWISE_NOT, return ~x;)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::CEIL, return std::ceil(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::COS, return std::cos(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::COUNT_LEADING_ZEROS,
                     return x ? __builtin_clz(x) : 32;)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::EXPONENT, return std::exp(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::FLOOR, return std::floor(x);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::IS_FINITE,
                     return (x == x) && (std::abs(x) != INFINITY);)
  DEFINE_UNARY_OP_FN(expr::UnaryOpType::LOGARITHM, return std::log(x);)
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
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = UnaryOpFn<op, T>::fn(in[i][j]);
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
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::FLOOR, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::IS_FINITE, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::LOGARITHM, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::LOGICAL_NOT, bool)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::NEGATE, float, half, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::POPCOUNT, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SIGNUM, float, half, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SIN, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::TANH, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::ROUND, float, half)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SQRT, float, half, int)
INSTANTIATE_OP(UnaryOp, expr::UnaryOpType::SQUARE, float, half)

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

    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = BinaryOpFn<op, T>::fn(in1[i][j], in2[i][j]);
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

template <typename InType>
class
[[poplar::constraint("elem(*data) != elem(*deltas)")]]
ScaledAdd : public Vertex {
public:
  InOut<Vector<InType>> data;
  Input<Vector<InType, ONE_PTR>> deltas;
  InType K;

  bool compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      data[i] += K * deltas[i];
    }
    return true;
  }
};

template class ScaledAdd<float>;
template class ScaledAdd<half>;
template class ScaledAdd<int>;
template class ScaledAdd<unsigned>;

template <typename InType>
class
[[poplar::constraint("elem(**data) != elem(**deltas)")]]
ScaledAdd2D : public Vertex {
public:
  Vector<InOut<Vector<InType>>> data;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> deltas;
  InType K;

  bool compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      for (unsigned j = 0; j < data[i].size(); ++j) {
        data[i][j] += K * deltas[i][j];
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
    for (unsigned i = 0; i < A.size(); ++i) {
      for (unsigned j = 0; j < A[i].size(); ++j) {
        A[i][j] *= B[i][j];
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
    for (unsigned i = 0; i < out.size(); ++i) {
      out[i] = 0;
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
    for (unsigned i = 0; i < dst.size(); ++i) {
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
    for (unsigned i = 0; i != dst.size(); ++i) {
      for (unsigned j = 0; j != dst[i].size(); ++j) {
        dst[i][j] = static_cast<DstType>(src[i][j]);
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

}
