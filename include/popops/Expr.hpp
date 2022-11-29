// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Expressions with elements of tensors.
 *
 */

#ifndef __popops_Expr_hpp__
#define __popops_Expr_hpp__
#include <cassert>
#include <memory>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplar/TypeTraits.hpp>
#include <popops/ExprOp.hpp>
#include <string>
#include <type_traits>
#include <vector>

namespace popops {
namespace expr {

/** Type to represent element expressions.
 *
 *  This class represents an expression that can be applied to elements
 *  of tensors.
 *
 *  The Expr type is an abstract type which can be instantiated by its
 *  sub-classes to build up expressions, for example:
 *  `Tanh(Add(Square(_1), Const(3))))`.
 *
 *  Expressions can be applied to tensors with the popops::map() and
 *  popops::mapInPlace() functions.
 *
 */
class Expr {
protected:
  using ExprClassID = void (*)(void);
  ExprClassID classId;
  Expr(ExprClassID classId) : classId(classId) {}

public:
  virtual ~Expr();

  template <class T> bool isA() const { return classId == T::getClassId(); }

  template <class T> T *getAs() {
    if (!isA<T>())
      return 0;
    return static_cast<T *>(this);
  }

  template <class T> const T *getAs() const {
    if (!isA<T>())
      return 0;
    return static_cast<const T *>(this);
  }

  virtual std::unique_ptr<Expr> clone() const = 0;

  virtual std::string name(const std::vector<poplar::Tensor> &) const = 0;

  virtual bool deepEquals(const Expr &other) const = 0;

  virtual void print(std::ostream &os, unsigned indent = 0,
                     bool prettyPrint = true) const = 0;
};

std::ostream &operator<<(std::ostream &os, const Expr &expr);
bool deepEquals(const Expr &a, const Expr &b);

template <class T> class ExprType : public Expr {
  static void loc();
  static ExprClassID getClassId() { return &loc; }

public:
  ExprType() : Expr(getClassId()) {}
  friend class Expr;
};

/// A class that can contain any expression, useful for building up expression
/// trees dynamically where the type of the outermost expression may change.
class Any {
  std::unique_ptr<Expr> expr;

public:
  Any(const Expr &expr) : expr(expr.clone()) {}

  Any(const Any &expr) : expr(expr.expr->clone()) {}

  Any(Any &&) = default;

  Any &operator=(const Any &other) {
    expr = other.expr->clone();
    return *this;
  }

  Any &operator=(Any &&other) = default;

  operator Expr &() { return *expr; }
  operator const Expr &() const { return *expr; }
  std::string name(const std::vector<poplar::Tensor> &inputs) const {
    return expr->name(inputs);
  }
};

/// A class to represent constant expressions.
class Const : public ExprType<Const> {
  poplar::TypeTraits typeTraits;
  poplar::Type type;
  std::unique_ptr<char[]> data;

protected:
  template <typename T> Const(T x, bool isHalfType) {
    static_assert(std::is_integral<T>::value ||
                      std::is_floating_point<T>::value,
                  "Constant expression values should be integrals or floats");
    typeTraits = poplar::TypeTraits::make<T>();
    if (isHalfType) {
      type = poplar::HALF;
    } else {
      type = poplar::equivalent_device_type<T>().value;
    }
    data.reset(new char[typeTraits.size]);
    const char *p = reinterpret_cast<const char *>(&x);
    std::copy(p, p + typeTraits.size, data.get());
  }

public:
  template <typename T, typename = typename std::enable_if<
                            poplar::TypeTraits::isSimpleType<T>(), T>::type>
  Const(T x) : Const(x, false) {}

  Const(poplar::TypeTraits typeTraits_, poplar::Type type_, const char *data_)
      : typeTraits(std::move(typeTraits_)), type(type_) {
    data.reset(new char[typeTraits.size]);
    std::copy(data_, data_ + typeTraits.size, data.get());
  }
  Const(Const &&) = default;
  Const &operator=(Const &&) = default;
  Const(const Const &other)
      : Const(other.typeTraits, other.type, other.data.get()) {}
  Const &operator=(const Const &other) {
    Const tmp{other};
    std::swap(*this, tmp);
    return *this;
  }

  char *getData() const { return data.get(); }

  const poplar::TypeTraits &getTypeTraits() const { return typeTraits; }

  const poplar::Type &getType() const { return type; }

  std::string printValue() const;

  double getDataAsDouble() const;

  std::uint64_t getDataForUnsignedIntegral() const;

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new Const(typeTraits, type, data.get()));
  }
  std::string name(const std::vector<poplar::Tensor> &) const override;

  bool deepEquals(const Expr &other) const override;

  void print(std::ostream &os, unsigned indent = 0,
             bool prettyPrint = true) const override;
};

/// A class to represent constant expressions of type \c half.
class ConstHalf : public Const {
public:
  ConstHalf(float x) : Const(x, true) {}
  ConstHalf(ConstHalf &&) = default;
  ConstHalf &operator=(ConstHalf &&) = default;
  ConstHalf(const ConstHalf &other) : Const(other) {}
  ConstHalf &operator=(const ConstHalf &other) {
    ConstHalf tmp{other};
    std::swap(*this, tmp);
    return *this;
  }
};

inline ConstHalf operator"" _half(long double x) {
  assert(x <= std::numeric_limits<float>::max());
  return ConstHalf(static_cast<float>(x));
}

/// A class to represent cast expressions.
class Cast : public ExprType<Cast> {
  std::unique_ptr<Expr> a;
  poplar::Type bType;

public:
  Cast(const Expr &a_, const poplar::Type bType_)
      : a(a_.clone()), bType(bType_) {}
  Cast(Cast &&other) = default;
  Cast &operator=(Cast &&other) = default;
  Cast(const Cast &other) : Cast(*other.a, other.bType) {}
  Cast &operator==(const Cast &other) {
    Cast tmp{other};
    std::swap(*this, tmp);
    return *this;
  }

  const Expr &getLHS() const { return *a; }
  const poplar::Type &getRHSType() const { return bType; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new Cast(*a, bType));
  }
  std::string name(const std::vector<poplar::Tensor> &inputs) const override;

  bool deepEquals(const Expr &other) const override;

  void print(std::ostream &os, unsigned indent = 0,
             bool prettyPrint = true) const override;
};

class PlaceHolder : public ExprType<PlaceHolder> {
  unsigned index;

public:
  PlaceHolder(unsigned index) : index(index) {}
  PlaceHolder(PlaceHolder &&) = default;
  PlaceHolder &operator=(PlaceHolder &&) = default;
  PlaceHolder(const PlaceHolder &other) : PlaceHolder(other.index) {}
  PlaceHolder &operator=(const PlaceHolder &other) {
    PlaceHolder tmp{other};
    std::swap(*this, tmp);
    return *this;
  }

  unsigned getIndex() const { return index; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new PlaceHolder(index));
  }
  std::string name(const std::vector<poplar::Tensor> &inputs) const override;

  bool deepEquals(const Expr &other) const override;

  void print(std::ostream &os, unsigned indent = 0,
             bool prettyPrint = true) const override;
};

const PlaceHolder _1(1);
const PlaceHolder _2(2);
const PlaceHolder _3(3);
const PlaceHolder _4(4);
const PlaceHolder _5(5);
const PlaceHolder _6(6);
const PlaceHolder _7(7);
const PlaceHolder _8(8);
const PlaceHolder _9(9);
const PlaceHolder _10(10);
const PlaceHolder _11(11);
const PlaceHolder _12(12);
const PlaceHolder _13(13);
const PlaceHolder _14(14);
const PlaceHolder _15(15);
const PlaceHolder _16(16);
const PlaceHolder _17(17);
const PlaceHolder _18(18);
const PlaceHolder _19(19);
const PlaceHolder _20(20);

/// A class to represent expressions with unary operators.
class UnaryOp : public ExprType<UnaryOp> {
  UnaryOpType type;
  std::unique_ptr<Expr> a;

public:
  UnaryOp(UnaryOpType type, const Expr &a) : type(type), a(a.clone()) {}
  UnaryOp(UnaryOp &&) = default;
  UnaryOp &operator=(UnaryOp &&) = default;
  UnaryOp(const UnaryOp &other) : UnaryOp(other.type, *other.a) {}
  UnaryOp &operator=(const UnaryOp &other) {
    UnaryOp tmp(other);
    std::swap(*this, tmp);
    return *this;
  }

  UnaryOpType getOpType() const { return type; }

  const Expr &getArg() const { return *a; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new UnaryOp(type, *a));
  }
  std::string name(const std::vector<poplar::Tensor> &inputs) const override;
  std::string exprName(const std::vector<poplar::Tensor> &inputs) const {
    return a->name(inputs);
  };

  bool deepEquals(const Expr &other) const override;

  void print(std::ostream &os, unsigned indent = 0,
             bool prettyPrint = true) const override;
};

#define POPLIBS_DEFINE_EXPR_UNARY_OP(Name, Op)                                 \
  class Name : public UnaryOp {                                                \
  public:                                                                      \
    Name(const Expr &a) : UnaryOp(UnaryOpType::Op, a) {}                       \
  };

#define POPLIBS_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(Name, Op, Sym)                 \
  POPLIBS_DEFINE_EXPR_UNARY_OP(Name, Op)                                       \
  inline Name operator Sym(const Expr &a) { return Name(a); }

POPLIBS_DEFINE_EXPR_UNARY_OP(Abs, ABSOLUTE)
POPLIBS_DEFINE_EXPR_UNARY_OP(Asin, ASIN)
POPLIBS_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(BitwiseNot, BITWISE_NOT, ~)
POPLIBS_DEFINE_EXPR_UNARY_OP(Cbrt, CBRT)
POPLIBS_DEFINE_EXPR_UNARY_OP(Erf, ERF)
POPLIBS_DEFINE_EXPR_UNARY_OP(Ceil, CEIL)
POPLIBS_DEFINE_EXPR_UNARY_OP(Cos, COS)
POPLIBS_DEFINE_EXPR_UNARY_OP(Exp, EXPONENT)
POPLIBS_DEFINE_EXPR_UNARY_OP(Expm1, EXPONENT_MINUS_ONE)
POPLIBS_DEFINE_EXPR_UNARY_OP(Exp2, EXPONENT2)
POPLIBS_DEFINE_EXPR_UNARY_OP(Floor, FLOOR)
POPLIBS_DEFINE_EXPR_UNARY_OP(GeluErf, GELU_ERF)
POPLIBS_DEFINE_EXPR_UNARY_OP(Inv, INVERSE)
POPLIBS_DEFINE_EXPR_UNARY_OP(IsFinite, IS_FINITE)
POPLIBS_DEFINE_EXPR_UNARY_OP(IsInf, IS_INF)
POPLIBS_DEFINE_EXPR_UNARY_OP(IsNaN, IS_NAN)
POPLIBS_DEFINE_EXPR_UNARY_OP(Log, LOGARITHM)
POPLIBS_DEFINE_EXPR_UNARY_OP(Log1p, LOGARITHM_ONE_PLUS)
POPLIBS_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(Not, LOGICAL_NOT, !)
POPLIBS_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(Neg, NEGATE, -)
POPLIBS_DEFINE_EXPR_UNARY_OP(Signum, SIGNUM)
POPLIBS_DEFINE_EXPR_UNARY_OP(Sin, SIN)
POPLIBS_DEFINE_EXPR_UNARY_OP(Tan, TAN)
POPLIBS_DEFINE_EXPR_UNARY_OP(Tanh, TANH)
POPLIBS_DEFINE_EXPR_UNARY_OP(Round, ROUND)
POPLIBS_DEFINE_EXPR_UNARY_OP(Trunc, TRUNC)
POPLIBS_DEFINE_EXPR_UNARY_OP(Sqrt, SQRT)
POPLIBS_DEFINE_EXPR_UNARY_OP(Square, SQUARE)
POPLIBS_DEFINE_EXPR_UNARY_OP(Sigmoid, SIGMOID)
POPLIBS_DEFINE_EXPR_UNARY_OP(Rsqrt, RSQRT)

/// A class to represent expressions with binary operators.
class BinaryOp : public ExprType<BinaryOp> {
  BinaryOpType type;
  std::unique_ptr<Expr> a, b;

public:
  BinaryOp(BinaryOpType type, const Expr &a, const Expr &b)
      : type(type), a(a.clone()), b(b.clone()) {}
  BinaryOp(BinaryOp &&) = default;
  BinaryOp &operator=(BinaryOp &&) = default;
  BinaryOp(const BinaryOp &other) : BinaryOp(other.type, *other.a, *other.b) {}
  BinaryOp &operator=(const BinaryOp &other) {
    BinaryOp tmp{other};
    std::swap(*this, tmp);
    return *this;
  }

  BinaryOpType getOpType() const { return type; }

  const Expr &getLHS() const { return *a; }
  const Expr &getRHS() const { return *b; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new BinaryOp(type, *a, *b));
  }
  std::string name(const std::vector<poplar::Tensor> &inputs) const override;
  std::string exprName(const std::vector<poplar::Tensor> &inputs) const {
    return a->name(inputs) + "_" + b->name(inputs);
  }

  bool deepEquals(const Expr &other) const override;

  void print(std::ostream &os, unsigned indent = 0,
             bool prettyPrint = true) const override;
};

#define POPLIBS_DEFINE_EXPR_BINARY_OP(Name, Op)                                \
  class Name : public BinaryOp {                                               \
  public:                                                                      \
    Name(const Expr &a, const Expr &b) : BinaryOp(BinaryOpType::Op, a, b) {}   \
  };

#define POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Name, Op, Sym)                \
  POPLIBS_DEFINE_EXPR_BINARY_OP(Name, Op)                                      \
  template <typename T>                                                        \
  inline typename std::enable_if<!std::is_base_of<Expr, T>::value &&           \
                                     poplar::TypeTraits::isSimpleType<T>(),    \
                                 Name>::type                                   \
  operator Sym(const T &a, const Expr &b) {                                    \
    return Name(Const(a), b);                                                  \
  }                                                                            \
  template <typename T>                                                        \
  inline typename std::enable_if<!std::is_base_of<Expr, T>::value &&           \
                                     poplar::TypeTraits::isSimpleType<T>(),    \
                                 Name>::type                                   \
  operator Sym(const Expr &a, const T &b) {                                    \
    return Name(a, Const(b));                                                  \
  }                                                                            \
  inline Name operator Sym(const Expr &a, const Expr &b) { return Name(a, b); }

POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Add, ADD, +)
POPLIBS_DEFINE_EXPR_BINARY_OP(Atan2, ATAN2)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseAnd, BITWISE_AND, &)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseOr, BITWISE_OR, |)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseXor, BITWISE_XOR, ^)
POPLIBS_DEFINE_EXPR_BINARY_OP(BitwiseXnor, BITWISE_XNOR)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Divide, DIVIDE, /)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Equal, EQUAL, ==)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Gte, GREATER_THAN_EQUAL, >=)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Gt, GREATER_THAN, >)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Lte, LESS_THAN_EQUAL, <=)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(And, LOGICAL_AND, &&)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Or, LOGICAL_OR, ||)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Lt, LESS_THAN, <)
POPLIBS_DEFINE_EXPR_BINARY_OP(InvStdDevToVariance, INV_STD_DEV_TO_VARIANCE)
POPLIBS_DEFINE_EXPR_BINARY_OP(Max, MAXIMUM)
POPLIBS_DEFINE_EXPR_BINARY_OP(Min, MINIMUM)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Mul, MULTIPLY, *)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(NotEqual, NOT_EQUAL, !=)
POPLIBS_DEFINE_EXPR_BINARY_OP(Pow, POWER)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Rem, REMAINDER, %)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Shl, SHIFT_LEFT, <<)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Shr, SHIFT_RIGHT, >>)
POPLIBS_DEFINE_EXPR_BINARY_OP(ShrSE, SHIFT_RIGHT_SIGN_EXTEND)
POPLIBS_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Sub, SUBTRACT, -)
POPLIBS_DEFINE_EXPR_BINARY_OP(VarianceToInvStdDev, VARIANCE_TO_INV_STD_DEV)

/// A class to represent expressions with ternary operators.
class TernaryOp : public ExprType<TernaryOp> {
  TernaryOpType type;
  std::unique_ptr<Expr> a, b, c;

public:
  TernaryOp(TernaryOpType type, const Expr &a, const Expr &b, const Expr &c)
      : type(type), a(a.clone()), b(b.clone()), c(c.clone()) {}
  TernaryOp(TernaryOp &&) = default;
  TernaryOp &operator=(TernaryOp &&) = default;
  TernaryOp(const TernaryOp &other)
      : TernaryOp(other.type, *other.a, *other.b, *other.c) {}
  TernaryOp &operator=(const TernaryOp &other) {
    TernaryOp tmp{other};
    std::swap(*this, tmp);
    return *this;
  }

  TernaryOpType getOpType() const { return type; }

  const Expr &getArg0() const { return *a; }
  const Expr &getArg1() const { return *b; }
  const Expr &getArg2() const { return *c; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new TernaryOp(type, *a, *b, *c));
  }
  std::string name(const std::vector<poplar::Tensor> &inputs) const override;
  std::string exprName(const std::vector<poplar::Tensor> &inputs) const {
    return a->name(inputs) + "_" + b->name(inputs) + "_" + c->name(inputs);
  }

  bool deepEquals(const Expr &other) const override;

  void print(std::ostream &os, unsigned indent = 0,
             bool prettyPrint = true) const override;
};

#define POPLIBS_DEFINE_EXPR_TERNARY_OP(Name, Op)                               \
  class Name : public TernaryOp {                                              \
  public:                                                                      \
    Name(const Expr &a, const Expr &b, const Expr &c)                          \
        : TernaryOp(TernaryOpType::Op, a, b, c) {}                             \
  };

/** Computes the conditional ternary operation
 *
 * ```
 * c ? a : b
 * ```
 */
POPLIBS_DEFINE_EXPR_TERNARY_OP(Select, SELECT)
POPLIBS_DEFINE_EXPR_TERNARY_OP(Clamp, CLAMP)

} // namespace expr
} // namespace popops

#endif // __popops_Expr_hpp__
