// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef __popops_Expr_hpp__
#define __popops_Expr_hpp__
#include <memory>
#include <poplar/TypeTraits.hpp>
#include <cassert>
#include <type_traits>
#include <popops/ExprOp.hpp>

namespace popops { namespace expr {

/** Type to represent element expressions.
 *
 *  This class represents an expression that can be applied to elements
 *  of tensors.
 *
 *  The type is an abstract type which can be instatiated by its sub-classes
 *  to build up expressions e.g. `Tanh(Add(Square(_1), Const(3))))`.
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

  template <class T>
  bool isA() const { return classId == T::getClassId(); }

  template <class T>
  T *getAs() {
    if (!isA<T>())
      return 0;
    return static_cast<T *>(this);
  }

  template <class T>
  const T *getAs() const {
    if (!isA<T>())
      return 0;
    return static_cast<const T *>(this);
  }

  virtual std::unique_ptr<Expr> clone() const = 0;
};

template <class T>
class ExprType : public Expr {
  static void loc();
  static ExprClassID getClassId() { return &loc; }
public:
  ExprType() : Expr(getClassId()) {}
  friend class Expr;
};

class Const : public ExprType<Const> {
  poplar::TypeTraits typeTraits;
  std::unique_ptr<char[]> data;
public:
  template <typename T>
  Const(T x) {
    static_assert(std::is_integral<T>::value ||
                  std::is_floating_point<T>::value,
                  "Constant expression values should be integrals or floats");
    typeTraits = poplar::TypeTraits::make<T>();
    data.reset(new char[typeTraits.size]);
    const char *p = reinterpret_cast<const char *>(&x);
    std::copy(p, p + typeTraits.size, data.get());
  }
  Const(poplar::TypeTraits typeTraits_,
        const char *data_) : typeTraits(std::move(typeTraits_)) {
    data.reset(new char[typeTraits.size]);
    std::copy(data_, data_ + typeTraits.size, data.get());
  }

  char *getData() const { return data.get(); }

  const poplar::TypeTraits &getTypeTraits() const { return typeTraits; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new Const(typeTraits, data.get()));
  }

};

class PlaceHolder : public ExprType<PlaceHolder> {
  unsigned index;
public:
  PlaceHolder(unsigned index) : index(index) {}

  unsigned getIndex() const { return index; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new PlaceHolder(index));
  }
};

extern PlaceHolder _1;
extern PlaceHolder _2;
extern PlaceHolder _3;
extern PlaceHolder _4;

class UnaryOp : public ExprType<UnaryOp> {
  UnaryOpType type;
  std::unique_ptr<Expr> a;
public:
  UnaryOp(UnaryOpType type, const Expr &a) : type(type), a(a.clone()) {}

  UnaryOpType getOpType() const { return type; }

  const Expr &getArg() const { return *a; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new UnaryOp(type, *a));
  }
};

#define POPLIBS_DEFINE_EXPR_UNARY_OP(Name, Op) \
class Name : public UnaryOp { \
  public: Name(const Expr &a) : UnaryOp(UnaryOpType::Op, a) {} \
};

POPLIBS_DEFINE_EXPR_UNARY_OP(Abs, ABSOLUTE)
POPLIBS_DEFINE_EXPR_UNARY_OP(BitwiseNot, BITWISE_NOT)
POPLIBS_DEFINE_EXPR_UNARY_OP(Ceil, CEIL)
POPLIBS_DEFINE_EXPR_UNARY_OP(Cos, COS)
POPLIBS_DEFINE_EXPR_UNARY_OP(Exp, EXPONENT)
POPLIBS_DEFINE_EXPR_UNARY_OP(Expm1, EXPONENT_MINUS_ONE)
POPLIBS_DEFINE_EXPR_UNARY_OP(Floor, FLOOR)
POPLIBS_DEFINE_EXPR_UNARY_OP(IsFinite, IS_FINITE)
POPLIBS_DEFINE_EXPR_UNARY_OP(Log, LOGARITHM)
POPLIBS_DEFINE_EXPR_UNARY_OP(Log1p, LOGARITHM_ONE_PLUS)
POPLIBS_DEFINE_EXPR_UNARY_OP(Not, LOGICAL_NOT)
POPLIBS_DEFINE_EXPR_UNARY_OP(Neg, NEGATE)
POPLIBS_DEFINE_EXPR_UNARY_OP(Signum, SIGNUM)
POPLIBS_DEFINE_EXPR_UNARY_OP(Sin, SIN)
POPLIBS_DEFINE_EXPR_UNARY_OP(Tanh, TANH)
POPLIBS_DEFINE_EXPR_UNARY_OP(Round, ROUND)
POPLIBS_DEFINE_EXPR_UNARY_OP(Sqrt, SQRT)
POPLIBS_DEFINE_EXPR_UNARY_OP(Square, SQUARE)

class BinaryOp : public ExprType<BinaryOp> {
  BinaryOpType type;
  std::unique_ptr<Expr> a, b;
public:
  BinaryOp(BinaryOpType type, const Expr &a, const Expr &b) :
    type(type), a(a.clone()), b(b.clone()) {}

  BinaryOpType getOpType() const { return type; }

  const Expr &getLHS() const { return *a; }
  const Expr &getRHS() const { return *b; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new BinaryOp(type, *a, *b));
  }
};

#define POPLIBS_DEFINE_EXPR_BINARY_OP(Name, Op) \
class Name : public BinaryOp { \
  public: \
  Name(const Expr &a, const Expr &b) : \
    BinaryOp(BinaryOpType::Op, a, b) {} \
};

POPLIBS_DEFINE_EXPR_BINARY_OP(Add, ADD)
POPLIBS_DEFINE_EXPR_BINARY_OP(Atan2, ATAN2)
POPLIBS_DEFINE_EXPR_BINARY_OP(BitwiseAnd, BITWISE_AND)
POPLIBS_DEFINE_EXPR_BINARY_OP(BitwiseOr, BITWISE_OR)
POPLIBS_DEFINE_EXPR_BINARY_OP(Divide, DIVIDE)
POPLIBS_DEFINE_EXPR_BINARY_OP(Equal, EQUAL)
POPLIBS_DEFINE_EXPR_BINARY_OP(Gte, GREATER_THAN_EQUAL)
POPLIBS_DEFINE_EXPR_BINARY_OP(Gt, GREATER_THAN)
POPLIBS_DEFINE_EXPR_BINARY_OP(Lte, LESS_THAN_EQUAL)
POPLIBS_DEFINE_EXPR_BINARY_OP(And, LOGICAL_AND)
POPLIBS_DEFINE_EXPR_BINARY_OP(Or, LOGICAL_OR)
POPLIBS_DEFINE_EXPR_BINARY_OP(Lt, LESS_THAN)
POPLIBS_DEFINE_EXPR_BINARY_OP(Max, MAXIMUM)
POPLIBS_DEFINE_EXPR_BINARY_OP(Min, MINIMUM)
POPLIBS_DEFINE_EXPR_BINARY_OP(Mul, MULTIPLY)
POPLIBS_DEFINE_EXPR_BINARY_OP(NotEqual, NOT_EQUAL)
POPLIBS_DEFINE_EXPR_BINARY_OP(Pow, POWER)
POPLIBS_DEFINE_EXPR_BINARY_OP(Rem, REMAINDER)
POPLIBS_DEFINE_EXPR_BINARY_OP(Shl, SHIFT_LEFT)
POPLIBS_DEFINE_EXPR_BINARY_OP(Shr, SHIFT_RIGHT)
POPLIBS_DEFINE_EXPR_BINARY_OP(ShrSE, SHIFT_RIGHT_SIGN_EXTEND)
POPLIBS_DEFINE_EXPR_BINARY_OP(Sub, SUBTRACT)

class TernaryOp : public ExprType<TernaryOp> {
  TernaryOpType type;
  std::unique_ptr<Expr> a, b, c;
public:
  TernaryOp(TernaryOpType type, const Expr &a, const Expr &b, const Expr &c) :
    type(type), a(a.clone()), b(b.clone()), c(c.clone()) {}

  TernaryOpType getOpType() const { return type; }

  const Expr &getArg0() const { return *a; }
  const Expr &getArg1() const { return *b; }
  const Expr &getArg2() const { return *c; }

  std::unique_ptr<Expr> clone() const override {
    return std::unique_ptr<Expr>(new TernaryOp(type, *a, *b, *c));
  }
};

#define POPLIBS_DEFINE_EXPR_TERNARY_OP(Name, Op) \
class Name : public TernaryOp { \
  public: \
  Name(const Expr &a, const Expr &b, const Expr &c) : \
    TernaryOp(TernaryOpType::Op, a, b, c) {} \
};

POPLIBS_DEFINE_EXPR_TERNARY_OP(Select, SELECT)
POPLIBS_DEFINE_EXPR_TERNARY_OP(Clamp, CLAMP)

}} // end namespace popops::expr

#endif // __popops_Expr_hpp__
