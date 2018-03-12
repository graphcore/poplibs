#include <popops/Expr.hpp>

namespace popops::expr {

Expr::~Expr() {}

PlaceHolder _1(1);
PlaceHolder _2(2);
PlaceHolder _3(3);
PlaceHolder _4(4);

template<> void ExprType<Const>::loc() {}
template<> void ExprType<PlaceHolder>::loc() {}
template<> void ExprType<UnaryOp>::loc() {}
template<> void ExprType<BinaryOp>::loc() {}
template<> void ExprType<TernaryOp>::loc() {}

} // popops::expr
