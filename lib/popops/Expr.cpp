#include <popops/Expr.hpp>

namespace popops {
namespace expr {

Expr::~Expr() {}

PlaceHolder _1(1);
PlaceHolder _2(2);
PlaceHolder _3(3);
PlaceHolder _4(4);
PlaceHolder _5(5);
PlaceHolder _6(6);
PlaceHolder _7(7);
PlaceHolder _8(8);
PlaceHolder _9(9);
PlaceHolder _10(10);
PlaceHolder _11(11);
PlaceHolder _12(12);
PlaceHolder _13(13);
PlaceHolder _14(14);
PlaceHolder _15(15);
PlaceHolder _16(16);
PlaceHolder _17(17);
PlaceHolder _18(18);
PlaceHolder _19(19);
PlaceHolder _20(20);

template<> void ExprType<Const>::loc() {}
template<> void ExprType<Cast>::loc() {}
template<> void ExprType<PlaceHolder>::loc() {}
template<> void ExprType<UnaryOp>::loc() {}
template<> void ExprType<BinaryOp>::loc() {}
template<> void ExprType<TernaryOp>::loc() {}

}} // popops::expr
