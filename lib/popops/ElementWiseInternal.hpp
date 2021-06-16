// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_ElementWiseInternal_hpp
#define popops_ElementWiseInternal_hpp

#include <poplar/Tensor.hpp>
#include <popops/Expr.hpp>

#include <optional>
#include <vector>

namespace popops {

const poplar::Tensor &
getTensorFromPlaceHolder(const expr::PlaceHolder &p,
                         const std::vector<poplar::Tensor> &ts);

const poplar::Type &
getTypeFromPlaceHolder(const expr::PlaceHolder &p,
                       const std::vector<poplar::Type> &tTypes);

std::unordered_map<const expr::Expr *, poplar::Type>
getConstType(const expr::Expr &expr, const std::vector<poplar::Type> &tTypes);

struct ExprAndType {
  std::unique_ptr<expr::Expr> expression;
  poplar::Type type;
};

// Recursively walk up the expression tree and replace expressions with
// simplified expressions where possible
ExprAndType optimise(const expr::Expr &expr,
                     const std::vector<poplar::Type> &tTypes);

} // end namespace popops

#endif // popops_ElementWiseInternal_hpp
