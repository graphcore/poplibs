// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "ElementWiseInternal.hpp"

#include "ExprOpUtil.hpp"
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/logging.hpp>
#include <popops/Expr.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poputil;
using namespace poplibs_support;

namespace popops {

static inline bool isRelational(expr::UnaryOpType op) {
  switch (op) {
  case expr::UnaryOpType::IS_FINITE:
  case expr::UnaryOpType::IS_INF:
  case expr::UnaryOpType::IS_NAN:
    return true;
  default:
    return false;
  }
}

static inline bool isRelational(expr::BinaryOpType op) {
  switch (op) {
  case expr::BinaryOpType::EQUAL:
  case expr::BinaryOpType::GREATER_THAN_EQUAL:
  case expr::BinaryOpType::GREATER_THAN:
  case expr::BinaryOpType::LESS_THAN_EQUAL:
  case expr::BinaryOpType::LESS_THAN:
  case expr::BinaryOpType::NOT_EQUAL:
    return true;
  default:
    return false;
  }
}

static inline bool isLogical(expr::UnaryOpType op) {
  switch (op) {
  case expr::UnaryOpType::LOGICAL_NOT:
    return true;
  default:
    return false;
  }
}

static inline bool isLogical(expr::BinaryOpType op) {
  switch (op) {
  case expr::BinaryOpType::LOGICAL_AND:
  case expr::BinaryOpType::LOGICAL_OR:
    return true;
  default:
    return false;
  }
}

static inline std::optional<Type> getTypeFromConst(const expr::Expr &expr) {
  const auto *constExpr = expr.getAs<expr::Const>();
  if (!constExpr)
    return std::nullopt;
  return constExpr->getType();
}

const Tensor &getTensorFromPlaceHolder(const expr::PlaceHolder &p,
                                       const std::vector<Tensor> &ts) {
  auto index = p.getIndex() - 1;
  if (index > ts.size()) {
    throw poplibs_error("Invalid placeholder _" + std::to_string(index + 1) +
                        " in expression");
  }
  return ts[index];
}

const Type &getTypeFromPlaceHolder(const expr::PlaceHolder &p,
                                   const std::vector<Type> &tTypes) {
  auto index = p.getIndex() - 1;
  if (index > tTypes.size()) {
    throw poplibs_error("Invalid placeholder _" + std::to_string(index + 1) +
                        " in expression");
  }
  return tTypes[index];
}

static std::optional<Type>
inferType(const expr::Expr &expr, const std::vector<Type> &tTypes,
          std::unordered_map<const expr::Expr *, Type> &constTypes,
          std::vector<const expr::Expr *> &unknown) {
  if (expr.isA<expr::Const>()) {
    unknown.push_back(&expr);
    return {};
  } else if (const expr::Cast *cast = expr.getAs<expr::Cast>()) {
    std::vector<const expr::Expr *> subExprUnknown;
    static_cast<void>(
        inferType(cast->getLHS(), tTypes, constTypes, subExprUnknown));
    if (!subExprUnknown.empty()) {
      std::stringstream errStr;
      errStr
          << "Could not infer the type(s) of some constant(s) in expression ";
      expr.print(errStr, 0, false);
      throw poplibs_error(errStr.str());
    }
    return cast->getRHSType();
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    return getTypeFromPlaceHolder(*p, tTypes);
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    auto opType = u->getOpType();
    bool propagateTypeUp = !isRelational(opType) && !isLogical(opType);
    std::vector<const expr::Expr *> tmp;
    std::vector<const expr::Expr *> &subExprUnknown =
        propagateTypeUp ? unknown : tmp;
    auto argType = inferType(u->getArg(), tTypes, constTypes, subExprUnknown);
    if (!propagateTypeUp) {
      if (!subExprUnknown.empty()) {
        std::stringstream errStr;
        errStr
            << "Could not infer the type(s) of some constant(s) in expression ";
        expr.print(errStr, 0, false);
        throw poplibs_error(errStr.str());
      }
      return BOOL;
    }
    return argType;
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto opType = b->getOpType();
    bool propagateTypeUp = !isRelational(opType) && !isLogical(opType);
    std::vector<const expr::Expr *> tmp;
    std::vector<const expr::Expr *> &subExprUnknown =
        propagateTypeUp ? unknown : tmp;
    auto lhsType = inferType(b->getLHS(), tTypes, constTypes, subExprUnknown);
    auto rhsType = inferType(b->getRHS(), tTypes, constTypes, subExprUnknown);
    if (!lhsType && rhsType) {
      lhsType = rhsType;
      for (const auto e : subExprUnknown)
        constTypes[e] = *rhsType;
      subExprUnknown.clear();
    }
    if (!rhsType && lhsType) {
      rhsType = lhsType;
      for (const auto e : subExprUnknown)
        constTypes[e] = *lhsType;
      subExprUnknown.clear();
    }
    if (lhsType != rhsType) {
      assert(bool(lhsType));
      assert(bool(rhsType));
      std::stringstream errStr;
      errStr << "Inferred type of lhs (" << *lhsType
             << ") does not match inferred type of rhs (" << *rhsType << ") in "
             << "expression ";
      expr.print(errStr, 0, false);
      throw poplibs_error(errStr.str());
    }
    if (!propagateTypeUp) {
      if (!subExprUnknown.empty()) {
        std::stringstream errStr;
        errStr
            << "Could not infer the type(s) of some constant(s) in expression ";
        expr.print(errStr, 0, false);
        throw poplibs_error(errStr.str());
      }
      return BOOL;
    }
    return lhsType;
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto opType = t->getOpType();
    if (opType == expr::TernaryOpType::SELECT) {
      auto predType = inferType(t->getArg2(), tTypes, constTypes, unknown);
      if (!predType || *predType != BOOL) {
        std::stringstream errStr;
        if (!predType) {
          errStr << "Could not infer type ";
        } else {
          errStr << "Inferred type (" << *predType << ") ";
        }
        errStr << "of condition argument of Select operator in expression ";
        expr.print(errStr, 0, false);
        errStr << ". Must be bool.";
        throw poplibs_error(errStr.str());
      }

      auto lhsType = inferType(t->getArg0(), tTypes, constTypes, unknown);
      auto rhsType = inferType(t->getArg1(), tTypes, constTypes, unknown);
      if (!lhsType && rhsType) {
        lhsType = rhsType;
        for (const auto e : unknown)
          constTypes[e] = *rhsType;
        unknown.clear();
      }
      if (lhsType && !rhsType) {
        rhsType = lhsType;
        for (const auto e : unknown)
          constTypes[e] = *lhsType;
        unknown.clear();
      }
      if (!lhsType && !rhsType) {
        // If both lhs and rhs don't have a type then try and deduce it from
        // constants.
        lhsType = getTypeFromConst(t->getArg0());
        rhsType = getTypeFromConst(t->getArg1());
        if (lhsType == rhsType) {
          for (const auto e : unknown)
            constTypes[e] = *lhsType;
          unknown.clear();
        }
      }

      if (lhsType != rhsType) {
        assert(bool(lhsType));
        assert(bool(rhsType));
        std::stringstream errStr;
        errStr << "Inferred type of lhs (" << *lhsType
               << ") does not match inferred type of rhs (" << *rhsType
               << ") in Select expression ";
        expr.print(errStr, 0, false);
        throw poplibs_error(errStr.str());
      }
      return lhsType;
    } else {
      assert(opType == expr::TernaryOpType::CLAMP);
      auto argType = inferType(t->getArg0(), tTypes, constTypes, unknown);
      if (!argType) {
        std::stringstream errStr;
        errStr
            << "Could not infer type of arguments/result in Clamp expression ";
        expr.print(errStr, 0, false);
        throw poplibs_error(errStr.str());
      }
      auto lowerType = inferType(t->getArg1(), tTypes, constTypes, unknown);
      if (!lowerType) {
        lowerType = argType;
        for (const auto e : unknown)
          constTypes[e] = *argType;
        unknown.clear();
      }
      auto higherType = inferType(t->getArg2(), tTypes, constTypes, unknown);
      if (!higherType) {
        higherType = argType;
        for (const auto e : unknown)
          constTypes[e] = *argType;
        unknown.clear();
      }
      return argType;
    }
  }
  POPLIB_UNREACHABLE();
}

std::unordered_map<const expr::Expr *, Type>
getConstType(const expr::Expr &expr, const std::vector<Type> &tTypes) {
  std::unordered_map<const expr::Expr *, Type> constTypes;
  std::vector<const expr::Expr *> unknown;
  std::optional<Type> type;
  // inferType throws with a specific error message which we rethrow
  // giving the full context - i.e. the full expression in question
  // and types of placeholders etc.
  try {
    type = inferType(expr, tTypes, constTypes, unknown);
  } catch (const poplibs_error &e) {
    std::stringstream errStr;
    errStr << "Error inferring types in expression:\n";
    expr.print(errStr, 0, true);
    if (!tTypes.empty()) {
      errStr << "\n\nwith PlaceHolder types:";
      for (std::size_t i = 0; i < tTypes.size(); ++i) {
        errStr << "\n  " << i << ":" << tTypes[i];
      }
    }
    errStr << "\n\n" << e.what();
    throw poplibs_error(errStr.str());
  }

  if (!type || !unknown.empty()) {
    std::stringstream errStr;
    errStr << "Could not infer types in expression:\n";
    expr.print(errStr, 0, true);
    throw poplibs_error(errStr.str());
  }
  return constTypes;
}

ExprAndType optimise(const expr::Expr &expr,
                     const std::vector<poplar::Type> &tTypes) {
  if (const expr::Const *c = expr.getAs<expr::Const>()) {
    return {c->clone(), c->getType()};
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    return {p->clone(), getTypeFromPlaceHolder(*p, tTypes)};
  } else if (const expr::Cast *c = expr.getAs<expr::Cast>()) {
    auto info = optimise(c->getLHS(), tTypes);
    return {std::unique_ptr<expr::Expr>(
                new expr::Cast(*info.expression, c->getRHSType())),
            c->getRHSType()};
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    auto info = optimise(u->getArg(), tTypes);
    return {std::unique_ptr<expr::Expr>(
                new expr::UnaryOp(u->getOpType(), *info.expression)),
            info.type};
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    const expr::Const *c = b->getRHS().getAs<expr::Const>();
    auto infoLhs = optimise(b->getLHS(), tTypes);
    const auto opType = b->getOpType();
    if (opType == expr::BinaryOpType::POWER && c) {
      double value = c->getDataAsDouble();
      if (value == 0.5) {
        return {std::unique_ptr<expr::Expr>(new expr::UnaryOp(
                    expr::UnaryOpType::SQRT, *infoLhs.expression)),
                infoLhs.type};
      } else if (value == -0.5) {
        return {std::unique_ptr<expr::Expr>(new expr::UnaryOp(
                    expr::UnaryOpType::RSQRT, *infoLhs.expression)),
                infoLhs.type};
      } else if (value == 1) {
        // This cast has the same source and destination types and should be
        // a copy that gets elided.
        return {std::unique_ptr<expr::Expr>(
                    new expr::Cast(*infoLhs.expression, infoLhs.type)),
                infoLhs.type};
      } else if (value == -1) {
        return {std::unique_ptr<expr::Expr>(new expr::UnaryOp(
                    expr::UnaryOpType::INVERSE, *infoLhs.expression)),
                infoLhs.type};
      } else if (value == 2) {
        return {std::unique_ptr<expr::Expr>(new expr::UnaryOp(
                    expr::UnaryOpType::SQUARE, *infoLhs.expression)),
                infoLhs.type};
      }
    } else if ((opType == expr::BinaryOpType::REMAINDER ||
                opType == expr::BinaryOpType::DIVIDE) &&
               c) {
      const auto rhsTraits = c->getTypeTraits();
      bool isLhsUnsignedAndIntegral = infoLhs.type == UNSIGNED_SHORT ||
                                      infoLhs.type == UNSIGNED_INT ||
                                      infoLhs.type == UNSIGNED_CHAR;
      bool isRhsUnsignedAndIntegral =
          rhsTraits.isIntegral && !rhsTraits.isSigned;

      if (isLhsUnsignedAndIntegral && isRhsUnsignedAndIntegral) {
        // only allow types upto UNSIGED_INT as there are no codelets
        // that support larger types
        const unsigned value =
            static_cast<unsigned>(c->getDataForUnsignedIntegral());
        if (value && !(value & (value - 1))) {
          if (opType == expr::BinaryOpType::REMAINDER) {
            logging::popops::debug(
                "REMAINDER op optimised to an BITWISE_AND for type {} with "
                "AND value {}",
                infoLhs.type, value - 1);

            return {std::unique_ptr<expr::Expr>(new expr::BinaryOp(
                        expr::BinaryOpType::BITWISE_AND, *infoLhs.expression,
                        expr::Const(value - 1))),
                    infoLhs.type};

          } else {
            const unsigned log2Val = ceilLog2(value);
            logging::popops::debug(
                "DIVIDE op optimised to an SHR for type {} with shift "
                "value {}",
                infoLhs.type, log2Val);

            return {std::unique_ptr<expr::Expr>(new expr::BinaryOp(
                        expr::BinaryOpType::SHIFT_RIGHT, *infoLhs.expression,
                        expr::Const(log2Val))),
                    infoLhs.type};
          }
        }
      }
    }
    auto argRhs = optimise(b->getRHS(), tTypes);
    return {std::unique_ptr<expr::Expr>(new expr::BinaryOp(
                b->getOpType(), *infoLhs.expression, *argRhs.expression)),
            infoLhs.type};
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto arg0Info = optimise(t->getArg0(), tTypes);
    auto arg1Info = optimise(t->getArg1(), tTypes);
    auto arg2Info = optimise(t->getArg2(), tTypes);
    return {std::unique_ptr<expr::Expr>(new expr::TernaryOp(
                t->getOpType(), *arg0Info.expression, *arg1Info.expression,
                *arg2Info.expression)),
            arg0Info.type};
  } else {
    throw poputil::poplibs_error("Unsupported expression");
  }
}

} // end namespace popops
