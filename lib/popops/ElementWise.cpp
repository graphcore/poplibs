#include "popops/ElementWise.hpp"

#include "poputil/Broadcast.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/Compiler.hpp"
#include <unordered_map>
#include <boost/optional.hpp>
#include "ExprOpUtil.hpp"

using namespace poputil;
using namespace poplar;
using namespace poplar::program;

using popops::expr::UnaryOpType;
using popops::expr::BinaryOpType;
using popops::expr::TernaryOpType;

namespace popops {

static Type outputType(const Type &inType, enum UnaryOpType op) {
  if (op == UnaryOpType::IS_FINITE
      || op == UnaryOpType::LOGICAL_NOT) {
    return BOOL;
  } else {
    return inType;
  }
}

static Type outputType(const Type &inType, BinaryOpType op) {
  if (op == BinaryOpType::EQUAL
      || op == BinaryOpType::GREATER_THAN_EQUAL
      || op == BinaryOpType::GREATER_THAN
      || op == BinaryOpType::LESS_THAN_EQUAL
      || op == BinaryOpType::LOGICAL_AND
      || op == BinaryOpType::LOGICAL_OR
      || op == BinaryOpType::LESS_THAN
      || op == BinaryOpType::NOT_EQUAL) {
    return BOOL;
  } else {
    return inType;
  }
}

static Type outputType(const Type &inType,
                       TernaryOpType /*op*/) {
  return inType;
}

static std::string vertexName(TernaryOpType op) {
  switch(op) {
  case TernaryOpType::CLAMP:
    return "popops::Clamp";
  case TernaryOpType::SELECT:
    return "popops::Select";
  }
  throw poputil::poplib_error("Op not supported");
}

static std::string debugName(UnaryOpType op) {
  switch(op) {
  case UnaryOpType::ABSOLUTE:
    return "Absolute";
  case UnaryOpType::BITWISE_NOT:
    return "BitwiseNot";
  case UnaryOpType::CEIL:
    return "Ceil";
  case UnaryOpType::COS:
    return "Cos";
  case UnaryOpType::EXPONENT:
    return "Exponent";
  case UnaryOpType::FLOOR:
    return "Floor";
  case UnaryOpType::IS_FINITE:
    return "IsFinite";
  case UnaryOpType::LOGARITHM:
    return "Logarithm";
  case UnaryOpType::LOGICAL_NOT:
    return "LogicalNot";
  case UnaryOpType::NEGATE:
    return "Negate";
  case UnaryOpType::ROUND:
      return "Round";
  case UnaryOpType::SIGNUM:
    return "Signum";
  case UnaryOpType::SIN:
    return "Sin";
  case UnaryOpType::TANH:
    return "Tanh";
  case UnaryOpType::SQRT:
    return "Sqrt";
  case UnaryOpType::SQUARE:
    return "Square";
  }
  throw poputil::poplib_error("Op not supported");
}

static std::string debugName(BinaryOpType op) {
  switch(op) {
    case BinaryOpType::ADD:
      return "Add";
    case BinaryOpType::ATAN2:
      return "Atan2";
    case BinaryOpType::BITWISE_AND:
      return "BitwiseAnd";
    case BinaryOpType::BITWISE_OR:
      return "BitwiseOr";
    case BinaryOpType::DIVIDE:
      return "Divide";
    case BinaryOpType::EQUAL:
      return "Equal";
    case BinaryOpType::GREATER_THAN_EQUAL:
      return "GreaterThanEqual";
    case BinaryOpType::GREATER_THAN:
      return "GreaterThan";
    case BinaryOpType::LESS_THAN_EQUAL:
      return "LessThanEqual";
    case BinaryOpType::LOGICAL_AND:
      return "LogicalAnd";
    case BinaryOpType::LOGICAL_OR:
      return "LogicalOr";
    case BinaryOpType::LESS_THAN:
      return "LessThan";
    case BinaryOpType::MAXIMUM:
      return "Maximum";
    case BinaryOpType::MINIMUM:
      return "Minimum";
    case BinaryOpType::MULTIPLY:
      return "Multiply";
    case BinaryOpType::NOT_EQUAL:
      return "NotEqual";
    case BinaryOpType::POWER:
      return "Power";
    case BinaryOpType::REMAINDER:
      return "Remainder";
    case BinaryOpType::SHIFT_LEFT:
      return "ShiftLeft";
    case BinaryOpType::SHIFT_RIGHT:
      return "ShiftRight";
    case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
      return "ShiftRightSignExtend";
    case BinaryOpType::SUBTRACT:
      return "Subtract";
  }
  throw poputil::poplib_error("Op not supported");
}

static std::string debugName(TernaryOpType op) {
  switch(op) {
  case TernaryOpType::CLAMP:
    return "Clamp";
  case TernaryOpType::SELECT:
    return "Select";
  }
  throw poputil::poplib_error("Op not supported");
}



static unsigned
compareTileMapDistributions(Graph &graph, std::vector<Tensor> in) {
  std::vector<unsigned> tileScore(in.size());
  std::vector<unsigned> distributionScore(in.size());

  for (unsigned i = 0; i < in.size(); ++i) {
    const auto mapping = graph.getTileMapping(in[i]);

    for (const auto &tile : mapping) {
      if (tile.size() != 0) {
        tileScore[i]++;
        distributionScore[i] += tile.size();
      }
    }
  }

  unsigned best = 0;
  for (unsigned i = 1; i < in.size(); ++i) {
    // Select the tensor which is spread onto the most tiles
    if (tileScore[i] > tileScore[best]) {
      best = i;
    }

    // If two tensors share the same number of tiles, then select the one
    // which has the fewest overall regions
    if (tileScore[i] == tileScore[best] &&
        distributionScore[i] < distributionScore[best]) {
      best = i;
    }
  }

  return best;
}

static Tensor unaryOp(Graph &graph, Tensor in, Sequence &prog,
                      UnaryOpType op, const std::string &debugPrefix_) {
  const auto debugPrefix = debugPrefix_ + "/Op/" + debugName(op);
  const auto inType = in.elementType();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  const auto outType = outputType(inType, op);
  auto out = graph.clone(outType, in, debugPrefix + "/Out");

  auto inFlat = in.flatten();
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});
  const auto mapping = graph.getTileMapping(inFlat);
  const auto grainSize = target.getVectorWidth(inType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);
    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("popops::UnaryOp", op,
                                              inType),
                               {{"in", inFlat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return out;
}

static Tensor binaryOp(Graph &graph, Tensor in1, Tensor in2, Sequence &prog,
                       BinaryOpType op, const std::string &debugPrefix_) {
  const auto debugPrefix = debugPrefix_ + "/Op/" + debugName(op);
  const auto in1Type = in1.elementType();
  const auto in2Type = in2.elementType();

  if (in1Type != in2Type) {
    throw poputil::poplib_error("Binary Op must have same type for "
                               "both operands: " + debugPrefix);
  }

  if (in1.shape() != in2.shape()) {
    throw poputil::poplib_error("Binary Op must have same shape for "
                               "both operands: " + debugPrefix);
  }

  unsigned tensorSelection = compareTileMapDistributions(graph, {in1, in2});

  const auto outType = outputType(in1Type, op);
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  auto out = graph.clone(outType, (tensorSelection == 0) ? in1 : in2,
                         debugPrefix + "/Out");


  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {&in1Flat, &in2Flat});
  const auto mapping = graph.getTileMapping(outFlat);

  const auto grainSize = target.getVectorWidth(in1Type);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, mapping[tile]);
    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("popops::BinaryOp", op,
                                              in1Type),
                               {{"in1", in1Flat.slices(regions)},
                                {"in2", in2Flat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return out;
}


static Tensor ternaryOp(Graph &graph, Tensor in1, Tensor in2, Tensor in3,
                        Sequence &prog, TernaryOpType op,
                        const std::string &debugPrefix_) {
  const auto debugPrefix = debugPrefix_ + "/Op/" + debugName(op);
  const auto in1Type = in1.elementType();
  const auto in2Type = in2.elementType();
  const auto in3Type = in3.elementType();

  if (in1Type != in2Type) {
    throw poputil::poplib_error("Ternary Op must have same type for "
                               "all input operands: " + debugPrefix);
  }

  if (in1.shape() != in2.shape() || in1.shape() != in3.shape()) {
    throw poputil::poplib_error("Ternary Op must have same shape for "
                               "all input operands: " + debugPrefix);
  }

  std::vector<Tensor> tensors = {in1, in2, in3};

  int tensorSelection = compareTileMapDistributions(graph, tensors);

  const auto outType = outputType(in1Type, op);
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  Tensor toClone = tensors[tensorSelection];

  auto out = graph.clone(outType, toClone, debugPrefix + "/Out");


  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto in3Flat = in3.flatten();
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {&in1Flat, &in2Flat, &in3Flat});
  const auto mapping = graph.getTileMapping(outFlat);

  const auto grainSize = target.getVectorWidth(in1Type);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto vertexRegions =
      splitRegionsBetweenWorkers(target, mapping[tile],
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex(vertexName(op), in1Type),
                               {{"in1", in1Flat.slices(regions)},
                                {"in2", in2Flat.slices(regions)},
                                {"in3", in3Flat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return out;
}

static bool isRelational(expr::UnaryOpType op) {
  switch (op) {
  case expr::UnaryOpType::IS_FINITE:
    return true;
  default:
    return false;
  }
}

static bool isRelational(expr::BinaryOpType op) {
  switch (op) {
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

static const Tensor &
getTensorFromPlaceHolder(const expr::PlaceHolder &p,
                          const std::vector<Tensor> &ts) {
  auto index = p.getIndex() - 1;
  if (index > ts.size()) {
    throw poplib_error("Invalid placeholder _" + std::to_string(index + 1) +
                       " in expression");
  }
  return ts[index];
}

static boost::optional<Type>
inferType(const expr::Expr &expr,
          const std::vector<Tensor> &ts,
          std::unordered_map<const expr::Expr *, Type> &constTypes,
          std::vector<const expr::Expr *> &unknown) {
  if (expr.isA<expr::Const>()) {
    unknown.push_back(&expr);
    return {};
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    return getTensorFromPlaceHolder(*p, ts).elementType();
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    auto opType = u->getOpType();
    auto argType = inferType(u->getArg(), ts, constTypes, unknown);
    if (isRelational(opType)) {
      if (!unknown.empty())
        throw poplib_error("Cannot infer constant types in expression");
      return BOOL;
    }
    return argType;
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto opType = b->getOpType();
    auto lhsType = inferType(b->getLHS(), ts, constTypes, unknown);
    auto rhsType = inferType(b->getRHS(), ts, constTypes, unknown);
    if (!lhsType && rhsType) {
      lhsType = rhsType;
      for (const auto e : unknown)
        constTypes[e] = *rhsType;
      unknown.clear();
    }
    if (!rhsType && lhsType) {
      rhsType = lhsType;
      for (const auto e : unknown)
        constTypes[e] = *lhsType;
      unknown.clear();
    }
    if (lhsType != rhsType)
      throw poplib_error("Arguments of binary operator in expression do not "
                         "have the same type");
    if (isRelational(opType)) {
      if (!unknown.empty())
        throw poplib_error("Cannot infer constant types in expression");
      return BOOL;
    }
    return lhsType;
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto opType = t->getOpType();
    if (opType == TernaryOpType::SELECT) {
      auto predType = inferType(t->getArg2(), ts, constTypes, unknown);
      if (!predType || *predType != BOOL)
        throw poplib_error("Invalid type of condition argument of "
                           "select operator in expression");

      auto lhsType = inferType(t->getArg0(), ts, constTypes, unknown);
      auto rhsType = inferType(t->getArg1(), ts, constTypes, unknown);
      if (!lhsType && rhsType) {
        lhsType = rhsType;
        for (const auto e : unknown)
          constTypes[e] = *rhsType;
        unknown.clear();
      }
      if (rhsType && !lhsType) {
        rhsType = lhsType;
        for (const auto e : unknown)
          constTypes[e] = *lhsType;
        unknown.clear();
      }
      if (lhsType != rhsType)
        throw poplib_error("Arguments of select operator in expression do not "
                           "have the same type");
      return lhsType;
    } else {
      assert(opType == TernaryOpType::CLAMP);
      auto argType = inferType(t->getArg0(), ts, constTypes, unknown);
      if (!argType)
        throw poplib_error("Cannot infer type in clamp expression");
      auto lowerType = inferType(t->getArg1(), ts, constTypes, unknown);
      if (!lowerType) {
        lowerType = argType;
        for (const auto e : unknown)
          constTypes[e] = *argType;
        unknown.clear();
      }
      auto higherType = inferType(t->getArg2(), ts, constTypes, unknown);
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

static Tensor
map(Graph &graph,
    const expr::Expr &expr,
    const std::vector<Tensor> &ts,
    program::Sequence &prog,
    const std::string &debugPrefix,
    const std::unordered_map<const expr::Expr *, Type> constTypes,
    bool topLevel) {
  if (const expr::Const *c = expr.getAs<expr::Const>()) {
    assert(constTypes.find(&expr) != constTypes.end());
    return graph.addConstant(constTypes.at(&expr), {},
                             c->getData(), c->getTypeTraits(), false);
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    const auto &t =  getTensorFromPlaceHolder(*p, ts);
    if (topLevel)
      return graph.clone(t);
    return t;
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    auto opType = u->getOpType();
    auto t1 = map(graph, u->getArg(), ts, prog, debugPrefix,
                  constTypes, false);
    return unaryOp(graph, t1, prog, opType, debugPrefix);
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto opType = b->getOpType();
    auto lhs = map(graph, b->getLHS(), ts, prog, debugPrefix,
                   constTypes, false);
    auto rhs = map(graph, b->getRHS(), ts, prog, debugPrefix,
                   constTypes, false);
    broadcastToMatch(lhs, rhs);
    return binaryOp(graph, lhs,  rhs, prog, opType, debugPrefix);
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto opType = t->getOpType();
    if (opType == TernaryOpType::SELECT) {
      auto lhs = map(graph, t->getArg0(), ts, prog, debugPrefix,
                     constTypes, false);
      auto rhs = map(graph, t->getArg1(), ts, prog, debugPrefix,
                     constTypes, false);
      auto pred = map(graph, t->getArg2(), ts, prog, debugPrefix,
                      constTypes, false);
      broadcastToMatch(lhs, rhs);
      return ternaryOp(graph, lhs,  rhs, pred, prog, opType, debugPrefix);
    } else {
      assert(opType == TernaryOpType::CLAMP);
      auto in = map(graph, t->getArg0(), ts, prog, debugPrefix,
                     constTypes, false);
      auto lower = map(graph, t->getArg1(), ts, prog, debugPrefix,
                     constTypes, false);
      auto upper = map(graph, t->getArg2(), ts, prog, debugPrefix,
                      constTypes, false);
      return ternaryOp(graph, in, lower, upper, prog, opType, debugPrefix);
    }
  }
  POPLIB_UNREACHABLE();
 }

Tensor map(Graph &graph, const expr::Expr &expr,
           const std::vector<Tensor> &ts,
           program::Sequence &prog,
           const std::string &debugPrefix,
           const std::vector<std::string> &options) {
  std::unordered_map<const expr::Expr *, Type> constTypes;
  std::vector<const expr::Expr *> unknown;
  auto type = inferType(expr, ts, constTypes, unknown);
  if (!type || !unknown.empty())
    throw poplib_error("Cannot infer type of expression");
  return map(graph, expr, ts, prog, debugPrefix, constTypes, true);
}

void mapInPlace(Graph &graph, const expr::Expr &expr,
                const std::vector<Tensor> &ts,
                program::Sequence &prog,
                const std::string &debugPrefix,
                const std::vector<std::string> &options) {
  // TODO: This method of creating an explicit intermediate tensor
  // is quite inefficient. We can replace this with specialized vertices
  // that write back the result in place.
  auto result = map(graph, expr, ts, prog, debugPrefix, options);
  prog.add(Copy(result, ts[0]));
  return;
}

} // namespace popops
