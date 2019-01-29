#include "popops/ElementWise.hpp"

#include "ExprOpUtil.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/Compiler.hpp"
#include <unordered_map>
#include <boost/optional.hpp>
#include <cassert>

using namespace poputil;
using namespace poplar;
using namespace poplar::program;

using popops::expr::UnaryOpType;
using popops::expr::BinaryOpType;
using popops::expr::TernaryOpType;
using popops::expr::BroadcastOpType;

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
  throw poputil::poplibs_error("Op not supported");
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
  case UnaryOpType::COUNT_LEADING_ZEROS:
    return "CountLeadingZeros";
  case UnaryOpType::EXPONENT:
    return "Exponent";
  case UnaryOpType::EXPONENT_MINUS_ONE:
    return "ExponentMinusOne";
  case UnaryOpType::FLOOR:
    return "Floor";
  case UnaryOpType::IS_FINITE:
    return "IsFinite";
  case UnaryOpType::LOGARITHM:
    return "Logarithm";
  case UnaryOpType::LOGARITHM_ONE_PLUS:
    return "LogarithmOnePlus";
  case UnaryOpType::LOGICAL_NOT:
    return "LogicalNot";
  case UnaryOpType::NEGATE:
    return "Negate";
  case UnaryOpType::POPCOUNT:
    return "Popcount";
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
  case UnaryOpType::SIGMOID:
    return "Sigmoid";
  case UnaryOpType::RSQRT:
    return "Rsqrt";
  }
  throw poputil::poplibs_error("Op not supported");
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
  throw poputil::poplibs_error("Op not supported");
}

static std::string debugName(BroadcastOpType op) {
  switch(op) {
    case BroadcastOpType::ADD:
      return "Add";
    case BroadcastOpType::INV_STD_DEV_TO_VARIANCE:
      return "InvStdDevToVariance";
    case BroadcastOpType::SUBTRACT:
      return "Subtract";
    case BroadcastOpType::MULTIPLY:
      return "Multiply";
    case BroadcastOpType::VARIANCE_TO_INV_STD_DEV:
      return "VarianceToInvStdDev";
  }
  throw poputil::poplibs_error("Op not supported");
}

static std::string debugName(TernaryOpType op) {
  switch(op) {
  case TernaryOpType::CLAMP:
    return "Clamp";
  case TernaryOpType::SELECT:
    return "Select";
  }
  throw poputil::poplibs_error("Op not supported");
}

static BroadcastOpType binaryToBroadcastOp(BinaryOpType op) {
  switch(op) {
    case BinaryOpType::ADD:
      return BroadcastOpType::ADD;
    case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
      return BroadcastOpType::INV_STD_DEV_TO_VARIANCE;
    case BinaryOpType::MULTIPLY:
      return BroadcastOpType::MULTIPLY;
    case BinaryOpType::SUBTRACT:
      return BroadcastOpType::SUBTRACT;
    case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
      return BroadcastOpType::VARIANCE_TO_INV_STD_DEV;
    default:
      throw poputil::poplibs_error("Op not supported");
  }
}

static bool checkForBroadcastOp(BinaryOpType op,
                                std::pair<Tensor, bool> lhs,
                                std::pair<Tensor, bool> rhs) {
  if(op == BinaryOpType::INV_STD_DEV_TO_VARIANCE ||
                               op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) {
    if(!lhs.second)
      throw poputil::poplibs_error("Op only supports InPlace");
    if(rhs.first.rank() != 0)
      throw poputil::poplibs_error("Op requires a scalar second operand");
  }
  if(lhs.first.rank() != rhs.first.rank() && rhs.first.rank() == 0) {
    if(lhs.second) {
      if(op == BinaryOpType::ADD ||
         op == BinaryOpType::INV_STD_DEV_TO_VARIANCE ||
         op == BinaryOpType::VARIANCE_TO_INV_STD_DEV ||
         op == BinaryOpType::SUBTRACT ||
         op == BinaryOpType::MULTIPLY ) {
          return true;
      }
    }
  }
  return false;
}
static Tensor unaryOp(Graph &graph, Tensor in, Sequence &prog,
                      UnaryOpType op, bool inPlace,
                      const std::string &debugPrefix_) {
  const auto debugPrefix = debugPrefix_ + "/Op/" + debugName(op);
  const auto inType = in.elementType();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  const auto outType = outputType(inType, op);
  Tensor out;
  if (inPlace) {
    out = in;
  } else {
    out = graph.clone(outType, in, debugPrefix + "/Out");
    poputil::mapOutputForElementWiseOp(graph, {in}, out);
  }

  auto inFlat = in.flatten();
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat});
  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(inType),
                         target.getAtomicStoreGranularity());

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap =  mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    if (tileContiguousRegions.size() == 1 ) {
      // If mapping of the output tensor on this tile is only region or regions
      // from one variable, force a gather (in case of more than one region)
      // to get all data to a single edge.
      // The decision to make a vertex supervisor may also have to account
      // for the total elements as the overhead and work balance may not be
      // very good for small vector sizes.
      // TODO: Use profiled results for selection
      const auto vertexTemplate =
          templateVertex(inPlace ? "popops::UnaryOp1DInPlaceSupervisor" :
                                   "popops::UnaryOp1DSupervisor",
                         op, inType);
      auto v = inPlace ?
        graph.addVertex(cs, vertexTemplate,
                        {{"inOut", concat(inFlat.slices(thisTileMap))}}):
        graph.addVertex(cs, vertexTemplate,
                        {{"in", concat(inFlat.slices(thisTileMap))},
                         {"out", concat(outFlat.slices(thisTileMap))}});
        graph.setTileMapping(v, tile);
    } else {
      const auto vertexTemplate =
          templateVertex(inPlace ? "popops::UnaryOp2DInPlace" :
                                   "popops::UnaryOp2D",
                         op, inType);
      auto vertexRegions =
          splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                     grainSize, 2 * grainSize);
      for (const auto &regions : vertexRegions) {
        VertexRef v = inPlace ?
            graph.addVertex(cs, vertexTemplate,
                            {{"inOut", inFlat.slices(regions)}}) :
            graph.addVertex(cs, vertexTemplate,
                            {{"in", inFlat.slices(regions)},
                           {"out", outFlat.slices(regions)}});
        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(cs));
  return out;
}

static Tensor binaryOp(Graph &graph, Tensor in1, Tensor in2,
                       Sequence &prog, BinaryOpType op, bool inPlace,
                       bool nonCopyBroadcast,
                       const std::string &debugPrefix_) {
  const auto debugPrefix = debugPrefix_ + "/Op/" + (nonCopyBroadcast ?
              debugName(binaryToBroadcastOp(op)) : debugName(op));
  const auto in1Type = in1.elementType();
  const auto in2Type = in2.elementType();

  if (in1Type != in2Type) {
    throw poputil::poplibs_error("Binary Op must have same type for "
                               "both operands: " + debugPrefix);
  }

  if (in1.shape() != in2.shape() && !nonCopyBroadcast) {
    throw poputil::poplibs_error("Binary Op must have same shape for "
                               "both operands: " + debugPrefix);
  }

  const auto outType = outputType(in1Type, op);
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  Tensor out;
  if (inPlace) {
    out = in1;
  } else {
    out = graph.clone(outType, in1, debugPrefix + "/Out");
    poputil::mapOutputForElementWiseOp(graph, {in1, in2}, out);
  }

  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto outFlat = out.flatten();
  if (nonCopyBroadcast) {
    assert(in2Flat.numElements() == 1);
    graph.reorderToSimplify(&outFlat, {&in1Flat});
  } else {
    graph.reorderToSimplify(&outFlat, {&in1Flat, &in2Flat});
  }
  const auto mapping = graph.getTileMapping(outFlat);

  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(in1Type),
                         target.getAtomicStoreGranularity());

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap =  mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    if (tileContiguousRegions.size() == 1 ) {
      // If mapping of the output tensor on this tile is only region or regions
      // from one variable, force a gather (in case of more than one region)
      // to get all data to a single edge.
      //
      // The decision to make a vertex supervisor may also have to account
      // for the total elements as the overhead and work balance may not be
      // very good for small vector sizes.
      // TODO: Use profiled results for selection
      if(nonCopyBroadcast) {
        const auto vertexTemplate =
            templateVertex("popops::BroadcastOp1DInPlaceSupervisor",
                                    binaryToBroadcastOp(op), in1Type);
        auto v = graph.addVertex(cs, vertexTemplate,
                            {{"data", concat(outFlat.slices(thisTileMap))},
                             {"B", in2}});
        graph.setTileMapping(v, tile);
      }
      else {
        const auto vertexTemplate =
            templateVertex(inPlace ? "popops::BinaryOp1DInPlaceSupervisor" :
                                     "popops::BinaryOp1DSupervisor",
                           op, in1Type);
        auto v = inPlace ?
            graph.addVertex(cs, vertexTemplate,
                            {{"in1Out", concat(outFlat.slices(thisTileMap))},
                             {"in2", concat(in2Flat.slices(thisTileMap))}}) :
            graph.addVertex(cs, vertexTemplate,
                            {{"in1", concat(in1Flat.slices(thisTileMap))},
                             {"in2", concat(in2Flat.slices(thisTileMap))},
                             {"out", concat(outFlat.slices(thisTileMap))}});
       graph.setTileMapping(v, tile);
     }
    } else {
      if(nonCopyBroadcast) {
        const auto vertexTemplate =
            templateVertex("popops::BroadcastOp2DInPlace",
                                    binaryToBroadcastOp(op), in1Type);
        auto vertexRegions =
          splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                     grainSize, 2 * grainSize);
        for (const auto &regions : vertexRegions) {
          auto v = graph.addVertex(cs, vertexTemplate,
                            {{"data", concat(outFlat.slices(regions))},
                             {"B", concat(in2Flat.slices(regions))}});
          graph.setTileMapping(v, tile);
        }
      }
      else {
        const auto vertexTemplate =
              templateVertex(inPlace ? "popops::BinaryOp2DInPlace" :
                                       "popops::BinaryOp2D",
                             op, in1Type);
        auto vertexRegions =
          splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                     grainSize, 2 * grainSize);

        for (const auto &regions : vertexRegions) {
          auto v = inPlace ?
                graph.addVertex(cs, vertexTemplate,
                                {{"in1Out", outFlat.slices(regions)},
                                 {"in2", in2Flat.slices(regions)}}) :
                graph.addVertex(cs, vertexTemplate,
                                {{"in1", in1Flat.slices(regions)},
                                 {"in2", in2Flat.slices(regions)},
                                 {"out", outFlat.slices(regions)}});
          graph.setTileMapping(v, tile);
        }
      }
    }
  }
  prog.add(Execute(cs));
  return out;
}

static Tensor ternaryOp(Graph &graph, Tensor in1, Tensor in2, Tensor in3,
                        Sequence &prog, TernaryOpType op, bool inPlace,
                        const std::string &debugPrefix_) {
  const auto debugPrefix = debugPrefix_ + "/Op/" + debugName(op);
  const auto in1Type = in1.elementType();
  const auto in2Type = in2.elementType();
  const auto in3Type = in3.elementType();

  if (in1Type != in2Type) {
    throw poputil::poplibs_error("Ternary Op must have same type for "
                               "all input operands: " + debugPrefix);
  }

  if (in1.shape() != in2.shape() || in1.shape() != in3.shape()) {
    throw poputil::poplibs_error("Ternary Op must have same shape for "
                               "all input operands: " + debugPrefix);
  }

  const auto outType = outputType(in1Type, op);
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  Tensor out;
  if (inPlace) {
    out = in1;
  } else {
    out = graph.clone(outType, in1, debugPrefix + "/Out");
    poputil::mapOutputForElementWiseOp(graph, {in1, in2, in3}, out);
  }

  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto in3Flat = in3.flatten();
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {&in1Flat, &in2Flat, &in3Flat});
  const auto mapping = graph.getTileMapping(outFlat);

  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(in1Type),
                         target.getAtomicStoreGranularity());
  const auto opVertexName = vertexName(op) + (inPlace ? "InPlace" : "");

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto vertexRegions =
      splitRegionsBetweenWorkers(target, mapping[tile],
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = inPlace ?
            graph.addVertex(cs,
                               templateVertex(opVertexName, in1Type),
                               {{"in1Out", in1Flat.slices(regions)},
                                {"in2", in2Flat.slices(regions)},
                                {"in3", in3Flat.slices(regions)}}) :
            graph.addVertex(cs,
                               templateVertex(opVertexName, in1Type),
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

static bool isLogical(expr::UnaryOpType op) {
  switch (op) {
  case expr::UnaryOpType::LOGICAL_NOT:
    return true;
  default:
    return false;
  }
}

static bool isLogical(expr::BinaryOpType op) {
  switch (op) {
  case expr::BinaryOpType::LOGICAL_AND:
  case expr::BinaryOpType::LOGICAL_OR:
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
    throw poplibs_error("Invalid placeholder _" + std::to_string(index + 1) +
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
    if (isRelational(opType) || isLogical(opType)) {
      if (!unknown.empty())
        throw poplibs_error("Cannot infer constant types in expression");
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
      throw poplibs_error("Arguments of binary operator in expression do not "
                         "have the same type");
    if (isRelational(opType) || isLogical(opType)) {
      if (!unknown.empty())
        throw poplibs_error("Cannot infer constant types in expression");
      return BOOL;
    }
    return lhsType;
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto opType = t->getOpType();
    if (opType == TernaryOpType::SELECT) {
      auto predType = inferType(t->getArg2(), ts, constTypes, unknown);
      if (!predType || *predType != BOOL)
        throw poplibs_error("Invalid type of condition argument of "
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
        throw poplibs_error("Arguments of select operator in expression do not "
                           "have the same type");
      return lhsType;
    } else {
      assert(opType == TernaryOpType::CLAMP);
      auto argType = inferType(t->getArg0(), ts, constTypes, unknown);
      if (!argType)
        throw poplibs_error("Cannot infer type in clamp expression");
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

// Recursively walk up the expression tree and do inPlace operations if
// conditions are met
// topLevel :
//   If true indicates root node
// constructGraph :
//   If true, graph is constructed as the expression tree is traversed. The
//   inPlaceExpr is used if inPlace flag is set
//   If false, no graph is constructed but inPlaceExpr may be set if a
//   placeholder expression with index 1 is found
// inPlace :
//   If true an attempt is made to do an in-place operation. An inplace
//   operation succeeds if placeholder with index 1 is on the leftmost traversal
//   path
//
// Further in-place optimisations are possble by traversing the tree and
// transforming the operations.
static std::pair<Tensor, bool>
map(Graph &graph,
    const expr::Expr &expr,
    const std::vector<Tensor> &ts,
    program::Sequence &prog,
    const std::string &debugPrefix,
    const std::unordered_map<const expr::Expr *, Type> constTypes,
    bool topLevel,
    bool constructGraph,
    bool inPlace,
    const expr::Expr *&inPlaceExpr) {

   if (!constructGraph)
    assert(!inPlace);
  if (const expr::Const *c = expr.getAs<expr::Const>()) {
    assert(constTypes.find(&expr) != constTypes.end());
    auto ct = graph.addConstant(constTypes.at(&expr), {},
                               c->getData(), c->getTypeTraits(), false);
    graph.setTileMapping(ct, 0);
    return {ct, false};
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    const auto &t =  getTensorFromPlaceHolder(*p, ts);
    const auto index = p->getIndex();
    bool useInPlace;
    if (!constructGraph) {
      // record expression only when graph is not constructed. The last
      // expression with placeholder = 1 is recorded
      if (index == 1)
        inPlaceExpr = p;
      useInPlace = false;
    } else {
      useInPlace = inPlace && index == 1 && inPlaceExpr == p;
      if (topLevel &&
          (!useInPlace || (useInPlace && index != 1))) {
        return {graph.clone(t), useInPlace};
      }
    }
    return {t, useInPlace};
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    auto opType = u->getOpType();
    auto t = map(graph, u->getArg(), ts, prog, debugPrefix, constTypes, false,
                 constructGraph, inPlace, inPlaceExpr);
    if (constructGraph) {
      return {unaryOp(graph, t.first, prog, opType, t.second, debugPrefix),
              t.second};
    } else {
      return t;
    }
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto opType = b->getOpType();
    auto lhs = map(graph, b->getLHS(), ts, prog, debugPrefix, constTypes, false,
                   constructGraph, inPlace, inPlaceExpr);
    auto rhs = map(graph, b->getRHS(), ts, prog, debugPrefix, constTypes, false,
                  constructGraph, false, inPlaceExpr);
    if (constructGraph) {
      const bool nonCopyBroadcast = checkForBroadcastOp(opType, lhs, rhs);
      if(!nonCopyBroadcast)
        broadcastToMatch(lhs.first, rhs.first);
      return {binaryOp(graph, lhs.first, rhs.first, prog, opType, lhs.second,
                       nonCopyBroadcast, debugPrefix), lhs.second};
    } else {
      return lhs;
    }
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto opType = t->getOpType();
    if (opType == TernaryOpType::SELECT) {
      auto lhs = map(graph, t->getArg0(), ts, prog, debugPrefix, constTypes,
                     false, constructGraph, inPlace, inPlaceExpr);
      auto rhs = map(graph, t->getArg1(), ts, prog, debugPrefix, constTypes,
                     false, constructGraph, false, inPlaceExpr);
      auto pred = map(graph, t->getArg2(), ts, prog, debugPrefix, constTypes,
                      false, constructGraph, false, inPlaceExpr);
      if (constructGraph) {
        broadcastToMatch(lhs.first, rhs.first);
        return {ternaryOp(graph, lhs.first, rhs.first, pred.first, prog, opType,
                          lhs.second, debugPrefix), lhs.second};
      } else {
        return lhs;
      }
    } else {
      assert(opType == TernaryOpType::CLAMP);
      auto in = map(graph, t->getArg0(), ts, prog, debugPrefix, constTypes,
                    false, constructGraph, inPlace, inPlaceExpr);
      auto lower = map(graph, t->getArg1(), ts, prog, debugPrefix, constTypes,
                       false, constructGraph, false, inPlaceExpr);
      auto upper = map(graph, t->getArg2(), ts, prog, debugPrefix, constTypes,
                       false, constructGraph, false, inPlaceExpr);
      if (constructGraph) {
        return {ternaryOp(graph, in.first, lower.first, upper.first, prog,
                          opType, in.second, debugPrefix), in.second};
      } else {
        return in;
      }
    }
  }
  POPLIB_UNREACHABLE();
}

static std::unordered_map<const expr::Expr *, Type>
getConstType(const expr::Expr &expr, const std::vector<Tensor> &ts) {
  std::unordered_map<const expr::Expr *, Type> constTypes;
  std::vector<const expr::Expr *> unknown;
  auto type = inferType(expr, ts, constTypes, unknown);
  if (!type || !unknown.empty())
    throw poplibs_error("Cannot infer type of expression");
  return constTypes;
}

Tensor map(Graph &graph, const expr::Expr &expr,
           const std::vector<Tensor> &ts,
           program::Sequence &prog,
           const std::string &debugPrefix,
           const std::vector<std::string> &options) {
  auto constTypes = getConstType(expr, ts);
  const expr::Expr *inplaceExpr = nullptr;
  return map(graph, expr, ts, prog, debugPrefix, constTypes, true, true, false,
             inplaceExpr).first;
}

void mapInPlace(Graph &graph, const expr::Expr &expr,
                const std::vector<Tensor> &ts,
                program::Sequence &prog,
                const std::string &debugPrefix,
                const std::vector<std::string> &options) {
  auto constTypes = getConstType(expr, ts);
  const expr::Expr *inPlaceExpr = nullptr;
  const bool doInPlace = !ts[0].containsAliases() && !ts[0].containsConstant();
  if (doInPlace) {
    // As the tree is traveresed, find the last expression which uses the
    // tensor used for in-place operation as a placeholder
    map(graph, expr, ts, prog, debugPrefix, constTypes, true, false, false,
        inPlaceExpr);
  }
  auto t = map(graph, expr, ts, prog, debugPrefix, constTypes, true, true,
               doInPlace, inPlaceExpr);
  // If in-place operations were not performed, then copy the final result
  // into the tensor supplied.
  // @TODO Optimisation: If placeholder _1 is not used, a copy may be done
  // early enough to avoid this copy and use in-place operations after that
  // copy. Or, the unary, binary and ternary operations must allow an output
  // tensor to be given as an argument (the current method either uses one of
  // the input tensors if the operation is in-place, or creates and output
  // tensor)
  if (!t.second) {
    prog.add(Copy(t.first, ts[0]));
  }
}

} // namespace popops
