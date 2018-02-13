#include "ElementWiseDef.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

using namespace poputil;
using namespace poplar;
using namespace poplar::program;

namespace popops {

static Type outputType(const Type &inType, enum UnaryOp op) {
  if (op == IS_FINITE
      || op == LOGICAL_NOT) {
    return BOOL;
  } else {
    return inType;
  }
}

static Type outputType(const Type &inType, enum BinaryOp op) {
  if (op == EQUAL
      || op == GREATER_THAN_EQUAL
      || op == GREATER_THAN
      || op == LESS_THAN_EQUAL
      || op == LOGICAL_AND
      || op == LOGICAL_OR
      || op == LESS_THAN
      || op == NOT_EQUAL) {
    return BOOL;
  } else {
    return inType;
  }
}

static Type outputType(const Type &inType,
                       enum TernaryOp /*op*/) {
  return inType;
}

static std::string vertexName(enum UnaryOp op) {
  switch(op) {
  case ABSOLUTE:
    return "popops::Absolute";
  case BITWISE_NOT:
    return "popops::BitwiseNot";
  case CEIL:
    return "popops::Ceil";
  case COS:
    return "popops::Cos";
  case EXPONENT:
    return "popops::Exponent";
  case FLOOR:
    return "popops::Floor";
  case IS_FINITE:
    return "popops::IsFinite";
  case LOGARITHM:
    return "popops::Logarithm";
  case LOGICAL_NOT:
    return "popops::LogicalNot";
  case NEGATE:
    return "popops::Negate";
  case ROUND:
      return "popops::Round";
  case SIGNUM:
    return "popops::Signum";
  case SIN:
    return "popops::Sin";
  case TANH:
    return "popops::Tanh";
  case SQRT:
    return "popops::Sqrt";
  case SQUARE:
    return "popops::Square";
  }
  throw poputil::poplib_error("Op not supported");
}

static std::string vertexName(enum BinaryOp op) {
  switch(op) {
    case ADD:
      return "popops::Add";
    case ATAN2:
      return "popops::Atan2";
    case BITWISE_AND:
      return "popops::BitwiseAnd";
    case BITWISE_OR:
      return "popops::BitwiseOr";
    case DIVIDE:
      return "popops::Divide";
    case EQUAL:
      return "popops::Equal";
    case GREATER_THAN_EQUAL:
      return "popops::GreaterThanEqual";
    case GREATER_THAN:
      return "popops::GreaterThan";
    case LESS_THAN_EQUAL:
      return "popops::LessThanEqual";
    case LOGICAL_AND:
      return "popops::LogicalAnd";
    case LOGICAL_OR:
      return "popops::LogicalOr";
    case LESS_THAN:
      return "popops::LessThan";
    case MAXIMUM:
      return "popops::Maximum";
    case MINIMUM:
      return "popops::Minimum";
    case MULTIPLY:
      return "popops::Multiply";
    case NOT_EQUAL:
      return "popops::NotEqual";
    case POWER:
      return "popops::Power";
    case REMAINDER:
      return "popops::Remainder";
    case SHIFT_LEFT:
      return "popops::ShiftLeft";
    case SHIFT_RIGHT:
      return "popops::ShiftRight";
    case SHIFT_RIGHT_SIGN_EXTEND:
      return "popops::ShiftRightSignExtend";
    case SUBTRACT:
      return "popops::Subtract";
  }
  throw poputil::poplib_error("Op not supported");
}

static std::string vertexName(enum TernaryOp op) {
  switch(op) {
  case CLAMP:
    return "popops::Clamp";
  case SELECT:
    return "popops::Select";
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
                      enum UnaryOp op, const std::string &debugPrefix) {

  const auto inType = in.elementType();
  const auto &target = graph.getTarget();
  const auto dataPathWidth = target.getDataPathWidth();
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
                               templateVertex(vertexName(op), inType),
                               {{"in", inFlat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return out;
}

static Tensor binaryOp(Graph &graph, Tensor in1, Tensor in2, Sequence &prog,
                       enum BinaryOp op, const std::string &debugPrefix) {
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
  const auto dataPathWidth = target.getDataPathWidth();
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
                               templateVertex(vertexName(op), in1Type),
                               {{"in1", in1Flat.slices(regions)},
                                {"in2", in2Flat.slices(regions)},
                                {"out", outFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return out;
}


static Tensor ternaryOp(Graph &graph, Tensor in1, Tensor in2, Tensor in3,
                        Sequence &prog, enum TernaryOp op,
                        const std::string &debugPrefix) {
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
  const auto dataPathWidth = target.getDataPathWidth();
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
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return out;
}

Tensor add(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::ADD, debugPrefix + "/Op/Add");
}

Tensor abs(Graph &graph, Tensor A, Sequence &prog,
           const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::ABSOLUTE, debugPrefix + "/Op/Abs");
}

Tensor atan2(Graph &graph, Tensor A, Tensor B, Sequence &prog,
             const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::ATAN2,
                  debugPrefix + "/Op/Atan2");
}

Tensor bitwiseAnd(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::BITWISE_AND,
                  debugPrefix + "/Op/BitwiseAnd");
}

Tensor bitwiseOr(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::BITWISE_OR,
                  debugPrefix + "/Op/BitwiseOr");
}

Tensor bitwiseNot(Graph &graph, Tensor A, Sequence &prog,
           const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::BITWISE_NOT,
                 debugPrefix + "/Op/BitwiseNot");
}

Tensor ceil(Graph &graph, Tensor A, Sequence &prog,
            const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::CEIL, debugPrefix + "/Op/Ceil");
}

Tensor cos(Graph &graph, Tensor A, Sequence &prog,
            const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::COS, debugPrefix + "/Op/Cos");
}

Tensor div(Graph &graph, Tensor A, Tensor B, Sequence &prog,
          const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::DIVIDE, debugPrefix + "/Op/Div");
}


Tensor div(Graph &graph, float k, Tensor A, Sequence &prog,
          const std::string &debugPrefix) {
  const auto dType = A.elementType();
  auto B = graph.addConstant<float>(dType, A.shape(), k);
  return binaryOp(graph, B, A, prog, BinaryOp::DIVIDE, debugPrefix + "/Op/Div");
}


Tensor eq(Graph &graph, Tensor A, Tensor B, Sequence &prog,
          const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::EQUAL,
                  debugPrefix + "/Op/Equal");
}

Tensor exp(Graph &graph, Tensor A, Sequence &prog,
           const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::EXPONENT, debugPrefix + "/Op/Exp");
}

Tensor floor(Graph &graph, Tensor A, Sequence &prog,
             const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::FLOOR, debugPrefix + "/Op/Floor");
}

Tensor gt(Graph &graph, Tensor A, Tensor B, Sequence &prog,
          const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::GREATER_THAN,
                  debugPrefix + "/Op/Gteq");
}

Tensor gteq(Graph &graph, Tensor A, Tensor B, Sequence &prog,
            const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::GREATER_THAN_EQUAL,
                  debugPrefix + "/Op/Gteq");
}

Tensor isFinite(Graph &graph, Tensor A, Sequence &prog,
                const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::IS_FINITE,
                 debugPrefix + "/Op/IsFinite");
}

Tensor lt(Graph &graph, Tensor A, Tensor B, Sequence &prog,
          const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::LESS_THAN,
                  debugPrefix + "/Op/Lteq");
}

Tensor lteq(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::LESS_THAN_EQUAL,
                  debugPrefix + "/Op/Lteq");
}

Tensor log(Graph &graph, Tensor A, Sequence &prog,
           const std::string &debugPrefix) {
   return unaryOp(graph, A, prog, UnaryOp::LOGARITHM, debugPrefix + "/Op/Log");
}

Tensor logicalAnd(Graph &graph, Tensor A, Tensor B, Sequence &prog,
                  const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::LOGICAL_AND,
                  debugPrefix + "/Op/And");
}

Tensor logicalNot(Graph &graph, Tensor A, Sequence &prog,
                  const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::LOGICAL_NOT, debugPrefix + "/Op/Not");
}

Tensor logicalOr(Graph &graph, Tensor A, Tensor B, Sequence &prog,
                 const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::LOGICAL_OR,
                  debugPrefix + "/Op/Or");
}

Tensor max(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::MAXIMUM,
                  debugPrefix + "/Op/Max");
}

Tensor min(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::MINIMUM,
                  debugPrefix + "/Op/Min");
}

Tensor mul(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::MULTIPLY,
                  debugPrefix + "/Op/Mul");
}

Tensor neq(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::NOT_EQUAL,
                  debugPrefix + "/Op/Neq");
}

Tensor neg(Graph &graph, Tensor A, Sequence &prog,
           const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::NEGATE, debugPrefix + "/Op/Neg");
}

Tensor pow(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::POWER, debugPrefix + "/Op/Pow");
}

Tensor rem(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::REMAINDER,
                  debugPrefix + "/Op/Rem");
}

Tensor round(Graph &graph, Tensor A, Sequence &prog,
             const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::ROUND, debugPrefix + "/Op/Round");
}


Tensor shiftLeft(Graph &graph, Tensor A, Tensor B, Sequence &prog,
                 const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::SHIFT_LEFT,
                  debugPrefix + "/Op/Lsl");
}

Tensor shiftRight(Graph &graph, Tensor A, Tensor B, Sequence &prog,
                  const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::SHIFT_RIGHT,
                  debugPrefix + "/Op/Lsr");
}

Tensor shiftRightSignExtend(Graph &graph, Tensor A, Tensor B, Sequence &prog,
                            const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::SHIFT_RIGHT_SIGN_EXTEND,
                  debugPrefix + "/Op/Asr");
}

Tensor signum(Graph &graph, Tensor A, Sequence &prog,
              const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::SIGNUM, debugPrefix + "/Op/Signum");
}

Tensor sin(Graph &graph, Tensor A, Sequence &prog,
           const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::SIN, debugPrefix + "/Op/Sin");
}

Tensor sub(Graph &graph, Tensor A, Tensor B, Sequence &prog,
           const std::string &debugPrefix) {
  return binaryOp(graph, A, B, prog, BinaryOp::SUBTRACT,
                  debugPrefix + "/Op/Sub");
}

Tensor tanh(Graph &graph, Tensor A, Sequence &prog,
            const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::TANH, debugPrefix + "/Op/Tanh");
}

Tensor sqrt(Graph &graph, Tensor A, Sequence &prog,
            const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::SQRT, debugPrefix + "/Op/Sqrt");
}

Tensor square(Graph &graph, Tensor A, Sequence &prog,
            const std::string &debugPrefix) {
  return unaryOp(graph, A, prog, UnaryOp::SQUARE, debugPrefix + "/Op/Square");
}

Tensor select(Graph &graph, Tensor A, Tensor B, Tensor pred, Sequence &prog,
              const std::string &debugPrefix) {
  return ternaryOp(graph, A, B, pred, prog, TernaryOp::SELECT,
                   debugPrefix + "/Op/Select");
}

Tensor clamp(Graph &graph, Tensor A, Tensor lowerBound, Tensor upperBound,
             Sequence &prog, const std::string &debugPrefix) {
  return ternaryOp(graph, A, lowerBound, upperBound, prog, TernaryOp::CLAMP,
                   debugPrefix + "/Op/Clamp");
}

} // namespace popops
