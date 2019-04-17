#include "popops/ElementWise.hpp"

#include "ExprOpUtil.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/gcd.hpp"
#include <unordered_map>
#include <unordered_set>
#include <boost/optional.hpp>
#include "poplibs_support/OptionParsing.hpp"
#include <tbb/parallel_for.h>

#include <cassert>
#include <algorithm>

#include "ExprOpUtil.hpp"
#include "PerformanceEstimation.hpp"


using namespace poputil;
using namespace poplar;
using namespace poplar::program;

using popops::expr::UnaryOpType;
using popops::expr::BinaryOpType;
using popops::expr::TernaryOpType;
using popops::expr::BroadcastOpType;

namespace popops {

namespace {

struct MapOptions {
  bool enableVectorBroadcastOptimisations = true;
};

MapOptions parseOptionFlags(const OptionFlags &options) {
  MapOptions mapOpts;
  const poplibs::OptionSpec mapSpec{
    { "enableVectorBroadcastOptimisations",
    poplibs::OptionHandler::createWithBool(
        mapOpts.enableVectorBroadcastOptimisations)}
  };
  for (const auto &entry : options) {
    mapSpec.parse(entry.first, entry.second);
  }
  return mapOpts;
}

Type outputType(const Type &inType, enum UnaryOpType op) {
  if (op == UnaryOpType::IS_FINITE
      || op == UnaryOpType::LOGICAL_NOT) {
    return BOOL;
  } else {
    return inType;
  }
}

Type outputType(const Type &inType, BinaryOpType op) {
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

Type outputType(const Type &inType,
                TernaryOpType /*op*/) {
  return inType;
}

std::string vertexName(TernaryOpType op) {
  switch(op) {
  case TernaryOpType::CLAMP:
    return "popops::Clamp";
  case TernaryOpType::SELECT:
    return "popops::Select";
  }
  throw poputil::poplibs_error("Op not supported");
}

std::string debugName(UnaryOpType op) {
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
  case UnaryOpType::INVERSE:
    return "Inverse";
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

std::string debugName(BinaryOpType op) {
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
    case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
      return "InvStdDevToVariance";
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
    case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
      return "VarianceToInvStdDev";
  }
  throw poputil::poplibs_error("Op not supported");
}

std::string debugName(BroadcastOpType op) {
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

std::string debugName(TernaryOpType op) {
  switch(op) {
  case TernaryOpType::CLAMP:
    return "Clamp";
  case TernaryOpType::SELECT:
    return "Select";
  }
  throw poputil::poplibs_error("Op not supported");
}

BroadcastOpType binaryOpToBroadcastOp(BinaryOpType op) {
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

unsigned matchingDimension(Tensor in1, Tensor in2) {
  for(unsigned i = 0; i < in1.rank(); i++) {
    if(in2.dim(i) != 1) {
      return i;
    }
  }
  return 0;
}

bool haveScalarBroadcastVertexForOp(BinaryOpType op, bool inPlace,
                                    const Type &dType) {
  if (!inPlace) {
    return false;
  }

  switch (op) {
    case BinaryOpType::ADD:
    case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
    case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
    case BinaryOpType::SUBTRACT:
    case BinaryOpType::MULTIPLY:
      return (dType == HALF ||
              dType == FLOAT);
    default:
      return false;
  }
  POPLIB_UNREACHABLE();
}

bool haveInnerVectorBroadcastVertexForOp(BinaryOpType op, bool inPlace,
                                         const Type &dType) {
  if (dType != HALF &&
      dType != FLOAT) {
    return false;
  }
  switch (op) {
    case BinaryOpType::ADD:
    case BinaryOpType::SUBTRACT:
      return inPlace;
    case BinaryOpType::MULTIPLY:
      return !inPlace;
    default:
      return false;
  }
  POPLIB_UNREACHABLE();
}

bool haveOuterVectorBroadcastVertexForOp(BinaryOpType op, bool inPlace,
                                         const Type &dType) {
  if (!inPlace) {
    return false;
  }
  if (dType != HALF &&
      dType != FLOAT) {
    return false;
  }
  switch (op) {
    case BinaryOpType::ADD:
    case BinaryOpType::SUBTRACT:
    case BinaryOpType::MULTIPLY:
      return true;
    default:
      return false;
  }
  POPLIB_UNREACHABLE();
}


Tensor unaryOp(Graph &graph, Tensor in, Sequence &prog,
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

struct OpEvalResult {
  VertexInfo info;
  Tensor output;
};


/** Generate vertices to perform an element-wise operation on a tile.
 *
 *  \param graph            The graph to add vertices to.
 *  \param in1              LHS input operand.
 *  \param in2              RHS input operand.
 *  \param out              Output operand. If in-place this will be the same
 *                          as the LHS input operand `in1`.
 *  \param intervals        Contiguous regions for the output operand on this
 *                          tile.
 *  \param tile             The tile to add vertices to.
 *  \param cs               The compute set to add vertices to.
 *  \param op               Binary operation to perform.
 *  \param inPlace          Whether or not this operation is performed in-place
 *                          on the LHS input operand.
 */
void binaryOpGeneral(Graph &graph,
                     const Tensor &in1,
                     const Tensor &in2,
                     const Tensor &out,
                     const std::vector<std::vector<Interval>> &intervals,
                     unsigned tile,
                     const ComputeSet &cs,
                     BinaryOpType op,
                     bool inPlace) {
  const auto dType = in1.elementType();
  const auto &target = graph.getTarget();
  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(dType),
                         target.getAtomicStoreGranularity());
  // Single contiguous region, supervisor vertex.
  if (intervals.size() == 1) {
    const auto vertexClass =
        templateVertex(inPlace ? "popops::BinaryOp1DInPlaceSupervisor" :
                                 "popops::BinaryOp1DSupervisor",
                       op, dType);
    auto outRegion = concat(out.flatten().slices(intervals));
    auto in1Region = concat(in1.flatten().slices(intervals));
    auto in2Region = concat(in2.flatten().slices(intervals));
    auto v = graph.addVertex(cs, vertexClass);
    graph.connect(v["in2"], in2Region);
    if (inPlace) {
      graph.connect(v["in1Out"], outRegion);
    } else {
      graph.connect(v["in1"], in1Region);
      graph.connect(v["out"], outRegion);
    }
    graph.setTileMapping(v, tile);

  // Multiple contiguous regions, 2D vertices.
  } else {
    const auto vertexClass =
          templateVertex(inPlace ? "popops::BinaryOp2DInPlace" :
                                   "popops::BinaryOp2D",
                         op, dType);
    auto vertexRegions =
      splitRegionsBetweenWorkers(target, intervals,
                                 grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, vertexClass);
      auto outRegions = out.flatten().slices(regions);
      auto in1Regions = in1.flatten().slices(regions);
      auto in2Regions = in2.flatten().slices(regions);
      graph.connect(v["in2"], in2Regions);
      if (inPlace) {
        graph.connect(v["in1Out"], outRegions);
      } else {
        graph.connect(v["in1"], in1Regions);
        graph.connect(v["out"], outRegions);
      }
      graph.setTileMapping(v, tile);
    }
  }
}

void binaryOpGeneral(Graph &graph,
                     const Tensor &in1,
                     const Tensor &in2,
                     const Tensor &out,
                     Sequence &prog,
                     BinaryOpType op,
                     bool inPlace,
                     const std::string &debugPrefix="") {
  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto outFlat = out.flatten();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);
  graph.reorderToSimplify(&outFlat, {&in1Flat, &in2Flat});
  const auto mapping = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap =  mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    binaryOpGeneral(graph, in1Flat, in2Flat, outFlat,
                    tileContiguousRegions, tile, cs, op, inPlace);
  }
  prog.add(Execute(cs));
}

/** Generate vertices to perform an element-wise operation where
 *  the second operand is just one underlying unique element.
 *
 *  This assumes each element of the outer vector in `intervals`
 *  contains regions which are both contiguous in memory and
 *  cover a single unique underlying element in in2.
 *
 *  \param graph            The graph to add vertices to.
 *  \param in1              LHS input operand.
 *  \param in2              RHS input operand, the input that is broadcast.
 *  \param out              Output operand. If in-place this will be the same
 *                          as the LHS input operand `in1`.
 *  \param intervals        Contiguous regions for the output operand on this
 *                          tile.
 *  \param tile             The tile to add vertices to.
 *  \param cs               The compute set to add vertices to.
 *  \param op               Binary operation to perform.
 *  \param inPlace          Whether or not this operation is performed in-place
 *                          on the LHS input operand.
 *  \param uniformScalar    Whether or not the scalar for each contiguous
 *                          region in `intervals` is the same. If true this
 *                          allows use of smaller vertices in the 2-dimensional
 *                          case.
 */
void binaryOpBroadcastScalar(
    Graph &graph,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &out,
    const std::vector<std::vector<Interval>> &intervals,
    unsigned tile,
    const ComputeSet &cs,
    BinaryOpType op,
    bool inPlace,
    bool uniformScalar) {

  // Only in-place ops supported currently.
  assert(inPlace);

  const auto &target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto dType = in1.elementType();
  const auto grainSize =
      std::max<unsigned>(target.getVectorWidth(dType),
                         target.getAtomicStoreGranularity());

  // Use a simple heuristic to decide if creating multiple supervisor
  // vertices for multiple regions will be poorly balanced over
  // using 2D vertices.
  auto nRegions = intervals.size();
  bool useSupervisor =
    intervals.size() == 1 ||
    (intervals.size() < numWorkers &&
     std::all_of(intervals.begin(), intervals.end(),
                 [&](const std::vector<Interval> &is) {
                   auto elems =
                     std::accumulate(is.begin(), is.end(), std::size_t(0),
                       [](std::size_t total, const Interval &i) {
                         return total + i.size();
                       });
                   return elems >= ((numWorkers + 1) / 2) * grainSize;
                 }));

  if (useSupervisor) {
    const auto vertexClass =
      templateVertex("popops::BroadcastScalar1DInPlaceSupervisor",
                     binaryOpToBroadcastOp(op), dType);
    for (const auto &regions : intervals) {
      const auto outRegion = concat(out.flatten().slices(regions));
      const auto in2Region = concat(in2.flatten().slices(regions));
      const auto in2ScalarRegion = in2Region[0];
      const auto v = graph.addVertex(cs, vertexClass,
                                     {{"data", outRegion},
                                      {"B", in2ScalarRegion}});
      graph.setTileMapping(v, tile);
    }
  } else {
    const auto vertexClass =
      templateVertex(uniformScalar ? "popops::BroadcastScalar2DDataInPlace"
                                   : "popops::BroadcastScalar2DInPlace",
                     binaryOpToBroadcastOp(op), dType);
    const auto vertexRegions =
      splitRegionsBetweenWorkers(target, intervals,
                                 grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      const auto outRegions = out.flatten().slices(regions);
      const auto in2Regions = in2.flatten().slices(regions);
      const auto v = graph.addVertex(cs, vertexClass,
                                     {{"data", outRegions}});
      if (uniformScalar) {
        auto in2ScalarRegion = concat(in2Regions).flatten()[0];
        graph.connect(v["B"], in2ScalarRegion);
      } else {
        // Take the first element in each region as the scalar.
        // We know that this must be the same element for all in each
        // region, otherwise calling this function is invalid.
        auto in2ScalarRegions = in2Regions;
        for (auto &region : in2ScalarRegions) {
          region = region[0];
        }
        graph.connect(v["B"], in2ScalarRegions);
      }
      graph.setTileMapping(v, tile);
    }
  }
}


void binaryOpBroadcastScalar(Graph &graph,
                             const Tensor &in1,
                             const Tensor &in2,
                             const Tensor &out,
                             Sequence &prog,
                             BinaryOpType op,
                             bool inPlace,
                             const std::string &debugPrefix) {
  // Tensors in1, in2 and out will be the same broadcast shape.
  assert(in1.shape() == in2.shape() &&
         in2.shape() == out.shape());

  auto in1Flat = in1.flatten();
  auto outFlat = out.flatten();
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);
  graph.reorderToSimplify(&outFlat, {&in1Flat});
  const auto mapping = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap =  mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    binaryOpBroadcastScalar(graph, in1Flat, in2, outFlat,
                            tileContiguousRegions, tile,
                            cs, op, inPlace, true /* uniformScalar */);
  }
  prog.add(Execute(cs));
}

/** Generate vertices to perform an element-wise operation where
 *  the second operand is in the innermost (contiguous in memory)
 *  dimension. If certain requirements are not met this will back
 *  out. We guarantee that if this function does not return success
 *  then nothing will be added to the graph/compute set given.
 *
 *  \param graph            The graph to add vertices to.
 *  \param in1              LHS input operand.
 *  \param in2              RHS input operand, the input that is broadcast.
 *  \param out              Output operand. If in-place this will be the same
 *                          as the LHS input operand `in1`.
 *  \param intervals        Contiguous regions for the output operand on this
 *                          tile.
 *  \param numPatternElems  The length of the portion of `in2` that will
 *                          be broadcast for each contiguous region of
 *                          `out`.
 *  \param tile             The tile to add vertices to.
 *  \param cs               The compute set to add vertices to.
 *  \param op               Binary operation to perform.
 *  \param inPlace          Whether or not this operation is performed in-place
 *                          on the LHS input operand.
 *
 *  \return Whether or not we added the operation to the graph. If false
 *          it is guaranteed that nothing was added to the graph. If true
 *          the operation is part of the given compute set upon return.
 */
bool binaryOpBroadcastInnerVector(
    Graph &graph,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &out,
    const std::vector<std::vector<Interval>> &intervals,
    std::size_t numPatternElems,
    unsigned tile,
    const ComputeSet &cs,
    BinaryOpType op,
    bool inPlace) {

  const auto dType = in1.elementType();
  const auto &target = graph.getTarget();

  // TODO: In any case, account for the fact that only numPatternElements
  // that are multiples of certain numbers will make for performant
  // channel op vertices.

  auto canUseSupervisorVertex = [&](std::size_t size, std::size_t subSize) {
    return (size % subSize) == 0 &&
           (size / subSize) <= (target.getRptCountMax() & ~1UL) * 6;
  };
  auto packCount = [](std::size_t size, std::size_t subSize) -> std::uint16_t {
    auto blockCount = size / subSize;
    return ((blockCount / 6) << 3) | blockCount % 6;
  };

  // See if we can use channel ops vertices.
  if ((op == BinaryOpType::ADD || op == BinaryOpType::SUBTRACT) && inPlace) {
    float scale = op == BinaryOpType::SUBTRACT ? -1.0f : 1.0f;

    if (intervals.size() == 1) {
      const auto outRegion = concat(out.flatten().slices(intervals));
      auto in2Region = concat(in2.flatten().slices(intervals))
                       .slice(0, numPatternElems);
      auto vertexClass =
        templateVertex(scale == 1.0f ? "popops::AddToChannel"
                                     : "popops::ScaledAddToChannel",
                       dType);
      if (canUseSupervisorVertex(outRegion.numElements(),
                                 in2Region.numElements())) {
        std::uint16_t actsBlockCountPacked =
          packCount(outRegion.numElements(), in2Region.numElements());
        auto v = graph.addVertex(cs, vertexClass);
        graph.connect(v["addend"], in2Region);
        graph.connect(v["acts"], outRegion);
        graph.setInitialValue(v["actsBlockCountPacked"], actsBlockCountPacked);
        if (scale != 1.0f) {
          graph.setInitialValue(v["scale"], scale);
        }
        graph.setTileMapping(v, tile);
        return true;
      }
    }

    // Split work based on the size of the pattern.
    auto vertexClass =
      templateVertex(scale == 1.0f ? "popops::AddToChannel2D"
                                   : "popops::ScaledAddToChannel2D",
                     dType);
    const auto maxAddendLen =
      graph.getMaxVertexFieldValue(vertexClass, "addendLen");
    // If numPatternElems were some ludicrous number that doesn't
    // actually fit in numPatternElems then we could handle it and still
    // use channel ops but for now it seems unlikely.
    if (numPatternElems <= maxAddendLen &&
        (intervalSequenceNumElements(intervals) % numPatternElems) == 0) {
      const auto maxBlockCount = std::min<unsigned>(
          graph.getMaxVertexFieldValue(vertexClass, "actsBlockCount"),
          target.getRptCountMax()
        );
      auto vertexRegions =
        splitRegionsBetweenWorkers(target, intervals,
                                   numPatternElems,
                                   numPatternElems,
                                   maxBlockCount * numPatternElems);
      for (const auto &regions : vertexRegions) {
        auto outRegions = out.flatten().slices(regions);
        auto in2Regions = in2.flatten().slices(regions);
        for (auto &region : in2Regions) {
          region = region.slice(0, numPatternElems);
        }
        std::vector<std::uint16_t> addendLen(outRegions.size(),
                                             numPatternElems);
        std::vector<std::uint16_t> actsBlockCount(outRegions.size());
        for (std::size_t i = 0; i < outRegions.size(); ++i) {
          assert((outRegions[i].numElements() % numPatternElems) == 0);
          actsBlockCount[i] = outRegions[i].numElements() / numPatternElems;
        }
        auto v = graph.addVertex(cs, vertexClass);
        graph.setInitialValue(v["n"], outRegions.size());
        graph.connect(v["addend"], in2Regions);
        graph.connect(v["acts"], outRegions);
        graph.setInitialValue(v["addendLen"], std::move(addendLen));
        graph.setInitialValue(v["actsBlockCount"], std::move(actsBlockCount));
        if (scale != 1.0f) {
          graph.setInitialValue(v["scale"], scale);
        }
        graph.setTileMapping(v, tile);
      }
      return true;
    }
  }

  if (op == BinaryOpType::MULTIPLY && !inPlace) {
    if (intervals.size() == 1) {
      auto outRegion = concat(out.flatten().slices(intervals));
      auto in1Region = concat(in1.flatten().slices(intervals));
      auto in2Region = concat(in2.flatten().slices(intervals))
                       .slice(0, numPatternElems);
      auto vertexClass = templateVertex("popops::ChannelMul", dType);
      if (canUseSupervisorVertex(outRegion.numElements(),
                                 in2Region.numElements())) {
        std::uint16_t actsBlockCountPacked =
          packCount(outRegion.numElements(), in2Region.numElements());
        auto v = graph.addVertex(cs, vertexClass);
        graph.connect(v["actsIn"], in1Region);
        graph.connect(v["scale"], in2Region);
        graph.connect(v["actsOut"], outRegion);
        graph.setInitialValue(v["actsBlockCountPacked"], actsBlockCountPacked);
        graph.setTileMapping(v, tile);
        return true;
      }
    }

    // Split work based on the size of the pattern.
    auto vertexClass = templateVertex("popops::ChannelMul2D", dType);
    const auto maxScaleLen =
      graph.getMaxVertexFieldValue(vertexClass, "scaleLen");
    // If numPatternElems were some ludicrous number that doesn't
    // actually fit in numPatternElems then we could handle it and still
    // use channel ops but for now it seems unlikely.
    if (numPatternElems <= maxScaleLen &&
        (intervalSequenceNumElements(intervals) % numPatternElems) == 0) {
      const auto maxBlockCount = std::min<unsigned>(
          graph.getMaxVertexFieldValue(vertexClass, "actsBlockCount"),
          target.getRptCountMax()
        );
      auto vertexRegions =
        splitRegionsBetweenWorkers(target, intervals,
                                   numPatternElems,
                                   numPatternElems,
                                   maxBlockCount * numPatternElems);
      for (const auto &regions : vertexRegions) {
        auto outRegions = out.flatten().slices(regions);
        auto in1Regions = in1.flatten().slices(regions);
        auto in2Regions = in2.flatten().slices(regions);
        for (auto &region : in2Regions) {
          region = region.slice(0, numPatternElems);
        }
        std::vector<std::uint16_t> scaleLen(outRegions.size(), numPatternElems);
        std::vector<std::uint16_t> actsBlockCount(outRegions.size());
        for (std::size_t i = 0; i < outRegions.size(); ++i) {
          assert((outRegions[i].numElements() % numPatternElems) == 0);
          actsBlockCount[i] = outRegions[i].numElements() / numPatternElems;
        }
        auto v = graph.addVertex(cs, vertexClass);
        graph.setInitialValue(v["n"], outRegions.size());
        graph.connect(v["scale"], in2Regions);
        graph.connect(v["actsIn"], in1Regions);
        graph.connect(v["actsOut"], outRegions);
        graph.setInitialValue(v["scaleLen"], std::move(scaleLen));
        graph.setInitialValue(v["actsBlockCount"], std::move(actsBlockCount));
        graph.setTileMapping(v, tile);
      }
      return true;
    }
  }

  return false;
}

/** Generate vertices to perform an element-wise operation where
 *  the second operand's elements are repeated `broadcastFactor`
 *  number of times each, and whose total (pre-broadcast) no.
 *  of elements is `numPatternElems`. If `broadcastFactor` *
 *  `numPatternElems` is less than the no. of elements in the first
 *  operand the pattern will repeat.
 *
 *  If certain requirements are not met this will back out. We
 *  guarantee that if this function does not return success then
 *  nothing will be added to the graph/compute set given.
 *
 *  \param graph            The graph to add vertices to.
 *  \param in1              LHS input operand.
 *  \param in2              RHS input operand, the input that is broadcast.
 *  \param out              Output operand. If in-place this will be the same
 *                          as the LHS input operand `in1`.
 *  \param intervals        Contiguous regions for the output operand on this
 *                          tile.
 *  \param numPatternElems  The length of the portion of `in2` that will
 *                          be broadcast for each contiguous region of
 *                          `out`.
 *  \param broadcastFactor  The number of times each element of the RHS input
 *                          is repeated over the output before moving to the
 *                          next.
 *  \param tile             The tile to add vertices to.
 *  \param cs               The compute set to add vertices to.
 *  \param op               Binary operation to perform.
 *  \param inPlace          Whether or not this operation is performed in-place
 *                          on the LHS input operand.
 *
 *  \return Whether or not the operation was added to the graph. If false
 *          it is guaranteed that nothing was added to the graph. If true
 *          the operation is part of the given compute set upon return.
 */
bool binaryOpBroadcastOuterVector(
    Graph &graph,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &out,
    const std::vector<std::vector<Interval>> &intervals,
    std::size_t numPatternElems,
    std::size_t broadcastFactor,
    unsigned tile,
    const ComputeSet &cs,
    BinaryOpType op,
    bool inPlace) {

  const auto dType = in1.elementType();
  const auto &target = graph.getTarget();

  // TODO: Probably we should also keep track of what parts of the
  // given pattern are contiguous. If they are not contiguous it may
  // be a space saving to use a 2D scalar broadcast vertex rather
  // than gathering the elements of the pattern.

  auto canUseOuterVectorVertex =
    [&](const std::vector<std::vector<Interval>> &intervals)  {
    const auto nElems = intervalSequenceNumElements(intervals);
    if ((nElems % broadcastFactor) != 0) {
      return false;
    }
    return true;
  };

  if (canUseOuterVectorVertex(intervals)) {
    auto outRegion = concat(out.flatten().slices(intervals));
    auto in2Region = concat(in2.flatten().slices(intervals))
                     .slice(0, numPatternElems * broadcastFactor)
                     .subSample(broadcastFactor, 0);
    // TODO: Create a vertex type which only handles 64-bit aligned
    // and a multiple of 64-bit sized data, and use the aligned
    // and non-aligned variants depending on the broadcastFactor.
    // For now this vertex will always handle misaligned rows.
    auto vertexClass =
      templateVertex("popops::BroadcastVectorOuterInPlaceSupervisor",
                     binaryOpToBroadcastOp(op), dType);
    auto maxColumns = graph.getMaxVertexFieldValue(vertexClass, "columns");
    auto maxRows = graph.getMaxVertexFieldValue(vertexClass, "rows");
    auto rows = outRegion.numElements() / broadcastFactor;
    if (broadcastFactor <= maxColumns &&
        rows <= maxRows) {
      auto v = graph.addVertex(cs, vertexClass,
                               {{"data", outRegion},
                                {"B", in2Region}});
      graph.setInitialValue(v["columns"], broadcastFactor);
      graph.setInitialValue(v["rows"], rows);
      graph.setTileMapping(v, tile);
      return true;
    }
  }

  return false;
}

// Describes a pattern of broadcast that we can detect and
// use to produce a more efficient element-wise op where an
// operand is broadcasted.
struct BroadcastPattern {
  std::size_t innerFactor = 1;
  std::vector<Interval> pattern;
  std::size_t outerFactor = 1;
  std::size_t patternElements() const {
    return std::accumulate(pattern.begin(),
                           pattern.end(),
                           std::size_t(0),
                           [](std::size_t total, const Interval &i) {
                             return total + i.size();
                           });
  }
  std::size_t numElements() const {
    return patternElements() * innerFactor * outerFactor;
  }
};

// Takes a series of intervals and run-length encodes these.
// Analysis returns whether there is any common pattern we can
// use for broadcasting.
class BroadcastPatternAnalysis {
  // Run-length encoded pattern N elements -> broadcast vector index
  std::vector<std::pair<std::size_t, std::size_t>> encoded;
public:
  void append(const Interval &i) {
    std::size_t iOffset = 0;
    // Repeating elements run-length encoded.
    if (!encoded.empty() &&
        i.begin() == encoded.back().second) {
      encoded.back().first += i.size();
      ++iOffset;
      if (iOffset == i.size()) {
        return;
      }
    }
    // Otherwise add new entries to the encoded.
    while (iOffset < i.size()) {
      encoded.emplace_back(1, i.begin() + iOffset);
      ++iOffset;
    }
  }

  void analyse(std::vector<BroadcastPattern> &out) const {
    if (encoded.empty()) {
      return;
    }

    auto it = encoded.begin();
    while (it != encoded.end()) {
      auto lastIt = it;

      std::unordered_set<std::size_t> seen{lastIt->second};
      out.emplace_back();
      auto &pattern = out.back();
      pattern.innerFactor = lastIt->first;
      pattern.pattern.emplace_back(lastIt->second,
                                   lastIt->second + 1);
      std::size_t index = 0;
      std::size_t offset = 0;
      bool haveCompletePattern = false;
      // Iterator storing a restart point if we discover the
      // current pattern does not match the previous.
      auto restartIt = it;
      for (it = std::next(lastIt); it != encoded.end(); ++it) {
        if (index + offset == 0) {
          restartIt = it;
        }

        // If the innerFactor changes then we need to start
        // a new pattern.
        if (it->first != pattern.innerFactor) {
          break;
        }

        if (!haveCompletePattern && seen.count(it->second)) {
          haveCompletePattern = true;
          index = 0;
          offset = 0;
        }

        if (haveCompletePattern) {
          // If the pattern does not match the current pattern then
          // we need to start a new one.
          if (pattern.pattern[index].begin() + offset != it->second) {
            break;
          }
        } else {
          // Extend the last interval of the pattern if possible.
          if (it->second == pattern.pattern.back().end()) {
            pattern.pattern.back() =
              Interval(pattern.pattern.back().begin(),
                       pattern.pattern.back().end() + 1);
          } else {
            // Otherwise add a new interval.
            pattern.pattern.emplace_back(it->second, it->second + 1);
          }
        }

        ++offset;
        if (offset >= pattern.pattern[index].size()) {
          ++index;
          offset = 0;
          if (index >= pattern.pattern.size()) {
            index = 0;
            ++pattern.outerFactor;
          }
        }

        lastIt = restartIt;
      }
    }
  }
};


// Given a set of broadcast patterns that cover the given set of
// contiguous intervals, split the intervals such that there is
// a single contiguous region for each pattern. We require that
// a pattern already does not cross the boundaries between two
// contiguous regions.
//
// i.e.:
//
//   1 contiguous region : many patterns
//
// is transformed into:
//
//   1 contiguous region : 1 pattern.
//
std::vector<std::vector<Interval>>
splitContiguousRegionsByPattern(
    std::vector<std::vector<Interval>> intervals,
    const std::vector<BroadcastPattern> &patterns) {
  if (intervals.size() == patterns.size()) {
    return intervals;
  }
  std::vector<std::vector<Interval>> newIntervals;
  newIntervals.reserve(patterns.size());
  auto pIt = patterns.begin();
  for (auto &regions : intervals) {
    auto beginIt = regions.begin();
    auto endIt = beginIt;
    std::size_t offset = 0;
    while (endIt != regions.end()) {
      auto remainingElems = pIt->numElements();
      while (remainingElems > 0) {
        auto n = std::min(remainingElems, endIt->size() - offset);
        remainingElems -= n;
        offset += n;
        if (offset == endIt->size()) {
          ++endIt;
          offset = 0;
        }
      }
      newIntervals.emplace_back();
      auto &newRegions = newIntervals.back();
      newRegions.reserve(std::distance(beginIt, endIt) +
                         (offset > 0 ? 1 : 0));
      std::move(beginIt, endIt, std::back_inserter(newRegions));
      // If there is an offset left, split the interval at endIt
      if (offset) {
        newRegions.emplace_back(endIt->begin(), endIt->begin() + offset);
        *endIt = Interval(endIt->begin() + offset, endIt->end());
        offset = 0;
      }
      beginIt = endIt;
      ++pIt;
    }
  }
  return newIntervals;
}


// Construct a binary op where the second operand is broadcasted
// before the binary op is applied to it and the first operand to
// produce the output. We can perform more optimal operations in
// these cases.
//
// TODO: This currently only handles the RHS of the op being
// the broadcasted operand though this is slightly arbitrary as a
// restriction for not in-place ops as for commutative operations
// the operands could be switched. For non-commutative operations
// new vertex types would be needed.
void constructBroadcastBinaryOp(Graph &graph,
                                Sequence &prog,
                                const Tensor &in1_,
                                const Tensor &in2_,
                                const Tensor &out_,
                                BinaryOpType op,
                                bool inPlace,
                                const std::string &debugPrefix) {
  // Tensors in1, in2 and out will be the same broadcast shape.
  assert(in1_.shape() == in2_.shape() &&
         in2_.shape() == out_.shape());
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto dType = in1_.elementType();

  auto in1 = in1_.flatten();
  auto in2 = in2_.flatten();
  auto out = out_.flatten();
  graph.reorderToSimplify(&out, {&in1, &in2});
  const auto outMapping = graph.getTileMapping(out);

  std::vector<std::vector<BroadcastPattern>>
    tilePatterns(numTiles);
  std::vector<std::vector<std::vector<Interval>>>
    tileContiguousRegions(numTiles);

  tbb::parallel_for(unsigned(0), numTiles, [&](unsigned tile) {

    // We work with the contiguous intervals of the output with
    // respect to unique elements of the broadcasted input.
    std::vector<std::size_t> aliases;
    auto outRegions =
      graph.getSortedContiguousRegions(out, outMapping[tile]);
    auto in2Regions =
      graph.getSortedContiguousRegions(in2, outMapping[tile],
                                       false, &aliases);

    // Build a map from region start to the representative interval
    // for the underlying elements using the returned aliases.
    const auto aliasMap = [&] {
      std::map<std::size_t, std::size_t> m;
      std::size_t i = 0;
      for (const auto &regions : in2Regions) {
        for (const auto &region : regions) {
          m[region.begin()] = aliases[i++];
        }
      }
      return m;
    }();

    // Determine any patterns on each tile.
    std::vector<BroadcastPattern> patterns;
    patterns.reserve(outRegions.size());
    for (const auto &regions : outRegions) {
      BroadcastPatternAnalysis analysis;
      // Iterate contiguous regions of the output tensor and find
      // the sequence of unique element of the broadcasted tensor
      // which contributes to each element of the output.
      for (const auto &region : regions) {
        auto it = std::prev(aliasMap.upper_bound(region.begin()));
        auto lastIt = it;
        // Because the aliased interval begin is <= the region begin,
        // the first index in the aliases may not be the same as
        // region.begin()
        std::size_t beginOffset = region.begin() - lastIt->first;
        while (++it != aliasMap.end() &&
               it->first < region.end()) {
          auto size = it->first - (lastIt->first + beginOffset);
          analysis.append(Interval{lastIt->second + beginOffset,
                                   lastIt->second + beginOffset + size});
          lastIt = it;
          beginOffset = 0;
        }
        auto size = region.end() - (lastIt->first + beginOffset);
        analysis.append(Interval{lastIt->second + beginOffset,
                                 lastIt->second + beginOffset + size});
      }
      analysis.analyse(patterns);
    }
    tilePatterns[tile] = std::move(patterns);
    tileContiguousRegions[tile] = std::move(outRegions);
  });

  // Predicates for being able to use different methods on each tile.
  auto scalarBroadcastablePredicate = [](const BroadcastPattern &p) {
    return p.innerFactor > 1 &&
           p.patternElements() == 1 &&
           p.outerFactor == 1;
  };
  auto innerVectorBroadcastablePredicate = [](const BroadcastPattern &p) {
    return p.innerFactor == 1 && p.outerFactor > 1;
  };
  auto outerVectorBroadcastablePredicate = [](const BroadcastPattern &p) {
    return p.patternElements() > 1 && p.innerFactor > 1;
  };

  // Generate vertices from the analyses
  auto cs = graph.addComputeSet(debugPrefix);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    if (tileContiguousRegions[tile].empty()) {
      continue;
    }
    if (!tilePatterns[tile].empty()) {
      const auto &patterns = tilePatterns[tile];
      if (std::all_of(patterns.begin(), patterns.end(),
                      scalarBroadcastablePredicate) &&
          patterns.size() == tileContiguousRegions[tile].size() &&
          haveScalarBroadcastVertexForOp(op, inPlace, dType)) {
        bool uniformScalar =
          std::all_of(std::next(patterns.begin()), patterns.end(),
                      [&](const BroadcastPattern &p) {
                        return p.pattern == patterns.front().pattern;
                      });
        // TODO: Allow this method to handle cases where there are multiple
        // patterns per contiguous region. i.e. remove the
        // patterns.size() == tileContiguousRegions[tile].size() condition
        // and use splitContiguousRegionsByPattern to break up contiguous
        // regions for use.
        binaryOpBroadcastScalar(graph, in1, in2, out,
                                tileContiguousRegions[tile],
                                tile, cs, op, inPlace, uniformScalar);
        continue;
      }
      // TODO: Currently there is a restriction that all inner vector
      // broadcasts in a 2D vertex have the same length. This is to
      // make work division easy.
      if (std::all_of(patterns.begin(), patterns.end(),
                      innerVectorBroadcastablePredicate) &&
          std::all_of(std::next(patterns.begin()), patterns.end(),
                      [&](const BroadcastPattern &p) {
                        return (p.patternElements() ==
                                patterns.front().patternElements());
                      }) &&
          patterns.size() == tileContiguousRegions[tile].size() &&
          haveInnerVectorBroadcastVertexForOp(op, inPlace, dType)) {
        if (binaryOpBroadcastInnerVector(graph, in1, in2, out,
                                         tileContiguousRegions[tile],
                                         patterns[0].patternElements(),
                                         tile, cs, op, inPlace)) {
          continue;
        }
      }
      // TODO: Currently we only have a 1D vertex to perform this
      // kind of operation.
      if (std::any_of(patterns.begin(), patterns.end(),
                      outerVectorBroadcastablePredicate) &&
          haveOuterVectorBroadcastVertexForOp(op, inPlace, dType) &&
          patterns.size() == 1) {
        if (binaryOpBroadcastOuterVector(graph, in1, in2, out,
                                         tileContiguousRegions[tile],
                                         patterns[0].patternElements(),
                                         patterns[0].innerFactor,
                                         tile, cs, op, inPlace)) {
          continue;
        }
      }
    }
    // Always fall back on the general op for this tile if no valid specialised
    // op could be generated
    binaryOpGeneral(graph, in1, in2, out, tileContiguousRegions[tile],
                    tile, cs, op, inPlace);
  }
  prog.add(Execute(cs));
}


void validateBinaryOpInputs(const Tensor &in1,
                            const Tensor &in2,
                            const std::string &debugPrefix) {
  if (in1.elementType() != in2.elementType()) {
    throw poputil::poplibs_error("Binary Op must have same type for "
                                 "both operands: " + debugPrefix);
  }

  if (in1.shape() == in2.shape()) {
    return;
  }

  if (!canBroadcastToMatch(in1, in2)) {
    throw poputil::poplibs_error("Binary Op operands must be the same "
                                 "shape or be a valid broadcast of "
                                 "either tensor. See Broadcast.hpp header "
                                 "for specifics.");
  }
}

Tensor binaryOp(Graph &graph, Tensor in1, Tensor in2,
                Sequence &prog, BinaryOpType op, bool inPlace,
                const MapOptions &options, const std::string &debugPrefix_) {
  const auto debugPrefix = debugPrefix_ + "/Op/" + debugName(op);

  const auto in1Type = in1.elementType();
  const auto in2Type = in2.elementType();
  const bool in2IsScalar = in2.numElements() == 1;

  validateBinaryOpInputs(in1, in2, debugPrefix);

  // Broadcast the inputs to have the same shape here to cover all paths
  // for binary ops

  broadcastToMatch(in1, in2);
  const auto outType = outputType(in1Type, op);

  Tensor out;
  if (inPlace) {
    out = in1;
  } else {
    out = graph.clone(outType, in1, debugPrefix + "/Out");
    poputil::mapOutputForElementWiseOp(graph, {in1, in2}, out);
  }

  // Special case for scalar broadcast, because knowing this is a binary
  // op and that the broadcasted tensor is a single element means we
  // know what the most efficient way to implement this is across tiles.
  if (in2IsScalar && haveScalarBroadcastVertexForOp(op, inPlace, in1Type)) {
    // Single element broadcast
    binaryOpBroadcastScalar(graph, in1, in2, out, prog, op,
                            inPlace, debugPrefix);
    return out;
  }

  // Vector broadcast special case. We try and find the most efficient
  // way to perform the binary operation on each tile.
  if (options.enableVectorBroadcastOptimisations) {
    constructBroadcastBinaryOp(graph, prog, in1, in2, out, op, inPlace,
                               debugPrefix);
    return out;
  }

  // General case which works for any given tensors and ops.
  binaryOpGeneral(graph, in1, in2, out, prog, op, inPlace, debugPrefix);
  return out;
}

Tensor ternaryOp(Graph &graph, Tensor in1, Tensor in2, Tensor in3,
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

bool isRelational(expr::UnaryOpType op) {
  switch (op) {
  case expr::UnaryOpType::IS_FINITE:
    return true;
  default:
    return false;
  }
}

bool isRelational(expr::BinaryOpType op) {
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

bool isLogical(expr::UnaryOpType op) {
  switch (op) {
  case expr::UnaryOpType::LOGICAL_NOT:
    return true;
  default:
    return false;
  }
}

bool isLogical(expr::BinaryOpType op) {
  switch (op) {
  case expr::BinaryOpType::LOGICAL_AND:
  case expr::BinaryOpType::LOGICAL_OR:
    return true;
  default:
    return false;
  }
}

const Tensor &
getTensorFromPlaceHolder(const expr::PlaceHolder &p,
                         const std::vector<Tensor> &ts) {
  auto index = p.getIndex() - 1;
  if (index > ts.size()) {
    throw poplibs_error("Invalid placeholder _" + std::to_string(index + 1) +
                       " in expression");
  }
  return ts[index];
}

boost::optional<Type>
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
std::pair<Tensor, bool>
map(Graph &graph,
    const expr::Expr &expr,
    const std::vector<Tensor> &ts,
    program::Sequence &prog,
    const std::string &debugPrefix,
    const std::unordered_map<const expr::Expr *, Type> constTypes,
    bool topLevel,
    bool constructGraph,
    bool inPlace,
    const expr::Expr *&inPlaceExpr,
    const MapOptions &options) {

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
                 constructGraph, inPlace, inPlaceExpr, options);
    if (constructGraph) {
      return {unaryOp(graph, t.first, prog, opType, t.second, debugPrefix),
              t.second};
    } else {
      return t;
    }
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto opType = b->getOpType();
    auto lhs = map(graph, b->getLHS(), ts, prog, debugPrefix, constTypes, false,
                   constructGraph, inPlace, inPlaceExpr, options);
    auto rhs = map(graph, b->getRHS(), ts, prog, debugPrefix, constTypes, false,
                  constructGraph, false, inPlaceExpr, options);
    if (constructGraph) {
      return {binaryOp(graph, lhs.first, rhs.first, prog, opType, lhs.second,
                       options, debugPrefix), lhs.second};
    } else {
      return lhs;
    }
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto opType = t->getOpType();
    if (opType == TernaryOpType::SELECT) {
      auto lhs = map(graph, t->getArg0(), ts, prog, debugPrefix, constTypes,
                  false, constructGraph, inPlace, inPlaceExpr, options);
      auto rhs = map(graph, t->getArg1(), ts, prog, debugPrefix, constTypes,
                  false, constructGraph, false, inPlaceExpr, options);
      auto pred = map(graph, t->getArg2(), ts, prog, debugPrefix, constTypes,
                  false, constructGraph, false, inPlaceExpr, options);
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
                  false, constructGraph, inPlace, inPlaceExpr, options);
      auto lower = map(graph, t->getArg1(), ts, prog, debugPrefix, constTypes,
                    false, constructGraph, false, inPlaceExpr, options);
      auto upper = map(graph, t->getArg2(), ts, prog, debugPrefix, constTypes,
                    false, constructGraph, false, inPlaceExpr, options);
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

std::unordered_map<const expr::Expr *, Type>
getConstType(const expr::Expr &expr, const std::vector<Tensor> &ts) {
  std::unordered_map<const expr::Expr *, Type> constTypes;
  std::vector<const expr::Expr *> unknown;
  auto type = inferType(expr, ts, constTypes, unknown);
  if (!type || !unknown.empty())
    throw poplibs_error("Cannot infer type of expression");
  return constTypes;
}

} // end anonymous namespace

Tensor map(Graph &graph, const expr::Expr &expr,
           const std::vector<Tensor> &ts,
           program::Sequence &prog,
           const std::string &debugPrefix,
           const OptionFlags &options) {
  auto opts = parseOptionFlags(options);
  auto constTypes = getConstType(expr, ts);
  const expr::Expr *inplaceExpr = nullptr;
  return map(graph, expr, ts, prog, debugPrefix, constTypes, true, true, false,
             inplaceExpr, opts).first;
}

void mapInPlace(Graph &graph, const expr::Expr &expr,
                const std::vector<Tensor> &ts,
                program::Sequence &prog,
                const std::string &debugPrefix,
                const OptionFlags &options) {
  auto opts = parseOptionFlags(options);
  auto constTypes = getConstType(expr, ts);
  const expr::Expr *inPlaceExpr = nullptr;
  const bool doInPlace = !ts[0].containsAliases() && !ts[0].containsConstant();
  if (doInPlace) {
    // As the tree is traveresed, find the last expression which uses the
    // tensor used for in-place operation as a placeholder
    map(graph, expr, ts, prog, debugPrefix, constTypes, true, false, false,
        inPlaceExpr, opts);
  }
  auto t = map(graph, expr, ts, prog, debugPrefix, constTypes, true, true,
               doInPlace, inPlaceExpr, opts);
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
