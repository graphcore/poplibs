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
#include <boost/optional.hpp>
#include "poplibs_support/OptionParsing.hpp"

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

static unsigned matchingDimension(Tensor in1, Tensor in2) {
  for(unsigned i = 0; i < in1.rank(); i++) {
    if(in2.dim(i) != 1) {
      return i;
    }
  }
  return 0;
}
static bool checkForBroadcastOp(BinaryOpType op,
                                std::pair<Tensor, bool> lhs,
                                std::pair<Tensor, bool> rhs,
                                bool vectorOptimise) {

  if(op == BinaryOpType::INV_STD_DEV_TO_VARIANCE ||
                               op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) {
    if(!lhs.second)
      throw poputil::poplibs_error("Op only supports InPlace");
    if(rhs.first.rank() != 0)
      throw poputil::poplibs_error("Op requires a scalar second operand");
  }

  // Is it possible to broadcast the scalar without a copy, given the
  // operations avilable?
  if(lhs.first.rank() != rhs.first.rank() && rhs.first.numElements() == 1) {
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
  if(vectorOptimise) {
    if(op == BinaryOpType::ADD ||
       op == BinaryOpType::MULTIPLY ||
       op == BinaryOpType::SUBTRACT) {
      if(detectVectorBroadcastOperands(lhs.first, rhs.first)) {
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

struct OpEvalResult {
  VertexInfo info;
  Tensor output;
};

static OpEvalResult binaryOpSameSize(Graph &graph,
                      const Tensor &in1,
                      const Tensor &in2,
                      Tensor &out,
                      Sequence &prog, BinaryOpType op, bool inPlace,
                      const std::string &debugPrefix="",
                      const bool generateVertices=false) {

  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto outFlat = out.flatten();
  const auto in1Type = in1Flat.elementType();
  const auto &target = graph.getTarget();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);
  VertexInfo costingInfo = {0};
  graph.reorderToSimplify(&outFlat, {&in1Flat, &in2Flat});
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
      const auto vertexTemplate =
          templateVertex(inPlace ? "popops::BinaryOp1DInPlaceSupervisor" :
                                   "popops::BinaryOp1DSupervisor",
                         op, in1Type);
      costingInfo.vertices += numWorkerContexts;
      costingInfo.slices += numWorkerContexts;
      if(generateVertices) {
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
        costingInfo.vertices ++;
        costingInfo.slices += regions.size();
        if(generateVertices) {
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
  if(generateVertices) {
    prog.add(Execute(cs));
  }
  return {costingInfo, out};

}

static Tensor binaryOpIn2Scalar(Graph &graph,
                      const Tensor &in1,
                      const Tensor &in2,
                      Tensor &out,
                      Sequence &prog, BinaryOpType op, bool inPlace,
                      const std::string &debugPrefix) {
  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto outFlat = out.flatten();
  const auto in1Type = in1Flat.elementType();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);
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
      const auto vertexTemplate =
          templateVertex("popops::BroadcastOp1DInPlaceSupervisor",
                                  binaryToBroadcastOp(op), in1Type);
      auto v = graph.addVertex(cs, vertexTemplate,
                          {{"data", concat(outFlat.slices(thisTileMap))},
                           {"B", in2Flat.reshape({})}});
      graph.setTileMapping(v, tile);
    }
    else {
      const auto vertexTemplate =
          templateVertex("popops::BroadcastOp2DInPlace",
                                  binaryToBroadcastOp(op), in1Type);
      auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize);
      for (const auto &regions : vertexRegions) {
        auto v = graph.addVertex(cs, vertexTemplate,
                          {{"data", outFlat.slices(regions)},
                             {"B", in2Flat.reshape({})}});
        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(cs));
  return out;
}

struct CombinedRegionInfo {
  unsigned long regions;
  unsigned long index;
};


static CombinedRegionInfo  countContiguousRegionsWithSameAddend(
                const std::vector<std::vector<Interval>> &tileContiguousRegions,
                const unsigned innerDimsSize,
                const unsigned in2Size,
                const unsigned tensorSize) {
  CombinedRegionInfo combinedRegions = {0};

  for( auto &regions : tileContiguousRegions) {
    auto regionsToProcess = regions.size();
    unsigned i = 0;
    while(i < regionsToProcess) {
      auto baseIndex = (regions[i].begin() / innerDimsSize ) % in2Size;
      auto baseIndexEnd = (regions[i].end() / innerDimsSize ) % in2Size;
      if(baseIndex != baseIndexEnd || regions[i].size() > innerDimsSize) {
        combinedRegions.regions = 0;
        return combinedRegions;
      }
      const auto &baseRegion = regions[i];
      auto j = i+1;
      unsigned regionsConsumed = 1;
      while(j < regionsToProcess) {
        const auto index = (regions[j].begin() / innerDimsSize ) % in2Size;
        const auto indexEnd = ((regions[j].end()-1) / innerDimsSize ) % in2Size;

        if(index == baseIndex &&
              index == indexEnd &&
              regions[j].size() == baseRegion.size()) {
          regionsConsumed++;
        }
        else {
          break;
        }
        j++;
      }
      i += regionsConsumed;
      combinedRegions.regions ++;
      combinedRegions.index = baseIndex;
    }
  }
  return combinedRegions;
}

static std::vector<Interval> divideRegionsRespectingBoundary(
                  std::vector<std::vector<Interval>> tileContiguousRegions,
                  unsigned boundary) {
  std::vector<Interval> dividedRegions;
  for( auto &regions : tileContiguousRegions) {
    for( auto &region : regions) {
      auto begin = region.begin();
      auto end = region.end();
      while(begin != end) {
        if(begin % boundary) {
          // The input region doesn't start on a boundary
          auto length = std::min(end-begin, boundary - (begin % boundary));
          dividedRegions.push_back({begin, begin+length});
          begin += length;
        }
        else if((end-begin < boundary) && (begin/boundary) == (end/boundary)) {
          // The input region begins on a boundary and doesn't cross a boundary
          dividedRegions.push_back({begin, end});
          begin = end;
        }
        else{
          // The input region begins on a boundary and crosses a boundary
          dividedRegions.push_back({begin, begin + boundary});
          begin += boundary;
        }
      }
    }
  }
  return dividedRegions;
}

static OpEvalResult binaryOpIn2Vector(Graph &graph,
                                Tensor in1, Tensor in2, Tensor out,
                                Sequence &prog,
                                const BinaryOpType op, const bool inPlace,
                                const std::string &debugPrefix="",
                                const bool generateVertices=false) {
  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto in2Size = in2Flat.dim(0);
  auto outFlat = out.flatten();
  const auto in1Type = in1Flat.elementType();
  const auto &target = graph.getTarget();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  const auto dimsIn1 = in1.shape();
  const auto matchingDim = matchingDimension(in1,
                                            extendDimensionsToMatch(in1, in2));

  VertexInfo costingInfo = {0};
  auto rowLength = dimsIn1.back();
  const bool innerDim = (matchingDim == in1.rank()-1);

  const auto mapping = graph.getTileMapping(in1Flat);
  const unsigned innerDimsSize = innerDim ? 1 :
            std::accumulate(dimsIn1.begin()+matchingDim+1, dimsIn1.end(), 1,
                                                std::multiplies<unsigned>());
  const auto grainSize = std::max<unsigned>(target.getVectorWidth(in1Type),
                         target.getAtomicStoreGranularity());

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
                      graph.getSortedContiguousRegions(in1Flat, mapping[tile]);
    if(tileContiguousRegions.size()) {
      std::vector<Interval> vertexRegions;
      CombinedRegionInfo combinedRegions;
      bool oneCombinedRegion =  false;
      if(!innerDim){
        // Check if all regions on the tile will use the same single, scalar
        // addend
        combinedRegions = countContiguousRegionsWithSameAddend(
                                                    tileContiguousRegions,
                                                    innerDimsSize,
                                                    in2Size,
                                                    in1Flat.numElements());
         if(tileContiguousRegions.size() == 1 && combinedRegions.regions == 1 ){
            oneCombinedRegion =true;
          }
      }
      if( !oneCombinedRegion ) {
        // For each tile split the elements with no region bridging a boundary
        // where the simple codelets cannot continue to work.
        vertexRegions = divideRegionsRespectingBoundary(tileContiguousRegions,
                                          innerDim ? rowLength : innerDimsSize);
      }

      if(innerDim) {
        if(vertexRegions.size() == 1) {
          auto regionSize = vertexRegions[0].size();
          auto in2Start = vertexRegions[0].begin() % rowLength;
          costingInfo.vertices += numWorkerContexts;
          costingInfo.slices += numWorkerContexts;
          if(generateVertices) {
           auto v = graph.addVertex(cs,
                   templateVertex("popops::BinaryOp1DInPlaceSupervisor",
                   op, in1Type),
                   {{"in1Out", in1Flat.slice(vertexRegions[0])},
                    {"in2", in2Flat.slice({in2Start, in2Start + regionSize})}});
            graph.setTileMapping(v, tile);
          }
        }
        else {
          auto workerRegions = splitRegionsBetweenWorkers(target, vertexRegions,
                                                grainSize, 2 * grainSize);
          for (const auto &regions : workerRegions) {
            std::vector<Interval> in2Regions;
            for(unsigned i = 0; i < regions.size(); i++) {
              in2Regions.push_back({regions[i].begin() % rowLength,
                        regions[i].size() + (regions[i].begin() % rowLength)});
            }
            costingInfo.slices += regions.size();
            costingInfo.vertices ++;
            if(generateVertices) {
              auto v = graph.addVertex(cs,
                  templateVertex("popops::BinaryOp2DInPlace", op, in1Type),
                                  {{"in1Out", in1Flat.slices(regions)},
                                   {"in2", in2Flat.slices(in2Regions)}});;
              graph.setTileMapping(v, tile);
            }
          }
        }
      }
      else {
        if(vertexRegions.size() == 1) {
          costingInfo.vertices += numWorkerContexts;
          costingInfo.slices += numWorkerContexts;
          if(generateVertices) {
            auto index = (vertexRegions[0].begin() / innerDimsSize ) % in2Size;
            auto v = graph.addVertex(cs,
                        templateVertex("popops::BroadcastOp1DInPlaceSupervisor",
                          binaryToBroadcastOp(op), in1Type),
                        {{"data", in1Flat.slice(vertexRegions[0])},
                        {"B", in2Flat[index]}});
            graph.setTileMapping(v, tile);
          }
        }
        else if(oneCombinedRegion) {
          costingInfo.vertices += numWorkerContexts;
          costingInfo.slices += numWorkerContexts;
          if(generateVertices) {
            std::vector<Interval> slicesToProcess;
            for(auto regions : tileContiguousRegions) {
              slicesToProcess.insert(slicesToProcess.end(),
                                        regions.begin(), regions.end());
            }
            auto v = graph.addVertex(cs,
                        templateVertex("popops::BroadcastOp1DInPlaceSupervisor",
                          binaryToBroadcastOp(op), in1Type),
                        {{"data", concat(in1Flat.slices(slicesToProcess))},
                        {"B", in2Flat[combinedRegions.index]}});
            graph.setTileMapping(v, tile);
          }
        }
        else {
          auto workerRegions =
              splitRegionsBetweenWorkers(target, vertexRegions,
                                       grainSize, 2 * grainSize);
          for (const auto &regions : workerRegions) {
             std::vector<Interval> in2Regions;
            // Check if the regions present on the tile access multiple indices
            bool manyIndices = false;
            for(unsigned i = 0; i < regions.size(); i++) {
              auto index = (regions[i].begin() / innerDimsSize) % in2Size;
              in2Regions.push_back({index, index+1});
              if(index != in2Regions[0].begin()) {
                manyIndices = true;
              }
            }
            costingInfo.slices += regions.size();
            costingInfo.vertices ++;
            if(generateVertices) {
              auto v = manyIndices ?
                    graph.addVertex(cs,
                    templateVertex("popops::BroadcastOpBVector2DInPlace",
                                     binaryToBroadcastOp(op), in1Type),
                                    {{"data", in1Flat.slices(regions)},
                                     {"B", in2Flat.slices(in2Regions)}}) :
                    graph.addVertex(cs,
                    templateVertex("popops::BroadcastOp2DInPlace",
                                    binaryToBroadcastOp(op), in1Type),
                                    {{"data", in1Flat.slices(regions)},
                                     {"B", in2Flat[in2Regions[0].begin()]}});
              graph.setTileMapping(v, tile);
            }
          }
        }
      }
    }
  }
  if(generateVertices) {
    prog.add(Execute(cs));
  }
  return {costingInfo, out};
}

static OpEvalResult binaryOpUsingChannelOp(Graph &graph,
                      Tensor in1, Tensor in2, Tensor out,
                      BinaryOpType op, Sequence &prog, bool inPlace,
                      const std::string &debugPrefix="",
                      const bool generateVertices=false) {
  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto outFlat = out.flatten();
  const auto in1Type = in1Flat.elementType();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);
  const auto mapping = graph.getTileMapping(in1Flat);
  const auto numWorkerContexts = target.getNumWorkerContexts();
  VertexInfo costingInfo = {0};

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap =  mapping[tile];
    const auto tileContiguousRegions = graph.getSortedContiguousRegions(in1Flat,
                                                                  thisTileMap);
    // flatten the contiguous tile region vector so we can search through all
    // regions.
    // This can have the effect of introducing copies where there are
    // multiple regions that are non-contiguous.  It is expected that the
    // benefit of using channel add/mul vertices will make this worthwhile.
    std::vector<Interval> tileRegions;
    for(auto &regions : tileContiguousRegions ){
      tileRegions.insert(tileRegions.end(),regions.begin(), regions.end());
    }

    std::vector<Interval> slicesToProcess;
    auto regionsToProcess = tileRegions.size();
    unsigned i = 0;
    while(i < regionsToProcess) {
      unsigned begin, end, addendBegin, addendEnd;
      unsigned thisBlockCount = 1, regionsConsumed = 1;
      auto firstRegionRemaining = tileRegions[i].size();
      auto regionBegin = tileRegions[i].begin();
      auto regionEnd = tileRegions[i].end();
      while(firstRegionRemaining) {
        if(tileRegions[i].size() >= in2.numElements() ||
          (regionEnd - regionBegin + (regionBegin % in2.dim(0))) > in2.dim(0)) {
          // The region may be longer than the addend, or span past a point
          // where the addend needs to be restarted.

          // Find begin, end matching the beginning/end of the addend,
          // rounded up / down so they remain within the region
          begin = (regionBegin + (in2.numElements()-1))/ in2.numElements();
          begin *= in2.numElements();
          end = (regionEnd )/ in2.numElements();
          end *= in2.numElements();

          if(begin != regionBegin) {
            // Deal with a tail end of addend at the region start
            auto length = begin - regionBegin;
            addendBegin = regionBegin % in2.dim(0);
            addendEnd = addendBegin + length;
            slicesToProcess.push_back({regionBegin, begin});
            regionBegin = begin;
            firstRegionRemaining -= length;
         }
          else if(end != regionEnd) {
            // Deal with part of the addend start at the region end
            auto length = regionEnd - end;
            addendBegin = end % in2.dim(0);
            addendEnd = addendBegin + length;
            firstRegionRemaining -= length;
            slicesToProcess.push_back({end, regionEnd});
            regionEnd = end;
          }
          else {
            // Identify (potentially multiple)
            // whole addend slices within the region
            thisBlockCount = (end - begin) /in2.numElements();
            firstRegionRemaining -= (end - begin);
            addendBegin = 0;
            addendEnd = in2.dim(0);
            slicesToProcess.push_back({begin, end});
         }

        }
        else {
          // The base region is smaller than the addend, and may be followed by
          // another region which uses the same slice of addend
          auto length = regionEnd - regionBegin;
          addendBegin = regionBegin % in2.dim(0);
          addendEnd = addendBegin + length;

          firstRegionRemaining = 0;
          auto j = i+1;
          slicesToProcess.push_back({regionBegin, regionEnd});
          // Check each consecutive region for 2 properties:
          // 1. Begins at a point that the same element of the addend can be
          //    applied as for the base region
          // 2. Has the same length as the base region

          while(j < regionsToProcess) {
            if(tileRegions[j].size() == length &&
                          tileRegions[j].begin() % in2.dim(0) == addendBegin) {
              thisBlockCount++;
              slicesToProcess.push_back({tileRegions[j].begin(),
                                                        tileRegions[j].end()});
            }
            else {
              break;
            }
            j++;
          }
          regionsConsumed += thisBlockCount-1;
        }
        costingInfo.vertices += numWorkerContexts;
        costingInfo.slices += thisBlockCount;
        costingInfo.addendLen = std::max(costingInfo.addendLen,
                                                        addendEnd-addendBegin);
        if(generateVertices) {
          auto actsBlockCountPacked = ((thisBlockCount / 6) << 3)
                                        | (thisBlockCount % 6);
          if(op == BinaryOpType::MULTIPLY) {
              auto v = graph.addVertex(cs,
                          templateVertex("popops::ChannelMul", in1Type),
                          {{"actsIn", concat(in1Flat.slices(slicesToProcess))},
                          {"actsOut", concat(outFlat.slices(slicesToProcess))},
                          {"scale", in2.slice(addendBegin, addendEnd)}});
              graph.setInitialValue(v["actsBlockCountPacked"],
                                                        actsBlockCountPacked);
              graph.setTileMapping(v, tile);
          }
          else {
            auto v = graph.addVertex(cs,templateVertex(op == BinaryOpType::ADD ?
                          "popops::AddToChannel" : "popops::ScaledAddToChannel",
                          in1Type),
                          {{"acts", concat(in1Flat.slices(slicesToProcess))},
                          {"addend", in2.slice(addendBegin, addendEnd)}});
            graph.setInitialValue(v["actsBlockCountPacked"],
                                                        actsBlockCountPacked);
            if (op == BinaryOpType::SUBTRACT) {
              graph.setInitialValue(v["scale"], -1.0);
            }
            graph.setTileMapping(v, tile);
          }
        }
        slicesToProcess.clear();
      }
      i += regionsConsumed;
    }
  }
  if(generateVertices) {
    prog.add(Execute(cs));
  }
  if(inPlace) {
    return {costingInfo, in1};
  }
  else{
    return {costingInfo, out};
  }
}

// Create a partial broadcast operand with the matching dimension duplicated
// by a broadcast ratio = innermost dimension size.  The matching dimension
// will be shuffled to become the 2nd innermost dimension
static Tensor createPartialBroadcastOperand(Tensor in1, Tensor in2) {

  auto matchingDimReshape = matchingDimension(in1, in2);
  const auto in1Shape = in1.shape();
  auto broadcastRatio = in1.dim(in1.rank()-1);
  in2 = in2.broadcast(broadcastRatio, matchingDimReshape);
  in2 = in2.reshape({broadcastRatio, in1.dim(matchingDimReshape)});

  return in2.transpose().flatten();
}

static BinaryOpMethod chooseBinaryOpMethod(Graph &graph, Sequence &prog,
                              BinaryOpType op,
                              bool inPlace, Tensor in1, Tensor in2,
                              Tensor out, unsigned matchingDim,
                              const std::vector<unsigned> &dimsShuffled)
{

  auto in1Shaped = in1.dimShuffle(dimsShuffled);
  std::vector<Costs> costs;
  const auto target = graph.getTarget();

  // Can we use add to channel / channelMul?
  if( (inPlace && ( op == BinaryOpType::ADD || op == BinaryOpType::SUBTRACT)) ||
      (!inPlace && op == BinaryOpType::MULTIPLY)) {

    auto in2Broadcast = extendDimensionsToMatch(in1, in2);

    if(dimsShuffled.back() == matchingDim) {
      // Dim to broadcast is stored contiguously in memory, use channel ops
      auto costResult = binaryOpUsingChannelOp(graph, in1Shaped,
                                    in2Broadcast.flatten(),
                                    out, op, prog, inPlace);

      costs.push_back(simpleBinaryOpCostEstimate(BinaryOpMethod::CHANNEL_OP,
                      costResult.info, dimsShuffled, matchingDim, in1, target));
    }
    else if(in1.rank() >= 2){
      // Dim to broadcast is not stored contiguously in memory, so partially
      // broadcast by copying, then use channel ops.
      unsigned matchingDimReshape = matchingDimension(in1Shaped,
                                        in2Broadcast.dimShuffle(dimsShuffled));

      in2Broadcast = createPartialBroadcastOperand(in1Shaped,
                                        in2Broadcast.dimShuffle(dimsShuffled));
      in1Shaped = in1Shaped.dimShufflePartial({matchingDimReshape},
                                             {in1Shaped.rank()-2});
      auto costResult = binaryOpUsingChannelOp(graph, in1Shaped, in2Broadcast,
                                                        out, op, prog, inPlace);
      costs.push_back(simpleBinaryOpCostEstimate(
              BinaryOpMethod::BROADCAST_AND_CHANNEL_OP,
              costResult.info, dimsShuffled, matchingDimReshape, in1, target));
    }
  }
  // Broadcasting using the broadcast op method - repetition of a scalar by the
  // vertex code
  if( inPlace &&
      (op == BinaryOpType::ADD ||
       op == BinaryOpType::SUBTRACT ||
       op == BinaryOpType::MULTIPLY)) {

    auto in2Reshaped = extendDimensionsToMatch(in1, in2);
    auto costResult = binaryOpIn2Vector(graph, in1.dimShuffle(dimsShuffled),
                                    in2Reshaped.dimShuffle(dimsShuffled),
                                    in1.dimShuffle(dimsShuffled),
                                    prog, op, inPlace);

    costs.push_back(simpleBinaryOpCostEstimate(BinaryOpMethod::VECTOR_BROADCAST,
                      costResult.info, dimsShuffled, matchingDim, in1, target));
  }
  // Broadcasting using default copy to broadcast method
  broadcastToMatch(in1, in2);
  auto costResult = binaryOpSameSize(graph,  in1, in2, in1, prog, op, inPlace);
  costs.push_back(simpleBinaryOpCostEstimate(BinaryOpMethod::COPY_BROADCAST,
                      costResult.info, dimsShuffled, matchingDim, in1, target));

  // Find the lowest cost recorded for each of the allowed methods
  std::uint64_t minCost = std::numeric_limits<std::uint64_t>::max();
  BinaryOpMethod minCostMethod = BinaryOpMethod::COPY_BROADCAST;
  for(auto cost : costs) {
    if(cost.copy + cost.vertices < minCost) {
      minCostMethod = cost.method;
      minCost = cost.copy + cost.vertices;
    }
  }
  return  minCostMethod;
}

// Detect which dimensions of a tensor are grouped in memory.  Use that result
// and the remining dimensions to construct a vector to use to dimShuffle
// the tensor for improved grouping.
static std::vector<unsigned> interpretDimGroupings(Graph &graph,
                                          const Tensor &in1){

  auto grouping = detectDimGroupings(graph, in1);

  std::vector<unsigned> dimsShuffled;
  for(auto group : grouping) {
    if(std::find(dimsShuffled.begin(), dimsShuffled.end(), group.first) ==
                                                          dimsShuffled.end()) {
      dimsShuffled.insert(dimsShuffled.begin(), group.first);
    }
  }

  for(unsigned i = 0; i < in1.rank(); i++) {
    if(std::find(dimsShuffled.begin(), dimsShuffled.end(), i) ==
                                                          dimsShuffled.end()) {
      dimsShuffled.insert(dimsShuffled.begin(), i);
    }
  }
  return dimsShuffled;
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

  Tensor out;
  if(inPlace) {
    out = in1;
  }
  else {
    out = graph.clone(outType, in1, debugPrefix + "/Out");
    poputil::mapOutputForElementWiseOp(graph, {in1, in2}, out);
  }

  if(in1.shape() != in2.shape() && in2.numElements() == 1 && nonCopyBroadcast) {
    // Single element broadcast
    return binaryOpIn2Scalar(graph, in1, in2, out, prog, op, inPlace,
                                                                debugPrefix);
  }
  else if(in1.shape() != in2.shape() && nonCopyBroadcast) {
    auto in2DimsMatched = extendDimensionsToMatch(in1, in2);
    // Vector broadcast
    auto bestShape = interpretDimGroupings(graph, in1);
    auto matchingDim = matchingDimension(in1, in2DimsMatched);

    auto method = chooseBinaryOpMethod(graph, prog, op, inPlace, in1, in2, out,
                                              matchingDim, bestShape);

    if(method == BinaryOpMethod::VECTOR_BROADCAST) {

      return binaryOpIn2Vector(graph, in1.dimShuffle(bestShape),
                                  in2DimsMatched.dimShuffle(bestShape),
                                  out.dimShuffle(bestShape),
                                  prog, op, inPlace, debugPrefix, true).output;
    }
    else if(method == BinaryOpMethod::CHANNEL_OP ||
            method == BinaryOpMethod::BROADCAST_AND_CHANNEL_OP ) {

      auto in1Shaped = in1.dimShuffle(bestShape);
      auto outShaped = out.dimShuffle(bestShape);

      if(method == BinaryOpMethod::BROADCAST_AND_CHANNEL_OP ){
        auto matchingDimReshape = matchingDimension(in1Shaped,
                                        in2DimsMatched.dimShuffle(bestShape));

        in2DimsMatched = createPartialBroadcastOperand(in1Shaped,
                                    in2DimsMatched.dimShuffle(bestShape));
        in1Shaped = in1Shaped.dimShufflePartial({matchingDimReshape},
                                                {in1Shaped.rank()-2});
        outShaped = outShaped.dimShufflePartial({matchingDimReshape},
                                                {outShaped.rank()-2});
      }
      if(!inPlace) {
        graph.setTileMapping(out, graph.getTileMapping(in1));
      }
      outShaped = binaryOpUsingChannelOp(graph, in1Shaped,
                                  in2DimsMatched.flatten(), outShaped,
                                  op, prog, inPlace, debugPrefix, true).output;
      return out;
    }
    else {
      assert(method == BinaryOpMethod::COPY_BROADCAST);
    }
  }
  broadcastToMatch(in1, in2);
  return binaryOpSameSize(graph, in1, in2, out, prog, op, inPlace,
                                                      debugPrefix, true).output;
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
    const expr::Expr *&inPlaceExpr,
    bool vectorOptimise) {

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
                 constructGraph, inPlace, inPlaceExpr, vectorOptimise);
    if (constructGraph) {
      return {unaryOp(graph, t.first, prog, opType, t.second, debugPrefix),
              t.second};
    } else {
      return t;
    }
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto opType = b->getOpType();
    auto lhs = map(graph, b->getLHS(), ts, prog, debugPrefix, constTypes, false,
                   constructGraph, inPlace, inPlaceExpr, vectorOptimise);
    auto rhs = map(graph, b->getRHS(), ts, prog, debugPrefix, constTypes, false,
                  constructGraph, false, inPlaceExpr, vectorOptimise);
    if (constructGraph) {
      const bool nonCopyBroadcast = checkForBroadcastOp(opType, lhs, rhs,
                                                              vectorOptimise);
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
                  false, constructGraph, inPlace, inPlaceExpr, vectorOptimise);
      auto rhs = map(graph, t->getArg1(), ts, prog, debugPrefix, constTypes,
                  false, constructGraph, false, inPlaceExpr, vectorOptimise);
      auto pred = map(graph, t->getArg2(), ts, prog, debugPrefix, constTypes,
                  false, constructGraph, false, inPlaceExpr, vectorOptimise);
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
                  false, constructGraph, inPlace, inPlaceExpr, vectorOptimise);
      auto lower = map(graph, t->getArg1(), ts, prog, debugPrefix, constTypes,
                    false, constructGraph, false, inPlaceExpr, vectorOptimise);
      auto upper = map(graph, t->getArg2(), ts, prog, debugPrefix, constTypes,
                    false, constructGraph, false, inPlaceExpr, vectorOptimise);
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
           const OptionFlags &options) {

  bool enableVectorBroadcastOptimisations = true;
  const poplibs::OptionSpec mapSpec{
    { "enableVectorBroadcastOptimisations",
    poplibs::OptionHandler::createWithBool(
        enableVectorBroadcastOptimisations)}
  };
  for (const auto &entry : options) {
    mapSpec.parse(entry.first, entry.second);
  }

  auto constTypes = getConstType(expr, ts);
  const expr::Expr *inplaceExpr = nullptr;
  return map(graph, expr, ts, prog, debugPrefix, constTypes, true, true, false,
             inplaceExpr, enableVectorBroadcastOptimisations).first;
}

void mapInPlace(Graph &graph, const expr::Expr &expr,
                const std::vector<Tensor> &ts,
                program::Sequence &prog,
                const std::string &debugPrefix,
                const OptionFlags &options) {

  bool enableVectorBroadcastOptimisations = true;
  const poplibs::OptionSpec mapSpec{
    { "enableVectorBroadcast",
    poplibs::OptionHandler::createWithBool(
        enableVectorBroadcastOptimisations)}
  };
  for (const auto &entry : options) {
    mapSpec.parse(entry.first, entry.second);
  }

  auto constTypes = getConstType(expr, ts);
  const expr::Expr *inPlaceExpr = nullptr;
  const bool doInPlace = !ts[0].containsAliases() && !ts[0].containsConstant();
  if (doInPlace) {
    // As the tree is traveresed, find the last expression which uses the
    // tensor used for in-place operation as a placeholder
    map(graph, expr, ts, prog, debugPrefix, constTypes, true, false, false,
        inPlaceExpr, enableVectorBroadcastOptimisations);
  }
  auto t = map(graph, expr, ts, prog, debugPrefix, constTypes, true, true,
               doInPlace, inPlaceExpr, enableVectorBroadcastOptimisations);
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
