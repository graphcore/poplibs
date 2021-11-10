// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popops/ElementWise.hpp"
#include "ElementWiseInternal.hpp"
#include "ElementWiseUtilInternal.hpp"
#include "ExprOpUtil.hpp"
#include "ScalarMultiply.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "popops/NaN.hpp"
#include "popops/PerformanceEstimation.hpp"
#include "popops/Rearrange.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VarStructure.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional.hpp>
#include <iostream>
#include <tbb/parallel_for.h>
#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include <cassert>

#include <cstdio>
#include <fstream>
#include <queue>
#include <stack>

#include "ExpressionGenerator.hpp"

using namespace poputil;
using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace popops::expr;

using popops::expr::BinaryOpType;
using popops::expr::TernaryOpType;
using popops::expr::UnaryOpType;

namespace popops {

// namespace popops {

namespace {

enum class ternaryOpTensorsMap {
  TENSOR1_SCALAR2_SCALAR3 = 0,
  SCALAR1_SCALAR2_TENSOR3,
  TENSOR1_TENSOR2_SCALAR3,
  TENSOR1_TENSOR2_TENSOR3
};

enum ternaryOpCodelets {
  CLAMP = 0,
  BROADCAST_CLAMP,
  SELECT,
  BROADCAST_SELECT,
  BROADCAST_SELECTOR_SELECT,
  NR_OF_CODELETS
};

struct MapOptions {
  bool enableVectorBroadcastOptimisations = true;
  bool enableGenerateCodelet = true;

  // By default if there is only a single operation we will not fuse. For tests
  // we will need to force it on.
  bool forceGenerateCodelet = false;

  // optimise expressions where possible
  bool enableExpressionOptimizations = true;
};

MapOptions parseOptionFlags(const OptionFlags &options) {
  MapOptions mapOpts;
  const poplibs::OptionSpec mapSpec{
      {"enableVectorBroadcastOptimisations",
       poplibs::OptionHandler::createWithBool(
           mapOpts.enableVectorBroadcastOptimisations)},
      {"enableGenerateCodelet",
       poplibs::OptionHandler::createWithBool(mapOpts.enableGenerateCodelet)},
      {"forceGenerateCodelet",
       poplibs::OptionHandler::createWithBool(mapOpts.forceGenerateCodelet)},
      {"enableExpressionOptimizations",
       poplibs::OptionHandler::createWithBool(
           mapOpts.enableExpressionOptimizations)}};
  for (const auto &entry : options) {
    mapSpec.parse(entry.first, entry.second);
  }
  return mapOpts;
}

Type outputType(const Type &inType, enum UnaryOpType op) {
  if (op == UnaryOpType::IS_FINITE || op == UnaryOpType::IS_INF ||
      op == UnaryOpType::IS_NAN || op == UnaryOpType::LOGICAL_NOT) {
    return BOOL;
  } else {
    return inType;
  }
}

Type outputType(const Type &inType, BinaryOpType op) {
  if (op == BinaryOpType::EQUAL || op == BinaryOpType::GREATER_THAN_EQUAL ||
      op == BinaryOpType::GREATER_THAN || op == BinaryOpType::LESS_THAN_EQUAL ||
      op == BinaryOpType::LOGICAL_AND || op == BinaryOpType::LOGICAL_OR ||
      op == BinaryOpType::LESS_THAN || op == BinaryOpType::NOT_EQUAL) {
    return BOOL;
  } else {
    return inType;
  }
}

Type outputType(const Type &inType, TernaryOpType /*op*/) { return inType; }

std::string vertexName(TernaryOpType op) {
  switch (op) {
  case TernaryOpType::CLAMP:
    return "Clamp";
  case TernaryOpType::SELECT:
    return "Select";
  }
  throw poputil::poplibs_error("Op not supported");
}

// Describes a pattern of broadcast that we can detect and
// use to produce a more efficient element-wise op where an
// operand is broadcasted.
struct BroadcastPattern {
  std::size_t innerFactor = 1;
  std::vector<Interval> region;
  std::size_t outerFactor = 1;
  std::size_t regionNumElements() const {
    return std::accumulate(
        region.begin(), region.end(), std::size_t(0),
        [](std::size_t total, const Interval &i) { return total + i.size(); });
  }
  std::size_t numElements() const {
    return regionNumElements() * innerFactor * outerFactor;
  }
};

bool isBinaryOpCommutative(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::ADD:
  case BinaryOpType::BITWISE_AND:
  case BinaryOpType::BITWISE_OR:
  case BinaryOpType::BITWISE_XOR:
  case BinaryOpType::BITWISE_XNOR:
  case BinaryOpType::EQUAL:
  case BinaryOpType::LOGICAL_AND:
  case BinaryOpType::LOGICAL_OR:
  case BinaryOpType::MAXIMUM:
  case BinaryOpType::MINIMUM:
  case BinaryOpType::MULTIPLY:
  case BinaryOpType::NOT_EQUAL:
    return true;
  case BinaryOpType::DIVIDE:
  case BinaryOpType::GREATER_THAN_EQUAL:
  case BinaryOpType::GREATER_THAN:
  case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
  case BinaryOpType::LESS_THAN_EQUAL:
  case BinaryOpType::LESS_THAN:
  case BinaryOpType::POWER:
  case BinaryOpType::REMAINDER:
  case BinaryOpType::SHIFT_LEFT:
  case BinaryOpType::SHIFT_RIGHT:
  case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
  case BinaryOpType::SUBTRACT:
  case BinaryOpType::ATAN2:
  // VARIANCE_TO_INV_STD_DEV is strictly speaking commutative, but the two
  // operands are not used in a symmetrical way
  case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
    return false;
  }
  throw poputil::poplibs_error("Op not supported");
}

bool haveScalarBroadcastVertexForOp(BinaryOpType op, bool inPlace,
                                    const Type &dType) {
  switch (op) {
  case BinaryOpType::ADD:
  case BinaryOpType::DIVIDE:
  case BinaryOpType::GREATER_THAN:
  case BinaryOpType::GREATER_THAN_EQUAL:
  case BinaryOpType::LESS_THAN:
  case BinaryOpType::LESS_THAN_EQUAL:
  case BinaryOpType::MAXIMUM:
  case BinaryOpType::MINIMUM:
  case BinaryOpType::REMAINDER:
  case BinaryOpType::MULTIPLY:
  case BinaryOpType::SUBTRACT:
    return (dType == HALF || dType == FLOAT || dType == INT ||
            dType == UNSIGNED_INT || dType == BOOL ||
            dType == UNSIGNED_LONGLONG || dType == LONGLONG);

  case BinaryOpType::EQUAL:
  case BinaryOpType::NOT_EQUAL:
    return (dType == HALF || dType == FLOAT || dType == INT ||
            dType == UNSIGNED_INT || dType == BOOL || dType == SHORT ||
            dType == UNSIGNED_SHORT || dType == UNSIGNED_LONGLONG ||
            dType == LONGLONG);

  case BinaryOpType::LOGICAL_AND:
  case BinaryOpType::LOGICAL_OR:
    return dType == BOOL;

  case BinaryOpType::ATAN2:
  case BinaryOpType::POWER:
  case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
  case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
    return (dType == HALF || dType == FLOAT);

  case BinaryOpType::BITWISE_AND:
  case BinaryOpType::BITWISE_OR:
  case BinaryOpType::BITWISE_XOR:
  case BinaryOpType::BITWISE_XNOR:
    return (dType == INT || dType == UNSIGNED_INT || dType == SHORT ||
            dType == UNSIGNED_SHORT || dType == UNSIGNED_LONGLONG ||
            dType == LONGLONG);

  case BinaryOpType::SHIFT_LEFT:
  case BinaryOpType::SHIFT_RIGHT:
    return (dType == INT || dType == UNSIGNED_INT ||
            dType == UNSIGNED_LONGLONG || dType == LONGLONG);

  case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
    return dType == INT || dType == LONGLONG;
  default:
    return false;
  }
  POPLIB_UNREACHABLE();
}

bool haveInnerVectorBroadcastVertexForOp(BinaryOpType op, bool inPlace,
                                         const Type &dType) {
  if (dType != HALF && dType != FLOAT) {
    return false;
  }
  switch (op) {
  case BinaryOpType::ADD:
  case BinaryOpType::DIVIDE:
  case BinaryOpType::SUBTRACT:
  case BinaryOpType::MULTIPLY:
    return true;
  default:
    return false;
  }
  POPLIB_UNREACHABLE();
}

bool haveOuterVectorBroadcastVertexForOp(BinaryOpType op, bool inPlace,
                                         const Type &dType) {
  if (dType != HALF && dType != FLOAT) {
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

bool unaryUsesNonLinearityVertex(UnaryOpType op) {
  return (op == UnaryOpType::TANH || op == UnaryOpType::RELU ||
          op == UnaryOpType::SIGMOID);
}

unsigned getUnaryOpVectorWidth(UnaryOpType op, const Tensor &in,
                               const Tensor &out) {
  // The dispatch methods in elementWiseCodelets.cpp indicate
  // how many elements are processed per loop.
  if ((in.elementType() == HALF || in.elementType() == FLOAT) &&
      out.elementType() == BOOL) {
    // BOOL outputs always written in groups of 4 to avoid subword writes
    return 4;
  }
  if (in.elementType() == HALF && out.elementType() == HALF) {
    // Non linearity implemented with a smaller vector width
    return unaryUsesNonLinearityVertex(op) ? 2 : 4;
  }
  if (in.elementType() == FLOAT && out.elementType() == FLOAT) {
    // Non linearity implemented with a smaller vector width
    return unaryUsesNonLinearityVertex(op) ? 1 : 2;
  }
  if (in.elementType() == SHORT || in.elementType() == UNSIGNED_SHORT) {
    if (op == UnaryOpType::BITWISE_NOT) {
      // Only this operation is defined for unsigned short and short
      return 4;
    }
    POPLIB_UNREACHABLE();
  }
  // All other cases are not vectorised
  return 1;
}

unsigned getBinaryOpVectorWidth(BinaryOpType op, const Tensor &in,
                                const Tensor &out) {
  // The dispatch methods in elementWiseCodelets.cpp indicate
  // how many elements are processed per loop.
  if ((in.elementType() == HALF || in.elementType() == FLOAT) &&
      out.elementType() == BOOL) {
    return 4;
  }
  if (in.elementType() == HALF && out.elementType() == HALF) {
    return 4;
  }
  if (in.elementType() == FLOAT && out.elementType() == FLOAT) {
    return 2;
  }
  if (in.elementType() == SHORT || in.elementType() == UNSIGNED_SHORT) {
    if (op == BinaryOpType::BITWISE_OR || op == BinaryOpType::BITWISE_AND) {
      return 4;
    }
    return 2;
  }
  return 1;
}

unsigned maxVertexElementsPerRegion(const Target &target, const Type &outType,
                                    const ternaryOpCodelets op,
                                    const bool inPlace = false) {

  auto typeToIndexCovertor = [](const Type &eType) {
    if (eType == BOOL) {
      return 0;
    } else if (eType == INT) {
      return 1;
    } else if (eType == HALF) {
      return 2;
    } else if (eType == FLOAT) {
      return 3;
    } else if (eType == UNSIGNED_INT) {
      return 4;
    } else {
      throw poplibs_error("Requested type to index conversion doesn't exist");
    }
  };

  /* Assembler codelet implementations indicate how many elements are processed
   * per HW loop. If HW loop isn't in use or a codelet has only C implementation
   * them UINT_MAX shall be returned */
  constexpr unsigned convMap[2][NR_OF_CODELETS][5] = {
      {
          // None inPlace
          {0, UINT_MAX, 2, 1, UINT_MAX},                      // Clamp
          {0, UINT_MAX, 2, 1, UINT_MAX},                      // BroadcastClamp
          {UINT_MAX, 1, UINT_MAX, 1, UINT_MAX},               // Select
          {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX}, // BroadcastSelect
          {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX,
           UINT_MAX} // BroadcastSelectorSelect
      },
      {
          // inPlace
          {0, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX},        // Clamp
          {0, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX},        // BroadcastClamp
          {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX}, // Select
          {0, 0, 0, 0, 0},                                    // BroadcastSelect
          {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX,
           UINT_MAX} // BroadcastSelectorSelect
      }};

  unsigned inPlaceIdx = static_cast<unsigned>(inPlace);
  unsigned nrElements = convMap[inPlaceIdx][op][typeToIndexCovertor(outType)];

  if (nrElements == 0) {
    std::stringstream ss;
    ss << "Failed to extract number of elements for " << std::to_string(op)
       << " of type " << outType;
    throw poplibs_error(ss.str());
  }
  if (nrElements == UINT_MAX) {
    return nrElements;
  }

  return target.getRptCountMax() * nrElements;
}

struct MaxElements {
  // Constraints in 1D-MultiVertex/2D-worker vertices due to the maximum loops
  unsigned repeat;
  // Constraints in 1D MultiVertex related to work division
  unsigned division;
  unsigned regionSize;
};

constexpr unsigned maxDivisibleValue = (UINT_MAX / 0xAAAB) - 5;
constexpr unsigned maxDivisibleValueNl = (UINT_MAX / 0xAAAB);

MaxElements maxVertexElementsPerRegion(const Graph &graph, const Target &target,
                                       UnaryOpType op, const Tensor &in,
                                       const Tensor &out,
                                       const std::string &codeletName2D,
                                       const std::string &codeletName1D,
                                       bool inPlace) {
  MaxElements result;
  const auto vectorWidth = getUnaryOpVectorWidth(op, in, out);
  // Assembler codelet implementations indicate how many elements are processed
  // per HW loop. If HW loop isn't in use or a codelet has only C implementation
  // then UINT_MAX shall be returned

  // Filter out codelets that don't depend on RPT instruction
  if (in.elementType() == SHORT || in.elementType() == UNSIGNED_SHORT) {
    // There are very few unary ops allowed using shot/unsigned short
    result.repeat = target.getRptCountMax() * vectorWidth;
    result.division = vectorWidth * maxDivisibleValue;
    result.regionSize = result.repeat;
  } else if (in.elementType() != HALF && out.elementType() != FLOAT) {
    // Integer functions with no vectorisation and no division required
    result.repeat = UINT_MAX;
    result.division = UINT_MAX;
    result.regionSize = result.repeat;
  } else if (inPlace && unaryUsesNonLinearityVertex(op)) {
    // Non linearities implemented using vertices shared with the non linearity
    // function.  Limited by the field sizes allowed
    unsigned nFieldSize = graph.getMaxVertexFieldValue(codeletName1D, "n");
    result.repeat = std::min(nFieldSize, target.getRptCountMax() * vectorWidth);
    // Note - work division function has less headroom, so it is correct that
    // there is no muliply by vectorWidth
    result.division = std::min(nFieldSize, maxDivisibleValueNl);
    result.regionSize = std::min(
        static_cast<unsigned>(graph.getMaxFieldDim(codeletName2D, "inOut", 1)),
        target.getRptCountMax());
  } else {
    // Normal float, half functions
    result.repeat = target.getRptCountMax() * vectorWidth;
    result.division = vectorWidth * maxDivisibleValue;
    result.regionSize = result.repeat;
  }
  return result;
}

MaxElements maxVertexElementsPerRegion(const Target &target, BinaryOpType op,
                                       const Tensor &in, const Tensor &out) {
  MaxElements result;
  // Assembler codelet implementations indicate how many elements are processed
  // per HW loop. If HW loop isn't in use or a codelet has only C implementation
  // then UINT_MAX shall be returned
  const auto vectorWidth = getBinaryOpVectorWidth(op, in, out);

  // Filter out codelets that don't depend on RPT instruction
  if (in.elementType() == SHORT || in.elementType() == UNSIGNED_SHORT) {
    result.repeat = target.getRptCountMax() * vectorWidth;
    result.division = vectorWidth * maxDivisibleValue;
    result.regionSize = result.repeat;
  } else if (in.elementType() != HALF && out.elementType() != FLOAT) {
    result.repeat = UINT_MAX;
    result.division = UINT_MAX;
    result.regionSize = result.repeat;
  } else {
    result.repeat = target.getRptCountMax() * vectorWidth;
    result.division = vectorWidth * maxDivisibleValue;
    result.regionSize = result.repeat;
  }

  return result;
}

// Check if we can use a MultiVertex, or if the regions to process
// prevent it.  This can be due to either having multiple regions or if the
// region is too large.
bool validateRegionSizeForMultiVertex(
    const std::vector<std::vector<Interval>> &intervals,
    MaxElements maxRegionSize, const unsigned numWorkers) {
  const auto numElems = intervalSequenceNumElements(intervals);
  if (numElems > maxRegionSize.division) {
    return false;
  }
  if (maxRegionSize.repeat == UINT_MAX) {
    return true;
  }
  if (numElems > maxRegionSize.repeat * numWorkers) {
    return false;
  }
  return true;
}

Tensor unaryOp(Graph &graph, Tensor in, Sequence &prog, UnaryOpType op,
               bool inPlace, const DebugNameAndId &dnai) {
  const std::string layer = "Op/" + debugName(op);
  const auto inType = in.elementType();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet({dnai, layer});
  const auto numWorkers = target.getNumWorkerContexts();

  logging::popops::debug("  UnaryOp{} DebugStr: {}", inPlace ? "InPlace" : "",
                         dnai.getPathName() + "/" + layer);
  const auto outType = outputType(inType, op);
  Tensor out;
  if (inPlace) {
    out = in;
  } else {
    out = createOutputForElementWiseOp(graph, {in}, outType,
                                       {dnai, layer + "/Out"});
  }

  if (inPlace) {
    logging::popops::debug("  inOut{}({}): {}", in.shapeToString(),
                           in.elementType(), in.getDebugStr());
  } else {
    logging::popops::debug("  in{}({}):  {}", in.shapeToString(),
                           in.elementType(), in.getDebugStr());
    logging::popops::debug("  out{}({}): {}", out.shapeToString(),
                           out.elementType(), out.getDebugStr());
  }
  logging::popops::debug("  {}{} = {} {}{}", out.getVarStr(),
                         out.shapeToString(), debugName(op), in.getVarStr(),
                         in.shapeToString());

  auto inFlat = in.flatten();
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {&inFlat}, false);
  const auto mapping = graph.getTileMapping(outFlat);
  const auto grainSize = std::max<unsigned>(target.getVectorWidth(inType),
                                            target.getAtomicStoreGranularity());

  const auto vertexTemplateMultiVertex = templateVertex(
      inPlace ? "popops::UnaryOp1DInPlace" : "popops::UnaryOp1D", op, inType);
  const auto vertexTemplate2D = templateVertex(
      inPlace ? "popops::UnaryOp2DInPlace" : "popops::UnaryOp2D", op, inType);
  const auto elementLimit =
      maxVertexElementsPerRegion(graph, target, op, in, out, vertexTemplate2D,
                                 vertexTemplateMultiVertex, inPlace);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    if (tileContiguousRegions.size() == 1 &&
        validateRegionSizeForMultiVertex(tileContiguousRegions, elementLimit,
                                         numWorkers)) {
      // If mapping of the output tensor on this tile is only region or regions
      // from one variable, force a gather (in case of more than one region)
      // to get all data to a single edge.
      // The decision to make a MultiVertex ("1D") may also have to account
      // for the total elements as the overhead and work balance may not be
      // very good for small vector sizes.
      // TODO: T12936 Use profiled results for selection.
      logging::popops::trace("  Tile: {} Producing: 1 {} vertex", tile,
                             vertexTemplateMultiVertex);
      auto inData = concat(inFlat.slices(tileContiguousRegions));
      auto v =
          inPlace
              ? graph.addVertex(cs, vertexTemplateMultiVertex,
                                {{"inOut", inData}})
              : graph.addVertex(
                    cs, vertexTemplateMultiVertex,
                    {{"in", inData},
                     {"out", concat(outFlat.slices(tileContiguousRegions))}});
      // Vertices for these ops have an extra field
      if (inPlace && unaryUsesNonLinearityVertex(op)) {
        graph.setInitialValue(v["n"], inData.numElements());
      }
      graph.setTileMapping(v, tile);
    } else {
      const auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, grainSize, 2 * grainSize, UINT_MAX,
          elementLimit.regionSize);
      if (vertexRegions.size()) {
        logging::popops::trace("  Tile: {} Producing: {} {} vertices", tile,
                               vertexRegions.size(), vertexTemplate2D);
      }
      for (const auto &regions : vertexRegions) {
        VertexRef v = inPlace
                          ? graph.addVertex(cs, vertexTemplate2D,
                                            {{"inOut", inFlat.slices(regions)}})
                          : graph.addVertex(cs, vertexTemplate2D,
                                            {{"in", inFlat.slices(regions)},
                                             {"out", outFlat.slices(regions)}});
        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(cs, {dnai}));
  return out;
}

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
void binaryOpGeneral(Graph &graph, const Tensor &in1, const Tensor &in2,
                     const Tensor &out,
                     const std::vector<std::vector<Interval>> &intervals,
                     unsigned tile, const ComputeSet &cs, BinaryOpType op,
                     bool inPlace) {
  const auto dType = in1.elementType();
  const auto &target = graph.getTarget();
  const auto grainSize = std::max<unsigned>(target.getVectorWidth(dType),
                                            target.getAtomicStoreGranularity());

  const auto elementLimit = maxVertexElementsPerRegion(target, op, in1, out);
  // Single contiguous region, use MultiVertex ("1D").
  if (intervals.size() == 1 &&
      validateRegionSizeForMultiVertex(intervals, elementLimit,
                                       target.getNumWorkerContexts())) {
    const auto vertexClass = templateVertex(
        inPlace ? "popops::BinaryOp1DInPlace" : "popops::BinaryOp1D", op,
        dType);
    logging::popops::trace("  Tile: {} Producing: 1 {} vertex", tile,
                           vertexClass);
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
    const auto vertexClass = templateVertex(
        inPlace ? "popops::BinaryOp2DInPlace" : "popops::BinaryOp2D", op,
        dType);
    const auto vertexRegions =
        splitRegionsBetweenWorkers(target, intervals, grainSize, 2 * grainSize,
                                   UINT_MAX, elementLimit.regionSize);
    if (vertexRegions.size()) {
      logging::popops::trace("  Tile: {} Producing: {} {} vertices", tile,
                             vertexRegions.size(), vertexClass);
    }
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

void binaryOpGeneral(Graph &graph, Tensor in1, Tensor in2, const Tensor &out,
                     Sequence &prog, BinaryOpType op, bool inPlace,
                     const DebugNameAndId &dnai) {
  // First try to regroup to match the grouping of the output tensor.
  {
    std::vector<Copy> copies;
    const DebugNameAndId transposeDnai{dnai, "TransposeInputs"};
    const auto cs = graph.addComputeSet(transposeDnai);
    if (!inPlace) {
      in1 = rearrange::regroupIfBeneficial(graph, in1, out, copies, cs,
                                           transposeDnai);
    }
    in2 = rearrange::regroupIfBeneficial(graph, in2, out, copies, cs,
                                         transposeDnai);

    for (auto &copy : copies) {
      prog.add(std::move(copy));
    }
    prog.add(Execute(cs));
  }

  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto outFlat = out.flatten();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet({dnai});
  graph.reorderToSimplify(&outFlat, {&in1Flat, &in2Flat}, false);
  const auto mapping = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    binaryOpGeneral(graph, in1Flat, in2Flat, outFlat, tileContiguousRegions,
                    tile, cs, op, inPlace);
  }
  prog.add(Execute(cs, {dnai}));
}

bool binaryOpBroadcastScalar(
    Graph &graph, const Tensor &in1, const Tensor &in2, const Tensor &out,
    const std::vector<std::vector<Interval>> &intervals, unsigned tile,
    const ComputeSet &cs, BinaryOpType op, bool inPlace, bool uniformScalar,
    bool exitIfInefficient = false) {
  return popops::createVertexBinaryOpBroadcastScalar(
      graph, in1, in2, out, intervals, tile, cs, op, inPlace, uniformScalar,
      exitIfInefficient);
}

void binaryOpBroadcastScalar(Graph &graph, const Tensor &in1, const Tensor &in2,
                             const Tensor &out, Sequence &prog, BinaryOpType op,
                             bool inPlace, const DebugNameAndId &dnai) {
  // Tensors in1, in2 and out will be the same broadcast shape.
  assert(in1.shape() == in2.shape() && in2.shape() == out.shape());

  auto in1Flat = in1.flatten();
  auto outFlat = out.flatten();
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto cs = graph.addComputeSet({dnai});
  graph.reorderToSimplify(&outFlat, {&in1Flat}, false);
  const auto mapping = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    binaryOpBroadcastScalar(graph, in1Flat, in2, outFlat, tileContiguousRegions,
                            tile, cs, op, inPlace, true /* uniformScalar */);
  }
  prog.add(Execute(cs, {dnai}));
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
    Graph &graph, const Tensor &in1, const Tensor &in2, const Tensor &out,
    const std::vector<std::vector<Interval>> &intervals,
    const std::vector<BroadcastPattern> &patterns, unsigned tile,
    const ComputeSet &cs, BinaryOpType op, bool inPlace, Sequence &prog,
    const DebugNameAndId &dnai) {
  const auto dType = in1.elementType();
  const auto &target = graph.getTarget();

  // The VectorInner implementation for the float vertices for ADD, SUB, MUL
  // is sometimes slightly faster, sometimes slower (depending on the size of
  // both operands) than broadcasting the second operand (with copies) and then
  // performing a BinaryOp, because the vertices for BinaryOp are optimised
  // (vectorised). So we don't select the VectorInner vertices for those
  // operations but only for DIVIDE which is advantageous for almost all the
  // input sizes.
  if (dType == FLOAT && op != BinaryOpType::DIVIDE) {
    return false;
  }

  // We want all patterns to be a multiple of pattern elements otherwise we
  // cannot divide work using a simplistic algorithm. We can lift this
  // restriction once we use cost based estimates to balance work through
  // creating 1D MultiVertex and/or 2D single worker vertices.
  const auto numPatternElems = patterns.front().regionNumElements();
  assert(dType == FLOAT || dType == HALF);
  const auto elemsPer64Bits = dType == HALF ? 4 : 2;
  // The implementation for half vertices is slow for the non-vectorised add,
  // mul cases. Avoid using this case until a better implementation is written.
  // The slow path is picked when the vector that is repeated has a length
  // which is not a multiple of the vectorWidth
  if (op != BinaryOpType::DIVIDE && dType == HALF &&
      numPatternElems % elemsPer64Bits != 0) {
    return false;
  }

  auto innerVectorBroadcastablePredicate = [](const BroadcastPattern &p) {
    return p.innerFactor == 1 && p.outerFactor > 1;
  };

  // Split tile contiguous regions based on broadcast patterns to form a new
  // set of regions which match patterns
  std::vector<std::vector<Interval>> splitIntervals(patterns.size());
  auto intervalIt = intervals.begin();

  // iterator for each interval with a contiguous set of intervals
  auto cIntervalIt = intervalIt->begin();
  std::size_t offset = cIntervalIt->begin();

  for (auto [pIt, splitIntervalsIt] =
           std::tuple{patterns.begin(), splitIntervals.begin()};
       pIt != patterns.end() && intervalIt != intervals.end();
       ++pIt, ++splitIntervalsIt) {
    // Every pattern is constrained to be inner broadcast.
    // Restriction on size of each pattern as we do not want to split any
    // pattern when dividing work across workers.
    // In future we could lift this restriction but this is the most common
    // case and we address it for now.
    //
    if (!innerVectorBroadcastablePredicate(*pIt) ||
        pIt->regionNumElements() != numPatternElems) {
      return false;
    }

    // consume a region one at a time and possibly over multiple patterns.
    auto numRemaining = pIt->numElements();
    while (numRemaining != 0) {
      // Take the minimum of the remaining part of the current region and the
      // size of the pattern.
      const auto size = std::min(
          cIntervalIt->size() + offset - cIntervalIt->begin(), numRemaining);
      splitIntervalsIt->emplace_back(offset, offset + size);
      // reduce pattern by elements consumed
      numRemaining -= size;
      offset += size;
      if (offset == cIntervalIt->end()) {
        if (++cIntervalIt == intervalIt->end()) {
          if (++intervalIt == intervals.end()) {
            break;
          }
          cIntervalIt = intervalIt->begin();
        }
        offset = cIntervalIt->begin();
      }
    }
  }
  const auto numWorkers = target.getNumWorkerContexts();
  auto checkUseMultiVertex = [&](std::size_t size, std::size_t subSize) {
    const unsigned dataBlockCountPackedMaxFieldSize = 0x1fff;
    const unsigned rptCountMaxFieldSize =
        (target.getRptCountMax() & ~1UL) * numWorkers;
    return (size % subSize) == 0 &&
           (size / subSize) <=
               std::min(dataBlockCountPackedMaxFieldSize, rptCountMaxFieldSize);
  };
  // The division of work among 6 workers is done when creating the vertex
  // (contrary to other types of vertices that do that in the device code).
  //
  // The amount of work to do is expressed by:
  //        dataBlockCount = data.size() / B.size();
  // i.e. how many times the 'B' vector fits inside 'data'
  // This is divided by 6; the quotient and remainder of this division is
  // packed into 'dataBlockCountPacked'
  //
  //                         15 14 13 12 11 10            4  3  2  1  0
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  // dataBlockCountPacked:  |           13 bits               | 3 bits |
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  //
  //                        |                                 |        |
  //                        +---------------+-----------------+----+---+
  //                                        |                      |
  //                            floor(dataBlockCount/6)    dataBlockCount % 6
  //
  auto packCount = [=](std::size_t size, std::size_t subSize) -> std::uint16_t {
    auto blockCount = size / subSize;
    return ((blockCount / numWorkers) << 3) | blockCount % numWorkers;
  };

  const auto elementLimit = maxVertexElementsPerRegion(target, op, in1, out);

  // Use a 1D MultiVertex to reduce memory use if there are a small number
  // of suitable intervals
  if (splitIntervals.size() <= 2) {
    auto intervalVectorLength = [](const std::vector<Interval> &iVector) {
      std::size_t len = 0;
      for (const auto &i : iVector)
        len += i.size();
      return len;
    };
    bool canUseMultiVertex = true;
    for (const auto &splitInterval : splitIntervals) {
      if (!checkUseMultiVertex(intervalVectorLength(splitInterval),
                               numPatternElems) ||
          !validateRegionSizeForMultiVertex({splitInterval}, elementLimit,
                                            target.getNumWorkerContexts())) {
        canUseMultiVertex = false;
        break;
      }
    }
    if (canUseMultiVertex) {
      for (const auto &splitInterval : splitIntervals) {
        const auto &outRegion = concat(out.flatten().slices(splitInterval));
        const auto &in1Region = concat(in1.flatten().slices(splitInterval));
        const auto &in2Region = concat(in2.flatten().slices(splitInterval))
                                    .slice(0, numPatternElems);
        std::string vertexName = inPlace
                                     ? "popops::BroadcastVectorInner1DInPlace"
                                     : "popops::BroadcastVectorInner1D";
        auto vertexClass = templateVertex(vertexName, op, dType);
        logging::popops::trace("  Tile: {} Producing: 1 {} vertex", tile,
                               vertexClass);
        std::uint16_t dataBlockCountPacked =
            packCount(outRegion.numElements(), in2Region.numElements());
        auto v = graph.addVertex(cs, vertexClass);
        graph.connect(v["B"], in2Region);
        graph.connect(v["data"], in1Region);
        if (!inPlace) {
          graph.connect(v["out"], outRegion);
        }
        graph.setInitialValue(v["dataBlockCountPacked"], dataBlockCountPacked);
        graph.setTileMapping(v, tile);
      }
      return true;
    }
  }

  // Cannot use MultiVertex, split work based on the size of the pattern.
  std::string vertexName = inPlace ? "popops::BroadcastVectorInner2DInPlace"
                                   : "popops::BroadcastVectorInner2D";
  auto vertexClass = templateVertex(vertexName, op, dType);
  const auto maxWorkListElemLen =
      graph.getMaxVertexFieldValue(vertexClass, "workList");
  // If numPatternElems were some ludicrous number that doesn't
  // actually fit in numPatternElems then we could handle it and still
  // use channel ops but for now it seems unlikely.
  if (numPatternElems <= maxWorkListElemLen &&
      (intervalSequenceNumElements(splitIntervals) % numPatternElems) == 0) {
    const auto maxBlockCount =
        std::min<unsigned>(maxWorkListElemLen, target.getRptCountMax());
    const auto maxSize =
        std::min(static_cast<unsigned>(maxBlockCount * numPatternElems),
                 elementLimit.regionSize);
    const auto vertexRegions =
        splitRegionsBetweenWorkers(target, splitIntervals, numPatternElems,
                                   numPatternElems, UINT_MAX, maxSize);
    if (vertexRegions.size()) {
      logging::popops::trace("  Tile: {} Producing: {} {} vertices", tile,
                             vertexRegions.size(), vertexClass);
    }
    for (const auto &regions : vertexRegions) {
      auto outRegions = out.flatten().slices(regions);
      auto in1Regions = in1.flatten().slices(regions);
      auto in2Regions = in2.flatten().slices(regions);

      for (auto &region : in2Regions) {
        region = region.slice(0, numPatternElems);
      }

      const auto numEdges = outRegions.size();
      std::vector<std::uint16_t> workList;
      workList.reserve(2 * numEdges + 1);
      if (numEdges > (1 + maxWorkListElemLen)) {
        throw poplibs_error("2D vector inner has > " +
                            std::to_string(maxWorkListElemLen) + " edges");
      }
      workList.push_back(numEdges - 1);
      for (std::size_t i = 0; i < numEdges; ++i) {
        assert((outRegions[i].numElements() % numPatternElems) == 0);
        workList.push_back(numPatternElems);
        workList.push_back(outRegions[i].numElements() / numPatternElems);
      }

      auto workListTensor =
          graph.addConstant(UNSIGNED_SHORT, {workList.size()}, workList.data(),
                            {dnai, "workList"});
      graph.setTileMapping(workListTensor, tile);

      auto v = graph.addVertex(cs, vertexClass);
      graph.connect(v["B"], in2Regions);
      graph.connect(v["data"], in1Regions);
      if (!inPlace) {
        graph.connect(v["out"], outRegions);
      }
      graph.connect(v["workList"], workListTensor);
      graph.setTileMapping(v, tile);
    }
    return true;
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
 *  \param regions          Contiguous regions for the output operand on this
 *                          tile.
 *  \param patterns         Broadcast patterns on that tile. Only support the
 *                          case where a single pattern exist for each region.
 *                          Meaning that the pattern is the same for all the
 *                          intervals of a single region.
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
    Graph &graph, const Tensor &in1, const Tensor &in2, const Tensor &out,
    const std::vector<std::vector<Interval>> &regions,
    const std::vector<BroadcastPattern> &patterns, unsigned tile,
    const ComputeSet &cs, BinaryOpType op, bool inPlace) {

  const auto dType = in1.elementType();
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(dType);
  const auto numWorkers = target.getNumWorkerContexts();

  // TODO: T12937 Consider tracking which parts of the given pattern are
  // contiguous. If they are not contiguous it may be a space-saving to use a 2D
  // scalar broadcast vertex rather than gathering the elements of the pattern.

  auto canUseOuterVectorVertex = [&](const BroadcastPattern &pattern) {
    if ((pattern.numElements() % pattern.innerFactor) != 0) {
      return false;
    }
    return true;
  };
  const auto elementLimit = maxVertexElementsPerRegion(target, op, in1, out);

  if ((std::all_of(patterns.begin(), patterns.end(),
                   canUseOuterVectorVertex)) &&
      patterns.size() != 0 &&
      validateRegionSizeForMultiVertex(regions, elementLimit, numWorkers)) {
    std::vector<Tensor> outRegion(patterns.size()), in1Region(patterns.size()),
        in2Region(patterns.size());
    unsigned regionIndex = 0;
    unsigned intervalIndex = 0;
    unsigned intervalOffset = 0;
    for (unsigned int i = 0; i < patterns.size(); i++) {
      std::vector<Interval> section =
          cutRegionSection(regions[regionIndex], patterns[i].numElements(),
                           intervalIndex, intervalOffset, regionIndex);
      outRegion[i] = concat(out.flatten().slices(section));
      in1Region[i] = concat(in1.flatten().slices(section));
      in2Region[i] = concat(in2.flatten().slices(section))
                         .slice(0, patterns[i].regionNumElements() *
                                       patterns[i].innerFactor)
                         .subSample(patterns[i].innerFactor, 0);
    }

    // Select for 4 possible cases based on 2 decisions:
    // 1. Is alignment is assured on resuming each row
    // 2. Are rows are short so using 1 worker per row is more efficient
    // If those 2 conditions are not met for all patterns, the function
    // returns false
    bool rowVertex = true;
    bool columnVertex = true;

    for (const auto &p : patterns) {
      if (!(p.innerFactor < numWorkers * vectorWidth)) {
        rowVertex = false;
        ;
      }
      if (p.innerFactor < numWorkers * vectorWidth) {
        columnVertex = false;
      }
    }

    std::string vertexName;
    if (rowVertex) {
      vertexName = inPlace ? "popops::BroadcastVectorOuterByRow1DInPlace"
                           : "popops::BroadcastVectorOuterByRow1D";
    } else if (columnVertex) {
      vertexName = inPlace ? "popops::BroadcastVectorOuterByColumn1DInPlace"
                           : "popops::BroadcastVectorOuterByColumn1D";
    } else { // Mix of row and column vertices
      return false;
    }

    std::vector<std::string> vertexClass(patterns.size());
    for (unsigned int i = 0; i < patterns.size(); i++) {
      vertexClass[i] =
          templateVertex(vertexName, op, dType,
                         patterns[i].innerFactor % vectorWidth ? true : false);
      auto maxC = graph.getMaxVertexFieldValue(vertexClass[i], "columns");
      auto maxR = graph.getMaxVertexFieldValue(vertexClass[i], "rows");
      auto rows = patterns[i].numElements() / patterns[i].innerFactor;

      if ((rows > maxR) || (patterns[i].innerFactor > maxC)) {
        return false;
      }
    }

    for (unsigned int i = 0; i < patterns.size(); i++) {
      logging::popops::trace("  Tile: {} Producing: 1 {} vertex", tile,
                             vertexClass[i]);
      auto v = graph.addVertex(cs, vertexClass[i],
                               {{"data", in1Region[i]}, {"B", in2Region[i]}});
      graph.setInitialValue(v["columns"], patterns[i].innerFactor);
      graph.setInitialValue(v["rows"], patterns[i].numElements() /
                                           patterns[i].innerFactor);
      if (!inPlace) {
        graph.connect(v["out"], outRegion[i]);
      }
      graph.setTileMapping(v, tile);
    }
    return true;
  }
  return false;
}

__attribute__((unused)) inline std::ostream &
operator<<(std::ostream &o, const BroadcastPattern &p) {
  o << "{innerFactor=" << p.innerFactor << ", outerFactor=" << p.outerFactor
    << ", region={";
  auto it = p.region.begin();
  o << "[" << it->begin() << "," << it->end() << ")";
  for (it = std::next(it); it != p.region.end(); ++it) {
    o << ",[" << it->begin() << "," << it->end() << ")";
  }
  o << "}}";
  return o;
}

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
    if (!encoded.empty() && i.begin() == encoded.back().second) {
      ++encoded.back().first;
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
      std::unordered_set<std::size_t> seen{it->second};
      out.emplace_back();
      auto &pattern = out.back();
      pattern.innerFactor = it->first;
      pattern.region.emplace_back(it->second, it->second + 1);
      std::size_t index = 0;
      std::size_t offset = 0;
      bool haveCompleteRegion = false;
      ++it;
      // Iterator storing a restart point if we discover the
      // current pattern does not match the previous. This
      // must always move on at points when the pattern
      // changes.
      auto restartIt = it;
      for (; it != encoded.end(); ++it) {
        if (!haveCompleteRegion || index + offset == 0) {
          restartIt = it;
        }

        // If the innerFactor changes then we need to start
        // a new pattern.
        if (it->first != pattern.innerFactor) {
          break;
        }

        if (!haveCompleteRegion && seen.count(it->second)) {
          haveCompleteRegion = true;
          index = 0;
          offset = 0;
        }

        if (haveCompleteRegion) {
          // If the region does not match the current pattern's region then
          // we need to start a new one.
          if (pattern.region[index].begin() + offset != it->second) {
            break;
          }
        } else {
          // Extend the last interval of the region if possible.
          if (it->second == pattern.region.back().end()) {
            pattern.region.back() = Interval(pattern.region.back().begin(),
                                             pattern.region.back().end() + 1);
          } else {
            // Otherwise add a new interval.
            pattern.region.emplace_back(it->second, it->second + 1);
          }
        }

        if (haveCompleteRegion) {
          ++offset;
          if (offset >= pattern.region[index].size()) {
            ++index;
            offset = 0;
            if (index >= pattern.region.size()) {
              index = 0;
              ++pattern.outerFactor;
            }
          }
        }
      }
      if (!haveCompleteRegion || index + offset == 0) {
        restartIt = it;
      }
      it = restartIt;
    }
  }
};

// Given a set of patterns, split them into simple patterns which describe
// broadcasts of only a single element. The simple patterns created will have an
// outer factor of one, and each pattern will reference only a single element.
//
std::vector<BroadcastPattern>
splitIntoScalarBroadcastPatterns(std::vector<BroadcastPattern> patterns) {
  // Set a reasonable upper bound on the number of single element patterns
  // that we will gather before giving up.  This avoids gathering a
  // ridiculous number of patterns which won't get processed later on anyhow,
  // but can slow things down.
  static constexpr unsigned maxPatternThreshold = 256;

  std::vector<BroadcastPattern> singlePatterns;
  for (const auto &p : patterns) {
    BroadcastPattern singlePattern;
    singlePattern.innerFactor = p.innerFactor;
    singlePattern.outerFactor = 1;
    singlePattern.region.resize(1);
    if (singlePatterns.size() + p.regionNumElements() * p.outerFactor >
        maxPatternThreshold) {
      singlePatterns.clear();
      return singlePatterns;
    }
    for (unsigned k = 0; k < p.outerFactor; k++) {
      for (unsigned i = 0; i < p.region.size(); i++) {
        for (unsigned j = p.region[i].begin(); j < p.region[i].end(); j++) {
          singlePattern.region[0] = {j, j + 1};
          singlePatterns.push_back(singlePattern);
        }
      }
    }
  }
  return singlePatterns;
}

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
splitContiguousRegionsByPattern(std::vector<std::vector<Interval>> intervals,
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
      newRegions.reserve(std::distance(beginIt, endIt) + (offset > 0 ? 1 : 0));
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

void validatePatterns(
    std::size_t totalElems,
    const std::vector<std::vector<BroadcastPattern>> &patternsByTile,
    const bool reversed = false) {
  std::size_t totalPatternElems = 0;
  for (const auto &tilePatterns : patternsByTile) {
    for (const auto &pattern : tilePatterns) {
      totalPatternElems += pattern.numElements();
    }
  }

  if (totalElems != totalPatternElems) {
    std::stringstream ss;
    ss << "Failed to validate the broadcast pattern, total elements ("
       << totalElems << ") doesn't match the total "
       << (reversed ? "reversed " : "") << "pattern elements ("
       << totalPatternElems << ")";
    throw poplibs_error(ss.str());
  }
}

// Construct a binary op where one operand is broadcasted
// before the binary op is applied to it and the other operand to
// produce the output. We can perform more optimal operations in
// these cases.
// The second operand is always checked for broadcasting into the first,
// while doing the reverse (first operand broadcasted into the second) has
// some restrictions.
void constructBroadcastBinaryOp(Graph &graph, Sequence &prog, Tensor in1,
                                bool in1HasAliases, Tensor in2,
                                bool in2HasAliases, Tensor out, BinaryOpType op,
                                bool inPlace, const DebugNameAndId &dnai) {
  // Tensors in1, in2 and out will be the same broadcast shape.
  assert(in1.shape() == in2.shape() && in2.shape() == out.shape());
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto dType = in1.elementType();

  // The normal case is try to broadcast second operand into the first.
  // Only for non-inplace, commutative operators we will check if we can
  // broadcast the first operand into the second.
  bool checkReverse = !inPlace && isBinaryOpCommutative(op);

  auto regroupOperandIfBeneficial = [&](Tensor &operand) {
    const double threshold = 0.1;
    const auto outGrouping = poputil::detectDimGroupings(graph, out);
    if (outGrouping.empty()) {
      return;
    }
    const auto outInnerDim = outGrouping[0].first;
    logging::popops::debug("regroupOperandIfBeneficial: outGrouping=({},{})",
                           outGrouping[0].first, outGrouping[0].second);
    const auto unbroadcastOperand = graph.findUnbroadcastTensor(operand);
    assert(out.rank() == unbroadcastOperand.rank());
    if (unbroadcastOperand.dim(outInnerDim) == 1 ||
        unbroadcastOperand.numElements() > threshold * out.numElements()) {
      return;
    }

    // See if there is a grouping in the operand.
    unsigned operandGrouping = detectInnermostGrouping(
        graph, unbroadcastOperand.dimRoll(outInnerDim, out.rank() - 1));
    // If the grouping in the operand is incompatible with that of the output,
    // we copy to a better laid out tensor of the same (unbroadcasted) shape
    // for the upcoming broadcast operation.
    const auto vectorWidth = target.getVectorWidth(dType);
    if (gcd(outGrouping[0].second, operandGrouping) % vectorWidth != 0) {
      // Factor unbroadcasted dimensions out of output
      const auto unbroadcastShape = unbroadcastOperand.shape();
      auto outFactored = factorDims(out, unbroadcastShape);
      // createBroadcastOperand works with a flattened view of the
      // dimensions of the input tensor which are to be broadcast, with
      // all other dimensions being 1. Work out the shape we want to recover
      // the shape of the broadcasted dimensions.
      auto newOperandShape = std::vector<std::size_t>(out.rank(), 1);
      newOperandShape.insert(newOperandShape.end(), unbroadcastShape.begin(),
                             unbroadcastShape.end());
      auto newUnbroadcastOperandFactored =
          poputil::createBroadcastOperand(
              graph, outFactored.flatten(out.rank(), out.rank() * 2), dType,
              out.rank(), false, {dnai})
              .reshape(newOperandShape);
      assert(newUnbroadcastOperandFactored.rank() == outFactored.rank());

      // Reshape to copy between unbroadcast tensor and newly created tensor.
      const auto newUnbroadcastOperand =
          unfactorDims(newUnbroadcastOperandFactored, out.rank());
      prog.add(Copy(unbroadcastOperand, newUnbroadcastOperand, false, {dnai}));

      // Use factored views of output and the newly created tensor to
      // broadcastToMatch as this requires that the size of each dimension of
      // the given tensors must either be equal or 1.
      broadcastToMatch(outFactored, newUnbroadcastOperandFactored);
      // Now unfactor new broadcasted to give the correct view matching the
      // original operand.
      const auto newOperand =
          unfactorDims(newUnbroadcastOperandFactored, out.rank());
      assert(newOperand.shape() == operand.shape());
      operand = std::move(newOperand);
    }
  };

  if (in2HasAliases) {
    regroupOperandIfBeneficial(in2);
  }
  if (checkReverse && in1HasAliases) {
    regroupOperandIfBeneficial(in1);
  }

  in1 = in1.flatten();
  in2 = in2.flatten();
  out = out.flatten();
  graph.reorderToSimplify(&out, {&in1, &in2}, false);
  const auto outMapping = graph.getTileMapping(out);

  std::vector<std::vector<BroadcastPattern>> tilePatterns(numTiles),
      tilePatternsReverse(numTiles);
  std::vector<std::vector<std::vector<Interval>>> tileContiguousRegions(
      numTiles);

  // Generates broadcast patterns relative to broadcasting 'operand' into 'out'
  // for the specified tile.
  auto generatePatterns = [&](unsigned tile, Tensor &operand,
                              std::vector<std::vector<Interval>> outRegions) {
    // We work with the contiguous intervals of the output with
    // respect to unique elements of the broadcasted input.
    std::vector<std::size_t> aliases;
    auto in2Regions = graph.getSortedContiguousRegions(
        operand, outMapping[tile], false, &aliases);

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
        while (++it != aliasMap.end() && it->first < region.end()) {
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
    return patterns;
  };

  tbb::parallel_for(unsigned(0), numTiles, [&](unsigned tile) {
    tileContiguousRegions[tile] =
        graph.getSortedContiguousRegions(out, outMapping[tile]);

    if (in2HasAliases) {
      tilePatterns[tile] =
          generatePatterns(tile, in2, tileContiguousRegions[tile]);
    }
    if (checkReverse && in1HasAliases) {
      tilePatternsReverse[tile] =
          generatePatterns(tile, in1, tileContiguousRegions[tile]);
    }
  });

  // Vaguely validate that the patterns cover exactly the elements of the
  // output. This should be covered in unit tests in future but for now
  // this will stop anything silly.
  if (in2HasAliases) {
    validatePatterns(out.numElements(), tilePatterns);
  }
  if (checkReverse && in1HasAliases) {
    validatePatterns(out.numElements(), tilePatternsReverse, true);
  }

  // Predicates for being able to use different methods on each tile.
  auto scalarBroadcastablePredicate = [](const BroadcastPattern &p) {
    return p.innerFactor > 1 && p.regionNumElements() == 1 &&
           p.outerFactor == 1;
  };

  auto outerVectorBroadcastablePredicate = [](const BroadcastPattern &p) {
    return p.regionNumElements() > 1 && p.innerFactor > 1;
  };

  // Toggle this on to log the region data for the patterns detected on each
  // tile. This is a lot of information which is only useful for internal debug
  // hence is hidden behind this toggle.
  static constexpr bool logRegions = false;

  // Generate vertices from the analyses
  auto cs = graph.addComputeSet({dnai});

  logging::popops::debug("BinaryOp begin DebugStr: {}", dnai.getPathName());
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    if (tileContiguousRegions[tile].empty()) {
      continue;
    }
    if (!tilePatterns[tile].empty()) {
      if (logging::popops::shouldLog(logging::Level::Trace)) {
        logging::popops::trace("tile {}: contiguousRegions={}, patterns={}",
                               tile, tileContiguousRegions[tile].size(),
                               tilePatterns[tile].size());
        for (std::size_t i = 0; i < tilePatterns[tile].size(); ++i) {
          const auto &pattern = tilePatterns[tile][i];
          logging::popops::trace("  pattern[{}].innerFactor={} "
                                 "regionNumElements()={} "
                                 "outerFactor={}",
                                 i, pattern.innerFactor,
                                 pattern.regionNumElements(),
                                 pattern.outerFactor);
          if (logRegions) {
            std::stringstream ss;
            if (!pattern.region.empty()) {
              ss << ",[" << pattern.region[0].begin() << ","
                 << pattern.region[0].end() << ")";
              for (std::size_t i = 1; i < pattern.region.size(); ++i) {
                ss << ",[" << pattern.region[i].begin() << ","
                   << pattern.region[i].end() << ")";
              }
            }
            logging::popops::trace("  pattern[{}].region={}", i, ss.str());
          }
        }
      }

      // --------------------------------------
      // First consider the scalar broadcast option.  If the implementation is
      // inefficient this will just return false to fall through to try the
      // other cases
      auto broadcastScalar =
          [&](const std::vector<BroadcastPattern> &patterns,
              const std::vector<std::vector<Interval>> &contiguousRegions,
              Tensor &in1, Tensor &in2) {
            if ((std::all_of(patterns.begin(), patterns.end(),
                             scalarBroadcastablePredicate) ||
                 patterns.size() != contiguousRegions.size()) &&
                haveScalarBroadcastVertexForOp(op, inPlace, dType)) {

              auto singlePatterns = splitIntoScalarBroadcastPatterns(patterns);
              if (singlePatterns.size() != 0) {
                const auto splitRegions = splitContiguousRegionsByPattern(
                    contiguousRegions, singlePatterns);

                bool uniformScalar = std::all_of(
                    std::next(singlePatterns.begin()), singlePatterns.end(),
                    [&](const BroadcastPattern &p) {
                      return p.region == singlePatterns.front().region;
                    });

                if (binaryOpBroadcastScalar(graph, in1, in2, out, splitRegions,
                                            tile, cs, op, inPlace,
                                            uniformScalar, true)) {
                  return true;
                }
              }
            }
            return false;
          };
      // First check if we can broadcast the second operand into the first
      // and then (if allowed) the first into the second
      if ((in2HasAliases &&
           broadcastScalar(tilePatterns[tile], tileContiguousRegions[tile], in1,
                           in2)) ||
          (checkReverse && in1HasAliases &&
           broadcastScalar(tilePatternsReverse[tile],
                           tileContiguousRegions[tile], in2, in1))) {
        continue;
      }

      // --------------------------------------
      // Now consider the Inner Vector broadcast.
      // TODO: T12938 Lift the restriction that all inner vector broadcasts in a
      // 2D vertex have the same length.
      auto broadcastInnerVector =
          [&](const std::vector<BroadcastPattern> &patterns,
              const std::vector<std::vector<Interval>> &contiguousRegions,
              Tensor &in1, Tensor &in2) {
            if (haveInnerVectorBroadcastVertexForOp(op, inPlace, dType)) {
              if (binaryOpBroadcastInnerVector(
                      graph, in1, in2, out, contiguousRegions, patterns, tile,
                      cs, op, inPlace, prog, dnai)) {
                return true;
              }
            }
            return false;
          };

      // First check if we can broadcast the second operand into the first
      // and then (if allowed) the first into the second
      if ((in2HasAliases &&
           broadcastInnerVector(tilePatterns[tile], tileContiguousRegions[tile],
                                in1, in2)) ||
          (checkReverse && in1HasAliases &&
           broadcastInnerVector(tilePatternsReverse[tile],
                                tileContiguousRegions[tile], in2, in1))) {
        continue;
      }

      // --------------------------------------
      // Now consider the Outer Vector broadcast.
      // TODO: T12939 Currently we only have a 1D vertex to perform this kind of
      // operation. When the 2D function becomes available, the 1D vertex should
      // probably be selected every time that there is more than one pattern
      // (T13312).
      auto broadcastOuterVector =
          [&](const std::vector<BroadcastPattern> &patterns,
              const std::vector<std::vector<Interval>> &contiguousRegions,
              Tensor &in1, Tensor &in2) {
            if (std::all_of(patterns.begin(), patterns.end(),
                            outerVectorBroadcastablePredicate) &&
                haveOuterVectorBroadcastVertexForOp(op, inPlace, dType) &&
                patterns.size() <= target.getNumWorkerContexts()) {
              if (binaryOpBroadcastOuterVector(graph, in1, in2, out,
                                               contiguousRegions, patterns,
                                               tile, cs, op, inPlace)) {
                return true;
              }
            }
            return false;
          };
      // First check if we can broadcast the second operand into the first
      // and then (if allowed) the first into the second
      if ((in2HasAliases &&
           broadcastOuterVector(tilePatterns[tile], tileContiguousRegions[tile],
                                in1, in2)) ||
          (checkReverse && in1HasAliases &&
           broadcastOuterVector(tilePatternsReverse[tile],
                                tileContiguousRegions[tile], in2, in1))) {
        continue;
      }
    }
    // Always fall back on the general op for this tile if no valid specialised
    // op could be generated
    binaryOpGeneral(graph, in1, in2, out, tileContiguousRegions[tile], tile, cs,
                    op, inPlace);
  }
  prog.add(Execute(cs, {dnai}));
}

void validateBinaryOpInputs(BinaryOpType op, const Tensor &in1,
                            const Tensor &in2, const DebugNameAndId &dnai) {
  if (in1.elementType() != in2.elementType()) {
    throw poputil::poplibs_error("Binary Op must have same type for "
                                 "both operands: " +
                                 dnai.getPathName());
  }

  if ((op == BinaryOpType::INV_STD_DEV_TO_VARIANCE ||
       op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) &&
      in2.numElements() != 1) {
    throw poputil::poplibs_error("Second operand must be a tensor with a single"
                                 " element for invStdDev to/from variance "
                                 "conversion.");
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

Tensor binaryOp(Graph &graph, Tensor in1, Tensor in2, Sequence &prog,
                BinaryOpType op, bool inPlace, const MapOptions &options,
                const DebugNameAndId &dnai) {
  const std::string layer = "Op/" + debugName(op);

  const auto in1Type = in1.elementType();
  const auto in2Type = in2.elementType();
  const bool in1IsScalar = in1.numElements() == 1;
  const bool in2IsScalar = in2.numElements() == 1;
  validateBinaryOpInputs(op, in1, in2, {dnai, layer});

  logging::popops::debug("  BinaryOp{} DebugStr: {}", inPlace ? "InPlace" : "",
                         dnai.getPathName() + "/" + layer);
  if (inPlace) {
    logging::popops::debug("  in2{}({})  : {}", in2.shapeToString(),
                           in2.elementType(), in2.getDebugStr());
    logging::popops::debug("  inOut{}({}): {}", in1.shapeToString(),
                           in1.elementType(), in1.getDebugStr());
  } else {

    logging::popops::debug("  in1{}({}): {}", in1.shapeToString(),
                           in1.elementType(), in1.getDebugStr());
    logging::popops::debug("  in2{}({}): {}", in2.shapeToString(),
                           in2.elementType(), in2.getDebugStr());
  }
  // Broadcast the inputs to have the same shape here to cover all paths
  // for binary ops

  broadcastToMatch(in1, in2);
  const auto outType = outputType(in1Type, op);

  Tensor out;
  if (inPlace) {
    out = in1;
  } else {
    out = createOutputForElementWiseOp(graph, {in1, in2}, outType,
                                       {dnai, layer + "/Out"});
  }

  if (!inPlace) {
    logging::popops::debug("  out{}({}): {}", out.shapeToString(),
                           out.elementType(), out.getDebugStr());
  }

  logging::popops::debug("  {}{} = {}{} {} {}{}", out.getVarStr(),
                         out.shapeToString(), in1.getVarStr(),
                         in1.shapeToString(), debugName(op), in2.getVarStr(),
                         in2.shapeToString());

  // Special case for scalar broadcast, because knowing this is a binary
  // op and that the broadcasted tensor is a single element means we
  // know what the most efficient way to implement this is across tiles.
  if (haveScalarBroadcastVertexForOp(op, inPlace, in1Type)) {
    // If it's the second operand to be a scalar we can always do it ...
    if (in2IsScalar) {
      binaryOpBroadcastScalar(graph, in1, in2, out, prog, op, inPlace,
                              {dnai, layer});
      return out;
      // ... if it's the first operand we have a couple of checks to do.
    } else if (in1IsScalar && !inPlace && isBinaryOpCommutative(op)) {
      binaryOpBroadcastScalar(graph, in2, in1, out, prog, op, inPlace,
                              {dnai, layer});
      return out;
    }
  }

  // Vector broadcast special case. We try and find the most efficient
  // way to perform the binary operation on each tile.
  bool in1HasAliases = in1.containsAliases();
  bool in2HasAliases = in2.containsAliases();
  if (options.enableVectorBroadcastOptimisations &&
      (in1HasAliases || in2HasAliases)) {
    constructBroadcastBinaryOp(graph, prog, in1, in1HasAliases, in2,
                               in2HasAliases, out, op, inPlace, {dnai, layer});
    return out;
  }

  // General case which works for any given tensors and ops.
  binaryOpGeneral(graph, in1, in2, out, prog, op, inPlace, {dnai, layer});
  return out;
}

Tensor ternaryOp(Graph &graph, Tensor in1, Tensor in2, Tensor in3,
                 Sequence &prog, TernaryOpType op, bool inPlace,
                 const DebugNameAndId &dnai) {

  const std::string layer = "Op/" + debugName(op);
  const auto in1Type = in1.elementType();
  const auto in2Type = in2.elementType();

  const auto in1Size = in1.numElements();
  const auto in2Size = in2.numElements();
  const auto in3Size = in3.numElements();

  // Used to create out tensor. Special case is BroadcastSelect that require in3
  // instead of in1
  auto referenceTensor = in1;
  ternaryOpCodelets codeletOp = ternaryOpCodelets::CLAMP;

  std::string opVertexName = "popops::";
  ternaryOpTensorsMap connectionPattern =
      ternaryOpTensorsMap::TENSOR1_TENSOR2_TENSOR3;

  if (in1Type != in2Type)
    throw poputil::poplibs_error("Ternary Op must have same type for "
                                 "first two operands: " +
                                 dnai.getPathName() + "/" + layer);

  std::vector<Tensor> inputs;
  if (op == TernaryOpType::CLAMP) {
    if (in2Size == 1 && in3Size == 1) {
      referenceTensor = in1;
      codeletOp = ternaryOpCodelets::BROADCAST_CLAMP;
      connectionPattern = ternaryOpTensorsMap::TENSOR1_SCALAR2_SCALAR3;
      opVertexName += "Broadcast" + vertexName(op);
      inputs = {in1};
    } else {
      broadcastToMatch(in1, in2, in3);
      referenceTensor = in1;
      codeletOp = ternaryOpCodelets::CLAMP;
      connectionPattern = ternaryOpTensorsMap::TENSOR1_TENSOR2_TENSOR3;
      opVertexName += vertexName(op);
      inputs = {in1, in2, in3};
    }
  } else if (op == TernaryOpType::SELECT) {
    if (in1Size == 1 && in2Size == 1 && in3Size > 1) {
      referenceTensor = in3;
      codeletOp = ternaryOpCodelets::BROADCAST_SELECT;
      connectionPattern = ternaryOpTensorsMap::SCALAR1_SCALAR2_TENSOR3;
      opVertexName += "Broadcast" + vertexName(op);
      inputs = {in3};
    } else if (in3Size == 1) {
      broadcastToMatch(in1, in2);
      referenceTensor = in1;
      codeletOp = ternaryOpCodelets::BROADCAST_SELECTOR_SELECT;
      connectionPattern = ternaryOpTensorsMap::TENSOR1_TENSOR2_SCALAR3;
      opVertexName += "BroadcastSelector" + vertexName(op);
      inputs = {in1, in2};
    } else {
      broadcastToMatch(in1, in2, in3);
      referenceTensor = in1;
      codeletOp = ternaryOpCodelets::SELECT;
      connectionPattern = ternaryOpTensorsMap::TENSOR1_TENSOR2_TENSOR3;
      opVertexName += vertexName(op);
      inputs = {in1, in2, in3};
    }
  } else {
    throw poplibs_error("Unhandled element-wise ternary op type");
  }

  const auto outType = outputType(in1Type, op);
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet({dnai, layer});

  Tensor out;
  if (inPlace) {
    out = referenceTensor;
  } else {
    out = createOutputForElementWiseOp(graph, inputs, outType,
                                       {dnai, layer + "/Out"});
  }

  auto in1Flat = in1.flatten();
  auto in2Flat = in2.flatten();
  auto in3Flat = in3.flatten();
  auto outFlat = out.flatten();

  // Reorder tensors only if output is bigger than 1 as it works as a reference
  // for all the rest. Only vectors required for reorder.
  if (outFlat.numElements() > 1) {
    std::vector<Tensor *> toReorder;
    if (in1Size > 1)
      toReorder.push_back(&in1Flat);
    if (in2Size > 1)
      toReorder.push_back(&in2Flat);
    if (in3Size > 1)
      toReorder.push_back(&in3Flat);

    graph.reorderToSimplify(&outFlat, toReorder, false);
  }

  const auto mapping = graph.getTileMapping(outFlat);

  const auto grainSize = std::max<unsigned>(target.getVectorWidth(in1Type),
                                            target.getAtomicStoreGranularity());

  // Check if inPlace vertex required
  opVertexName += (inPlace ? "InPlace" : "");
  const auto inOutName = inPlace ? "in1Out" : "in1";

  const auto elementLimit =
      maxVertexElementsPerRegion(target, out.elementType(), codeletOp, inPlace);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, mapping[tile], grainSize,
                                   2 * grainSize, UINT_MAX, elementLimit);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, templateVertex(opVertexName, in1Type));

      // Connect scalar (aka one element tensor) directly otherwise slice it
      switch (connectionPattern) {
      case ternaryOpTensorsMap::TENSOR1_SCALAR2_SCALAR3: {
        graph.connect(v[inOutName], in1Flat.slices(regions));
        graph.connect(v["in2"], in2Flat[0]);
        graph.connect(v["in3"], in3Flat[0]);
        break;
      }
      case ternaryOpTensorsMap::SCALAR1_SCALAR2_TENSOR3: {
        graph.connect(v[inOutName], in1Flat[0]);
        graph.connect(v["in2"], in2Flat[0]);
        graph.connect(v["in3"], in3Flat.slices(regions));
        break;
      }
      case ternaryOpTensorsMap::TENSOR1_TENSOR2_SCALAR3: {
        graph.connect(v[inOutName], in1Flat.slices(regions));
        graph.connect(v["in2"], in2Flat.slices(regions));
        graph.connect(v["in3"], in3Flat[0]);
        break;
      }
      case ternaryOpTensorsMap::TENSOR1_TENSOR2_TENSOR3:
      default: {
        graph.connect(v[inOutName], in1Flat.slices(regions));
        graph.connect(v["in2"], in2Flat.slices(regions));
        graph.connect(v["in3"], in3Flat.slices(regions));
        break;
      }
      }

      if (!inPlace)
        graph.connect(v["out"], outFlat.slices(regions));

      graph.setTileMapping(v, tile);
    }
  }

  prog.add(Execute(cs, {dnai}));

  return out;
}

boost::optional<unsigned> getLowestTileMapping(const Graph &graph,
                                               const Tensor &tensor) {
  auto tensorSimplified = tensor.flatten();
  graph.reorderToSimplify(&tensorSimplified, {}, false);
  auto mapping = graph.getTileMapping(tensorSimplified, false);
  auto isNonEmpty = [](const std::vector<Interval> &intervals) {
    return !intervals.empty();
  };
  auto match = std::find_if(mapping.begin(), mapping.end(), isNonEmpty);
  if (match == mapping.end())
    return boost::none;
  unsigned tile = match - mapping.begin();
  return tile;
}

/// Return the lowest value which is not none, or none if all the values are
/// none.
boost::optional<unsigned>
getLowest(const std::vector<boost::optional<unsigned>> &args) {
  boost::optional<unsigned> lowest;
  for (auto &arg : args) {
    if (!arg)
      continue;
    if (lowest)
      lowest = std::min(*lowest, *arg);
    else
      lowest = *arg;
  }
  return lowest;
}

/// Given an expression infer the tile used for that expression.
/// If the expression uses multiple tiles we arbitrarily return the
/// lowest tile number. The tiles inferred for constants are added to the
/// \a constTiles map.
boost::optional<unsigned>
inferTile(const Graph &graph, const expr::Expr &expr,
          const std::vector<Tensor> &ts,
          std::unordered_map<const expr::Expr *, unsigned> &constTiles,
          std::vector<const expr::Expr *> &unknown) {
  if (expr.isA<expr::Const>() || expr.isA<expr::Cast>()) {
    unknown.push_back(&expr);
    return {};
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    return getLowestTileMapping(graph, getTensorFromPlaceHolder(*p, ts));
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    return inferTile(graph, u->getArg(), ts, constTiles, unknown);
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto lhsTile = inferTile(graph, b->getLHS(), ts, constTiles, unknown);
    auto rhsTile = inferTile(graph, b->getRHS(), ts, constTiles, unknown);
    // Arbitrarily return the lowest tile that appears in any sub-expression.
    auto commonTile = getLowest({lhsTile, rhsTile});
    if (commonTile) {
      for (const auto e : unknown)
        constTiles[e] = *commonTile;
      unknown.clear();
    }
    return commonTile;
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto arg0Type = inferTile(graph, t->getArg0(), ts, constTiles, unknown);
    auto arg1Type = inferTile(graph, t->getArg1(), ts, constTiles, unknown);
    auto arg2Type = inferTile(graph, t->getArg2(), ts, constTiles, unknown);
    // Arbitrarily return the lowest tile that appears in any sub-expression.
    auto commonTile = getLowest({arg0Type, arg1Type, arg2Type});
    if (commonTile) {
      for (const auto e : unknown)
        constTiles[e] = *commonTile;
      unknown.clear();
    }
    return commonTile;
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
// Further in-place optimisations are possible by traversing the tree and
// transforming the operations.
std::pair<Tensor, bool>
map(Graph &graph, const expr::Expr &expr, const std::vector<Tensor> &ts,
    program::Sequence &prog, const poplar::DebugNameAndId &dnai,
    const std::unordered_map<const expr::Expr *, Type> constTypes,
    const std::unordered_map<const expr::Expr *, unsigned> constTiles,
    bool topLevel, bool constructGraph, bool inPlace,
    const expr::Expr *&inPlaceExpr, const MapOptions &options) {

  if (!constructGraph)
    assert(!inPlace);
  if (const expr::Const *c = expr.getAs<expr::Const>()) {
    assert(constTypes.find(&expr) != constTypes.end());
    auto ct = graph.addConstant(constTypes.at(&expr), {}, c->getData(),
                                c->getTypeTraits(), false, {dnai, "<const>"});
    unsigned tile = 0;
    auto match = constTiles.find(&expr);
    if (match != constTiles.end())
      tile = match->second;
    graph.setTileMapping(ct, tile);
    return {ct, false};
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    const auto &t = getTensorFromPlaceHolder(*p, ts);
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
      if (topLevel && (!useInPlace || (useInPlace && index != 1))) {
        // We are asked to return the very tensor specified by the placeholder
        // ('t'). We could simply return it, but we are requested not to do an
        // in-place operation, so we just make a copy.
        auto t2 = graph.clone(t, {dnai});
        prog.add(Copy(t, t2, false, {dnai}));
        return {t2, useInPlace};
      }
    }
    return {t, useInPlace};
  } else if (const expr::Cast *c = expr.getAs<expr::Cast>()) {
    auto t = map(graph, c->getLHS(), ts, prog, {dnai}, constTypes, constTiles,
                 false, constructGraph, inPlace, inPlaceExpr, options);
    if (constructGraph) {
      return {cast(graph, t.first, c->getRHSType(), prog, {dnai}), t.second};
    } else {
      return {graph.clone(c->getRHSType(), t.first, {dnai}), t.second};
    }
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    auto opType = u->getOpType();
    auto t = map(graph, u->getArg(), ts, prog, {dnai}, constTypes, constTiles,
                 false, constructGraph, inPlace, inPlaceExpr, options);
    if (constructGraph) {
      return {unaryOp(graph, t.first, prog, opType, t.second, {dnai}),
              t.second};
    } else {
      return t;
    }
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    auto opType = b->getOpType();
    auto lhs = map(graph, b->getLHS(), ts, prog, {dnai}, constTypes, constTiles,
                   false, constructGraph, inPlace, inPlaceExpr, options);
    auto rhs = map(graph, b->getRHS(), ts, prog, {dnai}, constTypes, constTiles,
                   false, constructGraph, false, inPlaceExpr, options);
    if (constructGraph) {
      return {binaryOp(graph, lhs.first, rhs.first, prog, opType, lhs.second,
                       options, {dnai}),
              lhs.second};
    } else {
      return lhs;
    }
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    auto opType = t->getOpType();
    if (opType == TernaryOpType::SELECT) {
      auto lhs =
          map(graph, t->getArg0(), ts, prog, {dnai}, constTypes, constTiles,
              false, constructGraph, inPlace, inPlaceExpr, options);
      auto rhs =
          map(graph, t->getArg1(), ts, prog, {dnai}, constTypes, constTiles,
              false, constructGraph, false, inPlaceExpr, options);
      auto pred =
          map(graph, t->getArg2(), ts, prog, {dnai}, constTypes, constTiles,
              false, constructGraph, false, inPlaceExpr, options);
      if (constructGraph) {
        return {ternaryOp(graph, lhs.first, rhs.first, pred.first, prog, opType,
                          lhs.second, {dnai}),
                lhs.second};
      } else {
        return lhs;
      }
    } else {
      assert(opType == TernaryOpType::CLAMP);
      auto in =
          map(graph, t->getArg0(), ts, prog, {dnai}, constTypes, constTiles,
              false, constructGraph, inPlace, inPlaceExpr, options);
      auto lower =
          map(graph, t->getArg1(), ts, prog, {dnai}, constTypes, constTiles,
              false, constructGraph, false, inPlaceExpr, options);
      auto upper =
          map(graph, t->getArg2(), ts, prog, {dnai}, constTypes, constTiles,
              false, constructGraph, false, inPlaceExpr, options);
      if (constructGraph) {
        return {ternaryOp(graph, in.first, lower.first, upper.first, prog,
                          opType, in.second, {dnai}),
                in.second};
      } else {
        return in;
      }
    }
  }
  POPLIB_UNREACHABLE();
}

std::unordered_map<const expr::Expr *, unsigned>
getConstTile(const Graph &graph, const expr::Expr &expr,
             const std::vector<Tensor> &ts) {
  std::unordered_map<const expr::Expr *, unsigned> constTiles;
  std::vector<const expr::Expr *> unknown;
  inferTile(graph, expr, ts, constTiles, unknown);
  return constTiles;
}

} // end anonymous namespace

bool createVertexBinaryOpBroadcastScalar(
    Graph &graph, const Tensor &in1, const Tensor &in2, const Tensor &out,
    const std::vector<std::vector<Interval>> &intervals, unsigned tile,
    const ComputeSet &cs, BinaryOpType op, bool inPlace, bool uniformScalar,
    bool exitIfInefficient) {
  if (intervals.size() == 0) {
    return true;
  }
  const auto &target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto inType = in1.elementType();
  const auto outType = out.elementType();
  const auto grainSize = std::max<unsigned>(target.getVectorWidth(inType),
                                            target.getAtomicStoreGranularity());

  const auto elementLimit = maxVertexElementsPerRegion(target, op, in1, out);
  const auto vertexRegions =
      splitRegionsBetweenWorkers(target, intervals, grainSize, 2 * grainSize,
                                 UINT_MAX, elementLimit.regionSize);

  auto elemsInRegion = [](const std::vector<Interval> &region) {
    return std::accumulate(
        region.begin(), region.end(), std::size_t(0),
        [](std::size_t total, const Interval &i) { return total + i.size(); });
  };

  // MultiVertex cycle estimates based on the number of contiguous regions
  std::uint64_t cycleEst1D = 0;
  for (auto region : intervals) {
    auto numElems = elemsInRegion(region);
    cycleEst1D += popops::internal::broadcastArithmetic1DCycleEstimate(
                      target, op, inType, outType, inPlace, numElems)
                      .cycles;
  }

  // The 2D regions are split among worker contexts to distribute the elements
  // uniformly among the workers. Since the workers are balanced according to
  // the number of elements and not processor cycles, workers with more regions
  // to process are likely to take longer to complete. This heuristic is used
  // to estimate the cycle cost for the most loaded worker.
  auto max2DRegions = std::max_element(
      vertexRegions.begin(), vertexRegions.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.size() < rhs.size(); });
  std::vector<std::size_t> max2DRegionSizes(max2DRegions->size());
  for (unsigned i = 0; i < max2DRegions->size(); ++i) {
    max2DRegionSizes[i] = elemsInRegion((*max2DRegions)[i]);
  }
  auto cycleEst2D =
      popops::internal::broadcastArithmeticCycleEstimate(
          target, op, inType, outType, inPlace, uniformScalar, max2DRegionSizes)
          .cycles;

  // Cycles estimates for MultiVertex are in machine cycles whereas 2D estimates
  // are in thread cycles.
  bool useMultiVertex = cycleEst1D < cycleEst2D * target.getNumWorkerContexts();

  if (useMultiVertex && exitIfInefficient) {
    // If necessary insert criteria for exit, having chosen 1D MultiVertex
    // over workers here.
  }

  // Having chosen worker vertices over 1D MultiVertex, exit if that is
  // inefficient.
  if (!useMultiVertex && exitIfInefficient) {
    // Calculate the total number of elements on the tile, and the number
    // of regions these are split into for the workers
    std::size_t totalElems = intervalSequenceNumElements(intervals);
    unsigned regions = 0;
    for (const auto &vertexRegion : vertexRegions) {
      regions += vertexRegion.size();
    }
    const auto elementsPerWorkerRegion = totalElems / regions;

    // Use a heuristic based on avoiding assigning many workers a very small
    // amount of work to decide if we should exit and abandon this method
    if (regions > numWorkers && elementsPerWorkerRegion < 2 * grainSize) {
      return false;
    }
  }
  auto isVarianceConversionBinaryOp = [](BinaryOpType &op) {
    return (op == BinaryOpType::VARIANCE_TO_INV_STD_DEV) ||
           (op == BinaryOpType::INV_STD_DEV_TO_VARIANCE);
  };
  if (useMultiVertex &&
      validateRegionSizeForMultiVertex(intervals, elementLimit, numWorkers)) {
    std::string vertexClass;
    if (isVarianceConversionBinaryOp(op) && (inType != outType)) {
      assert(!inPlace);
      auto vertexName = "popops::BroadcastScalar2Types1D";
      vertexClass = templateVertex(vertexName, op, inType, outType);
    } else {
      auto vertexName = inPlace ? "popops::BroadcastScalar1DInPlace"
                                : "popops::BroadcastScalar1D";
      vertexClass = templateVertex(vertexName, op, inType);
    }
    if (intervals.size()) {
      logging::popops::trace("  Tile: {} Producing: {} {} vertices", tile,
                             intervals.size(), vertexClass);
    }
    for (const auto &regions : intervals) {
      const auto outRegion = concat(out.flatten().slices(regions));
      const auto in1Region = concat(in1.flatten().slices(regions));
      // We know that for this interval the second operand is a
      // scalar value. There are two cases:
      //  - If the operand is a tensor of scalars, use a slice.
      //  - If the operand is a single scalar, just use it directly
      assert(!regions.empty());
      const auto in2ScalarRegion = in2.numElements() > 1
                                       ? in2.flatten()[regions.front().begin()]
                                       : in2.reshape({});
      const auto v = graph.addVertex(
          cs, vertexClass, {{"data", in1Region}, {"B", in2ScalarRegion}});
      if (!inPlace) {
        graph.connect(v["out"], outRegion);
      }
      graph.setTileMapping(v, tile);
    }
  } else {
    std::string vertexClass;
    std::vector<std::size_t> bShape;
    if (isVarianceConversionBinaryOp(op) && (inType != outType)) {
      assert(!inPlace);
      assert(uniformScalar);
      auto vertexName = "popops::BroadcastScalar2Types2DData";
      vertexClass = templateVertex(vertexName, op, inType, outType);
    } else {
      std::string vertexName = uniformScalar ? "popops::BroadcastScalar2DData"
                                             : "popops::BroadcastScalar2D";
      if (!uniformScalar) {
        bShape.push_back(1);
      }
      if (inPlace) {
        vertexName += "InPlace";
      }
      vertexClass = templateVertex(vertexName, op, inType);
    }
    if (vertexRegions.size()) {
      logging::popops::trace("  Tile: {} Producing: {} {} vertices", tile,
                             vertexRegions.size(), vertexClass);
    }
    for (const auto &regions : vertexRegions) {
      const auto outRegions = out.flatten().slices(regions);
      const auto in1Regions = in1.flatten().slices(regions);
      const auto v = graph.addVertex(cs, vertexClass, {{"data", in1Regions}});
      if (!inPlace) {
        graph.connect(v["out"], outRegions);
      }
      assert(!regions.empty());
      if (in2.numElements() == 1) {
        graph.connect(v["B"], in2.reshape(bShape));
      } else if (uniformScalar) {
        const auto &region = regions.front();
        assert(!region.empty());
        const auto in2ScalarRegion = in2.flatten()[region.front().begin()];
        graph.connect(v["B"], in2ScalarRegion);
      } else {
        // Take the first element in each region as the scalar.
        // We know that this must be the same element for all in each
        // region, otherwise calling this function is invalid.
        std::vector<Tensor> in2ScalarRegions;
        in2ScalarRegions.reserve(regions.size());
        for (const auto &region : regions) {
          assert(!region.empty());
          in2ScalarRegions.push_back(in2.flatten()[region.front().begin()]);
        }
        graph.connect(v["B"], in2ScalarRegions);
      }
      graph.setTileMapping(v, tile);
    }
  }
  return true;
}

static std::vector<Type> getTypesFromTensors(const std::vector<Tensor> &ts) {
  std::vector<Type> types;
  types.reserve(ts.size());
  std::transform(ts.begin(), ts.end(), std::back_inserter(types),
                 [](const auto &t) { return t.elementType(); });
  return types;
}

Tensor map(Graph &graph, const expr::Expr &expr, const std::vector<Tensor> &ts,
           program::Sequence &prog, const poplar::DebugContext &debugContext,
           const OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(ts, expr, options));

  logging::popops::debug("MapExpression DebugStr:{}", debugContext);

  auto opts = parseOptionFlags(options);

  const auto tTypes = getTypesFromTensors(ts);

  if ((ts.size() == 2) && expr.deepEquals(expr::Mul(_1, _2)) &&
      inputsMatchMixedPrecisionScalarMultiplyPattern(ts[0], ts[1], true)) {
    return scalarMultiply(graph, ts[0], ts[1], prog, di, options);
  }

  std::unique_ptr<expr::Expr> newExpr;
  if (opts.enableExpressionOptimizations) {
    newExpr = optimise(expr, tTypes).expression;
  }
  const auto &optExpr = opts.enableExpressionOptimizations ? *newExpr : expr;

  auto constTypes = getConstType(optExpr, tTypes);
  // If the user hasn't overridden 'enableGenerateCodelet' to be false and all
  // of the inputs don't alias and are the same size we can generate a codelet
  // to execute this map.
  const auto canGenerateCodelet =
      analyseExpr(optExpr, ts, opts.forceGenerateCodelet);
  if (opts.enableGenerateCodelet && canGenerateCodelet.isSupported) {
    return generateAndExecuteMappedOperations(
        graph, optExpr, ts, constTypes, prog, false,
        canGenerateCodelet.allInputsScalar, {di});
  }

  auto constTiles = getConstTile(graph, optExpr, ts);
  const expr::Expr *inplaceExpr = nullptr;
  auto output = map(graph, optExpr, ts, prog, {di}, constTypes, constTiles,
                    true, true, false, inplaceExpr, opts)
                    .first;
  di.addOutput(output);
  return output;
} // namespace popops

void mapInPlace(Graph &graph, const expr::Expr &expr,
                const std::vector<Tensor> &ts, program::Sequence &prog,
                const poplar::DebugContext &debugContext,
                const OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(ts, expr, options));

  logging::popops::debug("MapInPlaceExpression DebugStr:{}", debugContext);

  auto opts = parseOptionFlags(options);

  const auto tTypes = getTypesFromTensors(ts);

  if ((ts.size() == 2) && expr.deepEquals(expr::Mul(_1, _2)) &&
      inputsMatchMixedPrecisionScalarMultiplyPattern(ts[0], ts[1])) {
    scalarMultiplyInplace(graph, ts[0], ts[1], prog, di, options);
    return;
  }

  std::unique_ptr<expr::Expr> newExpr;
  if (opts.enableExpressionOptimizations) {
    newExpr = optimise(expr, tTypes).expression;
  }
  const auto &optExpr = opts.enableExpressionOptimizations ? *newExpr : expr;

  auto constTypes = getConstType(optExpr, tTypes);
  // If the user hasn't overridden 'enableGenerateCodelet' to be false and all
  // of the inputs don't alias and are the same size we can generate a codelet
  // to execute this map.
  const auto canGenerateCodelet =
      analyseExpr(optExpr, ts, opts.forceGenerateCodelet);
  if (opts.enableGenerateCodelet && canGenerateCodelet.isSupported) {
    generateAndExecuteMappedOperations(graph, optExpr, ts, constTypes, prog,
                                       true, canGenerateCodelet.allInputsScalar,
                                       {di});
    return;
  }

  auto constTiles = getConstTile(graph, optExpr, ts);
  const expr::Expr *inPlaceExpr = nullptr;
  const bool doInPlace = !ts[0].containsAliases() && !ts[0].containsConstant();
  if (doInPlace) {
    // As the tree is traversed, find the last expression which uses the
    // tensor used for in-place operation as a placeholder
    map(graph, optExpr, ts, prog, {di}, constTypes, constTiles, true, false,
        false, inPlaceExpr, opts);
  }
  auto t = map(graph, optExpr, ts, prog, {di}, constTypes, constTiles, true,
               true, doInPlace, inPlaceExpr, opts);
  // If in-place operations were not performed, then copy the final result
  // into the tensor supplied.
  // TODO T12943 Optimisation: If placeholder _1 is not used, a copy may be done
  // early enough to avoid this copy and use in-place operations after that
  // copy. Or, the unary, binary and ternary operations must allow an output
  // tensor to be given as an argument (the current method either uses one of
  // the input tensors if the operation is in-place, or creates and output
  // tensor).
  if (!t.second) {
    prog.add(Copy(t.first, ts[0], false, {di}));
  }
}

} // namespace popops

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const popops::expr::UnaryOpType &op) {
  return poplar::ProfileValue(popops::expr::debugName(op));
}

template <>
poplar::ProfileValue toProfileValue(const popops::expr::BinaryOpType &op) {
  return poplar::ProfileValue(popops::expr::debugName(op));
}

template <>
poplar::ProfileValue toProfileValue(const popops::expr::TernaryOpType &op) {
  return poplar::ProfileValue(popops::expr::debugName(op));
}
} // namespace poputil
