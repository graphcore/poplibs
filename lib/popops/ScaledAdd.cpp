// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popops/ScaledAdd.hpp"
#include "poplar/Program.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Rearrange.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional.hpp>

#include <optional>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const popops::ScaledAddSpecialisation &p) {
  switch (p) {
  case popops::ScaledAddSpecialisation::DEFAULT:
    return poplar::ProfileValue("Default");
  case popops::ScaledAddSpecialisation::X_MINUS_AX_PLUS_BY:
    return poplar::ProfileValue("X_minus_AX_plus_BY");
  }
  return {};
}
} // namespace poputil

namespace popops {

namespace {

// Check if we can use a supervisor vertex, or if the regions to process
// prevent it.  This can be due to either having multiple regions or if the
// region is too large.
bool validateRegionSizeForSupervisorVertex(
    const std::vector<std::vector<Interval>> &intervals,
    unsigned maxRegionSize) {
  if (maxRegionSize == UINT_MAX) {
    return true;
  }
  const auto numElems = intervalSequenceNumElements(intervals);
  if (numElems > maxRegionSize) {
    return false;
  }
  return true;
}

struct ScaledAddOptions {
  bool optimizeForSpeed = false;
  double floatToHalfTolerance = 1e-6;
};

ScaledAddOptions parseOptionFlags(const OptionFlags &options) {
  ScaledAddOptions scaledAddOpts;
  const poplibs::OptionSpec scaledAddSpec{
      {"optimizeForSpeed",
       poplibs::OptionHandler::createWithBool(scaledAddOpts.optimizeForSpeed)},
      {"scaleFloatToHalfTolerance", poplibs::OptionHandler::createWithDouble(
                                        scaledAddOpts.floatToHalfTolerance)},
  };
  for (const auto &entry : options) {
    scaledAddSpec.parse(entry.first, entry.second);
  }
  return scaledAddOpts;
}

static inline bool shouldRegroupBeforeCast(const Target &target, Type from,
                                           Type to) {
  // As a rough estimate of what will be more runtime efficient, we attempt
  // regrouping before the cast if there is less data to move.
  return target.getTypeSize(from) < target.getTypeSize(to);
}

void createVertices(Graph &graph, Tensor A, std::optional<Tensor> scaleA,
                    Tensor B, Tensor scaleB, const bool doSubtract,
                    const bool doaXPlusbY,
                    const ScaledAddSpecialisation speciality, Sequence &prog,
                    bool attemptRegroup, const DebugNameAndId &dnai,
                    const ScaledAddOptions &opts) {
  // <half,float> vertices are unconstrained

  const auto addConstraints =
      (A.elementType() == HALF || A.elementType() == FLOAT) &&
      !(A.elementType() == FLOAT && B.elementType() == FLOAT) &&
      !(A.elementType() == FLOAT && B.elementType() == HALF) &&
      !(A.elementType() == HALF && B.elementType() == FLOAT &&
        speciality == ScaledAddSpecialisation::DEFAULT) &&
      opts.optimizeForSpeed;
  if (!A.isParallelWriteable())
    throw poputil::poplibs_error("Trying to accumulate to tensor that cannot be"
                                 " written in parallel");
  if (A.shape() != B.shape())
    throw poputil::poplibs_error("Input Tensors for scaled arithmetic must"
                                 " have the same shape");

  // `scaleA` is only used when `doaXPlusbY==true`
  if (doaXPlusbY && scaleA->elementType() != scaleB.elementType())
    throw poputil::poplibs_error("Scale factors must be of the same type");
  if (speciality == ScaledAddSpecialisation::X_MINUS_AX_PLUS_BY) {
    if (!doaXPlusbY)
      throw poputil::poplibs_error(
          "Scaled add X-aX+bY is only supported together "
          "with doaXPlusbY option");
    if (doSubtract)
      throw poputil::poplibs_error("Subtraction not supported with X-aX+bY");
  }

  if (doaXPlusbY) {
    logging::popops::debug("aXPlusbY Tensor Debug: {}", dnai.getPathName());
    logging::popops::debug("  inOutA{}({}): {}", A.shapeToString(),
                           A.elementType(), A.getDebugStr());
    logging::popops::debug("  inputB{}({}): {}", B.shapeToString(),
                           B.elementType(), B.getDebugStr());
    logging::popops::debug("  scaleA{}({}): {} ", scaleA->shapeToString(),
                           scaleA->elementType(), scaleA->getDebugStr());
    logging::popops::debug("  scaleB{}({}): {}", scaleB.shapeToString(),
                           scaleB.elementType(), scaleB.getDebugStr());
    logging::popops::debug("  {}{} = {}{} * {}{} + {}{} * {}{}", A.getVarStr(),
                           A.shapeToString(), A.getVarStr(), A.shapeToString(),
                           scaleA->getVarStr(), scaleA->shapeToString(),
                           B.getVarStr(), B.shapeToString(), scaleB.getVarStr(),
                           scaleB.shapeToString());
  } else {
    logging::popops::debug("ScaledAdd Tensor Debug: {}", dnai.getPathName());
    logging::popops::debug("  inOutA{}({}): {}", A.shapeToString(),
                           A.elementType(), A.getDebugStr());
    logging::popops::debug("  inputB{}({}): {}", B.shapeToString(),
                           B.elementType(), B.getDebugStr());
    logging::popops::debug("  scaleB{}({}): {}", scaleB.shapeToString(),
                           scaleB.elementType(), scaleB.getDebugStr());
    logging::popops::debug("  {}{} += {}{} * {}{}", A.getVarStr(),
                           A.shapeToString(), B.getVarStr(), B.shapeToString(),
                           scaleB.getVarStr(), scaleB.shapeToString());
  }

  const auto &target = graph.getTarget();
  const auto dataType = A.elementType();
  const auto deltaType = B.elementType();
  const auto scaleType = scaleB.elementType();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet({dnai, "AddTo"});
  const auto vectorWidth = target.getVectorWidth(dataType);
  const auto numWorkers = target.getNumWorkerContexts();

  bool vertexHasTolerance = false;
  std::string codeletName2D;
  std::string codeletNameSupervisor;
  if (doSubtract && doaXPlusbY) {
    // The 'mixed' vertex (with 'half' data and 'float' scales) has a
    // 'tolerance' field ...
    vertexHasTolerance = (dataType == HALF) && (scaleType == FLOAT);

    codeletName2D = templateVertex("popops::aXMinusbY2D", dataType, scaleType,
                                   addConstraints);
    codeletNameSupervisor = templateVertex("popops::aXMinusbYSupervisor",
                                           dataType, scaleType, addConstraints);
  } else if (doSubtract && !doaXPlusbY) {
    codeletName2D = templateVertex("popops::ScaledSubtract2D", dataType,
                                   scaleType, addConstraints);
    codeletNameSupervisor =
        templateVertex("popops::ScaledSubtractSupervisor", dataType, deltaType,
                       scaleType, addConstraints);
  } else if (!doSubtract && doaXPlusbY) {
    if (speciality == ScaledAddSpecialisation::X_MINUS_AX_PLUS_BY) {
      codeletName2D =
          templateVertex("popops::XMinusaXPlusbY2D", dataType, addConstraints);
      codeletNameSupervisor = templateVertex("popops::XMinusaXPlusbYSupervisor",
                                             dataType, addConstraints);
    } else {
      // The 'mixed' vertex (with 'half' data and 'float' scales) has a
      // 'tolerance' field ...
      vertexHasTolerance = (dataType == HALF) && (scaleType == FLOAT);

      codeletName2D = templateVertex("popops::aXPlusbY2D", dataType, scaleType,
                                     addConstraints);
      codeletNameSupervisor = templateVertex(
          "popops::aXPlusbYSupervisor", dataType, scaleType, addConstraints);
    }
  } else if (!doSubtract && !doaXPlusbY) {
    vertexHasTolerance =
        (dataType == HALF) && (deltaType == HALF) && (scaleType == FLOAT);
    codeletName2D = templateVertex("popops::ScaledAdd2D", dataType, deltaType,
                                   scaleType, addConstraints);
    codeletNameSupervisor =
        templateVertex("popops::ScaledAddSupervisor", dataType, deltaType,
                       scaleType, addConstraints);
  }

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max2DInnerElements =
      std::min<std::size_t>(graph.getMaxFieldDim(codeletName2D, "A", 1),
                            target.getRptCountMax() * vectorWidth);

  const auto codeletNameSupervisorForSizingOnly =
      templateVertex("popops::ScaledAddSupervisor", dataType, deltaType,
                     scaleType, addConstraints);

  const auto maxSupervisorElements = std::min<std::size_t>(
      graph.getMaxVertexFieldValue(codeletNameSupervisorForSizingOnly, "size"),
      target.getRptCountMax() * vectorWidth * numWorkers);

  if (attemptRegroup) {
    // Ideally we'd perform the potential regroup on the simplified view
    // but currently the detection of grouping relies on the shape given.
    B = popops::rearrange::regroupIfBeneficial(graph, B, A, prog, {dnai});
  }

  auto aFlat = A.flatten();
  auto bFlat = B.flatten();
  graph.reorderToSimplify(&aFlat, {&bFlat}, false);
  const auto mapping = graph.getTileMapping(aFlat);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    if (mapping[tile].empty())
      continue;
    const auto grainSize = target.getVectorWidth(dataType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);

    if (tileContiguousRegions.size() == 1 &&
        validateRegionSizeForSupervisorVertex(tileContiguousRegions,
                                              maxSupervisorElements)) {
      const auto aContiguous = concat(aFlat.slices(tileContiguousRegions));
      const auto bContiguous = concat(bFlat.slices(tileContiguousRegions));

      VertexRef v = graph.addVertex(cs, codeletNameSupervisor,
                                    {{"A", aContiguous},
                                     {"B", bContiguous},
                                     {"scaleB", scaleB.reshape({1})}});
      if (doaXPlusbY) {
        graph.connect(v["scaleA"], scaleA->reshape({1}));
      }

      graph.setInitialValue(v["size"], aContiguous.numElements());
      if (vertexHasTolerance) {
        graph.setInitialValue(v["tolerance"], opts.floatToHalfTolerance);
      }
      graph.setTileMapping(v, tile);
    } else {
      auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, grainSize, 2 * grainSize, UINT32_MAX,
          max2DInnerElements);
      for (const auto &regions : vertexRegions) {
        VertexRef v = graph.addVertex(cs, codeletName2D,
                                      {{"A", aFlat.slices(regions)},
                                       {"B", bFlat.slices(regions)},
                                       {"scaleB", scaleB}});

        if (doaXPlusbY) {
          graph.connect(v["scaleA"], *scaleA);
        }

        if (vertexHasTolerance) {
          graph.setInitialValue(v["tolerance"], opts.floatToHalfTolerance);
        }
        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(cs, {dnai}));
}

void scaledAritTensorImpl(Graph &graph, Tensor A, Tensor scaleA, Tensor B,
                          Tensor scaleB, Sequence &prog, bool subtract,
                          const ScaledAddSpecialisation speciality,
                          const DebugNameAndId &dnai,
                          const poplar::OptionFlags &options) {
  const auto opts = parseOptionFlags(options);
  const auto dataTypeA = A.elementType();
  const auto scaleAType = scaleA.elementType();
  const auto scaleBType = scaleB.elementType();
  const std::string layer = subtract ? "scaledSubtract" : "scaledAdd";
  bool axpby = true;

  auto scaleType =
      (scaleAType == FLOAT || scaleBType == FLOAT) ? FLOAT : dataTypeA;

  if (scaleAType != scaleType) {
    scaleA = cast(graph, scaleA, scaleType, prog, {dnai, layer + "scaleA"});
  }
  // We only support half,float axpby vertex. Synthesize using mul and scaledAdd
  if (A.elementType() != HALF && A.elementType() != FLOAT) {
    mulInPlace(graph, A, scaleA, prog, {dnai, layer});
    axpby = false;
  }

  bool regroupBeforeCast =
      shouldRegroupBeforeCast(graph.getTarget(), B.elementType(), dataTypeA);
  if (regroupBeforeCast) {
    B = popops::rearrange::regroupIfBeneficial(graph, B, A, prog,
                                               {dnai, layer + "/regroupB"});
  }
  if (dataTypeA == HALF && B.elementType() == FLOAT && !subtract &&
      speciality == ScaledAddSpecialisation::DEFAULT) {
    // There is a specialisation for these vertices so don't cast.
  } else {
    const auto cs = graph.addComputeSet({dnai, layer + "cast"});
    if (dataTypeA != B.elementType()) {
      B = cast(graph, B, dataTypeA, cs, {dnai, layer + "/B"});
    }
    if (scaleBType != scaleType) {
      scaleB = cast(graph, scaleB, scaleType, cs, {dnai, layer + "/scaleB"});
    }
    prog.add(Execute(cs, {dnai}));
  }
  createVertices(graph, A, scaleA, B, scaleB, subtract, axpby, speciality, prog,
                 !regroupBeforeCast, {dnai}, opts);
}

void scaledAritConstImpl(Graph &graph, Tensor A, float scaleA, Tensor B,
                         float scaleB, Sequence &prog, bool subtract,
                         const ScaledAddSpecialisation speciality,
                         const DebugNameAndId &dnai,
                         const poplar::OptionFlags &options) {
  const auto opts = parseOptionFlags(options);
  const auto target = graph.getTarget();
  const auto targetType = A.elementType();
  const std::string layer = subtract ? "scaledSubtract" : "scaledAdd";

  // we only support half,float axpby. Synthesize using mul and scaledAdd
  if (A.elementType() != HALF && A.elementType() != FLOAT && scaleA != 1.0f) {
    const auto scaleATensor =
        graph.addConstant(targetType, {}, scaleA, {dnai, layer + "/scaleA"});
    graph.setTileMapping(scaleATensor, 0);
    mulInPlace(graph, A, scaleATensor, prog, {dnai, layer});
    scaleA = 1.0f;
  }

  bool regroupBeforeCast =
      shouldRegroupBeforeCast(target, B.elementType(), targetType);
  if (regroupBeforeCast) {
    B = popops::rearrange::regroupIfBeneficial(graph, B, A, prog,
                                               {dnai, layer + "/regroupB"});
  }

  if (B.elementType() != targetType &&
      !(targetType == HALF && B.elementType() == FLOAT)) {
    B = cast(graph, B, targetType, prog, {dnai, layer + "/B"});
  }
  if (subtract) {
    scaleB = -scaleB;
  }
  auto scaleType = targetType;
  // If the data is HALF type, when we cast the scale values to HALF will
  // they be accurate enough? If they lose accuracy, we keep them as FLOAT
  if (speciality == ScaledAddSpecialisation::DEFAULT && targetType == HALF) {
    auto tolerance = opts.floatToHalfTolerance;
    if (!poputil::checkAccuracyWhenCast(target, scaleA, FLOAT, HALF,
                                        tolerance) ||
        !poputil::checkAccuracyWhenCast(target, scaleB, FLOAT, HALF,
                                        tolerance)) {
      scaleType = FLOAT;
    }
  }
  // The vertex uses a tensor scale
  const auto scaleATensor =
      graph.addConstant(scaleType, {}, scaleA, {dnai, "/scaleA"});
  const auto scaleBTensor =
      graph.addConstant(scaleType, {}, scaleB, {dnai, "/scaleB"});
  graph.setTileMapping(scaleATensor, 0);
  graph.setTileMapping(scaleBTensor, 0);
  createVertices(graph, A, scaleATensor, B, scaleBTensor, false, scaleA != 1.0f,
                 speciality, prog, !regroupBeforeCast, {dnai}, opts);
}

bool specialisedVertexExists(const Type &A, const Type &B, const Type &scaleB,
                             bool subtract) {
  // There are specialisations for
  // float,half,float * add/subtract
  // float,half,half  * add/subtract
  // half,float,float * add
  // half,float,half  * add
  // half,half,float  * add
  return ((A == FLOAT && B == HALF) || (A == HALF && B == HALF && !subtract) ||
          (A == HALF && B == FLOAT && !subtract)) &&
         (scaleB == HALF || scaleB == FLOAT);
}

bool specialisedVertexExists(const Tensor &A, const Tensor &B,
                             const Tensor &scaleB, bool subtract) {
  return specialisedVertexExists(A.elementType(), B.elementType(),
                                 scaleB.elementType(), subtract);
}

} // namespace
void scaledAddTo(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
                 Sequence &prog, const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(A, B, scaleB, options));
  const auto targetType = A.elementType();
  const std::string layer = "scaledAdd";

  const auto opts = parseOptionFlags(options);
  if (A.elementType() == HALF && B.elementType() == HALF &&
      scaleB.elementType() == FLOAT) {
    // The vertex will select float or half scale based on the accuracy of the
    // scale, using the tolerance option
    createVertices(graph, A, std::nullopt, B, scaleB, false, false,
                   ScaledAddSpecialisation::DEFAULT, prog,
                   /* attemptRegroup */ true, {di}, opts);
  } else {
    bool regroupBeforeCast = false;
    if (!specialisedVertexExists(A, B, scaleB, false)) {
      regroupBeforeCast = shouldRegroupBeforeCast(graph.getTarget(),
                                                  B.elementType(), targetType);
      if (regroupBeforeCast) {
        B = popops::rearrange::regroupIfBeneficial(graph, B, A, prog,
                                                   {di, layer + "/regroupB"});
      }
      const auto cs = graph.addComputeSet({di, layer + "/cast"});
      if (B.elementType() != targetType) {
        B = cast(graph, B, targetType, cs, {di, layer + "/B"});
      }
      if (scaleB.elementType() != targetType) {
        scaleB = cast(graph, scaleB, targetType, cs, {di, layer + "/scaleB"});
      }
      prog.add(Execute(cs, {di}));
    }
    createVertices(graph, A, std::nullopt, B, scaleB, false, false,
                   ScaledAddSpecialisation::DEFAULT, prog, !regroupBeforeCast,
                   {di}, opts);
  }
}

void scaledAddTo(Graph &graph, Tensor A, Tensor B, float scaleB, Sequence &prog,
                 const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(A, B, scaleB, options));

  const auto opts = parseOptionFlags(options);
  const auto targetType = A.elementType();

  bool regroupBeforeCast =
      shouldRegroupBeforeCast(graph.getTarget(), B.elementType(), targetType);
  if (regroupBeforeCast) {
    B = popops::rearrange::regroupIfBeneficial(graph, B, A, prog,
                                               {di, "scaledAdd/regroupB"});
  }
  if (B.elementType() != targetType &&
      !specialisedVertexExists(A, B, B, false)) {
    B = cast(graph, B, targetType, prog, {di, "scaledAdd/B"});
  }
  auto scaleType = B.elementType();
  if ((A.elementType() == HALF || A.elementType() == FLOAT) &&
      (B.elementType() == HALF || B.elementType() == FLOAT)) {
    // Consider doing arithmetic as float internally to the codelet if scale
    // can't be correctly represented as a half, using this function:
    scaleType = poputil::checkAccuracyWhenCast(graph.getTarget(), scaleB, FLOAT,
                                               HALF, opts.floatToHalfTolerance)
                    ? HALF
                    : FLOAT;
    if (scaleType == HALF &&
        !specialisedVertexExists(A.elementType(), B.elementType(), scaleType,
                                 false)) {
      scaleType = B.elementType();
    }
  }
  const auto scaleBTensor =
      graph.addConstant(scaleType, {}, scaleB, {di, "/scaleB"});
  graph.setTileMapping(scaleBTensor, 0);
  createVertices(graph, A, std::nullopt, B, scaleBTensor, false, false,
                 ScaledAddSpecialisation::DEFAULT, prog, !regroupBeforeCast,
                 {di}, opts);
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
                        Sequence &prog,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(A, B, scaleB, options));

  const auto opts = parseOptionFlags(options);
  const auto targetType = A.elementType();
  const std::string layer = "scaledSub";

  if (A.elementType() == HALF && B.elementType() == HALF &&
      scaleB.elementType() == FLOAT) {
    // The vertex will select float or half scale based on the accuracy of the
    // scale, using the tolerance option
    createVertices(graph, A, std::nullopt, B, scaleB, true, false,
                   ScaledAddSpecialisation::DEFAULT, prog,
                   /* attemptRegroup */ true, {di}, opts);
  } else {
    bool regroupBeforeCast =
        shouldRegroupBeforeCast(graph.getTarget(), B.elementType(), targetType);
    if (regroupBeforeCast) {
      B = popops::rearrange::regroupIfBeneficial(graph, B, A, prog,
                                                 {di, layer + "/regroupB"});
    }
    const auto cs = graph.addComputeSet({di, layer + "/cast"});
    if (B.elementType() != targetType) {
      B = cast(graph, B, targetType, cs, {di, layer + "/B"});
    }
    if (scaleB.elementType() != targetType) {
      scaleB = cast(graph, scaleB, targetType, cs, {di, layer + "/scaleB"});
    }
    prog.add(Execute(cs, {di}));

    createVertices(graph, A, std::nullopt, B, scaleB, true, false,
                   ScaledAddSpecialisation::DEFAULT, prog, !regroupBeforeCast,
                   {di}, opts);
  }
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, float scaleB,
                        Sequence &prog,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  scaledAddTo(graph, A, B, -scaleB, prog, debugContext, options);
}

void scaledAddTo(Graph &graph, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB,
                 Sequence &prog, const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(A, scaleA, B, scaleB, options));

  scaledAritTensorImpl(graph, A, scaleA, B, scaleB, prog, false,
                       ScaledAddSpecialisation::DEFAULT, {di}, options);
}

void scaledAddTo(Graph &graph, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB,
                 Sequence &prog, const ScaledAddSpecialisation speciality,
                 const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(A, scaleA, B, scaleB, speciality, options));

  scaledAritTensorImpl(graph, A, scaleA, B, scaleB, prog, false, speciality,
                       {di}, options);
}

void scaledAddTo(Graph &graph, Tensor A, float scaleA, Tensor B, float scaleB,
                 Sequence &prog, const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(A, B, scaleA, scaleB, options));

  scaledAritConstImpl(graph, A, scaleA, B, scaleB, prog, false,
                      ScaledAddSpecialisation::DEFAULT, {di}, options);
}

void scaledAddTo(Graph &graph, Tensor A, float scaleA, Tensor B, float scaleB,
                 Sequence &prog, const ScaledAddSpecialisation speciality,
                 const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(A, B, scaleA, scaleB, speciality, options));

  scaledAritConstImpl(graph, A, scaleA, B, scaleB, prog, false, speciality,
                      {di}, options);
}

void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A,
                        poplar::Tensor scaleA, poplar::Tensor B,
                        poplar::Tensor scaleB, poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(A, scaleA, B, scaleB, options));

  scaledAritTensorImpl(graph, A, scaleA, B, scaleB, prog, true,
                       ScaledAddSpecialisation::DEFAULT, {di}, options);
}

void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A, float scaleA,
                        poplar::Tensor B, float scaleB,
                        poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(A, B, scaleA, scaleB, options));

  scaledAritConstImpl(graph, A, scaleA, B, scaleB, prog, true,
                      ScaledAddSpecialisation::DEFAULT, {di}, options);
}

} // namespace popops
