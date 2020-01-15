// Copyright (c) Graphcore Ltd, All rights reserved.
#include "popops/ScaledAdd.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional.hpp>

#include "poplar/Program.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

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
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    const auto &regions = intervals[i];
    const unsigned regionElements = std::accumulate(
        regions.begin(), regions.end(), 0,
        [](std::size_t total, const Interval &i) { return total + i.size(); });
    if (regionElements > maxRegionSize) {
      return false;
    }
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

ComputeSet scaledArithmeticConstImpl(Graph &graph, Tensor A, float scaleA,
                                     Tensor B, float scaleB, Type scaleType,
                                     const ScaledAddSpecialisation speciality,
                                     const std::string &debugPrefix,
                                     const poplar::OptionFlags &options) {
  auto opts = parseOptionFlags(options);

  const auto addConstraints =
      (A.elementType() == HALF || A.elementType() == FLOAT) &&
      opts.optimizeForSpeed;
  if (!A.isParallelWriteable())
    throw poputil::poplibs_error("Trying to accumulate to tensor that cannot be"
                                 " written in parallel");
  const auto &target = graph.getTarget();
  const auto dataType = A.elementType();
  const auto deltaType = B.elementType();

  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");
  const auto vectorWidth = target.getVectorWidth(dataType);
  const auto numWorkers = target.getNumWorkerContexts();

  std::string codeletName2D;
  std::string codeletNameSupervisor;
  if (speciality == ScaledAddSpecialisation::X_MINUS_AX_PLUS_BY) {
    codeletName2D = templateVertex("popops::XMinusaXPlusbY2D", dataType, true,
                                   addConstraints);
    codeletNameSupervisor = templateVertex("popops::XMinusaXPlusbYSupervisor",
                                           dataType, true, addConstraints);
  } else if (scaleA != 1.0f) {
    codeletName2D =
        templateVertex("popops::aXPlusbY2D", dataType, true, addConstraints);
    codeletNameSupervisor = templateVertex("popops::aXPlusbYSupervisor",
                                           dataType, true, addConstraints);
  } else {
    codeletName2D = templateVertex("popops::ScaledAdd2D", dataType, deltaType,
                                   scaleType, true, addConstraints);
    codeletNameSupervisor =
        templateVertex("popops::ScaledAddSupervisor", dataType, deltaType,
                       scaleType, true, addConstraints);
  }
  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max2DInnerElements =
      std::min<std::size_t>(graph.getMaxFieldDim(codeletName2D, "A", 1),
                            target.getRptCountMax() * vectorWidth);

  const auto maxSupervisorElements = std::min<std::size_t>(
      graph.getMaxVertexFieldValue(codeletNameSupervisor, "size"),
      target.getRptCountMax() * vectorWidth * numWorkers);

  auto aFlat = A.flatten();
  auto bFlat = B.flatten();
  graph.reorderToSimplify(&aFlat, {&bFlat});
  const auto mapping = graph.getTileMapping(aFlat);

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dataType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);

    if (tileContiguousRegions.size() == 1 &&
        tileContiguousRegions[0].size() == 1 &&
        validateRegionSizeForSupervisorVertex(tileContiguousRegions,
                                              maxSupervisorElements)) {
      const auto &region = tileContiguousRegions[0][0];
      auto v = graph.addVertex(
          cs, codeletNameSupervisor,
          {{"A", aFlat.slice(region)}, {"B", bFlat.slice(region)}});
      graph.setTileMapping(v, tile);
      graph.setInitialValue(v["size"], aFlat.slice(region).numElements());
      if (scaleA == 1.0f) {
        graph.setInitialValue(v["scaleB"], scaleB);
      } else {
        graph.setInitialValue(v["scaleA"], scaleA);
        graph.setInitialValue(v["scaleB"], scaleB);
      }
    } else {
      auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, grainSize, 2 * grainSize, UINT32_MAX,
          max2DInnerElements);

      for (const auto &regions : vertexRegions) {
        auto v = graph.addVertex(
            cs, codeletName2D,
            {{"A", aFlat.slices(regions)}, {"B", bFlat.slices(regions)}});

        graph.setTileMapping(v, tile);
        if (scaleA == 1.0f) {
          graph.setInitialValue(v["scaleB"], scaleB);
        } else {
          graph.setInitialValue(v["scaleA"], scaleA);
          graph.setInitialValue(v["scaleB"], scaleB);
        }
      }
    }
  }
  return cs;
}

ComputeSet scaledArithmeticTensorImpl(
    Graph &graph, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB,
    boost::optional<float> tolerance, const bool doSubtract,
    const bool doaXPlusbY, const ScaledAddSpecialisation speciality,
    const std::string &debugPrefix, const poplar::OptionFlags &options) {
  auto opts = parseOptionFlags(options);
  const auto addConstraints =
      (A.elementType() == HALF || A.elementType() == FLOAT) &&
      !(A.elementType() == FLOAT && B.elementType() == HALF) &&
      opts.optimizeForSpeed;
  if (!A.isParallelWriteable())
    throw poputil::poplibs_error("Trying to accumulate to tensor that cannot be"
                                 " written in parallel");
  if (A.shape() != B.shape())
    throw poputil::poplibs_error("Input Tensors for scaled arithmetic must"
                                 " have the same shape");
  if (scaleA.elementType() != scaleB.elementType())
    throw poputil::poplibs_error("Scale factors must be of the same type");

  if (speciality == ScaledAddSpecialisation::X_MINUS_AX_PLUS_BY) {
    if (!doaXPlusbY)
      throw poputil::poplibs_error(
          "Scaled add X-aX+bY is only supported together "
          "with doaXPlusbY option");
    if (doSubtract)
      throw poputil::poplibs_error("Subtraction not supported with X-aX+bY");
  }

  const auto &target = graph.getTarget();
  const auto dataType = A.elementType();
  const auto deltaType = B.elementType();
  const auto scaleType = scaleA.elementType();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");
  const auto vectorWidth = target.getVectorWidth(dataType);
  const auto numWorkers = target.getNumWorkerContexts();

  std::string codeletName2D;
  std::string codeletNameSupervisor;
  if (doSubtract && doaXPlusbY) {
    codeletName2D =
        templateVertex("popops::aXMinusbY2D", dataType, false, addConstraints);
    codeletNameSupervisor = templateVertex("popops::aXMinusbYSupervisor",
                                           dataType, false, addConstraints);
  } else if (doSubtract && !doaXPlusbY) {
    codeletName2D =
        templateVertex("popops::ScaledSubtract2D", dataType, addConstraints);
    codeletNameSupervisor = templateVertex("popops::ScaledSubtractSupervisor",
                                           dataType, deltaType, addConstraints);
  } else if (!doSubtract && doaXPlusbY) {
    if (speciality == ScaledAddSpecialisation::X_MINUS_AX_PLUS_BY) {
      codeletName2D = templateVertex("popops::XMinusaXPlusbY2D", dataType,
                                     false, addConstraints);
      codeletNameSupervisor = templateVertex("popops::XMinusaXPlusbYSupervisor",
                                             dataType, false, addConstraints);
    } else {
      codeletName2D =
          templateVertex("popops::aXPlusbY2D", dataType, false, addConstraints);
      codeletNameSupervisor = templateVertex("popops::aXPlusbYSupervisor",
                                             dataType, false, addConstraints);
    }
  } else if (!doSubtract && !doaXPlusbY) {
    codeletName2D = templateVertex("popops::ScaledAdd2D", dataType, deltaType,
                                   scaleType, false, addConstraints);
    codeletNameSupervisor =
        templateVertex("popops::ScaledAddSupervisor", dataType, deltaType,
                       scaleType, false, addConstraints);
  }

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max2DInnerElements =
      std::min<std::size_t>(graph.getMaxFieldDim(codeletName2D, "A", 1),
                            target.getRptCountMax() * vectorWidth);

  const auto codeletNameSupervisorForSizingOnly =
      templateVertex("popops::ScaledAddSupervisor", dataType, deltaType,
                     scaleType, true, addConstraints);

  const auto maxSupervisorElements = std::min<std::size_t>(
      graph.getMaxVertexFieldValue(codeletNameSupervisorForSizingOnly, "size"),
      target.getRptCountMax() * vectorWidth * numWorkers);

  auto aFlat = A.flatten();
  auto bFlat = B.flatten();
  graph.reorderToSimplify(&aFlat, {&bFlat});
  const auto mapping = graph.getTileMapping(aFlat);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dataType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    graph.setTileMapping(scaleB, tile);

    if (tileContiguousRegions.size() == 1 &&
        tileContiguousRegions[0].size() == 1 &&
        validateRegionSizeForSupervisorVertex(tileContiguousRegions,
                                              maxSupervisorElements)) {
      const auto &region = tileContiguousRegions[0][0];

      VertexRef v = graph.addVertex(cs, codeletNameSupervisor,
                                    {{"A", aFlat.slice(region)},
                                     {"B", bFlat.slice(region)},
                                     {"scaleB", scaleB.reshape({1})}});
      if (doaXPlusbY) {
        graph.connect(v["scaleA"], scaleA.reshape({1}));
      }

      graph.setInitialValue(v["size"], aFlat.slice(region).numElements());
      if (tolerance) {
        graph.setInitialValue(v["tolerance"], tolerance.get());
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
          graph.connect(v["scaleA"], scaleA);
        }

        if (tolerance) {
          graph.setInitialValue(v["tolerance"], tolerance.get());
        }
        graph.setTileMapping(v, tile);
      }
    }
  }
  return cs;
}

// Add a compute set to a graph if it is not already added
ComputeSet addOrGetCs(Graph &graph, boost::optional<ComputeSet> &cs,
                      const std::string &prefix) {
  if (!cs.is_initialized()) {
    cs = graph.addComputeSet(prefix + "/cast");
  }
  return *cs;
}

void scaledAritTensorImpl(Graph &graph, Tensor A, Tensor scaleA, Tensor B,
                          Tensor scaleB, Sequence &prog, bool subtract,
                          const ScaledAddSpecialisation speciality,
                          const std::string &debugPrefix,
                          const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const std::string debugPrefixSubAdd =
      subtract ? "/scaledSubtract" : "/scaledAdd";
  const auto fnPrefix = debugPrefix + debugPrefixSubAdd;
  bool axpby = true;
  if (scaleA.elementType() != targetType) {
    scaleA = cast(graph, scaleA, targetType, prog, fnPrefix + "/scaleA");
  }
  // We don't support float axpby vertex. Synthesize using mul and scaledAdd
  if (A.elementType() == FLOAT) {
    mulInPlace(graph, A, scaleA, prog, fnPrefix);
    axpby = false;
  }

  boost::optional<ComputeSet> cs;
  if (targetType != B.elementType()) {
    B = cast(graph, B, targetType, addOrGetCs(graph, cs, fnPrefix),
             fnPrefix + "/B");
  }
  if (scaleB.elementType() != targetType) {
    scaleB = cast(graph, scaleB, targetType, addOrGetCs(graph, cs, fnPrefix),
                  fnPrefix + "/scaleB");
  }
  if (cs.is_initialized()) {
    prog.add(Execute(*cs));
  }
  auto cs1 = scaledArithmeticTensorImpl(graph, A, scaleA, B, scaleB,
                                        boost::none, subtract, axpby,
                                        speciality, debugPrefix, options);
  prog.add(Execute(cs1));
}

void scaledAritConstImpl(Graph &graph, Tensor A, float scaleA, Tensor B,
                         float scaleB, Sequence &prog, bool subtract,
                         const ScaledAddSpecialisation speciality,
                         const std::string &debugPrefix,
                         const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const std::string debugPrefixSubAdd =
      subtract ? "/scaledSubtract" : "/scaledAdd";
  const auto fnPrefix = debugPrefix + debugPrefixSubAdd;

  // we do not support float axpby. Synthesize using mul and scaledAdd
  if (A.elementType() == FLOAT && scaleA != 1.0f) {
    mulInPlace(graph, A, scaleA, prog, fnPrefix);
    scaleA = 1.0f;
  }
  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, prog, fnPrefix + "/B");
  }
  if (subtract) {
    scaleB = -scaleB;
  }
  auto cs = scaledArithmeticConstImpl(graph, A, scaleA, B, scaleB, targetType,
                                      speciality, debugPrefix, options);
  prog.add(Execute(cs));
}

bool specialisedVertexExists(const Tensor &A, const Tensor &B,
                             const Tensor &scaleB) {
  return A.elementType() == FLOAT && B.elementType() == HALF &&
         (scaleB.elementType() == HALF || scaleB.elementType() == FLOAT);
}

} // namespace
void scaledAddTo(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
                 Sequence &prog, const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const auto castPrefix = debugPrefix + "/scaledAdd";

  if (A.elementType() == HALF && B.elementType() == HALF &&
      scaleB.elementType() == FLOAT) {

    auto opts = parseOptionFlags(options);
    // The vertex will select float or half scale based on the accuracy of the
    // scale, using the tolerance option
    auto cs = scaledArithmeticTensorImpl(
        graph, A, scaleB, B, scaleB, opts.floatToHalfTolerance, false, false,
        ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
    prog.add(Execute(cs));

  } else {
    if (!specialisedVertexExists(A, B, scaleB)) {
      boost::optional<ComputeSet> cs;
      if (B.elementType() != targetType) {
        B = cast(graph, B, targetType, addOrGetCs(graph, cs, castPrefix),
                 castPrefix + "/B");
      }
      if (scaleB.elementType() != targetType) {
        scaleB =
            cast(graph, scaleB, targetType, addOrGetCs(graph, cs, castPrefix),
                 castPrefix + "/scaleB");
      }
      if (cs.is_initialized()) {
        prog.add(Execute(*cs));
      }
    }
    auto cs1 = scaledArithmeticTensorImpl(
        graph, A, scaleB, B, scaleB, boost::none, false, false,
        ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
    prog.add(Execute(cs1));
  }
}

void scaledAddTo(Graph &graph, Tensor A, Tensor B, float scaleB, Sequence &prog,
                 const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {
  auto opts = parseOptionFlags(options);
  const auto targetType = A.elementType();

  if (B.elementType() != targetType && !specialisedVertexExists(A, B, B)) {
    B = cast(graph, B, targetType, prog, debugPrefix + "/scaledAdd/B");
  }
  bool useFloatScale = false;
  auto scaleType =
      specialisedVertexExists(A, B, B) ? B.elementType() : targetType;
  if ((A.elementType() == HALF || A.elementType() == FLOAT) &&
      B.elementType() == HALF) {
    // Consider doing arithmetic as float internally to the codelet if scale
    // can't be correctly represented as a half, using this function:
    useFloatScale = !(poputil::checkAccuracyWhenCast(
        graph.getTarget(), scaleB, FLOAT, HALF, opts.floatToHalfTolerance));
  }
  auto cs = scaledArithmeticConstImpl(
      graph, A, 1.0, B, scaleB, useFloatScale ? FLOAT : scaleType,
      ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
  prog.add(Execute(cs));
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
                        Sequence &prog, const std::string &debugPrefix,
                        const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const auto castPrefix = debugPrefix + "/scaledSub";
  boost::optional<ComputeSet> cs;

  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, addOrGetCs(graph, cs, castPrefix),
             castPrefix + "/B");
  }
  if (scaleB.elementType() != targetType) {
    scaleB = cast(graph, scaleB, targetType, addOrGetCs(graph, cs, castPrefix),
                  castPrefix + "/scaleB");
  }
  if (cs.is_initialized()) {
    prog.add(Execute(*cs));
  }
  auto cs1 = scaledArithmeticTensorImpl(
      graph, A, scaleB, B, scaleB, boost::none, true, false,
      ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
  prog.add(Execute(cs1));
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, float scaleB,
                        Sequence &prog, const std::string &debugPrefix,
                        const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, prog, debugPrefix + "/ScaledSub/B");
  }
  auto cs = scaledArithmeticConstImpl(graph, A, 1.0, B, -scaleB, targetType,
                                      ScaledAddSpecialisation::DEFAULT,
                                      debugPrefix, options);
  prog.add(Execute(cs));
}

void scaledAddTo(Graph &graph, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB,
                 Sequence &prog, const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {

  scaledAritTensorImpl(graph, A, scaleA, B, scaleB, prog, false,
                       ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
}

void scaledAddTo(Graph &graph, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB,
                 Sequence &prog, const ScaledAddSpecialisation speciality,
                 const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {

  scaledAritTensorImpl(graph, A, scaleA, B, scaleB, prog, false, speciality,
                       debugPrefix, options);
}

void scaledAddTo(Graph &graph, Tensor A, float scaleA, Tensor B, float scaleB,
                 Sequence &prog, const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {

  scaledAritConstImpl(graph, A, scaleA, B, scaleB, prog, false,
                      ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
}

void scaledAddTo(Graph &graph, Tensor A, float scaleA, Tensor B, float scaleB,
                 Sequence &prog, const ScaledAddSpecialisation speciality,
                 const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {

  scaledAritConstImpl(graph, A, scaleA, B, scaleB, prog, false, speciality,
                      debugPrefix, options);
}

void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A,
                        poplar::Tensor scaleA, poplar::Tensor B,
                        poplar::Tensor scaleB, poplar::program::Sequence &prog,
                        const std::string &debugPrefix,
                        const poplar::OptionFlags &options) {

  scaledAritTensorImpl(graph, A, scaleA, B, scaleB, prog, true,
                       ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
}

void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A, float scaleA,
                        poplar::Tensor B, float scaleB,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix,
                        const poplar::OptionFlags &options) {

  scaledAritConstImpl(graph, A, scaleA, B, scaleB, prog, true,
                      ScaledAddSpecialisation::DEFAULT, debugPrefix, options);
}

} // namespace popops
