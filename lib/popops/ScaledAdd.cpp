#include "popops/ElementWise.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "ExprOpUtil.hpp"
#include "popops/ScaledAdd.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops::expr;

namespace popops {

void scaledArithmeticConstImpl(Graph &graph, Tensor A, float scaleA, Tensor B,
           float scaleB,
           Sequence &prog, const std::string &debugPrefix) {

   if (!A.isParallelWriteable())
    throw poputil::poplibs_error("Trying to accumulate to tensor that cannot be"
                                 " written in parallel");
  if (A.numElements() != B.numElements())
    throw poputil::poplibs_error("Input Tensors for scaled arithmetic must"
                                 " have the same number of elements");
  const auto &target = graph.getTarget();
  const auto dType = A.elementType();
  const auto dataBType = B.elementType();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");
  const auto vectorWidth = target.getVectorWidth(dType);

  const auto codeletName2D = scaleA != 1.0f ?
                        templateVertex("popops::aXPlusbY2D", dType, true) :
                        templateVertex("popops::ScaledAdd2D", dType, true);

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max2DInnerElements = std::min<std::size_t>(
    graph.getMaxFieldDim(codeletName2D, "A", 1),
    target.getRptCountMax() * vectorWidth);

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
    const auto grainSize = target.getVectorWidth(dType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    if (tileContiguousRegions.size() == 1 &&
        tileContiguousRegions[0].size() == 1) {
      const auto &region = tileContiguousRegions[0][0];
      auto v = scaleA == 1.0f ?
            graph.addVertex(cs, templateVertex("popops::ScaledAddSupervisor",
                                              dType, dataBType, true),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)}}) :
            graph.addVertex(cs, templateVertex("popops::aXPlusbYSupervisor",
                                              dType, true),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)}});
      graph.setTileMapping(v, tile);
      if(scaleA == 1.0f) {
        graph.setInitialValue(v["scaleB"], scaleB);
      }
      else {
        graph.setInitialValue(v["scaleA"], scaleA);
        graph.setInitialValue(v["scaleB"], scaleB);
      }
    } else {
      auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize,
                                   max2DInnerElements);
      for (const auto &regions : vertexRegions) {
        auto v = graph.addVertex(cs, codeletName2D,
                                 {{"A", aFlat.slices(regions)},
                                  {"B", bFlat.slices(regions)}});

        graph.setTileMapping(v, tile);
        if(scaleA == 1.0f) {
          graph.setInitialValue(v["scaleB"], scaleB);
        }
        else {
          graph.setInitialValue(v["scaleA"], scaleA);
          graph.setInitialValue(v["scaleB"], scaleB);
        }
      }
    }
  }
  prog.add(Execute(cs));
}


void scaledArithmeticTensorImpl(Graph &graph, Tensor A, Tensor scaleA_,
                                Tensor B,
                                Tensor scaleB_,
                                const bool doSubtract,
                                const bool doaXPlusbY,
                                Sequence &prog,
                                const std::string &debugPrefix) {
  if (!A.isParallelWriteable())
    throw poputil::poplibs_error("Trying to accumulate to tensor that cannot be"
                                 " written in parallel");
  if (A.numElements() != B.numElements())
    throw poputil::poplibs_error("Input Tensors for scaled arithmetic must"
                                 " have the same number of elements");
  if (scaleA_.numElements() != 1)
    throw poputil::poplibs_error("The scaleA Tensor for scaled arithmetic must"
                                 " have a single element");
  if (scaleB_.numElements() != 1)
    throw poputil::poplibs_error("The scaleB Tensor for scaled arithmetic must"
                                 " have a single element");
  const auto &target = graph.getTarget();
  const auto dType = A.elementType();
  const auto dataBType = B.elementType();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");
  const auto vectorWidth = target.getVectorWidth(dType);

  const auto codeletName2D = doSubtract ?
          templateVertex("popops::ScaledSubtract2D", dType) :
          templateVertex("popops::ScaledAdd2D", dType, false);

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max2DInnerElements = std::min<std::size_t>(
    graph.getMaxFieldDim(codeletName2D, "A", 1),
    target.getRptCountMax() * vectorWidth);

  auto aFlat = A.flatten();
  auto bFlat = B.flatten();
  auto scaleA = scaleA_.reshape({});
  auto scaleB = scaleB_.reshape({});
  graph.reorderToSimplify(&aFlat, {&bFlat});
  const auto mapping = graph.getTileMapping(aFlat);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
   // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    graph.setTileMapping(scaleB, tile);

    if (tileContiguousRegions.size() == 1 &&
        tileContiguousRegions[0].size() == 1) {
      const auto &region = tileContiguousRegions[0][0];

      const auto v = doSubtract ?
          graph.addVertex(cs, templateVertex("popops::ScaledSubtractSupervisor",
                                              dType, dataBType),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)},
                                              {"scaleB", scaleB}}) :
          doaXPlusbY ?
          graph.addVertex(cs, templateVertex("popops::aXPlusbYSupervisor",
                                              dType, false),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)},
                                              {"scaleA", scaleA},
                                              {"scaleB", scaleB}}) :
          graph.addVertex(cs, templateVertex("popops::ScaledAddSupervisor",
                                              dType, dataBType, false),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)},
                                              {"scaleB", scaleB}});
      graph.setTileMapping(v, tile);
    } else {
      auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize,
                                   max2DInnerElements);
      for (const auto &regions : vertexRegions) {

        auto v = doaXPlusbY ?
              graph.addVertex(cs, templateVertex("popops::aXPlusbY2D",
                                  dType, false),
                                 {{"A", aFlat.slices(regions)},
                                  {"B", bFlat.slices(regions)},
                                  {"scaleA", scaleA},
                                  {"scaleB", scaleB}}) :
              graph.addVertex(cs, codeletName2D,
                                 {{"A", aFlat.slices(regions)},
                                  {"B", bFlat.slices(regions)},
                                  {"scaleB", scaleB}});

        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(cs));
}
////////////////////////////////////////////////////////////////////////////////
// The following functions produce expressions which the mapInPlace
// function will optimise into addInPlace operations.  Vertices are then
// created by the scaledArithmeticTensorImpl function above.
////////////////////////////////////////////////////////////////////////////////


// TODO T8195- The cases below which accepts floats should call mapInPlace.
// However as mapInPlace doedn't support a float being passed through to the
// addInPlace vertex, that would be inefficient. A constant tensor would be
// generated instead.. So for now the float version of addinPlace below
// calls scaledArithmeticConstImpl directly instead.
void scaledAddTo(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
           Sequence &prog, const std::string &debugPrefix) {
  if (A.elementType() != B.elementType()) {
    scaledArithmeticTensorImpl(graph, A, scaleB, B, scaleB, false, false,
                                                           prog, debugPrefix);
  }
  else {
    mapInPlace(graph, Add(_1, Mul(_2, _3)), {A, B, scaleB}, prog, debugPrefix);
  }
}

void scaledAddTo(Graph &graph, Tensor A, Tensor B, float scaleB,
           Sequence &prog, const std::string &debugPrefix) {
    scaledArithmeticConstImpl(graph, A, 1.0, B, scaleB, prog, debugPrefix);
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
                        Sequence &prog, const std::string &debugPrefix) {
  mapInPlace(graph, Sub(_1, Mul(_2, _3)), {A, B, scaleB}, prog, debugPrefix);
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, float scaleB,
                        Sequence &prog, const std::string &debugPrefix) {
  scaledArithmeticConstImpl(graph, A, 1.0, B, -scaleB, prog, debugPrefix);
}

void scaledAddTo(Graph &graph, Tensor A, Tensor scaleA,
           Tensor B, Tensor scaleB,
           Sequence &prog, const std::string &debugPrefix) {
  mapInPlace(graph, Add(Mul(_1, _2), Mul(_3, _4)), {A, scaleA, B, scaleB},
                                                            prog, debugPrefix);
}
void scaledAddTo(Graph &graph, Tensor A, float scaleA,
           Tensor B,  float scaleB,
           Sequence &prog, const std::string &debugPrefix) {
  scaledArithmeticConstImpl(graph, A, scaleA, B, scaleB, prog, debugPrefix);
}


}
