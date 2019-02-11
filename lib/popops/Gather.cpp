// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#include "popops/Gather.hpp"

#include "popops/DynamicSlice.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"

#include <algorithm>
#include <iterator>

using namespace poplar;

namespace {

// Transposes the given scatterIndices such that the indexVectorDim becomes
// the most-minor dimension.
Tensor transposeIndexVectorDimToLast(Tensor indices, unsigned indexVectorDim) {
  if (indices.rank() == indexVectorDim) {
    return indices;
  }

  if (indices.rank() == indexVectorDim + 1) {
    return indices;
  }

  std::vector<unsigned> permutation(indices.rank());

  const auto front = std::begin(permutation);
  const auto mid = std::next(std::begin(permutation), indexVectorDim);
  const auto back = std::end(permutation) - 1;

  std::iota(front, mid, 0);
  std::iota(mid, back, indexVectorDim + 1);
  *back = indexVectorDim;

  return indices.dimShuffle(permutation);
}

Tensor canonicalizeGatherIndices(Tensor startIndices, unsigned indexVectorDim) {
  // Transpose the non-index-vector dimensions to the front.
  Tensor startIndicesT =
      transposeIndexVectorDimToLast(startIndices, indexVectorDim);

  const bool indicesAreScalar = startIndicesT.rank() == indexVectorDim;

  // The number of dimensions in scatterIndices that are index dimensions.
  const std::size_t indexDimsInScatterIndices = indicesAreScalar ? 0 : 1;

  // If there is only one index (i.e. scatterIndices has rank 1 and this
  // scatter is really just a dynamic update slice) add a leading degenerate
  // dimension for uniformity.  Otherwise create a "collapsed" leading dimension
  // that subsumes all of the non-index-vector dimensions.
  std::vector<std::size_t> shape = startIndicesT.shape();
  if (shape.size() == indexDimsInScatterIndices) {
    shape.insert(shape.begin(), 1);
    return startIndicesT.reshape(shape);
  }

  if (indicesAreScalar) {
    return startIndicesT.reshape({startIndicesT.numElements()});
  }

  // Collapse all but the dimensions (0 or 1) in scatterIndices containing
  // the index vectors.
  std::vector<std::size_t> newShape = {
      startIndicesT.numElements() / shape.back(), shape.back()};
  return startIndicesT.reshape(newShape);
}

poplar::Tensor
adjustBatchDimsInAccumulator(std::vector<std::size_t> startIndicesShape,
                             const poplar::Tensor &accumulator,
                             std::size_t indexVectorDim) {
  std::vector<std::size_t> bounds;

  if (indexVectorDim < startIndicesShape.size()) {
    bounds.resize(startIndicesShape.size() - 1);
  } else {
    bounds.resize(startIndicesShape.size());
  }

  const auto indicesShape = [&startIndicesShape](std::size_t dim) {
    return startIndicesShape[dim];
  };

  const auto begin = std::begin(bounds);
  const auto mid = begin + indexVectorDim;
  const auto end = std::end(bounds);

  std::iota(begin, mid, 0);
  std::iota(mid, end, indexVectorDim + 1);

  std::transform(begin, end, begin, indicesShape);

  if (bounds.empty()) {
    return accumulator.squeeze({0});
  } else {
    const auto shape = accumulator.shape();
    bounds.insert(std::end(bounds), std::begin(shape) + 1, std::end(shape));

    return accumulator.reshape(bounds);
  }
}

poplar::Tensor expandIndexVectorIntoOperandSpace(
    poplar::Graph &graph, poplar::Tensor indices,
    std::vector<unsigned> scatterDimsToOperandDims, std::size_t rank) {
  poplar::Tensor zero = graph.addConstant(indices.elementType(), {1}, 0);
  graph.setTileMapping(zero, 0);
  std::vector<poplar::Tensor> expandedIndexComponents;

  for (unsigned i = 0; i < rank; ++i) {
    auto indexVectorDimItr = std::find(std::begin(scatterDimsToOperandDims),
                                       std::end(scatterDimsToOperandDims), i);

    auto indexVectorDimIndex =
        std::distance(std::begin(scatterDimsToOperandDims), indexVectorDimItr);

    if (std::end(scatterDimsToOperandDims) != indexVectorDimItr) {
      poplar::Tensor component =
          indices.slice(indexVectorDimIndex, indexVectorDimIndex + 1);
      expandedIndexComponents.push_back(component);
    } else {
      expandedIndexComponents.push_back(zero);
    }
  }

  return poplar::concat(expandedIndexComponents);
}

std::size_t scatterLoopTripCount(std::vector<std::size_t> indicesShape,
                                 unsigned indexVectorDim) {
  if (indexVectorDim < indicesShape.size()) {
    return std::accumulate(std::begin(indicesShape), std::end(indicesShape), 1,
                           std::multiplies<std::size_t>()) /
           indicesShape[indexVectorDim];
  } else {
    return std::accumulate(std::begin(indicesShape), std::end(indicesShape), 1,
                           std::multiplies<std::size_t>());
  }
}

poplar::Tensor createGatherLoopAccumulatorInitValue(
    poplar::Graph &graph, poplar::Type type,
    std::vector<std::size_t> sliceSizes,
    std::vector<std::size_t> collapsedSliceDims,
    std::size_t gatherLoopTripCount, poplar::program::Sequence &prog) {
  std::vector<std::size_t> shape(1 + sliceSizes.size());

  const auto isSliceDim = [&collapsedSliceDims](std::size_t dim) {
    return !std::binary_search(std::begin(collapsedSliceDims),
                               std::end(collapsedSliceDims), dim);
  };

  const auto sliceSize = [&sliceSizes](std::size_t dim) {
    return sliceSizes[dim];
  };

  auto begin = std::begin(shape) + 1;
  auto end = std::end(shape);

  shape.front() = gatherLoopTripCount;

  std::iota(begin, end, 0);
  auto itr = std::stable_partition(begin, end, isSliceDim);
  std::transform(begin, itr, begin, sliceSize);

  shape.erase(itr, shape.end());

  poplar::Tensor result = graph.addVariable(type, shape);
  poputil::mapTensorLinearly(graph, result);

  return result;
}

poplar::Tensor permuteBatchAndOffsetDims(poplar::Tensor accumulator,
                                         std::vector<std::size_t> offsetDims,
                                         std::size_t outputRank) {
  std::vector<unsigned> permutation(outputRank);

  const auto isOffsetDim = [&](unsigned dim) {
    return !std::binary_search(std::begin(offsetDims), std::end(offsetDims),
                               dim);
  };

  std::iota(std::begin(permutation), std::end(permutation), 0);
  std::stable_partition(std::begin(permutation), std::end(permutation),
                        isOffsetDim);

  return accumulator.dimShuffle(permutation);
}

poplar::Tensor padVectorWithZeros(poplar::Graph &graph, poplar::Tensor t,
                                  std::size_t prepend = 0,
                                  std::size_t append = 0) {
  poplar::Tensor prefix = graph.addConstant(t.elementType(), {prepend}, 0);
  poplar::Tensor suffix = graph.addConstant(t.elementType(), {append}, 0);
  graph.setTileMapping(prefix, 0);
  graph.setTileMapping(suffix, 0);
  return poplar::concat({prefix, t, suffix});
}

namespace tf_compat {
poplar::Tensor dynamicSlice(poplar::Graph &graph, poplar::Tensor input,
                            poplar::Tensor indices,
                            std::vector<std::size_t> sliceSizes,
                            poplar::program::Sequence &prog) {
  auto type = indices.elementType();
  if (type == poplar::INT) {
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  std::vector<std::size_t> sliceDims;
  std::vector<std::size_t> newSliceSizes;
  poplar::Tensor sliceIndices;
  for (unsigned d = 0; d < sliceSizes.size(); d++) {
    auto t = indices.index({d}).reshape({1});
    bool sameShape = sliceSizes[d] == input.shape()[d];
    unsigned int index;
    bool zeroIndex = t.getConstantValue(&index) && (index == 0);

    if (!(sameShape && zeroIndex)) {
      if (sliceDims.size() == 0) {
        sliceIndices = t;
      } else {
        sliceIndices = poplar::concat(sliceIndices, t, 0);
      }
      sliceDims.push_back(d);
      newSliceSizes.push_back(sliceSizes[d]);
    }
  }

  // Add the dynamic slice operations to `prog`. This automatically
  // creates the required compute set.
  poplar::Tensor out;

  if (sliceDims.size() > 0) {
    out = popops::dynamicSlice(graph, input, sliceIndices, sliceDims,
                               newSliceSizes, prog);
  } else {
    poplar::Tensor copy = graph.clone(input);
    prog.add(poplar::program::Copy(input, copy));
    out = copy;
  }

  return out;
}

poplar::Tensor dynamicUpdateSlice(poplar::Graph &graph, poplar::Tensor input,
                                  poplar::Tensor update, poplar::Tensor indices,
                                  std::vector<std::size_t> sliceSizes,
                                  poplar::program::Sequence &prog) {

  auto type = indices.elementType();
  if (type == poplar::INT) {
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  std::vector<std::size_t> sliceDims;
  std::vector<std::size_t> newSliceSizes;
  poplar::Tensor sliceIndices;
  for (unsigned d = 0; d < sliceSizes.size(); d++) {
    auto t = indices.index({d}).reshape({1});
    bool sameShape = sliceSizes[d] == update.shape()[d];
    unsigned int index;
    bool zeroIndex = t.getConstantValue(&index) && (index == 0);

    if (!(sameShape && zeroIndex)) {
      if (sliceDims.size() == 0) {
        sliceIndices = t;
      } else {
        sliceIndices = poplar::concat(sliceIndices, t);
      }
      sliceDims.push_back(d);
      newSliceSizes.push_back(update.shape()[d]);
    }
  }

  if (sliceDims.size() > 0) {
    popops::dynamicUpdate(graph, input, update, sliceIndices, sliceDims,
                          newSliceSizes, prog);
  } else {
    prog.add(poplar::program::Copy(update, input));
  }

  return input;
}
} // namespace tf_compat

template <typename BodyType>
poplar::program::Sequence countedLoop(poplar::Graph &graph, std::size_t count,
                                      BodyType body) {
  poplar::program::Sequence prog;
  poplar::Tensor inductionVar = graph.addVariable(poplar::INT, {});
  poputil::mapTensorLinearly(graph, inductionVar);
  prog.add(poplar::program::Assign(inductionVar, 0));

  inductionVar = inductionVar.reshape({1});
  poplar::program::Sequence bodyProg = body(inductionVar);
  auto one = graph.addConstant(poplar::INT, {1}, 1);
  graph.setTileMapping(one, 0);
  popops::addInPlace(graph, inductionVar, one, bodyProg);

  prog.add(poplar::program::Repeat(count, bodyProg));

  return prog;
}

} // namespace

namespace popops {

poplar::Tensor gather(poplar::Graph &graph, const poplar::Tensor &operand,
                      const poplar::Tensor &indices, std::size_t indexVectorDim,
                      std::vector<std::size_t> offsetDims,
                      std::vector<std::size_t> sliceSizes,
                      std::vector<std::size_t> collapsedSliceDims,
                      std::vector<unsigned> startIndexMap,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix) {
  std::size_t gatherLoopTripCount =
      scatterLoopTripCount(indices.shape(), indexVectorDim);

  poplar::Tensor canonicalStartIndices =
      canonicalizeGatherIndices(indices, indexVectorDim);

  poplar::Tensor accumulator = createGatherLoopAccumulatorInitValue(
      graph, operand.elementType(), sliceSizes, collapsedSliceDims,
      gatherLoopTripCount, prog);

  const bool hasScalarIndices = canonicalStartIndices.rank() == 1;

  // for (int i = 0; i < canonicalStartIndices; ++i)
  prog.add(countedLoop(graph, gatherLoopTripCount, [&](poplar::Tensor i) {
    poplar::program::Sequence prog;

    poplar::Tensor indexVector;
    if (hasScalarIndices) {
      indexVector =
          tf_compat::dynamicSlice(graph, canonicalStartIndices, i, {1}, prog);
    } else {
      indexVector = padVectorWithZeros(graph, i, 0, 1);
      indexVector =
          tf_compat::dynamicSlice(graph, canonicalStartIndices, indexVector,
                                  {1, canonicalStartIndices.dim(1)}, prog);
      indexVector = indexVector.squeeze({0});
    }

    poplar::Tensor gatheredSliceStart = expandIndexVectorIntoOperandSpace(
        graph, indexVector, startIndexMap, operand.rank());

    poplar::Tensor gatheredSlice = tf_compat::dynamicSlice(
        graph, operand, gatheredSliceStart, sliceSizes, prog);

    poplar::Tensor gatheredSliceWithDimsCollapsed =
        gatheredSlice.squeeze(collapsedSliceDims);

    poplar::Tensor gatheredSliceForUpdate =
        gatheredSliceWithDimsCollapsed.expand({0});

    poplar::Tensor indexVectorIntoAccum =
        padVectorWithZeros(graph, i, 0, gatheredSliceWithDimsCollapsed.rank());

    tf_compat::dynamicUpdateSlice(graph, accumulator, gatheredSliceForUpdate,
                                  indexVectorIntoAccum,
                                  gatheredSliceForUpdate.shape(), prog);

    return prog;
  }));

  poplar::Tensor accumulatorDecanonicalized = adjustBatchDimsInAccumulator(
      indices.shape(), accumulator, indexVectorDim);

  const auto outputRank = (indices.rank() == indexVectorDim ? 0 : -1) +
                          offsetDims.size() + indices.rank();
  return permuteBatchAndOffsetDims(accumulatorDecanonicalized, offsetDims,
                                   outputRank);
}

} // namespace popops
