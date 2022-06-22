// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/Scatter.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include "popops/DynamicSlice.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Loop.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"

#include <algorithm>
#include <boost/optional.hpp>
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

// Canonicalizes the scatterIndices tensor in order to keep them uniform while
// performing the scatter operation.
Tensor canonicalizeScatterIndices(Tensor scatterIndices,
                                  unsigned indexVectorDim) {
  // Transpose the non-index-vector dimensions to the front.
  Tensor scatterIndicesT =
      transposeIndexVectorDimToLast(scatterIndices, indexVectorDim);

  const bool indicesAreScalar = scatterIndicesT.rank() == indexVectorDim;

  // The number of dimensions in scatterIndices that are index dimensions.
  const std::size_t indexDimsInScatterIndices = indicesAreScalar ? 0 : 1;

  // If there is only one index (i.e. scatterIndices has rank 1 and this
  // scatter is really just a dynamic update slice) add a leading degenerate
  // dimension for uniformity.  Otherwise create a "collapsed" leading dimension
  // that subsumes all of the non-index-vector dimensions.
  std::vector<std::size_t> shape = scatterIndicesT.shape();
  if (shape.size() == indexDimsInScatterIndices) {
    shape.insert(shape.begin(), 1);
    return scatterIndicesT.reshape(shape);
  }

  if (indicesAreScalar) {
    return scatterIndicesT.reshape({scatterIndicesT.numElements()});
  }

  // Collapse all but the dimensions (0 or 1) in scatterIndices containing
  // the index vectors.
  std::vector<std::size_t> newShape = {
      scatterIndicesT.numElements() / shape.back(), shape.back()};
  return scatterIndicesT.reshape(newShape);
}

// Permutes the `updates` tensor such that all the scatter dims appear in the
// major dimensions and all the window dimensions appear in the minor
// dimensions.
Tensor
permuteScatterAndWindowDims(Tensor updates,
                            const std::vector<unsigned> updateWindowDims) {
  std::vector<unsigned> permutation(updates.rank());

  std::iota(std::begin(permutation), std::end(permutation), 0);

  const auto isWindowDim = [&updateWindowDims](unsigned dim) {
    return !std::binary_search(std::begin(updateWindowDims),
                               std::end(updateWindowDims), dim);
  };

  std::stable_partition(std::begin(permutation), std::end(permutation),
                        isWindowDim);

  return updates.dimShuffle(permutation);
}

// Expands or contracts the scatter indices in the updates tensor.
Tensor adjustScatterDims(std::vector<std::size_t> scatterIndicesShape,
                         Tensor updates, unsigned indexVectorDim) {
  unsigned rank = scatterIndicesShape.size();

  if (indexVectorDim < scatterIndicesShape.size()) {
    rank--;
  }

  auto shape = updates.shape();
  if (rank == 0) {
    shape.insert(shape.begin(), 1);

    // If there are no scatter dims, this must be a dynamic-update-slice kind of
    // scatter. In this case, we prepend a degenerate dimension to work
    // uniformly in the while loop.
    return updates.reshape(shape);
  }

  auto begin = std::begin(shape);
  auto collapse = std::next(std::begin(shape), rank);
  auto end = std::end(shape);

  std::vector<std::size_t> newShape;
  newShape.push_back(
      std::accumulate(begin, collapse, 1, std::multiplies<std::size_t>()));
  newShape.insert(std::end(newShape), collapse, end);

  return updates.reshape(newShape);
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

namespace tf_compat {
poplar::Tensor dynamicSlice(poplar::Graph &graph, poplar::Tensor input,
                            poplar::Tensor indices,
                            std::vector<std::size_t> sliceSizes,
                            poplar::program::Sequence &prog,
                            const poplar::OptionFlags &optionFlags,
                            const DebugNameAndId &dnai) {
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
                               newSliceSizes, prog, {dnai}, optionFlags);
  } else {
    poplar::Tensor copy = graph.clone(input, {dnai});
    prog.add(poplar::program::Copy(input, copy, false, {dnai}));
    out = copy;
  }

  return out;
}

poplar::Tensor dynamicUpdateSlice(poplar::Graph &graph, poplar::Tensor input,
                                  poplar::Tensor update, poplar::Tensor indices,
                                  std::vector<std::size_t> sliceSizes,
                                  poplar::program::Sequence &prog,
                                  const poplar::OptionFlags &optionFlags,
                                  const DebugNameAndId &dnai) {

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
                          newSliceSizes, prog, {dnai}, optionFlags);
  } else {
    prog.add(poplar::program::Copy(update, input, false, {dnai}));
  }

  return input;
}
} // namespace tf_compat

// We extract out individual components from the smaller index and concatenate
// them (interspersing zeros as needed) into the larger index.
poplar::Tensor expandIndexVectorIntoOperandSpace(
    poplar::Graph &graph, poplar::Tensor indices,
    std::vector<unsigned> scatterDimsToOperandDims, std::size_t rank,
    const DebugNameAndId &dnai) {
  poplar::Tensor zero =
      graph.addConstant(indices.elementType(), {1}, 0, {dnai, "zero"});
  graph.setTileMapping(zero, 0);
  std::vector<poplar::Tensor> expandedIndexComponents;
  expandedIndexComponents.reserve(rank);

  for (auto i = 0u; i < rank; ++i) {
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

poplar::Tensor padVectorWithZeros(poplar::Graph &graph, poplar::Tensor t,
                                  const DebugNameAndId &dnai,
                                  std::size_t prepend = 0,
                                  std::size_t append = 0) {
  poplar::Tensor prefix =
      graph.addConstant(t.elementType(), {prepend}, 0, {dnai, "prefix"});
  poplar::Tensor suffix =
      graph.addConstant(t.elementType(), {append}, 0, {dnai, "suffix"});
  graph.setTileMapping(prefix, 0);
  graph.setTileMapping(suffix, 0);
  return poplar::concat({prefix, t, suffix});
}

// High Level Algorithm.
//
// 1. Canonicalize the scatterIndices tensor such that it has rank 2, where
//    each row is an index into the operand.
// 2. Canonicalize the updates tensor such that is has rank `num_window_dims+1`
//    and the scatter dim is the most-major dimension.
// 3. Iterate over the set of indices in the canonicalized scatterIndices
//    tensor using a while loop, updating the operand for each such index. Each
//    iteration of this while loop performs the following:
//      a. Pick the index from scatterIndices for this iteration.
//      b. Transfrom this index into an index into the operand space.
//      c. Extract the slice to be used to update from the updates tensor.
//      d. Extract the slice to update from the operand tensor.
//      e. Write the updated value of the slice into the operand tensor.
void scatterInternal(
    poplar::Graph &graph, const poplar::Tensor &operand,
    const poplar::Tensor &indices, const poplar::Tensor &updates,
    std::size_t indexVectorDim, std::vector<unsigned> updateWindowDims,
    std::vector<std::size_t> insertWindowDims,
    std::vector<unsigned> scatterDimsToOperandDims,
    boost::optional<popops::UpdateComputationFunc &> updateComputation,
    poplar::program::Sequence &prog, poplar::OptionFlags optionFlags,
    const DebugNameAndId &dnai) {

  // If the updates tensor is empty, there is no need to update the operand. We
  // can return the operand as is.
  if (updates.numElements() == 0) {
    return;
  }

  // Compute the trip count for the while loop to be used for scatter. This
  // should be the number of indices we should scatter into the operand.
  const std::size_t scatterLoopTrip =
      scatterLoopTripCount(indices.shape(), indexVectorDim);

  // Canonicalize the scatterIndices, after which the size of its most-major
  // dimension must be same as the while loop trip count.
  poplar::Tensor canonicalScatterIndices =
      canonicalizeScatterIndices(indices, indexVectorDim);

  // Canonicalize the updates, after which the size of its most-major dimension
  // must be same as the while loop trip count.
  poplar::Tensor canonicalUpdates =
      permuteScatterAndWindowDims(updates, updateWindowDims);
  poplar::Tensor adjustedCanonicalUpdates =
      adjustScatterDims(indices.shape(), canonicalUpdates, indexVectorDim);

  const bool hasScalarIndices = canonicalScatterIndices.rank() == 1;

  // The while loop that implements the scatter operation.
  // for (i = 0; i < scatterLoopTripCount; ++i)
  prog.add(popops::countedLoop(
      graph, scatterLoopTrip,
      [&](poplar::Tensor i) {
        poplar::program::Sequence prog({}, {dnai});

        // Pick the index to scatter from scatterIndices based on the
        // inductionVar and transform that to an index into the `operand` space.
        poplar::Tensor indexVector;
        if (hasScalarIndices) {
          indexVector =
              tf_compat::dynamicSlice(graph, canonicalScatterIndices, i, {1},
                                      prog, optionFlags, {dnai});
        } else {
          std::size_t indexVectorSize = canonicalScatterIndices.dim(1);

          indexVector = padVectorWithZeros(graph, i, {dnai}, 0, 1);
          indexVector = tf_compat::dynamicSlice(
              graph, canonicalScatterIndices, indexVector, {1, indexVectorSize},
              prog, optionFlags, {dnai});
          indexVector = indexVector.squeeze({0});
        }

        poplar::Tensor scatterSliceStart = expandIndexVectorIntoOperandSpace(
            graph, indexVector, scatterDimsToOperandDims, operand.rank(),
            {dnai});

        // Extract the slice to be used to update from `updates` tensor for the
        // inductionVar corresponding to this iteration of the while loop.
        poplar::Tensor indexIntoUpdates = padVectorWithZeros(
            graph, i, {dnai}, 0, adjustedCanonicalUpdates.rank() - 1);

        auto updateSliceBounds = adjustedCanonicalUpdates.shape();
        updateSliceBounds[0] = 1;
        poplar::Tensor updateSlice = tf_compat::dynamicSlice(
            graph, adjustedCanonicalUpdates, indexIntoUpdates,
            updateSliceBounds, prog, optionFlags, {dnai});

        poplar::Tensor updateSliceWithDimsInserted = updateSlice.squeeze({0});
        for (auto dim : insertWindowDims) {
          updateSliceWithDimsInserted =
              updateSliceWithDimsInserted.expand({dim});
        }

        // Take the existing slice from the input tensor
        const auto updateSliceShape = updateSliceWithDimsInserted.shape();

        // If there is a user defined update computation
        if (updateComputation) {
          poplar::Tensor existingSlice = tf_compat::dynamicSlice(
              graph, operand, scatterSliceStart, updateSliceShape, prog,
              optionFlags, {dnai});

          // Combine the existing slice with the update slice using the user
          // provided update computation.
          updateSliceWithDimsInserted = (*updateComputation)(
              graph, existingSlice, updateSliceWithDimsInserted, prog);
        }

        // Copy the new slice into the tensor
        tf_compat::dynamicUpdateSlice(
            graph, operand, updateSliceWithDimsInserted, scatterSliceStart,
            updateSliceShape, prog, optionFlags, {dnai});

        return prog;
      },
      {dnai}));
}
} // namespace

namespace popops {

void scatter(poplar::Graph &graph, const poplar::Tensor &operand,
             const poplar::Tensor &indices, const poplar::Tensor &updates,
             std::size_t indexVectorDim, std::vector<unsigned> updateWindowDims,
             std::vector<std::size_t> insertWindowDims,
             std::vector<unsigned> scatterDimsToOperandDims,
             poplar::program::Sequence &prog,
             const poplar::DebugContext &debugContext,
             const poplar::OptionFlags &optionFlags) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(operand, indices, updates, indexVectorDim, updateWindowDims,
              insertWindowDims, scatterDimsToOperandDims));

  return scatterInternal(graph, operand, indices, updates, indexVectorDim,
                         updateWindowDims, insertWindowDims,
                         scatterDimsToOperandDims, boost::none, prog,
                         optionFlags, {di});
}

void scatter(poplar::Graph &graph, const poplar::Tensor &operand,
             const poplar::Tensor &indices, const poplar::Tensor &updates,
             std::size_t indexVectorDim, std::vector<unsigned> updateWindowDims,
             std::vector<std::size_t> insertWindowDims,
             std::vector<unsigned> scatterDimsToOperandDims,
             UpdateComputationFunc &updateComputation,
             poplar::program::Sequence &prog,
             const poplar::DebugContext &debugContext,
             const poplar::OptionFlags &optionFlags) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(operand, indices, updates, indexVectorDim, updateWindowDims,
              insertWindowDims, scatterDimsToOperandDims));
  return scatterInternal(graph, operand, indices, updates, indexVectorDim,
                         updateWindowDims, insertWindowDims,
                         scatterDimsToOperandDims, {updateComputation}, prog,
                         optionFlags, {di});
}

} // namespace popops
