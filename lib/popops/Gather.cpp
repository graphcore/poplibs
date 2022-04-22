// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/Gather.hpp"
#include "GatherInternal.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/DynamicSlice.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Pad.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>

using namespace poplar;

namespace logging = poplibs_support::logging;

namespace poputil {
template <> poplar::ProfileValue toProfileValue(const popops::GatherParams &p) {
  poplar::ProfileValue::Map v;
  v.insert({"lookupParallelSplit", toProfileValue(p.maxElementsPerTile)});
  return v;
}
} // namespace poputil

namespace {
// Transposes the given indices such that the indexVectorDim becomes the
// most-minor dimension.
Tensor transposeIndexVectorDimToLast(const Tensor &indices,
                                     unsigned indexVectorDim) {
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

// The canonicalized indices is a 2D tensor where each row represents a single
// slice, and each column represents a coordinate into the input tensor.
Tensor canonicalizeGatherIndices(const Tensor &startIndices,
                                 unsigned indexVectorDim,
                                 const std::vector<unsigned> &startIndexMap) {
  // Transpose the non-index-vector dimensions to the front.
  Tensor startIndicesT =
      transposeIndexVectorDimToLast(startIndices, indexVectorDim);

  const bool indicesAreScalar = startIndicesT.rank() == indexVectorDim;

  // The number of dimensions in startIndices that are index dimensions.
  const std::size_t indexDimsInScatterIndices = indicesAreScalar ? 0 : 1;

  // If there is only one index (i.e. indicesAreScalar has rank 1 and this
  // scatter is really just a dynamic update slice) add a leading degenerate
  // dimension for uniformity.  Otherwise create a "collapsed" leading dimension
  // that subsumes all of the non-index-vector dimensions.
  std::vector<std::size_t> shape = startIndicesT.shape();
  if (shape.empty()) {
    startIndicesT = startIndicesT.reshape({1, 1});
  } else if (shape.size() == indexDimsInScatterIndices) {
    shape.insert(shape.begin(), 1);
    startIndicesT = startIndicesT.reshape(shape);
  } else if (indicesAreScalar) {
    startIndicesT = startIndicesT.reshape({startIndicesT.numElements(), 1});
  } else {
    // Collapse all but the dimensions (0 or 1) in startIndices containing
    // the index vectors.
    std::vector<std::size_t> newShape = {
        startIndicesT.numElements() / shape.back(), shape.back()};
    startIndicesT = startIndicesT.reshape(newShape);
  }

  // Reorganise the indices tensor to match the canonicalized input
  // This is kind of like a compile-time matmul with a permutation matrix
  std::vector<unsigned> permutation(startIndicesT.dim(1));
  boost::iota(permutation, 0);
  boost::sort(permutation, [&](unsigned a, unsigned b) {
    return startIndexMap[a] < startIndexMap[b];
  });

  std::vector<Tensor> permutedStartIndices(startIndicesT.dim(1));

  auto permuteIndices = [&](unsigned idx) {
    return startIndicesT.slice(idx, idx + 1, 1);
  };

  boost::transform(permutation, permutedStartIndices.begin(), permuteIndices);

  return concat(permutedStartIndices, 1);
}

// The canonicalized input tensor has its axis permuted such that all of its
// sliced axes are after the non-sliced axes.
Tensor canonicalizeGatherInput(const Tensor &input,
                               const std::vector<unsigned> &startIndexMap) {
  std::vector<unsigned> permutation(input.rank());
  boost::iota(permutation, 0);

  auto dimPred = [&](std::size_t dim) {
    return std::find(startIndexMap.begin(), startIndexMap.end(), dim) !=
           startIndexMap.end();
  };

  boost::stable_partition(permutation, dimPred);
  return input.dimShuffle(permutation);
}

// The gather result is in a canonical form where axes have been permuted such
// that the zero-th dimension is a batch axis, followed by sliced axes, followed
// by the non-sliced axes. This function applies an inverse of that permutation
// which restores the original order of slice dimensions and keeps the batch
// dimension at axis zero.
Tensor
adjustSliceDimensionsInAccumulator(const Tensor &accumulator,
                                   const std::vector<unsigned> &startIndexMap) {
  // Find the permutation of the non-batch axes which was applied to the gather
  // input prior to the gather.
  // Note that startIndexMap refers to the dimensions in the gather input which
  // (unlike the accumulator) doesn't have the batch dimension.
  std::vector<unsigned> permutation(accumulator.rank() - 1);
  boost::iota(permutation, 0);

  auto dimPred = [&](std::size_t dim) {
    return std::find(startIndexMap.begin(), startIndexMap.end(), dim) !=
           startIndexMap.end();
  };
  // Recreate the permutation by putting the sliced axes first.
  boost::stable_partition(permutation, dimPred);

  // Reverse the permutation and add an extra 0 dimension at the beginning to
  // keep the batch dimension of the accumulator at the beginning and offset all
  // the other dimensions by one to account for this extra dimension.
  std::vector<unsigned> inversePermutation(permutation.size() + 1);
  inversePermutation[0] = 0;
  for (auto i = 0ul; i < permutation.size(); ++i) {
    inversePermutation[permutation[i] + 1] = i + 1;
  }

  return accumulator.dimShuffle(inversePermutation);
}

// The canonicalized input tensor has its axis permuted such that all of its
// sliced axes after the non-sliced axes. This function transforms the
// sliceSizes so that it still refers to the same dimensions in the
// canonicalized input tensor.
std::vector<std::size_t>
canonicalizeSliceSizes(const std::vector<std::size_t> &sliceSizes,
                       const std::vector<unsigned> &startIndexMap) {
  std::vector<unsigned> permutation(sliceSizes.size());
  boost::iota(permutation, 0);

  auto dimPred = [&](std::size_t dim) {
    return std::find(startIndexMap.begin(), startIndexMap.end(), dim) !=
           startIndexMap.end();
  };

  boost::stable_partition(permutation, dimPred);

  std::vector<std::size_t> newSliceSizes(sliceSizes.size());
  for (uint i = 0; i < sliceSizes.size(); ++i) {
    newSliceSizes[i] = sliceSizes[permutation[i]];
  }

  return newSliceSizes;
}

// This function "expands" the gather dimensions back to the indices shape.
// For example, if we have an input of shape [2, 3, 4], scalar indices of shape
// [1, 2, 3], and we are taking whole slices from dimension 0 of the input, the
// accumulator would be of shape [6, 3, 4]. We want to reshape this back to
// [1, 2, 3, 3, 4],
Tensor
adjustBatchDimsInAccumulator(const std::vector<std::size_t> &startIndicesShape,
                             const Tensor &accumulator,
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

// Undo the partitioning of the canonicalization on the accumulator. This
// shuffles the dimensions back to how the users expects for the output.
Tensor permuteBatchAndOffsetDims(const Tensor &accumulator,
                                 const std::vector<std::size_t> &offsetDims,
                                 std::size_t outputRank) {
  std::vector<unsigned> permutation;
  permutation.reserve(outputRank);

  std::size_t batch_idx_counter = 0;
  std::size_t offset_idx_counter = outputRank - offsetDims.size();

  for (std::size_t dim = 0; dim < outputRank; ++dim) {
    bool is_offset_dim = std::find(offsetDims.begin(), offsetDims.end(), dim) !=
                         offsetDims.end();
    if (is_offset_dim) {
      permutation.push_back(offset_idx_counter++);
    } else {
      permutation.push_back(batch_idx_counter++);
    }
  }

  return accumulator.dimShuffle(permutation);
}
} // namespace

namespace popops {

poplar::Tensor createGatherInput(poplar::Graph &graph, const poplar::Type &type,
                                 const std::vector<std::size_t> &inputShape,
                                 const std::vector<std::size_t> &sliceSizes,
                                 std::vector<unsigned> startIndexMap,
                                 const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(type, inputShape, sliceSizes, startIndexMap));

  std::vector<unsigned> permutation(inputShape.size());
  boost::iota(permutation, 0);

  auto dimPred = [&](std::size_t dim) {
    return std::find(startIndexMap.begin(), startIndexMap.end(), dim) !=
           startIndexMap.end();
  };

  boost::stable_partition(permutation, dimPred);
  std::vector<std::size_t> canonShape;
  canonShape.reserve(inputShape.size());
  for (auto i = 0ul; i < inputShape.size(); ++i) {
    canonShape.emplace_back(inputShape[permutation[i]]);
  }

  std::vector<std::size_t> canonSliceSizes;
  canonSliceSizes.reserve(startIndexMap.size());
  std::sort(startIndexMap.begin(), startIndexMap.end());
  for (unsigned i = 0; i < startIndexMap.size(); ++i) {
    canonSliceSizes.push_back(sliceSizes[startIndexMap[i]]);
  }

  auto input = internal::createGatherInputTensor(graph, type, canonShape,
                                                 canonSliceSizes, {di});

  std::vector<unsigned> inversePermutation(inputShape.size());
  for (auto i = 0ul; i < inputShape.size(); ++i) {
    inversePermutation[permutation[i]] = i;
  }

  input = input.dimShuffle(inversePermutation);
  di.addOutput(input);
  return input;
}

Tensor gather(Graph &graph, const Tensor &input, const Tensor &indices,
              std::size_t indexVectorDim,
              const std::vector<std::size_t> &offsetDims,
              const std::vector<std::size_t> &sliceSizes,
              const std::vector<std::size_t> &collapsedSliceDims,
              const std::vector<unsigned> &startIndexMap,
              program::Sequence &prog,
              const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(input, indices, indexVectorDim, offsetDims,
                            sliceSizes, collapsedSliceDims, startIndexMap));

  logging::popops::info("gather input={}, indices={}, name={}", input.shape(),
                        indices.shape(), debugContext.getPathName());

  auto canonicalizedIndices =
      canonicalizeGatherIndices(indices, indexVectorDim, startIndexMap);
  auto canonicalizedInput = canonicalizeGatherInput(input, startIndexMap);

  auto canonSliceSizes = canonicalizeSliceSizes(sliceSizes, startIndexMap);

  for (uint i = canonicalizedIndices.dim(1); i < canonSliceSizes.size(); ++i) {
    canonicalizedInput = canonicalizedInput.slice(0, canonSliceSizes[i], i);
  }

  canonSliceSizes.resize(canonicalizedIndices.dim(1));

  auto result =
      internal::gather(graph, canonicalizedInput, canonicalizedIndices,
                       canonSliceSizes, prog, {di});

  // Permute the dimensions in the result tensor to put the sliced and
  // non-sliced axes back into their original positions.
  result = adjustSliceDimensionsInAccumulator(result, startIndexMap);

  // Remove collapsed slice dimensions.
  auto offsetCollapsedSliceDims = collapsedSliceDims;
  boost::transform(offsetCollapsedSliceDims, offsetCollapsedSliceDims.begin(),
                   [](std::size_t dim) { return dim + 1; });
  result = result.squeeze(offsetCollapsedSliceDims);

  // Expand batch axes.
  result =
      adjustBatchDimsInAccumulator(indices.shape(), result, indexVectorDim);

  // Permute the dimensions into the expected output format.
  const auto outputRank = (indices.rank() == indexVectorDim ? 0 : -1) +
                          offsetDims.size() + indices.rank();
  auto output = permuteBatchAndOffsetDims(result, offsetDims, outputRank);
  di.addOutput(output);
  return output;
}

Tensor createGatherInput(Graph &graph, const Type &type,
                         const std::vector<std::size_t> &operandShape,
                         unsigned axis, GatherParams params,
                         const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(type, operandShape, axis, params));

  if (operandShape[axis] > params.maxElementsPerTile) {
    std::vector<std::size_t> newOperandShape = operandShape;
    if (operandShape[axis] % 2 == 1) {
      newOperandShape[axis] += 1;

      auto output =
          createGatherInput(graph, type, newOperandShape, axis, params, {di})
              .slice(0, operandShape[axis], axis);
      di.addOutput(output);
      return output;
    } else {
      newOperandShape[axis] /= 2;
      newOperandShape.insert(newOperandShape.begin() + axis + 1, 2);

      auto output =
          createGatherInput(graph, type, newOperandShape, axis, params, {di})
              .reshape(operandShape);
      di.addOutput(output);
      return output;
    }
  } else {
    const std::vector<std::size_t> sliceSizes = {1};

    std::vector<unsigned> permutation(operandShape.size());
    boost::iota(permutation, 0);
    std::swap(permutation.front(), permutation[axis]);

    std::vector<std::size_t> canonShape = operandShape;
    for (unsigned i = 0; i < operandShape.size(); ++i) {
      canonShape[i] = operandShape[permutation[i]];
    }

    auto input = internal::createGatherInputTensor(graph, type, canonShape,
                                                   sliceSizes, {di});

    auto output = input.dimShuffle(permutation);
    di.addOutput(output);
    return output;
  }
}

Tensor gather(Graph &graph, const Tensor &input, const Tensor &indices,
              unsigned axis, program::Sequence &prog, GatherParams params,
              const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, indices, axis, params));

  if (input.dim(axis) > params.maxElementsPerTile) {
    if (input.dim(axis) % 2 == 1) {
      return gather(graph, pad(graph, input, 0, 1, axis), indices, axis, prog,
                    params, {di});
    }

    auto shape = input.shape();
    shape[axis] /= 2;
    shape.insert(shape.begin() + axis + 1, 2);

    auto one = graph.addConstant(UNSIGNED_INT, {}, 1, {di, "const_1"});
    graph.setTileMapping(one, 0);

    auto indicesDiv = shiftRight(graph, indices, one, prog, {di});
    auto indicesRem = bitwiseAnd(graph, indices, one, prog, {di});
    auto indicesPred = eq(graph, indicesRem, one, prog, {di});

    auto result = gather(graph, input.reshape(shape), indicesDiv, axis, prog,
                         params, {di, "halved"});

    // The odd and even slice pairs from the split gather
    auto even = result.slice(0, 1, axis + 1);
    auto odd = result.slice(1, 2, axis + 1);

    auto s = odd.shape();
    std::fill(s.begin(), s.end(), 1);
    s[axis] = indicesPred.numElements();
    indicesPred = indicesPred.reshape(s);

    poputil::broadcastToMatch(indicesPred, odd.shape());
    auto output =
        select(graph, odd, even, indicesPred, prog).squeeze({axis + 1});
    di.addOutput(output);
    return output;
  }

  const std::vector<std::size_t> sliceSizes = {1};

  std::vector<unsigned> inputPermutation(input.rank());
  boost::iota(inputPermutation, 0);
  std::swap(inputPermutation.front(), inputPermutation[axis]);

  auto output =
      internal::gather(graph, input.dimShuffle(inputPermutation),
                       indices.flatten().expand({1}), sliceSizes, prog, {di});
  output = output.squeeze({1});

  std::vector<unsigned> outputPermutation(output.rank());
  boost::iota(outputPermutation, 0);
  std::swap(outputPermutation.front(), outputPermutation[axis]);

  auto output_ = output.dimShuffle(outputPermutation);
  di.addOutput(output_);
  return output_;
}

} // namespace popops
