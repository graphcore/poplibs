// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poputil/VarStructure.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"

#include "poputil/DebugInfo.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <algorithm>

using namespace poplar;
using namespace poplibs_support;
using namespace poputil;

namespace poputil {

// Given a vector 'indices', generate into it all possible
// values in the interval [0, limits).
/// E.g. if 'limits' = {2,3}, then 'indices[2] will sequence
/// through the following values:
///    {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {0, 0}
/// The algorithm behaves like binary addition, carrying the
/// increment into the next when the possibilities
/// for the current dimension have been exhausted.
template <typename T> struct GenerateIndices {
  GenerateIndices(T &i, const T &l) : indices(i), limits(l) {
    assert(indices.size() == limits.size() &&
           "indices and indices' limits dimension must match");
  }
  // generate next sequence value into 'indices'
  void next() {
    carryDim = 0;
    while (carryDim < indices.size()) {
      ++indices[carryDim];
      if (indices[carryDim] < limits[carryDim]) {
        break;
      }
      indices[carryDim] = 0;
      carryDim++;
    }
  }
  bool complete() { return carryDim >= indices.size(); }

  T &indices;
  const T &limits;
  std::size_t carryDim;
};

// The order in which different parameters are permuted
enum class PermutationOrder {
  // The possible values of the first parameter in the list
  // are iterated first, followed by the second until all
  // permutations have been iterated.
  Forward
};

// For a list of parameters, each with their own possible values,
// iterate the possible permutations of the values of all parameters
// and pass them to a user-provided functor.
//
// The order in which we permute these is defined by the given
// permutation order.
//
// TODO: T12984 Unit test this and consider exposing in public API.
template <typename F>
static inline void permute(const std::vector<std::vector<std::size_t>> &params,
                           const PermutationOrder order, const F &f) {
  // We can only do forward - we specify an option so the order
  // is clearly defined.
  assert(order == PermutationOrder::Forward);
  std::vector<std::size_t> permutation(params.size());
  std::vector<std::size_t> indices(params.size());

  std::vector<std::size_t> indicesLimits;
  for (const auto &p : params)
    indicesLimits.push_back(p.size());
  GenerateIndices genI(indices, indicesLimits);

  do {
    for (std::size_t i = 0; i < params.size(); ++i) {
      permutation[i] = params[i][indices[i]];
    }
    f(indices, permutation);
    genI.next();
  } while (!genI.complete());
}

Tensor createPartitionableTensor(Graph &graph, const Type &type,
                                 const std::vector<std::size_t> &shape,
                                 const std::vector<std::size_t> &nPartitions,
                                 const poplar::DebugContext &debugContext) {
  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(type, shape, nPartitions));

  logging::poputil::debug("createPartitionableTensor '{}' with shape={} and "
                          "nPartitions={}",
                          debugContext.getPathName(), shape, nPartitions);

  if (shape.size() != nPartitions.size()) {
    throw poplibs_error("createPartitionableTensor: shape.size() (" +
                        std::to_string(shape.size()) +
                        ") != "
                        "nPartitions.size() (" +
                        std::to_string(nPartitions.size()) + ")");
  }

  // If any dimension is 0 the resulting tensor has no elements anyway
  // so just return an appropriately shaped tensor.
  if (std::any_of(shape.begin(), shape.end(),
                  [](std::size_t n) { return n == 0; })) {
    return graph.addVariable(type, shape, {di});
  }

  // The problem we want to solve is that we want the slice of the returned
  // tensor corresponding to each partition to be a single contiguous region.
  // For this to be true we must add a variable which has the shape of this
  // partition of the full tensor as its innermost dimensions.
  //
  // We also want to minimise the number of underlying variables so as to
  // simplify the returned tensor expression and reduce graph construction time
  // and compile time. In order to create as few variables as possible (and not
  // just create one variable per partition) we can combine multiple partitions
  // into one variable by adding an outer dimension to the partition's shape
  // with number of elements equal to the number of partitions.
  //
  // e.g. I want a variable {10,20} divided into 5 partitions by dividing
  // dimension 1 into 5 partitions: each partition must be of shape {10,4} and
  // I can create 1 variable with shape {5,10,4}. Each of the 5 slices of this
  // variable has the desired shape of partition I wanted and is formed of a
  // single contiguous region.
  //
  // If the number of partitions given for a particular dimension does not
  // divide that dimension evenly, we will have 2 different numbers of elements
  // in that dimension for 2 sets of partitions. We will no longer be able to
  // simply add our extra outer dimension to the variable we create as there
  // will be a partition that has a different number of elements in one of the
  // inner dimensions. In this case we create 2 separate variables.
  //
  // Generalising to splitting multiple dimensions to produce our partitions we
  // need 2^(number of dimensions which do not divide evenly) variables e.g.
  // for 2 dimensions that are not evenly divided:
  //
  // +----------------+-----------+
  // |                |           |
  // |                |           |
  // |                |           |
  // |                |           |
  // |                |           |
  // |       v1       |     v2    |
  // |                |           |
  // |                |           |
  // |                |           |
  // |                |           |
  // |                |           |
  // |                |           |
  // +----------------------------+
  // |                |           |
  // |       v3       |     v4    |
  // |                |           |
  // +----------------+-----------+
  //
  // To achieve this we add a variable for each of the possible shapes in any
  // of the partitions resulting from this split of this shape.  We list the
  // different sizes of each dimension and permute all the shapes we need.
  //
  std::vector<std::vector<std::size_t>> varDimSizes(shape.size());
  std::vector<std::vector<std::size_t>> varElemsPerSplit(shape.size());
  for (std::size_t d = 0; d < shape.size(); ++d) {
    const auto elemsPerSplit = gccs::ceildiv(shape[d], nPartitions[d]);
    const auto rem = shape[d] % elemsPerSplit;
    varDimSizes[d].push_back(shape[d] - rem);
    varElemsPerSplit[d].push_back(elemsPerSplit);
    // Add the remainder if there is any (i.e. this dimension cannot be evenly
    // split into the desired number of partitions).
    if (rem != 0) {
      varDimSizes[d].push_back(rem);
      varElemsPerSplit[d].push_back(rem);
    }
  }

  // To create the variables for each shape of the partitions found we create
  // variables with the number of partitions in each dimension as the
  // outer-most dimensions of the shape, and the shape of the partition as the
  // inner-most dimensions. This ensures that each of the partitions found in
  // this variable are a single contiguous region.
  //
  // We then shuffle and flatten the number of partitions back into each
  // respective dimension to give the shape of each variable we expect ready to
  // be stitched back together.
  //
  std::vector<Tensor> vars;
  permute(varDimSizes, PermutationOrder::Forward,
          [&](const std::vector<std::size_t> &i,
              const std::vector<std::size_t> &permutedShape) {
            const auto nDims = permutedShape.size();
            // The first `nDims` dimensions give the number of partitions in
            // each dimension. The second `nDims` dimensions give the shape of
            // the partition.
            std::vector<std::size_t> splitPermutedShape(nDims * 2);
            // The inverse permutation shuffles the number of partitions in each
            // dimension next to the shape of the partition in that dimension.
            std::vector<unsigned> inversePermutation(nDims * 2);
            for (std::size_t d = 0; d < nDims; ++d) {
              const auto elemsPerSplit = varElemsPerSplit[d][i[d]];
              assert(permutedShape[d] % elemsPerSplit == 0);
              const auto nSplits = permutedShape[d] / elemsPerSplit;
              splitPermutedShape[d] = nSplits;
              inversePermutation[d * 2] = d;
              splitPermutedShape[nDims + d] = elemsPerSplit;
              inversePermutation[d * 2 + 1] = nDims + d;
            }
            // Shuffle then flatten the number of partitions in each dimension
            // together with the shape of the partition in that dimension
            // (through a reshape to the permuted shape).
            vars.push_back(graph.addVariable(type, splitPermutedShape, {di})
                               .dimShuffle(inversePermutation)
                               .reshape(permutedShape));
          });

  // Finally, stitch the variables created back together to form the full
  // tensor to return.
  //
  // We treat this like a reduction. We have some number of partials = the
  // number of variables we created that we want to concat together. We can
  // only concat in one dimension at a time hence we iterate and concat in one
  // dimension at a time.
  //
  // That we iterate from the inner-most dimension to outer-most is purely an
  // implementation choice so that we output the result of each stage of
  // concatenation in the first `n` elements of the array rather than every
  // `vars.size() / n`th element of the array.
  std::size_t dim = shape.size() - 1;
  auto nPartials = vars.size();
  do {
    // The factor by which we reduce the number of individual variables in
    // the list. This is the number of differing sizes of variables in this
    // dimension.
    const auto factor = varDimSizes[dim].size();
    const auto nextNPartials = nPartials / factor;

    // Each output for this dimension is formed of the concat of `factor`
    // variables in that dimension. Based on the order in which we permuted the
    // shapes above, in each stage the array of variables is essentially a 2D
    // array with shape {factor, nextNPartials} and so we index the array as if
    // it were a flattened 2D array of this shape.
    std::vector<Tensor> toConcat(factor);
    for (std::size_t i = 0; i < nextNPartials; ++i) {
      for (std::size_t o = 0; o < factor; ++o) {
        toConcat[o] = vars[o * nextNPartials + i];
      }
      vars[i] = concat(toConcat, dim);
    }
    --dim;
    nPartials = nextNPartials;
  } while (nPartials != 1);
  const auto result = vars[0];

  di.addOutput(result);
  return result;
}

void iterateTensorPartitions(
    const Tensor &t, const std::vector<std::size_t> &nPartitions,
    const std::function<void(const std::vector<std::size_t> &i,
                             const Tensor &s)> &f) {
  if (t.rank() != nPartitions.size()) {
    throw poplibs_error("iterateTensorPartitions: t.rank() (" +
                        std::to_string(t.rank()) +
                        ") != "
                        "nPartitions.size() (" +
                        std::to_string(nPartitions.size()) + ")");
  }

  const auto shape = t.shape();
  std::vector<std::size_t> i(shape.size(), 0);
  GenerateIndices genI(i, nPartitions);
  do {
    Tensor slice = t;
    for (std::size_t d = 0; d < shape.size(); ++d) {
      const auto ceil = gccs::ceildiv(shape[d], nPartitions[d]);
      slice = slice.slice(std::min(i[d] * ceil, shape[d]),
                          std::min((i[d] + 1) * ceil, shape[d]), d);
    }
    f(i, slice);
    genI.next();
  } while (!genI.complete());
}

unsigned detectInnermostGrouping(const Graph &graph, const Tensor &t0) {
  if (t0.rank() == 0)
    throw poplibs_error("Cannot detect channel grouping of 0-rank tensor");

  if (t0.numElements() == 0)
    return 1;

  // Sample the first point in the inner dimension
  auto t = t0;
  while (t.rank() != 1)
    t = t[0];

  // Perform a binary search to find the largest contiguous slice in
  // the inner dimension.
  auto lower = 1U;
  auto upper = t.numElements();
  while (lower != upper) {
    // Find a mid-point such that lower < mid <= upper
    auto mid = upper - (upper - lower) / 2;
    if (t.slice(0, mid).isContiguous()) {
      lower = mid;
    } else {
      upper = mid - 1;
    }
  }

  // Find the largest contiguous region on a tile as an estimate of grouping
  const auto tileMapping = graph.getTileMapping(t);
  std::size_t maxRegionSize = 0;
  for (const auto &regions : tileMapping) {
    if (regions.empty())
      continue;
    const auto maxIt = std::max_element(
        regions.begin(), regions.end(),
        [](const poplar::Interval &a, const poplar::Interval &b) {
          return a.size() < b.size();
        });
    maxRegionSize = std::max(maxRegionSize, maxIt->size());
  }

  // Use the greatest common divisor between channel grouping detected on a tile
  // and contiguous regions of the tensor. Note that in the case when a group
  // is partially mapped to a tile, GCD doesn't  give the correct result.
  auto grouping = std::gcd(maxRegionSize, upper);

  // The channel grouping must divide the number of channels
  if (t.numElements() % grouping != 0)
    grouping = 1;
  return grouping;
}

std::vector<GroupingInfo> detectDimGroupings(const Graph &graph,
                                             const Tensor &t) {
  std::vector<GroupingInfo> info;

  auto dims = t.rank();
  auto groupedT = t;
  unsigned totalGrouping = 1;
  while (true) {
    unsigned grouping = 1;
    unsigned groupedDim = 0;

    for (std::size_t d = 0; d < dims; ++d) {
      // Skip singular dimensions
      if (groupedT.dim(d) == 1)
        continue;
      // Detect grouping of this dim along with previous groupings
      auto permutation =
          groupedT.dimRoll(d, dims - 1).flatten(dims - 1, groupedT.rank());
      auto g = detectInnermostGrouping(graph, permutation);
      // Even though we may already have found some grouping, the new
      // grouping we find may not be a multiple of totalGrouping if
      // there is a grouping in a weirdly sized combination of dimensions
      // so bottom out at 1 so that the gcd below gives the desired result.
      auto thisGrouping = g % totalGrouping ? 1u : g / totalGrouping;
      thisGrouping = std::gcd<unsigned>(thisGrouping, groupedT.dim(d));
      if (thisGrouping > grouping) {
        groupedDim = d;
        grouping = thisGrouping;
      }
    }

    // No more groupings to be found, we're done.
    if (grouping == 1)
      break;

    info.emplace_back(groupedDim, grouping);
    totalGrouping *= grouping;
    assert((groupedT.dim(groupedDim) % grouping) == 0);
    // Roll the grouping to the back for the next round
    groupedT =
        groupedT
            .reshapePartial(groupedDim, groupedDim + 1,
                            {groupedT.dim(groupedDim) / grouping, grouping})
            .dimRoll(groupedDim + 1, dims);
  }

  return info;
}

} // end namespace poputil
