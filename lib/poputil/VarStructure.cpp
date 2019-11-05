#include "poputil/VarStructure.hpp"

#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/logging.hpp"

#include "poputil/exceptions.hpp"

#include <algorithm>

using namespace poplar;
using namespace poplibs_support;
using namespace poputil;

namespace poputil {

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
// TODO: T12984 Unit test this - move it to be public in some way?
template <typename F>
static inline void permute(const std::vector<std::vector<std::size_t>> &params,
                           const PermutationOrder order, const F &f) {
  // We can only do forward - we specify an option so the order
  // is clearly defined.
  assert(order == PermutationOrder::Forward);
  std::vector<std::size_t> permutation(params.size());
  std::vector<std::size_t> indices(params.size());
  std::size_t carryDim;
  do {
    for (std::size_t i = 0; i < params.size(); ++i) {
      permutation[i] = params[i][indices[i]];
    }
    f(indices, permutation);
    // Like binary addition, we carry the increment into the next
    // parameter's set of possible values when the possibilities
    // for the current parameter have been exhausted.
    carryDim = 0;
    while (carryDim < indices.size()) {
      ++indices[carryDim];
      if (indices[carryDim] < params[carryDim].size()) {
        break;
      }
      indices[carryDim] = 0;
      carryDim++;
    }

    // When we have reached the end of every set of possible
    // values for all parameters we are done
  } while (carryDim < indices.size());
}

Tensor createPartitionableTensor(Graph &graph, const Type &type,
                                 const std::vector<std::size_t> &shape,
                                 const std::vector<std::size_t> &nPartitions,
                                 const std::string &debugName) {
  logging::debug("createPartitionableTensor '{}' with shape={} and "
                 "nPartitions={}",
                 debugName, shape, nPartitions);

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
    return graph.addVariable(type, shape, debugName);
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
    const auto ceil = ceildiv(shape[d], nPartitions[d]);
    const auto rem = shape[d] % ceil;
    varDimSizes[d].push_back(shape[d] - rem);
    varElemsPerSplit[d].push_back(ceil);
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
              splitPermutedShape[nDims + d] = permutedShape[d] / nSplits;
              inversePermutation[d * 2 + 1] = nDims + d;
            }
            // Shuffle then flatten the number of partitions in each dimension
            // together with the shape of the partition in that dimension
            // (through a reshape to the permuted shape).
            vars.push_back(
                graph.addVariable(type, splitPermutedShape, debugName)
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
  std::size_t carryDim;
  do {
    Tensor slice = t;
    for (std::size_t d = 0; d < shape.size(); ++d) {
      const auto ceil = ceildiv(shape[d], nPartitions[d]);
      slice = slice.slice(std::min(i[d] * ceil, shape[d]),
                          std::min((i[d] + 1) * ceil, shape[d]), d);
    }
    f(i, slice);
    // Like binary addition, we carry the increment into the next
    // dimension's partitions when the partitions for the current
    // dimension have been exhausted.
    carryDim = 0;
    while (carryDim < i.size()) {
      ++i[carryDim];
      if (i[carryDim] < nPartitions[carryDim]) {
        break;
      }
      i[carryDim] = 0;
      carryDim++;
    }
  } while (carryDim < i.size());
}

} // end namespace poputil
