// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_MatMulUtils_hpp
#define popsparse_MatMulUtils_hpp

#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>

#include "MatMulOptions.hpp"
#include "popsparse/FullyConnectedParams.hpp"
#include "popsparse/MatMulParams.hpp"
#include "popsparse/SparseTensor.hpp"

namespace popsparse {
namespace dynamic {

/// Get equivalent fully connected layer parameters for the
/// given matrix multiplication parameters.
FullyConnectedParams getFullyConnectedParams(const MatMulParams &options);

/// Get option flags for equivalent fully connected layer for
/// the given matrix multiplication options.
poplar::OptionFlags getFullyConnectedOptions(const MatMulOptions &options);

// [B, G * C] -> [G, B, C]
static inline poplar::Tensor fcActsToMatrix(const poplar::Tensor &t,
                                            const std::size_t numGroups) {
  assert(t.rank() == 2);
  assert(t.dim(1) % numGroups == 0);
  return t.reshapePartial(1, 2, {numGroups, t.dim(1) / numGroups})
      .dimRoll(1, 0);
}

// [G, B, C] -> [B, G * C]
static inline poplar::Tensor matrixToFCActs(const poplar::Tensor &t,
                                            const std::size_t numGroups) {
  assert(t.rank() == 3);
  assert(t.dim(0) == numGroups);
  return t.dimRoll(0, 1).flatten(1, 3);
}

SparseTensor sparseMatrixToFCWeights(const SparseTensor &t);

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_MatMulUtils_hpp
