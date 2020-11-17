// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_BSNonLinearity_hpp
#define popsparse_BSNonLinearity_hpp

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popsparse/experimental/BlockSparse.hpp>

namespace popsparse {
namespace experimental {

poplar::Tensor bsSoftmaxInternal(
    poplar::Graph &graph, poplar::Tensor sparseTensor, bool inPlace,
    unsigned blockRow, unsigned blockCol, unsigned blockRows,
    unsigned blockCols, const unsigned char *sparsity,
    popsparse::experimental::SubBlockMask subBlockMaskType,
    poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

poplar::Tensor bsSoftmaxGradInternal(
    poplar::Graph &graph, poplar::Tensor sparseOut,
    poplar::Tensor sparseOutGrad, unsigned blockRow, unsigned blockCol,
    unsigned blockRows, unsigned blockCols, const unsigned char *sparsity,
    poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

} // namespace experimental
} // namespace popsparse

#endif // popsparse_BSNonLinearity_hpp