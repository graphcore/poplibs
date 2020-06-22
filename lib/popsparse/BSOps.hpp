// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_BSOps_hpp
#define popsparse_BSOps_hpp

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popsparse/experimental/BlockSparseMatMul.hpp>

namespace popsparse {
namespace experimental {

poplar::Tensor slice(const poplar::Tensor &sparseTensor, std::size_t coord,
                     unsigned dimension, unsigned blockRow, unsigned blockCol,
                     unsigned blockRows, unsigned blockCols,
                     bool columnMajorBlock, const unsigned char *sparsity);

void applySubBlockMask(poplar::Graph &graph, const poplar::Tensor &sparseTensor,
                       SubBlockMask subBlockMask, unsigned blockRow,
                       unsigned blockCol, unsigned blockRows,
                       unsigned blockCols, const unsigned char *sparsity,
                       unsigned numGroups, poplar::program::Sequence &prog,
                       const std::string &debugPrefix);

} // namespace experimental
} // namespace popsparse

#endif // popsparse_BSOps_hpp