// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_BSUtils_hpp
#define popsparse_BSUtils_hpp

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popsparse/experimental/BlockSparse.hpp>

namespace popsparse {
namespace experimental {

void bsCreateMaskTensor(poplar::Graph &graph, unsigned blockRow,
                        unsigned blockCol, unsigned blockRows,
                        unsigned blockCols, const unsigned char *sparsity,
                        popsparse::experimental::SubBlockMask subBlockMaskType,
                        float maskedValue, float unMaskedValue,
                        const poplar::Type &dataType,
                        std::vector<poplar::Tensor> &maskBlocks,
                        std::vector<unsigned> &diagBlockIdxs,
                        std::vector<bool> &emptyRowsMask,
                        const std::string &debugStr = "");

} // namespace experimental
} // namespace popsparse

#endif // popsparse_BSUtils_hpp