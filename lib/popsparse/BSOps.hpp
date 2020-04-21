// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_BSOps_hpp
#define popsparse_BSOps_hpp

#include <poplar/Tensor.hpp>

namespace popsparse {
namespace experimental {

poplar::Tensor slice(const poplar::Tensor &sparseTensor, std::size_t coord,
                     unsigned dimension, unsigned blockRow, unsigned blockCol,
                     unsigned blockRows, unsigned blockCols,
                     bool columnMajorBlock, const unsigned char *sparsity);

} // namespace experimental
} // namespace popsparse

#endif // popsparse_BSOps_hpp