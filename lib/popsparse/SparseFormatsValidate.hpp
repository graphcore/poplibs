// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Contains private header for validation of Sparse Formats

#ifndef _poplibs_popsparse_SparseFormatsValidate_hpp_
#define _poplibs_popsparse_SparseFormatsValidate_hpp_

#include <array>
#include <string>
#include <vector>

namespace popsparse {

// Validate CSR representation. Throws exceptions if format representations
// are violated.
void validateCSR(unsigned numRows, unsigned numColumns,
                 const std::array<std::size_t, 2> &blockDimensions,
                 const std::size_t numNzValues,
                 const std::vector<std::size_t> &rowIndices,
                 const std::vector<std::size_t> &columnIndices);

// Validate CSC representation. Throws exceptions if format representations
// are violated.
void validateCSC(unsigned numRows, unsigned numColumns,
                 const std::array<std::size_t, 2> &blockDimensions,
                 const std::size_t numNzValues,
                 const std::vector<std::size_t> &rowIndices,
                 const std::vector<std::size_t> &columnIndices);

// Validate COO representation. Throws exceptions if format representations
// are violated.
void validateCOO(unsigned numRows, unsigned numColumns,
                 const std::array<std::size_t, 2> &blockDimensions,
                 const std::size_t numNzValues,
                 const std::vector<std::size_t> &rowIndices,
                 const std::vector<std::size_t> &columnIndices);

} // namespace popsparse
#endif // _poplibs_popsparse_SparseFormatsValidate_hpp_
