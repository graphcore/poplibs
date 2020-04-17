// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef POPSPARSE_BLOCK_SPARSE_H
#define POPSPARSE_BLOCK_SPARSE_H

#include <array>
#include <string>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>

namespace popsparse {
namespace experimental {

// This is to define the sparsity mask inside a block
//
// NONE                 all elements are not zeroed out
//
// ZeroUpperTriangle:   Elements in upper triangle, above the diagonal are
//                      zeroed out.
//                      The diagonal is defined across the whole non-sparse
//                      matrix dimensions, where row index is equal to column
//                      index. Same for ZeroLowerTriangle case.
//
// ZeroLowerTriangle:   Elements in lower triangle, below the diagonal are
//                      zeroed out.
enum class SubBlockMask { None, ZeroUpperTriangle, ZeroLowerTriangle };

/* This function computes softmax on a sparse tensor.
 *
 * \param graph         The Poplar graph
 *
 * \param sparseTensor  The input sparse 2D tensor. It must be in a blocksparse
 *                      format.
 *
 * \param dim           dim[0] : number of rows of the original dense tensor.
 *                      dim[1] : number of columns of the original dense tensor.
 *
 * \param blockSize     blockSize[0] :block size of the row.
 *                      blockSize[1] :block size of the column.
 *
 * \param sparsity      The 2D sparsity mask for the block sparse
 *                      tensor, in which '1' is a non zero block and '0'
 *                      is a zero block.
 *
 * \param subBlockMaskType  Subblock mask type. Elements in upper(low) triangle
 *                          are filled by zeros in the result
 *
 * \param prog          A reference to a program sequence which will
 *                      be appended with the code to perform the softmax.
 */
poplar::Tensor bsSoftmax(poplar::Graph &graph, poplar::Tensor sparseTensor,
                         const std::array<int, 2> &dim,
                         const std::array<int, 2> &blockSize,
                         const std::vector<unsigned char> &sparsity,
                         SubBlockMask subBlockMaskType,
                         poplar::program::Sequence &prog,
                         const std::string &debugStr = "");

/* This function computes softmax on a sparse tensor in place.
 *
 * \param graph         The Poplar graph
 *
 * \param sparseTensor  The input sparse 2D tensor. It must be in a blocksparse
 *                      format.
 *
 * \param dim           dim[0] : number of rows of the original dense tensor.
 *                      dim[1] : number of columns of the original dense tensor.
 *
 * \param blockSize     blockSize[0] :block size of the row.
 *                      blockSize[1] :block size of the column.
 *
 * \param sparsity      The 2D sparsity mask for the block sparse
 *                      tensor, in which '1' is a non zero block and '0'
 *                      is a zero block.
 *
 * \param subBlockMaskType  Subblock mask type. Elements in upper(low) triangle
 *                          are filled by zeros in the result
 *
 * \param prog          A reference to a program sequence which will
 *                      be appended with the code to perform the softmax.
 */
void bsSoftmaxInPlace(poplar::Graph &graph, poplar::Tensor sparseTensor,
                      const std::array<int, 2> &dim,
                      const std::array<int, 2> &blockSize,
                      const std::vector<unsigned char> &sparsity,
                      SubBlockMask subBlockMaskType,
                      poplar::program::Sequence &prog,
                      const std::string &debugStr = "");

/* This function computes softmax gradient on a sparse tensor.
 *
 * \param graph         The Poplar graph
 *
 * \param sparseOut     The outer (activation) sparse 2D tensor. It must be in a
 *                      blocksparse format.
 *
 * \param sparseOutGrad The outer gradient sparse 2D tensor. It must be in a
 *                      blocksparse format.
 *
 * \param dim           dim[0] : number of rows of the original dense tensor.
 *                      dim[1] : number of columns of the original dense tensor.
 *
 * \param blockSize     blockSize[0] :block size of the row.
 *                      blockSize[1] :block size of the column.
 *
 * \param sparsity      The 2D sparsity mask for the block sparse
 *                      tensor, in which '1' is a non zero block and '0'
 *                      is a zero block.
 *
 * \param prog          A reference to a program sequence which will
 *                      be appended with the code to perform the softmax.
 */
poplar::Tensor bsSoftmaxGrad(poplar::Graph &graph, poplar::Tensor sparseOut,
                             poplar::Tensor sparseOutGrad,
                             const std::array<int, 2> &dim,
                             const std::array<int, 2> &blockSize,
                             const std::vector<unsigned char> &sparsity,
                             poplar::program::Sequence &prog,
                             const std::string &debugStr = "");

} // namespace experimental
} // namespace popsparse

#endif // POPSPARSE_BLOCK_SPARSE_H
