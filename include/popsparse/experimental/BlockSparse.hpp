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

/// Define the sparsity mask inside a block.
///
/// The diagonal is defined across sll the non-sparse
/// matrix dimensions, where the row index is equal to the column
/// index.
enum class SubBlockMask {
  /// No elements are zeroed out.
  None,
  /// Elements in the upper triangle, above the diagonal, are zeroed out.
  ZeroUpperTriangle,
  /// Elements in the lower triangle, below the diagonal, are zeroed out.
  ZeroLowerTriangle
};

/** This function computes softmax on a sparse tensor.
 *
 * \param graph         The Poplar graph.
 *
 * \param sparseTensor  The input sparse 2D tensor. It must be in a block-sparse
 *                      format.
 *
 * \param dim[0]  Number of rows of the original dense tensor.
 * \param dim[1]  Number of columns of the original dense tensor.
 *
 * \param blockSize[0] Block size of the rows.
 * \param blockSize[1] Block size of the columns.
 *
 * \param sparsity      The 2D sparsity mask for the block-sparse
 *                      tensor, in which '1' is a non zero block and '0'
 *                      is a zero block.
 *
 * \param subBlockMaskType  Sub-block mask type. Elements in upper (or lower)
 *                          triangle are filled by zeroes in the result.
 *
 * \param prog          A reference to the program sequence to which
 *                      the code to perform the softmax will be appended.
 */
poplar::Tensor bsSoftmax(poplar::Graph &graph, poplar::Tensor sparseTensor,
                         const std::array<int, 2> &dim,
                         const std::array<int, 2> &blockSize,
                         const std::vector<unsigned char> &sparsity,
                         SubBlockMask subBlockMaskType,
                         poplar::program::Sequence &prog,
                         const std::string &debugStr = "");

/** This function computes softmax on a sparse tensor, in place.
 *
 * \param graph         The Poplar graph.
 *
 * \param sparseTensor  The input sparse 2D tensor. It must be in a block-sparse
 *                      format.
 *
 * \param dim[0]  Number of rows of the original dense tensor.
 * \param dim[1]  Number of columns of the original dense tensor.
 *
 * \param blockSize[0]  Block size of the rows.
 * \param blockSize[1]  Block size of the columns.
 *
 * \param sparsity      The 2D sparsity mask for the block-sparse
 *                      tensor, in which '1' is a non zero block and '0'
 *                      is a zero block.
 *
 * \param subBlockMaskType  Sub-block mask type. Elements in upper (or lower)
 *                          triangle are filled by zeroes in the result.
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

/** This function computes softmax gradient on a sparse tensor.
 *
 * \param graph         The Poplar graph
 *
 * \param sparseOut     The outer (activation) sparse 2D tensor. It must be in
 *                      block-sparse format.
 *
 * \param sparseOutGrad The outer gradient sparse 2D tensor. It must be in a
 *                      block-sparse format.
 *
 * \param dim[0]        Number of rows of the original dense tensor.
 * \param dim[1]        Number of columns of the original dense tensor.
 *
 * \param blockSize[0]  Block size of the rows.
 * \param blockSize[1]  Block size of the columns.
 *
 * \param sparsity      The 2D sparsity mask for the block-sparse
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
