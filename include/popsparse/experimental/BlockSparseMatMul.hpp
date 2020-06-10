// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef POPSPARSE_BLOCK_SPARSE_MATMUL_H
#define POPSPARSE_BLOCK_SPARSE_MATMUL_H

#include "BlockSparse.hpp"
#include <array>
#include <poplar/Graph.hpp>

namespace popsparse {
namespace experimental {

class BSMatMulImpl;

/**
 * This class supports block-sparse matrix multiplication.
 *
 * The class only saves the sparsity mask, the matrix size, the block size,
 * and the data type, which are used to generate the computation graph.
 *
 * The matrix data is passed in when function \c bsMatMul() or \c bsUpdate()
 * gets called.
 *
 * The purpose of this design is to reuse the instance of this class when only
 * the data of the matrix is changed, and the matrix sparsity does not change.
 *
 * The current implementation is based on Zoltan to generate the hypergraph
 * partition for all tiles. Zoltan usually runs 2 minutes for ~16k non-zero
 * blocks, which is expensive if it runs for every matrix multiplication.
 *
 * The right matrix is always sparse, and the left matrix can be dense or
 * sparse.
 */
class BSMatMulParams {

public:
  /** This constructor is for a dense matrix (left side) multiplying a sparse
   * matrix (right side).
   *
   *
   * \param dim[0]        Number of rows in the left-hand matrix.
   * \param dim[1]        Number of columns in the left-hand matrix.
   * \param dim[2]        If the right matrix needs to be transposed,
   *                      this is the number of rows in the right-hand
   *                      matrix. Otherwise, it is number of columns
   *                      in the right-hand matrix.
   *
   * \param blockSize[0]  Block size of the rows in the left-hand matrix.
   * \param blockSize[1]  Block size of the columns in the left-hand matrix.
   * \param blockSize[2]  Block size of the columns in the right-hand matrix.
   *
   * \param rhsSparsity   The 2D sparsity mask for the right-hand block-sparse
   *                      matrix, in which '1' is a non-zero block and '0'
   *                      is a zero block.
   *
   * \param rhsNeedTranspose   True if the right-hand matrix needs to be
   *                           transposed.
   *                           This is to support backward passes.
   *
   * \param inDataType      Input data type.
   *
   * \param outDataType     Output data type.
   *
   * \param partialDataType Partial data type.
   *
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &rhsSparsity,
                 bool rhsNeedTranspose, poplar::Type inDataType,
                 poplar::Type outDataType, poplar::Type partialDataType);

  /** This constructor is for a sparse matrix multiplied by a sparse matrix.
   * It is not supported.
   */
  /* we can not find this use case in AI models.
   *
   * \param dim[0]        Number of rows in the left-hand matrix.
   * \param dim[1]        Number of columns in the left-hand matrix.
   * \param dim[2]        If the right matrix needs to be transposed,
   *                      this is the number of rows in the right-hand
   *                      matrix. Otherwise, it is number of columns
   *                      in the right-hand matrix.
   *
   * \param blockSize[0]  Block size of the rows in the left-hand matrix.
   * \param blockSize[1]  Block size of the columns in the left-hand matrix.
   * \param blockSize[2]  Block size of the columns in the right-hand matrix.
   *
   *
   * \param lhsSparsity     The 2D sparsity mask for the left-hand block-sparse
   *                        matrix, in which '1' is a non-zero block and '0'
   *                        is a zero block.
   *
   * \param lhsNeedTranspose   True if the left-hand matrix needs to be
   *                           transposed.
   *                           This is to support the backward pass.
   *
   * \param rhsSparsity   The 2D sparsity mask for the right-hand block-sparse
   *                      matrix, in which '1' is a non-zero block and '0'
   *                      is a zero block.
   *
   * \param rhsNeedTranspose   True if the right-hand matrix needs to be
   *                           transposed.
   *                           This is to support the backward pass.
   *
   * \param inDataType      Input data type.
   *
   * \param outDataType     Output data type.
   *
   * \param partialDataType Partial data type.
   *
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &lhsSparsity,
                 bool lhsNeedTranspose,
                 const std::vector<unsigned char> &rhsSparsity,
                 bool rhsNeedTranspose, poplar::Type inDataType,
                 poplar::Type outDataType, poplar::Type partialDataType);

  /** This constructor is for a dense matrix multiplying a dense matrix.
   *  The multiply is performed as a sparse operation and the result stored
   *  as a sparse matrix.
   *
   * \param dim[0]          Number of rows in the left-hand matrix.
   * \param dim[1]          Number of columns in the left-hand matrix.
   * \param dim[2]          Number of columns in the right-hand matrix.
   *
   * \param blockSize[0]    Block size of the rows in the left-hand matrix.
   * \param blockSize[1]    Block size of the columns in the left-hand matrix.
   * \param blockSize[2]    Block size of the columns in the right-hand matrix.
   *                        The block size of the columns in the left-hand
   *                        matrix equals the block size of the rows in the
   *                        right-hand matrix.
   *
   * \param resSparsity     The 2D sparsity mask for the result block-sparse
   *                        matrix, in which '1' is a non-zero block and '0'
   *                        is a zero block.
   *
   * \param inDataType      Input data type.
   *
   * \param outDataType     Output data type.
   *
   * \param partialDataType Partial data type.
   *
   * \param SubBlockMask    The mask inside a block. See \c SubBlockMask in
   *                        ``BlockSparse.hpp`` for details.
   *
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &resSparsity,
                 poplar::Type inDataType, poplar::Type outDataType,
                 poplar::Type partialDataType,
                 SubBlockMask subBlockMask = SubBlockMask::None);

  BSMatMulParams(BSMatMulParams &&other);

  ~BSMatMulParams();

  std::unique_ptr<BSMatMulImpl> impl;
};

/**
 * Create a tensor for use as the left operand of block-sparse matrix
 * multiplication.
 *
 * \param graph           The Poplar graph.
 *
 * \param bsMatMul        The object for block-sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type.
 *
 * \param name            The debug name of the created matrix.
 *
 * \returns               If the left matrix is a dense matrix, the return
 *                        tensor is just a regular 2D matrix. If it is a sparse
 *                        matrix, the return tensor is an array of non-zero
 *                        blocks.
 */
poplar::Tensor createBSMatMulInputLHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name);

/**
 * Create a tensor for use as the right operand of block-sparse matrix
 * multiplication.
 *
 * \param graph           The Poplar graph.
 *
 * \param bsMatMul        The object for block-sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type.
 *
 * \param name            The debug name of the created matrix.
 *
 * \returns               The return tensor is an array of non-zero
 *                        blocks for the block-sparse matrix
 */
poplar::Tensor createBSMatMulInputRHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name);

/* This function multiplies the left-hand matrix by the right-hand matrix.
 *
 * \param graph         The Poplar graph.
 *
 * \param bsMatMul      The object for block-sparse information, includes the
 *                      sparsity mask, the matrix size, the block size,
 *                      and the data type.
 *
 * \param prog          A reference to a program sequence to which the code
 *                      to perform the multiplication will be appended.
 *
 * \param lhsMatrix     If \p BSMatMulParams is for a dense x sparse multiply,
 *                      then this is the left-hand dense matrix.
 *                      If \p BSMatMulParams is for a sparse x sparse multiply,
 *                      then this is the non-zero blocks of the left sparse
 *                      matrix.
 *
 * \param rhsMatrix     A tensor for an array of non-zero blocks in the
 *                      right-hand sparse matrix.
 *
 * \param options       A structure containing options for how the
 *                      multiplication should be implemented.
 *
 *                        * `memory_cycle_ratio`: for computing the
 *                          weight of hyper graph node. This may be only a
 *                          temporary option.
 *
 *                          w = memory_cycle_ratio * mem_weight +
 *                             (1.0 - memory_cycle_ratio) * cycle_weight
 *
 * \param debugPrefix   A debug prefix added to compute set and tensor
 *                      names.
 *
 * \returns             The tensor holding the result of the
 *                      multiplication. This tensor will be created, added to
 *                      the graph and mapped to tiles.
 */
poplar::Tensor bsMatMul(poplar::Graph &graph, const BSMatMulParams &bsMatMul,
                        poplar::program::Sequence &prog,
                        const poplar::Tensor &lhsMatrix,
                        const poplar::Tensor &rhsMatrix,
                        const poplar::OptionFlags &options = {},
                        const std::string &debugPrefix = "");

} // namespace experimental
} // namespace popsparse

#endif // POPSPARSE_BLOCK_SPARSE_MATMUL_H
