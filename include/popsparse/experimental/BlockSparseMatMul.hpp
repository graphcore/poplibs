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
   *                      Block size must be divisible by 16 for FP16 and
   *                      divisible by 8 for FP32
   *
   * \param rhsSparsity     The 2D sparsity mask for right hand block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block.
   *                        For group operation this parameter is
   *                        concatenated sparsity masks for all ops in a group.
   *
   * \param rhsNeedTranspose   Whether the right hand matrix need be transposed.
   *                           This is mostly to support backward pass.
   *                        If this parameter is true:
   *                         - dim, blockSize must conform to transposed shape
   *                         - rhsSparsity must be in original, non-transposed
   *                           order
   *                         - rhsMatrix in bsMatMul() must contain data within
   *                           blocks in original, non-transposed order
   *
   * \param inDataType      Input data type.
   *
   * \param outDataType     Output data type.
   *
   * \param partialDataType Partial data type.
   *
   * \param numGroupsIn     number of groups for group operation
   *                        or 1 for non-group operation
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &rhsSparsity,
                 bool rhsNeedTranspose, poplar::Type inDataType,
                 poplar::Type outDataType, poplar::Type partialDataType,
                 unsigned numGroupsIn = 1);

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
   *                        Block size must be divisible by 16 for FP16 and
   *                        divisible by 8 for FP32
   *
   * \param resSparsity     The 2D sparsity mask for the result block-sparse
   *                        matrix, in which '1' is a non-zero block and '0'
   *                        is a zero block.
   *
   * \param resSparsity     The 2D sparsity mask for the result block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block.
   *                        For group operation this parameter is
   *                        concatenated sparsity masks for all ops in a group.
   *
   * \param outDataType     Output data type.
   *
   * \param partialDataType Partial data type.
   *
   * \param SubBlockMask    The mask inside a block. See \c SubBlockMask in
   *                        ``BlockSparse.hpp`` for details.
   *
   * \param numGroupsIn     number of groups for group operation
   *                        or 1 for non-group operation
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &resSparsity,
                 poplar::Type inDataType, poplar::Type outDataType,
                 poplar::Type partialDataType,
                 SubBlockMask subBlockMask = SubBlockMask::None,
                 unsigned numGroupsIn = 1);

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
 * \param debugContext    Optional debug information.
 *
 * \param options         matmul options, see bsMatmul for details
 *
 * \returns               For non-grouped BSMatMulParams object,
 *                        if the left matrix is a dense matrix, the return
 *                        tensor is just a regular 2D matrix. If it is a sparse
 *                        matrix, the return tensor is an array of non-zero
 *                        blocks.
 *                        For group BSMatMulParams object,
 *                        the return tensor is concatenated along 0 dimension
 *                        for all ops in a group.
 *                        tensor for all matrices in a group.
 */
poplar::Tensor createBSMatMulInputLHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const poplar::DebugContext &debugContext,
                                      const poplar::OptionFlags &options = {});

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
 * \param debugContext    Optional debug information.
 *
 * \param options         matmul options, see bsMatmul for details
 *
 * \returns               For non-grouped BSMatMulParams object,
 *                        if the right matrix is a dense matrix, the return
 *                        tensor is just a regular 2D matrix. If it is a sparse
 *                        matrix, the return tensor is an array of non zero
 *                        blocks.
 *                        For group BSMatMulParams object,
 *                        the return tensor is concatenated along 0 dimension
 *                        for all ops in a group.
 *                        tensor for all matrices in a group.
 */
poplar::Tensor createBSMatMulInputRHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const poplar::DebugContext &debugContext,
                                      const poplar::OptionFlags &options = {});

/* This function multiplies the left-hand matrix by the right-hand matrix.
 *
 * \param graph         The Poplar graph.
 *
 * \param bsMatMul        The object for block sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type
 *
 * \param prog            A reference to a program sequence which will
 *                        be appended with the code to perform the
 *                        multiplication.
 *
 * \param lhsMatrix       if BSMatMulParams is for dense x sparse, this is
 *                        the left hand dense matrix,
 *                        if BSMatMulParams is for sparse x sparse, this is
 *                        the non zero blocks of the left sparse matrix.
 *                        For group BSMatMulParams object,
 *                        it should be concatenated along 0 dimension
 *                        for all tensors in a group.
 *
 * \param rhsMatrix       a tensor for an array of non zero blocks in the right
 *                        hand sparse matrix.
 *                        For group BSMatMulParams object,
 *                        it should be concatenated along 0 dimension
 *                        for all tensors in a group.
 *
 * \param options         The structure describing options on how the
 *                        multiplication should be implemented.
 *
 *                        option "memoryCycleRatio" is for computing the
 *                        weight of hyper graph node. This may be only a
 *                        temporary option
 *                           w = memory_cycle_ratio * mem_weight +
 *                               (1.0 - memory_cycle_ratio) * cycle_weight
 *
 *                        option "numberOfPass": the number of passes to
 *                        serialize the matmul
 *
 *                        option "partitionMethod":
 *                          if it is "block", the matmul computation graph is
 *                          created for each non zero blocks, zoltan is used
 *                          to partition the graph;
 *                          if it is "block-naive", the matmul computation
 *                          graph is created for each non zero blocks, a
 *                          greedy algorithm is used to partition the graph;
 *                          if it is "strip", the graph is created for columns
 *                          or rows.
 *
 * \param debugContext    Optional debug information.
 * \returns               The tensor holding the result of the
 *                        multiplication. This tensor will be created, added to
 *                        the graph and mapped to tiles.
 *                        For group BSMatMulParams object,
 *                        the return tensor is concatenated along 0 dimension
 *                        for all ops in a group.
 */
poplar::Tensor bsMatMul(poplar::Graph &graph, const BSMatMulParams &bsMatMul,
                        poplar::program::Sequence &prog,
                        const poplar::Tensor &lhsMatrix,
                        const poplar::Tensor &rhsMatrix,
                        const poplar::OptionFlags &options = {},
                        const poplar::DebugContext &debugContext = {});

} // namespace experimental
} // namespace popsparse

#endif // POPSPARSE_BLOCK_SPARSE_MATMUL_H
