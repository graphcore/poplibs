// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef POPSPARSE_BLOCK_SPARSE_MATMUL_H
#define POPSPARSE_BLOCK_SPARSE_MATMUL_H

#include <array>
#include <poplar/Graph.hpp>

namespace popsparse {
namespace experimental {

class BSMatMulImpl;

/*
 * BSMatMulParams is to support block sparse matrix multiplication
 *
 * The class only saves the sparsity mask, the matrix size, the block size,
 * and the data type, which are used to generate the computation graph.
 *
 * The matrix data is passed in when function bsMatMul() or bsUpdate() gets
 * called.
 *
 * The purpose of this design is to reuse the instance of this class when only
 * the data of the matrix is changed, and the matrix sparsity does not change.
 *
 * The current implementation is based on Zoltan to generate the hyper graph
 * partition for all tiles. Zoltan usually runs 2 minutes for ~16k non zero
 * blocks, which is expensive if it runs for every matrix multiplication.
 *
 * The right matrix is always sparse, and the left matrix can be dense or sparse
 */
class BSMatMulParams {

public:
  /* This construct is for dense matrix (left side) multiply sparse matrix
   * (right side).
   *
   * \param dim          dim[0] : number of rows of left hand matrix
   *                     dim[1] : number of columns of left hand matrix
   *                     dim[2] : if right matrix need be transposed, it is
   *                              number of rows of right hand matrix; otherwise
   *                              it is number of columns of right hand matrix
   *
   * \param blockSize       blockSize[0] :block size of the row of left hand
   *                        blockSize[1] :block size of the column of left hand
   *                        blockSize[2] :block size of the column of right hand
   *
   * \param rhsSparsity     The 2D sparsity mask for right hand block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block
   *
   * \param rhsNeedTranspose   Whether the right hand matrix need be transposed
   *                           This is to support backward pass
   *
   * \param inDataType      input data type
   *
   * \param outDataType     output data type
   *
   * \param partialDataType partial data type
   *
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &rhsSparsity,
                 bool rhsNeedTranspose, poplar::Type inDataType,
                 poplar::Type outDataType, poplar::Type partialDataType);

  /* This construct is for sparse matrix multiply sparse matrix.
   * It is not supported yet, we can not find this use case in AI models
   *
   * \param dim          dim[0] : number of rows of left hand matrix
   *                     dim[1] : number of columns of left hand matrix
   *                     dim[2] : it is the number of rows or columns of the
   *                              right hand matrix, depends on the transpose
   *                              flag for each side.
   *
   * \param blockSize       blockSize[0] :block size of the row of left hand
   *                        blockSize[1] :block size of the column of left hand
   *                        blockSize[2] :block size of the column of right hand
   *
   * \param lhsSparsity     The 2D sparsity mask for left hand block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block
   *
   * \param lhsNeedTranspose   Whether the left hand matrix need be transposed
   *                           This is to support backward pass
   *
   * \param rhsSparsity     The 2D sparsity mask for right hand block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block
   *
   * \param rhsNeedTranspose   Whether the right hand matrix need be transposed
   *                           This is to support backward pass
   *
   * \param inDataType      input data type
   *
   * \param outDataType     output data type
   *
   * \param partialDataType partial data type
   *
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &lhsSparsity,
                 bool lhsNeedTranspose,
                 const std::vector<unsigned char> &rhsSparsity,
                 bool rhsNeedTranspose, poplar::Type inDataType,
                 poplar::Type outDataType, poplar::Type partialDataType);

  /* This construct is for dense matrix multiply dense matrix,
   *  in sparse way and store the result as a sparse matrix
   *
   * (right side).
   *
   * \param dim          dim[0] : number of rows of left hand matrix
   *                     dim[1] : number of columns of left hand matrix
   *                     dim[2] : number of columns of right hand matrix
   *
   * \param blockSize       blockSize[0] :block size of the row of left hand
   *                        blockSize[1] :block size of the column of left hand
   *                        blockSize[2] :block size of the col of right
   *                                      hand.The block size of the column of
   *                                      left hand equals to the block size
   *                                      of the row of right hand
   *
   * \param resSparsity     The 2D sparsity mask for the result block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block
   *
   * \param inDataType      input data type
   *
   * \param outDataType     output data type
   *
   * \param partialDataType partial data type
   *
   */
  BSMatMulParams(const std::array<int, 3> &dim,
                 const std::array<int, 3> &blockSize,
                 const std::vector<unsigned char> &resSparsity,
                 poplar::Type inDataType, poplar::Type outDataType,
                 poplar::Type partialDataType);

  BSMatMulParams(BSMatMulParams &&other);

  ~BSMatMulParams();

  std::unique_ptr<BSMatMulImpl> impl;
};

/**
 * Create a tensor that is used as the left operand of block sparse matrix
 * multiplication.
 *
 * \param graph           The Poplar graph.
 *
 * \param bsMatMul        The object for block sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type
 *
 * \param name            The debug name of the required matrix.
 *
 * \returns               If the left matrix is a dense matrix, the return
 *                        tensor is just a regular 2D matrix. If it is a sparse
 *                        matrix, the return tensor is an array of non zero
 *                        blocks.
 */
poplar::Tensor createBSMatMulInputLHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name);

/**
 * Create a tensor that is used as the right operand of block sparse matrix
 * multiplication.
 *
 * \param graph           The Poplar graph.
 *
 * \param bsMatMul        The object for block sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type
 *
 * \param name            The debug name of the required matrix.
 *
 * \returns               The return tensor is an array of non zero
 *                        blocks for block sparse matrix
 */
poplar::Tensor createBSMatMulInputRHS(poplar::Graph &graph,
                                      const BSMatMulParams &bsMatMul,
                                      const std::string &name);

/* This function is to multiply left hand matrix by the right hand matrix
 *
 * \param graph           The Poplar graph
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
 *                        the non zero blocks of the left sparse matrix
 *
 * \param rhsMatrix       a tensor for an array of non zero blocks in the right
 *                        hand sparse matrix
 *
 * \param options         The structure describing options on how the
 *                        multiplication should be implemented.
 *                        option "memory_cycle_ratio" is for computing the
 *                        weight of hyper graph node. This may be only a
 *                        temporary option
 *                           w = memory_cycle_ratio * mem_weight +
 *                               (1.0 - memory_cycle_ratio) * cycle_weight
 *
 * \param debugPrefix     A debug prefix added to compute set and tensor
 *                        names.
 *
 * \returns               The tensor holding the result of the
 *                        multiplication. This tensor will be created, added to
 *                        the graph and mapped to tiles.
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
