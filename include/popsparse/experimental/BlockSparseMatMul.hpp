// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef POPSPARSE_BLOCK_SPARSE_MATMUL_H
#define POPSPARSE_BLOCK_SPARSE_MATMUL_H

#include <poplar/Graph.hpp>

namespace popsparse {
namespace experimental {

/*
 * BlockSparseMatMul is to support block sparse matrix multiplication
 *
 * The class only saves the sparsity mask, the matrix size, the block size,
 * and the data type, which are used to generate the computation graph.
 *
 * The matrix data is passed in when function bsMatMul() or bsUpdate() got
 * called.
 *
 * The purpose of this design is to reuse the instance of this class when only
 * the data of the matrix is changed, and the matrix structure does not change.
 *
 * The current implementaztion is based on Zoltan to generate the hyper graph
 * partition for all tiles. Zoltan usually runs 2 minutes for ~16k non zero
 * blocks, which is expensive if it runs for every matrix multiplication.
 */
class BlockSparseMatMul {

public:
  /* This construct is for dense matrix (left side) multiply sparse matrix
   * (right side).
   *
   * \param graph           The Poplar graph, which is needed here to get some
   *                        information of IPU, such as number of tiles
   *
   * \param lhsDim          lhsDim[0] : number of rows of left hand matrix
   *                        lhsDim[1] : number of columns of left hand matrix
   *
   * \param rhsSparsity     The 2D sparsity mask for right hand block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block
   *
   * \param rhsDim          rhsDim[0] : number of rows of right hand matrix
   *                        rhsDim[1] : number of columns of right hand matrix
   *
   * \param rhsTransposed   Whether the right hand matrix need be transposed
   *                        This is to support backward pass
   *
   * \param blockSize       blockSize[0] :block size of the row of right hand
   *                        blockSize[1] :block size of the column of right hand
   *                        blockSize[2] :block size of the row of left
   *                                      hand.The block size of the column of
   *                                      left hand equals to the block size
   *                                      of the row of right hand
   *
   * \param inDataType      input data type
   *
   * \param outDataType     output data type
   *
   * \param partialDataType partial data type
   */
  BlockSparseMatMul(poplar::Graph &graph, uint lhsDim[2],
                    unsigned char *rhsSparisity, uint rhsDim[2],
                    bool rhsTransposed, uint blockSize[3],
                    poplar::Type inDataType, poplar::Type outDataType,
                    poplar::Type partialDataType);

  /* This construct is for sparse matrix multiply sparse matrix.
   *
   * \param graph           The Poplar graph, which is needed here to get some
   *                        information of IPU, such as number of tiles
   *
   * \param lhsSparsity     The 2D sparsity mask for left hand block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block
   *
   * \param lhsDim          lhsDim[0] : number of rows of left hand matrix
   *                        lhsDim[1] : number of columns of left hand matrix
   *
   * \param lhsTransposed   Whether the left hand matrix need be transposed
   *
   * \param rhsSparsity     The 2D sparsity mask for right hand block sparse
   *                        matrix, in which '1' is a non zero block and '0'
   *                        is a zero block
   *
   * \param rhsDim          rhsDim[0] : number of rows of right hand matrix
   *                        rhsDim[1] : number of columns of right hand matrix
   *
   * \param rhsTransposed   Whether the right hand matrix need be transposed
   *
   * \param blockSize       blockSize[0] : block size of the row of left hand
   *                        blockSize[1] : block size of the column of left hand
   *                        blockSize[2] : block size of the column of right
   *                                       hand. The block size of the column of
   *                                       left hand equals to the block size of
   *                                       the row of right hand
   *
   * \param inDataType      input data type
   *
   * \param outDataType     output data type
   *
   * \param partialDataType partial data type
   */
  BlockSparseMatMul(poplar::Graph &graph, unsigned char *lhsSparisity,
                    uint lhsDim[2], bool lhsTransposed,
                    unsigned char *rhsSparisity, uint rhsDim[2],
                    bool rhsTransposed, uint blockSize[3],
                    poplar::Type inDataType, poplar::Type outDataType,
                    poplar::Type partialDataType);
};

/* This function is to multiply left hand matrix by the right hand matrix
 *
 * \param bsMatMul        The struct for block sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type
 *
 * \param graph           The Poplar graph, which is needed here to get some
 *                        information of IPU, such as number of tiles
 *
 * \param lhsMatrix       if BlockSparseMatMul is for dense x sparse, this is
 *                        the left hand dense matrix,
 *                        if BlockSparseMatMul is for sparse x sparse, this is
 *                        the non zero blocks of the left sparse matrix
 *
 * \param rhsMatrix       a 2D tensor for non zero blocks in the right hand
 *                        sparse matrix
 *
 * \param options         The structure describing options on how the
 *                        multiplication should be implemented.
 *
 * \param debugPrefix     A debug prefix added to compute set and tensor
 *                        names.
 *
 * \returns               The tensor holding the result of the
 *                        multiplication. This tensor will be created, added to
 *                        the graph and mapped to tiles.
 */
poplar::Tensor bsMatMul(BlockSparseMatMul &bsMatMul, poplar::Graph &graph,
                        poplar::Tensor &lhsMatrix, poplar::Tensor &rhsMatrix,
                        const poplar::OptionFlags &options = {},
                        const std::string &debugPrefix = "");

/* This function is to update the weight of the sparse matrix
 *
 * \param bsMatMul        The struct for block sparse information, includes the
 *                        sparsity mask, the matrix size, the block size,
 *                        and the data type
 *
 * \param graph           The Poplar graph, which is needed here to get some
 *                        information of IPU, such as number of tiles
 *
 * \param matrixBlocks    a 2D tensor for non zero blocks in the sparse matrix
 *                        it need be consistent with the sparsity saved in
 *                        this class
 *
 * \param dw              The gradient for the sparse matrix
 *                        Note: this is a dense matrix since the gradient is
 *                              computed as dy * x(transpose). The weight, which
 *                              is outside non-zero blocks, will be ignored.
 *
 * \param options         The structure describing options on how the
 *                        weight update should be implemented.
 *
 * \param debugPrefix     A debug prefix added to compute set and tensor names
 */
void bsUpdate(BlockSparseMatMul &bsMatMul, poplar::Graph &graph,
              poplar::Tensor &matirxBlocks, poplar::Tensor &dw,
              const poplar::OptionFlags &options = {},
              const std::string &debugPrefix = "");

} // namespace experimental
} // namespace popsparse

#endif // POPSPARSE_BLOCK_SPARSE_MATMUL_H