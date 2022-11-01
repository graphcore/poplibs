// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Sparse matrix multiply operations.
 */

#ifndef popsparse_MatMul_hpp
#define popsparse_MatMul_hpp

#include <poplar/Graph.hpp>
#include <popsparse/PlanningCache.hpp>
#include <popsparse/SparseStorageFormats.hpp>
#include <popsparse/SparseTensor.hpp>

namespace popsparse {
namespace dynamic {

class MatMulParams;

/**
 * Create a sparse tensor that is used as the left-hand operand in a
 * sparse * dense matrix multiplication.
 *
 * The following options are available:
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 [=0.6]
 *
 *      The maximum proportion of available memory on each tile that this
 *      layer should consume temporarily during the course of the operation.
 *
 *    * `metaInfoBucketOversizeProportion` Decimal between 0 and 1 [=0.3]
 *
 *      This specifies additional elements to allocate in each bucket of
 *      meta-information as a proportion of the required size for a perfectly
 *      uniformly distributed sparsity pattern.
 *
 *    * `partialsType` poplar::Type [=poplar::FLOAT]
 *
 *      The type to use for partial results.
 *
 *    * `sharedBuckets` (true, false) [=true]
 *
 *      If set, forces the same buckets to be used whether or not the
 *      sparse (left-hand) operand is transposed or not. Saves memory
 *      at the expense of runtime.
 *
 * \param graph     The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params    Parameters for the matrix multiplication.
 * \param debugContext Optional debug information.
 * \param options   Implementation options for the matrix multiplication.
 * \param cache     Optional pointer to planning cache to use.
 *
 * \returns         A sparse tensor with sparse representation of left-hand
 *                  operand for the matrix multiplication.
 */
SparseTensor createSparseDenseMatMulLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

/**
 * Create a dense tensor that is used as the right-hand operand in a
 * sparse * dense matrix multiplication.
 *
 * \param graph     The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params    Parameters for the matrix multiplication.
 * \param debugContext Optional debug information.
 * \param options   Implementation options for the matrix multiplication.
 * \param cache     Optional pointer to planning cache to use.
 *
 * \returns         A dense tensor for use as right-hand operand for the matrix
 *                  multiplication.
 */
poplar::Tensor createSparseDenseMatMulRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

/**
 * Perform a sparse * dense matrix multiplication, yielding a dense result.
 *
 * The sparse left-hand operand tensor is made up of meta information for the
 * sparsity and the non-zero values of the matrix. This sparse tensor must have
 * been created with createSparseDenseMatMulLHS.
 *
 * If the sparse left-hand operand was created for the sparse equivalent of a
 * dense matrix multiplication:
 *
 *   [groups][m][k] * [groups][k][n] = [groups][m][n]
 *
 * Then the same sparse left-hand operand can be used to calculate the above
 * as well as:
 *
 *   [groups][k][m] * [groups][m][n] = [groups][k][n]
 *
 * through the use of the \p transposeLHS parameter. \p transposeRHS is also
 * provided for convenience.
 *
 *
 * \param graph         The Poplar graph.
 * \param lhs           The sparse left-hand operand to the matrix
 *                      multiplication.
 * \param rhs           The dense right-hand operand to the matrix
 *                      multiplication.
 * \param prog          A reference to a program sequence which will be
 *                      appended with the code to perform the matrix
 *                      multiplication.
 * \param transposeLHS  Whether or not to transpose the left-hand operand
 *                      before multiplying.
 * \param transposeRHS  Whether or not to transpose the right-hand operand
 *                      before multiplying.
 * \param debugContext  Optional debug information.
 * \param options       Implementation options for the matrix multiplication.
 * \param cache         Optional pointer to planning cache to use.
 *
 * \returns             The tensor holding the dense result of the matrix
 *                      multiplication. The tensor will be created, added to
 *                      the graph, and mapped to tiles.
 */
poplar::Tensor sparseDenseMatMul(
    poplar::Graph &graph, const SparseTensor &lhs, const poplar::Tensor &rhs,
    poplar::program::Sequence &prog, bool transposeLHS = false,
    bool transposeRHS = false, const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);
} // end namespace dynamic

namespace static_ {

class MatMulParams;
/**
 * Create a sparse tensor that is used as the left-hand operand in a
 * [sparse * dense] matrix multiplication. The matrix multiplication performed
 * is
 *         [groups][m][k] * [groups][k][n] = [groups][m][n]
 *
 *            sparse      *    dense       = dense
 *
 * The following options are available:
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 [=0.6]
 *
 *      The maximum proportion of available memory on each tile that this
 *      layer should consume temporarily during the course of the operation.
 *
 * Partials type is restricted to have the same type as data.
 *
 * \param graph     The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params    Parameters for the matrix multiplication.
 * \param csrLHS    CSR representation of the left hand side sparse matrix.
 * \param debugContext Optional debug information.
 * \param options   Implementation options for the matrix multiplication.
 * \param cache     Optional pointer to planning cache to use.
 *
 * \returns         A sparse tensor with sparse representation of left-hand
 *                  operand for the matrix multiplication. The sparse tensor is
 *                  mapped and it's non-zero values can be copied from the
 *                  host using an host equivalent created by using a
 * partitioner.
 */
template <typename T>
SparseTensor
createSparseDenseMatMulLHS(poplar::Graph &graph, const poplar::Type &inputType,
                           const MatMulParams &params,
                           const CSRMatrix<T> &csrLHS,
                           const poplar::DebugContext &debugContext = {},
                           const poplar::OptionFlags &options = {},
                           PlanningCache *cache = nullptr) = delete;
template <>
SparseTensor createSparseDenseMatMulLHS<float>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<float> &csrLHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache);
template <>
SparseTensor createSparseDenseMatMulLHS<double>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<double> &csrLHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache);

/**
 * Create a sparse tensor that is used as the right-hand operand in a
 * dense * sparse matrix multiplication. The matrix multiplication performed
 * is
 *         [groups][n][k] * [groups][k][m] = [groups][n][m]
 *
 *           dense        *    sparse      = dense
 *
 * Partials type is restricted to have the same type as data.
 *
 * \param graph     The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params    Parameters for the matrix multiplication.
 * \param csrRHS    CSR representation of the right hand side sparse matrix.
 * \param debugContext Optional debug information.
 * \param options   Implementation options for the matrix multiplication.
 *                  see static_::createSparseDenseMatMulLHS()
 * \param cache     Optional pointer to planning cache to use.
 *
 * \returns         A sparse tensor with sparse representation of right-hand
 *                  operand for the matrix multiplication.  The sparse tensor
 *                  is mapped and it's non-zero values can be copied from
 *                  the host to the using a host equivalent created by using a
 *                  partitioner static_::partitioner()
 */
template <typename T>
SparseTensor
createDenseSparseMatMulRHS(poplar::Graph &graph, const poplar::Type &inputType,
                           const MatMulParams &params,
                           const CSRMatrix<T> &csrRHS,
                           const poplar::DebugContext &debugContext = {},
                           const poplar::OptionFlags &options = {},
                           PlanningCache *cache = nullptr) = delete;
template <>
SparseTensor createDenseSparseMatMulRHS<float>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<float> &csrRHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache);
template <>
SparseTensor createDenseSparseMatMulRHS<double>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<double> &csrRHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache);

/**
 * Create a dense tensor that is used as the right-hand operand in a
 * sparse * dense matrix multiplication. The matrix multiplication performed
 * is
 *         [groups][m][k] * [groups][k][n] = [groups][m][n]
 *
 *            sparse      *    dense       = dense
 *
 * Partials type is restricted to have the same type as data.
 *
 * \param graph     The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params    Parameters for the matrix multiplication. The parameters
 *                  should be created for a sparse * dense multiplication.
 * \param csrLHS    CSR representation of the left hand side sparse matrix.
 * \param debugContext Optional debug information.
 * \param options   Implementation options for the matrix multiplication.
 *                  see static_::createSparseDenseMatMulLHS()
 * \param cache     Optional pointer to planning cache to use.
 * \returns         A dense tensor for use as right-hand operand for the matrix
 *                  multiplication. The returned dense tensor is of shape
 *                  [groups][m][n]. The tensor will be created, added to the
 *                  graph, and mapped to tiles.
 */
template <typename T>
poplar::Tensor
createSparseDenseMatMulRHS(poplar::Graph &graph, const poplar::Type &inputType,
                           const MatMulParams &params,
                           const CSRMatrix<T> &csrLHS,
                           const poplar::DebugContext &debugContext = {},
                           const poplar::OptionFlags options = {},
                           PlanningCache *cache = nullptr) = delete;
template <>
poplar::Tensor createSparseDenseMatMulRHS<float>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<float> &csrLHS,
    const poplar::DebugContext &debugContext, const poplar::OptionFlags options,
    PlanningCache *cache);
template <>
poplar::Tensor createSparseDenseMatMulRHS<double>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<double> &csrLHS,
    const poplar::DebugContext &debugContext, const poplar::OptionFlags options,
    PlanningCache *cache);

/**
 * Create a dense tensor that is used as the left-hand operand in a
 * dense * sparse matrix multiplication.
 *
 *         [groups][n][k] * [groups][k][m] = [groups][n][m]
 *
 *           dense        *    sparse      = dense
 *
 * Partials type is restricted to have the same type as data.
 *
 * \param graph     The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params    Parameters for the matrix multiplication. Must be created
 *                  for Dense * Sparse multiplication.
 * \param csrRHS    CSR representation of the right hand side sparse matrix
 * \param debugContext Optional debug information.
 * \param options   Implementation options for the matrix multiplication.
 *                  see static_::createSparseDenseMatMulLHS()
 * \param cache     Optional pointer to planning cache to use.
 * \returns         A dense tensor for use as right-hand operand for the matrix
 *                  multiplication. The returned dense tensor is of shape
 *                  [groups][n][m]. The tensor will be created, added to the
 *                  graph, and mapped to tiles.
 */
template <typename T>
poplar::Tensor
createDenseSparseMatMulLHS(poplar::Graph &graph, const poplar::Type &inputType,
                           const MatMulParams &params,
                           const CSRMatrix<T> &csrRHS,
                           const poplar::DebugContext &debugContext = {},
                           const poplar::OptionFlags options = {},
                           PlanningCache *cache = nullptr) = delete;
template <>
poplar::Tensor createDenseSparseMatMulLHS<float>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<float> &csrRHS,
    const poplar::DebugContext &debugContext, const poplar::OptionFlags options,
    PlanningCache *cache);
template <>
poplar::Tensor createDenseSparseMatMulLHS<double>(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<double> &csrRHS,
    const poplar::DebugContext &debugContext, const poplar::OptionFlags options,
    PlanningCache *cache);

/**
 * Perform a sparse * dense matrix multiplication, yielding a dense result.
 *
 * The sparse left-hand operand is a sparse representation of the matrix
 *
 * If the sparse left-hand operand was created for the sparse equivalent of a
 * dense matrix multiplication:
 *
 *   [groups][m][k] * [groups][k][n] = [groups][m][n]
 *
 * Then the same sparse left-hand operand can be used to calculate the above
 * as well as:
 *
 *   [groups][k][m] * [groups][m][n] = [groups][k][n]
 *
 * through the use of the \p transposeLHS parameter. \p transposeRHS is also
 * provided for convenience.
 *
 * Partials type is restricted to have the same type as data.
 *
 * Note: Transposition of the sparse LHS operand is not yet supported.
 *
 * \param graph         The Poplar graph.
 * \param lhs           The sparse left-hand operand to the matrix
 *                      multiplication.
 * \param rhs           The dense right-hand operand to the matrix
 *                      multiplication.
 * \param prog          A reference to a program sequence which will be
 *                      appended with the code to perform the matrix
 *                      multiplication.
 * \param transposeLHS  Whether or not to transpose the left-hand operand
 *                      before multiplying.
 * \param transposeRHS  Whether or not to transpose the right-hand operand
 *                      before multiplying.
 * \param debugContext  Optional debug information.
 * \param options       Implementation options for the matrix multiplication.
 *                      see static_::createSparseDenseMatMulLHS()
 * \param cache         Optional pointer to planning cache to use.
 *
 * \returns             The tensor holding the dense result of the matrix of
 *                      multiplication. The tensor will be created, added to
 *                      the graph, and mapped to tiles.
 */
poplar::Tensor sparseDenseMatMul(
    poplar::Graph &graph, const SparseTensor &lhs, const poplar::Tensor &rhs,
    poplar::program::Sequence &prog, bool transposeLHS = false,
    bool transposeRHS = false, const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

/**
 * Perform a dense * sparse matrix multiplication, yielding a dense result.
 * The sparse right-hand operand is a sparse representation of the matrix.
 *
 * If the sparse right-hand operand was created for the sparse equivalent of a
 * dense matrix multiplication:
 *
 *   [groups][n][k] * [groups][k][m] = [groups][n][k]
 *
 * Then the same sparse left-hand operand can be used to calculate the above
 * as well as:
 *
 *   [groups][k][m] * [groups][m][n] = [groups][k][n]
 *
 * through the use of the \p transposeLHS parameter. \p transposeRHS is also
 * provided for convenience.
 *
 * Partials type is restricted to have the same type as data.
 *
 * Note: Transposition of the sparse right hand operand is not yet supported.
 *
 * \param graph         The Poplar graph.
 * \param lhs           The dense left-hand operand to the matrix
 *                      multiplication.
 * \param rhs           The sparse right-hand operand to the matrix
 *                      multiplication.
 * \param prog          A reference to a program sequence which will be
 *                      appended with the code to perform the matrix
 *                      multiplication.
 * \param transposeLHS  Whether or not to transpose the left-hand operand
 *                      before multiplying.
 * \param transposeRHS  Whether or not to transpose the right-hand operand
 *                      before multiplying.
 * \param debugContext  Optional debug information.
 * \param options       Implementation options for the matrix multiplication.
 *                      see static_::createSparseDenseMatMulLHS()
 * \param cache         Optional pointer to planning cache to use.
 *
 * \returns             The tensor holding the dense result of the matrix
 *                      multiplication. The tensor will be created, added to
 *                      the graph, and mapped to tiles.
 */
poplar::Tensor denseSparseMatMul(
    poplar::Graph &graph, const poplar::Tensor &lhs, const SparseTensor &rhs,
    poplar::program::Sequence &prog, bool transposeLHS = false,
    bool transposeRHS = false, const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

} // namespace static_

} // end namespace popsparse

#endif // popsparse_MatMul_hpp
