// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions and data types for performing matrix multiplies on the IPU.
 *
 */

#ifndef poplin_MatMul_hpp
#define poplin_MatMul_hpp
#include "poplin/Convolution.hpp"
#include <iosfwd>
#include <map>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace poplin {
namespace matmul {

class PlanningCache;

} // namespace matmul

/** Multiply two matrices.
 *  Calculates `C = A * B` where \p A and \p B are matrices.
 *
 *  **Matrix multiply options**
 *    * `availableMemoryProportion` Decimal between 0 and 1 (inclusive) [=0.6]
 *
 *      See createWeights().
 *
 *    * `fullyConnectedPass` (NONE, INFERENCE_FWD, TRAINING_FWD, TRAINING_BWD,
 *      TRAINING_WU) [=NONE]
 *
 *      Optimize the plan for the specified type of pass. Note the
 *      abbreviations: FWD (forward), BWD (backward), WU (weight-update).
 *
 *    * `inputRHSIsPreArranged` (true, false) [=false]
 *
 *      Indicates to matMul functions whether the input data has already been
 *      re-arranged (using preArrangeMatMulInputRHS()). This allows data to be
 *      re-arranged once then used many times.
 *
 *    * `use128BitConvUnitLoad` (true, false) [=false]
 *
 *      If true, weights are loaded into the convolution unit 128-bits at a
 *      time. Otherwise, they are loaded 64-bits at a time. Not all codelets
 *      support 128-bit loads. This option affects memory usage and cycle count.
 *
 *    * `enableMultiStageReduce` (true, false) [=true]
 *
 *      If true, perform the reduction following the matrix multiplication in
 *      multiple stages if it would significantly reduce code size. This comes
 *      at the cost of increasing the number of cycles.
 *
 *    * `enableFastReduce` (true, false) [=false]
 *
 *      If true, use a faster reduction vertex if the data types and widths
 *      allow it.  This comes at the cost of further constraints on memory
 *      allocation
 *
 *    * `remapOutputTensor`       (true, false) [=true]
 *
 *       If true, the output of the convolution is remapped if the output
 *       is detected to have a poor layout.
 *
 *    * `partialsType` (half, float) [=float]
 *
 *      See createWeights().
 *
 *  \param graph           The Poplar graph.
 *  \param A               The left argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param B               The right argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         multiplication.
 *  \param outputType      Optional via overloaded function. Element type of
 *                         returned tensor. The default is \p A.elementType()
 *                         if omitted.
 *  \param debugContext    Optional debug information.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented.
 *  \param cache           Optional pointer to a planning cache to use.
 *
 *  \returns               The tensor holding the result of the multiplication.
 *                         This tensor will be created, added to the graph and
 *                         mapped to tiles.
 */
/*[INTERNAL OPTIONS]
 *    * `planConstraints` JSON string
 *
 *      See createWeights().
 *
 *    * `useAggressiveRegrouping` (true, false) [=false]
 *
 *      See createWeights().
 *
 *   * `gatherOutput` (true, false) [=false]
 *     Gather output of the matrix multipy into a single variable
 */
/** Matrix multiply with explicitly defined output type. */
poplar::Tensor matMul(poplar::Graph &graph, const poplar::Tensor &A,
                      const poplar::Tensor &B, poplar::program::Sequence &prog,
                      const poplar::Type &outputType,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      matmul::PlanningCache *cache = nullptr);

/** Matrix multiply where output type is the same as input \p A. */
poplar::Tensor matMul(poplar::Graph &graph, const poplar::Tensor &A,
                      const poplar::Tensor &B, poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      matmul::PlanningCache *cache = nullptr);

/** Report the convolution plan corresponding to the parameters and options
 * provided.
 *
 *  \param out             Stream to write report to.
 *  \param graph           The Poplar graph.
 *  \param inputType       Element type of input tensors.
 *  \param outputType      Element type of output tensor.
 *  \param aShape          Shape of input tensor A.
 *  \param bShape          Shape of input tensor B.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented.
 *  \param cache           Optional pointer to a planning cache to use.
 */
void matMulReportPlan(std::ostream &out, const poplar::Graph &graph,
                      const poplar::Type &inputType,
                      const poplar::Type &outputType,
                      const std::vector<std::size_t> &aShape,
                      const std::vector<std::size_t> &bShape,
                      const poplar::OptionFlags &options = {},
                      matmul::PlanningCache *cache = nullptr);

/** Multiply two grouped matrices.
 *
 *  Calculates `C[g] = A[g] * B[g]` where `A[g]` and `B[g]` are matrices for
 *  each element in the group, and `g` is an element of the set {0, 1, ...,
 *  `G`-1}.
 *
 *  The multiplication is done for every element in the group. The first
 *  dimension of the matrices is the group dimension with value equal to G.
 *
 *  \param graph           The Poplar graph.
 *  \param A               The left argument to the grouped multiplication. This
 *                         3D tensor must be already mapped to tiles.
 *  \param B               The right argument to the grouped multiplication.
 *                         This 3D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         multiplication.
 *  \param outputType      Data type to be used for the returned tensor.
 *  \param debugContext    Optional debug information.
 *  \param options         The structure describing options on how the
 *                         grouped multiplication should be implemented. See
 *                         matMul().
 *  \param cache           Optional pointer to a planning cache to use.
 *
 *  \returns               The tensor holding the result of the grouped
 *                         multiplication. This tensor will be created, added to
 *                         the graph and mapped to tiles.
 */
poplar::Tensor matMulGrouped(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B,
                             poplar::program::Sequence &prog,
                             const poplar::Type &outputType,
                             const poplar::DebugContext &debugContext = {},
                             const poplar::OptionFlags &options = {},
                             matmul::PlanningCache *cache = nullptr);

/** Report the convolution plan corresponding to the \p params and \p options
 * provided.
 *
 *  \param out             Stream to write report to.
 *  \param graph           The Poplar graph.
 *  \param inputType       Element type of input tensors.
 *  \param outputType      Element type of output tensor.
 *  \param aShape          Shape of input tensor A.
 *  \param bShape          Shape of input tensor B.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented.
 *  \param cache           Optional pointer to a planning cache to use.
 */
void matMulGroupedReportPlan(std::ostream &out, const poplar::Graph &graph,
                             const poplar::Type &inputType,
                             const poplar::Type &outputType,
                             const std::vector<std::size_t> &aShape,
                             const std::vector<std::size_t> &bShape,
                             const poplar::OptionFlags &options = {},
                             matmul::PlanningCache *cache = nullptr);

/** Multiply two matrices and add to a third (with a scaling factor).
 *
 *  Calculates `C += k * A * B` where \p A, \p B are matrices and \p k is a
 *  constant scalar.
 *
 *  \param graph           The Poplar graph.
 *  \param C               The matrix to add to. This
 *                         2D tensor must be already mapped to tiles.
 *  \param k               The constant or a single element tensor to multiply
 *                         the result of the multiplication. If \p k is a
 *                         tensor, it must be of the same type as \p A
 *  \param A               The left argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param B               The right argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         multiplication and add.
 *  \param debugContext    Optional debug information.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented. See matMul().
 *  \param cache           Optional pointer to a planning cache to use.
 */
/** Matrix multiply and accumulate with a scalar scaling factor. */
void matMulAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
               const poplar::Tensor &A, const poplar::Tensor &B,
               poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext = {},
               const poplar::OptionFlags &options = {},
               matmul::PlanningCache *cache = nullptr);

/** Matrix multiply and accumulate with a single-element scaling factor. */
void matMulAcc(poplar::Graph &graph, const poplar::Tensor &C,
               const poplar::Tensor &k, const poplar::Tensor &A,
               const poplar::Tensor &B, poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext = {},
               const poplar::OptionFlags &options = {},
               matmul::PlanningCache *cache = nullptr);

/** Grouped matrix multiply and accumulate.
 *
 *  Multiply two grouped matrices and add to a third (with a scaling factor).
 *
 *  Calculates `C[g] += k * A[g] * B[g]` where `A[g]`, `B[g]` are matrices and
 *  \p k is a constant scalar. g is element of the set g = {0, 1, ..., G-1}
 *
 *  The multiplication is done for every element in the group. The first
 *  dimension of the matrices is the group dimension with value equal to G.
 *
 *  \param graph           The Poplar graph.
 *  \param C               The matrix to add to. This
 *                         3D tensor must be already mapped to tiles.
 *  \param k               The constant or a single element tensor to multiply
 *                         the result of the multiplication. If \p k is a
 *                         tensor, it must be of the same type as \p A
 *  \param A               The left argument to the grouped multiplication. This
 *                         3D tensor must be already mapped to tiles.
 *  \param B               The right argument to the multiplication. This
 *                         3D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the grouped
 *                         multiplication and add.
 *  \param debugContext    Optional debug information.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented. See matMul().
 *  \param cache           Optional pointer to planning cache to use.
 */
/** Grouped matrix multiply and accumulate with a scalar scaling factor.
 */
void matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
                      const poplar::Tensor &A, const poplar::Tensor &B,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      matmul::PlanningCache *cache = nullptr);

/** Grouped matrix multiply and accumulate with a single-element scaling factor.
 */
void matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C,
                      const poplar::Tensor &k, const poplar::Tensor &A,
                      const poplar::Tensor &B, poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      matmul::PlanningCache *cache = nullptr);

/** Create a tensor that is used as the left operand of matrix multiplication.
 *
 * The types of the input and and output tensors are specified separately.
 * This will create a 2D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a matrix multiplication with this
 * tensor as the left argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param inputType       The input data type.
 * \param outputType      The data type of the returned tensor.
 * \param aShape          The shape of the required matrix.
 * \param bShape          The shape of the matrix that the required matrix will
 *                        be multiplied by.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the multiplication. See
 *                        matMul().
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p aShape. The
 *                        tensor will have been mapped to tiles.
 */
poplar::Tensor createMatMulInputLHS(poplar::Graph &graph,
                                    const poplar::Type &inputType,
                                    const poplar::Type &outputType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options = {},
                                    matmul::PlanningCache *cache = nullptr);

/** Create a tensor that is used as the left operand of matrix multiplication.
 *
 * The type of both input and output tensors is specified by \p dataType.
 * This will create a 2D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a matrix multiplication with this
 * tensor as the left argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param dataType        The data type of both the input and output tensors.
 * \param aShape          The shape of the required matrix.
 * \param bShape          The shape of the matrix that the required matrix will
 *                        be multiplied by.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the multiplication. See
 *                        matMul().
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p aShape. The
 *                        tensor will have been mapped to tiles.
 */
poplar::Tensor createMatMulInputLHS(poplar::Graph &graph,
                                    const poplar::Type &dataType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options = {},
                                    matmul::PlanningCache *cache = nullptr);

/**
 * Create a tensor that is used as the left operand of a grouped matrix
 * multiplication.
 *
 * This will create a 3D tensor in the graph. The ordering and tile mapping of
 * the tensor will be set to make a grouped matrix multiplication with this
 * tensor as the left argument efficient.
 *
 * The first dimension of the required matrix and the matrix it multiplies by
 * must the number of groups.
 *
 * \param graph           The Poplar graph.
 * \param type            The data type of the required matrix.
 * \param aShape          The grouped shape [g, r, c] of the required matrix.
 * \param bShape          The grouped shape [g, r, c] of the matrix that the
 *                        required matrix will be multiplied by.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the multiplication. See
 *                        matMul().
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and grouped shape \p aShape.
 *                        The tensor will have been mapped to tiles.
 */
poplar::Tensor
createMatMulGroupedInputLHS(poplar::Graph &graph, const poplar::Type &inputType,
                            const poplar::Type &outputType,
                            const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape,
                            const poplar::DebugContext &debugContext,
                            const poplar::OptionFlags &options = {},
                            matmul::PlanningCache *cache = nullptr);

/**
 * Create a tensor that is used as the right operand of matrix multiplication.
 *
 * This will create a 2D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a matrix multiplication with this
 * tensor as the right argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param inputType       The input data type.
 * \param outputType      The data type of the returned tensor.
 * \param aShape          The shape of the matrix that the required matrix will
 *                        be multiplied by.
 * \param bShape          The shape of the required matrix.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the multiplication. See
 *                        matMul().
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p bShape. The
 *                        tensor will have been mapped to tiles.
 */
poplar::Tensor createMatMulInputRHS(poplar::Graph &graph,
                                    const poplar::Type &inputType,
                                    const poplar::Type &outputType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options = {},
                                    matmul::PlanningCache *cache = nullptr);

/**
 * Overloaded function for when inputType == outputType (represented by the
 * dataType parameter).
 */
poplar::Tensor createMatMulInputRHS(poplar::Graph &graph,
                                    const poplar::Type &dataType,
                                    const std::vector<std::size_t> &aShape,
                                    const std::vector<std::size_t> &bShape,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options = {},
                                    matmul::PlanningCache *cache = nullptr);

/**
 * Create a tensor that is used as the right operand of grouped matrix
 * multiplication.
 *
 * This will create a 3D tensor in the graph. The ordering and tile mapping of
 * the tensor will be set to make a grouped matrix multiplication with this
 * tensor as the right argument efficient.
 *
 * The first dimension of the required matrix and the matrix it multiplies by
 * must the number of groups.
 *
 * \param graph           The Poplar graph.
 * \param type            The data type of the required matrix.
 * \param aShape          The grouped shape [g, r, c] of the matrix that the
 *                        required matrix will be multiplied by.
 * \param bShape          The grouped shape [g, r, c] of the required matrix.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the multiplication. See
 *                        matMul().
 * \param cache           Optional pointer to planning cache to use.
 *
 * \returns               A matrix of type \p type and grouped shape \p bShape.
 *                        The tensor will have been mapped to tiles.
 */
poplar::Tensor
createMatMulGroupedInputRHS(poplar::Graph &graph, const poplar::Type &inputType,
                            const poplar::Type &outputType,
                            const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape,
                            const poplar::DebugContext &debugContext,
                            const poplar::OptionFlags &options = {},
                            matmul::PlanningCache *cache = nullptr);

/** Pre-arrange right-hand side input.
 *
 *  Re-arrange memory for RHS operand to an upcoming matmul operation.
 *  This allows the rearrangement of the memory of a tensor that would
 *  otherwise be rearranged as part of the matmul operation for efficiency.
 *
 *  Use this function and the matMul*() functions with the
 *  `inputRHSIsPreArranged` option flag to do any re-arrangement necessary
 *  once and then re-use that input multiple times.
 *
 *  Only valid for fully connected layers.
 *
 *  \param graph          The Poplar graph.
 *  \param aShape         The shape of the left argument to the multiplication.
 *  \param B              The right argument to the multiplication. This
 *                        2D tensor must be already mapped to tiles.
 *  \param prog           A reference to a program sequence which will
 *                        be appended with the code to perform the
 *                        arrangement.
 *  \param outputType     Optional via overloaded function. Element type of
 *                        returned tensor. The default is \p B.elementType()
 *                        if omitted.
 *  \param debugContext    Optional debug information.
 *  \param options        Flags describing options for how the multiplication
 *                        should be implemented. See matMul().
 *  \param cache          Optional pointer to planning cache to use.
 *
 *  \returns              New tensor holding the rearranged input. This tensor
 *                        has the same shape as the given tensor.
 */
/** Pre-arrange input with explicitly defined output type. */
poplar::Tensor preArrangeMatMulInputRHS(
    poplar::Graph &graph, const std::vector<std::size_t> &aShape,
    const poplar::Tensor &B, poplar::program::Sequence &prog,
    const poplar::Type &outputType,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {},
    matmul::PlanningCache *cache = nullptr);

/** Pre-arrange input where the output type is the same as \p B. */
poplar::Tensor preArrangeMatMulInputRHS(
    poplar::Graph &graph, const std::vector<std::size_t> &aShape,
    const poplar::Tensor &B, poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {},
    matmul::PlanningCache *cache = nullptr);

/** Pre-arrange grouped input with explicitly defined output type. */
poplar::Tensor preArrangeMatMulGroupedInputRHS(
    poplar::Graph &graph, const std::vector<std::size_t> &aShape,
    const poplar::Tensor &B, poplar::program::Sequence &prog,
    const poplar::Type &outputType,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {},
    matmul::PlanningCache *cache = nullptr);

/**
 * Transposes a grouped matrix tensor.
 *
 * \param A               Tensor to transpose
 *
 * \returns               Transposed tensor
 */
poplar::Tensor transposeGroupedMatrix(const poplar::Tensor &A);

/**
 * Parameters to define a Matrix multiplication.
 *
 * \p C = \p A * \p B
 */
struct MatMulParams {
  /// Input type (of A & B)
  poplar::Type inputType;
  /// Output type (of C)
  poplar::Type outputType;

  /// Shape of the lhs input matrix (A)
  std::vector<std::size_t> aShape;
  /// Shape of the rhs input matrix (B)
  std::vector<std::size_t> bShape;

  friend bool operator<(const MatMulParams &a, const MatMulParams &b);
};

/**
 * A tuple containing the required parameters to preplan a matmul:
 *       - matmul-specific target for tile / IPU sizing
 *       - matmul parameters
 *       - implementation options (see matMul() above)
 *
 * All entries must have matching machine parameters.
 */
using MatMulPlanParams = std::tuple<const poplar::Target *, const MatMulParams,
                                    const poplar::OptionFlags *>;

/**
 * Mapping of pointers to matrix multiplication option flags to the
 * corresponding convolution option flags.
 */
using MatMulToConvOptions =
    std::unordered_map<const poplar::OptionFlags *, poplar::OptionFlags>;

/**
 * Obtain the set of convolution parameters corresponding to the user supplied
 * set of parameters for matrix multiplication.
 *
 *  \param matmuls        Set of Matrix multiplication parameter tuples
 *  \param matmulToConvOpts Convolution options corresponding to every matrix
 *                        multiplication options.
 *
 *  \returns              Set of Convolution parameters
 */
std::set<ConvPlanParams>
matMulGetConvPlanParams(const std::set<MatMulPlanParams> &matmuls,
                        MatMulToConvOptions &matmulToConvOpts);

/** \deprecated Use preplan() instead.
 *
 * Plan the specified matrix multiplications.
 * \param matmuls   A set of parameters to preplan matmuls
 * \param cache     The planning cache to update
 */
void preplanMatMuls(const std::set<MatMulPlanParams> &matmuls,
                    matmul::PlanningCache &cache);

/** Provides an interface to validate the matmul options. Presence of
 *  invalid key or a value will throw an exception.
 *
 *  \param options        Flags describing options for how the multiplication
 *                        should be implemented. See matMul().
 */
void matmulValidateOptions(const poplar::OptionFlags &options);

namespace matmul {

/** \deprecated Use poplin::PlanningCache instead. */
class PlanningCache {
public:
  PlanningCache();
  ~PlanningCache();

  /** Returns the number of entries currently stored in the cache. */
  std::size_t size() const;

  poplin::PlanningCache &getImpl();

private:
  poplin::PlanningCache impl;
};

} // namespace matmul

} // namespace poplin

#endif // poplin_MatMul_hpp
