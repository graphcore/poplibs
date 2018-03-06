#ifndef poplin_MatMul_hpp
#define poplin_MatMul_hpp
#include <iosfwd>
#include <map>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <tuple>

namespace poplin {

/** Class used to cache the calculation of plans for implementing matrix
 *  multiplication operations.
 */
class PlanningCache;

enum class FullyConnectedPass {
  NONE,
  INFERENCE_FWD,
  TRAINING_FWD,
  TRAINING_BWD,
  TRAINING_WU,
};

/** Options to control the implementation of matrix multiplication */
struct MatMulOptions {
  /** Type used for partial sum calculation */
  poplar::Type partialsType = poplar::FLOAT;
  /// The fully connected pass this multiplication corresponds to. If this
  /// variable is not set to NONE look for a joint plan that avoids the need to
  /// exchange weights. In the forward and backward passes the weight matrix is
  /// assumed to be the right hand side operand of the multiplication. In the
  /// weight update pass we arrange for the result to have the same layout as
  /// the weights so it can be added to the weights without any exchange.
  FullyConnectedPass fullyConnectedPass = FullyConnectedPass::NONE;
  /** An optional pointer to a planning cache. */
  PlanningCache *cache = nullptr;
  bool operator<(const MatMulOptions &other) const {
    return std::tie(partialsType, fullyConnectedPass) <
             std::tie(other.partialsType, other.fullyConnectedPass);
  }
};

/** Multiply two matrices.
 *
 *  Calculates C = A * B where A and B are matrices.
 *
 *  \param graph           The poplar graph.
 *  \param A               The left argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param B               The right argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         multiplication.
 *  \param debugPrefix     A debug prefix added to compute set and tensor
 *                         names.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented.
 *
 *  \returns               The tensor holding the result of the multiplication.
 *                         This tensor will be created, added to the graph and
 *                         mapped to tiles.
 */
poplar::Tensor
matMul(poplar::Graph &graph, const poplar::Tensor &A, const poplar::Tensor &B,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix = "",
       const MatMulOptions &options = MatMulOptions());

void matMulReportPlan(std::ostream &out,
                      const poplar::Graph &graph,
                      const poplar::Type &dType,
                      const std::vector<std::size_t> &aShape,
                      const std::vector<std::size_t> &bShape,
                      const MatMulOptions &options = MatMulOptions());

/** Multiply two grouped matrices.
 *
 *  Calculates C[g] = A[g] * B[g] where A[g] and B[g] are matrices for
 *  each element in the group. g is element of the set {0, 1, ..., G-1}
 *
 *  The multiplication is done for every element in the group. The first
 *  dimension of the matrices is the group dimension with value equal to G.
 *
 *  \param graph           The poplar graph.
 *  \param A               The left argument to the grouped multiplication. This
 *                         3D tensor must be already mapped to tiles.
 *  \param B               The right argument to the grouped multiplication.
 *                         This 3D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         multiplication.
 *  \param debugPrefix     A debug prefix added to compute set and tensor
 *                         names.
 *  \param options         The structure describing options on how the
 *                         grouped multiplication should be implemented.
 *
 *  \returns               The tensor holding the result of the grouped
 *                         multiplication. This tensor will be created, added to
 *                         the graph and mapped to tiles.
 */
poplar::Tensor
matMulGrouped(poplar::Graph &graph, const poplar::Tensor &A,
              const poplar::Tensor &B, poplar::program::Sequence &prog,
              const std::string &debugPrefix = "",
              const MatMulOptions &options = MatMulOptions());

void matMulGroupedReportPlan(std::ostream &out,
                             const poplar::Graph &graph,
                             const poplar::Type &dType,
                             const std::vector<std::size_t> &aShape,
                             const std::vector<std::size_t> &bShape,
                             const MatMulOptions &options = MatMulOptions());

/** Multiply two matrices and add to a third (with a scaling factor).
 *
 *  Calculates C += k * A * B where A, B are matrices and k is a constant
 *  scalar.
 *
 *  \param graph           The poplar graph.
 *  \param C               The matrix to add to. This
 *                         2D tensor must be already mapped to tiles.
 *  \param k               The constant to multiply the result of the
 *                         multiplication.
 *  \param A               The left argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param B               The right argument to the multiplication. This
 *                         2D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         multiplication and add.
 *  \param debugPrefix     A debug prefix added to compute set and tensor
 *                         names.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented.
 */
void
matMulAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
          const poplar::Tensor &A, const poplar::Tensor &B,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "",
          const MatMulOptions &options = MatMulOptions());

/** Multiply two grouped matrices and add to a third (with a scaling factor).
 *
 *  Calculates C[g] += k * A[g] * B[g] where A[g], B[g] are matrices and k is a
 *  constant scalar. g is element of the set g = {0, 1, ..., G-1}
 *
 *  The multiplication is done for every element in the group. The first
 *  dimension of the matrices is the group dimension with value equal to G
 *
 *  \param graph           The poplar graph.
 *  \param C               The matrix to add to. This
 *                         3D tensor must be already mapped to tiles.
 *  \param k               The constant to multiply the result of the
 *                         multiplication.
 *  \param A               The left argument to the grouped multiplication. This
 *                         3D tensor must be already mapped to tiles.
 *  \param B               The right argument to the multiplication. This
 *                         3D tensor must be already mapped to tiles.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the grouped
 *                         multiplication and add.
 *  \param debugPrefix     A debug prefix added to compute set and tensor
 *                         names.
 *  \param options         The structure describing options on how the
 *                         multiplication should be implemented.
 */
void
matMulGroupedAcc(poplar::Graph &graph, const poplar::Tensor &C, float k,
          const poplar::Tensor &A, const poplar::Tensor &B,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "",
          const MatMulOptions &options = MatMulOptions());

/**
 * Create an tensor that is used as the left operand of matrix multiplication.
 *
 * This will create a 2D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a matrix multiplication with this
 * tensor as the left argument efficient.
 *
 * \param graph           The poplar graph.
 * \param type            The data type of the required matrix.
 * \param aShape          The shape of the required matrix.
 * \param bShape          The shape of the matrix that the required matrix will
 *                        be multiplied by.
 * \param name            The debug name of the required matrix.
 * \param options         The implementation options of the multiplication.
 *
 * \returns               A matrix of type \type and shape \aShape. The
 *                        tensor will have been mapped to tiles.
 */
poplar::Tensor
createMatMulInputLHS(poplar::Graph &graph,
                     const poplar::Type &type,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options = MatMulOptions());

/**
 * Create an tensor that is used as the left operand of a grouped matrix
 * multiplication.
 *
 * This will create a 3D tensor in the graph. The ordering and tile mapping of
 * the tensor will be set to make a grouped matrix multiplication with this
 * tensor as the left argument efficient.
 *
 * The first dimension of the required matrix and the matrix it multiplies by
 * must the number of groups.
 *
 * \param graph           The poplar graph.
 * \param type            The data type of the required matrix.
 * \param aShape          The shape of the required matrix.
 * \param bShape          The shape of the matrix that the required matrix will
 *                        be multiplied by.
 * \param name            The debug name of the required matrix.
 * \param options         The implementation options of the multiplication.
 *
 * \returns               A matrix of type \type and shape \aShape. The
 *                        tensor will have been mapped to tiles.
 */
poplar::Tensor
createMatMulGroupedInputLHS(poplar::Graph &graph,
                           const poplar::Type &type,
                           const std::vector<std::size_t> &aShape,
                           const std::vector<std::size_t> &bShape,
                           const std::string &name,
                           const MatMulOptions &options = MatMulOptions());

/**
 * Create an tensor that is used as the right operand of matrix multiplication.
 *
 * This will create a 2D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a matrix multiplication with this
 * tensor as the right argument efficient.
 *
 * \param graph           The poplar graph.
 * \param type            The data type of the required matrix
 * \param aShape          The shape of the matrix that the required matrix will
 *                        be multiplied by.
 * \param bShape          The shape of the required matrix.
 * \param name            The debug name of the required matrix.
 * \param options         The implementation options of the multiplication.
 *
 * \returns               A matrix of type \type and shape \bShape. The tensor
 *                        will have been mapped to tiles.
 */
poplar::Tensor
createMatMulInputRHS(poplar::Graph &graph,
                     const poplar::Type &type,
                     const std::vector<std::size_t> &aShape,
                     const std::vector<std::size_t> &bShape,
                     const std::string &name,
                     const MatMulOptions &options = MatMulOptions());


/**
 * Create an tensor that is used as the right operand of grouped matrix
 * multiplication.
 *
 * This will create a 3D tensor in the graph. The ordering and tile mapping of
 * the tensor will be set to make a grouped matrix multiplication with this
 * tensor as the right argument efficient.
 *
 * The first dimension of the required matrix and the matrix it multiplies by
 * must the number of groups.
 *
 * \param graph           The poplar graph.
 * \param type            The data type of the required matrix.
 * \param aShape          The shape of the matrix that the required matrix will
 *                        be multiplied by.
 * \param bShape          The shape of the required matrix.
 * \param name            The debug name of the required matrix.
 * \param options         The implementation options of the multiplication.
 *
 * \returns               A matrix of type \type and shape \bShape. The tensor
 *                        will have been mapped to tiles.
 */
poplar::Tensor
createMatMulGroupedInputRHS(poplar::Graph &graph,
                            const poplar::Type &type,
                            const std::vector<std::size_t> &aShape,
                            const std::vector<std::size_t> &bShape,
                            const std::string &name,
                            const MatMulOptions &options = MatMulOptions());

/**
 * Transposes a grouped matrix tensor
 *
 * \param A               Tensor to transpose
 *
 * \returns               Transposed tensor
 */
poplar::Tensor
transposeGroupedMatrix(const poplar::Tensor &A);

struct Plan;

class PlanningCacheImpl;

class PlanningCache {
public:
  std::unique_ptr<PlanningCacheImpl> impl;
  PlanningCache();
  ~PlanningCache();
};

} // namespace poplin

#endif // poplin_MatMul_hpp
