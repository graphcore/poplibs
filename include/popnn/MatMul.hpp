#ifndef __poplin_MatMul_hpp__
#define __poplin_MatMul_hpp__
#include <map>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <tuple>

namespace poplin {

/** Class used to cache the calculation of plans for implementing matrix
 *  multiplication operations.
 */
class PlanningCache;

/** Options to control the implementation of matrix multiplication */
struct MatMulOptions {
  /** Type used for partial sum calculation */
  std::string partialsType = "float";
  /** By default the output of a matrix multiply has a storage order with
      the columns (inner dimension) contiguous. If this flag is set the
      output will be row contiguous. */
  bool outputIsRowContiguous = false;
  /** Whether the left hand argument is going to be used in another
   *  matrix multiplication and in that other multiplication used either
   *  as a right hand side argument or on the left in a transposed form. */
  bool leftHandArgUsedInTranspose = false;
  /** An optional pointer to a planning cache. */
  PlanningCache *cache = nullptr;
  bool operator<(const MatMulOptions &other) const {
    return std::tie(partialsType, leftHandArgUsedInTranspose) <
             std::tie(other.partialsType, other.leftHandArgUsedInTranspose);
  }
};

/** Multiply two matrices.
 *
 *  Calculates C = op(A * B) where op is identity or transpose.
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
matMul(poplar::Graph &graph, poplar::Tensor A, poplar::Tensor B,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix = "",
       const MatMulOptions &options = MatMulOptions());

/** Multiply two matrices and add to a third (with a scaling factor).
 *
 *  Calculates C += k * A * B where A, B are matrices and k is a
 *  constant scalar.
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
matMulAcc(poplar::Graph &graph, poplar::Tensor C, float k,
          poplar::Tensor A, poplar::Tensor B,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "",
          const MatMulOptions &options = MatMulOptions());

/** Create an input for matrix multiplication.
 *
 *  This will create a 2D tensor in the graph. The ordering and tile mapping
 *  of the tensor will be set to make a matrix multiplication with this
 *  tensor as the left argument efficient.
 *
 *  \param graph           The poplar graph.
 *  \param type            The data type of the required matrix.
 *  \param aShape          The shape of the required matrix.
 *  \param B               The matrix the required matrix will be multiplied
 *                         by.
 *  \param name            The debug name of the required matrix.
 *  \param options         The implementation options of the multiplication.
 *
 *  \returns               A matrix of type \type and shape \aShape. The
 *                         tensor will have been mapped to tiles.
 */
poplar::Tensor
createMatMulInputA(poplar::Graph &graph,
                   const std::string &type,
                   const std::vector<std::size_t> &aShape,
                   const poplar::Tensor &B,
                   const std::string &name,
                   const MatMulOptions &options = MatMulOptions());

struct Plan;

class PlanningCache {
public:
  PlanningCache();
  ~PlanningCache();
  friend Plan getPlan(const poplar::Graph &graph,
                      std::string dType,
                      std::vector<std::size_t> aShape,
                      std::vector<std::size_t> bShape,
                      MatMulOptions options);
private:
  struct Params {
    std::string dType;
    std::vector<std::size_t> aShape;
    std::vector<std::size_t> bShape;
    MatMulOptions options;
    Params(std::string dType, std::vector<std::size_t> aShape,
           std::vector<std::size_t> bShape, MatMulOptions options) :
      dType(std::move(dType)), aShape(std::move(aShape)),
      bShape(std::move(bShape)), options(std::move(options)) {}
    bool operator<(const Params &other) const {
      return std::tie(dType, aShape, bShape, options) <
             std::tie(other.dType, other.aShape, other.bShape, other.options);
    }
  };
  std::map<Params, std::unique_ptr<Plan>> plans;
};

} // namespace poplin

#endif // __poplin_MatMul_hpp__
