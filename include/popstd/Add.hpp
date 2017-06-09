#ifndef __popstd_Add_hpp__
#define __popstd_Add_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popstd {

/** Add the elements of one tensor multiplied by a scalar to another tensor.
 *
 *  Performs the calculations A += k * B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param k            The scalar to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void addTo(poplar::Graph &graph,
           poplar::Tensor A, poplar::Tensor B,
           float k, poplar::program::Sequence &prog,
           const std::string &debugPrefix = "");



/** Add the elements of one tensor multiplied by a scalar to another tensor.
 *
 *  Performs the calculations A += B
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 */
void addTo(poplar::Graph &graph,
           poplar::Tensor A, poplar::Tensor B,
           poplar::program::Sequence &prog,
           const std::string &debugPrefix = "");

}

#endif // __popstd_Add_hpp__
