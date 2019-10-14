// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_ScaledAdd_hpp
#define popops_ScaledAdd_hpp
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Add the elements of one tensor multiplied by a scalar to another tensor.
 *
 *  Performs the calculations A += scaleB * B
 *
 *  The operation is performed after casting B to type A.
 *
 * **Scaled add options**
 *
 *    * `optimizeForSpeed` (true, false) [=false]
 *
 *      The scaledAdd vertices default to being optimized to aid memory
 *      allocation. To optimise them for speed instead, set this option to true.
 *
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The scalar to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations.
 */
void scaledAddTo(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 float scaleB, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "",
                 const poplar::OptionFlags &options = {});

/** Add the elements of one tensor each multiplied by a (scalar) tensor to
 *  another tensor.
 *
 *  Performs the calculations A += scaleB * B
 *
 *  The operation is performed after casting scaleB and B to type A
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The scalar tensor to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 poplar::Tensor scaleB, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "",
                 const poplar::OptionFlags &options = {});

/** Subtract the elements of one tensor multiplied by a scalar from another
 * tensor.
 *
 *  Performs the calculations A -= scaleB * B
 *
 *  The operation is performed after casting B to type A.
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor providing the elements to subtract
 *                     (must be of the same shape as \A).
 * \param scaleB       The scalar to multiply elements of B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph,
                        poplar::Tensor A, poplar::Tensor B,
                        float scaleB, poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {});

/** Subtract the elements of one tensor each multiplied by a (scalar) tensor
 *  from another tensor.
 *
 *  Performs the calculations A -= scaleB * B
 *
 *  The operation is performed after casting scaleB, and B to type A
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor providing the elements to subtract
 *                     (must be of the same shape as \A).
 * \param scaleB       The scalar tensor to multiply elements of B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor B,
                 poplar::Tensor scaleB, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "",
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The 2 scaling factors are (scalar) tensors.
 *
 *  Performs the calculations A = scaleA * A + scaleB * B
 *
 *  The operation is performed after casting scaleA, scaleB and B to type A
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The scalar tensor to multiply elements of A with before
 *                     addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The scalar tensor to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, poplar::Tensor scaleA,
                 poplar::Tensor B, poplar::Tensor scaleB,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix ="",
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The 2 scaling factors are constants.
 *
 *  Performs the calculations A = scaleA * A + scaleB * B
 *
 * If A and B are of different types, B is first cast to type A and the
 * operation performed.
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The constant to multiply elements of A with before
 *                     addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The constant to multiply elements of B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, float scaleA,
                 poplar::Tensor B, float scaleB,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix ="",
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and subtract the scaled elements of another
 *  tensor to it. The 2 scaling factors are (scalar) tensors.
 *
 *  Performs the calculations A = scaleA * A - scaleB * B
 *
 *  The operation is performed after casting scaleA, scaleB and B to type A
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The scalar tensor to multiply elements of A with before
 *                     subtraction.
 * \param B            The second tensor to subtract elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The scalar tensor to multiply elements of B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     subtract will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph,
                 poplar::Tensor A, poplar::Tensor scaleA,
                 poplar::Tensor B, poplar::Tensor scaleB,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix ="",
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and subtract the scaled elements of
 *  another tensor to it. The 2 scaling factors are constants.
 *
 *  Performs the calculations A = scaleA * A - scaleB * B
 *
 * If A and B are of different types, B is first cast to type A and the
 * operation performed.
 *
 * \param graph        The poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The constant to multiply elements of A with before
 *                     subtraction.
 * \param B            The second tensor to subtract elements from (must be of
 *                     the same shape as \A).
 * \param scaleB       The constant to multiply elements of B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     subtract will be appended.
 * \param debugPrefix  A debug prefix to add to any tensors/compute set names.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph,
                 poplar::Tensor A, float scaleA,
                 poplar::Tensor B, float scaleB,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix ="",
                 const poplar::OptionFlags &options = {});
}

#endif // popops_ScaledAdd_hpp
