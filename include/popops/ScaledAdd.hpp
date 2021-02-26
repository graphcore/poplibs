// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions for scaling and adding tensors.
 *
 */

#ifndef popops_ScaledAdd_hpp
#define popops_ScaledAdd_hpp
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

enum class ScaledAddSpecialisation { DEFAULT, X_MINUS_AX_PLUS_BY };

/** Add the elements of one tensor multiplied by a scalar to another tensor.
 *
 *  Performs the calculations \p A += \p scaleB * \p B
 *
 *  The operation is performed after casting \p B to the type of \p A.
 *
 * **Scaled add options**
 *
 *    * `optimizeForSpeed` (true, false) [=false]
 *
 *      The scaledAdd vertices default to being optimized to aid memory
 *      allocation. To optimise them for speed instead, set this option to true.
 *
 *    * `scaleFloatToHalfTolerance` (double) [=1e-6]
 *
 *      Where the tensors \p A, \p B are of type half and a \p scaleB
 *      is provided as a float or a tensor of type float, it is possible to
 *      to implement the scaledAddTo in half precision if \p scaleB can be cast
 *      to half precision with acceptable accuracy.  Otherwise full precision
 *      arithmetic can be used internally, but at the cost of speed.
 *      Floating point arithmetic will be selected if the relative error in
 *      casting is greater than the relative tolerance. \n
 *      Only applies to \c scaledAddTo() with \p scaleB.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The scalar to multiply elements of \p B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations.
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, poplar::Tensor B,
                 float scaleB, poplar::program::Sequence &prog,
                 const poplar::DebugContext &debugContext = {},
                 const poplar::OptionFlags &options = {});

/** Add the elements of one tensor each multiplied by a (scalar) tensor to
 *  another tensor.
 *
 *  Performs the calculations \p A += \p scaleB * \p B
 *
 *  The operation is performed after casting \p scaleB and \p B to the type
 *  of \p A.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The scalar tensor to multiply elements of \p B with
 *                     before addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, poplar::Tensor B,
                 poplar::Tensor scaleB, poplar::program::Sequence &prog,
                 const poplar::DebugContext &debugContext = {},
                 const poplar::OptionFlags &options = {});

/** Subtract the elements of one tensor multiplied by a scalar from another
 * tensor.
 *
 *  Performs the calculations \p A -= \p scaleB * \p B
 *
 *  The operation is performed after casting \p B to type \p A.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor providing the elements to subtract
 *                     (must be of the same shape as \p A).
 * \param scaleB       The scalar to multiply elements of \p B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A,
                        poplar::Tensor B, float scaleB,
                        poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {});

/** Subtract the elements of one tensor each multiplied by a (scalar) tensor
 *  from another tensor.
 *
 *  Performs the calculations \p A -= \p scaleB * \p B
 *
 *  The operation is performed after casting \p scaleB, and \p B to the type
 *  of \p A.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param B            The second tensor providing the elements to subtract
 *                     (must be of the same shape as \p A).
 * \param scaleB       The scalar tensor to multiply elements of \p B with
 *                     before subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A,
                        poplar::Tensor B, poplar::Tensor scaleB,
                        poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The two scaling factors are (scalar) tensors.
 *
 *  Performs the calculations \p A = \p scaleA * \p A + \p scaleB * \p B
 *
 *  The operation is performed after casting \p scaleA, \p scaleB and \p B to
 *  the type of \p A.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The scalar tensor to multiply elements of \p A with
 *                     before addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The scalar tensor to multiply elements of \p B with
 *                     before addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, poplar::Tensor scaleA,
                 poplar::Tensor B, poplar::Tensor scaleB,
                 poplar::program::Sequence &prog,
                 const poplar::DebugContext &debugContext = {},
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The two scaling factors are (scalar) tensors.
 *
 *  Performs the calculations \p A = \p scaleA' * \p A + \p scaleB * \p B
 *  where scaleA' is a function of scaleA specified by the "speciality" option.
 *
 *  The operation is performed after casting \p scaleA, \p scaleB and \p B to
 *  the type of \p A.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The scalar tensor to multiply elements of \p A with
 *                     before addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The scalar tensor to multiply elements of \p B with
 *                     before addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param speciality   Choice of ScaledAdd expression formulation
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, poplar::Tensor scaleA,
                 poplar::Tensor B, poplar::Tensor scaleB,
                 poplar::program::Sequence &prog,
                 const ScaledAddSpecialisation speciality,
                 const poplar::DebugContext &debugContext = {},
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The two scaling factors are constants.
 *
 *  Performs the calculations \p A = \p scaleA * \p A + \p scaleB * \p B
 *
 * If \p A and \p B are of different types, \p B is first cast to the type of
 * \p A and the operation performed.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The constant to multiply elements of \p A with before
 *                     addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The constant to multiply elements of \p B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, float scaleA,
                 poplar::Tensor B, float scaleB,
                 poplar::program::Sequence &prog,
                 const poplar::DebugContext &debugContext = {},
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and add the scaled elements of another
 *  tensor to it. The two scaling factors are constants.
 *
 *  Performs the calculations \p A = \p scaleA' * \p A + \p scaleB * \p B
 *  where scaleA' is a function of scaleA specified by the "speciality" option.
 *
 * If \p A and \p B are of different types, \p B is first cast to the type of
 * \p A and the operation performed.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The constant to multiply elements of \p A with before
 *                     addition.
 * \param B            The second tensor to add elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The constant to multiply elements of \p B with before
 *                     addition.
 * \param prog         A sequence program to which the code performing the
 *                     add will be appended.
 * \param speciality   Choice of ScaledAdd expression formulation
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledAddTo(poplar::Graph &graph, poplar::Tensor A, float scaleA,
                 poplar::Tensor B, float scaleB,
                 poplar::program::Sequence &prog,
                 const ScaledAddSpecialisation speciality,
                 const poplar::DebugContext &debugContext = {},
                 const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and subtract the scaled elements of another
 *  tensor to it. The two scaling factors are (scalar) tensors.
 *
 *  Performs the calculations \p A = \p scaleA * \p A - \p scaleB * \p B
 *
 *  The operation is performed after casting \p scaleA, \p scaleB and \p B to
 *  the type of \p A.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The scalar tensor to multiply elements of \p A with
 *                     before subtraction.
 * \param B            The second tensor to subtract elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The scalar tensor to multiply elements of \p B with
 *                     before subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     subtract will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A,
                        poplar::Tensor scaleA, poplar::Tensor B,
                        poplar::Tensor scaleB, poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {});

/** Scale the elements of one tensor and subtract the scaled elements of
 *  another tensor to it. The two scaling factors are constants.
 *
 *  Performs the calculations \p A = \p scaleA * \p A - \p scaleB * \p B
 *
 * If \p A and \p B are of different types, \p B is first cast to the type of
 * \p A and the
 * operation performed.
 *
 * \param graph        The Poplar graph.
 * \param A            The destination tensor.
 * \param scaleA       The constant to multiply elements of \p A with before
 *                     subtraction.
 * \param B            The second tensor to subtract elements from (must be of
 *                     the same shape as \p A).
 * \param scaleB       The constant to multiply elements of \p B with before
 *                     subtraction.
 * \param prog         A sequence program to which the code performing the
 *                     subtract will be appended.
 * \param debugContext Optional debug information.
 * \param options      A list of flags to control optimizations. See
 *                     scaledAddTo().
 */
void scaledSubtractFrom(poplar::Graph &graph, poplar::Tensor A, float scaleA,
                        poplar::Tensor B, float scaleB,
                        poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {});
} // namespace popops

#endif // popops_ScaledAdd_hpp
