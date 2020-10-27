// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#ifndef popnn_NonLinearity_hpp
#define popnn_NonLinearity_hpp

#include <popnn/NonLinearityDef.hpp>

#ifndef __POPC__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {

#define DEF_NONLINEARITY_INPLACE(fn, nlType)                                   \
  inline void fn##InPlace(poplar::Graph &graph, poplar::Tensor t,              \
                          poplar::program::Sequence &prog,                     \
                          const poplar::DebugContext &debugContext = {}) {     \
    nonLinearityInPlace(graph, nlType, t, prog, debugContext);                 \
  }                                                                            \
  inline void fn##InPlace(poplar::Graph &graph, poplar::Tensor t,              \
                          float &nonLinearityScaling,                          \
                          poplar::program::Sequence &prog,                     \
                          const poplar::DebugContext &debugContext = {}) {     \
    nonLinearityInPlace(graph, nlType, t, nonLinearityScaling, prog,           \
                        debugContext);                                         \
  }

#define DEF_NONLINEARITY_(fn, nlType)                                          \
  inline poplar::Tensor fn(poplar::Graph &graph, poplar::Tensor t,             \
                           poplar::program::Sequence &prog,                    \
                           const poplar::DebugContext &debugContext = {}) {    \
    return nonLinearity(graph, nlType, t, prog, debugContext);                 \
  }                                                                            \
  inline poplar::Tensor fn(poplar::Graph &graph, poplar::Tensor t,             \
                           float &nonLinearityScaling,                         \
                           poplar::program::Sequence &prog,                    \
                           const poplar::DebugContext &debugContext = {}) {    \
    return nonLinearity(graph, nlType, t, nonLinearityScaling, prog,           \
                        debugContext);                                         \
  }

#define DEF_NONLINEARITY(fn, nlType)                                           \
  DEF_NONLINEARITY_INPLACE(fn, nlType)                                         \
  DEF_NONLINEARITY_(fn, nlType)

/** Update tensor \p t by applying the given non-linearity in-place.
 *
 * \param graph             The graph to add the operation to.
 * \param nonLinearityType  The type of non-linearity to apply to \p t.
 * \param t                 The tensor to apply the non-linearity to.
 * \param prog              The sequence to add the operation to.
 * \param debugContext      Optional debug information.
 */
void nonLinearityInPlace(poplar::Graph &graph,
                         NonLinearityType nonLinearityType, poplar::Tensor t,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext = {});

/** Update tensor \p t by applying the given non-linearity in-place.
 *
 * \param graph             The graph to add the operation to.
 * \param nonLinearityType  The type of non-linearity to apply to \p t.
 * \param t                 The tensor to apply the non-linearity to.
 * \param cs                The compute set to add vertices to.
 * \param debugContext      Optional debug information.
 */
void nonLinearityInPlace(poplar::Graph &graph,
                         NonLinearityType nonLinearityType, poplar::Tensor t,
                         poplar::ComputeSet &cs,
                         const poplar::DebugContext &debugContext = {});

/** Update tensor \p t by applying the given non-linearity in-place and return
 *  the scaling factor by which outputs from this operation are multiplied in
 *  \p nonLinearityScaling.
 *
 * For NonLinearityType other than SOFTMAX_SCALED \p nonLinearityScaling will be
 * 1.0f upon return.
 *
 * \param graph               The graph to add the operation to.
 * \param nonLinearityType    The type of non-linearity to apply to \p t.
 * \param t                   The tensor to apply the non-linearity to.
 * \param nonLinearityScaling Reference to a float which will be overwritten
 *                            with the scaling factor by which outputs from
 *                            this operation in \p t are multiplied.
 * \param prog                The sequence to add the operation to.
 * \param debugContext        Optional debug information.
 */
void nonLinearityInPlace(poplar::Graph &graph,
                         NonLinearityType nonLinearityType, poplar::Tensor t,
                         float &nonLinearityScaling,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext = {});

/** Update tensor \p t by applying the given non-linearity in-place and return
 *  the scaling factor by which outputs from this operation are multiplied in
 *  \p nonLinearityScaling.
 *
 * For NonLinearityType other than SOFTMAX_SCALED \p nonLinearityScaling will be
 * 1.0f upon return.
 *
 * \param graph               The graph to add the operation to.
 * \param nonLinearityType    The type of non-linearity to apply to \p t.
 * \param t                   The tensor to apply the non-linearity to.
 * \param nonLinearityScaling Reference to a float which will be overwritten
 *                            with the scaling factor by which outputs from
 *                            this operation in \p t are multiplied.
 * \param cs                  The compute set to add vertices to.
 * \param debugContext        Optional debug information.
 */
void nonLinearityInPlace(poplar::Graph &graph,
                         NonLinearityType nonLinearityType, poplar::Tensor t,
                         float &nonLinearityScaling, poplar::ComputeSet &cs,
                         const poplar::DebugContext &debugContext = {});

/** Apply the given non-linearity to tensor \p t and return the result.
 *
 * \param graph             The graph to add the operation to.
 * \param nonLinearityType  The type of non-linearity to apply.
 * \param t                 The tensor to apply the non-linearity to.
 * \param prog              The sequence to add the operation to.
 * \param debugContext        Optional debug information.
 *
 * \returns A new tensor containing the contents of \p t with the given
 *          non-linearity applied.
 */
poplar::Tensor nonLinearity(poplar::Graph &graph,
                            NonLinearityType nonLinearityType, poplar::Tensor t,
                            poplar::program::Sequence &prog,
                            const poplar::DebugContext &debugContext = {});

/** Apply the given non-linearity to tensor \p t and return the result. Also
 *  returns the scaling factor by which outputs from this operation are
 *  multiplied in \p nonLinearityScaling.
 *
 * For NonLinearityType other than SOFTMAX_SCALED \p nonLinearityScaling will be
 * 1.0f upon return.
 *
 * \param graph               The graph to add the operation to.
 * \param nonLinearityType    The type of non-linearity to apply to \p t.
 * \param t                   The tensor to apply the non-linearity to.
 * \param nonLinearityScaling Reference to a float which will be overwritten
 *                            with the scaling factor by which outputs from
 *                            this operation in \p t are multiplied.
 * \param prog                The sequence to add the operation to.
 * \param debugContext        Optional debug information.
 *
 * \returns A new tensor containing the contents of \p t with the given
 *          non-linearity applied.
 */
poplar::Tensor nonLinearity(poplar::Graph &graph,
                            NonLinearityType nonLinearityType, poplar::Tensor t,
                            float &nonLinearityScaling,
                            poplar::program::Sequence &prog,
                            const poplar::DebugContext &debugContext = {});

DEF_NONLINEARITY(sigmoid, NonLinearityType::SIGMOID)
DEF_NONLINEARITY(relu, NonLinearityType::RELU)
DEF_NONLINEARITY(tanh, NonLinearityType::TANH)
DEF_NONLINEARITY(gelu, NonLinearityType::GELU)
DEF_NONLINEARITY(softmax, NonLinearityType::SOFTMAX)
DEF_NONLINEARITY(softmaxStable, NonLinearityType::SOFTMAX_STABLE)
DEF_NONLINEARITY(scaledSoftmaxStable, NonLinearityType::SOFTMAX_SCALED)

/** Computes and returns the input gradient for a non-linearity from the
 *  activations and gradients at the output of the non-linearity.
 *
 * \param graph             The graph to add the operation to.
 * \param nonLinearityType  The type of non-linearity to compute the input
 *                          gradient for.
 * \param act               The output activations from the non-linearity.
 *                          For the GELU non-linearity only this is the
 *                          input to the non-linearity.
 * \param outGradient       The gradients at the output of the non-linearity.
 * \param cs                The compute set to add vertices to.
 * \param debugContext      Optional debug information.
 *
 * \returns A new tensor with the calculated gradient for the input of the
 *          non-linearity.
 */
poplar::Tensor
nonLinearityInputGradient(poplar::Graph &graph,
                          NonLinearityType nonLinearityType, poplar::Tensor act,
                          poplar::Tensor outGradient, poplar::ComputeSet &cs,
                          const poplar::DebugContext &debugContext = {});

/** Computes and returns the input gradient for a non-linearity from the
 *  activations and gradients at the output of the non-linearity.
 *
 * \param graph             The graph to add the operation to.
 * \param nonLinearityType  The type of non-linearity to compute the input
 *                          gradient for.
 * \param act               The output activations from the non-linearity.
 *                          For the GELU non-linearity only this is the
 *                          input to the non-linearity.
 * \param outGradient       The gradients at the output of the non-linearity.
 * \param prog              The sequence to add the operation to.
 * \param debugContext      Optional debug information.
 *
 * \returns A new tensor with the calculated gradient for the input of the
 *          non-linearity.
 */
poplar::Tensor nonLinearityInputGradient(
    poplar::Graph &graph, NonLinearityType nonLinearityType, poplar::Tensor act,
    poplar::Tensor outGradient, poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {});

} // end namespace popnn

#endif // !__POPC__

#endif // popnn_NonLinearity_hpp
