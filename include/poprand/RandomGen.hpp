// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poprand_RandomGen_hpp
#define poprand_RandomGen_hpp

#include "poputil/exceptions.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <utility>

namespace poprand {

/** Apply dropout to a tensor.
 *
 *  The elements of tensor \p input are multiplied by a mask consisting of a
 *  sequence of randomly generated 1 or 0. The keep probability of the dropout
 *  P(1) = \p keepProbability.
 * The contents of the mask depend on the keep probability, seed, seed modifier
 * and layout of the reference tensor.
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the dropout mask.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param input            The input tensor to be masked.
 *  \param reference        A tensor that specifies the layout of the output
 *                          tensor.
 *                          Must be the same shape as the input.
 *  \param keepProbability  The probability of keeping an input value.
 *  \param scale            Scales the output tensor. This is typically the
 *                          inverse of the dropout probability, (1 / P(1)).
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor dropout(poplar::Graph &graph, const poplar::Tensor *seed,
                       const uint32_t seedModifier, const poplar::Tensor &input,
                       const poplar::Tensor &reference, double keepProbability,
                       double scale, poplar::program::Sequence &prog,
                       const poplar::DebugContext &debugContext = {});

/** Apply dropout to a tensor.
 *
 *  The elements of tensor \p input are multiplied by a mask consisting of a
 *  sequence of randomly generated 1 or 0. The keep probability of the dropout
 *  P(1) = \p keepProbability.
 *  The contents of the mask depend on the keep probability, seed, seed
 *  modifier and layout of the reference tensor.
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the dropout mask.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param input            The input tensor to be masked.
 *  \param reference        A tensor that specifies the layout of the output
 *                          tensor. Must be the same shape as the input.
 *  \param keepProbability  The probability of keeping an input value.
 *  \param scale            Scales the output tensor. This is typically the
 *                          inverse of the dropout probability, (1 / P(1)).
 *  \param outputClonesRef  When true, the output tensor is a clone of the
 *                          reference tensors. When false, the output tensor
 *                          is a clone of the input tensor.
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor dropout(poplar::Graph &graph, const poplar::Tensor *seed,
                       const uint32_t seedModifier, const poplar::Tensor &input,
                       const poplar::Tensor &reference, double keepProbability,
                       double scale, bool outputClonesRef,
                       poplar::program::Sequence &prog,
                       const poplar::DebugContext &debugContext = {});

/** Apply shaped dropout to a tensor.
 *
 *  The elements of tensor \p input are multiplied by a mask consisting of a
 *  sequence of randomly generated 1 or 0. The keep probability of the dropout
 *  P(1) = \p keepProbability.
 *
 *  Shaped dropout allows row, column and dimension wise dropout, versus
 *  element-wise standard dropout. The shape of the dropout must be compatible
 *  (broadcastable) to \p input.
 *
 * The contents of the mask depend on the keep probability, seed, seed modifier
 * and layout of the reference tensor.
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the dropout mask.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param input            The input tensor to be masked.
 *  \param reference        A tensor that specifies the shape and layout of the
 *                          dropout. Must be broadcastable to the input.
 *  \param keepProbability  The probability of keeping an input value.
 *  \param scale            Scales the output tensor. This is typically the
 *                          inverse of the dropout probability, (1 / P(1)).
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor shapedDropout(poplar::Graph &graph, const poplar::Tensor *seed,
                             const uint32_t seedModifier,
                             const poplar::Tensor &input,
                             const poplar::Tensor &reference,
                             double keepProbability, double scale,
                             poplar::program::Sequence &prog,
                             const poplar::DebugContext &debugContext = {});

/** Uniform distribution in a given interval with \p maxVal > \p minVal.
 *
 *  Generates random data with uniform distribution in the interval [\p minVal,
 *  \p maxVal]. The output may be of type \c float, \c half or \c int.
 *
 *  For type \c int, data is generated in the interval [\p minVal, \p maxVal]
 *  with uniform probability if (\p maxVal - \p minVal) is a power of 2.
 *  Otherwise there will be a small bias in the probability generated, with the
 *  bias directly proportional to the ratio (\p maxVal - \p minVal + 1 ) / 2^32.
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the distribution.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param reference        A tensor that specifies the layout of the output
 *                          tensor.
 *  \param outType          Type of the output tensor. One of \c float, \c half
 *                          or \c int.
 *  \param minVal           The minimum value of the distribution.
 *  \param maxVal           The maximum value of the distribution.
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor with elements having a uniform distribution of random
 *           values.
 */
poplar::Tensor uniform(poplar::Graph &graph, const poplar::Tensor *seed,
                       uint32_t seedModifier, const poplar::Tensor &reference,
                       const poplar::Type &outType, double minVal,
                       double maxVal, poplar::program::Sequence &prog,
                       const poplar::DebugContext &debugContext = {});

/** Log-uniform distribution over a closed interval [\p minVal, \p maxVal]
 *
 *  Generates random data log-uniformly distributed in the closed interval
 *  [\p minVal, \p maxVal]. The output may be of type \c float, \c half or
 *  \c int. The base of the log can be specified, but defaults to the natural
 *  base.
 *
 *  The actual interval of the samples depends on the representable values of
 *  the \p outType and is a subset of the initial interval; the interval will be
 *  squeezed inward to the next representable values of \p outType. For example,
 *  for \c half, the interval [2049.0, 4098.0] would be squeezed to
 *  [2050.0, 4096.0].
 *  Depending on the interval's representability, this may cause spikes in the
 *  distribution at the boundaries - careful choice of interval is suggested.
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the distribution.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param reference        A tensor that specifies the layout of the output
 *                          tensor.
 *  \param outType          Type of the output tensor. One of \c float, \c half
 *                          or \c int.
 *  \param minVal           The minimum value of the distribution.
 *  \param maxVal           The maximum value of the distribution.
 *  \param prog             The program to add this operation to.
 *  \param base             Optional base of the log / exponent of the
 *                          underlying uniform distribution. Defaults to Euler's
 *                          number (natural base).
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor the same size as \p reference with elements having a
 *           log-uniform distribution of random values of type \p outType.
 *
 *  \throw poputil::poplibs_error If \p minVal < 1
 *  \throw poputil::poplibs_error If \p maxVal <= \p minVal
 *  \throw poputil::poplibs_error If \p minVal and \p maxVal are not suitable
 *         for the \p outType (for example the range is too narrow)
 */
poplar::Tensor logUniform(poplar::Graph &graph, const poplar::Tensor *seed,
                          uint32_t seedModifier,
                          const poplar::Tensor &reference,
                          const poplar::Type &outType, double minVal,
                          double maxVal, poplar::program::Sequence &prog,
                          double base = M_E,
                          const poplar::DebugContext &debugContext = {});

/** Bernoulli distribution which has the value 1 with the specified probability.
 *
 *  Generates a tensor with random values of 0 and 1, determined by \p prob.
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the distribution.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param reference        A tensor that specifies the layout of the output
 *                          tensor.
 *  \param outType          Type of the output tensor. One of \c float, \c half
 *                          or \c int.
 *  \param prob             Probability of an element being 1.
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor bernoulli(poplar::Graph &graph, const poplar::Tensor *seed,
                         uint32_t seedModifier, const poplar::Tensor &reference,
                         const poplar::Type &outType, double prob,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext = {});

/** Normal distribution with given mean and standard deviation.
 *
 *  Generates random data with a normal (Gaussian) distribution. The mean
 *  is given by \p mean and the standard deviation by \p stdDev.
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the distribution.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param reference        A tensor that specifies the layout of the output
 *                          tensor.
 *  \param outType          Type of the output tensor. One of \c float or
 *                          \c half.
 *  \param mean             The mean value of the distribution.
 *  \param stdDev           The standard deviation of the distribution.
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor normal(poplar::Graph &graph, const poplar::Tensor *seed,
                      uint32_t seedModifier, const poplar::Tensor &reference,
                      const poplar::Type &outType, double mean, double stdDev,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {});

/** Truncated normal distribution.
 *
 *  Generates a distribution derived from a normal distribution with mean
 *  \p mean and standard deviation \p stdDev. This normal distribution is
 *  truncated symmetrically about the mean at
 *  (\p mean - \p alpha * \p stdDev) and (\p mean + \p alpha * \p stdDev)
 *
 *  \param graph            The graph to add this operation to.
 *  \param seed             If not null, this is a pair of 32-bit integers used
 *                          to seed the random number generator that generates
 *                          the distribution.
 *  \param seedModifier     Provides a further modification of the seed value.
 *                          Ignored if \p seed is null.
 *  \param reference        A tensor that specifies the layout of the output
 *                          tensor.
 *  \param outType          Type of the output tensor. One of \c float or
 *                          \c half.
 *  \param mean             The mean value of the distribution.
 *  \param stdDev           The standard deviation of the distribution.
 *  \param alpha            Defines the minimum and maximum values of the
 *                          distribution.
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor truncatedNormal(poplar::Graph &graph, const poplar::Tensor *seed,
                               uint32_t seedModifier,
                               const poplar::Tensor &reference,
                               const poplar::Type &outType, double mean,
                               double stdDev, double alpha,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext = {});

/** Sets the random number generator seed on all tiles.
 *
 *  \param graph            The graph to add this operation to.
 *  \param masterSeed       A 64-bit integer to seed the random number
 *                          on every tile.
 *  \param seedModifier     Provides a further modification of the seed value.
 *  \param prog             The program to add this operation to.
 *  \param debugContext     Optional debug information.
 */
void setSeed(poplar::Graph &graph, const poplar::Tensor &masterSeed,
             uint32_t seedModifier, poplar::program::Sequence &prog,
             const poplar::DebugContext &debugContext = {});

} // namespace poprand

#endif // poprand_RandomGen_hpp
