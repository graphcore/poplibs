// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poprand_RandomGen_hpp
#define poprand_RandomGen_hpp

#include "poputil/exceptions.hpp"
#include <array>
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
 *  \param debugPrefix      A prefix string for debugging.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor dropout(poplar::Graph &graph, const poplar::Tensor *seed,
                       const uint32_t seedModifier, const poplar::Tensor &input,
                       const poplar::Tensor &reference, double keepProbability,
                       double scale, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "");

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
 *  \param debugPrefix      A prefix string for debugging.
 *
 *  \returns A tensor with elements having a uniform distribution of random
 *           values.
 */
poplar::Tensor uniform(poplar::Graph &graph, const poplar::Tensor *seed,
                       uint32_t seedModifier, const poplar::Tensor &reference,
                       const poplar::Type &outType, double minVal,
                       double maxVal, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "");

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
 *  \param debugPrefix      A prefix string for debugging.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor bernoulli(poplar::Graph &graph, const poplar::Tensor *seed,
                         uint32_t seedModifier, const poplar::Tensor &reference,
                         const poplar::Type &outType, double prob,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "");

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
 *  \param debugPrefix      A prefix string for debugging.
 *
 *  \returns A tensor with elements randomly set to either zero or the scaled
 *           input value.
 */
poplar::Tensor normal(poplar::Graph &graph, const poplar::Tensor *seed,
                      uint32_t seedModifier, const poplar::Tensor &reference,
                      const poplar::Type &outType, double mean, double stdDev,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

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
 *  \param debugPrefix      A prefix string for debugging.
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
                               const std::string &debugPrefix = "");

/** Sets the random number generator seed on all tiles.
 *
 *  \param graph            The graph to add this operation to.
 *  \param masterSseed      A 64-bit integer to seed the random number
 *                          on every tile.
 *  \param seedModifier     Provides a further modification of the seed value.
 *  \param prog             The program to add this operation to.
 *  \param debugPrefix      A prefix string for debugging.
 */
void setSeed(poplar::Graph &graph, const poplar::Tensor &masterSeed,
             uint32_t seedModifier, poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

} // namespace poprand

#endif // poprand_RandomGen_hpp
