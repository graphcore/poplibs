// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions to generate histograms of data.
 *
 */

#ifndef popops_GatherStatistics_hpp
#define popops_GatherStatistics_hpp

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Gather a histogram representing the statistics of the input tensor.
 *
 * Compare each element of \p input to each value in the \p levels
 * tensor. Where \p input <= \p levels[N] and \p input < \p levels[N-1], the
 * histogram entry for that range will be incremented by 1. The lowest and
 * highest histogram entries are bounded only by \p levels[0] and \p
 * levels[N-1], respectively. The function returns a histogram tensor with a
 * size one greater than the size of the \p levels tensor.
 *
 * **Histogram options**
 *
 *   * `useFloatArithmetic` (true, false) [=false]
 *
 *     If true, use float arithmetic internally and return a float result
 *     rather than an unsigned int result.  This has the benefit of
 *     simplicity and speed, but integer accuracy limited by the 32-bit float
 *     data format (integers > 16,777,216 are not all exactly represented).
 *
 * \param graph           The Poplar graph.
 * \param input           The input tensor on which to gather histogram
 *                        statistics.
 * \param levels          The levels defining the comparisons to carry out in
 *                        generating the histogram output.
 * \param absoluteOfInput If true, the absolute value of each input is
 *                        calculated before comparison to the \p levels data.
 * \param prog            A sequence program to which the code performing the
 *                        histogram will be appended.
 * \param debugPrefix     A debug prefix to add to any tensors/compute set
 *                        names.
 * \param options         A list of options to control the operation of the
 *                        histogram function.
 *
 * \return                A tensor of type unsigned int that contains the
 *                        levels + 1 histogram results. If the option
 *                        `useFloatArithmetic` is "true" the returned tensor
 *                        will have type float.
 */

poplar::Tensor histogram(poplar::Graph &graph, const poplar::Tensor &input,
                         const poplar::Tensor &levels, bool absoluteOfInput,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {});

/** Fill a tensor with a histogram representing the statistics of the input
 *  tensor.
 *
 *  Performs the same function as histogram() but writes the output to
 *  \p output. This must be one element larger than the \p levels tensor and
 *  have elements of type float or unsigned integer. The type of the output
 *  tensor will determine the type of arithmetic used internally, as described
 *  above.
 *
 *  This function allows histogram results to be accumulated over a number of
 *  calls using the \p updateOutput parameter.
 *
 * \param graph           The Poplar graph.
 * \param input           The input tensor on which to gather histogram
 *                        statistics.
 * \param input           The output tensor which will store the histogram
 *                        results.
 * \param updateOutput    If true, the histogram counts will be added to the
 *                        values already in \p output.
 * \param levels          The levels defining the comparisons to carry out in
 *                        generating the histogram output.
 * \param absoluteOfInput If true, the absolute value of each input is
 *                        calculated before comparison to the \p levels data.
 * \param prog            A sequence program to which the code performing the
 *                        histogram will be appended.
 * \param debugPrefix     A debug prefix to add to any tensors/compute set
 *                        names.
 */
void histogram(poplar::Graph &graph, const poplar::Tensor &input,
               poplar::Tensor &output, bool updateOutput,
               const poplar::Tensor &levels, bool absoluteOfInput,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");
} // namespace popops

#endif // popops_GatherStatistics_hpp
