// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_GatherStatistics_hpp
#define popops_GatherStatistics_hpp

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Gather a histogram representing the statistics of the input tensor.
 *
 * Compare each element of the \p input tensor to each level in the \p levels
 * tensor. Where "\p input <= level[N] and \p input < level[N-1] an
 * input is counted as between levels and that histogram entry will be
 * incremented by 1.
 * The function returns a histogram tensor with a size one greater than the
 * size of the \p levels tensor, as the upper and lower histogram entries are
 * bounded only by the upper and lower level respectively.
 *
 * ** Histogram options **
 *
 *     * `useFloatArithmetic` (true, false) [=false]
 *
 *       If true, use float arithmetic internally and return a float result
 *       rather than an unsigned int result.  This has the benefit of
 *       simplicity and speed but integer accuracy limited by the 32 bit float
 *       data format (Integers > 16777216 are not all exactly represented).
 *
 * \param graph           The Poplar graph.
 * \param input           The input tensor on which to gather histogram
 *                        statistics.
 * \param levels          The levels defining the comparisons to carry out in
 *                        generating the histogram output.
 * \param absoluteOfInput If true, the absolute value of each input is
 *                        calculated before comparison to the levels data.
 * \param prog            A sequence program to which the code performing the
 *                        add will be appended.
 * \param debugPrefix     A debug prefix to add to any tensors/compute set
 *                        names.
 * \param options         A list of flags to control the operation of the
 *                        histogram function.
 *
 * \return                A tensor of type unsigned int that contains the
 *                        levels + 1 histogram results. If the option
 *                        `useFloatArithmetic` is `true` the returned tensor
 *                        will have type float.
 */

poplar::Tensor histogram(poplar::Graph &graph, const poplar::Tensor &input,
                         const poplar::Tensor &levels, bool absoluteOfInput,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {});

// An alternative function which requires the output tensor to be provided.  The
// output must contain one more element than the \p levels tensor and have
// element type float or unsigned integer which will determine the type of
// arithmetic used internally as described above.
//
// This function allows histogram results to be accumulated over a number of
// calls using the /p updateOutput parameter.

void histogram(poplar::Graph &graph, const poplar::Tensor &input,
               poplar::Tensor &output, bool updateOutput,
               const poplar::Tensor &levels, bool absoluteOfInput,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");
} // namespace popops

#endif // popops_GatherStatistics_hpp
