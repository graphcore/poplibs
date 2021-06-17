// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file Loop.hpp
 *
 * Functions to provide counted loops of programs.
 *
 */

#ifndef popops_Loop_hpp
#define popops_Loop_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/VertexTemplates.hpp>

namespace popops {

using CountedLoopBodyType =
    std::function<poplar::program::Program(const poplar::Tensor &)>;

/** Create a loop program with constant initial count, increment and end value.
 *  The loop count is passed to the body program.
 *
 *  The program is equivalent to:
 *  \code
 *  for(unsigned i = begin; i != end; i += step){
 *    body;
 *  }
 *  \endcode
 *
 * \param graph        The graph the loop program will be added to.
 * \param begin        Initial counter value.
 * \param end          Counter end value (exclusive).
 * \param step         The increment added on each loop pass (must be greater
 *                     than zero).
 * \param body         The loop body program to run on each loop pass.
 * \param debugContext Optional debug information.
 *
 * \return             A program providing the above loop function.
 */
poplar::program::Sequence
countedLoop(poplar::Graph &graph, std::size_t begin, std::size_t end,
            size_t step, const CountedLoopBodyType &body,
            const poplar::DebugContext &debugContext = {});

/** Create a loop program which executes \p count times.
 *  The loop count is passed to the body program.
 *
 *  The program is equivalent to:
 *  \code
 *  for(unsigned i = 0; i != count; i += 1){
 *    body;
 *  }
 *  \endcode
 *  This is equivalent to poplar::Program::Repeat but with a loop counter
 *  that is passed to the body program. (It is actually implemented using
 *  poplar::Program::RepeatWhileTrue with a test for the count variable
 *  reaching \p count.)
 *
 * \param graph        The graph the loop program will be added to.
 * \param count        Number of loop iterations to execute.
 * \param body         The loop body program to run on each loop pass.
 * \param debugContext Optional debug information.
 *
 * \return             A program providing the above loop function.
 */
poplar::program::Sequence
countedLoop(poplar::Graph &graph, std::size_t count,
            const CountedLoopBodyType &body,
            const poplar::DebugContext &debugContext = {});

poplar::Tensor addForLoopCounterVertex(poplar::Graph &graph,
                                       const poplar::Tensor &count,
                                       const poplar::Tensor &countLimit,
                                       int countStep, unsigned tile,
                                       poplar::program::Sequence &prog,
                                       const poplar::DebugContext &di);

/** Create a for-loop program with constant initial count and increment, and a
 *  tensor as the end value.
 *  The use of a tensor as the loop end value means that the number of
 *  iterations can be calculated at run time.
 *  The loop count variable \p count is provided by the program that calls the
 *  loop program so it can be passed to the body program.
 *
 *  The program is equivalent to:
 *  \code
 *  for(unsigned count = initialCount; count != countLimit; count += countStep){
 *    body;
 *  }
 *  \endcode
 *
 * \param graph        The graph the loop program will be added to.
 * \param count        The loop count tensor, available to the \p body program
 *                     with element type INT or UNSIGNED_INT. Value initialised
 *                     by this function.
 * \param initialCount Initial counter value.
 * \param countLimit   Count limit tensor.
 * \param countStep    The increment added to the \p count tensor on each loop
 *                     pass.
 * \param body         The loop body program to run on each loop pass.
 * \param debugContext Optional debug information.
 *
 * \return             A program providing the above loop function.
 */

poplar::program::Sequence
countedForLoop(poplar::Graph &graph, const poplar::Tensor &count,
               int initialCount, const poplar::Tensor &countLimit,
               int countStep, const poplar::program::Program &body,
               const poplar::DebugContext &debugContext = {});

/** Create a for loop program with constant initial count and increment and a
 *  tensor as the end value.
 *  The use of a tensor as the loop end value means that the number of
 *  iterations can be calculated at run time.
 *  The count tensor is created internally and is not available to the body
 *  program.
 *
 *  The program is equivalent to:
 *  \code
 *  for(unsigned count = initialCount; count != countLimit; count += countStep){
 *    body;
 *  }
 *  \endcode
 *
 * \param graph        The graph the loop program will be added to.
 * \param initialCount Initial counter value.
 * \param countLimit   Count limit tensor.
 * \param countStep    The increment added to the \p count tensor on each loop
 *                     pass.
 * \param body         The loop body program to run on each loop pass.
 * \param debugContext Optional debug information.
 *
 * \return             A program providing the above loop function
 */

poplar::program::Sequence
countedForLoop(poplar::Graph &graph, int initialCount,
               const poplar::Tensor &countLimit, int countStep,
               const poplar::program::Program &body,
               const poplar::DebugContext &debugContext = {});

} // namespace popops

#endif // popops_Loop_hpp
