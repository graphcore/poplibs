// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file Loop.hpp
 *
 * Functions to provide counted loops of programs.
 *
 */

#ifndef poputil_Loop_hpp
#define poputil_Loop_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <string>

namespace poputil {

using CountedLoopBodyType =
    std::function<poplar::program::Program(const poplar::Tensor &)>;

/// This function creates a loop with counter set to initial value of \p begin,
/// and iterate up to the value of \p end (exclusive). \p step must be greater
/// than 0.
inline poplar::program::Sequence countedLoop(poplar::Graph &graph,
                                             std::size_t begin, std::size_t end,
                                             size_t step,
                                             const std::string &debugPrefix,
                                             const CountedLoopBodyType &body) {
  if (step == 0) {
    throw poputil::poplibs_error(
        "countedLoop: step must be greater than zero.");
  }

  poplar::program::Sequence prog;
  if (begin >= end) {
    throw poputil::poplibs_error("countedLoop: begin must be less than end");
  }

  poplar::Tensor tInductionVar = graph.addVariable(poplar::UNSIGNED_INT, {1});
  poplar::Tensor tBegin =
      graph.addConstant(poplar::UNSIGNED_INT, {1}, begin,
                        debugPrefix + "/begin-" + std::to_string(begin));
  poplar::Tensor tStep =
      graph.addConstant(poplar::UNSIGNED_INT, {1}, step,
                        debugPrefix + "/step-" + std::to_string(step));

  graph.setTileMapping(tInductionVar, 0);
  graph.setTileMapping(tBegin, graph.getTileMapping(tInductionVar));
  graph.setTileMapping(tStep, graph.getTileMapping(tInductionVar));

  prog.add(poplar::program::Copy(tBegin, tInductionVar));

  poplar::program::Sequence bodyProg = body(tInductionVar);
  popops::addInPlace(graph, tInductionVar, tStep, bodyProg);

  std::size_t count = (end - begin + step - 1) / step;
  prog.add(poplar::program::Repeat(count, bodyProg));

  return prog;
}

/// This function repeats \p body \p count times. This is a shortcut for
/// `countedLoop(graph, 0, count, 1, debugPrefix, body)`.
inline poplar::program::Sequence countedLoop(poplar::Graph &graph,
                                             std::size_t count,
                                             const std::string &debugPrefix,
                                             const CountedLoopBodyType &body) {
  return countedLoop(graph, 0, count, 1, debugPrefix, body);
}

} // namespace poputil

#endif // poputil_Loop_hpp
