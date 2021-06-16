// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "popops/Loop.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

namespace popops {

poplar::program::Sequence
countedLoop(poplar::Graph &graph, std::size_t begin, std::size_t end,
            size_t step, const CountedLoopBodyType &body,
            const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(begin, end, step));

  if (step == 0) {
    throw poputil::poplibs_error(
        "countedLoop: step must be greater than zero.");
  }

  poplar::program::Sequence prog({}, {di});
  if (begin >= end) {
    throw poputil::poplibs_error("countedLoop: begin must be less than end");
  }

  poplar::Tensor tInductionVar =
      graph.addVariable(poplar::UNSIGNED_INT, {1}, {di});
  poplar::Tensor tBegin = graph.addConstant(
      poplar::UNSIGNED_INT, {1}, begin, {di, "begin-" + std::to_string(begin)});
  poplar::Tensor tStep = graph.addConstant(
      poplar::UNSIGNED_INT, {1}, step, {di, "step-" + std::to_string(step)});

  graph.setTileMapping(tInductionVar, 0);
  graph.setTileMapping(tBegin, graph.getTileMapping(tInductionVar));
  graph.setTileMapping(tStep, graph.getTileMapping(tInductionVar));

  prog.add(poplar::program::Copy(tBegin, tInductionVar, false, {di}));

  poplar::program::Sequence bodyProg = {body(tInductionVar)};
  popops::addInPlace(graph, tInductionVar, tStep, bodyProg, {di});

  std::size_t count = (end - begin + step - 1) / step;
  prog.add(poplar::program::Repeat(count, bodyProg, {di}));
  di.addOutputs(DI_ARGS(prog));
  return prog;
}

poplar::program::Sequence
countedLoop(poplar::Graph &graph, std::size_t count,
            const CountedLoopBodyType &body,
            const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(count));

  auto prog = countedLoop(graph, 0, count, 1, body, {di});
  di.addOutputs(DI_ARGS(prog));
  return prog;
}

poplar::Tensor addForLoopCounterVertex(poplar::Graph &graph,
                                       const poplar::Tensor &count,
                                       const poplar::Tensor &countLimit,
                                       int countStep, unsigned tile,
                                       poplar::program::Sequence &prog,
                                       const poplar::DebugContext &di) {
  auto predicate = graph.addVariable(poplar::UNSIGNED_INT, {}, di);
  graph.setTileMapping(predicate, tile);

  auto cs = graph.addComputeSet(di);
  const auto vertex =
      graph.addVertex(cs, poputil::templateVertex("popops::ForLoopCounter",
                                                  count.elementType()));
  graph.setTileMapping(vertex, tile);

  graph.connect(vertex["count"], count.reshape({}));
  graph.connect(vertex["limit"], countLimit.reshape({}));
  graph.connect(vertex["comparisonResult"], predicate);
  graph.setInitialValue(vertex["increment"], countStep);

  prog.add(poplar::program::Execute(cs));

  return predicate;
}

poplar::program::Sequence
countedForLoop(poplar::Graph &graph, const poplar::Tensor &count,
               int initialCount, const poplar::Tensor &countLimit,
               int countStep, const poplar::program::Program &body,
               const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(count, initialCount, countLimit, countStep));

  const auto tensorType = count.elementType();
  if (tensorType != countLimit.elementType()) {
    throw poputil::poplibs_error(
        "countedForLoop: count and countLimit tensors must have the same type");
  }
  if (tensorType != poplar::UNSIGNED_INT && tensorType != poplar::INT) {
    throw poputil::poplibs_error("countedForLoop: count must have type INT "
                                 " or UNSIGNED_INT");
  }
  poplar::program::Sequence prog;
  // An initialiser, decremented by 1 step, so that when pre-incremented in the
  // `cond` program the loop body has a count variable visible that counts:
  // initialCount, initialCount + countStep, initialCount +2 * countStep ...
  auto initialiser = graph.addConstant<unsigned>(
      count.elementType(), {}, static_cast<unsigned>(initialCount - countStep),
      di);
  graph.setTileMapping(initialiser, 0);
  prog.add(poplar::program::Copy(initialiser, count));

  poplar::program::Sequence cond;
  auto predicate =
      addForLoopCounterVertex(graph, count, countLimit, countStep, 0, cond, di);

  prog.add(poplar::program::RepeatWhileTrue(cond, predicate.reshape({}), body,
                                            {di, "countedLoop"}));
  di.addOutputs(DI_ARGS(prog));
  return prog;
}

poplar::program::Sequence
countedForLoop(poplar::Graph &graph, int initialCount,
               const poplar::Tensor &countLimit, int countStep,
               const poplar::program::Program &body,
               const poplar::DebugContext &debugContext) {
  auto count = graph.addVariable(countLimit.elementType(), {}, debugContext);
  graph.setTileMapping(count, 0);
  return countedForLoop(graph, count, initialCount, countLimit, countStep, body,
                        debugContext);
}

} // namespace popops
