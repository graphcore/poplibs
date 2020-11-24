// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef _poplibs_popops_CollectivesProgram_hpp_
#define _poplibs_popops_CollectivesProgram_hpp_

#include "poplibs_support/Compiler.hpp"
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <cassert>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplibs_support/Visitor.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/exceptions.hpp>

// The collectives operation is created as a program that calls
// ReduceScatter:
//   Index = (RingIndex(repId) - 1) % N
//   bufferA = slice(MyFragments, Index)
//   Index = (Index + 1) % N
//   Repeat (N - 1) {
//     bufferB = CrossReplica(bufferA)
//     bufferA = slice(MyFragments, Index)
//     bufferA = add(bufferA, bufferB)
//     Index = (Index + 1) % N
//   }
//
// AllGather:
//   Index = RingIndex(repId)
//   MyFragment[Index] = bufferA  // bufferA initialised by scatter
//   Index = (Index + 1) % N
//   Repeat (N - 1) {
//     bufferB = CrossReplica(bufferA)
//     bufferA = bufferB
//     MyFragment[Index] = bufferA
//     Index = (Index + 1) % N
//
//
// Note this tries to represent the program for going both ways around
// the ring so many fields need a clockwise and anticlockwise version
//
//

namespace popops {

enum Direction { CLOCKWISE, ANTICLOCKWISE };

// This struct is used for the cross replica copy
// and the copies in the switch. For the switch
// we do not currently ever have both types but it may be
// something we are able to do in the future
template <class CopyType> struct BufferCopies {
  boost::optional<CopyType> clockwiseCopy;
  boost::optional<CopyType> anticlockwiseCopy;
  poplar::program::Sequence createProgram() const {
    // at least one of these should be populated
    assert(clockwiseCopy || anticlockwiseCopy);
    if (!clockwiseCopy) {
      return anticlockwiseCopy.get();
    } else if (!anticlockwiseCopy) {
      return clockwiseCopy.get();
    }
    return poplar::program::Sequence(clockwiseCopy.get(),
                                     anticlockwiseCopy.get());
  }

  void setCopy(CopyType copy, const Direction direction) {
    if (direction == Direction::CLOCKWISE) {
      assert(!static_cast<bool>(clockwiseCopy));
      clockwiseCopy = copy;
    } else if (direction == Direction::ANTICLOCKWISE) {
      assert(!static_cast<bool>(anticlockwiseCopy));
      anticlockwiseCopy = copy;
    }
  }
};

struct ReduceProg {
  poplar::Tensor A;
  poplar::Tensor B;
  popops::CollectiveOperator op;
  poplar::DebugNameAndId dnai;
  ReduceProg(poplar::Tensor A, poplar::Tensor B, popops::CollectiveOperator op,
             const poplar::DebugNameAndId &dnai_)
      : A(A), B(B), op(op), dnai(dnai_) {}

  ReduceProg operator+(const ReduceProg &other) const {
    assert(op == other.op);
    return ReduceProg(concat(A, other.A), concat(B, other.B), op, dnai);
  }
};

struct CollectivesProgram {

  CollectivesProgram(const poplar::DebugNameAndId &dnai);

  unsigned repeatCounter = 0;
  // Program to rearrange the input before the start of the loop.
  poplar::program::Sequence rearrangePre;
  // Program to rearrange the output at the end of the loop.
  poplar::program::Sequence rearrangePost;
  poplar::Tensor undefTensor; // These will be undeffed in the sequence
  // The src buffer that the slice program will slice into
  // The reduce scatter step returns this buffer, all gather doesn't set this
  boost::optional<poplar::Tensor> srcBuffer;
  boost::optional<poplar::Tensor> dstBuffer; // only used in reduce scatter
  poplar::program::Sequence
      initIndex; // program to set sliceTensor to ring index
  poplar::program::Sequence
      incrementIndex; // program to  update sliceIndex per iteration
  std::vector<BufferCopies<poplar::program::CrossReplicaCopy>>
      exchangeProg; // the cross replica copy may be expanded to multiple
                    // copies to avoid deadlocks

  poplar::program::Sequence sliceFragments; // dynamic slice of tensor
  boost::optional<ReduceProg> reduceProg;   // only used in reduce scatter
  poplar::program::Sequence allgatherCopy;  // only used in all gather
  poplar::program::Sequence
      firstGatherCopy; // on first iteration copy is from scatter output
};

static poplar::program::Sequence sequenceFromCrossReplicaCopies(
    const std::vector<BufferCopies<poplar::program::CrossReplicaCopy>>
        &bufferCopies,
    const poplar::DebugNameAndId &dnai) {
  poplar::program::Sequence s({}, {dnai});
  for (const auto &b : bufferCopies) {
    s.add(b.createProgram());
  }
  return s;
}

static void opInPlace(poplar::Graph &graph, popops::CollectiveOperator op,
                      const poplar::Tensor &a, const poplar::Tensor &b,
                      poplar::program::Sequence &prog,
                      const poplar::DebugNameAndId &dnai) {
  switch (op) {
  case CollectiveOperator::ADD:
    addInPlace(graph, a, b, prog, {dnai});
    break;
  case CollectiveOperator::MUL:
    mulInPlace(graph, a, b, prog, {dnai});
    break;
  case CollectiveOperator::MIN:
    minInPlace(graph, a, b, prog, {dnai});
    break;
  case CollectiveOperator::MAX:
    maxInPlace(graph, a, b, prog, {dnai});
    break;
  case CollectiveOperator::LOGICAL_AND:
    logicalAndInPlace(graph, a, b, prog, {dnai});
    break;
  case CollectiveOperator::LOGICAL_OR:
    logicalOrInPlace(graph, a, b, prog, {dnai});
    break;
  case CollectiveOperator::SQUARE_ADD:
    throw poputil::poplibs_error("Collective reduction using the SQUARE_ADD "
                                 "operation is not yet supported");
  case CollectiveOperator::LOCAL:
    POPLIB_UNREACHABLE();
    break;
  }
}

static poplar::program::Sequence
opInPlace(poplar::Graph &graph, const boost::optional<ReduceProg> &reduceProg) {
  poplar::program::Sequence prog;
  if (!reduceProg) {
    return prog;
  }
  opInPlace(graph, reduceProg->op, reduceProg->A, reduceProg->B, prog,
            reduceProg->dnai);
  return prog;
}

poplar::program::Sequence
unidirectionalSequence(CollectivesProgram &program, poplar::Graph &graph,
                       const poplar::DebugNameAndId &dnai) {
  using namespace poplar::program;
  const auto sliceFunction =
      graph.addFunction(std::move(program.sliceFragments));
  Sequence loopBody(
      {std::move(program.incrementIndex),
       sequenceFromCrossReplicaCopies(program.exchangeProg, {dnai}),
       std::move(program.allgatherCopy), Call(sliceFunction, {dnai}),
       opInPlace(graph, program.reduceProg)},
      {dnai});
  return Sequence(
      {WriteUndef(program.undefTensor, {dnai}), std::move(program.rearrangePre),
       std::move(program.initIndex), std::move(program.firstGatherCopy),
       Call(sliceFunction, {dnai}),
       Repeat(program.repeatCounter, std::move(loopBody), {dnai}),
       std::move(program.rearrangePost)},
      {dnai});
}
// Create a program that does a clockwise and anticlockwise collective
// simultaneously
poplar::program::Sequence
bidirectionalSequence(CollectivesProgram &clockwise,
                      CollectivesProgram &anticlockwise, poplar::Graph &graph,
                      const poplar::DebugNameAndId &dnai) {
  assert(clockwise.repeatCounter == anticlockwise.repeatCounter);
  using namespace poplar::program;
  const auto sliceFunction =
      graph.addFunction(Sequence(std::move(clockwise.sliceFragments),
                                 std::move(anticlockwise.sliceFragments)));
  boost::optional<ReduceProg> combinedReduceProg;
  assert(static_cast<bool>(clockwise.reduceProg) ==
         static_cast<bool>(anticlockwise.reduceProg));
  if (clockwise.reduceProg && anticlockwise.reduceProg) {
    combinedReduceProg =
        clockwise.reduceProg.get() + anticlockwise.reduceProg.get();
  }
  Sequence loopBody(
      {std::move(clockwise.incrementIndex),
       std::move(anticlockwise.incrementIndex),
       sequenceFromCrossReplicaCopies(clockwise.exchangeProg, {dnai}),
       sequenceFromCrossReplicaCopies(anticlockwise.exchangeProg, {dnai}),
       std::move(clockwise.allgatherCopy),
       std::move(anticlockwise.allgatherCopy), Call(sliceFunction, {dnai}),
       opInPlace(graph, combinedReduceProg)},
      {dnai});

  return Sequence(
      {WriteUndef(concat(clockwise.undefTensor, anticlockwise.undefTensor),
                  {dnai}),
       std::move(clockwise.rearrangePre), std::move(anticlockwise.rearrangePre),
       std::move(clockwise.initIndex), std::move(anticlockwise.initIndex),
       std::move(clockwise.firstGatherCopy),
       std::move(anticlockwise.firstGatherCopy), Call(sliceFunction, {dnai}),
       Repeat(clockwise.repeatCounter, std::move(loopBody), {dnai}),
       std::move(clockwise.rearrangePost),
       std::move(anticlockwise.rearrangePost)},
      {dnai});
}

// Create the sequence needed for the meet in the middle collective
poplar::program::Sequence meetInMiddleReduceScatterSequence(
    CollectivesProgram &clockwise, CollectivesProgram &anticlockwise,
    poplar::Graph &subGraph, poplar::program::Sequence combineBuffersProg,
    unsigned controlTile, const poplar::DebugNameAndId &dnai) {
  using namespace poplar;
  using namespace poplar::program;
  auto graph = subGraph.getTopLevelGraph();
  const auto isFirstStep = graph.addVariable(BOOL, {}, {dnai, "isFirstStep"});
  const auto trueConst = graph.addConstant(BOOL, {}, true, {dnai, "trueConst"});
  const auto falseConst =
      graph.addConstant(BOOL, {}, false, {dnai, "falseConst"});
  const auto zeroConst =
      graph.addConstant(UNSIGNED_INT, {}, 0, {dnai, "zeroConst"});
  const auto lastConst = graph.addConstant(
      UNSIGNED_INT, {}, clockwise.repeatCounter - 1, {dnai, "lastConst"});
  const auto loopCounter =
      graph.addVariable(UNSIGNED_INT, {}, {dnai, "loopCounter"});
  graph.setTileMapping(isFirstStep, controlTile);
  graph.setTileMapping(trueConst, controlTile);
  graph.setTileMapping(falseConst, controlTile);
  graph.setTileMapping(loopCounter, controlTile);
  graph.setTileMapping(zeroConst, controlTile);
  graph.setTileMapping(lastConst, controlTile);

  const auto clockwiseSliceFunction =
      graph.addFunction(std::move(clockwise.sliceFragments));
  const auto anticlockwiseSliceFunction =
      graph.addFunction(std::move(anticlockwise.sliceFragments));

  using namespace popops::expr;
  Sequence isLastProg({}, {dnai});
  auto isLastStep =
      popops::map(graph, _1 == _2, {loopCounter, lastConst}, isLastProg);

  Sequence incrementLoopCounter({}, {dnai});
  popops::mapInPlace(graph, _1 + 1, {loopCounter}, incrementLoopCounter,
                     {dnai});

  assert(clockwise.repeatCounter - 1 == anticlockwise.repeatCounter);
  // I think it is possible to remove the anticlockwise slice for before the
  // loop and use conditionals within the loop to do it
  Sequence loopBody(
      {std::move(clockwise.incrementIndex),
       sequenceFromCrossReplicaCopies(clockwise.exchangeProg, {dnai}),
       // here unconditionally create the cross replica copy. In the first
       // step this will transfer the uninitialised data but as the rest of the
       // repeat will be conditional on it not being step 0 it won't be
       // used and it will be overwritten in the next iteration of the repeat
       // It being done unconditionally means it can be merged with the
       // clockwise cross replica copy
       sequenceFromCrossReplicaCopies(anticlockwise.exchangeProg, {dnai}),
       Call(clockwiseSliceFunction, {dnai}),
       opInPlace(subGraph, clockwise.reduceProg),
       If(isFirstStep,
          Sequence({Copy(falseConst, isFirstStep, false, {dnai})}, {dnai}),
          Sequence(
              {std::move(isLastProg),
               If(isLastStep, Sequence({std::move(combineBuffersProg)}, {dnai}),
                  Sequence({Call(anticlockwiseSliceFunction, {dnai}),
                            opInPlace(subGraph, anticlockwise.reduceProg)},
                           {dnai}),
                  {dnai})},
              {dnai}),
          {dnai}),
       std::move(anticlockwise.incrementIndex),
       std::move(incrementLoopCounter)},
      {dnai});
  return Sequence(
      {WriteUndef(concat(clockwise.undefTensor, anticlockwise.undefTensor),
                  {dnai}),
       std::move(clockwise.rearrangePre), std::move(anticlockwise.rearrangePre),
       Copy(std::move(trueConst), isFirstStep, false, {dnai}),
       Copy(falseConst, isLastStep, false, {dnai}),
       Copy(std::move(zeroConst), std::move(loopCounter), false, {dnai}),
       std::move(clockwise.initIndex), std::move(anticlockwise.initIndex),
       Call(clockwiseSliceFunction, {dnai}),
       // TODO: T12922 Put this in first iteration of repeat loop.
       Call(anticlockwiseSliceFunction, {dnai}),
       Repeat(clockwise.repeatCounter, std::move(loopBody), {dnai}),
       std::move(clockwise.rearrangePost),
       std::move(anticlockwise.rearrangePost)},
      {dnai});
}

// Create the sequence needed for the meet in the middle collective
poplar::program::Sequence
meetInMiddleAllGatherSequence(CollectivesProgram &clockwise,
                              CollectivesProgram &anticlockwise,
                              poplar::Graph &subGraph, unsigned controlTile,
                              const poplar::DebugNameAndId &dnai) {
  using namespace poplar;
  using namespace poplar::program;
  auto graph = subGraph.getTopLevelGraph();
  const auto isFirstStep = graph.addVariable(BOOL, {}, {dnai, "isFirstStep"});
  const auto trueConst = graph.addConstant(BOOL, {}, true, {dnai, "trueConst"});
  const auto falseConst =
      graph.addConstant(BOOL, {}, false, {dnai, "falseConst"});
  const auto zeroConst =
      graph.addConstant(UNSIGNED_INT, {}, 0, {dnai, "zeroConst"});
  const auto lastConst = graph.addConstant(
      UNSIGNED_INT, {}, clockwise.repeatCounter - 1, {dnai, "lastConst"});
  const auto loopCounter =
      graph.addVariable(UNSIGNED_INT, {}, {dnai, "loopCounter"});
  graph.setTileMapping(isFirstStep, controlTile);
  graph.setTileMapping(trueConst, controlTile);
  graph.setTileMapping(falseConst, controlTile);
  graph.setTileMapping(loopCounter, controlTile);
  graph.setTileMapping(zeroConst, controlTile);
  graph.setTileMapping(lastConst, controlTile);

  const auto clockwiseSliceFunction =
      graph.addFunction(std::move(clockwise.sliceFragments));
  const auto anticlockwiseSliceFunction =
      graph.addFunction(std::move(anticlockwise.sliceFragments));

  using namespace popops::expr;
  Sequence isLastProg({}, {dnai});
  auto isLastStep =
      popops::map(graph, _1 == _2, {loopCounter, lastConst}, isLastProg);

  Sequence incrementLoopCounter({}, {dnai});
  popops::mapInPlace(graph, _1 + 1, {loopCounter}, incrementLoopCounter);

  assert(clockwise.repeatCounter - 1 == anticlockwise.repeatCounter);
  // In the loopbody i can choose to either put the anticlockwise slice or
  // the allgatherCopy behind the `if`. We have chosen the slice as it gives the
  // opportunity for the allgatherCopy to be merged with the clockwise one. This
  // decision should be reviewed if we ever merge the slice copies.
  Sequence loopBody(
      {std::move(clockwise.incrementIndex),
       std::move(anticlockwise.incrementIndex),
       sequenceFromCrossReplicaCopies(clockwise.exchangeProg, {dnai}),
       // here unconditionally create the cross replica copy. In the first
       // step this will transfer the uninitialised data but as the rest of the
       // repeat will be conditional on it not being step 0 it won't be
       // used and it will be overwritten in the next iteration of the repeat
       // It being done unconditionally means it can be merged with the
       // clockwise cross replica copy (same for the gather copy)
       sequenceFromCrossReplicaCopies(anticlockwise.exchangeProg, {dnai}),
       std::move(clockwise.allgatherCopy),
       std::move(anticlockwise.allgatherCopy),
       Call(clockwiseSliceFunction, {dnai}), std::move(isLastProg),
       If(isLastStep, Sequence({}, {dnai}),
          Sequence({Call(anticlockwiseSliceFunction, {dnai})}, {dnai}), {dnai}),
       std::move(incrementLoopCounter)},
      DebugContext(dnai));
  return Sequence(
      {WriteUndef(concat(clockwise.undefTensor, anticlockwise.undefTensor),
                  {dnai}),
       std::move(clockwise.rearrangePre), std::move(anticlockwise.rearrangePre),
       Copy(std::move(trueConst), std::move(isFirstStep), false, {dnai}),
       Copy(std::move(falseConst), std::move(isLastStep), false, {dnai}),
       Copy(std::move(zeroConst), std::move(loopCounter), false, {dnai}),
       std::move(clockwise.initIndex), std::move(anticlockwise.initIndex),
       std::move(clockwise.firstGatherCopy),
       std::move(anticlockwise.firstGatherCopy),
       Call(clockwiseSliceFunction, {dnai}),
       Repeat(clockwise.repeatCounter, std::move(loopBody), {dnai}),
       std::move(clockwise.rearrangePost),
       std::move(anticlockwise.rearrangePost)},
      DebugContext(dnai));
}

} // namespace popops
#endif
