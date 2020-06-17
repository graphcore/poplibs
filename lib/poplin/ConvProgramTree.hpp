// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplin_ConvProgramTree_H
#define poplin_ConvProgramTree_H

#include <boost/optional.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <vector>

namespace poplin {

// This object reflects the control program that is constructed to create a
// convolution.
struct ConvProgramTree {
  struct TransformPreProgram {
    explicit TransformPreProgram(poplar::Graph &graph,
                                 const std::string &debugPrefix);

    void lower(poplar::program::Sequence &prog);

    std::vector<poplar::Tensor> writeUndef;

    std::vector<poplar::program::Copy> preTranspose;
    poplar::ComputeSet transposeCS;
    std::vector<poplar::program::Copy> postTranspose;
  };

  struct TransformPostSerialProgram {
    explicit TransformPostSerialProgram(poplar::Graph &graph,
                                        const std::string &debugPrefix);

    void lower(poplar::program::Sequence &prog);

    poplar::ComputeSet castCS;
    std::vector<poplar::program::Copy> copies;
  };

  struct ComputeSetsGroup {
    ComputeSetsGroup(poplar::ComputeSet convolveCS) : convolveCS(convolveCS) {}

    void lower(poplar::program::Sequence &prog);

    // The pre and post will be added by the function creating the vertices
    // if the input requires a cast then a pre compute set is needed
    boost::optional<poplar::ComputeSet> pre;
    poplar::ComputeSet convolveCS;
    poplar::program::Sequence postProg;
    // if the output requires a cast then a post compute set is needed
    boost::optional<poplar::ComputeSet> post;
  };

  ConvProgramTree(poplar::Graph &graph, const Plan &plan,
                  const poplar::Type &type, const std::string &debugPrefix);

  // lower the program tree as has been built up into the sequence passed in.
  void lower(poplar::program::Sequence &prog);

  // the following shows how the control code structure is after lowering,
  // assuming 2 levels in the hierarchy.
  //
  // serially split convolutions:
  //  - initProg
  //  - transformPreSerial
  //  - repeat(loopCount)
  //    - slice
  //    - transformPre[level=0]
  //    - transformPre[level=1]
  //    - convolve[level=1]
  //    - transformPost[level=1]
  //    - reduce[level=0]
  //    - transformPost[level=0]
  //    - update/addInPlace
  //    - loopPost
  //  - transformPostSerial
  //  - finalizeProg
  //
  // when there is no serial splits the Repeat(1) is transformed into a
  // Sequence and the slice, update and loopPost stages don't exist.
  //
  // For the time being we only support serially splitting a single channel and
  // as such there is at most one loop per level in the hierarchy. When
  // a decision has been made about how to implement multiple dimensions of
  // serial slices, this can be nailed down better.
  //
  // Program run before all others at any level of the hierarchy.
  poplar::program::Sequence initProg;
  // Transformations applied before and after the convolution at each level.
  // outer vector is indexed by level in hierarchy.
  std::vector<TransformPreProgram> transformPre;
  std::vector<std::vector<poplar::program::Copy>> transformPost;
  // Transformations applied before and after the loop.
  TransformPreProgram transformPreSerial;
  TransformPostSerialProgram transformPostSerial;
  // program used to increment loop counter
  poplar::program::Sequence loopPost;
  // the number of loop iterations (or: number of serial splits).
  unsigned loopCount;
  // the dynamic slice and update/add that happens at
  // the beginning and end of each loop iteration.
  poplar::program::Sequence slice, update;
  // the core convolution op.
  ComputeSetsGroup convolveCSGroup;
  // any post-conv reductions that might be required. first vector indexed by
  // level in the hierarchy, second vector is indexed by the reduction depth.
  std::vector<std::vector<poplar::ComputeSet>> reduceComputeSets;
  // Program run after all others at any level of the hierarchy.
  poplar::program::Sequence finalizeProg;

  // this variable is required for the Padder workaround in Convolution.cpp,
  // once T5913 is resolved this can be removed.
  poplar::Tensor copyWritten;
};

} // namespace poplin

#endif // poplin_ConvProgramTree_H
