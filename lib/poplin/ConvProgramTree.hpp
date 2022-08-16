// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplin_ConvProgramTree_H
#define poplin_ConvProgramTree_H

#include <ConvOptions.hpp>
#include <ConvPlan.hpp>
#include <boost/optional.hpp>
#include <map>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <vector>

namespace poplin {

struct Plan;

// This object reflects the control program that is constructed to create a
// convolution.
struct ConvProgramTree {
  struct TransformPreProgram {
    explicit TransformPreProgram(poplar::Graph &graph,
                                 const poplar::DebugNameAndId &dnai);

    void lower(poplar::program::Sequence &prog,
               const poplar::DebugNameAndId &dnai);

    std::vector<poplar::Tensor> writeUndef;

    std::vector<poplar::program::Copy> preTransposeActs;
    std::vector<poplar::program::Copy> preTransposeWeights;
    std::vector<poplar::ComputeSet> transposeCSActs;
    std::vector<poplar::ComputeSet> transposeCSWeights;
    std::vector<poplar::program::Copy> postTransposeCtrl;
    std::vector<poplar::program::Copy> postTransposeActs;
    std::vector<poplar::program::Copy> postTransposeWeights;
  };

  struct TransformPostSerialProgram {
    explicit TransformPostSerialProgram(poplar::Graph &graph,
                                        const poplar::DebugNameAndId &dnai);

    void lower(poplar::program::Sequence &prog,
               const poplar::DebugNameAndId &dnai);

    poplar::ComputeSet castCS;
    std::vector<poplar::program::Copy> copies;
  };

  using PostProg =
      std::map<poplar::Type, std::pair<std::vector<poplar::Tensor>,
                                       std::vector<poplar::Tensor>>>;
  struct ComputeSetsGroup {
    ComputeSetsGroup(poplar::ComputeSet convolveCS) : convolveCS(convolveCS) {}

    void lower(poplar::program::Sequence &prog,
               const poplar::DebugNameAndId &dnai);

    // The pre and post will be added by the function creating the vertices
    // if the input requires a cast then a pre compute set is needed
    boost::optional<poplar::ComputeSet> pre;
    poplar::ComputeSet convolveCS;
    PostProg postProg;

    // if the output requires a cast then a post compute set is needed
    boost::optional<poplar::ComputeSet> post;
  };

  ConvProgramTree(poplar::Graph &graph, const Plan &plan,
                  const poplar::DebugNameAndId &dnai);

  ConvProgramTree(poplar::Graph &graph, const poplar::DebugNameAndId &dnai);

  // lower the program tree as has been built up into the sequence passed in.
  void lower(poplar::Graph &graph, poplar::program::Sequence &prog,
             const boost::optional<Plan> &plan,
             bool insertTransformsCycleCountProgs,
             const poplar::DebugNameAndId &dnai);

  // the following shows how the control code structure is after lowering,
  // assuming 2 levels in the hierarchy.
  //
  // serially split convolutions:
  //  - weightsTranspose (optional)
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
  // create transposed/flipped weights if this is for the backwards pass.
  TransformPreProgram weightsTranspose;
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
  // the slice to occur at the start of an optional extra loop iteration to
  // handle e.g. the last serial slice of the output channels when the
  // number of channels isn't a multiple of the number of slices.
  boost::optional<poplar::program::Sequence> lastSlice, lastUpdate;
  // the core convolution op.
  ComputeSetsGroup convolveCSGroup;
  // Any post-conv reductions or casts that might be required. First vector is
  // indexed by the level in the hierarchy, the second vector is indexed by the
  // reduction depth.
  std::vector<std::vector<poplar::ComputeSet>> reduceOrCastComputeSets;
  // Program run after all others at any level of the hierarchy.
  poplar::program::Sequence finalizeProg;

  // this variable is required for the Padder workaround in Convolution.cpp,
  // once T5913 is resolved this can be removed.
  // TODO: T12874 Specialise std::hash for poplar::Type and use an unordered
  // container here.
  std::vector<poplar::Tensor> copyWritten;

  // When using this structure to create an output only, avoid most of the work
  // by not adding vertices
  bool createVertices = true;
};

} // namespace poplin

#endif // poplin_ConvProgramTree_H
