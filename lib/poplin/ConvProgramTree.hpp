// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplin_ConvProgramTree_H
#define poplin_ConvProgramTree_H

#include <boost/optional.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace poplin {

// This object reflects the control program that is constructed to create a
// convolution.
struct ConvProgramTree {
  struct ComputeSetsGroup {
    ComputeSetsGroup(poplar::ComputeSet convolveCS) : convolveCS(convolveCS) {}

    poplar::program::Sequence createProgram() const;

    // The pre and post will be added by the function creating the vertices
    // if the input requires a cast then a pre compute set is needed
    boost::optional<poplar::ComputeSet> pre;
    poplar::ComputeSet convolveCS;
    poplar::program::Sequence postProg;
    // if the output requires a cast then a post compute set is needed
    boost::optional<poplar::ComputeSet> post;
  };

  ConvProgramTree(unsigned numLevels,
                  const std::vector<unsigned> &serialSplitsPerLevel,
                  poplar::Tensor copyWritten, poplar::ComputeSet convolveCS);

  // lower the program tree as has been built up into the sequence passed in.
  void lower(poplar::program::Sequence &prog);

  unsigned numLevels;

  // Program run before all others at any level of the hierarchy.
  poplar::program::Sequence initProg;
  // Transformations applied before and after the convolution at each level.
  std::vector<poplar::program::Sequence> transformPre, transformPost;
  // Programs to run at the start and end of each loop if present.
  std::vector<std::vector<poplar::program::Sequence>> loopPre, loopPost,
      transformPostSerial;
  std::vector<std::vector<unsigned>> loopCounts;
  std::vector<std::vector<poplar::ComputeSet>> reduceComputeSets;
  ComputeSetsGroup convolveCSGroup;
  // Program run after all others at any level of the hierarchy.
  poplar::program::Sequence finalizeProg;

  // this variable is required for the Padder workaround in Convolution.cpp,
  // once T5913 is resolved this can be removed.
  poplar::Tensor copyWritten;
};

} // namespace poplin

#endif // poplin_ConvProgramTree_H
