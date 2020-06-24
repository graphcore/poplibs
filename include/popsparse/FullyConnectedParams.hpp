// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_FullyConnectedParams_hpp
#define popsparse_FullyConnectedParams_hpp

#include "SparsityParams.hpp"

#include <cassert>
#include <cmath>
#include <ostream>

namespace popsparse {
namespace dynamic {

class FullyConnectedParams {

  /// Sparsity parameters
  SparsityParams sparsityParams;

  /// Proportion of weights which are non-zero in range [0,1]
  double nzRatio;

  // These are parameters which define a fully connected layer.
  //   Matrix multiplications for the different passes are the following
  //    For Pass = FC_INFERENCE and FC_TRAINING_FWD
  //      [numGroups][outputChannelsPerGroup][inputChannelsPerGroup] *
  //      [numGroups][inputChannelsPerGroup][batchSize]
  //
  //    For Pass = FC_TRAINING_GRADA
  //      [numGroups][inputChannelsPerGroup][outputChannelsPerGroup] *
  //      [numGroups][outputChannelsPerGroup][batchSize]
  //
  //    For Pass = FC_TRAINING_GRADW
  //      [numGroups][outputChannelsPerGroup][batchSize] *
  //      [numGroups][batchSize][inputChannelsPerGroup]
  std::size_t batchSize;
  std::size_t numGroups;
  std::size_t inputChannelsPerGroup;
  std::size_t outputChannelsPerGroup;

public:
  static FullyConnectedParams
  createWithNzRatio(const SparsityParams &sparsityParams, double nzRatio,
                    std::size_t batchSize, std::size_t numGroups,
                    std::size_t inputChannels, std::size_t outputChannels);
  static FullyConnectedParams
  createWithNumNonZeroValues(const SparsityParams &sparsityParams,
                             std::size_t numNonZeroElems, std::size_t batchSize,
                             std::size_t numGroups, std::size_t inputChannels,
                             std::size_t outputChannels);

  std::size_t getBatchSize() const { return batchSize; }
  std::size_t getNumGroups() const { return numGroups; }
  std::size_t getInputChannelsPerGroup() const { return inputChannelsPerGroup; }
  std::size_t getOutputChannelsPerGroup() const {
    return outputChannelsPerGroup;
  }
  struct SparsityParams getSparsityParams() const {
    return sparsityParams;
  }
  double getNzRatio() const;
  std::size_t getNumNonZeroValues() const;

  friend bool operator<(const FullyConnectedParams &a,
                        const FullyConnectedParams &b);
};

std::ostream &operator<<(std::ostream &os, const FullyConnectedParams &p);

} // namespace dynamic
} // namespace popsparse

#endif // popsparse_FullyConnectedParams_hpp
