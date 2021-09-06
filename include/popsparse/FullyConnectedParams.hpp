// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Parameters used for fully-connected layers using sparse tensors.
 */

#ifndef popsparse_FullyConnectedParams_hpp
#define popsparse_FullyConnectedParams_hpp

#include "SparsityParams.hpp"

#include <cassert>
#include <cmath>
#include <ostream>

namespace popsparse {
namespace dynamic {

class FullyConnectedParams {

  /// Sparsity parameters.
  SparsityParams sparsityParams;

  /// Proportion of weights which are non-zero in range [0,1].
  double nzRatio;

  std::size_t batchSize;
  std::size_t numGroups;
  std::size_t inputChannelsPerGroup;
  std::size_t outputChannelsPerGroup;

public:
  /** @name Fully connected parameters
   *  These are the parameters which define a fully connected layer.
   *
   *  Matrix multiplications for the different passes are as follows
   *
   *  - For pass = \c FC_INFERENCE or \c FC_TRAINING_FWD
   *
   *    [\p numGroups][\p outputChannelsPerGroup][\p inputChannelsPerGroup] *
   *    [\p numGroups][\p inputChannelsPerGroup][\p batchSize]
   *
   *  - For pass = \c FC_TRAINING_GRADA
   *
   *    [\p numGroups][\p inputChannelsPerGroup][\p outputChannelsPerGroup] *
   *    [\p numGroups][\p outputChannelsPerGroup][\p batchSize]
   *
   *  - For pass = \c FC_TRAINING_GRADW
   *
   *    [\p numGroups][\p outputChannelsPerGroup][\p batchSize] *
   *    [\p numGroups][\p batchSize][\p inputChannelsPerGroup]
   */
  ///@{
  /** Create parameters with the specified ratio of non-zero elements. */
  static FullyConnectedParams
  createWithNzRatio(const SparsityParams &sparsityParams, double nzRatio,
                    std::size_t batchSize, std::size_t numGroups,
                    std::size_t inputChannels, std::size_t outputChannels);
  /** Create parameters with the specified number of non-zero elements. */
  static FullyConnectedParams
  createWithNumNonZeroValues(const SparsityParams &sparsityParams,
                             std::size_t numNonZeroElems, std::size_t batchSize,
                             std::size_t numGroups, std::size_t inputChannels,
                             std::size_t outputChannels);
  ///@}

  std::size_t getBatchSize() const { return batchSize; }
  std::size_t getNumGroups() const { return numGroups; }
  std::size_t getInputChannels() const {
    return numGroups * inputChannelsPerGroup;
  }
  std::size_t getOutputChannels() const {
    return numGroups * outputChannelsPerGroup;
  }
  std::size_t getInputChannelsPerGroup() const { return inputChannelsPerGroup; }
  std::size_t getOutputChannelsPerGroup() const {
    return outputChannelsPerGroup;
  }
  const SparsityParams &getSparsityParams() const { return sparsityParams; }
  double getNzRatio() const;
  std::size_t getNumNonZeroValues() const;

  friend bool operator<(const FullyConnectedParams &a,
                        const FullyConnectedParams &b);
  friend bool operator==(const FullyConnectedParams &a,
                         const FullyConnectedParams &b);
  friend bool operator!=(const FullyConnectedParams &a,
                         const FullyConnectedParams &b);
};

std::ostream &operator<<(std::ostream &os, const FullyConnectedParams &p);

} // namespace dynamic
} // namespace popsparse

#endif // popsparse_FullyConnectedParams_hpp
