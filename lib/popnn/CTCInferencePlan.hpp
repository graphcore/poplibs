// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popnn_CTCInferencePlan_hpp
#define popnn_CTCInferencePlan_hpp

#include <poplibs_support/Algorithm.hpp>
#include <popnn/CTCInference.hpp>

namespace popnn {
namespace ctc {

struct CtcInferencePlannerParams {
  poplar::Type inType;
  poplar::Type partialsType;
  poplar::Type outType;
  unsigned batchSize;
  unsigned maxTime;
  unsigned maxLabelLength;
  unsigned numClasses;
  unsigned beamWidth;
};

// A placeholder structure with potential partitions to allow mapping
// and other basic functions to be implemented similar to those used for loss
template <typename T> struct CtcInferencePartition {
  T batch;
  T time;
  T label;
  T beam;
  T classes;
};

class InferencePlan {
  poplar::Interval partition(unsigned fullSize, unsigned partitions,
                             unsigned index) const {
    const auto partitionSize = poplibs_support::ceildiv(fullSize, partitions);
    const auto begin = std::min(partitionSize * index, fullSize);
    const auto end = std::min(partitionSize * (index + 1), fullSize);
    return {begin, end};
  }

public:
  CtcInferencePlannerParams params;
  CtcInferencePartition<unsigned> parallel;

  // Given a batch size and partition index, return range of batch elements
  // represented in this partition
  poplar::Interval partitionBatch(unsigned batchSize, unsigned index) const {
    return partition(batchSize, parallel.batch, index);
  }

  // Given a time size and partition index, return range of time elements
  // represented in this partition
  poplar::Interval partitionTime(unsigned timeSize, unsigned index) const {
    return partition(timeSize, parallel.time, index);
  }
  unsigned getTile(unsigned batch, unsigned time, unsigned label) const {
    return batch * (parallel.time * parallel.label) // Batch
           + time * parallel.label                  // Time
           + label;                                 // Label
  }
  // Tile allocation when splitting across batch and time dimensions only
  unsigned getTile(unsigned batch, unsigned time) const {
    return batch * (parallel.time * parallel.label) // Batch
           + time;                                  // Time
  }
  std::unique_ptr<InferencePlan> clone() const {
    return std::make_unique<InferencePlan>(*this);
  };
};

bool operator<(const InferencePlan &a, const InferencePlan &b) noexcept;
bool operator==(const InferencePlan &a, const InferencePlan &b) noexcept;

std::ostream &operator<<(std::ostream &o, const InferencePlan &p);

} // namespace ctc
} // namespace popnn

#endif // #ifndef popnn_CTCInferencePlan_hpp
