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

template <typename T> struct CtcInferencePartition {
  // Simple initial partition parameters, see plan assignment for a description
  // TODO - Documnet a fully thought out plan here
  T batch;
  T time;

  T copy;
  T extend;
  T extendVerticesPerPartition;
  T merge;
  T output;
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

  // The larger of the `classes` and `beam` partitions is the total number
  // of broadcast inputs, and replicas of the beam history that we will build.
  // In this simple model of splitting up the work, copy candidates and extend
  // candidates are generated with vertices allocated on overlapping tiles, so
  // the maximum of the 2 parameters is used here. In a more complete solution
  // we could choose between overlapping (total=max) or sequential (total=sum)
  // allocation of vertices
  unsigned batchEntryPartitions(void) const {
    return std::max(parallel.merge, std::max(parallel.extend, parallel.copy));
  }

  poplar::Interval partitionBatchEntry(unsigned size, unsigned index) const {
    return partition(size, batchEntryPartitions(), index);
  }

  poplar::Interval partitionMerge(unsigned mergeSize, unsigned index) const {
    return partition(mergeSize, parallel.merge, index);
  }

  poplar::Interval partitionOutput(unsigned outSize, unsigned index) const {
    return partition(outSize, parallel.output, index);
  }

  poplar::Interval partitionCopy(unsigned copySize, unsigned index) const {
    return partition(copySize, parallel.copy, index);
  }

  poplar::Interval partitionExtend(unsigned extendSize, unsigned index) const {
    return partition(extendSize, parallel.extend, index);
  }

  poplar::Interval partitionExtendVertices(unsigned extendSize,
                                           unsigned index) const {
    return partition(extendSize, parallel.extendVerticesPerPartition, index);
  }

  unsigned getTile(unsigned batch, unsigned time, unsigned batchEntry) const {
    return batch * (parallel.time * batchEntryPartitions()) // Batch
           + time * batchEntryPartitions()                  // Time
           + batchEntry;                                    // Batch entry
  }
  // Tile allocation when splitting across batch and time dimensions only
  unsigned getTile(unsigned batch, unsigned time) const {
    return batch * (parallel.time * batchEntryPartitions()) // Batch
           + time;                                          // Time
  }

  unsigned numTiles() const { return parallel.batch * batchEntryPartitions(); }

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
