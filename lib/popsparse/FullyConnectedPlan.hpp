// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_FullyConnectedPlan_hpp
#define popsparse_FullyConnectedPlan_hpp

#include <ostream>
#include <vector>

#include <poplar/OptionFlags.hpp>
#include <poplar/Target.hpp>
#include <poplar/Type.hpp>

#include "FullyConnectedPNMapping.hpp"
#include "FullyConnectedVector.hpp"
#include "popsparse/FullyConnected.hpp"

#include <popsolver/Model.hpp>

namespace popsparse {

namespace dynamic {
class FullyConnectedParams;
} // end namespace dynamic

namespace fullyconnected {

template <typename T> struct Estimates {
  Estimates() = default;
  Estimates(const T cycles, const T tempBytes)
      : cycles(cycles), tempBytes(tempBytes) {}

  T cycles;
  T tempBytes;
};

using Cost = Estimates<popsolver::DataType>;

inline bool operator==(Cost a, Cost b) {
  return a.cycles == b.cycles && a.tempBytes == b.tempBytes;
}

inline bool operator!=(Cost a, Cost b) { return !(a == b); }

std::ostream &operator<<(std::ostream &os, const Cost &c);

// Method used on tile in the plan
enum class OnTileMethod {
  // Codelet for Forward pass with full meta-info
  Forward,
  // Codelet for GradA pass with full meta-info
  GradA,
  // Codelet for GradA pass reusing forward pass meta-info
  Transpose,
  // Codelet for GradW pass reusing forward pass meta-info
  GradW,
};

std::ostream &operator<<(std::ostream &os, const OnTileMethod &m);

// This structure describes how to implement the passes of a fully
// connected layer.
struct Plan {
  // The grain-size in terms of elements of each dimension which the
  // vertex used can handle multiples of.
  Vector<unsigned> grouping;
  // This structure describes how different dimensions of a particular
  // sparse-dense matmul are partitioned to spread computational load
  // and memory usage across tiles.
  Vector<unsigned> partition;
  // This describes how buckets are partitioned for the initial
  // exchange and compute step. TODO: This might be better off
  // represented as a broadcast factor or similar (i.e.
  // partition / initialDistributionBucketPartition) as that's really
  // what we're doing. It would also play better with multiple levels of
  // hierarchy I believe (needs thinking through).
  Vector<unsigned> initialDistributionBucketPartition;
  // Method to use for mapping partitions to processor nodes.
  PartitionToPNMappingOrder mappingOrder;
  // Number of non-zero elements per bucket.
  unsigned nzElemsPerBucket;
  // Number of meta-info elements per bucket (Forward pass).
  unsigned fwdMetaInfoElemsPerBucket;
  // Number of meta-info elements per bucket (GradA pass).
  unsigned gradAMetaInfoElemsPerBucket;
  // Method used on-tile for each pass.
  OnTileMethod fwdMethod;
  OnTileMethod gradAMethod;
  OnTileMethod gradWMethod;

  // returns true if the same bucket is shared between passes
  bool sharedBuckets() const { return gradAMethod == OnTileMethod::Transpose; }
};

std::ostream &operator<<(std::ostream &os, const Plan &p);

std::array<std::vector<std::size_t>, 3>
getPartitionStartIndices(const popsparse::dynamic::FullyConnectedParams &params,
                         const Plan &plan);

std::tuple<Plan, Cost>
getPlan(const poplar::Target &target, const poplar::Type &inputType,
        const popsparse::dynamic::FullyConnectedParams &params,
        const poplar::OptionFlags &options = {},
        popsparse::dynamic::PlanningCache *cache = nullptr);

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedPlan_hpp
