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
  // Codelet for Forward pass with full meta-info for element-wise sparsity.
  Forward,
  // Codelet for GradA pass with full meta-info for element-wise sparsity.
  GradA,
  // Codelet for GradA pass reusing forward pass meta-info for element-wise
  // sparsity.
  Transpose,
  // Codelet for GradW pass reusing forward pass meta-info for element-wise
  // sparsity.
  GradW,
  // Codelet for Forward pass with full meta-info for block-sparsity utilising
  // AMP.
  ForwardAMPBlock,
  // Codelet for GradA pass reusing forward pass meta-info for block-sparsity
  // utilising AMP.
  TransposeAMPBlock,
  // Codelet for GradW pass reusing forward pass meta-info for block-sparsity.
  GradWBlock,
  // Codelet for GradW pass reusing forward pass meta-info for block-sparsity
  // and using AMP to split block column dimension.
  GradWAMPBlock,
};

std::ostream &operator<<(std::ostream &os, const OnTileMethod &m);

struct Method {
  // The grain-size in terms of elements of each dimension which the
  // vertex used can handle multiples of. The X/Y grouping will
  // typically be the block-size.
  Vector<unsigned> grouping;
  // Method used on-tile for each pass.
  OnTileMethod fwd;
  OnTileMethod gradA;
  OnTileMethod gradW;
};

std::ostream &operator<<(std::ostream &os, const Method &m);

struct ExchangeAndMappingPlan {
  // Method to use for mapping partitions to processor nodes.
  PartitionToPNMapping fwdMapping;
  PartitionToPNMapping gradAMapping;
  PartitionToPNMapping gradWMapping;
};

std::ostream &operator<<(std::ostream &os, const ExchangeAndMappingPlan &p);

// This structure describes how to implement the passes of a fully
// connected layer.
struct Plan {
  Method method;
  // This structure describes how different dimensions of a particular
  // sparse-dense matmul are partitioned to spread computational load
  // and memory usage across tiles.
  Vector<unsigned> partition;

  // This gives the number of partitions handled in the initial distribution
  // phase. This must be an exact divisor of partitions.
  Vector<unsigned> initialDistributionPartitions;
  // This gives the number of partitions handled in the propagation
  // phase. This must be an exact divisor of partition.
  Vector<unsigned> gradWPropagationPartitions;
  // Determines tile layout and patterns/methods used to exchange data
  // in different stages of the operation.
  ExchangeAndMappingPlan exchangePlan;
  // Number of non-zero elements per bucket.
  unsigned nzElemsPerBucket;
  // Number of meta-info elements per bucket (Forward pass).
  unsigned fwdMetaInfoElemsPerBucket;
  // Number of meta-info elements per bucket (GradA pass).
  unsigned gradAMetaInfoElemsPerBucket;
  // returns true if the same bucket is shared between passes
  bool sharedBuckets() const {
    return method.gradA == OnTileMethod::Transpose ||
           method.gradA == OnTileMethod::TransposeAMPBlock;
  }
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
