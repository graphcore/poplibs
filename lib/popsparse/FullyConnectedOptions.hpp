// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_FullyConnectedOptions_hpp
#define popsparse_FullyConnectedOptions_hpp

#include <poplar/OptionFlags.hpp>
#include <poplar/Type.hpp>

#include <ostream>
#include <tuple>

#include "poplibs_support/PlanConstraints.hpp"

#include "SparsePartitionerOptions.hpp"

namespace popsparse {
namespace fullyconnected {

struct Options {
  // The proportion of available memory on each tile that this layer should
  // at most consume temporarily during the course of the operation.
  double availableMemoryProportion = 0.6;
  // This gives additional elements to allocate in each bucket of meta-info
  // as a proportion of the required size for a perfectly uniformly distributed
  // sparsity pattern.
  double metaInfoBucketOversizeProportion = 0.3;
  // Indicates which passes are present for the operation of the layer
  // as a whole. It is assumed that the forward pass is always present.
  bool doGradAPass = false;
  bool doGradWPass = false;
  // What type to use for partial results.
  poplar::Type partialsType = poplar::FLOAT;
  // If set, forces the buckets to be used for all three passes to be the same
  bool sharedBuckets = true;
  // Enable structured rearrangements
  bool enableStructuredRearrangements = true;
  PartitionerOptions partitioner;
  // Constraints on the plan used
  poplibs_support::PlanConstraints planConstraints;

  friend bool operator<(const Options &a, const Options &b);
  friend bool operator==(const Options &a, const Options &b);
  friend bool operator!=(const Options &a, const Options &b);
  friend std::ostream &operator<<(std::ostream &os, const Options &o);
};

Options parseOptionFlags(const poplar::OptionFlags &flags);

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedOptions_hpp
