
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_BalancedPartitioner_hpp
#define popsparse_BalancedPartitioner_hpp

#include "HyperGraphPartitioner.hpp"

namespace popsparse {
namespace experimental {

class BalancedPartitioner : public HyperGraphPartitioner {
public:
  BalancedPartitioner() = default;

  virtual ~BalancedPartitioner() = default;

  virtual float partitionGraph(const HyperGraphData &graphData, int nPartition,
                               std::vector<int> &nodeAssignment) override;

public:
  static void partition(const std::vector<float> &nodeW, int nPartition,
                        std::vector<int> &nodeAssignment);
};

} // namespace experimental
} // namespace popsparse

#endif