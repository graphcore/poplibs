// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_ZoltanPartitioner_hpp
#define popsparse_ZoltanPartitioner_hpp

#include "HyperGraphPartitioner.hpp"

namespace popsparse {
namespace experimental {

/*
A wrapper around Zoltan partitioning library
*/
class ZoltanPartitioner : public HyperGraphPartitioner {
public:
  enum class PartitionType { BLOCK, HYPERGRAPH };

  ZoltanPartitioner(PartitionType partitionTypeIn)
      : partitionType(partitionTypeIn) {}

  virtual ~ZoltanPartitioner() = default;

  virtual float partitionGraph(const HyperGraphData &graphData, int nPartition,
                               std::vector<int> &nodeAssignment) override;

  PartitionType partitionType;
};

} // namespace experimental
} // namespace popsparse

#endif