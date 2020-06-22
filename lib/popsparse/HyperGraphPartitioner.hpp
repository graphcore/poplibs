// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphPartitioner_hpp
#define popsparse_HyperGraphPartitioner_hpp

#include <vector>

namespace popsparse {
namespace experimental {

/*
This is an abstract hypergraph description.
It uses the same terms as Zoltan.

Example:
Given hypergraph:
A1 -> V1
A1 -> V2
B1 -> V1
B2 -> V1
B2 -> V2

Nodes:
id label weight
0  A1    2.0
1  B1    1.0
2  B2    1.0
3  V1    1.0
4  V2    1.0

Hyperedges:
E1 [A1, V1, V2]
E2 [B1, V1]
E3 [B2, V1, V2]

The HyperGraphData will contain:
nodes: 5

weights: [2.0, 1.0, 1.0, 1.0, 1.0]

pins: [0, 3, 4, 1, 3, 2, 3, 4]

hyperEdges: [0, 3, 5]

*/
struct HyperGraphData {
  // Each pin correspond to one node
  // Pin element value is some index of weights list
  // pins is a flattened list of all hyperedges
  std::vector<unsigned int> pins;
  // Offsets into pins where each offset represents a start of a new hyperedge
  std::vector<unsigned int> hyperEdges;
  // Nodes weights.
  std::vector<float> weights;
  // Total number of nodes. The same as weights.size()
  unsigned int nodes;
};

/*
Abstract partitioner class.
Currently implemented by Zoltan partitioner only.
*/
class HyperGraphPartitioner {
public:
  virtual ~HyperGraphPartitioner() = default;

  /*
  Partition a hypergraph
  graphData - input hypergraph
  nPartition - number of partitions
  resulting partAssignment - assignment of nodes on partitions
  */
  virtual bool partitionGraph(const HyperGraphData &graphData, int nPartition,
                              std::vector<int> &partAssignment) = 0;
};

} // namespace experimental
} // namespace popsparse

#endif