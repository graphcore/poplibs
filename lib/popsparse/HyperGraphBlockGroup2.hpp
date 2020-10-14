// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphBlockGroup2_hpp
#define popsparse_HyperGraphBlockGroup2_hpp

#include "HyperGraphBlockGroup.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace popsparse {
namespace experimental {

/*
Partitions graph in a following way:
Collects V nodes into tiles along a partition, row, column or partition, column,
row. With it it tries to make number of multiplications per tile the same even
if for this we have to "spill" tile to different row/column. Also, the order of
split on last dimension is alternates, folllowing zig-zag pattern. This is to
maximize the probability of putting the same block of matrix A(B) on a current
tile when we have to collect the blocks for the current tile beyond the last
dimension in case when we have non-even split. Example:

Mat C[4][4] Mat B is dense (for simplicity)
P = 2
4 x 4 x 2 = 32 vertices

Case 1.
16 tiles
32 / 16 = 2 vertices per tile

column-row split:
k part 1:
 0  3  4  7
 0  3  4  7
 1  2  5  6
 1  2  5  6

k part 2:
 8 11 12 15
 8 11 12 15
 9 10 13 14
 9 10 13 14

Case 2.
10 tiles
32 / 10 = 3.2 vertices per tile

column-row split:
partition 1:
 0  2  2  5   // For tile 2 the same block of matrix A is used for 2nd and 3rd
columns 0  2  3  4 0  1  3  4 1  1  3  4

partition 2:
 5  7  8  9
 5  7  8  9
 6  7  8  9
 6  6  8  9   // For tile 6 the same block of matrix A is used for 1st and 2nd
columns

The best number of P and whether to do row/column or column/row split
is chosen based on the best memory statistics.
The start k number of P is calculated based on a model with uniform
sparsity. Then several searches around this number is performed.
*/
class HyperGraphBlockGroup2 : public HyperGraphBlockGroup {

public:
  HyperGraphBlockGroup2(BlockMatrix &A, BlockMatrix &B,
                        poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
                        poplar::Type partialDataTypeIn, int nTileIn);

  virtual ~HyperGraphBlockGroup2() = default;

  virtual void partitionGraph() override;

private:
  // Populates compute nodes V
  virtual void populateNodesV(int nRowC, int nColC,
                              const std::vector<std::vector<int>> &) override;

  // Performs preliminary or final partitioning of the graph
  void partitionGraphSearch(std::size_t &maxBytesOnTile, bool final);

protected:
  virtual void mapCNodes(poplar::Graph &graph) override;

  // Populates node V
  ComputeNode populateNodeV(
      unsigned row, unsigned col, unsigned p, int &nodeId,
      const std::vector<std::pair<unsigned int, unsigned int>> &aList,
      const std::vector<std::pair<unsigned int, unsigned int>> &bList);
};

} // namespace experimental
} // namespace popsparse

#endif