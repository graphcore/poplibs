#ifndef _FullyConnectedPlan_hpp_
#define _FullyConnectedPlan_hpp_
#include <string>
#include <map>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>

namespace poplin {

struct Plan {
  struct Partition {
    unsigned tilesPerColumn;
    unsigned tilesPerRow;
    Partition(unsigned tilesPerColumn, unsigned tilesPerRow) :
      tilesPerColumn(tilesPerColumn), tilesPerRow(tilesPerRow) {}
  };
  Partition ipuPartition;
  Plan(const Partition &ipuPartition) : ipuPartition(ipuPartition) {}
};

Plan getPlan(const poplar::Graph &graph,
             std::string dType,
             std::vector<std::size_t> aShape,
             std::vector<std::size_t> bShape,
             MatMulOptions options);

} // End namespace fc.

#endif // _FullyConnectedPlan_hpp_
