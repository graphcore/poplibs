// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "popops/Zero.hpp"

#include "popops/Fill.hpp"

namespace popops {

void zero(poplar::Graph &graph, poplar::Tensor t,
          const std::vector<poplar::Interval> &tileRegions, unsigned tile,
          poplar::ComputeSet zeroCS) {
  fill(graph, t, tileRegions, tile, zeroCS, 0);
}

void zero(poplar::Graph &graph, const poplar::Tensor &t, unsigned tile,
          poplar::ComputeSet zeroCS) {
  fill(graph, t, tile, zeroCS, 0);
}

void zero(poplar::Graph &graph, const poplar::Tensor &t,
          const std::vector<std::vector<poplar::Interval>> &mapping,
          poplar::ComputeSet zeroCS) {
  fill(graph, t, mapping, zeroCS, 0);
}

void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, const std::string &debugPrefix) {
  fill(graph, t, prog, 0, debugPrefix);
}

} // end namespace popops
