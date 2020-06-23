// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popops/Fill.hpp"

namespace popops {

void fill(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, const void *fillValue,
          const poplar::TypeTraits &traits, const std::string &debugPrefix) {
  const poplar::Tensor fillTensor =
      graph.addConstant(t.elementType(), t.shape(), fillValue, traits, true,
                        debugPrefix + "/fill");
  graph.setTileMapping(fillTensor, graph.getTileMapping(t));
  prog.add(poplar::program::Copy(fillTensor, t));
}

} // namespace popops