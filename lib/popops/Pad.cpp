// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#include "Padder.hpp"
#include <cstdlib>
#include <poplar/Graph.hpp>

namespace popops {

poplar::Tensor pad(poplar::Graph &graph,
                   const poplar::Tensor &t,
                   const std::vector<ptrdiff_t> &pLows,
                   const std::vector<ptrdiff_t> &pUpps,
                   float val) {
  padding::ConstPadder padder(graph, val);
  return padder.getPaddedTensor(t, pLows, pUpps);
}

poplar::Tensor pad(const poplar::Tensor &t,
                   const std::vector<ptrdiff_t> &pLows,
                   const std::vector<ptrdiff_t> &pUpps,
                   padding::Type type) {
  auto ptrPadder = padding::getPtrPadder(type);
  return ptrPadder->getPaddedTensor(t, pLows, pUpps);
}

poplar::Tensor pad(poplar::Graph &graph,
                   const poplar::Tensor &t,
                   ptrdiff_t pLow,
                   ptrdiff_t pUpp,
                   unsigned dim,
                   float val) {
  padding::ConstPadder padder(graph, val);
  return padder.getPartPaddedTensor(t, dim, pLow, pUpp);
}

poplar::Tensor pad(const poplar::Tensor &t,
                   ptrdiff_t pLow,
                   ptrdiff_t pUpp,
                   unsigned dim,
                   const padding::Type type) {
  auto ptrPadder = padding::getPtrPadder(type);
  return ptrPadder->getPartPaddedTensor(t, dim, pLow, pUpp);
}
}
