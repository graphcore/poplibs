// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#include "Padder.hpp"
#include <cstdlib>
#include <poplar/Graph.hpp>

namespace popops {
namespace {
template <typename T>
poplar::Tensor padImpl(poplar::Graph &graph, const poplar::Tensor &t,
                       const std::vector<std::ptrdiff_t> &pLows,
                       const std::vector<std::ptrdiff_t> &pUpps, T val,
                       padding::MappingMethod mappingMethod) {
  padding::ValuePadder<T> padder(graph, val, mappingMethod);
  return padder.getPaddedTensor(t, pLows, pUpps);
}

template <typename T>
poplar::Tensor padImpl(poplar::Graph &graph, const poplar::Tensor &t,
                       std::ptrdiff_t pLow, std::ptrdiff_t pUpp, unsigned dim,
                       T val, padding::MappingMethod mappingMethod) {
  padding::ValuePadder<T> padder(graph, val, mappingMethod);
  return padder.getPartPaddedTensor(t, dim, pLow, pUpp);
}
} // namespace

poplar::Tensor pad(poplar::Graph &graph, const poplar::Tensor &t,
                   const std::vector<std::ptrdiff_t> &paddingLower,
                   const std::vector<std::ptrdiff_t> &paddingUpper, float val,
                   padding::MappingMethod mappingMethod) {
  return padImpl(graph, t, paddingLower, paddingUpper, val, mappingMethod);
}

poplar::Tensor pad(poplar::Graph &graph, const poplar::Tensor &t,
                   const std::vector<std::ptrdiff_t> &paddingLower,
                   const std::vector<std::ptrdiff_t> &paddingUpper, int val,
                   padding::MappingMethod mappingMethod) {
  return padImpl(graph, t, paddingLower, paddingUpper, val, mappingMethod);
}

poplar::Tensor pad(poplar::Graph &graph, const poplar::Tensor &t,
                   const std::vector<std::ptrdiff_t> &paddingLower,
                   const std::vector<std::ptrdiff_t> &paddingUpper,
                   const poplar::Tensor &val,
                   padding::MappingMethod mappingMethod) {
  return padImpl(graph, t, paddingLower, paddingUpper, val, mappingMethod);
}

poplar::Tensor pad(poplar::Graph &graph, const poplar::Tensor &t,
                   std::ptrdiff_t paddingLower, std::ptrdiff_t paddingUpper,
                   unsigned dim, float val,
                   padding::MappingMethod mappingMethod) {
  return padImpl(graph, t, paddingLower, paddingUpper, dim, val, mappingMethod);
}

poplar::Tensor pad(poplar::Graph &graph, const poplar::Tensor &t,
                   std::ptrdiff_t paddingLower, std::ptrdiff_t paddingUpper,
                   unsigned dim, int val,
                   padding::MappingMethod mappingMethod) {
  return padImpl(graph, t, paddingLower, paddingUpper, dim, val, mappingMethod);
}

poplar::Tensor pad(poplar::Graph &graph, const poplar::Tensor &t,
                   std::ptrdiff_t paddingLower, std::ptrdiff_t paddingUpper,
                   unsigned dim, const poplar::Tensor &val,
                   padding::MappingMethod mappingMethod) {
  return padImpl(graph, t, paddingLower, paddingUpper, dim, val, mappingMethod);
}

poplar::Tensor pad(const poplar::Tensor &t, const std::vector<ptrdiff_t> &pLows,
                   const std::vector<ptrdiff_t> &pUpps, padding::Type type) {
  auto ptrPadder = padding::getPtrPadder(type);
  return ptrPadder->getPaddedTensor(t, pLows, pUpps);
}

poplar::Tensor pad(const poplar::Tensor &t, ptrdiff_t pLow, ptrdiff_t pUpp,
                   unsigned dim, const padding::Type type) {
  auto ptrPadder = padding::getPtrPadder(type);
  return ptrPadder->getPartPaddedTensor(t, dim, pLow, pUpp);
}
} // namespace popops
