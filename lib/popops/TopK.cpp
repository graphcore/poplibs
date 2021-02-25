// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popops/TopK.hpp>

#include <popops/Encoding.hpp>

#include <poputil/DebugInfo.hpp>
#include <poputil/TileMapping.hpp>

#include <poplibs_support/logging.hpp>
#include <poplibs_support/print.hpp>

#include "BitonicTopK.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poputil;

namespace poputil {

template <> ProfileValue toProfileValue(const popops::TopKParams &p) {
  ProfileValue::Map v;
  v.emplace("k", toProfileValue(p.k));
  v.emplace("largest", toProfileValue(p.largest));
  v.emplace("sortOrder", toProfileValue(p.sortOrder));
  return v;
}

} // namespace poputil

namespace popops {

TopKParams::TopKParams(unsigned k, bool largest, SortOrder sortOrder) noexcept
    : k(k), largest(largest), sortOrder(sortOrder) {}

std::ostream &operator<<(std::ostream &os, const TopKParams &p) {
  os << "{"
     << "k=" << p.k << ", largest=" << (p.largest ? "true" : "false")
     << ", sortOrder=" << p.sortOrder << "}";
  return os;
}

Tensor createTopKInput(Graph &graph, const Type &type,
                       const std::vector<std::size_t> &shape,
                       const TopKParams &params,
                       const DebugContext &debugContext) {
  logging::popops::info("createTopKInput(shape={}, params={}, debugPath='{}')",
                        shape, params, debugContext.getPathName());
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(shape, params),
                                 "createTopKInput");
  const auto t = bitonic::createTopKInputImpl(graph, type, shape, {di});
  di.addOutput(t);
  return t;
}

Tensor topK(Graph &graph, Sequence &prog, const Tensor &t,
            const TopKParams &params, const DebugContext &debugContext) {
  logging::popops::info("topK(shape={}, params={}, debugPath='{}')", t.shape(),
                        params, debugContext.getPathName());
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, params), "topK");
  const bool sorted = params.sortOrder != SortOrder::NONE;
  const bool ascending = params.sortOrder != SortOrder::DESCENDING;
  const auto result = bitonic::topKImpl(graph, prog, t, std::nullopt, params.k,
                                        params.largest, sorted, ascending, {di})
                          .first;
  di.addOutput(result);
  return result;
}

std::pair<Tensor, Tensor> topKKeyValue(Graph &graph, Sequence &prog,
                                       const Tensor &key, const Tensor &value,
                                       const TopKParams &params,
                                       const DebugContext &debugContext) {
  logging::popops::info("topKKeyValue(shape={}, params={}, debugPath='{}')",
                        key.shape(), params, debugContext.getPathName());
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(key, value, params),
                                 "topKKeyValue");
  const bool sorted = params.sortOrder != SortOrder::NONE;
  const bool ascending = params.sortOrder != SortOrder::DESCENDING;
  const auto result =
      bitonic::topKImpl(graph, prog, key, value, params.k, params.largest,
                        sorted, ascending, {di});
  di.addOutput(result.first);
  di.addOutput(result.second);
  return result;
}

std::pair<Tensor, Tensor>
topKWithPermutation(Graph &graph, Sequence &prog, const Tensor &t,
                    const TopKParams &params,
                    const DebugContext &debugContext) {
  logging::popops::info(
      "topKWithPermutation(shape={}, params={}, debugPath='{}')", t.shape(),
      params, debugContext.getPathName());
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, params));
  const bool sorted = params.sortOrder != SortOrder::NONE;
  const bool ascending = params.sortOrder != SortOrder::DESCENDING;

  const auto numElems = t.numElements();
  const auto n = t.dim(t.rank() - 1);
  const auto b = numElems / t.dim(t.rank() - 1);
  const auto indicesToPermute =
      createTopKInput(graph, UNSIGNED_INT, t.shape(), params, {di, "indices"});
  if (b == 1) {
    iota(graph, indicesToPermute.flatten(), 0u, prog, {di});
  } else {
    // T34944, a 2-dimensional iota API with a second dimension in which to
    // broadcast would be useful here.
    std::vector<std::size_t> singleBatchIndices(n);
    std::iota(singleBatchIndices.begin(), singleBatchIndices.end(), 0);
    const auto iota =
        graph.addConstant(UNSIGNED_INT, {1, n}, ArrayRef(singleBatchIndices),
                          {di, "indicesInitializer"});
    poputil::mapTensorLinearly(graph, iota);
    prog.add(Copy(iota.broadcast(b, 0).flatten(), indicesToPermute.flatten()));
  }
  const auto result =
      bitonic::topKImpl(graph, prog, t, indicesToPermute, params.k,
                        params.largest, sorted, ascending, {di});
  di.addOutput(result.first);
  di.addOutput(result.second);
  return result;
}

} // end namespace popops
