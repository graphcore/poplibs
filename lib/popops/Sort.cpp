// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include "popops/Sort.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include <poplibs_support/Algorithms.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/TopK.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <algorithm>

using namespace poplibs_support;

namespace popops {

poplar::Tensor sort(poplar::Graph &graph, const poplar::Tensor &t, unsigned dim,
                    poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, dim));

  poplar::Tensor result;
  const auto n = t.dim(dim);
  const bool largest = true;
  const popops::TopKParams params(n, largest, popops::SortOrder::ASCENDING,
                                  false);
  auto in = t.dimRoll(dim, t.rank() - 1);
  result = popops::topK(graph, prog, in, params, {di});
  result = result.dimRoll(result.rank() - 1, dim);
  di.addOutput(result);
  return result;
}

void sortInPlace(poplar::Graph &graph, const poplar::Tensor &t, unsigned dim,
                 poplar::program::Sequence &prog,
                 const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, dim));

  if (dim >= t.rank()) {
    throw poputil::poplibs_error(
        "Chosen sort dimension does not refer to a valid "
        "dimension in the input tensor");
  }

  const auto n = t.dim(dim);
  const bool largest = true;
  const popops::TopKParams params(n, largest, popops::SortOrder::ASCENDING,
                                  false);
  auto in = t.dimRoll(dim, t.rank() - 1);
  // We just use the non in-place API with a copy after the fact
  // as with bitonic sort there is not much difference.
  auto result = popops::topK(graph, prog, in, params, {di});
  result = result.dimRoll(result.rank() - 1, dim);
  prog.add(poplar::program::Copy(result, t));
}

poplar::Tensor sortKeyValue(poplar::Graph &graph, const poplar::Tensor &k,
                            const poplar::Tensor &v, unsigned dim,
                            poplar::program::Sequence &prog,
                            const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(k, v, dim));

  auto key = k;
  auto value = v;
  const auto n = k.dim(dim);
  const bool largest = true;
  const TopKParams params(n, largest, popops::SortOrder::ASCENDING, false);
  key = key.dimRoll(dim, key.rank() - 1);
  value = value.dimRoll(dim, value.rank() - 1);
  std::tie(key, value) =
      popops::topKKeyValue(graph, prog, key, value, params, {di});
  key = key.dimRoll(key.rank() - 1, dim);
  value = value.dimRoll(value.rank() - 1, dim);

  di.addOutput(value);
  return value;
}

void sortKeyValueInPlace(poplar::Graph &graph, const poplar::Tensor &k,
                         const poplar::Tensor &v, unsigned dim,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(k, v, dim));
  if (k.shape() != v.shape()) {
    throw poputil::poplibs_error(
        "Key and Value arguments to sortKeyValue must be the same shape");
  }

  if (dim >= k.rank()) {
    throw poputil::poplibs_error(
        "Chosen sort dimension does not refer to a valid "
        "dimension in the input tensor");
  }

  auto key = k;
  auto value = v;
  const auto n = k.dim(dim);
  const bool largest = true;
  const TopKParams params(n, largest, popops::SortOrder::ASCENDING, false);
  key = key.dimRoll(dim, key.rank() - 1);
  value = value.dimRoll(dim, value.rank() - 1);
  // We just use the non in-place API with a copy after the fact
  // as with bitonic sort there is not much difference.
  std::tie(key, value) =
      popops::topKKeyValue(graph, prog, key, value, params, {di});
  key = key.dimRoll(key.rank() - 1, dim);
  value = value.dimRoll(value.rank() - 1, dim);

  prog.add(poplar::program::Copy(std::move(key), k));
  prog.add(poplar::program::Copy(std::move(value), v));
}

} // namespace popops
