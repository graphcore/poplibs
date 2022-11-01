// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PlanningCacheImpl.hpp"
#include <poplin/MatMul.hpp>
#include <popsparse/PlanningCache.hpp>
#include <poputil/DebugInfo.hpp>

namespace poputil {
template <>
poplar::ProfileValue
toProfileValue(const popsparse::dynamic::PlanningCache &t) {
  return poplar::ProfileValue("<popsparse::dynamic::PlanningCache>");
}

template <>
poplar::ProfileValue
toProfileValue(const popsparse::static_::PlanningCache &t) {
  return poplar::ProfileValue("<popsparse::static_::PlanningCache>");
}
} // namespace poputil

namespace popsparse {
namespace dynamic {

PlanningCache::PlanningCache() : impl(new PlanningCacheImpl()) {}

PlanningCache::PlanningCache(poplin::PlanningCache *planningCache)
    : impl(new PlanningCacheImpl(planningCache)) {}

PlanningCache::~PlanningCache() = default;

} // end namespace dynamic

namespace static_ {
PlanningCache::PlanningCache() : impl(new PlanningCacheImpl()) {}
PlanningCache::~PlanningCache() = default;
} // end namespace static_

} // end namespace popsparse
