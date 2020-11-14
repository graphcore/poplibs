// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PlanningCacheImpl.hpp"
#include <popsparse/PlanningCache.hpp>
#include <poputil/DebugInfo.hpp>

namespace poputil {
template <>
poplar::ProfileValue
toProfileValue(const popsparse::dynamic::PlanningCache &t) {
  return poplar::ProfileValue("<popsparse::dynamic::PlanningCache>");
}
} // namespace poputil

namespace popsparse {
namespace dynamic {

PlanningCache::PlanningCache() {
  impl = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl());
}

PlanningCache::~PlanningCache() = default;

} // end namespace dynamic
} // end namespace popsparse
