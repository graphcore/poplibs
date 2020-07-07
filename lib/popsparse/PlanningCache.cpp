// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popsparse/PlanningCache.hpp>

#include "PlanningCacheImpl.hpp"

namespace popsparse {
namespace dynamic {

PlanningCache::PlanningCache() {
  impl = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl());
}

PlanningCache::~PlanningCache() = default;

} // end namespace dynamic
} // end namespace popsparse
