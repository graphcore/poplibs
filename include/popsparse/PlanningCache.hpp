// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Caching of plans for dynamically sparse operations.
 */

#ifndef popsparse_PlanningCache_hpp
#define popsparse_PlanningCache_hpp

#include <memory>

namespace poplin {

class PlanningCache;

} // namespace poplin

namespace popsparse {
namespace dynamic {

class PlanningCacheImpl;

/** Class used to cache the calculation of plans for dynamically sparse
 *  operations. This is optional and speeds up graph construction for these
 *  operations.
 */
class PlanningCache {
public:
  PlanningCache();
  PlanningCache(poplin::PlanningCache *planningCache);
  ~PlanningCache();
  std::unique_ptr<PlanningCacheImpl> impl;
};

} // end namespace dynamic

namespace static_ {

class PlanningCacheImpl;

/** Class used to cache the calculation of plans for staticaally sparse
 *  operations. This is optional and speeds up graph construction for these
 *  operations.
 */
class PlanningCache {
public:
  PlanningCache();
  PlanningCache(poplin::PlanningCache *planningCache);
  ~PlanningCache();
  std::unique_ptr<PlanningCacheImpl> impl;
};

} // end namespace static_

} // end namespace popsparse

#endif // popsparse_PlanningCache_hpp
