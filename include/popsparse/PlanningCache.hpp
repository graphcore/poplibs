// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_PlanningCache_hpp
#define popsparse_PlanningCache_hpp

#include <memory>

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
  ~PlanningCache();
  std::unique_ptr<PlanningCacheImpl> impl;
};

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_PlanningCache_hpp
