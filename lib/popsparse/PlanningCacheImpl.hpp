// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_PlanningCacheImpl_hpp
#define popsparse_PlanningCacheImpl_hpp

#include <map>

// Stuff that needs storing in the cache
#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPlan.hpp"
#include <poplin/MatMul.hpp>
#include <popsparse/FullyConnectedParams.hpp>

namespace popsparse {
namespace dynamic {

class PlanningCacheImpl {
  using DenseCacheType = poplin::PlanningCache;
  // If no dense cache provided then create one managed by this struct
  // If it does already exist denseCache will point to this and this
  // unique pointer will be empty
  std::unique_ptr<DenseCacheType> matMulCache;

public:
  struct Key {
    FullyConnectedParams params;
    fullyconnected::Options options;
    Key(FullyConnectedParams params, fullyconnected::Options options)
        : params(std::move(params)), options(std::move(options)) {}
    Key() = default;
    bool operator<(const Key &other) const {
      return std::tie(params, options) < std::tie(other.params, other.options);
    }
    bool operator==(const Key &other) const {
      return std::tie(params, options) == std::tie(other.params, other.options);
    }
    bool operator!=(const Key &other) const { return !(*this == other); }
  };

  std::map<Key, std::tuple<fullyconnected::Plan, fullyconnected::Cost>> plans;
  DenseCacheType *denseCache;

  PlanningCacheImpl(DenseCacheType *denseCache) : denseCache(denseCache) {}
  PlanningCacheImpl()
      : matMulCache(new DenseCacheType()), denseCache(matMulCache.get()) {}
};

} // end namespace dynamic
} // end namespace popsparse

#endif // popsparse_PlanningCacheImpl_hpp
