// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_PlanningCacheImpl_hpp
#define popsparse_PlanningCacheImpl_hpp

#include <map>

// Stuff that needs storing in the cache
#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPlan.hpp"
#include "MatMulOptions.hpp"
#include "StaticMatMulPartitioner.hpp"
#include "popsparse/MatMulParams.hpp"
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

namespace static_ {
class PlanningCacheImpl {
public:
  struct Key {
    MatMulParams params;
    MatMulOptions options;
    // The row and column indices of the static matrix representation obtained
    // after canocialisation of a CSR matrix
    unsigned blockLength;
    std::vector<std::size_t> rowIndices;
    std::vector<std::size_t> columnIndices;
    Key(MatMulParams params, MatMulOptions options, unsigned blockLength,
        std::vector<std::size_t> rowIndices,
        std::vector<std::size_t> columnIndices)
        : params(std::move(params)), options(std::move(options)),
          blockLength(blockLength), rowIndices(std::move(rowIndices)),
          columnIndices(std::move(columnIndices)) {}
    Key() = default;
    bool operator<(const Key &other) const {
      return std::tie(params, options, blockLength, rowIndices, columnIndices) <
             std::tie(other.params, other.options, other.blockLength,
                      other.rowIndices, other.columnIndices);
    }
    bool operator==(const Key &other) const {
      return std::tie(params, options, blockLength, rowIndices,
                      columnIndices) ==
             std::tie(other.params, other.options, other.blockLength,
                      other.rowIndices, other.columnIndices);
    }
    bool operator!=(const Key &other) const { return !(*this == other); }
  };

  std::map<Key, Partition> plans;
};
} // end namespace static_

} // end namespace popsparse

#endif // popsparse_PlanningCacheImpl_hpp
