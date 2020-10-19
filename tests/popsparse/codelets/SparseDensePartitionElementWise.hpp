// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef _popsparse_SparseDensePartitionElementWise_hpp
#define _popsparse_SparseDensePartitionElementWise_hpp

#include "SparseDenseUtils.hpp"
#include <poplar/Graph.hpp>
#include <random>

// Test functions to generate sparse tensor data and metadata for testing
// codelets that use elementwise sparse data

template <typename RandomEngine>
std::vector<std::array<unsigned, 2>>
generateSparseIndices(RandomEngine &randomEngine,
                      const std::vector<std::size_t> &shape, std::size_t n);

std::tuple<unsigned, unsigned, unsigned>
getForwardWorkerPartition(const poplar::Target &target,
                          const poplar::Type &inputType,
                          const unsigned bColumns, const unsigned aRows,
                          const std::vector<unsigned> &aRowColumnCounts);

std::tuple<unsigned, unsigned>
getGradWWorkerPartition(const poplar::Target &target,
                        const poplar::Type &inputType, const unsigned bColumns,
                        const unsigned aRows,
                        const std::vector<unsigned> &aRowColumnCounts);

template <typename RandomEngine>
std::vector<std::vector<unsigned>> generateMetaInfoAndPartition(
    RandomEngine &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &partialType, VertexType vertexType,
    unsigned xPartition = 0, unsigned yPartition = 0);

#endif // _popsparse_SparseDensePartitionElementWise_hpp
