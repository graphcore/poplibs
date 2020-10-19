// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef _popsparse_SparseDensePartitionBlock_hpp
#define _popsparse_SparseDensePartitionBlock_hpp

#include "SparseDenseUtils.hpp"
#include <poplar/Graph.hpp>
#include <random>

// Test functions to generate sparse tensor data and metadata for testing
// codelets that use blockwise sparse data

bool isGradWVertex(VertexType vt);

template <typename RandomEngine>
std::vector<std::array<unsigned, 2>> generateBlockSparseIndices(
    RandomEngine &randomEngine, const std::vector<std::size_t> &shape,
    const std::vector<std::size_t> &blockSize, std::size_t n);

std::vector<unsigned int>
getForwardWorkerPartition(const poplar::Target &target, unsigned bColumns);

unsigned getGradWWorkerPartition(const poplar::Target &target,
                                 const std::vector<unsigned> &aRowColumnCounts);

template <typename RandomEngine>
std::vector<std::vector<unsigned>> generateMetaInfoAndPartition(
    RandomEngine &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const std::vector<std::size_t> &blockSize, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &partialType, VertexType vertexType,
    unsigned xPartition = 0, unsigned yPartition = 0);

#endif // _popsparse_SparseDensePartitionBlock_hpp
