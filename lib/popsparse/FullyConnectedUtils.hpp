// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popsparse_FullyConnectedUtils_hpp
#define popsparse_FullyConnectedUtils_hpp

#include <poplar/Tensor.hpp>

#include "SparseStorageInternal.hpp"
#include "popsparse/SparseTensor.hpp"

#include <ostream>

namespace popsparse {
namespace fullyconnected {

unsigned calculateSubGroupId(unsigned numRowGroups, unsigned numSubRowGroups,
                             unsigned rowGroupIndex, unsigned subRowGroupIndex);

// returns Row and Sub-row indices for a given subgroup id
std::pair<unsigned, unsigned> getGroupIndices(unsigned subGroupId,
                                              unsigned numRowGroups,
                                              unsigned numSubRowGroups);

// returns distance for Row and Sub-row indices between two sub-groups
std::pair<std::size_t, std::size_t>
distanceToSubGroup(unsigned srcId, unsigned dstId, unsigned numRowGroups,
                   unsigned numSubRowGroups);

poplar::Tensor factorDims(const poplar::Tensor &t,
                          const std::vector<std::size_t> &factors,
                          unsigned startDim = 0);
poplar::Tensor unfactorDims(const poplar::Tensor &t, unsigned numDims,
                            unsigned startDim = 0);

poplar::Tensor inputExternalToInternalShape(const poplar::Tensor &t,
                                            std::size_t numGroups);

poplar::Tensor inputInternalToExternalShape(const poplar::Tensor &t,
                                            std::size_t numGroups);

poplar::Tensor weightsExternalToInternalShape(const poplar::Tensor &t,
                                              std::size_t elemsPerBucket);
poplar::Tensor weightsInternalToExternalShape(const poplar::Tensor &t,
                                              std::size_t elemsPerBucket);
popsparse::dynamic::SparseTensor
weightsExternalToInternalShape(const popsparse::dynamic::SparseTensor &t,
                               std::size_t metaInfoElemsPerBucket,
                               std::size_t nzValuesPerBucket);

popsparse::dynamic::SparseTensor
weightsInternalToExternalShape(const popsparse::dynamic::SparseTensor &t,
                               std::size_t metaInfoElemsPerBucket,
                               std::size_t nzValuesPerBucket);

poplar::Tensor weightsInternalSliceBuckets(const poplar::Tensor &t,
                                           std::size_t offset,
                                           std::size_t numElems);
popsparse::dynamic::SparseTensor
weightsInternalSliceBuckets(const popsparse::dynamic::SparseTensor &t,
                            std::size_t offset, std::size_t numElems);

popsparse::dynamic::SparseTensor
packWeights(const popsparse::dynamic::SparseTensor &buckets,
            std::size_t metaInfoElemsPerBucket, std::size_t nzValuesPerBucket,
            const poplar::Tensor &overflowInfo);

std::tuple<poplar::Tensor, poplar::Tensor>
unpackWeights(const poplar::Tensor &metaInfo, std::size_t overflowInfoElems,
              std::size_t metaInfoElemsPerBucket);

std::tuple<popsparse::dynamic::SparseTensor, poplar::Tensor>
unpackWeights(const popsparse::dynamic::SparseTensor &weights,
              std::size_t overflowInfoElems, std::size_t metaInfoElemsPerBucket,
              std::size_t nzValuesPerBucket);

// Get the number of meta-info elements
std::size_t getNumOverflowInfoElems(std::size_t metaInfoTypeBits,
                                    std::size_t xSplits, std::size_t ySplits,
                                    std::size_t zSplits);

// Splits the work for the output tile given by size 'numRows' x 'numColumns'.
// The work is split at most across a given number of workers. 'rowWeights' is
//  'numRows' sized vector  that computes the number of non zero elements in
// each of the non-sparse rows.
std::vector<Tile>
splitTileBetweenWorkers(std::size_t numRows, std::size_t numColumns,
                        std::size_t numWorkers,
                        const std::vector<std::size_t> &rowWeights = {});

// Convert between 2 different ways of representing number of non-zero elements
// in the API.
double convertAbsoluteNzElemsToRatio(std::size_t numGroups,
                                     std::size_t inputSize,
                                     std::size_t outputSize,
                                     std::size_t numNonZeroElems);
std::size_t convertRatioNzElemsToAbsolute(std::size_t numGroups,
                                          std::size_t inputSize,
                                          std::size_t outputSize,
                                          double nonZeroRatio);

} // end namespace fullyconnected
} // end namespace popsparse

#endif // popsparse_FullyConnectedUtils_hpp
