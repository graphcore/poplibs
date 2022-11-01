// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//

#ifndef _poplibs_popsparse_StaticSparseMatMulPartitioner_hpp_
#define _poplibs_popsparse_StaticSparseMatMulPartitioner_hpp_

#include "MatMulOptions.hpp"
#include "poplar/Interval.hpp"
#include "poplar/Target.hpp"
#include "poplar/Type.hpp"
#include "poplibs_support/print.hpp"
#include "popsparse/MatMulParams.hpp"
#include "popsparse/PlanningCache.hpp"
#include "popsparse/SparseStorageFormats.hpp"

namespace popsparse {
namespace static_ {

// Partition to split dimensions of the static matrix
struct Partition {
  // Information on the number of splits of the columns of the sparse lhs matrix
  struct BandInfo {
    // The column block for each band
    std::vector<unsigned> boundaries;
    // The number of NZ blocks per band
    std::vector<unsigned> nzBlocksPerBand;
    // The maximum number of column blocks. Can be obtained from band boundaries
    // but is kept as a separate field to avoid recomputing it.
    unsigned maxBandSize;
    const unsigned getNumColumnBands() const { return boundaries.size() - 1; }
  } bandInfo;

  // grain size of the columns of the dense rhs matrix
  unsigned nGrainSize;
  // grain size of the rows of the dense rhs matrix/columns of sparse lhs matrix
  unsigned kGrainSize;
  // grain size of the rows of the sparse lhs matrix
  unsigned mGrainSize;
  unsigned nSplit;
  // Permutation of the output rows of the sparse matrix
  std::vector<unsigned> rowPermutations;
  // Permutation of the input columns of the sparse matrix
  std::vector<unsigned> columnPermutations;
};

inline std::ostream &operator<<(std::ostream &os, const Partition &p) {
  os << "\n"
     << " nSplit=" << p.nSplit << "\n"
     << " Bands= " << p.bandInfo.getNumColumnBands() << "\n"
     << "  boundaries:" << poplibs_support::toString(p.bandInfo.boundaries)
     << "\n"
     << "  nzBlocks:" << poplibs_support::toString(p.bandInfo.nzBlocksPerBand)
     << "\n"
     << " Permutations\n"
     << "  row:" << poplibs_support::toString(p.rowPermutations) << "\n"
     << "  column:" << poplibs_support::toString(p.columnPermutations);
  return os;
}

// Build partition of N given number of splits. The partition is guaranteed to
// have non-zero length intervals
std::vector<poplar::Interval>
buildPartitionForN(std::size_t n, std::size_t numSplits, std::size_t grainSize);

// Give a parition of N, allocate tiles to each of the partition respecting the
// grain size. The number of elements in the tile partition is always
// guraranteed to be equal to the number of partitions of N. Some of the
// sizes of the partition entries may be zero if there are insufficient
// tiles to allocate.
std::vector<poplar::Interval>
buildTilePartitionForN(const std::vector<poplar::Interval> &partitionOfN,
                       std::size_t numTiles, std::size_t tileGrain);

std::vector<unsigned>
allocateTilesForBands(const std::vector<unsigned> &nzBlocksPerColumnBand,
                      unsigned numTiles, unsigned tileGrain);

unsigned getTileGrainSize(const poplar::Target &target);

unsigned getMGrainSize(const poplar::Target &target, const poplar::Type &type,
                       unsigned blockLength);

unsigned getKGrainSize(const poplar::Target &target, const poplar::Type &type,
                       unsigned blockLength);

// Designs a partition given various constraints imposed by the codelets used.
// \param csr      Sparse lhs matrix in a sparse-dense matmul
// \param params   Matrix multiplication parameters
// \param target   The target to build the partition on
// \param dataType The data type of the input and output
// \param options  Options
// \param cache    Pointer to planning cache
template <typename T>
Partition getPartition(const CSRMatrix<T> &csr, const MatMulParams &params,
                       const poplar::Target &target,
                       const poplar::Type &dataType,
                       const MatMulOptions &options, PlanningCache *cache);

} // namespace static_

} // namespace popsparse
#endif // _poplibs_popsparse_StaticSparseMatMulPartitioner_hpp_
