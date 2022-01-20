// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef popops_RearrangeInternal_hpp
#define popops_RearrangeInternal_hpp

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/logging.hpp>
#include <poputil/exceptions.hpp>
#include <vector>

using namespace poplibs_support;

namespace popops {
namespace internal {

bool canSplitTranspose(unsigned numMatrices, unsigned numWorkers) {
  return numWorkers % numMatrices == 0 && numMatrices < numWorkers;
}

std::vector<unsigned>
createSplitTranspose1DWorkList(unsigned rows, unsigned cols, unsigned matrices,
                               unsigned numWorkers, unsigned blockSize) {
  unsigned workersPerMatrix = numWorkers / matrices;
  unsigned rowsDbs = rows / blockSize;
  unsigned colsDbs = cols / blockSize;

  // Only allow a number of matrices that are a sub-multiple of the number of
  // workers.
  if (numWorkers % matrices != 0) {
    poputil::poplibs_error("number of workers must be an integer multiple of "
                           "the number of matrices");
  }
  if (matrices >= numWorkers) {
    poputil::poplibs_error("number of matrices must be less than the number of "
                           "workers");
  }

  // split rows first as the vertices are faster if split along that dimension.
  // split along column dimension once rows are split. We could use costs based
  // on actual transpose estimates to do the splitting.
  auto rowsPerWorker = ceildiv(rowsDbs, workersPerMatrix);
  auto workersUsed = ceildiv(rowsDbs, rowsPerWorker);

  auto workersPerCols = workersPerMatrix / workersUsed;
  auto colsPerWorker = ceildiv(colsDbs, workersPerCols);
  std::vector<unsigned> workList;
  workList.reserve(4 * numWorkers);

  // build worklist
  for (unsigned t = 0; t != matrices; ++t) {
    for (unsigned r = 0; r < rowsDbs; r += rowsPerWorker) {
      for (unsigned c = 0; c < colsDbs; c += colsPerWorker) {
        unsigned allocRows = std::min(rowsDbs - r, rowsPerWorker);
        unsigned allocCols = std::min(colsDbs - c, colsPerWorker);
        unsigned inIndex = t * rowsDbs * blockSize * colsDbs * blockSize +
                           r * blockSize * colsDbs * blockSize + c * blockSize;
        unsigned outIndex = t * rowsDbs * blockSize * colsDbs * blockSize +
                            c * blockSize * rowsDbs * blockSize + r * blockSize;
        workList.push_back(inIndex / blockSize);
        workList.push_back(outIndex / blockSize);
        workList.push_back(allocRows);
        workList.push_back(allocCols);
      }
    }
  }
  logging::popops::trace("createSplitTranspose1DWorkList: matrices {}, rows {} "
                         "cols {}, worklist {}",
                         matrices, rows, cols, workList);
  return workList;
}

} // namespace internal
} // namespace popops
#endif
