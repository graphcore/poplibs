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

std::vector<unsigned> createSplitTranspose1DWorkList(unsigned rows,
                                                     unsigned cols,
                                                     unsigned matrices,
                                                     unsigned numWorkers) {
  unsigned workersPerMatrix = numWorkers / matrices;
  unsigned rowsD4 = rows / 4;
  unsigned colsD4 = cols / 4;

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
  auto rowsPerWorker = ceildiv(rowsD4, workersPerMatrix);
  auto workersUsed = ceildiv(rowsD4, rowsPerWorker);

  auto workersPerCols = workersPerMatrix / workersUsed;
  auto colsPerWorker = ceildiv(colsD4, workersPerCols);
  std::vector<unsigned> workList;
  workList.reserve(4 * numWorkers);

  // build worklist
  for (unsigned t = 0; t != matrices; ++t) {
    for (unsigned r = 0; r < rowsD4; r += rowsPerWorker) {
      for (unsigned c = 0; c < colsD4; c += colsPerWorker) {
        unsigned allocRows = std::min(rowsD4 - r, rowsPerWorker);
        unsigned allocCols = std::min(colsD4 - c, colsPerWorker);
        unsigned inIndex =
            t * rowsD4 * 4 * colsD4 * 4 + r * 4 * colsD4 * 4 + c * 4;
        unsigned outIndex =
            t * rowsD4 * 4 * colsD4 * 4 + c * 4 * rowsD4 * 4 + r * 4;
        workList.push_back(inIndex / 4);
        workList.push_back(outIndex / 4);
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
