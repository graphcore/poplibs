// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "MatMulOptions.hpp"
#include "MatMulTensorMetaData.hpp"
#include "SparseMetaInfo.hpp"
#include "SparsePartitionerImpl.hpp"
#include "SparseStorageInternal.hpp"
#include "StaticMatMulPartitioner.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Pad.hpp"
#include "popops/Reduce.hpp"
#include "popsparse/MatMul.hpp"
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <gccs/Algorithm.hpp>
#include <numeric>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <popops/Rearrange.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace popsparse;
using namespace poputil;

namespace {

std::vector<unsigned>
inversePermutation(const std::vector<unsigned> &permutations) {
  const auto numRowBlocks = permutations.size();
  std::vector<unsigned> inversePerm(numRowBlocks);
  for (unsigned i = 0; i != numRowBlocks; ++i) {
    inversePerm[permutations[i]] = i;
  }
  return inversePerm;
}

// The codelets require padding because of over reads for block sparsity
// with blocks > 1 (we could make it a function of )
unsigned paddingForPartials(unsigned blockLength, unsigned typeSize) {
  // TODO: Check these
  unsigned bytes = 0;
  if (blockLength == 1) {
    bytes = 0;
  } else if (blockLength == 4) {
    bytes = 32;
  } else if (blockLength == 8) {
    bytes = 64;
  } else if (blockLength == 16) {
    bytes = 128;
  }
  return gccs::ceildiv(bytes, typeSize);
}

// This depends on the type and blocklength but we just use max here to
// simplify the codelet code.
unsigned paddingForDenseRhs(unsigned blockLength, unsigned typeSize) {
  return blockLength == 1 ? 0 : 16 / typeSize;
}

// Information kept for each row on a tile
struct RowInfo {
  unsigned relativeColIndex;
  unsigned relativeRowIndex;
  RowInfo(unsigned relativeColIndex, unsigned relativeRowIndex)
      : relativeColIndex(relativeColIndex), relativeRowIndex(relativeRowIndex) {
  }
};

// This is information kept per worker: this is a general way of keeping
// information for both element and block sparsity as the dimension we split
// for the two are different.
// For element: split both the n and rows.
// For block: split only n.
struct WorkDivision {
  // unique row indices assigned to worker
  std::vector<unsigned> rows;
  // interval of n on this tile assigned to this worker
  Interval nInterval;
  // group to which this worker belong to. Workers belonging to the same group
  // share the same rows. We could derive this by matching rows, but kept
  // simple here to avoid the check later
  unsigned group;
  WorkDivision() = default;
};

// A rudimentary work division amongst workers
std::vector<WorkDivision>
divideWork(std::unordered_map<unsigned, std::vector<RowInfo>> &rowInfo,
           const std::vector<unsigned> &uniqueRowIndices, unsigned numWorkers,
           unsigned n, unsigned nGrainSize, unsigned blockLength) {
  std::vector<WorkDivision> workDivision(numWorkers);
  if (blockLength == 1) {
    std::vector<std::pair<unsigned, unsigned>> rowAndSizePerRow;
    rowAndSizePerRow.reserve(rowInfo.size());
    unsigned totalWork = 0;
    for (auto &r : uniqueRowIndices) {
      auto numColsInRow = rowInfo[r].size();
      rowAndSizePerRow.emplace_back(r, numColsInRow);
      totalWork += numColsInRow;
    }

    unsigned nSplit = 1;
    if (rowAndSizePerRow.size() <= numWorkers / 2 && (n % 8 == 0)) {
      nSplit = 2;
    }
    bool singleRowPerWorker =
        nSplit == 1 && rowAndSizePerRow.size() <= numWorkers;
    const auto usedWorkers = numWorkers / nSplit;
    auto it = rowAndSizePerRow.begin();
    for (unsigned w = 0; w != usedWorkers; ++w) {
      unsigned metric = 0;
      while (metric * usedWorkers < totalWork && it != rowAndSizePerRow.end()) {
        workDivision[w].rows.push_back(it->first);
        metric += it->second;
        ++it;
        if (singleRowPerWorker) {
          break;
        }
      }
    }

    // now nSplit
    assert(n % nGrainSize == 0);
    const auto nNumGrains = n / nGrainSize;
    const unsigned nGrainsPerWorker = gccs::ceildiv(nNumGrains, nSplit);
    for (unsigned w = 0; w != numWorkers; ++w) {
      unsigned workerPerSplitIndex = w * nSplit / numWorkers;
      unsigned nGrainBegin =
          std::min(workerPerSplitIndex * nGrainsPerWorker, nNumGrains);
      unsigned nGrainEnd =
          std::min((workerPerSplitIndex + 1) * nGrainsPerWorker, nNumGrains);
      workDivision[w].nInterval =
          Interval(nGrainBegin * nGrainSize, nGrainEnd * nGrainSize);
      // All nSplits get the same rows and we have already found the the row
      // split for split index = 0
      if (workerPerSplitIndex != 0) {
        workDivision[w].rows = workDivision[w % (numWorkers / nSplit)].rows;
      }
      workDivision[w].group = w % (numWorkers / nSplit);
    }
  } else {
    // for block length != 1, we split n as have a grain in k
    const unsigned nPerWorker = gccs::ceildiv(n, numWorkers);
    for (unsigned w = 0; w != numWorkers; ++w) {
      unsigned nBegin = std::min(w * nPerWorker, n);
      unsigned nEnd = std::min((w + 1) * nPerWorker, n);
      workDivision[w].nInterval = Interval(nBegin, nEnd);
      workDivision[w].rows = uniqueRowIndices;
    }
  }
  return workDivision;
}

// Create a work list from work division that the codelet uses
std::vector<unsigned>
buildWorkList(const Target &target,
              std::unordered_map<unsigned, std::vector<RowInfo>> &rowsInfo,
              const std::vector<WorkDivision> &workDivision, unsigned n,
              const Type &dataType, const Type &partialsType,
              unsigned blockLength) {
  std::vector<unsigned> workList;
  const auto numWorkers = workDivision.size();
  // reserve space for per worker header
  const unsigned headerEntriesPerWorker = blockLength == 1 ? 6 : 2;
  workList.resize(headerEntriesPerWorker * numWorkers);
  const unsigned dataTypeSize = target.getTypeSize(dataType);
  const unsigned partialsTypeSize = target.getTypeSize(partialsType);

  // work list are completely different for block length of 1 and the others as
  // the codelets targeted are completely different.
  if (blockLength == 1) {
    std::vector<unsigned> offsetIntoWorklist(numWorkers);
    std::vector<unsigned> offsetInNZ(numWorkers);
    std::vector<unsigned> relativeRowNumber(numWorkers);
    unsigned nzOff = 0;
    unsigned rowOff = 0;
    // generate worklist only once per group
    std::unordered_set<unsigned> generatedGroups;
    for (unsigned w = 0; w != numWorkers; ++w) {
      const auto group = workDivision[w].group;
      auto found = generatedGroups.find(group) != generatedGroups.end();
      if (found)
        continue;
      generatedGroups.insert(group);
      assert(group < numWorkers);
      offsetIntoWorklist[group] =
          std::distance(workList.begin(), workList.end());
      offsetInNZ[group] = nzOff;
      relativeRowNumber[group] = rowOff;

      for (unsigned r = 0; r != workDivision[w].rows.size(); ++r) {
        unsigned row = workDivision[w].rows[r];
        const auto &rowInfo = rowsInfo[row];
        workList.push_back(rowInfo.size() - (dataType == HALF));
        for (const auto &c : rowInfo) {
          unsigned entry = c.relativeColIndex * n * dataTypeSize;
          workList.push_back(entry);
        }
        nzOff += rowInfo.size();
      }
      rowOff += workDivision[w].rows.size();
    }
    // fill in work list header
    for (unsigned w = 0; w != numWorkers; ++w) {
      const auto group = workDivision[w].group;
      const auto intervalSize = workDivision[w].nInterval.size();
      workList[headerEntriesPerWorker * w] = workDivision[w].rows.size();
      workList[headerEntriesPerWorker * w + 1] = offsetIntoWorklist[group];
      workList[headerEntriesPerWorker * w + 2] = offsetInNZ[group];
      workList[headerEntriesPerWorker * w + 3] = intervalSize;
      workList[headerEntriesPerWorker * w + 4] =
          intervalSize ? workDivision[w].nInterval.begin() : 0;
      workList[headerEntriesPerWorker * w + 5] = relativeRowNumber[group];
    }
  } else {
    // Every worker processes the same rows
    workList.push_back(workDivision[0].rows.size() - 1);
    for (unsigned r = 0; r != workDivision[0].rows.size(); ++r) {
      unsigned row = workDivision[0].rows[r];
      const auto &rowInfo = rowsInfo[row];
      workList.push_back(static_::block::convertToImplOffset(
          partialsTypeSize, rowInfo[0].relativeRowIndex * n));
      workList.push_back(rowInfo.size() - 1);
      for (const auto &c : rowInfo) {
        workList.push_back(static_::block::convertToImplOffset(
            dataTypeSize, c.relativeColIndex * n));
      }
    }
    // fill in work list header
    for (unsigned w = 0; w != numWorkers; ++w) {
      const auto intervalSize = workDivision[w].nInterval.size();
      workList[headerEntriesPerWorker * w] =
          intervalSize ? workDivision[w].nInterval.begin() : 0;
      workList[headerEntriesPerWorker * w + 1] = intervalSize;
    }
  }
  return workList;
}

// Allocation per tile
struct TileAllocation {
  // Interval of non-zero values allocated to each tile
  Interval nzInterval;
  // the mapping of the index of the split of the columns of the dense rhs
  // matrix. i.e. if nSplit is 3, this has values {0, 1, 2}
  unsigned nIndex;
  // band to which the nz values belong to
  unsigned band;
  TileAllocation(const Interval &nzInterval, unsigned nIndex, unsigned band)
      : nzInterval(nzInterval), nIndex(nIndex), band(band) {}
  TileAllocation() = default;
};

struct Plan {

  // number of splits of the columns of the dense rhs matrix
  unsigned nSplit = 1;

  // grain size of the columns of the dense rhs matrix
  unsigned nGrainSize = 4;

  // grain size of the rows of the dense rhs matrix
  unsigned kGrainSize = 4;

  // grain size of the rows of the sparse lhs matrix
  unsigned mGrainSize = 4;

  // Columns of the sparse matrix are split into bands. This gives the
  // boundaries of the bands
  std::vector<unsigned> columnBandBoundaries;

  // Allocation per tile: gives the column indices of the NZ values allocated
  // to each tile. when nSplit != 1, NZ values are shared between tiles but
  // worklist cannot be shared as long as the N split interval may not be the
  // same.
  std::vector<TileAllocation> tileAllocation;

  // The interval of the columns of the dense rhs matrix for each split index.
  // eg: if n is the number of dense columns and if nSplit = 3, this holds
  // {[0, n1), [n1, n2), [n2, n)}
  std::vector<Interval> partitionOfN;

  // Row and column permutations applied in pre-processing stage
  std::vector<unsigned> rowPermutations;
  std::vector<unsigned> columnPermutations;

  unsigned getNumColumnBands() const { return columnBandBoundaries.size() - 1; }
};

// Copy partition into Plan
Plan createPlanFromPartition(static_::Partition &&partition) {
  Plan p;
  p.nSplit = partition.nSplit;
  p.nGrainSize = partition.nGrainSize;
  p.kGrainSize = partition.kGrainSize;
  p.mGrainSize = partition.mGrainSize;
  p.columnBandBoundaries = std::move(partition.bandInfo.boundaries);
  p.rowPermutations = std::move(partition.rowPermutations);
  p.columnPermutations = std::move(partition.columnPermutations);
  return p;
}

template <typename T>
void checkParamsAndCSRConsistency(const static_::MatMulParams &params,
                                  const CSRMatrix<T> &csrLhs,
                                  const poplar::DebugContext &debugContext) {
  const auto blockLength = csrLhs.getBlockDimensions()[0];
  assert(csrLhs.numRows % blockLength == 0);
  assert(!params.isTransposed());
  if (csrLhs.numRows / blockLength + 1 != csrLhs.rowIndices.size()) {
    throw poplibs_error("Inconsistent CSR matrix: number of rows do not match "
                        "row indices size for " +
                        debugContext.getPathName() + " (rowIndices.size())[" +
                        std::to_string(csrLhs.rowIndices.size()) + "]!=[" +
                        std::to_string(csrLhs.numRows / blockLength + 1) +
                        "](csrLhs.numRows / blockLength + 1)");
  }
  if (params.getM() != csrLhs.numRows) {
    throw poplibs_error("Number of rows of sparse matrix in MatMulParams does "
                        "not match number of rows in CSR matrix for " +
                        debugContext.getPathName() + " (" +
                        std::to_string(params.getM()) +
                        "!=" + std::to_string(csrLhs.numRows) + ")");
  }
  if (params.getK() != csrLhs.numColumns) {
    throw poplibs_error("Number of columns of sparse matrix in MatMulParams "
                        "does not match number of columns in CSR matrix for " +
                        debugContext.getPathName() + " (" +
                        std::to_string(params.getK()) +
                        "!=" + std::to_string(csrLhs.numColumns) + ")");
  }
}

// Convert a CSR matrix to COO where column indices are renumbered according to
// the permutations in the plan and non-zero blocks are ordered in the result as
// follows:
//   1. All non-zero blocks for a column band/row
//   2. All rows for a column band
//   3. All column bands
//
// Additionally if `needNZValues` is set, non-zero blocks for a single column
// band/row pair are sorted by the permuted column ordering, and nzValues are
// copied to the result. For FLOAT type and block size 16, each non-zero 16x16
// block is split into a left 16x8 and right 16x8 because of the codelets used
// for that specific case.
template <typename T>
COOMatrix<T> convertToPartitionedCOO(const CSRMatrix<T> &fullMatrix,
                                     const Plan &plan, const Target &target,
                                     const Type &dataType, bool needNZValues) {
  const auto &bandBoundaries = plan.columnBandBoundaries;
  const auto blockLength = fullMatrix.getBlockDimensions()[0];
  CSRMatrix<T> csr;
  csr.rowIndices.reserve(fullMatrix.rowIndices.size());
  csr.columnIndices.resize(fullMatrix.columnIndices.size());
  if (needNZValues) {
    csr.nzValues.resize(fullMatrix.nzValues.size());
  }
  const unsigned blockArea = blockLength * blockLength;
  const auto inverseColumnPermutations =
      inversePermutation(plan.columnPermutations);

  // Reorder elements such that columns for each row are sorted
  unsigned numRowBlocks = fullMatrix.rowIndices.size() - 1;
  csr.rowIndices.push_back(0);

  for (unsigned rp = 0; rp != numRowBlocks; ++rp) {
    const auto r = plan.rowPermutations[rp];
    unsigned blocksInRow =
        (fullMatrix.rowIndices[r + 1] - fullMatrix.rowIndices[r]) / blockArea;
    csr.rowIndices.push_back(csr.rowIndices.back() + blocksInRow * blockArea);
    if (blocksInRow == 0) {
      continue;
    }
    std::vector<unsigned> sortedIndices;
    sortedIndices.resize(blocksInRow);
    std::iota(sortedIndices.begin(), sortedIndices.end(),
              fullMatrix.rowIndices[r] / blockArea);

    if (needNZValues) {
      std::stable_sort(
          sortedIndices.begin(), sortedIndices.end(),
          [&](unsigned a, unsigned b) {
            return inverseColumnPermutations[fullMatrix.columnIndices[a] /
                                             blockLength] <
                   inverseColumnPermutations[fullMatrix.columnIndices[b] /
                                             blockLength];
          });
    }

    unsigned baseIndex = csr.rowIndices[rp] / blockArea;
    for (unsigned b = 0; b != blocksInRow; ++b) {
      csr.columnIndices[b + baseIndex] =
          inverseColumnPermutations[fullMatrix.columnIndices[sortedIndices[b]] /
                                    blockLength] *
          blockLength;
      if (needNZValues) {
        std::copy(fullMatrix.nzValues.begin() + sortedIndices[b] * blockArea,
                  fullMatrix.nzValues.begin() +
                      (sortedIndices[b] + 1) * blockArea,
                  csr.nzValues.begin() + (b + baseIndex) * blockArea);
      }
    }
  }

  std::vector<unsigned> perRowConsumed(numRowBlocks);
  COOMatrix<T> coo(fullMatrix.getBlockDimensions());
  coo.rowIndices.reserve(fullMatrix.rowIndices.size());
  coo.columnIndices.reserve(fullMatrix.columnIndices.size());
  std::vector<T> nzValues;

  if (needNZValues) {
    coo.nzValues.reserve(fullMatrix.nzValues.size());
    nzValues.reserve(fullMatrix.nzValues.size());
  }

  for (unsigned b = 0; b != bandBoundaries.size() - 1; ++b) {
    unsigned colStart = bandBoundaries[b];
    unsigned colEnd = bandBoundaries[b + 1];
    if (colStart == colEnd) {
      continue;
    }
    for (unsigned r = 0; r != numRowBlocks; ++r) {
      auto colBeginIt = csr.columnIndices.begin() +
                        csr.rowIndices[r] / blockArea + perRowConsumed[r];
      auto colEndIt =
          csr.columnIndices.begin() + csr.rowIndices[r + 1] / blockArea;
      unsigned colToCheck = colEnd == 0 ? colEnd : colEnd - 1;
      auto it =
          std::upper_bound(colBeginIt, colEndIt, colToCheck * blockLength);
      unsigned numBlocks = std::distance(colBeginIt, it);

      // copy columns and row indices
      std::copy(colBeginIt, it, std::back_inserter(coo.columnIndices));
      if (needNZValues) {
        auto nzBeginIt = csr.nzValues.begin() + csr.rowIndices[r] +
                         perRowConsumed[r] * blockArea;
        std::copy(nzBeginIt, nzBeginIt + numBlocks * blockArea,
                  std::back_inserter(nzValues));
      }
      std::fill_n(std::back_inserter(coo.rowIndices), numBlocks,
                  r * blockLength);
      perRowConsumed[r] += numBlocks;
    }
  }

  if (needNZValues) {
    // Copy nzValues to the result.
    // For some block sizes elements within the block need to be reordered
    // but the relative ordering of the blocks doesn't change in the copy.
    for (unsigned t = 0; t != target.getNumTiles(); ++t) {
      auto nzInterval = plan.tileAllocation[t].nzInterval;
      auto nIndex = plan.tileAllocation[t].nIndex;
      if (nzInterval.size() == 0 || nIndex != 0) {
        continue;
      }
      for (auto i = nzInterval.begin(); i != nzInterval.end(); ++i) {
        if (dataType == FLOAT && blockLength == 16) {
          // Special handling for FLOAT 16x16 where the matrix is split
          // as two 16x8 matrices
          for (unsigned colSplit = 0; colSplit != 2; ++colSplit) {
            for (unsigned row = 0; row != blockLength; ++row) {
              unsigned beginIndex = i * blockArea + row * blockLength +
                                    blockLength / 2 * colSplit;
              std::copy(nzValues.begin() + beginIndex,
                        nzValues.begin() + beginIndex + blockLength / 2,
                        std::back_inserter(coo.nzValues));
            }
          }
        } else {
          std::copy(nzValues.begin() + i * blockArea,
                    nzValues.begin() + (i + 1) * blockArea,
                    std::back_inserter(coo.nzValues));
        }
      }
    }
  }
  return coo;
}

Tensor createAndMapNZTensor(Graph &graph, std::size_t numNZValues,
                            const Plan &plan, unsigned numTiles,
                            const Type &type, unsigned blockLength,
                            const poplar::DebugContext &debugContext) {
  auto nzFullTensor =
      graph.addVariable(type, {numNZValues}, {debugContext, "nz"});
  if (plan.nSplit > 1) {
    const unsigned nzGrainSize = std::lcm(plan.nGrainSize, blockLength);
    mapTensorLinearly(graph, nzFullTensor, 0, nzGrainSize);
  } else {
    const auto blockArea = blockLength * blockLength;
    for (unsigned t = 0; t != numTiles; ++t) {
      const auto nzInterval = plan.tileAllocation[t].nzInterval;
      if (nzInterval.size() == 0) {
        continue;
      }
      auto tileNZ = nzFullTensor.slice(nzInterval.begin() * blockArea,
                                       nzInterval.end() * blockArea);
      graph.setTileMapping(tileNZ, t);
    }
  }
  return nzFullTensor;
}

Tensor createSparseDenseMatMulRHSGivenPlan(
    Graph &graph, const Plan &plan, unsigned numGroups, unsigned k,
    unsigned numColumns, unsigned blockLength, const Type &inputType,
    const poplar::DebugContext &debugContext) {
  assert(k % plan.kGrainSize == 0);
  assert(k % blockLength == 0);
  auto rhs = graph.addVariable(
      inputType, {numGroups, k / blockLength, numColumns, blockLength},
      {debugContext, "rhs"});

  mapTensorLinearly(graph, rhs, 0, plan.nGrainSize * plan.kGrainSize);
  return rhs.dimShuffle({0, 1, 3, 2}).reshape({numGroups, k, numColumns});
}

// Construct graph
Tensor constructGraph(Graph &graph, const Tensor nzFullTensor,
                      const std::vector<std::size_t> &rowIndices,
                      const std::vector<std::size_t> &columnIndices,
                      const Tensor &rhs, Sequence &prog, unsigned groups,
                      unsigned m, unsigned k, unsigned n, unsigned blockLength,
                      const Plan &plan, const Type &partialsType,
                      const DebugNameAndId &dnai, bool verboseLogging) {
  const auto dataType = rhs.elementType();
  const auto &target = graph.getTarget();
  // No support for groups yet
  assert(n == rhs.dim(2));
  assert(groups == 1);
  assert(m % blockLength == 0);
  const auto numTiles = target.getNumTiles();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto blockArea = blockLength * blockLength;

  // Pad rhs as required by the grain decided by the plan
  auto nPadded = gccs::alignNext(n, plan.nGrainSize);
  auto out = graph.addVariable(
      dataType, {groups, m / blockLength, nPadded, blockLength}, {dnai, "out"});
  poputil::mapTensorLinearly(graph, out, 0, plan.nGrainSize * plan.mGrainSize);
  auto rhsPadded = rhs;
  if (nPadded != n) {
    rhsPadded = createSparseDenseMatMulRHSGivenPlan(graph, plan, groups, k,
                                                    nPadded, blockLength,
                                                    dataType, {dnai, "padded"});
    // The padded tensor is partially written
    prog.add(WriteUndef(rhsPadded));
    auto zero = graph.addConstant(dataType, {1}, 0);
    graph.setTileMapping(zero, 0);
    // Only zero out the edges, but ensure that the edge zeroed out is of a
    // nice size so that fast memsets could be used.
    auto sliceToZero =
        rhsPadded.slice(gccs::alignPrev(n, plan.nGrainSize), nPadded, 2);
    prog.add(Copy(zero.broadcast(sliceToZero.numElements(), 0),
                  sliceToZero.flatten()));
    // Copy the actual tensor into the padded one. Some entries that were zeroed
    // will be over-written.
    prog.add(Copy(rhs, rhsPadded.slice(0, n, 2)));
  }

  // we want [groups][k / blockLength][nPadded][blockLength]
  auto rhsReshaped =
      rhsPadded.reshapePartial(1, 2, {k / blockLength, blockLength})
          .dimShuffle({0, 1, 3, 2});

  // Apply column permutation
  rhsReshaped = concat(rhsReshaped.slices(plan.columnPermutations, 1), 1);

  auto cs = graph.addComputeSet({dnai});

  auto getLocalIndex = [](std::vector<unsigned> &uniqueIndices,
                          unsigned index) {
    auto it = std::find(uniqueIndices.begin(), uniqueIndices.end(), index);
    unsigned localIndex;
    if (it == uniqueIndices.end()) {
      uniqueIndices.push_back(index);
      localIndex = uniqueIndices.size() - 1;
    } else {
      localIndex = std::distance(uniqueIndices.begin(), it);
    }
    return localIndex;
  };

  std::vector<std::unordered_map<unsigned, std::vector<RowInfo>>> tileRowInfo(
      numTiles);
  std::vector<Tensor> partialOut(numTiles);
  std::vector<Copy> tileRhsCopies;
  tileRhsCopies.reserve(numTiles);
  std::vector<Tensor> tileRhsPaddedForWriteUndef;
  tileRhsPaddedForWriteUndef.reserve(numTiles);

  const unsigned dataTypeSize = target.getTypeSize(dataType);
  const unsigned partialsTypeSize = target.getTypeSize(partialsType);

  for (unsigned t = 0; t != numTiles; ++t) {
    auto nzInterval = plan.tileAllocation[t].nzInterval;
    if (nzInterval.size() == 0) {
      continue;
    }
    auto nIndex = plan.tileAllocation[t].nIndex;
    auto nInterval = plan.partitionOfN[nIndex];
    assert(nInterval.size() != 0);

    std::vector<unsigned> uniqueColIndices;
    std::vector<unsigned> uniqueRowIndices;
    for (auto i = nzInterval.begin(); i != nzInterval.end(); ++i) {
      auto colBlock = columnIndices[i] / blockLength;
      auto tileColIndex = getLocalIndex(uniqueColIndices, colBlock);
      auto rowBlock = rowIndices[i] / blockLength;
      auto tileRowIndex = getLocalIndex(uniqueRowIndices, rowBlock);
      tileRowInfo[t][rowBlock].emplace_back(tileColIndex, tileRowIndex);
    }
    auto tileNZ = nzFullTensor.slice(nzInterval.begin() * blockArea,
                                     nzInterval.end() * blockArea);
    auto tileRhs = concat(rhsReshaped[0].slices(uniqueColIndices, 0))
                       .slice(nInterval, 1)
                       .flatten();
    auto tileRhsPadded = tileRhs;
    if (blockLength != 1) {
      // The codelets for blockLength > 1 over-read rhs and also feed the data
      // to the AMP. This has the potential to cause exceptions. This guarantees
      // that we can zero that data.
      tileRhsPadded =
          graph.addVariable(dataType,
                            {tileRhs.numElements() +
                             paddingForDenseRhs(blockLength, dataTypeSize)},
                            {dnai, "paddedRhs"});
      tileRhsCopies.push_back(
          Copy(tileRhs, tileRhsPadded.slice(0, tileRhs.numElements()), false,
               {dnai}));
      tileRhsPaddedForWriteUndef.push_back(tileRhsPadded);
      graph.setTileMapping(tileRhsPadded, t);
    }
    const auto workDivision =
        divideWork(tileRowInfo[t], uniqueRowIndices, numWorkers,
                   nInterval.size(), plan.nGrainSize, blockLength);

    if (verboseLogging) {
      logging::popsparse::trace("work div tile {} : ", t);
      for (unsigned w = 0; w != workDivision.size(); ++w) {
        logging::popsparse::trace("    worker {}: group {}, rows {}, n "
                                  "interval {}",
                                  w, workDivision[w].group,
                                  workDivision[w].rows,
                                  workDivision[w].nInterval);
      }
    }

    if (verboseLogging) {
      logging::popsparse::debug(
          "Unique rows/cols/nz on tile {} : [{}, {}, {}] ", t,
          uniqueRowIndices.size(), uniqueColIndices.size(), nzInterval.size());
      const auto [minItC, maxItC] =
          std::minmax_element(uniqueColIndices.begin(), uniqueColIndices.end());
      unsigned minC = minItC == uniqueColIndices.end() ? ~0 : *minItC;
      unsigned maxC = maxItC == uniqueColIndices.end() ? ~0 : *maxItC;
      logging::popsparse::trace("uniqueColBlockIndices {} : min/max {}/{} size "
                                "{} : {}",
                                t, minC, maxC, uniqueColIndices.size(),
                                uniqueColIndices);

      const auto [minItR, maxItR] =
          std::minmax_element(uniqueRowIndices.begin(), uniqueRowIndices.end());
      unsigned minR = minItR == uniqueRowIndices.end() ? ~0 : *minItR;
      unsigned maxR = maxItR == uniqueRowIndices.end() ? ~0 : *maxItR;
      logging::popsparse::trace("uniqueRowBlockIndices {} : min/max {}/{} size "
                                "{} : {}",
                                t, minR, maxR, uniqueRowIndices.size(),
                                uniqueRowIndices);
    }

    auto workList =
        buildWorkList(target, tileRowInfo[t], workDivision, nInterval.size(),
                      dataType, partialsType, blockLength);

    if (verboseLogging) {
      logging::popsparse::trace("worklist {} : {}", t, workList);
    }

    auto workListT = graph.addConstant(UNSIGNED_SHORT, {workList.size()},
                                       workList.data(), {dnai, "workList"});
    graph.setTileMapping(workListT, t);
    const unsigned paddingElems =
        paddingForPartials(blockLength, partialsTypeSize);
    const std::vector<std::size_t> partialsShape = {
        uniqueRowIndices.size(), nInterval.size(), blockLength};
    const auto partialsTensorElems =
        std::accumulate(partialsShape.begin(), partialsShape.end(), 1U,
                        std::multiplies<std::size_t>());
    auto partialOutTile =
        graph.addVariable(partialsType, {partialsTensorElems + paddingElems},
                          {dnai, "partialTile"});
    graph.setTileMapping(partialOutTile, t);
    partialOut[t] =
        partialOutTile.slice(0, partialsTensorElems).reshape(partialsShape);
    if (blockLength == 1) {
      auto v = graph.addVertex(
          cs,
          templateVertex("popsparse::StaticSparseDenseElementWise", dataType,
                         partialsType),
          {{"in", tileRhsPadded},
           {"out", partialOutTile.flatten()},
           {"nz", tileNZ},
           {"workList", workListT}});
      graph.setInitialValue(v["numZ"], nInterval.size());
      graph.setTileMapping(v, t);
    } else {
      auto v = graph.addVertex(
          cs,
          templateVertex("popsparse::StaticSparseDenseMatMulBlock", dataType,
                         partialsType, blockLength, blockLength),
          {{"s", tileRhsPadded},
           {"q", partialOutTile.flatten()},
           {"r", tileNZ},
           {"metaInfo", workListT}});
      graph.setInitialValue(
          v["zeroInfo"], static_::block::convertToImplOffset(
                             partialsTypeSize, partialOutTile.numElements()));
      graph.setInitialValue(v["sSize"],
                            static_::block::convertToImplOffset(
                                dataTypeSize, tileRhs.numElements()));
      graph.setTileMapping(v, t);
    }
  }

  // The padded rhs is partially written and needs to be WriteUndef'ed
  if (!tileRhsPaddedForWriteUndef.empty()) {
    prog.add(WriteUndef(concat(tileRhsPaddedForWriteUndef)));
  }

  for (const auto &copy : tileRhsCopies) {
    prog.add(copy);
  }

  prog.add(Execute(cs, {dnai, "compute"}));
  // map of [rowBlock, n split index] -> tile and local rowBlock index
  std::unordered_map<std::pair<unsigned, unsigned>,
                     std::vector<std::pair<unsigned, unsigned>>,
                     boost::hash<std::pair<unsigned, unsigned>>>
      rowToTileMapping;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    auto nIndex = plan.tileAllocation[tile].nIndex;
    for (const auto &rowInfo : tileRowInfo[tile]) {
      rowToTileMapping[std::make_pair(rowInfo.first, nIndex)].emplace_back(
          tile, rowInfo.second[0].relativeRowIndex);
    }
  }

  // connect the reductions
  std::vector<popops::SingleReduceOp> singleReduceOps;
  singleReduceOps.reserve(plan.partitionOfN.size() * m);
  std::vector<Tensor> outSlices;
  Tensor flattenedZeroSlices = graph.addVariable(dataType, {0});
  graph.setTileMapping(flattenedZeroSlices, 0);

  for (auto nIndex = 0U; nIndex != plan.partitionOfN.size(); ++nIndex) {
    auto nInterval = plan.partitionOfN[nIndex];
    for (auto rowBlock = 0U; rowBlock != m / blockLength; ++rowBlock) {
      auto rowIndexPairIt =
          rowToTileMapping.find(std::make_pair(rowBlock, nIndex));

      if (rowIndexPairIt == rowToTileMapping.end()) {
        flattenedZeroSlices = concat(
            flattenedZeroSlices, out[0][rowBlock].slice(nInterval).flatten());
      } else {
        std::vector<Tensor> rowPartials;
        rowPartials.reserve(rowIndexPairIt->second.size());

        for (const auto [tile, partialIndex] : rowIndexPairIt->second) {
          rowPartials.push_back(partialOut[tile][partialIndex].expand({0}));
        }
        popops::SingleReduceOp reduceOp(concat(rowPartials), {0},
                                        {popops::Operation::ADD}, dataType);
        singleReduceOps.push_back(std::move(reduceOp));
        outSlices.push_back(out[0][rowBlock].slice(nInterval, 0));
        if (verboseLogging) {
          logging::popsparse::debug("reduction sources row {} : {}", rowBlock,
                                    rowPartials.size());
        }
      }
    }
  }
  const auto numZeroRows =
      flattenedZeroSlices.valid() ? flattenedZeroSlices.numElements() / n : 0;

  if (!outSlices.empty()) {
    popops::reduceMany(graph, singleReduceOps, outSlices, prog,
                       {dnai, "scatter-add"});
  }

  if (numZeroRows) {
    auto zero = graph.addConstant(dataType, {1}, 0);
    graph.setTileMapping(zero, 0);
    Copy(zero.broadcast(flattenedZeroSlices.numElements(), 0),
         flattenedZeroSlices);
  }
  logging::popsparse::debug("Statistics of sparse operand: name {} : non-zero "
                            "rows {}, zero rows {}, number of nz elems {}, "
                            "num bands {}",
                            dnai.getPathName(), m - numZeroRows, numZeroRows,
                            nzFullTensor.numElements(),
                            plan.getNumColumnBands());

  // We must apply the inverse permutation applied to the rows
  const auto invRowPermutation = inversePermutation(plan.rowPermutations);
  out = concat(out.slices(invRowPermutation, 1), 1);
  return out.dimShuffle({0, 1, 3, 2}).flatten(1, 3).slice(0, n, 2);
}

void validateOperand(const std::array<std::size_t, 3> &dimSizes,
                     const std::array<std::size_t, 3> &grainSize,
                     const std::array<std::string, 3> &dimStr,
                     unsigned numGroups, unsigned blockLength,
                     const std::string &debugString) {
  const std::vector<unsigned> supportedBlockLengths = {1, 4, 8, 16};
  if (std::find(supportedBlockLengths.begin(), supportedBlockLengths.end(),
                blockLength) == supportedBlockLengths.end()) {
    throw poplibs_error("Block length of " + std::to_string(blockLength) +
                        " not supported for " + debugString);
  }
  if (numGroups != 1) {
    throw poplibs_error("Only number of groups = 1 supported, but is " +
                        std::to_string(numGroups) + " for " + debugString);
  }
  for (auto d = 0; d != 3; ++d) {
    if (dimSizes[d] % grainSize[d]) {
      throw poplibs_error("Number of " + dimStr[d] + " " +
                          std::to_string(dimSizes[d]) +
                          " must be multiple of " +
                          std::to_string(grainSize[d]) + " for " + debugString);
    }
  }
}

// Create a plan given the input CSR matrix, the size of the matrices and
// optional plan constraints.
template <typename T>
Plan createPlan(const CSRMatrix<T> &csr, const static_::MatMulParams &params,
                const Target &target, const Type &dataType,
                const static_::MatMulOptions &options,
                static_::PlanningCache *cache) {
  const auto numTiles = target.getNumTiles();
  const auto tileGrain = static_::getTileGrainSize(target);
  auto partition =
      static_::getPartition(csr, params, target, dataType, options, cache);

  auto plan = createPlanFromPartition(std::move(partition));
  unsigned numColumnBands = plan.getNumColumnBands();
  const auto &nzBlocksPerColumnBand = partition.bandInfo.nzBlocksPerBand;
  const auto nPadded = gccs::alignNext(params.getN(), plan.nGrainSize);

  std::vector<TileAllocation> tileAllocation(numTiles);
  plan.partitionOfN =
      static_::buildPartitionForN(nPadded, plan.nSplit, plan.nGrainSize);
  const auto tilePartitionOfN =
      static_::buildTilePartitionForN(plan.partitionOfN, numTiles, tileGrain);

  for (unsigned splitN = 0; splitN != plan.partitionOfN.size(); ++splitN) {
    // start off allocating linearly to tiles starting from tileSplitBegin
    unsigned startTile = tilePartitionOfN[splitN].begin();
    unsigned band = 0, numRemaining = 0, nzBlocksPerTile = 0, nzStart = 0;
    unsigned curTile;

    const auto tilesPerBand = static_::allocateTilesForBands(
        nzBlocksPerColumnBand, tilePartitionOfN[splitN].size(), tileGrain);

    for (unsigned tile = 0; tile != tilePartitionOfN[splitN].size(); ++tile) {
      if (numRemaining == 0) {
        if (band == numColumnBands) {
          break;
        }
        curTile = startTile;
        nzBlocksPerTile =
            tilesPerBand[band]
                ? gccs::ceildiv(nzBlocksPerColumnBand[band], tilesPerBand[band])
                : 0;
        numRemaining = nzBlocksPerColumnBand[band];
        logging::popsparse::trace(
            "Allocation: Band {}:, NZ blocks {}, tiles req"
            " {}, max/tile {}, start tile {}",
            band, nzBlocksPerColumnBand[band], tilesPerBand[band],
            nzBlocksPerTile, startTile);
      }

      unsigned nzBlocksThisTile = std::min(nzBlocksPerTile, numRemaining);
      if (nzBlocksThisTile) {
        tileAllocation[curTile] = TileAllocation(
            Interval(nzStart, nzStart + nzBlocksThisTile), splitN, band);
        nzStart += nzBlocksThisTile;
        numRemaining -= nzBlocksThisTile;
        ++curTile;
      }
      if (numRemaining == 0) {
        // move start tile to next band. We need to do this to guarantee that
        // tile grain size is respected.
        startTile += tilesPerBand[band];
        ++band;
      }
    }
  }

  plan.tileAllocation = std::move(tileAllocation);
  return plan;
}

template <typename T>
Tensor createSparseDenseMatMulRHSInternal(
    poplar::Graph &graph, const poplar::Type &inputType,
    const static_::MatMulParams &params, const CSRMatrix<T> &csrLHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags optionFlags, static_::PlanningCache *cache) {
  checkParamsAndCSRConsistency(params, csrLHS, debugContext);
  auto options = static_::parseMatMulOptionFlags(optionFlags);
  const auto &target = graph.getTarget();
  const unsigned blockLength = csrLHS.getBlockDimensions()[0];
  validateOperand(
      {params.getM(), params.getK(), params.getN()},
      {std::gcd(blockLength,
                static_::getMGrainSize(target, inputType, blockLength)),
       std::gcd(blockLength,
                static_::getKGrainSize(target, inputType, blockLength)),
       1},
      {"rows of sparse lhs", "columns of sparse lhs", "columns of dense rhs"},
      params.getNumGroups(), blockLength, debugContext.getPathName());
  options.partialsType = inputType;
  const auto plan =
      createPlan(csrLHS, params, graph.getTarget(), inputType, options, cache);
  return createSparseDenseMatMulRHSGivenPlan(
      graph, plan, params.getNumGroups(), params.getK(), params.getN(),
      blockLength, inputType, {debugContext});
}

template <typename T>
static_::SparseTensor createSparseDenseMatMulLHSInternal(
    poplar::Graph &graph, const poplar::Type &inputType,
    const static_::MatMulParams &params, const CSRMatrix<T> &csrLhs,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &optionFlags, static_::PlanningCache *cache) {

  checkParamsAndCSRConsistency(params, csrLhs, debugContext);
  auto options = static_::parseMatMulOptionFlags(optionFlags);
  options.partialsType = inputType;
  const auto plan =
      createPlan(csrLhs, params, graph.getTarget(), inputType, options, cache);
  auto coo =
      convertToPartitionedCOO(csrLhs, plan, graph.getTarget(), inputType, true);
  auto nzFullTensor = createAndMapNZTensor(
      graph, coo.nzValues.size(), plan, graph.getTarget().getNumTiles(),
      inputType, csrLhs.getBlockDimensions()[0], debugContext);
  auto csr = csrLhs;
  canonicalizeCSR(csr);
  static_::PlanningCacheImpl::Key key(
      params, options, csrLhs.getBlockDimensions()[0],
      std::move(csr.rowIndices), std::move(csr.columnIndices));

  std::unique_ptr<TensorMetaDataBase> opMetaData =
      std::make_unique<static_::MatMulTensorMetaData>(key);

  return static_::SparseTensor(nzFullTensor, std::move(opMetaData));
}

Tensor sparseDenseMatMulInternal(
    poplar::Graph &graph, const Tensor &lhs, const poplar::Tensor &rhs_,
    const static_::MatMulParams &params, unsigned blockLength,
    const std::vector<std::size_t> &rowIndices,
    const std::vector<std::size_t> &columnIndices,
    poplar::program::Sequence &prog, bool transposeLHS, bool transposeRHS,
    const poplar::DebugNameAndId &dnai, const static_::MatMulOptions &options,
    static_::PlanningCache *cache) {
  const auto dataType = rhs_.elementType();
  const auto rhs = transposeRHS ? rhs_.dimRoll(1, 2) : rhs_;
  auto numRhsRows = rhs.dim(1);
  auto numRhsColumns = rhs.dim(2);
  if (lhs.elementType() != dataType) {
    throw poplibs_error("Data types of lhs and rhs do not match in matmul " +
                        dnai.getPathName());
  }
  if (params.getNumGroups() != rhs.dim(0)) {
    throw poplibs_error("Number of groups used to create sparse tensor does "
                        "not match groups in dense RHS in " +
                        dnai.getPathName());
  }
  if (params.getK() != numRhsRows) {
    throw poplibs_error("Number of columns of left matrix must match number of "
                        "rows of right matrix in " +
                        dnai.getPathName());
  }
  if (params.getN() != numRhsColumns) {
    throw poplibs_error("Dimension of dense RHS operand do not match those used"
                        " for creating LHS SparseTensor in " +
                        dnai.getPathName());
  }
  const auto &target = graph.getTarget();
  validateOperand(
      {params.getM(), params.getK(), numRhsColumns},
      {std::gcd(blockLength,
                static_::getMGrainSize(target, dataType, blockLength)),
       std::gcd(blockLength,
                static_::getKGrainSize(target, dataType, blockLength)),
       1},
      {"rows of sparse lhs", "columns of sparse lhs", "columns of sparse rhs"},
      params.getNumGroups(), blockLength, dnai.getPathName());

  std::vector<float> nzValues;
  nzValues.resize(rowIndices.back());
  CSRMatrix<float> csr(params.getM(), params.getK(), nzValues, columnIndices,
                       rowIndices, {blockLength, blockLength});

  const auto plan =
      createPlan(csr, params, graph.getTarget(), dataType, options, cache);
  const auto referenceRhs = createSparseDenseMatMulRHSGivenPlan(
      graph, plan, params.getNumGroups(), params.getK(), numRhsColumns,
      blockLength, dataType, {dnai});
  auto rhsMaybeRegrouped = popops::rearrange::regroupIfBeneficial(
      graph, rhs, referenceRhs, prog, {dnai});

  auto coo =
      convertToPartitionedCOO(csr, plan, graph.getTarget(), dataType, false);

  return constructGraph(
      graph, lhs, coo.rowIndices, coo.columnIndices, rhsMaybeRegrouped, prog,
      params.getNumGroups(), params.getM(), params.getK(), numRhsColumns,
      blockLength, plan, options.partialsType, {dnai}, options.verboseLogging);
}

template <typename T>
void applyInversePlanOperations(COOMatrix<T> &coo, const Plan &plan,
                                const Target &target,
                                const poplar::Type &dataType,
                                const std::string &name) {
  // Only sequential mapping of NZ intervals is supported. Check if that is
  // indeed the case.
  unsigned beginIndex = 0;
  for (unsigned t = 0; t != target.getNumTiles(); ++t) {
    auto nzInterval = plan.tileAllocation[t].nzInterval;
    auto nIndex = plan.tileAllocation[t].nIndex;
    if (nzInterval.size() == 0 || nIndex != 0) {
      continue;
    }
    if (beginIndex != nzInterval.begin()) {
      throw poplibs_error("Found inconsistent sparse representation while "
                          "converting device -> host in " +
                          name);
    }
    beginIndex += nzInterval.size();
  }
  // if permutation is identity for both rows and columns there's nothing to be
  // done
  const auto inverseRowPermutations = inversePermutation(plan.rowPermutations);
  const auto inverseColumnPermutations =
      inversePermutation(plan.columnPermutations);
  if (inverseRowPermutations == plan.rowPermutations &&
      inverseColumnPermutations == plan.columnPermutations) {
    return;
  }
  const auto rowBlockLength = coo.getBlockDimensions()[0];
  const auto columnBlockLength = coo.getBlockDimensions()[1];
  assert(coo.rowIndices.size() == coo.columnIndices.size());
  assert(coo.nzValues.size() ==
         coo.rowIndices.size() * rowBlockLength * columnBlockLength);

  for (unsigned b = 0; b != coo.rowIndices.size(); ++b) {
    coo.rowIndices[b] =
        plan.rowPermutations[coo.rowIndices[b] / rowBlockLength] *
        rowBlockLength;
    coo.columnIndices[b] =
        plan.columnPermutations[coo.columnIndices[b] / columnBlockLength] *
        columnBlockLength;
  }
}

template <typename T>
void applyInverseNZOperations(COOMatrix<T> &coo, const Type &dataType) {
  // special handling for FLOAT and block length of 16
  const auto rowBlockLength = coo.getBlockDimensions()[0];
  const auto columnBlockLength = coo.getBlockDimensions()[1];
  unsigned blockLengthForSpecialHandling = 16;
  const auto specialCase = dataType == FLOAT &&
                           rowBlockLength == blockLengthForSpecialHandling &&
                           columnBlockLength == blockLengthForSpecialHandling;
  if (specialCase) {
    const auto blockArea = rowBlockLength * columnBlockLength;
    std::vector<T> tempNZValues(blockArea);
    for (unsigned b = 0; b != coo.rowIndices.size(); ++b) {
      std::copy(coo.nzValues.begin() + blockArea * b,
                coo.nzValues.begin() + blockArea * (b + 1),
                tempNZValues.begin());
      for (unsigned i = 0; i != rowBlockLength; ++i) {
        std::copy(tempNZValues.begin() + i * columnBlockLength / 2,
                  tempNZValues.begin() + (i + 1) * columnBlockLength / 2,
                  coo.nzValues.begin() + blockArea * b + i * columnBlockLength);
        std::copy(tempNZValues.begin() +
                      (blockArea + i * columnBlockLength) / 2,
                  tempNZValues.begin() +
                      (blockArea + (i + 1) * columnBlockLength) / 2,
                  coo.nzValues.begin() + blockArea * b +
                      (2 * i + 1) * columnBlockLength / 2);
      }
    }
  }
}

static_::MatMulTensorMetaData
getMatMulMetaData(const static_::SparseTensor &st,
                  const std::string &debugString) {
  auto m = dynamic_cast<const static_::MatMulTensorMetaData *>(
      st.getOpMetaData().getData());
  if (m == nullptr) {
    throw poplibs_error("Sparse tensor doesn't contain valid metadata in " +
                        debugString);
  }
  return *m;
}

} // unnamed namespace

namespace popsparse {
namespace static_ {
template <typename T>
Tensor createSparseDenseMatMulRHSImpl(poplar::Graph &graph,
                                      const poplar::Type &inputType,
                                      const MatMulParams &params,
                                      const CSRMatrix<T> &csrLHS,
                                      const poplar::DebugContext &debugContext,
                                      const poplar::OptionFlags optionFlags,
                                      PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, inputType, optionFlags));
  logging::popsparse::debug("Creating dense RHS for Matmul {} : sparse[{}, "
                            "{}] * dense [{}, {}]",
                            debugContext.getPathName(), params.getM(),
                            params.getK(), params.getK(), params.getN());
  if (params.isTransposed()) {
    throw poplibs_error("MatMul params are not created for a sparse * dense "
                        "multiplication in " +
                        debugContext.getPathName());
  }
  return createSparseDenseMatMulRHSInternal(graph, inputType, params, csrLHS,
                                            debugContext, optionFlags, cache);
}

template <typename T>
Tensor createDenseSparseMatMulLHSImpl(poplar::Graph &graph,
                                      const poplar::Type &inputType,
                                      const MatMulParams &params,
                                      const CSRMatrix<T> &csrRHS,
                                      const poplar::DebugContext &debugContext,
                                      const poplar::OptionFlags optionFlags,
                                      PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, inputType, optionFlags));
  logging::popsparse::debug("Creating dense LHS for Matmul {} : dense[{}, "
                            "{}] * sparse [{}, {}]",
                            debugContext.getPathName(), params.getN(),
                            params.getK(), params.getK(), params.getM());
  if (!params.isTransposed()) {
    throw poplibs_error("MatMul params are not created for a dense * sparse "
                        "multiplication in " +
                        debugContext.getPathName());
  }
  auto csrLHS = csrTranspose(csrRHS.numRows, csrRHS.numColumns, csrRHS);
  auto newParams = MatMulParams::createForSparseDense(
      params.getNumGroups(), params.getM(), params.getK(), params.getN());
  return createSparseDenseMatMulRHSInternal(graph, inputType, newParams, csrLHS,
                                            debugContext, optionFlags, cache)
      .dimShuffle({0, 2, 1});
}

poplar::Tensor sparseDenseMatMul(poplar::Graph &graph, const SparseTensor &lhs_,
                                 const poplar::Tensor &rhs_,
                                 poplar::program::Sequence &prog,
                                 bool transposeLHS, bool transposeRHS,
                                 const poplar::DebugContext &debugContext,
                                 const poplar::OptionFlags &optionFlags,
                                 PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(lhs_, rhs_, optionFlags, cache));
  const auto &lhs = lhs_;
  const auto &opMetaData = getMatMulMetaData(lhs, debugContext.getPathName());
  const auto &planningKey = opMetaData.planningKey;

  const auto &params = planningKey.params;
  const auto &options = planningKey.options;
  const auto blockLength = planningKey.blockLength;

  auto newOptions = parseMatMulOptionFlags(optionFlags);
  newOptions.partialsType = rhs_.elementType();
  logging::popsparse::debug("Constructing graph for static spase Matmul {} : "
                            "sparse[{}, {}]{}  * dense [{}, {}]{}",
                            debugContext.getPathName(), params.getM(),
                            params.getK(), transposeLHS ? "'" : "", rhs_.dim(1),
                            rhs_.dim(2), transposeRHS ? "'" : "");

  if (options != newOptions) {
    throw poplibs_error("Options passed to sparseDenseMatMul do not match "
                        "options used to create sparse LHS for " +
                        debugContext.getPathName());
  }

  if (transposeLHS) {
    throw poplibs_error("Transpose of LHS operand not yet supported");
  }

  return sparseDenseMatMulInternal(
      graph, lhs.getNzValuesTensor(), rhs_, params, blockLength,
      planningKey.rowIndices, planningKey.columnIndices, prog, transposeLHS,
      transposeRHS, {{di}, "sparseDense"}, options, cache);
}

poplar::Tensor denseSparseMatMul(
    poplar::Graph &graph, const poplar::Tensor &lhs, const SparseTensor &rhs,
    poplar::program::Sequence &prog, bool transposeLHS, bool transposeRHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &optionFlags, PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(lhs, rhs, optionFlags, cache));
  const auto &opMetaData = getMatMulMetaData(rhs, debugContext.getPathName());
  const auto &planningKey = opMetaData.planningKey;
  const auto &params = planningKey.params;
  const auto &options = planningKey.options;
  const auto blockLength = planningKey.blockLength;

  logging::popsparse::debug("Constructing graph for static sparse Matmul {} : "
                            "dense[{}, {}]{}  * sparse [{}, {}]{}",
                            debugContext.getPathName(), lhs.dim(1), lhs.dim(2),
                            transposeLHS ? "'" : "", params.getK(),
                            params.getM(), transposeRHS ? "'" : "");
  auto newOptions = parseMatMulOptionFlags(optionFlags);
  newOptions.partialsType = lhs.elementType();
  if (options != newOptions) {
    throw poplibs_error("Options passed to denseSparseMatMul do not match "
                        "options used to create sparse RHS for " +
                        debugContext.getPathName());
  }

  if (transposeRHS) {
    throw poplibs_error("Transpose of sparse RHS in dense * sparse is not yet "
                        "supported");
  }

  return sparseDenseMatMulInternal(
             graph, rhs.getNzValuesTensor(), lhs, params, blockLength,
             planningKey.rowIndices, planningKey.columnIndices, prog, false,
             !transposeLHS, {{di}, "denseSparse"}, options, cache)
      .dimShuffle({0, 2, 1});
}

template <typename T>
SparseTensor createSparseDenseMatMulLHSImpl(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<T> &csrLHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &optionFlags, PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  logging::popsparse::debug("Creating sparse LHS for Matmul {} : sparse[{}, "
                            "{}] * dense [{}, {}]",
                            debugContext.getPathName(), params.getM(),
                            params.getK(), params.getK(), params.getN());

  if (params.isTransposed()) {
    throw poplibs_error("MatMulParams not created for Sparse * Dense matrix "
                        "multiplication in " +
                        debugContext.getPathName());
  }
  return createSparseDenseMatMulLHSInternal<T>(
      graph, inputType, params, csrLHS, debugContext, optionFlags, cache);
}

template <typename T>
SparseTensor createDenseSparseMatMulRHSImpl(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<T> &csrRHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  logging::popsparse::debug("Creating sparse RHS for Matmul {} : dense[{}, "
                            "{}] * sparse [{}, {}]",
                            debugContext.getPathName(), params.getN(),
                            params.getK(), params.getK(), params.getM());
  if (!params.isTransposed()) {
    throw poplibs_error("MatMulParams not created for Dense * Sparse matrix "
                        "multiplication in " +
                        debugContext.getPathName());
  }
  auto csrLHS = csrTranspose(csrRHS.numRows, csrRHS.numColumns, csrRHS);
  auto newParams = MatMulParams::createForSparseDense(
      params.getNumGroups(), params.getM(), params.getK(), params.getN());
  return createSparseDenseMatMulLHSInternal<T>(
      graph, inputType, newParams, csrLHS, debugContext, options, cache);
}

// Instantiations of methods that require sparsity representations
template <>
Tensor createSparseDenseMatMulRHS(poplar::Graph &graph,
                                  const poplar::Type &inputType,
                                  const MatMulParams &params,
                                  const CSRMatrix<float> &csrLHS,
                                  const poplar::DebugContext &debugContext,
                                  const poplar::OptionFlags optionFlags,
                                  PlanningCache *cache) {
  return createSparseDenseMatMulRHSImpl<float>(
      graph, inputType, params, csrLHS, debugContext, optionFlags, cache);
}

template <>
Tensor createSparseDenseMatMulRHS(poplar::Graph &graph,
                                  const poplar::Type &inputType,
                                  const MatMulParams &params,
                                  const CSRMatrix<double> &csrLHS,
                                  const poplar::DebugContext &debugContext,
                                  const poplar::OptionFlags optionFlags,
                                  PlanningCache *cache) {
  return createSparseDenseMatMulRHSImpl<double>(
      graph, inputType, params, csrLHS, debugContext, optionFlags, cache);
}

template <>
Tensor createDenseSparseMatMulLHS(poplar::Graph &graph,
                                  const poplar::Type &inputType,
                                  const MatMulParams &params,
                                  const CSRMatrix<float> &csrRHS,
                                  const poplar::DebugContext &debugContext,
                                  const poplar::OptionFlags optionFlags,
                                  PlanningCache *cache) {
  return createDenseSparseMatMulLHSImpl<float>(
      graph, inputType, params, csrRHS, debugContext, optionFlags, cache);
}

template <>
Tensor createDenseSparseMatMulLHS(poplar::Graph &graph,
                                  const poplar::Type &inputType,
                                  const MatMulParams &params,
                                  const CSRMatrix<double> &csrRHS,
                                  const poplar::DebugContext &debugContext,
                                  const poplar::OptionFlags optionFlags,
                                  PlanningCache *cache) {
  return createDenseSparseMatMulLHSImpl<double>(
      graph, inputType, params, csrRHS, debugContext, optionFlags, cache);
}

template <>
SparseTensor createSparseDenseMatMulLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<float> &csrLHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &optionFlags, PlanningCache *cache) {
  return createSparseDenseMatMulLHSImpl<float>(
      graph, inputType, params, csrLHS, debugContext, optionFlags, cache);
}

template <>
SparseTensor createSparseDenseMatMulLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<double> &csrLHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &optionFlags, PlanningCache *cache) {
  return createSparseDenseMatMulLHSImpl<double>(
      graph, inputType, params, csrLHS, debugContext, optionFlags, cache);
}

template <>
SparseTensor createDenseSparseMatMulRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<float> &csrRHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  return createDenseSparseMatMulRHSImpl<float>(graph, inputType, params, csrRHS,
                                               debugContext, options, cache);
}

template <>
SparseTensor createDenseSparseMatMulRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const MatMulParams &params, const CSRMatrix<double> &csrRHS,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  return createDenseSparseMatMulRHSImpl<double>(
      graph, inputType, params, csrRHS, debugContext, options, cache);
}

// convert from matrix representation to NZ values with an internal COO
// structure.
// TODO: Move these to a separate file once the conversion to partitioned COO
//       and dependents are moved to a separate file.
template <typename T>
std::vector<T> PartitionerImpl::getBandedNZValues(const CSCMatrix<T> &matrix,
                                                  const std::string &name) {
  return PartitionerImpl::getBandedNZValues(
      cscToCSR(matrix.numRows, matrix.numColumns, matrix), name);
}

template <typename T>
std::vector<T> PartitionerImpl::getBandedNZValues(const CSRMatrix<T> &matrix,
                                                  const std::string &name) {
  CSRMatrix<T> csrT;
  auto newParams = params;
  if (params.isTransposed()) {
    newParams = MatMulParams::createForSparseDense(
        params.getNumGroups(), params.getM(), params.getK(), params.getN());
    csrT = csrTranspose(matrix.numRows, matrix.numColumns, matrix);
  }
  auto newOptions = options;
  const CSRMatrix<T> &csr = params.isTransposed() ? csrT : matrix;
  checkParamsAndCSRConsistency(newParams, csr, {name});
  newOptions.partialsType = inputType;
  const auto plan =
      createPlan(csr, newParams, target, inputType, newOptions, cache);
  auto coo = convertToPartitionedCOO(csr, plan, target, inputType, true);

  // apply inverse operations so that partitioned COO is consistent with the
  // representation of the matrix given to the partitioner.
  applyInversePlanOperations(coo, plan, target, inputType, name);

  // check if the sparsity structure is the same if it is not empty
  if (!rowIndices.empty()) {
    if (rowIndices != coo.rowIndices || columnIndices != coo.columnIndices) {
      throw poplibs_error("Sparsity representation does not match previously"
                          "used for the same partitioner object " +
                          name);
    }
  }
  rowIndices = std::move(coo.rowIndices);
  columnIndices = std::move(coo.columnIndices);
  numRows = matrix.numRows;
  numColumns = matrix.numColumns;
  blockLength = matrix.getBlockDimensions()[0];
  return coo.nzValues;
}

template <typename T>
std::vector<T> PartitionerImpl::getBandedNZValues(const COOMatrix<T> &matrix,
                                                  const std::string &name) {
  return PartitionerImpl::getBandedNZValues(
      cooToCSR(matrix.numRows, matrix.numColumns, matrix), name);
}

// Convert NZ values with a COO structure into a COO sparse matrix
// representation.
template <typename T>
COOMatrix<T>
PartitionerImpl::bandedNZValuesToCOO(const std::vector<T> &nzValues,
                                     const std::string &name) const {
  const auto blockArea = blockLength * blockLength;
  if (nzValues.size() != rowIndices.size() * blockArea) {
    throw poplibs_error("Inconsistency between sparsity information and "
                        "partitioner object " +
                        name);
  }
  auto coo = COOMatrix<T>(numRows, numColumns, nzValues, columnIndices,
                          rowIndices, {blockLength, blockLength});
  applyInverseNZOperations(coo, inputType);
  return coo;
}

// convert NZ values with a COO structure into a CSR sparse matrix
// representation.
template <typename T>
CSRMatrix<T>
PartitionerImpl::bandedNZValuesToCSR(const std::vector<T> &nzValues,
                                     const std::string &name) const {
  auto coo = PartitionerImpl::bandedNZValuesToCOO(nzValues, name);
  return cooToCSR(numRows, numColumns, coo);
}

template <typename T>
CSCMatrix<T>
PartitionerImpl::bandedNZValuesToCSC(const std::vector<T> &nzValues,
                                     const std::string &name) const {
  auto coo = PartitionerImpl::bandedNZValuesToCOO(nzValues, name);
  return csrToCSC(numRows, numColumns, cooToCSR(numRows, numColumns, coo));
}

// Instantiations of templated member methods
template std::vector<float>
PartitionerImpl::getBandedNZValues(const CSCMatrix<float> &matrix,
                                   const std::string &name);
template std::vector<double>
PartitionerImpl::getBandedNZValues(const CSCMatrix<double> &matrix,
                                   const std::string &name);
template std::vector<float>
PartitionerImpl::getBandedNZValues(const CSRMatrix<float> &matrix,
                                   const std::string &name);
template std::vector<double>
PartitionerImpl::getBandedNZValues(const CSRMatrix<double> &matrix,
                                   const std::string &name);
template std::vector<float>
PartitionerImpl::getBandedNZValues(const COOMatrix<float> &matrix,
                                   const std::string &name);
template std::vector<double>
PartitionerImpl::getBandedNZValues(const COOMatrix<double> &matrix,
                                   const std::string &name);
template COOMatrix<float>
PartitionerImpl::bandedNZValuesToCOO(const std::vector<float> &nzValues,
                                     const std::string &name) const;
template COOMatrix<double>
PartitionerImpl::bandedNZValuesToCOO(const std::vector<double> &nzValues,
                                     const std::string &name) const;
template CSRMatrix<float>
PartitionerImpl::bandedNZValuesToCSR(const std::vector<float> &nzValues,
                                     const std::string &name) const;
template CSRMatrix<double>
PartitionerImpl::bandedNZValuesToCSR(const std::vector<double> &nzValues,
                                     const std::string &name) const;
template CSCMatrix<float>
PartitionerImpl::bandedNZValuesToCSC(const std::vector<float> &nzValues,
                                     const std::string &name) const;
template CSCMatrix<double>
PartitionerImpl::bandedNZValuesToCSC(const std::vector<double> &nzValues,
                                     const std::string &name) const;
} // namespace static_
} // namespace popsparse
