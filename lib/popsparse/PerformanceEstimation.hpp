// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/logging.hpp"

using namespace poplibs_support;

static inline std::uint64_t zeroPartialsCycles(unsigned numPartials,
                                               unsigned numWorkerContexts,
                                               bool floatPartials, bool block) {
  std::uint64_t cycles = block ? 3 : 5;
  if (numPartials) {
    unsigned vectorWidth = floatPartials ? 2 : 4;
    cycles += 10 + (numPartials + vectorWidth - 1) / vectorWidth;
  }
  return numPartials * numWorkerContexts;
}

static inline std::uint64_t sparseGatherElementWiseCycles(unsigned numIndices,
                                                          unsigned numWorkers,
                                                          bool floatData) {
  std::uint64_t supervisorCycles = 18;
  std::uint64_t workerCycles = 23;
  std::uint64_t remainderCycles = 0, vectorCycles = 0;
  unsigned cyclesPerVector = floatData ? 3 : 5;
  if (floatData) {
    auto numVectors = ((numIndices / 2) + numWorkers - 1) / numWorkers;
    if (numIndices & 0x1) {
      remainderCycles += 5;
    }
    if (numVectors) {
      vectorCycles += 5 + (numVectors - 1) * 3;
    }
  } else {
    // two additional cycles for
    auto numVectors = ((numIndices / 4) + numWorkers - 1) / numWorkers;
    if ((numIndices & 0x3) == 1) {
      remainderCycles += 10;
    } else if ((numIndices & 0x3) == 2) {
      remainderCycles += 10;
    } else if ((numIndices & 0x3) == 3) {
      remainderCycles += 14;
    }
    if (numVectors) {
      vectorCycles += 8 + (numVectors - 1) * 5;
    }
  }

  workerCycles +=
      std::max(vectorCycles, vectorCycles + remainderCycles -
                                 (vectorCycles > 0 ? cyclesPerVector : 0));
  return supervisorCycles + workerCycles * numWorkers;
}

// Should be called such that numY has one entry for many X where numY is
// an average. If the effect of different sizes of Y has to be taken into
// account, numX should be 1.
static inline std::uint64_t sparseDenseGradWElementwiseMultiply(
    unsigned numBuckets, unsigned numBucketsWithInfoForPN,
    unsigned averageSubgroupsPerBucket, unsigned numX, unsigned numZ,
    const std::vector<unsigned> &numY, bool floatInput,
    bool /* floatPartials */, unsigned numWorkerContexts) {

  std::uint64_t supervisorOverhead = 37;
  std::uint64_t supervisorCyclesWithBucketsNotForPN = 38;

  // assume uniform distribution of location. For the initial distribution this
  // can be set such that subgroup found if the very first.
  double avgSubgroupsForFirstMatch =
      static_cast<double>(averageSubgroupsPerBucket - 1) / 2;

  std::uint64_t supervisorCyclesWithBucketsForPN =
      (avgSubgroupsForFirstMatch + 1) * 26;

  std::uint64_t totalSupervisorCycles =
      supervisorOverhead + supervisorCyclesWithBucketsNotForPN +
      supervisorCyclesWithBucketsForPN * numBucketsWithInfoForPN;

  std::uint64_t workerCyclesOverhead = 26;
  std::uint64_t workerLoopCycles = 0;

  if (floatInput) {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    if (numZ == 4) {
      xOverhead = 8;
      workerCyclesOverhead += 2;
    } else if (numZ == 2) {
      xOverhead = 6;
      workerCyclesOverhead += 4;
    } else {
      xOverhead = 7;
      workerCyclesOverhead += 6;
    }

    for (const auto &y : numY) {
      if (numZ == 4) {
        innerCycles += 5 * y;
      } else if (numZ == 2) {
        innerCycles += 4 * y;
      } else {
        std::uint64_t yCycles = 9;
        if (numZ / 2) {
          yCycles += numZ / 2;
        }
        if (numZ & 0x1) {
          yCycles += 1;
        }
        innerCycles += yCycles * y;
      }
    }
    workerLoopCycles += (innerCycles + xOverhead) * numX;
  } else {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    if (numZ == 8) {
      workerCyclesOverhead += 2;
      xOverhead = 10;
    } else if (numZ == 4) {
      workerCyclesOverhead += 4;
      xOverhead = 9;
    } else {
      workerCyclesOverhead += 7;
      xOverhead = 6;
    }
    for (const auto &y : numY) {
      if (numZ == 8) {
        innerCycles += 5 * y;
      } else if (numZ == 4) {
        innerCycles += 4 * y;
      } else {
        std::uint64_t yCycles = 11;
        if (numZ / 4) {
          yCycles += numZ / 4;
        }
        if (numZ & 0x2) {
          yCycles += 3;
        }
        if (numZ & 0x1) {
          yCycles += 4;
        }
        innerCycles += yCycles * y;
      }
    }
    workerLoopCycles += (innerCycles + xOverhead) * numX;
  }

  uint64_t totalWorkerCycles = workerCyclesOverhead + workerLoopCycles;
  return totalWorkerCycles * numWorkerContexts + totalSupervisorCycles;
}

// Should be called such that numY has one entry for many X where numY is
// an average. If the effect of different sizes of Y has to be taken into
// account, numX should be 1.
static inline std::uint64_t sparseDenseBlockMultiply(
    unsigned numBuckets, unsigned numBucketsWithInfoForPN,
    unsigned averageSubgroupsPerBucket, unsigned numXBlocks, unsigned numZ,
    unsigned numBlockRows, unsigned numBlockCols,
    const std::vector<unsigned> &numYBlocks, bool floatInput,
    bool floatPartials, unsigned numWorkerContexts, unsigned numConvUnits,
    bool retainX) {

  // logging::popsparse::trace("sparseDenseElementwiseMultiply: numBuckets={},
  // numBucketsWithInfoForPN={}, averageSubgroupsPerBucket={}, numX={}, numZ={},
  // numY[0]={}, numWorkers={}", numBuckets, numBucketsWithInfoForPN,
  // averageSubgroupsPerBucket, numX, numZ, numY[0], numWorkerContexts);

  // we use 64-bit coefficient loading per Block
  uint64_t numCoeffLoadCyclesPerBlock =
      (numBlockRows * numBlockCols) / (floatInput ? 2 : 4);

  uint64_t numWeightLoadsPerBlock = 1;

  std::uint64_t supervisorOverhead = 40;
  std::uint64_t supervisorCyclesWithBucketsNotForPN =
      33 * (numBuckets - numBucketsWithInfoForPN) * averageSubgroupsPerBucket;

  // assume uniform distribution of location. For the initial distribution this
  // can be set such that subgroup found if the very first.
  double avgSubgroupsForFirstMatch =
      static_cast<double>(averageSubgroupsPerBucket - 1) / 2;

  std::uint64_t supervisorCyclesWithBucketsForPN =
      (avgSubgroupsForFirstMatch + 1) * 46 + 56;

  std::uint64_t totalSupervisorCycles =
      supervisorOverhead + supervisorCyclesWithBucketsNotForPN +
      supervisorCyclesWithBucketsForPN * numBucketsWithInfoForPN;

  // The overhead is the cycles for retention and depends on numZ.
  // We just take the worst case as the difference is small
  std::uint64_t workerCyclesOverhead = 26;
  std::uint64_t workerLoopCycles = 0;
  std::uint64_t supervisorBlockLoadCycles = 0;
  std::uint64_t innerOverhead = 0;
  std::uint64_t cyclesPerZ = 0;
  assert(numBlockRows == numBlockRows);
  switch (numBlockRows) {
  case 4:
    cyclesPerZ = floatInput ? 4 : 2;
    workerCyclesOverhead = 26;
    if (floatPartials) {
      if (numZ == 1) {
        innerOverhead = floatInput ? 14 : 17;
      } else if (numZ == 2) {
        innerOverhead = floatInput ? 14 : 18;
      } else {
        innerOverhead = floatInput ? 14 : 19;
      }
    } else {
      if (numZ == 1) {
        innerOverhead = 15;
      } else if (numZ == 2) {
        innerOverhead = 16;
      } else {
        innerOverhead = 17;
      }
    }
    break;
  case 8:
    cyclesPerZ = floatInput ? 8 : 4;
    workerCyclesOverhead = floatInput ? 23 : (floatPartials ? 20 : 23);
    if (floatPartials) {
      if (numZ == 1) {
        innerOverhead = 22;
      } else if (numZ == 2) {
        innerOverhead = floatInput ? 22 : 23;
      } else {
        innerOverhead = floatInput ? 23 : 23;
      }
    } else {
      if (numZ == 1) {
        innerOverhead = 19;
      } else if (numZ == 2) {
        innerOverhead = 20;
      } else {
        innerOverhead = 20;
      }
    }
    break;
  case 16:
    numWeightLoadsPerBlock = numConvUnits == 8 ? 2 : 1;
    cyclesPerZ = 4;
    if (floatPartials) {
      workerCyclesOverhead = 23;
      if (numZ == 1) {
        innerOverhead = 24;
      } else if (numZ == 2) {
        innerOverhead = 25;
      } else {
        innerOverhead = 24;
      }
    } else {
      if (numConvUnits == 16) {
        workerCyclesOverhead = 22;
        if (numZ == 1) {
          innerOverhead = 22;
        } else if (numZ == 2) {
          innerOverhead = 23;
        } else {
          innerOverhead = 22;
        }
      } else if (numConvUnits == 8) {
        workerCyclesOverhead = 23;
        if (numZ == 1) {
          innerOverhead = 24;
        } else if (numZ == 2) {
          innerOverhead = 25;
        } else {
          innerOverhead = 24;
        }
      } else {
        assert(0 && "Unhandled no. of conv units");
      }
    }
    break;
  }

  auto innerCycles = innerOverhead + numZ * cyclesPerZ;
  for (const auto &y : numYBlocks) {
    supervisorBlockLoadCycles +=
        y * (numCoeffLoadCyclesPerBlock + 3 + (12 * numWeightLoadsPerBlock));
    workerLoopCycles +=
        y * (numWeightLoadsPerBlock * (innerCycles - retainX * 3));
  }
  supervisorBlockLoadCycles =
      (supervisorBlockLoadCycles + retainX) * numXBlocks + 2;
  workerLoopCycles *= numXBlocks;
  uint64_t totalWorkerCycles = workerCyclesOverhead + workerLoopCycles;
  totalSupervisorCycles += supervisorBlockLoadCycles;
  return totalWorkerCycles * numWorkerContexts + totalSupervisorCycles;
}

// Should be called such that numY has one entry for many X where numY is
// an average. If the effect of different sizes of Y has to be taken into
// account, numX should be 1.
static inline std::uint64_t sparseDenseBlockMultiplyGradW(
    unsigned numBuckets, unsigned numBucketsWithInfoForPN,
    unsigned averageSubgroupsPerBucket, unsigned numXBlocks, unsigned numZ,
    unsigned numBlockRows, unsigned numBlockCols,
    const std::vector<unsigned> &numYBlocks, bool floatInput,
    bool floatPartials, unsigned numWorkerContexts) {
  std::uint64_t supervisorOverhead = 37;
  std::uint64_t supervisorCyclesWithBucketsNotForPN = 38;

  // assume uniform distribution of location. For the initial distribution this
  // can be set such that subgroup found if the very first.
  double avgSubgroupsForFirstMatch =
      static_cast<double>(averageSubgroupsPerBucket - 1) / 2;

  std::uint64_t supervisorCyclesWithBucketsForPN =
      (avgSubgroupsForFirstMatch + 1) * 26;

  std::uint64_t totalSupervisorCycles =
      supervisorOverhead + supervisorCyclesWithBucketsNotForPN +
      supervisorCyclesWithBucketsForPN * numBucketsWithInfoForPN;

  std::uint64_t workerCyclesOverhead = 26;
  std::uint64_t workerLoopCycles = 0;
  const auto blockArea = numBlockRows * numBlockCols;
  std::uint64_t xOverhead = 0;
  std::uint64_t yOverhead = 0;
  std::uint64_t cyclesPerZ = 0;

  if (blockArea == 16) {
    if (floatPartials) {
      if (floatInput) {
        workerCyclesOverhead = 27;
        xOverhead = 9;
        yOverhead = 45;
        cyclesPerZ = 12;
      } else {
        workerCyclesOverhead = 26;
        xOverhead = 9;
        yOverhead = 33;
        cyclesPerZ = 6;
      }
    } else {
      workerCyclesOverhead = 26;
      xOverhead = 9;
      yOverhead = 29;
      cyclesPerZ = 6;
    }
  } else if (numBlockRows == 8 && numBlockCols == 8) {
    if (floatInput) {
      assert(floatPartials);
      workerCyclesOverhead = 24;
      const auto numColumnLoops = 2;
      const auto numRowLoops = 8;
      const auto rowLoopOverhead = 5;
      const auto columnLoopOverhead = 10;
      xOverhead = 7;
      yOverhead = 8 + numRowLoops * (rowLoopOverhead +
                                     numColumnLoops * (columnLoopOverhead));
      const auto cyclesPerInnerLoop = 3;
      cyclesPerZ = numRowLoops * numColumnLoops * cyclesPerInnerLoop;
    } else {
      // Identical for half/float inputs
      workerCyclesOverhead = 26;
      const auto numRowLoops = 8;
      const auto rowLoopOverhead = 13;
      const auto cyclesPerInnerLoop = 3;
      xOverhead = 9;
      yOverhead = 9 + numRowLoops * rowLoopOverhead;
      cyclesPerZ = numRowLoops * cyclesPerInnerLoop;
    }
  } else if (blockArea == 256) {
    if (floatPartials) {
      // float input doesn't have an assembler codelet
      workerCyclesOverhead = 27;
      xOverhead = 9;
      yOverhead = 505;
      cyclesPerZ = 96;
    } else {
      workerCyclesOverhead = 27;
      xOverhead = 9;
      yOverhead = 441;
      cyclesPerZ = 96;
    }
  }

  std::uint64_t innerLoopCycles = 0;
  for (const auto &y : numYBlocks) {
    innerLoopCycles += y * (yOverhead + numZ * cyclesPerZ);
  }

  workerLoopCycles += numXBlocks * (xOverhead + innerLoopCycles);
  uint64_t totalWorkerCycles = workerCyclesOverhead + workerLoopCycles;
  return totalWorkerCycles * numWorkerContexts + totalSupervisorCycles;
}

static inline std::uint64_t sparseDenseBlockMultiplyGradWAmp(
    unsigned numBuckets, unsigned numBucketsWithInfoForPN,
    unsigned averageSubgroupsPerBucket, unsigned numXBlocks, unsigned numZ,
    unsigned numBlockRows, unsigned numBlockCols,
    const std::vector<unsigned> &numYBlocks, bool floatInput,
    bool floatPartials, unsigned numWorkerContexts, unsigned numConvUnits) {
  std::uint64_t supervisorOverhead = 37;
  std::uint64_t supervisorCyclesWithBucketsNotForPN = 38;

  // assume uniform distribution of location. For the initial distribution this
  // can be set such that subgroup found if the very first.
  double avgSubgroupsForFirstMatch =
      static_cast<double>(averageSubgroupsPerBucket - 1) / 2;

  std::uint64_t supervisorCyclesWithBucketsForPN =
      (avgSubgroupsForFirstMatch + 1) * 26;

  std::uint64_t totalSupervisorCycles =
      supervisorOverhead + supervisorCyclesWithBucketsNotForPN +
      supervisorCyclesWithBucketsForPN * numBucketsWithInfoForPN;

  // Supervisor cycles: split into first and next for the first pass
  // and subsequent passes because we exploit the fact that there is worker
  // imbalance. We assume for all cases, the worker imbalance is large enough
  // to account for all the excess cycles before a sync. The other way to do it
  // would be loop around and find the max and min.
  std::uint64_t sXOverheadFirst = 23;
  std::uint64_t sXOverheadOther = 8;
  std::uint64_t sYOverheadFirst = 32;
  std::uint64_t sYOverheadOther = 10;
  std::uint64_t sZOverheadFirst = 15;
  std::uint64_t sZOverheadOther = 4;
  std::uint64_t sRunCycles = 8;
  std::uint64_t sWeightLoad = 0;
  std::uint64_t wRetention = 0;
  std::uint64_t wNonRetention = 0;
  std::uint64_t wZOffRetention = 0;

  if (numBlockRows == 16 && numBlockCols == 16) {
    sWeightLoad = 234;
    assert(!floatInput);
    if (numConvUnits == 4 && !floatInput) {
      sRunCycles = 17;
      wZOffRetention = 2;
      // times 2 for non retention because of 2 passes of 8x16
      if (floatPartials) {
        wRetention = 11;
        wNonRetention = 2 * 33 - wZOffRetention;
      } else {
        wRetention = 9;
        wNonRetention = 2 * 31 - wZOffRetention;
      }
    } else {
      wZOffRetention = 4;
      wRetention = 10;
      wNonRetention = 33;
    }
  } else if (numBlockRows == 8 && numBlockCols == 8) {
    sWeightLoad = 114;

    if (floatInput) {
      wZOffRetention = 2;
      wRetention = 8;
      wNonRetention = 34;
    } else {
      wZOffRetention = 4;
      wRetention = floatPartials ? 8 : 7;
      wNonRetention = 28;
    }
  } else if (numBlockRows == 4 && numBlockCols == 4) {
    sWeightLoad = 54;
    wRetention = 6;
    if (floatInput) {
      wZOffRetention = 1;
      wNonRetention = 21;
    } else {
      wZOffRetention = 2;
      wNonRetention = floatPartials ? 18 : 17;
    }
  }

  totalSupervisorCycles += (numXBlocks - 1) * sXOverheadOther + sXOverheadFirst;
  uint64_t supervisorLoopCycles = 0;
  std::uint64_t workerCycles = wRetention;
  std::uint64_t innerLoopCycles = 0;

  for (const auto &y : numYBlocks) {
    supervisorLoopCycles += sYOverheadFirst + (y - 1) * sYOverheadOther;
    const auto numBlocks = ceildiv(numZ, floatInput ? 8U : 16U);
    supervisorLoopCycles +=
        (sZOverheadFirst + (numBlocks - 1) * sZOverheadOther +
         numBlocks * (sRunCycles + sWeightLoad)) *
        y;
    innerLoopCycles +=
        y * (wNonRetention * numBlocks - (numBlocks - 1) * wZOffRetention);
  }
  totalSupervisorCycles += supervisorLoopCycles * numXBlocks;
  workerCycles += numXBlocks * innerLoopCycles;
  return workerCycles * numWorkerContexts + totalSupervisorCycles;
}

// Should be called such that numY has one entry for many X where numY is
// an average. If the effect of different sizes of Y has to be taken into
// account, numX should be 1.
static inline std::uint64_t sparseDenseElementwiseMultiply(
    unsigned numBuckets, unsigned numBucketsWithInfoForPN,
    unsigned averageSubgroupsPerBucket, unsigned numX, unsigned numZ,
    const std::vector<unsigned> &numY, bool floatInput,
    bool /* floatPartials */, unsigned numWorkerContexts) {

  std::uint64_t supervisorOverhead = 30;
  std::uint64_t supervisorCyclesWithBucketsNotForPN =
      32 * (numBuckets - numBucketsWithInfoForPN) * averageSubgroupsPerBucket;

  // assume uniform distribution of location. For the initial distribution this
  // can be set such that subgroup found if the very first.
  double avgSubgroupsForFirstMatch =
      static_cast<double>(averageSubgroupsPerBucket - 1) / 2;

  std::uint64_t supervisorCyclesWithBucketsForPN =
      (avgSubgroupsForFirstMatch + 1) * 44 + 46;

  std::uint64_t totalSupervisorCycles =
      supervisorOverhead + supervisorCyclesWithBucketsNotForPN +
      supervisorCyclesWithBucketsForPN * numBucketsWithInfoForPN;

  std::uint64_t workerCyclesOverhead = 0;
  std::uint64_t workerLoopCycles = 0;

  if (floatInput) {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    if (numZ == 4) {
      workerCyclesOverhead += 22;
      xOverhead = 13;
    } else if (numZ == 2) {
      workerCyclesOverhead += 24;
      xOverhead = 11;
    } else {
      workerCyclesOverhead += 27;
      xOverhead = 12;
    }
    for (const auto &y : numY) {
      if (numZ == 4) {
        innerCycles += 3 * y;
      } else if (numZ == 2) {
        innerCycles += 2 * y;
      } else {
        if (numZ / 4) {
          innerCycles += 12 + numZ / 4 * 3 * y;
        }
        if (numZ & 0x2) {
          innerCycles += 8 + numZ / 2 * 2 * y;
        }
        if (numZ & 0x1) {
          innerCycles += 8 + 2 * y;
        }
        innerCycles += (numZ & 0x3) ? 3 : 1;
      }
    }
    workerLoopCycles += (xOverhead + innerCycles) * numX;
  } else {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    if (numZ == 8) {
      workerCyclesOverhead += 25;
      xOverhead = 15;
    } else if (numZ == 4) {
      workerCyclesOverhead += 27;
      xOverhead = 11;
    } else {
      workerCyclesOverhead += 30;
      xOverhead = 11;
    }
    for (const auto &y : numY) {
      if (numZ == 8) {
        innerCycles += 3 * y;
      } else if (numZ == 4) {
        innerCycles += 2 * y;
      } else {
        if (numZ / 8) {
          innerCycles += 16 + numZ / 8 * 3 * y;
        }
        if (numZ & 0x4) {
          innerCycles += 13 + 2 * y;
        }
        if (numZ & 0x3) {
          innerCycles += 11 + 2 * y * (numZ & 0x3);
        }
      }
    }
    workerLoopCycles += (xOverhead + innerCycles) * numX;
  }

  uint64_t totalWorkerCycles = workerCyclesOverhead + workerLoopCycles;
  return totalWorkerCycles * numWorkerContexts + totalSupervisorCycles;
}

// Should be called such that numY has one entry for many X where numY is
// an average. If the effect of different sizes of Y has to be taken into
// account, numX should be 1.
static inline std::uint64_t sparseDenseGradAElementwiseMultiply(
    unsigned numBuckets, unsigned numBucketsWithInfoForPN,
    unsigned averageSubgroupsPerBucket, unsigned numX, unsigned numZ,
    const std::vector<unsigned> &numY, bool floatInput, bool floatPartials,
    unsigned numWorkerContexts) {

  std::uint64_t supervisorOverhead = 30;
  std::uint64_t supervisorCyclesWithBucketsNotForPN =
      32 * (numBuckets - numBucketsWithInfoForPN) * averageSubgroupsPerBucket;

  // assume uniform distribution of location. For the initial distribution this
  // can be set such that subgroup found if the very first.
  double avgSubgroupsForFirstMatch =
      static_cast<double>(averageSubgroupsPerBucket - 1) / 2;

  std::uint64_t supervisorCyclesWithBucketsForPN =
      (avgSubgroupsForFirstMatch + 1) * 44 + 46;

  std::uint64_t totalSupervisorCycles =
      supervisorOverhead + supervisorCyclesWithBucketsNotForPN +
      supervisorCyclesWithBucketsForPN * numBucketsWithInfoForPN;

  std::uint64_t workerCyclesOverhead = 0;
  std::uint64_t workerLoopCycles = 0;

  if (floatInput) {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    if (numZ == 4) {
      workerCyclesOverhead += 20;
      xOverhead = 13;
    } else if (numZ == 2) {
      workerCyclesOverhead += 22;
      xOverhead = 11;
    } else {
      workerCyclesOverhead += 30;
      xOverhead = 12;
    }
    for (const auto &y : numY) {
      if (numZ == 4) {
        innerCycles += 3 * y;
      } else if (numZ == 2) {
        innerCycles += 2 * y;
      } else {
        if (numZ / 4) {
          innerCycles += 10 + numZ / 4 * 3 * y;
        }
        if (numZ & 0x2) {
          innerCycles += 8 + numZ / 2 * 2 * y;
        }
        if (numZ & 0x1) {
          innerCycles += 8 + 2 * y;
        }
        innerCycles += (numZ & 0x3) ? 3 : 1;
      }
    }
    workerLoopCycles += (xOverhead + innerCycles) * numX;
  } else {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    if (numZ == 8) {
      workerCyclesOverhead += 23;
      xOverhead = 15;
    } else if (numZ == 4) {
      workerCyclesOverhead += 25;
      xOverhead = 11;
    } else {
      workerCyclesOverhead += 29;
      xOverhead = 10;
    }
    for (const auto &y : numY) {
      if (numZ == 8) {
        innerCycles += 3 * y;
      } else if (numZ == 4) {
        innerCycles += 2 * y;
      } else {
        if (numZ / 8) {
          innerCycles += 14 + numZ / 8 * 3 * y;
        }
        if (numZ & 0x4) {
          innerCycles += 13 + 2 * y;
        }
        if (numZ & 0x3) {
          innerCycles += 10 + 2 * y * (numZ & 0x3);
        }
      }
    }
    workerLoopCycles += (xOverhead + innerCycles) * numX;
  }

  uint64_t totalWorkerCycles = workerCyclesOverhead + workerLoopCycles;
  return totalWorkerCycles * numWorkerContexts + totalSupervisorCycles;
}

// Should be called such that numY has one entry for many X where numY is
// an average. If the effect of different sizes of Y has to be taken into
// account, numX should be 1.
// The X and Y given to this function should be the same that given for the
// forward as it uses the same meta-information
static inline std::uint64_t sparseDenseTransposeElementwiseMultiply(
    unsigned numBuckets, unsigned numBucketsWithInfoForPN,
    unsigned averageSubgroupsPerBucket, unsigned numX, unsigned numZ,
    const std::vector<unsigned> &numY, bool floatInput, bool floatPartials,
    unsigned numWorkerContexts) {

  std::uint64_t supervisorOverhead = 30;
  std::uint64_t supervisorCyclesWithBucketsNotForPN =
      32 * (numBuckets - numBucketsWithInfoForPN) * averageSubgroupsPerBucket;

  // assume uniform distribution of location. For the initial distribution this
  // can be set such that subgroup found if the very first.
  double avgSubgroupsForFirstMatch =
      static_cast<double>(averageSubgroupsPerBucket - 1) / 2;

  std::uint64_t supervisorCyclesWithBucketsForPN =
      (avgSubgroupsForFirstMatch + 1) * 44 + 68;

  std::uint64_t totalSupervisorCycles =
      supervisorOverhead + supervisorCyclesWithBucketsNotForPN +
      supervisorCyclesWithBucketsForPN * numBucketsWithInfoForPN;

  std::uint64_t workerCyclesOverhead = 0;
  std::uint64_t workerLoopCycles = 0;

  if (floatInput) {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    workerCyclesOverhead = 21;
    for (const auto &yTotal : numY) {
      xOverhead = 22;
      const auto y = (yTotal + numWorkerContexts - 1) / numWorkerContexts;
      innerCycles += 8;
      if (numZ / 2) {
        innerCycles += (2 + numZ / 2 * 3) * y;
      }
      if (numZ & 0x1) {
        innerCycles += 4 * y;
      }
    }
    workerLoopCycles += (xOverhead + innerCycles) * numX;
  } else {
    std::uint64_t innerCycles = 0;
    std::uint64_t xOverhead = 0;
    if (numZ == 8) {
      workerCyclesOverhead += 18;
      xOverhead = 16;
    } else if (numZ == 4) {
      workerCyclesOverhead += 18;
      xOverhead = 19;
    } else {
      workerCyclesOverhead += 24;
      xOverhead = 20;
    }
    for (const auto &yTotal : numY) {
      const auto y = (yTotal + numWorkerContexts - 1) / numWorkerContexts;
      if (numZ == 8) {
        innerCycles += 10 * y;
      } else if (numZ == 4) {
        innerCycles += 7 * y;
      } else {
        innerCycles += 8;
        if (numZ / 4) {
          innerCycles += (2 + numZ / 4 * 5) * y;
        }
        if (numZ & 0x3) {
          innerCycles += (5 + 3 * (numZ & 0x3)) * y;
        }
      }
    }
    workerLoopCycles += (xOverhead + innerCycles) * numX;
  }

  uint64_t totalWorkerCycles = workerCyclesOverhead + workerLoopCycles;
  return totalWorkerCycles * numWorkerContexts + totalSupervisorCycles;
}

static inline std::uint64_t getCastCycleEstimate(unsigned outputSize,
                                                 unsigned partialsVectorWidth,
                                                 unsigned outputVectorWidth,
                                                 unsigned numWorkers) {
  const auto outputPerWorker = (outputSize + numWorkers - 1) / numWorkers;
  std::uint64_t loadPartialsCycles =
      (outputPerWorker + partialsVectorWidth - 1) / partialsVectorWidth;
  std::uint64_t writeOutputCycles =
      (outputPerWorker + outputVectorWidth - 1) / outputVectorWidth;
  std::uint64_t cycles = std::max(loadPartialsCycles, writeOutputCycles);
  return (cycles + 26) * numWorkers;
}

static inline std::uint64_t
getReduceCycleEstimate(unsigned outSize, unsigned partialsSize,
                       unsigned dataPathWidth, bool isOutTypeFloat,
                       bool isPartialsFloat, unsigned numWorkers) {
  unsigned cycles = 0;

  // Supervisor vertex, and new implementation
  if (isPartialsFloat) {
    cycles = 32;
    // Float - workers process 4 at once, and account for remainder loops
    auto loops = outSize / 4;
    if (outSize & 1)
      loops++;
    if (outSize & 2)
      loops++;
    // Account for time at full load - all workers busy
    auto loopsDividedBetweenWorkers = loops / numWorkers;
    // and a remainder where only some are busy which can be a shorter loop
    if (loops % numWorkers) {
      if (outSize & 3)
        cycles += (2 * partialsSize + 13);
      else
        loopsDividedBetweenWorkers++;
    }

    if (isOutTypeFloat)
      cycles += (3 * partialsSize + 7) * loopsDividedBetweenWorkers;
    else
      cycles += (3 * partialsSize + 6) * loopsDividedBetweenWorkers;
  } else {
    cycles = 32;
    // Half - workers process 8 at once, and account for remainder loops
    auto loops = outSize / 8;
    if (outSize & 1)
      loops++;
    if (outSize & 2)
      loops++;
    if (outSize & 4)
      loops++;
    // Account for time at full load - all workers busy
    auto loopsDividedBetweenWorkers = loops / numWorkers;
    // and a remainder where only some are busy which can be a shorter loop
    if (loops % numWorkers) {
      if (outSize & 7)
        cycles += (2 * partialsSize + 11);
      else
        loopsDividedBetweenWorkers++;
    }

    if (isOutTypeFloat)
      cycles += (3 * partialsSize + 9) * loopsDividedBetweenWorkers;
    else
      cycles += (3 * partialsSize + 8) * loopsDividedBetweenWorkers;
  }
  cycles = cycles * numWorkers;

  return cycles;
}

// Supervisor transpose estimate assuming we have numTransposes contiguous in
// memory based on supervisor half transpose vertex with a guaranteed minimum
// grain size.
static inline std::uint64_t
getTransposeCycleEstimate(unsigned numTransposes, unsigned numSrcRows,
                          unsigned numSrcColumns, const poplar::Type &dataType,
                          unsigned numWorkers) {

  const std::uint64_t supervisorCycles = 7;

  assert(dataType == poplar::FLOAT || dataType == poplar::HALF);
  const bool is4ByteType = dataType == poplar::FLOAT;

  const unsigned grainSize = is4ByteType ? 2 : 4;
  const std::uint64_t cyclesPerGrain = is4ByteType ? 3 : 4;

  std::uint64_t maxWorkerCycles = 0;
  assert(numSrcRows % grainSize == 0);
  assert(numSrcColumns % grainSize == 0);
  const auto numSrcRowGrains = numSrcRows / grainSize;
  const auto numSrcColumnGrains = numSrcColumns / grainSize;
  const std::uint64_t maxTransposesPerWorker =
      ceildiv(numTransposes, numWorkers);
  if (numSrcRowGrains == 1 && numSrcColumnGrains == 1) {
    if (maxTransposesPerWorker == 1) {
      maxWorkerCycles = 17 + 12;
    } else {
      maxWorkerCycles = 17 + 20 + (maxTransposesPerWorker - 2) * cyclesPerGrain;
    }
  } else if (numSrcColumnGrains == 1) {
    maxWorkerCycles =
        27 + maxTransposesPerWorker *
                 (15 + (20 + cyclesPerGrain * (numSrcRowGrains - 2)));
  } else {
    maxWorkerCycles =
        29 + maxTransposesPerWorker *
                 (18 + (numSrcRowGrains *
                        (12 + cyclesPerGrain * (numSrcColumnGrains - 2))));
  }

  return supervisorCycles + numWorkers * maxWorkerCycles;
}

#endif // _performance_estimation_h_
