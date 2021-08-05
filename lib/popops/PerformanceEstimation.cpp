// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popops/PerformanceEstimation.hpp"

#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/forceInterleavedEstimates.hpp"
#include "poplibs_support/gcd.hpp"

#include <poputil/cyclesTables.hpp>
#include <poputil/exceptions.hpp>

#include <cmath>

using namespace poplar;
using namespace poplibs_support;

namespace popops {
namespace internal {

std::uint64_t basicOpSupervisorOverhead(const bool isScaledPtr64Type) {

  // common supervisor overhead
  std::uint64_t cycles = 11;

  // extra 2 cycles needed to unpack A and B pointers if they are scaled.
  if (isScaledPtr64Type) {
    cycles += 2;
  }

  return cycles;
}

/* Cycle cost computation for basic operations */
std::uint64_t basicOpLoopCycles(const unsigned numElems,
                                const unsigned vectorSize,
                                const unsigned cyclesPerVector) {
  return cyclesPerVector * (numElems + vectorSize - 1) / vectorSize;
}

std::uint64_t
getMultiSliceCycleEstimate(const MultiSliceTargetParameters &targetParams,
                           const unsigned elemsPerSlice,
                           const unsigned numOffsets,
                           const double proportionOfIndexableRange,
                           const bool useOnePointDistribution) {
  (void)useOnePointDistribution;
  assert(numOffsets != 0);

  unsigned vectorWidth;
  unsigned cyclesPerVector;
  switch (targetParams.bytesPerElem) {
  case 1:
    vectorWidth = 4;
    cyclesPerVector = 3;
    break;
  case 2:
    vectorWidth = 2;
    cyclesPerVector = 2;
    break;
  case 4:
    assert(targetParams.dataPathWidth % (targetParams.bytesPerElem * 8) == 0);
    vectorWidth =
        (targetParams.dataPathWidth / (targetParams.bytesPerElem * 8));
    cyclesPerVector = 2;
    break;
  default:
    assert(false && "getMultiSliceCycleEstimate for unhandled element size");
    break;
  }
  constexpr std::uint64_t proAndEpilogueCycles = 13;
  // Almost exactly the same for each copy function assuming fastest
  // (aligned) path.
  constexpr std::uint64_t cyclesOverheadPerOffsetInRange = 19;
  constexpr std::uint64_t cyclesOverheadPerOffsetOutOfRange = 8;
  // Note the assumption that every offset in the vertex requires copying.
  // This could be pessimistic for a multi-stage multiSlice operation
  // where vertices in the first stage may try to slice indices which
  // are not part of partition of the sliced tensor on that tile and so
  // skip the copying cycles in a data-dependent way.
  const auto vectorsPerOffset = ceildiv(elemsPerSlice, vectorWidth);

  const unsigned numOffsetsInRange =
      useOnePointDistribution
          ? numOffsets
          : std::ceil(numOffsets * proportionOfIndexableRange);
  const unsigned numOffsetsOutOfRange = numOffsets - numOffsetsInRange;

  const std::uint64_t offsetsInRangeCycles =
      numOffsetsInRange *
      (cyclesOverheadPerOffsetInRange + (vectorsPerOffset * cyclesPerVector));
  const std::uint64_t offsetsOutOfRangeCycles =
      numOffsetsOutOfRange * cyclesOverheadPerOffsetOutOfRange;
  const std::uint64_t coreCycles =
      offsetsInRangeCycles + offsetsOutOfRangeCycles;
  return proAndEpilogueCycles + coreCycles;
}

std::uint64_t getMultiUpdateOpCycleEstimate(
    const MultiUpdateOpTargetParameters &targetParams, bool floatData,
    bool subWordWritesRequired, const unsigned elemsPerSlice,
    const unsigned numOffsets, const Operation op, const bool scaled,
    const double maxProportionOfIndexableRangePerWorker,
    const bool useOnePointDistribution) {

  std::uint64_t cycles = 3; // load size, zero check and exitz.

  if (numOffsets == 0) {
    return cycles;
  }

  // pre-outer loop overhead.
  cycles += floatData ? 24 : 25;

  if (scaled) {
    cycles += 2;
  }

  // outer loop overhead, before and after the inner loop.
  // cycle cost is data dependent on values of offsets.
  std::uint64_t cyclesPerOffsetInRange = floatData ? 11 : 12;
  cyclesPerOffsetInRange += (!floatData && scaled) ? 1u : 0u;
  const std::uint64_t cyclesPerOffsetOutOfRange = 5;

  const unsigned bytesPerElem = floatData ? 4 : 2;

  // inner loop cost.
  // Note gcd is used here for e.g. CPU where the atomic write size is 1.
  const unsigned bytesPerAtom = lcm(targetParams.atomicWriteSize, bytesPerElem);
  const unsigned elemsPerAtom = bytesPerAtom / bytesPerElem;
  // for the assembly implementation elemsPerSlice % vectorWidth == 0 must be
  // zero.
  if (subWordWritesRequired) {
    assert(!floatData);
    // Not based on anything in particular other than per-element cost in
    // generated code for C++ being high (even higher for half type).
    cyclesPerOffsetInRange += elemsPerSlice * 20;
  } else {
    assert(elemsPerSlice != 0 && elemsPerSlice % elemsPerAtom == 0);
    cyclesPerOffsetInRange += (elemsPerSlice / elemsPerAtom - 1) * 3;
  }

  const unsigned numOffsetsInRange =
      useOnePointDistribution
          ? numOffsets
          : std::ceil(numOffsets * maxProportionOfIndexableRangePerWorker);
  const unsigned numOffsetsOutOfRange = numOffsets - numOffsetsInRange;
  cycles += cyclesPerOffsetInRange * numOffsetsInRange;
  cycles += cyclesPerOffsetOutOfRange * numOffsetsOutOfRange;
  constexpr unsigned supervisorCycles = 25;
  return cycles * targetParams.numWorkerContexts + supervisorCycles;
}

// Returns the cycles of one to the 'cast_XXX_XXX_core' functions in assembly,
// or the equivalent section of code in one of the C++ codelets.
static uint64_t castWorkerCycles(const unsigned numElems, const Type &fromType,
                                 const Type &toType,
                                 const CastTargetParameters &targetParams) {
  std::uint64_t cycles = 0;
  const bool isCharFloat = (fromType == UNSIGNED_CHAR ||
                            fromType == SIGNED_CHAR || fromType == CHAR) &&
                           (toType == FLOAT || toType == HALF);
  const bool isFloatChar =
      (fromType == FLOAT || fromType == HALF) &&
      (toType == UNSIGNED_CHAR || toType == SIGNED_CHAR || toType == CHAR);

  const bool isDstLongLong = toType == UNSIGNED_LONGLONG || toType == LONGLONG;
  if (isDstLongLong) {
    std::uint64_t cyclesPerElem =
        (fromType == INT || fromType == CHAR || fromType == SHORT) + 4;
    return 3 + numElems * cyclesPerElem;
  }

  if (isCharFloat || isFloatChar) {
    // These assembly functions have a common structure, using an atom-sized
    // (2/4 elems) pipelined loop and a "0,1,2 or 3" remainder section.
    auto workCycles = [&](auto atomSize, auto fillDrainCycles,
                          auto cyclesPerLoop, auto rem1, auto rem2, auto rem3) {
      unsigned c = 0;
      unsigned nAtom = numElems / atomSize;
      if (nAtom > 0) {
        c += fillDrainCycles + (nAtom - 1) * cyclesPerLoop;
      }
      unsigned rem = numElems % atomSize;
      if (rem == 0) {
        return c + 2;
      } else if (rem == 1) {
        return c + rem1;
      } else if (rem == 2) {
        return c + rem2;
      } else if (rem == 3) {
        return c + rem3;
      } else {
        throw poputil::poplibs_error("in castWorkerCycles/workCycles, "
                                     "remainder must be 0..3, cannot be " +
                                     std::to_string(rem));
      }
    };
    // setup clamping when casting FROM int8 types
    unsigned clampSetupCycles = (fromType == UNSIGNED_CHAR) ? 3 : 4;
    cycles += 4; // all functions start with a 4 instruction sequence.
    // CastFromInt8.S
    if (fromType == UNSIGNED_CHAR && toType == HALF) {
      cycles += workCycles(4, 14, 9, 12, 14, 22);
    } else if ((fromType == SIGNED_CHAR || fromType == CHAR) &&
               toType == HALF) {
      cycles += workCycles(4, 15, 13, 12, 14, 22);
    } else if (fromType == UNSIGNED_CHAR && toType == FLOAT) {
      cycles += workCycles(4, 14, 10, 9, 15, 21);
    } else if ((fromType == SIGNED_CHAR || fromType == CHAR) &&
               toType == FLOAT) {
      cycles += workCycles(2, 9, 7, 10, 0, 0);
      // CastToInt8.S:
    } else if (fromType == HALF) {
      cycles += clampSetupCycles + workCycles(4, 19, 16, 14, 21, 29);
    } else if (fromType == FLOAT) {
      cycles += clampSetupCycles + workCycles(4, 18, 14, 13, 21, 28);
    }
  } else {
    const auto dataPathWidth = targetParams.dataPathWidth / 8;
    const auto fromLoadWidth = dataPathWidth / targetParams.fromTypeSize;
    const auto toStoreWidth = dataPathWidth / targetParams.toTypeSize;

    // We take a guess that the vector width possible for the op will be a
    // function of the available load/store bandwidth and number
    // of read/write ports. e.g. f32v2tof16 is 2 reads of 64-bit and 1 write
    // of 64-bits and f16v2tof32, a 64-bit write is the bottleneck.
    constexpr unsigned readPorts = 2, writePorts = 1;
    const bool conversionIsAuxPipeline =
        (fromType == FLOAT || fromType == HALF) &&
        (toType == FLOAT || toType == HALF);

    // If not aux pipeline (i.e. not floating point conversion) we give an
    // innaccurate guess anyhow as we will be using C++ code to perform
    // the conversion.
    const auto opVectorWidth =
        conversionIsAuxPipeline
            ? std::min(fromLoadWidth * readPorts, toStoreWidth * writePorts)
            : 1;

    // We then get the number of cycles to calculate each of these vectors.
    // NOTE: We don't use interleaved memory currently hence we don't utilise
    // multiple read ports. We do assume we use separate memory elements for
    // load/store to overlap loads/stores where possible.
    const auto loadCyclesPerVector =
        opVectorWidth /
        std::min(opVectorWidth,
                 fromLoadWidth *
                     (getForceInterleavedEstimates() ? readPorts : 1));
    const auto storeCyclesPerVector =
        opVectorWidth / std::min(opVectorWidth, toStoreWidth);
    const auto cyclesPerVector =
        std::max(loadCyclesPerVector, storeCyclesPerVector) +
        !conversionIsAuxPipeline;

    // Prologue cycles based on Half_Float_core assuming alignment and enough
    // elements to process.
    cycles += 19;
    // Cycles for processing vectors
    cycles += cyclesPerVector * (numElems / opVectorWidth);

    // Rough estimation of cycles for processing remainders. This should be
    // relatively insignificant so rough is fine.
    const auto remainingElems = numElems % opVectorWidth;
    const auto maxRemainderBit = ceilLog2(opVectorWidth);
    for (unsigned i = 0; i < maxRemainderBit; ++i) {
      const auto remainder = (1u << i);
      // Check the remainder. Conservative in that some paths will
      // exit early.
      cycles += 1;
      if (remainingElems & remainder) {
        // 2 cycles, 1 to convert and 1 to store assuming input is
        // already loaded in memory from vector path.
        cycles += 2;
      }
    }
  }
  return cycles;
}

std::uint64_t getCast2DCycleEstimate(const CastTargetParameters &targetParams,
                                     const Type &fromType, const Type &toType,
                                     std::vector<unsigned> &elemCounts) {
  std::uint64_t cycles = 5;
  for (unsigned i = 0; i != elemCounts.size(); ++i) {
    // Outer-loop cycles including call plus core function per-vector
    cycles +=
        11 + castWorkerCycles(elemCounts[i], fromType, toType, targetParams);
  }
  return cycles;
}

std::uint64_t
getCast1DSingleWorkerCycleEstimate(const CastTargetParameters &targetParams,
                                   const Type &fromType, const Type &toType,
                                   const unsigned numElems) {
  // Estimate written based on vertices with assembly implementations.
  // Not realistic for others.
  constexpr std::uint64_t getParamsCycles = 3;
  // Get parameters, call core function and exitz
  return getParamsCycles + 2 +
         castWorkerCycles(numElems, fromType, toType, targetParams);
}

std::uint64_t getCast1DCycleEstimate(const CastTargetParameters &targetParams,
                                     const Type &fromType, const Type &toType,
                                     const unsigned workerElems,
                                     const unsigned workerCount,
                                     const unsigned workerLast,
                                     const unsigned deltaLast) {

  // Work out workers doing unique workloads from the partitionParams and
  // find the maximum cycles for any of them.
  unsigned max = workerElems;
  unsigned maxM4 = workerElems - 4;
  unsigned numMaxWorkers = workerCount;
  unsigned numMaxM4Workers = targetParams.numWorkerContexts - workerCount;

  // Worker entry from the supervisor, including exitz and call.
  std::uint64_t maxCycles = 0;
  if (workerLast < workerCount) {
    numMaxWorkers--;
    maxCycles = std::max(maxCycles, castWorkerCycles(max - deltaLast, fromType,
                                                     toType, targetParams));
  } else {
    numMaxM4Workers--;
    maxCycles =
        std::max(maxCycles, castWorkerCycles(maxM4 - deltaLast, fromType,
                                             toType, targetParams));
  }
  if (numMaxWorkers) {
    maxCycles = std::max(maxCycles,
                         castWorkerCycles(max, fromType, toType, targetParams));
  }
  if (numMaxM4Workers) {
    maxCycles = std::max(
        maxCycles, castWorkerCycles(maxM4, fromType, toType, targetParams));
  }
  const std::uint64_t fromSupervisorWorkerCycles = 23;
  maxCycles += fromSupervisorWorkerCycles;

  // setzi, runall, sync, br
  // Assumes runall takes 6 cycles workers are balanced such that
  // sync takes 6 cycles and br takes 6 cycles.
  return 19 + targetParams.numWorkerContexts * maxCycles;
}

static std::uint64_t
_fillCycleEstimate(std::uint64_t size, const FillTargetParameters &targetParams,
                   const Type &type) {
  const bool is8Bits = type == BOOL || type == CHAR || type == UNSIGNED_CHAR ||
                       type == SIGNED_CHAR;
  const bool isHalf = type == HALF;
  const bool is64Bit = type == UNSIGNED_LONGLONG || type == LONGLONG;

  if (is64Bit) {
    return 5 + size;
  }

  const auto width = targetParams.dataPathWidth / (isHalf ? 16 : 32);

  if (is8Bits) {
    // Exact execution times for Fill vertices for 8 bits depend on memory byte
    // alignment of the tensor. For small sizes, this is a rough estimate.

    // For less than 64 bytes (16 words) we do st32, while for more we do st64
    if (size < 64) {
      return 28 + size / 4;
    } else {
      return 39 + size / 8;
    }
  }
  if (isHalf) {
    // Cycle breakdown:
    //
    //  In an eight byte interval there is one 8-byte-aligned address, two
    //  four-byte-aligned addresses and four two-byte-aligned addresses. So if
    //  the aligned addresses are chosen randomly, then on average
    //  two-byte-alignment will occur ~57% of the time, four-byte-alignement
    //  will occur ~29% of the time and eight byte alignment will occur ~14% of
    //  the time. So the return value should slightly bias towards
    //  2-byte-aligment.
    //
    //      size  | 2-byte-aligned | 4-byte-aligned | 8-byte-aligned | return
    //   ---------+----------------+----------------+----------------+---------
    //    2 bytes | 15             | 16             | 16             | 15
    //    4 bytes | 20             | 23             | 22             | 21
    //    8 bytes | 30             | 24             | 22             | 27
    //   16 bytes | 31             | 25             | 23             | 28
    switch (size) {
    case 1:
      return 15;
    case 2:
      return 21;
    default:
      return 26 + size / width;
    }
  }
  // Cycle breakdown:
  //
  // + 16 cycles for pre-loop code, such as loading data and checking alignment.
  // + 6 cycles for all the post-loop checks.
  // + 1 cycle if there are 4 bytes left after the loop.
  // + 1 cycle to on average account for 4 byte alignemnt rather than 8.
  //   There's an additional two cycles if the data is only 4 byte aligned
  //   rather than 8 byte aligned, and as 4 byte alignments are twice as likely
  //   to occur than 8 byte alignments this function could bias towards 4 byte
  //   alignments however in practise the cycle difference is so small, that the
  //   result of the bias is negligble and the returned cycles tend to the
  //   average.
  //
  // This ends up being roughly right in the small cases too:
  //
  //      size  | 4-byte-aligned | 8-byte-aligned | return
  //   ---------+----------------+----------------+----------
  //    4 bytes | 24             | 23             | 24
  //    8 bytes | 25             | 23             | 24
  //   16 bytes | 26             | 24             | 25
  return 16 + size / width + 6 + (size % width == 1) + 1;
}

std::uint64_t getFill1DCycleEstimate(const FillTargetParameters &targetParams,
                                     const Type &type,
                                     const unsigned numElems) {
  return _fillCycleEstimate(numElems, targetParams, type);
}

std::uint64_t getFill2DCycleEstimate(const FillTargetParameters &targetParams,
                                     const Type &type,
                                     const std::vector<unsigned> &numElems) {
  std::uint64_t cycles = 5;
  for (unsigned i = 0; i < numElems.size(); ++i)
    cycles += _fillCycleEstimate(numElems[i], targetParams, type);

  // 64-bit 2D doesn't share code with the other functions
  const auto is64Bit = type == UNSIGNED_LONGLONG || type == LONGLONG;
  if (is64Bit) {
    cycles -= 2 * numElems.size();
    return cycles;
  }

  // All cases other than 64-bit
  cycles += (type != HALF);

  // The 1d fill function includes overhead from loading variables which takes 5
  // cycles for half types and 6 cycles for other types, but the 2d fill
  // function has an additional per-loop overhead of 3 cycles, so subtract two
  // cycles for each call to fill to account for the difference (three cycles
  // for non-halves).
  cycles -= (2 + (type != HALF)) * numElems.size();
  return cycles;
}

std::uint64_t getScaledArithmeticSupervisorCycleEstimate(
    const ScaledArithmeticTargetParameters &targetParams, const Type &dataType,
    const Type &dataBType, const bool isConstant, const bool memConstrained,
    const ScaledArithmeticOp operation, const layout::Vector &aLayout,
    const layout::Vector &bLayout, const unsigned numElems) {
  if (dataType == INT || dataType == UNSIGNED_INT) {
    std::uint64_t supervisorCycles = 53 // constant overhead
                                     + (26 * (numElems / 3)); // main loop

    if (operation == ScaledArithmeticOp::SUBTRACT && !isConstant) {
      supervisorCycles += 1;
    }

    if (numElems % 3 == 0) {
      supervisorCycles += 6; // 6 cycle branch to skip the remainder loop
    } else {
      supervisorCycles += 6                        // --rem
                          + (26 * (numElems % 3)); // remainder loop
    }
    supervisorCycles += 8; // constant epilogue overhead.
    if (!isConstant) {
      supervisorCycles += 6;
    }
    return supervisorCycles;
  } else {
    assert(dataType == HALF || dataType == FLOAT);
  }

  // calculate count, rem and final
  const unsigned totalVectors = numElems / targetParams.vectorWidth;
  const unsigned remainingElems = numElems % targetParams.vectorWidth;

  const unsigned vectorsPerWorker =
      totalVectors / targetParams.numWorkerContexts;
  const unsigned remainingVectors =
      totalVectors % targetParams.numWorkerContexts;

  std::uint64_t perTypeSupervisorOverhead = 21;
  // scaled add and subtract for float and half maybe require an extra (bubble)
  // cycle to unpack the pointer.
  if (aLayout == layout::Vector::ScaledPtr64) {
    perTypeSupervisorOverhead += 6;
  }

  std::uint64_t supervisorCycles = perTypeSupervisorOverhead +
                                   basicOpSupervisorOverhead() +
                                   +(remainingElems == 0 ? 7 : 13) + 12;

  if (operation == ScaledArithmeticOp::AXPLUSBY && !isConstant) {
    supervisorCycles += 12 + poputil::internal::getUnpackCost(aLayout) +
                        poputil::internal::getUnpackCost(bLayout);
  }
  if (operation == ScaledArithmeticOp::SUBTRACT && !isConstant) {
    supervisorCycles += 7;
  }
  if (!isConstant) {
    // setzi + bri, but the branch skips a setzi already counted so just + 6.
    supervisorCycles += 6;
  }

  std::vector<unsigned> workerCycles;
  workerCycles.reserve(targetParams.numWorkerContexts);
  // Specific mixed precision half, float version
  if (dataType == HALF && dataBType == FLOAT) {
    const auto innerLoopCycles = 4;
    for (unsigned wid = 0; wid < targetParams.numWorkerContexts; ++wid) {
      std::uint64_t cycles = 16; // constant worker prologue cycles
      const auto numVectors = vectorsPerWorker + (wid < remainingVectors);
      if (numVectors != 0) {
        cycles += 8 + (innerLoopCycles * (numVectors - 1));
      }
      cycles += 2; // workerID == rem
      if (wid == remainingVectors) {
        cycles += 1; // final == 0?
        if (remainingElems != 0) {
          cycles += 2; // check if at least 2 remain
          if (remainingElems >= 2) {
            cycles += 5; // process 2 of the remainder.
          }
          cycles += 2; // check if 1 remains
          if (remainingElems % 2) {
            cycles += 7; // process final half
          }
        }
      }
      cycles += 1; // exitz
      workerCycles.push_back(cycles);
    }
  }
  // (half,half), (float, half) and (float, float) versions
  else {
    // half/float case handled above
    assert(dataType != HALF || dataBType != FLOAT);
    unsigned innerLoopCycles =
        memConstrained ? 2 : (dataType == dataBType ? 3 : 4);

    if (getForceInterleavedEstimates() && (dataType == dataBType)) {
      // Reduce inner loop cycles by one for (half,half), (float, float) when
      // using interleaved memory.
      innerLoopCycles -= 1;
    }

    for (unsigned wid = 0; wid < targetParams.numWorkerContexts; ++wid) {
      std::uint64_t cycles = 15; // constant worker prologue cycles
      const auto numVectors = vectorsPerWorker + (remainingVectors < 0);
      if (numVectors != 0) {
        cycles += 6 // inner loop constant overhead
                  + (innerLoopCycles * (numVectors - 1)); // loop cycles
      }
      cycles += 2; // workerID == rem
      if (wid == remainingVectors) {
        cycles += 1; // final == 0?
        if (remainingElems != 0) {
          if (dataType == FLOAT) {
            cycles += 8; // process final float.
          } else {
            cycles += 5; // unpack triPtr and check if at least 2 remain
            if (remainingElems >= 2) {
              cycles += 7; // process 2 of the remainder.
              if (remainingElems == 3) {
                cycles += 6; // process final half
              }
            }
          }
        }
      }
      cycles += 1; // exitz
      workerCycles.push_back(cycles);
    }
  }

  const auto maxWorkerCycles =
      *std::max_element(std::begin(workerCycles), std::end(workerCycles));
  return supervisorCycles + maxWorkerCycles * targetParams.numWorkerContexts;
}

} // namespace internal
} // namespace popops
