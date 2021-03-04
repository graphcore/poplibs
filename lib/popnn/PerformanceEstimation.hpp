// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearity.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <poplibs_support/FlopEstimation.hpp>

#include <algorithm>
#include <boost/optional.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

inline uint64_t getNonLinearityOpCycles(popnn::NonLinearityType nlType,
                                        bool isFloat) {
  // Based off the worst-case cycles from arch_man for float/half
  // transcendental ops.
  uint64_t opCycles;
  switch (nlType) {
  case popnn::NonLinearityType::RELU:
    opCycles = 1;
    break;
  case popnn::NonLinearityType::SIGMOID:
    opCycles = (isFloat ? 5 : 2);
    break;
  case popnn::NonLinearityType::TANH:
    opCycles = (isFloat ? 5 : 1);
    break;
  case popnn::NonLinearityType::GELU:
    // TODO: T12914 These are just placeholders. Change these when the
    // nonlinearity is coded in assembly.
    opCycles = isFloat ? 10 : 5;
    break;
  default:
    throw poputil::poplibs_error("Unhandled non-linearity type");
    break;
  }
  return opCycles;
}

inline uint64_t getLossTransformCycles(const bool isFloat, const bool isSoftmax,
                                       const std::size_t size) {
  uint64_t cycles =
      5                     // vertex overhead;
      + (isSoftmax ? 6 : 5) // loads of pointers
      + 5                   // get base and pointer shifts
      + (isFloat ? 0 : 1)   // shift size for halves
      + 2                   // 2 load aheads
      + 1                   // repeat instruction
      + (isSoftmax ? 9 : 4) * (isFloat ? size : size / 2)             // loop
      + (isFloat ? 0 : (2 + (size & 0x1 ? (isSoftmax ? 11 : 6) : 0))) // RMW
      + 1; // exit instruction
  return cycles;
}
struct UnpackCosts {
  unsigned outLayout;
  unsigned inLayout;
  unsigned fwdInLayout;
  unsigned fwdOutLayout;
  unsigned startPosLayout;
  unsigned workListLayout;
};

// Return cycles per vector, used by the planner
inline std::uint64_t poolVertexCyclesPerVector(bool isMaxPool, bool isBwdPass) {
  if (isMaxPool && isBwdPass) {
    return 5;
  }
  return 3;
}
// Overhead for the big row loop, used by the planner
inline std::uint64_t poolVertexCyclesPerRow(void) { return 30; }

inline uint64_t getPoolingCycles(
    const unsigned initInfo, const unsigned chansPerGroupD,
    const unsigned numChanGroupsM1, const std::vector<unsigned short> &startPos,
    const std::vector<std::vector<unsigned short>> &workList,
    const boost::optional<UnpackCosts> &unpackCosts_, const bool isMaxPool,
    const bool isBwdPass, const unsigned numWorkers,
    const bool planningEstimates = false) {

  // For planning we have a "worklist" that only contains sizes.
  const unsigned itemsPerWorklistEntry = planningEstimates ? 1 : 3;
  const unsigned workListLengthItemIndex = planningEstimates ? 0 : 2;
  // Unpack costs for use when planning - passed in for profiling
  const UnpackCosts defaultUnpackCosts = {2, 2, 2, 2, 2, 2};
  const UnpackCosts &unpackCosts =
      unpackCosts_ ? unpackCosts_.get() : defaultUnpackCosts;
  // per-worker cycles
  const auto workerCycles = [&](unsigned wId) {
    std::uint64_t cycles = 4    // load vertex state
                           + 2  // scale initInfo
                           + 2  // get $WSR and load identity
                           + 7  // divide init work
                           + 2; // Fetch and shift vertex slice size for init
    // maybe unpack outPtrPtr
    cycles += unpackCosts.outLayout;

    // calculate how much initialisation each worker does.
    const auto initElems = [&] {
      const unsigned numElems = initInfo * chansPerGroupD / numWorkers;
      const unsigned extra = wId < (initInfo - numElems * numWorkers);

      return (numElems + extra);
    }();

    // init loop overhead, number of rpt loop cycles, number of brnzdec cycles.
    cycles += (4 + initElems) * (numChanGroupsM1 + 1);

    cycles += 5    // load startPosPtr, numRows and startPos
              + 1; // bnz numRows

    // maybe unpack startPosPtr
    cycles += unpackCosts.startPosLayout;

    // if numRows is zero this worker is done.
    const unsigned numRows =
        wId == 0 ? startPos[0] : startPos[wId] - startPos[wId - 1];
    if (numRows == 0) {
      return cycles + 1; // exitz
    }

    cycles += 2 // save startPos, load inPtrPtr and workListBase
              + (isMaxPool ? 1 : 2); // unpack inPtrPtr, maybe load scale

    // load and (possibly) unpack acts pointer pointers
    if (isBwdPass) {
      cycles += 6 + unpackCosts.outLayout + unpackCosts.inLayout;
    }

    cycles += 3 // unpack workListBase, store it
              + 2 + unpackCosts.workListLayout * 2 // Unpack worklist ptr
              + 1;                                 // decrement numRows

    for (unsigned row = 0; row < numRows; ++row) {
      cycles += 13; // row_loop overhead

      const unsigned sPos = wId == 0 ? 0 : startPos[wId - 1];
      const unsigned numWorkItems = workList[sPos + row].size();
      for (unsigned w = 0; w < numWorkItems; w += itemsPerWorklistEntry) {
        cycles += 20; // work_loop overhead
        for (unsigned cg = 0; cg < numChanGroupsM1 + 1u; ++cg) {
          if (cg != 0) {   // Pointer offsets avoided on the 1st pass
            cycles += 2    // load inSliceSize, outSliceSize
                      + 2; // move pointers on by inSliceSize, outSliceSize

            if (isBwdPass) {
              cycles += 2; // move pointers on by inSliceSize, outSliceSize
            }
          }
          cycles += 1; // reload chansPerGroupD

          for (unsigned c = 0; c < chansPerGroupD; ++c) {
            // rpt loop cycles.
            const auto rptCycles = [&] {
              // numElementsM1, aka the rpt count
              const unsigned n =
                  workList[sPos + row][w + workListLengthItemIndex];

              if (isBwdPass) {
                return 7 + 5 * n;
              } else if (isMaxPool) {
                return 4 + 3 * n;
              } else {
                return 5 + 3 * n;
              }
            }();

            cycles += 2           // chans_per_group_loop overhead
                      + rptCycles // innermost loop
                      + 1;        // brnzdec chansPerGroupD
          }
          ++cycles; // brnzdec numChanGroupsM1
        }
        cycles += 3; // reload, decrement and brnz numWorkItems
      }
      cycles += 2; // reload numRows and brnzdec
    }
    return cycles + 1; // exitz
  };

  // calculate how long each worker take
  std::vector<std::uint64_t> allWorkerCycles;
  for (unsigned wId = 0; wId < numWorkers; ++wId) {
    allWorkerCycles.push_back(workerCycles(wId));
  }

  return 7                                          // supervisor overhead
         + *boost::max_element(allWorkerCycles) * 6 // longest worker
         + 6                                        // br $lr
      ;
}

// Cycles to calculate x + y in the log domain;
// where a = max(x, y)
//       b = min(x, y)
// x + y => a + log(1 + exp(b - a))
inline uint64_t logAddCycles() {
  return 4    // min/max assign x,y to a,b
         + 1  // b - a
         + 3  // exp(ans)
         + 1  // 1 + ans
         + 6  // log(ans)
         + 1; // a + ans
}

// Cycles to calculate x * y in the log domain;
// x * y => x + y
inline uint64_t logMulCycles() {
  return 1; // x + y
}

/*
We make the assumption that the label rarely has duplicate adjacent symbols
(e.g. "abcdabc" has no adjacent duplicates and is more representative than
"aabbaa"). Therefore we model the number of paths to a non-blank class in an
extended label as 3.

   t-1    t
(-) X--\  X
        \
         \
(a) X-----X


(-) X--\  X
        \
         \  We can't jump from a to a (duplicate character).
(a) X-----X And we expect this won't happen very often, so we don't
     \      consider it occuring in the estimates.
      \
(-) X--\  X
        \
         \
(b) X-----X We can jump from a to b

  figure 1: Valid transitions to non blank class nodes at t

For transitions to the blank class at t, the number of pathways is always 2
(can't skip over a non-blank class). Since there's approximately the same number
of blank to non-blank in the extended label (between each non-blank class is a
blank, plus blank at start and end).

To calculate maximum number of add/mul for alpha/beta, consider the following
two paths in figure 2:

    t-1    t
(-)  X--\  X
         \
          \
(a)  X~~~~~X

  figure 2: Calculating maximum number of add/mul required for alpha/beta

For each path, we need to multiply the alpha at that node (at time t-1), with
the probability of the next class (at time t). Then alpha at time t is the sum
of the results for all the paths.

For the straight lined path from (-) to (a), we multiply alpha of the input node
(-) at time t-1 with the probability of (a) at time t. And similarly for the
wiggly lined path from (a) to (a), it's the alpha at the input node (a)
multiplied by probability of (a) at time t. These two results (arriving at
output (a) via straight and wiggly lines) are summed together to produce alpha
at the output node (a) at time t.

This can be expressed by the following equation for output node (a) at time t:

Where:
  alpha[a](t) => Alpha at node 'a' at time 't'
  P[a](t) => Probability of class 'a' at time 't'

  equation 1: alpha[a](t) = alpha[-](t-1)*P[a](t) + alpha[a](t-1)*P[a](t)

Which is equivalent to:

  equation 2: alpha[a](t) = P[a](t)*(alpha[-](t-1) + alpha[a](t-1))

From the equation 2, it's apparent that we need 1 add and a multiply operation,
and it follows for three paths we need 2 add operations and a multiply. So for a
given class in l (and including one adjacent blank), the number of operations
required is: 3 add, 2 multiply (2 add + 1 mul for non-blank class, 1 add + 1 mul
for blank class). However the partial result calculating the previous blank
symbol can be reused if the previous symbol is different can be reused to
require only 2 add and 2 multiply operations.

We have illustrated computing alpha here, however it's equivalent costing for
beta as it is just the same operation with the data reversed.

For costing the gradient we have two vertices, gradGivenAlpha and gradGivenBeta.
The equation for gradient is the following:

  equation 3: grad = (alpha * beta) / prob

By considering gradGivenBeta, we can avoid the division of `prob` in equation 3.
This is by:

  1.  Do the sum part of the alpha operation
  2.  grad = grad + (sum * beta)
  3.  alpha = sum * prob

*/
inline uint64_t alphaFlops(unsigned t, unsigned l, const poplar::Type &type,
                           bool extraBlank) {
  auto flopsPerInputElement = 2 * poplibs_support::flopsForLogAdd() +
                              2 * poplibs_support::flopsForLogMultiply();
  auto flopsLastBlank = poplibs_support::flopsForLogAdd() +
                        poplibs_support::flopsForLogMultiply();
  return poplibs_support::convertToTypeFlops(
      flopsPerInputElement * t * l + (extraBlank ? flopsLastBlank : 0), type);
}

inline uint64_t betaFlops(unsigned t, unsigned l, const poplar::Type &type,
                          bool extraBlank) {
  return alphaFlops(t, l, type, extraBlank);
}

inline uint64_t gradGivenAlphaFlops(unsigned t, unsigned l,
                                    const poplar::Type &type, bool extraBlank) {
  auto gradFlopsPerInputElement = 2 * (poplibs_support::flopsForLogMultiply() +
                                       poplibs_support::flopsForLogAdd());
  auto gradFlopsLastBlank = 2 * (poplibs_support::flopsForLogAdd() +
                                 poplibs_support::flopsForLogMultiply());
  auto flops = betaFlops(t, l, type, extraBlank) +
               (t * l * gradFlopsPerInputElement +
                (extraBlank ? gradFlopsLastBlank : 0));
  return poplibs_support::convertToTypeFlops(flops, type);
}

inline uint64_t gradGivenBetaFlops(unsigned t, unsigned l,
                                   const poplar::Type &type, bool extraBlank) {
  return gradGivenAlphaFlops(t, l, type, extraBlank);
}

// Estimated cycles for a given region of size {t, l}
inline uint64_t alphaCycles(unsigned t, unsigned l, bool extraBlank) {
  auto readWriteCost =
      (1    // -> Writing to output
       + 3) // -> Reading from 3 inputs (previous non-blank in label, in
            // addition to current blank and non-blank at previous t)
      * 2;  // For both blank and non-blank class
  auto arithmeticCost = (2 * logAddCycles() + 2 * logMulCycles()); // Each path
  auto controlFlowCost = 2; // Conditional for duplicate non-blank class

  auto cyclesPerSymbolAndAdjacentBlankPerT =
      readWriteCost + arithmeticCost + controlFlowCost;

  auto cycles = cyclesPerSymbolAndAdjacentBlankPerT * l * t;
  if (extraBlank) {
    cycles += logAddCycles() + logMulCycles();
  }
  return cycles;
}

// Beta is approximately the same as alpha in the inverse direction, so
// estimates are equivalent
inline uint64_t betaCycles(unsigned t, unsigned l, bool extraBlank) {
  return alphaCycles(t, l, extraBlank);
}

// For gradGiven[Alpha/Beta], this calculates the compliment [Beta/Alpha], and
// with the working state, combines them to compute the gradient
inline uint64_t gradGivenAlphaCycles(unsigned t, unsigned l, bool extraBlank) {
  // This is based on beta vertex, but does additional work
  auto baseBetaCalculationCost = betaCycles(t, l, extraBlank);

  // The following estimates are calculating grad, given beta has been
  // calculated (and alpha in a previous step)
  auto readWriteCost =
      (2    // -> Writing to output (indirection to classes from extened label)
       + 1) // -> Read previous grad
      * 2;  // For both blank and non-blank class
  auto arithmeticCost =
      (logMulCycles() +
       logAddCycles()) // multiply alpha & beta, and add to previous grad
      * 2;             // For both blank and non-blank class

  auto cyclesPerSymbolAndAdjacentBlankPerT = readWriteCost + arithmeticCost;
  auto cycles =
      baseBetaCalculationCost + cyclesPerSymbolAndAdjacentBlankPerT * l * t;
  if (extraBlank) {
    cycles += logAddCycles() + logMulCycles();
  }
  return cycles;
}

// Similarly, gradGivenBeta is approximately the same as gradGivenAlpha, so
// estimates are equivalent
inline uint64_t gradGivenBetaCycles(unsigned t, unsigned l, bool extraBlank) {
  return gradGivenAlphaCycles(t, l, extraBlank);
}

#endif // _performance_estimation_h_
