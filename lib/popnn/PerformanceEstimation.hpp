// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearity.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"

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
  unsigned outInnerLayout;
  unsigned inLayout;
  unsigned inInnerLayout;
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
inline std::uint64_t poolVertexCyclesPerRow(void) { return 54; }
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
  const UnpackCosts defaultUnpackCosts = {2, 1, 2, 1, 2, 2, 2, 2};
  const UnpackCosts &unpackCosts =
      unpackCosts_ ? unpackCosts_.get() : defaultUnpackCosts;
  // per-worker cycles
  const auto workerCycles = [&](unsigned wId) {
    std::uint64_t cycles = 4   // load vertex state
                           + 1 // scale initInfo
                           + 2 // get $WSR and load identity
                           + 7 // divide init work
        ;
    // maybe unpack outPtrPtr
    cycles += unpackCosts.outLayout;

    // calculate how much initialisation each worker does.
    const auto initElems = [&] {
      const unsigned numElems = initInfo * chansPerGroupD;
      const unsigned extra = wId < (initInfo - numElems * numWorkers);

      return (numElems + extra) * 8;
    }();
    // init loop overhead, number of rpt loop cycles, number of brnzdec cycles.
    cycles += (2 + initElems) * numChanGroupsM1;

    cycles += 5   // load startPosPtr, numRows and startPos
              + 1 // bnz numRows
        ;

    // maybe unpack outPtr and startPosPtr
    cycles += unpackCosts.outInnerLayout;
    cycles += unpackCosts.startPosLayout;

    // if numRows is zero this worker is done.
    const unsigned numRows =
        wId == 0 ? startPos[0] : startPos[wId] - startPos[wId - 1];
    if (numRows == 0) {
      return cycles + 1; // exitz
    }

    cycles += 2 // save startPos, load inPtrPtr and workListBase
              + (isMaxPool ? 1 : 2) // unpack inPtrPtr, maybe load scale
              + unpackCosts.inInnerLayout;

    // load and (possibly) unpack acts pointer pointers
    if (isBwdPass) {
      cycles += 6 + unpackCosts.outLayout + unpackCosts.inLayout;
    }

    cycles += 2   // unpack workListBase
              + 1 // decrement numRows
        ;

    for (unsigned row = 0; row < numRows; ++row) {
      cycles += 13 + unpackCosts.workListLayout; // row_loop overhead

      const unsigned sPos = wId == 0 ? 0 : startPos[wId - 1];
      const unsigned numWorkItems = workList[sPos + row].size();
      for (unsigned w = 0; w < numWorkItems; w += itemsPerWorklistEntry) {
        cycles += 20; // work_loop overhead
        for (unsigned cg = 0; cg < numChanGroupsM1 + 1u; ++cg) {
          cycles += 2 // reload outPos and inPos
                    + unpackCosts.outLayout + unpackCosts.inLayout +
                    2   // reload chansPerGroupD, decrement it
                    + 4 // move pointers on by outPos and inPos
              ;

          if (isBwdPass) {
            cycles += unpackCosts.outInnerLayout + unpackCosts.inInnerLayout +
                      unpackCosts.fwdInLayout + unpackCosts.fwdOutLayout +
                      4 // move pointers on by outPos and inPos
                ;
          }

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
                      + 1         // brnzdec chansPerGroupD
                ;
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
#endif // _performance_estimation_h_
