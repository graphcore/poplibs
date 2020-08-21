// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <cstdint>

static inline std::uint64_t histogramSupervisorByLimitEstimate(
    unsigned elements, unsigned histogramCount, bool isAbsolute, bool isHalf,
    unsigned numWorkers, unsigned vectorWidth, unsigned unpackCostHistogram = 0,
    unsigned unpackCostLimits = 0) {
  const auto remainder = ((histogramCount - 1) % numWorkers) != 0;
  const auto maxLimits = remainder + (histogramCount - 1) / numWorkers;
  uint64_t workerCycles = 19 + unpackCostHistogram + unpackCostLimits;
  if (isHalf) {
    workerCycles += 3; // Pre-loop overhead
    uint64_t dataLoopCycles = 17;

    if (elements & 1) {
      dataLoopCycles += 3 + isAbsolute;
    }
    if (elements & 2) {
      dataLoopCycles += 3 + isAbsolute;
    }
    dataLoopCycles += (3 + isAbsolute) * (elements / vectorWidth);

    workerCycles += dataLoopCycles * maxLimits;
  } else {
    workerCycles += 3; // Pre-loop overhead
    uint64_t dataLoopCycles = 11;

    if (elements & 1) {
      dataLoopCycles += 2 + isAbsolute;
    }
    dataLoopCycles += (3 + isAbsolute) * (elements / vectorWidth);

    workerCycles += dataLoopCycles * maxLimits;
  }
  // post process
  workerCycles += 8 + (histogramCount - 1) * 2;

  return workerCycles * numWorkers;
}

static inline std::uint64_t histogramSupervisorByDataEstimate(
    unsigned elements, unsigned histogramCount, bool isAbsolute, bool isHalf,
    unsigned numWorkers, unsigned vectorWidth, unsigned unpackCostHistogram = 0,
    unsigned unpackCostLimits = 0) {

  const auto vectors = elements / vectorWidth;
  const auto workerVectors = (vectors + numWorkers - 1) / numWorkers;
  uint64_t workerCycles = 23 + unpackCostHistogram + unpackCostLimits;
  if (isHalf) {
    workerCycles += 5; // Pre-loop overhead
    uint64_t dataLoopCycles = 19;

    if (elements & 1) {
      dataLoopCycles += 3 + isAbsolute;
    }
    if (elements & 2) {
      dataLoopCycles += 3 + isAbsolute;
    }
    dataLoopCycles += (3 + isAbsolute) * workerVectors;

    workerCycles += dataLoopCycles * (histogramCount - 1);
  } else {
    workerCycles += 4; // Pre-loop overhead
    uint64_t dataLoopCycles = 17;

    if (elements & 1) {
      dataLoopCycles += 3 + isAbsolute;
    }
    dataLoopCycles += (3 + isAbsolute) * workerVectors;

    workerCycles += dataLoopCycles * (histogramCount - 1);
  }
  // post process (worker 0)
  workerCycles += 10;

  return workerCycles * numWorkers;
}
#endif // _performance_estimation_h_
