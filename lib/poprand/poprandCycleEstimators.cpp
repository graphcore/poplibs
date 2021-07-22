// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poprandCycleEstimators.hpp"
#include "poplibs_support/FlopEstimation.hpp"

using namespace poplar;
using namespace poplibs_support;

namespace poprand {

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Uniform)(const VertexIntrospector &vertex,
                                  const Target &target, const Type &type) {
  uint64_t cycles = 4; // supervisor call
  CODELET_FIELD(out);
  if (type == INT) {
    CODELET_SCALAR_VAL(scale, float);

    if (scale == 0) {
      // prep-exit cycles/thread
      cycles += 22 * target.getNumWorkerContexts();
      // rpt loop cycles for every st64
      cycles += out.size() / 2;
    } else {
      // prep cycles per thread
      cycles += 22 * target.getNumWorkerContexts();
      // rpt loop cycles per element
      cycles += 6 * (out.size() / 2);
      cycles += (out.size() % 2) * 2;
    }
  } else {
    const auto dataPathWidth = target.getDataPathWidth();

    bool isFloat = type == FLOAT;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 2 : 4);

    cycles += 4 * (out.size() + vectorWidth - 1) / vectorWidth;
    if (isFloat) {
      cycles += 22 * target.getNumWorkerContexts();
      cycles += ((out.size() % vectorWidth) != 0) * 4;
    } else {
      cycles += 24 * target.getNumWorkerContexts();
      cycles += ((out.size() % vectorWidth) != 0) * 12;
    }
  }
  // Flops for creating a random number is arbitrarily assumed to be 1
  std::uint64_t flops = static_cast<std::uint64_t>(out.size()) *
                        (1 + flopsForMultiply() + flopsForAdd());
  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Bernoulli)(const VertexIntrospector &vertex,
                                    const Target &target, const Type &type) {
  CODELET_FIELD(out);
  const auto dataPathWidth = target.getDataPathWidth();

  bool isHalf = type == HALF;
  unsigned vectorWidth = dataPathWidth / (isHalf ? 4 : 2);

  uint64_t cycles = 4; // supervisor overhead (Per tile?)

  cycles += (out.size() + vectorWidth - 1) / vectorWidth;
  if (isHalf) {
    cycles += 21 * target.getNumWorkerContexts();
    cycles += ((out.size() % vectorWidth) != 0) * 9;
  } else {
    cycles += 23 * target.getNumWorkerContexts();
    cycles += ((out.size() % vectorWidth) != 0);
  }
  return {cycles, convertToTypeFlops(out.size(), type)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Normal)(const VertexIntrospector &vertex,
                                 const Target &target, const Type &type) {
  CODELET_FIELD(out);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 4; // Supervisor overhead
  bool isFloat = type == FLOAT;
  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

  cycles += 4 * (out.size() + vectorWidth - 1) / vectorWidth;
  if (isFloat) {
    cycles += 22 * target.getNumWorkerContexts();
    cycles += ((out.size() % vectorWidth) != 0) * 3;
  } else {
    cycles += 24 * target.getNumWorkerContexts();
    cycles += ((out.size() % vectorWidth) != 0) * 12;
  }
  // Flops for creating a random number is arbitrarily assumed to be 1
  std::uint64_t flops = static_cast<std::uint64_t>(out.size()) *
                        (1 + flopsForMultiply() + flopsForAdd());
  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(TruncatedNormal)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(iterations, unsigned);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 4; // supervisor overhead
  bool isFloat = type == FLOAT;
  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

  if (isFloat) {
    cycles += 26 * target.getNumWorkerContexts();
    cycles +=
        (11 * iterations + 5) * (out.size() + vectorWidth - 1) / vectorWidth;
    cycles += (out.size() % vectorWidth != 0) * (13 * iterations + 2);
  } else {
    cycles += 28 * target.getNumWorkerContexts();
    cycles +=
        (12 * iterations + 5) * (out.size() + vectorWidth - 1) / vectorWidth;
    cycles += (out.size() % vectorWidth != 0) * (12 * iterations + 10);
  }
  // Flops for creating a random number is arbitrarily assumed to be 1
  std::uint64_t flops = static_cast<std::uint64_t>(out.size()) *
                        (1 + flopsForMultiply() + flopsForAdd());
  return {cycles, flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Dropout)(const VertexIntrospector &vertex,
                                  const Target &target, const Type &type) {
  CODELET_FIELD(out);

  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 6; // supervisor overhead
  bool isFloat = type == FLOAT;
  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

  cycles += 24 * target.getNumWorkerContexts();
  cycles += 2 * (out.size() + vectorWidth - 1) / vectorWidth;
  cycles += (out.size() % vectorWidth != 0) * (isFloat ? 1 : 9);
  // Flops for creating a random number is arbitrarily assumed to be 1
  std::uint64_t flops =
      static_cast<std::uint64_t>(out.size()) * (1 + flopsForMultiply());
  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(SetSeed)(const VertexIntrospector &vertex,
                                  const Target &target) {
  return 14 + 27 * target.getNumWorkerContexts();
}

poputil::internal::PerfEstimatorTable makePerfFunctionTable() {
  return {
      CYCLE_ESTIMATOR_ENTRY(poprand, TruncatedNormal, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, TruncatedNormal, HALF),

      CYCLE_ESTIMATOR_ENTRY(poprand, Normal, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, Normal, HALF),

      CYCLE_ESTIMATOR_ENTRY(poprand, Bernoulli, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, Bernoulli, HALF),
      CYCLE_ESTIMATOR_ENTRY(poprand, Bernoulli, INT),

      CYCLE_ESTIMATOR_ENTRY(poprand, Uniform, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, Uniform, HALF),
      CYCLE_ESTIMATOR_ENTRY(poprand, Uniform, INT),

      CYCLE_ESTIMATOR_ENTRY(poprand, Dropout, HALF),
      CYCLE_ESTIMATOR_ENTRY(poprand, Dropout, FLOAT),

      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(poprand, SetSeed),
  };
}

} // end namespace poprand
