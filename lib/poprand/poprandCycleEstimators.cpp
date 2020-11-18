// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poprandCycleEstimators.hpp"

using namespace poplar;

namespace poprand {

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(UniformSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  uint64_t cycles = 4; // supervisor call
  if (type == INT) {
    CODELET_FIELD(out);
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
    CODELET_FIELD(out);
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
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(BernoulliSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
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

  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(NormalSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
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

  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(TruncatedNormalSupervisor)(
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
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(DropoutSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(out);

  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 6; // supervisor overhead
  bool isFloat = type == FLOAT;
  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

  cycles += 24 * target.getNumWorkerContexts();
  cycles += 2 * (out.size() + vectorWidth - 1) / vectorWidth;
  cycles += (out.size() % vectorWidth != 0) * (isFloat ? 1 : 9);

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(SetSeedSupervisor)(const VertexIntrospector &vertex,
                                             const Target &target) {
  return 14 + 27 * target.getNumWorkerContexts();
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return {
      CYCLE_ESTIMATOR_ENTRY(poprand, TruncatedNormalSupervisor, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, TruncatedNormalSupervisor, HALF),

      CYCLE_ESTIMATOR_ENTRY(poprand, NormalSupervisor, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, NormalSupervisor, HALF),

      CYCLE_ESTIMATOR_ENTRY(poprand, BernoulliSupervisor, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, BernoulliSupervisor, HALF),
      CYCLE_ESTIMATOR_ENTRY(poprand, BernoulliSupervisor, INT),

      CYCLE_ESTIMATOR_ENTRY(poprand, UniformSupervisor, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poprand, UniformSupervisor, HALF),
      CYCLE_ESTIMATOR_ENTRY(poprand, UniformSupervisor, INT),

      CYCLE_ESTIMATOR_ENTRY(poprand, DropoutSupervisor, HALF),
      CYCLE_ESTIMATOR_ENTRY(poprand, DropoutSupervisor, FLOAT),

      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(poprand, SetSeedSupervisor),
  };
}

} // end namespace poprand
