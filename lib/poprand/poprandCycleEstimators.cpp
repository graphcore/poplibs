#include "poprandCycleEstimators.hpp"

using namespace poplar;

namespace poprand {

constexpr unsigned WARMUP_ITERATIONS = 4;

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Uniform)(const VertexIntrospector &vertex,
                                   const Target &target,
                                   const Type &type) {
  if (type == INT) {
    CODELET_FIELD(out);
    CODELET_SCALAR_VAL(saveRestoreSeed, bool);

    uint64_t cycles = 5;  // overhead + broadcast offset
    if (saveRestoreSeed) {
      cycles += 5;        // to set up seeds in CSR
      cycles += WARMUP_ITERATIONS;
    }
    for (auto i = 0u; i != out.size(); ++i) {
      // use modulo instruction
      cycles += out[i].size() * 23;
    }
    if ((saveRestoreSeed)) {
      // save seeds
      cycles += 6;
    }
    return cycles;
  } else {
    CODELET_FIELD(out);
    CODELET_SCALAR_VAL(saveRestoreSeed, bool);
    const auto dataPathWidth = target.getDataPathWidth();

    uint64_t cycles = 7;  // overhead + broadcast offset
    if (saveRestoreSeed) {
      cycles += 5;        // to set up seeds in CSR
      cycles += WARMUP_ITERATIONS;
    }
    bool isFloat = type == FLOAT;
    unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

    for (auto i = 0u; i != out.size(); ++i) {
      cycles += 3; // overhead to load pointers + rpt + brnzdec
      // rand gen/convert/axpb
      cycles += (out[i].size() + vectorWidth - 1) / vectorWidth * 3;
    }
    if ((saveRestoreSeed)) {
      // save seeds
      cycles += 6;
    }
    return cycles;
  }
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Bernoulli)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(saveRestoreSeed, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 7;  // overhead to form and broadcast 1.0. float/int
                        // should take less
  if (saveRestoreSeed) {
    cycles += 5;          // to set up seeds in CSR
    cycles += WARMUP_ITERATIONS;
  }
  bool isFloat = type == FLOAT;
  unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

  for (auto i = 0u; i != out.size(); ++i) {
    cycles += 3; // overhead to load pointers + rpt + brnzdec
    // use f16v4rmask for half and f32v2mask for int/float + store64
    // assumption that rmask ignores NaNs (as it seems from archman)
    cycles += (out[i].size() + vectorWidth - 1) / vectorWidth * 1;
  }
  if (saveRestoreSeed) {
    // save seeds
    cycles += 6;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Normal)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(saveRestoreSeed, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 7;  // overhead to store stdDev into CSR. broadcast mean
  if (saveRestoreSeed) {
    cycles += 5;        // to set up seeds in CSR
    cycles += WARMUP_ITERATIONS;
  }
  bool isFloat = type == FLOAT;
  unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

  for (auto i = 0u; i != out.size(); ++i) {
    cycles += 3; // overhead to load pointers + rpt + brnzdec
    // use f16v4grand for half and f32v2grand for int/float + store64
    // and axpby
    cycles += (out[i].size() + vectorWidth - 1) / vectorWidth * 2;
  }
  if (saveRestoreSeed) {
    // save seeds
    cycles += 6;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(TruncatedNormal)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &type) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(iterations, unsigned);
  CODELET_SCALAR_VAL(saveRestoreSeed, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 8;  // overhead to store stdDev into CSR. broadcast mean
                        // store constants in stack
  if (saveRestoreSeed) {
    cycles += 5;          // to set up seeds in CSR
    cycles += WARMUP_ITERATIONS;
  }
  bool isFloat = type == FLOAT;
  unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

  for (auto i = 0u; i != out.size(); ++i) {
    cycles += 3; // overhead to load pointer + brnzdec + init mask
    // 6 cycles per iter + axpby + store + 5 (for triangular/uniform)
    cycles += (out[i].size() + vectorWidth - 1)
              / vectorWidth * ( 6 * iterations + 6);
  }
  if (saveRestoreSeed) {
    // save seeds
    cycles += 6;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DropoutSupervisor)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             const Type &type) {
  CODELET_SCALAR_VAL(numElems, unsigned);
  const auto dataPathWidth = target.getDataPathWidth();
  unsigned vectorWidth = target.getVectorWidth(type);
  unsigned numContexts = target.getNumWorkerContexts();
  unsigned numVectors = (numElems + vectorWidth - 1) / vectorWidth;
  unsigned numVectorsPerContext = (numVectors + numContexts - 1) / numContexts;
  std::uint64_t cycles = numContexts * numVectorsPerContext * 2 + 40;
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(SetSeedSupervisor)
  (const VertexIntrospector &vertex,
   const Target &target) {
  return 14 + 22 * target.getNumWorkerContexts();
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(GetSeedsSupervisor)
  (const VertexIntrospector &vertex,
   const Target &target) {
  return 14 + 7 * target.getNumWorkerContexts();
}


poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
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

    CYCLE_ESTIMATOR_ENTRY(poprand, DropoutSupervisor, HALF),
    CYCLE_ESTIMATOR_ENTRY(poprand, DropoutSupervisor, FLOAT),

    CYCLE_ESTIMATOR_ENTRY(poprand, SetSeedSupervisor),
    CYCLE_ESTIMATOR_ENTRY(poprand, GetSeedsSupervisor),

  };
};

} // end namespace poprand
