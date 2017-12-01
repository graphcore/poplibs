#include "poprandCycleEstimators.hpp"
#include <poplar/HalfFloat.hpp>

using namespace poplar;

namespace poprand {

constexpr unsigned WARMUP_ITERATIONS = 4;

template <class OutType>
MAKE_CYCLE_ESTIMATOR(Uniform, vertex, target) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(saveRestoreSeed, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 7;  // overhead + broadcast offset
  if (saveRestoreSeed) {
    cycles += 5;        // to set up seeds in CSR
    cycles += WARMUP_ITERATIONS;
  }
  bool isFloat = std::is_same<OutType, float>::value;
  unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

  for (auto i = 0; i != out.size(); ++i) {
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

// NB: Specialising template functions is usually considered bad!
// In this case the usage and is so restricted that it doesn't matter.
template <>
MAKE_CYCLE_ESTIMATOR(Uniform<int>, vertex, target) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(saveRestoreSeed, bool);

  uint64_t cycles = 5;  // overhead + broadcast offset
  if (saveRestoreSeed) {
    cycles += 5;        // to set up seeds in CSR
    cycles += WARMUP_ITERATIONS;
  }
  for (auto i = 0; i != out.size(); ++i) {
    // use modulo instruction
    cycles += out[i].size() * 23;
  }
  if ((saveRestoreSeed)) {
    // save seeds
    cycles += 6;
  }
  return cycles;
}

template <class OutType>
MAKE_CYCLE_ESTIMATOR(Bernoulli, vertex, target) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(saveRestoreSeed, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 7;  // overhead to form and broadcast 1.0. float/int
                        // should take less
  if (saveRestoreSeed) {
    cycles += 5;          // to set up seeds in CSR
    cycles += WARMUP_ITERATIONS;
  }
  bool isFloat = std::is_same<OutType, float>::value;
  unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

  for (auto i = 0; i != out.size(); ++i) {
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

template <class OutType>
MAKE_CYCLE_ESTIMATOR(Normal, vertex, target) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(saveRestoreSeed, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  uint64_t cycles = 7;  // overhead to store stdDev into CSR. broadcast mean
  if (saveRestoreSeed) {
    cycles += 5;        // to set up seeds in CSR
    cycles += WARMUP_ITERATIONS;
  }
  bool isFloat = std::is_same<OutType, float>::value;
  unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

  for (auto i = 0; i != out.size(); ++i) {
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

template <class OutType>
MAKE_CYCLE_ESTIMATOR(TruncatedNormal, vertex, target) {
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
  bool isFloat = std::is_same<OutType, float>::value;
  unsigned vectorWidth =  dataPathWidth / (isFloat ? 32 : 16);

  for (auto i = 0; i != out.size(); ++i) {
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

poplibs::CycleEstimatorTable cyclesFunctionTable = {
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, TruncatedNormal, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, TruncatedNormal, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Normal, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Normal, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Bernoulli, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Bernoulli, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Bernoulli, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Uniform, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Uniform, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(poprand, Uniform, int)
};

} // end namespace poprand
