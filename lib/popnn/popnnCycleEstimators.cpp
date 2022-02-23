// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popnnCycleEstimators.hpp"

#include "PerformanceEstimation.hpp"
#include "PoolingDefUtil.hpp"
#include "poplibs_support/FlopEstimation.hpp"
#include "popnn/NonLinearity.hpp"
#include "popnn/NonLinearityDefUtil.hpp"
#include "popnn/PoolingDef.hpp"
#include <poplibs_support/Algorithm.hpp>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/range/irange.hpp>
#include <cassert>
#include <cmath>

using namespace poplar;
using namespace poplibs_support;

// Macro to create entries in cycle estimator table
#define INSTANTIATE_NL_CYCLE_ESTIMATOR(v)                                      \
  CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, popnn::NonLinearityType::GELU),       \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, popnn::NonLinearityType::GELU),    \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, popnn::NonLinearityType::SWISH),  \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, popnn::NonLinearityType::SWISH)

#define INSTANTIATE_NL_GRAD_CYCLE_ESTIMATOR(v)                                 \
  CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, popnn::NonLinearityType::SIGMOID),    \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, popnn::NonLinearityType::SIGMOID), \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, popnn::NonLinearityType::RELU),   \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, popnn::NonLinearityType::RELU),    \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, popnn::NonLinearityType::TANH),   \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, popnn::NonLinearityType::TANH),    \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, popnn::NonLinearityType::GELU),   \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, popnn::NonLinearityType::GELU),    \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, popnn::NonLinearityType::SWISH),  \
      CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, popnn::NonLinearityType::SWISH)

namespace popnn {

static std::uint64_t nonlinearityFlops(const NonLinearityType &nlType) {
  // assume single flop for all non-linearities other than GELU
  if (nlType == NonLinearityType::GELU) {
    return 8;
  } else if (nlType == NonLinearityType::SWISH) {
    return 2;
  }
  return 1;
}

static std::uint64_t nonlinearityGradFlops(const NonLinearityType &nlType) {
  switch (nlType) {
  case NonLinearityType::GELU:
    return 15;
  case NonLinearityType::SWISH:
    return 5;
  case NonLinearityType::RELU:
    return 1;
  case NonLinearityType::SIGMOID:
    return 2;
  case NonLinearityType::TANH:
    return 2;
  default:
    throw poputil::poplibs_error("Unhandled non-linearity type");
  }
}

VertexPerfEstimate
nonLinearity1DCycleEstimator(const VertexIntrospector &vertex,
                             const Target &target, const Type &type,
                             const NonLinearityType &nlType, bool inPlace) {
  bool isNonInPlaceSwish = nlType == NonLinearityType::SWISH && inPlace;
  bool isFloat = type == FLOAT;
  CODELET_FIELD(data);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto vectorWidth = target.getVectorWidth(type);
  const auto n = data.size();

  const auto numVectors = n / vectorWidth;
  const auto remainder = n % vectorWidth;

  // If any worker handles an extra vector due to the remainder
  // we take the longest worker hence rounded up.
  const auto vectorsPerWorker = (numVectors + numWorkers - 1) / numWorkers;

  const auto opCycles = getNonLinearityOpCycles(nlType, isFloat, false);
  const auto vectorLoopCycles = getNonLinearityOpCycles(nlType, isFloat, true);

  // These cycle estimates follow the aligned path. Slightly optimistic.
  // The cost of misalignment is ~9 cycles for half, less for float.
  std::uint64_t cycles = 9; // Supervisor vertex overhead
  std::uint64_t workerCycles =
      3 + // Load input pointer, output pointer and size
              isNonInPlaceSwish
          ? 1
          : 0 +     // Branch into shared code
                5 + // Divide & Remainder to split work between workers
                2 + // Get worker ID
                2 + // Check 64-bit aligned and branch
                5 + // Setup remainders and size for worker
                3 + // Offset worker's pointers and branch if done
                (vectorsPerWorker ? 1 : 0) *
                    (2 + opCycles + // Warm up pipeline, rpt
                     (vectorsPerWorker - 1) * vectorLoopCycles + 1 +
                     opCycles); // Handle remaining element from pipeline

  // possibly unpack pointers
  workerCycles +=
      poputil::internal::getUnpackCost(data.getProfilerVectorLayout(0));

  // Add remainder handling cycles. This handling could be slightly overlapped
  // with other workers if the worker doing the remainder had less vector
  // work than the others. Some of these transcendental ops may take
  // less time anyway so we'll just stick with the simpler estimation.
  if (isFloat) {
    workerCycles +=
        3 + // Test worker ID to handle remainder, test remainder, branch
        ((remainder & 1) ? 1 : 0) * (2 + opCycles); // Handle 32-bit remainder
  } else {
    workerCycles +=
        2 + // Test worker ID to handle remainder with
        1 + // branch for 32-bit remainder
        ((remainder & 2) ? 1 : 0) * (2 + opCycles) + // Handle 32-bit remainder
        1 + // branch for 16-bit remainder
        ((remainder & 1) ? 1 : 0) * (3 + opCycles); // Handle 16-bit remainder
  }

  std::uint64_t flops = n * nonlinearityFlops(nlType);

  return {cycles + (workerCycles * numWorkers),
          convertToTypeFlops(flops, type)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(NonLinearity1D)(const VertexIntrospector &vertex,
                                         const Target &target, const Type &type,
                                         const NonLinearityType &nlType) {
  return nonLinearity1DCycleEstimator(vertex, target, type, nlType, false);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(NonLinearity1DInPlace)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const NonLinearityType &nlType) {
  return nonLinearity1DCycleEstimator(vertex, target, type, nlType, true);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(NonLinearityGrad1D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const NonLinearityType &nlType) {
  bool isFloat = type == FLOAT;
  const auto vectorWidth = target.getVectorWidth(type);
  const auto numWorkers = target.getNumWorkerContexts();
  CODELET_FIELD(inGrad);
  CODELET_FIELD(outGrad);
  CODELET_FIELD(out);
  const auto n = inGrad.size();
  assert(outGrad.size() == n);
  assert(out.size() == n);

  const auto inGradLayout = inGrad.getProfilerVectorLayout(0);
  assert(inGradLayout == outGrad.getProfilerVectorLayout(0) &&
         inGradLayout == out.getProfilerVectorLayout(0));

  const auto numVectors = n / vectorWidth;
  const auto remainder = n % vectorWidth;
  const auto vectorsPerWorker = (numVectors + numWorkers - 1) / numWorkers;
  const auto opCycles = getNonLinearityGradOpCycles(nlType, isFloat, false);
  const auto vectorCycles = getNonLinearityGradOpCycles(nlType, isFloat, true);

  std::uint64_t cycles = 9; // Supervisor vertex overhead
  std::uint64_t workerCycles =
      3 + // Load vertex state
      5 + // Split work between workers
      2 + // Get worker ID
      3 + // Add remaining vectors to relevant workers
      3 + // Offset pointers to data
      3 + // Pre-load inputs, and generate ones if needed
      1 + // Branch if no vectors
      (vectorsPerWorker ? 1 : 0) * (4 + // Warm up the pipeline
                                    (vectorsPerWorker - 1) * vectorCycles +
                                    1); // Store remaining element

  // get real pointers from scaled pointers
  if (inGradLayout == layout::Vector::ScaledPtr64) {
    workerCycles += poputil::internal::getUnpackCost(inGradLayout) + 2;
  }

  if (isFloat) {
    workerCycles += 2 + // Pick a worker to handle the remainder, branch
                    2 + // Check for remainder
                    (remainder ? 1 : 0) * (opCycles + 1);
  } else {
    workerCycles += 2 + // Pick a worker to handle remainders, branch
                    2 + // Check for 32-bit remainder
                    ((remainder & 2) ? 1 : 0) * (opCycles + 2) +
                    2 + // Check for 16-bit remainder
                    ((remainder & 1) ? 1 : 0) * (opCycles + 4);
  }

  return {cycles + (workerCycles * numWorkers),
          convertToTypeFlops(n * nonlinearityGradFlops(nlType), type)};
}

VertexPerfEstimate
nonLinearity2DCycleEstimator(const VertexIntrospector &vertex,
                             const Target &target, const Type &type,
                             const NonLinearityType &nlType, bool inPlace) {
  bool isNonInPlaceSwish = nlType == NonLinearityType::SWISH && inPlace;
  bool isFloat = type == FLOAT;
  const auto vectorWidth = target.getVectorWidth(type);
  CODELET_FIELD(data);
  const auto n0 = data.size();
  assert(n0 > 0);

  std::uint64_t cycles = 5; // Vertex overhead

  const auto opCycles = getNonLinearityOpCycles(nlType, isFloat, false);
  const auto vectorLoopCycles = getNonLinearityOpCycles(nlType, isFloat, true);

  cycles += isNonInPlaceSwish
                ? 1
                : 0 +     // Load out pointer
                      2 + // Load base pointer, DeltaN pointer
                      5 + // Unpack base pointer, n0, DeltaN pointer
                      2;  // Set mask for inner loop, sub for brnzdec

  // Following 64-bit aligned path
  unsigned totalElements = 0;
  const std::uint64_t flopsPerElement = nonlinearityFlops(nlType);
  for (std::size_t i = 0; i < n0; ++i) {
    const auto n1 = data[i].size();
    const auto numVectors = n1 / vectorWidth;
    const auto remainder = n1 % vectorWidth;

    cycles += 4 +                 // Load DeltaN, calculate inner pointer and n1
              1 +                 // Load next out ptr or copy data ptr
              (isFloat ? 0 : 2) + // Test 32-bit aligned
              (isFloat ? 2 : 3) + // Test 64-bit aligned
              2 +                 // Shift to get num vectors, branch if 0
              (numVectors ? 1 : 0) * (2 + opCycles + // Warm up pipeline
                                      (numVectors - 1) * vectorLoopCycles + 1 +
                                      opCycles); // Handle last element

    if (isFloat) {
      cycles += 2 + // Check for remainder, branch
                (remainder ? 1 : 0) * (2 + opCycles);
    } else {
      cycles += 2 + // Check for 32-bit remainder, branch
                ((remainder & 2) ? 1 : 0) * (2 + opCycles) +
                2 + // Check for 16-bit remainder, branch
                ((remainder & 1) ? 1 : 0) * (3 + opCycles);
    }

    cycles += 1; // brnzdec

    // assume one flop per element regardless of non-linearity type
    totalElements += n1;
  }

  return {cycles, convertToTypeFlops(totalElements * flopsPerElement, type)};
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(NonLinearity2D)(const VertexIntrospector &vertex,
                                         const Target &target, const Type &type,
                                         const NonLinearityType &nlType) {
  return nonLinearity2DCycleEstimator(vertex, target, type, nlType, false);
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(NonLinearity2DInPlace)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const NonLinearityType &nlType) {
  return nonLinearity2DCycleEstimator(vertex, target, type, nlType, true);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(NonLinearityGrad2D)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const NonLinearityType &nlType) {
  bool isFloat = type == FLOAT;
  const auto vectorWidth = target.getVectorWidth(type);
  CODELET_FIELD(inGrad);
  CODELET_FIELD(outGrad);
  CODELET_FIELD(out);
  const auto n0 = inGrad.size();
  assert(outGrad.size() == n0);
  assert(out.size() == n0);
  assert(n0 > 0);

  std::uint64_t cycles = 5; // Vertex overhead

  cycles += 4 + // Load vertex state
            3 + // Load DeltaN base/n0, generate ones if needed
            3 + // Calculate DeltaN pointer
            2;  // Set mask for inner loop, sub for brnzdec

  const auto opCycles = getNonLinearityGradOpCycles(nlType, isFloat, false);
  const auto vectorCycles = getNonLinearityGradOpCycles(nlType, isFloat, true);

  unsigned totalElements = 0;
  for (std::size_t i = 0; i < n0; ++i) {
    const auto n1 = inGrad[i].size();
    assert(outGrad[i].size() == n1);
    assert(out[i].size() == n1);
    const auto numVectors = n1 / vectorWidth;
    const auto remainder = n1 % vectorWidth;

    cycles += 6 + // Load DeltaN, calculate inner pointer/n1, shift for n1 vecs
              3 + // Pre-load inputs for pipeline, branch if 0
              (numVectors ? 1 : 0) * (4 + // Warm up pipeline
                                          // Store last element
                                      (numVectors - 1) * vectorCycles + 1);

    if (isFloat) {
      cycles += 2 + // Check for remainder
                (remainder ? 1 : 0) * (1 + opCycles);
    } else {
      cycles += 2 + // Check for 32-bit remainder
                ((remainder & 2) ? 1 : 0) * (2 + opCycles) +
                2 + // Check for 16-bit remainder
                ((remainder & 1) ? 1 : 0) * (4 + opCycles);
    }
    totalElements += n1;
    cycles += 1; // brnzdec
  }

  return {cycles, convertToTypeFlops(
                      totalElements * nonlinearityGradFlops(nlType), type)};
}

VertexPerfEstimate
poolingCycleEstimator(const VertexIntrospector &vertex, const Target &target,
                      const Type &dataType, const PoolingType &pType,
                      bool isBwdPass, bool isGradientScale = false) {
  CODELET_SCALAR_VAL(initInfo, unsigned short);
  CODELET_SCALAR_VAL(chansPerGroupDM1, unsigned short);
  CODELET_SCALAR_VAL(numChanGroupsM1, unsigned short);
  CODELET_VECTOR_VALS(startPos, unsigned short);
  CODELET_VECTOR_2D_VALS(workList, unsigned short);
  CODELET_FIELD(out);
  CODELET_FIELD(in);

  const auto numWorkers = target.getNumWorkerContexts();

  const auto startPosLayout =
      vertex.getFieldInfo("startPos").getProfilerVectorLayout(0);
  const auto workListLayout =
      vertex.getFieldInfo("workList").getProfilerVectorListLayout();

  UnpackCosts unpackCosts;
  unpackCosts.outLayout =
      poputil::internal::getUnpackCost(out.getProfilerVectorLayout(0));

  unpackCosts.inLayout =
      poputil::internal::getUnpackCost(in.getProfilerVectorLayout(0));
  unpackCosts.fwdOutLayout = unpackCosts.outLayout;
  unpackCosts.fwdInLayout = unpackCosts.inLayout;

  unpackCosts.startPosLayout = poputil::internal::getUnpackCost(startPosLayout);
  unpackCosts.workListLayout = poputil::internal::getUnpackCost(workListLayout);

  auto cycles = getPoolingCycles(
      initInfo, chansPerGroupDM1 + 1, numChanGroupsM1, startPos, workList,
      unpackCosts, pType == PoolingType::MAX, isBwdPass, numWorkers);

  // compute flops
  unsigned totalWorkItems = 0;
  for (unsigned wId = 0; wId < numWorkers; ++wId) {
    const unsigned numRows =
        wId == 0 ? startPos[0] : startPos[wId] - startPos[wId - 1];
    for (unsigned row = 0; row < numRows; ++row) {
      const unsigned sPos = wId == 0 ? 0 : startPos[wId - 1];
      for (unsigned w = 0; w != workList[sPos + row].size(); w += 3) {
        totalWorkItems += workList[sPos + row][w + 2];
      }
    }
  }
  std::uint64_t numChanGroups = numChanGroupsM1 + 1;
  const auto scaleFactor = dataType == poplar::HALF ? 4 : 2;
  const auto chansPerGroup = (chansPerGroupDM1 + 1) * scaleFactor;
  std::uint64_t flops = 0;

  if (dataType == FLOAT || dataType == HALF) {
    flops += numChanGroups * totalWorkItems * chansPerGroup;

    if (pType == PoolingType::AVG || pType == PoolingType::SUM ||
        isGradientScale) {
      // additional flops for division/scale
      flops += numChanGroups * chansPerGroup * initInfo;
    }
  }
  return {cycles, convertToTypeFlops(flops, dataType)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(MaxPooling)(const VertexIntrospector &vertex,
                                     const Target &target, const Type &type) {
  return poolingCycleEstimator(vertex, target, type, PoolingType::MAX, false);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(MaxPoolingGradientScale)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  return poolingCycleEstimator(vertex, target, type, PoolingType::MAX, false,
                               true);
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(SumPooling)(const VertexIntrospector &vertex,
                                     const Target &target, const Type &type) {
  return poolingCycleEstimator(vertex, target, type, PoolingType::SUM, false);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(SelectiveScaling)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  // TODO: T5436 Improve this estimate.
  return 10;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ROIAlignForward)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  // TODO
  return 10;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ROIAlignBackward)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  // TODO
  return 10;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(MaxPoolingGrad)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  (void)type;
  return poolingCycleEstimator(vertex, target, type, PoolingType::MAX, true);
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(LossSumSquaredTransform)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fpType) {
  const bool isFloat = fpType == FLOAT;
  const auto size = vertex.getFieldInfo("probs").size();
  const auto isSoftmax = false;
  std::uint64_t flops = static_cast<std::uint64_t>(size) * 3;
  auto cycles = getLossTransformCycles(isFloat, isSoftmax, size);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(LossCrossEntropyTransform)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &fpType) {
  const bool isFloat = fpType == FLOAT;
  const auto size = vertex.getFieldInfo("probs").size();
  const auto isSoftmax = true;
  std::uint64_t flops = static_cast<std::uint64_t>(size) * 6 + 2;
  auto cycles = getLossTransformCycles(isFloat, isSoftmax, size);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ReduceMaxClassGather)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &labelType) {
  std::uint64_t supervisorCycles = 5 + // Vertex overhead
                                   4;  // Supervisor call + sync
  CODELET_FIELD(activations);
  CODELET_SCALAR_VAL(size, unsigned);
  CODELET_SCALAR_VAL(workerSize, unsigned);
  const auto numWorkers = target.getNumWorkerContexts();
  // Check the divisor chosen is large enough to process all inputs
  // with the target number of workers and the grain size.
  assert(workerSize * numWorkers >= size);
  std::uint64_t cycles;
  if (inType == FLOAT || inType == HALF) {
    // Assembly, supervisor implementation
    // Size is the size of the whole tensor, divisor indicates the region an
    // individual worker operates on.  So each worker does divisor inner loop
    // passes unless size is small, in which case one worker does size inner
    // loop passes, and all the others do nothing.
    cycles = 3 + // Load acts pointer, size, divisor
             2 + // Get worker ID
             4 + // Calculate the worker's region
             3 + // Calculate N, sub 1 for first element, branch if no work.
             1 + // Offset pointer for worker
             3 + // Load first element as max, setup pointers
             1 + // rpt
             std::min(workerSize - 1, size - 1) * 3 +
             3 + // Handle remaining element from loop
             6 + // Calculate max index from max act pointer
             4;  // Load maxValue/maxIndex pointers, store (+f16->f32 for half)
  } else {
    // Compiled, 1 worker (pseudo supervisor) version for other types
    const auto nOutputs = (size + workerSize - 1) / workerSize;
    cycles = 22 +                                // Net overhead
             nOutputs * ((workerSize * 6) + 25); // Inner, outer loop overhead
  }
  return {cycles * numWorkers + supervisorCycles,
          convertToTypeFlops(size, inType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ReduceMaxNClassGather)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const bool sorted) {
  CODELET_FIELD(activations);
  CODELET_SCALAR_VAL(divisorLog2, unsigned short);
  CODELET_SCALAR_VAL(numK, unsigned short);

#ifndef NDEBUG
  const auto numWorkers = target.getNumWorkerContexts();
#endif
  const auto divisor = (1u << divisorLog2);
  // Check the divisor chosen is large enough to process all inputs
  // with the target number of workers and the grain size.
  assert(divisor * numWorkers >= activations.size());

  const auto nOutputs = (activations.size() + divisor - 1) / divisor;

  std::uint64_t cycles = 10 + // Initial set up.
                         2;   // Enter nOutputsloop.

  // Gather is assumed to have (roughly) the same cycles as sparse but in a
  // loop. It also doesn't benefit from compile time optimizations.
  for (unsigned i = 0; i < nOutputs; ++i) {
    cycles += 23; // Rough estimate for the first add.

    // For the first K we have a guaranteed push op.
    for (unsigned i = 1; i < numK; ++i) {
      cycles += 13 +              // Setup
                std::log(i) * 20; // log(i) loop.
    }

    // Assumes the worst case. This would be the case of activations being
    // sorted in assending order.
    cycles += (activations.size() - numK) * (13 + std::log(numK) * 20);

    // As we are working on the indices we do a bit at the end to store the
    // actual values as well and transform the indices.
    cycles += 8 * numK;

    if (sorted) {
      for (int i = numK; i >= 1; --i) {
        cycles += 10;               // Setup.
        cycles += 20 * std::log(i); // log(k-1) pop operation.
      }
    }
  }

  return cycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ReduceMaxNClassSparse)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    const bool sorted) {
  CODELET_SCALAR_VAL(numK, unsigned short);
  CODELET_FIELD(activations);

  std::uint64_t cycles = 10 + // Initial set up.
                         2;   // Enter N loop.

  cycles += 23; // Rough estimate for the first add.

  // For the first K we have a guaranteed push op.
  for (int i = 1; i < numK; ++i) {
    cycles += 13 +              // Setup
              std::log(i) * 20; // log(i) loop.
  }

  // Assumes the worst case. This would be the case of activations being
  // sorted in assending order.
  cycles += (activations.size() - numK) * (13 + std::log(numK) * 20);

  // As we are working on the indices we do a bit at the end to store the actual
  // values as well and transform the indices.
  cycles += 8 * numK;

  // Sorting is very expensive but even if requested by the user it will ony be
  // performed on the very last reduction.
  if (sorted) {
    for (int i = numK; i >= 1; --i) {
      cycles += 10;               // Setup.
      cycles += 20 * std::log(i); // log(k-1) pop operation.
    }
  }
  return cycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ReduceMaxClassSparse)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inOutType, const Type &labelType) {
  std::uint64_t cycles = 5; // Vertex overhead
  CODELET_FIELD(activations);
  CODELET_FIELD(labels);
  const auto numActs = activations.size();
  assert(numActs == labels.size());
  if (inOutType == HALF || inOutType == FLOAT) {
    // Assembly implementation
    cycles += 2 + // Load acts start/end pointer
              3 + // Calculate N, sub 1 for first element
              3 + // Load first element as max, setup pointers
              1 + // rpt
              (numActs - 1) * 3 + 3 + // Handle remaining element from loop
              6 + // Calculate max index from max act pointer
              4;  // Load maxValue/maxIndex pointers, store
  } else {
    // Compiled versions for other types
    cycles += 18 +         // Net overhead
              numActs * 6; // Loop cycles
  }
  return cycles;
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ReduceMinClassGather)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &labelType) {
  std::uint64_t supervisorCycles = 5 + // Vertex overhead
                                   4;  // Supervisor call + sync
  CODELET_FIELD(activations);
  CODELET_SCALAR_VAL(size, unsigned);
  CODELET_SCALAR_VAL(workerSize, unsigned);
  const auto numWorkers = target.getNumWorkerContexts();
  // Check the divisor chosen is large enough to process all inputs
  // with the target number of workers and the grain size.
  assert(workerSize * numWorkers >= size);
  std::uint64_t cycles;
  if (inType == FLOAT || inType == HALF) {
    // Assembly, supervisor implementation
    // Size is the size of the whole tensor, divisor indicates the region an
    // individual worker operates on.  So each worker does divisor inner loop
    // passes unless size is small, in which case one worker does size inner
    // loop passes, and all the others do nothing.
    cycles = 3 + // Load acts pointer, size, divisor
             2 + // Get worker ID
             4 + // Calculate the worker's region
             3 + // Calculate N, sub 1 for first element, branch if no work.
             1 + // Offset pointer for worker
             3 + // Load first element as max, setup pointers
             1 + // rpt
             std::min(workerSize - 1, size - 1) * 3 +
             3 + // Handle remaining element from loop
             6 + // Calculate min index from min act pointer
             4;  // Load minValue/minIndex pointers, store (+f16->f32 for half)
  } else {
    // Compiled, 1 worker (pseudo supervisor) version for other types
    const auto nOutputs = (size + workerSize - 1) / workerSize;
    cycles = 22 +                                // Net overhead
             nOutputs * ((workerSize * 6) + 25); // Inner, outer loop overhead
  }
  return {cycles * numWorkers + supervisorCycles,
          convertToTypeFlops(size, inType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ReduceMinClassSparse)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inOutType, const Type &labelType) {
  std::uint64_t cycles = 5; // Vertex overhead
  CODELET_FIELD(activations);
  CODELET_FIELD(labels);
  const auto numActs = activations.size();
  assert(numActs == labels.size());
  if (inOutType == HALF || inOutType == FLOAT) {
    // Assembly implementation
    cycles += 2 + // Load acts start/end pointer
              3 + // Calculate N, sub 1 for first element
              3 + // Load first element as max, setup pointers
              1 + // rpt
              (numActs - 1) * 3 + 3 + // Handle remaining element from loop
              6 + // Calculate min index from min act pointer
              4;  // Load minValue/minIndex pointers, store
  } else {
    // Compiled version for other types
    cycles += 18 +         // Total overhead
              numActs * 6; // Loop cycles
  }
  return {cycles, convertToTypeFlops(numActs, inOutType)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(CalcAccuracy)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &labelType) {
  std::uint64_t cycles = 5; // Vertex overhead
  CODELET_FIELD(maxPerBatch);
  CODELET_FIELD(expected);
  const auto batchSize = maxPerBatch.size();
  assert(batchSize == expected.size());

  cycles += 4 + // Load maxPerBatch start/end, sub, shift for num elements
            2 + // Load expected and numCorrect pointer
            1 + // Load initial numCorrect value
            1;  // rpt

  cycles += batchSize * (2 + // Load maxPerBatch/expected
                         1 + // cmpeq
                         1); // add

  cycles += 1; // Store final numCorrect

  // calc accuracy does no FP operations
  return cycles;
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCAlpha)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &outType, const Type symbolType, const bool isLastLabel) {
  CODELET_FIELD(label);
  // Label contains the previous symbol as a dependency, not an extra result
  return {alphaCycles(1, label.size() - 1, false),
          alphaFlops(1, label.size() - 1, outType, false)};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCBeta)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &outType, const Type symbolType, const bool isFirstLabel) {
  CODELET_FIELD(label);
  // Label contains the previous symbol as a dependency, not an extra result
  return {betaCycles(1, label.size() - 1, false),
          betaFlops(1, label.size() - 1, outType, false)};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCGradGivenAlpha)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &outType, const Type symbolType, const bool isFirstLabel) {
  CODELET_FIELD(label);
  // Label contains the previous symbol as a dependency, not an extra result
  return {gradGivenAlphaCycles(1, label.size() - 1, false),
          gradGivenAlphaFlops(1, label.size() - 1, outType, false)};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCGradGivenBeta)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &outType, const Type symbolType, const bool isLastLabel) {
  CODELET_FIELD(label);
  // Label contains the previous symbol as a dependency, not an extra result
  return {gradGivenBetaCycles(1, label.size() - 1, false),
          gradGivenBetaFlops(1, label.size() - 1, outType, false)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCGenerateCopyCandidates)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &partialsType, const Type symbolType) {
  // TODO: cycle estimator
  return {0, 0};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCGenerateExtendCandidates)(
    const VertexIntrospector &vertex, const Target &target, const Type &inType,
    const Type &partialsType, const Type symbolType) {
  // TODO: cycle estimator
  return {0, 0};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCMergeCandidates)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &partialsType, const Type symbolType) {
  // TODO: cycle estimator
  return {0, 0};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCSelectCopyCandidates)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &partialsType, const Type symbolType) {
  // TODO: cycle estimator
  return {0, 0};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCSelectExtendCandidates)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &partialsType, const Type symbolType) {
  // TODO: cycle estimator
  return {0, 0};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCRankCandidates)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &partialsType, const Type symbolType) {
  CODELET_SCALAR_VAL(totalCandidates, unsigned);
  CODELET_SCALAR_VAL(firstCandidateToRank, unsigned);
  CODELET_SCALAR_VAL(lastCandidateToRank, unsigned);
  CODELET_SCALAR_VAL(beamwidth, unsigned);

  const std::size_t supervisorCycles =
      14 +                  // Vertex state load
      (5 + 6) * beamwidth + // Initialise to zero loop
      8;                    // runall, exit if complete

  const auto numToRank = lastCandidateToRank - firstCandidateToRank;
  const auto numWorkers = target.getNumWorkerContexts();
  const auto maxToRankPerWorker =
      poplibs_support::ceildiv(numToRank, numWorkers);

  std::size_t workerCycles = 13;           // Non loop
  workerCycles += maxToRankPerWorker * 28; // Outer loop body, including odd one
  workerCycles +=
      maxToRankPerWorker * 3 * ((totalCandidates - 1) / 2); // rpt loop content

  workerCycles += 15; // Assume each worker will copy only once

  std::size_t flops = totalCandidates * numToRank * 2;
  return {supervisorCycles + workerCycles * numWorkers, flops};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCReduceCandidates)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &partialsType, const Type symbolType) {
  CODELET_SCALAR_VAL(totalCandidates, unsigned);

  const std::size_t supervisorCycles = 24 + // Exit if complete
                                       12;  // runall, exit

  std::size_t workerCycles = 13 + // Pre loop, processing 1
                             (totalCandidates - 1) / 2 + // Actual loop
                             6;                          // Post loop

  const auto numWorkers = target.getNumWorkerContexts();
  const unsigned numItemsToReduce = 5;
  return {supervisorCycles + workerCycles * numWorkers,
          totalCandidates * numItemsToReduce};
}
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(CTCUpdate)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &partialsType, const Type symbolType) {

  CODELET_SCALAR_VAL(beamwidth, unsigned);

  const std::size_t supervisorCycles = 12 + // Load pointers, vertex state
                                       16;  // Exit branch and function calls
  // Slowest worker path
  const std::size_t workerCycles = 16 +            // Pre loop
                                   beamwidth * 7 + // Loop body
                                   1;              // Post loop

  const auto numWorkers = target.getNumWorkerContexts();
  return {supervisorCycles + numWorkers * workerCycles, 0};
}
VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(CTCGenerateOutput)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type symbolType) {
  // TODO: cycle estimator
  return {0, 0};
}

poputil::internal::PerfEstimatorTable makePerfFunctionTable() {
  return {
      CYCLE_ESTIMATOR_ENTRY(popnn, LossSumSquaredTransform, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, LossSumSquaredTransform, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, LossCrossEntropyTransform, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, LossCrossEntropyTransform, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, HALF, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, INT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, UNSIGNED_INT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, FLOAT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, HALF, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, INT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, UNSIGNED_INT, INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, UNSIGNED_INT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, UNSIGNED_INT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, FLOAT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, INT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, INT, INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, FLOAT, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, HALF, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, INT, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassGather, UNSIGNED_INT, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, FLOAT, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, HALF, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, INT, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxNClassSparse, UNSIGNED_INT, true),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, HALF, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, INT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, UNSIGNED_INT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, FLOAT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, HALF, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, INT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassGather, UNSIGNED_INT, INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassSparse, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassSparse, INT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassSparse, UNSIGNED_INT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassSparse, FLOAT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassSparse, UNSIGNED_INT, INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMinClassSparse, INT, INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, SumPooling, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, SumPooling, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGradientScale, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGradientScale, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, SelectiveScaling, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, SelectiveScaling, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, ROIAlignForward, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ROIAlignForward, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, ROIAlignBackward, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(popnn, ROIAlignBackward, HALF),

      CYCLE_ESTIMATOR_ENTRY(popnn, CTCAlpha, FLOAT, FLOAT, UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCAlpha, HALF, HALF, UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCAlpha, HALF, FLOAT, UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCBeta, FLOAT, FLOAT, UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCBeta, HALF, HALF, UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCBeta, HALF, FLOAT, UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenAlpha, FLOAT, FLOAT,
                            UNSIGNED_INT, true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenAlpha, HALF, HALF, UNSIGNED_INT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenAlpha, HALF, FLOAT, UNSIGNED_INT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenBeta, FLOAT, FLOAT, UNSIGNED_INT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenBeta, HALF, HALF, UNSIGNED_INT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenBeta, HALF, FLOAT, UNSIGNED_INT,
                            true),

      CYCLE_ESTIMATOR_ENTRY(popnn, CTCAlpha, FLOAT, FLOAT, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCAlpha, HALF, HALF, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCAlpha, HALF, FLOAT, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCBeta, FLOAT, FLOAT, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCBeta, HALF, HALF, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCBeta, HALF, FLOAT, UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenAlpha, FLOAT, FLOAT,
                            UNSIGNED_INT, false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenAlpha, HALF, HALF, UNSIGNED_INT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenAlpha, HALF, FLOAT, UNSIGNED_INT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenBeta, FLOAT, FLOAT, UNSIGNED_INT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenBeta, HALF, HALF, UNSIGNED_INT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGradGivenBeta, HALF, FLOAT, UNSIGNED_INT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGenerateCopyCandidates, FLOAT, FLOAT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGenerateCopyCandidates, HALF, FLOAT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGenerateCopyCandidates, HALF, HALF,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGenerateExtendCandidates, FLOAT, FLOAT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGenerateExtendCandidates, HALF, FLOAT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGenerateExtendCandidates, HALF, HALF,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCMergeCandidates, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCMergeCandidates, HALF, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, CTCSelectCopyCandidates, FLOAT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCSelectCopyCandidates, HALF, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, CTCSelectExtendCandidates, FLOAT,
                            UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCSelectExtendCandidates, HALF,
                            UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, CTCRankCandidates, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCRankCandidates, HALF, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, CTCReduceCandidates, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCReduceCandidates, HALF, UNSIGNED_INT),

      CYCLE_ESTIMATOR_ENTRY(popnn, CTCUpdate, HALF, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCUpdate, FLOAT, UNSIGNED_INT),
      CYCLE_ESTIMATOR_ENTRY(popnn, CTCGenerateOutput, UNSIGNED_INT),

      INSTANTIATE_NL_GRAD_CYCLE_ESTIMATOR(NonLinearityGrad1D),
      INSTANTIATE_NL_CYCLE_ESTIMATOR(NonLinearity1DInPlace),
      CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity1D, FLOAT,
                            popnn::NonLinearityType::SWISH),
      CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity1D, HALF,
                            popnn::NonLinearityType::SWISH),

      INSTANTIATE_NL_GRAD_CYCLE_ESTIMATOR(NonLinearityGrad2D),
      INSTANTIATE_NL_CYCLE_ESTIMATOR(NonLinearity2DInPlace),
      CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity2D, FLOAT,
                            popnn::NonLinearityType::SWISH),
      CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity2D, HALF,
                            popnn::NonLinearityType::SWISH)};
}

} // end namespace popnn
