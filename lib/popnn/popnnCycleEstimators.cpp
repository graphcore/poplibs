#include "popnnCycleEstimators.hpp"

#include "popnn/NonLinearity.hpp"
#include "popnn/NonLinearityDefUtil.hpp"
#include "popnn/PoolingDef.hpp"
#include "PoolingDefUtil.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

// Macro to create entries in cycle estimator table
#define INSTANTIATE_NL_CYCLE_ESTIMATOR(v, ...) \
        CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, \
                              popnn::NonLinearityType::SIGMOID), \
        CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, \
                              popnn::NonLinearityType::SIGMOID), \
        CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, \
                              popnn::NonLinearityType::RELU), \
        CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, \
                              popnn::NonLinearityType::RELU), \
        CYCLE_ESTIMATOR_ENTRY(popnn, v, FLOAT, \
                              popnn::NonLinearityType::TANH), \
        CYCLE_ESTIMATOR_ENTRY(popnn, v, HALF, \
                              popnn::NonLinearityType::TANH)

namespace popnn {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearitySupervisor)(
                              const VertexIntrospector &vertex,
                              const Target &target,
                              const Type &type,
                              const NonLinearityType &nlType) {
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

  // We do 2 ops per vector.
  const auto opCycles = getNonLinearityOpCycles(nlType, isFloat);
  const auto vectorLoopCycles = opCycles * 2;

  // These cycle estimates follow the aligned path. Slightly optimistic.
  // The cost of misalignment is ~9 cycles for half, less for float.
  std::uint64_t cycles = 9; // Supervisor vertex overhead
  std::uint64_t workerCycles =
    2 + // Load input pointer and size
    5 + // Divide & Remainder to split work between workers
    2 + // Shift scaled pointer, get base address
    2 + // Get worker ID
    2 + // Check 64-bit aligned and branch
    5 + // Setup remainders and size for worker
    2 + // Offset worker's pointer and branch if done
    (vectorsPerWorker ? 1 : 0) *
      (2 + opCycles + // Warm up pipeline, rpt
       (vectorsPerWorker - 1) * vectorLoopCycles +
       1 + opCycles); // Handle remaining element from pipeline

  // Add remainder handling cycles. This handling could be slightly overlapped
  // with other workers if the worker doing the remainder had less vector
  // work than the others. Some of these transcendental ops may take
  // less time anyway so we'll just stick with the simpler estimation.
  if (isFloat) {
    workerCycles +=
      3 + // Test worker ID to handle remainder, test remainder, branch
      ((remainder & 1) ? 1 : 0) * (2 + opCycles);  // Handle 32-bit remainder
  } else {
    workerCycles +=
      2 + // Test worker ID to handle remainder with
      1 + // branch for 32-bit remainder
      ((remainder & 2) ? 1 : 0) * (2 + opCycles) + // Handle 32-bit remainder
      1 + // branch for 16-bit remainder
      ((remainder & 1) ? 1 : 0) * (3 + opCycles);  // Handle 16-bit remainder
  }

  return cycles + (workerCycles * numWorkers);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearityGradSupervisor)(
                              const VertexIntrospector &vertex,
                              const Target &target,
                              const Type &type,
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

  const auto numVectors = n / vectorWidth;
  const auto remainder = n % vectorWidth;
  const auto vectorsPerWorker = (numVectors + numWorkers - 1) / numWorkers;

  std::uint64_t cycles = 9; // Supervisor vertex overhead
  std::uint64_t workerCycles =
    8 + // Load vertex state, get real pointers from scaled pointers
    5 + // Split work between workers
    2 + // Get worker ID
    3 + // Add remaining vectors to relevant workers
    3 + // Offset pointers to data
    3 + // Pre-load inputs, and generate ones if needed
    1 + // Branch if no vectors
    (vectorsPerWorker ? 1 : 0) *
      (4 + // Warm up the pipeline
       (vectorsPerWorker - 1) * 3 +
       1); // Store remaining element

  if (isFloat) {
    workerCycles += 2 + // Pick a worker to handle the remainder, branch
                    2 + // Check for remainder
                    (remainder ? 1 : 0) * 4;
  } else {
    workerCycles += 2 + // Pick a worker to handle remainders, branch
                    2 + // Check for 32-bit remainder
                    ((remainder & 2) ? 1 : 0) * 5 +
                    2 + // Check for 16-bit remainder
                    ((remainder & 1) ? 1 : 0) * 7;
  }

  return cycles + (workerCycles * numWorkers);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearity2D)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type,
                                          const NonLinearityType &nlType) {
  bool isFloat = type == FLOAT;
  const auto vectorWidth = target.getVectorWidth(type);
  CODELET_FIELD(data);
  const auto n0 = data.size();
  assert(n0 > 0);

  std::uint64_t cycles = 5; // Vertex overhead

  // We do 2 ops per vector.
  const auto opCycles = getNonLinearityOpCycles(nlType, isFloat);
  const auto vectorLoopCycles = opCycles * 2;

  cycles += 2 + // Load base pointer, DeltaN pointer
            5 + // Unpack base pointer, n0, DeltaN pointer
            2;  // Set mask for inner loop, sub for brnzdec

  // Following 64-bit aligned path
  for (std::size_t i = 0; i < n0; ++i) {
    const auto n1 = data[i].size();
    const auto numVectors = n1 / vectorWidth;
    const auto remainder = n1 % vectorWidth;

    cycles += 4 + // Load DeltaN, calculate inner pointer and n1
              (isFloat ? 0 : 2) + // Test 32-bit aligned
              2 + // Test 64-bit aligned
              2 + // Shift to get num vectors, branch if 0
              (numVectors ? 1 : 0) *
                (2 + opCycles + // Warm up pipeline
                 (numVectors - 1) * vectorLoopCycles +
                 1 + opCycles); // Handle last element

    if (isFloat) {
      cycles += 2 + // Check for remainder, branch
                (remainder ? 1 : 0) * (2 + opCycles);
    } else {
      cycles +=
        2 + // Check for 32-bit remainder, branch
        ((remainder & 2) ? 1 : 0) * (2 + opCycles) +
        2 + // Check for 16-bit remainder, branch
        ((remainder & 1) ? 1 : 0) * (3 + opCycles);
    }

    cycles += 1; // brnzdec
  }

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearityGrad2D)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type,
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

  for (std::size_t i = 0; i < n0; ++i) {
    const auto n1 = inGrad[i].size();
    assert(outGrad[i].size() == n1);
    assert(out[i].size() == n1);
    const auto numVectors = n1 / vectorWidth;
    const auto remainder = n1 % vectorWidth;

    cycles += 6 + // Load DeltaN, calculate inner pointer/n1, shift for n1 vecs
              3 + // Pre-load inputs for pipeline, branch if 0
              (numVectors ? 1 : 0) *
                (4 + // Warm up pipeline
                 (numVectors - 1) * 3 +
                 1); // Store last element

    if (isFloat) {
      cycles += 2 + // Check for remainder
                (remainder ? 1 : 0) * 4;
    } else {
      cycles += 2 + // Check for 32-bit remainder
                ((remainder & 2) ? 1 : 0) * 5 +
                2 + // Check for 16-bit remainder
                ((remainder & 1) ? 1 : 0) * 7;
    }

    cycles += 1; // brnzdec
  }

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(MaxPooling)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      const Type &type) {
  unsigned numCycles = 10;
  bool isFloat = type == FLOAT;
  const auto out = vertex.getFieldInfo("out");
  const auto windowSizes = vertex.getFieldInfo("windowSizes");
  const auto windowSizeValues =
    windowSizes.getInitialValues<unsigned short>(target);
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  unsigned inIndex = 0;
  CODELET_FIELD(in);
  assert(windowSizes.size() == out.size());
  for (unsigned i = 0; i < out.size(); ++i) {
    for (unsigned w = 0; w < windowSizeValues[i]; ++w) {
      assert(out[i].size() == in[inIndex].size());
    }
    inIndex += windowSizeValues[i];
    auto numVectors = (out[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // TODO: This is too optimistic
    numCycles += 1 + numVectors * (1 + windowSize);
  }
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledSumPooling)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &type,
                                            const popnn::PoolingType pType) {
  unsigned numCycles = 10;
  bool isFloat = type == FLOAT;
  const auto out = vertex.getFieldInfo("out");
  const auto windowSizes = vertex.getFieldInfo("windowSizes");
  const auto windowSizeValues =
    windowSizes.getInitialValues<unsigned short>(target);
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  const unsigned scaleCycles = pType == popnn::PoolingType::AVG ? 1 : 0;
  CODELET_FIELD(in);
  unsigned inIndex = 0;
  for (unsigned i = 0; i < out.size(); ++i) {
    for (unsigned w = 0; w < windowSizeValues[i]; ++w) {
      assert(out[i].size() == in[inIndex].size());
      inIndex++;
    }
    auto numVectors = (out[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // load ptr/vec/ load data/add for windowSize
    // axpby and store
    numCycles += 2 + scaleCycles + numVectors * (3 + 2 * windowSize);
  }
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(MaxPoolingGrad)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type) {
  unsigned numCycles = 10;
  bool isFloat = type == FLOAT;
  const auto inGrad = vertex.getFieldInfo("inGrad");
  const auto windowSizes = vertex.getFieldInfo("windowSizes");
  const auto windowSizeValues =
    windowSizes.getInitialValues<unsigned short>(target);
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  // Expected implementation per group:
  // load group of actIn
  // for windowsize:
  // load actOut
  //  compare
  //  res<<=14 (covert to 0.5/0)
  //  mac
  // getacc
  // double
  // store
  CODELET_FIELD(in);
  CODELET_FIELD(out);
  CODELET_FIELD(outGrad);
  unsigned inIndex = 0;
  for (unsigned i = 0; i < inGrad.size(); ++i) {
    assert(inGrad[i].size() == in[i].size());
    for (unsigned w = 0; w < windowSizeValues[i]; ++w) {
      assert(inGrad[i].size() == outGrad[inIndex].size());
      assert(inGrad[i].size() == out[inIndex].size());
      inIndex++;
    }
    auto numVectors = (inGrad[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // TODO: This is too optimistic
    numCycles += 5 + numVectors * (5 + windowSize * 3);
  }
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(SumPoolingGrad)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type) {
  unsigned numCycles = 10;
  bool isFloat = type == FLOAT;
  const auto inGrad = vertex.getFieldInfo("inGrad");
  const auto windowSizes = vertex.getFieldInfo("windowSizes");
  const auto windowSizeValues =
    windowSizes.getInitialValues<unsigned short>(target);
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  // Expected implementation per group:
  // for windowsize:
  // load deltaIn
  //  acc
  // getacc
  // axpby
  // double
  // store
#ifndef NDEBUG
  CODELET_FIELD(outGrad);
  unsigned inIndex = 0;
#endif
  for (unsigned i = 0; i < inGrad.size(); ++i) {
    for (unsigned w = 0; w < windowSizeValues[i]; ++w)
      assert(inGrad[i].size() == outGrad[inIndex++].size());
    auto numVectors = (inGrad[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // TODO: This is too optimistic
    numCycles += 2 + numVectors * (4 + windowSize * 1);
  }
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(LossSumSquaredTransform)
  (const VertexIntrospector &vertex,
   const Target &target,
   const Type &fpType) {
  const bool isFloat = fpType == FLOAT;
  const auto size = vertex.getFieldInfo("probs").size();
  const auto isSoftmax = false;
  return getLossTransformCycles(isFloat, isSoftmax, size);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(LossCrossEntropyTransform)
  (const VertexIntrospector &vertex,
   const Target &target,
   const Type &fpType) {
  const bool isFloat = fpType == FLOAT;
  const auto size = vertex.getFieldInfo("probs").size();
  const auto isSoftmax = true;
  return getLossTransformCycles(isFloat, isSoftmax, size);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMaxClassGather)
  (const VertexIntrospector &vertex,
   const Target &target,
   const Type &fpType,
   const Type &labelType) {
  std::uint64_t cycles = 5 + // Vertex overhead
                         4;  // Supervisor call + sync
  CODELET_FIELD(activations);
  CODELET_SCALAR_VAL(size, unsigned);
  CODELET_SCALAR_VAL(divisorLog2, unsigned short);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto divisor = (1u << divisorLog2);
  // Check the divisor chosen is large enough to process all inputs
  // with the target number of workers and the grain size.
  assert(divisor * numWorkers >= size);
  const auto isFloat = (fpType == FLOAT);

  cycles += 3 + // Load acts pointer, size, divisor
            2 + // Get worker ID
            4 + // Calculate the worker's region
            3 + // Calculate N, sub 1 for first element, branch if no work.
            1 + // Offset pointer for worker
            3 + // Load first element as max, setup pointers
            1 + // rpt
            std::min(divisor - 1, size - 1) * 3 +
            3 + // Handle remaining element from loop
            6 + // Calculate max index from max act pointer
            4;  // Load maxValue/maxIndex pointers, store (+ f16->f32 for half)

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMaxClassSparse)
  (const VertexIntrospector &vertex,
   const Target &target,
   const Type &labelType) {
  std::uint64_t cycles = 5; // Vertex overhead
  CODELET_FIELD(activations);
  CODELET_FIELD(labels);
  const auto numActs = activations.size();
  assert(numActs == labels.size());

  cycles += 2 + // Load acts start/end pointer
            3 + // Calculate N, sub 1 for first element
            3 + // Load first element as max, setup pointers
            1 + // rpt
            (numActs - 1) * 3 +
            3 + // Handle remaining element from loop
            6 + // Calculate max index from max act pointer
            4;  // Load maxValue/maxIndex pointers, store

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CalcAccuracy)(const VertexIntrospector &vertex,
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

  cycles += batchSize *
              (2 + // Load maxPerBatch/expected
               1 + // cmpeq
               1); // add

  cycles += 1; // Store final numCorrect

  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BatchNormEstimates)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &inType,
                                              const Type &partialsType) {
  unsigned numCycles = 5;
  const auto mean = vertex.getFieldInfo("mean");
  const auto acts = vertex.getFieldInfo("acts");
  const unsigned n = mean.size();
  const auto batchSize = acts[0].size();
#ifndef NDEBUG
  unsigned actsIdx = 0;
#endif
  for (unsigned i = 0; i != n; ++i) {
    const unsigned numActs = mean[i].size();
    numCycles += (batchSize + 7) * numActs;
    for (unsigned a = 0; a != numActs; ++a) {
      assert(acts[actsIdx++].size() == batchSize);
    }
  }
  return numCycles;
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return
  {
    CYCLE_ESTIMATOR_ENTRY(popnn, BatchNormEstimates, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, BatchNormEstimates, HALF, FLOAT),

    CYCLE_ESTIMATOR_ENTRY(popnn, LossSumSquaredTransform, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, LossSumSquaredTransform, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, LossCrossEntropyTransform, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, LossCrossEntropyTransform, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, FLOAT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, HALF, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassGather, HALF, INT),

    CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, ReduceMaxClassSparse, INT),

    CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, INT),

    CYCLE_ESTIMATOR_ENTRY(popnn, SumPoolingGrad, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, SumPoolingGrad, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, FLOAT, PoolingType::AVG),
    CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, FLOAT, PoolingType::SUM),
    CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, HALF, PoolingType::AVG),
    CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, HALF, PoolingType::SUM),

    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, HALF),

    INSTANTIATE_NL_CYCLE_ESTIMATOR(NonLinearityGradSupervisor),
    INSTANTIATE_NL_CYCLE_ESTIMATOR(NonLinearitySupervisor),
    INSTANTIATE_NL_CYCLE_ESTIMATOR(NonLinearityGrad2D),
    INSTANTIATE_NL_CYCLE_ESTIMATOR(NonLinearity2D)
  };
};

} // end namespace popnn
