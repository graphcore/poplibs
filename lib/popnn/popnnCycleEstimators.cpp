#include "popnnCycleEstimators.hpp"

#include "popnn/NonLinearity.hpp"
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
  std::vector<unsigned> regionSizes;
  const auto data = vertex.getFieldInfo("data");
  regionSizes.push_back(data.size());
  return getNonLinearityCycles(regionSizes, nlType, isFloat, false, true,
                               target.getDataPathWidth(),
                               target.getNumWorkerContexts());
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearityGradSupervisor)(
                              const VertexIntrospector &vertex,
                              const Target &target,
                              const Type &type,
                              const NonLinearityType &nlType) {
  bool isFloat = type == FLOAT;
  std::vector<unsigned> regionSizes;
  const auto inGrad = vertex.getFieldInfo("inGrad");
  CODELET_FIELD(outGrad);
  CODELET_FIELD(out);
  assert(outGrad.size() == inGrad.size());
  assert(outGrad.size() == out.size());
  regionSizes.push_back(inGrad.size());
  return getBwdNonlinearityDerivativeCycles(regionSizes, nlType, isFloat, false,
                                            true, target.getDataPathWidth(),
                                            target.getNumWorkerContexts());
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearity2D)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type,
                                          const NonLinearityType &nlType) {
  bool isFloat = type == FLOAT;
  std::vector<unsigned> regionSizes;
  const auto data = vertex.getFieldInfo("data");
  for (unsigned i=0;i<data.size(); ++i)
    regionSizes.push_back(data[i].size());
  return getNonLinearityCycles(regionSizes, nlType, isFloat, true, false,
                               target.getDataPathWidth(),
                               target.getNumWorkerContexts());
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearityGrad2D)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type,
                                              const NonLinearityType &nlType) {
  bool isFloat = type == FLOAT;
  std::vector<unsigned> regionSizes;
  const auto inGrad = vertex.getFieldInfo("inGrad");
  CODELET_FIELD(outGrad);
  CODELET_FIELD(out);
  assert(outGrad.size() == inGrad.size());
  for (unsigned i = 0; i < inGrad.size(); ++i) {
    assert(outGrad[i].size() == inGrad[i].size());
    assert(outGrad[i].size() == out[i].size());
    regionSizes.push_back(inGrad[i].size());
  }
  return getBwdNonlinearityDerivativeCycles(regionSizes, nlType, isFloat, true,
                                            false, target.getDataPathWidth(),
                                            target.getNumWorkerContexts());
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
MAKE_CYCLE_ESTIMATOR_NAME(LossSoftmaxTransform)
  (const VertexIntrospector &vertex,
   const Target &target,
   const Type &fpType) {
  const bool isFloat = fpType == FLOAT;
  const auto size = vertex.getFieldInfo("probs").size();
  const auto isSoftmax = true;
  return getLossTransformCycles(isFloat, isSoftmax, size);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CalcAccuracy)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &fpType,
                                        const Type &labelType) {
  std::uint64_t cycles = 5; // Vertex overhead

  CODELET_FIELD(activations);
  CODELET_FIELD(labels);
  const auto batchSize = activations.size();
  assert(labels.size() == batchSize);

  cycles += 3 + // Load activations outer start/end pointer, calc length
            1 + // Load labels pointer
            2 + // Load numCorrect pointer, then current value
            1;  // Sub 1 for brnzdec

  const auto classesPerBatch = activations[0].size();
  for (std::size_t b = 0; b < batchSize; ++b) {
    cycles += 3; // Load inner start/end pointer and calc length

    const auto numClasses = activations[b].size();
    assert(numClasses == classesPerBatch);

    // Cycles to find index of max class
    // This is worst case, where all activations are sorted in ascending order
    // and every iteration takes the branch.
    //
    cycles += 1;   // Set counter tracking current class index (to -1 for below
                   // loop to work).
    cycles += numClasses *
              (1 + // [M] load next activation
                   // [A] cmpgt this activation
               1 + // [A] min to give 0.0/1.0
               1 + // [M] add to counter tracking current class index
                   // [A] convert comparison result to int
               2 + // [M] move comparison result to MRF, then branch
               1 + // [M] potentially update current max activation
                   // [A] potentially update current max class index
               2); // [M] cmpeq curr class index with class count, branch to
                   //     loop.

    // Cycles to check if the max class index was the expected
    cycles += 1 + // load expected class index
              1 + // cmpeq expected with max class index
              1 + // add to running numCorrect total
              1;  // outer loop brnzdec
  }

  cycles += 1; // store final numCorrect total
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

    CYCLE_ESTIMATOR_ENTRY(popnn, LossSoftmaxTransform, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, LossSoftmaxTransform, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, FLOAT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, HALF, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, CalcAccuracy, HALF, INT),

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
