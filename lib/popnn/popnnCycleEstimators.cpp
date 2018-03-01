#include "popnnCycleEstimators.hpp"

#include "popnn/NonLinearity.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popnn {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearity)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  bool isFloat = type == FLOAT;
  std::vector<unsigned> regionSizes;
  const auto data = vertex.getFieldInfo("data");
  auto nonLinearityType =
    vertex.getFieldInfo("nonLinearityType").getInitialValue<unsigned>(target);
  for (unsigned i=0;i<data.size(); ++i)
    regionSizes.push_back(data[i].size());
  return getNonLinearityCycles(regionSizes,
                               NonLinearityType(nonLinearityType), isFloat,
                               target.getDataPathWidth());
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NonLinearityGrad)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &type) {
  bool isFloat = type == FLOAT;
  uint64_t cycles = 5;
  const auto inGrad = vertex.getFieldInfo("inGrad");
  auto nonLinearityType =
    vertex.getFieldInfo("nonLinearityType").getInitialValue<unsigned>(target);
  CODELET_FIELD(outGrad);
  CODELET_FIELD(out);
  assert(outGrad.size() == inGrad.size());
  for (unsigned i = 0; i < inGrad.size(); ++i) {
    assert(outGrad[i].size() == inGrad[i].size());
    assert(outGrad[i].size() == out[i].size());
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    unsigned numVectors = (inGrad[i].size() + vectorWidth - 1) / vectorWidth;
    switch (nonLinearityType) {
    case NON_LINEARITY_SIGMOID:
      cycles += 5 + numVectors * 3;
      break;
    case NON_LINEARITY_RELU: {
      const unsigned vertexOverhead = 2    // run instruction
                                      + 7; // remaining vertex overhead
      cycles += vertexOverhead + numVectors * 3;
      }
      break;
    case NON_LINEARITY_TANH:
      cycles += 5 + numVectors * 3;
      break;
    default:
      throw std::runtime_error("Invalid nonlinearity type");
    }
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
    windowSizes.getInitialValues<unsigned>(target);
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
                                            const Type &type) {
  unsigned numCycles = 10;
  bool isFloat = type == FLOAT;
  const auto out = vertex.getFieldInfo("out");
  const auto windowSizes = vertex.getFieldInfo("windowSizes");
  const auto windowSizeValues =
    windowSizes.getInitialValues<unsigned>(target);
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  const auto scaleOutput =
    vertex.getFieldInfo("scaleOutput").getInitialValue<bool>(target);
  const unsigned scaleCycles = scaleOutput ? 1 : 0;
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
    windowSizes.getInitialValues<unsigned>(target);
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
    windowSizes.getInitialValues<unsigned>(target);
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
MAKE_CYCLE_ESTIMATOR_NAME(CalcLoss)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &fpType,
                                    const Type &labelType) {
  // TODO
  CODELET_FIELD(batchIn);
  CODELET_FIELD(batchDeltaOut);
  CODELET_FIELD(probs);
  assert(batchIn.size() == batchDeltaOut.size());
  const auto batchSize = batchIn.size();
  for (unsigned batchNum = 0; batchNum < batchSize; ++batchNum) {
#ifndef NDEBUG
    auto in = batchIn[batchNum];
    auto deltaOut = batchDeltaOut[batchNum];
    assert(in.size() == deltaOut.size());
    assert(probs.size() == in.size());
#endif
  }
  return 0;
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

    CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, FLOAT, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, HALF, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, HALF, INT),

    CYCLE_ESTIMATOR_ENTRY(popnn, SumPoolingGrad, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, SumPoolingGrad, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearityGrad, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearityGrad, HALF),

    CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity, HALF)
  };
};

} // end namespace popnn
