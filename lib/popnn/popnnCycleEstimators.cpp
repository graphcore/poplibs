#include "popnnCycleEstimators.hpp"
#include <poplar/HalfFloat.hpp>

#include "popnn/NonLinearity.hpp"
#include "PerformanceEstimation.hpp"

namespace popnn {

template <class FPType>
MAKE_CYCLE_ESTIMATOR(NonLinearity, vertex, target) {
  bool isFloat = std::is_same<FPType, float>::value;
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

template <class FPType>
MAKE_CYCLE_ESTIMATOR(NonLinearityGrad, vertex, target) {
  bool isFloat = std::is_same<FPType, float>::value;
  uint64_t cycles = 5;
  const auto inGrad = vertex.getFieldInfo("inGrad");
  auto nonLinearityType =
    vertex.getFieldInfo("nonLinearityType").getInitialValue<unsigned>(target);
  for (unsigned i = 0; i < inGrad.size(); ++i) {
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

template <class FPType>
MAKE_CYCLE_ESTIMATOR(MaxPooling, vertex, target) {
  unsigned numCycles = 10;
  bool isFloat = std::is_same<FPType, float>::value;
  const auto out = vertex.getFieldInfo("out");
  const auto windowSizes = vertex.getFieldInfo("windowSizes");
  const auto windowSizeValues =
    windowSizes.getInitialValues<unsigned>(target);
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  for (unsigned i = 0; i < out.size(); ++i) {
    auto numVectors = (out[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // TODO: This is too optimistic
    numCycles += 1 + numVectors * (1 + windowSize);
  }
  return numCycles;
}

template <class FPType>
MAKE_CYCLE_ESTIMATOR(ScaledSumPooling, vertex, target) {
  unsigned numCycles = 10;
  bool isFloat = std::is_same<FPType, float>::value;
  const auto out = vertex.getFieldInfo("out");
  const auto windowSizes = vertex.getFieldInfo("windowSizes");
  const auto windowSizeValues =
    windowSizes.getInitialValues<unsigned>(target);
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  const auto scaleOutput =
    vertex.getFieldInfo("scaleOutput").getInitialValue<bool>(target);
  const unsigned scaleCycles = scaleOutput ? 1 : 0;
  for (unsigned i = 0; i < out.size(); ++i) {
    auto numVectors = (out[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // load ptr/vec/ load data/add for windowSize
    // axpby and store
    numCycles += 2 + scaleCycles + numVectors * (3 + 2 * windowSize);
  }
  return numCycles;
}

template <class FPType>
MAKE_CYCLE_ESTIMATOR(MaxPoolingGrad, vertex, target) {
  unsigned numCycles = 10;
  bool isFloat = std::is_same<FPType, float>::value;
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
  for (unsigned i = 0; i < inGrad.size(); ++i) {
    auto numVectors = (inGrad[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // TODO: This is too optimistic
    numCycles += 5 + numVectors * (5 + windowSize * 3);
  }
  return numCycles;
}

template <class FPType>
MAKE_CYCLE_ESTIMATOR(SumPoolingGrad, vertex, target) {
  unsigned numCycles = 10;
  bool isFloat = std::is_same<FPType, float>::value;
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
  for (unsigned i = 0; i < inGrad.size(); ++i) {
    auto numVectors = (inGrad[i].size() + vectorWidth - 1) / vectorWidth;
    auto windowSize = windowSizeValues[i];
    // TODO: This is too optimistic
    numCycles += 2 + numVectors * (4 + windowSize * 1);
  }
  return numCycles;
}

MAKE_CYCLE_ESTIMATOR(CalcLoss, vertex, target) {
  return 0;
}

MAKE_CYCLE_ESTIMATOR(BatchNormEstimates, vertex, target) {
  unsigned numCycles = 5;
  const auto mean = vertex.getFieldInfo("mean");
  const auto acts = vertex.getFieldInfo("acts");
  const unsigned n = mean.size();
  const unsigned batchSize = acts[0].size();
  for (unsigned i = 0; i != n; ++i) {
    const unsigned numActs = mean[i].size();
    numCycles += (batchSize + 7) * numActs;
  }
  return numCycles;
}

using half = poplar::half;
poplibs::CycleEstimatorTable cyclesFunctionTable = {
  TYPED_CYCLE_ESTIMATOR_ENTRY(popnn, BatchNormEstimates, float, float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popnn, BatchNormEstimates, half, float),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, float, unsigned int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, float, int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, half, unsigned int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popnn, CalcLoss, half, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, SumPoolingGrad, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, SumPoolingGrad, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, MaxPoolingGrad, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, ScaledSumPooling, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, MaxPooling, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearityGrad, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearityGrad, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popnn, NonLinearity, half)
};

} // end namespace popnn
