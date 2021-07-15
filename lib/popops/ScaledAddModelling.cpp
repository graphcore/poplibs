// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "ScaledAddModelling.hpp"

#include <poplar/Target.hpp>
#include <poplar/Type.hpp>

#include <popops/PerformanceEstimation.hpp>

using namespace poplar;
using namespace popops::internal;

namespace popops {
namespace modelling {

ScaledAddEstimates
modelContiguousScaledAdd(const Target &target, const Type &dataType,
                         const Type &dataBType, const bool scaleIsConstant,
                         const bool isMemConstrained, popsolver::Model &m,
                         const popsolver::Variable &mNumElems,
                         const std::string &debugPrefix) {
  ScaledArithmeticTargetParameters targetParams(target, dataType);
  ScaledAddEstimates e(m.zero());

  e.cycles = m.call<unsigned>(
      {mNumElems},
      [targetParams, dataType, dataBType, scaleIsConstant,
       isMemConstrained](const std::vector<unsigned> &values) {
        const unsigned numElems = values[0];
        // NOTE: vector layout is a small one-off cost, and not currently
        // easy to get from the target so just assuming.
        const auto vectorLayout = layout::Vector::ScaledPtr64;
        const auto cycles = getScaledArithmeticSupervisorCycleEstimate(
            targetParams, dataType, dataBType, scaleIsConstant,
            isMemConstrained, ScaledArithmeticOp::ADD, vectorLayout,
            vectorLayout, numElems);
        return popsolver::DataType{cycles};
      },
      debugPrefix + ".cycles");
  return e;
}

} // end namespace modelling
} // end namespace popops
