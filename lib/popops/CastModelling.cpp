// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CastModelling.hpp"

#include <poplar/Target.hpp>
#include <poplar/Type.hpp>

#include <popops/PerformanceEstimation.hpp>

using namespace poplar;

namespace popops {

using namespace internal;

namespace modelling {

CastEstimates modelContiguousCast(const poplar::Target &target,
                                  const poplar::Type &inType,
                                  const poplar::Type &outType,
                                  popsolver::Model &m,
                                  const popsolver::Variable &mNumElems,
                                  const std::string &debugPrefix) {
  CastEstimates e(m.zero());

  // Modelling for non-floating point cast not currently handled.
  assert((inType == FLOAT || inType == HALF) &&
         (outType == FLOAT || outType == HALF));

  if (inType != outType) {
    const CastTargetParameters targetParams{target, inType, outType};
    e.cycles = m.call<unsigned>(
        {mNumElems},
        [targetParams, inType, outType](const std::vector<unsigned> &values) {
          const unsigned numElems = values[0];
          const auto cycles =
              getCast1DCycleEstimate(targetParams, inType, outType, numElems);
          return popsolver::DataType{cycles};
        },
        debugPrefix + ".cycles");
  }

  return e;
}

} // end namespace modelling
} // end namespace popops
