// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "FillModelling.hpp"

#include <popops/PerformanceEstimation.hpp>

#include <gccs/Algorithm.hpp>

using namespace poplar;
namespace popsolver = gccs::popsolver;

namespace popops {

using namespace internal;

namespace modelling {

FillEstimates modelContiguousFill(const Target &target, const Type &type,
                                  popsolver::Model &m,
                                  const popsolver::Variable &numElems,
                                  const std::string &debugPrefix) {
  FillEstimates e(m.zero());

  const FillTargetParameters targetParams{target};
  const auto numWorkerContexts = target.getNumWorkerContexts();
  const auto vectorWidth = target.getVectorWidth(type);
  e.cycles = m.call<unsigned>(
      {numElems},
      [targetParams, numWorkerContexts, vectorWidth,
       type](const std::vector<unsigned> &values) {
        const unsigned numElems = values[0];
        const auto numGrains = gccs::ceildiv(numElems, vectorWidth);
        const auto maxGrainsPerWorker =
            std::max(gccs::ceildiv(numGrains, numWorkerContexts), 2u);
        const auto maxElemsPerWorker =
            std::min(numElems, maxGrainsPerWorker * vectorWidth);
        const auto cycles =
            getFill1DCycleEstimate(targetParams, type, maxElemsPerWorker);
        return popsolver::DataType{cycles * numWorkerContexts};
      },
      debugPrefix + ".cycles");
  return e;
}

} // end namespace modelling
} // end namespace popops
