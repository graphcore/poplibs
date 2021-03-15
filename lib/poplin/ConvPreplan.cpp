// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "poplin/ConvPreplan.hpp"
#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/Convolution.hpp"
#include "poplin/MatMul.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/exceptions.hpp"

namespace logging = poplibs_support::logging;

namespace poplin {

void preplan(const std::set<ConvPlanParams> &convs,
             const std::set<MatMulPlanParams> &matmuls, PlanningCache &cache) {
  POPLIN_TRACEPOINT();
  if (convs.empty() && matmuls.empty())
    return;

  MatMulToConvOptions matmulOptsPtrToConvOpts;
  auto matmulConvs = matMulGetConvPlanParams(matmuls, matmulOptsPtrToConvOpts);

  std::set<poplin::ConvPlanKey> convsImpl;
  for (auto &conv : convs) {
    const ConvOptions options(*std::get<2>(conv));
    convsImpl.emplace(std::get<1>(conv), options);
  }
  for (auto &conv : matmulConvs) {
    const ConvOptions options(*std::get<2>(conv));
    convsImpl.emplace(std::get<1>(conv), options);
  }
  auto &commonTarget = (convs.size() > 0) ? *std::get<0>(*(convs.cbegin()))
                                          : *std::get<0>(*(matmuls.cbegin()));
  preplanConvolutionsImpl(commonTarget, convsImpl, cache);
}

void preplanMatMuls(const std::set<MatMulPlanParams> &matmuls,
                    matmul::PlanningCache &cache) {
  preplan({}, matmuls, cache.getImpl());
}

void preplanConvolutions(poplar::Graph &graph,
                         const std::set<ConvPlanParams> &convs,
                         PlanningCache &cache) {
  logging::poplin::warn("poplin::preplanConvolution() is deprecated! "
                        "Use poplin::preplan() instead");
  preplan(convs, {}, cache);
}

void preplanConvolutions(const std::set<ConvPlanParams> &convs,
                         PlanningCache &cache) {
  logging::poplin::warn("poplin::preplanConvolution() is deprecated! "
                        "Use poplin::preplan() instead");
  preplan(convs, {}, cache);
}

} // namespace poplin
