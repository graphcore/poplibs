// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplinCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>
#include <poplin/codelets.hpp>

namespace poplin {

const std::vector<std::string> winogradCodelets = {
    {"poplin::WgdKernelTransform<float,4,4,3,3>",
     "poplin::WgdKernelTransform<half,4,4,3,3>", "poplin::WgdPartials<float>",
     "poplin::WgdPartials<half>", "poplin::WgdReduce<float,4,4>",
     "poplin::WgdReduce<half,4,4>",
     "poplin::WgdInverseTransform<float,4,4,3,3>",
     "poplin::WgdInverseTransform<half,4,4,3,3>",
     "poplin::WgdConvComplete<float>", "poplin::WgdConvComplete<half>"}};

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("poplin", "poplin.gp", loc));
  poplibs::registerPerfFunctions(graph, makePerfFunctionTable());

  // The winograd codelets are not currently supported and do not have correct
  // cycle estimators.
  auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                          const poplar::Target &device) {
    return std::uint64_t(0);
  };
  for (const auto &codelet : winogradCodelets) {
    graph.registerPerfEstimator(codelet, zeroEstimator);
  }
}

} // namespace poplin
