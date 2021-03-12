// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popops/AllTrue.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include "popops/Reduce.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/exceptions.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

Tensor allTrue(Graph &graph, Tensor in, Sequence &prog,
               const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(in));

  const auto inType = in.elementType();

  if (inType != BOOL) {
    throw poputil::poplibs_error(
        "Operation allTrue only takes boolean tensors");
  }
  auto inFlat = in.flatten();
  auto output = reduce(graph, inFlat, inType, {0},
                       popops::Operation::LOGICAL_AND, prog, {di});
  di.addOutput(output);
  return output;
}

} // end namespace popops
