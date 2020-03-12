// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popops/AllTrue.hpp"

#include "popops/Reduce.hpp"
#include "poputil/exceptions.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

Tensor allTrue(Graph &graph, Tensor in, Sequence &prog,
               const std::string &debugPrefix) {
  const auto inType = in.elementType();

  if (inType != BOOL) {
    throw poputil::poplibs_error(
        "Operation allTrue only takes boolean tensors");
  }
  auto inFlat = in.flatten();
  return reduce(graph, inFlat, inType, {0}, popops::Operation::LOGICAL_AND,
                prog, debugPrefix);
}

} // end namespace popops
