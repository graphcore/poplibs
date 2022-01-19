// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/logging.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"

#include <cassert>

using namespace poplar;

namespace popnn {

void checkTensorShape(Tensor acts) {
  const auto rank = acts.rank();
  if (rank < 2) {
    throw poputil::poplibs_error("Norm supported for tensors of rank > 1");
  }
}

void checkNormTensorTypes(const Type &inputType, const Target &target,
                          Type &partialsType) {
  if (target.getTypeSize(partialsType) < target.getTypeSize(inputType)) {
    poplibs_support::logging::popops::warn(
        "Ignoring normalisation partialsType ({})"
        " which is smaller than the input/output type ({})",
        partialsType.toString(), inputType.toString());
    partialsType = inputType;
  }
}

Tensor preProcessNormActs(const Tensor &acts) {
  return acts.rank() == 2 ? acts.expand({2}) : acts;
}

Tensor postProcessNormActs(const Tensor &acts, unsigned originalActsRank) {
  if (originalActsRank == 2) {
    assert(acts.rank() == 3 && acts.dim(2) == 1);
    return acts.squeeze({2});
  }
  return acts;
}
} // namespace popnn
