// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef _MultiUpdateOp_hpp_
#define _MultiUpdateOp_hpp_

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include "popops/OperationDef.hpp"

#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

namespace popops {

template <typename T> T updateOp(Operation op, T x, T y) {
  switch (op) {
  case popops::Operation::ADD:
    return x + y;
  case popops::Operation::MAX:
    return x > y ? x : y;
  default:
    assert(0 && "Operation not supported");
    return x;
  }
}

} // namespace popops

#endif // #ifndef _MultiUpdateOp_hpp_
