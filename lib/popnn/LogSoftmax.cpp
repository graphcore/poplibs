// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "NonLinearityInternal.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/ElementWise.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "popops/Reduce.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <cmath>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

namespace logging = poplibs_support::logging;

namespace {

// computes log of the softmax along the innermost dimension
Tensor logSoftmaxImpl(Graph &graph, Tensor t, bool inPlace, Sequence &prog,
                      const DebugNameAndId &dnai) {
  const std::string fnStr = "LogSoftmax";
  const auto dType = t.elementType();
  logging::popnn::info("logSoftmax t={}, name={}", t.shape(),
                       dnai.getPathName() + fnStr);
  if (t.rank() < 1) {
    throw poplibs_error("input tensor to logSoftmax must have at least 1 "
                        "dimension");
  }

  const bool expandDimension = t.rank() == 1;
  if (expandDimension) {
    t = t.expand({0});
  }

  // Switch innermost dimension to outer as softmax is done over it
  const auto rank = t.rank();
  auto tShuf = t.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  const auto innerDimSize = t.dim(rank - 1);

  bool needsCopy = !inPlace;
  auto max = popops::reduce(graph, tShuf, {0}, popops::Operation::MAX, prog,
                            {dnai, fnStr})
                 .expand({0})
                 .broadcast(innerDimSize, 0);

  if (needsCopy) {
    tShuf = popops::sub(graph, tShuf, max, prog, {dnai, fnStr});
  } else {
    popops::subInPlace(graph, tShuf, max, prog, {dnai, fnStr});
  }

  auto tExp = popops::exp(graph, tShuf, prog, {dnai, fnStr});

  // For half types we can improve accuracy by scaling the result so that the
  // sum of the values is max half instead of 1.0.
  auto sumF = popops::reduce(graph, tExp, poplar::FLOAT, {0},
                             popops::Operation::ADD, prog, {dnai, fnStr});

  // Keep at higher precision though is strictly not necessary as the input is
  // always guaranteed to be > 1. The tensor dimension is already smaller than
  // the original; so the cost of doing this in higher precision should be low
  // as long as the outer dimension is not large compared to the number of tiles
  // this is split over.
  auto sum = popops::map(graph, expr::Cast(expr::Log(expr::_1), dType), {sumF},
                         prog, {dnai, fnStr});

  // Do not fuse with above expression to allow efficient use of broadcast
  // vertices.
  popops::subInPlace(graph, tShuf, sum, prog, {dnai, fnStr});

  // Shuffle dimensions back to original ordering and return.
  // If inPlace == true then this is the same as the original tensor.
  auto tRet = tShuf.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(tRet.shape() == t.shape());
  return expandDimension ? tRet.squeeze({0}) : tRet;
}

} // end anonymous namespace

namespace popnn {

void logSoftmaxInPlace(Graph &graph, Tensor t, Sequence &prog,
                       const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t));

  logSoftmaxImpl(graph, t, true, prog, {di});
}

Tensor logSoftmax(Graph &graph, Tensor t, Sequence &prog,
                  const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t));

  auto output = logSoftmaxImpl(graph, t, false, prog, {di});
  di.addOutput(output);
  return output;
}

} // end namespace popnn
