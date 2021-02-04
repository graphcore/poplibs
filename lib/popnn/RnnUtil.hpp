// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef popnn_RnnUtil_hpp
#define popnn_RnnUtil_hpp

#include <boost/optional.hpp>
#include <cassert>
#include <cstdint>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Rearrange.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/OptionParsing.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VarStructure.hpp>
#include <poputil/VertexTemplates.hpp>

namespace popnn {
namespace Rnn {

// Flatten a 3D tensor to a 2D tensor such that the innermost dimension is the
// product of outputs(or inputs) and units
inline poplar::Tensor flattenUnits(const poplar::Tensor &t) {
  return t.dimShuffle({1, 0, 2}).reshape({t.dim(1), t.dim(0) * t.dim(2)});
}

// unflatten a 2D tensor which has units flattened in it's innermost dimension.
// The resultant 3D tensor view has the unit dimension as the outermost
// dimension
inline poplar::Tensor unflattenUnits(const poplar::Tensor &t, size_t num_unit) {
  return t.reshape({t.dim(0), num_unit, t.dim(1) / num_unit})
      .dimShuffle({1, 0, 2});
}

// Given a tensor of rank 2 that is laid out in memory such that groups of
// elements in the outermost dimension are contiguous try to rearrange it
// so groups of elements in the innermost dimension are contiguous.
// Returns either the original tensor or a copy of the original tensor
// with the same shape but a updated memory layout.
inline poplar::Tensor tryGroupedPartialTranspose(
    poplar::Graph &graph, poplar::Tensor t, unsigned desiredGrouping,
    poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  unsigned outerSize = t.dim(0);
  unsigned innerSize = t.dim(1);
  if (innerSize % desiredGrouping) {
    desiredGrouping = gcd(innerSize, desiredGrouping);
  }
  if (desiredGrouping == 1) {
    return t;
  }
  const auto outerGrouping =
      poputil::detectInnermostGrouping(graph, t.transpose());
  if (outerGrouping == 1) {
    return t;
  }
  auto groupedView = t.reshape({outerSize / outerGrouping, outerGrouping,
                                innerSize / desiredGrouping, desiredGrouping})
                         .dimShuffle({0, 2, 3, 1});
  auto cs = graph.addComputeSet({dnai, "groupedPartialTranspose"});
  auto partiallyTransposed =
      popops::rearrange::partialTranspose(graph, groupedView, cs, {dnai});
  prog.add(poplar::program::Execute(cs, {dnai}));
  return partiallyTransposed.dimShuffle({0, 2, 1, 3})
      .reshape({outerSize, innerSize});
}

} // namespace Rnn
} // namespace popnn

#endif // #ifndef popnn_RnnUtil_hpp
