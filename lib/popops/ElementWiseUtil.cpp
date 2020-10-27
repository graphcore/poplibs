// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/ElementWiseUtil.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"

using namespace poplar;
using namespace poputil;
using namespace poplibs_support;

namespace popops {

// The aim of this function is to create an output that best maintains the
// layout of its input through an element-wise op. Primarily this function
// is motivated by cases where a tensor has a binary op performed on it
// with a constant or a broadcasted tensor where we should pick the
// other tensor without aliases/constants to clone as the output.
//
// Where there are multiple suitable inputs (which don't contain aliases
// or constants) we have a go at choosing which one would be best to
// clone for the output. The current order of preference is:
//
//  * Tensor with the minimum maximum elements per-tile.
//  * Tensor occupying the most tiles.
//  * Tensor with the fewest contiguous regions.
//
// This is not especially well founded and could do with a good test
// case to decide which tensor we would really want to choose in this
// case. For now mostly there isn't a choice between multiple or the
// tensors we choose between are laid out pretty much identically
// so we just pick the first anyway.
//
Tensor createOutputForElementWiseOp(Graph &graph,
                                    const std::vector<Tensor> &inputs,
                                    const Type &outputType,
                                    const poplar::DebugContext &debugContext) {
  const auto debugName = debugContext.getPathName();
  if (inputs.size() < 1) {
    throw poplibs_error("createOutputForElementWiseOp: Must provide at "
                        "least one input tensor as a reference but none "
                        "were given");
  }

  for (std::size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i - 1].shape() != inputs[i].shape()) {
      throw poplibs_error("createOutputForElementWiseOp '" + debugName +
                          "': "
                          "Shapes of input tensors do not match");
    }
  }

  std::vector<unsigned> tilesOccupied(inputs.size());
  std::vector<unsigned> numRegions(inputs.size());
  std::vector<size_t> maxTileElements(inputs.size());
  std::vector<bool> parallelWriteable(inputs.size());

  // Gather info on distribution of inputs.
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].isParallelWriteable())
      continue;
    parallelWriteable[i] = true;
    // Simplify the input so that when we get the tile mapping we
    // have the intersection of the tile mapping and the contiguous
    // regions of the tensor.
    auto inputSimplified = inputs[i].flatten();
    graph.reorderToSimplify(&inputSimplified, {});
    const auto mapping = graph.getTileMapping(inputSimplified);
    for (const auto &tileMapping : mapping) {
      if (!tileMapping.empty()) {
        tilesOccupied[i]++;
        numRegions[i] += tileMapping.size();
        const std::size_t tileElements = std::accumulate(
            tileMapping.begin(), tileMapping.end(), std::size_t(0),
            [](std::size_t t, const Interval &i) { return t + i.size(); });
        maxTileElements[i] = std::max(maxTileElements[i], tileElements);
      }
    }
  }

  // If an input tensor has a suitable distribution then clone it to
  // create the output.
  int best = -1;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    // If not parallel writeable, either this has constant elements with
    // indeterminate mapping, or some elements alias others, and likely
    // the resulting tensor will not be well distributed.
    if (!parallelWriteable[i])
      continue;

    // Select the tensor with the minimum maximum tile elements
    if (best < 0 || maxTileElements[i] < maxTileElements[best]) {
      best = i;
    } else if (maxTileElements[i] == maxTileElements[best]) {
      // If both have the same maximum, select the tensor which is spread onto
      // the most tiles, or if two tensors share the same number of tiles, then
      // select the one which has the fewest overall regions
      if ((tilesOccupied[i] > tilesOccupied[best]) ||
          (tilesOccupied[i] == tilesOccupied[best] &&
           numRegions[i] < numRegions[best])) {
        best = i;
      }
    }
  }

  // When there's no suitable input put a scalar output with the first input and
  // don't warn about it.
  if (best < 0 && inputs[0].numElements() == 1) {
    best = 0;
  }

  // Clone output either based on a suitable input tensor or just based on
  // the first given tensor with a linear mapping applied (with given
  // restrictions on grain size and no. of grains per-tile).
  Tensor output;
  if (best >= 0) {
    output = graph.clone(outputType, inputs[best], debugName);
  } else {
    logging::popops::warn(
        "createOutputForElementWiseOp '{}' ({}): No suitable input "
        "found, creating new variable with linear tile mapping",
        debugName, inputs[0].shape());
    output = graph.addVariable(outputType, inputs[0].shape(), debugName);
    poputil::mapTensorLinearly(graph, output);
  }
  return output;
}

std::vector<Interval> cutRegionSection(const std::vector<Interval> &region,
                                       const unsigned secLength,
                                       unsigned &index, unsigned &offset,
                                       unsigned &regionIndex) {
  std::vector<Interval> section;

  // Besides the very first interval which may be a partial interval, copy as
  // many whole intervals as possible.
  auto start = offset;
  offset += secLength;
  while ((index < region.size()) && (offset >= region[index].size())) {
    offset -= region[index].size();
    section.emplace_back(start + region[index].begin(), region[index].end());
    index++;
    start = 0;
  }

  // Include the final partial interval if the section does not finish on an
  // interval boundary.
  if ((index < region.size()) && (offset > 0)) {
    section.emplace_back(start + region[index].begin(),
                         offset + region[index].begin());
  }

  if (index == region.size()) {
    regionIndex++;
    index = 0;
    assert(offset == 0);
  }

  return section;
}

} // end namespace popops
