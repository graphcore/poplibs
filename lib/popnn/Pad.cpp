#include "Pad.hpp"

#include <cstdlib>
#include <poplar/Graph.hpp>
#include "Zero.hpp"

using namespace poplar;
using namespace poplar::program;

static void validatePadArgs(poplar::Tensor in,
                            const std::vector<std::size_t> &dims,
                            const std::vector<std::size_t> &beforePadding) {
  // Check the number of dimensions match.
  const auto numDims = in.getDimensionality();
  if (dims.size() != numDims) {
    std::abort();
  }
  if (beforePadding.size() != numDims) {
    std::abort();
  }
  // Check the size of the output is greater than the size of the input.
  for (unsigned i = 0; i != numDims; ++i) {
    if (in.dim(i) + beforePadding[i] > dims[i])
      std::abort();
  }
}

poplar::Tensor
pad(poplar::Graph &graph, poplar::Tensor t,
    const std::vector<std::size_t> &dims,
    const std::vector<std::size_t> &beforePadding) {
  const auto type = graph.getTensorElementType(t);
  validatePadArgs(t, dims, beforePadding);
  for (unsigned i = 0; i < dims.size(); ++i) {
    if (dims[i] == t.dim(i))
      continue;
    auto beforePadDims = t.dims();
    beforePadDims[i] = beforePadding[i];
    auto beforePadding = graph.addConstantTensor(type, beforePadDims, 0);
    t = concat(beforePadding, t, i);
    auto afterPadDims = t.dims();
    afterPadDims[i] = dims[i] - t.dim(i);
    auto afterPadding = graph.addConstantTensor(type, afterPadDims, 0);
    t = concat(t, afterPadding, i);
  }

  return t;
}
