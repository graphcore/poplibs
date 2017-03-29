#include "popstd/Pad.hpp"

#include <cstdlib>
#include <poplar/Graph.hpp>
#include "popstd/Zero.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popstd {

static void validatePadArgs(poplar::Tensor in,
                            const std::vector<std::size_t> &shape,
                            const std::vector<std::size_t> &beforePadding) {
  // Check the ranks match.
  const auto rank = in.rank();
  if (shape.size() != rank) {
    std::abort();
  }
  if (beforePadding.size() != rank) {
    std::abort();
  }
  // Check the size of the output is greater than the size of the input.
  for (unsigned i = 0; i != rank; ++i) {
    if (in.dim(i) + beforePadding[i] > shape[i])
      std::abort();
  }
}

poplar::Tensor
pad(poplar::Graph &graph, poplar::Tensor t,
    const std::vector<std::size_t> &shape,
    const std::vector<std::size_t> &beforePadding) {
  const auto type = graph.getTensorElementType(t);
  validatePadArgs(t, shape, beforePadding);
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (shape[i] == t.dim(i))
      continue;
    auto beforePadShape = t.shape();
    beforePadShape[i] = beforePadding[i];
    auto beforePadding = graph.addConstantTensor(type, beforePadShape, 0);
    t = concat(beforePadding, t, i);
    auto afterPadShape = t.shape();
    afterPadShape[i] = shape[i] - t.dim(i);
    auto afterPadding = graph.addConstantTensor(type, afterPadShape, 0);
    t = concat(t, afterPadding, i);
  }

  return t;
}

poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t, std::size_t newSize,
    std::size_t beforePadding, unsigned dim) {
  if (dim >= t.rank()) {
    std::abort();
  }
  auto shape = t.shape();
  std::vector<std::size_t> beforePaddingVector(t.rank());
  shape[dim] = newSize;
  beforePaddingVector[dim] = beforePadding;
  return pad(graph, t, shape, beforePaddingVector);
}

} // end namespace popstd
