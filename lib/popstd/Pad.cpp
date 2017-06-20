#include "popstd/Pad.hpp"

#include <cstdlib>
#include <poplar/Graph.hpp>
#include "popstd/Zero.hpp"
#include "popstd/exceptions.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popstd {

static void validatePadArgs(poplar::Tensor in,
                            const std::vector<std::ptrdiff_t> &paddingLower,
                            const std::vector<std::ptrdiff_t> &paddingUpper) {
  // Check the ranks match.
  const auto rank = in.rank();
  if (paddingLower.size() != rank ||
      paddingUpper.size() != rank) {
    std::abort();
  }
  for (unsigned i = 0; i != rank; ++i) {
    if (paddingLower[i] + static_cast<int>(in.dim(i)) < 0 ||
        paddingUpper[i] + static_cast<int>(in.dim(i)) < 0 ||
        paddingLower[i] + paddingUpper[i] + static_cast<int>(in.dim(i)) < 0) {
      throw popstd::poplib_error("Cannot truncate dimension by more than the "
                                 "size of the dimension");
    }
  }
}

poplar::Tensor
pad(poplar::Graph &graph, poplar::Tensor t,
    const std::vector<std::ptrdiff_t> &paddingLower,
    const std::vector<std::ptrdiff_t> &paddingUpper) {
  const auto type = t.elementType();
  validatePadArgs(t, paddingLower, paddingUpper);
  for (unsigned i = 0; i < t.rank(); ++i) {
    if (paddingLower[i] > 0) {
      auto paddingShape = t.shape();
      paddingShape[i] = paddingLower[i];
      auto padding = graph.addConstantTensor(type, paddingShape, 0);
      t = concat(padding, t, i);
    } else if (paddingLower[i] < 0) {
      t = t.slice(-paddingLower[i], t.dim(i), i);
    }
    if (paddingUpper[i] > 0) {
      auto paddingShape = t.shape();
      paddingShape[i] = paddingUpper[i];
      auto padding = graph.addConstantTensor(type, paddingShape, 0);
      t = concat(t, padding, i);
    } else if (paddingUpper[i] < 0) {
      t = t.slice(0, t.dim(i) + paddingUpper[i], i);
    }
  }
  return t;
}

poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t, std::ptrdiff_t paddingLower,
    std::ptrdiff_t paddingUpper, unsigned dim) {
  if (dim >= t.rank()) {
    std::abort();
  }
  std::vector<std::ptrdiff_t> paddingLowerVector(t.rank());
  paddingLowerVector[dim] = paddingLower;
  std::vector<std::ptrdiff_t> paddingUpperVector(t.rank());
  paddingUpperVector[dim] = paddingUpper;
  return pad(graph, t, paddingLowerVector, paddingUpperVector);
}

} // end namespace popstd
