#include "Pad.hpp"

#include <cstdlib>
#include <poplar/Graph.hpp>
#include "Zero.hpp"

using namespace poplar;
using namespace poplar::program;

static void validatePadArgs(poplar::Graph &graph, poplar::Tensor in,
                            poplar::Tensor out,
                            const std::vector<std::size_t> &beforePadding) {
  // Check the number of dimensions match.
  const auto numDims = in.getDimensionality();
  if (out.getDimensionality() != numDims) {
    std::abort();
  }
  // Check the data types match.
  if (graph.getTensorElementType(in) != graph.getTensorElementType(out)) {
    std::abort();
  }
  // Check the size of the output is greater than the size of the input.
  for (unsigned i = 0; i != numDims; ++i) {
    if (in.dim(i) + beforePadding[i] > out.dim(i))
      std::abort();
  }
}

poplar::program::Program
pad(poplar::Graph &graph,
    poplar::Tensor in, poplar::Tensor out,
    const std::vector<std::size_t> &beforePadding,
    const std::vector<unsigned> &outTileMapping) {
  validatePadArgs(graph, in, out, beforePadding);
  if (in.dims() == out.dims()) {
    return Copy(out, in);
  }
  const auto numDims = in.getDimensionality();
  const auto &copyDstBegin = beforePadding;
  std::vector<std::size_t> copyDstEnd(in.getDimensionality());
  for (unsigned i = 0; i != numDims; ++i) {
    copyDstEnd[i] = copyDstBegin[i] + in.dim(i);
  }
  // Zero the output tensor and then copy the input to the non-padding
  // slice of the output.
  // TODO It would be more efficient to only zero the padding.
  Sequence prog;
  prog.add(zero(graph, out, outTileMapping));
  prog.add(Copy(out.slice(copyDstBegin, copyDstEnd), in));
  return prog;
}
