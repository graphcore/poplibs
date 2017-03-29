#include "popstd/Regroup.hpp"
#include "util/gcd.hpp"
#include "popstd/VertexTemplates.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;

namespace popstd {

Tensor
regroup(Tensor t, unsigned outerDim, unsigned innerDim, unsigned newGroupSize) {
  assert(innerDim != outerDim);
  assert(innerDim < t.rank());
  assert(outerDim < t.rank());
  std::vector<std::size_t> shape = t.shape();
  unsigned numElements = shape[outerDim] * shape[innerDim];
  assert(numElements % newGroupSize == 0);
  std::vector<unsigned> shuffle1;
  std::vector<std::size_t> reshapeDims;
  shuffle1.reserve(shape.size());
  reshapeDims.reserve(shape.size());
  for (unsigned dim = 0; dim != shape.size(); ++dim) {
    if (dim == innerDim)
      continue;
    if (dim == outerDim)
      continue;
    shuffle1.push_back(dim);
    reshapeDims.push_back(shape[dim]);
  }
  shuffle1.push_back(outerDim);
  reshapeDims.push_back(numElements / newGroupSize);
  shuffle1.push_back(innerDim);
  reshapeDims.push_back(newGroupSize);
  std::vector<unsigned> shuffle2(shape.size());
  for (unsigned dim = 0; dim != shape.size(); ++dim) {
    shuffle2[shuffle1[dim]] = dim;
  }
  return t.dimShuffle(shuffle1).reshape(reshapeDims).dimShuffle(shuffle2);
}

Tensor
regroup(Tensor in, unsigned outChansPerGroup) {
  auto rank = in.rank();
  assert(rank == 4 || rank == 5);
  return regroup(in, rank - 4, rank - 1, outChansPerGroup);
}

} // end namespace popstd
