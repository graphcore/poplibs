// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace poplin {

template <typename T>
class [[poplar::constraint("elem(**src) != elem(**dst)")]] Transpose2d
    : public Vertex {
public:
  Transpose2d();

  Vector<Input<Vector<T, ONE_PTR, 8>>> src;
  Vector<Output<Vector<T, ONE_PTR, 8>>, ONE_PTR> dst;
  // TODO: T12869 Specialize the vertex based on the value of this field to
  // avoid extra memory usage.
  const unsigned short numSrcRows;
  const unsigned short numSrcColumns;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const auto numTranspositions = src.size();
    for (unsigned i = 0; i != numTranspositions; ++i) {
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dst[i][x * numSrcRows + y] = src[i][y * numSrcColumns + x];
        }
      }
    }
    return true;
  }
};

template class Transpose2d<float>;
template class Transpose2d<half>;

} // end namespace poplin
