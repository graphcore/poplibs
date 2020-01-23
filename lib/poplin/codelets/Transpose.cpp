// Copyright (c) Graphcore Ltd, All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

#if defined(VECTOR_AVAIL_SCALED_PTR64)
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::SCALED_PTR64;
#else
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::ONE_PTR;
#endif

namespace poplin {

template <typename T>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Transpose
    : public Vertex {
public:
  Transpose();

  Input<Vector<T, PTR_ALIGN64, 8>> src;
  Output<Vector<T, PTR_ALIGN64, 8>> dst;
  const unsigned short numSrcRowsD4;
  const unsigned short numSrcColumnsD4;
  const unsigned short numTranspositionsM1;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const unsigned numTranspositions = numTranspositionsM1 + 1;
    const unsigned numSrcColumns = numSrcColumnsD4 * 4;
    const unsigned numSrcRows = numSrcRowsD4 * 4;
    for (unsigned t = 0; t != numTranspositions; ++t) {
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dst[t * numSrcRows * numSrcColumns + x * numSrcRows + y] =
              src[t * numSrcRows * numSrcColumns + y * numSrcColumns + x];
        }
      }
    }
    return true;
  }
};

template class Transpose<half>;

} // end namespace poplin
