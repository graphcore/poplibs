// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

namespace popops {

template <typename T>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Transpose1D
    : public MultiVertex {
public:
  Transpose1D();

  Input<Vector<T, COMPACT_PTR, 8>> src;
  Output<Vector<T, COMPACT_PTR, 8>> dst;
  const unsigned short numSrcRowsD4;
  const unsigned short numSrcColumnsD4;
  // Each worker will process a contiguous span of matrices (each one comprises
  // 'matrixSize' contiguous elements) from 'data[offs]'.
  // The first 'workerCount' workers (from wid=0 to wid=workerCount-1) will
  // process 'numTranspositions' matrices ('numTranspositions' always > 0), and
  // (6-workerCount) workers (from wid=workerCount to wid=5) will process
  // (numTranspositions-1) matrices.
  // Note that (6-workerCount) and/or (numTranspositions-1) could be zero.
  const unsigned short numTranspositions;
  const unsigned short workerCount;

  IS_EXTERNAL_CODELET(true);

  bool compute(unsigned wid) {
    const unsigned numSrcColumns = numSrcColumnsD4 * 4;
    const unsigned numSrcRows = numSrcRowsD4 * 4;
    const unsigned matrixSize = numSrcRows * numSrcColumns;

    unsigned n = numTranspositions - 1;
    unsigned offs = wid * n + ((wid < workerCount) ? wid : workerCount);
    n += (wid < workerCount);

    T *srcPtr = &src[offs * matrixSize];
    T *dstPtr = &dst[offs * matrixSize];

    for (unsigned t = 0; t != n; ++t) {
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dstPtr[t * matrixSize + x * numSrcRows + y] =
              srcPtr[t * matrixSize + y * numSrcColumns + x];
        }
      }
    }
    return true;
  }
};

template class Transpose1D<half>;
template class Transpose1D<unsigned short>;
template class Transpose1D<short>;

} // end namespace popops
