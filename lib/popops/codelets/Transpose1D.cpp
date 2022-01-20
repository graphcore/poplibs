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

#include "inlineAssemblerTranspose.hpp"

template <typename T>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Transpose1D
    : public MultiVertex {

#if __IPU_ARCH_VERSION__ == 21
  static constexpr bool ext = !std::is_same<T, quarter>::value;
  static constexpr unsigned subTransposeSize =
      std::is_same<T, quarter>::value ? 8 : 4;
#else
  static constexpr bool ext = true;
  static constexpr unsigned subTransposeSize = 4;
#endif //__IPU_ARCH_VERSION__

public:
  Transpose1D();

  Input<Vector<T, COMPACT_PTR, 8>> src;
  Output<Vector<T, COMPACT_PTR, 8>> dst;
  const unsigned short numSrcRowsD4Or8;
  const unsigned short numSrcColumnsD4Or8;
  // Each worker will process a contiguous span of matrices (each one comprises
  // 'matrixSize' contiguous elements) from 'data[offs]'.
  // The first 'workerCount' workers (from wid=0 to wid=workerCount-1) will
  // process 'numTranspositions' matrices ('numTranspositions' always > 0), and
  // (6-workerCount) workers (from wid=workerCount to wid=5) will process
  // (numTranspositions-1) matrices.
  // Note that (6-workerCount) and/or (numTranspositions-1) could be zero.
  const unsigned short numTranspositions;
  const unsigned short workerCount;

  IS_EXTERNAL_CODELET(ext);

  bool compute(unsigned wid) {
    const unsigned matrixSize = numSrcRowsD4Or8 * numSrcColumnsD4Or8 *
                                subTransposeSize * subTransposeSize;

    unsigned n = numTranspositions - 1;
    unsigned offs = wid * n + ((wid < workerCount) ? wid : workerCount);
    n += (wid < workerCount);

    const T *srcPtr = &src[offs * matrixSize];
    T *dstPtr = &dst[offs * matrixSize];

    for (unsigned t = 0; t != n; ++t) {
      transposeRowsColumnsFast(srcPtr + t * matrixSize, dstPtr + t * matrixSize,
                               numSrcRowsD4Or8, numSrcColumnsD4Or8);
    }
    return true;
  }
};

#ifdef __IPU__
#if __IPU_ARCH_VERSION__ == 21

template class Transpose1D<quarter>;

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

template class Transpose1D<half>;
template class Transpose1D<unsigned short>;
template class Transpose1D<short>;

} // end namespace popops
