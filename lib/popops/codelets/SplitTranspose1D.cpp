// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
static constexpr auto SHORT_SPAN = VectorLayout::SHORT_SPAN;

namespace popops {

#include "inlineAssemblerTranspose.hpp"

template <typename T>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] SplitTranspose1D
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
  SplitTranspose1D();

  Input<Vector<T, COMPACT_PTR, 8>> src;
  Output<Vector<T, COMPACT_PTR, 8>> dst;
  const unsigned short numSrcRowsD4Or8;
  const unsigned short numSrcColumnsD4Or8;
  Input<Vector<unsigned short, SHORT_SPAN>> workList;

  IS_EXTERNAL_CODELET(ext);

  bool compute(unsigned wid) {
    // 4 entries per worker in work list
    if (wid >= workList.size() / 4) {
      return true;
    }
    const unsigned numSrcColumns = numSrcColumnsD4Or8 * subTransposeSize;
    const unsigned numSrcRows = numSrcRowsD4Or8 * subTransposeSize;
    const unsigned matrixSize = numSrcRows * numSrcColumns;
    const unsigned short *workListPtr = &workList[wid * 4];

    // All fields in the work list are divided by subTransposeSize
    unsigned inIndex = workListPtr[0] * subTransposeSize;
    unsigned outIndex = workListPtr[1] * subTransposeSize;
    const unsigned allocatedRowsD4Or8 = workListPtr[2];
    const unsigned allocatedColsD4Or8 = workListPtr[3];

    // the start of the transposition can be obtained by dividing by the matrix
    // size.
    const auto transposition = inIndex / matrixSize;
    const T *srcPtr = &src[transposition * matrixSize];
    T *dstPtr = &dst[transposition * matrixSize];

    // remove full transpositions from in and out index
    inIndex -= transposition * matrixSize;
    outIndex -= transposition * matrixSize;

    // derive the start row and column assigned to this worker within the
    // transposition assigned to it.
    const unsigned startRow =
        (inIndex * numSrcRows - outIndex) / (numSrcColumns * numSrcRows - 1);
    const unsigned startColumn = inIndex - startRow * numSrcColumns;

    srcPtr += startRow * numSrcColumns + startColumn;
    dstPtr += startColumn * numSrcRows + startRow;
    transposeRowsColumnsFast(srcPtr, dstPtr, numSrcRowsD4Or8,
                             numSrcColumnsD4Or8, allocatedRowsD4Or8,
                             allocatedColsD4Or8);
    return true;
  }
};

#ifdef __IPU__
#if __IPU_ARCH_VERSION__ == 21

template class SplitTranspose1D<quarter>;

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

template class SplitTranspose1D<half>;
template class SplitTranspose1D<unsigned short>;
template class SplitTranspose1D<short>;

} // end namespace popops
