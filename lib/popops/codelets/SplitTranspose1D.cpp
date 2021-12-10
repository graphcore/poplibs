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

template <typename T>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] SplitTranspose1D
    : public MultiVertex {
public:
  SplitTranspose1D();

  Input<Vector<T, COMPACT_PTR, 8>> src;
  Output<Vector<T, COMPACT_PTR, 8>> dst;
  const unsigned short numSrcRowsD4;
  const unsigned short numSrcColumnsD4;
  Input<Vector<unsigned short, SHORT_SPAN>> workList;

  IS_EXTERNAL_CODELET(true);

  bool compute(unsigned wid) {
    // 4 entries per worker in work list
    if (wid >= workList.size() / 4) {
      return true;
    }
    const unsigned numSrcColumns = numSrcColumnsD4 * 4;
    const unsigned numSrcRows = numSrcRowsD4 * 4;
    const unsigned matrixSize = numSrcRows * numSrcColumns;
    const unsigned short *workListPtr = &workList[wid * 4];

    // All fields in the work list are divided by 4
    unsigned inIndex = workListPtr[0] * 4;
    unsigned outIndex = workListPtr[1] * 4;
    const unsigned allocatedRows = workListPtr[2] * 4;
    const unsigned allocatedCols = workListPtr[3] * 4;

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

    for (unsigned x = startRow; x != startRow + allocatedRows; ++x) {
      for (unsigned y = startColumn; y != startColumn + allocatedCols; ++y) {
        dstPtr[y * numSrcRows + x] = srcPtr[x * numSrcColumns + y];
      }
    }
    return true;
  }
};

template class SplitTranspose1D<half>;
template class SplitTranspose1D<unsigned short>;
template class SplitTranspose1D<short>;

} // end namespace popops
