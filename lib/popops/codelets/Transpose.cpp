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
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Transpose1DSingleWorker
    : public Vertex {

#if __IPU_ARCH_VERSION__ == 21
  static const bool ext = !std::is_same<T, quarter>::value;
  static constexpr unsigned subTransposeSize =
      std::is_same<T, quarter>::value ? 8 : 4;
#else
  static const bool ext = true;
  static constexpr unsigned subTransposeSize = 4;
#endif //__IPU_ARCH_VERSION__

public:
  Transpose1DSingleWorker();

  Input<Vector<T, COMPACT_PTR, 8>> src;
  Output<Vector<T, COMPACT_PTR, 8>> dst;
  const unsigned short numSrcRowsD4Or8;
  const unsigned short numSrcColumnsD4Or8;
  const unsigned short numTranspositionsM1;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    const unsigned matrixSize = numSrcRowsD4Or8 * numSrcColumnsD4Or8 *
                                subTransposeSize * subTransposeSize;
    const unsigned numTranspositions = numTranspositionsM1 + 1;
    for (unsigned t = 0; t != numTranspositions; ++t) {
      transposeRowsColumnsFast(&src[t * matrixSize], &dst[t * matrixSize],
                               numSrcRowsD4Or8, numSrcColumnsD4Or8);
    }
    return true;
  }
};

#ifdef __IPU__
#if __IPU_ARCH_VERSION__ == 21

template class Transpose1DSingleWorker<quarter>;

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

template class Transpose1DSingleWorker<half>;
template class Transpose1DSingleWorker<unsigned short>;
template class Transpose1DSingleWorker<short>;

} // end namespace popops
