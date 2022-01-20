// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popops {

#include "inlineAssemblerTranspose.hpp"

template <typename T>
class [[poplar::constraint("elem(**src) != elem(**dst)")]] Transpose2D
    : public Vertex {

#if __IPU_ARCH_VERSION__ == 21
  static const bool ext = !std::is_same<T, quarter>::value;
#else
  static const bool ext = true;
#endif //__IPU_ARCH_VERSION__

public:
  Transpose2D();

  Vector<Input<Vector<T, ONE_PTR, 8>>> src;
  Vector<Output<Vector<T, ONE_PTR, 8>>, ONE_PTR> dst;
  // TODO: T12869 Specialize the vertex based on the value of this field to
  // avoid extra memory usage.
  const unsigned short numSrcRows;
  const unsigned short numSrcColumns;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    const auto numTranspositions = src.size();
    for (unsigned i = 0; i != numTranspositions; ++i) {
      transposeRowsColumns(&src[i][0], &dst[i][0], numSrcRows, numSrcColumns);
    }
    return true;
  }
};

#ifdef __IPU__
#if __IPU_ARCH_VERSION__ == 21

template class Transpose2D<quarter>;

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

template class Transpose2D<float>;
template class Transpose2D<unsigned int>;
template class Transpose2D<int>;
template class Transpose2D<half>;
template class Transpose2D<unsigned short>;
template class Transpose2D<short>;

} // end namespace popops
