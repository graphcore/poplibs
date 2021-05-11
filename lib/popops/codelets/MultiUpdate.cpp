// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popops {

// Update single slices from multiple offsets \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors. the updates are added to the core tensor
// indices that are not within the range of [baseOffset,
// baseOffset + numBaseElements) are ignored.
template <typename Type> class MultiUpdate : public Vertex {
public:
  MultiUpdate();

  IS_EXTERNAL_CODELET(true);

  Input<Vector<unsigned>> offsets; // in \a baseT
  InOut<Vector<Type, ONE_PTR>> baseT;
  Input<Vector<Type, ONE_PTR>> subT;
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  const unsigned short regionSize; // stride between slices

  bool compute() {
    for (unsigned o = 0; o != offsets.size(); ++o) {
      auto baseIdx = offsets[o];

      // the assembly uses this same logic here but without bounds checks on
      // baseIdx for speed reasons so assert it here instead.
      assert(baseIdx < (1 << 31));
      assert(numBaseElements < (1 << 31));
      baseIdx -= baseOffset;
      if (baseIdx >= numBaseElements) {
        // this slice is not a part of baseT so we can skip it.
        continue;
      }

      for (unsigned e = 0; e != regionSize; ++e) {
        baseT[baseIdx * regionSize + e] = subT[o * regionSize + e];
      }
    }
    return true;
  }
};

template class MultiUpdate<float>;
template class MultiUpdate<half>;
template class MultiUpdate<int>;
template class MultiUpdate<unsigned>;
template class MultiUpdate<bool>;
template class MultiUpdate<char>;
template class MultiUpdate<unsigned char>;
template class MultiUpdate<signed char>;

} // namespace popops
