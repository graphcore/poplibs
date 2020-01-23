// Copyright (c) Graphcore Ltd, All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

namespace popops {

// Add single slices from multiple offsets \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors. the updates are added to the core tensor
// indices that are not within the range of [baseOffset,
// baseOffset + numBaseElements) are ignored.
template <typename Type, bool subwordWritesSupported>
class MultiUpdateAdd : public Vertex {
public:
  MultiUpdateAdd();

  static const bool isExternal =
      std::is_same<Type, float>::value ||
      (std::is_same<Type, half>::value && !subwordWritesSupported);
  IS_EXTERNAL_CODELET(isExternal);
  Input<Type> scale;
  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR, 4>> subT;
  InOut<Vector<Type, COMPACT_PTR, 4>> baseT;
  const unsigned short regionSize; // stride between slices
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension

  bool compute() {
    // Perform calculation in single precision for half data so that stochastic
    // rounding will occur. TODO: T12921 Replace with a mix.
    // For halves, accumulate in float so that stochastic rounding will take
    // effect.
    using ScaleType =
        std::conditional_t<std::is_same<Type, half>::value, float, Type>;
    // load scale once
    const auto scaleL = ScaleType(*scale);

    unsigned restrictedRegionSize = regionSize;
    if (std::is_same<Type, half>::value && !subwordWritesSupported)
      restrictedRegionSize &= ~0x1;
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

      for (unsigned e = 0; e != restrictedRegionSize; ++e) {
        const auto addend =
            scaleL * ScaleType(subT[o * restrictedRegionSize + e]);
        baseT[baseIdx * restrictedRegionSize + e] += addend;
      }
    }
    return true;
  }
};

template class MultiUpdateAdd<half, true>;
template class MultiUpdateAdd<half, false>;
template class MultiUpdateAdd<float, false>;
template class MultiUpdateAdd<int, false>;
template class MultiUpdateAdd<unsigned, false>;

} // namespace popops
