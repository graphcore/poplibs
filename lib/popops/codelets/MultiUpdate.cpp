// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "MultiSliceUpdateCommon.hpp"
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
template <typename Type> class MultiUpdate : public MultiVertex {
public:
  MultiUpdate();

  IS_EXTERNAL_CODELET(true);

  Input<Vector<unsigned>> offsets; // in \a baseT
  InOut<Vector<Type, ONE_PTR, 4>> baseT;
  Input<Vector<Type, ONE_PTR>> subT;
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  const unsigned short regionSize; // stride between slices
  const bool indicesAreSorted;     // indices are sorted in increasing order
  const bool splitSingleRegion;    // Use in the case of a single offset and
                                   // alignment constraints are met. Used only
                                   // in assembler
  // in the slice dimension (ceil numBaseElements / numWorkers). Required only
  // by assembler
  const unsigned maxElementsPerWorker;

  bool compute(unsigned wid) {
    if (wid == 0) {
      unsigned offsetIndexBegin = 0;
      unsigned offsetIndexEnd = offsets.size();
      if (indicesAreSorted) {
        offsetIndexBegin =
            lowerBinarySearch(reinterpret_cast<const int *>(&offsets[0]),
                              offsets.size(), baseOffset);
        offsetIndexEnd =
            upperBinarySearch(reinterpret_cast<const int *>(&offsets[0]),
                              offsets.size(), baseOffset + numBaseElements);
      }
      for (unsigned o = offsetIndexBegin; o != offsetIndexEnd; ++o) {
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

#if __IPU_ARCH_VERSION__ == 21
template class MultiUpdate<quarter>;
#endif
} // namespace popops
