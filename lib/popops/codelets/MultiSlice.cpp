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

// Copy single slices from multiple offsets \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
template <typename Type> class MultiSlice : public MultiVertex {
public:
  MultiSlice();

  IS_EXTERNAL_CODELET(true);

  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR>> baseT;
  Output<Vector<Type, ONE_PTR>> subT;
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  const unsigned short regionSize; // stride between slices
  const bool indicesAreSorted;     // indices are sorted in increasing order
  // (ceil numOffsets / numWorkers). Required only
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
          subT[o * regionSize + e] = baseT[baseIdx * regionSize + e];
        }
      }
    }
    return true;
  }
};
template class MultiSlice<float>;
template class MultiSlice<half>;
template class MultiSlice<int>;
template class MultiSlice<unsigned>;
template class MultiSlice<bool>;
template class MultiSlice<char>;
template class MultiSlice<unsigned char>;
template class MultiSlice<signed char>;

} // namespace popops
