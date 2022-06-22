// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

using namespace poplar;

static constexpr auto COMPACT_DELTAN = poplar::VectorListLayout::COMPACT_DELTAN;

namespace popops {

// Copy each \numSubElements regions from \a in to
// \a out regions [\a offset : \a offset + \a numInElements)
// Where the offset given is larger than numBaseElements, behaviour is not
// properly specified.  Options could be baseSlice=offset % numBaseElements,
// or as implemented if(offset>=numBaseElements) baseSlice=0;
template <typename InType> class DynamicUpdateSlice2D : public Vertex {
public:
  DynamicUpdateSlice2D();

  Input<unsigned> offset; // in out
  // 16bit element shall follow VectorList to compensate packing
  // and make next VectorList 4bytes aligned
  InOut<VectorList<InType, COMPACT_DELTAN>> baseT;
  const unsigned short numSubElements; // in the slice dimension
  Input<VectorList<InType, COMPACT_DELTAN>> subT;
  const unsigned short numRegions;
  const unsigned numBaseElements; // in the slice dimension MSB used to
                                  // indicate if invalid indices must be
                                  // remapped

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const unsigned numBaseElementsActual = numBaseElements & 0x7fffffffu;
    for (unsigned r = 0; r != numRegions; ++r) {
      auto regionSize = baseT[r * numBaseElementsActual].size();
      unsigned baseSlice = offset;
      if (numBaseElements & 0x80000000u) {
        if (baseSlice >= numBaseElementsActual)
          baseSlice = 0;
      } else {
        if (baseSlice >= numBaseElementsActual)
          return true;
      }
      unsigned subIdx = r * numSubElements;

      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto baseIdx = r * numBaseElementsActual + baseSlice;
        for (unsigned e = 0; e != regionSize; e++) {
          baseT[baseIdx][e] = subT[subIdx][e];
        }
        subIdx++;
        baseSlice++;
        if (baseSlice >= numBaseElementsActual)
          baseSlice -= numBaseElementsActual;
      }
    }
    return true;
  }
};
template class DynamicUpdateSlice2D<float>;
template class DynamicUpdateSlice2D<half>;
template class DynamicUpdateSlice2D<int>;
template class DynamicUpdateSlice2D<unsigned>;
template class DynamicUpdateSlice2D<bool>;
template class DynamicUpdateSlice2D<char>;
template class DynamicUpdateSlice2D<unsigned char>;
template class DynamicUpdateSlice2D<signed char>;
template class DynamicUpdateSlice2D<unsigned long long>;
template class DynamicUpdateSlice2D<long long>;

} // namespace popops
