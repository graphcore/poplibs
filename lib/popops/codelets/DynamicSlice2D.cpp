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

// Copy slices [\a offset : \a offset + \a numSubElements) of regions of
// \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change.
// Where the offset given is larger than numBaseElements, behaviour is not
// properly specified.  Options could be baseSlice=offset % numBaseElements,
// or as implemented if(offset>=numBaseElements) baseSlice=0;
template <typename InType> class DynamicSlice2D : public Vertex {
public:
  DynamicSlice2D();

  Input<unsigned> offset; // in \a baseT
  // 16bit element shall follow VectorList to compensate packing
  // and make next VectorList 4bytes aligned
  Input<VectorList<InType, COMPACT_DELTAN>> baseT;
  const unsigned short numSubElements; // in the slice dimension
  Output<VectorList<InType, COMPACT_DELTAN>> subT;
  const unsigned short numRegions;
  const unsigned numBaseElements; // in the slice dimension:  MSB used to
                                  // indicate if invalid indices must be
                                  // remapped

  IS_EXTERNAL_CODELET(true);

  void compute() {
    for (unsigned r = 0; r != numRegions; ++r) {
      const auto numBaseElementsActual = numBaseElements & 0x7fffffffu;
      auto regionSize = baseT[r * numBaseElementsActual].size();
      unsigned baseSlice = offset;
      unsigned subIdx = r * numSubElements;
      if (numBaseElements & 0x80000000u) {
        if (baseSlice >= numBaseElementsActual)
          baseSlice = 0;
      } else {
        if (baseSlice >= numBaseElementsActual)
          return;
      }
      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto baseIdx = r * numBaseElementsActual + baseSlice;
        for (unsigned e = 0; e != regionSize; e++) {
          subT[subIdx][e] = baseT[baseIdx][e];
        }
        subIdx++;
        baseSlice++;
        if (baseSlice >= numBaseElementsActual)
          baseSlice -= numBaseElementsActual;
      }
    }
  }
};
template class DynamicSlice2D<float>;
template class DynamicSlice2D<half>;
template class DynamicSlice2D<int>;
template class DynamicSlice2D<unsigned>;
template class DynamicSlice2D<bool>;
template class DynamicSlice2D<char>;
template class DynamicSlice2D<signed char>;
template class DynamicSlice2D<unsigned char>;
template class DynamicSlice2D<unsigned long long>;
template class DynamicSlice2D<long long>;

} // namespace popops
