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
template <typename InType> class DynamicUpdateSlice2d : public Vertex {
public:
  DynamicUpdateSlice2d();

  Input<unsigned> offset; // in out
  // 16bit element shall follow VectorList to compensate packing
  // and make next VectorList 4bytes aligned
  InOut<VectorList<InType, COMPACT_DELTAN>> baseT;
  const unsigned short numSubElements; // in the slice dimension
  Input<VectorList<InType, COMPACT_DELTAN>> subT;
  const unsigned short numRegions;
  const unsigned numBaseElements; // in the slice dimension

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned r = 0; r != numRegions; ++r) {
      auto regionSize = baseT[r * numBaseElements].size();
      unsigned baseSlice = offset;
      if (baseSlice >= numBaseElements)
        baseSlice = 0;
      unsigned subIdx = r * numSubElements;

      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto baseIdx = r * numBaseElements + baseSlice;
        for (unsigned e = 0; e != regionSize; e++) {
          baseT[baseIdx][e] = subT[subIdx][e];
        }
        subIdx++;
        baseSlice++;
        if (baseSlice >= numBaseElements)
          baseSlice -= numBaseElements;
      }
    }
    return true;
  }
};
template class DynamicUpdateSlice2d<float>;
template class DynamicUpdateSlice2d<half>;
template class DynamicUpdateSlice2d<int>;
template class DynamicUpdateSlice2d<unsigned>;
template class DynamicUpdateSlice2d<bool>;
template class DynamicUpdateSlice2d<char>;
template class DynamicUpdateSlice2d<unsigned char>;
template class DynamicUpdateSlice2d<signed char>;

} // namespace popops
