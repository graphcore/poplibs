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

// Copy slices [\a offset : \a offset + \a numSubElements) of regions of
// \a baseT to \a subT.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change
// Where the offset given is larger than numBaseElements, behaviour is not
// properly specified.  Options could be baseSlice=offset % numBaseElements,
// or as implemented if(offset>=numBaseElements) baseSlice=0;
template <typename InType> class DynamicSlice1D : public MultiVertex {
public:
  DynamicSlice1D();

  Input<unsigned> offset; // in \a baseT
  Input<Vector<InType, ONE_PTR>> baseT;
  Output<Vector<InType, ONE_PTR>> subT;
  const unsigned numBaseElements; // in the slice dimension: MSB used to
                                  // indicate if invalid indices must be
                                  // remapped
  const unsigned numSubElements;  // in the slice dimension
  const unsigned regionSize;      // stride between slices

  IS_EXTERNAL_CODELET(true);

  void compute(unsigned wid) {
    const auto numBaseElementsActual = numBaseElements & 0x7fffffffu;
    unsigned elementsPerWorker = (regionSize + numWorkers() - 1) / numWorkers();
    unsigned workerOffset = wid * elementsPerWorker;
    unsigned baseSlice = offset;
    if (numBaseElements & 0x80000000u) {
      if (baseSlice >= numBaseElementsActual)
        baseSlice = 0;
    } else {
      if (baseSlice >= numBaseElementsActual)
        return;
    }
    for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
      for (unsigned e = 0; e != elementsPerWorker; e++) {
        if (workerOffset + e >= regionSize)
          // vertices may have empty or truncated regions
          break;
        subT[subSlice * regionSize + workerOffset + e] =
            baseT[baseSlice * regionSize + workerOffset + e];
      }
      baseSlice++;
      if (baseSlice >= numBaseElementsActual)
        baseSlice = 0;
    }
  }
};
template class DynamicSlice1D<float>;
template class DynamicSlice1D<half>;
template class DynamicSlice1D<int>;
template class DynamicSlice1D<unsigned>;
template class DynamicSlice1D<bool>;
template class DynamicSlice1D<char>;
template class DynamicSlice1D<unsigned char>;
template class DynamicSlice1D<signed char>;
template class DynamicSlice1D<unsigned long long>;
template class DynamicSlice1D<long long>;

} // namespace popops
