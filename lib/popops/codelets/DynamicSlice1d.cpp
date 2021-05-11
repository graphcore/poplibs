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
template <typename InType>
class DynamicSlice1d : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  DynamicSlice1d();

  Input<unsigned> offset; // in \a baseT
  Input<Vector<InType, ONE_PTR>> baseT;
  Output<Vector<InType, ONE_PTR>> subT;
  const unsigned numBaseElements; // in the slice dimension
  const unsigned numSubElements;  // in the slice dimension
  const unsigned regionSize;      // stride between slices

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const unsigned numWorkers = CTXT_WORKERS;
    unsigned elementsPerWorker = (regionSize + numWorkers - 1) / numWorkers;

    for (unsigned worker = 0; worker != numWorkers; ++worker) {
      unsigned workerOffset = worker * elementsPerWorker;
      unsigned baseSlice = offset;
      if (baseSlice >= numBaseElements)
        baseSlice = 0;
      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        for (unsigned e = 0; e != elementsPerWorker; e++) {
          if (workerOffset + e >= regionSize)
            // vertices may have empty or truncated regions
            break;
          subT[subSlice * regionSize + workerOffset + e] =
              baseT[baseSlice * regionSize + workerOffset + e];
        }
        baseSlice++;
        if (baseSlice >= numBaseElements)
          baseSlice = 0;
      }
    }
    return true;
  }
};
template class DynamicSlice1d<float>;
template class DynamicSlice1d<half>;
template class DynamicSlice1d<int>;
template class DynamicSlice1d<unsigned>;
template class DynamicSlice1d<bool>;
template class DynamicSlice1d<char>;
template class DynamicSlice1d<unsigned char>;
template class DynamicSlice1d<signed char>;

} // namespace popops
