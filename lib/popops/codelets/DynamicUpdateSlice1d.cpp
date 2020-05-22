// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popops {

template <typename InType> constexpr bool isBool() {
  return std::is_same<InType, bool>::value;
}

// Copy each \numSubElements regions from \a in to
// \a out regions [\a offset : \a offset + \a numInElements)
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change
// Where the offset given is larger than numBaseElements, behaviour is not
// properly specified.  Options could be baseSlice=offset % numBaseElements,
// or as implemented if(offset>=numBaseElements) baseSlice=0;
template <typename InType>
class DynamicUpdateSlice1d
    : public SupervisorVertexIf<!isBool<InType>() && ASM_CODELETS_ENABLED> {
public:
  DynamicUpdateSlice1d();

  Input<unsigned> offset; // in \a baseT
  InOut<Vector<InType, ONE_PTR>> baseT;
  Input<Vector<InType, ONE_PTR>> subT;
  const unsigned numBaseElements; // in the slice dimension
  const unsigned numSubElements;  // in the slice dimension
  const unsigned regionSize;      // stride between slices

  IS_EXTERNAL_CODELET(!isBool<InType>());

  bool compute() {
    const unsigned numWorkers = NUM_WORKERS;
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
          baseT[baseSlice * regionSize + workerOffset + e] =
              subT[subSlice * regionSize + workerOffset + e];
        }
        baseSlice++;
        if (baseSlice >= numBaseElements)
          baseSlice = 0;
      }
    }
    return true;
  }
};
template class DynamicUpdateSlice1d<float>;
template class DynamicUpdateSlice1d<half>;
template class DynamicUpdateSlice1d<int>;
template class DynamicUpdateSlice1d<unsigned>;
template class DynamicUpdateSlice1d<bool>;

} // namespace popops
