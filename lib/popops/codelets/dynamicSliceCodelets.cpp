#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;

namespace popops {

// Copy slices [\a offset : \a offset + \a numOutElements) of regions of
// \a baseT to \a subT.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change.
// Where the offset given is larger than numBaseElements, behaviour is not
// properly specified.  Options could be baseSlice=offset % numBaseElements,
// or as implemented if(offset>=numBaseElements) baseSlice=0;
template <typename InType>
class DynamicSlice2d : public Vertex {
public:
  DynamicSlice2d();

  Input<unsigned> offset; // in \a baseT
  // 16bit element shall follow VectorList to compensate packing
  // and make next VectorList 4bytes aligned
  Input<VectorList<InType, DELTAN>> baseT;
  const unsigned short numSubElements; // in the slice dimension
  Output<VectorList<InType, DELTAN>> subT;
  const unsigned short numRegions;
  const unsigned numBaseElements;      // in the slice dimension

  static const bool isBool = std::is_same<InType,bool>::value;
  IS_EXTERNAL_CODELET(!isBool);


  bool compute() {
    for (unsigned r = 0; r != numRegions; ++r) {
      auto regionSize = baseT[r * numBaseElements].size();
      unsigned baseSlice = offset;
      unsigned subIdx = r * numSubElements;
      if(baseSlice >= numBaseElements)
        baseSlice=0;

      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto baseIdx = r * numBaseElements + baseSlice;
        for (unsigned e = 0; e != regionSize; e++) {
          subT[subIdx][e]= baseT[baseIdx][e];
        }
        subIdx++;
        baseSlice++;
        if(baseSlice >= numBaseElements)
          baseSlice-=numBaseElements;
      }
    }
    return true;
  }
};
template class DynamicSlice2d<float>;
template class DynamicSlice2d<half>;
template class DynamicSlice2d<int>;
template class DynamicSlice2d<unsigned>;
template class DynamicSlice2d<bool>;

// Copy slices [\a offset : \a offset + \a numOutElements) of regions of
// \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change
// Where the offset given is larger than numBaseElements, behaviour is not
// properly specified.  Options could be baseSlice=offset % numBaseElements,
// or as implemented if(offset>=numBaseElements) baseSlice=0;
template <typename InType>
class DynamicSliceSupervisor : public SupervisorVertex {
public:
  DynamicSliceSupervisor();

  Input<unsigned> offset; // in \a baseT
  Input<Vector<InType, ONE_PTR>> baseT;
  Output<Vector<InType, ONE_PTR>> subT;
  const unsigned numBaseElements;  // in the slice dimension
  const unsigned short numSubElements;   // in the slice dimension
  const unsigned short regionSize;       // stride between slices

  static const bool isBool = std::is_same<InType,bool>::value;
  IS_EXTERNAL_CODELET(!isBool);

  bool compute() {
    const unsigned numWorkers = NUM_WORKERS;
    unsigned elementsPerWorker = (regionSize + numWorkers -1) / numWorkers;

    for (unsigned worker = 0; worker != numWorkers; ++worker) {
      unsigned workerOffset = worker * elementsPerWorker;
      unsigned baseSlice = offset;
      if (baseSlice >= numBaseElements)
        baseSlice=0;
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
          baseSlice=0;
      }
    }
    return true;
  }
};
template class DynamicSliceSupervisor<float>;
template class DynamicSliceSupervisor<half>;
template class DynamicSliceSupervisor<int>;
template class DynamicSliceSupervisor<unsigned>;
template class DynamicSliceSupervisor<bool>;

// Copy single slices from multiple offsets \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
template <typename Type>
class MultiSlice : public Vertex {
public:
  MultiSlice();

  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR>> baseT;
  Output<Vector<Type, ONE_PTR>> subT;
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  const unsigned short regionSize; // stride between slices

  bool compute() {
    for (unsigned o = 0; o != offsets.size(); ++o) {
      auto baseIdx = offsets[o];
      if (baseIdx < baseOffset ||
          baseIdx >= baseOffset + numBaseElements) {
        // this slice is not a part of baseT so we can skip it.
        continue;
      }
      baseIdx -= baseOffset;

      for (unsigned e = 0; e != regionSize; ++e) {
        subT[o * regionSize + e] = baseT[baseIdx * regionSize + e];
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

// Update single slices from multiple offsets \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors. the updates are added to the core tensor
// indices that are not within the range of [baseOffset,
// baseOffset + numBaseElements) are ignored.
template <typename Type>
class MultiUpdate : public Vertex {
public:
  MultiUpdate();

  Input<Vector<unsigned>> offsets; // in \a baseT
  InOut<Vector<Type, ONE_PTR>> baseT;
  Input<Vector<Type, ONE_PTR>> subT;
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  const unsigned short regionSize; // stride between slices

  bool compute() {
    for (unsigned o = 0; o != offsets.size(); ++o) {
      auto baseIdx = offsets[o];
      if (baseIdx < baseOffset ||
          baseIdx >= baseOffset + numBaseElements) {
        // this slice is not a part of baseT so we can skip it.
        continue;
      }
      baseIdx -= baseOffset;

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

// Add single slices from multiple offsets \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors. the updates are added to the core tensor
// indices that are not within the range of [baseOffset,
// baseOffset + numBaseElements) are ignored.
template <typename Type, bool subwordWritesSupported>
class MultiUpdateAdd : public Vertex {
public:
  MultiUpdateAdd();

  static const bool isExternal = std::is_same<Type, float>::value
                              || (std::is_same<Type, half>::value
                                  && !subwordWritesSupported);
  IS_EXTERNAL_CODELET(isExternal);
  Input<Type> scale;
  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR, 4>> subT;
  InOut<Vector<Type, SCALED_PTR32, 4>> baseT;
  const unsigned short regionSize; // stride between slices
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension

  bool compute() {
    // perform calculation in single precision for half data so that stochastic
    // rounding will occur. TODO: replace with a mix
    // For halves, accumulate in float so that stochastic rounding will take
    // effect.
    using ScaleType = std::conditional_t<std::is_same<Type, half>::value, float,
                                         Type>;
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
      if (baseIdx > numBaseElements) {
        // this slice is not a part of baseT so we can skip it.
        continue;
      }

      for (unsigned e = 0; e != restrictedRegionSize; ++e) {
        const auto addend = scaleL *
                            ScaleType(subT[o * restrictedRegionSize + e]);
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

// Copy each \numSubElements regions from \a in to
// \a out regions [\a offset : \a offset + \a numInElements)
// Where the offset given is larger than numBaseElements, behaviour is not
// properly specified.  Options could be baseSlice=offset % numBaseElements,
// or as implemented if(offset>=numBaseElements) baseSlice=0;
template <typename InType>
class DynamicUpdateSlice2d : public Vertex {
public:
  DynamicUpdateSlice2d();

  Input<unsigned> offset; // in out
  // 16bit element shall follow VectorList to compensate packing
  // and make next VectorList 4bytes aligned
  InOut<VectorList<InType, DELTAN>> baseT;
  const unsigned short numSubElements; // in the slice dimension
  Input<VectorList<InType, DELTAN>> subT;
  const unsigned short numRegions;
  const unsigned numBaseElements;      // in the slice dimension

  static const bool isBool = std::is_same<InType,bool>::value;
  IS_EXTERNAL_CODELET(!isBool);

  bool compute() {
    for (unsigned r = 0; r != numRegions; ++r) {
      auto regionSize = baseT[r * numBaseElements].size();
      unsigned baseSlice = offset;
      if(baseSlice >= numBaseElements)
        baseSlice=0;
      unsigned subIdx = r * numSubElements;

      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto baseIdx = r * numBaseElements + baseSlice;
        for (unsigned e = 0; e != regionSize; e++) {
          baseT[baseIdx][e] = subT[subIdx][e];
        }
        subIdx++;
        baseSlice++;
        if(baseSlice >= numBaseElements)
          baseSlice-=numBaseElements;
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
class DynamicUpdateSliceSupervisor : public SupervisorVertex {
public:
  DynamicUpdateSliceSupervisor();

  Input<unsigned> offset; // in \a baseT
  InOut<Vector<InType, ONE_PTR>> baseT;
  Input<Vector<InType, ONE_PTR>> subT;
  const unsigned numBaseElements;        // in the slice dimension
  const unsigned short numSubElements;   // in the slice dimension
  const unsigned short regionSize;       // stride between slices

  static const bool isBool = std::is_same<InType,bool>::value;
  IS_EXTERNAL_CODELET(!isBool);

  bool compute() {
    const unsigned numWorkers = NUM_WORKERS;
    unsigned elementsPerWorker = (regionSize + numWorkers -1) / numWorkers;

    for (unsigned worker = 0; worker != numWorkers; ++worker) {
      unsigned workerOffset = worker * elementsPerWorker;
      unsigned baseSlice =offset;
      if (baseSlice >= numBaseElements)
        baseSlice=0;
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
          baseSlice=0;
      }
    }
    return true;
  }
};
template class DynamicUpdateSliceSupervisor<float>;
template class DynamicUpdateSliceSupervisor<half>;
template class DynamicUpdateSliceSupervisor<int>;
template class DynamicUpdateSliceSupervisor<unsigned>;
template class DynamicUpdateSliceSupervisor<bool>;

}
