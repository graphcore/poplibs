#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath>
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popops {

// Copy slices [\a offset : \a offset + \a numOutElements) of regions of
// \a baseT to \a subT.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change
template <typename InType>
class DynamicSlice : public Vertex {
public:
  Input<unsigned> offset; // in \a baseT
  // [region*numBaseElements+sliceIdx][os]
  Vector<Input<Vector<InType>>, ONE_PTR> baseT;
  // [region*numSubElements+sliceIdx][os]
  Vector<Output<Vector<InType, ONE_PTR>>, ONE_PTR> subT;
  unsigned short numBaseElements;  // in the slice dimension
  unsigned short numSubElements; // int the slice dimension
  unsigned short numRegions;

  bool compute() {
    for (unsigned r = 0; r != numRegions; ++r) {
      auto regionSize = baseT[r * numBaseElements].size();
      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto subIdx = r * numSubElements + subSlice;
        auto baseSlice = (offset + subSlice) % numBaseElements;
        auto baseIdx = r * numBaseElements + baseSlice;
        for (unsigned e = 0; e != regionSize; e++) {
          subT[subIdx][e] = baseT[baseIdx][e];
        }
      }
    }
    return true;
  }
};
template class DynamicSlice<float>;
template class DynamicSlice<half>;
template class DynamicSlice<int>;
template class DynamicSlice<bool>;

// Copy slices [\a offset : \a offset + \a numOutElements) of regions of
// \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change
template <typename InType>
class DynamicSlice2d : public SupervisorVertex {
public:
  Input<unsigned> offset; // in \a baseT
  Input<Vector<InType>> baseT;
  Output<Vector<InType>> subT;
  unsigned numBaseElements;  // in the slice dimension
  unsigned numSubElements;   // in the slice dimension
  unsigned regionSize;       // stride between slices
  unsigned elementsPerWorker;// number of elements to copy
  SimOnlyField<unsigned> numWorkers;
  bool compute() {
    assert(baseT.size() == numBaseElements * regionSize);
    assert(subT.size() == numSubElements * regionSize);
    for (unsigned worker = 0; worker != numWorkers; ++worker) {
      unsigned vertexOffset = worker * elementsPerWorker;
      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto baseSlice = (offset + subSlice) % numBaseElements;
        for (unsigned e = 0; e != elementsPerWorker; e++) {
          if (vertexOffset + e >= regionSize)
            // vertices may have empty or truncated regions
            break;
          subT[subSlice * regionSize + vertexOffset + e] =
            baseT[baseSlice * regionSize + vertexOffset + e];
        }
      }
    }
    return true;
  }
};
template class DynamicSlice2d<float>;
template class DynamicSlice2d<half>;
template class DynamicSlice2d<int>;
template class DynamicSlice2d<bool>;


// Copy each \numSubElements regions from \a in to
// \a out regions [\a offset : \a offset + \a numInElements)
template <typename InType>
class DynamicUpdateSlice : public Vertex {
public:
  Input<unsigned> offset; // in out
  // [region*numBaseElements+sliceIdx][os]
  Vector<InOut<Vector<InType>>, ONE_PTR> baseT;
  // [region*numSubElements+sliceIdx][os]
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> subT;
  unsigned short numBaseElements;  // in the slice dimension
  unsigned short numSubElements; // in the slice dimension
  unsigned short numRegions;

  bool compute() {
    for (unsigned r = 0; r != numRegions; ++r) {
      auto regionSize = baseT[r * numBaseElements].size();
      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto subIdx = r * numSubElements + subSlice;
        auto baseSlice = (offset + subSlice) % numBaseElements;
        auto baseIdx = r * numBaseElements + baseSlice;
        for (unsigned e = 0; e != regionSize; e++) {
          baseT[baseIdx][e] = subT[subIdx][e];
        }
      }
    }
    return true;
  }
};
template class DynamicUpdateSlice<float>;
template class DynamicUpdateSlice<half>;
template class DynamicUpdateSlice<int>;
template class DynamicUpdateSlice<bool>;

// Copy each \numSubElements regions from \a in to
// \a out regions [\a offset : \a offset + \a numInElements)
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
// The slice calculation is currently performed modulo \a numBaseElements but
// this is subject to change
template <typename InType>
class DynamicUpdateSlice2d : public SupervisorVertex {
public:
  Input<unsigned> offset; // in \a baseT
  InOut<Vector<InType>> baseT;
  Input<Vector<InType>> subT;
  unsigned numBaseElements;  // in the slice dimension
  unsigned numSubElements;   // in the slice dimension
  unsigned regionSize;       // stride between slices
  unsigned elementsPerWorker;// number of elements to copy
  SimOnlyField<unsigned> numWorkers;
  bool compute() {
    assert(baseT.size() == numBaseElements * regionSize);
    assert(subT.size() == numSubElements * regionSize);
    for (unsigned worker = 0; worker != numWorkers; ++worker) {
      unsigned vertexOffset = worker * elementsPerWorker;
      for (unsigned subSlice = 0; subSlice != numSubElements; ++subSlice) {
        auto baseSlice = (offset + subSlice) % numBaseElements;
        for (unsigned e = 0; e != elementsPerWorker; e++) {
          if (vertexOffset + e >= regionSize)
            // vertices may have empty or truncated regions
            break;
          baseT[baseSlice * regionSize + vertexOffset + e] =
            subT[subSlice * regionSize + vertexOffset + e];
        }
      }
    }
    return true;
  }
};
template class DynamicUpdateSlice2d<float>;
template class DynamicUpdateSlice2d<half>;
template class DynamicUpdateSlice2d<int>;
template class DynamicUpdateSlice2d<bool>;

}
