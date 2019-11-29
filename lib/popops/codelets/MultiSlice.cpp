#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;

namespace popops {

// Copy single slices from multiple offsets \a baseT to \a subT.
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors.
template <typename Type> class MultiSlice : public Vertex {
public:
  MultiSlice();

  static const bool isBool = std::is_same<Type, bool>::value;
  IS_EXTERNAL_CODELET(!isBool);

  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR>> baseT;
  Output<Vector<Type, ONE_PTR>> subT;
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  const unsigned short regionSize; // stride between slices

  bool compute() {
    for (unsigned o = 0; o != offsets.size(); ++o) {
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
    return true;
  }
};
template class MultiSlice<float>;
template class MultiSlice<half>;
template class MultiSlice<int>;
template class MultiSlice<unsigned>;
template class MultiSlice<bool>;

} // namespace popops
