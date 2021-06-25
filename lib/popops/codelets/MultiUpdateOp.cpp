// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "MultiUpdateOp.hpp"
#include "poplar/TileConstants.hpp"
#include <type_traits>

using namespace poplar;

namespace popops {

template <typename Type, bool subwordWritesSupported>
constexpr bool hasAssembly() {
  return std::is_same<Type, float>::value ||
         (std::is_same<Type, half>::value && !subwordWritesSupported);
}

template <typename Type, bool subwordWritesSupported, Operation op>
class MultiUpdateOp : public MultiVertex {
public:
  MultiUpdateOp();

  IS_EXTERNAL_CODELET((hasAssembly<Type, subwordWritesSupported>()));
  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR, 4>> subT;
  InOut<Vector<Type, COMPACT_PTR, 4>> baseT;
  const unsigned short regionSize; // stride between slices
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  // in the slice dimension (ceil numBaseElements / numWorkers). Required only
  // by assembler
  const unsigned maxElementsPerWorker;

  bool compute(unsigned wid) {
    if (wid == 0) {
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
        if (baseIdx >= numBaseElements) {
          // this slice is not a part of baseT so we can skip it.
          continue;
        }

        for (unsigned e = 0; e != restrictedRegionSize; ++e) {
          const auto dstIndex = baseIdx * restrictedRegionSize + e;
          baseT[dstIndex] =
              updateOp(op, baseT[dstIndex], subT[o * restrictedRegionSize + e]);
        }
      }
    }
    return true;
  }
};

template class MultiUpdateOp<half, true, Operation::MAX>;
template class MultiUpdateOp<half, false, Operation::MAX>;
template class MultiUpdateOp<float, false, Operation::MAX>;
template class MultiUpdateOp<int, false, Operation::MAX>;
template class MultiUpdateOp<unsigned, false, Operation::MAX>;

template <typename Type, bool subwordWritesSupported, Operation op>
class ScaledMultiUpdateOp : public MultiVertex {
public:
  ScaledMultiUpdateOp();

  IS_EXTERNAL_CODELET((hasAssembly<Type, subwordWritesSupported>()));
  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR, 4>> subT;
  InOut<Vector<Type, COMPACT_PTR, 4>> baseT;
  const unsigned short regionSize; // stride between slices
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  // in the slice dimension (ceil numBaseElements / numWorkers). Required only
  // by assembler
  const unsigned maxElementsPerWorker;
  Input<Type> scale;

  bool compute(unsigned wid) {
    if (wid == 0) {
      // Perform calculation in single precision for half data so that s
      // stochastic rounding will occur. TODO: T12921 Replace with a mix.
      // For halves, accumulate in float so that stochastic rounding will take
      // effect.
      using ScaleType =
          std::conditional_t<std::is_same<Type, half>::value, float, Type>;
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
        if (baseIdx >= numBaseElements) {
          // this slice is not a part of baseT so we can skip it.
          continue;
        }

        for (unsigned e = 0; e != restrictedRegionSize; ++e) {
          const Type addend =
              scaleL * ScaleType(subT[o * restrictedRegionSize + e]);
          const auto srcDstIndex = baseIdx * restrictedRegionSize + e;
          baseT[srcDstIndex] = updateOp(op, baseT[srcDstIndex], addend);
        }
      }
    }
    return true;
  }
};

template class ScaledMultiUpdateOp<half, true, Operation::ADD>;
template class ScaledMultiUpdateOp<half, false, Operation::ADD>;
template class ScaledMultiUpdateOp<float, false, Operation::ADD>;
template class ScaledMultiUpdateOp<int, false, Operation::ADD>;
template class ScaledMultiUpdateOp<unsigned, false, Operation::ADD>;

} // namespace popops