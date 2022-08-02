// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "MultiUpdateOp.hpp"
#include "MultiSliceUpdateCommon.hpp"
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
  Input<Vector<Type, ONE_PTR, 8>> subT;
  InOut<Vector<Type, COMPACT_PTR, 8>> baseT;
  const unsigned short regionSize; // stride between slices
  const bool indicesAreSorted;     // indices are sorted in increasing order
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  // in the slice dimension (ceil numBaseElements / numWorkers). Required only
  // by assembler
  const unsigned maxElementsPerWorker;

  void compute(unsigned wid) {
    unsigned restrictedRegionSize = regionSize;
    if (std::is_same<Type, half>::value && !subwordWritesSupported)
      restrictedRegionSize &= ~0x1;

    constexpr unsigned minAtomSize = 4;
    constexpr bool hasAtomicWriteGranularity =
        sizeof(baseT[0]) % minAtomSize == 0;
    const auto split = multiUpdateOpWorkerDivision(
        hasAtomicWriteGranularity, indicesAreSorted, baseOffset,
        baseOffset + numBaseElements, wid, maxElementsPerWorker);

    // This worker has not been assigned any base offsets
    const unsigned thisWorkerBaseElems = split.offsetEnd - split.offsetBegin;
    if (thisWorkerBaseElems == 0) {
      return;
    }

    unsigned offsetIndexBegin = 0;
    unsigned offsetIndexEnd = offsets.size();
    if (indicesAreSorted) {
      offsetIndexBegin =
          lowerBinarySearch(reinterpret_cast<const int *>(&offsets[0]),
                            offsets.size(), baseOffset);
      offsetIndexEnd =
          upperBinarySearch(reinterpret_cast<const int *>(&offsets[0]),
                            offsets.size(), baseOffset + numBaseElements);
    }

    for (unsigned o = offsetIndexBegin; o != offsetIndexEnd; ++o) {
      auto baseIdx = offsets[o];

      // the assembly uses this same logic here but without bounds checks on
      // baseIdx for speed reasons so assert it here instead.
      assert(baseIdx < (1 << 31));
      assert(numBaseElements < (1 << 31));
      if (baseIdx - split.offsetBegin >= thisWorkerBaseElems) {
        // this slice is not a part of baseT so we can skip it.
        continue;
      }
      baseIdx -= baseOffset;

      for (unsigned e = 0; e != restrictedRegionSize; ++e) {
        const auto dstIndex = baseIdx * restrictedRegionSize + e;
        baseT[dstIndex] =
            updateOp(op, baseT[dstIndex], subT[o * restrictedRegionSize + e]);
      }
    }
  }
};

template class MultiUpdateOp<half, true, Operation::MAX>;
template class MultiUpdateOp<half, false, Operation::MAX>;
template class MultiUpdateOp<float, false, Operation::MAX>;
template class MultiUpdateOp<int, false, Operation::MAX>;
template class MultiUpdateOp<unsigned, false, Operation::MAX>;

template <typename Type, typename SType, bool subwordWritesSupported,
          Operation op>
class ScaledMultiUpdateOp : public MultiVertex {
public:
  ScaledMultiUpdateOp();

  IS_EXTERNAL_CODELET((hasAssembly<Type, subwordWritesSupported>()));
  Input<Vector<unsigned>> offsets; // in \a baseT
  Input<Vector<Type, ONE_PTR, 8>> subT;
  InOut<Vector<Type, COMPACT_PTR, 8>> baseT;
  const unsigned short regionSize; // stride between slices
  const bool indicesAreSorted;     // indices are sorted in increasing order
  const unsigned baseOffset;       // in the slice dimension
  const unsigned numBaseElements;  // in the slice dimension
  // in the slice dimension (ceil numBaseElements / numWorkers)
  const unsigned maxElementsPerWorker;
  Input<SType> scale;

  void compute(unsigned wid) {
    // Perform calculation in single precision for half data so that s
    // stochastic rounding will occur. TODO: T12921 Replace with a mix.
    // For halves, accumulate in float so that stochastic rounding will take
    // effect.
    using ScaleType =
        std::conditional_t<std::is_same<SType, half>::value, float, SType>;
    // load scale once
    const auto scaleL = ScaleType(*scale);

    unsigned restrictedRegionSize = regionSize;
    if (std::is_same<Type, half>::value && !subwordWritesSupported)
      restrictedRegionSize &= ~0x1;

    // split across base elements or regionSize dimension
    constexpr unsigned minAtomSize = 4;
    constexpr bool hasAtomicWriteGranularity =
        sizeof(baseT[0]) % minAtomSize == 0;
    const auto split = multiUpdateOpWorkerDivision(
        hasAtomicWriteGranularity, indicesAreSorted, baseOffset,
        baseOffset + numBaseElements, wid, maxElementsPerWorker);

    // This worker has not been assigned any base offsets
    const unsigned thisWorkerBaseElems = split.offsetEnd - split.offsetBegin;
    if (thisWorkerBaseElems == 0) {
      return;
    }

    unsigned offsetIndexBegin = 0;
    unsigned offsetIndexEnd = offsets.size();
    if (indicesAreSorted) {
      offsetIndexBegin =
          lowerBinarySearch(reinterpret_cast<const int *>(&offsets[0]),
                            offsets.size(), baseOffset);
      offsetIndexEnd =
          upperBinarySearch(reinterpret_cast<const int *>(&offsets[0]),
                            offsets.size(), baseOffset + numBaseElements);
    }

    for (unsigned o = offsetIndexBegin; o != offsetIndexEnd; ++o) {
      auto baseIdx = offsets[o];
      // the assembly uses this same logic here but without bounds checks on
      // baseIdx for speed reasons so assert it here instead.
      assert(baseIdx < (1 << 31));
      assert(numBaseElements < (1 << 31));
      if (baseIdx - split.offsetBegin >= thisWorkerBaseElems) {
        // this slice is not a part of baseT so we can skip it.
        continue;
      }

      baseIdx -= baseOffset;
      for (unsigned e = 0; e != restrictedRegionSize; ++e) {
        const Type addend =
            scaleL * ScaleType(subT[o * restrictedRegionSize + e]);
        const auto srcDstIndex = baseIdx * restrictedRegionSize + e;
        baseT[srcDstIndex] = updateOp(op, baseT[srcDstIndex], addend);
      }
    }
  }
};

template class ScaledMultiUpdateOp<half, half, true, Operation::ADD>;
template class ScaledMultiUpdateOp<half, float, true, Operation::ADD>;
template class ScaledMultiUpdateOp<half, half, false, Operation::ADD>;
template class ScaledMultiUpdateOp<half, float, false, Operation::ADD>;
template class ScaledMultiUpdateOp<float, float, false, Operation::ADD>;
template class ScaledMultiUpdateOp<int, int, false, Operation::ADD>;
template class ScaledMultiUpdateOp<unsigned, unsigned, false, Operation::ADD>;

} // namespace popops
