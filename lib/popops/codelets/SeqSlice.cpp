// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>
#include <type_traits>

#ifdef __IPU__
#include "colossus/tileimplconsts.h"
#endif

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popops {

#ifdef __IPU__
template <typename Type>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] SequenceSliceCopy
    : public Vertex {
  Input<Vector<Type, ONE_PTR>> src;
  Input<Vector<Type, ONE_PTR>> dst;
  unsigned short nAtoms; // must be non-zero

public:
  IS_EXTERNAL_CODELET(false);
  bool compute() {
    const Type *srcPtr = &src[0];
    Type *dstPtr = &dst[0];
    unsigned n = nAtoms;

    if (std::is_same<Type, half>::value) {
      // Minimise sub-word writes
      if (uintptr_t(dstPtr) & 0x2) {
        // initial misaligned output element
        auto x = *srcPtr++;
        *dstPtr++ = x;
        --n;
        if (n == 0)
          return true;
      }
      unsigned short loopCount = n / 2;
      n &= 1; // There may be an odd element at the end
      auto *dstAlignedPtr = (half2 *)dstPtr;
      half a = *srcPtr++;
      half b = *srcPtr++;
      for (unsigned i = 0; i != loopCount; ++i) {
        half2 word{a, b};
        a = *srcPtr++;
        b = *srcPtr++;
        *dstAlignedPtr++ = word;
      }
      if (n) {
        (*dstAlignedPtr)[0] = a;
      }
    } else {
      auto prev = *srcPtr++;
      unsigned short loopCount = n - 1;
      for (unsigned i = 0; i != loopCount; ++i) {
        *dstPtr++ = prev;
        prev = *srcPtr++;
      }
      *dstPtr = prev;
    }
    return true;
  }
};
template class SequenceSliceCopy<half>;
template class SequenceSliceCopy<float>;
#endif

// Dynamically update slices from a source to a destination tensor.
// Both src and dst offsets plus the number of elements sliced are configured at
// runtime.
// The vertex can be connected to a subset of a tensor, with slicing clipped by
// both src and dst boundaries.
// Update slices from multiple offsets \a baseT to \a subT.
// Each slice has its own size.
// srcOffsetT and dstOffsetT are absolute
// This variant takes a 2d input and calculates the offsets given the start
// address of the base and sub Tensors. the updates are added to the core tensor
// indices that are not within the range of [baseOffset,
// baseOffset + numBaseElements) are ignored.
template <typename Type>
class [[poplar::constraint("elem(*srcT) != elem(*dstT)")]] SequenceSlice
    : public SupervisorVertex {
  // TODO: make this a supervisor vertex so multiple offsets can be handled
  // by multiple workers. Can't split into multple vertices as they would all
  // have the same dstT
public:
  SequenceSlice();

  IS_EXTERNAL_CODELET(false);

  Input<Vector<Type, ONE_PTR, 4>> srcT;
  InOut<Vector<Type, ONE_PTR, 4>> dstT;
  Input<Vector<unsigned>> srcOffsetT;          // in \a baseT
  Input<Vector<unsigned, ONE_PTR>> dstOffsetT; // in \a subT
  Input<Vector<unsigned, ONE_PTR>> nElementsT;
  Output<Vector<unsigned, ONE_PTR>> tmpT; // size should be 3*srcOffsetT.size()
  const unsigned srcFirst;
  const unsigned dstFirst;         // in the slice dimension
  const unsigned numSrcElements;   // in the slice dimension
  const unsigned numDstElements;   // in the slice dimension
  const unsigned short regionSize; // stride between elements

#ifdef __IPU__
  __attribute__((target("supervisor")))
#endif
  bool
  compute() {
    for (unsigned o = 0; o != srcOffsetT.size(); ++o) {
      auto srcOffset = srcOffsetT[o];
      auto dstOffset = dstOffsetT[o];
      int nE = nElementsT[o];
      if (nE == 0)
        continue;
      // Clip the starts and lengths so neither src nor dst accesses will
      // overflow.
      auto startSkip = 0u;
      if (srcFirst > srcOffset)
        startSkip = srcFirst - srcOffset;
      if (dstOffset + startSkip < dstFirst)
        startSkip = dstFirst - dstOffset;
      nE -= startSkip;
      const Type *srcPtr =
          &srcT[(srcOffset + startSkip - srcFirst) * regionSize];
      Type *dstPtr = &dstT[(dstOffset + startSkip - dstFirst) * regionSize];

      if (srcOffset + nE > numSrcElements)
        nE -= srcOffset + nE - numSrcElements;
      if (dstOffset + nE > numDstElements)
        nE -= dstOffset + nE - numDstElements;

      if (nE == 0)
        continue;
      auto nAtoms = nE * regionSize;

#ifdef __IPU__
      // Invoke a worker for each slice
      tmpT[3 * o + 0] = uintptr_t(srcPtr);
      tmpT[3 * o + 1] = uintptr_t(dstPtr);
      tmpT[3 * o + 2] = nAtoms;

      unsigned dummy;
      if (std::is_same<Type, half>::value) {
        asm volatile(
            " setzi %0, __runCodelet_popops__SequenceSliceCopy___half\n"
            " run %0, %1, 0\n"
            : "+r"(dummy) // actually scratch rather than output
            : "r"(uintptr_t(&tmpT[3 * o] - TMEM_REGION0_BASE_ADDR / 4)));
      } else {
        asm volatile(
            " setzi %0, __runCodelet_popops__SequenceSliceCopy___float\n"
            " run %0, %1, 0\n"
            : "+r"(dummy) // actually scratch rather than output
            : "r"(uintptr_t(&tmpT[3 * o] - TMEM_REGION0_BASE_ADDR / 4)));
      }

#else
      for (unsigned i = 0; i != nAtoms; ++i) {
        *dstPtr++ = *srcPtr++;
      }
#endif
    }
#ifdef __IPU__
    asm volatile(" sync %0\n" ::"i"(TEXCH_SYNCZONE_LOCAL));
#endif
    return true;
  }
};

template class SequenceSlice<float>;
template class SequenceSlice<half>;
template class SequenceSlice<int>;
template class SequenceSlice<unsigned>;

} // namespace popops
