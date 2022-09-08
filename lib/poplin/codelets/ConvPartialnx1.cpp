// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <tuple>
#include <type_traits>

#include "ConvPartialsStridesPacking.hpp"
#include "convCastSupport.hpp"
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

#ifdef __IPU__
#include "StackSizeDefs.hpp"
#include "inlineAssemblerConv.hpp"
#endif

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;
static constexpr auto COMPACT_DELTAN = poplar::VectorListLayout::COMPACT_DELTAN;

namespace poplin {

template <typename FPType, typename AccumType, bool useLimitedVer,
          unsigned numConvUnits, unsigned convInputLoadElems>
constexpr bool hasAssembly() {
  if (std::is_same_v<FPType, quarter>) {
    return false;
  }
  constexpr std::size_t loadElems64Bits = std::is_same_v<FPType, half> ? 4 : 2;
  if (convInputLoadElems != loadElems64Bits) {
    return false;
  }
  return !(std::is_same<AccumType, half>() && std::is_same<FPType, float>()) &&
         useLimitedVer == true &&
         (numConvUnits == 8 ||
          (numConvUnits == 16 &&
           (std::is_same<AccumType, half>() || std::is_same<FPType, float>())));
}

/**
 * Compute nx1 convolutions and accumulate them with partial sums in memory.
 * useLimitedVer is "true" if there are constraints imposed on
 * - size of strides are bounded to strides supported by ISA
 * - worklists-offsets are bounded to fit 16-bits
 * - worklists-number of elements <= maximum count supported by rpt instruction
 **/
template <class FPType, class AccumType, bool useLimitedVer, bool use128BitLoad,
          unsigned numConvUnits, unsigned convInputLoadElems, bool disableSR>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartialnx1
    : public std::conditional_t<
          hasAssembly<FPType, AccumType, useLimitedVer, numConvUnits,
                      convInputLoadElems>() &&
              EXTERNAL_CODELET,
          SupervisorVertex, MultiVertex> {
  static const bool needsAlignWorkers = false;

public:
  ConvPartialnx1();

  using WorkListType =
      std::conditional_t<useLimitedVer, unsigned short, unsigned>;
  using WorkListNumFieldType = std::conditional_t<useLimitedVer, short, int>;
  using UnsignedType =
      std::conditional_t<useLimitedVer, unsigned short, unsigned>;
  using SignedType = std::conditional_t<useLimitedVer, short, int>;
  using PackedStridesType =
      std::conditional_t<useLimitedVer, unsigned, unsigned long long>;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  static constexpr unsigned numStrideBits =
      useLimitedVer ? NUM_STRIDE_BITS : numStrideBitsUnlimited();

  // This value is
  // (inStrideX - 1 - (ampKernelHeight - 1) * inRowStride)
  //      * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  const PackedStridesType transformedInStride;
  // This output stride also encodes the flip parameter and is given as
  // -6 + outChansPerGroup * (actual output stride) if flipOut = false
  // -6 - outChansPerGroup * (actual output stride) if flipOut = true
  const PackedStridesType transformedOutStride;

  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, COMPACT_PTR, weightsAlign, use128BitLoad>>,
         ONE_PTR>
      weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 8, true>>, ONE_PTR> out;
  const unsigned zerosInfo;
  Input<VectorList<WorkListType, COMPACT_DELTAN>> worklists;
  const UnsignedType numOutGroupsM1;
  const UnsignedType numInGroups;
  const UnsignedType kernelOuterSizeM1;
  const UnsignedType kernelInnerElementsM1;

  const UnsignedType numConvGroupsM1;
  // The number of kernel elements we accumulate across within the AMP unit
  const UnsignedType ampKernelHeightM1;
  // transformedInRowStride is given by
  //  (inRowStride - 1) * inChansPerGroup / convInputLoadElems + 1
  const SignedType transformedInRowStride;
  const UnsignedType outChansPerGroup;
  const UnsignedType inChansPerGroup;

  IS_EXTERNAL_CODELET((hasAssembly<FPType, AccumType, useLimitedVer,
                                   numConvUnits, convInputLoadElems>()));

  // For assembly codelets, a true SupervisorVertex is used.
  void compute();

  // For C++ codelets, a MultiVertex is used.
  void compute(unsigned wid) {
    if (wid != 0) {
      return;
    }
    const auto numContexts = CTXT_WORKERS;

    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned ampKernelHeight = ampKernelHeightM1 + 1;
    const unsigned kernelOuterSize = kernelOuterSizeM1 + 1;
    const unsigned kernelInnerElements = kernelInnerElementsM1 + 1;

    const int convOutputStoreElems = std::is_same<AccumType, half>() ? 4 : 2;
    const auto packedTransformedInStrideReg = transformedInStride;
    const auto packedTransformedOutStrideReg = transformedOutStride;

    // Unpack registers strides into transformed strides
    int unpackedTransformedInStride =
        unpackAmpNx1Stride(numStrideBits, packedTransformedInStrideReg, 1);
    int unpackedTransformedOutStride =
        unpackAmpNx1Stride(numStrideBits, packedTransformedOutStrideReg, 0);
    const int secondPtrOffset =
        unpackAmpNx1Stride(numStrideBits, packedTransformedOutStrideReg, 1) +
        transformedInRowStride;
    const int unpackedTransformedInRowStride =
        unpackAmpNx1InRowStride(ampKernelHeight, secondPtrOffset);

    // Special case for half-half-8
    if ((numConvUnits == 8) && std::is_same<AccumType, half>() &&
        std::is_same<FPType, half>()) {
      // See createConvPartialAmpVertex for details
      unpackedTransformedOutStride -= 1;
    }

    // Convert back to elements stride
    unpackedTransformedOutStride *= convOutputStoreElems;

    unpackedTransformedInStride = reverseTransformedInStrideNx1(
        unpackedTransformedInStride, secondPtrOffset);

    const int inRowStride = reverseTransformedInRowStride(
        unpackedTransformedInRowStride, convInputLoadElems, inChansPerGroup);
    const int inStride = reverseTransformedInStride(
        unpackedTransformedInStride, convInputLoadElems, inChansPerGroup,
        ampKernelHeight, inRowStride);

    bool flipOut;
    int outStride;
    std::tie(flipOut, outStride) = reverseTransformedOutStride(
        unpackedTransformedOutStride, std::is_same<AccumType, float>(),
        numConvUnits, outChansPerGroup);

    quarter_metadata inMetadata, weightsMetadata;
    if constexpr (std::is_same<FPType, quarter>::value) {
      inMetadata = unpackMetadata(in.getMetadata());
      weightsMetadata = unpackMetadata(weights.getMetadata());
    }
    const unsigned numElems = zerosInfo;

    for (unsigned cg = 0; cg != numConvGroups; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        for (unsigned i = 0; i != numElems; ++i)
          out[cg * numOutGroups + og][i] = 0;
      }
    }
    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];
          for (unsigned ky = 0; ky < kernelOuterSize; ++ky) {
            for (unsigned kx = 0; kx < kernelInnerElements; ++kx) {
              const auto k = (ky * kernelInnerElements + kx);
              for (unsigned ctx = 0; ctx != numContexts; ++ctx) {
                const auto &wl = worklists[k * numContexts + ctx];
                unsigned wi = 0;
                while (wi < wl.size()) {
                  const auto accumTypeSize =
                      std::is_same<AccumType, float>() ? 4 : 2;
                  const auto typeSize = std::is_same<FPType, float>()  ? 4
                                        : std::is_same<FPType, half>() ? 2
                                                                       : 1;
                  const auto outOffset =
                      (wl[wi] * 8) / (outChansPerGroup * accumTypeSize);
                  // The numFieldElems values from worklist is less by 3
                  const auto numFieldElems =
                      static_cast<WorkListNumFieldType>(wl[wi + 1]) + 3;
                  const auto inOffset =
                      (wl[wi + 2] * 8) / (inChansPerGroup * typeSize);

                  wi += 3;
                  for (unsigned i = 0; i < numFieldElems; ++i) {
                    for (unsigned outChan = 0; outChan < outChansPerGroup;
                         ++outChan) {
                      const auto outIndex =
                          (outOffset + (flipOut ? -i : i) * outStride) *
                              outChansPerGroup +
                          outChan;
                      AccumType sum = out[cg * numOutGroups + og][outIndex];
                      for (unsigned ak = 0; ak < ampKernelHeight; ++ak) {
                        for (unsigned inChan = 0; inChan < inChansPerGroup;
                             ++inChan) {
                          const auto inIndex =
                              (inOffset + i * inStride) * inChansPerGroup +
                              ak * inRowStride * inChansPerGroup + inChan;
                          const auto weightIndex =
                              ky * ampKernelHeight * kernelInnerElements *
                                  outChansPerGroup * inChansPerGroup +
                              kx * outChansPerGroup * inChansPerGroup +
                              ak * kernelInnerElements * outChansPerGroup *
                                  inChansPerGroup +
                              outChan * inChansPerGroup + inChan;

                          sum += promoteType<FPType, AccumType>(
                                     in[cg * numInGroups + ig][inIndex],
                                     inMetadata) *
                                 promoteType<FPType, AccumType>(
                                     w[weightIndex], weightsMetadata);
                        }
                      }
                      out[cg * numOutGroups + og][outIndex] = sum;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

#define INSTANTIATE_DISABLE_SR(inType, parType, limited, weights128,           \
                               convUnits, inputElems)                          \
  template class ConvPartialnx1<inType, parType, limited, weights128,          \
                                convUnits, inputElems, true>;                  \
  template class ConvPartialnx1<inType, parType, limited, weights128,          \
                                convUnits, inputElems, false>

#define INSTANTIATE_LIMITED(inType, parType, weights128, convUnits,            \
                            inputElems)                                        \
  INSTANTIATE_DISABLE_SR(inType, parType, true, weights128, convUnits,         \
                         inputElems);                                          \
  INSTANTIATE_DISABLE_SR(inType, parType, false, weights128, convUnits,        \
                         inputElems)

#define INSTANTIATE_WEIGHTS_128(inType, parType, convUnits, inputElems)        \
  INSTANTIATE_LIMITED(inType, parType, true, convUnits, inputElems);           \
  INSTANTIATE_LIMITED(inType, parType, false, convUnits, inputElems)

INSTANTIATE_WEIGHTS_128(half, half, 8, 4);
INSTANTIATE_WEIGHTS_128(half, float, 8, 4);
INSTANTIATE_WEIGHTS_128(float, float, 8, 2);

INSTANTIATE_WEIGHTS_128(half, half, 16, 4);
INSTANTIATE_WEIGHTS_128(float, float, 16, 2);

INSTANTIATE_WEIGHTS_128(half, half, 8, 8);
INSTANTIATE_WEIGHTS_128(half, float, 8, 8);
INSTANTIATE_WEIGHTS_128(float, float, 8, 4);

INSTANTIATE_WEIGHTS_128(half, half, 16, 8);
INSTANTIATE_WEIGHTS_128(half, float, 16, 8);
INSTANTIATE_WEIGHTS_128(float, float, 16, 4);

#ifdef __IPU__

#if __IPU_ARCH_VERSION__ >= 21

template <typename UnsignedType, unsigned numConvUnits>
class WorkerClassNx1 : public Vertex {
public:
  static void compute() {}
};

// This needs to be an equivalent statement of the vertex state of the
// WorkerClassNx1 vertex.  The supervisor contructs this struct, and the
// worker accesses the same thing, as if it is a vertex state
struct WorkerStateNx1 {
  const quarter *inChanPtr;
  const quarter *metadataUnused; // The vertex state quarter input adds this ptr
  half *outChanPtr;
  unsigned strides;
  const unsigned *partitionList;
  const unsigned *partitionBase;
};

template <> class WorkerClassNx1<unsigned short, 16> : public Vertex {
public:
  Input<Vector<quarter, ONE_PTR, 8>> inChanPtr;
  InOut<Vector<half, ONE_PTR, 8>> outChanPtr;
  unsigned strides;
  Input<Vector<unsigned, ONE_PTR>> partitionList;
  Input<Vector<unsigned, ONE_PTR>> partitionBase;

  void compute() {
    unsigned deltaNData = *(&partitionList[0] + getWid());
    unsigned workListLength = deltaNData >> DELTAN_OFFSET_BITS;
    unsigned offset = deltaNData - (workListLength << DELTAN_OFFSET_BITS);
    const unsigned short *workListPtr =
        reinterpret_cast<const unsigned short *>(&partitionBase[0]);
    workListPtr += offset;

    constexpr auto outputVectorWidth = 4;
    const unsigned short *workListEndPtr = workListPtr + workListLength;

    while (workListPtr < workListEndPtr) {
      auto outPtr = ld64StepToIncPtr(&outChanPtr[0], *workListPtr++);
      int loops = *reinterpret_cast<const short *>(workListPtr++);
      auto inPtr = ld64StepToIncPtr(&inChanPtr[0], *workListPtr++);
      convQuarterHalfLoop<false>(inPtr, outPtr, loops, strides);
    }
  }
};

static __attribute__((always_inline)) unsigned
divideWork(const unsigned size, const unsigned vectorWidthShifts,
           const unsigned worker) {
  // Multiply by 0xaaab and shift by 18 is just a multiplication by 1/6 in
  // fixed point (Q14.18)
  return (((size >> vectorWidthShifts) + 5 - worker) * 0xaaab) >> 18;
}

// This needs to be an equivalent statement of the vertex state of the
// WorkerMemZero vertex.  The supervisor contructs this struct, and the
// worker accesses the same thing, as if it is a vertex state
struct WorkerMemZeroState {
  half *outPtr;
  unsigned zerosInfo;
};

class WorkerMemZero : public Vertex {
public:
  Output<Vector<half, ONE_PTR, 8>> outPtr;
  unsigned zerosInfo;

  void compute() {
    // All workers write the last 2 halves (treated as a float) which won't be
    // covered by the loop.  Only necessary where the number of elements is not
    // a multiple of 4 which the loop will deal with, but executed regardless.
    float *last2Elems = reinterpret_cast<float *>(&outPtr[0] + zerosInfo - 2);
    *last2Elems = 0.0f;

    const auto wid = getWid();
    unsigned loops = divideWork(zerosInfo, 2, wid);
    auto wrPtr = &outPtr[0];
    asm volatile(
        R"l(
          .align 8
          ld64step $azeros, $mzero, %[wrPtr]+=, %[workerOffset]
          rpt %[loops], (2f - 1f) / 8 -1
          1:
            {st64step $azeros, $mzero, %[wrPtr]+=, 6
             fnop}
           2:
        )l"
        : [wrPtr] "+r"(wrPtr)
        : [workerOffset] "r"(wid), [loops] "r"(loops)
        : "$m0", "memory");
  }
};

template <bool use128BitLoad, unsigned numConvUnits,
          unsigned convInputLoadElems, bool disableSR>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartialnx1<
    quarter, half, true, use128BitLoad, numConvUnits, convInputLoadElems,
    disableSR> : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

public:
  ConvPartialnx1();
  using FPType = quarter;
  using AccumType = half;

  using WorkListType = unsigned short;
  using WorkListNumFieldType = short;
  using UnsignedType = unsigned short;
  using SignedType = short;
  using PackedStridesType = unsigned;

  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  static constexpr unsigned numStrideBits = NUM_STRIDE_BITS;
  static constexpr unsigned strideMask = (1 << numStrideBits) - 1;
  // This value is
  // (inStrideX - 1 - (ampKernelHeight - 1) * inRowStride)
  //      * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  const PackedStridesType transformedInStride;
  // This output stride also encodes the flip parameter and is given as
  // -6 + outChansPerGroup * (actual output stride) if flipOut = false
  // -6 - outChansPerGroup * (actual output stride) if flipOut = true
  const PackedStridesType transformedOutStride;

  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, COMPACT_PTR, weightsAlign, use128BitLoad>>,
         ONE_PTR>
      weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 8, true>>, ONE_PTR> out;
  const unsigned zerosInfo;
  Input<VectorList<WorkListType, COMPACT_DELTAN>> worklists;
  const UnsignedType numOutGroupsM1;
  const UnsignedType numInGroups;
  const UnsignedType kernelOuterSizeM1;
  const UnsignedType kernelInnerElementsM1;

  const UnsignedType numConvGroupsM1;
  // The number of kernel elements we accumulate across within the AMP unit
  const UnsignedType ampKernelHeightM1;
  // The actual coding of this is
  //  (inRowSride - 1) * inChansPerGroup / convInputLoadElems + 1
  const SignedType transformedInRowStride;
  const UnsignedType outChansPerGroup;
  const UnsignedType inChansPerGroup;

  __attribute__((target("supervisor"))) void compute() {
    unsigned srStore;
    if constexpr (disableSR) {
      srStore = getFPICTL();
      putFPICTL(srStore & stocasticRoundingMask);
    }
    WorkerStateNx1 workerState;
    auto wlStatePtr = reinterpret_cast<unsigned *>(&worklists);
    workerState.partitionBase =
        reinterpret_cast<unsigned *>(*wlStatePtr & DELTAN_OFFSET_MASK);

    const auto weightsMetadata = *weights.getMetadata();
    const auto inMetadata = *in.getMetadata();
    setFp8Format(weightsMetadata, inMetadata);
    setFp8Scale(weightsMetadata, inMetadata);

    // A small amount of manipulation on the passed strides.
    // This could be avoided by packing differently for this vertex but this
    // way it's compatible with others
    constexpr auto packedStrideSize = 32;
    auto unpackedTransformedInStride =
        transformedInStride << (packedStrideSize - 2 * numStrideBits);
    auto inStride =
        (unpackedTransformedInStride >> (packedStrideSize - numStrideBits)) -
        (transformedInRowStride * 2);
    workerState.strides =
        packStrides(inStride & strideMask, transformedOutStride & strideMask,
                    numStrideBits);
    // Zeroing - using a worker function with 64 bit writes, rpt and bundles
    const unsigned numOutGroups = numOutGroupsM1 + 1;

    WorkerMemZeroState workerMemZeroState;
    workerMemZeroState.zerosInfo = zerosInfo;
    unsigned *workerFunction;
    SET_ADDR(workerFunction, "__runCodelet_poplin__WorkerMemZero");
    for (unsigned cg = 0; cg <= numConvGroupsM1; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        // Sync before changing vertex state
        syncWorkers();
        workerMemZeroState.outPtr = &out[cg * numOutGroups + og][0];
        runAll(workerFunction, &workerMemZeroState);
        // No need to sync until next time we want to run
      }
    }

    const unsigned ampKernelHeight = ampKernelHeightM1 + 1;
    const unsigned kernelInnerElements = kernelInnerElementsM1 + 1;

    for (unsigned cg = 0; cg <= numConvGroupsM1; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        workerState.outChanPtr = &out[cg * numOutGroups + og][0];
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          auto partitionList = reinterpret_cast<unsigned *>(*(wlStatePtr + 1) &
                                                            DELTAN_OFFSET_MASK);
          workerState.partitionList = partitionList;
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];

          for (unsigned ky = 0; ky <= kernelOuterSizeM1; ++ky) {
            for (unsigned kx = 0; kx < kernelInnerElements; ++kx) {

              // Amp kernel height loop extracted out - supervisor function,
              // affecting weight load.
              const auto weightIndex = ky * ampKernelHeight *
                                           kernelInnerElements *
                                           outChansPerGroup * inChansPerGroup +
                                       kx * outChansPerGroup * inChansPerGroup;
              SET_ADDR(
                  workerFunction,
                  "__runCodelet_poplin__WorkerClassNx1___unsigned_short_16")
              // Don't change weights or workerState until synced
              __builtin_ipu_put(reinterpret_cast<unsigned>(&w[weightIndex]),
                                CSR_S_CCCSLOAD__INDEX);
              syncWorkers();
              ampLoadWeights<use128BitLoad, numConvUnits>();
              workerState.inChanPtr = &in[cg * numInGroups + ig][0];
              workerState.partitionList = partitionList;

              runAll(workerFunction, &workerState);
              partitionList += CTXT_WORKERS;
            }
          }
          // Outer loops can change worker state too, so sync
          syncWorkers();
        }
      }
    }
    syncWorkers();
    if constexpr (disableSR) {
      putFPICTL(srStore);
    }
  }
};

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

INSTANTIATE_WEIGHTS_128(quarter, half, 16, 8);

} // end namespace poplin
