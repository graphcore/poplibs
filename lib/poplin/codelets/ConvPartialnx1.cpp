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
          unsigned numConvUnits>
constexpr bool hasAssembly() {
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
          unsigned numConvUnits>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartialnx1
    : public SupervisorVertexIf<
          hasAssembly<FPType, AccumType, useLimitedVer, numConvUnits>() &&
          ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;

public:
  ConvPartialnx1();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using WorkListNumFieldType =
      typename std::conditional<useLimitedVer, short, int>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType = typename std::conditional<useLimitedVer, short, int>::type;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;

  // This value is
  // (inStrideX - 1 - (ampKernelHeight - 1) * inRowStride)
  //      * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  // const SignedType transformedInStride;
  const signed transformedInStride;
  // This output stride also encodes the flip parameter and is given as
  // -6 + outChansPerGroup * (actual output stride) if flipOut = false
  // -6 - outChansPerGroup * (actual output stride) if flipOut = true
  // const SignedType transformedOutStride;
  const signed transformedOutStride;

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

  IS_EXTERNAL_CODELET(
      (hasAssembly<FPType, AccumType, useLimitedVer, numConvUnits>()));

  bool compute() {
    const unsigned numWorkers = CTXT_WORKERS;
    const unsigned convInputLoadElems =
        std::is_same<FPType, float>::value ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT
        : std::is_same<FPType, half>::value
            ? CONV_UNIT_INPUT_LOAD_ELEMS_HALF
            : CONV_UNIT_INPUT_LOAD_ELEMS_QUARTER;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned ampKernelHeight = ampKernelHeightM1 + 1;
    const unsigned kernelOuterSize = kernelOuterSizeM1 + 1;
    const unsigned kernelInnerElements = kernelInnerElementsM1 + 1;

    const int convOutputStoreElems = std::is_same<AccumType, half>() ? 4 : 2;
    const int packedTransformedInStrideReg = transformedInStride;
    const int packedTransformedOutStrideReg = transformedOutStride;
    const int secondPtrOffset = transformedInRowStride;

    // Unpack registers strides into transformed strides
    const int unpackedTransformedInRowStride =
        unpackAmpNx1InRowStride(ampKernelHeight, secondPtrOffset);
    int unpackedTransformedInStride =
        unpackAmpNx1Stride(NUM_STRIDE_BITS, packedTransformedInStrideReg, 1);
    int unpackedTransformedOutStride =
        unpackAmpNx1Stride(NUM_STRIDE_BITS, packedTransformedOutStrideReg, 0);

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

    const int inRowStride = reverseTransfromedInRowStride(
        unpackedTransformedInRowStride, convInputLoadElems, inChansPerGroup);
    const int inStride = reverseTransfromedInStride(
        unpackedTransformedInStride, convInputLoadElems, inChansPerGroup,
        ampKernelHeight, inRowStride);

    const auto usedContexts =
        worklists.size() / (kernelOuterSize * kernelInnerElements);

    bool flipOut;
    int outStride;
    std::tie(flipOut, outStride) = reverseTransfromedOutStride(
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
              for (unsigned context = 0; context < usedContexts; ++context) {
                const auto k = (ky * kernelInnerElements + kx);
                const auto &wl = worklists[k * usedContexts + context];
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
    return true;
  }
};

#ifdef __IPU__

#if __IPU_ARCH_VERSION__ >= 21

template <typename UnsignedType, unsigned numConvUnits>
class WorkerClassNx1 : public Vertex {
public:
  static bool compute() { return true; }
};

template <> class WorkerClassNx1<unsigned short, 16> : public Vertex {
public:
  static bool compute() {
    auto state = workerState<WorkerStateNx1>();

    unsigned deltaNData = *(state->partitionList + getWid());
    unsigned workListLength = deltaNData >> DELTAN_OFFSET_BITS;
    unsigned offset = deltaNData - (workListLength << DELTAN_OFFSET_BITS);
    const unsigned short *workListPtr =
        reinterpret_cast<const unsigned short *>(state->partitionBase);
    workListPtr += offset;

    constexpr auto outputVectorWidth = 4;
    constexpr auto inputVectorWidth = 8;
    const unsigned short *workListEndPtr = workListPtr + workListLength;

    while (workListPtr < workListEndPtr) {
      int loops = (*reinterpret_cast<const short *>(workListPtr + 1) + 3);
      auto outPtr = state->outChanPtr + workListPtr[0] * outputVectorWidth;
      auto inPtr = state->inChanPtr + workListPtr[2] * inputVectorWidth;
      workListPtr += 3;
      convQuarterHalfLoop(inPtr, outPtr, loops, state->strides);
    }
    return true;
  }
};

struct WorkerMemZeroState {
  half *outPtr;
  unsigned zerosInfo;
};

static __attribute__((always_inline)) unsigned
divideWork(const unsigned size, const unsigned vectorWidthShifts,
           const unsigned worker) {
  // Multiply by 0xaaab and shift by 18 is just a multiplication by 1/6 in
  // fixed point (Q14.18)
  return (((size >> vectorWidthShifts) + 5 - worker) * 0xaaab) >> 18;
}

class WorkerMemZero : public Vertex {
public:
  static bool compute() {
    auto state = workerState<WorkerMemZeroState>();
    asm volatile(" mov %[state], $mvertex_base\n" : [state] "=r"(state) : :);

    // All workers write the last 2 halves (treated as a float) which won't be
    // covered by the loop.  Only necessary where the number of elements is not
    // a multiple of 4 which the loop will deal with, but executed regardless.
    float *last2Elems =
        reinterpret_cast<float *>(state->outPtr + (state->zerosInfo) - 2);
    *last2Elems = 0.0f;

    const auto wid = getWid();
    unsigned loops = divideWork(state->zerosInfo, 2, wid);
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
        : [wrPtr] "+r"(state->outPtr)
        : [workerOffset] "r"(wid), [loops] "r"(loops)
        : "$m0", "memory");

    return true;
  }
};

template <bool useLimitedVer, bool use128BitLoad, unsigned numConvUnits>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartialnx1<
    quarter, half, useLimitedVer, use128BitLoad, numConvUnits>
    : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

public:
  ConvPartialnx1();
  using FPType = quarter;
  using AccumType = half;

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using WorkListNumFieldType =
      typename std::conditional<useLimitedVer, short, int>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType = typename std::conditional<useLimitedVer, short, int>::type;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;

  // This value is
  // (inStrideX - 1 - (ampKernelHeight - 1) * inRowStride)
  //      * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  // const SignedType transformedInStride;
  const signed transformedInStride;
  // This output stride also encodes the flip parameter and is given as
  // -6 + outChansPerGroup * (actual output stride) if flipOut = false
  // -6 - outChansPerGroup * (actual output stride) if flipOut = true
  // const SignedType transformedOutStride;
  const signed transformedOutStride;

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

  __attribute__((target("supervisor"))) bool compute() {
    WorkerStateNx1 workerState;
    auto wlStatePtr = reinterpret_cast<unsigned *>(&worklists);
    workerState.partitionBase =
        reinterpret_cast<unsigned *>(*wlStatePtr & DELTAN_OFFSET_MASK);

    const auto weightsMetaData = *weights.getMetadata();
    const auto inMetaData = *in.getMetadata();
    setFp8Format(weightsMetaData, inMetaData);
    setFp8Scale(weightsMetaData, inMetaData);

    // A small amount of manipulation on the passed strides.
    // This could be avoided by packing differently for this vertex but this
    // way it's compatible with others
    constexpr auto unsignedSize = 32;
    int unpackedTransformedInStride = transformedInStride
                                      << (unsignedSize - 2 * NUM_STRIDE_BITS);
    int inStride =
        (unpackedTransformedInStride >> (unsignedSize - NUM_STRIDE_BITS)) -
        transformedInRowStride;
    workerState.strides =
        packStrides(inStride, transformedOutStride & NUM_STRIDE_BITS_MASK);
    // Zeroing - using a worker function with 64 bit writes, rpt and bundles
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    WorkerMemZeroState workerMemZeroState;
    workerMemZeroState.zerosInfo = zerosInfo;
    for (unsigned cg = 0; cg != numConvGroups; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        workerMemZeroState.outPtr = &out[cg * numOutGroups + og][0];
        RUN_ALL("__runCodelet_poplin__WorkerMemZero", &workerMemZeroState);
      }
    }

    const unsigned ampKernelHeight = ampKernelHeightM1 + 1;
    const unsigned kernelOuterSize = kernelOuterSizeM1 + 1;
    const unsigned kernelInnerElements = kernelInnerElementsM1 + 1;

    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        workerState.outChanPtr = &out[cg * numOutGroups + og][0];
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          workerState.partitionList = reinterpret_cast<unsigned *>(
              *(wlStatePtr + 1) & DELTAN_OFFSET_MASK);

          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];

          for (unsigned ky = 0; ky < kernelOuterSize; ++ky) {
            for (unsigned kx = 0; kx < kernelInnerElements; ++kx) {

              // Amp kernel height loop extracted out - supervisor function,
              // affecting weight load.
              const auto weightIndex = ky * ampKernelHeight *
                                           kernelInnerElements *
                                           outChansPerGroup * inChansPerGroup +
                                       kx * outChansPerGroup * inChansPerGroup;

              ampLoadWeights<use128BitLoad, numConvUnits>(&w[weightIndex]);
              workerState.inChanPtr = &in[cg * numInGroups + ig][0];

              RUN_ALL("__runCodelet_poplin__WorkerClassNx1___unsigned_short_16",
                      &workerState)
              // Advance for the next loop
              workerState.partitionList += CTXT_WORKERS;
            }
          }
        }
      }
    }
    return true;
  }
};

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

template class ConvPartialnx1<float, float, true, false, 8>;
template class ConvPartialnx1<half, half, true, false, 8>;
template class ConvPartialnx1<half, float, true, false, 8>;
template class ConvPartialnx1<float, float, false, false, 8>;
template class ConvPartialnx1<half, half, false, false, 8>;
template class ConvPartialnx1<half, float, false, false, 8>;

template class ConvPartialnx1<float, float, true, true, 8>;
template class ConvPartialnx1<half, half, true, true, 8>;
template class ConvPartialnx1<half, float, true, true, 8>;
template class ConvPartialnx1<float, float, false, true, 8>;
template class ConvPartialnx1<half, half, false, true, 8>;
template class ConvPartialnx1<half, float, false, true, 8>;

template class ConvPartialnx1<float, float, true, false, 16>;
template class ConvPartialnx1<half, half, true, false, 16>;
template class ConvPartialnx1<float, float, false, false, 16>;
template class ConvPartialnx1<half, half, false, false, 16>;

template class ConvPartialnx1<float, float, true, true, 16>;
template class ConvPartialnx1<half, half, true, true, 16>;
template class ConvPartialnx1<float, float, false, true, 16>;
template class ConvPartialnx1<half, half, false, true, 16>;

template class ConvPartialnx1<quarter, half, false, false, 16>;
template class ConvPartialnx1<quarter, half, false, true, 16>;
template class ConvPartialnx1<quarter, half, true, false, 16>;
template class ConvPartialnx1<quarter, half, true, true, 16>;

} // end namespace poplin
