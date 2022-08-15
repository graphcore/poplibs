// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define __QUARTER_FOR_IPU__
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "ConvPartialsStridesPacking.hpp"
#include "convCastSupport.hpp"
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

#ifdef __IPU__
#include "inlineAssemblerConv.hpp"
#endif

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

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
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 * useLimitedVer is "true" if there are constraints imposed on
 * - size of strides are bounded to strides supported by ISA
 * - worklists-offsets are bounded to fit 16-bits
 * - worklists-number of elements <= maximum count supported by rpt instruction
 **/
template <class FPType, class AccumType, bool useLimitedVer, bool use128BitLoad,
          unsigned numConvUnits, unsigned convInputLoadElems, bool disableSR>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartial1x1Out
    : public std::conditional_t<
          hasAssembly<FPType, AccumType, useLimitedVer, numConvUnits,
                      convInputLoadElems>() &&
              EXTERNAL_CODELET,
          SupervisorVertex, MultiVertex> {
  static const bool needsAlignWorkers = false;

public:
  ConvPartial1x1Out();

  using WorkListType =
      std::conditional_t<useLimitedVer, unsigned short, unsigned>;
  using WorkListNumFieldType = std::conditional_t<useLimitedVer, short, int>;
  using UnsignedType =
      std::conditional_t<useLimitedVer, unsigned short, unsigned>;
  using SignedType = std::conditional_t<useLimitedVer, short, int>;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR, 4> in;
  Vector<Input<Vector<FPType, ONE_PTR, weightsAlign, use128BitLoad>>, ONE_PTR,
         4>
      weights;
  static constexpr unsigned outAlign = std::is_same_v<FPType, quarter> ? 8 : 16;
  Vector<Output<Vector<AccumType, ONE_PTR, outAlign, true>>, ONE_PTR, 4> out;
  Input<Vector<WorkListType, ONE_PTR, 4>> worklists;
  const UnsignedType numConvGroupsM1;
  // Actual value is 1 more than this
  const UnsignedType numOutGroupsM1;
  const UnsignedType numInGroups;
  // This value is
  // (inStrideX - 1) * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  const SignedType transformedInStride;
  const UnsignedType outChansPerGroup;
  // This stride encodes the flip out parameter
  const SignedType transformedOutStride;
  const UnsignedType inChansPerGroup;

  IS_EXTERNAL_CODELET((hasAssembly<FPType, AccumType, useLimitedVer,
                                   numConvUnits, convInputLoadElems>()));

  // For assembly codelets, a true SupervisorVertex is used.
  void compute();

  // For C++ codelets, a MultiVertex is used.
  void compute(unsigned wid) {
    const auto usedContexts = CTXT_WORKERS;

    // modify to set actual values used by vertex
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned numOutGroups = numOutGroupsM1 + 1;

    const int inStride = reverseTransformedInStride(
        transformedInStride, convInputLoadElems, inChansPerGroup);

    // For AMP 1x1 output stride is always 1 hence calling
    // reverseTransformedOutStride just to get flipOut parameter
    const bool flipOut =
        reverseTransformedOutStride(transformedOutStride,
                                    std::is_same<AccumType, float>(),
                                    numConvUnits, outChansPerGroup)
            .first;

    const UnsignedType accumTypeSize = std::is_same<AccumType, float>() ? 4 : 2;
    const UnsignedType typeSize = std::is_same<FPType, float>()
                                      ? 4
                                      : (std::is_same<FPType, half>() ? 2 : 1);

    // TODO(T35918): Use this struct inside the worklists' definition.
    struct WorklistEntry {
      WorkListType outOffset;
      WorkListNumFieldType numFieldElems;
      WorkListType inOffset;
    };
    const WorklistEntry *worklists_ =
        reinterpret_cast<const WorklistEntry *>(&worklists[0]);

    quarter_metadata inMetadata, weightsMetadata;
    if constexpr (std::is_same<FPType, quarter>::value) {
      inMetadata = unpackMetadata(in.getMetadata());
      weightsMetadata = unpackMetadata(weights.getMetadata());
    }
    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          const auto outRow = cg * numOutGroups + og;
          const auto inRow = cg * numInGroups + ig;
          const auto wRow = cg * numOutGroups * numInGroups +
                            ig * numOutGroups + (numOutGroups - 1 - og);
          WorklistEntry entry = worklists_[wid];
          // Decode the worklist offsets.
          entry.outOffset /= (outChansPerGroup * accumTypeSize) / 8;
          entry.inOffset /= (inChansPerGroup * typeSize) / 8;
          // The numFieldElems values from worklist is less by 3
          entry.numFieldElems += 3;

          for (unsigned i = 0; i < entry.numFieldElems; ++i) {
            for (unsigned outChan = 0; outChan < outChansPerGroup; ++outChan) {
              const auto outCol =
                  (entry.outOffset + (flipOut ? -i : i)) * outChansPerGroup;
              const auto inCol =
                  (entry.inOffset + i * inStride) * inChansPerGroup;
              const auto wCol = outChan * inChansPerGroup;

              float sum = 0;
              for (unsigned inChan = 0; inChan < inChansPerGroup; ++inChan) {
                sum += promoteType<FPType, float>(in[inRow][inCol + inChan],
                                                  inMetadata) *
                       promoteType<FPType, float>(weights[wRow][wCol + inChan],
                                                  weightsMetadata);
              }

              if (ig == 0)
                out[outRow][outCol + outChan] = sum;
              else
                out[outRow][outCol + outChan] += sum;
            }
          }
        }
      }
    }
  }
};

#define INSTANTIATE_DISABLE_SR(inType, parType, limited, weights128,           \
                               convUnits, inputElems)                          \
  template class ConvPartial1x1Out<inType, parType, limited, weights128,       \
                                   convUnits, inputElems, true>;               \
  template class ConvPartial1x1Out<inType, parType, limited, weights128,       \
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
template <typename UnsignedType, bool zeroPartials, unsigned numConvUnits>
class WorkerClass1x1 : public Vertex {
public:
  static void compute() {}
};

// This needs to be an equivalent statement of the vertex state of the
// WorkerClass1x1 vertex.  The supervisor contructs this struct, and the
// worker accesses the same thing, as if it is a vertex state
template <typename UnsignedType> struct WorkerState1x1 {
  const quarter *inChanPtr;
  const quarter *metadataUnused; // The vertex state quarter input adds this ptr
  half *outChanPtr;
  unsigned strides;
  const UnsignedType *partition;
};

template <bool zeroPartials>
class WorkerClass1x1<unsigned short, zeroPartials, 16> : public Vertex {
public:
  Input<Vector<quarter, ONE_PTR, 8>> inChanPtr;
  InOut<Vector<half, ONE_PTR, 8>> outChanPtr;
  unsigned strides;
  Input<Vector<unsigned short, ONE_PTR>> partition;

  void compute() {
    auto partitionOffset = 3 * getWid();
    int loops =
        *reinterpret_cast<const short *>(&partition[partitionOffset + 1]);
    auto inPtr =
        ld64StepToIncPtr(&inChanPtr[0], partition[partitionOffset + 2]);
    auto outPtr = ld64StepToIncPtr(&outChanPtr[0], partition[partitionOffset]);

    convQuarterHalfLoop<zeroPartials>(inPtr, outPtr, loops, strides);
  }
};

template class WorkerClass1x1<unsigned short, true, 16>;
template class WorkerClass1x1<unsigned short, false, 16>;

template <bool use128BitLoad, unsigned numConvUnits,
          unsigned convInputLoadElems, bool disableSR>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartial1x1Out<
    quarter, half, true, use128BitLoad, numConvUnits, convInputLoadElems,
    disableSR> : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

public:
  ConvPartial1x1Out();
  using FPType = quarter;
  using AccumType = half;

  using WorkListType = unsigned short;
  using WorkListNumFieldType = short;
  using UnsignedType = unsigned short;
  using SignedType = short;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR, 4> in;
  Vector<Input<Vector<FPType, ONE_PTR, weightsAlign, use128BitLoad>>, ONE_PTR,
         4>
      weights;
  Vector<Output<Vector<AccumType, ONE_PTR, 8, true>>, ONE_PTR, 4> out;
  Input<Vector<WorkListType, ONE_PTR, 4>> worklists;
  const UnsignedType numConvGroupsM1;
  // Actual value is 1 more than this
  const UnsignedType numOutGroupsM1;
  const UnsignedType numInGroups;
  // This value is
  // (inStrideX - 1) * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  const SignedType transformedInStride;
  const UnsignedType outChansPerGroup;
  // This stride encodes the flip out parameter
  const SignedType transformedOutStride;
  const UnsignedType inChansPerGroup;

  __attribute__((target("supervisor"))) void compute() {
    unsigned srStore;
    if constexpr (disableSR) {
      srStore = getFPICTL();
      putFPICTL(srStore & stocasticRoundingMask);
    }
    WorkerState1x1<UnsignedType> workerState;
    workerState.partition = &worklists[0];
    const auto weightsMetadata = *weights.getMetadata();
    const auto inMetadata = *in.getMetadata();
    setFp8Format(weightsMetadata, inMetadata);
    setFp8Scale(weightsMetadata, inMetadata);

    // modify to set actual values used by vertex
    const unsigned numOutGroups = numOutGroupsM1 + 1;

    // For AMP 1x1 output stride is always 1 unless flipOut = true
    constexpr auto outStrideThresh = -4;
    const auto flipOut = transformedOutStride < outStrideThresh;
    // Stride for memory initialisation
    const auto inOutStrides = flipOut ? -1 * (numConvUnits >> 1) + 1 : 1;

    static constexpr unsigned numStrideBits = NUM_STRIDE_BITS;
    static constexpr unsigned strideMask = (1 << numStrideBits) - 1;

    // Strides for use with tapack
    workerState.strides = packStrides(transformedInStride,
                                      inOutStrides & strideMask, numStrideBits);

    for (unsigned cg = 0; cg <= numConvGroupsM1; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        unsigned *workerFunction;
        SET_ADDR(workerFunction,
                 "__runCodelet_poplin__WorkerClass1x1___unsigned_short_true_16")
        // Don't change weights or workerState until synced
        syncWorkers();
        workerState.outChanPtr = &out[cg * numOutGroups + og][0];
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          const auto *w =
              &weights[cg * numOutGroups * numInGroups + ig * numOutGroups +
                       (numOutGroups - 1 - og)][0];
          // Don't change weights or workerState until synced
          __builtin_ipu_put(reinterpret_cast<unsigned>(w),
                            CSR_S_CCCSLOAD__INDEX);
          syncWorkers();
          ampLoadWeights<use128BitLoad, numConvUnits>();
          workerState.inChanPtr = &in[cg * numInGroups + ig][0];
          runAll(workerFunction, &workerState);
          SET_ADDR(
              workerFunction,
              "__runCodelet_poplin__WorkerClass1x1___unsigned_short_false_16")
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
