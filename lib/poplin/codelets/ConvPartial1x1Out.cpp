// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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
  Vector<Output<Vector<AccumType, ONE_PTR, 16, true>>, ONE_PTR, 4> out;
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
  bool compute();

  // For C++ codelets, a MultiVertex is used.
  bool compute(unsigned wid) {
    const auto usedContexts = CTXT_WORKERS;

    // modify to set actual values used by vertex
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned numOutGroups = numOutGroupsM1 + 1;

    const int inStride = reverseTransfromedInStride(
        transformedInStride, convInputLoadElems, inChansPerGroup);

    // For AMP 1x1 output stride is always 1 hence calling
    // reverseTransfromedOutStride just to get flipOut parameter
    const bool flipOut =
        reverseTransfromedOutStride(transformedOutStride,
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
    return true;
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
  static bool compute() { return true; }
};

template <> class WorkerClass1x1<unsigned short, true, 16> : public Vertex {
public:
  static bool compute() {
    auto partitionOffset = 3 * getWid();
    auto state = workerState<WorkerState1x1<unsigned short>>();
    int loops = *reinterpret_cast<const short *>(state->partition +
                                                 partitionOffset + 1) +
                3;
    constexpr auto outputVectorWidth = 4;
    constexpr auto inputVectorWidth = 8;

    auto outPtr = state->outChanPtr +
                  state->partition[partitionOffset] * outputVectorWidth;
    auto inPtr = state->inChanPtr +
                 state->partition[partitionOffset + 2] * inputVectorWidth;
    convQuarterHalfLoop<true>(inPtr, outPtr, loops, state->strides);
    return true;
  }
};

template <> class WorkerClass1x1<unsigned short, false, 16> : public Vertex {
public:
  static bool compute() {
    auto partitionOffset = 3 * getWid();
    auto state = workerState<WorkerState1x1<unsigned short>>();
    int loops = *reinterpret_cast<const short *>(state->partition +
                                                 partitionOffset + 1) +
                3;
    constexpr auto outputVectorWidth = 4;
    constexpr auto inputVectorWidth = 8;

    auto outPtr = state->outChanPtr +
                  state->partition[partitionOffset] * outputVectorWidth;
    auto inPtr = state->inChanPtr +
                 state->partition[partitionOffset + 2] * inputVectorWidth;
    convQuarterHalfLoop<false>(inPtr, outPtr, loops, state->strides);
    return true;
  }
};

template <bool useLimitedVer, bool use128BitLoad, unsigned numConvUnits,
          unsigned convInputLoadElems, bool disableSR>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartial1x1Out<
    quarter, half, useLimitedVer, use128BitLoad, numConvUnits,
    convInputLoadElems, disableSR> : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

public:
  ConvPartial1x1Out();
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
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR, 4> in;
  Vector<Input<Vector<FPType, ONE_PTR, weightsAlign, use128BitLoad>>, ONE_PTR,
         4>
      weights;
  Vector<Output<Vector<AccumType, ONE_PTR, 16, true>>, ONE_PTR, 4> out;
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

  __attribute__((target("supervisor"))) bool compute() {
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

    // Strides for use with tapack
    workerState.strides =
        packStrides(transformedInStride, inOutStrides & NUM_STRIDE_BITS_MASK);

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
          syncWorkers();
          ampLoadWeights<use128BitLoad, numConvUnits>(w);
          workerState.inChanPtr = &in[cg * numInGroups + ig][0];
          runAll(workerFunction, &workerState);
          SET_ADDR(
              workerFunction,
              "__runCodelet_poplin__WorkerClass1x1___unsigned_short_false_16")
        }
      }
    }
    syncWorkers();
    return true;
  }
};

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

INSTANTIATE_WEIGHTS_128(quarter, half, 16, 8);

} // end namespace poplin
