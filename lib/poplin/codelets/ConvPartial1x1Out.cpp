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

static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

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
 * Compute a sum of 1x1 convolutions over a subset of the input channels for
 * multiple output channels.
 * useLimitedVer is "true" if there are constraints imposed on
 * - size of strides are bounded to strides supported by ISA
 * - worklists-offsets are bounded to fit 16-bits
 * - worklists-number of elements <= maximum count supported by rpt instruction
 **/
template <class FPType, class AccumType, bool useLimitedVer, bool use128BitLoad,
          unsigned numConvUnits>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartial1x1Out
    : public SupervisorVertexIf<
          hasAssembly<FPType, AccumType, useLimitedVer, numConvUnits>() &&
          ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;

public:
  ConvPartial1x1Out();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using WorkListNumFieldType =
      typename std::conditional<useLimitedVer, short, int>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType = typename std::conditional<useLimitedVer, short, int>::type;
  static constexpr unsigned weightsAlign = use128BitLoad ? 16 : 8;
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, COMPACT_PTR, 4> in;
  Vector<Input<Vector<FPType, COMPACT_PTR, weightsAlign, use128BitLoad>>,
         COMPACT_PTR, 4>
      weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 16, true>>, COMPACT_PTR, 4> out;
  Input<Vector<WorkListType, COMPACT_PTR, 4>> worklists;
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

  IS_EXTERNAL_CODELET(
      (hasAssembly<FPType, AccumType, useLimitedVer, numConvUnits>()));

  bool compute() {
    const unsigned convInputLoadElems = std::is_same<FPType, float>::value
                                            ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT
                                            : CONV_UNIT_INPUT_LOAD_ELEMS_HALF;
    const unsigned usedContexts = CTXT_WORKERS;
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
#if __IPU_ARCH_VERSION__ >= 21
    if constexpr (std::is_same<FPType, quarter>::value) {
      inMetadata = unpackMetadata(in.getMetadata());
      weightsMetadata = unpackMetadata(weights.getMetadata());
    }
#endif
    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          const auto outRow = cg * numOutGroups + og;
          const auto inRow = cg * numInGroups + ig;
          const auto wRow = cg * numOutGroups * numInGroups +
                            ig * numOutGroups + (numOutGroups - 1 - og);
          for (unsigned context = 0; context < usedContexts; ++context) {
            WorklistEntry entry = worklists_[context];
            // Decode the worklist offsets.
            entry.outOffset /= (outChansPerGroup * accumTypeSize) / 8;
            entry.inOffset /= (inChansPerGroup * typeSize) / 8;
            // The numFieldElems values from worklist is less by 3
            entry.numFieldElems += 3;

            for (unsigned i = 0; i < entry.numFieldElems; ++i) {
              for (unsigned outChan = 0; outChan < outChansPerGroup;
                   ++outChan) {
                const auto outCol =
                    (entry.outOffset + (flipOut ? -i : i)) * outChansPerGroup;
                const auto inCol =
                    (entry.inOffset + i * inStride) * inChansPerGroup;
                const auto wCol = outChan * inChansPerGroup;

                float sum = 0;
                for (unsigned inChan = 0; inChan < inChansPerGroup; ++inChan) {
                  sum += promoteType<FPType, float>(in[inRow][inCol + inChan],
                                                    inMetadata) *
                         promoteType<FPType, float>(
                             weights[wRow][wCol + inChan], weightsMetadata);
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
    return true;
  }
};

#ifdef __IPU__

#if __IPU_ARCH_VERSION__ >= 21
template <typename UnsignedType, unsigned numConvUnits>
class WorkerClass1x1 : public Vertex {
public:
  static bool compute() { return true; }
};

template <> class WorkerClass1x1<unsigned short, 16> : public Vertex {
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
    if (state->firstTime) {
      asm volatile(
          R"l(
            .align 8
            mov $m0, %[wrPtr]
            rpt %[loops], (2f - 1f) / 8 - 1
            1:
              {st64step $azeros, $mzero, $m0+=, 1
               fnop}
              {st64step $azeros, $mzero, $m0+=, 1
               fnop}
              {st64step $azeros, $mzero, $m0+=, 1
               fnop}
              {st64step $azeros, $mzero, $m0+=,%[inOutStrides]
               fnop}
            2:
          )l"
          :
          : [wrPtr] "r"(outPtr), [loops] "r"(loops),
            [inOutStrides] "r"(state->inOutStrides)
          : "$m0", "memory");
    }
    convQuarterHalfLoop(inPtr, outPtr, loops, state->strides);
    return true;
  }
};

template <bool useLimitedVer, bool use128BitLoad, unsigned numConvUnits>
class [[poplar::constraint("elem(**in) != elem(**out)")]] ConvPartial1x1Out<
    quarter, half, useLimitedVer, use128BitLoad, numConvUnits>
    : public SupervisorVertex {
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
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, COMPACT_PTR, 4> in;
  Vector<Input<Vector<FPType, COMPACT_PTR, weightsAlign, use128BitLoad>>,
         COMPACT_PTR, 4>
      weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 16, true>>, COMPACT_PTR, 4> out;
  Input<Vector<WorkListType, COMPACT_PTR, 4>> worklists;
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
    const auto weightsMetaData = *weights.getMetadata();
    const auto inMetaData = *in.getMetadata();
    setFp8Format(weightsMetaData, inMetaData);
    setFp8Scale(weightsMetaData, inMetaData);

    // modify to set actual values used by vertex
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned numOutGroups = numOutGroupsM1 + 1;

    // For AMP 1x1 output stride is always 1 unless flipOut = true
    constexpr auto outStrideThresh = -4;
    const auto flipOut = transformedOutStride < outStrideThresh;
    // Stride for memory initialisation
    workerState.inOutStrides = flipOut ? -1 * (numConvUnits >> 1) + 1 : 1;

    auto inStride = (transformedInStride - 1) *
                    CONV_UNIT_INPUT_LOAD_ELEMS_HALF / inChansPerGroup;
    // Strides for use with tapack
    workerState.strides = packStrides(
        (1 + 4 * inStride), workerState.inOutStrides & NUM_STRIDE_BITS_MASK);

    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        workerState.outChanPtr = &out[cg * numOutGroups + og][0];
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          workerState.inChanPtr = &in[cg * numInGroups + ig][0];
          const auto *w =
              &weights[cg * numOutGroups * numInGroups + ig * numOutGroups +
                       (numOutGroups - 1 - og)][0];
          ampLoadWeights<use128BitLoad, numConvUnits>(w);
          workerState.firstTime = (ig == 0);
          RUN_ALL("__runCodelet_poplin__WorkerClass1x1___unsigned_short_16",
                  &workerState)
        }
      }
    }
    return true;
  }
};

#endif // __IPU_ARCH_VERSION__
#endif // __IPU__

template class ConvPartial1x1Out<half, half, true, false, 8>;
template class ConvPartial1x1Out<half, float, true, false, 8>;
template class ConvPartial1x1Out<float, half, true, false, 8>;
template class ConvPartial1x1Out<float, float, true, false, 8>;
template class ConvPartial1x1Out<half, half, false, false, 8>;
template class ConvPartial1x1Out<half, float, false, false, 8>;
template class ConvPartial1x1Out<float, half, false, false, 8>;
template class ConvPartial1x1Out<float, float, false, false, 8>;

template class ConvPartial1x1Out<half, half, true, true, 8>;
template class ConvPartial1x1Out<half, float, true, true, 8>;
template class ConvPartial1x1Out<float, half, true, true, 8>;
template class ConvPartial1x1Out<float, float, true, true, 8>;
template class ConvPartial1x1Out<half, half, false, true, 8>;
template class ConvPartial1x1Out<half, float, false, true, 8>;
template class ConvPartial1x1Out<float, half, false, true, 8>;
template class ConvPartial1x1Out<float, float, false, true, 8>;

template class ConvPartial1x1Out<half, half, true, false, 16>;
template class ConvPartial1x1Out<float, float, true, false, 16>;
template class ConvPartial1x1Out<half, half, false, false, 16>;
template class ConvPartial1x1Out<float, float, false, false, 16>;

template class ConvPartial1x1Out<half, half, true, true, 16>;
template class ConvPartial1x1Out<float, float, true, true, 16>;
template class ConvPartial1x1Out<half, half, false, true, 16>;
template class ConvPartial1x1Out<float, float, false, true, 16>;

template class ConvPartial1x1Out<quarter, half, true, false, 16>;
template class ConvPartial1x1Out<quarter, half, true, true, 16>;
template class ConvPartial1x1Out<quarter, half, false, false, 16>;
template class ConvPartial1x1Out<quarter, half, false, true, 16>;

} // end namespace poplin
