// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/QuarterFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

#include <type_traits>

#ifdef __IPU__
#include "inlineAssemblerConv.hpp"
#include "inlineAssemblerSLIC.hpp"
#include "inlineAssemblerSLICStride2.hpp"
#endif

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::ONE_PTR;
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTANELEMENTS;

namespace poplin {

template <typename FPType, typename AccumType, bool useShortTypes,
          unsigned windowWidth, unsigned numConvChains>
constexpr bool hasAssemblyVersion() {
  return std::is_same<FPType, half>::value &&
         (std::is_same<AccumType, float>::value ||
          std::is_same<AccumType, half>::value) &&
         useShortTypes && (windowWidth == 4) &&
         (numConvChains == 2 || numConvChains == 4);
}

template <typename FPType, typename AccumType, unsigned outStride,
          bool useShortTypes, unsigned windowWidth, unsigned numConvChains,
          unsigned convGroupsPerGroupVertexType>
class [[poplar::constraint(
    "elem(**in) != elem(**out)",
    "elem(**in) != elem(*outFieldBuffer)")]] ConvPartial1xNSLIC
    : public SupervisorVertexIf<
          hasAssemblyVersion<FPType, AccumType, useShortTypes, windowWidth,
                             numConvChains>() &&
          ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;

public:
  ConvPartial1xNSLIC();

  // Depending on whether strides/sizes fit, use short types for storage.
  using UnsignedType =
      std::conditional_t<useShortTypes, unsigned short, unsigned int>;
  using WorkListType =
      std::conditional_t<useShortTypes, unsigned short, unsigned int>;

  // A pointer for inputs per conv group group.
  Vector<Input<Vector<FPType, PTR_ALIGN64, 8>>, PTR_ALIGN32> in;
  // A weights pointer per sub-kernel/conv group group.
  Vector<Input<Vector<FPType, PTR_ALIGN64, 8>>, PTR_ALIGN32> weights;
  // A pointer for outputs per conv group group. Enforced to be 16-byte
  // aligned to allow use of ld2xst64pace instruction along with
  // `outFieldBuffer`.
  Vector<Output<Vector<AccumType, PTR_ALIGN64, 16, true>>, PTR_ALIGN32> out;
  // A pointer to a buffer with size of outputs per conv group group
  // + 8 bytes. Enforced to be 16-byte aligned and offset by 8 bytes to
  // allow use of ld2xst64pace instruction along with each entry in `out`.
  Output<Vector<AccumType, PTR_ALIGN64, 16, true>> outFieldBuffer;

  // Array with shape:
  // [numSubKernels * numWorkerContexts][numFieldRows * 3]
  Input<VectorList<WorkListType, DELTAN>> worklists;

  // Indicates which of 5 modes we operate in:
  //
  //  =0 ->  4 conv groups, 1 input channel,  1 output channel
  //     ->  8 conv groups, 1 input channel,  1 output channel
  //     -> 16 conv groups, 1 input channel,  1 output channel
  //  =1 ->  2 conv groups, 2 input channels, 2 output channels
  //  =2 ->  1 conv group,  4 input channels, 4 output channels
  //
  // This encodes conv groups per group, input channels per group
  // and output channels per group as these are tightly coupled (tripled?)
  // in this vertex.
  const unsigned char chansPerGroupLog2;
  // This essentially encodes whether numSubKernels is odd or not in a
  // way that avoids manipulating state on load in the assembly.
  const unsigned char outPtrLoadOffset;

  const UnsignedType numSubKernelsM1;
  const UnsignedType numConvGroupGroupsM1;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<FPType, AccumType, useShortTypes,
                                          windowWidth, numConvChains>()));

  bool compute() {
    constexpr unsigned outFieldBufferOffset = 200u / sizeof(AccumType);

    const unsigned chansPerGroup = 1 << chansPerGroupLog2;
    const unsigned convGroupsPerGroup =
        convGroupsPerGroupVertexType / chansPerGroup;

    const unsigned numSubKernels = numSubKernelsM1 + 1;
    const unsigned numConvGroupGroups = numConvGroupGroupsM1 + 1;

    for (unsigned cg = 0; cg < numConvGroupGroups; ++cg) {
      auto *lastOutBuffer = (!outPtrLoadOffset)
                                ? &outFieldBuffer[outFieldBufferOffset]
                                : &out[cg][0];
      auto *currOutBuffer = (!outPtrLoadOffset)
                                ? &out[cg][0]
                                : &outFieldBuffer[outFieldBufferOffset];
      for (unsigned kg = 0; kg < numSubKernels; ++kg) {
        const auto &w = weights[cg * numSubKernels + kg];

        for (unsigned context = 0; context < CTXT_WORKERS; ++context) {
          const auto &wl = worklists[kg * CTXT_WORKERS + context];
          unsigned wi = 0;
          while (wi < wl.size()) {
            const auto inOffset = wl[wi + 0];
            const auto outOffset = wl[wi + 1];
            const auto numFieldElems = wl[wi + 2];

            for (unsigned i = 0; i < numFieldElems * outStride;
                 i += outStride) {
              // From here we are replicating the core SLIC loop
              for (unsigned outChan = 0; outChan < chansPerGroup; ++outChan) {
                for (unsigned convGroup = 0; convGroup < convGroupsPerGroup;
                     ++convGroup) {

                  // Apply outStride to stride the output
                  const auto outIndex = (outOffset + i / outStride) *
                                            convGroupsPerGroup * chansPerGroup +
                                        convGroup * chansPerGroup + outChan;

                  // Implicitly zero partials on the first pass.
                  AccumType sum = 0;
                  if (kg > 0) {
                    sum = AccumType(lastOutBuffer[outIndex]);
                  }
                  for (unsigned k = 0; k < windowWidth; ++k) {
                    for (unsigned inChan = 0; inChan < chansPerGroup;
                         ++inChan) {
                      const auto inIndex = (inOffset + i + k) *
                                               convGroupsPerGroup *
                                               chansPerGroup +
                                           convGroup * chansPerGroup + inChan;
                      const auto weightIndex =
                          k * convGroupsPerGroup *
                              // outChansPerGroup * inChansPerGroup
                              chansPerGroup * chansPerGroup +
                          // convGroup * outChansPerGroup * inChansPerGroup
                          convGroup * chansPerGroup * chansPerGroup +
                          outChan * chansPerGroup + inChan;
                      sum += AccumType(in[cg][inIndex]) *
                             AccumType(w[weightIndex]);
                    }
                  }
                  currOutBuffer[outIndex] = sum;
                }
              }
            }
            wi += 3;
          }
        }
        std::swap(lastOutBuffer, currOutBuffer);
      }
    }
    return true;
  }
};

#ifdef __IPU__

#if __IPU_ARCH_VERSION__ >= 21

template <unsigned stride>
static __attribute__((always_inline)) void
f8v8hihoSLIC(const quarter *inPtr, half *partialsPtr, half *outPtr,
             unsigned strides, int loops, unsigned noImplicitZero,
             unsigned outVectorWidth) {
  // Process the first 4 output channels using weights=W0
  auto triAddr = __builtin_ipu_tapack(inPtr, partialsPtr, outPtr);
  if (noImplicitZero) {
    if (loops < 0) {
      F8v8HIHO_SLIC_LESS_THAN_5(TSLIC_F16V4_1x4_W0)
    } else {
      F8v8HIHO_SLIC(TSLIC_F16V4_1x4_W0)
    }
  } else {
    F8v8HIHO_SLIC_IMPLICIT_ZERO(TSLIC_F16V4_1x4_W0)
  }
  // Process the second 4 output channels using weights=W1
  triAddr = __builtin_ipu_tapack(inPtr, partialsPtr + outVectorWidth,
                                 outPtr + outVectorWidth);
  if (noImplicitZero) {
    if (loops < 0) {
      F8v8HIHO_SLIC_LESS_THAN_5(TSLIC_F16V4_1x4_W1)
    } else {
      F8v8HIHO_SLIC(TSLIC_F16V4_1x4_W1)
    }
  } else {
    F8v8HIHO_SLIC_IMPLICIT_ZERO(TSLIC_F16V4_1x4_W1)
  }
}

template <>
void f8v8hihoSLIC<2>(const quarter *inPtr, half *partialsPtr, half *outPtr,
                     unsigned strides, int loops, unsigned noImplicitZero,
                     unsigned outVectorWidth) {
  // Process the first 4 output channels using weights=W0
  auto triAddr = __builtin_ipu_tapack(inPtr, partialsPtr, outPtr);
  if (noImplicitZero) {
    F8v8HIHO_SLIC_STRIDE2(TSLIC_F16V4_1x4_W0)
  } else {
    F8v8HIHO_SLIC_IMPLICIT_ZERO_STRIDE2(TSLIC_F16V4_1x4_W0)
  }
  // Process the second 4 output channels using weights=W1
  triAddr = __builtin_ipu_tapack(inPtr, partialsPtr + outVectorWidth,
                                 outPtr + outVectorWidth);
  if (noImplicitZero) {
    F8v8HIHO_SLIC_STRIDE2(TSLIC_F16V4_1x4_W1)
  } else {
    F8v8HIHO_SLIC_IMPLICIT_ZERO_STRIDE2(TSLIC_F16V4_1x4_W1)
  }
}

template <typename UnsignedType, unsigned stride, unsigned numConvUnits>
class WorkerClass1xN : public Vertex {
public:
  static bool compute() { return true; }
};

template <typename UnsignedType, unsigned stride>
class WorkerClass1xN<UnsignedType, stride, 16> : public Vertex {
public:
  static bool compute() {
    auto state = workerState<WorkerState1xN>();
    unsigned deltaNData = *(state->partitionList + getWid());
    unsigned workListLength = deltaNData >> DELTAN_OFFSET_BITS;
    unsigned offset = deltaNData - (workListLength << DELTAN_OFFSET_BITS);
    const UnsignedType *workListPtr =
        reinterpret_cast<const UnsignedType *>(state->partitionBase);
    workListPtr += offset;

    int strides = (0) |       // 0b01 (0 for no stride to avoid overread)
                  (0 << 10) | // 0b10 (unused)
                  (2 << 20);  // 0b11 (out stride over 2nd call)
    const UnsignedType *workListEndPtr = workListPtr + workListLength;
    while (workListPtr < workListEndPtr) {
      constexpr unsigned inVectorWidth = 8;
      constexpr unsigned outVectorWidth = 4;
      constexpr unsigned outVectorsPerOuterLoop = 2;

      auto outPtr = state->outChanPtr +
                    outVectorsPerOuterLoop * outVectorWidth * workListPtr[1];
      auto partialsPtr = state->partialsChanPtr + outVectorsPerOuterLoop *
                                                      outVectorWidth *
                                                      workListPtr[1];
      auto inPtr = state->inChanPtr + workListPtr[0] * inVectorWidth;
      constexpr int slicPipeLength = stride == 1 ? 5 : 3;
      int loops =
          (CSR_W_REPEAT_COUNT__VALUE__MASK & workListPtr[2]) - slicPipeLength;
      f8v8hihoSLIC<stride>(inPtr, partialsPtr, outPtr, strides, loops,
                           state->noImplicitZero, outVectorWidth);
      workListPtr += 3;
    }
    return true;
  }
};

template class WorkerClass1xN<unsigned short, 1, 16>;
template class WorkerClass1xN<unsigned, 1, 16>;
template class WorkerClass1xN<unsigned short, 2, 16>;
template class WorkerClass1xN<unsigned, 2, 16>;

template <unsigned outStride, bool useShortTypes, unsigned windowWidth,
          unsigned numConvChains, unsigned convGroupsPerGroupVertexType>
class [[poplar::constraint(
    "elem(**in) != elem(**out)",
    "elem(**in) != "
    "elem(*"
    "outFieldBuffer"
    ")")]] ConvPartial1xNSLIC<quarter, half, outStride, useShortTypes,
                              windowWidth, numConvChains,
                              convGroupsPerGroupVertexType>
    : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

public:
  ConvPartial1xNSLIC();
  using FPType = quarter;
  using AccumType = half;

  // Depending on whether strides/sizes fit, use short types for storage.
  using UnsignedType =
      std::conditional_t<useShortTypes, unsigned short, unsigned int>;
  using WorkListType =
      std::conditional_t<useShortTypes, unsigned short, unsigned int>;

  // A pointer for inputs per conv group group.
  Vector<Input<Vector<FPType, PTR_ALIGN64, 8>>, PTR_ALIGN32> in;
  // A weights pointer per sub-kernel/conv group group.
  Vector<Input<Vector<FPType, PTR_ALIGN64, 8>>, PTR_ALIGN32> weights;
  // A pointer for outputs per conv group group. Enforced to be 16-byte
  // aligned to allow use of ld2xst64pace instruction along with
  // `outFieldBuffer`.
  Vector<Output<Vector<AccumType, PTR_ALIGN64, 16, true>>, PTR_ALIGN32> out;
  // A pointer to a buffer with size of outputs per conv group group
  // + 8 bytes. Enforced to be 16-byte aligned and offset by 8 bytes to
  // allow use of ld2xst64pace instruction along with each entry in `out`.
  Output<Vector<AccumType, PTR_ALIGN64, 16, true>> outFieldBuffer;

  // Array with shape:
  // [numSubKernels * numWorkerContexts][numFieldRows * 3]
  Input<VectorList<WorkListType, DELTAN>> worklists;

  // For a generic C++ vertex this will indicate which mode the vertex operates
  // in, specifying conv groups, input and output channels.
  // However this specialisation is for a specific vertex and these parameters
  // are implied by the assembler instructions used. So this parameter is
  // redundant.
  const unsigned char chansPerGroupLog2;
  // This essentially encodes whether numSubKernels is odd or not in a
  // way that avoids manipulating state on load in the assembly.
  const unsigned char outPtrLoadOffset;

  const UnsignedType numSubKernelsM1;
  const UnsignedType numConvGroupGroupsM1;

  __attribute__((target("supervisor"))) bool compute() {
    WorkerState1xN workerState;
    auto wlStatePtr = reinterpret_cast<unsigned *>(&worklists);
    workerState.partitionBase =
        reinterpret_cast<unsigned *>(*wlStatePtr & DELTAN_OFFSET_MASK);
    const auto weightsMetaData = *weights.getMetadata();
    const auto inMetaData = *in.getMetadata();
    setFp8Format(weightsMetaData, inMetaData);
    setFp8Scale(weightsMetaData, inMetaData);

    constexpr unsigned outFieldBufferOffset = 200u / sizeof(AccumType);
    const unsigned numSubKernels = numSubKernelsM1 + 1;
    const unsigned numConvGroupGroups = numConvGroupGroupsM1 + 1;

    for (unsigned cg = 0; cg < numConvGroupGroups; ++cg) {
      // Partials input read / output write alternate between a buffer and
      // the true output.  Pick the starting state, which will result in the
      // final output being written to the true output
      auto *lastOutBuffer = (!outPtrLoadOffset)
                                ? &outFieldBuffer[outFieldBufferOffset]
                                : &out[cg][0];
      auto *currOutBuffer = (!outPtrLoadOffset)
                                ? &out[cg][0]
                                : &outFieldBuffer[outFieldBufferOffset];
      for (unsigned kg = 0; kg < numSubKernels; ++kg) {
        workerState.noImplicitZero = kg;
        workerState.outChanPtr = currOutBuffer;
        workerState.partialsChanPtr = lastOutBuffer;
        const auto &w = weights[cg * numSubKernels + kg];

        workerState.partitionList =
            reinterpret_cast<unsigned *>(*(wlStatePtr + 1) &
                                         DELTAN_OFFSET_MASK) +
            kg * CTXT_WORKERS;

        slicLoadWeights<false, 16>(&w[0]);
        workerState.inChanPtr = &in[cg][0];
        if constexpr (outStride == 1) {
          RUN_ALL("__runCodelet_poplin__WorkerClass1xN___unsigned_short_1_16",
                  &workerState);
        } else {
          RUN_ALL("__runCodelet_poplin__WorkerClass1xN___unsigned_short_2_16",
                  &workerState);
        }
        std::swap(lastOutBuffer, currOutBuffer);
      }
    }
    return true;
  }
};

#endif

#endif

template class ConvPartial1xNSLIC<half, float, 1, false, 4, 2, 4>;
template class ConvPartial1xNSLIC<half, float, 1, true, 4, 2, 4>;
template class ConvPartial1xNSLIC<half, float, 2, false, 4, 2, 4>;
template class ConvPartial1xNSLIC<half, float, 2, true, 4, 2, 4>;

template class ConvPartial1xNSLIC<half, half, 1, false, 4, 2, 4>;
template class ConvPartial1xNSLIC<half, half, 1, true, 4, 2, 4>;
template class ConvPartial1xNSLIC<half, half, 2, false, 4, 2, 4>;
template class ConvPartial1xNSLIC<half, half, 2, true, 4, 2, 4>;

template class ConvPartial1xNSLIC<half, half, 1, false, 4, 4, 4>;
template class ConvPartial1xNSLIC<half, half, 1, true, 4, 4, 4>;
template class ConvPartial1xNSLIC<half, half, 2, false, 4, 4, 4>;
template class ConvPartial1xNSLIC<half, half, 2, true, 4, 4, 4>;

template class ConvPartial1xNSLIC<half, half, 1, false, 4, 4, 8>;
template class ConvPartial1xNSLIC<half, half, 1, true, 4, 4, 8>;
template class ConvPartial1xNSLIC<half, half, 2, false, 4, 4, 8>;
template class ConvPartial1xNSLIC<half, half, 2, true, 4, 4, 8>;

template class ConvPartial1xNSLIC<half, half, 1, false, 4, 4, 16>;
template class ConvPartial1xNSLIC<half, half, 1, true, 4, 4, 16>;
template class ConvPartial1xNSLIC<half, half, 2, false, 4, 4, 16>;
template class ConvPartial1xNSLIC<half, half, 2, true, 4, 4, 16>;

#if __IPU_ARCH_VERSION__ >= 21
template class ConvPartial1xNSLIC<quarter, half, 1, false, 4, 4, 8>;
template class ConvPartial1xNSLIC<quarter, half, 1, true, 4, 4, 8>;

template class ConvPartial1xNSLIC<quarter, half, 2, false, 4, 4, 8>;
template class ConvPartial1xNSLIC<quarter, half, 2, true, 4, 4, 8>;
#endif

} // end namespace poplin
