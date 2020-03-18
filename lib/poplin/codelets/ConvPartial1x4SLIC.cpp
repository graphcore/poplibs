// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

#include <type_traits>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::ONE_PTR;
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTANELEMENTS;

namespace poplin {

template <typename FPType, typename AccumType, bool useShortTypes>
constexpr bool hasAssemblyVersion() {
  return std::is_same<FPType, half>::value &&
         std::is_same<AccumType, float>::value && useShortTypes;
}

template <typename FPType, typename AccumType, bool useShortTypes>
class [[poplar::constraint(
    "elem(**in) != elem(**out)",
    "elem(**in) != elem(*outFieldBuffer)")]] ConvPartial1x4SLIC
    : public SupervisorVertexIf<
          hasAssemblyVersion<FPType, AccumType, useShortTypes>() &&
          ASM_CODELETS_ENABLED> {
  // when we add support for 1x3 this can become a template parameter.
  constexpr static unsigned windowWidth = 4u;

public:
  ConvPartial1x4SLIC();

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

  // Indicates which of 3 modes we operate in:
  //
  //  =0 -> 4 conv groups, 1 input channel, 1 output channel
  //  =1 -> 2 conv groups, 2 input channels, 2 output channels
  //  =2 -> 1 conv group, 4 input channels, 4 output channels
  //
  // This encodes conv groups per group, input channels per group
  // and output channels per group as these are tightly coupled (tripled?)
  // in this vertex.
  const unsigned char mode;
  // This essentially encodes whether numSubKernels is odd or not in a
  // way that avoids manipulating state on load in the assembly.
  const unsigned char outPtrLoadOffset;

  const UnsignedType numSubKernelsM1;
  const UnsignedType numConvGroupGroupsM1;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<FPType, AccumType, useShortTypes>()));

  bool compute() {
    const auto convInputLoadElems = std::is_same<FPType, float>::value
                                        ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT
                                        : CONV_UNIT_INPUT_LOAD_ELEMS_HALF;

    // this vertex requires that the product of the conv groups and the channels
    // per group is a product of 4.
    constexpr unsigned chansInConvGroups = 4u;

    const unsigned convGroupsPerGroup = chansInConvGroups >> mode;
    const unsigned outChansPerGroup = chansInConvGroups / convGroupsPerGroup;
    const unsigned inChansPerGroup = chansInConvGroups / convGroupsPerGroup;

    const unsigned numSubKernels = numSubKernelsM1 + 1;
    const unsigned numConvGroupGroups = numConvGroupGroupsM1 + 1;

    for (unsigned cg = 0; cg < numConvGroupGroups; ++cg) {
      auto *lastOutBuffer = (!outPtrLoadOffset)
                                ? &outFieldBuffer[8u / sizeof(AccumType)]
                                : &out[cg][0];
      auto *currOutBuffer = (!outPtrLoadOffset)
                                ? &out[cg][0]
                                : &outFieldBuffer[8u / sizeof(AccumType)];
      for (unsigned kg = 0; kg < numSubKernels; ++kg) {
        const auto &w = weights[cg * numSubKernels + kg];
        for (unsigned context = 0; context < NUM_WORKERS; ++context) {
          const auto &wl = worklists[kg * NUM_WORKERS + context];
          unsigned wi = 0;
          while (wi < wl.size()) {
            const auto inOffset = wl[wi + 0];
            const auto outOffset = wl[wi + 1];
            const auto numFieldElems = wl[wi + 2];

            for (unsigned i = 0; i < numFieldElems; ++i) {
              // From here we are replicating the core SLIC loop
              for (unsigned outChan = 0; outChan < outChansPerGroup;
                   ++outChan) {
                for (unsigned convGroup = 0; convGroup < convGroupsPerGroup;
                     ++convGroup) {
                  const auto outIndex =
                      (outOffset + i) * convGroupsPerGroup * outChansPerGroup +
                      convGroup * outChansPerGroup + outChan;
                  // Implicitly zero partials on the first pass.
                  AccumType sum = 0;
                  if (kg > 0) {
                    sum = AccumType(lastOutBuffer[outIndex]);
                  }
                  for (unsigned k = 0; k < windowWidth; ++k) {
                    for (unsigned inChan = 0; inChan < inChansPerGroup;
                         ++inChan) {
                      const auto inIndex = (inOffset + i + k) *
                                               convGroupsPerGroup *
                                               inChansPerGroup +
                                           convGroup * inChansPerGroup + inChan;
                      const auto weightIndex =
                          k * convGroupsPerGroup * outChansPerGroup *
                              inChansPerGroup +
                          convGroup * outChansPerGroup * inChansPerGroup +
                          outChan * inChansPerGroup + inChan;
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

template class ConvPartial1x4SLIC<half, float, false>;
template class ConvPartial1x4SLIC<half, float, true>;

} // end namespace poplin
