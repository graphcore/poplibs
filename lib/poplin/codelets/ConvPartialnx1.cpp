// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;
static constexpr auto COMPACT_DELTAN = poplar::VectorListLayout::COMPACT_DELTAN;

namespace poplin {

template <typename FPType, typename AccumType, bool useLimitedVer,
          unsigned numConvUnits>
constexpr bool hasAssembly() {
  return !(std::is_same<AccumType, half>() && std::is_same<FPType, float>()) &&
         useLimitedVer == true && numConvUnits == 8;
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

  // This value is
  // (inStrideX - 1 - (ampKernelHeight - 1) * inRowStride)
  //      * inChansPerGroup / convInputLoadElems + 1)
  // Where inStrideX is the actual stride
  const SignedType transformedInStride;
  // This output stride also encodes the flip parameter and is given as
  // -6 + outChansPerGroup * (actual output stride) if flipOut = false
  // -6 - outChansPerGroup * (actual output stride) if flipOut = true
  const SignedType transformedOutStride;
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
    const unsigned numWorkers = NUM_WORKERS;
    const unsigned convInputLoadElems = std::is_same<FPType, float>::value
                                            ? CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT
                                            : CONV_UNIT_INPUT_LOAD_ELEMS_HALF;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned ampKernelHeight = ampKernelHeightM1 + 1;
    const unsigned kernelOuterSize = kernelOuterSizeM1 + 1;
    const unsigned kernelInnerElements = kernelInnerElementsM1 + 1;

    int inRowStride =
        (transformedInRowStride - 1) * convInputLoadElems / inChansPerGroup + 1;

    const int inStride =
        (transformedInStride - 1) * convInputLoadElems / inChansPerGroup + 1 +
        (ampKernelHeight - 1) * inRowStride;

    const auto usedContexts =
        worklists.size() / (kernelOuterSize * kernelInnerElements);
    const auto outStrideThresh = std::is_same<AccumType, float>() ? -6 : -4;

    const auto flipOut = transformedOutStride < outStrideThresh;
    const int outStride =
        flipOut ? (-transformedOutStride + outStrideThresh) / outChansPerGroup
                : (transformedOutStride - outStrideThresh) / outChansPerGroup;

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
                  const auto typeSize = std::is_same<FPType, float>() ? 4 : 2;
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

                          sum += AccumType(in[cg * numInGroups + ig][inIndex] *
                                           w[weightIndex]);
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

template class ConvPartialnx1<half, half, true, false, 4>;
template class ConvPartialnx1<half, float, true, false, 4>;
template class ConvPartialnx1<half, half, false, false, 4>;
template class ConvPartialnx1<half, float, false, false, 4>;

template class ConvPartialnx1<half, half, true, true, 4>;
template class ConvPartialnx1<half, float, true, true, 4>;
template class ConvPartialnx1<half, half, false, true, 4>;
template class ConvPartialnx1<half, float, false, true, 4>;

template class ConvPartialnx1<float, float, true, false, 16>;
template class ConvPartialnx1<half, half, true, false, 16>;
template class ConvPartialnx1<float, float, false, false, 16>;
template class ConvPartialnx1<half, half, false, false, 16>;

template class ConvPartialnx1<float, float, true, true, 16>;
template class ConvPartialnx1<half, half, true, true, 16>;
template class ConvPartialnx1<float, float, false, true, 16>;
template class ConvPartialnx1<half, half, false, true, 16>;

} // end namespace poplin
