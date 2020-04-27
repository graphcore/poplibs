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

static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

namespace poplin {

template <typename FPType, typename AccumType, bool useLimitedVer,
          unsigned numConvUnits>
constexpr bool hasAssembly() {
  return !(std::is_same<AccumType, half>() && std::is_same<FPType, float>()) &&
         useLimitedVer == true && numConvUnits == 8;
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
    const auto usedContexts = NUM_WORKERS;
    // modify to set actual values used by vertex
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const int inStride =
        (transformedInStride - 1) * convInputLoadElems / inChansPerGroup + 1;
    bool flipOut =
        transformedOutStride < (std::is_same<AccumType, float>() ? -6 : -4);

    for (unsigned cg = 0; cg < numConvGroups; ++cg) {
      for (unsigned og = 0; og < numOutGroups; ++og) {
        for (unsigned ig = 0; ig < numInGroups; ++ig) {
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];
          for (unsigned context = 0; context < usedContexts; ++context) {
            const auto accumTypeSize = std::is_same<AccumType, float>() ? 4 : 2;
            const auto typeSize = std::is_same<FPType, float>() ? 4 : 2;
            const auto outOffset = (worklists[3 * context] * 8) /
                                   (outChansPerGroup * accumTypeSize);
            // The numFieldElems values from worklist is less by 3
            const int numFieldElems =
                static_cast<WorkListNumFieldType>(worklists[3 * context + 1]) +
                3;
            const auto inOffset =
                (worklists[3 * context + 2] * 8) / (inChansPerGroup * typeSize);

            for (unsigned i = 0; i < numFieldElems; ++i) {
              for (unsigned outChan = 0; outChan < outChansPerGroup;
                   ++outChan) {
                const auto outIndex =
                    (outOffset + (flipOut ? -i : i)) * outChansPerGroup +
                    outChan;
                if (ig == 0)
                  out[cg * numOutGroups + og][outIndex] = 0;
                float sum = 0;
                for (unsigned inChan = 0; inChan < inChansPerGroup; ++inChan) {
                  const auto inIndex =
                      (inOffset + i * inStride) * inChansPerGroup + inChan;
                  const auto weightIndex = outChan * inChansPerGroup + inChan;
                  sum += float(in[cg * numInGroups + ig][inIndex]) *
                         float(w[weightIndex]);
                }
                out[cg * numOutGroups + og][outIndex] += sum;
              }
            }
          }
        }
      }
    }
    return true;
  }
};

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

template class ConvPartial1x1Out<half, half, true, false, 4>;
template class ConvPartial1x1Out<half, float, true, false, 4>;
template class ConvPartial1x1Out<half, half, false, false, 4>;
template class ConvPartial1x1Out<half, float, false, false, 4>;

template class ConvPartial1x1Out<half, half, true, true, 4>;
template class ConvPartial1x1Out<half, float, true, true, 4>;
template class ConvPartial1x1Out<half, half, false, true, 4>;
template class ConvPartial1x1Out<half, float, false, true, 4>;

template class ConvPartial1x1Out<half, half, true, false, 16>;
template class ConvPartial1x1Out<float, float, true, false, 16>;
template class ConvPartial1x1Out<half, half, false, false, 16>;
template class ConvPartial1x1Out<float, float, false, false, 16>;

template class ConvPartial1x1Out<half, half, true, true, 16>;
template class ConvPartial1x1Out<float, float, true, true, 16>;
template class ConvPartial1x1Out<half, half, false, true, 16>;
template class ConvPartial1x1Out<float, float, false, true, 16>;

} // end namespace poplin
