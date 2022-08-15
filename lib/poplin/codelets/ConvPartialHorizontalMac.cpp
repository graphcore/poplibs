// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;
static constexpr auto COMPACT_DELTAN = poplar::VectorListLayout::COMPACT_DELTAN;

namespace poplin {

template <typename FPType, typename AccumType, bool useLimitedVer>
constexpr bool hasAssembly() {
  return !(std::is_same<AccumType, half>() && std::is_same<FPType, float>());
}

/* Perform a series of 1x1 convolutions using the HMAC instruction where the
 * axis of accumulation is across the vector.
 * useLimitedVer is "true" if there are constraints imposed on
 * - The number of input channels is a multiple of 2
 * - worklists items are bounded to fit 16-bits
 * - number of input channels per group divided by 2 or 4 depending on whether
 *   those channels are multiple of 2 and 4 respectively
 *    <= maximum count supported by rpt instruction
 */
template <class FPType, class AccumType, bool useLimitedVer>
class [[poplar::constraint(
    "elem(**in) != elem(**weights)")]] ConvPartialHorizontalMac
    : public SupervisorVertexIf<
          hasAssembly<FPType, AccumType, useLimitedVer>() &&
          ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;

public:
  ConvPartialHorizontalMac();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 8>>, ONE_PTR> out;
  const unsigned zerosInfo;
  Input<VectorList<WorkListType, COMPACT_DELTAN>> worklists;
  const UnsignedType numOutGroupsM1;

  // transformedInStride =  ("actual input stride" - 1) * inChansPerGroup
  const unsigned transformedInStride;
  // transformedOutStride =
  //   = (-1 * "actual output stride" - 1 * outChansPerGroup (if flip output)
  //   = +1 * "actual output stride" * outChansPerGroup
  // Due to a fact that HMAC codelet for half partials process 2 partials
  // in one loop iterration transformedOutStride was adjusted accordingly
  // e.g. if (AccumType == HALF) transformedOutStride /= 2
  const int transformedOutStride;
  const UnsignedType numInGroups;
  const UnsignedType kernelSizeM1;
  const UnsignedType numConvGroupsM1;
  const UnsignedType outChansPerGroup;
  const UnsignedType inChansPerGroup;

  IS_EXTERNAL_CODELET((hasAssembly<FPType, AccumType, useLimitedVer>()));

  void compute() {
    const unsigned numWorkers = CTXT_WORKERS;
    const unsigned kernelSize = kernelSizeM1 + 1;
    const auto usedContexts = worklists.size() / kernelSize;
    const unsigned numOutGroups = numOutGroupsM1 + 1;
    const unsigned numConvGroups = numConvGroupsM1 + 1;
    const auto outStride =
        (std::is_same<AccumType, float>() ? transformedOutStride
                                          : 2 * transformedOutStride) /
            static_cast<int>(outChansPerGroup) +
        1;
    const auto inStride = transformedInStride / inChansPerGroup;
    const unsigned numElems = zerosInfo;

    for (unsigned cg = 0; cg != numConvGroups; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        for (unsigned i = 0; i != numElems; ++i)
          out[cg * numOutGroups + og][i] = 0;
      }
    }

    for (unsigned cg = 0; cg != numConvGroups; ++cg) {
      for (unsigned og = 0; og != numOutGroups; ++og) {
        for (unsigned ig = 0; ig != numInGroups; ++ig) {
          const auto &w = weights[cg * numOutGroups * numInGroups +
                                  ig * numOutGroups + (numOutGroups - 1 - og)];

          for (unsigned k = 0; k != kernelSize; ++k) {
            for (unsigned context = 0; context < usedContexts; ++context) {
              const auto &wl = worklists[k * usedContexts + context];
              unsigned wi = 0;
              while (wi < wl.size()) {
                auto outOffset = wl[wi];
                auto numConv = wl[wi + 1];
                auto inOffset = wl[wi + 2];
                wi += 3;
                for (unsigned i = 0; i != numConv; ++i) {
                  for (unsigned oc = 0; oc != outChansPerGroup; ++oc) {
                    const auto outIndex =
                        (outOffset + i * outStride) * outChansPerGroup + oc;
                    AccumType sum = out[cg * numOutGroups + og][outIndex];
                    for (unsigned ic = 0; ic != inChansPerGroup; ++ic) {
                      const auto inIndex =
                          (inOffset + i * inStride) * inChansPerGroup + ic;
                      const auto weightIndex =
                          k * outChansPerGroup * inChansPerGroup +
                          oc * inChansPerGroup + ic;
                      sum += AccumType(in[cg * numInGroups + ig][inIndex] *
                                       w[weightIndex]);
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
};
template class ConvPartialHorizontalMac<float, float, true>;
template class ConvPartialHorizontalMac<float, float, false>;
template class ConvPartialHorizontalMac<half, float, true>;
template class ConvPartialHorizontalMac<half, float, false>;
template class ConvPartialHorizontalMac<half, half, true>;
template class ConvPartialHorizontalMac<half, half, false>;

} // end namespace poplin
