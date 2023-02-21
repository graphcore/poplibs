// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

namespace poplin {

template <typename FPType, typename AccumType, bool useLimitedVer>
constexpr bool hasAssembly() {
  return !(std::is_same<AccumType, half>() && std::is_same<FPType, float>());
}

/* Perform a single 1x1 convolutions using the HMAC instruction where the
 * axis of accumulation is across the vector
 * useLimitedVer is "true" if there are constraints imposed on
 * - The number of input channels is a multiple of 2
 * - worklists items are bounded to fit 16-bits
 * - number of input channels per group divided by 2 or 4 depending on whether
 *   those channels are multiple of 2 and 4 respectively
 *    <= maximum count supported by rpt instruction
 */
template <class FPType, class AccumType, bool useLimitedVer>
class [[poplar::constraint(
    "elem(**in) != elem(**weights)")]] ConvPartialHorizontalMac1x1
    : public SupervisorVertexIf<
          hasAssembly<FPType, AccumType, useLimitedVer>() &&
          ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;

public:
  ConvPartialHorizontalMac1x1();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using UnsignedType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 8>>, ONE_PTR> out;
  const unsigned zerosInfo;
  Input<Vector<WorkListType, ONE_PTR, 4>> worklists;
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
  const UnsignedType numConvGroupsM1;
  const UnsignedType outChansPerGroup;
  const UnsignedType inChansPerGroup;

  IS_EXTERNAL_CODELET((hasAssembly<FPType, AccumType, useLimitedVer>()));

  void compute() {
    const unsigned numWorkers = CTXT_WORKERS;
    const auto usedContexts = numWorkers;
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

          for (unsigned context = 0; context < usedContexts; ++context) {
            auto outOffset = worklists[context * 3 + 0];
            auto numConv = worklists[context * 3 + 1];
            auto inOffset = worklists[context * 3 + 2];
            for (unsigned i = 0; i != numConv; ++i) {
              for (unsigned oc = 0; oc != outChansPerGroup; ++oc) {
                const auto outIndex =
                    (outOffset + i * outStride) * outChansPerGroup + oc;
                AccumType sum =
                    zerosInfo ? out[cg * numOutGroups + og][outIndex] : 0;
                for (unsigned ic = 0; ic != inChansPerGroup; ++ic) {
                  const auto inIndex =
                      (inOffset + i * inStride) * inChansPerGroup + ic;
                  const auto weightIndex = oc * inChansPerGroup + ic;
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
};

template class ConvPartialHorizontalMac1x1<half, float, true>;

} // end namespace poplin
