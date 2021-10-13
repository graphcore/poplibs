// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

template <typename FPType, typename AccumType, bool useLimitedVer,
          unsigned convGroupsPerGroup>
constexpr bool hasAssembly() {
  constexpr bool floatActivations = std::is_same<FPType, float>();
  constexpr bool floatPartials = std::is_same<AccumType, float>();
  constexpr bool halfHalfVersion =
      !floatPartials && (convGroupsPerGroup == 4 || convGroupsPerGroup == 8 ||
                         convGroupsPerGroup == 16);

  constexpr bool halfFloatVersion = floatPartials && (convGroupsPerGroup == 4);
  return !floatActivations && useLimitedVer &&
         (halfHalfVersion || halfFloatVersion);
}

/* Perform a series of 1x1 convolutions in parallel using the MUL & ACC
 * instructions.
 */
template <class FPType, class AccumType, bool useLimitedVer,
          unsigned convGroupsPerGroup>
class [[poplar::constraint(
    "elem(**in) != elem(**weights)")]] ConvPartialVerticalMac
    : public SupervisorVertexIf<
          hasAssembly<FPType, AccumType, useLimitedVer, convGroupsPerGroup>() &&
          ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;
  // ConvGroupsPerGroup 8 and 16 codelets use load128 that requries 16 bytes
  // alignment
  static constexpr bool use128bitLoad = (convGroupsPerGroup > 4) ? true : false;
  static constexpr unsigned dataAlignment = use128bitLoad ? 16 : 8;

public:
  ConvPartialVerticalMac();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType =
      typename std::conditional<useLimitedVer, short, signed int>::type;
  Vector<Input<Vector<FPType, COMPACT_PTR, dataAlignment, use128bitLoad>>,
         ONE_PTR>
      in;
  Vector<Input<Vector<FPType, COMPACT_PTR, dataAlignment, use128bitLoad>>,
         ONE_PTR>
      weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 8>>, ONE_PTR> out;
  Output<Vector<AccumType, COMPACT_PTR, dataAlignment, use128bitLoad>> partials;
  const unsigned zerosInfo;
  const unsigned numInGroups;
  const unsigned numConvGroupsM1;
  Input<VectorList<WorkListType, COMPACT_DELTAN>> worklists;
  const SignedType inStride;
  const SignedType weightsStride;

  IS_EXTERNAL_CODELET(
      (hasAssembly<FPType, AccumType, useLimitedVer, convGroupsPerGroup>()));

  bool compute() {
    const unsigned numWorkers = CTXT_WORKERS;
    const auto usedContexts = worklists.size();
    const unsigned numConvGroups = numConvGroupsM1 + 1;

    const unsigned numElems = zerosInfo;
    for (unsigned cgg = 0; cgg != numConvGroups; ++cgg) {
      // zero out partials from each worker
      for (unsigned context = 0; context < usedContexts; ++context) {
        for (unsigned i = 0; i != numElems; ++i) {
          partials[context * numElems + i] = 0;
        }
      }

      // perform vmac operation
      for (unsigned ig = 0; ig != numInGroups; ++ig) {
        const auto &w = weights[cgg * numInGroups + ig];
        for (unsigned context = 0; context < usedContexts; ++context) {
          const auto &wl = worklists[context];
          unsigned wi = 0;
          unsigned prevOut = 0;
          while (wi < wl.size()) {
            // output offset is stored as deltas following the first value
            auto outOffset = wl[wi] + prevOut;
            prevOut = outOffset;

            auto kOffset = wl[wi + 1];
            auto inOffset = wl[wi + 2];
            auto numConv = wl[wi + 3] + 1;
            wi += 4;
            for (unsigned cg = 0; cg != convGroupsPerGroup; ++cg) {
              const auto outIndex = outOffset * convGroupsPerGroup + cg;
              AccumType &sum = partials[context * numElems + outIndex];
              for (unsigned i = 0; i != numConv; ++i) {
                const auto inIndex =
                    (inOffset + (inStride * i)) * convGroupsPerGroup + cg;
                const auto weightIndex =
                    (kOffset + (weightsStride * i)) * convGroupsPerGroup + cg;
                sum += AccumType(in[cgg * numInGroups + ig][inIndex] *
                                 w[weightIndex]);
              }
            }
          }
        }
      }

      // Reduce partials from all the worker contexts
      for (unsigned i = 0; i != numElems; ++i) {
        out[cgg][i] = partials[i];
        for (unsigned context = 1; context < usedContexts; ++context) {
          out[cgg][i] += partials[context * numElems + i];
        }
      }
    }
    return true;
  }
};

template class ConvPartialVerticalMac<half, float, true, 4>;
template class ConvPartialVerticalMac<half, float, false, 4>;
template class ConvPartialVerticalMac<half, half, true, 4>;
template class ConvPartialVerticalMac<half, half, false, 4>;

template class ConvPartialVerticalMac<half, float, true, 8>;
template class ConvPartialVerticalMac<half, float, false, 8>;
template class ConvPartialVerticalMac<half, half, true, 8>;
template class ConvPartialVerticalMac<half, half, false, 8>;

template class ConvPartialVerticalMac<half, float, true, 16>;
template class ConvPartialVerticalMac<half, float, false, 16>;
template class ConvPartialVerticalMac<half, half, true, 16>;
template class ConvPartialVerticalMac<half, half, false, 16>;

} // end namespace poplin
