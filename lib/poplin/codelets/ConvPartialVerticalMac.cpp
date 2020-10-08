// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

template <typename FPType, typename AccumType, bool useLimitedVer>
constexpr bool hasAssembly() {
  return std::is_same<AccumType, float>() && std::is_same<FPType, half>() &&
         useLimitedVer;
}

/* Perform a series of 1x1 convolutions in parallel using the MUL & ACC
 * instructions.
 */
template <class FPType, class AccumType, bool useLimitedVer>
class [[poplar::constraint(
    "elem(**in) != elem(**weights)")]] ConvPartialVerticalMac
    : public SupervisorVertexIf<
          hasAssembly<FPType, AccumType, useLimitedVer>() &&
          ASM_CODELETS_ENABLED> {
public:
  ConvPartialVerticalMac();

  using WorkListType =
      typename std::conditional<useLimitedVer, unsigned short, unsigned>::type;
  using SignedType =
      typename std::conditional<useLimitedVer, short, signed int>::type;
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> in;
  Vector<Input<Vector<FPType, COMPACT_PTR, 8>>, ONE_PTR> weights;
  Vector<Output<Vector<AccumType, COMPACT_PTR, 8>>, ONE_PTR> out;
  Output<Vector<AccumType, COMPACT_PTR, 8>> partials;
  const unsigned zerosInfo;
  const unsigned numInGroups;
  const unsigned numConvGroupsM1;
  Input<VectorList<WorkListType, COMPACT_DELTAN>> worklists;
  const SignedType inStride;
  const SignedType weightsStride;

  IS_EXTERNAL_CODELET((hasAssembly<FPType, AccumType, useLimitedVer>()));

  bool compute() {
    const unsigned convGroupsPerGroup = 4;
    const unsigned numWorkers = NUM_WORKERS;
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
          while (wi < wl.size()) {
            auto outOffset = wl[wi];
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
template class ConvPartialVerticalMac<float, float, true>;
template class ConvPartialVerticalMac<float, float, false>;
template class ConvPartialVerticalMac<half, float, true>;
template class ConvPartialVerticalMac<half, float, false>;
template class ConvPartialVerticalMac<half, half, true>;
template class ConvPartialVerticalMac<half, half, false>;

} // end namespace poplin
