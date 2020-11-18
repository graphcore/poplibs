// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include "popnn/PoolingDef.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
#if defined(VECTOR_AVAIL_SCALED_PTR32) &&                                      \
    defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::SCALED_PTR64;
static constexpr auto DELTANLAYOUT = poplar::VectorListLayout::DELTAN;
#else
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::ONE_PTR;
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTANLAYOUT = poplar::VectorListLayout::DELTANELEMENTS;
#endif

namespace popnn {

template <typename FPType>
class WORKER_ALIGN SumPooling
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
  IS_EXTERNAL_CODELET(true);

  Output<Vector<FPType, PTR_ALIGN64, 8>> out;
  Input<Vector<FPType, PTR_ALIGN64, 8>> in;
  // starting position within vector list for each context. The number
  // to process can be found from the difference from previous
  Input<Vector<unsigned short, PTR_ALIGN32, 4>> startPos;
  // Base offset for each entry in list
  //  - Kept as a pair with even entry for output and odd entry for input
  Input<Vector<unsigned short, PTR_ALIGN32, 4>> offsetBase;
  Input<VectorList<unsigned short, DELTANLAYOUT>> workList;

public:
  SumPooling();

  const unsigned short initInfo;
  const unsigned short numChanGroupsM1;
  // the following are scaled by the amount of FPType we can fit into 64-bits.
  const unsigned short chansPerGroupDM1;
  const unsigned inStrideD;
  const unsigned outStrideD;
  const unsigned inSliceSize;
  const unsigned outSliceSize;
  const FPType scale;

  bool compute() {
    const auto scaleFactor = std::is_same<FPType, half>::value ? 4 : 2;
    const auto numChanGroups = numChanGroupsM1 + 1;
    const auto chansPerGroup = (chansPerGroupDM1 + 1) * scaleFactor;
    const auto inStride = inStrideD * scaleFactor;
    const auto outStride = outStrideD * scaleFactor;

    // initialise output
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        out[cg * (initInfo * chansPerGroup) + i] = 0;
      }
    }

    // do pooling operation
    for (unsigned ctxtM1 = 0; ctxtM1 != NUM_WORKERS; ++ctxtM1) {
      const unsigned numRows =
          ctxtM1 == 0 ? startPos[0] : startPos[ctxtM1] - startPos[ctxtM1 - 1];
      const unsigned sPos = ctxtM1 == 0 ? 0 : startPos[ctxtM1 - 1];

      for (unsigned row = 0; row != numRows; ++row) {
        const auto pos = sPos + row;
        const auto outOffsetBase = offsetBase[2 * pos] * chansPerGroup;
        const auto inOffsetBase = offsetBase[2 * pos + 1] * chansPerGroup;
        const auto numWorkItems = workList[pos].size();
        // there are always three items for each work vector
        for (unsigned w = 0; w != numWorkItems; w += 3) {
          const auto outBeginOffset =
              workList[pos][w + 0] * chansPerGroup + outOffsetBase;
          const auto inBeginOffset =
              workList[pos][w + 1] * chansPerGroup + inOffsetBase;
          const auto numElements = workList[pos][w + 2] + 1;
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            const auto inBase = cg * inSliceSize;
            const auto outBase = cg * outSliceSize;
            for (unsigned c = 0; c != chansPerGroup; ++c) {
              unsigned outPos = outBeginOffset + c + outBase;
              unsigned inPos = inBeginOffset + c + inBase;
              for (unsigned f = 0; f != numElements; ++f) {
                out[outPos] += scale * in[inPos];

                outPos += outStride;
                inPos += inStride;
              }
            }
          }
        }
      }
    }
    return true;
  }
};

template class SumPooling<float>;
template class SumPooling<half>;

} // namespace popnn
