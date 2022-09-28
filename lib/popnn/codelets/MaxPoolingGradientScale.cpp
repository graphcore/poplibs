// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "popnn/PoolingDef.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::ONE_PTR;
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTANLAYOUT = poplar::VectorListLayout::DELTANELEMENTS;

namespace popnn {

template <typename FPType>
class WORKER_ALIGN MaxPoolingGradientScale : public Vertex {
public:
  MaxPoolingGradientScale();

  Output<Vector<FPType, PTR_ALIGN64, 8>> out;
  Input<Vector<FPType, PTR_ALIGN64, 8>> fwdActsOut;
  Input<Vector<FPType, PTR_ALIGN64, 8>> in;
  // starting position within vector list for each context. The number
  // to process can be found from the difference from previous
  Input<Vector<unsigned short, PTR_ALIGN32, 4>> startPos;
  // Base offset for each entry in list
  //  - Kept as a pair with even entry for output and odd entry for input
  Input<Vector<unsigned short, PTR_ALIGN32, 4>> offsetBase;
  // Each worklist entry describes work to be done per partial row and
  // contains input, output offsets, and number of elements for each kernal
  // position if it exists.
  // The following associated information is kept
  // - Starting position of the worklist entry for a context. There are always
  //   as many entries as the number of contexts. This also gives the number to
  //   worklist entries to be processed by each context.
  // - Base for input begin offset for each worklist entry. The actual offset is
  //   this + the entry in the worklist
  // - Base for output begin offset for each worklist entry. The actual offset
  //   is this + the entry in the worklist
  // The base for the offsets contains alternating values for output and input
  // respectively (i.e. output offset bases are at even and input offset bases
  // are at odd positions
  Input<VectorList<unsigned short, DELTANLAYOUT>> workList;
  const unsigned short initInfo;
  const unsigned short numChanGroupsM1;
  // the following are scaled by the amount of FPType we can fit into 64-bits.
  const unsigned short chansPerGroupDM1;
  const unsigned inStrideD;
  const unsigned outStrideD;
  const unsigned inSliceSize;
  const unsigned outSliceSize;

  void compute() {
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
    for (unsigned ctxtM1 = 0; ctxtM1 != CTXT_WORKERS; ++ctxtM1) {
      const unsigned numRows =
          ctxtM1 == 0 ? startPos[0] : startPos[ctxtM1] - startPos[ctxtM1 - 1];
      const unsigned sPos = ctxtM1 == 0 ? 0 : startPos[ctxtM1 - 1];

      // the first 4 loops are completely independent from each other so order
      // them in such a way that the more intermediate work required by a step
      // is the outer most loop.
      for (unsigned row = 0; row != numRows; ++row) {
        const auto pos = sPos + row;
        const auto outOffsetBase = offsetBase[2 * pos];
        const auto inOffsetBase = offsetBase[2 * pos + 1];
        const auto numWorkItems = workList[pos].size();
        // there are always three items for each work vector
        for (unsigned w = 0; w != numWorkItems; w += 3) {
          const auto outBeginOffset = workList[pos][w + 0] + outOffsetBase;
          const auto inBeginOffset = workList[pos][w + 1] + inOffsetBase;
          const auto numElements = workList[pos][w + 2] + 1;
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            const auto inBase = cg * inSliceSize;
            const auto outBase = cg * outSliceSize;
            for (unsigned c = 0; c != chansPerGroup / 2; ++c) {
              unsigned outPos =
                  (chansPerGroup * outBeginOffset) + outBase + c * 2;
              unsigned inPos = (chansPerGroup * inBeginOffset) + inBase + c * 2;
              for (unsigned f = 0; f != numElements; ++f) {
                out[outPos] += fwdActsOut[outPos] == in[inPos];
                out[outPos + 1] += fwdActsOut[outPos + 1] == in[inPos + 1];
                outPos += outStride;
                inPos += inStride;
              }
            }
          }
        }
      }
    }
    // Compute scale
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        if (out[cg * (initInfo * chansPerGroup) + i])
          out[cg * (initInfo * chansPerGroup) + i] =
              1.0f /
              static_cast<float>(out[cg * (initInfo * chansPerGroup) + i]);
      }
    }
  }
};
template class MaxPoolingGradientScale<float>;
template class MaxPoolingGradientScale<half>;

} // namespace popnn
