#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include "popnn/PoolingDef.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;

namespace popnn {

template <typename FPType>
class WORKER_ALIGN MaxPooling
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
  static FPType identity() {
    if (std::is_same<FPType, float>{}) {
      return -std::numeric_limits<FPType>::infinity();
    } else {
      // half type has no infinity so use the lowest finite value instead.
      return std::numeric_limits<FPType>::lowest();
    }
  }

  static FPType max(FPType lhs, FPType rhs) { return lhs > rhs ? lhs : rhs; }

public:
  MaxPooling();

  IS_EXTERNAL_CODELET(true);

  Vector<Output<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> out;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, SCALED_PTR32> in;
  // starting position within vector list for each context. The number
  // to process can be found from the difference from previous
  Input<Vector<unsigned short, SCALED_PTR32>> startPos;
  // Base offset for each entry in list
  //  - Kept as a pair with even entry for output and odd entry for input
  Input<Vector<unsigned short, SCALED_PTR32>> offsetBase;
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
  Input<VectorList<unsigned short, DELTAN>> workList;
  const unsigned short initInfo;
  const unsigned short numChanGroupsM1;
  // the following are scaled by the amount of FPType we can fit into 64-bits.
  const unsigned short chansPerGroupD;
  const unsigned inStrideD;
  const unsigned outStrideD;

  bool compute() {
    const auto scaleFactor = std::is_same<FPType, half>::value ? 4 : 2;
    const auto numChanGroups = numChanGroupsM1 + 1;
    const auto chansPerGroup = chansPerGroupD * scaleFactor;
    const auto inStride = inStrideD * scaleFactor;
    const auto outStride = outStrideD * scaleFactor;

    // initialise output
    for (unsigned cg = 0; cg != numChanGroups; ++cg) {
      for (unsigned i = 0; i != initInfo * chansPerGroup; ++i) {
        out[cg][i] = identity();
      }
    }

    // do pooling operation
    for (unsigned ctxtM1 = 0; ctxtM1 != NUM_WORKERS; ++ctxtM1) {
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
            const auto in_ = in[cg];
            auto out_ = out[cg];
            for (unsigned c = 0; c != chansPerGroup / 2; ++c) {
              unsigned outPos = (chansPerGroup * outBeginOffset) + c * 2;
              unsigned inPos = (chansPerGroup * inBeginOffset) + c * 2;
              for (unsigned f = 0; f != numElements; ++f) {
                out_[outPos] = max(out_[outPos], in_[inPos]);
                out_[outPos + 1] = max(out_[outPos + 1], in_[inPos + 1]);

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

template class MaxPooling<float>;
template class MaxPooling<half>;

} // namespace popnn
