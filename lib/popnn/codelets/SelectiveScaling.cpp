// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/VectorTypes.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
static constexpr auto CompactPTR = poplar::VectorLayout::COMPACT_PTR;
static constexpr auto CompactDELTAN = poplar::VectorListLayout::COMPACT_DELTAN;

namespace popnn {

template <typename FPType> class SelectiveScaling : public Vertex {
public:
  SelectiveScaling();

  IS_EXTERNAL_CODELET(false);
  Input<VectorList<unsigned short, CompactDELTAN>> scaleWorklist;
  Vector<InOut<Vector<FPType, CompactPTR, 8>>, CompactPTR> inOut;
  const unsigned short numChanGroups;
  const unsigned short chansPerGroup;

  bool compute() {
    // Scale output
    for (unsigned ctxt = 0; ctxt != NUM_WORKERS; ++ctxt) {
      for (unsigned w = 0; w != scaleWorklist[ctxt].size(); w += 3) {
        for (unsigned f = 0; f != scaleWorklist[ctxt][w + 1]; ++f) {
          for (unsigned cg = 0; cg != numChanGroups; ++cg) {
            for (unsigned c = 0; c != chansPerGroup; ++c) {
              unsigned outPos =
                  (f + scaleWorklist[ctxt][w]) * chansPerGroup + c;
              FPType scale = static_cast<FPType>(scaleWorklist[ctxt][w + 2]);
              inOut[cg][outPos] /= scale;
            }
          }
        }
      }
    }
    return true;
  }
};

template class SelectiveScaling<float>;
template class SelectiveScaling<half>;

} // namespace popnn
