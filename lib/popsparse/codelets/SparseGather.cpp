// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

template <typename FPType> static constexpr inline bool hasAssemblyVersion() {
  return true;
}

namespace popsparse {

template <typename FPType>
class [[poplar::constraint(
    "elem(*rIn) != elem(*indices)")]] SparseGatherElementWise
    : public MultiVertex {
public:
  // only allow 16-bits. We could change to a template parameter
  using IndexType = unsigned short;
  SparseGatherElementWise();

  // Pointers to buckets of sparse non-zero input values in r.
  Input<Vector<FPType, ONE_PTR, 8>> rIn;
  Output<Vector<FPType, ONE_PTR, 8>> rOut;
  Input<Vector<IndexType, ONE_PTR, 8>> indices;
  // encoding of how work is split
  unsigned short numIndices;
  unsigned short workerOffsets;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<FPType>()));

  void compute(unsigned wid) {
    if (wid == 0) {
      const auto vectorSize = std::is_same<FPType, float>() ? 2 : 4;
      unsigned numExcess = 0;
      for (unsigned c = 0, z = workerOffsets; c != CTXT_WORKERS; ++c, z >>= 1) {
        numExcess += (z & 0x1);
      }
      const auto sizeAtom = 8 / vectorSize;
      const auto n = (numIndices / vectorSize) * (vectorSize * CTXT_WORKERS) +
                     (numIndices % vectorSize) + (vectorSize * numExcess);
      for (unsigned i = 0; i != n; ++i) {
        rOut[i] = rIn[indices[i] / sizeAtom];
      }
    }
  }
};

template class SparseGatherElementWise<half>;
template class SparseGatherElementWise<float>;

} // end namespace popsparse
