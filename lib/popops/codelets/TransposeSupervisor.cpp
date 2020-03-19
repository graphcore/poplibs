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

namespace popops {

template <typename T>
class WORKER_ALIGN
    [[poplar::constraint("elem(*src) != elem(*dst)")]] TransposeSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  TransposeSupervisor();

  Input<Vector<T, COMPACT_PTR, 8>> src;
  Output<Vector<T, COMPACT_PTR, 8>> dst;
  const unsigned short numSrcRowsD4;
  const unsigned short numSrcColumnsD4;
  // There will be 'workerCount' workers (1 <= workerCount <= 6) transposing
  // 'numTranspositions' matrices ('numTranspositions' always >0) plus
  // (6-workerCount) workers transposing (numTranspositions-1) matrices.
  // Note that (6-workerCount) and/or (numTranspositions-1) could be zero.
  const unsigned short numTranspositions;
  const unsigned short workerCount;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned totalTranspositions =
        workerCount * numTranspositions +
        (CTXT_WORKERS - workerCount) * (numTranspositions - 1);

    const unsigned numSrcColumns = numSrcColumnsD4 * 4;
    const unsigned numSrcRows = numSrcRowsD4 * 4;
    for (unsigned t = 0; t != totalTranspositions; ++t) {
      for (unsigned x = 0; x != numSrcColumns; ++x) {
        for (unsigned y = 0; y != numSrcRows; ++y) {
          dst[t * numSrcRows * numSrcColumns + x * numSrcRows + y] =
              src[t * numSrcRows * numSrcColumns + y * numSrcColumns + x];
        }
      }
    }
    return true;
  }
};

template class TransposeSupervisor<half>;

} // end namespace popops
