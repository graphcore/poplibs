// Copyright (c) Graphcore Ltd, All rights reserved.
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

namespace poplin {

template <typename OutType, typename PartialsType>
class ReduceAdd : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  ReduceAdd();

  Vector<Input<Vector<PartialsType, ONE_PTR, 8, false>>, COMPACT_PTR, 4>
      partials;
  Output<Vector<OutType, COMPACT_PTR, 8>> out;
  const unsigned short numPartials;
  const unsigned short numElems;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i < numElems; ++i) {
      float sum = 0;
      for (unsigned j = 0; j < numPartials; ++j) {
        sum += float(partials[j][i]);
      }
      out[i] = sum;
    }
    return true;
  }
};

template class ReduceAdd<float, float>;
template class ReduceAdd<half, float>;
template class ReduceAdd<float, half>;
template class ReduceAdd<half, half>;

} // end namespace poplin
