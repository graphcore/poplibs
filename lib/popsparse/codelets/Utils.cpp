// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

template <typename FPType> static constexpr inline bool hasAssemblyVersion() {
  return false;
}

namespace popsparse {

template <typename Type>
class BufferIndexUpdate

    : public SupervisorVertexIf<hasAssemblyVersion<Type>() &&
                                ASM_CODELETS_ENABLED> {
public:
  BufferIndexUpdate();

  // Pointers to buckets of sparse non-zero input values in r.
  InOut<Type> index;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<Type>()));

  bool compute() {
    *index = 1 - *index;
    return true;
  }
};

template class BufferIndexUpdate<unsigned>;

} // end namespace popsparse
