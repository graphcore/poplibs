// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include "popops/EncodingConstants.hpp"
#include <cassert>
#include <poplar/HalfFloat.hpp>
#include <poplar/VectorTypes.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

namespace popops {

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template <typename IndexType, typename OutType> constexpr bool hasAssembly() {
  return std::is_same<IndexType, unsigned>{} && std::is_same<OutType, half>{};
}

template <typename IndexType, typename OutType>
class EncodeOneHot
    : public SupervisorVertexIf<hasAssembly<IndexType, OutType>() &&
                                ASM_CODELETS_ENABLED> {
public:
  EncodeOneHot();

  IS_EXTERNAL_CODELET((hasAssembly<IndexType, OutType>()));

  Input<Vector<IndexType>> indices;

  // InOut because we don't want to touch the entire output (popops::zero in cs
  // before zero'd memory for us)
  InOut<Vector<OutType, ONE_PTR, 8>> out;

  // slice length handled by this vertex
  unsigned sliceLength;

  // offset is the start index such that the range of indices
  // handled by this vertex is [offset, offset + sliceLength)
  unsigned offset;

  bool compute() {
    unsigned begin = 0;
    for (unsigned i = 0; i < indices.size(); ++i) {
      if ((indices[i] >= offset) && (indices[i] < offset + sliceLength)) {
        const auto index = begin + indices[i] - offset;
        if (index != MASKED_LABEL_CODE) {
          out[index] = 1;
        }
      }
      begin += sliceLength;
    }
    return true;
  }
};

template class EncodeOneHot<unsigned, float>;
template class EncodeOneHot<unsigned, half>;
template class EncodeOneHot<unsigned, unsigned>;
template class EncodeOneHot<unsigned, int>;
template class EncodeOneHot<int, float>;
template class EncodeOneHot<int, half>;
template class EncodeOneHot<int, unsigned>;
template class EncodeOneHot<int, int>;

} // namespace popops
