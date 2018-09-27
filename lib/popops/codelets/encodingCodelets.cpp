#include <poplar/Vertex.hpp>
#include <poplar/VectorTypes.hpp>
#include <poplar/HalfFloat.hpp>
#include "poplibs_support/ExternalCodelet.hpp"

#include <string.h>

using namespace poplar;

namespace popops {

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template <typename IndexType, typename OutType>
class EncodeOneHot : public SupervisorVertex {
  constexpr static bool isExternal() {
    return std::is_same<IndexType, unsigned>{}
      && std::is_same<OutType, half>{};
  }
public:
  IS_EXTERNAL_CODELET(isExternal());

  Input<Vector<IndexType>> indices;
  Output<Vector<OutType, ONE_PTR, 8>> out;
  // the output tensor has been flattened, so this field states how many
  // elements to be processed for each index.
  Input<Vector<unsigned, ONE_PTR>> sliceLength;
  Input<Vector<unsigned, ONE_PTR>> offsets;
  // This field could be removed as it is sum of the total slice Lengths
  unsigned outLength;

  bool compute() {
    for (unsigned i = 0; i < outLength; ++i) {
      out[i] = 0;
    }
    unsigned begin = 0;
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (indices[i] >= offsets[i] &&
          (indices[i] < offsets[i] + sliceLength[i])) {
        const auto index = begin + indices[i] - offsets[i];
        assert(index < outLength);
        out[index] = 1;
      }
      begin += sliceLength[i];
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
