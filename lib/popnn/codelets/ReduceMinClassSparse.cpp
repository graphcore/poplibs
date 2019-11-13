#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popnn {

template <typename InType> constexpr bool isIntegral() {
  return std::is_integral<InType>::value;
}

template <typename InOutType, typename LabelType>
class ReduceMinClassSparse : Vertex {
  constexpr static bool ext = !std::is_integral<InOutType>::value;

public:
  ReduceMinClassSparse();

  Input<Vector<InOutType>> activations;
  Input<Vector<LabelType, ONE_PTR>> labels;
  Output<InOutType> minValue;
  Output<LabelType> minIndex;

  IS_EXTERNAL_CODELET(ext);
  bool compute() {
    LabelType minI = 0;
    InOutType minV = activations[0];
    for (std::size_t i = 1; i < activations.size(); ++i) {
      if (activations[i] < minV) {
        minV = activations[i];
        minI = i;
      }
    }
    *minValue = minV;
    *minIndex = labels[minI];
    return true;
  }
};

template class ReduceMinClassSparse<float, unsigned int>;
template class ReduceMinClassSparse<float, int>;

template class ReduceMinClassSparse<unsigned int, unsigned int>;
template class ReduceMinClassSparse<unsigned int, int>;

template class ReduceMinClassSparse<int, unsigned int>;
template class ReduceMinClassSparse<int, int>;

} // namespace popnn
