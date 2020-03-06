// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
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
class ReduceMaxClassSparse : Vertex {
  constexpr static bool ext = !std::is_integral<InOutType>::value;

public:
  ReduceMaxClassSparse();

  Input<Vector<InOutType>> activations;
  Input<Vector<LabelType, ONE_PTR>> labels;
  Output<InOutType> maxValue;
  Output<LabelType> maxIndex;

  IS_EXTERNAL_CODELET(ext);
  bool compute() {
    LabelType maxI = 0;
    InOutType maxV = activations[0];
    for (std::size_t i = 1; i < activations.size(); ++i) {
      if (activations[i] > maxV) {
        maxV = activations[i];
        maxI = i;
      }
    }
    *maxValue = maxV;
    *maxIndex = labels[maxI];
    return true;
  }
};

template class ReduceMaxClassSparse<float, unsigned int>;
template class ReduceMaxClassSparse<float, int>;

template class ReduceMaxClassSparse<unsigned int, unsigned int>;
template class ReduceMaxClassSparse<unsigned int, int>;

template class ReduceMaxClassSparse<int, unsigned int>;
template class ReduceMaxClassSparse<int, int>;

} // namespace popnn
