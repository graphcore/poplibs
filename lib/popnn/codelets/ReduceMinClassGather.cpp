// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

// Same as ReduceMaxClassGather, but finds the minimum.
template <typename InType, typename LabelType>
class ReduceMinClassGather
    : public SupervisorVertexIf<!isIntegral<InType>() && ASM_CODELETS_ENABLED> {
  using OutType =
      typename std::conditional<isIntegral<InType>(), InType, float>::type;

public:
  ReduceMinClassGather();

  Input<Vector<InType, ONE_PTR>> activations;
  const LabelType index;
  Output<Vector<OutType, ONE_PTR>> minValue;
  Output<Vector<LabelType, ONE_PTR>> minIndex;
  const unsigned size; // Total num of activations for all the workers to do
  const unsigned workerSize; // Num of activations for 1 worker to do

  IS_EXTERNAL_CODELET(!isIntegral<InType>());

  bool compute() {
    // nOutputs is the number of workers, and of the pairs of outputs
    // (max+index)
    const auto nOutputs = (size + workerSize - 1) / workerSize;
    for (std::size_t i = 0; i < nOutputs; ++i) {
      LabelType minI = workerSize * i;
      InType minV = activations[minI];
      const auto end = (minI + workerSize > size) ? size : minI + workerSize;
      for (std::size_t j = minI + 1; j < end; ++j) {
        if (activations[j] < minV) {
          minV = activations[j];
          minI = j;
        }
      }
      minValue[i] = OutType(minV);
      minIndex[i] = minI + index;
    }
    return true;
  }
};

template class ReduceMinClassGather<float, unsigned int>;
template class ReduceMinClassGather<half, unsigned int>;
template class ReduceMinClassGather<int, unsigned int>;
template class ReduceMinClassGather<unsigned int, unsigned int>;

template class ReduceMinClassGather<float, int>;
template class ReduceMinClassGather<half, int>;
template class ReduceMinClassGather<int, int>;
template class ReduceMinClassGather<unsigned int, int>;

} // namespace popnn
