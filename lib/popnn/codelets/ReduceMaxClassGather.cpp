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

// Takes a contiguous set of activations and divides it in chunks to be
// processed each by a worker. Each worker will return the maximum value
// in its chunk, and its index. Note that the index is relative to the start of
// the whole contiguous set, not the worker's chunk, and also it adds to it
// the value of the 'index' iput field.
// Each worker processes 'workerSize' elements but never works past
// 'size' elements form the start of 'activations'.
// For instance is 'size'=105 and 'workerSize'=80 there will only be two
// workers doing work: Worker ID=0 will do the first 80 elements and
// Worker ID=1 will do the last 25 elements.
template <typename InType, typename LabelType>
class ReduceMaxClassGather
    : public SupervisorVertexIf<!isIntegral<InType>() && ASM_CODELETS_ENABLED> {
  using OutType =
      typename std::conditional<isIntegral<InType>(), InType, float>::type;

public:
  ReduceMaxClassGather();

  Input<Vector<InType, ONE_PTR>> activations;
  const LabelType index;
  Output<Vector<OutType, ONE_PTR>> maxValue;
  Output<Vector<LabelType, ONE_PTR>> maxIndex;
  const unsigned size; // Total num of activations for all the workers to do
  const unsigned workerSize; // Num of activations for 1 worker to do

  IS_EXTERNAL_CODELET(!isIntegral<InType>());

  bool compute() {
    // nOutputs is the number of workers, and of the pairs of outputs
    // (max+index)
    const auto nOutputs = (size + workerSize - 1) / workerSize;
    for (std::size_t i = 0; i < nOutputs; ++i) {
      LabelType maxI = workerSize * i;
      InType maxV = activations[maxI];
      const auto end = (maxI + workerSize > size) ? size : maxI + workerSize;
      for (std::size_t j = maxI + 1; j < end; ++j) {
        if (activations[j] > maxV) {
          maxV = activations[j];
          maxI = j;
        }
      }
      maxValue[i] = OutType(maxV);
      maxIndex[i] = maxI + index;
    }
    return true;
  }
};

template class ReduceMaxClassGather<float, unsigned int>;
template class ReduceMaxClassGather<half, unsigned int>;
template class ReduceMaxClassGather<int, unsigned int>;
template class ReduceMaxClassGather<unsigned int, unsigned int>;

template class ReduceMaxClassGather<float, int>;
template class ReduceMaxClassGather<half, int>;
template class ReduceMaxClassGather<int, int>;
template class ReduceMaxClassGather<unsigned int, int>;

} // namespace popnn
