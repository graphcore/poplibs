// Copyright (c) Graphcore Ltd, All rights reserved.
#include "MinHeapView.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popnn {

/*
  See the description of ReduceMaxNClassSparse for the general algorithm for
  calculating the top |numK| from the given |activations|. This version is only
  different in that it works on multiple batches of input at a time.
*/
template <typename FPType, bool Sort = false>
class ReduceMaxNClassGather : public Vertex {
public:
  ReduceMaxNClassGather();

  Input<Vector<FPType, ONE_PTR>> activations;
  const unsigned index;

  Output<Vector<FPType, ONE_PTR>> maxValues;

  Output<Vector<unsigned, ONE_PTR>> maxValuesIndices;

  unsigned numK;
  const unsigned size;
  const unsigned short divisorLog2;
  const bool shouldSort;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // Work is split between up to N workers based on the divisor
    // and outputs to each maxValue/Index output based on this
    const auto divisor = (1u << divisorLog2);
    const auto nOutputs = (size + divisor - 1) / divisor;
    for (std::size_t i = 0; i < nOutputs; ++i) {
      std::size_t offset = divisor * i;
      const auto end = (offset + divisor > size) ? size : offset + divisor;

      // Smallest value in the MaxHeap. Aka the smallest of all the largest
      // numbers.
      FPType smallest = activations[offset];
      int smallestIndex = -1;

      // To avoid having the extra vertex data associated with vector of vectors
      // and vector lists we just have one vector and treat it as a 2D vector.
      int topKIndex = numK * i;
      unsigned *currentPartialBucket = &maxValuesIndices[topKIndex];
      FPType *currentPartialBucketData = &maxValues[topKIndex];

      // Create an inplace view of the maxValues array as a min heap.
      MinHeapView<decltype(currentPartialBucket), decltype(activations), FPType>
          heapView{currentPartialBucket, activations};

      heapView.Push(offset, 0);

      size_t elements_in_heap = 1;
      for (std::size_t j = offset + 1; j < end; ++j) {

        if (elements_in_heap < numK) {
          heapView.Push(j, elements_in_heap);
          elements_in_heap++;
        } else if (heapView.IsLargerThanSmallest(activations[j])) {
          // Replace the smallest value in the heap with this value then shuffle
          // it to the correct place in the heap.
          heapView.ReplaceAndRepair(j, numK);
        }
      }

      // If this gather is uneven I.E the NumOuput*topK is greater than the size
      // we have to adjust the numK so we don't sort the bit at the end as well.
      // How much remains in the last iteration.
      unsigned remainder = end - offset;

      // If there is more numK than remaining elements
      unsigned adjustedNumK = numK > remainder ? remainder : numK;

      // If the remainder is smaller than numK is larger than topK we have to
      // fill the remaining numK slots with the smallest possible value for that
      // type.
      if (adjustedNumK != numK) {
        for (int k = adjustedNumK; k < numK; ++k) {
          currentPartialBucketData[k] = std::numeric_limits<FPType>::lowest();
          currentPartialBucket[k] = std::numeric_limits<unsigned>::max();
        }
      }
      // Sort if template parameter Sort is true and the runtime flag is set. If
      // the runtime flag will never be set (I.E compile time Sort=false) this
      // should be trivially eliminated by DCE.
      if (Sort && shouldSort) {
        heapView.Sort(adjustedNumK);
      }

      for (int k = 0; k < adjustedNumK; ++k) {
        currentPartialBucketData[k] = activations[currentPartialBucket[k]];

        // Add the index base to get the actual index.
        currentPartialBucket[k] += index;
      }
    }
    return true;
  }
};

// Unsorted.
template class ReduceMaxNClassGather<float>;
template class ReduceMaxNClassGather<int>;
template class ReduceMaxNClassGather<unsigned int>;

// Sorted outputs.
template class ReduceMaxNClassGather<float, true>;
template class ReduceMaxNClassGather<int, true>;
template class ReduceMaxNClassGather<unsigned int, true>;

} // namespace popnn
