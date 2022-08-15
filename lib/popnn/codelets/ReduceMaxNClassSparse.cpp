// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "MinHeapView.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popnn {

/*
  Calcuate the top |numK| values of the given |activations| and store them in
  the tensor |maxValues|. The algorithm works as follows.

  1. Create a Heap of the maxValues. We use a HeapView which treats the values
  as if they are a min heap and operates in place.
  2. Init the Heap by pushing the first |numK| values to it.
  3. For all the values after |numK| we check if they are larger than the
  smallest previously added. If so we add them to the heap and remove the
  smallest at the same time. We just overwrite the smallest then repair the
  tree.
  4. Repeat 3  until we have exhausted the input |activations|.
  5. If the Sort template parameter has been given we use pop to shuffle the
  heap into array sorted order before returning it.
*/
template <typename DataType, bool Sort = false>
class ReduceMaxNClassSparse : Vertex {
public:
  ReduceMaxNClassSparse();
  Input<Vector<DataType>> activations;

  Output<Vector<DataType>> maxValues;

  Output<Vector<unsigned, ONE_PTR>> maxValuesIndices;

  Input<Vector<unsigned, ONE_PTR>> labels;

  const bool shouldSort;

  unsigned numK;
  IS_EXTERNAL_CODELET(false);
  void compute() {
    // Create an inplace view of the maxValues array as a min heap.
    MinHeapView<decltype(maxValuesIndices), decltype(activations), unsigned>
        heapView{maxValuesIndices, activations};

    heapView.Push(0, 0);

    for (std::size_t i = 1; i < activations.size(); ++i) {
      if (i < numK) {
        // Initialize the heap with the first "k" values.
        heapView.Push(i, i);
      } else if (heapView.IsLargerThanSmallest(i)) {
        // Replace the smallest value in the heap with this value then shuffle
        // it to the correct place in the heap.
        heapView.ReplaceAndRepair(i, numK);
      }
    }

    // Sort if template parameter Sort is true and the runtime flag is set. If
    // the runtime flag will never be set (I.E compile time Sort=false) this
    // should be trivially eliminated by DCE.
    if (Sort && shouldSort) {
      heapView.Sort(numK);
    }

    for (int i = 0; i < numK; ++i) {
      maxValues[i] = activations[maxValuesIndices[i]];

      // Assign the max index its actual label. "i" is in the range of 0-size
      // where 0-size are relative to the activation context. A.K.A are a
      // subarray of the actual activations.
      maxValuesIndices[i] = labels[maxValuesIndices[i]];
    }
  }
};

// Unsorted.
template class ReduceMaxNClassSparse<float>;
template class ReduceMaxNClassSparse<half>;
template class ReduceMaxNClassSparse<int>;
template class ReduceMaxNClassSparse<unsigned int>;

// Sorted outputs.
template class ReduceMaxNClassSparse<float, true>;
template class ReduceMaxNClassSparse<half, true>;
template class ReduceMaxNClassSparse<int, true>;
template class ReduceMaxNClassSparse<unsigned int, true>;

} // namespace popnn
