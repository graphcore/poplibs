// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_Reduce_hpp
#define poplibs_test_Reduce_hpp

#include <cstddef>
#include <string>
#include <vector>

#include <poplibs_support/MultiArray.hpp>
#include <popops/Reduce.hpp>

namespace poplibs_test {
namespace reduce {

// Reduce a tensor along the given dimensions with the given operation.
template <typename T>
poplibs_support::MultiArray<T>
reduce(const poplibs_support::MultiArray<T> &input,
       const std::vector<std::size_t> &reduceDims, popops::Operation op) {

  if (reduceDims.empty()) {
    poplibs_support::MultiArray<T> output(input.shape());
    std::transform(input.data(), input.data() + input.numElements(),
                   output.data(), [op](const T x) {
                     if (op == popops::Operation::SQUARE_ADD) {
                       return x * x;
                     }
                     return x;
                   });
    return output;
  }

  // Is the given dimension one that should be reduced.
  auto isReducedDim = [&](std::size_t dim) {
    return std::find(reduceDims.begin(), reduceDims.end(), dim) !=
           reduceDims.end();
  };

  // Work out the number of output values and the shape of the output.
  poplibs_support::MultiArrayShape outputShape;
  for (unsigned i = 0; i < input.shape().size(); ++i) {
    if (!isReducedDim(i)) {
      outputShape.push_back(input.shape()[i]);
    }
  }

  // The output shape is scalar if reduction is on all dimensions
  if (!outputShape.size()) {
    outputShape.push_back(1);
  }

  poplibs_support::MultiArray<T> output(outputShape);
  std::for_each(output.data(), output.data() + output.numElements(),
                [op](T &element) {
                  // Initialise the contents of each output multi-array element
                  // according to the operation type.
                  switch (op) {
                  case popops::Operation::ADD:
                  case popops::Operation::SQUARE_ADD:
                  case popops::Operation::LOGICAL_OR:
                    element = 0;
                    break;
                  case popops::Operation::MUL:
                  case popops::Operation::LOGICAL_AND:
                    element = 1;
                    break;
                  case popops::Operation::MIN:
                    element = std::numeric_limits<T>::has_infinity
                                  ? std::numeric_limits<T>::infinity()
                                  : std::numeric_limits<T>::max();
                    break;
                  case popops::Operation::MAX:
                    element = std::numeric_limits<T>::has_infinity
                                  ? -std::numeric_limits<T>::infinity()
                                  : std::numeric_limits<T>::lowest();
                    break;
                  }
                });

  poplibs_support::forEachIndex(
      input.shape(), [&](const poplibs_support::MultiArrayShapeRange indices) {
        poplibs_support::MultiArrayShape outIndices;
        for (unsigned i = 0; i < indices.size(); ++i) {
          if (!isReducedDim(i)) {
            outIndices.push_back(indices[i]);
          }
        }

        // Treat output as scalar if reduction is on all dimensions
        if (!outIndices.size()) {
          outIndices.push_back(0);
        }

        // Accumulate over the appropriate output multi-array element.
        switch (op) {
        case popops::Operation::ADD:
          output[outIndices] += input[indices];
          break;
        case popops::Operation::SQUARE_ADD:
          output[outIndices] += input[indices] * input[indices];
          break;
        case popops::Operation::MUL:
          output[outIndices] *= input[indices];
          break;
        case popops::Operation::LOGICAL_AND:
          output[outIndices] = output[outIndices] && input[indices];
          break;
        case popops::Operation::LOGICAL_OR:
          output[outIndices] = output[outIndices] || input[indices];
          break;
        case popops::Operation::MIN:
          if (input[indices] < output[outIndices])
            output[outIndices] = input[indices];
          break;
        case popops::Operation::MAX:
          if (input[indices] > output[outIndices])
            output[outIndices] = input[indices];
          break;
        }
      });

  return output;
}

} // namespace reduce
} // namespace poplibs_test

#endif // poplibs_test_Reduce_hpp
