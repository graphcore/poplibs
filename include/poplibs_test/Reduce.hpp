#ifndef poplibs_test_Reduce_hpp
#define poplibs_test_Reduce_hpp

#include <vector>
#include <string>
#include <cstddef>

#include <popops/Reduce.hpp>

namespace poplibs_test {
namespace reduce {

// TODO: use poplibs_support::MultiArray.
template<typename T>
struct ReferenceTensor {
  std::vector<T> values;
  std::vector<std::size_t> shape;
};

// Reduce a tensor along the given dimensions with the given operation.
template<typename T>
ReferenceTensor<T> reduce(const ReferenceTensor<T> &input,
                         const std::vector<std::size_t> &reduceDims,
                         popops::Operation op) {

  if (reduceDims.empty()) {
    auto output = input;
    if (op == popops::Operation::SQUARE_ADD) {
      // Square the output.
      for (auto &val : output.values)
        val *= val;
    }
    return output;
  }

  // Is the given dimension one that should be reduced.
  auto isReducedDim = [&](std::size_t dim) {
    return std::find(reduceDims.begin(), reduceDims.end(), dim)
        != reduceDims.end();
  };

  ReferenceTensor<T> output;

  // Work out the number of output values and the shape of the output.
  std::size_t numOutVals = 1;
  for (unsigned i = 0; i < input.shape.size(); ++i) {
    if (!isReducedDim(i)) {
      numOutVals *= input.shape[i];
      output.shape.push_back(input.shape[i]);
    }
  }

  T initVal{};

  switch (op) {
  case popops::Operation::ADD:
  case popops::Operation::SQUARE_ADD:
  case popops::Operation::LOGICAL_OR:
    initVal = 0;
    break;
  case popops::Operation::MUL:
  case popops::Operation::LOGICAL_AND:
    initVal = 1;
    break;
  case popops::Operation::MIN:
    initVal = std::numeric_limits<T>::max();
    break;
  case popops::Operation::MAX:
    initVal = std::numeric_limits<T>::lowest();
    break;
  }
  output.values = std::vector<T>(numOutVals, initVal);

  // The current input index we are at.
  std::vector<std::size_t> inputIndex(input.shape.size(), 0);

  // Loop through all the inputs.
  for (auto val : input.values) {

    // Calculate the corresponding flattened output index.
    std::size_t outIdx = 0;
    for (unsigned dim = 0; dim < inputIndex.size(); ++dim) {
      if (isReducedDim(dim))
        continue;
      outIdx *= input.shape[dim];
      outIdx += inputIndex[dim];
    }

    assert(outIdx < numOutVals);

    switch (op) {
    case popops::Operation::ADD:
      output.values[outIdx] += val;
      break;
    case popops::Operation::SQUARE_ADD:
      output.values[outIdx] += val * val;
      break;
    case popops::Operation::MUL:
      output.values[outIdx] *= val;
      break;
    case popops::Operation::LOGICAL_AND:
      output.values[outIdx] = output.values[outIdx] && val;
      break;
    case popops::Operation::LOGICAL_OR:
      output.values[outIdx] = output.values[outIdx] || val;
      break;
    case popops::Operation::MIN:
      if (val < output.values[outIdx])
        output.values[outIdx] = val;
      break;
    case popops::Operation::MAX:
      if (val > output.values[outIdx])
        output.values[outIdx] = val;
      break;
    }

    // Increment the input index.
    if (!inputIndex.empty()) {
      ++inputIndex.back();
      for (unsigned dim = inputIndex.size() - 1; dim > 0; --dim) {
        if (inputIndex[dim] >= input.shape[dim]) {
          inputIndex[dim] = 0;
          ++inputIndex[dim-1];
        }
      }
    }
  }

  return output;
}

}
}

#endif // poplibs_test_Reduce_hpp
