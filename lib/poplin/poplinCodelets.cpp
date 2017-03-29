#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <vector>
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace poplin {

template <typename FPType>
class MatMul1Partial : public Vertex {
public:
  Input<Vector<FPType>> in;
  Input<Vector<FPType>> weights;
  Output<float> out;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    float sum = 0;
    for (unsigned i = 0; i < in.size(); ++i) {
      sum += in[i] * weights[i];
    }
    *out = sum;
    return true;
  }

  uint64_t getCycleEstimate() const {
    bool isFloat = std::is_same<FPType, float>::value;
    return getMatMul1PartialCycleEstimate(isFloat, in.size(),
                                                 dataPathWidth);
  }
};

template class MatMul1Partial<float>;
template class MatMul1Partial<half>;

template <typename FPType>
class MatMul2 : public Vertex {
public:
  Input<Vector<FPType>> in;
  Vector<Input<Vector<FPType>>> weights;
  Output<Vector<float>> out;

  bool compute() {
    assert(in.size() == weights.size());
    for (auto &sum : out) {
      sum = 0.0;
    }
    for (unsigned i = 0; i != in.size(); ++i) {
      for (unsigned j = 0; j != out.size(); ++j) {
        assert(weights[i].size() == out.size());
        out[j] += in[i] * weights[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    return getMatMul2CycleEstimate(weights.size());
  }
};

template class MatMul2<float>;
template class MatMul2<half>;

template <typename FPType>
class MatMul3 : public Vertex {
public:
  Vector<Input<FPType>> d;
  Output<Vector<FPType>> dst;
  Vector<Input<Vector<FPType>>> in;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    const auto batchSize = d.size();
    for (unsigned i = 0; i < dst.size(); ++i) {
      float g = 0;
      for (unsigned b = 0; b < batchSize; ++b) {
        g += d[b] * in[b][i];
      }
      dst[i] = g;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto batchSize = d.size();
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (dst.size() + vectorWidth - 1) / vectorWidth;

    if (batchSize == 1) {
      // Assume a specialized version that accumulates directly into the
      // weight vector.
      // Inner loop involves multiplication by (*d * eta) and addition.
      return 5 + 2 * numVectors;
    } else if (batchSize <= 4) {
      // Assume a specialized version where each delta is loaded to a register
      auto deltaLoadCycles = batchSize * 3; // Load, conversion and multiply
                                            // by eta for each delta
      // Unrolled inner loop  involves multiplication by (*d * eta) and
      // addition for each element in batch
      return 5 + deltaLoadCycles + 2 * numVectors * batchSize;
    } else {
      // Use broadcast mac
      // 5 cycles to load/store accumulators in outer loop
      // Inner loop requires 2 cycles per mac to load vector and scalar
      // and  convert scalar to 32 bits in the case of halves.
      return 5 + numVectors * (5 + 2 * batchSize);
    }
  }
};

template class MatMul3<float>;
template class MatMul3<half>;

template <typename FPType>
class MatMul3Update : public Vertex {
public:
  Vector<Input<FPType>> d;
  InOut<Vector<FPType>> dst;
  Vector<Input<Vector<FPType>>> in;
  float K;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    const auto batchSize = d.size();
    for (unsigned i = 0; i < dst.size(); ++i) {
      float g = 0;
      for (unsigned b = 0; b < batchSize; ++b) {
        g += d[b] * in[b][i];
      }
      dst[i] += K * g;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto batchSize = d.size();
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    unsigned numVectors = (dst.size() + vectorWidth - 1) / vectorWidth;

    if (batchSize == 1) {
      // Assume a specialized version that accumulates directly into the
      // weight vector.
      // Inner loop involves multiplication by (*d * eta) and addition.
      return 5 + 2 * numVectors;
    } else if (batchSize <= 4) {
      // Assume a specialized version where each delta is loaded to a register
      auto deltaLoadCycles = batchSize * 3; // Load, conversion and multiply
                                            // by eta for each delta
      // Unrolled inner loop  involves multiplication by (*d * eta) and
      // addition for each element in batch
      return 5 + deltaLoadCycles + 2 * numVectors * batchSize;
    } else {
      // Use broadcast mac
      // 5 cycles to load/store accumulators in outer loop
      // Inner loop requires 2 cycles per mac to load vector and scalar
      // and  convert scalar to 32 bits in the case of halves.
      return 5 + numVectors * (5 + 2 * batchSize);
    }
  }
};

template class MatMul3Update<float>;
template class MatMul3Update<half>;

} // end namespace poplin
