#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>

using namespace poplar;

namespace popstd {

template <typename FPType>
class ScaledAdd : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> data;
  Vector<Input<Vector<FPType>>> deltas;

  float K;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(data.size() == deltas.size());
    for (unsigned i = 0; i < data.size(); ++i) {
      assert (deltas[i].size() == data[i].size());
      for (unsigned j = 0; j < data[i].size(); ++j) {
        data[i][j] += K * deltas[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < data.size(); ++i) {
      unsigned numElem = data[i].size();
      bool isFloat = std::is_same<FPType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Inner loop uses the axpy instruction.
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
    }
    return cycles;
  }
};

template class ScaledAdd<float>;
template class ScaledAdd<half>;


template <typename FPType>
class HadamardProd : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> A;
  Vector<Input<Vector<FPType>>> B;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(A.size() == B.size());
    for (unsigned i = 0; i < A.size(); ++i) {
      assert (A[i].size() == B[i].size());
      for (unsigned j = 0; j < A[i].size(); ++j) {
        A[i][j] *= B[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < A.size(); ++i) {
      unsigned numElem = A[i].size();
      bool isFloat = std::is_same<FPType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;
      cycles += 5 + (1 + numVectors * 2);
    }
    return cycles;
  }
};

template class HadamardProd<float>;
template class HadamardProd<half>;



template <typename FPType>
class Zero : public Vertex {
public:
  Output<Vector<FPType>> out;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned i = 0; i < out.size(); ++i) {
      out[i] = 0;
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // TODO: make this more accurate
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    auto zeroCycles = (out.size() + vectorWidth - 1) / vectorWidth;
    return 2 // run
           + 5 // vertex cycles
           + zeroCycles;
  }
};

template class Zero<float>;
template class Zero<half>;

template <typename FPType>
class Zero2D : public Vertex {
public:
  Vector<Output<Vector<FPType>>> out;

  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (auto &row : out) {
      for (auto &x : row) {
        x = 0;
      }
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const {
    // TODO: make this more accurate
    bool isFloat = std::is_same<FPType, float>::value;
    const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
    std::uint64_t cycles = 2 // run
                           + 5; // vertex overhead
    for (auto &row : out) {
      auto zeroCycles = (row.size() + vectorWidth - 1) / vectorWidth;
      auto const loopOverhead = 3;
      cycles += loopOverhead + zeroCycles;
    }
    return cycles;
  }
};

template class Zero2D<float>;
template class Zero2D<half>;


template <typename SrcType, typename DstType>
class Cast : public Vertex {
public:
  Input<Vector<SrcType>> src;
  Output<Vector<DstType>> dst;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    for (unsigned i = 0; i < dst.size(); ++i) {
      dst[i] = static_cast<DstType>(src[i]);
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    const auto floatVectorWidth = dataPathWidth / 32;
    return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
  }
};

template class Cast<float, half>;
template class Cast<half, float>;
template class Cast<float, float>;
template class Cast<half, half>;

template <typename SrcType, typename DstType>
class Cast2D : public Vertex {
public:
  Vector<Input<Vector<SrcType>>> src;
  Vector<Output<Vector<DstType>>> dst;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(src.size() == dst.size());
    for (unsigned i = 0; i != dst.size(); ++i) {
      assert(src[i].size() == dst[i].size());
      for (unsigned j = 0; j != dst[i].size(); ++j) {
        dst[i][j] = static_cast<DstType>(src[i][j]);
      }
    }
    return true;
  }

  std::uint64_t getCycleEstimate() const {
    const auto floatVectorWidth = dataPathWidth / 32;
    std::uint64_t cycles = 5;
    for (unsigned i = 0; i != dst.size(); ++i) {
      // Estimate based on 6 cycles of loop overhead per src / dst pointer pair:
      //
      // 1: load src
      // 2: load dst
      // 3: load length
      // 4: load src[0]
      // 5: { load src[1] ; convert src[0] }
      // 6: repeat
      cycles += 6 + (dst[i].size() + floatVectorWidth - 1) / floatVectorWidth;
    }
    return cycles;
  }
};

template class Cast2D<float, half>;
template class Cast2D<half, float>;
template class Cast2D<float, float>;
template class Cast2D<half, half>;


} // end namespace popstd
