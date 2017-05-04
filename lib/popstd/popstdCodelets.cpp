#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <cmath>

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
    // These are not valid for integer and boolean casts
    const auto floatVectorWidth = dataPathWidth / 32;
    return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
  }
};

template class Cast<float, float>;
template class Cast<float, half>;
template class Cast<float, int>;
template class Cast<float, bool>;

template class Cast<half, float>;
template class Cast<half, half>;
template class Cast<half, int>;
template class Cast<half, bool>;

template class Cast<int,float>;
template class Cast<int,half>;
template class Cast<int,int>;
template class Cast<int,bool>;

template class Cast<bool,float>;
template class Cast<bool,half>;
template class Cast<bool,int>;
template class Cast<bool,bool>;

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
      // These are not valid for integer and boolean casts
      cycles += 6 + (dst[i].size() + floatVectorWidth - 1) / floatVectorWidth;
    }
    return cycles;
  }
};

template class Cast2D<float, float>;
template class Cast2D<float, half>;
template class Cast2D<float, int>;
template class Cast2D<float, bool>;

template class Cast2D<half, float>;
template class Cast2D<half, half>;
template class Cast2D<half, int>;
template class Cast2D<half, bool>;

template class Cast2D<int,float>;
template class Cast2D<int,half>;
template class Cast2D<int,int>;
template class Cast2D<int,bool>;

template class Cast2D<bool,float>;
template class Cast2D<bool,half>;
template class Cast2D<bool,int>;
template class Cast2D<bool,bool>;


template <typename InType>
class Absolute : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        if (in[i][j] >= 0) {
          out[i][j] = in[i][j];
        } else {
          out[i][j] = -in[i][j];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // use f16v4/f32v4 absmax
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class Absolute<float>;
template class Absolute<half>;


template <typename InType>
class Add : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] + in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Use f16v4 and f32v2 variants
      // Assume ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class Add<float>;
template class Add<half>;


template <typename InType>
class Ceil : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = std::ceil(in[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 6;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      // use mul with 1.0 and use correct rounding mode
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
    }
    return cycles;
  }
};

template class Ceil<float>;
template class Ceil<half>;


template <typename InType>
class Divide : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] / in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      if (isFloat) {
        cycles += 5 + (1 + numElem);
      } else {
        // Convert to f32 using v2 and divide and convert back to f16
        cycles += 5 + (1 + (numElem + 1)/2 * 4);
      }
    }
    return cycles;
  }
};

template class Divide<float>;
template class Divide<half>;

template <typename InType>
class Equal : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] == in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Compare and AND with 0x1
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class Equal<float>;
template class Equal<half>;


template <typename InType>
class Exponent : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = std::exp(in[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

      if(isFloat) {
        cycles += 5 + (1 + numElem);

      } else {
        // Use f16v4 variant
        cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
      }
    }
    return cycles;
  }
};

template class Exponent<float>;
template class Exponent<half>;

template <typename InType>
class Floor : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = std::floor(in[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 6;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      // Use mul with 1.0 and use correct rounding mode
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
    }
    return cycles;
  }
};

template class Floor<float>;
template class Floor<half>;

template <typename InType>
class GreaterThan : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] > in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Compare and AND with 0x1
      // Assume that ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class GreaterThan<float>;
template class GreaterThan<half>;

template <typename InType>
class GreaterThanEqual : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] >= in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Compare and AND with 0x1
      // Assume ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class GreaterThanEqual<float>;
template class GreaterThanEqual<half>;


template <typename InType>
class LessThan : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] < in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Compare and AND with 0x1
      // Assume ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class LessThan<float>;
template class LessThan<half>;


template <typename InType>
class LessThanEqual : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] <= in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Compare and AND with 0x1
      // Assume ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class LessThanEqual<float>;
template class LessThanEqual<half>;

template <typename InType>
class Logarithm : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = std::log(in[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

      if(isFloat) {
        cycles += 5 + (1 + numElem);
      } else {
        // used f16v4 variant
        cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
      }
    }
    return cycles;
  }
};

template class Logarithm<float>;
template class Logarithm<half>;

template <typename InType>
class LogicalAnd : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] && in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      // Use AND on AUX side
      // Assume ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + 1) / 2  * 2);
    }
    return cycles;
  }
};

template class LogicalAnd<bool>;


template <typename InType>
class LogicalNot : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = !in[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // XOR on aux side
      cycles += 5 + (1 + (numElem + 1) / 2 );
    }
    return cycles;
  }
};

template class LogicalNot<bool>;


template <typename InType>
class LogicalOr : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] || in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      // OR on the aux side
      // Assume ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + 1) / 2  * 2);
    }
    return cycles;
  }
};

template class LogicalOr<bool>;


template <typename InType>
class Maximum : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = std::max(in1[i][j], in2[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Use f16v4/f32v2 variants
      // Assume that ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class Maximum<float>;
template class Maximum<half>;

template <typename InType>
class Minimum : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = std::min(in1[i][j], in2[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Use f16v4/f32v2 instructions
      // Assume that ld2xst64 cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class Minimum<float>;
template class Minimum<half>;

template <typename InType>
class Multiply : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] * in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Use f16v4/f32v2 instructions
      // Assume that ld2xst cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};

template class Multiply<float>;
template class Multiply<half>;


template <typename InType>
class NotEqual : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<bool>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j] != in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // Compare and AND with 0x1
      // Assume that ld2xst cannot be used
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 2);
    }
    return cycles;
  }
};


template class NotEqual<float>;
template class NotEqual<half>;


template <typename InType>
class Negate : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = -in[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 6;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
    }
    return cycles;
  }
};

template class Negate<float>;
template class Negate<half>;


template <typename InType>
class Power : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = std::pow(in1[i][j], in2[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      // The cycle count here is not correct
      cycles += 5 + (1 + numElem * 5);
    }
    return cycles;
  }
};

template class Power<float>;
template class Power<half>;


template <typename InType>
class Remainder : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = std::fmod(in1[i][j], in2[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned numElem = in1[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      if (isFloat) {
        // 64 bit loads and stores
        cycles += 5 + (1 + numElem);
      } else {
        // Convert to f32 using v2 and divide and convert back to f16
        cycles += 5 + (1 + (numElem + 1)/2 * 4);
      }
    }
    return cycles;
  }
};


template class Remainder<float>;
template class Remainder<half>;


template <typename InType>
class Signum : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] =  (0 < in[i][j]) - (in[i][j] < 0);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    // extra cycles to form constants
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // 64-bit AND to extract sign
      // OR to pad exponent
      // A compare to form mask to check against 0
      // AND with mask
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth * 4);
    }
    return cycles;
  }
};

template class Signum<float>;
template class Signum<half>;


template <typename InType>
class Tanh : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = std::tanh(in[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 6;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

      if(isFloat) {
        cycles += 5 + (1 + numElem);

      } else {
        /* Use f16v4 tanh instruction
         */
        cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
      }
    }
    return cycles;
  }
};

template class Tanh<float>;
template class Tanh<half>;


} // end namespace popstd
