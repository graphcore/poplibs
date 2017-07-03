#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <iostream>
#include <cmath>

using namespace poplar;

/* Cycle cost computation for basic operations */
static uint64_t basicOpLoopCycles(unsigned overhead,
                                  unsigned numElems,
                                  unsigned vectorSize,
                                  unsigned cyclesPerVector) {
  return overhead + (numElems + vectorSize - 1) / vectorSize  * cyclesPerVector;
}

/* Cycles for comparison operations which result in bool as output */
template<typename InType>
static uint64_t comparisonOpsCycles(unsigned dataPathWidth,
                                    unsigned numElems) {
  if (std::is_same<InType, float>::value) {
    unsigned vectorWidth = dataPathWidth / 32;
    if (sizeof(bool) == 4) {
      // for dataPathWidth = 64:
      // ld64/cmp, ldst64/and on aux
      return basicOpLoopCycles(5, numElems, vectorWidth, 2);
    } else if (sizeof(bool) == 2) {
      // for dataPathWidth = 64:
      // ld64/cmp, ld64/and, st32/sort16
      return basicOpLoopCycles(5, numElems, vectorWidth, 3);
    } else if (sizeof(bool) == 1) {
      // for dataPathWidth = 64:
      // (ld64/cmp, ld64/and, sort16, atom) * 2 on aux
      //   shuf8, shl16, or, st32 on main
      return basicOpLoopCycles(5, numElems, 4 / vectorWidth,
                               (4 / vectorWidth) * 4 + 5);
    }
  } else if (std::is_same<InType, half>::value) {
    unsigned vectorWidth = dataPathWidth / 32;
    if (sizeof(bool) == 4) {
      // for dataPathWidth = 64:
      // ld64/cmp, ld64/and
      // sort16, sort16/st64
      return basicOpLoopCycles(5, numElems, vectorWidth, 2 + 2 * vectorWidth);
    } else if (sizeof(bool) == 2) {
      // ldst64/cmp, ld64/amp
      return basicOpLoopCycles(5, numElems, vectorWidth, 2);
    } else if (sizeof(bool) == 1) {
      // for dataPathWidth = 64:
      // (ld64/cmp, ld64/and, sort16, atom) * 2 on aux
      //   shuf8, shl16, or, st32 on main
      return basicOpLoopCycles(5, numElems, 4 / vectorWidth,
                               (4 / vectorWidth) * 4 + 2);
    }
  } else if (std::is_same<InType, int>::value) {
    if (sizeof(bool) == 4) {
      return basicOpLoopCycles(5, numElems, 1, 4);
    } else if (sizeof(bool) == 2) {
      // (ld32, ld32, cmp) * 2, sort16, sort16, st32
      return basicOpLoopCycles(5, numElems, 2, 9);
    } else if (sizeof(bool) == 1) {
      // (ld32, ld32, cmp) * 4, sort16, sort16, sort8, st32
      return basicOpLoopCycles(5, numElems, 4, 16);
    }
  } else if (std::is_same<InType, bool>::value) {
    unsigned vectorWidth = dataPathWidth / sizeof(bool);
    // ld64/ xor(and), ld64st64
    return basicOpLoopCycles(5, numElems, vectorWidth, 2);
  }
  assert(0 && "Bool size not supported");
  return 0;
}

namespace popstd {

template <typename InType>
class ScaledAdd : public Vertex {
public:
  Vector<InOut<Vector<InType>>> data;
  Vector<Input<Vector<InType>>> deltas;

  InType K;
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
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
      }
      else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
      }
      else if (std::is_same<InType, int>::value) {
        vectorWidth = 1;
      }
      // Inner loop uses the axpy instruction.
      cycles += 5 + (1 + (numElem + vectorWidth - 1) / vectorWidth);
    }
    return cycles;
  }
};

template class ScaledAdd<float>;
template class ScaledAdd<half>;
template class ScaledAdd<int>;


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
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      unsigned vectorWidth = 1;

      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, int>::value) {
        // ld, abs, st
        cyclesPerVector = 3;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Absolute<float>;
template class Absolute<half>;
template class Absolute<int>;


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
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, int>::value) {
        // ld, ld, add, st
        cyclesPerVector = 4;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Add<float>;
template class Add<half>;
template class Add<int>;


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
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      // use mul with 1.0 and use correct rounding mode
      unsigned cyclesPerVector = 1;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Ceil<float>;
template class Ceil<half>;

template <typename InType>
class Cos : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = std::cos(in[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 6;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = 1;
      unsigned cyclesPerVector = 1;
      //TODO - this is the same as tanh, but needs to be corrected
      if (!isFloat) {
        vectorWidth = dataPathWidth / 16;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Cos<float>;
template class Cos<half>;

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
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        cyclesPerVector = 1;
      } else if (std::is_same<InType, half>::value) {
        // Convert to f32 using v2 and divide and convert back to f16
        vectorWidth = 2;
        cyclesPerVector = 4;
      } else if (std::is_same<InType, int>::value) {
        // ld into aux, ld into aux, div, st
        cyclesPerVector = 4;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Divide<float>;
template class Divide<half>;
template class Divide<int>;

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
      cycles += comparisonOpsCycles<InType>(dataPathWidth, in1.size());
    }
    return cycles;
  }
};

template class Equal<float>;
template class Equal<half>;
template class Equal<bool>;
template class Equal<int>;


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
      unsigned vectorWidth = 1;
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;

      if(!isFloat) {
        vectorWidth = dataPathWidth / 16;
        // Use f16v4 variant
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
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
      const unsigned overhead = 6;
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;

      // Use mul with 1.0 and use correct rounding mode
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      unsigned cyclesPerVector = 1;
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
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
      cycles += comparisonOpsCycles<InType>(dataPathWidth, in1.size());
    }
    return cycles;
  }
};

template class GreaterThan<float>;
template class GreaterThan<half>;
template class GreaterThan<int>;

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
      cycles += comparisonOpsCycles<InType>(dataPathWidth, in1.size());
    }
    return cycles;
  }
};

template class GreaterThanEqual<float>;
template class GreaterThanEqual<half>;
template class GreaterThanEqual<int>;


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
      cycles += comparisonOpsCycles<InType>(dataPathWidth, in1.size());
    }
    return cycles;
  }
};

template class LessThan<float>;
template class LessThan<half>;
template class LessThan<int>;


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
      cycles += comparisonOpsCycles<InType>(dataPathWidth, in1.size());
    }
    return cycles;
  }
};

template class LessThanEqual<float>;
template class LessThanEqual<half>;
template class LessThanEqual<int>;


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
      bool isFloat = std::is_same<InType, float>::value;
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      unsigned vectorWidth = 1;

      if(!isFloat) {
        // used f16v4 variant
        vectorWidth = dataPathWidth / 16;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
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
      unsigned overhead = 6;
      unsigned vectorWidth = dataPathWidth / sizeof(bool);
      unsigned cyclesPerVector = 2;

      // Use AND on AUX side
      // Assume ld2xst64 cannot be used
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
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
      unsigned overhead = 6;
      unsigned vectorWidth = dataPathWidth / sizeof(bool);
      unsigned cyclesPerVector = 1;

      // XOR on aux side
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
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
      unsigned overhead = 6;
      unsigned vectorWidth = dataPathWidth / sizeof(bool);
      unsigned cyclesPerVector = 2;

      // OR on the aux side
      // Assume ld2xst64 cannot be used
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
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
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;

      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, int>::value) {
        // ld, ld, max, st
        cyclesPerVector = 4;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Maximum<float>;
template class Maximum<half>;
template class Maximum<int>;


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

      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;

      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, int>::value) {
        // ld, ld, min, st
        cyclesPerVector = 4;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Minimum<float>;
template class Minimum<half>;
template class Minimum<int>;


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
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, int>::value) {
        // ld, ld, mul, st
        cyclesPerVector = 4;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Multiply<float>;
template class Multiply<half>;
template class Multiply<int>;


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
      cycles += comparisonOpsCycles<InType>(dataPathWidth, in1.size());
    }
    return cycles;
  }
};

template class NotEqual<float>;
template class NotEqual<half>;
template class NotEqual<int>;
template class NotEqual<bool>;


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

      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
      } else if (std::is_same<InType, int>::value) {
        // ld, sub, st
        cyclesPerVector = 3;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Negate<float>;
template class Negate<half>;
template class Negate<int>;


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
    uint64_t cycles = 7;
    for (unsigned i = 0; i < in1.size(); ++i) {
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = 1;
      unsigned cyclesPerVector = 3;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();

      // This cycles are wrong
      // Accuracy concerns using ln
      // pow(a,b) = exp(b * log(a))
      // Doesn't handle negative values yet
      if(!isFloat) {
        // used f16v4 variant: Accuracy converns using half precision log
        vectorWidth = dataPathWidth / 16;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
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
        if (std::is_same<InType, int>::value) {
          int r = in1[i][j] / in2[i][j];
          out[i][j] = in1[i][j] - r * in2[i][j];
        } else {
          out[i][j] = std::fmod(in1[i][j], in2[i][j]);
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;

      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
      } else if (std::is_same<InType, half>::value) {
        // Convert to f32 using v2 and divide and convert back to f16
        vectorWidth = 2;
        cyclesPerVector = 4;
      } else if (std::is_same<InType, int>::value) {
        // load on aux side, mod and store result from aux
        cyclesPerVector = 4;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};


template class Remainder<float>;
template class Remainder<half>;
template class Remainder<int>;


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
      unsigned cyclesPerVector = 4;
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
      // 64-bit AND to extract sign
      // OR to pad exponent
      // A compare to form mask to check against 0
      // AND with mask
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Signum<float>;
template class Signum<half>;

template <typename InType>
class Subtract : public Vertex {
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
        out[i][j] = in1[i][j] - in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, int>::value) {
        // ld, ld, sub, st
        cyclesPerVector = 4;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Subtract<float>;
template class Subtract<half>;
template class Subtract<int>;


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
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = 1;
      unsigned cyclesPerVector = 1;
      if (std::is_same<InType, float>::value) {
        // 64 bit load with sqrt, sqrt, 64 bit
        vectorWidth = 2;
        cyclesPerVector = 3;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = 2;
        cyclesPerVector = 3;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Tanh<float>;
template class Tanh<half>;


template <typename InType>
class Sqrt : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = std::sqrt(in[i][j]);
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 1;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 1;
      } else if (std::is_same<InType, int>::value) {
        // ld, mul, st
        cyclesPerVector = 3;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Sqrt<float>;
template class Sqrt<half>;

template <typename InType>
class Square : public Vertex {
public:
  Vector<Input<Vector<InType>>> in;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in.size() == out.size());
    for (unsigned i = 0; i != in.size(); ++i) {
      assert (in[i].size() == out[i].size());
      for (unsigned j = 0; j != in[i].size(); ++j) {
        out[i][j] = in[i][j] * in[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 6;
    for (unsigned i = 0; i < in.size(); ++i) {
      unsigned overhead = 6;
      unsigned numElem = in[i].size();
      bool isFloat = std::is_same<InType, float>::value;
      unsigned vectorWidth = 1;
      unsigned cyclesPerVector = 1;
      if (!isFloat) {
        vectorWidth = dataPathWidth / 16;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Square<float>;
template class Square<half>;


template <typename InType>
class Select : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;
  Vector<Input<Vector<bool>>> in3;
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    assert(in3.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      assert(in3[i].size() == in1[i].size());
      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1[i][j] : in2[i][j];
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned cyclesPerVector = 5;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;
      // ld in1, ld in2, ld in3, movz, st
      // it may be possible to load on the Aux side but then would
      // depend on bool size. If Aux side is used masks must be created after
      // expanding bools to match the input datum size
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Select<float>;
template class Select<half>;
template class Select<int>;


template <typename InType>
class Clamp : public Vertex {
public:
  Vector<Input<Vector<InType>>> in1;
  Vector<Input<Vector<InType>>> in2;  // lower bound
  Vector<Input<Vector<InType>>> in3;  // upper bound
  Vector<Output<Vector<InType>>> out;
  SimOnlyField<unsigned> dataPathWidth;

  bool compute() {
    assert(in1.size() == out.size());
    assert(in2.size() == in1.size());
    assert(in3.size() == in1.size());
    for (unsigned i = 0; i != in1.size(); ++i) {
      assert(in1[i].size() == out[i].size());
      assert(in2[i].size() == in1[i].size());
      assert(in3[i].size() == in1[i].size());

      for (unsigned j = 0; j != in1[i].size(); ++j) {
        out[i][j] = in1[i][j];
        if (out[i][j] < in2[i][j]) {
          out[i][j] = in2[i][j];
        }
        if (out[i][j] > in3[i][j]) {
          out[i][j] = in3[i][j];
        }
      }
    }
    return true;
  }

  uint64_t getCycleEstimate() const {
    uint64_t cycles = 5;
    for (unsigned i = 0; i < in1.size(); ++i) {
      unsigned cyclesPerVector = 1;
      unsigned overhead = 6;
      unsigned numElem = in1[i].size();
      unsigned vectorWidth = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = dataPathWidth / 32;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, half>::value) {
        vectorWidth = dataPathWidth / 16;
        cyclesPerVector = 2;
      } else if (std::is_same<InType, int>::value) {
        // ld, ld, ld, cmp, movz, cmp, st
        cyclesPerVector = 7;
      }
      cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                  cyclesPerVector);
    }
    return cycles;
  }
};

template class Clamp<float>;
template class Clamp<half>;
template class Clamp<int>;

} // end namespace popstd
