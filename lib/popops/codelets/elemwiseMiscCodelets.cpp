// Copyright (c) 2019, Graphcore Ltd, All rights reserved.
#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

#include <cassert>
#include <cstring>
#include <cmath>

#include "util.hpp"
#include "popops/ExprOp.hpp"
#include "popops/elementwiseCodelets.hpp"

namespace popops {

template <typename AType, typename BType, bool isConstant>
class
[[poplar::constraint("elem(*A) != elem(*B)")]]
ScaledAddSupervisor : public SupervisorVertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<BType>{} ? alignof(BType) : 8;
  }
public:
  ScaledAddSupervisor();

  IS_EXTERNAL_CODELET(true);

  InOut<Vector<AType, SPAN, minAlign()>> A;
  Input<Vector<BType, ONE_PTR, minAlign()>> B;
  const AType scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
        A[i] += scaleB * static_cast<AType>(B[i]);
    }
    return true;
  }
};

template <typename AType, typename BType>
class
[[poplar::constraint("elem(*A) != elem(*B)")]]
ScaledAddSupervisor <AType, BType, false> : public SupervisorVertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<BType>{} ? alignof(BType) : 8;
  }
public:
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<AType, SPAN, minAlign()>> A;
  Input<Vector<BType, ONE_PTR, minAlign()>> B;
  Input<AType> scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
        A[i] += *scaleB * static_cast<AType>(B[i]);
    }
    return true;
  }
};

template class ScaledAddSupervisor<float, float, true>;
template class ScaledAddSupervisor<half, half, true>;
template class ScaledAddSupervisor<int, int, true>;
template class ScaledAddSupervisor<unsigned, unsigned, true>;

template class ScaledAddSupervisor<float, float, false>;
template class ScaledAddSupervisor<half, half, false>;
template class ScaledAddSupervisor<int, int, false>;
template class ScaledAddSupervisor<unsigned, unsigned, false>;

template class ScaledAddSupervisor<half, float, true>;
template class ScaledAddSupervisor<half, float, false>;

template <typename InType, bool isConstant>
class
[[poplar::constraint("elem(**A) != elem(**B)")]]
ScaledAdd2D : public Vertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  ScaledAdd2D();

  IS_EXTERNAL_CODELET(true);

  Vector<InOut<Vector<InType, SPAN, minAlign()>>> A;
  Vector<Input<Vector<InType, ONE_PTR, minAlign()>>, ONE_PTR> B;
  const InType scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
          refOut[j] += scaleB * refIn[j];
      }
    }
    return true;
  }
};

template <typename InType>
class
[[poplar::constraint("elem(**A) != elem(**B)")]]
ScaledAdd2D <InType, false>: public Vertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  IS_EXTERNAL_CODELET(true);

  Vector<InOut<Vector<InType, SPAN, minAlign()>>> A;
  Vector<Input<Vector<InType, ONE_PTR, minAlign()>>, ONE_PTR> B;
  Input<InType> scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
          refOut[j] += *scaleB * refIn[j];
      }
    }
    return true;
  }
};

template class ScaledAdd2D<float, true>;
template class ScaledAdd2D<half, true>;
template class ScaledAdd2D<int, true>;
template class ScaledAdd2D<unsigned, true>;

template class ScaledAdd2D<float, false>;
template class ScaledAdd2D<half, false>;
template class ScaledAdd2D<int, false>;
template class ScaledAdd2D<unsigned, false>;

template <typename AType, typename BType>
class
[[poplar::constraint("elem(*A) != elem(*B)")]]
ScaledSubtractSupervisor : public SupervisorVertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<BType>{} ? alignof(BType) : 8;
  }
public:
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<AType, SPAN, minAlign()>> A;
  Input<Vector<BType, ONE_PTR, minAlign()>> B;
  Input<AType> scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
        A[i] -= *scaleB * static_cast<AType>(B[i]);
    }
    return true;
  }
};

template class ScaledSubtractSupervisor<float, float>;
template class ScaledSubtractSupervisor<half, half>;
template class ScaledSubtractSupervisor<int, int>;
template class ScaledSubtractSupervisor<unsigned, unsigned>;
template class ScaledSubtractSupervisor<half, float>;

template <typename InType>
class
[[poplar::constraint("elem(**A) != elem(**B)")]]
ScaledSubtract2D : public Vertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  IS_EXTERNAL_CODELET(true);

  Vector<InOut<Vector<InType, SPAN, minAlign()>>> A;
  Vector<Input<Vector<InType, ONE_PTR, minAlign()>>, ONE_PTR> B;
  Input<InType> scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
          refOut[j] -= *scaleB * refIn[j];
      }
    }
    return true;
  }
};

template class ScaledSubtract2D<float>;
template class ScaledSubtract2D<half>;
template class ScaledSubtract2D<int>;
template class ScaledSubtract2D<unsigned>;

template <typename InType, bool isConstant>
class
[[poplar::constraint("elem(*A) != elem(*B)")]]
aXPlusbYSupervisor : public SupervisorVertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  aXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<InType, SPAN, minAlign()>> A;
  Input<Vector<InType, ONE_PTR, minAlign()>> B;
  const InType scaleA;
  const InType scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
        A[i] = scaleA * A[i] + scaleB * B[i];
    }
    return true;
  }
};


template <typename InType>
class
[[poplar::constraint("elem(*A) != elem(*B)")]]
aXPlusbYSupervisor <InType, false>: public SupervisorVertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<InType, SPAN, minAlign()>> A;
  Input<Vector<InType, ONE_PTR, minAlign()>> B;
  Input<InType> scaleA;
  Input<InType> scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
        A[i] = *scaleA * A[i] + *scaleB * B[i];
    }
    return true;
  }
};

template class aXPlusbYSupervisor<half, true>;
template class aXPlusbYSupervisor<half, false>;

template <typename InType, bool isConstant>
class
[[poplar::constraint("elem(**A) != elem(**B)")]]
aXPlusbY2D : public Vertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  aXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  Vector<InOut<Vector<InType, SPAN, minAlign()>>> A;
  Vector<Input<Vector<InType, ONE_PTR, minAlign()>>, ONE_PTR> B;
  const InType scaleA;
  const InType scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
          refOut[j] = scaleA * refOut[j] + scaleB * refIn[j];
      }
    }
    return true;
  }
};

template <typename InType>
class
[[poplar::constraint("elem(**A) != elem(**B)")]]
aXPlusbY2D <InType, false>: public Vertex {
  constexpr static std::size_t minAlign() {
    // the floating point variants use ld2x64pace and therefore require
    // 64-bit alignment.
    return std::is_integral<InType>{} ? alignof(InType) : 8;
  }
public:
  IS_EXTERNAL_CODELET(true);

  Vector<InOut<Vector<InType, SPAN, minAlign()>>> A;
  Vector<Input<Vector<InType, ONE_PTR, minAlign()>>, ONE_PTR> B;
  Input<InType> scaleA;
  Input<InType> scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
          refOut[j] = *scaleA * refOut[j] + *scaleB * refIn[j];
      }
    }
    return true;
  }
};

template class aXPlusbY2D<half, true>;
template class aXPlusbY2D<half, false>;

template <typename FPType>
class
[[poplar::constraint("elem(**A) != elem(**B)")]]
HadamardProd : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> A;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> B;

  bool compute() {
    const unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      const unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] *= refIn[j];
      }
    }
    return true;
  }
};

template class HadamardProd<float>;
template class HadamardProd<half>;



template <typename InType>
class Zero : public Vertex {
public:
  Output<Vector<InType>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (auto &x : out) {
      x = 0;
    }
    return true;
  }
};

template class Zero<float>;
template class Zero<half>;
template class Zero<int>;
template class Zero<unsigned>;

template <typename FPType>
class Zero2d : public Vertex {
public:
  Vector<Output<Vector<FPType>>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (auto &row : out) {
      for (auto &x : row) {
        x = 0;
      }
    }
    return true;
  }
};

template class Zero2d<float>;
template class Zero2d<half>;

template <typename SrcType, typename DstType>
class
[[poplar::constraint("elem(*src) != elem(*dst)")]]
Cast : public Vertex {
public:
  Cast();

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf = std::is_same<SrcType,float>::value
            && std::is_same<DstType,half>::value;
  static const bool halfFloat = std::is_same<SrcType,half>::value
            && std::is_same<DstType,float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : 1;
  static const unsigned inAlign = ext ? 8 : 1;

  static const poplar::VectorLayout inLayout =
      inAlign == 8 ? SCALED_PTR64 : ONE_PTR;
  static const poplar::VectorLayout outLayout =
      outAlign == 4 ? SCALED_PTR32 :
                      (outAlign == 8 ? SCALED_PTR64 : ONE_PTR);

  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned numElems;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    for (unsigned i = 0; i < numElems; ++i) {
      dst[i] = static_cast<DstType>(src[i]);
    }
    return true;
  }
};

template class Cast<float, float>;
template class Cast<float, half>;
template class Cast<float, int>;
template class Cast<float, unsigned>;
template class Cast<float, bool>;

template class Cast<half, float>;
template class Cast<half, half>;
template class Cast<half, int>;
template class Cast<half, unsigned>;
template class Cast<half, bool>;

template class Cast<int, float>;
template class Cast<int, half>;
template class Cast<int, int>;
template class Cast<int, unsigned>;
template class Cast<int, bool>;

template class Cast<unsigned, float>;
template class Cast<unsigned, half>;
template class Cast<unsigned, int>;
template class Cast<unsigned, unsigned>;
template class Cast<unsigned, bool>;

template class Cast<bool, float>;
template class Cast<bool, half>;
template class Cast<bool, int>;
template class Cast<bool, unsigned>;
template class Cast<bool, bool>;

template <typename SrcType, typename DstType>
class
[[poplar::constraint("elem(**src) != elem(**dst)")]]
Cast2d : public Vertex {
public:

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf = std::is_same<SrcType,float>::value
            && std::is_same<DstType,half>::value;
  static const bool halfFloat = std::is_same<SrcType,half>::value
            && std::is_same<DstType,float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : 1;
  static const unsigned inAlign = ext ? 8 : 1;

  Vector<Input<Vector<SrcType, ONE_PTR, inAlign>>, ONE_PTR> src;
  Vector<Output<Vector<DstType, SPAN, outAlign>>> dst;

  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    const unsigned limI = dst.size();
    for (unsigned i = 0; i != limI; ++i) {
      const unsigned limJ = dst[i].size();
      auto const &refSrc = src[i];
      auto &refDst = dst[i];
      for (unsigned j = 0; j != limJ; ++j) {
        refDst[j] = static_cast<DstType>(refSrc[j]);
      }
    }
    return true;
  }
};

template class Cast2d<float, float>;
template class Cast2d<float, half>;
template class Cast2d<float, int>;
template class Cast2d<float, unsigned>;
template class Cast2d<float, bool>;

template class Cast2d<half, float>;
template class Cast2d<half, half>;
template class Cast2d<half, int>;
template class Cast2d<half, unsigned>;
template class Cast2d<half, bool>;

template class Cast2d<int, float>;
template class Cast2d<int, half>;
template class Cast2d<int, int>;
template class Cast2d<int, unsigned>;
template class Cast2d<int, bool>;

template class Cast2d<unsigned, float>;
template class Cast2d<unsigned, half>;
template class Cast2d<unsigned, int>;
template class Cast2d<unsigned, unsigned>;
template class Cast2d<unsigned, bool>;

template class Cast2d<bool, float>;
template class Cast2d<bool, half>;
template class Cast2d<bool, int>;
template class Cast2d<bool, unsigned>;
template class Cast2d<bool, bool>;

template <typename InType>
class Clamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;  // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3;  // upper bound
  Vector<Output<Vector<InType>>> out;

  static const bool ext = std::is_same<InType,float>::value
            || std::is_same<InType,half>::value;
  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {

      for (unsigned j = 0; j != out[i].size(); ++j) {
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
};

template class Clamp<float>;
template class Clamp<half>;
template class Clamp<int>;

template <typename InType>
class Select : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Vector<Input<Vector<bool,   ONE_PTR>>, ONE_PTR> in3;
  Vector<Output<Vector<InType, SPAN, 4>>> out;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1[i][j] : in2[i][j];
      }
    }
    return true;
  }
};

template class Select<float>;
template class Select<half>;
template class Select<int>;
template class Select<bool>;

template <typename InType>
class BroadcastClamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Input<InType> in2;
  Input<InType> in3;
  Vector<Output<Vector<InType>>> out;

  static const bool ext = std::is_same<InType,float>::value
            || std::is_same<InType,half>::value;
  IS_EXTERNAL_CODELET(ext);

  bool compute() {
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned j = 0; j < out[i].size(); ++j) {
        out[i][j] = in1[i][j];
        if (out[i][j] < *in2) {
          out[i][j] = *in2;
        }
        if (out[i][j] > *in3) {
          out[i][j] = *in3;
        }
      }
    }
    return true;
  }
};

template class BroadcastClamp<float>;
template class BroadcastClamp<half>;
template class BroadcastClamp<int>;

// 'Select' ternary operator where the selector (boolean third operand) is a
// tensor, while the 1st and 2nd operands are scalars (that are broadcasted
// into the output)
template <typename InType>
class BroadcastSelect : public Vertex {
public:
  Input<InType> in1;
  Input<InType> in2;
  Vector<Input<Vector<bool, ONE_PTR>>, ONE_PTR> in3;
  Vector<Output<Vector<InType, SPAN>>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      for (unsigned j = 0; j != out[i].size(); ++j) {
        out[i][j] = in3[i][j] ? in1 : in2;
      }
    }
    return true;
  }
};

template class BroadcastSelect<float>;
template class BroadcastSelect<half>;
template class BroadcastSelect<int>;
template class BroadcastSelect<bool>;

// 'Select' ternary operator where the selector (boolean third operand) is a
// scalar and needs broadcasting, while the 1st and 2nd operands are tensors
// Just copy 'in1', or 'in2', into 'out', based on the scalar 'in3'.
template <typename InType>
class BroadcastSelectorSelect : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Input<bool> in3;
  Vector<Output<Vector<InType, SPAN>>> out;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    const auto in = in3 ? in1 : in2;
    for (unsigned i = 0; i < out.size(); ++i) {
      for (unsigned j = 0; j < out[i].size(); ++j) {
        out[i][j] = in[i][j];
      }
    }
    return true;
  }
};

template class BroadcastSelectorSelect<float>;
template class BroadcastSelectorSelect<half>;
template class BroadcastSelectorSelect<int>;
template class BroadcastSelectorSelect<bool>;

template <typename InType>
class ClampInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;  // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3;  // upper bound

  bool compute() {
    for (unsigned i = 0; i != in1Out.size(); ++i) {
      for (unsigned j = 0; j != in1Out[i].size(); ++j) {
        if (in1Out[i][j] < in2[i][j]) {
          in1Out[i][j] = in2[i][j];
        }
        if (in1Out[i][j] > in3[i][j]) {
          in1Out[i][j] = in3[i][j];
        }
      }
    }
    return true;
  }
};

template class ClampInPlace<float>;
template class ClampInPlace<half>;
template class ClampInPlace<int>;

template <typename InType>
class BroadcastClampInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Input<InType> in2;
  Input<InType> in3;

  bool compute() {
    for (unsigned i = 0; i < in1Out.size(); ++i) {
      for (unsigned j = 0; j < in1Out[i].size(); ++j) {
        if (in1Out[i][j] < *in2) {
          in1Out[i][j] = *in2;
        }
        if (in1Out[i][j] > *in3) {
          in1Out[i][j] = *in3;
        }
      }
    }
    return true;
  }
};

template class BroadcastClampInPlace<float>;
template class BroadcastClampInPlace<half>;
template class BroadcastClampInPlace<int>;

template <typename InType>
class SelectInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Vector<Input<Vector<bool, ONE_PTR>>, ONE_PTR> in3;

  bool compute() {
    for (unsigned i = 0; i != in1Out.size(); ++i) {
      for (unsigned j = 0; j != in1Out[i].size(); ++j) {
        in1Out[i][j] = in3[i][j] ? in1Out[i][j] : in2[i][j];
      }
    }
    return true;
  }
};

template class SelectInPlace<float>;
template class SelectInPlace<half>;
template class SelectInPlace<int>;
template class SelectInPlace<bool>;

template <typename InType>
class BroadcastSelectorSelectInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Input<bool> in3;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    if (in3 == false) {
      for (unsigned i = 0; i != in1Out.size(); ++i) {
        for (unsigned j = 0; j != in1Out[i].size(); ++j) {
          in1Out[i][j] = in2[i][j];
        }
      }
    }
    return true;
  }
};

template class BroadcastSelectorSelectInPlace<float>;
template class BroadcastSelectorSelectInPlace<half>;
template class BroadcastSelectorSelectInPlace<int>;
template class BroadcastSelectorSelectInPlace<bool>;

}
