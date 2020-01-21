// Copyright (c) 2019, Graphcore Ltd, All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <math.h>

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "popops/ExprOp.hpp"
#include "popops/elementwiseCodelets.hpp"

#ifdef VECTOR_AVAIL_SHORT_SPAN
static constexpr auto SPAN_TYPE = poplar::VectorLayout::SHORT_SPAN;
#else
static constexpr auto SPAN_TYPE = poplar::VectorLayout::SPAN;
#endif
#ifdef VECTOR_AVAIL_SCALED_PTR64
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::SCALED_PTR64;
#else
static constexpr auto PTR_ALIGN64 = poplar::VectorLayout::ONE_PTR;
#endif
#ifdef VECTOR_AVAIL_SCALED_PTR32
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::SCALED_PTR32;
#else
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::ONE_PTR;
#endif

using namespace poplar;

namespace popops {
template <typename InputType, typename OutputType>
class CheckAccuracyWhenCast : public Vertex {
public:
  const float tolerance;
  Input<InputType> input;

  CheckAccuracyWhenCast();
  static bool computeImpl(InputType input, float tolerance) {
#ifdef __IPU__
    // Disable exceptions as the following can create numbers that are out of
    // range in half precision.  We need to store / restore the FP_CTL as
    // the worker will continue to run the actual scaledAdd code - done
    // outside this function
    __builtin_ipu_uput(0x00000000,
                       CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
#endif
    const auto castInput = static_cast<OutputType>(input);
    const auto relativeError = static_cast<InputType>(
        (static_cast<float>(std::fabs(input)) * tolerance));
    return relativeError > std::abs(static_cast<InputType>(castInput) - input);
  }

  bool compute() { return computeImpl(*input, tolerance); }
};

template <> class CheckAccuracyWhenCast<float, half> : public Vertex {
public:
  const float tolerance;
  Input<float> input;

  CheckAccuracyWhenCast();
  static bool computeImpl(float input, float tolerance) {
#ifdef __IPU__
    // Disable exceptions as the following can create numbers that are out of
    // range in half precision.  We need to store / restore the FP_CTL as
    // the worker will continue to run the actual scaledAdd code - done outside
    // this function
    __builtin_ipu_uput(0x00000000,
                       CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
    // Cast to half and back to float, decision is based on relative error
    const auto castInput = static_cast<half>(input);
    return (ipu::fabs(input) * tolerance) >
           ipu::fabs(static_cast<float>(castInput) - input);

#else
    const auto castInput = static_cast<half>(input);
    // As the CPU doesn't deal with halves correctly, then exclude out of
    // range numbers (as half) from being considered accurate.
    return std::fabs(input) > 65504
               ? false
               : (std::fabs(input) * tolerance) >
                     std::fabs(static_cast<float>(castInput) - input);

#endif
  }
  bool compute() { return computeImpl(*input, tolerance); }
};

template <typename AType>
using InputScaleType = Input<Vector<AType, PTR_ALIGN64, 8>>;

// Vector types for scaledAdd variants.  All float, half types use ld64 type
// instructions, therefore required 8 byte alignment and can have a more compact
// vertex state as a result, if we also constrain the size field.

template <typename AType>
using InOutAType2D =
    std::conditional_t<std::is_integral<AType>{},
                       Vector<InOut<Vector<AType, SPAN, alignof(AType)>>, SPAN>,
                       Vector<InOut<Vector<AType, SPAN_TYPE, 8>>, SPAN>>;

template <typename BType>
using InputBType2D = std::conditional_t<
    std::is_integral<BType>{},
    Vector<Input<Vector<BType, ONE_PTR, alignof(BType)>>, ONE_PTR>,
    Vector<Input<Vector<BType, PTR_ALIGN64, 8>>, ONE_PTR>>;

template <typename AType, typename BType, typename ScaleType>
using ComputeType =
    std::conditional_t<((std::is_same<float, AType>::value ||
                         std::is_same<float, BType>::value ||
                         std::is_same<float, ScaleType>::value)),
                       float, AType>;

template <typename AType, typename ScaleType> constexpr bool hasAssembly() {
  return std::is_same<AType, ScaleType>::value ||
         std::is_same<float, ScaleType>::value;
}

template <typename AType, typename BType, typename ScaleType, bool isConstant,
          bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] ScaledAddSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  ScaledAddSupervisor();
  using ComputeType = ComputeType<AType, BType, ScaleType>;

  IS_EXTERNAL_CODELET(true);

  InOut<Vector<AType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<BType, PTR_ALIGN64, 8>> B;
  const ScaleType scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] += static_cast<AType>(static_cast<ComputeType>(scaleB) *
                                 static_cast<ComputeType>(B[i]));
    }
    return true;
  }
};

#define DEF_SCALED_ADD_SUPER_VERTEX(SCALE_TYPE, SCALE, CONSTRAINTS,            \
                                    IS_CONSTANT, IS_CONSTRAINED)               \
  template <typename AType, typename BType, typename ScaleType>                \
  class CONSTRAINTS ScaledAddSupervisor<AType, BType, ScaleType, IS_CONSTANT,  \
                                        IS_CONSTRAINED>                        \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    ScaledAddSupervisor();                                                     \
    using ComputeType = ComputeType<AType, BType, ScaleType>;                  \
                                                                               \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<AType, PTR_ALIGN64, 8>> A;                                    \
    unsigned short size;                                                       \
    Input<Vector<BType, PTR_ALIGN64, 8>> B;                                    \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = size;                                                    \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        A[i] += static_cast<AType>(static_cast<ComputeType>(SCALE) *           \
                                   static_cast<ComputeType>(B[i]));            \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_SCALED_ADD_SUPER_VERTEX(const ScaleType, scaleB,
                            [[poplar::constraint("elem(*A) != elem(*B)")]],
                            true, true)
DEF_SCALED_ADD_SUPER_VERTEX(InputScaleType<ScaleType>, scaleB[0],
                            [[poplar::constraint("elem(*A) != elem(*B)")]],
                            false, true)
DEF_SCALED_ADD_SUPER_VERTEX(const ScaleType, scaleB, , true, false)
DEF_SCALED_ADD_SUPER_VERTEX(InputScaleType<ScaleType>, scaleB[0], , false,
                            false)

// Note that the <half, float, half, IS_CONSTANT, IS_CONSTRAINED> variant
// is not used at present as there is no 2D version.  It is tested however.
#define INSTANTIATE_SCALED_ADD_SUPER_VERTICES(IS_CONSTANT, IS_CONSTRAINED)     \
  template class ScaledAddSupervisor<float, float, float, IS_CONSTANT,         \
                                     IS_CONSTRAINED>;                          \
  template class ScaledAddSupervisor<half, half, half, IS_CONSTANT,            \
                                     IS_CONSTRAINED>;                          \
  template class ScaledAddSupervisor<half, float, half, IS_CONSTANT,           \
                                     IS_CONSTRAINED>;

INSTANTIATE_SCALED_ADD_SUPER_VERTICES(true, true)
INSTANTIATE_SCALED_ADD_SUPER_VERTICES(true, false)
INSTANTIATE_SCALED_ADD_SUPER_VERTICES(false, true)
INSTANTIATE_SCALED_ADD_SUPER_VERTICES(false, false)

template class ScaledAddSupervisor<half, half, float, true, false>;
template class ScaledAddSupervisor<half, half, float, true, true>;

template class ScaledAddSupervisor<float, half, half, false, false>;
template class ScaledAddSupervisor<float, half, half, true, false>;
template class ScaledAddSupervisor<float, half, float, false, false>;
template class ScaledAddSupervisor<float, half, float, true, false>;

#define DEF_SCALED_ADD_FLOAT_SCALE_SUPER_VERTEX(CONSTRAINTS, IS_CONSTRAINED)   \
  template <>                                                                  \
  class CONSTRAINTS                                                            \
      ScaledAddSupervisor<half, half, float, false, IS_CONSTRAINED>            \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    ScaledAddSupervisor();                                                     \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<half, PTR_ALIGN64, 8>> A;                                     \
    unsigned short size;                                                       \
    Input<Vector<half, PTR_ALIGN64, 8>> B;                                     \
    InputScaleType<float> scaleB;                                              \
    const float tolerance;                                                     \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = size;                                                    \
      if (CheckAccuracyWhenCast<float, half>::computeImpl(scaleB[0],           \
                                                          tolerance)) {        \
        const auto halfScale = static_cast<half>(scaleB[0]);                   \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          A[i] += halfScale * B[i];                                            \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          A[i] += static_cast<half>(scaleB[0] * static_cast<float>(B[i]));     \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_SCALED_ADD_FLOAT_SCALE_SUPER_VERTEX(
    [[poplar::constraint("elem(*A) != elem(*B)")]], true)
DEF_SCALED_ADD_FLOAT_SCALE_SUPER_VERTEX(, false)

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledAddSupervisor<int, int, int, false, false>;
template class ScaledAddSupervisor<unsigned, unsigned, unsigned, false, false>;
template class ScaledAddSupervisor<int, int, int, true, false>;
template class ScaledAddSupervisor<unsigned, unsigned, unsigned, true, false>;

template <typename AType, typename BType, typename ScaleType, bool isConstant,
          bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] ScaledAdd2D
    : public Vertex {
public:
  ScaledAdd2D();

  using ComputeType = ComputeType<AType, BType, ScaleType>;
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<AType> A;
  InputBType2D<BType> B;
  const ScaleType scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] += static_cast<AType>(static_cast<ComputeType>(scaleB) *
                                        static_cast<ComputeType>(refIn[j]));
      }
    }
    return true;
  }
};
#define DEF_SCALED_ADD_2D_VERTEX(SCALE_TYPE, SCALE, CONSTRAINTS, IS_CONSTANT,  \
                                 IS_CONSTRAINED)                               \
  template <typename AType, typename BType, typename ScaleType>                \
  class CONSTRAINTS                                                            \
      ScaledAdd2D<AType, BType, ScaleType, IS_CONSTANT, IS_CONSTRAINED>        \
      : public Vertex {                                                        \
  public:                                                                      \
    ScaledAdd2D();                                                             \
    using ComputeType = ComputeType<AType, BType, ScaleType>;                  \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<AType> A;                                                     \
    InputBType2D<BType> B;                                                     \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        for (unsigned j = 0; j < limJ; ++j) {                                  \
          refOut[j] += static_cast<AType>(static_cast<ComputeType>(SCALE) *    \
                                          static_cast<ComputeType>(refIn[j])); \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_SCALED_ADD_2D_VERTEX(const ScaleType, scaleB,
                         [[poplar::constraint("elem(**A) != elem(**B)")]], true,
                         true)
DEF_SCALED_ADD_2D_VERTEX(Input<ScaleType>, *scaleB,
                         [[poplar::constraint("elem(**A) != elem(**B)")]],
                         false, true)
DEF_SCALED_ADD_2D_VERTEX(const ScaleType, scaleB, , true, false)
DEF_SCALED_ADD_2D_VERTEX(Input<ScaleType>, *scaleB, , false, false)

template class ScaledAdd2D<float, float, float, true, true>;
template class ScaledAdd2D<half, half, half, true, true>;
template class ScaledAdd2D<float, float, float, false, true>;
template class ScaledAdd2D<half, half, half, false, true>;

template class ScaledAdd2D<float, float, float, false, false>;
template class ScaledAdd2D<half, half, half, false, false>;
template class ScaledAdd2D<float, float, float, true, false>;
template class ScaledAdd2D<half, half, half, true, false>;

template class ScaledAdd2D<float, half, half, false, false>;
template class ScaledAdd2D<float, half, half, true, false>;
template class ScaledAdd2D<float, half, float, false, false>;
template class ScaledAdd2D<float, half, float, true, false>;

#define DEF_SCALED_ADD_FLOAT_SCALE_2D_VERTEX(CONSTRAINTS, IS_CONSTRAINED)      \
  template <>                                                                  \
  class CONSTRAINTS ScaledAdd2D<half, half, float, false, IS_CONSTRAINED>      \
      : public Vertex {                                                        \
  public:                                                                      \
    ScaledAdd2D();                                                             \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<half> A;                                                      \
    InputBType2D<half> B;                                                      \
    Input<float> scaleB;                                                       \
    const float tolerance;                                                     \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = A.size();                                                \
      if (CheckAccuracyWhenCast<float, half>::computeImpl(scaleB,              \
                                                          tolerance)) {        \
        const auto halfScale = static_cast<half>(*scaleB);                     \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          unsigned limJ = A[i].size();                                         \
          auto const &refIn = B[i];                                            \
          auto &refOut = A[i];                                                 \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] += halfScale * refIn[j];                                 \
          }                                                                    \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          unsigned limJ = A[i].size();                                         \
          auto const &refIn = B[i];                                            \
          auto &refOut = A[i];                                                 \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] +=                                                       \
                static_cast<half>(*scaleB * static_cast<float>(refIn[j]));     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_SCALED_ADD_FLOAT_SCALE_2D_VERTEX(
    [[poplar::constraint("elem(**A) != elem(**B)")]], true)
DEF_SCALED_ADD_FLOAT_SCALE_2D_VERTEX(, false)

template class ScaledAdd2D<half, half, float, true, true>;
template class ScaledAdd2D<half, half, float, true, false>;

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledAdd2D<int, int, int, true, false>;
template class ScaledAdd2D<unsigned, unsigned, unsigned, true, false>;
template class ScaledAdd2D<int, int, int, false, false>;
template class ScaledAdd2D<unsigned, unsigned, unsigned, false, false>;

template <typename AType, typename BType, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] ScaledSubtractSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<AType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<BType, PTR_ALIGN64, 8>> B;
  InputScaleType<AType> scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] -= scaleB[0] * static_cast<AType>(B[i]);
    }
    return true;
  }
};

template <typename AType, typename BType>
class ScaledSubtractSupervisor<AType, BType, false>
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<AType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<BType, PTR_ALIGN64, 8>> B;
  InputScaleType<AType> scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] -= scaleB[0] * static_cast<AType>(B[i]);
    }
    return true;
  }
};
template class ScaledSubtractSupervisor<float, float, true>;
template class ScaledSubtractSupervisor<half, half, true>;
template class ScaledSubtractSupervisor<half, float, true>;

template class ScaledSubtractSupervisor<float, float, false>;
template class ScaledSubtractSupervisor<half, half, false>;
template class ScaledSubtractSupervisor<half, float, false>;

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledSubtractSupervisor<int, int, false>;
template class ScaledSubtractSupervisor<unsigned, unsigned, false>;

template <typename InType, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] ScaledSubtract2D
    : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<InType> A;
  InputBType2D<InType> B;
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

template <typename InType>
class ScaledSubtract2D<InType, false> : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<InType> A;
  InputBType2D<InType> B;
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

template class ScaledSubtract2D<float, true>;
template class ScaledSubtract2D<half, true>;
template class ScaledSubtract2D<float, false>;
template class ScaledSubtract2D<half, false>;

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledSubtract2D<int, false>;
template class ScaledSubtract2D<unsigned, false>;

template <typename InType, bool isConstant, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] aXPlusbYSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  aXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<InType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<InType, PTR_ALIGN64, 8>> B;
  const InType scaleA;
  const InType scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] = scaleA * A[i] + scaleB * B[i];
    }
    return true;
  }
};

#define DEF_AXPLUSBY_SUPER_VERTEX(SCALE_TYPE, PTR, CONSTRAINTS, IS_CONSTANT,   \
                                  IS_CONSTRAINED)                              \
  template <typename InType>                                                   \
  class CONSTRAINTS aXPlusbYSupervisor<InType, IS_CONSTANT, IS_CONSTRAINED>    \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    aXPlusbYSupervisor();                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<InType, PTR_ALIGN64, 8>> A;                                   \
    unsigned short size;                                                       \
    Input<Vector<InType, PTR_ALIGN64, 8>> B;                                   \
    SCALE_TYPE scaleA;                                                         \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = size;                                                    \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        A[i] = scaleA PTR * A[i] + scaleB PTR * B[i];                          \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXPLUSBY_SUPER_VERTEX(const InType, ,
                          [[poplar::constraint("elem(*A) != elem(*B)")]], true,
                          true)
DEF_AXPLUSBY_SUPER_VERTEX(InputScaleType<InType>, [0],
                          [[poplar::constraint("elem(*A) != elem(*B)")]], false,
                          true)
DEF_AXPLUSBY_SUPER_VERTEX(const InType, , , true, false)
DEF_AXPLUSBY_SUPER_VERTEX(InputScaleType<InType>, [0], , false, false)

template class aXPlusbYSupervisor<half, true, true>;
template class aXPlusbYSupervisor<half, false, true>;
template class aXPlusbYSupervisor<half, true, false>;
template class aXPlusbYSupervisor<half, false, false>;

template <typename InType, bool isConstant, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] aXPlusbY2D
    : public Vertex {
public:
  aXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<InType> A;
  InputBType2D<InType> B;
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
#define DEF_AXPLUSBY_2D_VERTEX(SCALE_TYPE, PTR, CONSTRAINTS, IS_CONSTANT,      \
                               IS_CONSTRAINED)                                 \
  template <typename InType>                                                   \
  class CONSTRAINTS aXPlusbY2D<InType, IS_CONSTANT, IS_CONSTRAINED>            \
      : public Vertex {                                                        \
  public:                                                                      \
    aXPlusbY2D();                                                              \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<InType> A;                                                    \
    InputBType2D<InType> B;                                                    \
    SCALE_TYPE scaleA;                                                         \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        for (unsigned j = 0; j < limJ; ++j) {                                  \
          refOut[j] = PTR scaleA * refOut[j] + PTR scaleB * refIn[j];          \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXPLUSBY_2D_VERTEX(const InType, ,
                       [[poplar::constraint("elem(**A) != elem(**B)")]], true,
                       true)
DEF_AXPLUSBY_2D_VERTEX(Input<InType>, *,
                       [[poplar::constraint("elem(**A) != elem(**B)")]], false,
                       true)
DEF_AXPLUSBY_2D_VERTEX(const InType, , , true, false)
DEF_AXPLUSBY_2D_VERTEX(Input<InType>, *, , false, false)

template class aXPlusbY2D<half, true, true>;
template class aXPlusbY2D<half, false, true>;
template class aXPlusbY2D<half, true, false>;
template class aXPlusbY2D<half, false, false>;

template <typename FPType>
class [[poplar::constraint("elem(**A) != elem(**B)")]] HadamardProd
    : public Vertex {
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

template <typename InType> class Zero : public Vertex {
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

template <typename FPType> class Zero2d : public Vertex {
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

// A couple of macros to instantiate more compactly the templates of the various
// Cast vertices, for all possible combinations of input and output types
// (float, half, signed/unsinged ints and bool)
#define INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, srcType)                  \
  template class CastVertexName<srcType, float>;                               \
  template class CastVertexName<srcType, half>;                                \
  template class CastVertexName<srcType, int>;                                 \
  template class CastVertexName<srcType, unsigned>;                            \
  template class CastVertexName<srcType, bool>;
#define INSTANTIATE_CAST(CastVertexName)                                       \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, float)                          \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, half)                           \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, int)                            \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, unsigned)                       \
  INSTANTIATE_CAST_BY_SRC_TYPE(CastVertexName, bool)

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] Cast : public Vertex {
public:
  Cast();

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

  static const poplar::VectorLayout inLayout =
      inAlign == 8 ? PTR_ALIGN64 : ONE_PTR;
  static const poplar::VectorLayout outLayout =
      outAlign == 4 ? PTR_ALIGN32 : (outAlign == 8 ? PTR_ALIGN64 : ONE_PTR);
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

INSTANTIATE_CAST(Cast)

#ifdef __IPU__
// The vertices defined by this template will be called by the supervisor
// vertex only
template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] CastWorker
    : public Vertex {
public:
  CastWorker();

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

  static const poplar::VectorLayout inLayout =
      inAlign == 8 ? PTR_ALIGN64 : ONE_PTR;
  static const poplar::VectorLayout outLayout =
      outAlign == 4 ? PTR_ALIGN32 : (outAlign == 8 ? PTR_ALIGN64 : ONE_PTR);
  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  const unsigned partitionParams;

  bool compute() {
    unsigned wId = getWsr();
    // Read the comment for partitionParams in CastSupervisor below to see
    // how work is partitioned to this worker.
    const unsigned deltaLast = partitionParams & 0x7;
    const unsigned workerLast = (partitionParams >> 3) & 0x7;
    const unsigned workerCount = (partitionParams >> 6) & 0x7;
    unsigned workerElems = partitionParams >> 9;
    unsigned offs = wId * workerElems;
    if (wId >= workerCount) {
      workerElems -= 4;
      offs -= (wId - workerCount) * 4;
    }
    if (wId == workerLast) {
      workerElems -= deltaLast;
    }
    for (unsigned i = 0; i < workerElems; ++i) {
      dst[offs + i] = static_cast<DstType>(src[offs + i]);
    }
    return true;
  }
};

// Note that we don't define here the following:
//    1. FLOAT<->HALF conversion (defined in assembly)
//    2. Identity conversions (XXX->XXX) and INT<->UNSIGNED as these will be
//       replaced with Copy() in popops::cast()
template class CastWorker<float, int>;
template class CastWorker<float, unsigned>;
template class CastWorker<float, bool>;

template class CastWorker<half, int>;
template class CastWorker<half, unsigned>;
template class CastWorker<half, bool>;

template class CastWorker<int, float>;
template class CastWorker<int, half>;
template class CastWorker<int, bool>;

template class CastWorker<unsigned int, float>;
template class CastWorker<unsigned int, half>;
template class CastWorker<unsigned int, bool>;

template class CastWorker<bool, float>;
template class CastWorker<bool, half>;
template class CastWorker<bool, int>;
template class CastWorker<bool, unsigned int>;

#endif

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(*src) != elem(*dst)")]] CastSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  CastSupervisor();

  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

  static const poplar::VectorLayout inLayout =
      inAlign == 8 ? PTR_ALIGN64 : ONE_PTR;
  static const poplar::VectorLayout outLayout =
      outAlign == 4 ? PTR_ALIGN32 : (outAlign == 8 ? PTR_ALIGN64 : ONE_PTR);

  Input<Vector<SrcType, inLayout, inAlign>> src;
  Output<Vector<DstType, outLayout, outAlign>> dst;
  // 'partitionParams' contains 4 bit fields defining how the work is
  // partitioned among workers:
  //
  //                           23 bits                    3     3     3
  //  +------------------------------------------------+-----+-----+-----+
  //  |                       Welems                   | Wcnt| Wlst| Dlst|
  //  +------------------------------------------------+-----+-----+-----+
  // MSB                                                               LSB
  //
  // The first 'Wcnt' (Worker Count) workers will process 'Welems' (Worker
  // Elements) elements each ('Welems' always a multiple of 4).
  // The other (6-'Wcnt') workers will process 'Welems-4' elems (could be none).
  // Need to correct the above for the last worker, (index 'Wlst =Worker Last),
  // that will process 'Dlst' (Delta Last, 0..3) fewer elements than specified
  // by 'Welems' or 'Welems-4'.
  // For instance:
  //
  // Total elements : 15  =>   WCnt=4, Welems=4, Wlst=3, Dlst=1
  //
  //  WkId 0     WkId 1     WkId 2     WkId 3     WkId 4     WkId 5
  // 4 elems    4 elems    4 elems    3 elems    0 elems    0 elems
  //   +---------------+------------------+  \
  //                   |                      \
  //                Wcnt=4                  Last one does 'Dlst' fewer elems.
  //
  //
  // Total elements : 30  =>   WCnt=2, Welems=8, Wlst=5, Dlst=2
  //
  //  WkId 0     WkId 1     WkId 2     WkId 3     WkId 4     WkId 5
  // 8 elems    8 elems    4 elems    4 elems    4 elems    2 elems
  // +--------+-------+                                        |
  //          |                                                |
  //        Wcnt=2                       Last one does 'Dlst' fewer elems.
  //
  // This ensures that all workers start on a 4-element boundary.
  const unsigned partitionParams;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const unsigned deltaLast = partitionParams & 0x7;
    const unsigned workerLast = (partitionParams >> 3) & 0x7;
    const unsigned workerCount = (partitionParams >> 6) & 0x7;
    const unsigned workerElems = partitionParams >> 9;
    const unsigned numElems = workerCount * workerElems +
                              (CTXT_WORKERS - workerCount) * (workerElems - 4) -
                              deltaLast;
    for (unsigned i = 0; i < numElems; ++i) {
      dst[i] = static_cast<DstType>(src[i]);
    }
    return true;
  }
};

INSTANTIATE_CAST(CastSupervisor)

template <typename SrcType, typename DstType>
class [[poplar::constraint("elem(**src) != elem(**dst)")]] Cast2d
    : public Vertex {
public:
  // Logic for the minimum aligment based on Src and Dst Type
  static const bool floatHalf =
      std::is_same<SrcType, float>::value && std::is_same<DstType, half>::value;
  static const bool halfFloat =
      std::is_same<SrcType, half>::value && std::is_same<DstType, float>::value;

  static const bool ext = halfFloat || floatHalf;
  static const unsigned outAlign = ext ? (halfFloat ? 8 : 4) : alignof(DstType);
  static const unsigned inAlign = ext ? 8 : alignof(SrcType);

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

INSTANTIATE_CAST(Cast2d)

template <typename InType> class Clamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2; // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3; // upper bound
  Vector<Output<Vector<InType>>> out;

  static const bool ext =
      std::is_same<InType, float>::value || std::is_same<InType, half>::value;
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

template <typename InType> class Select : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2;
  Vector<Input<Vector<bool, ONE_PTR>>, ONE_PTR> in3;
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

template <typename InType> class BroadcastClamp : public Vertex {
public:
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in1;
  Input<InType> in2;
  Input<InType> in3;
  Vector<Output<Vector<InType>>> out;

  static const bool ext =
      std::is_same<InType, float>::value || std::is_same<InType, half>::value;
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
template <typename InType> class BroadcastSelect : public Vertex {
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
template <typename InType> class BroadcastSelectorSelect : public Vertex {
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

template <typename InType> class ClampInPlace : public Vertex {
public:
  Vector<InOut<Vector<InType>>> in1Out;
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in2; // lower bound
  Vector<Input<Vector<InType, ONE_PTR>>, ONE_PTR> in3; // upper bound

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

template <typename InType> class BroadcastClampInPlace : public Vertex {
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

template <typename InType> class SelectInPlace : public Vertex {
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

template <typename InType, bool isConstant, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] aXMinusbYSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  aXMinusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<InType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<InType, PTR_ALIGN64, 8>> B;
  const InType scaleA;
  const InType scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] = scaleA * A[i] - scaleB * B[i];
    }
    return true;
  }
};

#define DEF_AXMINUSBY_SUPER_VERTEX(SCALE_TYPE, PTR, CONSTRAINTS, IS_CONSTANT,  \
                                   IS_CONSTRAINED)                             \
  template <typename InType>                                                   \
  class CONSTRAINTS aXMinusbYSupervisor<InType, IS_CONSTANT, IS_CONSTRAINED>   \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    aXMinusbYSupervisor();                                                     \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<InType, PTR_ALIGN64, 8>> A;                                   \
    unsigned short size;                                                       \
    Input<Vector<InType, PTR_ALIGN64, 8>> B;                                   \
    SCALE_TYPE scaleA;                                                         \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = size;                                                    \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        A[i] = scaleA PTR * A[i] - scaleB PTR * B[i];                          \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXMINUSBY_SUPER_VERTEX(InputScaleType<InType>, [0],
                           [[poplar::constraint("elem(*A) != elem(*B)")]],
                           false, true)
DEF_AXMINUSBY_SUPER_VERTEX(InputScaleType<InType>, [0], , false, false)

template class aXMinusbYSupervisor<half, false, true>;
template class aXMinusbYSupervisor<half, false, false>;

template <typename InType, bool isConstant, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] aXMinusbY2D
    : public Vertex {
public:
  aXMinusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<InType> A;
  InputBType2D<InType> B;
  const InType scaleA;
  const InType scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] = scaleA * refOut[j] - scaleB * refIn[j];
      }
    }
    return true;
  }
};
#define DEF_AXMINUSBY_2D_VERTEX(SCALE_TYPE, PTR, CONSTRAINTS, IS_CONSTANT,     \
                                IS_CONSTRAINED)                                \
  template <typename InType>                                                   \
  class CONSTRAINTS aXMinusbY2D<InType, IS_CONSTANT, IS_CONSTRAINED>           \
      : public Vertex {                                                        \
  public:                                                                      \
    aXMinusbY2D();                                                             \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<InType> A;                                                    \
    InputBType2D<InType> B;                                                    \
    SCALE_TYPE scaleA;                                                         \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        for (unsigned j = 0; j < limJ; ++j) {                                  \
          refOut[j] = PTR scaleA * refOut[j] - PTR scaleB * refIn[j];          \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXMINUSBY_2D_VERTEX(Input<InType>, *,
                        [[poplar::constraint("elem(**A) != elem(**B)")]], false,
                        true)
DEF_AXMINUSBY_2D_VERTEX(Input<InType>, *, , false, false)

template class aXMinusbY2D<half, false, true>;
template class aXMinusbY2D<half, false, false>;

template <typename InType, bool isConstant, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] XMinusaXPlusbYSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  XMinusaXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(false);

  InOut<Vector<InType, SCALED_PTR64, 8>> A;
  unsigned short size;
  Input<Vector<InType, SCALED_PTR64, 8>> B;
  const InType scaleA;
  const InType scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] = A[i] - scaleA * A[i] + scaleB * B[i];
    }
    return true;
  }
};

#define DEF_XMINUSAXPLUSBY_SUPER_VERTEX(SCALE_TYPE, PTR, CONSTRAINTS,          \
                                        IS_CONSTANT, IS_CONSTRAINED)           \
  template <typename InType>                                                   \
  class CONSTRAINTS                                                            \
      XMinusaXPlusbYSupervisor<InType, IS_CONSTANT, IS_CONSTRAINED>            \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    XMinusaXPlusbYSupervisor();                                                \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<InType, SCALED_PTR64, 8>> A;                                  \
    unsigned short size;                                                       \
    Input<Vector<InType, SCALED_PTR64, 8>> B;                                  \
    SCALE_TYPE scaleA;                                                         \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = size;                                                    \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        A[i] = A[i] - scaleA PTR * A[i] + scaleB PTR * B[i];                   \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_XMINUSAXPLUSBY_SUPER_VERTEX(const InType, ,
                                [[poplar::constraint("elem(*A) != elem(*B)")]],
                                true, true)
DEF_XMINUSAXPLUSBY_SUPER_VERTEX(InputScaleType<InType>, [0],
                                [[poplar::constraint("elem(*A) != elem(*B)")]],
                                false, true)
DEF_XMINUSAXPLUSBY_SUPER_VERTEX(const InType, , , true, false)
DEF_XMINUSAXPLUSBY_SUPER_VERTEX(InputScaleType<InType>, [0], , false, false)

template class XMinusaXPlusbYSupervisor<half, false, true>;
template class XMinusaXPlusbYSupervisor<half, false, false>;
template class XMinusaXPlusbYSupervisor<half, true, true>;
template class XMinusaXPlusbYSupervisor<half, true, false>;

template <typename InType, bool isConstant, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] XMinusaXPlusbY2D
    : public Vertex {
public:
  XMinusaXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<InType> A;
  InputBType2D<InType> B;
  const InType scaleA;
  const InType scaleB;

  bool compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] = refOut[j] - scaleA * refOut[j] + scaleB * refIn[j];
      }
    }
    return true;
  }
};
#define DEF_XMINUSAXPLUSBY_2D_VERTEX(SCALE_TYPE, PTR, CONSTRAINTS,             \
                                     IS_CONSTANT, IS_CONSTRAINED)              \
  template <typename InType>                                                   \
  class CONSTRAINTS XMinusaXPlusbY2D<InType, IS_CONSTANT, IS_CONSTRAINED>      \
      : public Vertex {                                                        \
  public:                                                                      \
    XMinusaXPlusbY2D();                                                        \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<InType> A;                                                    \
    InputBType2D<InType> B;                                                    \
    SCALE_TYPE scaleA;                                                         \
    SCALE_TYPE scaleB;                                                         \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        for (unsigned j = 0; j < limJ; ++j) {                                  \
          refOut[j] =                                                          \
              refOut[j] - PTR scaleA * refOut[j] + PTR scaleB * refIn[j];      \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_XMINUSAXPLUSBY_2D_VERTEX(const InType, ,
                             [[poplar::constraint("elem(**A) != elem(**B)")]],
                             true, true)
DEF_XMINUSAXPLUSBY_2D_VERTEX(Input<InType>, *,
                             [[poplar::constraint("elem(**A) != elem(**B)")]],
                             false, true)
DEF_XMINUSAXPLUSBY_2D_VERTEX(const InType, , , true, false)
DEF_XMINUSAXPLUSBY_2D_VERTEX(Input<InType>, *, , false, false)

template class XMinusaXPlusbY2D<half, true, true>;
template class XMinusaXPlusbY2D<half, true, false>;
template class XMinusaXPlusbY2D<half, false, true>;
template class XMinusaXPlusbY2D<half, false, false>;
} // namespace popops
