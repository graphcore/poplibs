// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <math.h>

#include "elementwiseCodelets.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "popops/ExprOp.hpp"

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

  // Do the actual casting and accuracy check. This must be done in a separate
  // function while T12725 is not solved to prevent LLVM schedulers from
  // reordering the corresponding instructions before the UPUT.
  static bool __attribute__((noinline))
  castAndCheck(InputType input, float tolerance) {
    const auto castInput = static_cast<OutputType>(input);
    const auto relativeError = static_cast<InputType>(
        (static_cast<float>(std::fabs(input)) * tolerance));
    return relativeError > std::abs(static_cast<InputType>(castInput) - input);
  }

  static bool computeImpl(InputType input, float tolerance) {
#ifdef __IPU__
    // Disable exceptions as the following can create numbers that are out of
    // range in half precision.  We need to store / restore the FP_CTL as
    // the worker will continue to run the actual scaledAdd code - done
    // outside this function
    __builtin_ipu_uput(0x00000000,
                       CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
#endif
    return castAndCheck(input, tolerance);
  }

  bool compute() { return computeImpl(*input, tolerance); }
};

template <> class CheckAccuracyWhenCast<float, half> : public Vertex {
public:
  const float tolerance;
  Input<float> input;

  CheckAccuracyWhenCast();

  // Do the actual casting and accuracy check. This must be done in a separate
  // function while T12725 is not solved to prevent LLVM schedulers from
  // reordering the corresponding instructions before the UPUT.
  static bool __attribute__((noinline))
  castAndCheck(float input, float tolerance) {
#ifdef __IPU__
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

  static bool computeImpl(float input, float tolerance) {
#ifdef __IPU__
    // Disable exceptions as the following can create numbers that are out of
    // range in half precision.  We need to store / restore the FP_CTL as
    // the worker will continue to run the actual scaledAdd code - done outside
    // this function
    __builtin_ipu_uput(0x00000000,
                       CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
#endif
    return castAndCheck(input, tolerance);
  }
  bool compute() { return computeImpl(*input, tolerance); }
};

// Check if two values x0, x1 are "accurate enough" when cast to HALF.
// For each value 'x' we compute a maximum acceptable error which is
// x*tolerance and then we check if (x-half(x)) is (strictly) smaller than
// that error.
// Return 'true' if both values pass the check.
bool checkAccuracyWhenCastFloatV2ToHalf(float x0, float x1, float tolerance) {
  if (std::fabs(x0) > 65504 || std::fabs(x1) > 65504)
    return false;
#ifdef __IPU__
  unsigned save_fp_ctl =
      __builtin_ipu_uget(CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
  __builtin_ipu_uput(0x00000000,
                     CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
#endif
  float maxErr0 = std::fabs(tolerance * x0);
  float maxErr1 = std::fabs(tolerance * x1);
  half x0half = static_cast<half>(x0);
  half x1half = static_cast<half>(x1);
  float diff0 = static_cast<float>(x0half) - x0;
  float diff1 = static_cast<float>(x1half) - x1;
#ifdef __IPU__
  __builtin_ipu_uput(save_fp_ctl,
                     CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK);
#endif
  return std::fabs(diff0) < maxErr0 && std::fabs(diff1) < maxErr1;
}

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

#define INSTANTIATE_SCALED_ADD_SUPER_VERTICES(IS_CONSTANT, IS_CONSTRAINED)     \
  template class ScaledAddSupervisor<float, float, float, IS_CONSTANT,         \
                                     IS_CONSTRAINED>;                          \
  template class ScaledAddSupervisor<half, half, half, IS_CONSTANT,            \
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

template class ScaledAddSupervisor<half, float, half, true, false>;
template class ScaledAddSupervisor<half, float, half, false, false>;
template class ScaledAddSupervisor<half, float, float, true, false>;
template class ScaledAddSupervisor<half, float, float, false, false>;

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
template class ScaledAdd2D<float, float, float, true, false>;
template class ScaledAdd2D<float, float, float, false, true>;
template class ScaledAdd2D<float, float, float, false, false>;

template class ScaledAdd2D<half, half, half, true, true>;
template class ScaledAdd2D<half, half, half, true, false>;
template class ScaledAdd2D<half, half, half, false, true>;
template class ScaledAdd2D<half, half, half, false, false>;

template class ScaledAdd2D<half, half, float, true, true>;
template class ScaledAdd2D<half, half, float, true, false>;

template class ScaledAdd2D<half, float, float, true, false>;
template class ScaledAdd2D<half, float, float, false, false>;

template class ScaledAdd2D<half, float, half, true, false>;
template class ScaledAdd2D<half, float, half, false, false>;

template class ScaledAdd2D<float, half, half, true, false>;
template class ScaledAdd2D<float, half, half, false, false>;

template class ScaledAdd2D<float, half, float, true, false>;
template class ScaledAdd2D<float, half, float, false, false>;

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

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledAdd2D<int, int, int, true, false>;
template class ScaledAdd2D<unsigned, unsigned, unsigned, true, false>;
template class ScaledAdd2D<int, int, int, false, false>;
template class ScaledAdd2D<unsigned, unsigned, unsigned, false, false>;

template <typename AType, typename BType, typename ScaleType,
          bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] ScaledSubtractSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<AType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<BType, PTR_ALIGN64, 8>> B;
  InputScaleType<ScaleType> scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] -= scaleB[0] * static_cast<AType>(B[i]);
    }
    return true;
  }
};

template <typename AType, typename BType>
class ScaledSubtractSupervisor<AType, BType, AType, false>
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

#define DEF_SCALED_SUB_FLOAT_SCALE_SUPER_VERTEX(CONSTRAINTS, IS_CONSTRAINED)   \
  template <>                                                                  \
  class CONSTRAINTS                                                            \
      ScaledSubtractSupervisor<half, half, float, IS_CONSTRAINED>              \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    ScaledSubtractSupervisor();                                                \
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
          A[i] -= halfScale * B[i];                                            \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          A[i] -= static_cast<half>(scaleB[0] * static_cast<float>(B[i]));     \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_SCALED_SUB_FLOAT_SCALE_SUPER_VERTEX(
    [[poplar::constraint("elem(*A) != elem(*B)")]], true)
DEF_SCALED_SUB_FLOAT_SCALE_SUPER_VERTEX(, false)

template class ScaledSubtractSupervisor<float, float, float, true>;
template class ScaledSubtractSupervisor<half, half, half, true>;
template class ScaledSubtractSupervisor<half, float, half, true>;

template class ScaledSubtractSupervisor<float, float, float, false>;
template class ScaledSubtractSupervisor<half, half, half, false>;
template class ScaledSubtractSupervisor<half, float, half, false>;

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledSubtractSupervisor<int, int, int, false>;
template class ScaledSubtractSupervisor<unsigned, unsigned, unsigned, false>;

template <typename DataType, typename ScaleType, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] ScaledSubtract2D
    : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  Input<ScaleType> scaleB;

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

template <typename DataType>
class ScaledSubtract2D<DataType, DataType, false> : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  Input<DataType> scaleB;

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

// Specialisation for the mixed case for half tensor with float scales.
#define DEF_SCALED_SUB_FLOAT_SCALE_2D_VERTEX(CONSTRAINTS, IS_CONSTRAINED)      \
  template <>                                                                  \
  class CONSTRAINTS ScaledSubtract2D<half, float, IS_CONSTRAINED>              \
      : public Vertex {                                                        \
  public:                                                                      \
    ScaledSubtract2D();                                                        \
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
            refOut[j] -= halfScale * refIn[j];                                 \
          }                                                                    \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          unsigned limJ = A[i].size();                                         \
          auto const &refIn = B[i];                                            \
          auto &refOut = A[i];                                                 \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] -=                                                       \
                static_cast<half>(*scaleB * static_cast<float>(refIn[j]));     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_SCALED_SUB_FLOAT_SCALE_2D_VERTEX(, false)
DEF_SCALED_SUB_FLOAT_SCALE_2D_VERTEX(
    [[poplar::constraint("elem(**A) != elem(**B)")]], true)

template class ScaledSubtract2D<float, float, true>;
template class ScaledSubtract2D<half, half, true>;
template class ScaledSubtract2D<float, float, false>;
template class ScaledSubtract2D<half, half, false>;

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledSubtract2D<int, int, false>;
template class ScaledSubtract2D<unsigned, unsigned, false>;

template <typename DataType, typename ScaleType, bool isConstant,
          bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] aXPlusbYSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  aXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<DataType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<DataType, PTR_ALIGN64, 8>> B;
  const ScaleType scaleA;
  const ScaleType scaleB;

  bool compute() {
    unsigned limI = size;
    for (unsigned i = 0; i < limI; ++i) {
      A[i] = scaleA * A[i] + scaleB * B[i];
    }
    return true;
  }
};

#define DEF_AXPLUSBY_SUPER_VERTEX(SCALE_DEF, PTR, CONSTRAINTS, IS_CONSTANT,    \
                                  IS_CONSTRAINED)                              \
  template <typename DataType>                                                 \
  class CONSTRAINTS                                                            \
      aXPlusbYSupervisor<DataType, DataType, IS_CONSTANT, IS_CONSTRAINED>      \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    aXPlusbYSupervisor();                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<DataType, PTR_ALIGN64, 8>> A;                                 \
    unsigned short size;                                                       \
    Input<Vector<DataType, PTR_ALIGN64, 8>> B;                                 \
    SCALE_DEF scaleA;                                                          \
    SCALE_DEF scaleB;                                                          \
                                                                               \
    bool compute() {                                                           \
      unsigned limI = size;                                                    \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        A[i] = scaleA PTR * A[i] + scaleB PTR * B[i];                          \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXPLUSBY_SUPER_VERTEX(const DataType, ,
                          [[poplar::constraint("elem(*A) != elem(*B)")]], true,
                          true)
DEF_AXPLUSBY_SUPER_VERTEX(InputScaleType<DataType>, [0],
                          [[poplar::constraint("elem(*A) != elem(*B)")]], false,
                          true)
DEF_AXPLUSBY_SUPER_VERTEX(const DataType, , , true, false)
DEF_AXPLUSBY_SUPER_VERTEX(InputScaleType<DataType>, [0], , false, false)

template class aXPlusbYSupervisor<half, half, true, true>;
template class aXPlusbYSupervisor<half, half, false, true>;
template class aXPlusbYSupervisor<half, half, true, false>;
template class aXPlusbYSupervisor<half, half, false, false>;

// This is for the vertex having data=HALF; scale values=FLOAT. This vertex
// has an extra 'tolerance' field, and extra code to check the accuracy of
// the scale values
#define DEF_AXPLUSBY_MIXED_SUPER(SCALE_DEF, PTR, CONSTRAINTS, IS_CONSTANT,     \
                                 IS_CONSTRAINED)                               \
  template <>                                                                  \
  class CONSTRAINTS                                                            \
      aXPlusbYSupervisor<half, float, IS_CONSTANT, IS_CONSTRAINED>             \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    aXPlusbYSupervisor();                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<half, PTR_ALIGN64, 8>> A;                                     \
    unsigned short size;                                                       \
    Input<Vector<half, PTR_ALIGN64, 8>> B;                                     \
    SCALE_DEF scaleA;                                                          \
    SCALE_DEF scaleB;                                                          \
    float tolerance;                                                           \
                                                                               \
    bool compute() {                                                           \
      bool castScalesToHalf =                                                  \
          !IS_CONSTANT && !checkAccuracyWhenCastFloatV2ToHalf(                 \
                              scaleA PTR, scaleB PTR, tolerance);              \
      unsigned limI = size;                                                    \
      if (castScalesToHalf) {                                                  \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          A[i] = static_cast<half>(scaleA PTR) * A[i] +                        \
                 static_cast<half>(scaleB PTR) * B[i];                         \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          A[i] = scaleA PTR * static_cast<float>(A[i]) +                       \
                 scaleB PTR * static_cast<float>(B[i]);                        \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXPLUSBY_MIXED_SUPER(const float, , , true, false)
DEF_AXPLUSBY_MIXED_SUPER(InputScaleType<float>, [0],
                         [[poplar::constraint("elem(*A) != elem(*B)")]], false,
                         true)
DEF_AXPLUSBY_MIXED_SUPER(InputScaleType<float>, [0], , false, false)

template <typename DataType, typename ScaleType, bool isConstant,
          bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] aXPlusbY2D
    : public Vertex {
public:
  aXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  const ScaleType scaleA;
  const ScaleType scaleB;

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
#define DEF_AXPLUSBY_2D_VERTEX(SCALE_DEF, PTR, CONSTRAINTS, IS_CONSTANT,       \
                               IS_CONSTRAINED)                                 \
  template <typename DataType>                                                 \
  class CONSTRAINTS                                                            \
      aXPlusbY2D<DataType, DataType, IS_CONSTANT, IS_CONSTRAINED>              \
      : public Vertex {                                                        \
  public:                                                                      \
    aXPlusbY2D();                                                              \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<DataType> A;                                                  \
    InputBType2D<DataType> B;                                                  \
    SCALE_DEF scaleA;                                                          \
    SCALE_DEF scaleB;                                                          \
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

DEF_AXPLUSBY_2D_VERTEX(const DataType, ,
                       [[poplar::constraint("elem(**A) != elem(**B)")]], true,
                       true)
DEF_AXPLUSBY_2D_VERTEX(Input<DataType>, *,
                       [[poplar::constraint("elem(**A) != elem(**B)")]], false,
                       true)
DEF_AXPLUSBY_2D_VERTEX(const DataType, , , true, false)
DEF_AXPLUSBY_2D_VERTEX(Input<DataType>, *, , false, false)

template class aXPlusbY2D<half, half, true, true>;
template class aXPlusbY2D<half, half, false, true>;
template class aXPlusbY2D<half, half, true, false>;
template class aXPlusbY2D<half, half, false, false>;

// This is for the vertex having data=HALF; scale values=FLOAT. This vertex
// has an extra 'tolerance' field, and extra code to check the accuracy of
// the scale values
#define DEF_AXPLUSBY_2D_MIXED(SCALE_DEF, PTR, CONSTRAINTS, IS_CONSTANT,        \
                              IS_CONSTRAINED)                                  \
  template <>                                                                  \
  class CONSTRAINTS aXPlusbY2D<half, float, IS_CONSTANT, IS_CONSTRAINED>       \
      : public Vertex {                                                        \
  public:                                                                      \
    aXPlusbY2D();                                                              \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<half> A;                                                      \
    InputBType2D<half> B;                                                      \
    SCALE_DEF scaleA;                                                          \
    SCALE_DEF scaleB;                                                          \
    float tolerance;                                                           \
                                                                               \
    bool compute() {                                                           \
      bool castScalesToHalf =                                                  \
          !IS_CONSTANT && !checkAccuracyWhenCastFloatV2ToHalf(                 \
                              PTR scaleA, PTR scaleB, tolerance);              \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        if (castScalesToHalf) {                                                \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = static_cast<half>(PTR scaleA) * refOut[j] +            \
                        static_cast<half>(PTR scaleB) * refIn[j];              \
          }                                                                    \
        } else {                                                               \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = PTR scaleA * static_cast<float>(refOut[j]) +           \
                        PTR scaleB * static_cast<float>(refIn[j]);             \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXPLUSBY_2D_MIXED(const float, , , true, false)
DEF_AXPLUSBY_2D_MIXED(Input<float>, *,
                      [[poplar::constraint("elem(**A) != elem(**B)")]], false,
                      true)
DEF_AXPLUSBY_2D_MIXED(Input<float>, *, , false, false)

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

template <typename DataType, typename ScaleType, bool isConstant,
          bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] aXMinusbYSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  aXMinusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<DataType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<DataType, PTR_ALIGN64, 8>> B;
  const ScaleType scaleA;
  const ScaleType scaleB;

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
  template <typename DataType>                                                 \
  class CONSTRAINTS                                                            \
      aXMinusbYSupervisor<DataType, DataType, IS_CONSTANT, IS_CONSTRAINED>     \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    aXMinusbYSupervisor();                                                     \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<DataType, PTR_ALIGN64, 8>> A;                                 \
    unsigned short size;                                                       \
    Input<Vector<DataType, PTR_ALIGN64, 8>> B;                                 \
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

DEF_AXMINUSBY_SUPER_VERTEX(InputScaleType<DataType>, [0],
                           [[poplar::constraint("elem(*A) != elem(*B)")]],
                           false, true)
DEF_AXMINUSBY_SUPER_VERTEX(InputScaleType<DataType>, [0], , false, false)

template class aXMinusbYSupervisor<half, half, false, true>;
template class aXMinusbYSupervisor<half, half, false, false>;

#define DEF_AXMINUSBY_MIXED_SUPER_VERTEX(SCALE_TYPE, PTR, CONSTRAINTS,         \
                                         IS_CONSTANT, IS_CONSTRAINED)          \
  template <>                                                                  \
  class CONSTRAINTS                                                            \
      aXMinusbYSupervisor<half, float, IS_CONSTANT, IS_CONSTRAINED>            \
      : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {                      \
  public:                                                                      \
    aXMinusbYSupervisor();                                                     \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<half, PTR_ALIGN64, 8>> A;                                     \
    unsigned short size;                                                       \
    Input<Vector<half, PTR_ALIGN64, 8>> B;                                     \
    SCALE_TYPE scaleA;                                                         \
    SCALE_TYPE scaleB;                                                         \
    float tolerance;                                                           \
                                                                               \
    bool compute() {                                                           \
      bool castScalesToHalf =                                                  \
          !IS_CONSTANT && !checkAccuracyWhenCastFloatV2ToHalf(                 \
                              scaleA PTR, scaleB PTR, tolerance);              \
                                                                               \
      unsigned limI = size;                                                    \
      if (castScalesToHalf) {                                                  \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          A[i] = static_cast<half>(scaleA PTR) * A[i] -                        \
                 static_cast<half>(scaleB PTR) * B[i];                         \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i) {                                  \
          A[i] = scaleA PTR * static_cast<float>(A[i]) -                       \
                 scaleB PTR * static_cast<float>(B[i]);                        \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXMINUSBY_MIXED_SUPER_VERTEX(InputScaleType<float>, [0],
                                 [[poplar::constraint("elem(*A) != elem(*B)")]],
                                 false, true)
DEF_AXMINUSBY_MIXED_SUPER_VERTEX(InputScaleType<float>, [0], , false, false)

template <typename DataType, typename ScaleType, bool isConstant,
          bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] aXMinusbY2D
    : public Vertex {
public:
  aXMinusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  const ScaleType scaleA;
  const ScaleType scaleB;

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
  template <typename DataType>                                                 \
  class CONSTRAINTS                                                            \
      aXMinusbY2D<DataType, DataType, IS_CONSTANT, IS_CONSTRAINED>             \
      : public Vertex {                                                        \
  public:                                                                      \
    aXMinusbY2D();                                                             \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<DataType> A;                                                  \
    InputBType2D<DataType> B;                                                  \
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

DEF_AXMINUSBY_2D_VERTEX(Input<DataType>, *,
                        [[poplar::constraint("elem(**A) != elem(**B)")]], false,
                        true)
DEF_AXMINUSBY_2D_VERTEX(Input<DataType>, *, , false, false)

template class aXMinusbY2D<half, half, false, true>;
template class aXMinusbY2D<half, half, false, false>;

// This is for the vertex having data=HALF; scale values=FLOAT. This vertex
// has an extra 'tolerance' field, and extra code to check the accuracy of
// the scale values.
#define DEF_AXMINUSBY_2D_MIXED_VERTEX(SCALE_DEF, PTR, CONSTRAINTS,             \
                                      IS_CONSTANT, IS_CONSTRAINED)             \
  template <>                                                                  \
  class CONSTRAINTS aXMinusbY2D<half, float, IS_CONSTANT, IS_CONSTRAINED>      \
      : public Vertex {                                                        \
  public:                                                                      \
    aXMinusbY2D();                                                             \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<half> A;                                                      \
    InputBType2D<half> B;                                                      \
    SCALE_DEF scaleA;                                                          \
    SCALE_DEF scaleB;                                                          \
    float tolerance;                                                           \
                                                                               \
    bool compute() {                                                           \
      bool castScalesToHalf =                                                  \
          !IS_CONSTANT && !checkAccuracyWhenCastFloatV2ToHalf(                 \
                              PTR scaleA, PTR scaleB, tolerance);              \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        if (castScalesToHalf) {                                                \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = static_cast<half>(PTR scaleA) * refOut[j] -            \
                        static_cast<half>(PTR scaleB) * refIn[j];              \
          }                                                                    \
        } else {                                                               \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = PTR scaleA * static_cast<float>(refOut[j]) -           \
                        PTR scaleB * static_cast<float>(refIn[j]);             \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      return true;                                                             \
    }                                                                          \
  };

DEF_AXMINUSBY_2D_MIXED_VERTEX(Input<float>, *,
                              [[poplar::constraint("elem(**A) != elem(**B)")]],
                              false, true)
DEF_AXMINUSBY_2D_MIXED_VERTEX(Input<float>, *, , false, false)

template <typename InType, bool isConstant, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] XMinusaXPlusbYSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  XMinusaXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(false);

  InOut<Vector<InType, PTR_ALIGN64, 8>> A;
  unsigned short size;
  Input<Vector<InType, PTR_ALIGN64, 8>> B;
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
    InOut<Vector<InType, PTR_ALIGN64, 8>> A;                                   \
    unsigned short size;                                                       \
    Input<Vector<InType, PTR_ALIGN64, 8>> B;                                   \
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
