// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <math.h>

#include "CheckAccuracyWhenCast.hpp"
#include "elementwiseCodelets.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "popops/ExprOp.hpp"

#ifdef VECTOR_AVAIL_SHORT_SPAN
static constexpr auto SPAN_TYPE = poplar::VectorLayout::SHORT_SPAN;
#else
static constexpr auto SPAN_TYPE = poplar::VectorLayout::SPAN;
#endif

#ifdef VECTOR_AVAIL_SCALED_PTR128
static constexpr auto PTR_ALIGN128 = poplar::VectorLayout::SCALED_PTR128;
#else
static constexpr auto PTR_ALIGN128 = poplar::VectorLayout::ONE_PTR;
#endif

using namespace poplar;

namespace popops {

template <typename AType>
using InputScaleType = Input<Vector<AType, PTR_ALIGN128, 16>>;

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
    Vector<Input<Vector<BType, ONE_PTR, 8>>, ONE_PTR>>;

template <typename AType, typename BType, typename ScaleType>
using ComputeType =
    std::conditional_t<((std::is_same<float, AType>::value ||
                         std::is_same<float, BType>::value ||
                         std::is_same<float, ScaleType>::value)),
                       float, AType>;

template <typename AType, typename BType, typename ScaleType,
          bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] ScaledAddSupervisor
    : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  ScaledAddSupervisor();
  using ComputeType = ComputeType<AType, BType, ScaleType>;

  IS_EXTERNAL_CODELET(!std::is_integral_v<AType>);

  InOut<Vector<AType, ONE_PTR, 8, memConstraints>> A;
  Input<Vector<BType, ONE_PTR, 8>> B;
  InputScaleType<ScaleType> scaleB;
  unsigned short size;

  void compute(unsigned wid) {
    unsigned idx = wid;
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         ++i, idx += CTXT_WORKERS) {
      A[idx] += static_cast<AType>(static_cast<ComputeType>(scaleB[0]) *
                                   static_cast<ComputeType>(B[idx]));
    }
  }
};

template <typename AType, typename BType, typename ScaleType>
class ScaledAddSupervisor<AType, BType, ScaleType, false> : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  ScaledAddSupervisor();
  using ComputeType = ComputeType<AType, BType, ScaleType>;

  IS_EXTERNAL_CODELET(!std::is_integral_v<AType>);

  InOut<Vector<AType, ONE_PTR, 8, false>> A;
  Input<Vector<BType, ONE_PTR, 8>> B;
  InputScaleType<ScaleType> scaleB;
  unsigned short size;

  void compute(unsigned wid) {
    unsigned idx = wid;
    // Where the vertex is used on IPU (for integral types) loading scale
    // avoids re-reading every loop
    const auto scale = static_cast<ComputeType>(scaleB[0]);
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         i++, idx += CTXT_WORKERS) {
      A[idx] += static_cast<AType>(scale * static_cast<ComputeType>(B[idx]));
    }
  }
};

template class ScaledAddSupervisor<float, float, float, true>;
template class ScaledAddSupervisor<float, float, float, false>;
template class ScaledAddSupervisor<half, half, half, true>;
template class ScaledAddSupervisor<half, half, half, false>;

template class ScaledAddSupervisor<float, half, half, false>;
template class ScaledAddSupervisor<float, half, float, false>;

template class ScaledAddSupervisor<half, float, half, false>;
template class ScaledAddSupervisor<half, float, float, false>;

#define DEF_SCALED_ADD_FLOAT_SCALE_SUPER_VERTEX(CONSTRAINTS, IS_CONSTRAINED)   \
  template <>                                                                  \
  class CONSTRAINTS ScaledAddSupervisor<half, half, float, IS_CONSTRAINED>     \
      : public MultiVertex {                                                   \
    static const bool needsAlignWorkers = false;                               \
                                                                               \
  public:                                                                      \
    ScaledAddSupervisor();                                                     \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<half, ONE_PTR, 8, IS_CONSTRAINED>> A;                         \
    Input<Vector<half, ONE_PTR, 8>> B;                                         \
    InputScaleType<float> scaleB;                                              \
    unsigned short size;                                                       \
    const float tolerance;                                                     \
                                                                               \
    void compute(unsigned wid) {                                               \
      unsigned limI = divideWork(size, 0, wid);                                \
      unsigned idx = wid;                                                      \
      if (checkAccuracyWhenCastComputeImpl<float, half>(scaleB[0],             \
                                                        tolerance)) {          \
        const auto halfScale = static_cast<half>(scaleB[0]);                   \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] += halfScale * B[idx];                                        \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] += static_cast<half>(scaleB[0] * static_cast<float>(B[idx])); \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

DEF_SCALED_ADD_FLOAT_SCALE_SUPER_VERTEX(
    [[poplar::constraint("elem(*A) != elem(*B)")]], true)
DEF_SCALED_ADD_FLOAT_SCALE_SUPER_VERTEX(, false)

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledAddSupervisor<int, int, int, false>;
template class ScaledAddSupervisor<unsigned, unsigned, unsigned, false>;

template <typename AType, typename BType, typename ScaleType,
          bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] ScaledAdd2D
    : public Vertex {
public:
  ScaledAdd2D();

  using ComputeType = ComputeType<AType, BType, ScaleType>;
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<AType> A;
  InputBType2D<BType> B;
  Input<ScaleType> scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] += static_cast<AType>(static_cast<ComputeType>(*scaleB) *
                                        static_cast<ComputeType>(refIn[j]));
      }
    }
  }
};

template <typename AType, typename BType, typename ScaleType>
class ScaledAdd2D<AType, BType, ScaleType, false> : public Vertex {
public:
  ScaledAdd2D();
  using ComputeType = ComputeType<AType, BType, ScaleType>;
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<AType> A;
  InputBType2D<BType> B;
  Input<ScaleType> scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] += static_cast<AType>(static_cast<ComputeType>(*scaleB) *
                                        static_cast<ComputeType>(refIn[j]));
      }
    }
  }
};

template class ScaledAdd2D<float, float, float, true>;
template class ScaledAdd2D<float, float, float, false>;

template class ScaledAdd2D<half, half, half, true>;
template class ScaledAdd2D<half, half, half, false>;

template class ScaledAdd2D<half, float, float, false>;
template class ScaledAdd2D<half, float, half, false>;

template class ScaledAdd2D<float, half, half, false>;
template class ScaledAdd2D<float, half, float, false>;

#define DEF_SCALED_ADD_FLOAT_SCALE_2D_VERTEX(CONSTRAINTS, IS_CONSTRAINED)      \
  template <>                                                                  \
  class CONSTRAINTS ScaledAdd2D<half, half, float, IS_CONSTRAINED>             \
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
    void compute() {                                                           \
      unsigned limI = A.size();                                                \
      if (checkAccuracyWhenCastComputeImpl<float, half>(scaleB, tolerance)) {  \
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
    }                                                                          \
  };

DEF_SCALED_ADD_FLOAT_SCALE_2D_VERTEX(
    [[poplar::constraint("elem(**A) != elem(**B)")]], true)
DEF_SCALED_ADD_FLOAT_SCALE_2D_VERTEX(, false)

// No memory constraints for integral versions as the code doesn't make
// use of it
template class ScaledAdd2D<int, int, int, false>;
template class ScaledAdd2D<unsigned, unsigned, unsigned, false>;

template <typename AType, typename BType, typename ScaleType,
          bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] ScaledSubtractSupervisor
    : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  IS_EXTERNAL_CODELET(!is_integral_v<AType>);

  InOut<Vector<AType, ONE_PTR, 8, memConstraints>> A;
  Input<Vector<BType, ONE_PTR, 8>> B;
  InputScaleType<ScaleType> scaleB;
  unsigned short size;

  void compute(unsigned wid) {
    unsigned idx = wid;
    // Where the vertex is used on IPU (for integral types) loading scale
    // avoids re-reading every loop
    const auto scale = scaleB[0];
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         i++, idx += CTXT_WORKERS) {
      A[idx] -= scale * static_cast<AType>(B[idx]);
    }
  }
};

template <typename AType, typename BType>
class ScaledSubtractSupervisor<AType, BType, AType, false>
    : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  IS_EXTERNAL_CODELET(!is_integral_v<AType>);

  InOut<Vector<AType, ONE_PTR, 8>> A;
  Input<Vector<BType, ONE_PTR, 8>> B;
  InputScaleType<AType> scaleB;
  unsigned short size;

  void compute(unsigned wid) {
    unsigned idx = wid;
    // Where the vertex is used on IPU (for integral types) loading scale
    // avoids re-reading every loop
    const auto scale = scaleB[0];
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         i++, idx += CTXT_WORKERS) {
      A[idx] -= scale * static_cast<AType>(B[idx]);
    }
  }
};

#define DEF_SCALED_SUB_FLOAT_SCALE_SUPER_VERTEX(CONSTRAINTS, IS_CONSTRAINED)   \
  template <>                                                                  \
  class CONSTRAINTS                                                            \
      ScaledSubtractSupervisor<half, half, float, IS_CONSTRAINED>              \
      : public MultiVertex {                                                   \
    static const bool needsAlignWorkers = false;                               \
                                                                               \
  public:                                                                      \
    ScaledSubtractSupervisor();                                                \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<half, ONE_PTR, 8, IS_CONSTRAINED>> A;                         \
    Input<Vector<half, ONE_PTR, 8>> B;                                         \
    InputScaleType<float> scaleB;                                              \
    unsigned short size;                                                       \
    const float tolerance;                                                     \
                                                                               \
    void compute(unsigned wid) {                                               \
      unsigned limI = divideWork(size, 0, wid);                                \
      unsigned idx = wid;                                                      \
      if (checkAccuracyWhenCastComputeImpl<float, half>(scaleB[0],             \
                                                        tolerance)) {          \
        const auto halfScale = static_cast<half>(scaleB[0]);                   \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] -= halfScale * B[idx];                                        \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] -= static_cast<half>(scaleB[0] * static_cast<float>(B[idx])); \
        }                                                                      \
      }                                                                        \
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

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] -= *scaleB * refIn[j];
      }
    }
  }
};

template <typename DataType>
class ScaledSubtract2D<DataType, DataType, false> : public Vertex {
public:
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  Input<DataType> scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] -= *scaleB * refIn[j];
      }
    }
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
    void compute() {                                                           \
      unsigned limI = A.size();                                                \
      if (checkAccuracyWhenCastComputeImpl<float, half>(scaleB, tolerance)) {  \
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

template <typename DataType, typename ScaleType, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] aXPlusbYSupervisor
    : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  aXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<DataType, ONE_PTR, 8, memConstraints>> A;
  Input<Vector<DataType, ONE_PTR, 8>> B;
  InputScaleType<DataType> scaleA;
  unsigned short size;
  InputScaleType<DataType> scaleB;

  void compute(unsigned wid) {
    unsigned idx = wid;
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         ++i, idx += CTXT_WORKERS) {
      A[idx] = scaleA[0] * A[idx] + scaleB[0] * B[idx];
    }
  }
};

template <typename DataType>
class aXPlusbYSupervisor<DataType, DataType, false> : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  aXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<DataType, ONE_PTR, 8, false>> A;
  Input<Vector<DataType, ONE_PTR, 8>> B;
  InputScaleType<DataType> scaleA;
  unsigned short size;
  InputScaleType<DataType> scaleB;

  void compute(unsigned wid) {
    unsigned idx = wid;
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         ++i, idx += CTXT_WORKERS) {
      A[idx] = scaleA[0] * A[idx] + scaleB[0] * B[idx];
    }
  }
};

template class aXPlusbYSupervisor<half, half, true>;
template class aXPlusbYSupervisor<half, half, false>;

template class aXPlusbYSupervisor<float, float, false>;

// This is for the vertex having data=HALF; scale values=FLOAT. This vertex
// has an extra 'tolerance' field, and extra code to check the accuracy of
// the scale values
#define DEF_AXPLUSBY_MIXED_SUPER(CONSTRAINTS, IS_CONSTRAINED)                  \
  template <>                                                                  \
  class CONSTRAINTS aXPlusbYSupervisor<half, float, IS_CONSTRAINED>            \
      : public MultiVertex {                                                   \
    static const bool needsAlignWorkers = false;                               \
                                                                               \
  public:                                                                      \
    aXPlusbYSupervisor();                                                      \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<half, ONE_PTR, 8, IS_CONSTRAINED>> A;                         \
    Input<Vector<half, ONE_PTR, 8>> B;                                         \
    InputScaleType<float> scaleA;                                              \
    unsigned short size;                                                       \
    InputScaleType<float> scaleB;                                              \
    float tolerance;                                                           \
                                                                               \
    void compute(unsigned wid) {                                               \
      bool castScalesToHalf = !checkAccuracyWhenCastFloatV2ToHalf(             \
          scaleA[0], scaleB[0], tolerance);                                    \
      unsigned limI = divideWork(size, 0, wid);                                \
      unsigned idx = wid;                                                      \
      if (castScalesToHalf) {                                                  \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] = static_cast<half>(scaleA[0]) * A[idx] +                     \
                   static_cast<half>(scaleB[0]) * B[idx];                      \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] = scaleA[0] * static_cast<float>(A[idx]) +                    \
                   scaleB[0] * static_cast<float>(B[idx]);                     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

DEF_AXPLUSBY_MIXED_SUPER([[poplar::constraint("elem(*A) != elem(*B)")]], true)
DEF_AXPLUSBY_MIXED_SUPER(, false)

template <typename DataType, typename ScaleType, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] aXPlusbY2D
    : public Vertex {
public:
  aXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  Input<DataType> scaleA;
  Input<DataType> scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] = *scaleA * refOut[j] + *scaleB * refIn[j];
      }
    }
  }
};
template <typename DataType>
class aXPlusbY2D<DataType, DataType, false> : public Vertex {
public:
  aXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  Input<DataType> scaleA;
  Input<DataType> scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] = *scaleA * refOut[j] + *scaleB * refIn[j];
      }
    }
  }
};

template class aXPlusbY2D<half, half, true>;
template class aXPlusbY2D<half, half, false>;

template class aXPlusbY2D<float, float, false>;

// This is for the vertex having data=HALF; scale values=FLOAT. This vertex
// has an extra 'tolerance' field, and extra code to check the accuracy of
// the scale values
#define DEF_AXPLUSBY_2D_MIXED(CONSTRAINTS, IS_CONSTRAINED)                     \
  template <>                                                                  \
  class CONSTRAINTS aXPlusbY2D<half, float, IS_CONSTRAINED> : public Vertex {  \
  public:                                                                      \
    aXPlusbY2D();                                                              \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<half> A;                                                      \
    InputBType2D<half> B;                                                      \
    Input<float> scaleA;                                                       \
    Input<float> scaleB;                                                       \
    float tolerance;                                                           \
                                                                               \
    void compute() {                                                           \
      bool castScalesToHalf =                                                  \
          !checkAccuracyWhenCastFloatV2ToHalf(*scaleA, *scaleB, tolerance);    \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        if (castScalesToHalf) {                                                \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = static_cast<half>(*scaleA) * refOut[j] +               \
                        static_cast<half>(*scaleB) * refIn[j];                 \
          }                                                                    \
        } else {                                                               \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = *scaleA * static_cast<float>(refOut[j]) +              \
                        *scaleB * static_cast<float>(refIn[j]);                \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

DEF_AXPLUSBY_2D_MIXED([[poplar::constraint("elem(**A) != elem(**B)")]], true)
DEF_AXPLUSBY_2D_MIXED(, false)

template <typename FPType>
class [[poplar::constraint("elem(**A) != elem(**B)")]] HadamardProd
    : public Vertex {
public:
  Vector<InOut<Vector<FPType>>> A;
  Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR> B;

  void compute() {
    const unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      const unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] *= refIn[j];
      }
    }
  }
};

template <typename DataType, typename ScaleType, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] aXMinusbYSupervisor
    : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  aXMinusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<DataType, ONE_PTR, 8, memConstraints>> A;
  Input<Vector<DataType, ONE_PTR, 8>> B;
  InputScaleType<DataType> scaleA;
  unsigned short size;
  InputScaleType<DataType> scaleB;

  void compute(unsigned wid) {
    unsigned idx = wid;
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         ++i, idx += CTXT_WORKERS) {
      A[idx] = scaleA[0] * A[idx] - scaleB[0] * B[idx];
    }
  }
};

template <typename DataType>
class aXMinusbYSupervisor<DataType, DataType, false> : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  aXMinusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<DataType, ONE_PTR, 8, false>> A;
  Input<Vector<DataType, ONE_PTR, 8>> B;
  InputScaleType<DataType> scaleA;
  unsigned short size;
  InputScaleType<DataType> scaleB;

  void compute(unsigned wid) {
    unsigned idx = wid;
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         ++i, idx += CTXT_WORKERS) {
      A[idx] = scaleA[0] * A[idx] - scaleB[0] * B[idx];
    }
  }
};

template class aXMinusbYSupervisor<half, half, true>;
template class aXMinusbYSupervisor<half, half, false>;

template class aXMinusbYSupervisor<float, float, false>;

#define DEF_AXMINUSBY_MIXED_SUPER_VERTEX(CONSTRAINTS, IS_CONSTRAINED)          \
  template <>                                                                  \
  class CONSTRAINTS aXMinusbYSupervisor<half, float, IS_CONSTRAINED>           \
      : public MultiVertex {                                                   \
    static const bool needsAlignWorkers = false;                               \
                                                                               \
  public:                                                                      \
    aXMinusbYSupervisor();                                                     \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOut<Vector<half, ONE_PTR, 8, IS_CONSTRAINED>> A;                         \
    Input<Vector<half, ONE_PTR, 8>> B;                                         \
    InputScaleType<float> scaleA;                                              \
    unsigned short size;                                                       \
    InputScaleType<float> scaleB;                                              \
    float tolerance;                                                           \
                                                                               \
    void compute(unsigned wid) {                                               \
      bool castScalesToHalf = !checkAccuracyWhenCastFloatV2ToHalf(             \
          scaleA[0], scaleB[0], tolerance);                                    \
                                                                               \
      unsigned limI = divideWork(size, 0, wid);                                \
      unsigned idx = wid;                                                      \
      if (castScalesToHalf) {                                                  \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] = static_cast<half>(scaleA[0]) * A[idx] -                     \
                   static_cast<half>(scaleB[0]) * B[idx];                      \
        }                                                                      \
      } else {                                                                 \
        for (unsigned i = 0; i < limI; ++i, idx += CTXT_WORKERS) {             \
          A[idx] = scaleA[0] * static_cast<float>(A[idx]) -                    \
                   scaleB[0] * static_cast<float>(B[idx]);                     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

DEF_AXMINUSBY_MIXED_SUPER_VERTEX([[poplar::constraint("elem(*A) != elem(*B)")]],
                                 true)
DEF_AXMINUSBY_MIXED_SUPER_VERTEX(, false)

template <typename DataType, typename ScaleType, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] aXMinusbY2D
    : public Vertex {
public:
  aXMinusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<DataType> A;
  InputBType2D<DataType> B;
  const ScaleType scaleA;
  const ScaleType scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] = scaleA * refOut[j] - scaleB * refIn[j];
      }
    }
  }
};
#define DEF_AXMINUSBY_2D_VERTEX(CONSTRAINTS, IS_CONSTRAINED)                   \
  template <typename DataType>                                                 \
  class CONSTRAINTS aXMinusbY2D<DataType, DataType, IS_CONSTRAINED>            \
      : public Vertex {                                                        \
  public:                                                                      \
    aXMinusbY2D();                                                             \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<DataType> A;                                                  \
    InputBType2D<DataType> B;                                                  \
    Input<DataType> scaleA;                                                    \
    Input<DataType> scaleB;                                                    \
                                                                               \
    void compute() {                                                           \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        for (unsigned j = 0; j < limJ; ++j) {                                  \
          refOut[j] = *scaleA * refOut[j] - *scaleB * refIn[j];                \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

DEF_AXMINUSBY_2D_VERTEX([[poplar::constraint("elem(**A) != elem(**B)")]], true)
DEF_AXMINUSBY_2D_VERTEX(, false)

template class aXMinusbY2D<half, half, true>;
template class aXMinusbY2D<half, half, false>;

template class aXMinusbY2D<float, float, false>;

// This is for the vertex having data=HALF; scale values=FLOAT. This vertex
// has an extra 'tolerance' field, and extra code to check the accuracy of
// the scale values.
#define DEF_AXMINUSBY_2D_MIXED_VERTEX(CONSTRAINTS, IS_CONSTRAINED)             \
  template <>                                                                  \
  class CONSTRAINTS aXMinusbY2D<half, float, IS_CONSTRAINED> : public Vertex { \
  public:                                                                      \
    aXMinusbY2D();                                                             \
    IS_EXTERNAL_CODELET(true);                                                 \
                                                                               \
    InOutAType2D<half> A;                                                      \
    InputBType2D<half> B;                                                      \
    Input<float> scaleA;                                                       \
    Input<float> scaleB;                                                       \
    float tolerance;                                                           \
                                                                               \
    void compute() {                                                           \
      bool castScalesToHalf =                                                  \
          !checkAccuracyWhenCastFloatV2ToHalf(*scaleA, *scaleB, tolerance);    \
      unsigned limI = A.size();                                                \
      for (unsigned i = 0; i < limI; ++i) {                                    \
        unsigned limJ = A[i].size();                                           \
        auto const &refIn = B[i];                                              \
        auto &refOut = A[i];                                                   \
        if (castScalesToHalf) {                                                \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = static_cast<half>(*scaleA) * refOut[j] -               \
                        static_cast<half>(*scaleB) * refIn[j];                 \
          }                                                                    \
        } else {                                                               \
          for (unsigned j = 0; j < limJ; ++j) {                                \
            refOut[j] = *scaleA * static_cast<float>(refOut[j]) -              \
                        *scaleB * static_cast<float>(refIn[j]);                \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

DEF_AXMINUSBY_2D_MIXED_VERTEX([[poplar::constraint("elem(**A) != elem(**B)")]],
                              true)
DEF_AXMINUSBY_2D_MIXED_VERTEX(, false)

template <typename InType, bool memConstraints>
class [[poplar::constraint("elem(*A) != elem(*B)")]] XMinusaXPlusbYSupervisor
    : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  XMinusaXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<InType, ONE_PTR, 8>> A;
  Input<Vector<InType, ONE_PTR, 8>> B;
  InputScaleType<InType> scaleA;
  unsigned short size;
  InputScaleType<InType> scaleB;

  void compute(unsigned wid) {
    unsigned idx = wid;
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         ++i, idx += CTXT_WORKERS) {
      A[idx] = A[idx] - scaleA[0] * A[idx] + scaleB[0] * B[idx];
    }
  }
};

template <typename InType>
class XMinusaXPlusbYSupervisor<InType, false> : public MultiVertex {
  static const bool needsAlignWorkers = false;

public:
  XMinusaXPlusbYSupervisor();
  IS_EXTERNAL_CODELET(true);

  InOut<Vector<InType, ONE_PTR, 8>> A;
  Input<Vector<InType, ONE_PTR, 8>> B;
  InputScaleType<InType> scaleA;
  unsigned short size;
  InputScaleType<InType> scaleB;

  void compute(unsigned wid) {
    unsigned idx = wid;
    for (unsigned i = 0; i < divideWork(size, 0, wid);
         ++i, idx += CTXT_WORKERS) {
      A[idx] = A[idx] - scaleA[0] * A[idx] + scaleB[0] * B[idx];
    }
  }
};

template class XMinusaXPlusbYSupervisor<half, true>;
template class XMinusaXPlusbYSupervisor<half, false>;

template <typename InType, bool memConstraints>
class [[poplar::constraint("elem(**A) != elem(**B)")]] XMinusaXPlusbY2D
    : public Vertex {
public:
  XMinusaXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<InType> A;
  InputBType2D<InType> B;
  Input<InType> scaleA;
  Input<InType> scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] = refOut[j] - *scaleA * refOut[j] + *scaleB * refIn[j];
      }
    }
  }
};

template <typename InType>
class XMinusaXPlusbY2D<InType, false> : public Vertex {
public:
  XMinusaXPlusbY2D();
  IS_EXTERNAL_CODELET(true);

  InOutAType2D<InType> A;
  InputBType2D<InType> B;
  Input<InType> scaleA;
  Input<InType> scaleB;

  void compute() {
    unsigned limI = A.size();
    for (unsigned i = 0; i < limI; ++i) {
      unsigned limJ = A[i].size();
      auto const &refIn = B[i];
      auto &refOut = A[i];
      for (unsigned j = 0; j < limJ; ++j) {
        refOut[j] = refOut[j] - *scaleA * refOut[j] + *scaleB * refIn[j];
      }
    }
  }
};

template class XMinusaXPlusbY2D<half, true>;
template class XMinusaXPlusbY2D<half, false>;
} // namespace popops
