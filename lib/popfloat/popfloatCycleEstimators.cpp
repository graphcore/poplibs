// Copyright (c) Graphcore Ltd, All rights reserved.
#include "popfloatCycleEstimators.hpp"
#include "codelets/popfloatCycleCount.hpp"
#include <cassert>
#include <cmath>
#include <experimental/popfloat/GfloatExprUtil.hpp>
#include <map>

using namespace poplar;

namespace experimental {
namespace popfloat {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(PackedGfloatParams)(const VertexIntrospector &vertex,
                                              const Target &target) {
  return (1);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastGf8ToHalf)(
    const VertexIntrospector &vertex, const Target &target, FormatType frmt) {
  // CODELET_VECTOR_VALS(param, CHAR);
  CODELET_FIELD(in);

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles = 0;

  totalCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_PROLOG;

  if (frmt == FormatType::ONE_FIVE_TWO_GF8) {
    iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_FP8_1_5_2_TO_FP16;
  } else {
    iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_NORM_ALIGN_PROLOG;
    if (frmt == FormatType::MAX_NORM_ALIGN_GF8) {
      iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_FP8_MAX_NORM_ALIGN;
    } else if (frmt == FormatType::MIN_NORM_ALIGN_GF8) {
      iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_FP8_MIN_NORM_ALIGN;
    }
  }

  iterCycles *=
      (in.size() + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastGf16ToFloat)(
    const VertexIntrospector &vertex, const Target &target, FormatType frmt) {
  // CODELET_VECTOR_VALS(param, CHAR);
  CODELET_FIELD(in);

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles = 0;

  totalCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_PROLOG;

  if (frmt == FormatType::BFLOAT16) {
    iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_BF16_TO_FP32;
  } else {
    iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_EN_DENORM_PROLOG;
    if (frmt == FormatType::ENABLE_DENORM_GF16) {
      iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_GF16_EN_DENORM;
    } else if (frmt == FormatType::NO_DENORM_GF16) {
      iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_GF16_NO_DENORM;
    }
  }

  iterCycles *=
      (in.size() + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16Param)(const VertexIntrospector &vertex,
                                               const Target &target) {
  CODELET_FIELD(gfStruct);

  std::uint64_t totalCycles = 0;
  totalCycles += POPFLOAT_CAST_GFLOAT16_PARAM_CALC_CYCLE_COUNT_PARAM;

  totalCycles += POPFLOAT_CAST_GFLOAT16_PARAM_CALC_CYCLE_COUNT_HALF_PARAMS;

  totalCycles += POPFLOAT_CAST_GFLOAT16_PARAM_CALC_CYCLE_COUNT_SR_MASK;

  return totalCycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType) {
  CODELET_FIELD(in);

  const bool isFloat = (inputType == FLOAT);
  const bool isFP8 = (outputType == CHAR);
  int gf16Class = POPFLOAT_GF16_CLASS_FP16;
  int enDnrm = 1;

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles = 0;
  if (isFP8) {
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_GFLOAT8_OUTPUT;
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SET_SAVE_AS_GFLOAT8;
  } else {
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_GFLOAT16_OUTPUT;
  }
  totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_PROLOG;

  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_CAST_LOOP;
  if (isFloat) {
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_HALF_SCALE_INPUT;
  } else {
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_SCALE_INPUT;
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_IN_TO_HALF;
  }

  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_LOAD_CLIP_SCALE_END;
  if (enDnrm) {
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_CALC_DENORM_MANT_MASK;
  }
  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ADD_CORRECTION;
  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_GEN_NAN_ON_OVERFLOW;

  switch (gf16Class) {
  case POPFLOAT_GF16_CLASS_FP16:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SAVE_FP16;
    break;
  case POPFLOAT_GF16_CLASS_FP8_1_5_2:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SAVE_FP8_1_5_2;
    break;
  case POPFLOAT_GF16_CLASS_FP8_MIN_NORM_ALIGN:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SAVE_FP8_MIN_NORM_ALIGN;
    break;
  case POPFLOAT_GF16_CLASS_FP8_MAX_NORM_ALIGN:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SAVE_FP8_MAX_NORM_ALIGN;
    break;
  default:
    iterCycles += 0;
    break;
  }

  iterCycles *=
      (in.size() + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16Sr)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16InPlace)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16SrInPlace)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType) {
  return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32Param)(const VertexIntrospector &vertex,
                                               const Target &target) {
  CODELET_FIELD(gfStruct);

  int srBits = 0;

  std::uint64_t totalCycles = 0;
  totalCycles += POPFLOAT_CAST_GFLOAT32_PARAM_CALC_CYCLE_COUNT_PARAM;

  if (srBits < POPFLOAT_NUM_FP32_MANTISSA_BITS) {
    totalCycles += POPFLOAT_CAST_GFLOAT32_PARAM_CALC_CYCLE_COUNT_SR_MASK;
  }

  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastGf8ToFloat)(const VertexIntrospector &vertex,
                                          const Target &target) {
  return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastFloatToGf8)(const VertexIntrospector &vertex,
                                          const Target &target) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastHalfToGf8)(
    const VertexIntrospector &vertex, const Target &target, FormatType frmt) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastFloatToGf16)(
    const VertexIntrospector &vertex, const Target &target, FormatType frmt) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType) {
  CODELET_FIELD(in);

  const bool isFloatOut = (outputType == FLOAT);

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles = 0;

  totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_PROLOG;
  if (!isFloatOut) {
    totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SET_SAVE_AS_GFLOAT16;
  }

  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ITER_START;
  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_CALC_DENORM_MASK;

  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ADD_CORRECTION;

  if (outputType == FLOAT) {
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SAVE_FP32;
  } else {
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SAVE_FP16;
  }

  iterCycles *=
      (in.size() + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32Sr)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32InPlace)(
    const VertexIntrospector &vertex, const Target &target) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32SrInPlace)(
    const VertexIntrospector &vertex, const Target &target) {
  return 1;
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastGf16ToFloat,
                            FormatType::BFLOAT16),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastGf16ToFloat,
                            FormatType::NO_DENORM_GF16),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastGf16ToFloat,
                            FormatType::ENABLE_DENORM_GF16),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastFloatToGf16,
                            FormatType::BFLOAT16),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastFloatToGf16,
                            FormatType::NO_DENORM_GF16),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastFloatToGf16,
                            FormatType::ENABLE_DENORM_GF16),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastGf8ToHalf,
                            FormatType::MIN_NORM_ALIGN_GF8),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastGf8ToHalf,
                            FormatType::ONE_FIVE_TWO_GF8),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastGf8ToHalf,
                            FormatType::MAX_NORM_ALIGN_GF8),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastHalfToGf8,
                            FormatType::MIN_NORM_ALIGN_GF8),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastHalfToGf8,
                            FormatType::ONE_FIVE_TWO_GF8),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastHalfToGf8,
                            FormatType::MAX_NORM_ALIGN_GF8),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastFloatToGf8),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastGf8ToFloat),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Param),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Param),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, PackedGfloatParams),
  };
  return table;
};

} // end namespace popfloat
} // end namespace experimental
