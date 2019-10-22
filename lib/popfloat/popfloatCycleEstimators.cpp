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
    const Type &inputType, const Type &outputType, const bool enNanoo,
    RoundType rMode) {
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

  if (rMode != RoundType::RZ) {
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SET_ROUND_MODE;
  }

  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_CAST_LOOP;
  if (isFloat) {
    if (enNanoo) {
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_HALF_SCALE_INPUT;
    } else {
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_HALF_LOAD_CLIP_SCALE;
    }
  } else {
    if (enNanoo) {
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_SCALE_INPUT;
    } else {
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_LOAD_CLIP_SCALE;
    }
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_IN_TO_HALF;
  }

  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_LOAD_CLIP_SCALE_END;
  if (enDnrm) {
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_CALC_DENORM_MANT_MASK;
  }
  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ADD_CORRECTION;
  if (enNanoo) {
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_GEN_NAN_ON_OVERFLOW;
  }

  switch (rMode) {
  case RoundType::RZ:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_ZERO;
    break;
  case RoundType::RN:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_NEAREST_EVEN;
    break;
  case RoundType::RA:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_NEAREST_AWAY;
    break;
  case RoundType::RU:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_POS_INF;
    break;
  case RoundType::RD:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_NEG_INF;
    break;
  case RoundType::SR:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_STOCHASTIC_ROUND_FULL;
    break;
  case RoundType::SX:
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_STOCHASTIC_ROUND_SHORT;
    break;
  case RoundType::INV:
    iterCycles += 0;
    break;
  }

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
    const Type &inputType, const Type &outputType, const bool probNan,
    SRDensityType dist) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16InPlace)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const bool enNanoo, RoundType rMode) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16SrInPlace)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const bool probNan, SRDensityType dist) {
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
    const Type &inputType, const Type &outputType, bool enNanoo,
    RoundType rMode) {
  CODELET_FIELD(in);

  const bool isFloatOut = (outputType == FLOAT);

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles = 0;

  totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_PROLOG;
  if (!isFloatOut) {
    totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SET_SAVE_AS_GFLOAT16;
  }

  if (rMode != RoundType::RZ) {
    totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SET_ROUND_MODE;
  }

  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ITER_START;
  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_CALC_DENORM_MASK;

  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ADD_CORRECTION;

  if (enNanoo) {
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_GEN_NAN_ON_OVERFLOAT;
  }

  switch (rMode) {
  case RoundType::RZ:
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_ZERO;
    break;
  case RoundType::RN:
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_NEAREST_EVEN;
    break;
  case RoundType::RA:
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_NEAREST_AWAY;
    break;
  case RoundType::RU:
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_POS_INF;
    break;
  case RoundType::RD:
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_NEG_INF;
    break;
  case RoundType::SR:
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_STOCHASTIC_ROUND_FULL;
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_HALF_MIN_SR;
    break;
  case RoundType::SX:
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_STOCHASTIC_ROUND_SHORT;
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_HALF_MIN_SR;
    break;
  case RoundType::INV:
    iterCycles += 0;
    break;
  }

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
    const Type &inputType, const Type &outputType, const bool enNanoo,
    SRDensityType dist) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32InPlace)(
    const VertexIntrospector &vertex, const Target &target, bool enNanoo,
    RoundType rMode) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32SrInPlace)(
    const VertexIntrospector &vertex, const Target &target, const bool enNanoo,
    SRDensityType dist) {
  return 1;
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            false, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            false, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            false, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            false, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            false, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            false, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT, HALF,
                            false, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, false, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, false, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, false, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, false, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, false, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, false, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, FLOAT,
                            FLOAT, false, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            false, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            false, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            false, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            false, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            false, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            false, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16, HALF, HALF,
                            false, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, false, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, false, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, false, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, false, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, false, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, false, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace,
                            FLOAT, false, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            false, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            false, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            false, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            false, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            false, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            false, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16InPlace, HALF,
                            false, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, false, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, false, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, false, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, false, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, false, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, false, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT,
                            FLOAT, false, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32, FLOAT, HALF,
                            true, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace, true,
                            RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace, true,
                            RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace, true,
                            RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace, true,
                            RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace, true,
                            RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace, true,
                            RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace, true,
                            RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace,
                            false, RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace,
                            false, RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace,
                            false, RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace,
                            false, RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace,
                            false, RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace,
                            false, RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32InPlace,
                            false, RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            FLOAT, false,
                            SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16Sr, HALF,
                            HALF, false, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            FLOAT, false,
                            SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat16SrInPlace,
                            HALF, false, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            FLOAT, true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32Sr, FLOAT,
                            HALF, false, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            true, SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(experimental::popfloat, CastToGfloat32SrInPlace,
                            false, SRDensityType::TRUNCATED_LOGIT_NORMAL),

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
