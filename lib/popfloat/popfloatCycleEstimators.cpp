// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popfloatCycleEstimators.hpp"
#include "codelets/popfloatCycleCount.hpp"
#include <cassert>
#include <cmath>
#include <map>
#include <popfloat/experimental/GfloatExprUtil.hpp>

using namespace poplar;

namespace popfloat {
namespace experimental {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(PackedGfloatParams)(const VertexIntrospector &vertex,
                                              const Target &target) {
  return (1);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastGf8ToHalfSupervisor)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastGf16ToFloatSupervisor)(
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16Supervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType, bool nanoo,
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16SrSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType, bool nanoo,
    SRDensityType srDensity) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16InPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, bool nanoo, RoundType rMode) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16SrInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, bool nanoo, SRDensityType srDensity) {
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastGf8ToFloatSupervisor)(
    const VertexIntrospector &vertex, const Target &target) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastFloatToGf8Supervisor)(
    const VertexIntrospector &vertex, const Target &target) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastHalfToGf8Supervisor)(
    const VertexIntrospector &vertex, const Target &target, FormatType frmt) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastFloatToGf16Supervisor)(
    const VertexIntrospector &vertex, const Target &target, FormatType frmt) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32Supervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType, bool nanoo,
    RoundType rMode) {
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

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32SrSupervisor)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &inputType, const Type &outputType, bool nanoo,
    SRDensityType srDensity) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32InPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, bool nanoo,
    RoundType rMode) {
  return 1;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32SrInPlaceSupervisor)(
    const VertexIntrospector &vertex, const Target &target, bool nanoo,
    SRDensityType srDensity) {
  return 1;
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Supervisor,
                            HALF, HALF, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16InPlaceSupervisor, HALF, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Supervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, true,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, true,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, true,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, true,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, true,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, true,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, true,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, false,
                            popfloat::experimental::RoundType::RZ),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, false,
                            popfloat::experimental::RoundType::RA),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, false,
                            popfloat::experimental::RoundType::RN),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, false,
                            popfloat::experimental::RoundType::RU),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, false,
                            popfloat::experimental::RoundType::RD),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, false,
                            popfloat::experimental::RoundType::SR),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32InPlaceSupervisor, false,
                            popfloat::experimental::RoundType::SX),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, false,
          popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, false,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16SrSupervisor,
                            HALF, HALF, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, false,
          popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, true,
          popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, false,
          popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrSupervisor, HALF, HALF, false,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, FLOAT, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          true, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          false, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat16SrInPlaceSupervisor, HALF, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF, true,
          popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF,
          false, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat16SrInPlaceSupervisor, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, FLOAT, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          true, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, FLOAT,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32SrSupervisor,
                            FLOAT, HALF, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrSupervisor, FLOAT, HALF,
          false, popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, true,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, false,
                            popfloat::experimental::SRDensityType::UNIFORM),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, true,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, false,
                            popfloat::experimental::SRDensityType::NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, true,
          popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, false,
          popfloat::experimental::SRDensityType::TRUNCATED_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, true,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, false,
                            popfloat::experimental::SRDensityType::BERNOULLI),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, true,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, false,
                            popfloat::experimental::SRDensityType::LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, false,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGISTIC),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, true,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental,
                            CastToGfloat32SrInPlaceSupervisor, false,
                            popfloat::experimental::SRDensityType::LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, false,
          popfloat::experimental::SRDensityType::TRUNCATED_LAPLACE),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, true,
          popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, false,
          popfloat::experimental::SRDensityType::LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, true,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),
      CYCLE_ESTIMATOR_ENTRY(
          popfloat::experimental, CastToGfloat32SrInPlaceSupervisor, false,
          popfloat::experimental::SRDensityType::TRUNCATED_LOGIT_NORMAL),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastGf16ToFloatSupervisor,
                            FormatType::BFLOAT16),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastGf16ToFloatSupervisor,
                            FormatType::NO_DENORM_GF16),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastGf16ToFloatSupervisor,
                            FormatType::ENABLE_DENORM_GF16),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastFloatToGf16Supervisor,
                            FormatType::BFLOAT16),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastFloatToGf16Supervisor,
                            FormatType::NO_DENORM_GF16),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastFloatToGf16Supervisor,
                            FormatType::ENABLE_DENORM_GF16),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastGf8ToHalfSupervisor,
                            FormatType::MIN_NORM_ALIGN_GF8),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastGf8ToHalfSupervisor,
                            FormatType::ONE_FIVE_TWO_GF8),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastGf8ToHalfSupervisor,
                            FormatType::MAX_NORM_ALIGN_GF8),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastHalfToGf8Supervisor,
                            FormatType::MIN_NORM_ALIGN_GF8),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastHalfToGf8Supervisor,
                            FormatType::ONE_FIVE_TWO_GF8),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastHalfToGf8Supervisor,
                            FormatType::MAX_NORM_ALIGN_GF8),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastFloatToGf8Supervisor),
      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastGf8ToFloatSupervisor),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat16Param),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, CastToGfloat32Param),

      CYCLE_ESTIMATOR_ENTRY(popfloat::experimental, PackedGfloatParams),
  };
  return table;
};

} // end namespace experimental
} // end namespace popfloat
