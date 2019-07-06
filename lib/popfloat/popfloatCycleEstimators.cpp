#include "popfloatCycleEstimators.hpp"
#include <map>
#include <cassert>
#include <cmath>
#include <popfloat/GfloatExprUtil.hpp>
#include "codelets/popfloatCycleCount.hpp"

using namespace poplar;
using namespace popfloat::gfexpr;

namespace popfloat {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastGf8ToHalfParam)
  (const VertexIntrospector &vertex,
   const Target             &target){
    return (POPFLOAT_CAST_GF8_TO_FP16_PARAM_CALC_CYCLE_COUNT);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastHalfToGf8Param)
  (const VertexIntrospector &vertex,
   const Target             &target){
    return (POPFLOAT_CAST_GF8_TO_FP16_PARAM_CALC_CYCLE_COUNT);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(PackedGfloatParams)
  (const VertexIntrospector &vertex,
   const Target             &target){
    return (1);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastGf8ToHalf)
  (const VertexIntrospector           &vertex,
   const Target                       &target,
   popfloat::gfexpr::GfloatFormatType  frmt){
  //CODELET_VECTOR_VALS(param, CHAR);
  CODELET_FIELD(in);

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles  = 0;

  totalCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_PROLOG;

  if(frmt == popfloat::gfexpr::GfloatFormatType::ONE_FIVE_TWO_GF8){
    iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_FP8_1_5_2_TO_FP16;
  } else {
    iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_NORM_ALIGN_PROLOG;
    if(frmt == popfloat::gfexpr::GfloatFormatType::MAX_NORM_ALIGN_GF8){
      iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_FP8_MAX_NORM_ALIGN;
    } else if(frmt == popfloat::gfexpr::GfloatFormatType::MIN_NORM_ALIGN_GF8){
      iterCycles += POPFLOAT_CAST_GF8_TO_FP16_CYCLE_COUNT_FP8_MIN_NORM_ALIGN;
    }
  }

  iterCycles *=(in.size()+ POPFLOAT_GF16_VEC_SIZE - 1)/ POPFLOAT_GF16_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastGf16ToFloatParam)
  (const VertexIntrospector &vertex,
   const Target             &target){
    return (POPFLOAT_CAST_GF16_TO_FP32_PARAM_CALC_CYCLE_COUNT);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastFloatToGf16Param)
  (const VertexIntrospector &vertex,
   const Target             &target){
    return (POPFLOAT_CAST_GF16_TO_FP32_PARAM_CALC_CYCLE_COUNT);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastGf16ToFloat)
  (const VertexIntrospector           &vertex,
   const Target                       &target,
   popfloat::gfexpr::GfloatFormatType  frmt){
  //CODELET_VECTOR_VALS(param, CHAR);
  CODELET_FIELD(in);

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles  = 0;

  totalCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_PROLOG;

  if(frmt == popfloat::gfexpr::GfloatFormatType::BFLOAT16){
    iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_BF16_TO_FP32;
  } else {
    iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_EN_DENORM_PROLOG;
    if(frmt == popfloat::gfexpr::GfloatFormatType::ENABLE_DENORM_GF16){
      iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_GF16_EN_DENORM;
    } else if(frmt == popfloat::gfexpr::GfloatFormatType::NO_DENORM_GF16){
      iterCycles += POPFLOAT_CAST_GF16_TO_FP32_CYCLE_COUNT_GF16_NO_DENORM;
    }
  }

  iterCycles  *=(in.size()+ POPFLOAT_GF32_VEC_SIZE - 1)/ POPFLOAT_GF32_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16Param)
  (const VertexIntrospector &vertex,
   const Target             &target){
  CODELET_FIELD(gfStruct);

  std::uint64_t totalCycles = 0;
  totalCycles += POPFLOAT_CAST_GFLOAT16_PARAM_CALC_CYCLE_COUNT_PARAM;

  totalCycles += POPFLOAT_CAST_GFLOAT16_PARAM_CALC_CYCLE_COUNT_HALF_PARAMS;

  totalCycles += POPFLOAT_CAST_GFLOAT16_PARAM_CALC_CYCLE_COUNT_SR_MASK;

  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16)
  (const VertexIntrospector           &vertex,
   const Target                       &target,
   const Type                         &inputType,
   const Type                         &outputType,
   const bool                          enNanoo,
   popfloat::gfexpr::GfloatRoundType   rMode){
  CODELET_FIELD(in);

  const bool isFloat   = (inputType  == FLOAT);
  const bool isFP8     = (outputType == CHAR);
  int gf16Class = POPFLOAT_GF16_CLASS_FP16;
  int enDnrm = 1;

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles  = 0;
  if(isFP8){
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_GFLOAT8_OUTPUT;
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SET_SAVE_AS_GFLOAT8;
  } else {
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_GFLOAT16_OUTPUT;
  }
  totalCycles   += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_PROLOG;

  if(rMode != popfloat::gfexpr::GfloatRoundType::RZ){
    totalCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_SET_ROUND_MODE;
  }

  iterCycles   += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_CAST_LOOP;
  if(isFloat){
    if(enNanoo){
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_HALF_SCALE_INPUT;
    } else {
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_HALF_LOAD_CLIP_SCALE;
    }
  } else {
    if(enNanoo){
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_SCALE_INPUT;
    } else {
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_LOAD_CLIP_SCALE;
    }
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_FLOAT_IN_TO_HALF;
  }

  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_LOAD_CLIP_SCALE_END;
  if(enDnrm){
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_CALC_DENORM_MANT_MASK;
  }
  iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ADD_CORRECTION;
  if(enNanoo){
    iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_GEN_NAN_ON_OVERFLOW;
  }

  switch(rMode){
    case popfloat::gfexpr::GfloatRoundType::RZ:
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_ZERO;
      break;
    case popfloat::gfexpr::GfloatRoundType::RN:
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_NEAREST_EVEN;
      break;
    case popfloat::gfexpr::GfloatRoundType::RA:
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_NEAREST_AWAY;
      break;
    case popfloat::gfexpr::GfloatRoundType::RU:
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_POS_INF;
      break;
    case popfloat::gfexpr::GfloatRoundType::RD:
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_ROUND_NEG_INF;
      break;
    case popfloat::gfexpr::GfloatRoundType::SR:
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_STOCHASTIC_ROUND_FULL;
      break;
    case popfloat::gfexpr::GfloatRoundType::SX:
      iterCycles += POPFLOAT_CAST_TO_GF16_CYCLE_COUNT_STOCHASTIC_ROUND_SHORT;
      break;
    case popfloat::gfexpr::GfloatRoundType::INV:
      iterCycles += 0;
      break;
  }

  switch(gf16Class){
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
  }

  iterCycles *=(in.size()+ POPFLOAT_GF16_VEC_SIZE - 1)/ POPFLOAT_GF16_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16Sr)
  (const VertexIntrospector                  &vertex,
   const Target                              &target,
   const Type                                &inputType,
   const Type                                &outputType,
   const bool                                 probNan,
   popfloat::gfexpr::GfloatSRDensityType dist){
  return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16InPlace)
  (const VertexIntrospector           &vertex,
   const Target                       &target,
   const bool                          enNanoo,
   popfloat::gfexpr::GfloatRoundType   rMode){
    return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat16SrInPlace)
  (const VertexIntrospector                  &vertex,
   const Target                              &target,
   const bool                                 probNan,
   popfloat::gfexpr::GfloatSRDensityType dist){
     return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32Param)
  (const VertexIntrospector &vertex,
   const Target             &target){
  CODELET_FIELD(gfStruct);

  int srBits = 0;

  std::uint64_t totalCycles = 0;
  totalCycles += POPFLOAT_CAST_GFLOAT32_PARAM_CALC_CYCLE_COUNT_PARAM;

  if(srBits < POPFLOAT_NUM_FP32_MANTISSA_BITS){
    totalCycles += POPFLOAT_CAST_GFLOAT32_PARAM_CALC_CYCLE_COUNT_SR_MASK;
  }

  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastHalfToGf8)
  (const VertexIntrospector           &vertex,
   const Target                       &target,
   popfloat::gfexpr::GfloatFormatType  frmt){
    return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastFloatToGf16)
  (const VertexIntrospector           &vertex,
   const Target                       &target,
   popfloat::gfexpr::GfloatFormatType  frmt){
    return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32)
  (const VertexIntrospector           &vertex,
   const Target                       &target,
   const Type                         &inputType,
   const Type                         &outputType,
   bool                                enNanoo,
   popfloat::gfexpr::GfloatRoundType   rMode){
  CODELET_FIELD(in);

  const bool isFloatOut = (outputType == FLOAT);

  std::uint64_t totalCycles = 0;
  std::uint64_t iterCycles = 0;

  totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_PROLOG;
  if(!isFloatOut){
    totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SET_SAVE_AS_GFLOAT16;
  }

  if(rMode != popfloat::gfexpr::GfloatRoundType::RZ) {
    totalCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SET_ROUND_MODE;
  }

  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ITER_START;
  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_CALC_DENORM_MASK;

  iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ADD_CORRECTION;

  if(enNanoo){
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_GEN_NAN_ON_OVERFLOAT;
  }

  switch(rMode){
    case popfloat::gfexpr::GfloatRoundType::RZ:
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_ZERO;
      break;
    case popfloat::gfexpr::GfloatRoundType::RN:
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_NEAREST_EVEN;
      break;
    case popfloat::gfexpr::GfloatRoundType::RA:
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_NEAREST_AWAY;
      break;
    case popfloat::gfexpr::GfloatRoundType::RU:
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_POS_INF;
      break;
    case popfloat::gfexpr::GfloatRoundType::RD:
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_ROUND_NEG_INF;
      break;
    case popfloat::gfexpr::GfloatRoundType::SR:
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_STOCHASTIC_ROUND_FULL;
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_HALF_MIN_SR;
      break;
    case popfloat::gfexpr::GfloatRoundType::SX:
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_STOCHASTIC_ROUND_SHORT;
      iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_HALF_MIN_SR;
      break;
    case popfloat::gfexpr::GfloatRoundType::INV:
      iterCycles += 0;
      break;
  }

  if(outputType == FLOAT){
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SAVE_FP32;
  } else {
    iterCycles += POPFLOAT_CAST_TO_GF32_CYCLE_COUNT_SAVE_FP16;
  }

  iterCycles *=(in.size()+ POPFLOAT_GF32_VEC_SIZE - 1)/ POPFLOAT_GF32_VEC_SIZE;
  totalCycles += iterCycles;
  return totalCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32Sr)
  (const VertexIntrospector                  &vertex,
   const Target                              &target,
   const Type                                &inputType,
   const Type                                &outputType,
   const bool                                 enNanoo,
   popfloat::gfexpr::GfloatSRDensityType dist){
  return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32InPlace)
  (const VertexIntrospector          &vertex,
   const Target                      &target,
   bool                               enNanoo,
   popfloat::gfexpr::GfloatRoundType  rMode){
    return 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CastToGfloat32SrInPlace)
  (const VertexIntrospector                  &vertex,
   const Target                              &target,
   const bool                                 enNanoo,
   popfloat::gfexpr::GfloatSRDensityType dist){
    return 1;
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  poplibs::CycleEstimatorTable table = {
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16, HALF, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::SX),


    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, true,
                          ::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, false,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, false,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, false,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, false,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, false,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, false,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, FLOAT, false,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, true,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RZ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RN),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RA),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RU),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::RD),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::SR),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32InPlace, false,
                          popfloat::gfexpr::GfloatRoundType::SX),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF, true,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF, true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF, true,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , true,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , true,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF , false,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, FLOAT, HALF , false,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , false,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Sr, HALF , HALF , false,
               popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, true ,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, true ,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, true ,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, true ,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, true ,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, true ,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, true ,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, false,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16SrInPlace, false,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, FLOAT, true,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, FLOAT, true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, FLOAT, true,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF , true,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF , true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF , true,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),


    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF , false,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF, false,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF, false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Sr, FLOAT, HALF, false,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, true,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, true,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, true,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, true,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, true,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, true,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::UNIFORM),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, false,
                  popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::LAPLACE),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGISTIC),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, false,
                          popfloat::gfexpr::GfloatSRDensityType::LOGIT_NORMAL),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32SrInPlace, false,
                popfloat::gfexpr::GfloatSRDensityType::TRUNCATED_LOGIT_NORMAL),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf16ToFloat,
                          popfloat::gfexpr::GfloatFormatType::BFLOAT16),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf16ToFloat,
                          popfloat::gfexpr::GfloatFormatType::NO_DENORM_GF16),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf16ToFloat,
                popfloat::gfexpr::GfloatFormatType::ENABLE_DENORM_GF16),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastFloatToGf16,
                          popfloat::gfexpr::GfloatFormatType::BFLOAT16),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastFloatToGf16,
                          popfloat::gfexpr::GfloatFormatType::NO_DENORM_GF16),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastFloatToGf16,
                popfloat::gfexpr::GfloatFormatType::ENABLE_DENORM_GF16),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf8ToHalf,
                          popfloat::gfexpr::GfloatFormatType::MIN_NORM_ALIGN_GF8
                          ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf8ToHalf,
                          popfloat::gfexpr::GfloatFormatType::ONE_FIVE_TWO_GF8),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf8ToHalf,
                          popfloat::gfexpr::GfloatFormatType::MAX_NORM_ALIGN_GF8
                          ),


    CYCLE_ESTIMATOR_ENTRY(popfloat, CastHalfToGf8,
                          popfloat::gfexpr::GfloatFormatType::MIN_NORM_ALIGN_GF8
                          ),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastHalfToGf8,
                          popfloat::gfexpr::GfloatFormatType::ONE_FIVE_TWO_GF8),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastHalfToGf8,
                          popfloat::gfexpr::GfloatFormatType::MAX_NORM_ALIGN_GF8
                          ),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat16Param),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastToGfloat32Param),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf16ToFloatParam),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastFloatToGf16Param),

    CYCLE_ESTIMATOR_ENTRY(popfloat, CastGf8ToHalfParam),
    CYCLE_ESTIMATOR_ENTRY(popfloat, CastHalfToGf8Param),

    CYCLE_ESTIMATOR_ENTRY(popfloat, PackedGfloatParams),
  };
  return table;
};

} // end namespace popfloat
