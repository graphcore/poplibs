#ifndef GFLOAT_PARAM_UTILS_H
#define GFLOAT_PARAM_UTILS_H

#include "GfloatConst.hpp"

using namespace poplar;
#define ENABLE_PARAM_PRINT

struct castToGfloat16Params {
  static void PrintParams(int  *gf16Param) {
    uint64_t  gf16ExpMask;
    uint64_t  gf16Qnan;
    uint64_t  gf16OutMask;
    uint64_t  gf16ClampF32In;
    uint32_t  gf16ClampF16In;
    uint32_t  gf16ClampOut;
    uint32_t  gf16ScaleFlt;
    uint32_t  gf16ScaleHlf;
    uint32_t  gf16MinOut;
    uint32_t  fp16Pwr2mMan10;
    uint32_t  fp16Pwr2m1;
    uint32_t  fp16Pwr2p10;

    std::memcpy(& gf16ExpMask,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(gf16ExpMask));
    std::memcpy(& gf16Qnan,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_QNAN_OUTPUT_OFFSET],
                sizeof(gf16Qnan));
    std::memcpy(& gf16OutMask,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_NORM_MAN_MASK_OFFSET],
                sizeof(gf16OutMask));

    std::memcpy(& gf16ClampF32In,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET],
                sizeof(gf16ClampF32In));
    std::memcpy(& gf16ClampF16In,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET],
                sizeof(gf16ClampF16In));
    std::memcpy(& gf16ClampOut,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET],
                sizeof(gf16ClampOut));
    std::memcpy(& gf16ScaleFlt,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET],
                sizeof(gf16ScaleFlt));
    std::memcpy(& gf16ScaleHlf,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET + 1],
                sizeof(gf16ScaleHlf));
    std::memcpy(& gf16MinOut,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET],
                sizeof(gf16MinOut));
    std::memcpy(&fp16Pwr2mMan10  ,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M_MAN_10_OFFSET],
                sizeof(fp16Pwr2mMan10) );
    std::memcpy(& fp16Pwr2m1,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M1_OFFSET],
                sizeof(fp16Pwr2m1));
    std::memcpy(& fp16Pwr2p10,
                &gf16Param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_10_OFFSET],
                sizeof(fp16Pwr2p10));

#ifdef ENABLE_PARAM_PRINT
    std::cout << "gf16Param::gf16ExpMask     = " <<
      std::hex << gf16ExpMask     << std::endl;
    std::cout << "gf16Param::gf16Qnan        = " <<
      std::hex << gf16Qnan        << std::endl;
    std::cout << "gf16Param::gf16OutMask     = " <<
      std::hex << gf16OutMask     << std::endl;
    std::cout << "gf16Param::gf16ClampF32In  = " <<
      std::hex << gf16ClampF32In  << std::endl;
    std::cout << "gf16Param::gf16ClampF16In  = " <<
      std::hex << gf16ClampF16In  << std::endl;
    std::cout << "gf16Param::gf16ClampOut    = " <<
      std::hex << gf16ClampOut    << std::endl;
    std::cout << "gf16Param::gf16ScaleFlt    = " <<
      std::hex << gf16ScaleFlt    << std::endl;
    std::cout << "gf16Param::gf16ScaleHlf    = " <<
      std::hex << gf16ScaleHlf    << std::endl;
    std::cout << "gf16Param::gf16MinOut      = " <<
      std::hex << gf16MinOut      << std::endl;
    std::cout << "gf16Param::fp16Pwr2mMan10  = " <<
      std::hex << fp16Pwr2mMan10  << std::endl;
    std::cout << "gf16Param::fp16Pwr2m1      = " <<
      std::hex << fp16Pwr2m1      << std::endl;
    std::cout << "gf16Param::fp16Pwr2p10     = " <<
      std::hex << fp16Pwr2p10     << std::endl;
#endif
  }
};

struct castToGfloat32Params {
  static void PrintParams(int *gf32Param){
    uint64_t     gf32OutMask;
    uint64_t     gf32ExpMask;
    uint64_t     gf32SgnMask;
    uint64_t     gf32Qnan;
    uint64_t     gf32SgnExpMask;
    uint64_t     gf32Bit23Mask;
    uint64_t     gf32ClampOut;
    uint64_t     gf32HalfMin;
    uint32_t     gf32MinValue;
    uint32_t     gf32MinValueBits;
    uint32_t     gf32MinNorm;
    uint32_t     gf32MinNormBits;
    uint32_t     gf32EnDnrm;

    std::memcpy(& gf32OutMask,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_NORM_MANT_MASK_OFFSET],
                sizeof(gf32OutMask));
    std::memcpy(& gf32ExpMask,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(gf32ExpMask));
    std::memcpy(& gf32SgnMask,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET],
                sizeof(gf32SgnMask));
    std::memcpy(& gf32Qnan,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_QNAN_MASK_OFFSET],
                sizeof(gf32Qnan));
    std::memcpy(& gf32SgnExpMask,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_SIGN_EXP_MASK_OFFSET],
                sizeof(gf32SgnExpMask));
    std::memcpy(& gf32Bit23Mask,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_BIT23_MASK_OFFSET],
                sizeof(gf32Bit23Mask));
    std::memcpy(& gf32ClampOut,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET],
                sizeof(gf32ClampOut));
    std::memcpy(& gf32MinValue,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_VALUE_OFFSET],
                sizeof(gf32MinValue));
    std::memcpy(& gf32MinValueBits,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_VALUE_OFFSET],
                sizeof(gf32MinValueBits));
    std::memcpy(& gf32HalfMin,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET],
                sizeof(gf32HalfMin));
    std::memcpy(& gf32MinNorm,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                sizeof(gf32MinNorm));
    std::memcpy(& gf32MinNormBits,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                sizeof(gf32MinNormBits));
    std::memcpy(& gf32EnDnrm,
                &gf32Param[POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET],
                sizeof(gf32EnDnrm));
#ifdef ENABLE_PARAM_PRINT
    std::cout << "gf32Param::gf32OutMask      = " <<
      std::hex << gf32OutMask      << std::endl;
    std::cout << "gf32Param::gf32ExpMask      = " <<
      std::hex << gf32ExpMask      << std::endl;
    std::cout << "gf32Param::gf32SgnMask      = " <<
      std::hex << gf32SgnMask      << std::endl;
    std::cout << "gf32Param::gf32Qnan         = " <<
      std::hex << gf32Qnan         << std::endl;
    std::cout << "gf32Param::gf32SgnExpMask   = " <<
      std::hex << gf32SgnExpMask   << std::endl;
    std::cout << "gf32Param::gf32Bit23Mask    = " <<
      std::hex << gf32Bit23Mask    << std::endl;
    std::cout << "gf32Param::gf32ClampOut     = " <<
      std::hex << gf32ClampOut     << std::endl;
    std::cout << "gf32Param::gf32HalfMin      = " <<
      std::hex << gf32HalfMin      << std::endl;
    std::cout << "gf32Param::gf32MinValue     = " <<
      std::hex << gf32MinValue     << std::endl;
    std::cout << "gf32Param::gf32MinValueBits = " <<
      std::hex << gf32MinValueBits << std::endl;
    std::cout << "gf32Param::gf32MinNorm      = " <<
      std::hex << gf32MinNorm      << std::endl;
    std::cout << "gf32Param::gf32MinNormBits  = " <<
      std::hex << gf32MinNormBits  << std::endl;
    std::cout << "gf32Param::gf32EnDnrm       = " <<
      std::hex << gf32EnDnrm       << std::endl;
#endif
  }
};

struct castGf8ToHalfParams {
  static void PrintParams(int *gf8Param) {
    uint64_t  gf8ExpMask;
    uint64_t  gf8MaxExp;
    uint64_t  gf8InClamp;
    uint32_t  gf8InClampBits;
    int       gf8ShrAlign;
    int       gf8SgnMask;

    std::memcpy(& gf8ExpMask,
                &gf8Param[POPFLOAT_GF8_TO_FP16_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(gf8ExpMask));
    std::memcpy(& gf8MaxExp,
                &gf8Param[POPFLOAT_GF8_TO_FP16_PARAM_MAX_EXPONENT_OFFSET],
                sizeof(gf8MaxExp));
    std::memcpy(& gf8InClamp,
                &gf8Param[POPFLOAT_GF8_TO_FP16_PARAM_CLAMP_INPUT_OFFSET],
                sizeof(gf8InClamp));
    std::memcpy(& gf8InClampBits,
                &gf8InClamp,
                sizeof(gf8InClampBits));
    std::memcpy(& gf8ShrAlign,
                &gf8Param[POPFLOAT_GF8_TO_FP16_PARAM_SHR_ALIGN_OFFSET],
                sizeof(gf8ShrAlign));
    std::memcpy(& gf8SgnMask,
                &gf8Param[POPFLOAT_GF8_TO_FP16_PARAM_SIGN_MASK_OFFSET],
                sizeof(gf8SgnMask));
#ifdef ENABLE_PARAM_PRINT
    std::cout << "gf8ToHalfParam::gf8ExpMask     = " <<
      std::hex << gf8ExpMask     << std::endl;
    std::cout << "gf8ToHalfParam::gf8MaxExp      = " <<
      std::hex << gf8MaxExp      << std::endl;
    std::cout << "gf8ToHalfParam::gf8InClamp     = " <<
      std::hex << gf8InClamp     << std::endl;
    std::cout << "gf8ToHalfParam::gf8InClampBits = " <<
      std::hex << gf8InClampBits << std::endl;
    std::cout << "gf8ToHalfParam::gf8ShrAlign    = " <<
      std::hex << gf8ShrAlign    << std::endl;
    std::cout << "gf8ToHalfParam::gf8SgnMask     = " <<
      std::hex << gf8SgnMask     << std::endl;
#endif
  }
};

struct castHalfToGf8Params {
  static void PrintParams(int   *gf8Param) {
    int       gf8ShrAlign;

    std::memcpy(&gf8ShrAlign,
                &gf8Param[POPFLOAT_FP16_TO_GF8_PARAM_SHR_ALIGN_OFFSET    ],
                sizeof(gf8ShrAlign)   );

#ifdef ENABLE_PARAM_PRINT
    std::cout << "halfToGf8Param::gf8ShrAlign = " <<
      std::hex << gf8ShrAlign << std::endl;
#endif
  }
};

struct castFloatToGf16Params {
  static void PrintParams(int   *gf16Param) {
    uint64_t gf16ExpMask;
    int      gf16ExpAlign;
    int      gf16ExpAlignBits;
    float    gf16MinNorm;
    int      gf16MinNormBits;
    int      gf16Shift;

    std::memcpy(& gf16ExpMask,
                &gf16Param[POPFLOAT_FP32_TO_GF16_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(gf16ExpMask));
    std::memcpy(& gf16ExpAlign,
                &gf16Param[POPFLOAT_FP32_TO_GF16_PARAM_EXP_ALIGN_OFFSET],
                sizeof(gf16ExpAlign));
    std::memcpy(& gf16ExpAlignBits,
                &gf16Param[POPFLOAT_FP32_TO_GF16_PARAM_EXP_ALIGN_OFFSET],
                sizeof(gf16ExpAlignBits));
    std::memcpy(& gf16MinNorm,
                &gf16Param[POPFLOAT_FP32_TO_GF16_PARAM_MIN_NORM_OFFSET],
                sizeof(gf16MinNorm));
    std::memcpy(& gf16MinNormBits,
                &gf16Param[POPFLOAT_FP32_TO_GF16_PARAM_MIN_NORM_OFFSET],
                sizeof(gf16MinNormBits));
    std::memcpy(& gf16Shift,
                &gf16Param[POPFLOAT_FP32_TO_GF16_PARAM_FP16_SHR_ALIGN_OFFSET],
                sizeof(gf16Shift));
#ifdef ENABLE_PARAM_PRINT
    std::cout << "floatToGf16Params::gf16ExpMask      = " <<
      std::hex << gf16ExpMask      << std::endl;
    std::cout << "floatToGf16Params::gf16ExpAlign     = " <<
      std::hex << gf16ExpAlign     << std::endl;
    std::cout << "floatToGf16Params::gf16ExpAlignBits = " <<
      std::hex << gf16ExpAlignBits << std::endl;
    std::cout << "floatToGf16Params::gf16MinNorm      = " <<
      std::hex << gf16MinNorm      << std::endl;
    std::cout << "floatToGf16Params::gf16MinNormBits  = " <<
      std::hex << gf16MinNormBits  << std::endl;
    std::cout << "floatToGf16Params::gf16Shift        = " <<
      std::hex << gf16Shift        << std::endl;
#endif
  }
};

struct castGf16ToFloatParams {
  static void PrintParams(int   *gf16Param) {
    uint64_t  gf16ExpMask;
    uint64_t  gf16Clamp;
    uint64_t  gf16ClampBits;
    float     gf16ExpAlign;
    int       gf16ExpAlignBits;
    float     gf16MinNorm;
    int       gf16MinNormBits;
    int       gf16Shift0;
    int       gf16Shift1;

    std::memcpy(& gf16ExpMask,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_EXP_MASK_OFFSET],
                sizeof(gf16ExpMask));
    std::memcpy(& gf16Clamp,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_CLAMP_OFFSET],
                sizeof(gf16Clamp));
    std::memcpy(& gf16ClampBits,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_CLAMP_OFFSET],
                sizeof(gf16ClampBits));
    std::memcpy(& gf16ExpAlign,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_EXP_ALIGN_OFFSET],
                sizeof(gf16ExpAlign));
    std::memcpy(& gf16ExpAlignBits,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_EXP_ALIGN_OFFSET],
                sizeof(gf16ExpAlignBits));
    std::memcpy(& gf16MinNorm,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_MIN_NORM_OFFSET],
                sizeof(gf16MinNorm));
    std::memcpy(& gf16MinNormBits,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_MIN_NORM_OFFSET],
                sizeof(gf16MinNormBits));
    std::memcpy(& gf16Shift0,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_SHIFT0_OFFSET],
                sizeof(gf16Shift0));
    std::memcpy(& gf16Shift1,
                &gf16Param[POPFLOAT_GF16_TO_FP32_PARAM_SHIFT1_OFFSET],
                sizeof(gf16Shift1));
#ifdef ENABLE_PARAM_PRINT
    std::cout << "gf16ToFloatParams::gf16ExpMask      = " <<
      std::hex << gf16ExpMask      << std::endl;
    std::cout << "gf16ToFloatParams::gf16Clamp        = " <<
      std::hex << gf16Clamp        << std::endl;
    std::cout << "gf16ToFloatParams::gf16ClampBits    = " <<
      std::hex << gf16ClampBits    << std::endl;
    std::cout << "gf16ToFloatParams::gf16ExpAlign     = " <<
      std::dec << gf16ExpAlign     << std::endl;
    std::cout << "gf16ToFloatParams::gf16ExpAlignBits = " <<
      std::hex << gf16ExpAlignBits << std::endl;
    std::cout << "gf16ToFloatParams::gf16MinNorm      = " <<
      std::dec << gf16MinNorm      << std::endl;
    std::cout << "gf16ToFloatParams::gf16MinNormBits  = " <<
      std::hex << gf16MinNormBits  << std::endl;
    std::cout << "gf16ToFloatParams::gf16Shift0       = " <<
      std::hex << gf16Shift0       << std::endl;
    std::cout << "gf16ToFloatParams::gf16Shift1       = " <<
      std::hex << gf16Shift1       << std::endl;
#endif
  }
};
#endif
