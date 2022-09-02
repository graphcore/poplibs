// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "asm/GfloatConst.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

class CastToGfloat32Param : public Vertex {
public:
  Input<Vector<int, ONE_PTR>> gfStruct;
  Output<Vector<int, SPAN, 8>> param;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  void compute() {
#ifdef __IPU__
    char4 gfPacked;

    std::memcpy(&gfPacked, &gfStruct[0], sizeof(uint32_t));

    uint32_t paramStruct = gfPacked[POPFLOAT_GF_STRUCT_PARAMS_OFFSET];
    unsigned int MANTISSA = gfPacked[POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET];
    unsigned int EXPONENT = gfPacked[POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET];
    int BIAS = gfPacked[POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET];
    bool EN_DENORM =
        ((paramStruct >> POPFLOAT_GF_STRUCT_ENDENORM_BIT_OFFSET) & 0x1) ||
        (EXPONENT == 0);
    bool EN_INF = ((paramStruct >> POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET) & 0x1);

    int32_t minNormExp = (1 - BIAS);
    uint32_t _f32MinNormBits = POPFLOAT_FP32_POWER2(minNormExp);

    float2 gf32HalfMin, fpOutClamp;
    float fpMinNorm, pwr2mMan, gf32Min;
    uintAsVec<float, uint32_t, 1>(&fpMinNorm, _f32MinNormBits);
    uintAsVec<float, uint32_t, 1>(&pwr2mMan, POPFLOAT_FP32_POWER2(-MANTISSA));
    gf32Min = EN_DENORM ? (fpMinNorm * pwr2mMan) : fpMinNorm;
    ;
    gf32HalfMin[0] = (float)gf32Min / (float)2.0;
    gf32HalfMin[1] = (float)gf32Min / (float)2.0;
    int32_t maxExpGf32;
    if (EXPONENT > 0) {
      maxExpGf32 = POPFLOAT_BIT_MASK(EXPONENT) - ((EN_INF ? 1 : 0) + BIAS);
    } else {
      maxExpGf32 = -BIAS;
    }

    float2 zeroV2;
    uintAsVec<float2, uint64_t, 1>(&zeroV2, 0);

    float expMask, sgnMask, sgnExpMask, bit23Mask;
    uintAsVec<float, uint32_t, 1>(&expMask, POPFLOAT_FP32_EXPONENT_MASK);
    uintAsVec<float, uint32_t, 1>(&sgnMask, POPFLOAT_FP32_SIGN_MASK);
    uintAsVec<float, uint32_t, 1>(&sgnExpMask, POPFLOAT_FP32_EXPONENT_MASK |
                                                   POPFLOAT_FP32_SIGN_MASK);
    uintAsVec<float, uint32_t, 1>(&bit23Mask,
                                  (1 << POPFLOAT_NUM_FP32_MANTISSA_BITS));

    float2 expMaskV2, sgnMaskV2, sgnExpMaskV2, bit23MaskV2;
    expMaskV2 = addF32v2(zeroV2, expMask);
    sgnMaskV2 = mulF32v2(zeroV2, sgnMask);
    sgnExpMaskV2 = addF32v2(zeroV2, sgnExpMask);
    bit23MaskV2 = addF32v2(zeroV2, bit23Mask);

    uint32_t outMask;
    outMask = (POPFLOAT_MAN_MASK(MANTISSA)
               << (POPFLOAT_NUM_FP32_MANTISSA_BITS - MANTISSA)) |
              POPFLOAT_FP32_EXPONENT_MASK | POPFLOAT_FP32_SIGN_MASK;

    int32_t outManMaskV2[POPFLOAT_GF32_VEC_SIZE];
    for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
      outManMaskV2[idx] = outMask;
    }

    uint32_t gf32ManMask;
    gf32ManMask = POPFLOAT_MAN_MASK(MANTISSA)
                  << (POPFLOAT_NUM_FP32_MANTISSA_BITS - MANTISSA);
    if (EXPONENT == 0) {
      gf32ManMask = (gf32ManMask << 1) & POPFLOAT_FP32_MANTISSA_MASK;
    }
    float maxValue;
    uintAsVec<float, uint32_t, 1>(&maxValue, POPFLOAT_FP32_POWER2(maxExpGf32) |
                                                 gf32ManMask);
    fpOutClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX] = maxValue;
    fpOutClamp[POPFLOAT_IPU_CLAMP_INDEX_MIN] = -maxValue;

    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_NORM_MANT_MASK_OFFSET],
                &outManMaskV2, sizeof(outManMaskV2));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET],
                &expMaskV2, sizeof(expMaskV2));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET],
                &sgnMaskV2, sizeof(sgnMaskV2));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_SIGN_EXP_MASK_OFFSET],
                &sgnExpMaskV2, sizeof(sgnExpMaskV2));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_BIT23_MASK_OFFSET],
                &bit23MaskV2, sizeof(bit23MaskV2));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET],
                &fpOutClamp, sizeof(fpOutClamp));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_HALF_MIN_OFFSET],
                &gf32HalfMin, sizeof(gf32HalfMin));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_VALUE_OFFSET], &gf32Min,
                sizeof(gf32Min));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET], &fpMinNorm,
                sizeof(fpMinNorm));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET],
                &EN_DENORM, sizeof(EN_DENORM));

    if (1) {
      int32_t minNormExp = (1 - BIAS);
      uint32_t gf16MinNorm = POPFLOAT_FP32_POWER2(minNormExp);
      uint32_t gf16BiasCorr = POPFLOAT_FP32_POWER2(1 - BIAS);

      uint32_t gf16AlignSh0 = EXPONENT + (16 - POPFLOAT_NUM_FP32_EXPONENT_BITS);
      uint32_t gf16AlignSh1 = (16 - POPFLOAT_NUM_FP32_EXPONENT_BITS) - EXPONENT;

      uint32_t gf32ManMask;
      gf32ManMask = POPFLOAT_MAN_MASK(MANTISSA)
                    << (POPFLOAT_NUM_FP32_MANTISSA_BITS - MANTISSA);
      float maxValue;
      if (EXPONENT > 0) {
        int32_t maxNormExp =
            POPFLOAT_BIT_MASK(EXPONENT) - ((EN_INF ? 1 : 0) + BIAS);
        uintAsVec<float, uint32_t, 1>(
            &maxValue, POPFLOAT_FP32_POWER2(maxNormExp) | gf32ManMask);
      } else {
        gf32ManMask = (gf32ManMask << 1) & POPFLOAT_FP32_MANTISSA_MASK;
        int32_t maxDnrmExp = minNormExp - 1;
        uintAsVec<float, uint32_t, 1>(
            &maxValue, POPFLOAT_FP32_POWER2(maxDnrmExp) | gf32ManMask);
      }
      float2 gf16ClampOut;
      gf16ClampOut[POPFLOAT_IPU_CLAMP_INDEX_MAX] = maxValue;
      gf16ClampOut[POPFLOAT_IPU_CLAMP_INDEX_MIN] = -maxValue;

      float sgnMask, expMask;
      uintAsVec<float, uint32_t, 1>(&sgnMask, POPFLOAT_FP32_SIGN_MASK);
      uintAsVec<float, uint32_t, 1>(&expMask, POPFLOAT_FP32_EXPONENT_MASK);

      float2 zeroV2;
      uintAsVec<float2, uint64_t, 1>(&zeroV2, 0);
      float2 sgnMaskV2, expMaskV2;
      expMaskV2 = addF32v2(zeroV2, expMask);
      sgnMaskV2 = mulF32v2(zeroV2, sgnMask);

      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_GF16_EXP_MASK_OFFSET],
                  &expMaskV2, sizeof(expMaskV2));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET],
                  &gf16ClampOut, sizeof(gf16ClampOut));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_EXP_ALIGN_OFFSET],
                  &gf16BiasCorr, sizeof(gf16BiasCorr));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                  &gf16MinNorm, sizeof(gf16MinNorm));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET],
                  &gf16AlignSh0, sizeof(gf16AlignSh0));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT1_OFFSET],
                  &gf16AlignSh1, sizeof(gf16AlignSh1));
    }

    if (1) {
      int32_t minNormExp = (1 - BIAS);
      uint32_t gf16MinNorm = POPFLOAT_FP32_POWER2(minNormExp);
      uint32_t gf16BiasCorr = BIAS << POPFLOAT_NUM_FP32_MANTISSA_BITS;

      uint32_t gf16AlignSh = EXPONENT + (16 - POPFLOAT_NUM_FP32_EXPONENT_BITS);

      float expMask;
      uintAsVec<float, uint32_t, 1>(&expMask, POPFLOAT_FP32_EXPONENT_MASK);

      float2 expMaskV2;
      uintAsVec<float2, uint64_t, 1>(&expMaskV2, 0);
      expMaskV2 = addF32v2(expMaskV2, expMask);

      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET],
                  &expMaskV2, sizeof(expMaskV2));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_PACK_EXP_ALIGN_OFFSET],
                  &gf16BiasCorr, sizeof(gf16BiasCorr));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                  &gf16MinNorm, sizeof(gf16MinNorm));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF32_PARAM_PACK_SHR_ALIGN_OFFSET],
                  &gf16AlignSh, sizeof(gf16AlignSh));
    }
#else
    // Not supported on non-ipu targets
    exit(1);
#endif // defined(__IPU__)
  }
};

} // end namespace experimental
} // end namespace popfloat
