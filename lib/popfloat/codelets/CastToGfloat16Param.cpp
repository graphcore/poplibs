// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "GfloatConst.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <ipudef.h>
#include <poplar/Vertex.hpp>

static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

class CastToGfloat16Param : public Vertex {
public:
  Input<Vector<int, ONE_PTR>> gfStruct;
  Output<Vector<int, SPAN, 8>> param;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
    char4 gfPacked;

    std::memcpy(&gfPacked, &gfStruct[0], sizeof(uint32_t));

    unsigned int MANTISSA = gfPacked[POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET];
    unsigned int EXPONENT = gfPacked[POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET];
    int BIAS = gfPacked[POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET];
    uint32_t gfPackedStruct = gfPacked[POPFLOAT_GF_STRUCT_PARAMS_OFFSET];
    bool EN_DENORM =
        ((gfPackedStruct >> POPFLOAT_GF_STRUCT_ENDENORM_BIT_OFFSET) & 0x1) ||
        (EXPONENT == 0);
    bool EN_INF =
        ((gfPackedStruct >> POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET) & 0x1);

    uint16_t expMask, sgnMask, genNan;
    expMask = POPFLOAT_FP16_EXPONENT_MASK;
    sgnMask = POPFLOAT_FP16_SIGN_MASK;
    genNan = POPFLOAT_FP16_GEN_QNAN;

    uint64_t expMaskV4, sgnMaskV4, srMaskV4, genNanV4;
    expMaskV4 = addF16v4(0, expMask);
    sgnMaskV4 = mulF16v4(0, sgnMask);
    genNanV4 = addF16v4(0, genNan);

    uint16_t outBitsMask = POPFLOAT_MAN_MASK(MANTISSA);
    outBitsMask <<= (POPFLOAT_NUM_FP16_MANTISSA_BITS - MANTISSA);
    outBitsMask =
        outBitsMask | POPFLOAT_FP16_EXPONENT_MASK | POPFLOAT_FP16_SIGN_MASK;
    ushort4 outBitsMaskV4;
    for (int idx = 0; idx != POPFLOAT_GF16_VEC_SIZE; ++idx) {
      outBitsMaskV4[idx] = outBitsMask;
    }

    uint32_t f16Pwr2M1 = 0x3800;
    ushort2 f16Pwr10;
    f16Pwr10[0] = 0x6400;
    f16Pwr10[1] = 0x1400;

    uint32_t pwr2mMan10Bits =
        POPFLOAT_FP32_POWER2(-(POPFLOAT_NUM_FP16_MANTISSA_BITS + MANTISSA));

    float pwr2mMan10;
    uintAsVec<float, uint32_t, 1>(&pwr2mMan10, pwr2mMan10Bits);

    short2 f16Pwr2mMan10;
    poplar::IeeeHalf hlfPwr2mMan10(pwr2mMan10);
    f16Pwr2mMan10[0] = hlfPwr2mMan10.bit16();

    int32_t maxExpInGf16;
    if (EXPONENT > 0) {
      maxExpInGf16 = (POPFLOAT_BIT_MASK(EXPONENT) - ((EN_INF ? 1 : 0) + BIAS));
    } else {
      maxExpInGf16 = (1 - BIAS);
    }

    int32_t scaleInExp;
    if ((EXPONENT == POPFLOAT_NUM_FP16_EXPONENT_BITS) && !EN_INF) {
      scaleInExp = maxExpInGf16 - POPFLOAT_FP16_MAX_EXP;
    } else {
      scaleInExp = BIAS - POPFLOAT_FP16_EXPONENT_BIAS;
    }

    uint32_t scaleInBits;
    scaleInBits = POPFLOAT_FP32_POWER2(scaleInExp);

    float scale;
    uintAsVec<float, uint32_t, 1>(&scale, scaleInBits);

    short2 hfScaleInBits;
    poplar::IeeeHalf hlfScaleIn(scale);
    hfScaleInBits[0] = hlfScaleIn.bit16();

    uint64_t clampIn;
    float maxValue;
    uintAsVec<float, uint32_t, 1>(
        &maxValue, POPFLOAT_FP32_POWER2(maxExpInGf16 - MANTISSA));
    if (EXPONENT > 0) {
      unsigned int MANT_SIZE = (MANTISSA + 1);
      maxValue *= (float)POPFLOAT_BIT_MASK(MANT_SIZE);
    } else {
      maxValue *= (float)POPFLOAT_BIT_MASK(MANTISSA);
    }

    poplar::IeeeHalf hlfScale(scale);

    short2 hfScale;
    hfScale[0] = hlfScale.bit16();
    hfScale[1] = hlfScale.bit16();
    vecAsUInt<short2, uint32_t, 1>(&hfScale, &scaleInBits);

    poplar::IeeeHalf hlfClampIn(maxValue);

    short2 hfClampIn;
    hfClampIn[POPFLOAT_IPU_CLAMP_INDEX_MAX] = hlfClampIn.bit16();
    hfClampIn[POPFLOAT_IPU_CLAMP_INDEX_MIN] =
        hlfClampIn.bit16() | POPFLOAT_FP16_SIGN_MASK;

    vecAsUInt<float, uint32_t, 1>(&scale, &scaleInBits);

    float2 fpClampIn;
    fpClampIn[POPFLOAT_IPU_CLAMP_INDEX_MAX] = maxValue;
    fpClampIn[POPFLOAT_IPU_CLAMP_INDEX_MIN] = -maxValue;

    int32_t minNormExp = (1 - BIAS);
    uint32_t fpMinNormBits = POPFLOAT_FP32_POWER2(minNormExp);

    float fpMinNorm, fpMinValue, fpHalfMinValue, fpMaxValue, pwr2mMan;
    uintAsVec<float, uint32_t, 1>(&fpMinNorm, fpMinNormBits);
    uintAsVec<float, uint32_t, 1>(&pwr2mMan, POPFLOAT_FP32_POWER2(-MANTISSA));

    fpMinValue = scale * (EN_DENORM ? (fpMinNorm * pwr2mMan) : fpMinNorm);
    fpHalfMinValue = (float)fpMinValue / (float)2.0;

    poplar::IeeeHalf hlfMinOut(fpMinValue);
    poplar::IeeeHalf hlfHalfMinOut(fpHalfMinValue);

    short2 fp16MinOut;
    fp16MinOut[0] = hlfMinOut.bit16();
    fp16MinOut[1] = hlfHalfMinOut.bit16();

    uint32_t gf32ManMask = POPFLOAT_MAN_MASK(MANTISSA)
                           << (POPFLOAT_NUM_FP32_MANTISSA_BITS - MANTISSA);

    if (EXPONENT == 0) {
      gf32ManMask = (gf32ManMask << 1) & POPFLOAT_FP32_MANTISSA_MASK;
    }

    int32_t maxExpOutGf16;
    if (EXPONENT > 0) {
      if (EXPONENT == POPFLOAT_NUM_FP16_EXPONENT_BITS) {
        maxExpOutGf16 = POPFLOAT_FP16_MAX_EXP;
      } else {
        maxExpOutGf16 = POPFLOAT_FP16_MIN_NORM + (1 << EXPONENT) - 2 - EN_INF;
      }
    } else {
      maxExpOutGf16 = POPFLOAT_FP16_MIN_NORM - 1;
    }
    uintAsVec<float, uint32_t, 1>(
        &fpMaxValue, POPFLOAT_FP32_POWER2(maxExpOutGf16) | gf32ManMask);

    poplar::IeeeHalf hlfCalmpOut(fpMaxValue);

    short2 gf16CalmpOut;
    gf16CalmpOut[POPFLOAT_IPU_CLAMP_INDEX_MAX] = hlfCalmpOut.bit16();
    gf16CalmpOut[POPFLOAT_IPU_CLAMP_INDEX_MIN] =
        hlfCalmpOut.bit16() | POPFLOAT_FP16_SIGN_MASK;

    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET],
                &expMaskV4, sizeof(expMaskV4));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET],
                &sgnMaskV4, sizeof(sgnMaskV4));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_QNAN_OUTPUT_OFFSET],
                &genNanV4, sizeof(genNanV4));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_NORM_MAN_MASK_OFFSET],
                &outBitsMaskV4, sizeof(outBitsMaskV4));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET],
                &fpClampIn, sizeof(fpClampIn));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET],
                &hfClampIn, sizeof(hfClampIn));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET],
                &gf16CalmpOut, sizeof(gf16CalmpOut));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET],
                &scaleInBits, sizeof(scaleInBits));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET + 1],
                &hfScaleInBits, sizeof(hfScaleInBits));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_MIN_OUTPUT_OFFSET],
                &fp16MinOut, sizeof(fp16MinOut));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M_MAN_10_OFFSET],
                &f16Pwr2mMan10, sizeof(f16Pwr2mMan10));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M1_OFFSET],
                &f16Pwr2M1, sizeof(f16Pwr2M1));
    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_10_OFFSET], &f16Pwr10,
                sizeof(f16Pwr10));

    uint32_t packShrAlign = POPFLOAT_NUM_FP16_MANTISSA_BITS - 7 + EXPONENT;

    std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_PACK_SHR_ALIGN_OFFSET],
                &packShrAlign, sizeof(packShrAlign));

    if (1) {
      uint32_t unpackShrAlign = POPFLOAT_NUM_FP16_MANTISSA_BITS - 7 + EXPONENT;
      uint32_t gf8SgnMask = POPFLOAT_FP8_V4_SIGN_MASK;

      uint16_t expMask, maxExp;
      expMask = POPFLOAT_FP16_EXPONENT_MASK;
      maxExp = 0x7800;

      uint64_t expMaskV4, maxExpV4;
      expMaskV4 = addF16v4(0, expMask);
      maxExpV4 = addF16v4(0, maxExp);

      uint32_t gf32ManMask = POPFLOAT_MAN_MASK(MANTISSA)
                             << (POPFLOAT_NUM_FP32_MANTISSA_BITS - MANTISSA);
      if (EXPONENT == 0) {
        gf32ManMask = (gf32ManMask << 1) & POPFLOAT_FP32_MANTISSA_MASK;
      }

      float fpMaxValue;
      int32_t maxExpOutGf16;
      if (EXPONENT > 0) {
        if (EXPONENT == POPFLOAT_NUM_FP16_EXPONENT_BITS) {
          maxExpOutGf16 = POPFLOAT_FP16_MAX_EXP;
        } else {
          maxExpOutGf16 = POPFLOAT_FP16_MIN_NORM + (1 << EXPONENT) - 2 - EN_INF;
        }
      } else {
        maxExpOutGf16 = POPFLOAT_FP16_MIN_NORM - 1;
      }
      uintAsVec<float, uint32_t, 1>(
          &fpMaxValue, POPFLOAT_FP32_POWER2(maxExpOutGf16) | gf32ManMask);

      poplar::IeeeHalf hlfMaxValue(fpMaxValue);
      short2 hlfClamp;
      hlfClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX] = hlfMaxValue.bit16();
      hlfClamp[POPFLOAT_IPU_CLAMP_INDEX_MIN] =
          hlfMaxValue.bit16() | POPFLOAT_FP16_SIGN_MASK;

      std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET],
                  &expMaskV4, sizeof(expMaskV4));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_MAX_EXPONENT_OFFSET],
                  &maxExpV4, sizeof(maxExpV4));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET],
                  &hlfClamp, sizeof(hlfClamp));
      std::memcpy(&param[POPFLOAT_CAST_TO_GF16_PARAM_UNPACK_SHR_ALIGN_OFFSET],
                  &unpackShrAlign, sizeof(unpackShrAlign));
      std::memcpy(&param[POPFLOAT_CAST_TO_GP16_PARAM_GF8_SIGN_MASK_OFFSET],
                  &gf8SgnMask, sizeof(gf8SgnMask));
    }

    return true;
  }
};

} // end namespace experimental
} // end namespace popfloat
