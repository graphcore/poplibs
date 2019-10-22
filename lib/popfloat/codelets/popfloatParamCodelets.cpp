#include "GfloatConst.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <ipudef.h>
#include <poplar/Vertex.hpp>

static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

#if defined(__IPU__) && !defined(POPLIBS_DISABLE_ASM_CODELETS)
#define EXTERNAL_CODELET true
#else
#define EXTERNAL_CODELET false
#endif

using namespace poplar;

namespace experimental {
namespace popfloat {

class PackedGfloatParams : public Vertex {
public:
  Output<Vector<int, ONE_PTR>> gfStruct;
  unsigned manBits;
  unsigned expBits;
  int expBias;
  unsigned enDenorm;
  unsigned enInf;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
    uint32_t param = 0;
    param += enDenorm << POPFLOAT_GF_STRUCT_ENDENORM_BIT_OFFSET;
    param += enInf << POPFLOAT_GF_STRUCT_ENINF_BIT_OFFSET;

    uint32_t gfPacked;
    char packed[4];
    packed[POPFLOAT_GF_STRUCT_MANTISSA_SIZE_OFFSET] = manBits;
    packed[POPFLOAT_GF_STRUCT_EXPONENT_SIZE_OFFSET] = expBits;
    packed[POPFLOAT_GF_STRUCT_EXP_BIAS_OFFSET] = expBias;
    packed[POPFLOAT_GF_STRUCT_PARAMS_OFFSET] = param;

    std::memcpy(&gfPacked, &packed, sizeof(gfPacked));
    gfStruct[0] = gfPacked;

    return true;
  }
};

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

class CastToGfloat32Param : public Vertex {
public:
  Input<Vector<int, ONE_PTR>> gfStruct;
  Output<Vector<int, SPAN, 8>> param;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
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
    return true;
  }
};

} // end namespace popfloat
} // end namespace experimental
