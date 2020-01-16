// Copyright (c) Graphcore Ltd, All rights reserved.
#include "popfloatCodelets.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <ipudef.h>
#include <popfloat/experimental/GfloatExpr.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>

static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

class CastToGfloat32SrInPlace : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<InOut<Vector<float, SPAN, 8>>, SPAN> inOut;
  Vector<uint32_t, ONE_PTR, 8> srMask;
  unsigned roundMode;
  bool enNanoo;
  Vector<unsigned, ONE_PTR, 8> corrParams;
  unsigned distParam;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
    uint64_t expMaskV2, sgnMaskV2, nanMaskV2, outManMaskV2, sgnExpMaskV2;
    uint64_t srMaskV2, bit23MaskV2, qnanMaskV2;
    float fpMinNorm, fpMinValue, fpHalfMinValue;
    unsigned int EN_DENORM;
    float2 fpOutClamp;
    nanMaskV2 = enNanoo ? (~0) : 0;
    std::memcpy(&outManMaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_NORM_MANT_MASK_OFFSET],
                sizeof(outManMaskV2));
    std::memcpy(&expMaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(expMaskV2));
    std::memcpy(&sgnMaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_SIGN_MASK_OFFSET],
                sizeof(sgnMaskV2));
    std::memcpy(&qnanMaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_QNAN_MASK_OFFSET],
                sizeof(qnanMaskV2));
    std::memcpy(&sgnExpMaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_SIGN_EXP_MASK_OFFSET],
                sizeof(sgnExpMaskV2));
    std::memcpy(&bit23MaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_BIT23_MASK_OFFSET],
                sizeof(bit23MaskV2));
    std::memcpy(&fpOutClamp,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET],
                sizeof(fpOutClamp));
    std::memcpy(&fpMinValue,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_VALUE_OFFSET],
                sizeof(fpMinValue));
    std::memcpy(&fpHalfMinValue,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_VALUE_OFFSET + 4],
                sizeof(fpHalfMinValue));
    std::memcpy(&fpMinNorm, &param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                sizeof(fpMinNorm));
    std::memcpy(&EN_DENORM,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_EN_DENORM_OFFSET],
                sizeof(EN_DENORM));

    float2 inV2;
    for (unsigned Idx = 0; Idx < inOut.size(); ++Idx) {
      unsigned len = inOut[Idx].size();
      unsigned nv = (len + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;

      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF32_VEC_SIZE) {
        unsigned maxPerCall =
            (len < POPFLOAT_GF32_VEC_SIZE) ? len : POPFLOAT_GF32_VEC_SIZE;
        for (unsigned idx = 0; idx != POPFLOAT_GF32_VEC_SIZE; ++idx) {
          inV2[idx] = 0;
          if (idx < maxPerCall) {
            inV2[idx] = inOut[Idx][POPFLOAT_GF32_VEC_SIZE * j + idx];
          }
        }

        uint64_t inValueV2 = 0;
        vecAsUInt<float2, uint64_t, 1>(&inV2, &inValueV2);

        float2 tmpV2;
        uint64_t isNanOrInf, nanValue;
        isNanOrInf = (~inValueV2) & expMaskV2;
        float2 zeroVec;
        uintAsVec<float2, uint64_t, 1>(&zeroVec, 0);
        uintAsVec<float2, uint64_t, 1>(&tmpV2, isNanOrInf);
        compareF32v2Eq(tmpV2, zeroVec, &isNanOrInf);
        uint64_t sgnV2, outValueV2, expV2;

        isNanOrInf = nanMaskV2 & isNanOrInf;
        nanValue = isNanOrInf & inValueV2;
        sgnV2 = inValueV2 & sgnMaskV2;
        outValueV2 = inValueV2 ^ sgnV2;
        expV2 = inValueV2 & expMaskV2;

        float2 fpExp;
        uintAsVec<float2, uint64_t, 1>(&fpExp, expV2);

        uint64_t manMaskV2, isDenormV2;
        compareF32v2Lt(fpExp, fpMinNorm, &isDenormV2);
        int minNorm;
        vecAsUInt<float, int, 1>(&fpMinNorm, &minNorm);
        manMaskV2 = outManMaskV2;
        if (EN_DENORM) {
          manMaskV2 = manMaskV2 & (~isDenormV2);
          float2 dnrmMan;
          dnrmMan = subF32v2(fpExp, fpHalfMinValue);
          uint64_t denormV2;
          vecAsUInt<float2, uint64_t, 1>(&dnrmMan, &denormV2);
          denormV2 = denormV2 | sgnExpMaskV2;
          denormV2 = denormV2 & isDenormV2;
          manMaskV2 = manMaskV2 | denormV2;
        }
        float2 fpOut;
        uintAsVec<float2, uint64_t, 1>(&fpOut, outValueV2);

        float2 corrV2;
        uintAsVec<float2, uint64_t, 1>(&corrV2, 0);

        uint64_t randBits;
        gfloat32_correction_sr(corrV2, manMaskV2, expV2, randBits,
                               fpHalfMinValue);

        fpOut = addF32v2(fpOut, corrV2);
        vecAsUInt<float2, uint64_t, 1>(&fpOut, &outValueV2);

        float gf32MaxValue = fpOutClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX];
        if (enNanoo) {
          float2 Out;
          uint64_t isGtVec, inVec, outVec;
          compareF32v2Gt(fpOut, gf32MaxValue, &isGtVec);
          vecAsUInt<float2, uint64_t, 1>(&fpOut, &inVec);
          inVec = inVec & (~isGtVec);
          outVec = qnanMaskV2 & isGtVec;
          outVec = outVec | inVec;
          uintAsVec<float2, uint64_t, 1>(&fpOut, outVec);
          fpOut = genQnanOverflowF32(fpOut, gf32MaxValue, qnanMaskV2);
        } else {
          fpOut = clipF32v2(fpOut, gf32MaxValue);
        }

        uint64_t maskOutV2;
        vecAsUInt<float2, uint64_t, 1>(&fpOut, &maskOutV2);
        maskOutV2 = maskOutV2 & manMaskV2;
        maskOutV2 = maskOutV2 | nanValue;
        maskOutV2 = maskOutV2 | sgnV2;
        uintAsVec<float2, uint64_t, 1>(&fpOut, maskOutV2);
        for (unsigned idx = 0; idx != maxPerCall; ++idx) {
          inOut[Idx][POPFLOAT_GF32_VEC_SIZE * j + idx] = fpOut[idx];
        }
      }
    }
    return true;
  }
};

} // end namespace experimental
} // end namespace popfloat
