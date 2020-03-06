// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
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

template <typename FPType> class CastToGfloat16InPlace : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<InOut<Vector<FPType, SPAN, 8>>, SPAN> inOut;
  Vector<uint32_t, ONE_PTR, 8> srMask;
  unsigned roundMode;
  bool enNanoo;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);
  bool compute() {
    unsigned int gf8AlignShr;
    uint64_t halfExpMaskV4, halfSgnMaskV4, outBitsMaskV4, enNanooInf, srMaskV4,
        halfGenQnanV4, minDnrmV4;
    short2 gf16CalmpOut;
    uint64_t clampF32In;
    uint32_t scaleIn, clampF16In;
    uint16_t twoPwrM10Mman, f16Pwr10;
    enNanooInf = enNanoo ? (~0) : 0;

    std::memcpy(&halfExpMaskV4,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(halfExpMaskV4));
    std::memcpy(&halfSgnMaskV4,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_SIGN_MASK_OFFSET],
                sizeof(halfSgnMaskV4));
    std::memcpy(&halfGenQnanV4,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_QNAN_OUTPUT_OFFSET],
                sizeof(halfGenQnanV4));
    std::memcpy(&outBitsMaskV4,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_NORM_MAN_MASK_OFFSET],
                sizeof(outBitsMaskV4));
    std::memcpy(&clampF32In,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP32_IN_OFFSET],
                sizeof(clampF32In));
    std::memcpy(&clampF16In,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_FP16_IN_OFFSET],
                sizeof(clampF16In));
    std::memcpy(&gf16CalmpOut,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET],
                sizeof(gf16CalmpOut));
    std::memcpy(&scaleIn,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_SCALE_INPUT_OFFSET],
                sizeof(scaleIn));
    std::memcpy(&twoPwrM10Mman,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_M_MAN_10_OFFSET],
                sizeof(twoPwrM10Mman));
    std::memcpy(&f16Pwr10, &param[POPFLOAT_CAST_TO_GF16_PARAM_POWER2_10_OFFSET],
                sizeof(f16Pwr10));
    std::memcpy(&minDnrmV4, &param[POPFLOAT_CAST_TO_GF16_PARAM_MIN_DNRM_OFFSET],
                sizeof(minDnrmV4));

    float maxValue, scale;

    uint16_t hfScale[2];
    uintAsVec<uint16_t, uint32_t, 2>(hfScale, scaleIn);
    uint16_t hfClamp[2];
    uintAsVec<uint16_t, uint32_t, 2>(hfClamp, clampF16In);
    maxValue = floatFromHalfBits(hfClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX]);
    uint32_t maxBits;
    vecAsUInt<float, uint32_t, 1>(&maxValue, &maxBits);

    short4 scaledIn;
    for (unsigned Idx = 0; Idx < inOut.size(); ++Idx) {
      unsigned len = inOut[Idx].size();
      unsigned nv = (len + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;
      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF16_VEC_SIZE) {
        unsigned maxPerCall =
            (len < POPFLOAT_GF16_VEC_SIZE) ? len : POPFLOAT_GF16_VEC_SIZE;
        for (unsigned idx = 0; idx != POPFLOAT_GF16_VEC_SIZE; ++idx) {
          float tmp = (float)0.0;
          if (idx < maxPerCall) {
            short inBits;
            std::memcpy(&inBits, &inOut[Idx][POPFLOAT_GF16_VEC_SIZE * j + idx],
                        sizeof(inBits));
            tmp = floatFromHalfBits(inBits);
            if (abs(tmp) > maxValue) {
              if (!enNanoo) {
                tmp = (tmp > 0) ? maxValue : -maxValue;
              } else {
                uint16_t qnanMask;
                qnanMask = POPFLOAT_FP16_GEN_QNAN;
                qnanMask |= (tmp > 0) ? 0 : POPFLOAT_FP16_SIGN_MASK;
                tmp = floatFromHalfBits(POPFLOAT_FP16_GEN_QNAN);
              }
            }
          }
          poplar::IeeeHalf scaledTmp(tmp * scale);
          scaledIn[idx] = scaledTmp.bit16();
        }

        uint64_t inValueV4 = 0;
        vecAsUInt<short4, uint64_t, 1>(&scaledIn, &inValueV4);
        uint64_t sgnV4, outValueV4, expV4, isNanOrInf, nanValue;
        sgnV4 = inValueV4 & halfSgnMaskV4;
        outValueV4 = inValueV4 ^ sgnV4;

        isNanOrInf = gfloat16_nan_or_inf(inValueV4, halfExpMaskV4, enNanooInf);

        nanValue = isNanOrInf & halfGenQnanV4;
        inValueV4 = inValueV4 & (~isNanOrInf);
        expV4 = inValueV4 & halfExpMaskV4;

        uint64_t manMaskV4 = outBitsMaskV4 | minDnrmV4;

        uint64_t corrV4 = gfloat16_correction(inValueV4, manMaskV4, expV4,
                                              (RoundType)roundMode);

        uint64_t maskOutV4;
        maskOutV4 = addF16v4(outValueV4, corrV4);
        float gf16MaxValue =
            floatFromHalfBits(gf16CalmpOut[POPFLOAT_IPU_CLAMP_INDEX_MAX]);
        if (enNanoo) {
          maskOutV4 =
              genQnanOverflowF16(maskOutV4, gf16MaxValue, halfGenQnanV4);
        } else {
          maskOutV4 = clipF16v4(maskOutV4, gf16MaxValue);
        }

        maskOutV4 = maskOutV4 & manMaskV4;
        maskOutV4 = maskOutV4 | nanValue;
        maskOutV4 = maskOutV4 | sgnV4;

        std::memcpy(&inOut[Idx][POPFLOAT_GF16_VEC_SIZE * j], &maskOutV4,
                    sizeof(maskOutV4));
      }
    }
    return true;
  }
};

template class CastToGfloat16InPlace<half>;
template class CastToGfloat16InPlace<float>;

} // end namespace experimental
} // end namespace popfloat
