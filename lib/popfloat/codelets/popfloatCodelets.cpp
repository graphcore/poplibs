#include "popfloatCodelets.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <experimental/popfloat/GfloatExpr.hpp>
#include <ipudef.h>
#include <poplar/Vertex.hpp>
#include <print.h>

static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

using namespace poplar;

namespace experimental {
namespace popfloat {

template <typename FPType, typename GFType>
class CastToGfloat16 : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<FPType, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<GFType, SPAN, 8>>, SPAN> out;
  Vector<uint32_t, ONE_PTR, 8> srMask;
  unsigned roundMode;
  bool enNanoo;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);
  bool compute() {
    unsigned int gf8AlignShr;
    uint64_t halfExpMaskV4, halfSgnMaskV4, outBitsMaskV4, minDnrmV4;
    short2 gf16CalmpOut;
    uint64_t clampF32In, enNanooInf, halfGenQnanV4;
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
    if (std::is_same<FPType, float>::value) {
      uintAsVec<float, uint32_t, 1>(&scale, scaleIn);

      float2 fpClamp;
      uintAsVec<float2, uint64_t, 1>(&fpClamp, clampF32In);

      maxValue = fpClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX];
    } else {
      uint16_t hfScale[2];
      uintAsVec<uint16_t, uint32_t, 2>(hfScale, scaleIn);

      uint16_t hfClamp[4];
      uintAsVec<uint16_t, uint32_t, 2>(hfClamp, clampF16In);
      maxValue = floatFromHalfBits(hfClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX]);
    }
    uint32_t maxBits;
    vecAsUInt<float, uint32_t, 1>(&maxValue, &maxBits);
    short4 scaledIn;

    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;
      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF16_VEC_SIZE) {
        unsigned maxPerCall =
            (len < POPFLOAT_GF16_VEC_SIZE) ? len : POPFLOAT_GF16_VEC_SIZE;
        for (unsigned idx = 0; idx != POPFLOAT_GF16_VEC_SIZE; ++idx) {
          float tmp = (float)0.0;
          if (idx < maxPerCall) {
            if (std::is_same<FPType, float>::value) {
              tmp = (float)in[Idx][POPFLOAT_GF16_VEC_SIZE * j + idx];
            } else {
              short inBits;
              std::memcpy(&inBits, &in[Idx][POPFLOAT_GF16_VEC_SIZE * j + idx],
                          sizeof(inBits));
              tmp = floatFromHalfBits(inBits);
            }
            uint32_t inBits;
            vecAsUInt<float, uint32_t, 1>(&maxValue, &maxBits);

            if (abs(tmp) > maxValue) {
              if (!enNanoo) {
                tmp = (tmp > 0) ? (float)maxValue : -((float)maxValue);
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
        std::memcpy(&out[Idx][POPFLOAT_GF16_VEC_SIZE * j], &maskOutV4,
                    sizeof(maskOutV4));
      }
    }
    return true;
  }
};

template class CastToGfloat16<float, float>;
template class CastToGfloat16<float, half>;
template class CastToGfloat16<half, half>;

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

template <typename FPType, typename GFType>
class CastToGfloat16Sr : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<FPType, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<GFType, SPAN, 8>>, SPAN> out;
  Vector<uint32_t, ONE_PTR, 8> srMask;
  unsigned roundMode;
  bool enNanoo;
  Vector<unsigned, ONE_PTR, 8> corrParams;
  unsigned distParam;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);
  bool compute() {
    unsigned int roundMode, gf8AlignShr;
    uint64_t halfExpMaskV4, halfSgnMaskV4, outBitsMaskV4, minDnrmV4;
    short2 gf16CalmpOut;
    uint64_t clampF32In, enNanooInf, srMaskV4, halfGenQnanV4;
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
    if (std::is_same<FPType, float>::value) {
      uintAsVec<float, uint32_t, 1>(&scale, scaleIn);

      float2 fpClamp;
      uintAsVec<float2, uint64_t, 1>(&fpClamp, clampF32In);

      maxValue = fpClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX];
    } else {
      uint16_t hfScale[2];
      uintAsVec<uint16_t, uint32_t, 2>(hfScale, scaleIn);

      uint16_t hfClamp[4];
      uintAsVec<uint16_t, uint32_t, 2>(hfClamp, clampF16In);
      maxValue = floatFromHalfBits(hfClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX]);
    }
    uint32_t maxBits;
    vecAsUInt<float, uint32_t, 1>(&maxValue, &maxBits);
    short4 scaledIn;

    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;
      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF16_VEC_SIZE) {
        unsigned maxPerCall =
            (len < POPFLOAT_GF16_VEC_SIZE) ? len : POPFLOAT_GF16_VEC_SIZE;
        for (unsigned idx = 0; idx != POPFLOAT_GF16_VEC_SIZE; ++idx) {
          float tmp = (float)0.0;
          if (idx < maxPerCall) {
            if (std::is_same<FPType, float>::value) {
              tmp = (float)in[Idx][POPFLOAT_GF16_VEC_SIZE * j + idx];
            } else {
              short inBits;
              std::memcpy(&inBits, &in[Idx][POPFLOAT_GF16_VEC_SIZE * j + idx],
                          sizeof(inBits));
              tmp = floatFromHalfBits(inBits);
            }
            uint32_t inBits;
            vecAsUInt<float, uint32_t, 1>(&maxValue, &maxBits);

            if (abs(tmp) > maxValue) {
              if (!enNanoo) {
                tmp = (tmp > 0) ? (float)maxValue : -((float)maxValue);
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

        uint64_t corrV4 =
            gfloat16_correction(inValueV4, manMaskV4, expV4, RoundType::SR);
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
        std::memcpy(&out[Idx][POPFLOAT_GF16_VEC_SIZE * j], &maskOutV4,
                    sizeof(maskOutV4));
      }
    }
    return true;
  }
};

template class CastToGfloat16Sr<float, float>;
template class CastToGfloat16Sr<float, half>;
template class CastToGfloat16Sr<half, half>;

template <typename FPType> class CastToGfloat16SrInPlace : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<InOut<Vector<FPType, SPAN, 8>>, SPAN> inOut;
  Vector<uint32_t, ONE_PTR, 8> srMask;
  unsigned roundMode;
  bool enNanoo;
  Vector<unsigned, ONE_PTR, 8> corrParams;
  unsigned distParam;

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

        uint64_t corrV4 =
            gfloat16_correction(inValueV4, manMaskV4, expV4, RoundType::SR);
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

template class CastToGfloat16SrInPlace<float>;
template class CastToGfloat16SrInPlace<half>;

template <FormatType FORMAT> class CastHalfToGf8 : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<half, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<char, SPAN, 4>>, SPAN> out;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
    unsigned int gf8AlignShr;
    uint64_t halfExpMaskV4, inValueV4, halfSgnMaskV4;

    uint16_t expMaks = POPFLOAT_FP16_EXPONENT_MASK;
    halfExpMaskV4 = addF16v4(0, expMaks);

    std::memcpy(&gf8AlignShr,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_PACK_SHR_ALIGN_OFFSET],
                sizeof(gf8AlignShr));

    uint16_t sgnMask = POPFLOAT_FP16_SIGN_MASK;
    halfSgnMaskV4 = mulF16v4(0, sgnMask);

    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;
      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF16_VEC_SIZE) {
        for (unsigned idx = 0; idx != POPFLOAT_GF16_VEC_SIZE; ++idx) {
          std::memcpy(&inValueV4, &in[Idx][POPFLOAT_GF16_VEC_SIZE * j],
                      sizeof(inValueV4));

          uint64_t sgnV4;
          sgnV4 = inValueV4 & halfSgnMaskV4;

          char4 gf8V4;
          char gfV8[2 * POPFLOAT_GF16_VEC_SIZE];
          if (FORMAT == FormatType::MAX_NORM_ALIGN_GF8) {
            uint16_t maxExpBits = 0x7800; // 32768.0

            uint64_t expV4, hfTmpV4;
            expV4 = inValueV4 & halfExpMaskV4;

            hfTmpV4 = addF16v4(0, maxExpBits);

            uint64_t isMaxExpV4, maxExpV4;
            compareF16v4Eq(expV4, hfTmpV4, &isMaxExpV4);

            maxExpV4 = inValueV4 | halfExpMaskV4;
            maxExpV4 = maxExpV4 & isMaxExpV4;
            inValueV4 = inValueV4 & ~isMaxExpV4;

            uint16_t twoBits = 0x4000; // 2.0

            inValueV4 = mulF16v4(hfTmpV4, twoBits);
            inValueV4 = inValueV4 | maxExpV4 | halfExpMaskV4;
          } else if (FORMAT == FormatType::MIN_NORM_ALIGN_GF8) {
            inValueV4 = (inValueV4 >> gf8AlignShr) << 8;
          }
          inValueV4 = inValueV4 | sgnV4;
          uintAsVec<char, uint64_t, 8>(gfV8, inValueV4);
          for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
            gf8V4[idx] = gfV8[2 * idx + 1];
          }
          std::memcpy(&out[Idx][POPFLOAT_GF16_VEC_SIZE * j], &gf8V4,
                      sizeof(gf8V4));
        }
      }
    }
    return true;
  }
};
template class CastHalfToGf8<FormatType::MIN_NORM_ALIGN_GF8>;
template class CastHalfToGf8<FormatType::ONE_FIVE_TWO_GF8>;
template class CastHalfToGf8<FormatType::MAX_NORM_ALIGN_GF8>;

template <FormatType FORMAT> class CastGf8ToHalf : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<char, SPAN, 4>>, SPAN> in;
  Vector<Output<Vector<half, SPAN, 8>>, SPAN> out;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
    unsigned int PROP_NAN, EN_DENORM, EN_INF;
    uint64_t halfExpMaskV4, outBitsMaskV4, halfSgnMaskV4;
    uint64_t enNanooInf;

    uint32_t gf8SgnMask, gf8ShrAlign;
    uint64_t maxExpV4;
    short2 hlfClamp;

    uint16_t halfGenQnan = POPFLOAT_FP16_GEN_QNAN;
    uint64_t halfGenQnanV4 = 0;
    halfGenQnanV4 = addF16v4(0, halfGenQnan);

    std::memcpy(&halfExpMaskV4,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(halfExpMaskV4));
    std::memcpy(&maxExpV4,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_MAX_EXPONENT_OFFSET],
                sizeof(maxExpV4));
    std::memcpy(&hlfClamp,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_CLAMP_OUTPUT_OFFSET],
                sizeof(hlfClamp));
    std::memcpy(&gf8ShrAlign,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_UNPACK_SHR_ALIGN_OFFSET],
                sizeof(gf8ShrAlign));
    std::memcpy(&gf8SgnMask,
                &param[POPFLOAT_CAST_TO_GP16_PARAM_GF8_SIGN_MASK_OFFSET],
                sizeof(gf8SgnMask));

    ushort4 fp16Sgn;
    for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
      fp16Sgn[idx] = POPFLOAT_FP16_SIGN_MASK;
    }
    vecAsUInt<ushort4, uint64_t, 1>(&fp16Sgn, &halfSgnMaskV4);

    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;

      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF16_VEC_SIZE) {
        char4 gf8V4;
        std::memcpy(&gf8V4, &in[Idx][POPFLOAT_GF16_VEC_SIZE * j],
                    sizeof(gf8V4));
        char gfV8[2 * POPFLOAT_GF16_VEC_SIZE];
        for (int idx = 0; idx < POPFLOAT_GF16_VEC_SIZE; ++idx) {
          gfV8[2 * idx + 0] = 0;
          gfV8[2 * idx + 1] = gf8V4[idx];
        }
        uint64_t maskOutV4, sgnV4, expV4;
        vecAsUInt<char, uint64_t, 8>(gfV8, &maskOutV4);

        sgnV4 = maskOutV4 & halfSgnMaskV4;
        maskOutV4 = maskOutV4 ^ sgnV4;

        if (FORMAT == FormatType::MAX_NORM_ALIGN_GF8) {
          uint16_t maxExpBits = 0x7800; // 32768.0

          uint64_t hfTmpV4;
          expV4 = maskOutV4 & halfExpMaskV4;
          hfTmpV4 = addF16v4(hfTmpV4, maxExpBits);

          uint64_t isMaxExpV4, maxExpV4;
          compareF16v4Eq(expV4, hfTmpV4, &isMaxExpV4);
          maxExpV4 = maskOutV4 | halfExpMaskV4;
          maxExpV4 = maxExpV4 & isMaxExpV4;
          maskOutV4 = maskOutV4 & ~isMaxExpV4;

          uint16_t hlf2Pm1Bits = 0x3800;
          maskOutV4 = mulF16v4(maskOutV4, hlf2Pm1Bits);
          maskOutV4 = maskOutV4 | maxExpV4;
        } else if (FORMAT == FormatType::MIN_NORM_ALIGN_GF8) {
          maskOutV4 = (maskOutV4 >> 8) << gf8ShrAlign;
        }
        float gf16MaxValue =
            floatFromHalfBits(hlfClamp[POPFLOAT_IPU_CLAMP_INDEX_MAX]);
        uint64_t hfpOut;
        if (PROP_NAN) {
          hfpOut = genQnanOverflowF16(maskOutV4, gf16MaxValue, halfGenQnanV4);
        } else {
          hfpOut = clipF16v4(maskOutV4, gf16MaxValue);
        }
        maskOutV4 = hfpOut | sgnV4;
        std::memcpy(&out[Idx][POPFLOAT_GF16_VEC_SIZE * j], &maskOutV4,
                    sizeof(maskOutV4));
      }
    }
    return true;
  }
};
template class CastGf8ToHalf<FormatType::MIN_NORM_ALIGN_GF8>;
template class CastGf8ToHalf<FormatType::ONE_FIVE_TWO_GF8>;
template class CastGf8ToHalf<FormatType::MAX_NORM_ALIGN_GF8>;

template <typename FPType, typename GFType>
class CastToGfloat32 : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<FPType, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<GFType, SPAN, 8>>, SPAN> out;
  Vector<uint32_t, ONE_PTR, 8> srMask;
  unsigned roundMode;
  bool enNanoo;

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
    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;

      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF32_VEC_SIZE) {
        unsigned maxPerCall =
            (len < POPFLOAT_GF32_VEC_SIZE) ? len : POPFLOAT_GF32_VEC_SIZE;
        for (unsigned idx = 0; idx != POPFLOAT_GF32_VEC_SIZE; ++idx) {
          inV2[idx] = 0;
          if (idx < maxPerCall) {
            inV2[idx] = in[Idx][POPFLOAT_GF32_VEC_SIZE * j + idx];
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

        if ((RoundType)roundMode != RoundType::RZ) {
          float2 corrV2;
          uintAsVec<float2, uint64_t, 1>(&corrV2, 0);

          if ((RoundType)roundMode == RoundType::SR) {
            uint64_t randBits;
            gfloat32_correction_sr(corrV2, manMaskV2, expV2, randBits,
                                   fpHalfMinValue);
          } else {
            gfloat32_correction_dr(corrV2, expMaskV2, inValueV2, manMaskV2,
                                   expV2, (RoundType)roundMode);
          }

          fpOut = addF32v2(fpOut, corrV2);
          vecAsUInt<float2, uint64_t, 1>(&fpOut, &outValueV2);
        }

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
          out[Idx][POPFLOAT_GF32_VEC_SIZE * j + idx] = fpOut[idx];
        }
      }
    }
    return true;
  }
};

template class CastToGfloat32<float, float>;
template class CastToGfloat32<float, half>;

class CastToGfloat32InPlace : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<InOut<Vector<float, SPAN, 8>>, SPAN> inOut;
  Vector<uint32_t, ONE_PTR, 8> srMask;
  unsigned roundMode;
  bool enNanoo;

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

        if ((RoundType)roundMode != RoundType::RZ) {
          float2 corrV2;
          uintAsVec<float2, uint64_t, 1>(&corrV2, 0);

          if ((RoundType)roundMode == RoundType::SR) {
            uint64_t randBits;
            gfloat32_correction_sr(corrV2, manMaskV2, expV2, randBits,
                                   fpHalfMinValue);
          } else {
            gfloat32_correction_dr(corrV2, expMaskV2, inValueV2, manMaskV2,
                                   expV2, (RoundType)roundMode);
          }

          fpOut = addF32v2(fpOut, corrV2);
          vecAsUInt<float2, uint64_t, 1>(&fpOut, &outValueV2);
        }

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

template <typename FPType, typename GFType>
class CastToGfloat32Sr : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<FPType, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<GFType, SPAN, 8>>, SPAN> out;
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
    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;

      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF32_VEC_SIZE) {
        unsigned maxPerCall =
            (len < POPFLOAT_GF32_VEC_SIZE) ? len : POPFLOAT_GF32_VEC_SIZE;
        for (unsigned idx = 0; idx != POPFLOAT_GF32_VEC_SIZE; ++idx) {
          inV2[idx] = 0;
          if (idx < maxPerCall) {
            inV2[idx] = in[Idx][POPFLOAT_GF32_VEC_SIZE * j + idx];
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
          out[Idx][POPFLOAT_GF32_VEC_SIZE * j + idx] = fpOut[idx];
        }
      }
    }
    return true;
  }
};

template class CastToGfloat32Sr<float, float>;
template class CastToGfloat32Sr<float, half>;

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

template <FormatType FORMAT> class CastFloatToGf16 : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<float, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<short, SPAN, 8>>, SPAN> out;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
    uint64_t expMaskV2, sgnMaskV2;
    float fpMinNorm, fpMinValue, fpHalfMinValue;
    unsigned int gf16AlignShr;
    float2 fpOutClamp;
    unsigned int gf16BiasCorr;

    std::memcpy(&expMaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_EXPONENT_MASK_OFFSET],
                sizeof(expMaskV2));
    std::memcpy(&gf16BiasCorr,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_PACK_EXP_ALIGN_OFFSET],
                sizeof(gf16BiasCorr));
    std::memcpy(&gf16AlignShr,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_PACK_SHR_ALIGN_OFFSET],
                sizeof(gf16AlignShr));
    std::memcpy(&fpMinNorm, &param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                sizeof(fpMinNorm));

    float2 inV2;

    float sgnMask;
    uintAsVec<float, uint32_t, 1>(&sgnMask, POPFLOAT_FP32_SIGN_MASK);

    uintAsVec<float2, uint64_t, 1>(&inV2, 0);
    inV2 = mulF32v2(inV2, sgnMask);
    vecAsUInt<float2, uint64_t, 1>(&inV2, &sgnMaskV2);

    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;
      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF32_VEC_SIZE) {
        unsigned maxPerCall =
            (len < POPFLOAT_GF32_VEC_SIZE) ? len : POPFLOAT_GF32_VEC_SIZE;
        for (unsigned idx = 0; idx != POPFLOAT_GF32_VEC_SIZE; ++idx) {
          inV2[idx] = 0;
          if (idx < maxPerCall) {
            inV2[idx] = in[Idx][POPFLOAT_GF32_VEC_SIZE * j + idx];
          }
        }

        uint64_t inValueV2 = 0;
        vecAsUInt<float2, uint64_t, 1>(&inV2, &inValueV2);

        short2 gf16Out;
        if (FORMAT == FormatType::BFLOAT16) {
          short4 shortOut;
          uintAsVec<short4, uint64_t, 1>(&shortOut, inValueV2);
          for (unsigned idx = 0; idx != POPFLOAT_GF32_VEC_SIZE; ++idx) {
            gf16Out[idx] = shortOut[2 * idx + 1];
          }
        } else {
          uint64_t sgnV2;
          sgnV2 = inValueV2 & sgnMaskV2;
          inValueV2 = inValueV2 ^ sgnV2;

          short2 gf16Sign;
          short4 signOut;
          uintAsVec<short4, uint64_t, 1>(&signOut, sgnV2);

          for (unsigned idx = 0; idx != POPFLOAT_GF32_VEC_SIZE; ++idx) {
            gf16Sign[idx] = signOut[2 * idx + 1];
          }

          uint32_t fpMinNormBits;
          vecAsUInt<float, uint32_t, 1>(&fpMinNorm, &fpMinNormBits);
          uintAsVec<float2, uint64_t, 1>(&inV2, inValueV2);
          uint64_t isDenormV2;
          compareF32v2Lt(inV2, fpMinNorm, &isDenormV2);
          float2 normV2, dnrmV2;
          normV2 = mulF32v2(inV2, (float)2.0);
          dnrmV2 = addF32v2(inV2, (float)fpMinNorm);
          uint64_t normOut, dnrmOut;
          vecAsUInt<float2, uint64_t, 1>(&normV2, &normOut);
          vecAsUInt<float2, uint64_t, 1>(&dnrmV2, &dnrmOut);

          normOut = normOut & (~isDenormV2);
          dnrmOut = dnrmOut & isDenormV2;
          inValueV2 = normOut | dnrmOut | sgnV2;
          uint32_t fpMinN;
          vecAsUInt<float, uint32_t, 1>(&fpMinNorm, &fpMinN);

          uint32_t fpBits[POPFLOAT_GF32_VEC_SIZE];
          uintAsVec<uint32_t, uint64_t, 2>(fpBits, inValueV2);
          for (unsigned idx = 0; idx != POPFLOAT_GF32_VEC_SIZE; ++idx) {
            fpBits[idx] = fpBits[idx] - fpMinN;
            gf16Out[idx] = (fpBits[idx] >> gf16AlignShr) & 0x7FFF;
            gf16Out[idx] = gf16Out[idx] | gf16Sign[idx];
          }
        }
        std::memcpy(&out[Idx][POPFLOAT_GF32_VEC_SIZE * j], &gf16Out,
                    sizeof(gf16Out));
      }
    }

    return true;
  }
};
template class CastFloatToGf16<FormatType::BFLOAT16>;
template class CastFloatToGf16<FormatType::NO_DENORM_GF16>;
template class CastFloatToGf16<FormatType::ENABLE_DENORM_GF16>;

template <FormatType FORMAT> class CastGf16ToFloat : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<short, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<float, SPAN, 8>>, SPAN> out;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() {
    uint64_t expMaskV2;
    uint32_t gf16BiasCorr, minNormBits;
    unsigned int gf16AlignSh0, gf16AlignSh1;
    float gf16MinNorm;
    float2 gf16ClampOut;

    std::memcpy(&expMaskV2,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_GF16_EXP_MASK_OFFSET],
                sizeof(expMaskV2));
    std::memcpy(&gf16ClampOut,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_CLAMP_OUTPUT_OFFSET],
                sizeof(gf16ClampOut));
    std::memcpy(&gf16BiasCorr,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_EXP_ALIGN_OFFSET],
                sizeof(gf16BiasCorr));
    std::memcpy(&minNormBits,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                sizeof(minNormBits));
    std::memcpy(&gf16MinNorm,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_MIN_NORM_OFFSET],
                sizeof(gf16MinNorm));
    std::memcpy(&gf16AlignSh0,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT0_OFFSET],
                sizeof(gf16AlignSh0));
    std::memcpy(&gf16AlignSh1,
                &param[POPFLOAT_CAST_TO_GF32_PARAM_UNPACK_SHIFT1_OFFSET],
                sizeof(gf16AlignSh1));

    unsigned int expManMask = (0x7FFF << gf16AlignSh0);
    for (unsigned Idx = 0; Idx < in.size(); ++Idx) {
      unsigned len = in[Idx].size();
      unsigned nv = (len + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;
      for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF32_VEC_SIZE) {
        uint64_t maskOutV2;
        float2 fp32V2;
        if (FORMAT == FormatType::QUANTISED_FP16) {
          short2 fp16V2;
          std::memcpy(&fp16V2, &in[Idx][POPFLOAT_GF32_VEC_SIZE * j],
                      sizeof(fp16V2));
          for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
            fp32V2[idx] = floatFromHalfBits(fp16V2[idx]);
          }
        } else {
          short2 gf16V2;
          std::memcpy(&gf16V2, &in[Idx][POPFLOAT_GF32_VEC_SIZE * j],
                      sizeof(gf16V2));
          int32_t gf32V2[POPFLOAT_GF32_VEC_SIZE];
          if (FORMAT == FormatType::BFLOAT16) {
            for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
              gf32V2[idx] = gf16V2[idx] << 16;
            }
            vecAsUInt<int32_t, uint64_t, 2>(gf32V2, &maskOutV2);
          } else {
            int32_t sgnV2[POPFLOAT_GF32_VEC_SIZE];
            gf32V2[0] = ((gf16V2[0] & 0xFFFF) << gf16AlignSh0);
            gf32V2[1] = ((gf16V2[1] & 0xFFFF) << gf16AlignSh0);
            for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
              sgnV2[idx] = (gf16V2[idx] & POPFLOAT_FP16_SIGN_MASK) << 16;
              gf32V2[idx] = gf32V2[idx] & expManMask;
              gf32V2[idx] = gf32V2[idx] + gf16BiasCorr;
            }

            uint64_t sgnOutV2;
            vecAsUInt<int32_t, uint64_t, 2>(sgnV2, &sgnOutV2);
            vecAsUInt<int32_t, uint64_t, 2>(gf32V2, &maskOutV2);
            uint64_t isDenormV2;
            isDenormV2 = maskOutV2 & expMaskV2;
            float2 expV2;
            uintAsVec<float2, uint64_t, 1>(&expV2, isDenormV2);
            compareF32v2Lt(expV2, gf16MinNorm, &isDenormV2);

            uint64_t dnrmMaskV2 = 0;
            if (FORMAT == FormatType::ENABLE_DENORM_GF16) {
              float2 dnrmOutV2;
              uintAsVec<float2, uint64_t, 1>(&dnrmOutV2, maskOutV2);

              dnrmOutV2 = subF32v2(dnrmOutV2, gf16MinNorm);

              vecAsUInt<float2, uint64_t, 1>(&dnrmOutV2, &dnrmMaskV2);
              dnrmMaskV2 = dnrmMaskV2 & isDenormV2;
            }

            float2 normOutV2;
            maskOutV2 = maskOutV2 & (~isDenormV2);
            uintAsVec<float2, uint64_t, 1>(&normOutV2, maskOutV2);
            normOutV2 = mulF32v2(normOutV2, (float)0.5);

            vecAsUInt<float2, uint64_t, 1>(&normOutV2, &maskOutV2);

            maskOutV2 = maskOutV2 | dnrmMaskV2;

            uintAsVec<float2, uint64_t, 1>(&fp32V2, maskOutV2);
            fp32V2 =
                clipF32v2(fp32V2, gf16ClampOut[POPFLOAT_IPU_CLAMP_INDEX_MAX]);

            vecAsUInt<float2, uint64_t, 1>(&fp32V2, &maskOutV2);
            maskOutV2 = maskOutV2 | sgnOutV2;
          }
          uintAsVec<float2, uint64_t, 1>(&fp32V2, maskOutV2);
        }
        out[Idx][POPFLOAT_GF32_VEC_SIZE * j + 0] = fp32V2[0];
        out[Idx][POPFLOAT_GF32_VEC_SIZE * j + 1] = fp32V2[1];
      }
    }
    return true;
  }
};
template class CastGf16ToFloat<FormatType::BFLOAT16>;
template class CastGf16ToFloat<FormatType::NO_DENORM_GF16>;
template class CastGf16ToFloat<FormatType::ENABLE_DENORM_GF16>;

class CastGf8ToFloat : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<char, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<float, SPAN, 8>>, SPAN> out;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() { return true; }
};

class CastFloatToGf8 : public Vertex {
public:
  Input<Vector<int, SPAN, 8>> param;
  Vector<Input<Vector<float, SPAN, 8>>, SPAN> in;
  Vector<Output<Vector<char, SPAN, 8>>, SPAN> out;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() { return true; }
};

} // end namespace popfloat
} // end namespace experimental
