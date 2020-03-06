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

} // end namespace experimental
} // end namespace popfloat
