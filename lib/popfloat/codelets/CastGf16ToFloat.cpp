// Copyright (c) Graphcore Ltd, All rights reserved.
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

} // end namespace popfloat
} // end namespace experimental
