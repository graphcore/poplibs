// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popfloatCodelets.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <popfloat/experimental/GfloatExpr.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>

#include "poplar/TileConstants.hpp"

static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

template <FormatType FORMAT>
class CastGf16ToFloatSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  CastGf16ToFloatSupervisor();

  Input<Vector<int, COMPACT_PTR, 8>> param;
  Input<Vector<short, COMPACT_PTR, 4>> in;
  Output<Vector<float, COMPACT_PTR, 8>> out;
  unsigned short elementsPerWorker;
  unsigned short lastWorkerParams;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  void compute() {
#ifdef __IPU__
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

    auto lastWorker = lastWorkerParams & 0xFF;
    auto remainder = (lastWorkerParams >> 8) & 0xFF;

    unsigned int expManMask = (0x7FFF << gf16AlignSh0);
    unsigned len =
        (CTXT_WORKERS * elementsPerWorker + lastWorker) * 2 + remainder;
    unsigned nv = (len + POPFLOAT_GF32_VEC_SIZE - 1) / POPFLOAT_GF32_VEC_SIZE;

    for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF32_VEC_SIZE) {
      uint64_t maskOutV2;
      float2 fp32V2;
      if (FORMAT == FormatType::QUANTISED_FP16) {
        short2 fp16V2;
        std::memcpy(&fp16V2, &in[POPFLOAT_GF32_VEC_SIZE * j], sizeof(fp16V2));
        for (int idx = 0; idx < POPFLOAT_GF32_VEC_SIZE; ++idx) {
          fp32V2[idx] = floatFromHalfBits(fp16V2[idx]);
        }
      } else {
        short2 gf16V2;
        std::memcpy(&gf16V2, &in[POPFLOAT_GF32_VEC_SIZE * j], sizeof(gf16V2));
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
      out[POPFLOAT_GF32_VEC_SIZE * j + 0] = fp32V2[0];
      out[POPFLOAT_GF32_VEC_SIZE * j + 1] = fp32V2[1];
    }
#else
    // Not supported on non-ipu targets
    exit(1);
#endif // defined(__IPU__)
  }
};
template class CastGf16ToFloatSupervisor<FormatType::BFLOAT16>;
template class CastGf16ToFloatSupervisor<FormatType::NO_DENORM_GF16>;
template class CastGf16ToFloatSupervisor<FormatType::ENABLE_DENORM_GF16>;

} // end namespace experimental
} // end namespace popfloat
