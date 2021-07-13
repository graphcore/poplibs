// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popfloatCodelets.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <ipudef.h>
#include <popfloat/experimental/GfloatExpr.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>

#include "poplar/TileConstants.hpp"

static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

template <FormatType FORMAT>
class CastGf8ToHalfSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  CastGf8ToHalfSupervisor();

  Input<Vector<int, COMPACT_PTR, 8>> param;
  Input<Vector<signed char, COMPACT_PTR, 4>> in;
  Output<Vector<half, COMPACT_PTR, 8>> out;
  unsigned short elementsPerWorker;
  unsigned short lastWorkerParams;

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

    auto lastWorker = lastWorkerParams & 0xFF;
    auto remainder = (lastWorkerParams >> 8) & 0xFF;

    unsigned len =
        (CTXT_WORKERS * elementsPerWorker + lastWorker) * 4 + remainder;
    unsigned nv = (len + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;

    for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF16_VEC_SIZE) {
      char4 gf8V4;
      std::memcpy(&gf8V4, &in[POPFLOAT_GF16_VEC_SIZE * j], sizeof(gf8V4));
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
      std::memcpy(&out[POPFLOAT_GF16_VEC_SIZE * j], &maskOutV4,
                  sizeof(maskOutV4));
    }
    return true;
  }
};
template class CastGf8ToHalfSupervisor<FormatType::MIN_NORM_ALIGN_GF8>;
template class CastGf8ToHalfSupervisor<FormatType::ONE_FIVE_TWO_GF8>;
template class CastGf8ToHalfSupervisor<FormatType::MAX_NORM_ALIGN_GF8>;

} // end namespace experimental
} // end namespace popfloat
