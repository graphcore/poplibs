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
class CastHalfToGf8Supervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  CastHalfToGf8Supervisor();

  Input<Vector<int, COMPACT_PTR, 8>> param;
  Input<Vector<half, COMPACT_PTR, 8>> in;
  Output<Vector<signed char, COMPACT_PTR, 4>> out;
  unsigned short elementsPerWorker;
  unsigned short lastWorkerParams;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  void compute() {
    unsigned int gf8AlignShr;
    uint64_t halfExpMaskV4, inValueV4, halfSgnMaskV4;

    uint16_t expMaks = POPFLOAT_FP16_EXPONENT_MASK;
    halfExpMaskV4 = addF16v4(0, expMaks);

    std::memcpy(&gf8AlignShr,
                &param[POPFLOAT_CAST_TO_GF16_PARAM_PACK_SHR_ALIGN_OFFSET],
                sizeof(gf8AlignShr));

    uint16_t sgnMask = POPFLOAT_FP16_SIGN_MASK;
    halfSgnMaskV4 = mulF16v4(0, sgnMask);

    auto lastWorker = lastWorkerParams & 0xFF;
    auto remainder = (lastWorkerParams >> 8) & 0xFF;

    unsigned len =
        (CTXT_WORKERS * elementsPerWorker + lastWorker) * 4 + remainder;
    unsigned nv = (len + POPFLOAT_GF16_VEC_SIZE - 1) / POPFLOAT_GF16_VEC_SIZE;
    for (unsigned j = 0; j != nv; ++j, len -= POPFLOAT_GF16_VEC_SIZE) {
      for (unsigned idx = 0; idx != POPFLOAT_GF16_VEC_SIZE; ++idx) {
        std::memcpy(&inValueV4, &in[POPFLOAT_GF16_VEC_SIZE * j],
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
        std::memcpy(&out[POPFLOAT_GF16_VEC_SIZE * j], &gf8V4, sizeof(gf8V4));
      }
    }
  }
};
template class CastHalfToGf8Supervisor<FormatType::MIN_NORM_ALIGN_GF8>;
template class CastHalfToGf8Supervisor<FormatType::ONE_FIVE_TWO_GF8>;
template class CastHalfToGf8Supervisor<FormatType::MAX_NORM_ALIGN_GF8>;

} // end namespace experimental
} // end namespace popfloat
