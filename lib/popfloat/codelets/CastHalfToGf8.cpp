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

} // end namespace experimental
} // end namespace popfloat
