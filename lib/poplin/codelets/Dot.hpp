// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/TileConstants.hpp"

namespace poplin {
namespace {
template <typename FloatType, bool AllowLoad2x64> struct Dot;

template <typename FloatType>
static std::ptrdiff_t combine(const FloatType *a, const FloatType *b) {
  return reinterpret_cast<std::ptrdiff_t>(a) |
         reinterpret_cast<std::ptrdiff_t>(b);
}

template <std::size_t N, typename FloatType>
static bool aligned(const FloatType *a, const FloatType *b) {
  static constexpr auto Mask = N - 1;
  return (combine<FloatType>(a, b) & Mask) == 0;
}

template <bool AllowLoad2x64> struct Dot<float, AllowLoad2x64> {
  static float compute(const float *&a, const float *&b, unsigned n) {
    float result = 0;
#ifdef __IPU__
    if (aligned<8>(a, b)) {
      auto n2 = n >> 1;
      if (n2) {
        auto const *a2 = reinterpret_cast<const float2 *>(a);
        auto const *b2 = reinterpret_cast<const float2 *>(b);
        static constexpr unsigned aaccMask = CSR_W_FP_CLR__ZAACC__MASK
                                             << CSR_W_FP_CLR__ZAACC__SHIFT;
        if (AllowLoad2x64) {
          uint2 packed_ptr = __builtin_ipu_tapack(a2, b2, 0);
          a2 += n2;
          b2 += n2;
          --n2;
          asm(
              R"(
                  {
                    ld2x64pace $a2:3, $a4:5, %[ptr]+=, $mzero, 0
                    setzi $a6, %[aaccMask]
                  }
                  .align 8
                  {
                    rpt %[n], 0
                    uput $FP_CLR, $a6
                  }
                  {
                    ld2x64pace $a2:3, $a4:5, %[ptr]+=, $mzero, 0
                    f32v2mac $a2:3, $a4:5
                  }
                  f32v2mac $a2:3, $a4:5
                  f32v2gina $a2:3, $azeros, 0
                  f32add %[result], $a2, $a3
                  )"
              : [n] "+r"(n2), [ptr] "+r"(packed_ptr), [result] "=r"(result)
              : [aaccMask] "n"(aaccMask)
              : "$a2:3", "$a4:5", "$a6");
        } else {
          --n2;
          asm(
              R"(
                  {
                    ld64step $a4:5, $mzero, %[b]+=, 1
                    setzi $a2, %[aaccMask]
                  }
                  .align 8
                  {
                    rpt %[n], 1
                    uput $FP_CLR, $a2
                  }
                  {
                    ld64step $a2:3, $mzero, %[a]+=, 1
                    fnop
                  }
                  {
                    ld64step $a4:5, $mzero, %[b]+=, 1
                    f32v2mac $a2:3, $a4:5
                  }
                  ld64step $a2:3, $mzero, %[a]+=, 1
                  f32v2mac $a2:3, $a4:5
                  f32v2gina $a2:3, $azeros, 0
                  f32add %[result], $a2, $a3
                  )"
              : [a] "+r"(a2), [b] "+r"(b2), [n] "+r"(n2), [result] "=r"(result)
              : [aaccMask] "n"(aaccMask)
              : "$a2:3", "$a4:5");
        }
        a = reinterpret_cast<const float *>(a2);
        b = reinterpret_cast<const float *>(b2);
      }
      n = n & 1;
    }
#endif

    while (n--) {
      result += *a++ * *b++;
    }

    return result;
  }
};

template <bool AllowLoad2x64> struct Dot<half, AllowLoad2x64> {
  static half compute(const half *&a, const half *&b, unsigned n) {
    half result = 0;
#ifdef __IPU__
    static constexpr unsigned aaccMask = CSR_W_FP_CLR__ZAACC__MASK
                                         << CSR_W_FP_CLR__ZAACC__SHIFT;
    auto combined = combine(a, b);
    if ((combined & 7) == 0) {
      auto n4 = n >> 2;
      if (n4) {
        const half4 *a4 = reinterpret_cast<const half4 *>(a);
        const half4 *b4 = reinterpret_cast<const half4 *>(b);
        half2 result2;

        if (AllowLoad2x64) {
          uint2 packed_ptr = __builtin_ipu_tapack(a4, b4, 0);
          a4 += n4;
          b4 += n4;
          --n4;
          asm(
              R"(
                {
                  ld2x64pace $a2:3, $a4:5, %[ptr]+=, $mzero, 0
                  setzi $a6, %[aaccMask]
                }
                .align 8
                {
                  rpt %[n], 0
                  uput $FP_CLR, $a6
                }
                {
                  ld2x64pace $a2:3, $a4:5, %[ptr]+=, $mzero, 0
                  f16v4cmac $a2:3, $a4:5
                }
                f16v4cmac $a2:3, $a4:5
                f16v2gina $a2, $azero, 0
                f16v2gina $a3, $azero, 1
                f16v2add %[result], $a2, $a3
                )"
              : [n] "+r"(n4), [ptr] "+r"(packed_ptr), [result] "=r"(result2)
              : [aaccMask] "n"(aaccMask)
              : "$a2:3", "$a4:5", "$a6");
        } else {
          --n4;
          asm(
              R"(
                {
                  ld64step $a4:5, $mzero, %[b]+=, 1
                  setzi $a2, %[aaccMask]
                }
                .align 8
                {
                  rpt %[n], 1
                  uput $FP_CLR, $a2
                }
                {
                  ld64step $a2:3, $mzero, %[a]+=, 1
                  fnop
                }
                {
                  ld64step $a4:5, $mzero, %[b]+=, 1
                  f16v4cmac $a2:3, $a4:5
                }
                ld64step $a2:3, $mzero, %[a]+=, 1
                f16v4cmac $a2:3, $a4:5
                f16v2gina $a2, $azero, 0
                f16v2gina $a3, $azero, 1
                f16v2add %[result], $a2, $a3
                )"
              : [a] "+r"(a4), [b] "+r"(b4), [n] "+r"(n4), [result] "=r"(result2)
              : [aaccMask] "n"(aaccMask)
              : "$a2:3", "$a4:5");
        }
        result = result2[0] + result2[1];
        a = reinterpret_cast<const half *>(a4);
        b = reinterpret_cast<const half *>(b4);
        n = n & 3;
      } // if (n4)
    }   // if (aligned)

    if ((combined & 3) == 0) {
      auto n2 = n >> 1;
      if (n2) {
        auto a2 = reinterpret_cast<const half2 *>(a);
        auto b2 = reinterpret_cast<const half2 *>(b);
        half2 result2;
        --n2;
        asm(
            R"(
              {
                ld32step $a3, $mzero, %[b]+=, 1
                setzi $a2, %[aaccMask]
              }
              .align 8
              {
                rpt %[n], 1
                uput $FP_CLR, $a2
              }
              {
                ld32step $a2, $mzero, %[a]+=, 1
                fnop
              }
              {
                ld32step $a3, $mzero, %[b]+=, 1
                f16v2cmac $a2, $a3
              }
              ld32step $a2, $mzero, %[a]+=, 1
              f16v2cmac $a2, $a3
              f16v2gina %[result], $azero, 0
              )"
            : [a] "+r"(a2), [b] "+r"(b2), [n] "+r"(n2), [result] "=r"(result2)
            : [aaccMask] "n"(aaccMask)
            : "$a2:3");
        result += result2[0] + result2[1];
        a = reinterpret_cast<const half *>(a2);
        b = reinterpret_cast<const half *>(b2);
        n = n & 1;
      } // if (n2)
    }   // if (aligned)

#endif

    while (n--) {
      result += *a++ * *b++;
    }
    return result;
  }
};
} // namespace
} // namespace poplin
