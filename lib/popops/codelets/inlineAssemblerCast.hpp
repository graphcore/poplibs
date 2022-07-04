// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <ipu_builtins.h>

// Flag value for rounding mode
#define TFPU_ROUND_ZERO 3

enum class TemplateInstructions {
  f32toi32,
  f32toui32,
  f32v4tof16,
  f16v8tof8,
  f16v2tof32,
  f8v4tof16
};

// Optimized pipelined processing for multiple of 4 elements
// UNSIGNED_CHAR => HALF
template <unsigned stride>
static __attribute__((always_inline)) void
castUCharHalfLoop(const unsigned char *inPtr, half *outPtr,
                  unsigned loopCount) {
  asm volatile(
      R"l(
              brnzdec     %[loopCount], 3f
              bri         4f
            .align 8
              nop
            3:
              ldz16step   $m0, $mzero, %[inPtr]+=, 1
              shuf8x8lo   $m0, $m0, $mzero
              st32        $m0, $mzero, %[outPtr], 0
              ld32        $a4, $mzero, %[outPtr], 0
              {ldz16step   $m0, $mzero, %[inPtr]+=,2* %[STRIDE] -1
              sort4x16lo  $a0, $a4, $azero }

              rpt         %[loopCount],((2f-1f)/8)-1
            1:
              {shuf8x8lo   $m0, $m0, $mzero
               sort4x16hi  $a1, $a4, $azero }
              {st32        $m0, $mzero, %[outPtr], 0
               f32fromui32 $a0, $a0       }
              {ld32        $a4, $mzero, %[outPtr], 0
               f32fromui32 $a1, $a1       }
              {ldz16step   $m0, $mzero, %[inPtr]+=, 1
               sort4x16lo  $a2, $a4, $azero }
              {shuf8x8lo   $m0, $m0, $mzero
               sort4x16hi  $a3, $a4, $azero }
              {st32        $m0, $mzero, %[outPtr], 0
               f32fromui32 $a2, $a2       }
              {ld32        $a4, $mzero, %[outPtr], 0
               f32fromui32 $a3, $a3       }
              {ldz16step   $m0, $mzero, %[inPtr]+=,2* %[STRIDE] -1
              f32v4tof16  $a0:1, $a0:3   }
              {st64step    $a0:1, $mzero, %[outPtr]+=, %[STRIDE]
               sort4x16lo  $a0, $a4, $azero }
            2:
              {shuf8x8lo   $m0, $m0, $mzero
               sort4x16hi  $a1, $a4, $azero }
              {st32        $m0, $mzero, %[outPtr], 0
               f32fromui32 $a0, $a0       }
              {ld32        $a4, $mzero, %[outPtr], 0
               f32fromui32 $a1, $a1       }
              sort4x16lo  $a2, $a4, $azero
              sort4x16hi  $a3, $a4, $azero
              f32fromui32 $a2, $a2
              f32fromui32 $a3, $a3
              f32v4tof16  $a0:1, $a0:3
              st64step    $a0:1, $mzero, %[outPtr]+=, %[STRIDE]
            4:
             )l"
      : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
      : [STRIDE] "i"(stride)
      : "$m0", "$a0:1", "$a2:3", "$a4", "memory");
}

// Optimized pipelined processing for multiple of 4 elements
// HALF => UCHAR or SCHAR
template <typename T, unsigned stride>
static __attribute__((always_inline)) void
castHalfUCharSCharLoop(const half *inPtr, T *outPtr, unsigned loopCount,
                       float2 limits) {
  asm volatile(
      R"l(
               brnzdec     %[loopCount], 3f
               bri         4f
               .align 8
               3:
                // Prologue (fill)
                ld64step    $a4:5, $mzero, %[inPtr]+=,  %[STRIDE]
                f16v2tof32  $a0:1, $a4
                f32clamp    $a0, $a0, %[limits]
                f32clamp    $a1, $a1, %[limits]
                f32int     $a0, $a0, %[ROUNDING_MODE]
                f32int     $a1, $a1, %[ROUNDING_MODE]
                f32toi32   $a0, $a0
                {add        %[loopCount], %[loopCount], -1
                f32toi32   $a1, $a1}

                // Main pipelined loop over blocks of 4 elements
                {brneg      %[loopCount], 5f
                sort4x16lo  $a0, $a0, $a1}
               6:
                {atom        $m0, $a0
                f16v2tof32  $a2:3, $a5}
                f32clamp    $a2, $a2, %[limits]
                f32clamp    $a3, $a3, %[limits]
                f32int $a2, $a2, %[ROUNDING_MODE]
                f32int $a3, $a3, %[ROUNDING_MODE]
                f32toi32   $a2, $a2
                f32toi32   $a3, $a3
                {ld64step    $a4:5, $mzero, %[inPtr]+=, %[STRIDE]
                sort4x16lo  $a2, $a2, $a3}

                {atom        $m1, $a2
                f16v2tof32  $a0:1, $a4}
                {sort8x8lo   $m0, $m0, $m1
                f32clamp    $a0, $a0, %[limits]}
                {st32step    $m0, $mzero, %[outPtr]+=, %[STRIDE]
                 f32clamp    $a1, $a1, %[limits]}
                 f32int $a0, $a0, %[ROUNDING_MODE]
                f32int $a1, $a1, %[ROUNDING_MODE]
                f32toi32  $a0, $a0
                f32toi32   $a1, $a1
                {brnzdec     %[loopCount], 6b
                sort4x16lo  $a0, $a0, $a1}
               5:
                 // Epilogue (drain)
                {atom        $m0, $a0
                f16v2tof32  $a2:3, $a5}
                f32clamp    $a2, $a2, %[limits]
                f32clamp    $a3, $a3, %[limits]
                f32int $a2, $a2, %[ROUNDING_MODE]
                f32int $a3, $a3, %[ROUNDING_MODE]
                f32toi32   $a2, $a2
                f32toi32   $a3, $a3
                sort4x16lo  $a2, $a2, $a3
                atom        $m1, $a2
                sort8x8lo   $m0, $m0, $m1
                st32step    $m0, $mzero, %[outPtr]+=, %[STRIDE]
               4:
           )l"
      : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
      : [limits] "r"(limits), [ROUNDING_MODE] "i"(TFPU_ROUND_ZERO),
        [STRIDE] "i"(stride)
      : "$m0", "$m1", "$a0:1", "$a2:3", "$a4:5", "memory");
}
// Optimized pipelined processing for multiple of 4 elements
// FLOAT => HALF or HALF => QUARTER
template <typename SrcType, typename DstType, TemplateInstructions instruction,
          unsigned stride>
static __attribute__((always_inline)) void
castFpDemoteLoop(const SrcType *inPtr, DstType *outPtr, unsigned loopCount) {
  if constexpr (instruction == TemplateInstructions::f32v4tof16) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f32v4tof16 \OPERANDS
         .endm
        )l" ::
            :);
  } else if constexpr (instruction == TemplateInstructions::f16v8tof8) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f16v8tof8 \OPERANDS
         .endm
        )l" ::
            :);
  }
  asm volatile(
      R"l(
          brnzdec   %[loopCount], 3f
          bri       4f
        .align 8
        3:
          ld64step $a0:1, $mzero, %[inPtr]+=,1
          ld64step $a2:3, $mzero, %[inPtr]+=,2* %[STRIDE] -1
          tapack $m0:1, %[inPtr], $mzero, %[outPtr]
          ld64step $azeros, $mzero, %[inPtr]+=,1
          mul $m2, %[loopCount], %[STRIDE] * 8
          add %[outPtr], %[outPtr], $m2
          setzi $m2, (2*  %[STRIDE])<<10 | %[STRIDE]
          rpt %[loopCount], (2f-1f)/8 -1
        1:
            {ld64step $a2:3, $mzero, %[inPtr]+=,2* %[STRIDE]
              cast $a0:1, $a0:3}
            {ldst64pace $a0:1, $a0:1, $m0:1+=, $m2, 6
              fnop}
        2:
          cast $a2:3, $a0:3
          st64step $a2:3, $mzero, %[outPtr]+=, %[STRIDE]
        4:

        // Remove the macro so it can be redefined later
        .purgem cast
    )l"
      : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
      : [STRIDE] "i"(stride)
      : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");
}

// Optimized pipelined processing for multiple of 4 elements
// HALF => FLOAT or QUARTER => HALF
template <typename SrcType, typename DstType, TemplateInstructions instruction,
          unsigned stride>
static __attribute__((always_inline)) void
castFpPromoteLoop(const SrcType *inPtr, DstType *outPtr, unsigned loopCount) {
  if constexpr (instruction == TemplateInstructions::f16v2tof32) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f16v2tof32 \OPERANDS
         .endm
        )l" ::
            :);
  } else if constexpr (instruction == TemplateInstructions::f8v4tof16) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f8v4tof16 \OPERANDS
         .endm
        )l" ::
            :);
  }
  asm volatile(
      R"l(
        brnzdec   %[loopCount], 3f
            bri       4f
        .align 8
        3:
          ld64step $a0:1, $mzero, %[inPtr]+=, %[STRIDE]
          tapack $m0:1, %[inPtr], $mzero, %[outPtr]
          mul $m2, %[loopCount], 16* %[STRIDE]
          add %[inPtr], %[outPtr], $m2
          ld64step $azeros, $mzero, %[outPtr]+=,1
          setzi $m2, 2* %[STRIDE]  | ( %[STRIDE] <<10)
          {rpt %[loopCount], (2f-1f)/8 -1;
           cast $a2:3, $a0}
        1:
          {ldst64pace $a0:1, $a2:3, $m0:1+=, $m2, 6
            cast $a2:3, $a1}
          {st64step $a2:3, $mzero, %[outPtr]+=,2* %[STRIDE]
            cast $a2:3, $a0}
        2:
          {st64step $a2:3, $mzero, %[inPtr]+=,1
            cast $a0:1, $a1}
          st64step $a0:1, $mzero, %[inPtr]+=, %[STRIDE]
        4:

        // remove the macro so it can be redefined later
        .purgem cast
    )l"
      : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
      : [STRIDE] "i"(stride)
      : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");
}
template <typename SrcType, typename DstType, bool charToFPType,
          unsigned stride>
class inLineAssemblerCast {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, SrcType inPtr, DstType outPtr) {
    // non-specialised template is never instantiated
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const float *, half *, false, stride> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const float *inPtr, half *outPtr) {
    castFpDemoteLoop<float, half, TemplateInstructions::f32v4tof16, stride>(
        inPtr, outPtr, loopCount);
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const half *, float *, false, stride> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const half *inPtr, float *outPtr) {
    castFpPromoteLoop<half, float, TemplateInstructions::f16v2tof32, stride>(
        inPtr, outPtr, loopCount);
    return;
  }
};

// Case where SrcType is float and DstType is an 8-bit (char/uchar) type
template <typename DstType, unsigned stride>
class inLineAssemblerCast<const float *, DstType *, false, stride> {
public:
  static __attribute__((always_inline)) unsigned
  singleCast(const float in, const float2 limits) {
    unsigned result;
    auto inFloat = static_cast<float>(in);
    asm volatile(
        R"l(  f32clamp $a0, %[inFloat], %[limits]
              f32int $a0, $a0, %[ROUNDING_MODE]
              f32toi32 $a0, $a0
              atom %[result], $a0
        )l"
        : [result] "=r"(result)
        : [inFloat] "r"(inFloat), [limits] "r"(limits),
          [ROUNDING_MODE] "i"(TFPU_ROUND_ZERO)
        : "$a0");
    return result;
  }

  static __attribute__((always_inline)) void loopBody(unsigned loopCount,
                                                      const float *inPtr,
                                                      DstType *outPtr,
                                                      float2 limits) {
    auto outPtrUnsigned = reinterpret_cast<unsigned *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      auto result0 = singleCast(inPtr[0], limits);
      auto result1 = singleCast(inPtr[1], limits);
      auto result2 = singleCast(inPtr[2], limits);
      auto result3 = singleCast(inPtr[3], limits);
      inPtr += 4 * stride;
      *outPtrUnsigned = combine8bit(result0, result1, result2, result3);
      outPtrUnsigned += stride;
    }
    return;
  }
};

// Case where SrcType is half and DstType is an 8-bit (char/uchar) type
template <typename DstType, unsigned stride>
class inLineAssemblerCast<const half *, DstType *, false, stride> {
public:
  static __attribute__((always_inline)) unsigned
  singleCast(const half in, const float2 limits) {
    unsigned result;
    auto inFloat = static_cast<float>(in);
    asm volatile(
        R"l(  f32clamp $a0, %[inFloat], %[limits]
              f32int $a0, $a0, %[ROUNDING_MODE]
              f32toi32 $a0, $a0
              atom %[result], $a0
        )l"
        : [result] "=r"(result)
        : [inFloat] "r"(inFloat), [limits] "r"(limits),
          [ROUNDING_MODE] "i"(TFPU_ROUND_ZERO)
        : "$a0");
    return result;
  }

  static __attribute__((always_inline)) void loopBody(unsigned loopCount,
                                                      const half *inPtr,
                                                      DstType *outPtr,
                                                      float2 limits) {
    castHalfUCharSCharLoop<DstType, stride>(inPtr, outPtr, loopCount, limits);
    return;
  }
};

template <typename SrcType, typename DstType, unsigned stride>
class inLineAssemblerCast<const SrcType *, DstType *, true, stride> {
public:
  static __attribute__((always_inline)) DstType singleCast(SrcType in,
                                                           float2 limits) {
    return static_cast<DstType>(in);
  }

  static __attribute__((always_inline)) void loopBody(unsigned loopCount,
                                                      const SrcType *inPtr,
                                                      DstType *outPtr,
                                                      float2 limits) {
    float2 *outPtr2 = reinterpret_cast<float2 *>(outPtr);
    half4 *outPtr4 = reinterpret_cast<half4 *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      DstType out0 = static_cast<DstType>(*inPtr++);
      DstType out1 = static_cast<DstType>(*inPtr++);
      DstType out2 = static_cast<DstType>(*inPtr++);
      DstType out3 = static_cast<DstType>(*inPtr++);
      if constexpr (std::is_same<DstType, float>::value) {
        *outPtr2++ = {out0, out1};
        *outPtr2-- = {out2, out3};
        outPtr2 += 2 * stride;
      } else {
        *outPtr4 = {out0, out1, out2, out3};
        outPtr4 += stride;
      }
      inPtr += (4 * stride - 4);
    }
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const unsigned char *, half *, true, stride> {
public:
  static __attribute__((always_inline)) half singleCast(unsigned char in,
                                                        float2 limits) {
    return static_cast<half>(in);
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const unsigned char *inPtr, half *outPtr,
           float2 limits) {
    castUCharHalfLoop<stride>(inPtr, outPtr, loopCount);
    return;
  }
};

#if __IPU_ARCH_VERSION__ == 21
template <unsigned stride>
class inLineAssemblerCast<const half *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) quarter
  singleCast(const half *in, float2 metadata0, float2 metadata1) {
    quarter result;
    half2 in2 = {*in, *in};
    asm volatile(
        R"l(  f16v2tof8 $a0, %[in2]
              atom %[result], $a0
        )l"
        : [result] "=r"(result)
        : [in2] "r"(in2)
        : "$a0");

    return result;
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const half *inPtr, quarter *outPtr,
           float2 metadata0, float2 metadata1) {
    castFpDemoteLoop<half, quarter, TemplateInstructions::f16v8tof8, stride>(
        inPtr, outPtr, loopCount);
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const quarter *, half *, true, stride> {
public:
  static __attribute__((always_inline)) half
  singleCast(const quarter *in, float2 metadata0, float2 metadata1) {
    half2 result;
    asm volatile(
        R"l(  ldb8 $a0, $mzero, %[in], 0
              f8v2tof16 %[result], $a0
        )l"
        : [result] "=r"(result)
        : [in] "r"(in)
        : "$a0");

    return result[0];
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, half *outPtr,
           float2 metadata0, float2 metadata1) {
    castFpPromoteLoop<quarter, half, TemplateInstructions::f8v4tof16, stride>(
        inPtr, outPtr, loopCount);
    return;
  }
};

template <TemplateInstructions instruction>
static __attribute__((always_inline)) unsigned
castQuarterToInt8(const quarter *in) {
  unsigned result;
  if constexpr (instruction == TemplateInstructions::f32toi32) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f32toi32 \OPERANDS
         .endm
        )l" ::
            :);
  } else if constexpr (instruction == TemplateInstructions::f32toui32) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f32toui32 \OPERANDS
         .endm
        )l" ::
            :);
  }
  asm volatile(
      R"l(
        ldb8 $a0, $mzero, %[in], 0
        f8v2tof16 $a0, $a0
        f16tof32 $a0, $a0
        cast $a0, $a0
        atom %[result], $a0

        // Remove macro definition to avoid later re-definition
        .purgem cast
        )l"
      : [result] "=r"(result)
      : [in] "r"(in)
      : "$a0");
  return result;
}

template <TemplateInstructions instruction>
static __attribute__((always_inline)) float
castQuarterToInt8V4(const float in) {
  float out;
  if constexpr (instruction == TemplateInstructions::f32toi32) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f32toi32 \OPERANDS
         .endm
        )l" ::
            :);
  } else if constexpr (instruction == TemplateInstructions::f32toui32) {
    asm volatile(
        R"l(.macro cast OPERANDS:vararg
            f32toui32 \OPERANDS
         .endm
        )l" ::
            :);
  }
  asm volatile(
      R"l(
         f8v4tof16  $a0:1, %[in]
         f16v2tof32 $a2:3, $a0
         cast $a2, $a2
         cast $a3, $a3
         f16v2tof32 $a0:1, $a1
         cast $a0, $a0
         cast $a1, $a1
         sort4x16lo $a0, $a0, $a1
         sort4x16lo $a2, $a2, $a3
         sort8x8lo  %[out], $a2, $a0

         .purgem cast
  )l"
      : [out] "=r"(out)
      : [in] "r"(in)
      : "$a0:1", "$a2:3");
  return out;
}

template <unsigned stride>
class inLineAssemblerCast<const quarter *, char *, true, stride> {
public:
  static __attribute__((always_inline)) char
  singleCast(const quarter *in, float2 metadata0, float2 metadata1) {
    return castQuarterToInt8<TemplateInstructions::f32toi32>(in);
  }
  static __attribute__((always_inline)) float vectorCast4(const float in) {
    return castQuarterToInt8V4<TemplateInstructions::f32toi32>(in);
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, char *outPtr,
           float2 metadata0, float2 metadata1) {
    const float *inPtr4 = reinterpret_cast<const float *>(inPtr);
    float *outPtr4 = reinterpret_cast<float *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      *outPtr4++ = vectorCast4(*inPtr4++);
      *outPtr4 = vectorCast4(*inPtr4);
      inPtr4 += 2 * stride - 1;
      outPtr4 += 2 * stride - 1;
    }
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const quarter *, unsigned char *, true, stride> {
public:
  static __attribute__((always_inline)) unsigned char
  singleCast(const quarter *in, float2 metadata0, float2 metadata1) {
    return castQuarterToInt8<TemplateInstructions::f32toui32>(in);
  }
  static __attribute__((always_inline)) float vectorCast4(const float in) {
    return castQuarterToInt8V4<TemplateInstructions::f32toui32>(in);
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, unsigned char *outPtr,
           float2 metadata0, float2 metadata1) {
    const float *inPtr4 = reinterpret_cast<const float *>(inPtr);
    float *outPtr4 = reinterpret_cast<float *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      *outPtr4++ = vectorCast4(*inPtr4++);
      *outPtr4 = vectorCast4(*inPtr4);
      inPtr4 += 2 * stride - 1;
      outPtr4 += 2 * stride - 1;
    }
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const quarter *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) void setFp8Config(float2 metadata) {
    __builtin_ipu_uput(metadata[0], CSR_W_FP_NFMT__INDEX & CSR_UPPER_MASK);
    __builtin_ipu_uput(metadata[1], CSR_W_FP_SCL__INDEX & CSR_UPPER_MASK);
  }

  static __attribute__((always_inline)) quarter
  singleCast(const quarter *in, float2 metadata0, float2 metadata1) {
    quarter result;
    setFp8Config(metadata0);
    asm volatile(
        R"l(  ldb8 $a0, $mzero, %[in], 0
              f8v2tof16 $a0, $a0
              uput $FP_SCL, %[scale]
              uput $FP_NFMT, %[format]
              f16v2tof8 $a0, $a0
              atom %[result], $a0
          )l"
        : [result] "=r"(result)
        : [in] "r"(in), [scale] "r"(metadata1[1]), [format] "r"(metadata1[0])
        : "$a0");

    return result;
  }

  static __attribute__((always_inline)) float2 vectorCast8(const float2 in,
                                                           float2 metadata) {
    float2 out;
    asm volatile(
        R"l( f8v4tof16  $a0:1, %[in0]
             f8v4tof16  $a2:3, %[in1]
             uput $FP_SCL, %[scale]
             uput $FP_NFMT, %[format]
             f16v8tof8  %[out], $a0:3
        )l"
        : [out] "=r"(out)
        : [in0] "r"(in[0]), [in1] "r"(in[1]), [scale] "r"(metadata[1]),
          [format] "r"(metadata[0])
        : "$a0:1", "$a2:3");
    return out;
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, quarter *outPtr,
           float2 metadata0, float2 metadata1) {
    const float2 *inPtr8 = reinterpret_cast<const float2 *>(inPtr);
    float2 *outPtr8 = reinterpret_cast<float2 *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      setFp8Config(metadata0);
      *outPtr8 = vectorCast8(*inPtr8, metadata1);
      inPtr8 += stride;
      outPtr8 += stride;
    }
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const char *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) quarter
  singleCast(const char *in, float2 metadata0, float2 metadata1) {
    quarter result;
    unsigned scratch;
    asm volatile(
        R"l(  lds8        $m0, $mzero, %[in], 0
              st32        $m0, $mzero, %[scratchPtr],0
              ld32        $a0, $mzero, %[scratchPtr],0
              f32fromi32  $a0, $a0
              f32tof16    $a0, $a0
              f16v2tof8   $a0, $a0
              atom        %[result], $a0
        )l"
        : [result] "=r"(result), [scratch] "=r"(scratch)
        : [in] "r"(in), [scratchPtr] "r"(&scratch)
        : "$a0", "$m0");

    return result;
  }

  static __attribute__((always_inline)) float vectorCast4(const char *in) {
    float2 out;
    unsigned scratch;
    asm volatile(
        R"l(  lds8        $m0, $mzero, %[in], 0
              st32        $m0, $mzero, %[scratchPtr],0
              ld32        $a0, $mzero, %[scratchPtr],0
              {lds8       $m0, $mzero, %[in], 1
               f32fromi32 $a0, $a0}
              st32        $m0, $mzero, %[scratchPtr],0
              ld32        $a1, $mzero, %[scratchPtr],0
              {lds8       $m0, $mzero, %[in], 2
               f32fromi32 $a1, $a1}
              st32        $m0, $mzero, %[scratchPtr],0
              ld32        $a2, $mzero, %[scratchPtr],0
              {lds8       $m0, $mzero, %[in], 3
               f32fromi32 $a2, $a2}
              st32        $m0, $mzero, %[scratchPtr],0
              ld32        $a3, $mzero, %[scratchPtr],0
              f32fromi32  $a3, $a3

              f32v4tof16  $a0:1, $a0:3
              mov         $a2:3, $azeros
              f16v8tof8   %[out], $a0:3
        )l"
        : [out] "=r"(out), [scratch] "=r"(scratch)
        : [in] "r"(in), [scratchPtr] "r"(&scratch)
        : "$a0:1", "$a2:3", "$m0");
    return out[0];
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const char *inPtr, quarter *outPtr,
           float2 metadata0, float2 metadata1) {
    float *outPtr4 = reinterpret_cast<float *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      *outPtr4++ = vectorCast4(inPtr);
      inPtr += 4;
      *outPtr4 = vectorCast4(inPtr);
      inPtr += 8 * stride - 4;
      outPtr4 += 2 * stride - 1;
    }
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const unsigned char *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) quarter
  singleCast(const unsigned char *in, float2 metadata0, float2 metadata1) {
    quarter result;
    asm volatile(
        R"l(  ldb8        $a0, $mzero, %[in], 0
              and         $a0, $a0, 0xff
              f32fromi32  $a0, $a0
              f32tof16    $a0, $a0
              f16v2tof8   $a0, $a0
              atom        %[result], $a0
        )l"
        : [result] "=r"(result)
        : [in] "r"(in)
        : "$a0");

    return result;
  }

  static __attribute__((always_inline)) float vectorCast4(float in) {
    float2 out;
    asm volatile(
        R"l(  and         $a0, %[in], 0xff
              f32fromi32  $a0, $a0
              roll8r      %[in], %[in], $azero
              and         $a1, %[in], 0xff
              f32fromi32  $a1, $a1
              roll8r      %[in], %[in], $azero
              and         $a2, %[in], 0xff
              f32fromi32  $a2, $a2
              roll8r      %[in], %[in], $azero
              and         $a3, %[in], 0xff
              f32fromi32  $a3, $a3

              f32v4tof16  $a0:1, $a0:3
              mov         $a2:3, $azeros
              f16v8tof8   %[out], $a0:3
        )l"
        : [out] "=r"(out), [in] "+r"(in)
        :
        : "$a0:1", "$a2:3");
    return out[0];
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const unsigned char *inPtr, quarter *outPtr,
           float2 metadata0, float2 metadata1) {
    const float *inPtr4 = reinterpret_cast<const float *>(inPtr);
    float *outPtr4 = reinterpret_cast<float *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      *outPtr4++ = vectorCast4(*inPtr4++);
      *outPtr4 = vectorCast4(*inPtr4);
      inPtr4 += 2 * stride - 1;
      outPtr4 += 2 * stride - 1;
    }
    return;
  }
};

// Setting Fp8 metadata:
// It's OK to write to reserved bits of $FP_NFMT, $FP_SCL
// We can't do many regular bit manipulations on the ARF register file
// instead:
// We can negate scale by doing a subtraction using denorm f16 bit arithmetic:
// negScaleIn6LSBs = 0x80 - scale
// To extract the `format` bit
// The `f32cmplt` instruction will effectively broadcast the sign bit
// as long as the input is a legal fp32 value.  This is ensured by
// masking out one of the exponent bits with `andc` even though one
// bit should be zero as it is unused

static __attribute__((always_inline)) float ldb8(const unsigned char *address) {
  float result;
  // TODO - Use intrinsic/builtin for this when one becomes available
  asm volatile(
      R"l(  ldb8 %[result], %[address], $mzero, 0
      )l"
      : [result] "=r"(result)
      : [address] "r"(address)
      :);
  return result;
}

static __attribute__((always_inline)) void
setFp8Config(const MetadataType *metadata) {
  float temp = ldb8(metadata);
  __builtin_ipu_uput(temp, CSR_W_FP_SCL__INDEX & CSR_UPPER_MASK);
  temp = __builtin_ipu_andc_f32(temp, 2.0f);
  temp = __builtin_ipu_cmplt(temp, 0.0f);
  __builtin_ipu_uput(temp, CSR_W_FP_NFMT__INDEX & CSR_UPPER_MASK);
}

static __attribute__((always_inline)) void
setFp8ConfigNegScale(const MetadataType *metadata) {
  auto scale = ldb8(metadata);
  // The value 2.0f = 0x40000000 so andc is an and with 0xbfffffff
  // This just makes sure that the exponent is not all 1's
  auto format = __builtin_ipu_cmplt(__builtin_ipu_andc_f32(scale, 2.0f), 0.0f);
  __builtin_ipu_uput(format, CSR_W_FP_NFMT__INDEX & CSR_UPPER_MASK);

  // TODO - Use intrinsics / builtins for these operations when we make
  // a method to initialise values that appear as bitmasks into $ARF registers
  // without converting to the float equivalent.
  asm volatile(
      R"l(  and  %[scale], %[scale], 0x3f
            setzi $a0, 0x80
            f16v2sub  %[scale], $a0, %[scale]
      )l"
      : [scale] "+r"(scale)
      :
      : "$a0");
  __builtin_ipu_uput(scale, CSR_W_FP_SCL__INDEX & CSR_UPPER_MASK);
}

static float2 extractMetadata(const MetadataType *metadata) {
  const auto scale = ldb8(metadata);
  auto format = __builtin_ipu_cmplt(__builtin_ipu_andc_f32(scale, 2.0f), 0.0f);
  return {format, scale};
}

static float2 extractMetadataNegScale(const MetadataType *metadata) {
  float2 result;
  result[1] = ldb8(metadata);
  result[0] =
      __builtin_ipu_cmplt(__builtin_ipu_andc_f32(result[1], 2.0f), 0.0f);
  asm volatile(
      R"l(  and       %[scale], %[scale], 0x3f
            setzi     $a0, 0x80
            f16v2sub  %[scale], $a0, %[scale]
      )l"
      : [scale] "+r"(result[1])
      :
      : "$a0");
  return result;
}

#else

// FP8 -> Half Mk2 Cast
// It's OK to overprocess because we have to deal with every byte input value of
// the initial quarter anyway.

template <unsigned stride>
class inLineAssemblerCast<const quarter *, half *, true, stride> {
public:
  static __attribute__((always_inline)) half4
  loopCast152(int loopCount, const quarter *in, half *out, float2 *metadata) {
    half4 lastResults;
    float2 stack[5];
    stack[0][0] = (*metadata)[1];    // scale
    stack[1][0] = 2.66851606673e+36; // 0x7c007c00 exponent mask
    stack[1][1] = 2.66851606673e+36; // 0x7c007c00 exponent mask
    stack[2][0] = 1.04226232351e+34; // 0x78007800 exponent offset
    stack[2][1] = 1.04226232351e+34; // 0x78007800 exponent offset

    // Cast fp8(152) -> half:
    // Shuffle into 16 bit fields abcdefgh => ab00cd00 ef00gh00
    // Temporary shift and compare with 0x08 shifted nan fp8 mask
    // Isolate mantisa with sign bits and mantisa
    // Check that the value is not denormalized
    // If so, mark as zero and merge to sign and mantissa
    // Multiply the values by 2^scale
    asm volatile(
        R"l(
              .equ stackOffsetMtoA6, 6
              .equ stackOffsetMtoA7, 7
              .equ stackOffsetMtoA2, 8
              .equ stackOffsetMtoA3, 9
              .equ stackOffsetA67, 3
              .equ stackOffsetA23, 4
              .equ stackOffsetExpMask, 1
              .equ stackOffsetExpOff, 2
              .equ stackOffsetScale, 0
              {
                ld32step      $m7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                uput   $FP_CTL, $azero   // disable FP exceptions
              }
              shuf8x8lo       $m8, $mzero, $m7
              shuf8x8hi       $m7, $mzero, $m7
              st32            $m8, %[stackPtr], $mzero, stackOffsetMtoA6
              st32            $m7, %[stackPtr], $mzero, stackOffsetMtoA7
              // Shift, preserve all 8 bits, but make 0x8000 -> 0x0800
              // So we can compare without hitting a -0.0 == 0.0 issue
              shr             $m8, $m8, 4
              shr             $m7, $m7, 4
              st32            $m8, %[stackPtr], $mzero, stackOffsetMtoA2
              st32            $m7, %[stackPtr], $mzero, stackOffsetMtoA3

              {
                ld64          $a6:7, %[stackPtr], $mzero, stackOffsetA67
                setzi         $a4, 0x3800
              }
              ld64            $a2:3, %[stackPtr], $mzero, stackOffsetExpMask
              bri             1f
            2:
              st64step        %[lastResults], $mzero, %[baseOut]+=, %[ctxtWorkers]
            1:
              {
                ld32step      $m7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                andc64        %[lastResults], $a6:7, $a2:3
              }
              {
                shuf8x8lo     $m8, $mzero, $m7
                andc64        $a2:3, $a2:3, $a6:7
              }
              {
                shuf8x8hi     $m7, $mzero, $m7
                f16v4cmpeq    $a2:3, $azeros, $a2:3
              }
              {
                ld64          $a4:5, %[stackPtr], $mzero, stackOffsetExpOff
                f16v4mul      $a6:7, $a4:BL, $a6:7
              }
              {
                st32          $m8, %[stackPtr], $mzero, stackOffsetMtoA6
                or64          %[lastResults],%[lastResults], $a4:5
              }
              {
                st32          $m7, %[stackPtr], $mzero, stackOffsetMtoA7
                andc64        $a6:7, $a6:7, $a2:3
              } 
              and64           %[lastResults], %[lastResults], $a2:3
              {
                ld64          $a2:3, %[stackPtr], $mzero, stackOffsetA23
                or64          %[lastResults], $a6:7, %[lastResults]
              }
              {
                shr           $m8, $m8, 4
                setzi         $a6, 0x0800
              }
            {
                shr           $m7, $m7, 4
                f16v4cmpeq    $a2:3, $a6:BL, $a2:3
              }
              {
                ld32         $a2, %[stackPtr], $mzero, stackOffsetScale
                or64         %[lastResults], $a2:3, %[lastResults]
              }             
              {
                st32         $m8, %[stackPtr], $mzero, stackOffsetMtoA2
                f32fromi32   $a2, $a2
              }
              {
                st32         $m7, %[stackPtr], $mzero, stackOffsetMtoA3    
                f32tof16     $a2, $a2
              }
              {
                ld64         $a6:7, %[stackPtr], $mzero, stackOffsetA67
                f16v2exp2    $a2, $a2
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetExpMask
                f16v4mul     %[lastResults], $a2:BL, %[lastResults]
              }
              {
                brnzdec      %[count], 2b
                setzi        $a4, 0x3800
              }
          )l"
        : [lastResults] "=r"(lastResults)
        : [baseIn] "r"(in), [baseOut] "r"(out), [count] "r"(loopCount),
          [ctxtWorkers] "r"(stride), [stackPtr] "r"(&stack)
        : "$a2:3", "$a4:5", "$a6:7", "$m7", "$m8", "memory");
    return lastResults;
  }

  static __attribute__((always_inline)) half4
  loopCast143(int loopCount, const quarter *in, half *out, float2 *metadata) {
    half4 lastResults;
    half2 bias = {7.0, 7.0};
    float2 stack[3];
    stack[0][0] = 2.83326107615e+14;  // 0x5780d780 clampfp16
    stack[0][1] = -1.18010406225e-38; // 0x80808080 sign mask
    stack[1][0] = 2.66851606673e+36;  // 0x7c007c00 exponent mask
    stack[1][1] = 2.66851606673e+36;  // 0x7c007c00 exponent mask
    stack[2][0] = (*metadata)[1];     // scale
    stack[2][1] = 6.01853466353e-36;  // -0.0 fp16 class 0x05000005

    // Cast fp8(143) -> half:
    // Shuffle into 16 bit fields abcdefgh => ab00cd00 ef00gh00
    // Isolate sign then shift right by 1 and recombine signs
    // Check that the values are not negative zeros (using f16v4class)
    // which is equivalent to nan in fp16
    // Clamp values to limits the range +/- 120, then Mark as 0xffff if no
    // clamp else 0x0000
    // Multiply the values by 2^(scale + bias)
    asm volatile(
        R"l(
              .equ mask12L, 0xfff
              .equ mask12U, 0xfff00000
              .equ maskClassResult ,0xff0000ff
              .equ cmpClassResult, 0x05000005

              .equ stackOffsetMtoA6, 0
              .equ stackOffsetMtoA7, 1
              .equ stackOffsetA67, 0
              .equ stackOffsetScale, 4
              .equ stackOffsetSignMask, 1
              .equ stackOffsetExpMask, 1
              .equ stackOffsetClampfp16, 0
              .equ stackOffsetFp16MZeroClass, 5

              ld32           $a5, %[stackPtr], $mzero, stackOffsetScale
              {
                ld32step     $m7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                f32fromi32   $a5, $a5
              }
              {
                ld32         $m2, %[stackPtr], $mzero, stackOffsetSignMask
                f32tof16     $a5, $a5
              }
              {
                ld64         %[lastResults], %[stackPtr], $mzero, stackOffsetExpMask
                f16v2add     $a5, $a5, %[bias]
              }
              {
                ld32         $a5, %[stackPtr], $mzero, stackOffsetClampfp16
                f16v2exp2    %[bias], $a5 // prepare scale
              }
              {
                and          $m4, $m7, $m2
                uput           $FP_CTL, $azero   // disable FP exceptions
              }
              xor            $m7, $m7, $m4
              shuf8x8lo      $m5, $mzero, $m4
              shuf8x8hi      $m4, $mzero, $m4
              shuf8x8lo      $m8, $mzero, $m7
              shuf8x8hi      $m7, $mzero, $m7
              shr            $m8, $m8, 0x1
              shr            $m7, $m7, 0x1
              or             $m8, $m8, $m5
              or             $m7, $m7, $m4
              st32           $m7, %[stackPtr], $mzero, stackOffsetMtoA7

              bri            1f
            2:
              st64step       $a0:1, $mzero, %[baseOut]+=, %[ctxtWorkers]
            1:
              {
                ld32step     $m7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                or           $a0, $azero, maskClassResult & mask12L
              }
              { 
                st32         $m8, %[stackPtr], $mzero, stackOffsetMtoA6
                or           $a0, $a0, maskClassResult & mask12U
              }
              {
                ld64         $a6:7, %[stackPtr], $mzero, stackOffsetA67
                mov          $a1, $a0
              }
              // Produces 0x05 for the value we want in 4 consecutive bytes:
              // 0xwwxxyyzz
              {
                and          $m4, $m7, $m2
                f16v4class   $a7, $a6:7
              }
              {
                xor          $m7, $m7, $m4
                sort4x16lo   $a6, $a7, $a7  //0xyyzzyyzz
              }
              {
                shuf8x8lo    $m5, $mzero, $m4
                sort4x16hi   $a7, $a7, $a7  //0xwwxxwwxx
              }
              {
                shuf8x8hi    $m4, $mzero, $m4
                and64        $a0:1, $a0:1, $a6:7  //$a1:0xww0000xx $a0:0xyy0000zz
              }
              shuf8x8lo      $m8, $mzero, $m7
              ld32           $a6, %[stackPtr], $mzero, stackOffsetFp16MZeroClass
              {
                shuf8x8hi    $m7, $mzero, $m7
                mov          $a7, $a6
              }
              // Reload as it is trampled
              // Compare: any byte left that =0x05 => 0xffff (==nan)
              {
                ld64         $a6:7, %[stackPtr], $mzero, stackOffsetA67
                f16v4cmpeq   $a0:1, $a0:1, $a6:7
              }
              {
                shr          $m8, $m8, 0x1
                or64         $a6:7, $a6:7, $a0:1
              }
              {
                shr          $m7, $m7, 0x1
                f16v4clamp   $a0:1, $a6:7, $a5
              }
              {
                or           $m8, $m8, $m5
                f16v4cmpeq   $a0:1, $a6:7, $a0:1
              }
              {
                or           $m7, $m7, $m4
                andc64       $a0:1, %[lastResults], $a0:1
              }
              {
                st32         $m7, %[stackPtr], $mzero, stackOffsetMtoA7
                or64         $a0:1, $a6:7, $a0:1
              }
              {
                brnzdec      %[count], 2b
                f16v4mul     $a0:1, %[bias]:BL, $a0:1         // Scale values
              }
              mov            %[lastResults], $a0:1
          )l"
        : [bias] "+r"(bias), [lastResults] "=r"(lastResults)
        : [baseIn] "r"(in), [baseOut] "r"(out), [count] "r"(loopCount),
          [ctxtWorkers] "r"(stride), [stackPtr] "r"(stack)
        : "$a0:1", "$a5", "$a6:7", "$m2", "$m4", "$m5", "$m7", "$m8", "memory");
    return lastResults;
  }
};
// Setting Fp8 metadata:
// Set the format to 0 if 152, otherwise 0.
// Convert the scale value to i32.

static void extractMetadata(const MetadataType *metadata,
                            float2 *unpackedMetadata) {
  uint32_t scaleSignMask = 0x3FFFFFF;
  asm volatile(
      R"l(  ldz8      $m0,  %[metadata], $mzero, 0
            
            and       $m1, $m0, 0x80
            shl       $m1, $m1, 23
            st32      $m1, %[unpackedPtr], $mzero, 0

			      and $m1, $m0, 0x20
            // 0x7FFFFFE0 mask if scale sign bit is active, else 0x0
            mul       $m2, $m1, %[scaleSignMask]
            shl       $m1, $m1, 0x1a
            and       $m0, $m0, 0x1f

            or        $m0, $m0, $m1
            or        $m0, $m0, $m2
            st32      $m0, %[unpackedPtr], $mzero, 0x1
      )l" ::[metadata] "r"(metadata),
      [unpackedPtr] "r"(unpackedMetadata), [scaleSignMask] "r"(scaleSignMask)
      : "$m0", "$m1", "$m2", "memory");
}

#endif
