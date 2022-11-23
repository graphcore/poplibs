// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <ipu_builtins.h>

// Flag value for rounding mode
#define TFPU_ROUND_ZERO 3

enum class TemplateInstructions {
  f32toi32,
  f32toui32,
  f32v4tof16,
  f32v4tof8Pseudo,
  f16v8tof8,
  f16v2tof32,
  f8v4tof16,
  f8v8tof32Pseudo
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
// FLOAT => HALF or HALF => QUARTER or FLOAT => QUARTER
template <typename SrcType, typename DstType, TemplateInstructions instruction,
          unsigned stride>
static __attribute__((always_inline)) void
castFpDemoteLoop(const SrcType *inPtr, DstType *outPtr, unsigned loopCount,
                 float *scale = nullptr) {
  if constexpr (instruction == TemplateInstructions::f32v4tof16 ||
                instruction == TemplateInstructions::f16v8tof8) {
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
        :
        [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
        : [STRIDE] "i"(stride)
        : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");
  } else if constexpr (instruction == TemplateInstructions::f32v4tof8Pseudo) {
    float2 stack[4];
    uint32_t *stackValues = (uint32_t *)(&stack[0]);

    // Note that the constants in `stackValues[2]` and `stackValues[3]` with
    // value 0x00040000 are used as arbitrary but valid fp16 inputs to
    // `f16v8tof8`. This is in addition to their use to mask the lowest bits
    // of the fp32 mantissa.
    stackValues[0] = 0x00008000; // Mantissa Bit 15
    stackValues[1] = 0x00008000; // Mantissa Bit 15
    stackValues[2] = 0x00007fff; // Lower 15 mantissa bit mask
    stackValues[3] = 0x00007fff; // Lower 15 mantissa bit mask
    stack[2][0] = *scale;

    // See the comments in `inLineAssemblerCast<const float *, quarter *, true,
    // stride>::singleCast()` for the implementation principles.
    asm volatile(
        R"l(
            .equ stackOffsetMantissaBit15, 0
            .equ stackOffsetLower15BitMask, 8
            .equ stackOffsetScale, 16
            .equ stackOffsetFirst2xFloat, 24

            brnzdec   %[loopCount], 3f
            bri       4f
          .align 8
            nop
          3:
            ld32 $a6, %[stack], $mzero, stackOffsetScale/4
            ld64 $a2:3, %[stack], $mzero, stackOffsetLower15BitMask/8
            ld64step $a0:1, $mzero, %[inPtr]+=,1
            f32v2mul $a0:1, $a6:B, $a0:1
            andc64 $a4:5, $a0:1, $a2:3
            {ld64 $a2:3, %[stack], $mzero, stackOffsetMantissaBit15/8
              f32v2cmpne $a0:1, $a0:1, $a4:5}
            {ld64step $a0:1, $mzero, %[inPtr]+=,2*%[STRIDE] -1
              and64 $a2:3, $a0:1, $a2:3}
            {ld64 $a2:3, %[stack], $mzero, stackOffsetLower15BitMask/8
              or64 $a4:5, $a4:5, $a2:3}
            f32v2mul $a0:1, $a6:B, $a0:1
            {ld64 $a2:3, %[stack], $mzero, stackOffsetMantissaBit15/8
              andc64 $a6:7, $a0:1, $a2:3}
            {st64 $a4:5, %[stack], $mzero, stackOffsetFirst2xFloat/8
              f32v2cmpne $a0:1, $a0:1, $a6:7}
            and64 $a4:5, $a0:1, $a2:3
            {ld64 $a4:5, %[stack], $mzero, stackOffsetFirst2xFloat/8
              or64 $a6:7, $a6:7, $a4:5}
            {ld32 $a6, %[stack], $mzero, stackOffsetScale/4
              f32v4tof16 $a0:1, $a4:7}
            {ld64 $a2:3, %[stack], $mzero, stackOffsetLower15BitMask/8
              f16v8tof8 $a4:5, $aq0}

            // Noted that $a2 and $a3 are each loaded with 0x00008000
            // which is interpreted in the following command as 2 valid half
            // values represented by 0x0000 and 0x8000.
            {rpt %[loopCount], (2f-1f)/8 -1
              fnop}
          1:
              {ld64step $a0:1, $mzero, %[inPtr]+=,1
                fnop}
              {st32step $a4, $mzero, %[outPtr]+=, %[STRIDE]
                f32v2mul $a0:1, $a6:B, $a0:1}
              {ld32 $a6, %[stack], $mzero, stackOffsetScale/4
                andc64 $a4:5, $a0:1, $a2:3}
              {ld64 $a2:3, %[stack], $mzero, stackOffsetMantissaBit15/8
                f32v2cmpne $a0:1, $a0:1, $a4:5}
              {ld64step $a0:1, $mzero, %[inPtr]+=,2*%[STRIDE] -1
                and64 $a2:3, $a0:1, $a2:3}
              {ld64 $a2:3, %[stack], $mzero, stackOffsetLower15BitMask/8
                or64 $a4:5, $a4:5, $a2:3}
              {nop
                f32v2mul $a0:1, $a6:B, $a0:1}
              {ld64 $a2:3, %[stack], $mzero, stackOffsetMantissaBit15/8
                andc64 $a6:7, $a0:1, $a2:3}
              {st64 $a4:5, %[stack], $mzero, stackOffsetFirst2xFloat/8
                f32v2cmpne $a0:1, $a0:1, $a6:7}
              {nop
                and64 $a4:5, $a0:1, $a2:3}
              {ld64 $a4:5, %[stack], $mzero, stackOffsetFirst2xFloat/8
                or64 $a6:7, $a6:7, $a4:5}
              {ld32 $a6, %[stack], $mzero, stackOffsetScale/4
                f32v4tof16 $a0:1, $a4:7}

              // Noted that $a2 and $a3 are each loaded with 0x00008000
              // which is interpreted in the following command as 2 valid half
              // values represented by 0x0000 and 0x8000.
              {ld64 $a2:3, %[stack], $mzero, stackOffsetLower15BitMask/8
                f16v8tof8 $a4:5, $aq0}
          2:
              st32step $a4, $mzero, %[outPtr]+=, 0
          4:
      )l"
        :
        [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
        : [STRIDE] "i"(stride), [stack] "r"(stack)
        : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
  }
}

// Optimized pipelined processing for multiple of 4 elements
// HALF => FLOAT or QUARTER => HALF or QUARTER => FLOAT
template <typename SrcType, typename DstType, TemplateInstructions instruction,
          unsigned stride>
static __attribute__((always_inline)) void
castFpPromoteLoop(const SrcType *inPtr, DstType *outPtr, unsigned loopCount,
                  float *scale = nullptr) {
  if constexpr (instruction == TemplateInstructions::f16v2tof32 ||
                instruction == TemplateInstructions::f8v4tof16) {
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
        :
        [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
        : [STRIDE] "i"(stride)
        : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");
  } else if constexpr (instruction == TemplateInstructions::f8v8tof32Pseudo) {
    // See the comments in `inLineAssemblerCast<const quarter *, float *, true,
    // stride>::singleCast()` for the implementation principles.
    asm volatile(
        R"l(
          brnzdec   %[loopCount], 3f
              bri       4f
          .align 8
          3:
            ld64step $a0:1, $mzero, %[inPtr]+=, %[STRIDE]
            f8v4tof16 $a2:3, $a0
            f16v2tof32 $a4:5, $a2
            f32v2mul $a4:5, %[scale]:B, $a4:5
            {st64step $a4:5, $mzero, %[outPtr]+=, 1
              f16v2tof32 $a4:5, $a3}
            {rpt %[loopCount], (2f-1f)/8 -1;
              f32v2mul $a4:5, %[scale]:B, $a4:5}
          1:{ld64step $a0:1, $mzero, %[inPtr]+=, %[STRIDE]
              f8v4tof16 $a2:3, $a1}
            {st64step $a4:5, $mzero, %[outPtr]+=, 1
              f16v2tof32 $a4:5, $a2}
            {nop
              f32v2mul $a4:5, %[scale]:B, $a4:5}
            {st64step $a4:5, $mzero, %[outPtr]+=, 1
              f16v2tof32 $a4:5, $a3}
            {nop
              f32v2mul $a4:5, %[scale]:B, $a4:5}
            {st64step $a4:5, $mzero, %[outPtr]+=, 1
              f8v4tof16 $a2:3, $a0}
            {nop
              f16v2tof32 $a4:5, $a2}
            {ld64step $azeros, $mzero, %[outPtr]+=, 4* %[STRIDE]-4
              f32v2mul $a4:5, %[scale]:B, $a4:5}
            {st64step $a4:5, $mzero, %[outPtr]+=, 1
              f16v2tof32 $a4:5, $a3}
            {nop
              f32v2mul $a4:5, %[scale]:B, $a4:5}
          2:
            {st64step $a4:5, $mzero, %[outPtr]+=, 1
              f8v4tof16 $a2:3, $a1}
            f16v2tof32 $a4:5, $a2
            f32v2mul $a4:5, %[scale]:B, $a4:5
            {st64step $a4:5, $mzero, %[outPtr]+=, 1
              f16v2tof32 $a4:5, $a3}
            f32v2mul $a4:5, %[scale]:B, $a4:5
            st64step $a4:5, $mzero, %[outPtr]+=, 0
          4:
      )l"
        :
        [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
        : [STRIDE] "i"(stride), [scale] "r"(*scale)
        : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "$a4:5", "memory");
  }
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

#if __IPU_ARCH_VERSION__ >= 21
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

template <unsigned stride>
class inLineAssemblerCast<const float *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) quarter
  singleCast(const float *in, float2 metadata0, float2 metadata1) {
    quarter result;
    float negScale1 = metadata1[1];

    // Accurate float to quarter cast, rounding to nearest, ties to even.
    // ------------------------------------------------------------------
    // In the following explanation quarter-f143 format is used without loss of
    // generality. The same principle should apply to quarter-f152. Machine
    // instructions do not exist to directly convert from float to quarter,
    // but can be done in the following steps.
    //  1. Scale float by pow(2.0f, -metadataScale)
    //  2. convert from float to half, taking rounding into account.
    //  3. convert from half to quarter (either f143 or f152 format) with unit
    //     metadata scale.
    //
    // The following table shows some example conversions of float to half and
    // followed by conversion to quarter-f143. Notice that some half numbers are
    // equidistant from the two adjacent quarter values. This condition is
    // called a tie. For example 8.5 is exactly equidistant from quarter values
    // 0x58 and 0x59. When a tie occurs the value is rounded to the nearest even
    // quarter number, 0x58 in this case.
    //
    //  Value         float         half        f143
    //  7.0           0x40e00000    0x4700      0x56
    //  7.25          0x40e80000    0x4740      0x56
    //  7.5           0x40f00000    0x4780      0x57
    //  7.75          0x40f80000    0x47c0      0x58
    //  8.0           0x41000000    0x4800      0x58
    //  8.5           0x41080000    0x4840      0x58
    //  9.0           0x41100000    0x4880      0x59
    //  9.5           0x41180000    0x4900      0x5a
    //
    // Consider the value 8.503906250 that is slightly higher than 8.5 with
    // the 11th significant mantissa bit set. This value should be rounded
    // to the nearest quarter-f143 value 0x59. However converting to an
    // intermediate type half would arrive at 0x58 as explained below.
    //
    // The half format has 10 mantissa bits, so 8.503906250 which has the 11th
    // mantissa bit set (and all lesser significant bits zero) happens to be
    // equidistant from the two adjacent half values and so on casting to
    // half it wil get rounded to the even half value, 0x4840 in this case.
    // This value converts to 0x58.
    //
    //  Value        float        half       f143-actual f143-expected
    //  8.503906250  0x41081000   0x4840     0x58        0x59
    //
    // The above example shows that rounding must be suppressed when casting
    // from float to half in step 2. This can be done by masking the least
    // significant 13 bits of the float value. However this could lead to
    // a float value with non-zero least significant 13 bits falsely appear
    // to be a tie between two adjacent quarter values. This can be avoided
    // by ensuring that bit 10 is non-zero if the least significant 13 bits
    // are non-zero.
    //
    // When a floating point value casts to a Half denorm value there is a loss
    // of 2 bits of precision.
    //  - The implicit leading 1 bit of the float number in the normal range.
    //  - The half bias is short by 1 bit compard to quarter-f152.
    // Due to the abvoe considerations a 15 bit mask is used (instead of 13).
    float mantissaBit15 = 4.5917748079e-41;
    float lowest15BitMask = mantissaBit15 - 1e-45f;
    asm volatile(
        R"l(
              // Scale fp32 value by multiplying by pow(2.0f, -scale)
              f32mul $a1, %[in], %[scale]

              // Clear lowest significant bits.
              andc $a2, $a1, %[lowest15BitMask]

              // Set bit 10 if the lowest significant 15 bits are non-zero.
              f32cmpne $a3, $a1, $a2
              and $a3, %[mantissaBit15], $a3
              or $a1, $a2, $a3

              // Cast to half here. Since the least significant bits are
              // cleared the IPU should not do any rounding. Bit 15 contains
              // the information about whether any of the lowest bits were
              // non-zero. This information will help the following
              // half->quarter casting instruction from falsely detecting a tie
              // in the distance between the scaled fp32 value and the two
              // nearest quarter representations, as explained in the comment
              // preceding the asm block.
              f32tof16 $a1, $a1
              f16v2tof8 $a0, $a1
              atom %[result], $a0
        )l"
        : [result] "=r"(result)
        : [scale] "r"(negScale1), [in] "r"(*in),
          [mantissaBit15] "r"(mantissaBit15),
          [lowest15BitMask] "r"(lowest15BitMask)
        : "$a0", "$a1", "$a2", "$a3");
    return result;
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const float *inPtr, quarter *outPtr,
           float2 metadata0, float2 metadata1) {
    float scale = metadata1[1];
    castFpDemoteLoop<float, quarter, TemplateInstructions::f32v4tof8Pseudo,
                     stride>(inPtr, outPtr, loopCount, &scale);
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const quarter *, float *, true, stride> {
public:
  static __attribute__((always_inline)) float
  singleCast(const quarter *in, float2 metadata0, float2 metadata1) {
    float scale0 = metadata0[1];
    float result;
    // Accurate quarter to float cast.
    // -------------------------------
    // Machine instructions do not exist to directly convert from quarter to
    // float, but can be done in the following steps.
    //  1. convert from quarter to half (either f143 or f152 format) with unit
    //     metadata scale. The range of quarter is a subset of the range of
    //     half as shown below.
    //       half ~ [2^-24, 2^15]
    //       quarter f143 ~ [2^-10, 2^7]
    //       quarter f152 ~ [2^-17, 2^15]
    //  2. convert from half to float.
    //  3. Scale float by pow(2.0f, metadataScale)
    asm volatile(
        R"l(  ldb8 $a0, $mzero, %[in], 0
              f8v2tof16 $a1, $a0
              f16tof32 $a1, $a1
              f32mul %[result], $a1, %[scale]
        )l"
        : [result] "=r"(result)
        : [scale] "r"(scale0), [in] "r"(in)
        : "$a0", "$a1");

    return result;
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, float *outPtr,
           float2 metadata0, float2 metadata1) {
    float scale = metadata0[1];
    castFpPromoteLoop<quarter, float, TemplateInstructions::f8v8tof32Pseudo,
                      stride>(inPtr, outPtr, loopCount, &scale);
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
setFp8Config(const MetadataType &metadata) {
  float temp = ldb8(&metadata);
  __builtin_ipu_uput(temp, CSR_W_FP_SCL__INDEX & CSR_UPPER_MASK);
  temp = __builtin_ipu_andc_f32(temp, 2.0f);
  temp = __builtin_ipu_cmplt(temp, 0.0f);
  __builtin_ipu_uput(temp, CSR_W_FP_NFMT__INDEX & CSR_UPPER_MASK);
}

static __attribute__((always_inline)) void
setFp8ConfigNegScale(const MetadataType &metadata) {
  auto scale = ldb8(&metadata);
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
static __attribute__((always_inline)) float
getScaleFloat(const MetadataType metadata) {
  signed char scaleExp =
      (metadata & 0x20) ? (metadata | 0xc0) : (metadata & 0x3f);
  return static_cast<float>(scaleExp);
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

#define ASSIGN_HEX_2_FLOAT(f, h, v)                                            \
  h = v;                                                                       \
  f = *((float *)&h);

#define ASSIGN_HEX_2_DOUBLE(f, h, v)                                           \
  h = v;                                                                       \
  f[0] = *((float *)&h);                                                       \
  f[1] = *((float *)&h);

// FP8 -> Half Mk2 Cast
// It's OK to overprocess because we have to deal with every byte input value of
// the initial quarter anyway.

template <unsigned stride>
class inLineAssemblerCast<const quarter *, half *, true, stride> {
public:
  static __attribute__((always_inline)) half4
  loopCast152(int loopCount, const quarter *in, half *out, float2 *metadata) {
    unsigned hexValue;
    half4 lastResults;
    float2 stack[4];

    stack[0][0] = (*metadata)[1];                        // scale
    ASSIGN_HEX_2_DOUBLE(stack[1], hexValue, 0x78007800); // exponent offset
    // stack[2] and stack[3] are used for scratch

    // Cast fp8(152) -> half:
    // Shuffle into 16 bit fields abcdefgh => ab00cd00 ef00gh00
    // Temporary shift and compare with 0x08 shifted nan fp8 mask
    // Isolate mantisa with sign bits and mantisa
    // Check that the value is not denormalized
    // If so, mark as zero and merge to sign and mantissa
    // Multiply the values by 2^scale
    asm volatile(
        R"l(
              .equ expMask, 0x7c00

              .equ stackOffsetScale, 0
              .equ stackOffsetExpOff, 8
              .equ stackOffsetA67, 16
              .equ stackOffsetMtoA6, 16
              .equ stackOffsetMtoA7, 20
              .equ stackOffsetA23, 24
              .equ stackOffsetMtoA2, 24
              .equ stackOffsetMtoA3, 28
              {
                ld32step      $m7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                uput          $FP_CTL, $azero   // disable FP exceptions
              }
              ld32            $a2, %[stackPtr], $mzero, stackOffsetScale/4
              {
                shuf8x8lo     $m8, $mzero, $m7
                f32fromi32    $a2, $a2
              }
              {
                shuf8x8hi     $m7, $mzero, $m7
                f32tof16      $a2, $a2
              }
              {
                st32          $m8, %[stackPtr], $mzero, stackOffsetMtoA6/4
                f16v2exp2     $a2, $a2
              }
              st32            $a2, %[stackPtr], $mzero, stackOffsetScale/4
              st32            $m7, %[stackPtr], $mzero, stackOffsetMtoA7/4
              // Shift, preserve all 8 bits, but make 0x8000 -> 0x0800
              // So we can compare without hitting a -0.0 == 0.0 issue
              shr             $m8, $m8, 4
              shr             $m7, $m7, 4
              {
                st32          $m8, %[stackPtr], $mzero, stackOffsetMtoA2/4
                setzi         $a2, expMask
              }
              {
                st32          $m7, %[stackPtr], $mzero, stackOffsetMtoA3/4
                sort4x16lo    $a2, $a2, $a2
              }
              {
                ld64          $a6:7, %[stackPtr], $mzero, stackOffsetA67/8
                setzi         $a4, 0x3800
              }
              {
                bri           1f
                mov           $a3, $a2
              }
            2:
              {
                st64step      %[lastResults], $mzero, %[baseOut]+=, %[ctxtWorkers]
                mov           $a3, $a2
              }
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
                ld64          $a4:5, %[stackPtr], $mzero, stackOffsetExpOff/8
                f16v4mul      $a6:7, $a4:BL, $a6:7
              }
              {
                st32          $m8, %[stackPtr], $mzero, stackOffsetMtoA6/4
                or64          %[lastResults],%[lastResults], $a4:5
              }
              {
                st32          $m7, %[stackPtr], $mzero, stackOffsetMtoA7/4
                andc64        $a6:7, $a6:7, $a2:3
              }
              and64           %[lastResults], %[lastResults], $a2:3
              {
                ld64          $a2:3, %[stackPtr], $mzero, stackOffsetA23/8
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
                ld32          $a2, %[stackPtr], $mzero, stackOffsetScale/4
                or64          %[lastResults], $a2:3, %[lastResults]
              }
              {
                st32          $m8, %[stackPtr], $mzero, stackOffsetMtoA2/4
                f16v4mul      %[lastResults], $a2:BL, %[lastResults]
              }
              {
                st32          $m7, %[stackPtr], $mzero, stackOffsetMtoA3/4
                setzi         $a2, expMask
              }
              {
                ld64          $a6:7, %[stackPtr], $mzero, stackOffsetA67/8
                sort4x16lo    $a2, $a2, $a2
              }
              {
                brnzdec       %[count], 2b
                setzi         $a4, 0x3800
              }
          )l"
        : [lastResults] "=r"(lastResults), [baseIn] "+r"(in),
          [baseOut] "+r"(out), [count] "+r"(loopCount)
        : [ctxtWorkers] "r"(stride), [stackPtr] "r"(&stack)
        : "$a2:3", "$a4:5", "$a6:7", "$m7", "$m8", "memory");
    return lastResults;
  }

  static __attribute__((always_inline)) half4
  loopCast143(int loopCount, const quarter *in, half *out, float2 *metadata) {
    unsigned hexValue;
    half4 lastResults;
    half2 bias = {7.0, 7.0};
    float2 stack[3];

    ASSIGN_HEX_2_FLOAT(stack[0][0], hexValue, 0x5780d780); // clampfp16
    ASSIGN_HEX_2_FLOAT(stack[0][1], hexValue, 0x80808080); // sign mask
    ASSIGN_HEX_2_DOUBLE(stack[1], hexValue, 0x7c007c00);   // exponent mask
    ASSIGN_HEX_2_FLOAT(stack[2][0], hexValue, 0x05000005); // fp16 class

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
              .equ maskClassResult, 0xff0000ff

              // Clampfp16 and SignMask are loaded once and then used as a
              // scratch
              .equ stackOffsetClampfp16, 0
              .equ stackOffsetMtoA6, 0
              .equ stackOffsetA67, 0
              .equ stackOffsetSignMask, 4
              .equ stackOffsetMtoA7, 4
              .equ stackOffsetExpMask, 8
              .equ stackOffsetFp16MZeroClass, 16

              {
                ld32step     $m7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                f32fromi32   %[scaleOrClamp], %[scaleOrClamp]
              }
              {
                ld32         $m2, %[stackPtr], $mzero, stackOffsetSignMask/4
                f32tof16     %[scaleOrClamp], %[scaleOrClamp]
              }
              {
                ld64         %[lastResults], %[stackPtr], $mzero, stackOffsetExpMask/8
                f16v2add     %[scaleOrClamp], %[scaleOrClamp], %[bias]
              }
              {
                ld32         %[scaleOrClamp], %[stackPtr], $mzero, stackOffsetClampfp16/4
                f16v2exp2    %[bias], %[scaleOrClamp] // prepare scale
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
              st32           $m7, %[stackPtr], $mzero, stackOffsetMtoA7/4

              bri            1f
            2:
              st64step       $a0:1, $mzero, %[baseOut]+=, %[ctxtWorkers]
            1:
              {
                ld32step     $m7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                or           $a0, $azero, maskClassResult & mask12L
              }
              {
                st32         $m8, %[stackPtr], $mzero, stackOffsetMtoA6/4
                or           $a0, $a0, maskClassResult & mask12U
              }
              {
                ld64         $a6:7, %[stackPtr], $mzero, stackOffsetA67/8
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
              ld32           $a6, %[stackPtr], $mzero, stackOffsetFp16MZeroClass/4
              {
                shuf8x8hi    $m7, $mzero, $m7
                mov          $a7, $a6
              }
              // Reload as it is trampled
              // Compare: any byte left that =0x05 => 0xffff (==nan)
              {
                ld64         $a6:7, %[stackPtr], $mzero, stackOffsetA67/8
                f16v4cmpeq   $a0:1, $a0:1, $a6:7
              }
              {
                shr          $m8, $m8, 0x1
                or64         $a6:7, $a6:7, $a0:1
              }
              {
                shr          $m7, $m7, 0x1
                f16v4clamp   $a0:1, $a6:7, %[scaleOrClamp]
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
                st32         $m7, %[stackPtr], $mzero, stackOffsetMtoA7/4
                or64         $a0:1, $a6:7, $a0:1
              }
              {
                brnzdec      %[count], 2b
                f16v4mul     $a0:1, %[bias]:BL, $a0:1         // Scale values
              }
              mov            %[lastResults], $a0:1
          )l"
        : [bias] "+r"(bias), [lastResults] "=r"(lastResults), [baseIn] "+r"(in),
          [baseOut] "+r"(out), [count] "+r"(loopCount),
          [scaleOrClamp] "+r"((*metadata)[1])
        : [ctxtWorkers] "r"(stride), [stackPtr] "r"(stack)
        : "$a0:1", "$a6:7", "$m2", "$m4", "$m5", "$m7", "$m8", "memory");
    return lastResults;
  }
};

template <unsigned stride>
class inLineAssemblerCast<const half *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) unsigned loopCast152(unsigned loopCount,
                                                             const half *in,
                                                             quarter *out,
                                                             float2 *metadata) {
    unsigned lastResults;
    unsigned hexValue;
    float2 stack[5];

    ASSIGN_HEX_2_DOUBLE(stack[0], hexValue, 0x7c007c00);   // exp mask
    ASSIGN_HEX_2_FLOAT(stack[1][0], hexValue, 0x00007800); // fp8 max exp
    stack[1][1] = (*metadata)[1];                          // scale
    ASSIGN_HEX_2_DOUBLE(stack[2], hexValue, 0x02000002);   // nan class mask
    ASSIGN_HEX_2_DOUBLE(stack[3], hexValue, 0x80008000);   // nan fp8
    ASSIGN_HEX_2_DOUBLE(stack[4], hexValue,
                        0xff0000ff); // upper lower class mask

    // Cast half -> fp8(152):
    // Multiply the values by 2^(scale + bias)
    // Get absolute values and mask only exponent
    // Limit the exponent values to the maximum
    // Multiply by 2 using f16v4add
    // Merge exp with rest bits
    // Check that the values are not negative zeros (using f16v4class)

    asm volatile(
        R"l(
              .equ stackOffsetExpMask, 0
              .equ stackOffsetMaxExp, 8
              .equ stackOffsetScale, 12
              .equ stackOffsetNanClassMask, 16
              .equ stackOffsetNanFp8, 24
              .equ stackOffsetUpLowClassMask, 32

              {
                ld32         $a2, %[stackPtr], $mzero, stackOffsetScale/4
                uput         $FP_CTL, $azero   // disable FP exceptions
              }
              f32fromi32     $a2, $a2
              f32sub         $a2, $azero, $a2
              f32tof16       $a2, $a2
              {
                ld64step     $a6:7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                f16v2exp2    $a2, $a2 // prepare scale
              }

              {
                ld64         $a0:1, %[stackPtr], $mzero, stackOffsetUpLowClassMask/8
                f16v4class   $a5, $a6:7
              }
              sort4x16lo     $a4, $a5, $a5
              sort4x16hi     $a5, $a5, $a5
              {
                ld64         $a0:1, %[stackPtr], $mzero, stackOffsetNanClassMask/8
                and64        $a4:5, $a4:5, $a0:1
              }
              {
                ld64         $a0:1, %[stackPtr], $mzero, stackOffsetNanFp8/8
                f16v4cmpeq   $a4:5, $a4:5, $a0:1
              }
              andc64         $a6:7, $a6:7, $a4:5
              and64          $a0:1, $a0:1, $a4:5
              or64           $a6:7, $a6:7, $a0:1
              {
                st32         $a2, %[stackPtr], $mzero, stackOffsetScale/4
                f16v4mul     $a6:7, $a2:BL, $a6:7
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetExpMask/8
                f16v4absadd  $a0:1, $a6:7, $azeros
              }
              f16v4cmple     $a4:5, $a2:BL, $a0:1
              or64           $a0:1, $a6:7, $a2:3
              and64          $a0:1, $a0:1, $a4:5
              f16v4add       $a6:7, $a6:7, $a6:7
              {
                bri          1f
                andc64       $a6:7, $a6:7, $a4:5
              }
            2:
              st32           %[lastResults], $mzero, %[baseOut], 0
              {
                ld32step     $azero, $mzero, %[baseOut]+=, %[ctxtWorkers]
                andc64       $a6:7, $a6:7, $a4:5
              }
            1:
              {
                ld64step     $a6:7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                or64         $a0:1, $a6:7, $a0:1
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetUpLowClassMask/8
                f16v4class   $a5, $a6:7
              }
              sort4x16lo     $a4, $a5, $a5
              sort4x16hi     $a5, $a5, $a5
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetNanClassMask/8
                and64        $a4:5, $a4:5, $a2:3
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetNanFp8/8
                f16v4cmpeq   $a4:5, $a4:5, $a2:3
              }
              andc64         $a6:7, $a6:7, $a4:5
              and64          $a2:3, $a2:3, $a4:5
              {
                ld32         $a2, %[stackPtr], $mzero, stackOffsetScale/4
                or64         $a6:7, $a6:7, $a2:3
              }
              {
                ld32         $a2, %[stackPtr], $mzero, stackOffsetMaxExp/4
                f16v4mul     $a6:7, $a2:BL, $a6:7       // Scale values
              }
              {
                atom         $m6, $a0
                f16v4absadd  $a4:5, $a6:7, $azeros
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetExpMask/8
                f16v4cmple   $a4:5, $a2:BL, $a4:5
              }
              {
                atom         %[lastResults], $a1
                or64         $a0:1, $a6:7, $a2:3
              }
              {
                sort8x8hi    %[lastResults], $m6, %[lastResults]
                and64        $a0:1, $a0:1, $a4:5
              }
              {
                brnzdec      %[count], 2b
                f16v4add     $a6:7, $a6:7, $a6:7
              }
          )l"
        : [lastResults] "=&r"(lastResults), [baseOut] "+r"(out),
          [baseIn] "+r"(in), [count] "+r"(loopCount)
        : [ctxtWorkers] "r"(stride), [stackPtr] "r"(stack)
        : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "$m6", "memory");
    return lastResults;
  }

  static __attribute__((always_inline)) unsigned loopCast143(unsigned loopCount,
                                                             const half *in,
                                                             quarter *out,
                                                             float2 *metadata) {
    unsigned lastResults;
    unsigned hexValue;
    float2 stack[11];

    stack[0][0] = (*metadata)[1];                          // scale
    ASSIGN_HEX_2_DOUBLE(stack[1], hexValue, 0x00800080);   // fp8 sign mask
    ASSIGN_HEX_2_FLOAT(stack[2][0], hexValue, 0x47004700); // 7.0 fp16 bias
    ASSIGN_HEX_2_DOUBLE(stack[3], hexValue, 0x80000080);   // fp8 fp32 sign mask
    ASSIGN_HEX_2_DOUBLE(stack[4], hexValue, 0x02000002);   // nan class mask
    ASSIGN_HEX_2_DOUBLE(stack[5], hexValue, 0x80008000);   // nan fp8
    ASSIGN_HEX_2_DOUBLE(stack[6], hexValue,
                        0xff0000ff); // upper/lower class mask
    ASSIGN_HEX_2_DOUBLE(stack[7], hexValue,
                        0xb000b000); // lowest representable fp8
    ASSIGN_HEX_2_DOUBLE(stack[8], hexValue, 0x007f007f);  // sign exp mask
    ASSIGN_HEX_2_DOUBLE(stack[9], hexValue, 0x7c007c00);  // exp mask
    ASSIGN_HEX_2_DOUBLE(stack[10], hexValue, 0x00400040); // correction

    // Cast half -> fp8(143):
    // Multiply the values by 2^(scale + bias)
    // Round values to nearest representable
    // Check that the values are not nans (using f16v4class)
    // Split sign and rest bits
    // Shift and sort the values accordingly and merge to the sign

    asm volatile(
        R"l(
              .equ fp16SignMask, 0x00008000
              .equ f16MantisaDiff, 7

              .equ stackOffsetScale, 0
              .equ stackOffsetFp8SignMask, 8
              .equ stackOffsetBias, 16
              .equ stackOffsetFp832SignMask, 24
              .equ stackOffsetNanClassMask, 32
              .equ stackOffsetNanFp8, 40
              .equ stackOffsetUpLowClassMask, 48
              .equ stackOffsetLowestFp16, 56
              .equ stackOffsetSignExpMask, 64
              .equ stackOffsetExpMask, 72
              .equ stackOffsetCorrections, 80
              .equ stackOffsetScratch, 88

              {
                ld32         $a0, %[stackPtr], $mzero, stackOffsetScale/4
                uput         $FP_CTL, $azero   // disable FP exceptions
              }
              {
                ld32         $a3, %[stackPtr], $mzero, stackOffsetBias/4
                f32fromi32   $a0, $a0
              }
              f32sub         $a0, $azero, $a0
              f32tof16       $a0, $a0
              f16v2sub       $a0, $a0, $a3
              {
                ld64step     $a6:7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                f16v2exp2    $a0, $a0 // prepare scale
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetUpLowClassMask/8
                f16v4class   $a1, $a6:7
              }
              {
                st32         $a0, %[stackPtr], $mzero, stackOffsetScale/8
                sort4x16lo   $a0, $a1, $a1
              }
              sort4x16hi     $a1, $a1, $a1
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetNanClassMask/8
                and64        $a0:1, $a0:1, $a2:3
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetNanFp8/8
                f16v4cmpeq   $a4:5, $a0:1, $a2:3
              }
              andc64         $a6:7, $a6:7, $a4:5
              {
                ld64         $a0:1, %[stackPtr], $mzero, stackOffsetFp8SignMask/8
                and64        $a2:3, $a2:3, $a4:5
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetLowestFp16/8
                or64         $a6:7, $a6:7, $a2:3
              }
              f16v4cmpge     $a2:3, $a2:3, $a6:7
              {
                ld32         $a2, %[stackPtr], $mzero, stackOffsetScale/4
                or64         $a4:5, $a2:3, $a4:5
              }
              {
                bri          1f
                f16v4mul     $a6:7, $a2:BL, $a6:7       // Scale values
              }
            2:
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetLowestFp16/8
                or64         $a6:7, $a6:7, $a2:3
              }
              {
                st32         %[lastResults], $mzero, %[baseOut], 0
                f16v4cmpge   $a2:3, $a2:3, $a6:7
              }
              {
                ld32         $a2, %[stackPtr], $mzero, stackOffsetScale/4
                or64         $a4:5, $a2:3, $a4:5
              }
              {
                ld32step     $azero, $mzero, %[baseOut]+=, %[ctxtWorkers]
                f16v4mul     $a6:7, $a2:BL, $a6:7       // Scale values
              }
            1:
              {
                st64         $a4:5, %[stackPtr], $mzero, stackOffsetScratch/8
                and64        $a0:1, $a6:7, $a0:1
              }
              ld64           $a2:3, %[stackPtr], $mzero, stackOffsetSignExpMask/8
              {
                ld64         $a4:5, %[stackPtr], $mzero, stackOffsetCorrections/8
                and64        $a2:3, $a6:7, $a2:3
              }
              f16v4cmpeq     $a2:3, $a2:3, $a4:5
              and64          $a0:1, $a0:1, $a2:3
              andc64         $a2:3, $a4:5, $a2:3
              {
                ld64         $a0:1, %[stackPtr], $mzero, stackOffsetExpMask/8
                or64         $a2:3, $a2:3, $a0:1
              }
              and64          $a0:1, $a6:7, $a0:1
              or64           $a2:3, $a2:3, $a0:1
              {
                ld64         $a4:5, %[stackPtr], $mzero, stackOffsetScratch/8
                f16v4sub     $a2:3, $a2:3, $a0:1
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetFp832SignMask/8
                f16v4absadd  $a6:7, $a6:7, $a2:3
              }
              {
                atom         %[lastResults], $a7
                mov          $a0:1, $a6:7
              }
              {
                atom         $m6, $a6
                and64        $a2:3, $a4:5, $a2:3
              }
              {
                shr          $m6, $m6, f16MantisaDiff
                sort4x16lo   $a7, $a2, $a3
              }
              {
                shr          %[lastResults], %[lastResults], f16MantisaDiff
                sort4x16hi   $a6, $a2, $a3
              }
              {
                sort8x8lo    $m6, $m6, %[lastResults]
                or           $a7, $a6, $a7
              }
                atom           $m4, $a7
              {
                ld64step       $a6:7, $mzero, %[baseIn]+=, %[ctxtWorkers]
                setzi        $a2, fp16SignMask
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetUpLowClassMask/8
                f16v4class   $a1, $a6:7
              }
              sort4x16lo     $a0, $a1, $a1
              {
                or           %[lastResults], $m6, $m4
                sort4x16hi     $a1, $a1, $a1
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetNanClassMask/8
                and64        $a0:1, $a0:1, $a2:3
              }
              {
                ld64         $a2:3, %[stackPtr], $mzero, stackOffsetNanFp8/8
                f16v4cmpeq   $a4:5, $a0:1, $a2:3
              }
              {
                ld64         $a0:1, %[stackPtr], $mzero, stackOffsetFp8SignMask/8
                andc64       $a6:7, $a6:7, $a4:5
              }
              {
                brnzdec      %[count], 2b
                and64        $a2:3, $a2:3, $a4:5
              }
          )l"
        : [lastResults] "=&r"(lastResults), [baseOut] "+r"(out),
          [baseIn] "+r"(in), [count] "+r"(loopCount)
        : [ctxtWorkers] "r"(stride), [stackPtr] "r"(stack)
        : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "$m4", "$m6", "memory");
    return lastResults;
  }
};

// Setting Fp8 metadata:
// Set the format to 0 if 152, otherwise -0.
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
