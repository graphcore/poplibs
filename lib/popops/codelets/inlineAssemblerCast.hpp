// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Change a number into a string
// The number must be known at the time the preprocessor runs
#define STR_(x) #x
#define STR(x) STR_(x)

// Flag value for rounding mode
#define TFPU_ROUND_ZERO 3

// Optimized pipelined processing for multiple of 4 elements
// UNSIGNED_CHAR => HALF
#define CAST_UCHAR_HALF_LOOP(STRIDE)                                           \
  asm volatile("brnzdec     %[loopCount], 3f\n"                                \
               "bri         4f \n"                                             \
               ".align 8\n"                                                    \
               " nop\n"                                                        \
               "3:\n"                                                          \
               " ldz16step   $m0, $mzero, %[inPtr]+=, 1\n"                     \
               " shuf8x8lo   $m0, $m0, $mzero\n"                               \
               " st32        $m0, $mzero, %[outPtr], 0\n"                      \
               " ld32        $a4, $mzero, %[outPtr], 0\n"                      \
               " {ldz16step   $m0, $mzero, %[inPtr]+=,2*" STRIDE "-1\n"        \
               " sort4x16lo  $a0, $a4, $azero }\n"                             \
               "\n"                                                            \
               " rpt         %[loopCount],((2f-1f)/8)-1\n"                     \
               "1:\n"                                                          \
               "{shuf8x8lo   $m0, $m0, $mzero\n"                               \
               " sort4x16hi  $a1, $a4, $azero }\n"                             \
               "{st32        $m0, $mzero, %[outPtr], 0\n"                      \
               " f32fromui32 $a0, $a0       }\n"                               \
               "{ld32        $a4, $mzero, %[outPtr], 0\n"                      \
               " f32fromui32 $a1, $a1       }\n"                               \
               "{ldz16step   $m0, $mzero, %[inPtr]+=, 1\n"                     \
               " sort4x16lo  $a2, $a4, $azero }\n"                             \
               "{shuf8x8lo   $m0, $m0, $mzero\n"                               \
               " sort4x16hi  $a3, $a4, $azero }\n"                             \
               "{st32        $m0, $mzero, %[outPtr], 0\n"                      \
               " f32fromui32 $a2, $a2       }\n"                               \
               "{ld32        $a4, $mzero, %[outPtr], 0\n"                      \
               " f32fromui32 $a3, $a3       }\n"                               \
               "{ldz16step   $m0, $mzero, %[inPtr]+=,2*" STRIDE "-1\n"         \
               "f32v4tof16  $a0:1, $a0:3   }\n"                                \
               "{st64step    $a0:1, $mzero, %[outPtr]+=," STRIDE "\n"          \
               " sort4x16lo  $a0, $a4, $azero }\n"                             \
               "2:\n"                                                          \
               " {shuf8x8lo   $m0, $m0, $mzero\n"                              \
               " sort4x16hi  $a1, $a4, $azero }\n"                             \
               " {st32        $m0, $mzero, %[outPtr], 0\n"                     \
               " f32fromui32 $a0, $a0       }\n"                               \
               "{ld32        $a4, $mzero, %[outPtr], 0\n"                      \
               " f32fromui32 $a1, $a1       }\n"                               \
               " sort4x16lo  $a2, $a4, $azero\n"                               \
               " sort4x16hi  $a3, $a4, $azero\n"                               \
               " f32fromui32 $a2, $a2\n"                                       \
               " f32fromui32 $a3, $a3\n"                                       \
               " f32v4tof16  $a0:1, $a0:3\n"                                   \
               " st64step    $a0:1, $mzero, %[outPtr]+=," STRIDE "\n"          \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               :                                                               \
               : "$m0", "$a0:1", "$a2:3", "$a4", "memory");

// Optimized pipelined processing for multiple of 4 elements
// HALF => UCHAR or SCHAR
#define CAST_HALF_UCHAR_SCHAR_LOOP(STRIDE)                                     \
  asm volatile("brnzdec     %[loopCount], 3f\n"                                \
               "bri         4f \n"                                             \
               ".align 8\n"                                                    \
               "3:\n"                                                          \
               " // Prologue (fill)\n"                                         \
               " ld64step    $a4:5, $mzero, %[inPtr]+=," STRIDE "\n"           \
               " f16v2tof32  $a0:1, $a4\n"                                     \
               " f32clamp    $a0, $a0, %[limits]\n"                            \
               " f32clamp    $a1, $a1, %[limits]\n"                            \
               " f32int     $a0, $a0, %[ROUNDING_MODE]\n"                      \
               " f32int     $a1, $a1, %[ROUNDING_MODE]\n"                      \
               " f32toi32   $a0, $a0\n"                                        \
               " {add        %[loopCount], %[loopCount], -1\n"                 \
               " f32toi32   $a1, $a1}\n"                                       \
               " \n"                                                           \
               " // Main pipelined loop over blocks of 4 elements\n"           \
               " {brneg      %[loopCount], 5f\n"                               \
               " sort4x16lo  $a0, $a0, $a1}\n"                                 \
               "6:\n"                                                          \
               " {atom        $m0, $a0\n"                                      \
               " f16v2tof32  $a2:3, $a5}\n"                                    \
               " f32clamp    $a2, $a2, %[limits]\n"                            \
               " f32clamp    $a3, $a3, %[limits]\n"                            \
               " f32int $a2, $a2, %[ROUNDING_MODE]\n"                          \
               " f32int $a3, $a3, %[ROUNDING_MODE]\n"                          \
               " f32toi32   $a2, $a2\n"                                        \
               " f32toi32   $a3, $a3\n"                                        \
               " {ld64step    $a4:5, $mzero, %[inPtr]+=," STRIDE "\n"          \
               " sort4x16lo  $a2, $a2, $a3}\n"                                 \
               "\n"                                                            \
               " {atom        $m1, $a2\n"                                      \
               " f16v2tof32  $a0:1, $a4}\n"                                    \
               " {sort8x8lo   $m0, $m0, $m1\n"                                 \
               " f32clamp    $a0, $a0, %[limits]}\n"                           \
               " {st32step    $m0, $mzero, %[outPtr]+=," STRIDE "\n"           \
               "  f32clamp    $a1, $a1, %[limits]}\n"                          \
               "  f32int $a0, $a0, %[ROUNDING_MODE]\n"                         \
               " f32int $a1, $a1, %[ROUNDING_MODE]\n"                          \
               " f32toi32  $a0, $a0\n"                                         \
               " f32toi32   $a1, $a1\n"                                        \
               " {brnzdec     %[loopCount], 6b\n"                              \
               " sort4x16lo  $a0, $a0, $a1}\n"                                 \
               "5:\n"                                                          \
               "  // Epilogue (drain)\n"                                       \
               " {atom        $m0, $a0\n"                                      \
               " f16v2tof32  $a2:3, $a5}\n"                                    \
               " f32clamp    $a2, $a2, %[limits]\n"                            \
               " f32clamp    $a3, $a3, %[limits]\n"                            \
               " f32int $a2, $a2, %[ROUNDING_MODE]\n"                          \
               " f32int $a3, $a3, %[ROUNDING_MODE]\n"                          \
               " f32toi32   $a2, $a2\n"                                        \
               " f32toi32   $a3, $a3\n"                                        \
               " sort4x16lo  $a2, $a2, $a3\n"                                  \
               " atom        $m1, $a2\n"                                       \
               " sort8x8lo   $m0, $m0, $m1\n"                                  \
               " st32step    $m0, $mzero, %[outPtr]+=," STRIDE "\n"            \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               : [limits] "r"(limits), [ROUNDING_MODE] "i"(TFPU_ROUND_ZERO)    \
               : "$m0", "$m1", "$a0:1", "$a2:3", "$a4:5", "memory");

// Optimized pipelined processing for multiple of 4 elements
// FLOAT => HALF or HALF => QUARTER
#define CAST_FP_DEMOTE_LOOP(STRIDE, INSTRUCTION)                               \
  asm volatile("    brnzdec   %[loopCount], 3f\n"                              \
               "    bri       4f\n"                                            \
               ".align 8\n"                                                    \
               "3:\n"                                                          \
               "  ld64step $a0:1, $mzero, %[inPtr]+=,1\n"                      \
               "  ld64step $a2:3, $mzero, %[inPtr]+=,2*" STRIDE "-1\n"         \
               "  tapack $m0:1, %[inPtr], $mzero, %[outPtr]\n"                 \
               "  ld64step $azeros, $mzero, %[inPtr]+=,1\n"                    \
               "  mul $m2, %[loopCount], " STRIDE "* 8\n"                      \
               "  add %[outPtr], %[outPtr], $m2\n"                             \
               "  setzi $m2, (2*" STRIDE ")<<10 | " STRIDE "\n"                \
               "  rpt %[loopCount], (2f-1f)/8 -1;"                             \
               "1:\n"                                                          \
               "      {ld64step $a2:3, $mzero, %[inPtr]+=,2*" STRIDE "\n"      \
               "     " INSTRUCTION " $a0:1, $a0:3}\n"                          \
               "      {ldst64pace $a0:1, $a0:1, $m0:1+=, $m2, 6\n"             \
               "       fnop}\n"                                                \
               "2:\n"                                                          \
               "  " INSTRUCTION " $a2:3, $a0:3\n"                              \
               "    st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"          \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               :                                                               \
               : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");

// Optimized pipelined processing for multiple of 4 elements
// HALF => FLOAT or QUARTER => HALF
#define CAST_FP_PROMOTE_LOOP(STRIDE, INSTRUCTION)                              \
  asm volatile("    brnzdec   %[loopCount], 3f\n"                              \
               "    bri       4f\n"                                            \
               ".align 8\n"                                                    \
               "3:\n"                                                          \
               "  ld64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"             \
               "  tapack $m0:1, %[inPtr], $mzero, %[outPtr]\n"                 \
               "  mul $m2, %[loopCount], 16*" STRIDE "\n"                      \
               "  add %[inPtr], %[outPtr], $m2\n"                              \
               "  ld64step $azeros, $mzero, %[outPtr]+=,1\n"                   \
               "  setzi $m2, 2*" STRIDE " | (" STRIDE "<<10)\n"                \
               "  {rpt %[loopCount], (2f-1f)/8 -1;"                            \
               " " INSTRUCTION " $a2:3, $a0}\n"                                \
               "1:\n"                                                          \
               "      {ldst64pace $a0:1, $a2:3, $m0:1+=, $m2, 6\n"             \
               "     " INSTRUCTION " $a2:3, $a1}\n"                            \
               "      {st64step $a2:3, $mzero, %[outPtr]+=,2*" STRIDE "\n"     \
               "     " INSTRUCTION " $a2:3, $a0}\n"                            \
               "2:\n"                                                          \
               "    {st64step $a2:3, $mzero, %[inPtr]+=,1\n"                   \
               "   " INSTRUCTION " $a0:1, $a1}\n"                              \
               "    st64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"           \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               :                                                               \
               : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");

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

template <> class inLineAssemblerCast<const float *, half *, false, 1> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const float *inPtr, half *outPtr) {
    CAST_FP_DEMOTE_LOOP(STR(1), "f32v4tof16")
    return;
  }
};

template <> class inLineAssemblerCast<const half *, float *, false, 1> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const half *inPtr, float *outPtr) {
    CAST_FP_PROMOTE_LOOP(STR(1), "f16v2tof32")
    return;
  }
};
template <>
class inLineAssemblerCast<const float *, half *, false, CTXT_WORKERS> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const float *inPtr, half *outPtr) {
    CAST_FP_DEMOTE_LOOP(STR(CTXT_WORKERS), "f32v4tof16")
    return;
  }
};

template <>
class inLineAssemblerCast<const half *, float *, false, CTXT_WORKERS> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const half *inPtr, float *outPtr) {
    CAST_FP_PROMOTE_LOOP(STR(CTXT_WORKERS), "f16v2tof32")
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
    if constexpr (stride == CTXT_WORKERS) {
      CAST_HALF_UCHAR_SCHAR_LOOP(STR(CTXT_WORKERS))
    } else {
      CAST_HALF_UCHAR_SCHAR_LOOP(STR(1))
    }
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
    if constexpr (stride == CTXT_WORKERS) {
      CAST_UCHAR_HALF_LOOP(STR(CTXT_WORKERS))
    } else {
      CAST_UCHAR_HALF_LOOP(STR(1))
    }
    return;
  }
};

template <typename SrcType, typename DstType, bool charToFPType,
          unsigned stride>
class inLineAssemblerCastFp8 {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, SrcType inPtr, DstType outPtr) {
    return;
  }
};

#if __IPU_ARCH_VERSION__ == 21
template <unsigned stride>
class inLineAssemblerCastFp8<const half *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) quarter
  singleCast(const half *in, float2 metaData0, float2 metaData1) {
    unsigned result;
    half2 in2 = {*in, *in};
    asm volatile(
        R"l(  f16v2tof8 $a0, %[in2]
              atom %[result], $a0
        )l"
        : [result] "=r"(result)
        : [in2] "r"(in2)
        : "$a0");

    return quarter(result);
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const half *inPtr, quarter *outPtr,
           float2 metaData0, float2 metaData1) {
    if constexpr (stride == CTXT_WORKERS) {
      CAST_FP_DEMOTE_LOOP(STR(CTXT_WORKERS), "f16v8tof8")
    } else {
      CAST_FP_DEMOTE_LOOP(STR(1), "f16v8tof8")
    }
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCastFp8<const quarter *, half *, true, stride> {
public:
  static __attribute__((always_inline)) half
  singleCast(const quarter *in, float2 metaData0, float2 metaData1) {
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
           float2 metaData0, float2 metaData1) {
    if constexpr (stride == CTXT_WORKERS) {
      CAST_FP_PROMOTE_LOOP(STR(CTXT_WORKERS), "f8v4tof16")
    } else {
      CAST_FP_PROMOTE_LOOP(STR(1), "f8v4tof16")
    }
    return;
  }
};

#define CAST_QUARTER_TO_INT8(INSTRUCTION)                                      \
  asm volatile("   ldb8 $a0, $mzero, %[in], 0 \n"                              \
               "   f8v2tof16 $a0, $a0 \n"                                      \
               "   f16tof32 $a0, $a0 \n"                                       \
               " " INSTRUCTION " $a0, $a0 \n"                                  \
               "   atom %[result], $a0 \n"                                     \
               : [result] "=r"(result)                                         \
               : [in] "r"(in)                                                  \
               : "$a0");

#define CAST_QUARTER_TO_INT8_V4(INSTRUCTION)                                   \
  asm volatile("f8v4tof16  $a0:1, %[in]\n"                                     \
               "   f16v2tof32 $a2:3, $a0\n"                                    \
               " " INSTRUCTION " $a2, $a2\n"                                   \
               " " INSTRUCTION " $a3, $a3\n"                                   \
               "   f16v2tof32 $a0:1, $a1\n"                                    \
               " " INSTRUCTION "  $a0, $a0\n"                                  \
               " " INSTRUCTION "  $a1, $a1\n"                                  \
               "   sort4x16lo $a0, $a0, $a1\n"                                 \
               "   sort4x16lo $a2, $a2, $a3\n"                                 \
               "   sort8x8lo  %[out], $a2, $a0\n"                              \
               : [out] "=r"(out)                                               \
               : [in] "r"(in)                                                  \
               : "$a0:1", "$a2:3");

template <unsigned stride>
class inLineAssemblerCastFp8<const quarter *, char *, true, stride> {
public:
  static __attribute__((always_inline)) char
  singleCast(const quarter *in, float2 metaData0, float2 metaData1) {
    unsigned result;
    CAST_QUARTER_TO_INT8("f32toi32")
    return result;
  }
  static __attribute__((always_inline)) float vectorCast4(const float in) {
    float out;
    CAST_QUARTER_TO_INT8_V4("f32toi32")

    return out;
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, char *outPtr,
           float2 metaData0, float2 metaData1) {
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
class inLineAssemblerCastFp8<const quarter *, unsigned char *, true, stride> {
public:
  static __attribute__((always_inline)) unsigned char
  singleCast(const quarter *in, float2 metaData0, float2 metaData1) {
    unsigned result;
    CAST_QUARTER_TO_INT8("f32toui32")
    return result;
  }
  static __attribute__((always_inline)) float vectorCast4(const float in) {
    float out;
    CAST_QUARTER_TO_INT8_V4("f32toui32")

    return out;
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, unsigned char *outPtr,
           float2 metaData0, float2 metaData1) {
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
class inLineAssemblerCastFp8<const quarter *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) void setFp8Config(float2 metaData) {
    asm volatile(
        R"l(  uput $FP_SCL, %[scale]
              uput $FP_NFMT, %[format]
        )l"
        :
        : [format] "r"(metaData[0]), [scale] "r"(metaData[1])
        :);
  }

  static __attribute__((always_inline)) quarter
  singleCast(const quarter *in, float2 metaData0, float2 metaData1) {
    unsigned result;
    setFp8Config(metaData0);
    asm volatile(
        R"l(  ldb8 $a0, $mzero, %[in], 0
              f8v2tof16 $a0, $a0
              uput $FP_SCL, %[scale]
              uput $FP_NFMT, %[format]
              f16v2tof8 $a0, $a0
              atom %[result], $a0
          )l"
        : [result] "=r"(result)
        : [in] "r"(in), [scale] "r"(metaData1[1]), [format] "r"(metaData1[0])
        : "$a0");

    return quarter(result);
  }

  static __attribute__((always_inline)) float2 vectorCast8(const float2 in,
                                                           float2 metaData) {
    float2 out;
    asm volatile(
        R"l( f8v4tof16  $a0:1, %[in0]
             f8v4tof16  $a2:3, %[in1]
             uput $FP_SCL, %[scale]
             uput $FP_NFMT, %[format]
             f16v8tof8  %[out], $a0:3
        )l"
        : [out] "=r"(out)
        : [in0] "r"(in[0]), [in1] "r"(in[1]), [scale] "r"(metaData[1]),
          [format] "r"(metaData[0])
        : "$a0:1", "$a2:3");
    return out;
  }

  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const quarter *inPtr, quarter *outPtr,
           float2 metaData0, float2 metaData1) {
    const float2 *inPtr8 = reinterpret_cast<const float2 *>(inPtr);
    float2 *outPtr8 = reinterpret_cast<float2 *>(outPtr);
    for (unsigned i = 0; i < loopCount; i++) {
      setFp8Config(metaData0);
      *outPtr8 = vectorCast8(*inPtr8, metaData1);
      inPtr8 += stride;
      outPtr8 += stride;
    }
    return;
  }
};

template <unsigned stride>
class inLineAssemblerCastFp8<const char *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) quarter
  singleCast(const char *in, float2 metaData0, float2 metaData1) {
    unsigned result;
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

    return quarter(result);
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
           float2 metaData0, float2 metaData1) {
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
class inLineAssemblerCastFp8<const unsigned char *, quarter *, true, stride> {
public:
  static __attribute__((always_inline)) quarter
  singleCast(const unsigned char *in, float2 metaData0, float2 metaData1) {
    unsigned result;
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

    return quarter(result);
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
           float2 metaData0, float2 metaData1) {
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

// Setting Fp8 meta data:
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

static __attribute__((always_inline)) void
setFp8Config(const unsigned char *metaData) {
  asm volatile(
      R"l(  ldb8 $a0, %[metaData], $mzero, 0
            uput $FP_SCL, $a0
            andc $a0, $a0, 0x40000000
            f32cmplt $a0, $a0, $azero
            uput $FP_NFMT, $a0
      )l"
      :
      : [metaData] "r"(metaData)
      : "$a0");
}

static __attribute__((always_inline)) void
setFp8ConfigNegScale(const unsigned char *metaData) {
  asm volatile(
      R"l(  ldb8 $a0, %[metaData], $mzero, 0
            andc $a1, $a0, 0x40000000
            f32cmplt $a1, $a1, $azero
            uput $FP_NFMT, $a1
            and  $a0, $a0, 0x3f
            setzi $a1, 0x80
            f16v2sub  $a0, $a1, $a0
            uput $FP_SCL, $a0
      )l"
      :
      : [metaData] "r"(metaData)
      : "$a0", "$a1");
}

static float2 extractMetaData(const unsigned char *metaData) {
  float2 result;
  asm volatile(
      R"l(  ldb8      %[scale], %[metaData], $mzero, 0
            andc      $a0, %[scale], 0x40000000
            f32cmplt  %[format], $a0, $azero
      )l"
      : [format] "=r"(result[0]), [scale] "=r"(result[1])
      : [metaData] "r"(metaData)
      : "$a0");
  return result;
}

static float2 extractMetaDataNegScale(const unsigned char *metaData) {
  float2 result;
  asm volatile(
      R"l(  ldb8      %[scale], %[metaData], $mzero, 0
            andc      $a0, %[scale], 0x40000000
            f32cmplt  %[format], $a0, $azero
            and       %[scale], %[scale], 0x3f 
            setzi     $a0, 0x80
            f16v2sub  %[scale], $a0, %[scale]
      )l"
      : [format] "=r"(result[0]), [scale] "=r"(result[1])
      : [metaData] "r"(metaData)
      : "$a0");
  return result;
}

#endif
