// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Change a number into a string
// The number must be known at the time the preprocessor runs
#define STR_(x) #x
#define STR(x) STR_(x)

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
               " f32int     $a0, $a0, 3\n"                                     \
               " f32int     $a1, $a1, 3\n"                                     \
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
               " f32int $a2, $a2, 3\n"                                         \
               " f32int $a3, $a3, 3\n"                                         \
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
               "  f32int $a0, $a0, 3\n"                                        \
               " f32int $a1, $a1, 3\n"                                         \
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
               " f32int $a2, $a2, 3\n"                                         \
               " f32int $a3, $a3, 3\n"                                         \
               " f32toi32   $a2, $a2\n"                                        \
               " f32toi32   $a3, $a3\n"                                        \
               " sort4x16lo  $a2, $a2, $a3\n"                                  \
               " atom        $m1, $a2\n"                                       \
               " sort8x8lo   $m0, $m0, $m1\n"                                  \
               " st32step    $m0, $mzero, %[outPtr]+=," STRIDE "\n"            \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               : [limits] "r"(limits)                                          \
               : "$m0", "$m1", "$a0:1", "$a2:3", "$a4:5", "memory");

// Optimized pipelined processing for multiple of 4 elements
// FLOAT => HALF
#define CAST_FLOAT_HALF_LOOP(STRIDE)                                           \
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
               "       f32v4tof16 $a0:1, $a0:3}\n"                             \
               "      {ldst64pace $a0:1, $a0:1, $m0:1+=, $m2, 6\n"             \
               "       fnop}\n"                                                \
               "2:\n"                                                          \
               "    f32v4tof16 $a2:3, $a0:3\n"                                 \
               "    st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"          \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               :                                                               \
               : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");

// Optimized pipelined processing for multiple of 4 elements
// HALF => FLOAT
#define CAST_HALF_FLOAT_LOOP(STRIDE)                                           \
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
               "    f16v2tof32 $a2:3, $a0}\n"                                  \
               "1:\n"                                                          \
               "      {ldst64pace $a0:1, $a2:3, $m0:1+=, $m2, 6\n"             \
               "       f16v2tof32 $a2:3, $a1}\n"                               \
               "      {st64step $a2:3, $mzero, %[outPtr]+=,2*" STRIDE "\n"     \
               "       f16v2tof32 $a2:3, $a0}\n"                               \
               "2:\n"                                                          \
               "    {st64step $a2:3, $mzero, %[inPtr]+=,1\n"                   \
               "    f16v2tof32 $a0:1, $a1}\n"                                  \
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
    CAST_FLOAT_HALF_LOOP(STR(1))
    return;
  }
};

template <> class inLineAssemblerCast<const half *, float *, false, 1> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const half *inPtr, float *outPtr) {
    CAST_HALF_FLOAT_LOOP(STR(1))
    return;
  }
};
template <>
class inLineAssemblerCast<const float *, half *, false, CTXT_WORKERS> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const float *inPtr, half *outPtr) {
    CAST_FLOAT_HALF_LOOP(STR(CTXT_WORKERS))
    return;
  }
};

template <>
class inLineAssemblerCast<const half *, float *, false, CTXT_WORKERS> {
public:
  static __attribute__((always_inline)) void
  loopBody(unsigned loopCount, const half *inPtr, float *outPtr) {
    CAST_HALF_FLOAT_LOOP(STR(CTXT_WORKERS))
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
    asm volatile("   f32clamp $a0, %[inFloat], %[limits]\n"
                 "   f32int $a0, $a0, 3\n"
                 "   f32toi32 $a0, $a0\n"
                 "   atom %[result], $a0\n"
                 : [result] "=r"(result)
                 : [inFloat] "r"(inFloat), [limits] "r"(limits)
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
    asm volatile("   f32clamp $a0, %[inFloat], %[limits]\n"
                 "   f32int $a0, $a0, 3\n"
                 "   f32toi32 $a0, $a0\n"
                 "   atom %[result], $a0\n"
                 : [result] "=r"(result)
                 : [inFloat] "r"(inFloat), [limits] "r"(limits)
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
