// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Macro producing a rpt loop for unary ops where the instruction supports
// processing of 64 bits in 1 instruction (f32v2... or f16v4...)
#define UNARY_LOOP_1_INSTRUCTION(INSTRUCTION, STRIDE, OPERAND)                 \
  asm volatile("    brnzdec   %[loopCount], 3f\n"                              \
               "    bri       4f\n"                                            \
               ".align 8 \n"                                                   \
               "3:\n"                                                          \
               "  ld64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"             \
               "  rpt %[loopCount], (2f-1f)/8 -1;"                             \
               "1:\n"                                                          \
               "      {ld64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"        \
               "     " INSTRUCTION " $a2:3, " OPERAND ", $a0:1}\n"             \
               "      {st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"       \
               "       fnop}\n"                                                \
               "2:\n"                                                          \
               "  " INSTRUCTION " $a2:3, " OPERAND ", $a0:1\n"                 \
               "    st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"          \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               :                                                               \
               : "$a0:1", "$a2:3", "memory");

// Macro producing a rpt loop for unary ops where the instruction supports
// processing of 64 bits in 2 instructions (2 of f32... or f16v2...)
#define UNARY_LOOP_2_INSTRUCTION(INSTRUCTION, STRIDE)                          \
  asm volatile("    brnzdec   %[loopCount], 3f\n"                              \
               "    bri       4f\n"                                            \
               ".align 8 \n"                                                   \
               "    nop\n /* rpt alignment*/"                                  \
               "3:\n"                                                          \
               "  ld64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"             \
               "  {rpt %[loopCount], (2f-1f)/8 -1;"                            \
               " " INSTRUCTION " $a2, $a0}\n"                                  \
               "1:\n"                                                          \
               "      {ld64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"        \
               "     " INSTRUCTION " $a3, $a1}\n"                              \
               "      {st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"       \
               "     " INSTRUCTION " $a2, $a0}\n"                              \
               "2:\n"                                                          \
               "  " INSTRUCTION " $a3, $a1\n"                                  \
               "    st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"          \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               :                                                               \
               : "$a0:1", "$a2:3", "memory");

// Macro producing a rpt loop for unary ops with type half by casting to fp32
// and using 4 fp32 instructions
#define UNARY_LOOP_4_INSTRUCTION_CAST(INSTRUCTION, STRIDE)                     \
  asm volatile("    brnzdec   %[loopCount], 3f\n"                              \
               "    bri       4f\n"                                            \
               ".align 8 \n"                                                   \
               "3:\n"                                                          \
               "  ld64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"             \
               "  f16v2tof32 $a2:3, $a1\n"                                     \
               "  {rpt %[loopCount], (2f-1f)/8 -1;"                            \
               " " INSTRUCTION " $a2, $a2}\n"                                  \
               "1:\n"                                                          \
               "      {nop;"                                                   \
               "     " INSTRUCTION " $a3, $a3}\n"                              \
               "      {nop;"                                                   \
               "       f16v2tof32 $a0:1, $a0}\n"                               \
               "      {nop;"                                                   \
               "     " INSTRUCTION " $a0, $a0}\n"                              \
               "      {nop;"                                                   \
               "     " INSTRUCTION " $a1, $a1}\n"                              \
               "      {ld64step $a0:1, $mzero, %[inPtr]+=," STRIDE "\n"        \
               "       f32v4tof16 $a2:3, $a0:3}\n"                             \
               "      {st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"       \
               "       f16v2tof32 $a2:3, $a1}\n"                               \
               "      {nop;"                                                   \
               "     " INSTRUCTION " $a2, $a2}\n"                              \
               "2:\n"                                                          \
               "    " INSTRUCTION " $a3, $a3\n"                                \
               "      f16v2tof32 $a0:1, $a0\n"                                 \
               "    " INSTRUCTION " $a0, $a0\n"                                \
               "    " INSTRUCTION " $a1, $a1\n"                                \
               "      f32v4tof16 $a2:3, $a0:3\n"                               \
               "      st64step $a2:3, $mzero, %[outPtr]+=," STRIDE "\n"        \
               "4:\n"                                                          \
               : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr),             \
                 [outPtr] "+r"(outPtr)                                         \
               :                                                               \
               : "$a0:1", "$a2:3", "memory");

// Default template picks up the case for half data, and stride = CTXT_WORKERS
template <popops::expr::UnaryOpType op, typename T, unsigned stride>
class inlineAssemblerUnaryOp {
public:
  static __attribute__((always_inline)) std::pair<const half4 *, half4 *>
  loopBody(unsigned loopCount, const half4 *inPtr, half4 *outPtr) {

    if constexpr (op == popops::expr::UnaryOpType::ABSOLUTE) {
      UNARY_LOOP_1_INSTRUCTION("f16v4absadd", "6", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::EXPONENT) {
      UNARY_LOOP_2_INSTRUCTION("f16v2exp", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::LOGARITHM) {
      UNARY_LOOP_2_INSTRUCTION("f16v2ln", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::INVERSE) {
      UNARY_LOOP_4_INSTRUCTION_CAST("f32oox", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::NEGATE) {
      UNARY_LOOP_1_INSTRUCTION("f16v4sub", "6", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::SQRT) {
      UNARY_LOOP_4_INSTRUCTION_CAST("f32sqrt", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::SQUARE) {
      UNARY_LOOP_1_INSTRUCTION("f16v4mul", "6", "$a0:1")
    } else if constexpr (op == popops::expr::UnaryOpType::RSQRT) {
      UNARY_LOOP_4_INSTRUCTION_CAST("f32oorx", "6")
    }
    return std::make_pair(inPtr, outPtr);
  }
};

template <popops::expr::UnaryOpType op>
class inlineAssemblerUnaryOp<op, float, CTXT_WORKERS> {
public:
  static __attribute__((always_inline)) std::pair<const float2 *, float2 *>
  loopBody(unsigned loopCount, const float2 *inPtr, float2 *outPtr) {

    if constexpr (op == popops::expr::UnaryOpType::ABSOLUTE) {
      UNARY_LOOP_1_INSTRUCTION("f32v2absadd", "6", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::EXPONENT) {
      UNARY_LOOP_2_INSTRUCTION("f32exp", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::LOGARITHM) {
      UNARY_LOOP_2_INSTRUCTION("f32ln", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::INVERSE) {
      UNARY_LOOP_2_INSTRUCTION("f32oox", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::NEGATE) {
      UNARY_LOOP_1_INSTRUCTION("f32v2sub", "6", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::SQRT) {
      UNARY_LOOP_2_INSTRUCTION("f32sqrt", "6")
    } else if constexpr (op == popops::expr::UnaryOpType::SQUARE) {
      UNARY_LOOP_1_INSTRUCTION("f32v2mul", "6", "$a0:1")
    } else if constexpr (op == popops::expr::UnaryOpType::RSQRT) {
      UNARY_LOOP_2_INSTRUCTION("f32oorx", "6")
    }
    return std::make_pair(inPtr, outPtr);
  }
};

template <popops::expr::UnaryOpType op>
class inlineAssemblerUnaryOp<op, half, 1> {
public:
  static __attribute__((always_inline)) std::pair<const half4 *, half4 *>
  loopBody(unsigned loopCount, const half4 *inPtr, half4 *outPtr) {

    if constexpr (op == popops::expr::UnaryOpType::ABSOLUTE) {
      UNARY_LOOP_1_INSTRUCTION("f16v4absadd", "1", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::EXPONENT) {
      UNARY_LOOP_2_INSTRUCTION("f16v2exp", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::LOGARITHM) {
      UNARY_LOOP_2_INSTRUCTION("f16v2ln", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::INVERSE) {
      UNARY_LOOP_4_INSTRUCTION_CAST("f32oox", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::NEGATE) {
      UNARY_LOOP_1_INSTRUCTION("f16v4sub", "1", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::SQRT) {
      UNARY_LOOP_4_INSTRUCTION_CAST("f32sqrt", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::SQUARE) {
      UNARY_LOOP_1_INSTRUCTION("f16v4mul", "1", "$a0:1")
    } else if constexpr (op == popops::expr::UnaryOpType::RSQRT) {
      UNARY_LOOP_4_INSTRUCTION_CAST("f32oorx", "1")
    }
    return std::make_pair(inPtr, outPtr);
  }
};

template <popops::expr::UnaryOpType op>
class inlineAssemblerUnaryOp<op, float, 1> {
public:
  static __attribute__((always_inline)) std::pair<const float2 *, float2 *>
  loopBody(unsigned loopCount, const float2 *inPtr, float2 *outPtr) {

    if constexpr (op == popops::expr::UnaryOpType::ABSOLUTE) {
      UNARY_LOOP_1_INSTRUCTION("f32v2absadd", "1", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::EXPONENT) {
      UNARY_LOOP_2_INSTRUCTION("f32exp", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::LOGARITHM) {
      UNARY_LOOP_2_INSTRUCTION("f32ln", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::INVERSE) {
      UNARY_LOOP_2_INSTRUCTION("f32oox", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::NEGATE) {
      UNARY_LOOP_1_INSTRUCTION("f32v2sub", "1", "$azeros")
    } else if constexpr (op == popops::expr::UnaryOpType::SQRT) {
      UNARY_LOOP_2_INSTRUCTION("f32sqrt", "1")
    } else if constexpr (op == popops::expr::UnaryOpType::SQUARE) {
      UNARY_LOOP_1_INSTRUCTION("f32v2mul", "1", "$a0:1")
    } else if constexpr (op == popops::expr::UnaryOpType::RSQRT) {
      UNARY_LOOP_2_INSTRUCTION("f32oorx", "1")
    }
    return std::make_pair(inPtr, outPtr);
  }
};
