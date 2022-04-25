// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

enum class TemplateInstructions {
  // f16v4 instructions
  f16v4absadd,
  f16v4mul,
  f16v4sub,
  // f16v2 instructions
  f16v2exp,
  f16v2ln,
  // f32v2 instructions
  f32v2absadd,
  f32v2sub,
  f32v2mul,
  // f32 instructions
  f32oox,
  f32sqrt,
  f32oorx,
  f32exp,
  f32ln
};

template <TemplateInstructions instruction>
static __attribute__((always_inline)) void selectInstruction() {
  if constexpr (instruction == TemplateInstructions::f16v4absadd) {
    asm volatile(R"l( .macro instruction OP1 OP2 OP3:vararg
                      f16v4absadd \OP1, $azeros, \OP3; .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f16v2exp) {
    asm volatile(R"l( .macro instruction OPERANDS:vararg
                      f16v2exp \OPERANDS; .endm;)l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f16v2ln) {
    asm volatile(R"l( .macro instruction OPERANDS:vararg
                      f16v2ln \OPERANDS; .endm;)l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32oox) {
    asm volatile(R"l( .macro instruction OPERANDS:vararg
                      f32oox \OPERANDS;.endm;)l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f16v4sub) {
    asm volatile(R"l( .macro instruction OP1 OP2 OP3:vararg
                      f16v4sub \OP1, $azeros, \OP3; .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32sqrt) {
    asm volatile(R"l( .macro instruction OPERANDS:vararg
                      f32sqrt \OPERANDS; .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f16v4mul) {
    asm volatile(R"l( .macro instruction OP1 OP2 OP3:vararg
                      f16v4mul \OP1, \OP2, \OP3;  .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32oorx) {
    asm volatile(R"l( .macro instruction OPERANDS:vararg
                      f32oorx \OPERANDS;  .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32v2absadd) {
    asm volatile(R"l( .macro instruction OP1 OP2 OP3:vararg
                      f32v2absadd \OP1, $azeros, \OP3;  .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32exp) {
    asm volatile(R"l( .macro instruction OPERANDS:vararg
                      f32exp \OPERANDS; .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32ln) {
    asm volatile(R"l( .macro instruction OPERANDS:vararg
                      f32ln \OPERANDS;  .endm;  )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32v2sub) {
    asm volatile(R"l( .macro instruction OP1 OP2 OP3:vararg
                      f32v2sub \OP1, $azeros, \OP3;  .endm; )l" ::
                     :);
  } else if constexpr (instruction == TemplateInstructions::f32v2mul) {
    asm volatile(R"l( .macro instruction OP1 OP2 OP3:vararg
                      f32v2mul \OP1, \OP2, \OP3;  .endm; )l" ::
                     :);
  }
}

// Function producing a rpt loop for unary ops where the instruction supports
// processing of 64 bits in 1 instruction (f32v2... or f16v4...)
template <typename T, TemplateInstructions instruction, unsigned stride>
static __attribute__((always_inline)) std::pair<const T *, T *>
unaryLoop1Instruction(unsigned loopCount, const T *inPtr, T *outPtr) {

  selectInstruction<instruction>();

  asm volatile(
      R"l(         brnzdec   %[loopCount], 3f
                   bri       4f
               .align 8
               3:
                 ld64step $a0:1, $mzero, %[inPtr]+=, %[stride]
                 rpt %[loopCount], (2f-1f)/8 -1
               1:
                     {ld64step $a0:1, $mzero, %[inPtr]+=, %[stride]
                     instruction $a2:3, $a0:1, $a0:1}
                     {st64step $a2:3, $mzero, %[outPtr]+=, %[stride]
                      fnop}
               2:
                   instruction $a2:3, $a0:1, $a0:1
                   st64step $a2:3, $mzero, %[outPtr]+=, %[stride]
               4:
               // Undefine the macro so it can be redefined later
               .purgem instruction
               )l"
      : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
      : [stride] "i"(stride)
      : "$a0:1", "$a2:3", "memory");
  return std::make_pair(inPtr, outPtr);
}

// Function producing a rpt loop for unary ops where the instruction supports
// processing of 64 bits in 2 instructions (2 of f32... or f16v2...)
template <typename T, TemplateInstructions instruction, unsigned stride>
static __attribute__((always_inline)) std::pair<const T *, T *>
unaryLoop2Instruction(unsigned loopCount, const T *inPtr, T *outPtr) {

  selectInstruction<instruction>();

  asm volatile(
      R"l(         brnzdec   %[loopCount], 3f
                   bri       4f
               .align 8
                   nop /* rpt alignment*/
               3:
                 ld64step $a0:1, $mzero, %[inPtr]+=, %[stride]
                 {rpt %[loopCount], (2f-1f)/8 -1
                 instruction $a2, $a0}
               1:
                     {ld64step $a0:1, $mzero, %[inPtr]+=, %[stride]
                     instruction $a3, $a1}
                     {st64step $a2:3, $mzero, %[outPtr]+=, %[stride]
                     instruction $a2, $a0}
               2:
                   instruction $a3, $a1
                   st64step $a2:3, $mzero, %[outPtr]+=, %[stride]
               4:
               // Undefine the macro so it can be redefined later
               .purgem instruction
               )l"
      : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
      : [stride] "i"(stride)
      : "$a0:1", "$a2:3", "memory");
  return std::make_pair(inPtr, outPtr);
}

// Function producing a rpt loop for unary ops with type half by casting to fp32
// and using 4 fp32 instructions
template <typename T, TemplateInstructions instruction, unsigned stride>
static __attribute__((always_inline)) std::pair<const T *, T *>
unaryLoop4InstructionCast(unsigned loopCount, const T *inPtr, T *outPtr) {

  selectInstruction<instruction>();

  asm volatile(
      R"l(         brnzdec   %[loopCount], 3f
                   bri       4f
               .align 8
               3:
                 ld64step $a0:1, $mzero, %[inPtr]+=, %[stride]
                 f16v2tof32 $a2:3, $a1
                 {rpt %[loopCount], (2f-1f)/8 -1
                  instruction $a2, $a2}
               1:
                     {nop
                      instruction $a3, $a3}
                     {nop
                      f16v2tof32 $a0:1, $a0}
                     {nop
                      instruction $a0, $a0}
                     {nop
                      instruction $a1, $a1}
                     {ld64step $a0:1, $mzero, %[inPtr]+=, %[stride]
                      f32v4tof16 $a2:3, $a0:3}
                     {st64step $a2:3, $mzero, %[outPtr]+=, %[stride]
                      f16v2tof32 $a2:3, $a1}
                     {nop
                      instruction $a2, $a2}
               2:
                     instruction $a3, $a3
                     f16v2tof32 $a0:1, $a0
                     instruction $a0, $a0
                     instruction $a1, $a1
                     f32v4tof16 $a2:3, $a0:3
                     st64step $a2:3, $mzero, %[outPtr]+=, %[stride]
               4:
               // Undefine the macro so it can be redefined later
               .purgem instruction
               )l"
      : [loopCount] "+r"(loopCount), [inPtr] "+r"(inPtr), [outPtr] "+r"(outPtr)
      : [stride] "i"(stride)
      : "$a0:1", "$a2:3", "memory");
  return std::make_pair(inPtr, outPtr);
}
// Default template picks up the case for half data
template <popops::expr::UnaryOpType op, typename T, unsigned stride>
class inlineAssemblerUnaryOp {
public:
  static __attribute__((always_inline)) std::pair<const half4 *, half4 *>
  loopBody(unsigned loopCount, const half4 *inPtr, half4 *outPtr) {

    if constexpr (op == popops::expr::UnaryOpType::ABSOLUTE) {
      return unaryLoop1Instruction<half4, TemplateInstructions::f16v4absadd,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::EXPONENT) {
      return unaryLoop2Instruction<half4, TemplateInstructions::f16v2exp,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::LOGARITHM) {
      return unaryLoop2Instruction<half4, TemplateInstructions::f16v2ln,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::INVERSE) {
      return unaryLoop4InstructionCast<half4, TemplateInstructions::f32oox,
                                       stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::NEGATE) {
      return unaryLoop1Instruction<half4, TemplateInstructions::f16v4sub,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::SQRT) {
      return unaryLoop4InstructionCast<half4, TemplateInstructions::f32sqrt,
                                       stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::SQUARE) {
      return unaryLoop1Instruction<half4, TemplateInstructions::f16v4mul,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::RSQRT) {
      return unaryLoop4InstructionCast<half4, TemplateInstructions::f32oorx,
                                       stride>(loopCount, inPtr, outPtr);
    }
    return std::make_pair(inPtr, outPtr);
  }
};

template <popops::expr::UnaryOpType op, unsigned stride>
class inlineAssemblerUnaryOp<op, float, stride> {
public:
  static __attribute__((always_inline)) std::pair<const float2 *, float2 *>
  loopBody(unsigned loopCount, const float2 *inPtr, float2 *outPtr) {

    if constexpr (op == popops::expr::UnaryOpType::ABSOLUTE) {
      return unaryLoop1Instruction<float2, TemplateInstructions::f32v2absadd,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::EXPONENT) {
      return unaryLoop2Instruction<float2, TemplateInstructions::f32exp,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::LOGARITHM) {
      return unaryLoop2Instruction<float2, TemplateInstructions::f32ln, stride>(
          loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::INVERSE) {
      return unaryLoop2Instruction<float2, TemplateInstructions::f32oox,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::NEGATE) {
      return unaryLoop1Instruction<float2, TemplateInstructions::f32v2sub,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::SQRT) {
      return unaryLoop2Instruction<float2, TemplateInstructions::f32sqrt,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::SQUARE) {
      return unaryLoop1Instruction<float2, TemplateInstructions::f32v2mul,
                                   stride>(loopCount, inPtr, outPtr);
    } else if constexpr (op == popops::expr::UnaryOpType::RSQRT) {
      return unaryLoop2Instruction<float2, TemplateInstructions::f32oorx,
                                   stride>(loopCount, inPtr, outPtr);
    }
    return std::make_pair(inPtr, outPtr);
  }
};
