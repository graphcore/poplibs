// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
// --------------------------------------------------------------------------
// 'Core' functions to perform conversion from floating point to 8 bit
// integer types.
//
// Each function processes a linear vector of elements:
// The input registers, as defined in CastCommon.inc, are:
//
//   $IN_PTR   Start of input data vector
//   $OUT_PTR  Start of output vector (aligned to 4 bytes)
//   $N_ELEMS  Number of elements to process
//
// The functions can clobber all MRF and ARF registers, except:
//    $lr (m10)
//    $CLAMP_MIN, $CLAMP_MAX (defined in CastCommon.inc)
//
// --------------------------------------------------------------------------
#include "CastCommon.inc"

// This is probably already defined in some include file
#define TFPU_ROUND_ZERO    3


#define REMAINDER             m5

#define MEM_WORD              m7
#define ACCUM                 m0
#define ACCUM1                m1
#define SCRATCH               m6


// Macro that specifies the instruction to round toward zero (truncation), used
// in the cast conversion.
.macro ROUND_TO_ZERO A_REG
  f32int      \A_REG, \A_REG, TFPU_ROUND_ZERO
.endm


.worker

//******************************************************************************
// Macro to be instantiated to create the functions for conversions of:
//
//   HALF => UNSIGNED_CHAR/SIGNED_CHAR
//
// Alignment required by the code in this macro is:
//  input  : 8 bytes
//  output : 4 bytes
//
// FN_NAME   Name of the function (label)
//
// F32TOINT  Either 'f32toui32' or 'f32toi32' for converting to
//           unsigned or signed char
//
// F32ROUND  A macro with one parameter, that contains the instruction to
//           perform the rounding, before calling F32TOINT.
//           Note that this macro can be empty (i.e. it doesn't contain an
//           instruction), the rounding will be the one performed by
//           f32toui32/f32toi32 (round to nearest, ties to even)
//
//******************************************************************************
.macro HALF_TO_CHAR_CORE  FN_NAME  F32TOINT  F32ROUND
.section .text.\FN_NAME\()
EXPORT_FN \FN_NAME
\FN_NAME:
  and         $REMAINDER, $N_ELEMS, 3

  // Convert N_ELEMS to the number of 4 elements atoms
  shr         $N_ELEMS, $N_ELEMS, 2
  brz         $N_ELEMS, .Lremainder_half_to_char\@

  add         $N_ELEMS, $N_ELEMS, -2  // because 1 atom of four is processed
                                      // in Prologue + Epilogue and 1 is for
                                      // brnzdec

  // Optimized pipelined processing for 4 elements atoms

  // Prologue (fill)
  ld64step    $a4:5, $mzero, $IN_PTR+=, 1
  f16v2tof32  $a0:1, $a4
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  f32clamp    $a1, $a1, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32ROUND   $a1
  \F32TOINT   $a0, $a0
  \F32TOINT   $a1, $a1

  // Main pipelined loop over blocks of 4 elements
 {brneg       $N_ELEMS, .L_half_to_8bit_loop_end\@
  sort4x16lo  $a0, $a0, $a1}
.L_half_to_8bit_loop\@:
 {atom        $ACCUM, $a0
  f16v2tof32  $a2:3, $a5}
  f32clamp    $a2, $a2, $CLAMP_LIMITS
  f32clamp    $a3, $a3, $CLAMP_LIMITS
  \F32ROUND   $a2
  \F32ROUND   $a3
  \F32TOINT   $a2, $a2
  \F32TOINT   $a3, $a3
 {ld64step    $a4:5, $mzero, $IN_PTR+=, 1
  sort4x16lo  $a2, $a2, $a3}

 {atom        $ACCUM1, $a2
  f16v2tof32  $a0:1, $a4}
 {sort8x8lo   $ACCUM, $ACCUM, $ACCUM1
  f32clamp    $a0, $a0, $CLAMP_LIMITS}
 {st32step    $ACCUM, $mzero, $OUT_PTR+=, 1
  f32clamp    $a1, $a1, $CLAMP_LIMITS}
  \F32ROUND   $a0
  \F32ROUND   $a1
  \F32TOINT   $a0, $a0
  \F32TOINT   $a1, $a1
 {brnzdec     $N_ELEMS, .L_half_to_8bit_loop\@
  sort4x16lo  $a0, $a0, $a1}
.L_half_to_8bit_loop_end\@:
  // Epilogue (drain)
 {atom        $ACCUM, $a0
  f16v2tof32  $a2:3, $a5}
  f32clamp    $a2, $a2, $CLAMP_LIMITS
  f32clamp    $a3, $a3, $CLAMP_LIMITS
  \F32ROUND   $a2
  \F32ROUND   $a3
  \F32TOINT   $a2, $a2
  \F32TOINT   $a3, $a3
  sort4x16lo  $a2, $a2, $a3
  atom        $ACCUM1, $a2
  sort8x8lo   $ACCUM, $ACCUM, $ACCUM1
  st32step    $ACCUM, $mzero, $OUT_PTR+=, 1


.Lremainder_half_to_char\@:
  // 0, 1, 2 or 3 trailing elements to process

  brz         $REMAINDER, .Lend_remainder_half_to_char\@ // bail out if 0
  // 1, 2 or 3 trailing elements to process
  ld32        $MEM_WORD, $mzero, $OUT_PTR, 0 // load existing word from memory
  cmpeq       $SCRATCH, $REMAINDER, 1
  brnz        $SCRATCH, .Lremainder_1_half_to_char\@
  // 2 or 3 trailing elements. Need to process at least two
  ld32step    $a0, $mzero, $IN_PTR+=, 1
  f16v2tof32  $a0:1, $a0
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  f32clamp    $a1, $a1, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32ROUND   $a1
  \F32TOINT   $a0, $a0
  \F32TOINT   $a1, $a1
  sort4x16lo  $a0, $a0, $a1
  atom        $ACCUM, $a0
  sort8x8lo   $ACCUM, $ACCUM, $mzero
  // Is there a third element?
  and         $SCRATCH, $REMAINDER, 1
  brnz        $SCRATCH, 1f
  // There were just two, mix together MEM_WORD and ACCUM and write to mem
  shl         $ACCUM, $ACCUM, 16
  sort4x16hi  $MEM_WORD, $m0, $MEM_WORD
  st32        $MEM_WORD, $mzero, $OUT_PTR, 0
  br          $lr
1:// last element of three
  ldb16       $a0, $mzero, $IN_PTR, 0
  f16tof32    $a0, $a0
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32TOINT   $a0, $a0
  and         $a0, $a0, 0xff
  atom        $SCRATCH, $a0
  sort4x16lo  $ACCUM, $ACCUM, $SCRATCH
  // Isolate the lone element in MEM_WORD (top byte)
  shr         $MEM_WORD, $MEM_WORD, 24
  shl         $MEM_WORD, $MEM_WORD, 24
  // mix together MEM_WORD and ACCUM and write to mem
  or          $MEM_WORD, $MEM_WORD, $ACCUM
  st32        $MEM_WORD, $mzero, $OUT_PTR, 0
  br          $lr

.Lremainder_1_half_to_char\@:
  // There was only 1 trailing element
  ldb16       $a0, $mzero, $IN_PTR, 0
  f16tof32    $a0, $a0
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32TOINT   $a0, $a0
  and         $a0, $a0, 0xff
  atom        $ACCUM, $a0
  andc        $MEM_WORD, $MEM_WORD, 0xff
  or          $MEM_WORD, $MEM_WORD, $ACCUM
  st32        $MEM_WORD, $mzero, $OUT_PTR, 0
.Lend_remainder_half_to_char\@:
  br          $lr

FN_SIZE \FN_NAME

.endm



//******************************************************************************
// Macro to be instantiated to create the functions for:
//
//   FLOAT => UNSIGNED_CHAR/SIGNED_CHAR
//
// Alignment required by the code in this macro is:
//  input  : 8 bytes
//  output : 4 bytes
//
// The parameters are the same as for HALF_TO_CHAR_CORE
//
//******************************************************************************
.macro FLOAT_TO_CHAR_CORE  FN_NAME  F32TOINT  F32ROUND

.section .text.\FN_NAME\()
EXPORT_FN \FN_NAME
\FN_NAME:
  and         $REMAINDER, $N_ELEMS, 3

  // Convert N_ELEMS to the number of 4 elements atoms
  shr         $N_ELEMS, $N_ELEMS, 2
  brz         $N_ELEMS, .Lremainder_float_to_char\@

  add         $N_ELEMS, $N_ELEMS, -2  // because 1 atom of four is processed
                                      // in Prologue + Epilogue and 1 is for
                                      // brnzdec

  // Optimized pipelined processing for 4 elements atoms

  // Prologue (fill)
  ld64step    $a0:1, $mzero, $IN_PTR+=, 1
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  f32clamp    $a1, $a1, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32ROUND   $a1
  \F32TOINT   $a0, $a0
  \F32TOINT   $a1, $a1
 {ld64step    $a2:3, $mzero, $IN_PTR+=, 1
  sort4x16lo  $a0, $a0, $a1}

  // Main pipelined loop over blocks of 4 elements
  brneg       $N_ELEMS, .L_float_to_8bit_loop_end\@
.L_float_to_8bit_loop\@:
 {atom        $ACCUM, $a0
  f32clamp    $a2, $a2, $CLAMP_LIMITS}
  f32clamp    $a3, $a3, $CLAMP_LIMITS
  \F32ROUND   $a2
  \F32ROUND   $a3
  \F32TOINT   $a2, $a2
  \F32TOINT   $a3, $a3
 {ld64step    $a0:1, $mzero, $IN_PTR+=, 1
  sort4x16lo  $a2, $a2, $a3}
 {atom        $ACCUM1, $a2
  f32clamp    $a0, $a0, $CLAMP_LIMITS}
 {sort8x8lo   $ACCUM, $ACCUM, $ACCUM1
  f32clamp    $a1, $a1, $CLAMP_LIMITS}
  \F32ROUND   $a0
  \F32ROUND   $a1
 {st32step    $ACCUM, $mzero, $OUT_PTR+=, 1
  \F32TOINT   $a0, $a0}
 {ld64step    $a2:3, $mzero, $IN_PTR+=, 1
  \F32TOINT   $a1, $a1}
 {brnzdec     $N_ELEMS, .L_float_to_8bit_loop\@
  sort4x16lo  $a0, $a0, $a1}
.L_float_to_8bit_loop_end\@:
  // Epilogue (drain)
 {atom        $ACCUM, $a0
  f32clamp    $a2, $a2, $CLAMP_LIMITS}
  f32clamp    $a3, $a3, $CLAMP_LIMITS
  \F32ROUND   $a2
  \F32ROUND   $a3
  \F32TOINT   $a2, $a2
  \F32TOINT   $a3, $a3
  sort4x16lo  $a2, $a2, $a3
  atom        $ACCUM1, $a2
  sort8x8lo   $ACCUM, $ACCUM, $ACCUM1
  st32step    $ACCUM, $mzero, $OUT_PTR+=, 1

.Lremainder_float_to_char\@:
  // 0, 1, 2 or 3 trailing elements to process

  brz         $REMAINDER, .Lend_remainder_float_to_char\@ // bail out if 0
  //   1 elems: 12 cycles
  //   2 elems: 19 cycles
  //   3 elems: 25 cycles
  // 1, 2 or 3 trailing elements to process
  ld32        $MEM_WORD, $mzero, $OUT_PTR, 0 // load existing word from memory
  cmpeq       $SCRATCH, $REMAINDER, 1
  brnz        $SCRATCH, .Lremainder_1_float_to_char\@
  // 2 or 3 trailing elements. Need to process at least two
  ld32step    $a0, $mzero, $IN_PTR+=, 1
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32TOINT   $a0, $a0
  ld32step    $a1, $mzero, $IN_PTR+=, 1
  f32clamp    $a1, $a1, $CLAMP_LIMITS
  \F32ROUND   $a1
  \F32TOINT   $a1, $a1
  atom        $m0, $a0
  atom        $m1, $a1
  shuf8x8lo   $ACCUM, $ACCUM, $m1
  // Is there a third element?
  and         $SCRATCH, $REMAINDER, 1
  brnz        $SCRATCH, 1f
  // There were just two, mix together MEM_WORD and ACCUM and write to mem
  shl         $ACCUM, $ACCUM, 16
  sort4x16hi  $MEM_WORD, $ACCUM, $MEM_WORD
  st32        $MEM_WORD, $mzero, $OUT_PTR, 0
  br          $lr
1:// last element of three
  ld32        $a0, $mzero, $IN_PTR, 0
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32TOINT   $a0, $a0
  and         $a0, $a0, 0xff
  atom        $m6, $a0
  sort4x16lo  $ACCUM, $ACCUM, $m6
  // Isolate the lone element in MEM_WORD (top byte)
  shr         $MEM_WORD, $MEM_WORD, 24
  shl         $MEM_WORD, $MEM_WORD, 24
  // mix together MEM_WORD and ACCUM and write to mem
  or          $MEM_WORD, $MEM_WORD, $ACCUM
  st32        $MEM_WORD, $mzero, $OUT_PTR, 0
  br          $lr

.Lremainder_1_float_to_char\@:
  // There was only 1 trailing element
  ld32        $a0, $mzero, $IN_PTR, 0
  f32clamp    $a0, $a0, $CLAMP_LIMITS
  \F32ROUND   $a0
  \F32TOINT   $a0, $a0
  and         $a0, $a0, 0xff
  atom        $ACCUM, $a0
  andc        $MEM_WORD, $MEM_WORD, 0xff
  or          $MEM_WORD, $MEM_WORD, $ACCUM
  st32        $MEM_WORD, $mzero, $OUT_PTR, 0
.Lend_remainder_float_to_char\@:
  br          $lr

FN_SIZE \FN_NAME
.endm


// Now create the code by instantiating the macros

HALF_TO_CHAR_CORE   cast_half_unsigned_char_core  f32toui32  ROUND_TO_ZERO
HALF_TO_CHAR_CORE   cast_half_signed_char_core    f32toi32   ROUND_TO_ZERO

FLOAT_TO_CHAR_CORE  cast_float_unsigned_char_core  f32toui32  ROUND_TO_ZERO
FLOAT_TO_CHAR_CORE  cast_float_signed_char_core    f32toi32   ROUND_TO_ZERO

#endif
