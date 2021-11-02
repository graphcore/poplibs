// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
// --------------------------------------------------------------------------
// 'Core' functions to perform casts FROM 8 bit types
//
// Each function processes a linear vector of elements:
// The input registers, as defined in CastCommon.inc, are:
//
//   $IN_PTR   Start of input data vector
//   $OUT_PTR  Start of output vector (aligned to 4 bytes)
//   $N_ELEMS  Number of elements to process
//
// The functions can clobber all MRF and ARF registers, except $lr (m10)
// --------------------------------------------------------------------------
#include "CastCommon.inc"

#define REMAINDER             m5

#define ACCUM                 m0
#define SCRATCH               m6

.worker

//******************************************************************************
// Process the trailing 0..3 signed/unsigned remainder elements remaining after
// the optimized 4-element loop, when converting to HALF.
//
//   LDCHAR      Either 'ldz8step' or 'lds8step' for unsigned or signed values
//   F32FROMINT  Either 'f32fromui32' or 'f32fromi32' for unsigned or signed
//
// Registers:
//   $IN_PTR   : Points to the first input element in the remainder
//   $OUT_PTR  : Points to the first output element where to store the remainder
//   $REMAINDER: Elements in the remainder (0, 1, 2 or 3)
//******************************************************************************
.macro PROCESS_REMAINDER_CHAR_TO_HALF  LDCHAR  F32FROMINT
  // 0, 1, 2 or 3 trailing elements to process
  brz         $REMAINDER, .Lend_remainder_char_to_half\@ // bail out if 0
  // was there only 1 element?
  cmpeq       $SCRATCH, $REMAINDER, 1
  brnz        $SCRATCH, .L_1_char_to_half\@

  // 2 or 3 trailing elements. Need to process at least two
  \LDCHAR     $ACCUM, $mzero, $IN_PTR+=, 1
  // Use the output word as storage to move from 'm' reg to 'a' reg
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0
 {\LDCHAR     $ACCUM, $mzero, $IN_PTR+=, 1
  \F32FROMINT $a0, $a0}
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a1, $mzero, $OUT_PTR, 0
 {cmpeq       $SCRATCH, $REMAINDER, 3
  \F32FROMINT $a1, $a1}
  f32v2tof16  $a0, $a0:1
  st32step    $a0, $mzero, $OUT_PTR+=, 1
  // Were there 3 trailing elems? if not, bail out
  brz         $SCRATCH, .Lend_remainder_char_to_half\@

.L_1_char_to_half\@:
  ldb16       $a2, $mzero, $OUT_PTR, 1 // load to-be-preserved hword from mem
  \LDCHAR     $ACCUM, $mzero, $IN_PTR+=, 1
  // Use the output word as storage to move from 'm' reg to 'a' reg
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0
  \F32FROMINT $a0, $a0
  f32tof16    $a0, $a0
  sort4x16lo  $a0, $a0, $a2 // mix last result and to-be-preserved hword
  st32step    $a0, $mzero, $OUT_PTR+=, 1
.Lend_remainder_char_to_half\@:
  br          $lr
.endm

//******************************************************************************
// UNSIGNED_CHAR => HALF
// Alignment required by the code in this function is:
//  input  : 2 bytes
//  output : 8 bytes
//******************************************************************************
FN_SECTION cast_unsigned_char_half_core 8 nop
FN_EXPORT cast_unsigned_char_half_core

  and         $REMAINDER, $N_ELEMS, 3
  shr         $N_ELEMS, $N_ELEMS, 2
  brz         $N_ELEMS, .Lremainder_unsigned_char_to_half

  add         $N_ELEMS, $N_ELEMS, -1  // because 1 batch of four is processed
                                      // in Prologue + Epilogue

  // Optimized pipelined processing for multiple of 4 elements

  // Prologue (fill)
  ldz16step   $m0, $mzero, $IN_PTR+=, 1 # load 2 x 8-bit values into MRF
  shuf8x8lo   $m0, $m0, $mzero          # change the 2 8-bit to 2 x 16-bit
  st32        $m0, $mzero, $OUT_PTR, 0  # store the 2x16-bit values
  ld32        $a4, $mzero, $OUT_PTR, 0  # load the 2x16-bit values into the ARF
 {ldz16step   $m0, $mzero, $IN_PTR+=, 1
  sort4x16lo  $a0, $a4, $azero }        # Create a zero-extended 32-bit uint

  // Main pipelined loop over blocks of 4 elements
  rpt         $N_ELEMS,((2f-1f)/8)-1
1:
 {shuf8x8lo   $m0, $m0, $mzero
  sort4x16hi  $a1, $a4, $azero }           # Create 2nd zero-extended 32-bit
 {st32        $m0, $mzero, $OUT_PTR, 0
  f32fromui32 $a0, $a0       }             # Convert 32-bit uint to float
 {ld32        $a4, $mzero, $OUT_PTR, 0
  f32fromui32 $a1, $a1       }             # Convert 2nd 32-bit uint to float
 {ldz16step   $m0, $mzero, $IN_PTR+=, 1
  sort4x16lo  $a2, $a4, $azero }
 {shuf8x8lo   $m0, $m0, $mzero
  sort4x16hi  $a3, $a4, $azero }
 {st32        $m0, $mzero, $OUT_PTR, 0
  f32fromui32 $a2, $a2       }             # Convert 3rd 32-bit uint to float
 {ld32        $a4, $mzero, $OUT_PTR, 0
  f32fromui32 $a3, $a3       }             # Convert 4th 32-bit uint to float
 {ldz16step   $m0, $mzero, $IN_PTR+=, 1
  f32v4tof16  $a0:1, $a0:3   }             # Convert the 4 floats to f16
 {st64step    $a0:1, $mzero, $OUT_PTR+=, 1 # Store out the 4xf16 values
  sort4x16lo  $a0, $a4, $azero }
2:
  // Epilogue (drain)
 {shuf8x8lo   $m0, $m0, $mzero
  sort4x16hi  $a1, $a4, $azero }
 {st32        $m0, $mzero, $OUT_PTR, 0
  f32fromui32 $a0, $a0       }
 {ld32        $a4, $mzero, $OUT_PTR, 0
  f32fromui32 $a1, $a1       }
  sort4x16lo  $a2, $a4, $azero
  sort4x16hi  $a3, $a4, $azero
  f32fromui32 $a2, $a2
  f32fromui32 $a3, $a3
  f32v4tof16  $a0:1, $a0:3
  st64step    $a0:1, $mzero, $OUT_PTR+=, 1

.Lremainder_unsigned_char_to_half:
  PROCESS_REMAINDER_CHAR_TO_HALF  ldz8step  f32fromui32
FN_SIZE cast_unsigned_char_half_core

//******************************************************************************
// SIGNED_CHAR => HALF
// Alignment required by the code in this function is:
//  input  : 1 byte
//  output : 4 bytes
//******************************************************************************
FN_SECTION cast_signed_char_half_core 8 
FN_EXPORT cast_signed_char_half_core

  and         $REMAINDER, $N_ELEMS, 3
  shr         $N_ELEMS, $N_ELEMS, 2
  brz         $N_ELEMS, .Lremainder_signed_char_to_half

  add         $N_ELEMS, $N_ELEMS, -1  // because 1 element is processed
                                      // in Prologue + Epilogue
  // Optimized pipelined processing for multiple of 4 value

  // Prologue (fill))
  lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0

  // Main pipelined loop over 4 elements
  rpt         $N_ELEMS, ((2f-1f)/8)-1
1:
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a2, $a0}
 {st32        $ACCUM, $mzero, $OUT_PTR, 0
  fnop}
 {ld32        $a0, $mzero, $OUT_PTR, 0
  fnop}
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a3, $a0}
 {st32        $ACCUM, $mzero, $OUT_PTR, 0
  f32v2tof16  $a4, $a2:3}
 {ld32        $a0, $mzero, $OUT_PTR, 0
  fnop}
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a2, $a0}
 {st32        $ACCUM, $mzero, $OUT_PTR, 0
  fnop}
 {ld32        $a0, $mzero, $OUT_PTR, 0
  fnop}
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a3, $a0}
 {st32        $ACCUM, $mzero, $OUT_PTR, 0
  f32v2tof16  $a5, $a2:3}
 {ld32        $a0, $mzero, $OUT_PTR, 0
  fnop}
 {st64step    $a4:5, $mzero, $OUT_PTR+=, 1
  fnop}
2:
  // Epilogue (drain)
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a2, $a0}
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a3, $a0}
 {st32        $ACCUM, $mzero, $OUT_PTR, 0
  f32v2tof16  $a4, $a2:3}
  ld32        $a0, $mzero, $OUT_PTR, 0
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a2, $a0}
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0
  f32fromi32  $a3, $a0
  f32v2tof16  $a5, $a2:3
  st64step    $a4:5, $mzero, $OUT_PTR+=, 1

.Lremainder_signed_char_to_half:
  PROCESS_REMAINDER_CHAR_TO_HALF  lds8step f32fromi32
FN_SIZE cast_signed_char_half_core


//******************************************************************************
// Process the trailing 0..3 signed/unsigned remainder elements remaining after
// the optimized 4-element loop, when converting to FLOAT.
//
//   LDCHAR      Either 'ldz8step' or 'lds8step' for unsigned or signed values
//   F32FROMINT  Either 'f32fromui32' or 'f32fromi32' for unsigned or signed
//
// Registers:
//   $IN_PTR   : Points to the first input element in the remainder
//   $OUT_PTR  : Points to the first output element where to store the remainder
//   $REMAINDER: Elements in the remainder (0, 1, 2 or 3)
//******************************************************************************
.macro PROCESS_REMAINDER_CHAR_TO_FLOAT  LDCHAR  F32FROMINT
  brnzdec     $REMAINDER, .Loop_char_to_float\@
  br          $lr
.Loop_char_to_float\@:
  \LDCHAR     $ACCUM, $mzero, $IN_PTR+=, 1
  // Use the output word as storage to move from 'm' reg to 'a' reg
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0
  \F32FROMINT $a0, $a0
  st32step    $a0, $mzero, $OUT_PTR+=, 1
  brnzdec     $REMAINDER, .Loop_char_to_float\@
  br          $lr
.endm

//******************************************************************************
// UNSIGNED_CHAR => FLOAT
// Alignment required by the code in this function is:
//  input  : 2 bytes
//  output : 8 bytes
//******************************************************************************
FN_SECTION cast_unsigned_char_float_core 8 nop
FN_EXPORT cast_unsigned_char_float_core
  and         $REMAINDER, $N_ELEMS, 3
  shr         $N_ELEMS, $N_ELEMS, 2
  brz         $N_ELEMS, .Lremainder_unsigned_char_to_float

  add         $N_ELEMS, $N_ELEMS, -1  // because 1 batch of four is processed
                                      // in Prologue + Epilogue

  // Optimized pipelined processing for multiple of 4 value

  // Prologue (fill)
  ldz16step   $m0, $mzero, $IN_PTR+=, 1 # load 2 x 8-bit values into MRF
  shuf8x8lo   $m0, $m0, $mzero          # change the 2 8-bitto 2 x 16-bit
  st32        $m0, $mzero, $OUT_PTR, 0  # store the 2x16-bit values
  ld32        $a4, $mzero, $OUT_PTR, 0  # load the 2x16-bit values into the ARF
 {ldz16step   $m0, $mzero, $IN_PTR+=, 1
  sort4x16lo  $a0, $a4, $azero}         # Create a zero-extended 32-bit uint

  // Main pipelined loop over 4 elements
  rpt         $N_ELEMS,((2f-1f)/8)-1
1:
 {shuf8x8lo   $m0, $m0, $mzero
  sort4x16hi  $a1, $a4, $azero}            # Create 2nd zero-extended 32-bit
 {st32        $m0, $mzero, $OUT_PTR, 0
  f32fromui32 $a0, $a0}                    # Convert 32-bit uint to float
 {ld32        $a4, $mzero, $OUT_PTR, 0
  f32fromui32 $a1, $a1}                    # Convert 2nd 32-bit uint to float
 {ldz16step   $m0, $mzero, $IN_PTR+=, 1
  sort4x16lo  $a2, $a4, $azero}
 {shuf8x8lo   $m0, $m0, $mzero
  sort4x16hi  $a3, $a4, $azero}
 {st32        $m0, $mzero, $OUT_PTR, 0
  f32fromui32 $a2, $a2}                    # Convert 3rd 32-bit uint to float
 {ld32        $a4, $mzero, $OUT_PTR, 0
  f32fromui32 $a3, $a3}                    # Convert 4th 32-bit uint to float
 {ldz16step   $m0, $mzero, $IN_PTR+=, 1
  fnop}
 {st64step    $a0:1, $mzero, $OUT_PTR+=, 1 # Store out the first pair of values
  sort4x16lo  $a0, $a4, $azero}
 {st64step    $a2:3, $mzero, $OUT_PTR+=, 1 # Store out the second pair
  fnop}
2:
  // Epilogue (drain)
 {shuf8x8lo   $m0, $m0, $mzero
  sort4x16hi  $a1, $a4, $azero}
 {st32        $m0, $mzero, $OUT_PTR, 0
  f32fromui32 $a0, $a0}
 {ld32        $a4, $mzero, $OUT_PTR, 0
  f32fromui32 $a1, $a1}
  sort4x16lo  $a2, $a4, $azero
  sort4x16hi  $a3, $a4, $azero
  f32fromui32 $a2, $a2
  f32fromui32 $a3, $a3
  st64step    $a0:1, $mzero, $OUT_PTR+=, 1
  st64step    $a2:3, $mzero, $OUT_PTR+=, 1

.Lremainder_unsigned_char_to_float:
  PROCESS_REMAINDER_CHAR_TO_FLOAT  ldz8step  f32fromui32
FN_SIZE cast_unsigned_char_float_core


//******************************************************************************
// SIGNED_CHAR => FLOAT
// Alignment required by the code in this function is:
//  input  : 1 byte
//  output : 8 bytes
//******************************************************************************
FN_SECTION cast_signed_char_float_core 8
FN_EXPORT cast_signed_char_float_core
  and         $REMAINDER, $N_ELEMS, 1
  shr         $N_ELEMS, $N_ELEMS, 1
  brz         $N_ELEMS, .Lremainder_signed_char_to_float

  add         $N_ELEMS, $N_ELEMS, -1  // because 1 element is processed
                                      // in Prologue + Epilogue

  // Pipelined processing over pairs of elements

  // Prologue (fill))
  lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0

  rpt         $N_ELEMS, ((2f-1f)/8)-1
1:
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a2, $a0}
 {st32        $ACCUM, $mzero, $OUT_PTR, 0
  fnop}
 {ld32        $a1, $mzero, $OUT_PTR, 0
  fnop}
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a3, $a1}
 {st32        $ACCUM, $mzero, $OUT_PTR, 0
  fnop}
 {ld32        $a0, $mzero, $OUT_PTR, 0
  fnop}
 {st64step    $a2:3, $mzero, $OUT_PTR+=, 1
  fnop}
2:
  // Epilogue (drain)
 {lds8step    $ACCUM, $mzero, $IN_PTR+=, 1
  f32fromi32  $a2, $a0}
  st32        $ACCUM, $mzero, $OUT_PTR, 0
  ld32        $a0, $mzero, $OUT_PTR, 0
  f32fromi32  $a3, $a0
  st64step    $a2:3, $mzero, $OUT_PTR+=, 1

.Lremainder_signed_char_to_float:
  PROCESS_REMAINDER_CHAR_TO_FLOAT  lds8step  f32fromi32
FN_SIZE cast_signed_char_float_core

#endif
