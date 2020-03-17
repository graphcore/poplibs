// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#include "poplar/StackSizeDefs.hpp"

#define SELECT_BOOL __runCodelet_popops__Select___bool

.globl SELECT_BOOL
.type SELECT_BOOL, @function

DEF_STACK_USAGE 0 SELECT_BOOL
.section .text.SELECT_BOOL
.align 8

// Constants
#define VERTEX_IN1_OFFSET 0
#define VERTEX_IN2_OFFSET 1
#define VERTEX_IN3_OFFSET 2
#define VERTEX_OUT_START_OFFSET 3
#define VERTEX_OUT_END_OFFSET 4

#define VALUES_PER_WORD 4

// Integer variables
#define out m0
#define in1_ptr m1
#define in2_ptr m2
#define in3_ptr m3
#define out_ptr m4
#define iRow m5
#define nRowRemain m6
#define nColRemain m7
#define nByteRemain m8 // NOTE: Same register as ifVal
#define values4 m9
#define value m10
#define trueVal m11 // NOTE: Same register as mask
#define ifVal m8 // NOTE: Same register as nByteRemain
#define mask m11 // NOTE: Same register as trueVal

  nop
SELECT_BOOL:
  // Load the vertex state.
  ld32 $out, $mvertex_base, $mzero, VERTEX_OUT_START_OFFSET
  ld32 $nRowRemain, $mvertex_base, $mzero, VERTEX_OUT_END_OFFSET
  brz $nRowRemain, row_loop_end
  add $nRowRemain, $nRowRemain, -1
  setzi $iRow, 0

row_loop_start:
  // Reload in pointers
  ld32 $in1_ptr, $mvertex_base, $mzero, VERTEX_IN1_OFFSET
  ld32 $in2_ptr, $mvertex_base, $mzero, VERTEX_IN2_OFFSET
  ld32 $in3_ptr, $mvertex_base, $mzero, VERTEX_IN3_OFFSET

  // Load in-rows
  ld32 $in1_ptr, $mzero, $in1_ptr, $iRow
  ld32 $in2_ptr, $mzero, $in2_ptr, $iRow
  ld32 $in3_ptr, $mzero, $in3_ptr, $iRow

  // Load next out pointer and calculate number of inner elements
  ld32step $out_ptr, $mzero, $out+=, 1
  ld32step $nColRemain, $mzero, $out+=, 1

  brz $nColRemain, load_next_row
  and $value, $nColRemain, 3 // Number of columns mod 4
  st32 $value, $mzero, $m12, 0 // Put it on the stack for later

column_loop_start:
  min $nByteRemain, $nColRemain, VALUES_PER_WORD
  sub $nColRemain, $nColRemain, $nByteRemain

  rpt $nByteRemain, ((byte_loop_end-byte_loop_start)/8)-1
byte_loop_start:
  {ldz8step $trueVal, $mzero, $in1_ptr+=, 1; fnop}
  {ldz8step $value, $mzero, $in2_ptr+=, 1; fnop}
  {ldz8step $ifVal, $mzero, $in3_ptr+=, 1; fnop}
  {movz $value, $ifVal, $trueVal; fnop}
  {roll8r $values4, $values4, $value; fnop}

byte_loop_end:
  brz $nColRemain, column_loop_end
  st32step $values4, $mzero, $out_ptr+=, 1
  bri column_loop_start

column_loop_end:
  ld32 $value, $mzero, $m12, 0 // Retrieve number of columns mod 2 from stack
  brnz $value, merge_partial_value
  bri write_value

merge_partial_value:
  shl $value, $value, 3 // Multiply by 8
  xnor $mask, $mzero, $mzero // Set mask to all ones
  shl $mask, $mask, $value // Align mask
  sub $value, 32, $value
  shr $values4, $values4, $value // Align (partial) value

  ld32 $value, $mzero, $out_ptr, 0
  and $value, $value, $mask
  or $values4, $values4, $value // Merge values

write_value:
  st32step $values4, $mzero, $out_ptr+=, 1

load_next_row:
  add $iRow, $iRow, 1
  brnzdec $nRowRemain, row_loop_start

row_loop_end:
  exitz $mzero

.size SELECT_BOOL, .-SELECT_BOOL

#endif
