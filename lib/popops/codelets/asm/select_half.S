// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

#define SELECT_HALF __runCodelet_popops__Select___half

// Constants
#define VERTEX_IN1_OFFSET 0
#define VERTEX_IN2_OFFSET 1
#define VERTEX_IN3_OFFSET 2
#define VERTEX_OUT_START_OFFSET 3
#define VERTEX_OUT_END_OFFSET 4

#define VALUES_PER_WORD 2

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
#define values2 m9
#define value m10
#define trueVal m11
#define ifVal m8 // NOTE: Same register as nByteRemain

FN_WORKER_ENTRY_POINT SELECT_HALF 8 nop
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
  and $value, $nColRemain, 1 // Number of columns mod 2
  st32 $value, $mzero, $m12, 0 // Put it on the stack for later

column_loop_start:
  min $nByteRemain, $nColRemain, VALUES_PER_WORD
  sub $nColRemain, $nColRemain, $nByteRemain

  rpt $nByteRemain, ((byte_loop_end-byte_loop_start)/8)-1
byte_loop_start:
  {ldz16step $trueVal, $mzero, $in1_ptr+=, 1; fnop}
  {ldz16step $value, $mzero, $in2_ptr+=, 1; fnop}
  {ldz8step $ifVal, $mzero, $in3_ptr+=, 1; fnop}
  {movz $value, $ifVal, $trueVal; fnop}
  {roll16 $values2, $values2, $value; fnop}

byte_loop_end:
  brz $nColRemain, column_loop_end
  st32step $values2, $mzero, $out_ptr+=, 1
  bri column_loop_start

column_loop_end:
  ld32 $value, $mzero, $m12, 0 // Retrieve number of columns mod 2 from stack
  brnz $value, merge_partial_value
  bri write_value

merge_partial_value:
  ld32 $value, $mzero, $out_ptr, 0
  sort4x16hi $values2, $values2, $value

write_value:
  st32step $values2, $mzero, $out_ptr+=, 1

load_next_row:
  add $iRow, $iRow, 1
  brnzdec $nRowRemain, row_loop_start

row_loop_end:
  exitz $mzero

FN_SIZE SELECT_HALF

#endif
