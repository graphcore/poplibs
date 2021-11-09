// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

#define SELECT_INT __runCodelet_popops__Select___int
#define SELECT_UNSIGNED_INT __runCodelet_popops__Select___unsigned_int
#define SELECT_FLOAT __runCodelet_popops__Select___float

// Constants
#define VERTEX_IN1_OFFSET 0
#define VERTEX_IN2_OFFSET 1
#define VERTEX_IN3_OFFSET 2
#define VERTEX_OUT_START_OFFSET 3
#define VERTEX_OUT_END_OFFSET 4

// Integer variables
#define out m0
#define in1 m1
#define in2 m2
#define in3 m3
#define in1_ptr m4
#define in2_ptr m5
#define in3_ptr m6
#define out_ptr m7
#define nRowRemain m8
#define nColRemain m9 // Note: Same register as trueVal
#define value m10
#define ifVal m11
#define trueVal m9 // Note: Same register as nColRemain

FN_WORKER_ENTRY_POINT SELECT_INT 8 nop
FN_EXPORT SELECT_UNSIGNED_INT
FN_EXPORT SELECT_FLOAT
  // Load the vertex state.
  ld32 $in1, $mvertex_base, $mzero, VERTEX_IN1_OFFSET
  ld32 $in2, $mvertex_base, $mzero, VERTEX_IN2_OFFSET
  ld32 $in3, $mvertex_base, $mzero, VERTEX_IN3_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_OUT_START_OFFSET
  ld32 $nRowRemain, $mvertex_base, $mzero, VERTEX_OUT_END_OFFSET
  brz $nRowRemain, row_loop_end
  add $nRowRemain, $nRowRemain, -1

row_loop_start:
  // Load in-rows
  ld32step $in1_ptr, $mzero, $in1+=, 1
  ld32step $in2_ptr, $mzero, $in2+=, 1
  ld32step $in3_ptr, $mzero, $in3+=, 1

  // Load next out pointer and calculate number of inner elements
  ld32step $out_ptr, $mzero, $out+=, 1
  ld32step $nColRemain, $mzero, $out+=, 1

  rpt $nColRemain, ((column_loop_end-column_loop_start)/8)-1
column_loop_start:
  {ld32step $trueVal, $mzero, $in1_ptr+=, 1; fnop}
  {ld32step $value, $mzero, $in2_ptr+=, 1; fnop}
  {ldz8step $ifVal, $mzero, $in3_ptr+=, 1; fnop}
  {movz $value, $ifVal, $trueVal; fnop}
  {st32step $value, $mzero, $out_ptr+=, 1; fnop}

column_loop_end:
  brnzdec $nRowRemain, row_loop_start

row_loop_end:
  exitz $mzero

FN_SIZE SELECT_INT

#endif
