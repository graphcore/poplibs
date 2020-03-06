// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Compact and efficient implementation of the common Binary operations in
// assembler.
// Half, float, inPlace and non-inplace versions
/* -------------------------------------------------------------------------- */

#include "poplibs_support/TileConstants.hpp"

// Registers for each of the passed parameters
// vertex state, all offsets are 8-bit
#define VERTEX_IN1_PTR_OFFSET 0
#define VERTEX_IN2_PTR_OFFSET 4
#define VERTEX_OUT_PTR_OFFSET 8
#define VERTEX_OUT_COUNT_OFFSET 12

#define VERTEX_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_INPLACE_OUT_COUNT_OFFSET 4
#define VERTEX_INPLACE_IN2_PTR_OFFSET 8

// vertex state, for broadcast ops
#define VERTEX_BROADCAST_IN1_PTR_OFFSET 0
#define VERTEX_BROADCAST_IN1_COUNT_OFFSET 4
#define VERTEX_BROADCAST_OUT_PTR_OFFSET 8
#define VERTEX_BROADCAST_IN2_PTR_OFFSET 12

#define VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET 0
#define VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET 4
#define VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET 8

// Register aliases

// integer variables
#define outPtr m0
#define in1Ptr m1
#define in2Ptr m2
#define outCount m3
#define in1Count m3

#define out m6
#define in1 m4
#define in2 m5
// Same as in2 to share fast/slow path decision code
#define outBroadcast m5
#define in12 m4:5
#define outLength m7
#define mloops m8

#define mloopAddress m9
#define mscratch m11


#define MANGLE_STR_COMMON(SUFFIX) __runCodelet_popops__common##SUFFIX
#define MANGLE_STR_FLOAT __runCodelet_popops__\INPLACE_STR\()___popops__expr__BinaryOpType__\NAME_STR\()_float
#define MANGLE_STR_HALF __runCodelet_popops__\INPLACE_STR\()___popops__expr__BinaryOpType__\NAME_STR\()_half

#define MANGLE_STR_BROADCAST_FLOAT __runCodelet_popops__\INPLACE_STR\()___popops__expr__BroadcastOpType__\NAME_STR\()_float
#define MANGLE_STR_BROADCAST_HALF __runCodelet_popops__\INPLACE_STR\()___popops__expr__BroadcastOpType__\NAME_STR\()_half

//******************************************************************************
// Macros for float/binary entry - inplace, non inplace
//******************************************************************************

.macro BINARY_OP_FLOAT_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type MANGLE_STR_FLOAT, @function
.align 4
  // load vertex state
MANGLE_STR_FLOAT:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_IN1_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET/4
  // Decrement as using brnzdec
  sub  $outCount,$outCount, 1
  setzi $mloopAddress,outerLoop_float_non_inplace_\OPERATION
  bri  outerLoop_float_non_inplace_\OPERATION
.size MANGLE_STR_FLOAT, . -MANGLE_STR_FLOAT
.endm

.macro BINARY_OP_FLOAT_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_FLOAT
.type MANGLE_STR_FLOAT, @function
.align 4
  // load vertex state
MANGLE_STR_FLOAT:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_IN2_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_INPLACE_OUT_COUNT_OFFSET/4
  // Decrement as using brnzdec
  sub  $outCount,$outCount, 1
  setzi $mloopAddress,outerLoop_float_inplace_\OPERATION
  bri  outerLoop_float_inplace_\OPERATION
.size MANGLE_STR_FLOAT, . -MANGLE_STR_FLOAT
.endm

//******************************************************************************
// Can we use the faster loop, based on data being in different memory
// elements?
// To help, here's a summary of the memory map,
// based on B0 architecture:
//
// 0x40000 +------------------+
// 0x44000 |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
//         |------------------|
// 0x60000 +------------------+
//         |                  |
// 0x68000 +------------------+
//         |                  |
//         +------------------+
//         |                  |
//         +------------------+
//         |                  |
//         +------------------+
//   0x80000
//
// Memory consists of 2 regions, 0x40000 to 0x60000 and 0x60000 to 0x80000.
// They are subdivided into memory elements, each of which can be accessed once
// per cycle.  The first region consists of 8 elements mapped linearly.
// The second region also has 8 elements, but they are combined into
// pairs as 4 interleaved element pairs. In interleaved memory "odd 64 bit"
// addresses belong to one element, "even 64 bit" belong to another.
// An element is 0x4000 in length, an element pair is 0x8000 in length.
//
// We have 2 non-overlapping arrays to access and have the start address of each
// and the length, which is the same for both.  To decide if we can access the
// whole array with a ld2x64pace instruction, one of the following can be true:
//
// 1. If min(end1, end2) > 0x60000 AND start1, start 2 differ by >0x8000.
// 2. If min(end1, end2) < 0x60000 AND start1, start2 differ by >0x4000
// 3. If max(start1, start2) >=0x60000 AND one start is odd, the other even
// 4. If max(end1, end2) < 0x60000 AND lower array end is in a lower element than the higher array start
// 5. If max(start1, start2) >= 0x60000 AND lower array end is in a lower element pair than the higher array start
// 6. Some side cases where one array is non-interleaved memory, another is not - covering
//    Odd/even addresses or the array ending before the start of interleaved memory
//
// All that logic would slow things down a lot, so being pragmatic we
// just check if the 2 addresses are 32k (1 interleaved element-pair) apart
//******************************************************************************
.macro CHECK_FAST_PATH ADDRESS1 ADDRESS2
  // If greater than 2 * elementSize plus the initial 2 loads before using the
  // ldst instruction apart we can use the fast loop (3x8 bytes to deal with the equals case)
  sub $mscratch, \ADDRESS1, \ADDRESS2
  abs $mscratch, $mscratch

  cmpult $mscratch, $mscratch, (TMEM_ELEMSIZE * 2) + 24

  // Returning 0 = fast loop, 1 = normal loop
.endm

//******************************************************************************
// Macro for float/broadcast loop implementation - inplace, non inplace
//******************************************************************************
.macro BINARY_OP_FLOAT OPERATION
.section .text.MANGLE_STR_COMMON(\@)
.type MANGLE_STR_COMMON(\@), @function
MANGLE_STR_COMMON(\@):
.align 8
 nop
  // Per input vector loop
outerLoop\@:
  br $mloopAddress

outerLoop_float_inplace_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  ld32step $outLength, $mzero, $in1Ptr+=, 1
  // Could save this cycle but that would affect many load/store instructions
  // below - if we need to optimise, use 2 macros
  mov $out, $in1
  bri outerLoop_float_continue\@

outerLoop_float_non_inplace_\OPERATION:
  ld32step $out, $mzero, $outPtr+=, 1
  ld32step $outLength, $mzero, $outPtr+=, 1
  ld32step $in1, $mzero, $in1Ptr+=, 1

outerLoop_float_continue\@:
  ld32step $in2, $mzero, $in2Ptr+=, 1

  CHECK_FAST_PATH $in1 $in2
  // Shift to account for items per loop
  shr $mloops, $outLength, 1
  brz $mscratch, fastLoop\@

defaultLoop\@:
  // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1+=, 1
  ld64step $a2:3, $mzero, $in2+=, 1

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1+=, 1
   f32v2\OPERATION $a2:3, $a0:1, $a2:3}
  {st64step $a2:3, $mzero, $out+=, 1
   fnop}
  {ld64step $a2:3, $mzero, $in2+=, 1
   fnop}
2:
  bri doRemainder\@
  .align 8         // Repeat alignment

fastLoop\@:
  ld2x64pace $a0:1, $a2:3, $in12+=, $mzero, 0b0000
  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld2x64pace $a0:1, $a2:3, $in12+=, $mzero, 0b0000
   f32v2\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $out+=, 1
   fnop}
2:
doRemainder\@:
  // Here we have always overread the 2 inputs by 2 floats, we may need 1 of them
  // if length is odd.  Don't process it if not required though
  and $outLength, $outLength, 1
  brz $outLength, 3f
  f32\OPERATION $a0, $a0, $a2
  st32 $a0, $mzero, $out, 0
3:
  brnzdec $outCount, outerLoop\@

  exitz $mzero
.size MANGLE_STR_COMMON(\@),\
   . -MANGLE_STR_COMMON(\@)
.endm

//******************************************************************************
// Macros for half/binary entry - inplace, non inplace
//******************************************************************************

.macro BINARY_OP_HALF_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_HALF

.global MANGLE_STR_HALF
.type MANGLE_STR_HALF, @function
.align 4
  // load vertex state
MANGLE_STR_HALF:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_IN1_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET/4
  // Decrement as using brnzdec
  sub  $outCount,$outCount, 1
  setzi $mloopAddress,outerLoop_half_non_inplace_\OPERATION
  bri  outerLoop_half_non_inplace_\OPERATION
.size MANGLE_STR_HALF, . -MANGLE_STR_HALF
.endm

.macro BINARY_OP_HALF_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_HALF

.global MANGLE_STR_HALF
.type MANGLE_STR_HALF, @function
.align 4
  // load vertex state
MANGLE_STR_HALF:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_IN2_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_INPLACE_OUT_COUNT_OFFSET/4
  // Decrement as using brnzdec
  sub  $outCount,$outCount, 1
  setzi $mloopAddress,outerLoop_half_inplace_\OPERATION
  bri  outerLoop_half_inplace_\OPERATION
.size MANGLE_STR_HALF, . -MANGLE_STR_HALF
.endm

//******************************************************************************
// Macro for half/binary loop implementation - inplace, non inplace
//******************************************************************************
.macro BINARY_OP_HALF OPERATION
.section .text.MANGLE_STR_COMMON(\@)
.type MANGLE_STR_COMMON(\@), @function
MANGLE_STR_COMMON(\@):
.align 8
 nop
  // Per input vector loop
outerLoop\@:
  br $mloopAddress
  // load vector pointer, size
outerLoop_half_inplace_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  ld32step $outLength, $mzero, $in1Ptr+=, 1
  // Could save this cycle but that would affect many load/store instructions
  // below - if we need to optimise, use 2 macros
  mov $out, $in1
  bri outerLoop_half_continue\@

outerLoop_half_non_inplace_\OPERATION:
  ld32step $out, $mzero, $outPtr+=, 1
  ld32step $outLength, $mzero, $outPtr+=, 1
  ld32step $in1, $mzero, $in1Ptr+=, 1

outerLoop_half_continue\@:
  ld32step $in2, $mzero, $in2Ptr+=, 1

  CHECK_FAST_PATH $in1 $in2
  // Shift to account for items per loop
  shr $mloops, $outLength, 2
  brz $mscratch, fastLoop\@

defaultLoop\@:
  // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1+=, 1
  ld64step $a2:3, $mzero, $in2+=, 1

  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1+=, 1
   f16v4\OPERATION $a2:3, $a0:1, $a2:3}
  {st64step $a2:3, $mzero, $out+=, 1
   fnop}
  {ld64step $a2:3, $mzero, $in2+=, 1
   fnop}
2:
  bri doRemainder\@

  .align 8         // Repeat alignment

fastLoop\@:
  ld2x64pace $a0:1, $a2:3, $in12+=, $mzero, 0b0000
  // Shift to account for items per loop
  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld2x64pace $a0:1, $a2:3, $in12+=, $mzero, 0b0000
   f16v4\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $out+=, 1
   fnop}
2:
doRemainder\@:
  // Here we have always overread the 2 inputs by 4 halves, we may need 1, 2 or 3 of them
  // if length is not a multiple of 4.  Don't process it if not required though
  and $mscratch, $outLength, 2
  brz $mscratch, 4f
  f16v2\OPERATION $a0, $a0, $a2
  // Store result, transfer operands so the same code to deal with 1 will work
  {st32step $a0, $mzero, $out+=, 1
   mov $a0, $a1}
  mov $a2, $a3
4:
  // Process a last one - zero out the unused half of both operands
  {and $mscratch, $outLength, 1
   sort4x16lo $a2, $a2, $azero}
  {brz $mscratch, 5f
   sort4x16lo $a0, $a0, $azero}
  {ldb16 $a1, $mzero, $out, 1
   f16v2\OPERATION $a0, $a0, $a2}
  sort4x16lo $a0, $a0, $a1
  st32 $a0, $mzero, $out, 0

5:
  brnzdec $outCount, outerLoop\@

  exitz $mzero
.size MANGLE_STR_COMMON(\@),\
   . -MANGLE_STR_COMMON(\@)
.endm
//******************************************************************************
// Macro for float/broadcast entry - inplace, non inplace
//******************************************************************************

.macro BROADCAST_OP_FLOAT_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_BROADCAST_FLOAT
.type MANGLE_STR_BROADCAST_FLOAT, @function

.align 4
  // load vertex state
MANGLE_STR_BROADCAST_FLOAT:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4
  ld32 $a2, $mzero, $in2Ptr, 0
  // Decrement as using brnzdec
  {sub  $in1Count,$in1Count, 1
   mov  $a3, $a2}
  setzi $mloopAddress,outerLoop_float_non_inplace_broadcast_\OPERATION
  bri  outerLoop_float_non_inplace_broadcast_\OPERATION
.size MANGLE_STR_BROADCAST_FLOAT, . -MANGLE_STR_BROADCAST_FLOAT
.endm


.macro BROADCAST_OP_FLOAT_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_FLOAT
.global MANGLE_STR_BROADCAST_FLOAT
.type MANGLE_STR_BROADCAST_FLOAT, @function

.align 4
  // load vertex state
MANGLE_STR_BROADCAST_FLOAT:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET/4
  ld32 $a2, $mzero, $in2Ptr, 0
  // Decrement as using brnzdec
  {sub  $in1Count,$in1Count, 1
   mov  $a3, $a2}
  setzi $mloopAddress,outerLoop_float_inplace_broadcast_\OPERATION
  bri  outerLoop_float_inplace_broadcast_\OPERATION
.size MANGLE_STR_BROADCAST_FLOAT, . -MANGLE_STR_BROADCAST_FLOAT
.endm


//******************************************************************************
// Macro for float/broadcast loop implementation - inplace, non inplace
//******************************************************************************
.macro BROADCAST_OP_FLOAT OPERATION
.section .text.MANGLE_STR_COMMON(\@)
.type MANGLE_STR_COMMON(\@), @function
MANGLE_STR_COMMON(\@):
.align 8
  // Per input vector loop
outerLoop\@:
  br $mloopAddress
  // load vector pointer, size
outerLoop_float_inplace_broadcast_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  mov $outBroadcast, $in1
  ld32step $outLength, $mzero, $in1Ptr+=, 1
  // Shift to account for items per loop
  shr $mloops, $outLength, 1
  // No fast path implemented for inplace broadcast operations
  bri defaultLoop\@

outerLoop_float_non_inplace_broadcast_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  ld32step $outBroadcast, $mzero, $outPtr+=, 1
  ld32step $outLength, $mzero, $in1Ptr+=, 1

  CHECK_FAST_PATH $in1 $in2
  // Shift to account for items per loop
  shr $mloops, $outLength, 1
  brz $mscratch, fastLoop\@

defaultLoop\@:
  // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1+=, 1
  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1+=, 1
   f32v2\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outBroadcast+=, 1
   fnop}
2:
  bri doRemainder\@

  .align 8
  nop //rpt alignment
fastLoop\@:
  add $mscratch, $mloops, -1
  brneg $mscratch, defaultLoop\@

  ld64step $a0:1, $mzero, $in1+=, 1
  {ld64step $a0:1, $mzero, $in1+=, 1
   f32v2\OPERATION $a4:5, $a0:1, $a2:3}
  tapack $in12, $in1, $mzero, $outBroadcast
  rpt $mscratch, (2f - 1f ) /8 - 1
1:
  {ldst64pace $a0:1, $a4:5, $in12+=, $mscratch, 0b0000
   f32v2\OPERATION $a4:5, $a0:1, $a2:3}
2:
  // Re-create the output address (we have overread 64 bits already) and store
  shr $mscratch, $in1, TMEM_BYTE_MAX_ADDRESS_WIDTH
  shr $in2, $in2, TMEM_BYTE_MAX_ADDRESS_WIDTH
  shl $in2, $in2, (32 - TMEM_BYTE_MAX_ADDRESS_WIDTH)
  or  $outBroadcast, $in2, $mscratch
  st64step $a4:5, $mzero, $outBroadcast+=,1
doRemainder\@:
  // Here we have always overread the input by one float pair, we may need 1 of them
  // if length is odd.  Don't process it if not required though
  and $outLength, $outLength, 1
  brz $outLength, 3f
  f32\OPERATION $a0, $a0, $a2
  st32 $a0, $mzero, $outBroadcast, 0
3:
  brnzdec $in1Count, outerLoop\@

  exitz $mzero
.size MANGLE_STR_COMMON(\@), . -MANGLE_STR_COMMON(\@)
.endm


//******************************************************************************
// Macro for half/broadcast entry - inplace, non inplace
//******************************************************************************

.macro BROADCAST_OP_HALF_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_BROADCAST_HALF
.global MANGLE_STR_BROADCAST_HALF
.type MANGLE_STR_BROADCAST_HALF, @function
.align 4
  // load vertex state
MANGLE_STR_BROADCAST_HALF:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4
  ldb16 $a2, $mzero, $in2Ptr, 0
  // Decrement as using brnzdec
  {sub  $in1Count,$in1Count, 1
   mov  $a3, $a2}
  setzi $mloopAddress,outerLoop_half_non_inplace_broadcast_\OPERATION
  bri  outerLoop_half_non_inplace_broadcast_\OPERATION

.size MANGLE_STR_BROADCAST_HALF, . -MANGLE_STR_BROADCAST_HALF
.endm

.macro BROADCAST_OP_HALF_IN_PLACE_ENTRY OPERATION INPLACE_STR NAME_STR
.section .text.MANGLE_STR_BROADCAST_HALF
.global MANGLE_STR_BROADCAST_HALF
.type MANGLE_STR_BROADCAST_HALF, @function
.align 4
  // load vertex state
MANGLE_STR_BROADCAST_HALF:
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET/4
  ldb16 $a2, $mzero, $in2Ptr, 0
  // Decrement as using brnzdec
  {sub  $in1Count,$in1Count, 1
   mov  $a3, $a2}
  setzi $mloopAddress,outerLoop_half_inplace_broadcast_\OPERATION
  bri  outerLoop_half_inplace_broadcast_\OPERATION

.size MANGLE_STR_BROADCAST_HALF, . -MANGLE_STR_BROADCAST_HALF
.endm

//******************************************************************************
// Macro for half/broadcast loop implementation - inplace, non inplace
//******************************************************************************
.macro BROADCAST_OP_HALF OPERATION
.section .text.MANGLE_STR_COMMON(\@)
.type MANGLE_STR_COMMON(\@), @function
MANGLE_STR_COMMON(\@):
.align 8
  // Per input vector loop
outerLoop\@:
  br $mloopAddress
  // load vector pointer, size
outerLoop_half_inplace_broadcast_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  mov $outBroadcast, $in1
  ld32step $outLength, $mzero, $in1Ptr+=, 1
  // Shift to account for items per loop
  shr $mloops, $outLength, 2
  // No fast path implemented for inplace broadcast operations
  bri defaultLoop\@

outerLoop_half_non_inplace_broadcast_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  ld32step $outBroadcast, $mzero, $outPtr+=, 1

  ld32step $outLength, $mzero, $in1Ptr+=, 1

  CHECK_FAST_PATH $in1 $in2
  // Shift to account for items per loop
  shr $mloops, $outLength, 2
  brz $mscratch, fastLoop\@

defaultLoop\@:
  // Pre load so we can pipeline the loop
  ld64step $a0:1, $mzero, $in1+=, 1
  rpt $mloops, (2f - 1f ) /8 - 1
1:
  {ld64step $a0:1, $mzero, $in1+=, 1
   f16v4\OPERATION $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $outBroadcast+=, 1
   fnop}
2:
  bri doRemainder\@
  .align 8

  nop //rpt alignment
fastLoop\@:
  add $mscratch, $mloops, -1
  brneg $mscratch, defaultLoop\@

  ld64step $a0:1, $mzero, $in1+=, 1
  {ld64step $a0:1, $mzero, $in1+=, 1
   f16v4\OPERATION $a4:5, $a0:1, $a2:3}
  tapack $in12, $in1, $mzero, $outBroadcast
  rpt $mscratch, (2f - 1f ) /8 - 1
1:
  {ldst64pace $a0:1, $a4:5, $in12+=, $mscratch, 0b0000
   f16v4\OPERATION $a4:5, $a0:1, $a2:3}
2:
  // Re-create the output address (we have overread 64 bits already) and store
  shr $mscratch, $in1, TMEM_BYTE_MAX_ADDRESS_WIDTH
  shr $in2, $in2, TMEM_BYTE_MAX_ADDRESS_WIDTH
  shl $in2, $in2, (32 - TMEM_BYTE_MAX_ADDRESS_WIDTH)
  or  $outBroadcast, $in2, $mscratch
  st64step $a4:5, $mzero, $outBroadcast+=,1

doRemainder\@:
  // Here we have always overread the input by 4 halves, we may need 1, 2 or 3 of them
  // if length is not a multiple of 4.  Don't process it if not required though
  and $mscratch, $outLength, 2
  brz $mscratch, 4f
  f16v2\OPERATION $a0, $a0, $a2
  // Store result, transfer operands so the same code to deal with 1 will work
  {st32step $a0, $mzero, $outBroadcast+=, 1
   mov $a0, $a1}
4:
  and $mscratch, $outLength, 1
  // Process a last 1.  $a2 was broadcast so doesn't require masking, but
  // we need to zero out the unused half of $a0
  {brz $mscratch, 5f
   sort4x16lo $a0, $a0, $azero}
  {ldb16 $a1, $mzero, $outBroadcast, 1
   f16v2\OPERATION $a0, $a0, $a2}
  sort4x16lo $a0, $a0, $a1
  st32 $a0, $mzero, $outBroadcast, 0

5:
  brnzdec $in1Count, outerLoop\@

  exitz $mzero
.size MANGLE_STR_COMMON(\@), . -MANGLE_STR_COMMON(\@)
.endm


//******************************************************************************
// Use the macros to create inplace and non inplace, float and half versions
// for each Binary op
//******************************************************************************

  BINARY_OP_HALF_IN_PLACE_ENTRY add BinaryOp2DInPlace ADD
  BINARY_OP_HALF_IN_PLACE_ENTRY sub BinaryOp2DInPlace SUBTRACT
  BINARY_OP_HALF_IN_PLACE_ENTRY mul BinaryOp2DInPlace MULTIPLY
  BINARY_OP_HALF_ENTRY add BinaryOp2D ADD
  BINARY_OP_HALF_ENTRY sub BinaryOp2D SUBTRACT
  BINARY_OP_HALF_ENTRY mul BinaryOp2D MULTIPLY

  BINARY_OP_FLOAT_IN_PLACE_ENTRY add BinaryOp2DInPlace ADD
  BINARY_OP_FLOAT_IN_PLACE_ENTRY sub BinaryOp2DInPlace SUBTRACT
  BINARY_OP_FLOAT_IN_PLACE_ENTRY mul BinaryOp2DInPlace MULTIPLY
  BINARY_OP_FLOAT_ENTRY add BinaryOp2D ADD
  BINARY_OP_FLOAT_ENTRY sub BinaryOp2D SUBTRACT
  BINARY_OP_FLOAT_ENTRY mul BinaryOp2D MULTIPLY

  BINARY_OP_HALF add
  BINARY_OP_HALF sub
  BINARY_OP_HALF mul

  BINARY_OP_FLOAT add
  BINARY_OP_FLOAT sub
  BINARY_OP_FLOAT mul

//******************************************************************************
// Use the macros to create inplace and non inplace, float and half entry points
// for each Broadcast op
//******************************************************************************

  BROADCAST_OP_HALF_IN_PLACE_ENTRY add BroadcastScalar2DDataInPlace ADD
  BROADCAST_OP_HALF_IN_PLACE_ENTRY sub BroadcastScalar2DDataInPlace SUBTRACT
  BROADCAST_OP_HALF_IN_PLACE_ENTRY mul BroadcastScalar2DDataInPlace MULTIPLY
  BROADCAST_OP_HALF_ENTRY add BroadcastScalar2DData ADD
  BROADCAST_OP_HALF_ENTRY sub BroadcastScalar2DData SUBTRACT
  BROADCAST_OP_HALF_ENTRY mul BroadcastScalar2DData MULTIPLY

  BROADCAST_OP_FLOAT_IN_PLACE_ENTRY add BroadcastScalar2DDataInPlace ADD
  BROADCAST_OP_FLOAT_IN_PLACE_ENTRY sub BroadcastScalar2DDataInPlace SUBTRACT
  BROADCAST_OP_FLOAT_IN_PLACE_ENTRY mul BroadcastScalar2DDataInPlace MULTIPLY
  BROADCAST_OP_FLOAT_ENTRY add BroadcastScalar2DData ADD
  BROADCAST_OP_FLOAT_ENTRY sub BroadcastScalar2DData SUBTRACT
  BROADCAST_OP_FLOAT_ENTRY mul BroadcastScalar2DData MULTIPLY

  BROADCAST_OP_FLOAT add
  BROADCAST_OP_FLOAT sub
  BROADCAST_OP_FLOAT mul

  BROADCAST_OP_HALF add
  BROADCAST_OP_HALF sub
  BROADCAST_OP_HALF mul

#endif
/* -------------------------------------------------------------------------- */
