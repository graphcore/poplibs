// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Compact and efficient implementation of the common Binary operations in
// assembler.
// Add, Sub, mul Half, float, inPlace and non-inplace versions
// And, Or short, unsigned short versions
/* -------------------------------------------------------------------------- */

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

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
#define MANGLE_STR __runCodelet_popops__\VERTEX_NAME\()___popops__expr__BinaryOpType__\OPERATION\()_\TYPE

//******************************************************************************
// Macros for binaryOp entry (all types) - inplace, non inplace
//******************************************************************************

.macro BINARY_OP_ENTRY TYPE VERTEX_NAME OPERATION LOOP_VARIANT

FN_WORKER_ENTRY_POINT MANGLE_STR
  // load vertex state
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_IN1_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET/4
  setzi $mloopAddress,outerLoop_\LOOP_VARIANT\()_non_inplace_\OPERATION
  // Decrement as using brnzdec
  brnzdec $outCount,  outerLoop_\LOOP_VARIANT\()_non_inplace_\OPERATION
  exitz $mzero
FN_SIZE MANGLE_STR
.endm

.macro BINARY_OP_IN_PLACE_ENTRY TYPE VERTEX_NAME OPERATION LOOP_VARIANT

FN_WORKER_ENTRY_POINT MANGLE_STR
  // load vertex state
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_INPLACE_IN2_PTR_OFFSET/4
  ld32 $outCount, $mzero, $mvertex_base, VERTEX_INPLACE_OUT_COUNT_OFFSET/4
  setzi $mloopAddress,outerLoop_\LOOP_VARIANT\()_inplace_\OPERATION
  // Decrement as using brnzdec
  brnzdec $outCount, outerLoop_\LOOP_VARIANT\()_inplace_\OPERATION
  exitz $mzero
FN_SIZE MANGLE_STR
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
.macro BINARY_OP_float OPERATION INSTRUCTIONv2 INSTRUCTIONv1

FN_SECTION MANGLE_STR_COMMON(\@) 8 nop

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
   \INSTRUCTIONv2 $a2:3, $a0:1, $a2:3}
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
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $out+=, 1
   fnop}
2:
doRemainder\@:
  // Here we have always overread the 2 inputs by 2 floats, we may need 1 of them
  // if length is odd.  Don't process it if not required though
  and $outLength, $outLength, 1
  brz $outLength, 3f
  \INSTRUCTIONv1 $a0, $a0, $a2
  st32 $a0, $mzero, $out, 0
3:
  brnzdec $outCount, outerLoop\@

  exitz $mzero
FN_SIZE MANGLE_STR_COMMON(\@)
.endm

//******************************************************************************
// Macro for 4x16bit data vector binaryOp loop implementation - inplace, non inplace
//******************************************************************************
.macro BINARY_OP_4x16bit OPERATION INSTRUCTIONv4 INSTRUCTIONv2="void"
FN_SECTION MANGLE_STR_COMMON(\@) 8 nop
  // Per input vector loop
outerLoop\@:
  br $mloopAddress
  // load vector pointer, size
outerLoop_4x16bit_inplace_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  ld32step $outLength, $mzero, $in1Ptr+=, 1
  // Could save this cycle but that would affect many load/store instructions
  // below - if we need to optimise, use 2 macros
  mov $out, $in1
  bri outerLoop_4x16bit_continue\@

outerLoop_4x16bit_non_inplace_\OPERATION:
  ld32step $out, $mzero, $outPtr+=, 1
  ld32step $outLength, $mzero, $outPtr+=, 1
  ld32step $in1, $mzero, $in1Ptr+=, 1

outerLoop_4x16bit_continue\@:
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
   \INSTRUCTIONv4 $a2:3, $a0:1, $a2:3}
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
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $mzero, $out+=, 1
   fnop}
2:
doRemainder\@:
  // Here we have always overread the 2 inputs by 4 halves, we may need 1, 2 or 3 of them
  // if length is not a multiple of 4.  Don't process it if not required though
  and $mscratch, $outLength, 2
  brz $mscratch, 4f
.ifc "\INSTRUCTIONv2","void"
  // Some instructions have no v2 equivalent.  At the moment this is only used for
  // bitwise ops which don't cause exceptions, but beware of the overprocessing!
  \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3
.else
  \INSTRUCTIONv2 $a4, $a0, $a2
.endif
  // Store result, transfer operands so the same code to deal with 1 will work
  {st32step $a4, $mzero, $out+=, 1
   mov $a0, $a1}
  mov $a2, $a3
4:
  // Process a last one - zero out the unused half of both operands
  {and $mscratch, $outLength, 1
   sort4x16lo $a2, $a2, $azero}
  {brz $mscratch, 5f
   sort4x16lo $a0, $a0, $azero}
.ifc "\INSTRUCTIONv2","void"
  // Some instructions have no v2 equivalent.  At the moment this is only used for
  // bitwise ops which don't cause exceptions, but beware of the overprocessing!
  {ldb16 $a1, $mzero, $out, 1
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
.else
  {ldb16 $a1, $mzero, $out, 1
   \INSTRUCTIONv2 $a4, $a0, $a2}
.endif
  sort4x16lo $a0, $a4, $a1
  st32 $a0, $mzero, $out, 0

5:
  brnzdec $outCount, outerLoop\@

  exitz $mzero
FN_SIZE MANGLE_STR_COMMON(\@)
.endm
//******************************************************************************
// Macro for broadcastOp entry (All types) - inplace, non inplace
//******************************************************************************

.macro BROADCAST_OP_ENTRY TYPE VERTEX_NAME OPERATION LOOP_VARIANT

FN_WORKER_ENTRY_POINT MANGLE_STR
  // load vertex state
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_IN1_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_IN2_PTR_OFFSET/4
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_BROADCAST_OUT_PTR_OFFSET/4
.ifc "\TYPE", "float"
  ld32 $a2, $mzero, $in2Ptr, 0
.else
  ldb16 $a2, $mzero, $in2Ptr, 0
.endif
  {setzi $mloopAddress,outerLoop_\LOOP_VARIANT\()_non_inplace_broadcast_\OPERATION
   mov  $a3, $a2}
  // Decrement as using brnzdec
  brnzdec $in1Count, outerLoop_\LOOP_VARIANT\()_non_inplace_broadcast_\OPERATION
  exitz $mzero
FN_SIZE MANGLE_STR
.endm

.macro BROADCAST_OP_IN_PLACE_ENTRY TYPE VERTEX_NAME OPERATION LOOP_VARIANT
FN_WORKER_ENTRY_POINT MANGLE_STR
  ld32 $in1Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_PTR_OFFSET/4
  ld32 $in1Count, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_INOUT_COUNT_OFFSET/4
  ld32 $in2Ptr, $mzero, $mvertex_base, VERTEX_BROADCAST_INPLACE_IN2_PTR_OFFSET/4
.ifc "\TYPE", "float"
  ld32 $a2, $mzero, $in2Ptr, 0
.else
  ldb16 $a2, $mzero, $in2Ptr, 0
.endif
  {setzi $mloopAddress,outerLoop_\LOOP_VARIANT\()_inplace_broadcast_\OPERATION
   mov  $a3, $a2}
  // Decrement as using brnzdec
  brnzdec $in1Count, outerLoop_\LOOP_VARIANT\()_inplace_broadcast_\OPERATION
  exitz $mzero
FN_SIZE MANGLE_STR
.endm


//******************************************************************************
// Macro for float/broadcast loop implementation - inplace, non inplace
//******************************************************************************
.macro BROADCAST_OP_float OPERATION INSTRUCTIONv2 INSTRUCTIONv1
FN_SECTION MANGLE_STR_COMMON(\@) 8
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
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
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
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
  tapack $in12, $in1, $mzero, $outBroadcast
  rpt $mscratch, (2f - 1f ) /8 - 1
1:
  {ldst64pace $a0:1, $a4:5, $in12+=, $mscratch, 0b0000
   \INSTRUCTIONv2 $a4:5, $a0:1, $a2:3}
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
  \INSTRUCTIONv1 $a0, $a0, $a2
  st32 $a0, $mzero, $outBroadcast, 0
3:
  brnzdec $in1Count, outerLoop\@

  exitz $mzero
FN_SIZE MANGLE_STR_COMMON(\@)
.endm

//******************************************************************************
// Macro for half,short, ushort/broadcast loop implementation - inplace, non inplace
//******************************************************************************
.macro BROADCAST_OP_4x16bit OPERATION INSTRUCTIONv4 INSTRUCTIONv2="void"
FN_WORKER_ENTRY_POINT MANGLE_STR_COMMON(\@) 8
  // Per input vector loop
outerLoop\@:
  br $mloopAddress
  // load vector pointer, size
outerLoop_4x16bit_inplace_broadcast_\OPERATION:
  ld32step $in1, $mzero, $in1Ptr+=, 1
  mov $outBroadcast, $in1
  ld32step $outLength, $mzero, $in1Ptr+=, 1
  // Shift to account for items per loop
  shr $mloops, $outLength, 2
  // No fast path implemented for inplace broadcast operations
  bri defaultLoop\@

outerLoop_4x16bit_non_inplace_broadcast_\OPERATION:
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
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
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
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
  tapack $in12, $in1, $mzero, $outBroadcast
  rpt $mscratch, (2f - 1f ) /8 - 1
1:
  {ldst64pace $a0:1, $a4:5, $in12+=, $mscratch, 0b0000
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
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
.ifc "\INSTRUCTIONv2", "void"
  // Some instructions have no v2 equivalent.  At the moment this is only used for
  // bitwise ops which don't cause exceptions, but beware of the overprocessing!
  \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3
.else
  \INSTRUCTIONv2 $a4, $a0, $a2
.endif
  // Store result, transfer operands so the same code to deal with 1 will work
  {st32step $a4, $mzero, $outBroadcast+=, 1
   mov $a0, $a1}
4:
  and $mscratch, $outLength, 1
  // Process a last 1.  $a2 was broadcast so doesn't require masking, but
  // we need to zero out the unused half of $a0
  {brz $mscratch, 5f
   sort4x16lo $a0, $a0, $azero}
.ifc "\INSTRUCTIONv2", "void"
  // Some instructions have no v2 equivalent.  At the moment this is only used for
  // bitwise ops which don't cause exceptions, but beware of the overprocessing!
  {ldb16 $a1, $mzero, $outBroadcast, 1
   \INSTRUCTIONv4 $a4:5, $a0:1, $a2:3}
.else
  {ldb16 $a1, $mzero, $outBroadcast, 1
   \INSTRUCTIONv2 $a4, $a0, $a2}
.endif
  sort4x16lo $a0, $a4, $a1
  st32 $a0, $mzero, $outBroadcast, 0

5:
  brnzdec $in1Count, outerLoop\@

  exitz $mzero
FN_SIZE MANGLE_STR_COMMON(\@)
.endm


//******************************************************************************
// Use the macros to create inplace and non inplace, float and half versions
// for each Binary op
//******************************************************************************

  BINARY_OP_IN_PLACE_ENTRY half BinaryOp2DInPlace ADD 4x16bit
  BINARY_OP_IN_PLACE_ENTRY half BinaryOp2DInPlace SUBTRACT 4x16bit
  BINARY_OP_IN_PLACE_ENTRY half BinaryOp2DInPlace MULTIPLY 4x16bit
  BINARY_OP_ENTRY half BinaryOp2D ADD 4x16bit
  BINARY_OP_ENTRY half BinaryOp2D SUBTRACT 4x16bit
  BINARY_OP_ENTRY half BinaryOp2D MULTIPLY 4x16bit

  BINARY_OP_IN_PLACE_ENTRY float BinaryOp2DInPlace ADD float
  BINARY_OP_IN_PLACE_ENTRY float BinaryOp2DInPlace SUBTRACT float
  BINARY_OP_IN_PLACE_ENTRY float BinaryOp2DInPlace MULTIPLY float
  BINARY_OP_ENTRY float BinaryOp2D ADD float
  BINARY_OP_ENTRY float BinaryOp2D SUBTRACT float
  BINARY_OP_ENTRY float BinaryOp2D MULTIPLY float

  BINARY_OP_float ADD f32v2add f32add
  BINARY_OP_float SUBTRACT f32v2sub f32sub
  BINARY_OP_float MULTIPLY f32v2mul f32mul

  BINARY_OP_4x16bit ADD f16v4add f16v2add
  BINARY_OP_4x16bit SUBTRACT f16v4sub f16v2sub
  BINARY_OP_4x16bit MULTIPLY f16v4mul f16v2mul

//******************************************************************************
// Use the macros to create inplace and non inplace, short and unsigned short versions
// for each Binary op
//******************************************************************************
  BINARY_OP_IN_PLACE_ENTRY short BinaryOp2DInPlace BITWISE___AND 4x16bit
  BINARY_OP_IN_PLACE_ENTRY short BinaryOp2DInPlace BITWISE___OR 4x16bit
  BINARY_OP_ENTRY short BinaryOp2D BITWISE___AND 4x16bit
  BINARY_OP_ENTRY short BinaryOp2D BITWISE___OR 4x16bit

  BINARY_OP_IN_PLACE_ENTRY unsigned_short BinaryOp2DInPlace BITWISE___AND 4x16bit
  BINARY_OP_IN_PLACE_ENTRY unsigned_short BinaryOp2DInPlace BITWISE___OR 4x16bit
  BINARY_OP_ENTRY unsigned_short BinaryOp2D BITWISE___AND 4x16bit
  BINARY_OP_ENTRY unsigned_short BinaryOp2D BITWISE___OR 4x16bit

  BINARY_OP_4x16bit BITWISE___AND and64
  BINARY_OP_4x16bit BITWISE___OR or64

//******************************************************************************
// Use the macros to create inplace and non inplace, float and half entry points
// for each Broadcast op
//******************************************************************************

  BROADCAST_OP_IN_PLACE_ENTRY half BroadcastScalar2DDataInPlace ADD 4x16bit
  BROADCAST_OP_IN_PLACE_ENTRY half BroadcastScalar2DDataInPlace SUBTRACT 4x16bit
  BROADCAST_OP_IN_PLACE_ENTRY half BroadcastScalar2DDataInPlace MULTIPLY 4x16bit
  BROADCAST_OP_ENTRY half BroadcastScalar2DData ADD 4x16bit
  BROADCAST_OP_ENTRY half BroadcastScalar2DData SUBTRACT 4x16bit
  BROADCAST_OP_ENTRY half BroadcastScalar2DData MULTIPLY 4x16bit

  BROADCAST_OP_IN_PLACE_ENTRY float BroadcastScalar2DDataInPlace ADD float
  BROADCAST_OP_IN_PLACE_ENTRY float BroadcastScalar2DDataInPlace SUBTRACT float
  BROADCAST_OP_IN_PLACE_ENTRY float BroadcastScalar2DDataInPlace MULTIPLY float
  BROADCAST_OP_ENTRY float BroadcastScalar2DData ADD float
  BROADCAST_OP_ENTRY float BroadcastScalar2DData SUBTRACT float
  BROADCAST_OP_ENTRY float BroadcastScalar2DData MULTIPLY float

  BROADCAST_OP_float ADD f32v2add f32add
  BROADCAST_OP_float SUBTRACT f32v2sub f32sub
  BROADCAST_OP_float MULTIPLY f32v2mul f32mul

  BROADCAST_OP_4x16bit ADD f16v4add f16v2add
  BROADCAST_OP_4x16bit SUBTRACT f16v4sub f16v2sub
  BROADCAST_OP_4x16bit MULTIPLY f16v4mul f16v2mul

//******************************************************************************
// Use the macros to create inplace and non inplace, short and unsigned short versions
// for each Broadcast op
//******************************************************************************
  BROADCAST_OP_IN_PLACE_ENTRY short BroadcastScalar2DDataInPlace BITWISE___AND 4x16bit
  BROADCAST_OP_IN_PLACE_ENTRY short BroadcastScalar2DDataInPlace BITWISE___OR 4x16bit
  BROADCAST_OP_ENTRY short BroadcastScalar2DData BITWISE___AND 4x16bit
  BROADCAST_OP_ENTRY short BroadcastScalar2DData BITWISE___OR 4x16bit

  BROADCAST_OP_IN_PLACE_ENTRY unsigned_short BroadcastScalar2DDataInPlace BITWISE___AND 4x16bit
  BROADCAST_OP_IN_PLACE_ENTRY unsigned_short BroadcastScalar2DDataInPlace BITWISE___OR 4x16bit
  BROADCAST_OP_ENTRY unsigned_short BroadcastScalar2DData BITWISE___AND 4x16bit
  BROADCAST_OP_ENTRY unsigned_short BroadcastScalar2DData BITWISE___OR 4x16bit

  BROADCAST_OP_4x16bit BITWISE___AND and64
  BROADCAST_OP_4x16bit BITWISE___OR or64

#endif
/* -------------------------------------------------------------------------- */
