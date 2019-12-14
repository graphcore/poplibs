// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__

#define VERTEX(ty) __runCodelet_popops__ScaledAdd2D___ ## ty
#define VERTEX_SUBTRACT(ty) __runCodelet_popops__ScaledSubtract2D___ ## ty
#define VERTEX_COMMON __ScaledAdd2D___
.globl VERTEX(int_int_int_true_false)
.type VERTEX(int_int_int_true_false), @function

.globl VERTEX(unsigned_int_unsigned_int_unsigned_int_true_false)
.type VERTEX(unsigned_int_unsigned_int_unsigned_int_true_false), @function


.globl VERTEX(int_int_int_false_false)
.type VERTEX(int_int_int_false_false), @function

.globl VERTEX(unsigned_int_unsigned_int_unsigned_int_false_false)
.type VERTEX(unsigned_int_unsigned_int_unsigned_int_false_false), @function

.globl VERTEX_SUBTRACT(int_false)
.type VERTEX_SUBTRACT(int_false), @function

.globl VERTEX_SUBTRACT(unsigned_int_false)
.type VERTEX_SUBTRACT(unsigned_int_false), @function

// constants
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 1
#define VERTEX_DATA_B_OFFSET 2
// 2 versions - one with scale as a constant, the other using a tensor scale
#define VERTEX_SCALE_B_OFFSET 3

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define k m3
#define dataPtr m4
#define dataSize m5
#define dataBPtr m6
#define data m7
#define dataB m8

.section .text.VERTEX(int_true)
.align 4

VERTEX(int_int_int_true_false):
VERTEX(unsigned_int_unsigned_int_unsigned_int_true_false):
  // load vertex state specific to the case where we multiply by a constant, k
  ld32 $k, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  bri  VERTEX_COMMON
.size VERTEX(int_true), .-VERTEX(int_int_int_true_false)

.align 4

VERTEX_SUBTRACT(int_false):
VERTEX_SUBTRACT(unsigned_int_false):
  // load vertex state specific to the case where we multiply by a tensor, factor
  ld32 $k, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  ld32 $k, $mzero, $k, 0
  mul  $k, $k, -1
  bri  VERTEX_COMMON
.size VERTEX_SUBTRACT(int), .-VERTEX_SUBTRACT(int_false)


.section .text.VERTEX_COMMON
.align 8
VERTEX(int_int_int_false_false):
VERTEX(unsigned_int_unsigned_int_unsigned_int_false_false):
  // load vertex state specific to the case where we multiply by a tensor, factor
  ld32 $k, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  ld32 $k, $mzero, $k, 0

VERTEX_COMMON:
  // load common vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
  ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET

  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1
.Louter_loop:
  // load inner pointers
  ld32step $dataPtr, $mzero, $outData+=, 1
  ld32step $dataSize, $mzero, $outData+=, 1
  ld32step $dataBPtr, $mzero, $outDataB+=, 1

  rpt $dataSize, (.Linner_loop_end - .Linner_loop_begin)/8-1
.Linner_loop_begin:
  {
    ld32 $data, $mzero, $dataPtr, 0
    fnop
  }
  {
    ld32step $dataB, $mzero, $dataBPtr+=, 1
    fnop
  }
  {
    mul $dataB, $dataB, $k
    fnop
  }
  {
    add $data, $data, $dataB
    fnop
  }
  {
    st32step $data, $mzero, $dataPtr+=, 1
    fnop
  }
.Linner_loop_end:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

.size VERTEX_COMMON, .-VERTEX(int_int_int_false_false)

#endif // __IPU__
