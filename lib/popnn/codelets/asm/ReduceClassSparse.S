// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popnn::ReduceMaxClassSparse vertex template
// variations.

// No restrictions

// TODO: T12903 Much of the inner portion of this code is identical to
// ReduceMaxClassGather. There is an opportunity to reuse code here.

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define VERTEX(FP_TYPE, LABEL_TYPE) \
  __runCodelet_popnn__Reduce\MIN_OR_MAX\()ClassSparse___##FP_TYPE##_##LABEL_TYPE

// Constants
#define ACTS_VOFFSET 0
#define LABELS_VOFFSET 2
#define MAXACT_VOFFSET 3
#define MAXINDEX_VOFFSET 4

#define LOG2_SIZEOF_FLOAT 2

// Register aliases
#define ACTS_PTR m0
#define N m1
#define MAX_PTR m2
#define MSCRATCH m10

#define ACT a0
#define MAX a1
#define ASCRATCH a6
//------------------------------------------------------------------------------
.macro INSTANTIATE_FLOAT MIN_OR_MAX COMPARE_OP MIN_OR_MAX_LOWER

DEF_STACK_USAGE 0 .text.VERTEX(float,unsigned_int)
.section .text.VERTEX(float,unsigned_int)
.globl VERTEX(float,unsigned_int)
.type VERTEX(float,unsigned_int), @function

.globl VERTEX(float,int)
.type VERTEX(float,int), @function

.align 8
.worker
  nop
VERTEX(float,unsigned_int):
VERTEX(float,int):
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  ld32 $N, $mvertex_base, $mzero, (ACTS_VOFFSET + 1)

  // Calculate no. of elements, sub 1 for first element loaded
  add $N, $N, -1

  // Load first act, which gives the max
  mov $MAX_PTR, $ACTS_PTR
  ld32step $MAX, $mzero, $MAX_PTR+=, 2
  ld32step $ACT, $mzero, $ACTS_PTR+=, 1

  // Doesn't matter what $ASCRATCH is on the first
  // loop iteration, $MAX_PTR will end up the same value
  rpt $N, (2f-1f)/8-1
1:
  { ld32step $ACT, $mzero, $ACTS_PTR+=, 1
    fnop }
  { atom $MSCRATCH, $ASCRATCH
    f32\COMPARE_OP $ASCRATCH, $ACT, $MAX }
  { movz $MAX_PTR, $MSCRATCH, $ACTS_PTR
    f32\MIN_OR_MAX_LOWER $MAX, $ACT, $MAX }
2:
  ld32step $azero, $mzero, $ACTS_PTR+=, 1
  atom $MSCRATCH, $ASCRATCH
  movz $MAX_PTR, $MSCRATCH, $ACTS_PTR

  // Calculate the index from $MAX_PTR
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  sub $MAX_PTR, $MAX_PTR, $ACTS_PTR
  shr $MAX_PTR, $MAX_PTR, LOG2_SIZEOF_FLOAT
  // $MAX_PTR always ends up 2 elements ahead by the end of the loop
  add $MAX_PTR, $MAX_PTR, -2
  // Load the actual index
  ld32 $MSCRATCH, $mvertex_base, $mzero, LABELS_VOFFSET
  ld32 $MAX_PTR, $MSCRATCH, $mzero, $MAX_PTR

  // Load maxValue/maxIndex pointers and store
  ld32 $MSCRATCH, $mvertex_base, $mzero, MAXACT_VOFFSET
  st32 $MAX, $MSCRATCH, $mzero, 0
  ld32 $MSCRATCH, $mvertex_base, $mzero, MAXINDEX_VOFFSET
  st32 $MAX_PTR, $MSCRATCH, $mzero, 0

  exitz $mzero

// Only set the size for the int version so we don't count it twice.
.size VERTEX(float,int), .-VERTEX(float,int)
.endm
//------------------------------------------------------------------------------
// Use the macros above to instantiate the vertex variants

INSTANTIATE_FLOAT Max cmpgt max
INSTANTIATE_FLOAT Min cmplt min


#endif // __IPU__
