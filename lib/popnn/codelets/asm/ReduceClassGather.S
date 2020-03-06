// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#ifdef __IPU__

// Assembly implementation of popnn::ReduceMaxClassGather vertex template
// variations.

// No restrictions

// TODO: T12903 Much of the inner portion of this code is identical to
// ReduceMaxClassSparse. There is an opportunity to reuse code here.

#include "poplibs_support/TileConstants.hpp"

#define VERTEX(fpType,labelType) \
  __runCodelet_popnn__Reduce@MIN_OR_MAX@ClassGather___ ## fpType ## _ ## labelType

// Constants
#define ACTS_VOFFSET 0
#define INDEX_VOFFSET 1
#define MAXACT_VOFFSET 2
#define MAXINDEX_VOFFSET 3
#define SIZE_VOFFSET 4
#define WSIZE_VOFFSET 5

#define LOG2_SIZEOF_FLOAT 2
#define LOG2_SIZEOF_HALF 1

// Supervisor register aliases
#define SUPER_BASE m0
#define WORKER_ENTRY m1

// Worker register aliases
#define WORKER_ID m0
#define ACTS_PTR m1
#define SIZE m2
#define WSIZE m3
#define OFFSET m4
#define END m5
#define N m6
#define MAX_PTR m7
#define MSCRATCH m11

#define ACT a0
#define MAX a1
#define ASCRATCH a6

.section .text.VERTEX(float, unsigned_int)
.globl VERTEX(float, unsigned_int)
.type VERTEX(float, unsigned_int), @function

.globl VERTEX(float, int)
.type VERTEX(float, int), @function

.align 8
.supervisor
  nop
VERTEX(float, unsigned_int):
VERTEX(float, int):
  setzi $WORKER_ENTRY, .Lfloat_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.Lfloat_worker:
.worker
  // Load starting vertex state
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  ld32 $SIZE, $mvertex_base, $mzero, SIZE_VOFFSET
  ld32 $WSIZE, $mvertex_base, $mzero, WSIZE_VOFFSET

  // Get worker ID
  get $WORKER_ID, $WSR
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Compute worker region
  mul $OFFSET, $WORKER_ID, $WSIZE
  add $END, $WORKER_ID, 1
  mul $END, $END, $WSIZE
  min $END, $END, $SIZE

  // Calculate number of elements, sub 1 for first element loaded below
  sub $N, $END, $OFFSET
  add $N, $N, -1

  // If there is no actual work for this worker, early exit
  brneg $N, .Lfloat_end

  // Offset pointer
  ld32step $azero, $mzero, $ACTS_PTR+=, $OFFSET

  // Initialise max
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
    f32@COMPARE_OP@ $ASCRATCH, $ACT, $MAX }
  { movz $MAX_PTR, $MSCRATCH, $ACTS_PTR
    f32@MIN_OR_MAX_LOWER@ $MAX, $ACT, $MAX }
2:
  // Handle remaining conditional set for the MAX_PTR, MAX is already handled.
  ld32step $azero, $mzero, $ACTS_PTR+=, 1
  atom $MSCRATCH, $ASCRATCH
  movz $MAX_PTR, $MSCRATCH, $ACTS_PTR

  // Calculate the index from $MAX_PTR
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  sub $MAX_PTR, $MAX_PTR, $ACTS_PTR
  shr $MAX_PTR, $MAX_PTR, LOG2_SIZEOF_FLOAT
  // $MAX_PTR always ends up 2 elements ahead by the end of the loop
  add $MAX_PTR, $MAX_PTR, -2
  // Add the offset for the vertex's activations to the max index
  ld32 $MSCRATCH, $mvertex_base, $mzero, INDEX_VOFFSET
  add $MAX_PTR, $MAX_PTR, $MSCRATCH

  // Load maxValue/maxIndex output pointers, store
  ld32 $MSCRATCH, $mvertex_base, $mzero, MAXACT_VOFFSET
  st32 $MAX, $MSCRATCH, $mzero, $WORKER_ID
  ld32 $MSCRATCH, $mvertex_base, $mzero, MAXINDEX_VOFFSET
  stm32 $MAX_PTR, $MSCRATCH, $WORKER_ID

.Lfloat_end:
  exitz $mzero

// Only set the size for the int version so we don't count it twice.
.size VERTEX(float, int), .-VERTEX(float, int)

.section .text.VERTEX(half, unsigned_int)
.globl VERTEX(half, unsigned_int)
.type VERTEX(half, unsigned_int), @function

.globl VERTEX(half, int)
.type VERTEX(half, int), @function

.align 8
  nop
.supervisor
VERTEX(half, unsigned_int):
VERTEX(half, int):
  setzi $WORKER_ENTRY, .Lhalf_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.Lhalf_worker:
.worker
  // Load starting vertex state
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  ld32 $SIZE, $mvertex_base, $mzero, SIZE_VOFFSET
  ld32 $WSIZE, $mvertex_base, $mzero, WSIZE_VOFFSET

  // Get worker ID
  get $WORKER_ID, $WSR
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Compute worker region
  mul $OFFSET, $WORKER_ID, $WSIZE
  add $END, $WORKER_ID, 1
  mul $END, $END, $WSIZE
  min $END, $END, $SIZE

  // Calculate number of elements, sub 1 for first element loaded below
  sub $N, $END, $OFFSET
  add $N, $N, -1

  // If there is no actual work for this worker, early exit
  brneg $N, .Lhalf_end

  // Offset pointer
  ldb16step $azero, $mzero, $ACTS_PTR+=, $OFFSET

  // Initialise max
  mov $MAX_PTR, $ACTS_PTR
  ldb16step $MAX, $mzero, $MAX_PTR+=, 2
  ldb16step $ACT, $mzero, $ACTS_PTR+=, 1

  // Doesn't matter what $ASCRATCH is on the first
  // loop iteration, $MAX_PTR will end up the same value
  rpt $N, (2f-1f)/8-1
1:
  { ldb16step $ACT, $mzero, $ACTS_PTR+=, 1
    fnop }
  { atom $MSCRATCH, $ASCRATCH
    f16v2@COMPARE_OP@ $ASCRATCH, $ACT, $MAX }
  { movz $MAX_PTR, $MSCRATCH, $ACTS_PTR
    f16v2@MIN_OR_MAX_LOWER@ $MAX, $ACT, $MAX }
2:
  // Handle remaining conditional set for the MAX_PTR, MAX is already handled.
  ldb16step $azero, $mzero, $ACTS_PTR+=, 1
  atom $MSCRATCH, $ASCRATCH
  movz $MAX_PTR, $MSCRATCH, $ACTS_PTR

  // Calculate the index from $MAX_PTR
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  sub $MAX_PTR, $MAX_PTR, $ACTS_PTR
  shr $MAX_PTR, $MAX_PTR, LOG2_SIZEOF_HALF
  // $MAX_PTR always ends up 2 elements ahead by the end of the loop
  add $MAX_PTR, $MAX_PTR, -2
  // Add the offset for the vertex's activations to the max index
  ld32 $MSCRATCH, $mvertex_base, $mzero, INDEX_VOFFSET
  add $MAX_PTR, $MAX_PTR, $MSCRATCH

  // Load maxValue/maxIndex output pointers, store
  { ld32 $MSCRATCH, $mvertex_base, $mzero, MAXACT_VOFFSET
    f16tof32 $MAX, $MAX }
  st32 $MAX, $MSCRATCH, $mzero, $WORKER_ID
  ld32 $MSCRATCH, $mvertex_base, $mzero, MAXINDEX_VOFFSET
  stm32 $MAX_PTR, $MSCRATCH, $WORKER_ID

.Lhalf_end:
  exitz $mzero

// Only set the size for the int version so we don't count it twice.
.size VERTEX(half, int), .-VERTEX(half, int)

#endif // __IPU__
