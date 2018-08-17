#ifdef __IPU__

// Assembly implementation of popnn::ReduceMaxClassGather vertex template
// variations.

// Restrictions:
//
//  * 4-byte aligned outputs

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define VERTEX(fpType,labelType) \
  __runCodelet_popnn__ReduceMaxClassGather___ ## fpType ## _ ## labelType

// Constants
#define ACTS_VOFFSET 0
#define INDEX_VOFFSET 1
#define MAXACT_VOFFSET 2
#define MAXINDEX_VOFFSET 3
#define SIZE_VOFFSET 4
#define DIVISORLOG2_VOFFSET (5 * 2) // 16-bit load

#define GRAINSIZE_HALF 2
#define LOG2_GRAINSIZE_HALF 1

// Supervisor register aliases
#define SUPER_BASE m0
#define WORKER_ENTRY m1

// Worker register aliases
#define WORKER_ID m0
#define ACTS_PTR m1
#define MAXINDEX_PTR m2
#define MAXACT_PTR m3
#define TOTALSIZE m3
#define SIZE m4
#define DIVISORLOG2 m5
#define OFFSET m6
#define END m7
#define N m8
#define N0 m8
#define N1 m9
#define MAXINDEX m10
#define MSCRATCH m11

#define ACT a0
#define MAXACT a1
#define GRAIN a2
#define ASCRATCH a6

.section .text.VERTEX(float, unsigned_int)
.globl VERTEX(float, unsigned_int)
.type VERTEX(float, unsigned_int), @function

.globl VERTEX(float, int)
.type VERTEX(float, int), @function

.align 4
VERTEX(float, unsigned_int):
VERTEX(float, int):
  setzi $WORKER_ENTRY, .Lfloat_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.Lfloat_worker:
  // Load starting vertex state
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  ld32 $SIZE, $mvertex_base, $mzero, SIZE_VOFFSET
  ldz16 $DIVISORLOG2, $mvertex_base, $mzero, DIVISORLOG2_VOFFSET

  // Get worker ID
  get $WORKER_ID, $WSR
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Compute worker region
  shl $OFFSET, $WORKER_ID, $DIVISORLOG2
  add $END, $WORKER_ID, 1
  shl $END, $END, $DIVISORLOG2
  min $END, $END, $SIZE

  // Calculate number of elements, sub 1 for first loaded below
  sub $N, $END, $OFFSET
  add $SIZE, $N, -1

  // If there is no actual work for this worker, early exit
  brneg $SIZE, .Lfloat_end

  // Offset pointer
  ld32step $azero, $mzero, $ACTS_PTR+=, $OFFSET

  // Get starting max values
  ld32step $MAXACT, $mzero, $ACTS_PTR+=, 1
  add $MAXINDEX, $N, -1

  brz $MAXINDEX, .Lfloat_loop_end
  // -2 for already handled value and for post-decrement brnzdec below
  add $N, $N, -2
.Lfloat_loop_start:
  ld32step $ACT, $mzero, $ACTS_PTR+=, 1
  f32cmpgt $ASCRATCH, $ACT, $MAXACT
  atom $MSCRATCH, $ASCRATCH
  brz $MSCRATCH, .Lfloat_not_greater
  { mov $MAXINDEX, $N
    mov $MAXACT, $ACT }
.Lfloat_not_greater:
  brnzdec $N, .Lfloat_loop_start
.Lfloat_loop_end:

  // Recover the max index from N above, add the index from vertex state
  sub $MAXINDEX, $SIZE, $MAXINDEX
  add $MAXINDEX, $MAXINDEX, $OFFSET
  ld32 $MSCRATCH, $mvertex_base, $mzero, INDEX_VOFFSET
  add $MAXINDEX, $MAXINDEX, $MSCRATCH

  // Load maxValue/maxIndex output pointers
  ld32 $MAXACT_PTR, $mvertex_base, $mzero, MAXACT_VOFFSET
  ld32 $MAXINDEX_PTR, $mvertex_base, $mzero, MAXINDEX_VOFFSET

  st32 $MAXACT, $MAXACT_PTR, $mzero, $WORKER_ID
  stm32 $MAXINDEX, $MAXINDEX_PTR, $WORKER_ID

.Lfloat_end:
  exitz $mzero

.size VERTEX(float, unsigned_int), .-VERTEX(float, unsigned_int)
.size VERTEX(float, int), .-VERTEX(float, int)


.section .text.VERTEX(half, unsigned_int)
.globl VERTEX(half, unsigned_int)
.type VERTEX(half, unsigned_int), @function

.globl VERTEX(half, int)
.type VERTEX(half, int), @function

.align 8
VERTEX(half, unsigned_int):
VERTEX(half, int):
  setzi $WORKER_ENTRY, .Lhalf_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

.Lhalf_worker:
  // Load starting vertex state
  ld32 $TOTALSIZE, $mvertex_base, $mzero, SIZE_VOFFSET
  ldz16 $DIVISORLOG2, $mvertex_base, $mzero, DIVISORLOG2_VOFFSET
  ld32 $MAXINDEX_PTR, $mvertex_base, $mzero, MAXINDEX_VOFFSET

  // Get worker ID
  get $WORKER_ID, $WSR
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  shl $N0, $WORKER_ID, LOG2_GRAINSIZE_HALF

  // If there are no outputs, it isn't safe to proceed
  shl $OFFSET, $N0, $DIVISORLOG2
  cmpult $MSCRATCH, $OFFSET, $TOTALSIZE
  brz $MSCRATCH, .Lhalf_end
.Lhalf_loop_outer_start:
  // Compute this region
  shl $OFFSET, $N0, $DIVISORLOG2
  add $END, $N0, 1
  shl $END, $END, $DIVISORLOG2
  min $END, $END, $TOTALSIZE

  sub $N1, $END, $OFFSET
  add $SIZE, $N1, -1

  // If this region is empty, this worker is done, so go to the store
  brneg $SIZE, .Lhalf_handle_maxact

  // Load pointer and offset
  ld32 $ACTS_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  ldb16step $azero, $mzero, $ACTS_PTR+=, $OFFSET

  // Get starting max values
  ldb16step $MAXACT, $mzero, $ACTS_PTR+=, 1
  add $MAXINDEX, $N1, -1

  // -2 for already handled value and for post-decrement brnzdec below
  add $N1, $N1, -2
  brneg $N1, .Lhalf_loop_end
.Lhalf_loop_start:
  ldb16step $ACT, $mzero, $ACTS_PTR+=, 1
  f16v2cmpgt $ASCRATCH, $ACT, $MAXACT
  atom $MSCRATCH, $ASCRATCH
  brz $MSCRATCH, .Lhalf_not_greater
  { mov $MAXINDEX, $N1
    mov $MAXACT, $ACT }
.Lhalf_not_greater:
  brnzdec $N1, .Lhalf_loop_start
.Lhalf_loop_end:

  // Recover max index from N above, add vertex offset, add worker offset
  sub $MAXINDEX, $SIZE, $MAXINDEX
  add $MAXINDEX, $MAXINDEX, $OFFSET
  ld32 $MSCRATCH, $mvertex_base, $mzero, INDEX_VOFFSET
  add $MAXINDEX, $MAXINDEX, $MSCRATCH
  stm32 $MAXINDEX, $MAXINDEX_PTR, $N0

  // Store maxAct away for storage later
  roll16 $GRAIN, $GRAIN, $MAXACT
  add $N0, $N0, 1
  add $MSCRATCH, $WORKER_ID, 1
  shl $MSCRATCH, $MSCRATCH, LOG2_GRAINSIZE_HALF
  cmpeq $MSCRATCH, $MSCRATCH, $N0
  brz $MSCRATCH, .Lhalf_loop_outer_start

.Lhalf_handle_maxact:
  ld32 $MAXACT_PTR, $mvertex_base, $mzero, MAXACT_VOFFSET

  // Check if we have the full word of output or not
  and $N0, $N0, 1
  brz $N0, .Lhalf_store_maxact

  // We have a single output to write, merge with the hi
  // half that is already in memory.
  ld32 $ASCRATCH, $MAXACT_PTR, $mzero, $WORKER_ID
  sort4x16hi $GRAIN, $GRAIN, $ASCRATCH

.Lhalf_store_maxact:
  // By happy coincidence, the 32-bit output slot we
  // want is exactly the worker ID
  st32 $GRAIN, $MAXACT_PTR, $mzero, $WORKER_ID
 
.Lhalf_end:
  exitz $mzero

.size VERTEX(half, unsigned_int), .-VERTEX(half, unsigned_int)
.size VERTEX(half, int), .-VERTEX(half, int)

#endif // __IPU__
