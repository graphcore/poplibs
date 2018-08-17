#ifdef __IPU__

// Assembly implementation of popnn::ReduceMaxClassGather vertex template
// variations.

// No restrictions

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define VERTEX(fpType,labelType) \
  __runCodelet_popnn__ReduceMaxClassSparse___ ## fpType ## _ ## labelType

// Constants
#define ACTS_VOFFSET 0
#define LABELS_VOFFSET 2
#define MAXACT_VOFFSET 3
#define MAXINDEX_VOFFSET 4

#define LOG2_SIZEOF_PTR 2

// Register aliases
#define ACTS_OUTER_PTR m0
#define LABELS_OUTER_PTR m1
#define ACT_PTR m2
#define MAXACT_PTR m2
#define MAXINDEX_PTR m3
#define SIZE m4
#define N m5
#define MAXINDEX m6
#define MSCRATCH m10

#define ACT a0
#define MAXACT a1
#define ASCRATCH a6

.section .text.VERTEX(float, unsigned_int)
.globl VERTEX(float, unsigned_int)
.type VERTEX(float, unsigned_int), @function

.globl VERTEX(float, int)
.type VERTEX(float, int), @function

.align 4
VERTEX(float, unsigned_int):
VERTEX(float, int):
  ld32 $ACTS_OUTER_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  ld32 $MSCRATCH, $mvertex_base, $mzero, (ACTS_VOFFSET + 1)
  ld32 $LABELS_OUTER_PTR, $mvertex_base, $mzero, LABELS_VOFFSET
  
  // Calculate no. of elements
  sub $SIZE, $MSCRATCH, $ACTS_OUTER_PTR
  shr $SIZE, $SIZE, LOG2_SIZEOF_PTR

  // Initialise maxValue/Index
  ld32step $ACT_PTR, $mzero, $ACTS_OUTER_PTR+=, 1
  ld32 $MAXACT, $ACT_PTR, $mzero, 0
  // sub 1 for brnzdec
  add $SIZE, $SIZE, -1
  mov $MAXINDEX, $SIZE

  // sub 1 for already loaded element
  add $N, $SIZE, -1
  brneg $N, .Lfloat_loop_end
.Lfloat_loop_begin:
  ld32step $ACT_PTR, $mzero, $ACTS_OUTER_PTR+=, 1
  ld32 $ACT, $ACT_PTR, $mzero, 0
  f32cmpgt $ASCRATCH, $ACT, $MAXACT
  atom $MSCRATCH, $ASCRATCH
  brz $MSCRATCH, .Lfloat_not_greater
  { mov $MAXINDEX, $N
    mov $MAXACT, $ACT }
.Lfloat_not_greater:
  brnzdec $N, .Lfloat_loop_begin
.Lfloat_loop_end:

  // Recover max index from N above, load label
  sub $MAXINDEX, $SIZE, $MAXINDEX
  ld32 $MAXINDEX, $LABELS_OUTER_PTR, $mzero, $MAXINDEX
  ld32 $MAXINDEX, $MAXINDEX, $mzero, 0

  // Load maxValue/maxIndex pointers
  ld32 $MAXACT_PTR, $mvertex_base, $mzero, MAXACT_VOFFSET
  ld32 $MAXINDEX_PTR, $mvertex_base, $mzero, MAXINDEX_VOFFSET
  
  // Store maxValue/maxIndex
  st32 $MAXACT, $MAXACT_PTR, $mzero, 0
  st32 $MAXINDEX, $MAXINDEX_PTR, $mzero, 0
  
.Lfloat_end:
  exitz $mzero

.size VERTEX(float, unsigned_int), .-VERTEX(float, unsigned_int)
.size VERTEX(float, int), .-VERTEX(float, int)


.section .text.VERTEX(half, unsigned_int)
.globl VERTEX(half, unsigned_int)
.type VERTEX(half, unsigned_int), @function

.globl VERTEX(half, int)
.type VERTEX(half, int), @function

.align 4
VERTEX(half, unsigned_int):
VERTEX(half, int):
  ld32 $ACTS_OUTER_PTR, $mvertex_base, $mzero, ACTS_VOFFSET
  ld32 $MSCRATCH, $mvertex_base, $mzero, (ACTS_VOFFSET + 1)
  ld32 $LABELS_OUTER_PTR, $mvertex_base, $mzero, LABELS_VOFFSET
  
  // Calculate no. of elements
  sub $SIZE, $MSCRATCH, $ACTS_OUTER_PTR
  shr $SIZE, $SIZE, LOG2_SIZEOF_PTR

  // Initialise maxValue/Index
  ld32step $ACT_PTR, $mzero, $ACTS_OUTER_PTR+=, 1
  ldb16 $MAXACT, $ACT_PTR, $mzero, 0

  // sub 1 for brnzdec
  add $SIZE, $SIZE, -1
  mov $MAXINDEX, $SIZE

  // sub 1 for already loaded element
  add $N, $SIZE, -1
  brneg $N, .Lhalf_loop_end
.Lhalf_loop_begin:
  ld32step $ACT_PTR, $mzero, $ACTS_OUTER_PTR+=, 1
  ldb16 $ACT, $ACT_PTR, $mzero, 0
  f16v2cmpgt $ASCRATCH, $ACT, $MAXACT
  atom $MSCRATCH, $ASCRATCH
  brz $MSCRATCH, .Lhalf_not_greater
  { mov $MAXINDEX, $N
    mov $MAXACT, $ACT }
.Lhalf_not_greater:
  brnzdec $N, .Lhalf_loop_begin
.Lhalf_loop_end:

  // Recover max index from N above, load label
  sub $MAXINDEX, $SIZE, $MAXINDEX
  ld32 $MAXINDEX, $LABELS_OUTER_PTR, $mzero, $MAXINDEX
  ld32 $MAXINDEX, $MAXINDEX, $mzero, 0

  // Load maxValue/maxIndex pointers
  ld32 $MAXACT_PTR, $mvertex_base, $mzero, MAXACT_VOFFSET
  ld32 $MAXINDEX_PTR, $mvertex_base, $mzero, MAXINDEX_VOFFSET
  
  // Store maxIndex
  st32 $MAXINDEX, $MAXINDEX_PTR, $mzero, 0

  // Store maxValue
  and $MSCRATCH, $MAXACT_PTR, 0x2
  brnz $MSCRATCH, .Lhalf_misaligned
  ld32 $ACT, $MAXACT_PTR, $mzero, 0
  sort4x16hi $ACT, $MAXACT, $ACT
  st32 $ACT, $MAXACT_PTR, $mzero, 0
  bri .Lhalf_end

.Lhalf_misaligned:
  andc $MAXACT_PTR, $MAXACT_PTR, 0x3
  ld32 $ACT, $MAXACT_PTR, $mzero, 0
  sort4x16lo $ACT, $ACT, $MAXACT
  st32 $ACT, $MAXACT_PTR, $mzero, 0
  
.Lhalf_end:
  exitz $mzero

.size VERTEX(half, unsigned_int), .-VERTEX(half, unsigned_int)
.size VERTEX(half, int), .-VERTEX(half, int)

#endif // __IPU__
