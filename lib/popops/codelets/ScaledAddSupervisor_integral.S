#ifdef __IPU__

#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty

.globl VERTEX(int)
.type VERTEX(int), @function

.globl VERTEX(unsigned_int)
.type VERTEX(unsigned_int), @function

// constants
#define VERTEX_DATA_START_OFFSET 0
#define VERTEX_DATA_END_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
#define VERTEX_K_OFFSET 3

#define LOG2_SIZEOF_INT 2

// integer variables
#define k m0
#define dataPtr m1
#define sizeD3M1 m2
#define rem m3
#define deltaPtr m4

#define data0 m5
#define data1 m6
#define data2 m7

// we trash $fp (m9) and $lr (m10) here but we keep $sp and use that to store
// them to the stack so this fine.
#define delta0 m8
#define delta1 m9
#define delta2 m10

// scratch registers, these use the same registers as the dataN registers but
// are only used in the prologue.
#define size m5
#define sizeD3 m6

.section .text.VERTEX
.align 4

VERTEX(int):
VERTEX(unsigned_int):
  ld32 $size, $m0, $mzero, VERTEX_DATA_END_OFFSET
  ld32 $dataPtr, $m0, $mzero, VERTEX_DATA_START_OFFSET
  ld32 $deltaPtr, $m0, $mzero, VERTEX_DELTAS_OFFSET
  // NOTE: this trashes the vertex state pointer because $k is mapped to $m0.
  ld32 $k, $m0, $mzero, VERTEX_K_OFFSET

  // store the registers we need to restore at the end to the stack. do this
  // after we have loaded the vertex state because this provides us with enough
  // cycles between loading $size and using it to avoid a cycle penalty.
  add $sp, $sp, -8
  st32 $fp, $sp, $mzero, 0
  st32 $lr, $sp, $mzero, 1

  // calculate size by doing (end-begin)/sizeof(int)
  sub $size, $size, $dataPtr
  shr $size, $size, LOG2_SIZEOF_INT

  // calculate n/3 using (n * 0xAAAB) >> 17
  // see recipe #1 for how these constants were derived:
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  setzi $sizeD3, 0xAAAB
  mul $sizeD3, $size, $sizeD3
  shr $sizeD3, $sizeD3, 17

  // calculate n%3 using n - (n/3)*3
  mul $rem, $sizeD3, 3
  sub $rem, $size, $rem

  // minus 1 from each size for the brnzdec
  add $sizeD3M1, $sizeD3, -1

  // skip the x3 loop if we have less than 3 to process. 6 cycles if
  // sizeD3M1 < 0, 1 otherwise.
  brneg $sizeD3M1, .Lloop1_prologue

.Lloop3:
  // we must make sure we won't use any of the resulting values from a load
  // until at least 6 cycles later. processing three at a time here enforces
  // that guarantee.
  ld32step $delta0, $mzero, $deltaPtr+=, 1
  ld32step $delta1, $mzero, $deltaPtr+=, 1
  ld32step $delta2, $mzero, $deltaPtr+=, 1

  ld32 $data0, $mzero, $dataPtr, 0
  ld32 $data1, $mzero, $dataPtr, 1
  ld32 $data2, $mzero, $dataPtr, 2

  mul $delta0, $delta0, $k
  mul $delta1, $delta1, $k
  mul $delta2, $delta2, $k

  add $data0, $data0, $delta0
  add $data1, $data1, $delta1
  add $data2, $data2, $delta2

  st32step $data0, $mzero, $dataPtr+=, 1
  st32step $data1, $mzero, $dataPtr+=, 1
  st32step $data2, $mzero, $dataPtr+=, 1

  // 6 cycle penalty here each iteration.
  brnzdec $sizeD3M1, .Lloop3

.Lloop1_prologue:
  // process the remainder (if any). 6 cycles if rem == 0, 1 otherwise.
  brz $rem, .Lepilogue

  // minus 1 from the remainder for the brnzdec. do this before we check for
  add $rem, $rem, -1
.Lloop1:
  ld32step $delta0, $mzero, $deltaPtr+=, 1
  ld32 $data0, $mzero, $dataPtr, 0

  // wait 4 more cycles before using $delta0
  delay 3

  mul $delta0, $delta0, $k
  add $data0, $data0, $delta0
  st32step $data0, $mzero, $dataPtr+=, 1

  // 6 cycle penalty here each iteration. perhaps it's worth unrolling this
  // loop as it will only ever be 2 iterations at most.
  brnzdec $rem, .Lloop1

  // intentional fallthrough

.Lepilogue:
  // restore $fp, $lr and $sp
  ld32step $fp, $mzero, $sp+=, 1
  ld32step $lr, $mzero, $sp+=, 1

  br $lr

#undef size
.size VERTEX(int), .-VERTEX(int)

#endif // __IPU__
