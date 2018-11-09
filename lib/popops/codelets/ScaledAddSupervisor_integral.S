#ifdef __IPU__

#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty

.globl VERTEX(int_true)
.type VERTEX(int_true), @function

.globl VERTEX(unsigned_int_true)
.type VERTEX(unsigned_int_true), @function

.globl VERTEX(int_false)
.type VERTEX(int_false), @function

.globl VERTEX(unsigned_int_false)
.type VERTEX(unsigned_int_false), @function

// constants
#define VERTEX_DATA_START_OFFSET 0
#define VERTEX_DATA_SIZE_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
//Constant (k) and non-constant (ptr to factor) versions
#define VERTEX_K_OFFSET 3
#define VERTEX_FACTOR_OFFSET 3

// integer variables
#define deltaPtr m0
#define dataPtr m1
#define sizeD3M1 m2
#define rem m3
#define k m4

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
#define dataSize m5
#define sizeD3 m6

.section .text.VERTEX(int_false)
.align 4
VERTEX(int_false):
VERTEX(unsigned_int_false):
  // keeping this before the branch means it doesn't cause a stall later
  add $sp, $sp, -8

  ld32 $k, $m0, $mzero, VERTEX_K_OFFSET
  ld32 $k, $mzero, $k, 0 // 6 cycles
  bri  VERTEX(int_common)// 6 cycles
.size VERTEX(int_false), .-VERTEX(int_false)

.section .text.VERTEX(int_true)
.align 4

VERTEX(int_true):
VERTEX(unsigned_int_true):
 // keeping this before the branch means it doesn't cause a stall later
  add $sp, $sp, -8

  ld32 $k, $m0, $mzero, VERTEX_K_OFFSET
  bri  VERTEX(int_common) //6 cycles
.size VERTEX(int_true), .-VERTEX(int_true)

.section .text.VERTEX(int_common)
.align 4

VERTEX(int_common):
  ld32 $dataSize, $m0, $mzero, VERTEX_DATA_SIZE_OFFSET
  ld32 $dataPtr, $m0, $mzero, VERTEX_DATA_START_OFFSET
  ld32 $deltaPtr, $m0, $mzero, VERTEX_DELTAS_OFFSET
  // NOTE: this trashes the vertex state pointer because $deltaPtr is mapped to $m0.

  setzi $sizeD3, 0xAAAB

  // store the registers we need to restore at the end to the stack
  st32 $fp, $sp, $mzero, 0
  st32 $lr, $sp, $mzero, 1

  // calculate n/3 using (n * 0xAAAB) >> 17
  // see recipe #1 for how these constants were derived:
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  mul $sizeD3, $dataSize, $sizeD3 // 6 cycles
  shr $sizeD3, $sizeD3, 17 // 6 cycles

  // calculate n%3 using n - (n/3)*3
  mul $rem, $sizeD3, 3 // 6 cycles
  sub $rem, $dataSize, $rem // 6 cycles

  // minus 1 from each size for the brnzdec
  add $sizeD3M1, $sizeD3, -1

  // skip the x3 loop if we have less than 3 to process. 6 cycles if
  // sizeD3M1 < 0, 1 otherwise.
  brneg $sizeD3M1, .Lloop1_prologue // 6 cycles

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

  // avoid register bubble
  nop

  mul $delta0, $delta0, $k
  mul $delta1, $delta1, $k
  mul $delta2, $delta2, $k

  add $data0, $data0, $delta0
  add $data1, $data1, $delta1
  add $data2, $data2, $delta2

  // avoid register bubble
  nop ; nop ; nop ; nop

  st32step $data0, $mzero, $dataPtr+=, 1
  st32step $data1, $mzero, $dataPtr+=, 1
  st32step $data2, $mzero, $dataPtr+=, 1

  // 6 cycle penalty here each iteration.
  brnzdec $sizeD3M1, .Lloop3

.Lloop1_prologue:
  // process the remainder (if any). 6 cycles if rem == 0, 1 otherwise.
  brz $rem, .Lepilogue

  // minus 1 from the remainder for the brnzdec. do this before we check for
  add $rem, $rem, -1 // 6 cycles
.Lloop1:
  ld32 $data0, $mzero, $dataPtr, 0
  ld32step $delta0, $mzero, $deltaPtr+=, 1

  mul $delta0, $delta0, $k // 6 cycles
  add $data0, $data0, $delta0 // 6 cycles
  st32step $data0, $mzero, $dataPtr+=, 1 // 6 cycles

  // 6 cycle penalty here each iteration. perhaps it's worth unrolling this
  // loop as it will only ever be 2 iterations at most.
  brnzdec $rem, .Lloop1

  // intentional fallthrough

.Lepilogue:
  // restore $fp, $lr and $sp
  ld32step $fp, $mzero, $sp+=, 1
  ld32step $lr, $mzero, $sp+=, 1

  br $lr // 6 cycles

.size VERTEX(int_common), .-VERTEX(int_common)

#endif // __IPU__
