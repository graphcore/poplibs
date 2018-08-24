#ifdef __IPU__


#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define OUT_OFF          4
#define OUT_OFFSET       8
#define IN_OFF           12
#define IN_OFFSET        16
#define NUM_PART_OFF     10
#define SCALE_OFF        0

#define DELTAN_SIZE_OFF  20
#define DELTAN_SIZE_CLR  12
#define SCPTR_SIZE_OFF   18
#define SCPTR_SIZE_CLR   14

#define LDCONST_MASK     ((1<<20)-1)

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

#define NUM_ELEM        m0
#define OUT_i_PTR       m0
#define OUT_j_PTR       m1
#define IN_i_PTR        m2
#define IN_j_PTR        m3
#define OUT_i_SIZE      m4
#define OUT_j_SIZE      m5
#define OUT_BASE        m6
#define IN_j_DELTA      m6
#define NUM_PART_PTR    m7
#define SCRATCH2        m7
#define SCRATCH         m8
#define NUM_PART        m9
#define IN_BASE         m10
#define IN_j_SIZE       m11
#define VALUES_0        a0
#define VALUES_1        a1
#define VALUES_2        a2
#define VALUES_3        a3
#define ACC_0           a2
#define ACC_1           a3
#define ASCRATCH_0      a5
#define ZAACC           a4
#define SCALE           a6
#define SCALE2          a7


// ld macros populate the arf (VALUES_0:4) with partial data that
// will be used as the input to accumulation instructions. Scratch
// holds the offset from an 8 byte aligned ptr
// ------------------------------------------------------- //
#define ld128_MIS_2_ _ld128_MIS_2(__COUNTER__)
#define _ld128_MIS_2(ID) ld128_MIS_2(ID)

#define ld128_MIS_2(ID) and $SCRATCH2, $SCRATCH, 0x7; \
brz $SCRATCH2, _aligned_case_##ID; \
and $SCRATCH2, $SCRATCH, 0x2; \
brz $SCRATCH2, _4_misaligned_##ID; \
ldb16 $VALUES_0, $IN_j_PTR, $SCRATCH, 0; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 1; \
{ldb16 $VALUES_1, $IN_j_PTR, $SCRATCH, 2; \
 roll16 $VALUES_0, $VALUES_0, $ASCRATCH_0}; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 3; \
{ldb16 $VALUES_2, $IN_j_PTR, $SCRATCH, 4; \
 roll16 $VALUES_1, $VALUES_1, $ASCRATCH_0}; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 5; \
{ldb16 $VALUES_3, $IN_j_PTR, $SCRATCH, 6; \
 roll16 $VALUES_2, $VALUES_2, $ASCRATCH_0}; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 7; \
{bri _end_ld_##ID; roll16 $VALUES_3, $VALUES_3, $ASCRATCH_0} \
_4_misaligned_##ID:; \
ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0; \
ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1; \
ld32 $VALUES_2, $IN_j_PTR, $SCRATCH, 2; \
ld32 $VALUES_3, $IN_j_PTR, $SCRATCH, 3; \
bri _end_ld_##ID; \
_aligned_case_##ID:; \
ld64 $VALUES_0:1, $IN_j_PTR, $SCRATCH, 0; \
ld64 $VALUES_2:3, $IN_j_PTR, $SCRATCH, 1; \
_end_ld_##ID:;

// ------------------------------------------------------- //
#define ld64_MIS_2_ _ld64_MIS_2(__COUNTER__)
#define _ld64_MIS_2(ID) ld64_MIS_2(ID)

#define ld64_MIS_2(ID) and $SCRATCH2, $SCRATCH, 0x3; \
brz $SCRATCH2, _aligned_case_##ID; \
ldb16 $VALUES_0, $IN_j_PTR, $SCRATCH, 0; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 1; \
{ldb16 $VALUES_1, $IN_j_PTR, $SCRATCH, 2; \
 roll16 $VALUES_0, $VALUES_0, $ASCRATCH_0}; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 3; \
{bri _end_ld_##ID; roll16 $VALUES_1, $VALUES_1, $ASCRATCH_0} \
_aligned_case_##ID:; \
ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0; \
ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1; \
_end_ld_##ID:;

// ------------------------------------------------------- //
#define ld32_MIS_2_ _ld32_MIS_2(__COUNTER__)
#define _ld32_MIS_2(ID) ld32_MIS_2(ID)

#define ld32_MIS_2(ID) and $SCRATCH2, $SCRATCH, 0x3; \
brz $SCRATCH2, _aligned_case_##ID; \
ldb16 $VALUES_0, $IN_j_PTR, $SCRATCH, 0; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 1; \
{bri _end_ld_##ID; roll16 $VALUES_0, $VALUES_0, $ASCRATCH_0} \
_aligned_case_##ID:; \
ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0; \
_end_ld_##ID:;

// ------------------------------------------------------- //


#define MANGLE(op, ptype, otype, paos, scale) \
            MANGLE_(op, ptype, otype, paos, scale)
#define MANGLE_(op, ptype, otype, paos, scale) \
__runCodelet_popops__Reduce___popops__##op##_##ptype##_##otype##\
_##paos##_##scale##_##false


#define OP @ReduceOp@
#define INIT_FLOAT @INIT_FLOAT@
#define INIT_HALF @INIT_HALF@

#define F16V4 f16v4@INSTRUCTION@
#define F32V2 f32v2@INSTRUCTION@

#define LABEL_SUFFIX _h_h_

#define SIZE_OF_IN_TYPE 2


#define REDUCE_HALF_HALF MANGLE(OP, half, half, false, false)
#define REDUCE_HALF_HALF_PAOS MANGLE(OP, half, half, true, false)
#define REDUCE_HALF_HALF_SCALE MANGLE(OP, half, half, false, true)
#define REDUCE_HALF_HALF_PAOS_SCALE MANGLE(OP, half, half, true, true)
#define REDUCE_FLOAT_FLOAT MANGLE(OP, float, float, false, false)
#define REDUCE_FLOAT_FLOAT_PAOS MANGLE(OP, float, float, true, false)
#define REDUCE_FLOAT_FLOAT_SCALE MANGLE(OP, float, float, false, true)
#define REDUCE_FLOAT_FLOAT_PAOS_SCALE MANGLE(OP, float, float, true, true)

.globl REDUCE_HALF_HALF
.type REDUCE_HALF_HALF, @function
.globl REDUCE_HALF_HALF_PAOS
.type REDUCE_HALF_HALF_PAOS, @function
.globl REDUCE_HALF_HALF_SCALE
.type REDUCE_HALF_HALF_SCALE, @function
.globl REDUCE_HALF_HALF_PAOS_SCALE
.type REDUCE_HALF_HALF_PAOS_SCALE, @function

.section .text.REDUCE_HALF_HALF, "ax"
.align 8
REDUCE_HALF_HALF:
REDUCE_HALF_HALF_PAOS:
REDUCE_HALF_HALF_SCALE:
REDUCE_HALF_HALF_PAOS_SCALE:
// ************************************************* //
// Load vertex state
// ************************************************* //
  ld32       $SCALE, $mvertex_base, $mzero, SCALE_OFF/4
  ld32       $OUT_BASE, $mvertex_base, $mzero, OUT_OFF/4
  ldz16      $OUT_i_PTR, $mvertex_base, $mzero, OUT_OFFSET/2
  ldz16      $NUM_PART_PTR, $mvertex_base, $mzero, NUM_PART_OFF/2
  {
    ld32       $IN_BASE, $mvertex_base, $mzero, IN_OFF/4
    f32tof16   $SCALE, $SCALE
  }
  {
    ldz16      $IN_i_PTR, $mvertex_base, $mzero, IN_OFFSET/2
    mov        $SCALE2, $SCALE
  }

// ************************************************* //
// Useful constants
// ************************************************* //
  {
    setzi    $SCRATCH, TMEM_REGION0_BASE_ADDR
    setzi    $ZAACC, ZAACC_BITMASK
  }

// ************************************************* //
// Unpack scaled pointers
// ************************************************* //
  shl        $OUT_i_PTR, $OUT_i_PTR, 2
  shl        $IN_i_PTR, $IN_i_PTR, 2
  shl        $NUM_PART_PTR, $NUM_PART_PTR, 2
  add        $OUT_i_PTR, $OUT_i_PTR, $SCRATCH
  add        $IN_i_PTR, $IN_i_PTR, $SCRATCH
  add        $NUM_PART_PTR, $NUM_PART_PTR, $SCRATCH

// ************************************************* //
// Extract size and bases
// ************************************************* //
  shr        $OUT_i_SIZE, $OUT_BASE, DELTAN_SIZE_OFF
  shl        $OUT_BASE, $OUT_BASE, DELTAN_SIZE_CLR
  shr        $OUT_BASE, $OUT_BASE, DELTAN_SIZE_CLR
  shl        $IN_BASE, $IN_BASE, DELTAN_SIZE_CLR
  shr        $IN_BASE, $IN_BASE, DELTAN_SIZE_CLR

// ************************************************* //
// Start loops, store in_i_ptr as rest every loop
// ************************************************* //
  st32       $IN_i_PTR, $mworker_base, $mzero, 1
  st32       $OUT_BASE, $mworker_base, $mzero, 2
  st32       $NUM_PART_PTR, $mworker_base, $mzero, 3
  st32       $OUT_i_PTR, $mworker_base, $mzero, 4
  brnzdec    $OUT_i_SIZE, _skip_e.LABEL_SUFFIX
  bri        _exit.LABEL_SUFFIX

_loop_over_reductions.LABEL_SUFFIX:
// ************************************************* //
// unpack offset and size
// ************************************************* //
  ld32       $OUT_BASE, $mworker_base, $mzero, 2
  ld32       $OUT_i_PTR, $mworker_base, $mzero, 4
  _skip_e.LABEL_SUFFIX:
  ld32step   $OUT_j_PTR, $mzero, $OUT_i_PTR+=, 1
  st32       $OUT_i_PTR, $mworker_base, $mzero, 4
  shr        $OUT_j_SIZE, $OUT_j_PTR, SCPTR_SIZE_OFF
  shl        $OUT_j_PTR, $OUT_j_PTR, SCPTR_SIZE_CLR
  shr        $OUT_j_PTR, $OUT_j_PTR, SCPTR_SIZE_CLR
  add        $OUT_j_PTR, $OUT_j_PTR, $OUT_BASE
  setzi      $IN_j_DELTA, 0

// ************************************************* //
// going to do 8 at a time and store remainder to memory
// ************************************************* //
  and        $SCRATCH, $OUT_j_SIZE, 0x3
  st32       $SCRATCH, $mworker_base, $mzero, 0
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 2

  ld32       $NUM_PART_PTR, $mworker_base, $mzero, 3
  ldz16step  $NUM_PART, $mzero, $NUM_PART_PTR+=, 1
  st32       $NUM_PART_PTR, $mworker_base, $mzero, 3

  brnzdec    $NUM_PART, _skip3.LABEL_SUFFIX
  shr        $SCRATCH, $OUT_j_SIZE, 1
  ldconst    $ACC_0, INIT_HALF
  rpt       $SCRATCH, 0
  {st32      $ACC_0, $mzero, $OUT_j_PTR+=, 1; fnop}
  and        $OUT_j_SIZE, $OUT_j_SIZE, 1
  brz        $OUT_j_SIZE, _out_j_size_end.LABEL_SUFFIX
  ld32       $ASCRATCH_0, $OUT_j_PTR, $mzero, 0
  sort4x16hi $ACC_0, $ACC_0, $ASCRATCH_0
  st32       $SCRATCH, $OUT_j_PTR, $mzero, 0
  bri        _out_j_size_end.LABEL_SUFFIX
  _skip3.LABEL_SUFFIX:

  st32       $NUM_PART, $mzero, $mworker_base, 5
  setzi      $IN_j_DELTA, 0

  brnzdec    $OUT_j_SIZE, _skip2.LABEL_SUFFIX
  bri        _out_j_size_remainder.LABEL_SUFFIX
  _skip2.LABEL_SUFFIX:

_out_j_loop.LABEL_SUFFIX:
  {
    ld32       $IN_i_PTR, $mworker_base, $mzero, 1
    setzi      $ACC_0, INIT_HALF & LDCONST_MASK
  }
  {
    ld32       $NUM_PART, $mzero, $mworker_base, 5
    or         $ACC_0, $ACC_0, INIT_HALF & ~LDCONST_MASK
  }
  mov        $ACC_1, $ACC_0

// ************************************************* //
// Loop over iputs accumulating
// ************************************************* //
_start_num_partials_loop.LABEL_SUFFIX:
  ld32step   $IN_j_PTR, $mzero, $IN_i_PTR+=, 1
  shr        $IN_j_SIZE, $IN_j_PTR, SCPTR_SIZE_OFF
  shl        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  shr        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  add        $IN_j_PTR, $IN_BASE, $IN_j_PTR

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE
  mov        $SCRATCH, $IN_j_DELTA

_in_j_loop_start.LABEL_SUFFIX:
  ld64_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V4 $ACC_0:3, $ACC_0:3, $VALUES_0:1
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop.LABEL_SUFFIX

// ************************************************* //
// end of 4 vector accumulating, scale and store
// ************************************************* //

  {
    add $IN_j_DELTA, $IN_j_DELTA, 8
    f16v4mul   $VALUES_0:1, $SCALE:7, $ACC_0:3
  }
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.LABEL_SUFFIX

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.LABEL_SUFFIX:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, 0

// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.LABEL_SUFFIX

  {
    ld32       $IN_i_PTR, $mworker_base, $mzero, 1
    setzi      $ACC_0, INIT_HALF & LDCONST_MASK
  }
  {
    ld32       $NUM_PART, $mzero, $mworker_base, 5
    or         $ACC_0, $ACC_0, INIT_HALF & ~LDCONST_MASK
  }

_start_num_partials_loop_2_rem.LABEL_SUFFIX:
  ld32step   $IN_j_PTR, $mzero, $IN_i_PTR+=, 1
  shr        $IN_j_SIZE, $IN_j_PTR, SCPTR_SIZE_OFF
  shl        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  shr        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  add        $IN_j_PTR, $IN_BASE, $IN_j_PTR

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE
  mov        $SCRATCH, $IN_j_DELTA

_in_j_loop_start_2_rem.LABEL_SUFFIX:
  ld32_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V4  $ACC_0:3, $ACC_0:3, $VALUES_0:1
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_2_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.LABEL_SUFFIX

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //

  {
    add $IN_j_DELTA, $IN_j_DELTA, 4
    f16v4mul   $VALUES_0:1, $SCALE:7, $ACC_0:3
  }
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.LABEL_SUFFIX

  {
    ld32       $IN_i_PTR, $mworker_base, $mzero, 1
    setzi      $ACC_0, INIT_HALF & LDCONST_MASK
  }
  {
    ld32       $NUM_PART, $mzero, $mworker_base, 5
    or         $ACC_0, $ACC_0, INIT_HALF & ~LDCONST_MASK
  }

_start_num_partials_loop_1_rem.LABEL_SUFFIX:
  ld32step   $IN_j_PTR, $mzero, $IN_i_PTR+=, 1
  shr        $IN_j_SIZE, $IN_j_PTR, SCPTR_SIZE_OFF
  shl        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  shr        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  add        $IN_j_PTR, $IN_BASE, $IN_j_PTR

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE
  mov        $SCRATCH, $IN_j_DELTA

_in_j_loop_start_1_rem.LABEL_SUFFIX:
  ldb16      $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V4   $ACC_0:3, $ACC_0:3, $VALUES_0:1
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.LABEL_SUFFIX

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //

  {
    ldb16 $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
    f16v4mul   $VALUES_0:1, $SCALE:7, $ACC_0:3
  }
  {
    add $IN_j_DELTA, $IN_j_DELTA, 2
    sort4x16lo $VALUES_0, $VALUES_0, $ASCRATCH_0
  }
  st32 $VALUES_0, $OUT_j_PTR, $mzero, 0

_out_j_size_end.LABEL_SUFFIX:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, 1
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.LABEL_SUFFIX
// ************************************************* //
// End of loops
// ************************************************* //
_exit.LABEL_SUFFIX:
  exitz      $mzero

.size REDUCE_HALF_HALF,\
              .-REDUCE_HALF_HALF





// ------------------------------------------------------- //
// ------------------------------------------------------- //
// Float Float
// ------------------------------------------------------- //
// ------------------------------------------------------- //


#define LABEL_SUFFIX _f_f_

#undef SIZE_OF_IN_TYPE
#define SIZE_OF_IN_TYPE 4

.globl REDUCE_FLOAT_FLOAT
.type REDUCE_FLOAT_FLOAT, @function
.globl REDUCE_FLOAT_FLOAT_PAOS
.type REDUCE_FLOAT_FLOAT_PAOS, @function
.globl REDUCE_FLOAT_FLOAT_SCALE
.type REDUCE_FLOAT_FLOAT_SCALE, @function
.globl REDUCE_FLOAT_FLOAT_PAOS_SCALE
.type REDUCE_FLOAT_FLOAT_PAOS_SCALE, @function

.section .text.REDUCE_FLOAT_FLOAT, "ax"
.align 8
REDUCE_FLOAT_FLOAT:
REDUCE_FLOAT_FLOAT_PAOS:
REDUCE_FLOAT_FLOAT_SCALE:
REDUCE_FLOAT_FLOAT_PAOS_SCALE:
// ************************************************* //
// Load vertex state
// ************************************************* //
  ld32       $SCALE, $mvertex_base, $mzero, SCALE_OFF/4
  ld32       $OUT_BASE, $mvertex_base, $mzero, OUT_OFF/4
  ldz16      $OUT_i_PTR, $mvertex_base, $mzero, OUT_OFFSET/2
  ldz16      $NUM_PART_PTR, $mvertex_base, $mzero, NUM_PART_OFF/2
  ld32       $IN_BASE, $mvertex_base, $mzero, IN_OFF/4
  {
    ldz16      $IN_i_PTR, $mvertex_base, $mzero, IN_OFFSET/2
    mov        $SCALE2, $SCALE
  }

// ************************************************* //
// Useful constants
// ************************************************* //
  {
    setzi    $SCRATCH, TMEM_REGION0_BASE_ADDR
    setzi    $ZAACC, ZAACC_BITMASK
  }

// ************************************************* //
// Unpack scaled pointers
// ************************************************* //
  shl        $OUT_i_PTR, $OUT_i_PTR, 2
  shl        $IN_i_PTR, $IN_i_PTR, 2
  shl        $NUM_PART_PTR, $NUM_PART_PTR, 2
  add        $OUT_i_PTR, $OUT_i_PTR, $SCRATCH
  add        $IN_i_PTR, $IN_i_PTR, $SCRATCH
  add        $NUM_PART_PTR, $NUM_PART_PTR, $SCRATCH

// ************************************************* //
// Extract size and bases
// ************************************************* //
  shr        $OUT_i_SIZE, $OUT_BASE, DELTAN_SIZE_OFF
  shl        $OUT_BASE, $OUT_BASE, DELTAN_SIZE_CLR
  shr        $OUT_BASE, $OUT_BASE, DELTAN_SIZE_CLR
  shl        $IN_BASE, $IN_BASE, DELTAN_SIZE_CLR
  shr        $IN_BASE, $IN_BASE, DELTAN_SIZE_CLR

// ************************************************* //
// Start loops, store in_i_ptr as rest every loop
// ************************************************* //
  st32       $IN_i_PTR, $mworker_base, $mzero, 1
  st32       $OUT_BASE, $mworker_base, $mzero, 2
  st32       $NUM_PART_PTR, $mworker_base, $mzero, 3
  st32       $OUT_i_PTR, $mworker_base, $mzero, 4
  brnzdec    $OUT_i_SIZE, _skip_e.LABEL_SUFFIX
  bri        _exit.LABEL_SUFFIX
  _skip_e.LABEL_SUFFIX:

_loop_over_reductions.LABEL_SUFFIX:
// ************************************************* //
// unpack offset and size
// ************************************************* //
  ld32       $OUT_BASE, $mworker_base, $mzero, 2
  ld32       $OUT_i_PTR, $mworker_base, $mzero, 4
  ld32step   $OUT_j_PTR, $mzero, $OUT_i_PTR+=, 1
  st32       $OUT_i_PTR, $mworker_base, $mzero, 4
  shr        $OUT_j_SIZE, $OUT_j_PTR, SCPTR_SIZE_OFF
  shl        $OUT_j_PTR, $OUT_j_PTR, SCPTR_SIZE_CLR
  shr        $OUT_j_PTR, $OUT_j_PTR, SCPTR_SIZE_CLR
  add        $OUT_j_PTR, $OUT_j_PTR, $OUT_BASE
  setzi      $IN_j_DELTA, 0

// ************************************************* //
// going to do 8 at a time and store remainder to memory
// ************************************************* //

  ld32       $NUM_PART_PTR, $mworker_base, $mzero, 3
  ldz16step  $NUM_PART, $mzero, $NUM_PART_PTR+=, 1
  st32       $NUM_PART_PTR, $mworker_base, $mzero, 3
  brnzdec    $NUM_PART, _skip3.LABEL_SUFFIX

  ldconst    $ACC_0, INIT_FLOAT
  rpt        $OUT_j_SIZE, 0
  {st32step  $ACC_0, $mzero, $OUT_j_PTR+=, 1; fnop}

  _skip3.LABEL_SUFFIX:
  st32       $NUM_PART, $mzero, $mworker_base, 5

  and        $SCRATCH, $OUT_j_SIZE, 0x1
  st32       $SCRATCH, $mworker_base, $mzero, 0
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 1
  setzi      $IN_j_DELTA, 0

  brnzdec    $OUT_j_SIZE, _skip2.LABEL_SUFFIX
  bri        _out_j_size_remainder.LABEL_SUFFIX
  _skip2.LABEL_SUFFIX:

_out_j_loop.LABEL_SUFFIX:

  {
    ld32       $IN_i_PTR, $mworker_base, $mzero, 1
    setzi      $ACC_0, INIT_FLOAT & LDCONST_MASK
  }
  {
    ld32       $NUM_PART, $mzero, $mworker_base, 5
    or         $ACC_0, $ACC_0, INIT_FLOAT & ~LDCONST_MASK
  }
  mov        $ACC_1, $ACC_0


// ************************************************* //
// Loop over iputs accumulating
// ************************************************* //
_start_num_partials_loop.LABEL_SUFFIX:
  ld32step   $IN_j_PTR, $mzero, $IN_i_PTR+=, 1
  shr        $IN_j_SIZE, $IN_j_PTR, 18
  shl        $IN_j_PTR, $IN_j_PTR, 14
  shr        $IN_j_PTR, $IN_j_PTR, 14
  add        $IN_j_PTR, $IN_BASE, $IN_j_PTR

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE
  mov        $SCRATCH, $IN_j_DELTA

_in_j_loop_start.LABEL_SUFFIX:
  ld64_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V2 $ACC_0:3, $ACC_0:3, $VALUES_0:1
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop.LABEL_SUFFIX

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //

  {
    add $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2mul   $VALUES_0:1, $SCALE:7, $ACC_0:3
  }
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.LABEL_SUFFIX

_out_j_size_remainder.LABEL_SUFFIX:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, 0

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.LABEL_SUFFIX

  {
    ld32       $IN_i_PTR, $mworker_base, $mzero, 1
    setzi      $ACC_0, INIT_FLOAT & LDCONST_MASK
  }
  {
    ld32       $NUM_PART, $mzero, $mworker_base, 5
    or         $ACC_0, $ACC_0, INIT_FLOAT & ~LDCONST_MASK
  }

_start_num_partials_loop_1_rem.LABEL_SUFFIX:
  ld32step   $IN_j_PTR, $mzero, $IN_i_PTR+=, 1
  shr        $IN_j_SIZE, $IN_j_PTR, SCPTR_SIZE_OFF
  shl        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  shr        $IN_j_PTR, $IN_j_PTR, SCPTR_SIZE_CLR
  add        $IN_j_PTR, $IN_BASE, $IN_j_PTR

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE
  mov        $SCRATCH, $IN_j_DELTA

_in_j_loop_start_1_rem.LABEL_SUFFIX:
  ld32_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V2   $ACC_0:3, $ACC_0:3, $VALUES_0:1
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.LABEL_SUFFIX

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //

  {
    add $IN_j_DELTA, $IN_j_DELTA, 2
    f32v2mul   $VALUES_0:1, $SCALE:7, $ACC_0:3
  }
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

_out_j_size_end.LABEL_SUFFIX:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, 1
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.LABEL_SUFFIX
// ************************************************* //
// End of loops
// ************************************************* //
_exit.LABEL_SUFFIX:
  exitz      $mzero

.size REDUCE_FLOAT_FLOAT,\
              .-REDUCE_FLOAT_FLOAT


#endif
