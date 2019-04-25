#ifdef __IPU__


#include "colossus/tilearch.h"
#include "colossus/tileimplconsts.h"

#define OUT_OFF          0
#define OUT_OFFSET       4
#define IN_OFF           8
#define IN_OFFSET        12
#define NUM_PART_OFF     6
#define SCALE_OFF        14

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

#define FLOAT_1_0 0x3f800000

// ld macros populate the arf (VALUES_0:4) with partial data that
// will be used as the input to accumulation instructions. Scratch
// holds the offset from an 8 byte aligned ptr
// ------------------------------------------------------- //

// ------------------------------------------------------- //
  .macro ld64_MIS_2_
and $SCRATCH2, $SCRATCH, 0x3
brz $SCRATCH2, 1f
ldb16step $VALUES_0, $IN_j_PTR, $SCRATCH+=, 1
ld32step $ASCRATCH_0, $IN_j_PTR, $SCRATCH+=,1
{ldb16step $VALUES_1, $IN_j_PTR, $SCRATCH+=, -3
 roll16 $VALUES_0, $VALUES_0, $ASCRATCH_0};
{bri 2f; roll16 $VALUES_1, $ASCRATCH_0, $VALUES_1}
1:
ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
2:
  .endm

// ------------------------------------------------------- //
#define ld32_MIS_2_ _ld32_MIS_2(__COUNTER__)
#define _ld32_MIS_2(ID) ld32_MIS_2(ID)

#define ld32_MIS_2(ID) ldb16 $VALUES_0, $IN_j_PTR, $SCRATCH, 0; \
ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 1; \
roll16 $VALUES_0, $VALUES_0, $ASCRATCH_0;


// ------------------------------------------------------- //


#define MANGLE(prefix, op, ptype, otype, alignOut) \
            MANGLE_(prefix, op, ptype, otype, alignOut)
#define MANGLE_(prefix, op, ptype, otype, alignOut) \
__runCodelet_popops__##prefix##___popops__##op##_##ptype##_##otype##_##false##_##alignOut

#define OP @ReduceOp@
#define INIT_FLOAT @INIT_FLOAT@
#define INIT_HALF @INIT_HALF@

#define F16V4 f16v4@INSTRUCTION@
#define F32V2 f32v2@INSTRUCTION@
#define F32V1 f32@INSTRUCTION@

#define LABEL_SUFFIX _h_h_

#define SIZE_OF_IN_TYPE 2

#define REDUCE_HALF_HALF(prefix, alignOut) MANGLE(prefix, OP, half, half, alignOut)
#define REDUCE_FLOAT_FLOAT(prefix, alignOut) MANGLE(prefix, OP, float, float, alignOut)

.type REDUCE_HALF_HALF(Reduce,common), @function

.section .text.REDUCE_HALF_HALF(Reduce,common), "ax"
// Instantiate two variants which call the same common function
.globl REDUCE_HALF_HALF(Reduce,0)
.type REDUCE_HALF_HALF(Reduce,0), @function
.globl REDUCE_HALF_HALF(ScaledReduce,0)
.type REDUCE_HALF_HALF(ScaledReduce,0), @function
.globl REDUCE_HALF_HALF(Reduce,1)
.type REDUCE_HALF_HALF(Reduce,1), @function
.globl REDUCE_HALF_HALF(ScaledReduce,1)
.type REDUCE_HALF_HALF(ScaledReduce,1), @function

.align 4
// ************************************************* //
// Load vertex state
// ************************************************* //
REDUCE_HALF_HALF(Reduce,common):
REDUCE_HALF_HALF(Reduce,0):
REDUCE_HALF_HALF(Reduce,1):
{
  bri        1f
  or      $SCALE, $azero, FLOAT_1_0
}
REDUCE_HALF_HALF(ScaledReduce,0):
REDUCE_HALF_HALF(ScaledReduce,1):
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/2
  shl        $SCRATCH, $SCRATCH, 2
  setzi      $SCRATCH2, TMEM_REGION0_BASE_ADDR
  add        $SCRATCH, $SCRATCH, $SCRATCH2
  ld32       $SCALE, $SCRATCH, $mzero, 0

1:
   {call       $IN_j_SIZE, _Reduce_load_state_process_common
  f32tof16   $SCALE, $SCALE}

_loop_over_reductions.LABEL_SUFFIX:
// ************************************************* //
// unpack offset and size
// ************************************************* //
  call       $IN_j_SIZE, _Reduce_outer_loop_setup

  and        $SCRATCH, $OUT_j_SIZE, 0x3
  st32       $SCRATCH, $mworker_base, $mzero, 0
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 2

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
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

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
  call       $SCRATCH2, _Reduce_ptr_fetch
  {
    mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE
    // clear $VALUES_1 to avoid FP exceptions even though the result is unused
    setzi      $VALUES_1, 0
  }

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
  call       $SCRATCH2, _Reduce_ptr_fetch
  {
    mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE
    // clear $VALUES_1 to avoid FP exceptions even though the result is unused
    setzi      $VALUES_1, 0
  }

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

.size REDUCE_HALF_HALF(Reduce,common),\
              .-REDUCE_HALF_HALF(Reduce,common)


// ------------------------------------------------------- //
// ------------------------------------------------------- //
// Float Float
// ------------------------------------------------------- //
// ------------------------------------------------------- //

#undef LABEL_SUFFIX
#define LABEL_SUFFIX _f_f_

#undef SIZE_OF_IN_TYPE
#define SIZE_OF_IN_TYPE 4

.type REDUCE_FLOAT_FLOAT(Reduce,common), @function

.section .text.REDUCE_FLOAT_FLOAT(Reduce,common), "ax"
// Instantiate two variants which call the same common function
.globl REDUCE_FLOAT_FLOAT(Reduce,0)
.type REDUCE_FLOAT_FLOAT(Reduce,0), @function
.globl REDUCE_FLOAT_FLOAT(ScaledReduce,0)
.type REDUCE_FLOAT_FLOAT(ScaledReduce,0), @function
.globl REDUCE_FLOAT_FLOAT(Reduce,1)
.type REDUCE_FLOAT_FLOAT(Reduce,1), @function
.globl REDUCE_FLOAT_FLOAT(ScaledReduce,1)
.type REDUCE_FLOAT_FLOAT(ScaledReduce,1), @function

.align 4
// ************************************************* //
// Load vertex state
// ************************************************* //
REDUCE_FLOAT_FLOAT(Reduce,common):
REDUCE_FLOAT_FLOAT(Reduce,0):
REDUCE_FLOAT_FLOAT(Reduce,1):
{
  bri        1f
  or      $SCALE, $azero, FLOAT_1_0
}
REDUCE_FLOAT_FLOAT(ScaledReduce,0):
REDUCE_FLOAT_FLOAT(ScaledReduce,1):
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/2
  shl        $SCRATCH, $SCRATCH, 2
  setzi      $SCRATCH2, TMEM_REGION0_BASE_ADDR
  add        $SCRATCH, $SCRATCH, $SCRATCH2
  ld32       $SCALE, $SCRATCH, $mzero, 0
1:
  call       $IN_j_SIZE, _Reduce_load_state_process_common

_loop_over_reductions.LABEL_SUFFIX:
// ************************************************* //
// unpack offset and size
// ************************************************* //
  call       $IN_j_SIZE, _Reduce_outer_loop_setup

  and        $SCRATCH, $OUT_j_SIZE, 0x1
  st32       $SCRATCH, $mworker_base, $mzero, 0
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 1

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
  call       $SCRATCH2, _Reduce_ptr_fetch

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start.LABEL_SUFFIX:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
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
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_1_rem.LABEL_SUFFIX:
  ld32      $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V1   $ACC_0, $ACC_0, $VALUES_0
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

.size REDUCE_FLOAT_FLOAT(Reduce,common),\
              .-REDUCE_FLOAT_FLOAT(Reduce,common)


#endif
