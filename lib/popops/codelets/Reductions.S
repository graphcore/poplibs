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


// all scratch offsets given in words
#define REM_SCRATCH      0
#define IN_PTR_SCRATCH   1
#define BASE_SCRATCH     2
#define NP_PTR_SCRATCH   3
#define OUT_PTR_SCRATCH  4
#define NP_SCRATCH       5
#define OUT_j_SIZE_SCRATCH  6


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
#define ASCRATCH_0      a5
#define ZAACC           a4
#define SCALE           a6
#define SCALE2          a7

// ld macros populate the arf (VALUES_0:4) with partial data that
// will be used as the input to the accumulation instuction

// ------------------------------------------------------- //
// Macro to load 64 bits when either 16 bit aligned or 32 bit aligned
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

#define IS_UPDATE @IS_UPDATE@

#if IS_UPDATE
#define UPD true
#define UPDATE_INSTR(scale_instr, ld_instr, acc_instr) \
{ld_instr , $OUT_j_PTR, $mzero, 0; \
 scale_instr}; \
acc_instr $VALUES_0:1, $VALUES_0:1, $VALUES_2:3
#else
#define UPD false
#define UPDATE_INSTR(scale_instr, ld_instr, acc_instr) scale_instr
#endif

// ------------------------------------------------------- //

#define UNPAREN(...) __VA_ARGS__
#define OP @ReduceOp@
#define INSTRUCTION @INSTRUCTION@

#define F16V8(instr) F16V8_(instr)
#define F16V8_(instr) f16v8##instr
#define F16V4(instr) F16V4_(instr)
#define F16V4_(instr) f16v4##instr
#define F32V4(instr) F32V4_(instr)
#define F32V4_(instr) f32v4##instr

#define LABEL_SUFFIX _h_f_
#define SIZE_OF_IN_TYPE 2

#define MANGLE(prefix, op, ptype, otype, specialisation) \
            _MANGLE_(prefix, op, ptype, otype, UPD, specialisation)
#define _MANGLE_(prefix, op, ptype, otype, update, specialisation) \
            MANGLE_(prefix, op, ptype, otype, update, specialisation)
#define MANGLE_(prefix, op, ptype, otype, update, specialisation) \
__runCodelet_popops__##prefix##___popops__##op##_##ptype##_##otype##_##update##_##specialisation


#define REDUCE_HALF_FLOAT(prefix, specialisation) MANGLE(prefix, OP, half, float, specialisation)
#define REDUCE_HALF_HALF(prefix, specialisation) MANGLE(prefix, OP, half, half, specialisation)
#define REDUCE_FLOAT_FLOAT(prefix, specialisation) MANGLE(prefix, OP, float, float, specialisation)
#define REDUCE_FLOAT_HALF(prefix, specialisation) MANGLE(prefix, OP, float, half, specialisation)

#define FLOAT_1_0 0x3f800000

.type REDUCE_HALF_FLOAT(Reduce,common), @function

.section .text.REDUCE_HALF_FLOAT(Reduce,common), "ax"

// Instantiate two variants which call the same common function
.globl REDUCE_HALF_FLOAT(Reduce,0)
.type REDUCE_HALF_FLOAT(Reduce,0), @function
.globl REDUCE_HALF_FLOAT(ScaledReduce,0)
.type REDUCE_HALF_FLOAT(ScaledReduce,0), @function
.globl REDUCE_HALF_FLOAT(Reduce,1)
.type REDUCE_HALF_FLOAT(Reduce,1), @function
.globl REDUCE_HALF_FLOAT(ScaledReduce,1)
.type REDUCE_HALF_FLOAT(ScaledReduce,1), @function

// ************************************************* //
// Load vertex state
// ************************************************* //
.align 4
REDUCE_HALF_FLOAT(Reduce,common):
REDUCE_HALF_FLOAT(Reduce,0):
REDUCE_HALF_FLOAT(Reduce,1):
{
  bri        1f
  or      $SCALE, $azero, FLOAT_1_0
}
REDUCE_HALF_FLOAT(ScaledReduce,0):
REDUCE_HALF_FLOAT(ScaledReduce,1):
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
  and        $SCRATCH, $OUT_j_SIZE, 0x7
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 3

  brnzdec    $OUT_j_SIZE, _skip2.LABEL_SUFFIX
  bri        _out_j_size_remainder.LABEL_SUFFIX
_skip2.LABEL_SUFFIX:

_out_j_loop.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_zero_and_load

// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
  st32      $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

_start_num_partials_loop.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start.LABEL_SUFFIX:
  call      $OUT_j_SIZE, _Reduce_ld128_MIS_2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION) $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop.LABEL_SUFFIX

  ld32      $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

// ************************************************* //
// end of 8 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 16
    f32v2gina  $VALUES_0:1, $azeros, 0
  }

  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
              ld64 $VALUES_2:3, f32v2add)
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
              ld64 $VALUES_2:3, f32v2add)
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
              ld64 $VALUES_2:3, f32v2add)
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
              ld64 $VALUES_2:3, f32v2add)
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.LABEL_SUFFIX

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.LABEL_SUFFIX:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH
  and        $SCRATCH, $OUT_j_SIZE, 4
  brz        $SCRATCH, _out_j_2_remainder.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_4_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start_4_rem.LABEL_SUFFIX:
  ld64_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_4_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_4_rem.LABEL_SUFFIX
// ************************************************* //
// end of 4 vector accumulating, scale and store
// ************************************************* //
  {
    add      $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
              ld64 $VALUES_2:3, f32v2add)
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
              ld64 $VALUES_2:3, f32v2add)
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1


// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_2_rem.LABEL_SUFFIX:
    call       $SCRATCH2, _Reduce_ptr_fetch
    mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.LABEL_SUFFIX:
  ld32_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_2_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.LABEL_SUFFIX

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 4
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld64 $VALUES_2:3, f32v2add)
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_1_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start_1_rem.LABEL_SUFFIX:
  ldb16      $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.LABEL_SUFFIX

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 2
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld32 $VALUES_2, f32v2add)
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

_out_j_size_end.LABEL_SUFFIX:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.LABEL_SUFFIX
// ************************************************* //
// End of loops
// ************************************************* //
_exit.LABEL_SUFFIX:
  exitz      $mzero

.size REDUCE_HALF_FLOAT(Reduce,common),\
            .-REDUCE_HALF_FLOAT(Reduce,common)


// ------------------------------------------------------- //
// ------------------------------------------------------- //
// HALf HALF
// ------------------------------------------------------- //
// ------------------------------------------------------- //

#undef LABEL_SUFFIX
#define LABEL_SUFFIX _h_h_
// Instantiate two variants which call the same common function
.globl REDUCE_HALF_HALF(Reduce,0)
.type REDUCE_HALF_HALF(Reduce,0), @function
.globl REDUCE_HALF_HALF(ScaledReduce,0)
.type REDUCE_HALF_HALF(ScaledReduce,0), @function
.globl REDUCE_HALF_HALF(Reduce,1)
.type REDUCE_HALF_HALF(Reduce,1), @function
.globl REDUCE_HALF_HALF(ScaledReduce,1)
.type REDUCE_HALF_HALF(ScaledReduce,1), @function

.type REDUCE_HALF_HALF(Reduce,common), @function

.section .text.REDUCE_HALF_HALF(Reduce,common), "ax"
.align 4
// ************************************************* //
// Load vertex state
// ************************************************* //
REDUCE_HALF_HALF(Reduce,0):
REDUCE_HALF_HALF(Reduce,1):
REDUCE_HALF_HALF(Reduce,common):
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
  and        $SCRATCH, $OUT_j_SIZE, 0x7
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 3

  brnzdec    $OUT_j_SIZE, _skip2.LABEL_SUFFIX
  bri        _out_j_size_remainder.LABEL_SUFFIX
  _skip2.LABEL_SUFFIX:

_out_j_loop.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_zero_and_load


// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
  st32      $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

_start_num_partials_loop.LABEL_SUFFIX:
   call       $SCRATCH2, _Reduce_ptr_fetch
   mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start.LABEL_SUFFIX:
  call      $OUT_j_SIZE, _Reduce_ld128_MIS_2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop.LABEL_SUFFIX

  ld32       $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

// ************************************************* //
// end of 8 vector accumulating, scale and store
// ************************************************* //
  // instead of f16v4mul could have done f16v4mac, for those that are bored of
  // the regular instruction set
  // TODO use f16v4gacc here as uput later removes need for gina
  {
    add $IN_j_DELTA, $IN_j_DELTA, 16
    f16v2gina  $VALUES_0, $azero, 0
  }
  f16v2gina $VALUES_1, $azero, 0
  UPDATE_INSTR(UNPAREN(f16v4mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld64 $VALUES_2:3, f16v4add)
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f16v2gina  $VALUES_0, $azero, 0
  }
  f16v2gina $VALUES_1, $azero, 0
  UPDATE_INSTR(UNPAREN(f16v4mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld64 $VALUES_2:3, f16v4add)
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.LABEL_SUFFIX

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.LABEL_SUFFIX:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH
  and        $SCRATCH, $OUT_j_SIZE, 4
  brz        $SCRATCH, _out_j_2_remainder.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_4_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_4_rem.LABEL_SUFFIX:
  ld64_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_4_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_4_rem.LABEL_SUFFIX
// ************************************************* //
// end of 4 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 8
    f16v2gina  $VALUES_0, $azero, 0
  }
  f16v2gina $VALUES_1, $azero, 0
  UPDATE_INSTR(UNPAREN(f16v4mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld64 $VALUES_2:3, f16v4add)

  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_2_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.LABEL_SUFFIX:
  ld32_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_2_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.LABEL_SUFFIX

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 4
    f16v2gina  $VALUES_0, $azero, 0
  }
  UPDATE_INSTR(UNPAREN(f16v4mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                          ld32 $VALUES_2, f16v4add)
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_1_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start_1_rem.LABEL_SUFFIX:
  ldb16      $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F16V8(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.LABEL_SUFFIX

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 2
    f16v2gina  $VALUES_0, $azero, 0
  }
  UPDATE_INSTR(UNPAREN(f16v4mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                          ldb16 $VALUES_2, f16v4add)
  ldb16 $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
  sort4x16lo $VALUES_0, $VALUES_0, $ASCRATCH_0
  st32 $VALUES_0, $OUT_j_PTR, $mzero, 0

_out_j_size_end.LABEL_SUFFIX:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.LABEL_SUFFIX
// ************************************************* //
// End of loops
// ************************************************* //
_exit.LABEL_SUFFIX:
  exitz      $mzero

.size REDUCE_HALF_HALF(Reduce,common),\
              .-REDUCE_HALF_HALF(Reduce,common)


// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Float
// -------------------------------------------------------- //
// -------------------------------------------------------- //


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
  call       $IN_j_SIZE, _Reduce_outer_loop_setup
  and        $SCRATCH, $OUT_j_SIZE, 0x3
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 2

  brnzdec    $OUT_j_SIZE, _skip2.LABEL_SUFFIX
  bri        _out_j_size_remainder.LABEL_SUFFIX
  _skip2.LABEL_SUFFIX:

_out_j_loop.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_zero_and_load

// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
_start_num_partials_loop.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start.LABEL_SUFFIX:
  // As we don't assume alignment here, it's better to load 4x32 bits
  // than conditionally load 2x64 bits or revert to loading 4x32 bits based
  // on checking alignment
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  ld32 $VALUES_2, $IN_j_PTR, $SCRATCH, 2
  ld32 $VALUES_3, $IN_j_PTR, $SCRATCH, 3
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V4(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop.LABEL_SUFFIX

// ************************************************* //
// end of 8 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 16
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld64 $VALUES_2:3, f32v2add)
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld64 $VALUES_2:3, f32v2add)
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.LABEL_SUFFIX

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.LABEL_SUFFIX:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH

// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_2_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.LABEL_SUFFIX:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V4(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_2_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.LABEL_SUFFIX

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld64 $VALUES_2:3, f32v2add)

  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load

_start_num_partials_loop_1_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_1_rem.LABEL_SUFFIX:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V4(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.LABEL_SUFFIX
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.LABEL_SUFFIX

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 4
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  UPDATE_INSTR(UNPAREN(f32v2mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                ld32 $VALUES_2, f32v2add)
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

_out_j_size_end.LABEL_SUFFIX:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.LABEL_SUFFIX
// ************************************************* //
// End of loops
// ************************************************* //
_exit.LABEL_SUFFIX:
  exitz      $mzero

.size REDUCE_FLOAT_FLOAT(Reduce,common),\
              .-REDUCE_FLOAT_FLOAT(Reduce,common)


// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Half
// -------------------------------------------------------- //
// -------------------------------------------------------- //



#undef LABEL_SUFFIX
#define LABEL_SUFFIX _f_h_

.globl REDUCE_FLOAT_HALF(Reduce,common)
.type REDUCE_FLOAT_HALF(Reduce,common), @function

.section .text.REDUCE_FLOAT_HALF(Reduce,common), "ax"

// Instantiate two variants which call the same common function
.globl REDUCE_FLOAT_HALF(Reduce,0)
.type REDUCE_FLOAT_HALF(Reduce,0), @function
.globl REDUCE_FLOAT_HALF(ScaledReduce,0)
.type REDUCE_FLOAT_HALF(ScaledReduce,0), @function
.globl REDUCE_FLOAT_HALF(Reduce,1)
.type REDUCE_FLOAT_HALF(Reduce,1), @function
.globl REDUCE_FLOAT_HALF(ScaledReduce,1)
.type REDUCE_FLOAT_HALF(ScaledReduce,1), @function

.align 4
// ************************************************* //
// Load vertex state
// ************************************************* //
REDUCE_FLOAT_HALF(Reduce,common):
REDUCE_FLOAT_HALF(Reduce,0):
REDUCE_FLOAT_HALF(Reduce,1):
{
  bri        1f
  or      $SCALE, $azero, FLOAT_1_0
}
REDUCE_FLOAT_HALF(ScaledReduce,0):
REDUCE_FLOAT_HALF(ScaledReduce,1):
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
  and        $SCRATCH, $OUT_j_SIZE, 0x3
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 2

  brnzdec    $OUT_j_SIZE, _skip2.LABEL_SUFFIX
  bri        _out_j_size_remainder.LABEL_SUFFIX
  _skip2.LABEL_SUFFIX:

_out_j_loop.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_zero_and_load

// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
_start_num_partials_loop.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start.LABEL_SUFFIX:
  // As we don't assume alignment here, it's better to load 4x32 bits
  // than conditionally load 2x64 bits or revert to loading 4x32 bits based
  // on checking alignment
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  ld32 $VALUES_2, $IN_j_PTR, $SCRATCH, 2
  ld32 $VALUES_3, $IN_j_PTR, $SCRATCH, 3
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V4(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  {
    brnz       $SCRATCH2, _in_j_loop_start.LABEL_SUFFIX
    // move ACC0 and ACC2 to VALUES0_1 and shift the remaining results to
    // the odd accumulators. This does not shift the accumulator delay line.
    f16v4stacc $VALUES_0:1, 0
  }
  {
    brnzdec   $NUM_PART, _start_num_partials_loop.LABEL_SUFFIX
    f32v2gina $VALUES_2:3, $azeros, 1
  }

// ************************************************* //
// end of 8 vector accumulating, scale and store
// ************************************************* //
  {
    add       $IN_j_DELTA, $IN_j_DELTA, 16
    // scale at higher precision
    f32v2mul $VALUES_0:1, $SCALE:B, $VALUES_0:1
  }

  f32v2mul $VALUES_2:3, $SCALE:B, $VALUES_2:3
#if IS_UPDATE == 1
  {
    // load value to update
    ld64       $VALUES_2:3, $OUT_j_PTR, $mzero, 0
    f32v4tof16 $VALUES_0:1, $VALUES_0:3
  }
  f16v4add $VALUES_0:1, $VALUES_0:1, $VALUES_2:3
#else
  f32v4tof16 $VALUES_0:1, $VALUES_0:3
#endif

  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    // The accumulator pipeline is still dirty and needs to be cleared for
    // next pass.
    uput       $FP_CLR, $ZAACC
  }

  brnzdec    $OUT_j_SIZE, _out_j_loop.LABEL_SUFFIX

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.LABEL_SUFFIX:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH

// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.LABEL_SUFFIX

  call       $SCRATCH2, _Reduce_zero_and_load

_start_num_partials_loop_2_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.LABEL_SUFFIX:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V4(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  {
    brnz       $SCRATCH2, _in_j_loop_start_2_rem.LABEL_SUFFIX
    // move ACC0 and ACC2 to VALUES0_1 and shift the remaining results to
    // the odd accumulators. This does not shift the accumulator delay line.
    f16v4stacc $VALUES_0:1, 0
  }
  {
    brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.LABEL_SUFFIX
    // scale up in higher precision
    f32v2mul $VALUES_0:1, $SCALE:B, $VALUES_0:1
  }
// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //

#if IS_UPDATE == 1
  {
    ld32       $VALUES_2, $OUT_j_PTR, $mzero, 0
    f32v2tof16 $VALUES_0, $VALUES_0:1
  }
  {
    add        $IN_j_DELTA, $IN_j_DELTA, 8
    f16v2add   $VALUES_0, $VALUES_0, $VALUES_2
  }
#else
  {
    add        $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2tof16 $VALUES_0, $VALUES_0:1
  }
#endif
  {
    st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1
    // The accumulator state is dirty and must be cleared for the next pass
    uput       $FP_CLR, $ZAACC
  }

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.LABEL_SUFFIX:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.LABEL_SUFFIX
  call       $SCRATCH2, _Reduce_zero_and_load

_start_num_partials_loop_1_rem.LABEL_SUFFIX:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_1_rem.LABEL_SUFFIX:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    F32V4(INSTRUCTION)   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.LABEL_SUFFIX
  {
    brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.LABEL_SUFFIX
    // move ACC0 and ACC2 to VALUES0_1 and shift the remaining results to
    // the odd accumulators. This does not shift the accumulator delay line.
    f16v4stacc $VALUES_0:1, 0
  }

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //
  {
    add       $IN_j_DELTA, $IN_j_DELTA, 4
    f32v2mul  $VALUES_0:1, $SCALE:B, $VALUES_0:1
  }
#if IS_UPDATE
  {
    ldb16      $VALUES_2, $OUT_j_PTR, $mzero, 0
    f32v2tof16 $VALUES_0, $VALUES_0:1
  }
  {
    ldb16    $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
    f16v2add $VALUES_0, $VALUES_0, $VALUES_2
  }
#else
  {
    ldb16       $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
    f32v2tof16  $VALUES_0, $VALUES_0:1
  }
#endif

  sort4x16lo $VALUES_0, $VALUES_0, $ASCRATCH_0
  {
    st32       $VALUES_0, $OUT_j_PTR, $mzero, 0
    // Clear accumulators for the next pass as they are dirty
    uput       $FP_CLR, $ZAACC
  }

_out_j_size_end.LABEL_SUFFIX:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.LABEL_SUFFIX

// ************************************************* //
// End of loops
// ************************************************* //
_exit.LABEL_SUFFIX:
  exitz      $mzero

.size REDUCE_FLOAT_HALF(Reduce,common),\
            .-REDUCE_FLOAT_HALF(Reduce,common)

#if IS_UPDATE == 0 // updating versions of these vertices aren't implemented
.type REDUCE_FLOAT_FLOAT(Reduce,2common), @function
.section .text.REDUCE_FLOAT_FLOAT(Reduce,2common), "ax"
.align 8
// accumulate values from a single edge
// This code is only used for acc and sqadd, so it's OK to accumulate extra
// zeros
REDUCE_FLOAT_FLOAT(Reduce,2common):
  //$m0: dest offset from MEMBASE in f32's
  //$m1: src offset from MEMBASE in bytes
  //$a0:3 accumulators
  //$a4 preamble/remainders
  //load *src, *dest, n
  ld32 $m0, $mvertex_base, $m15, 0 // load output pointer
  ldz16 $m1, $mvertex_base, $m15, 2
  shl $m1, $m1, 2 //
  {
  ldz16 $m2, $mvertex_base, $m15, 3
  f32v2add $a2:3, $azeros, $azeros
  }
  {
  setzi $SCRATCH, TMEM_REGION0_BASE_ADDR
  f32v2add $a0:1, $azeros, $azeros
  }
  {
  and   $m3, $m1, 0x7 //m3=nonzero if dst is not 64bit aligned
  setzi $ZAACC, ZAACC_BITMASK
  }
  {
  brz $m0, 9f
  uput  $FP_CLR, $ZAACC
  }
  brz  $m3, 1f
  // consume leading 64bit-misaligned value to give 64bit alignment
  ld32step $a0, $SCRATCH, $m1+=, 1
  add   $m2, $m2, -1
1: // m1 now 64bit aligned
  shr   $m3, $m2, 2
1:// outer loop when rpt count is too small
  // consume quads of aligned elements
  min   $m4, $m3, CSR_W_REPEAT_COUNT__VALUE__MASK
  sub   $m3, $m3, $m4
  rpt   $m4, (3f-2f)/8-1
2:
    {
    ld64step $a2:3, $SCRATCH, $m1+=, 1
    F32V4(INSTRUCTION) $a0:3
    }
	  {
    ld64step $a0:1, $SCRATCH, $m1+=, 1
    fnop
    }
3:
  brnz $m3, 1b
  {
  and $m2, $m2, 0x3 // remainder
  F32V4(INSTRUCTION) $a0:3
  }
  // consume remainder - 0-3 floats possible
  // arrive here by normal fall-through with partial sum in $a0:3, or in the
  // <5 entries case in which case $a0:3 are all zero
  mov $a1, $azero
  // all further accumulation is into the first accumulator
  {
  ld32step $a0, $SCRATCH, $m1+=, 1
  f32v2gina $a4:5, $azeros, 0
  }
  {
  rpt $m2, (2f-1f)/8 - 1
  f32v2add $a2:3, $azeros, $azeros
  }
1:
    {
    ld32step $a0, $SCRATCH, $m1+=, 1
    F32V4(INSTRUCTION) $a0:3
    }
2:
  f32v2gina $a0:1, $azeros, 0
9:
  f32v2add $a0:1, $a0:1, $a4:5
  {
  br $m10
  f32add $a0, $a0, $a1
  }
.size REDUCE_FLOAT_FLOAT(Reduce,2common), .-REDUCE_FLOAT_FLOAT(Reduce,2common)

.globl REDUCE_FLOAT_FLOAT(Reduce,2)
.type REDUCE_FLOAT_FLOAT(Reduce,2), @function
.section .text.REDUCE_FLOAT_FLOAT(Reduce,2), "ax"
.align 4
REDUCE_FLOAT_FLOAT(Reduce,2):
  call $m10, REDUCE_FLOAT_FLOAT(Reduce,2common)
  st32 $a0, $m0, $mzero
	exitnz $mzero
.size REDUCE_FLOAT_FLOAT(Reduce,2), .-REDUCE_FLOAT_FLOAT(Reduce,2)

.globl REDUCE_FLOAT_HALF(Reduce,2)
.type REDUCE_FLOAT_HALF(Reduce,2), @function
.section .text.REDUCE_FLOAT_HALF(Reduce,2), "ax"
.align 4
REDUCE_FLOAT_HALF(Reduce,2):
  call $m10, REDUCE_FLOAT_FLOAT(Reduce,2common)
  and $m1, $m0, 0x2
  andc $m0, $m0, 0x2
  {
  ld32 $a1, $m0, $mzero
  f32tof16 $a0, $a0
  }
  {
  brz $m1, 1f
  sort4x16hi $a2, $a0, $a1 // a2 = a1[16:31] | a0 [16:31]
  }
  sort4x16lo $a2, $a1, $a0 // a3 = a0[0:15]  | a1[0:15]
1:
  st32 $a2, $m0, $mzero
	exitnz $mzero
.size REDUCE_FLOAT_HALF(Reduce,2), .-REDUCE_FLOAT_HALF(Reduce,2)

.globl REDUCE_FLOAT_FLOAT(Reduce,3)
.type REDUCE_FLOAT_FLOAT(Reduce,3), @function
.section .text.REDUCE_FLOAT_FLOAT(Reduce,3), "ax"
.align 8
// Accumulate values from a single edge
// output must be 32bit aligned
// partials must be 64bit aligned
// numOutputs * sizeof(float) must be a multiple of 8bytes (else there will be
// a misload of partials
// Only half the accumulator width is used to simplify alignment constraints
REDUCE_FLOAT_FLOAT(Reduce,3):
#define LOG2_PARTIAL_SIZE 2 // 4 bytes per float
  ldz16 $m0, $mvertex_base, $mzero, 0 // load output pointer
  ldz16 $m1, $mvertex_base, $mzero, 1 // load partials pointer
  ldz16 $m2, $mvertex_base, $mzero, 2 // load numOutputs
  ldz16 $m3, $mvertex_base, $mzero, 3 // load numPartials
  // keep $m0 and $m1 as byte offsets, using $SCRATCH to hold memory base for
  // offset addressing below
 {setzi $SCRATCH, TMEM_REGION0_BASE_ADDR
  setzi $ZAACC, ZAACC_BITMASK}
  shl $m0, $m0, 2
  shl $m1, $m1, 2
  // $m4 = numOutputs/4; 2 elements per loop for float partials
  and $m4, $m2, 1<<(3-LOG2_PARTIAL_SIZE)-1
  brz $m4, 9f // branch if outputs a multiple of 8 bytes
  ld32 $m0, $mzero, $mzero, 0 // issue a null address read to stop
9:
  {shr $m4, $m2, 3-LOG2_PARTIAL_SIZE
  f32v2add $a2:3, $azeros, $azeros}
  // $m6 holds the rewind in bytes from the final partial back to the first for
  // the next output loop. Note the core loop makes an extra stride
  add $m6, $m3, 1
  mul $m6, $m6, $m2
  shl $m6, $m6, LOG2_PARTIAL_SIZE // float partials
  sub $m6, $m6, 8
  // $m7 = 64bit offset between consecutive partials for the same output
 {shr $m7, $m2, 3-LOG2_PARTIAL_SIZE
  uput  $FP_CLR, $ZAACC}
  bri 4f
1:
  // $m1 points to the first partial to be accumulated for this output
  // $m7 is the step between consecutive partials for the same output
  ld64step $a0:1, $SCRATCH, $m1+=, $m7
  rpt $m3, (3f-2f)/8-1
  // last pass will overread 8 bytes
2:
 {ld64step $a0:1, $SCRATCH, $m1+=, $m7
  F32V4(INSTRUCTION) $a0:3}
3:
 {sub $m1, $m1, $m6 // advance to first partial for the next output
  f32v2gina $a0:1, $azeros, 0}
  st32step $a0, $SCRATCH, $m0+=, 1
  st32step $a1, $SCRATCH, $m0+=, 1
4:
  brnzdec $m4, 1b
  exitnz $mzero
.size REDUCE_FLOAT_FLOAT(Reduce,3), .-REDUCE_FLOAT_FLOAT(Reduce,3)

.globl REDUCE_HALF_FLOAT(Reduce,3)
.type REDUCE_HALF_FLOAT(Reduce,3), @function
.section .text.REDUCE_HALF_FLOAT(Reduce,3), "ax"
.align 8
// Accumulate values from a single edge
// This is the same as the FLOAT_FLOAT version except for the f16 accumulation
// and output of twice as many outputs per loop
// output must be 32bit aligned
// partials must be 64bit aligned
// numOutputs * sizeof(float) must be a multiple of 8bytes (else there will be
// a misload of partials
// Only half the accumulator width is used to simplify alignment constraints
REDUCE_HALF_FLOAT(Reduce,3):
#undef LOG2_PARTIAL_SIZE
#define LOG2_PARTIAL_SIZE 1 // 2 bytes per half
  ldz16 $m0, $mvertex_base, $mzero, 0 // load output pointer
  ldz16 $m1, $mvertex_base, $mzero, 1 // load partials pointer
  ldz16 $m2, $mvertex_base, $mzero, 2 // load numOutputs
  ldz16 $m3, $mvertex_base, $mzero, 3 // load numPartials
  // keep $m0 and $m1 as byte offsets, using $SCRATCH to hold memory base for
  // offset addressing below
 {setzi $SCRATCH, TMEM_REGION0_BASE_ADDR
  setzi $ZAACC, ZAACC_BITMASK}
  shl $m0, $m0, 2
  shl $m1, $m1, 2
  // $m4 = numOutputs/4; 2 elements per loop for float partials
  and $m4, $m2, 1<<(3-LOG2_PARTIAL_SIZE)-1
  brz $m4, 9f // branch if outputs a multiple of 8 bytes
  ld32 $m0, $mzero, $mzero, 0 // issue a null address read to stop
9:
  {shr $m4, $m2, 3-LOG2_PARTIAL_SIZE
  f16v4add $a2:3, $azeros, $azeros}
  // $m6 holds the rewind in bytes from the final partial back to the first for
  // the next output loop. Note the core loop makes an extra stride
  add $m6, $m3, 1
  mul $m6, $m6, $m2
  shl $m6, $m6, LOG2_PARTIAL_SIZE // half partials
  sub $m6, $m6, 8
  // $m7 = 64bit offset between consecutive partials for the same output
 {shr $m7, $m2, 3-LOG2_PARTIAL_SIZE
  uput  $FP_CLR, $ZAACC}
  bri 4f
1:
  // $m1 points to the first partial to be accumulated for this output
  // $m7 is the step between consecutive partials for the same output
  ld64step $a0:1, $SCRATCH, $m1+=, $m7
  rpt $m3, (3f-2f)/8-1
  // last pass will overread 8 bytes
  // There is no f16v4sqadd, so use f16v8 for all types
2:
 {ld64step $a0:1, $SCRATCH, $m1+=, $m7
  F16V8(INSTRUCTION) $a0:3}
3:
 {sub $m1, $m1, $m6 // advance to first partial for the next output
  f32v2gina $a0:1, $azeros, 0}
  st32step $a0, $SCRATCH, $m0+=, 1
 {st32step $a1, $SCRATCH, $m0+=, 1
  f32v2gina $a0:1, $azeros, 0}
  st32step $a0, $SCRATCH, $m0+=, 1
  st32step $a1, $SCRATCH, $m0+=, 1
4:
  brnzdec $m4, 1b
  exitnz $mzero
.size REDUCE_HALF_FLOAT(Reduce,3), .-REDUCE_HALF_FLOAT(Reduce,3)

#endif

#endif
