#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts.h"

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

#define MANGLE(op, ptype, otype) \
            _MANGLE_(op, ptype, otype, UPD)
#define _MANGLE_(op, ptype, otype, update) \
            MANGLE_(op, ptype, otype, update)
#define MANGLE_(op, ptype, otype, update) \
__runCodelet_popops__Reduce___popops__##op##_##ptype##_##otype##_##update


#define REDUCE_HALF_FLOAT MANGLE(OP, half, float)
#define REDUCE_HALF_HALF MANGLE(OP, half, half)
#define REDUCE_FLOAT_FLOAT MANGLE(OP, float, float)
#define REDUCE_FLOAT_HALF MANGLE(OP, float, half)

.globl REDUCE_HALF_FLOAT
.type REDUCE_HALF_FLOAT, @function

.section .text.REDUCE_HALF_FLOAT, "ax"
.align 4
REDUCE_HALF_FLOAT:
// ************************************************* //
// Load vertex state
// ************************************************* //
  ld32       $SCALE, $mvertex_base, $mzero, SCALE_OFF/4
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
  {call       $SCRATCH2, _Reduce_zero_and_load
  zero       $VALUES_2:3}

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

.size REDUCE_HALF_FLOAT,\
            .-REDUCE_HALF_FLOAT





// ------------------------------------------------------- //
// ------------------------------------------------------- //
// HALf HALF
// ------------------------------------------------------- //
// ------------------------------------------------------- //






#undef LABEL_SUFFIX
#define LABEL_SUFFIX _h_h_

.globl REDUCE_HALF_HALF
.type REDUCE_HALF_HALF, @function

.section .text.REDUCE_HALF_HALF, "ax"
.align 4
REDUCE_HALF_HALF:
// ************************************************* //
// Load vertex state
// ************************************************* //
  ld32       $SCALE, $mvertex_base, $mzero, SCALE_OFF/4
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
  {call       $SCRATCH2, _Reduce_zero_and_load
  zero       $VALUES_2:3}


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

.size REDUCE_HALF_HALF,\
              .-REDUCE_HALF_HALF





// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Float
// -------------------------------------------------------- //
// -------------------------------------------------------- //



#undef LABEL_SUFFIX
#define LABEL_SUFFIX _f_f_

#undef SIZE_OF_IN_TYPE
#define SIZE_OF_IN_TYPE 4

.globl REDUCE_FLOAT_FLOAT
.type REDUCE_FLOAT_FLOAT, @function

.section .text.REDUCE_FLOAT_FLOAT, "ax"
.align 4
REDUCE_FLOAT_FLOAT:
// ************************************************* //
// Load vertex state
// ************************************************* //
  ld32       $SCALE, $mvertex_base, $mzero, SCALE_OFF/4
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
  {call       $SCRATCH2, _Reduce_zero_and_load
  zero       $VALUES_2:3}


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

.size REDUCE_FLOAT_FLOAT,\
              .-REDUCE_FLOAT_FLOAT





// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Half
// -------------------------------------------------------- //
// -------------------------------------------------------- //



#undef LABEL_SUFFIX
#define LABEL_SUFFIX _f_h_

.globl REDUCE_FLOAT_HALF
.type REDUCE_FLOAT_HALF, @function

.section .text.REDUCE_FLOAT_HALF, "ax"
.align 4
REDUCE_FLOAT_HALF:
// ************************************************* //
// Load vertex state
// ************************************************* //
  ld32       $SCALE, $mvertex_base, $mzero, SCALE_OFF/4
  {call       $IN_j_SIZE, _Reduce_load_state_process_common
  f32tof16   $SCALE, $SCALE}

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
  {call       $SCRATCH2, _Reduce_zero_and_load
  zero       $VALUES_2:3}


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
    f16v2gina  $VALUES_0, $azero, 0
  }
  UPDATE_INSTR(UNPAREN(f16v4mul   $VALUES_0:1, $SCALE:7, $VALUES_0:1),
                          ldb16 $VALUES_2, f16v4add)
  ldb16 $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
  sort4x16lo $VALUES_0, $VALUES_0, $ASCRATCH_0; \
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

.size REDUCE_FLOAT_HALF,\
            .-REDUCE_FLOAT_HALF

#endif
