#ifdef __IPU__

#include "colossus/tilearch.h"
#include "colossus/tileimplconsts.h"

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

#define MANGLE(op, ptype, otype, outAlign) \
            _MANGLE_(op, ptype, otype, UPD, outAlign)
#define _MANGLE_(op, ptype, otype, update, outAlign) \
            MANGLE_(op, ptype, otype, update, outAlign)
#define MANGLE_(op, ptype, otype, update, outAlign) \
__runCodelet_popops__Reduce___popops__##op##_##ptype##_##otype##_##update##_##outAlign


#define REDUCE_HALF_FLOAT(outAlign) MANGLE(OP, half, float, outAlign)
#define REDUCE_HALF_HALF(outAlign) MANGLE(OP, half, half, outAlign)
#define REDUCE_FLOAT_FLOAT(outAlign) MANGLE(OP, float, float, outAlign)
#define REDUCE_FLOAT_HALF(outAlign) MANGLE(OP, float, half, outAlign)


.type REDUCE_HALF_FLOAT(common), @function

.section .text.REDUCE_HALF_FLOAT(common), "ax"
.align 4
REDUCE_HALF_FLOAT(common):
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

.size REDUCE_HALF_FLOAT(common),\
            .-REDUCE_HALF_FLOAT(common)


// Instantiate two variants which call the same common function
.globl REDUCE_HALF_FLOAT(true) 
.type REDUCE_HALF_FLOAT(true), @function

.section .text.REDUCE_HALF_FLOAT(true), "ax"
.align 4
REDUCE_HALF_FLOAT(true):
bri REDUCE_HALF_FLOAT(common)
.size REDUCE_HALF_FLOAT(true),\
            .-REDUCE_HALF_FLOAT(true)

.globl REDUCE_HALF_FLOAT(false) 
.type REDUCE_HALF_FLOAT(false), @function

.section .text.REDUCE_HALF_FLOAT(false), "ax"
.align 4
REDUCE_HALF_FLOAT(false):
bri REDUCE_HALF_FLOAT(common)
.size REDUCE_HALF_FLOAT(false),\
            .-REDUCE_HALF_FLOAT(false)



// ------------------------------------------------------- //
// ------------------------------------------------------- //
// HALf HALF
// ------------------------------------------------------- //
// ------------------------------------------------------- //






#undef LABEL_SUFFIX
#define LABEL_SUFFIX _h_h_

.type REDUCE_HALF_HALF(common), @function

.section .text.REDUCE_HALF_HALF(common), "ax"
.align 4
REDUCE_HALF_HALF(common):
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

.size REDUCE_HALF_HALF(common),\
              .-REDUCE_HALF_HALF(common)


// Instantiate two variants which call the same common function
.globl REDUCE_HALF_HALF(true) 
.type REDUCE_HALF_HALF(true), @function

.section .text.REDUCE_HALF_HALF(true), "ax"
.align 4
REDUCE_HALF_HALF(true):
bri REDUCE_HALF_HALF(common)
.size REDUCE_HALF_HALF(true),\
            .-REDUCE_HALF_HALF(true)

.globl REDUCE_HALF_HALF(false) 
.type REDUCE_HALF_HALF(false), @function

.section .text.REDUCE_HALF_HALF(false), "ax"
.align 4
REDUCE_HALF_HALF(false):
bri REDUCE_HALF_HALF(common)
.size REDUCE_HALF_HALF(false),\
            .-REDUCE_HALF_HALF(false)




// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Float
// -------------------------------------------------------- //
// -------------------------------------------------------- //


#undef LABEL_SUFFIX
#define LABEL_SUFFIX _f_f_

#undef SIZE_OF_IN_TYPE
#define SIZE_OF_IN_TYPE 4

.type REDUCE_FLOAT_FLOAT(common), @function

.section .text.REDUCE_FLOAT_FLOAT(common), "ax"
.align 4
REDUCE_FLOAT_FLOAT(common):
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

.size REDUCE_FLOAT_FLOAT(common),\
              .-REDUCE_FLOAT_FLOAT(common)


// Instantiate two variants which call the same common function
.globl REDUCE_FLOAT_FLOAT(true) 
.type REDUCE_FLOAT_FLOAT(true), @function

.section .text.REDUCE_FLOAT_FLOAT(true), "ax"
.align 4
REDUCE_FLOAT_FLOAT(true):
bri REDUCE_FLOAT_FLOAT(common)
.size REDUCE_FLOAT_FLOAT(true),\
            .-REDUCE_FLOAT_FLOAT(true)

.globl REDUCE_FLOAT_FLOAT(false) 
.type REDUCE_FLOAT_FLOAT(false), @function

.section .text.REDUCE_FLOAT_FLOAT(false), "ax"
.align 4
REDUCE_FLOAT_FLOAT(false):
bri REDUCE_FLOAT_FLOAT(common)
.size REDUCE_FLOAT_FLOAT(false),\
            .-REDUCE_FLOAT_FLOAT(false)



// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Half
// -------------------------------------------------------- //
// -------------------------------------------------------- //



#undef LABEL_SUFFIX
#define LABEL_SUFFIX _f_h_

.globl REDUCE_FLOAT_HALF(common)
.type REDUCE_FLOAT_HALF(common), @function

.section .text.REDUCE_FLOAT_HALF(common), "ax"
.align 4
REDUCE_FLOAT_HALF(common):
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

.size REDUCE_FLOAT_HALF(common),\
            .-REDUCE_FLOAT_HALF(common)


// Instantiate two variants which call the same common function
.globl REDUCE_FLOAT_HALF(true) 
.type REDUCE_FLOAT_HALF(true), @function

.section .text.REDUCE_FLOAT_HALF(true), "ax"
.align 4
REDUCE_FLOAT_HALF(true):
bri REDUCE_FLOAT_HALF(common)
.size REDUCE_FLOAT_HALF(true),\
            .-REDUCE_FLOAT_HALF(true)

.globl REDUCE_FLOAT_HALF(false) 
.type REDUCE_FLOAT_HALF(false), @function

.section .text.REDUCE_FLOAT_HALF(false), "ax"
.align 4
REDUCE_FLOAT_HALF(false):
bri REDUCE_FLOAT_HALF(common)
.size REDUCE_FLOAT_HALF(false),\
            .-REDUCE_FLOAT_HALF(false)

#endif
