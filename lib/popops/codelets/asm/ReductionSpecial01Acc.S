// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
// Accumulator versions of reduction specialisations 0,1
// Covering operations: add, square add.
//
// The difference in these functions vs ReductionSpecial01NoAcc.S is that the accumulators
// are used and therefore also that the vectorwidth is 8 at most instead of 4.
//
// This is a general vertex capable of flexible operation at the expense of speed.
// We have a vector of regions to output - each of which can be a different size.
// Each output region is the reduction of a number of partials (can be different
// per out region).
// We cope with any length output/partial:
//
// If the desired outputs are [A0 A1] [B0 B1 B2]
// We could have partials:
//
// [a0 a1 a2 a3 a4 a5 a6 a7]
// [a8 a9]
// [a10 a11]
// [b0 b1 b2]
// [b3 b4 b5 b6 b7 b8]
//
// So that's 2 outputs A - with 3 partials, B with 2 partials.
// Partials 'fold' so A0 = Sum(add a's with even suffix)
//                    A1 = Sum(add a's with odd suffix)
//
// The implementation has an outer loop per reduction (_loop_over_reductions) {
//    The next loop deals with (in the Acc, half case) groups of 8 partials,
//    so if we have [A0 A1 ... A15] we make 2 passesto deal with 16
//    elements in each of the partials. (_out_j_loop) {
//        The next loop is over the partials themselves (_start_num_partials_loop) {
//             And finally over the repetitions of the output length in a partial (_in_j_loop_start) {
//                operation:8-half or 2-float width accumulation
//             }
//        }
//    }
//    Then loop - like the `_start_num_partials_loop` to deal with 4's of halves:
//    Outer loop - per partial,
//         inner loop per repetition of the output length within the partial
//
//    Then loop - like the `_start_num_partials_loop` to deal with 2's of halves:
//    Outer loop - per partial,
//        inner loop per repetition of the output length within the partial
//
//    Then loop - like the `_start_num_partials_loop` to deal with 2's of halves:
//    Outer loop - per partial,
//        inner loop per repetition of the output length within the partial
// }
//
// The fact that each of the lengths involved can generate non-aligned data slows things down a lot
//
// Performance notes:
// Per reduction overhead is 25 cycles to unpack vertex state, find number of
// partials per reduction etc...
// There are around 10 cycles overhead per partial
// The largest inner loop can take between 8 and 14 cycles for 8 halves
// depending on word alignment.
// Or 7 cycles for 4 floats, regardless of word alignment

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "MathConstants.S"
#include "poplar/StackSizeDefs.hpp"

// Register, vertex state definitions
#include "ReductionsCommon.h.S"

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

.macro ld32_MIS_2_
  ldb16 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ldb16 $ASCRATCH_0, $IN_j_PTR, $SCRATCH, 1
  roll16 $VALUES_0, $VALUES_0, $ASCRATCH_0
.endm

// ------------------------------------------------------- //
// Macros to apply scaling and conditionally update the output.

.macro DO_SCALE_AND_UPDATE_HALF_1 UPDATE
  // Apply scale as 32 bit for improved accuracy, especially as scale itself
  // can be small
.ifc "\UPDATE","true"
  {ldb16 $VALUES_2, $OUT_j_PTR, $mzero, 0
   f32v2mul $VALUES_0:1, $SCALE:B, $VALUES_0:1}
  {ldb16 $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
   f32v2tof16 $VALUES_0, $VALUES_0:1}
  f16v2add $VALUES_0, $VALUES_0, $VALUES_2
.else
  f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1
  {ldb16 $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
    f32v2tof16 $VALUES_0, $VALUES_0:1}
.endif
.endm

.macro DO_SCALE_AND_UPDATE_HALF_2 UPDATE
  // Apply scale as 32 bit for improved accuracy, especially as scale itself
  // can be small
.ifc "\UPDATE","true"
  {ld32 $VALUES_2, $OUT_j_PTR, $mzero, 0
   f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1}
  f32v2tof16 $VALUES_0, $VALUES_0:1
  f16v2add $VALUES_0, $VALUES_0, $VALUES_2
.else
  f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1
  f32v2tof16 $VALUES_0, $VALUES_0:1
.endif
.endm

.macro DO_SCALE_AND_UPDATE_HALF_4 UPDATE
  // Apply scale as 32 bit for improved accuracy, especially as scale itself
  // can be small
.ifc "\UPDATE","true"
  f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1
  f32v2gina  $VALUES_4:5, $azeros, 0
  f32v2mul   $VALUES_2:3, $SCALE:B, $VALUES_4:5
  {ld64 $VALUES_2:3, $OUT_j_PTR, $mzero, 0
   f32v4tof16 $VALUES_0:1, $VALUES_0:3}
  f16v4add $VALUES_0:1, $VALUES_0:1, $VALUES_2:3
.else
  f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1
  f32v2gina  $VALUES_4:5, $azeros, 0
  f32v2mul   $VALUES_2:3, $SCALE:B, $VALUES_4:5
  f32v4tof16 $VALUES_0:1, $VALUES_0:3
.endif
.endm

// ------------------------------------------------------- //
// The float macro deals with multiple sizes
.macro DO_SCALE_AND_UPDATE_FLOAT UPDATE SIZE
.ifc "\UPDATE","true"
  .ifc "\SIZE","2"
   {ld64 $VALUES_2:3, $OUT_j_PTR, $mzero, 0
    f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1}
  .else
   {ld32 $VALUES_2, $OUT_j_PTR, $mzero, 0
    f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1}
  .endif
  f32v2add $VALUES_0:1, $VALUES_0:1, $VALUES_2:3
.else
  f32v2mul   $VALUES_0:1, $SCALE:B, $VALUES_0:1
.endif
.endm

// Name mangling
#define REDUCE_HALF_FLOAT(prefix, specialisation)  __runCodelet_popops__##prefix##___\
popops__\OP\()_half_float_\UPDATE\()_popops__ReductionSpecialisation__##specialisation

#define REDUCE_HALF_HALF(prefix, specialisation) __runCodelet_popops__##prefix##___\
popops__\OP\()_half_half_\UPDATE\()_popops__ReductionSpecialisation__##specialisation

#define REDUCE_FLOAT_FLOAT(prefix, specialisation) __runCodelet_popops__##prefix##___\
popops__\OP\()_float_float_\UPDATE\()_popops__ReductionSpecialisation__##specialisation

#define REDUCE_FLOAT_HALF(prefix, specialisation) __runCodelet_popops__##prefix##___\
popops__\OP\()_float_half_\UPDATE\()_popops__ReductionSpecialisation__##specialisation

//------------------------------------------------------------------------------
// HALF FLOAT
//------------------------------------------------------------------------------

.macro INSTANTIATE_HALF_FLOAT UPDATE INSTRUCTION OP
.equ SIZE_OF_IN_TYPE, 2

.type REDUCE_HALF_FLOAT(Reduce,common), @function

DEF_STACK_USAGE 0 .text.REDUCE_HALF_FLOAT(Reduce,common)
.section .text.REDUCE_HALF_FLOAT(Reduce,common), "ax"

// Instantiate two variants which call the same common function
.globl REDUCE_HALF_FLOAT(Reduce,DEFAULT)
.type REDUCE_HALF_FLOAT(Reduce,DEFAULT), @function
.globl REDUCE_HALF_FLOAT(ScaledReduce,DEFAULT)
.type REDUCE_HALF_FLOAT(ScaledReduce,DEFAULT), @function
.globl REDUCE_HALF_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_HALF_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS), @function
.globl REDUCE_HALF_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_HALF_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS), @function

// ************************************************* //
// Load vertex state
// ************************************************* //
.align 4
REDUCE_HALF_FLOAT(Reduce,common):
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_HALF_FLOAT(Reduce,DEFAULT):
REDUCE_HALF_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS):
#else
REDUCE_HALF_FLOAT(Reduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_HALF_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
#endif
{
  bri        2f
  or         $SCALE, $azero, FLOAT_1_0
}
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_HALF_FLOAT(ScaledReduce,DEFAULT):
REDUCE_HALF_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/2
  setzi      $SCRATCH2, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $SCRATCH2, $mzero, $SCRATCH
#else
REDUCE_HALF_FLOAT(ScaledReduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_HALF_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/4
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif

2:
  call       $IN_j_SIZE, _Reduce_load_state_process_common

_loop_over_reductions.\@:
// ************************************************* //
// unpack offset and size
// ************************************************* //
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  call       $IN_j_SIZE, _Reduce_outer_loop_setup
#else
  // Implement a call to function pointer by pre-loading the return register
  // with the address of the following instruction, followed by branching
  // to the function pointer address.
  setzi      $IN_j_SIZE, 3f
  ld32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  br         $SCRATCH
#endif
3:
  and        $SCRATCH, $OUT_j_SIZE, 0x7
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 3

  brnzdec    $OUT_j_SIZE, _skip2.\@
  bri        _out_j_size_remainder.\@
_skip2.\@:

_out_j_loop.\@:
  call       $SCRATCH2, _Reduce_zero_and_load

// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
  st32      $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

_start_num_partials_loop.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start.\@:
  call      $OUT_j_SIZE, _Reduce_ld128_MIS_2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.\@
  brnzdec    $NUM_PART, _start_num_partials_loop.\@

  ld32      $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

// ************************************************* //
// end of 8 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 16
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.\@

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.\@:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH
  and        $SCRATCH, $OUT_j_SIZE, 4
  brz        $SCRATCH, _out_j_2_remainder.\@

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_4_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start_4_rem.\@:
  ld64_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_4_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_4_rem.\@
// ************************************************* //
// end of 4 vector accumulating, scale and store
// ************************************************* //
  {
    add      $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1


// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.\@

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_2_rem.\@:
    call       $SCRATCH2, _Reduce_ptr_fetch
    mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.\@:
  ld32_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_2_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.\@

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 4
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.\@

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_1_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start_1_rem.\@:
  ldb16      $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.\@

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 2
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 1
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

_out_j_size_end.\@:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.\@
// ************************************************* //
// End of loops
// ************************************************* //
_exit.\@:
  exitz      $mzero

.size REDUCE_HALF_FLOAT(Reduce,common),\
            .-REDUCE_HALF_FLOAT(Reduce,common)

.endm
// ------------------------------------------------------- //
// ------------------------------------------------------- //
// HALF HALF
// ------------------------------------------------------- //
// ------------------------------------------------------- //

.macro INSTANTIATE_HALF_HALF UPDATE INSTRUCTION OP
.equ SIZE_OF_IN_TYPE, 2

// Instantiate two variants which call the same common function
.globl REDUCE_HALF_HALF(Reduce,DEFAULT)
.type REDUCE_HALF_HALF(Reduce,DEFAULT), @function
.globl REDUCE_HALF_HALF(ScaledReduce,DEFAULT)
.type REDUCE_HALF_HALF(ScaledReduce,DEFAULT), @function
.globl REDUCE_HALF_HALF(Reduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_HALF_HALF(Reduce,SCALAR___OUTPUT___REGIONS), @function
.globl REDUCE_HALF_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_HALF_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS), @function

.type REDUCE_HALF_HALF(Reduce,common), @function

DEF_STACK_USAGE  0  .text.REDUCE_HALF_HALF(Reduce,common)
.section .text.REDUCE_HALF_HALF(Reduce,common), "ax"
.align 4
// ************************************************* //
// Load vertex state
// ************************************************* //
REDUCE_HALF_HALF(Reduce,common):
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_HALF_HALF(Reduce,DEFAULT):
REDUCE_HALF_HALF(Reduce,SCALAR___OUTPUT___REGIONS):
#else
REDUCE_HALF_HALF(Reduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_HALF_HALF(Reduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
#endif
{
  bri        2f
  or         $SCALE, $azero, FLOAT_1_0
}
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_HALF_HALF(ScaledReduce,DEFAULT):
REDUCE_HALF_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/2
  setzi      $SCRATCH2, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $SCRATCH2, $mzero, $SCRATCH
#else
REDUCE_HALF_HALF(ScaledReduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_HALF_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/4
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif

2:
 call       $IN_j_SIZE, _Reduce_load_state_process_common

_loop_over_reductions.\@:
// ************************************************* //
// unpack offset and size
// ************************************************* //
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  call       $IN_j_SIZE, _Reduce_outer_loop_setup
#else
  // Implement a call to function pointer by pre-loading the return register
  // with the address of the following instruction, followed by branching
  // to the function pointer address.
  setzi      $IN_j_SIZE, 3f
  ld32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  br         $SCRATCH
#endif
3:
  and        $SCRATCH, $OUT_j_SIZE, 0x7
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 3

  brnzdec    $OUT_j_SIZE, _skip2.\@
  bri        _out_j_size_remainder.\@
_skip2.\@:

_out_j_loop.\@:
  call       $SCRATCH2, _Reduce_zero_and_load


// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
  st32      $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

_start_num_partials_loop.\@:
   call       $SCRATCH2, _Reduce_ptr_fetch
   mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start.\@:
  call      $OUT_j_SIZE, _Reduce_ld128_MIS_2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.\@
  brnzdec    $NUM_PART, _start_num_partials_loop.\@

  ld32       $OUT_j_SIZE, $mworker_base, $mzero, OUT_j_SIZE_SCRATCH

// ************************************************* //
// end of 8 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 16
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_HALF_4 \UPDATE
  {st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
   f32v2gina  $VALUES_0:1, $azeros, 0}

  DO_SCALE_AND_UPDATE_HALF_4 \UPDATE
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.\@

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.\@:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH
  and        $SCRATCH, $OUT_j_SIZE, 4
  brz        $SCRATCH, _out_j_2_remainder.\@

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_4_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_4_rem.\@:
  ld64_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_4_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_4_rem.\@
// ************************************************* //
// end of 4 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_HALF_4 \UPDATE
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1


// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.\@

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_2_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.\@:
  ld32_MIS_2_      // trashes scratch2
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_2_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.\@

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 4
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_HALF_2 \UPDATE
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.\@

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_1_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, 2 // size of half

_in_j_loop_start_1_rem.\@:
  ldb16      $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f16v8\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.\@

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 2
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_HALF_1 \UPDATE

  sort4x16lo $VALUES_0, $VALUES_0, $ASCRATCH_0
  st32 $VALUES_0, $OUT_j_PTR, $mzero, 0

_out_j_size_end.\@:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.\@
// ************************************************* //
// End of loops
// ************************************************* //
_exit.\@:
  exitz      $mzero

.size REDUCE_HALF_HALF(Reduce,common),\
              .-REDUCE_HALF_HALF(Reduce,common)

.endm
// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Float
// -------------------------------------------------------- //
// -------------------------------------------------------- //

.macro INSTANTIATE_FLOAT_FLOAT UPDATE INSTRUCTION OP
.equ SIZE_OF_IN_TYPE, 4

.type REDUCE_FLOAT_FLOAT(Reduce,common), @function

DEF_STACK_USAGE 0 .text.REDUCE_FLOAT_FLOAT(Reduce,common)

.section .text.REDUCE_FLOAT_FLOAT(Reduce,common), "ax"

// Instantiate two variants which call the same common function
.globl REDUCE_FLOAT_FLOAT(Reduce,DEFAULT)
.type REDUCE_FLOAT_FLOAT(Reduce,DEFAULT), @function
.globl REDUCE_FLOAT_FLOAT(ScaledReduce,DEFAULT)
.type REDUCE_FLOAT_FLOAT(ScaledReduce,DEFAULT), @function
.globl REDUCE_FLOAT_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_FLOAT_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS), @function
.globl REDUCE_FLOAT_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_FLOAT_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS), @function

.align 4
// ************************************************* //
// Load vertex state
// ************************************************* //
REDUCE_FLOAT_FLOAT(Reduce,common):
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_FLOAT_FLOAT(Reduce,DEFAULT):
REDUCE_FLOAT_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS):
#else
REDUCE_FLOAT_FLOAT(Reduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_FLOAT_FLOAT(Reduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
#endif
{
  bri        2f
  or      $SCALE, $azero, FLOAT_1_0
}
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_FLOAT_FLOAT(ScaledReduce,DEFAULT):
REDUCE_FLOAT_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/2
  setzi      $SCRATCH2, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $SCRATCH2, $mzero, $SCRATCH
#else
REDUCE_FLOAT_FLOAT(ScaledReduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_FLOAT_FLOAT(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/4
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif

2:
  call       $IN_j_SIZE, _Reduce_load_state_process_common

_loop_over_reductions.\@:
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  call       $IN_j_SIZE, _Reduce_outer_loop_setup
#else
  // Implement a call to function pointer by pre-loading the return register
  // with the address of the following instruction, followed by branching
  // to the function pointer address.
  setzi      $IN_j_SIZE, 3f
  ld32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  br         $SCRATCH
#endif
3:
  and        $SCRATCH, $OUT_j_SIZE, 0x3
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 2

  brnzdec    $OUT_j_SIZE, _skip2.\@
  bri        _out_j_size_remainder.\@
_skip2.\@:

_out_j_loop.\@:
  call       $SCRATCH2, _Reduce_zero_and_load

// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
_start_num_partials_loop.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start.\@:
  // As we don't assume alignment here, it's better to load 4x32 bits
  // than conditionally load 2x64 bits or revert to loading 4x32 bits based
  // on checking alignment
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  ld32 $VALUES_2, $IN_j_PTR, $SCRATCH, 2
  ld32 $VALUES_3, $IN_j_PTR, $SCRATCH, 3
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f32v4\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start.\@
  brnzdec    $NUM_PART, _start_num_partials_loop.\@

// ************************************************* //
// end of 8 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 16
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2
  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

  brnzdec    $OUT_j_SIZE, _out_j_loop.\@

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.\@:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH

// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.\@

  call       $SCRATCH2, _Reduce_zero_and_load


_start_num_partials_loop_2_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch

  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.\@:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f32v4\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_2_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.\@

// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 2

  st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.\@

  call       $SCRATCH2, _Reduce_zero_and_load

_start_num_partials_loop_1_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_1_rem.\@:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f32v4\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.\@
  brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.\@

// ************************************************* //
// end of 1 vector accumulating, scale and store
// ************************************************* //
  {
    add $IN_j_DELTA, $IN_j_DELTA, 4
    f32v2gina  $VALUES_0:1, $azeros, 0
  }
  DO_SCALE_AND_UPDATE_FLOAT \UPDATE 1
  st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1

_out_j_size_end.\@:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.\@
// ************************************************* //
// End of loops
// ************************************************* //
_exit.\@:
  exitz      $mzero

.size REDUCE_FLOAT_FLOAT(Reduce,common),\
              .-REDUCE_FLOAT_FLOAT(Reduce,common)
.endm

// -------------------------------------------------------- //
// -------------------------------------------------------- //
// Float Half
// -------------------------------------------------------- //
// -------------------------------------------------------- //
.macro INSTANTIATE_FLOAT_HALF UPDATE INSTRUCTION OP

.equ SIZE_OF_IN_TYPE, 4
.globl REDUCE_FLOAT_HALF(Reduce,common)
.type REDUCE_FLOAT_HALF(Reduce,common), @function

DEF_STACK_USAGE 0 .text.REDUCE_FLOAT_HALF(Reduce,common)

.section .text.REDUCE_FLOAT_HALF(Reduce,common), "ax"

// Instantiate two variants which call the same common function
.globl REDUCE_FLOAT_HALF(Reduce,DEFAULT)
.type REDUCE_FLOAT_HALF(Reduce,DEFAULT), @function
.globl REDUCE_FLOAT_HALF(ScaledReduce,DEFAULT)
.type REDUCE_FLOAT_HALF(ScaledReduce,DEFAULT), @function
.globl REDUCE_FLOAT_HALF(Reduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_FLOAT_HALF(Reduce,SCALAR___OUTPUT___REGIONS), @function
.globl REDUCE_FLOAT_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS)
.type REDUCE_FLOAT_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS), @function

.align 4
// ************************************************* //
// Load vertex state
// ************************************************* //
REDUCE_FLOAT_HALF(Reduce,common):
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_FLOAT_HALF(Reduce,DEFAULT):
REDUCE_FLOAT_HALF(Reduce,SCALAR___OUTPUT___REGIONS):
#else
REDUCE_FLOAT_HALF(Reduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_FLOAT_HALF(Reduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
#endif
{
  bri        2f
  or      $SCALE, $azero, FLOAT_1_0
}
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
REDUCE_FLOAT_HALF(ScaledReduce,DEFAULT):
REDUCE_FLOAT_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  ldz16      $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/2
  setzi      $SCRATCH2, TMEM_REGION0_BASE_ADDR
  ld32       $SCALE, $SCRATCH2, $mzero, $SCRATCH
#else
REDUCE_FLOAT_HALF(ScaledReduce,DEFAULT):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align64
  bri        1f

REDUCE_FLOAT_HALF(ScaledReduce,SCALAR___OUTPUT___REGIONS):
  setzi      $SCRATCH, _Reduce_outer_loop_setup_out_align32

1:
  st32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  ld32       $SCRATCH, $mvertex_base, $mzero, SCALE_OFF/4
  ld32       $SCALE, $mzero, $SCRATCH, 0
#endif

2:
  call       $IN_j_SIZE, _Reduce_load_state_process_common

_loop_over_reductions.\@:
// ************************************************* //
// unpack offset and size
// ************************************************* //
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTORLIST_AVAIL_DELTAN)
  call       $IN_j_SIZE, _Reduce_outer_loop_setup
#else
  // Implement a call to function pointer by pre-loading the return register
  // with the address of the following instruction, followed by branching
  // to the function pointer address.
  setzi      $IN_j_SIZE, 3f
  ld32       $SCRATCH, $mworker_base, $mzero, FN_REDUCE_OUTER_LOOP_SCRATCH
  br         $SCRATCH
#endif
3:
  and        $SCRATCH, $OUT_j_SIZE, 0x3
  st32       $SCRATCH, $mworker_base, $mzero, REM_SCRATCH
  mul        $NUM_ELEM, $OUT_j_SIZE, SIZE_OF_IN_TYPE
  shr        $OUT_j_SIZE, $OUT_j_SIZE, 2

  brnzdec    $OUT_j_SIZE, _skip2.\@
  bri        _out_j_size_remainder.\@
_skip2.\@:

_out_j_loop.\@:
  call       $SCRATCH2, _Reduce_zero_and_load

// ************************************************* //
// Loop over inputs accumulating
// ************************************************* //
_start_num_partials_loop.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start.\@:
  // As we don't assume alignment here, it's better to load 4x32 bits
  // than conditionally load 2x64 bits or revert to loading 4x32 bits based
  // on checking alignment
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  ld32 $VALUES_2, $IN_j_PTR, $SCRATCH, 2
  ld32 $VALUES_3, $IN_j_PTR, $SCRATCH, 3
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f32v4\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  {
    brnz       $SCRATCH2, _in_j_loop_start.\@
    // move ACC0 and ACC2 to VALUES0_1 and shift the remaining results to
    // the odd accumulators. This does not shift the accumulator delay line.
    f16v4stacc $VALUES_0:1, 0
  }
  {
    brnzdec   $NUM_PART, _start_num_partials_loop.\@
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
.ifc "\UPDATE","true"
  {
    // load value to update
    ld64       $VALUES_2:3, $OUT_j_PTR, $mzero, 0
    f32v4tof16 $VALUES_0:1, $VALUES_0:3
  }
  f16v4add $VALUES_0:1, $VALUES_0:1, $VALUES_2:3
.else
  f32v4tof16 $VALUES_0:1, $VALUES_0:3
.endif

  {
    st64step   $VALUES_0:1, $mzero, $OUT_j_PTR+=, 1
    // The accumulator pipeline is still dirty and needs to be cleared for
    // next pass.
    uput       $FP_CLR, $ZAACC
  }

  brnzdec    $OUT_j_SIZE, _out_j_loop.\@

// ************************************************* //
// 4 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_size_remainder.\@:
  ld32       $OUT_j_SIZE, $mworker_base, $mzero, REM_SCRATCH

// ************************************************* //
// 2 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_2_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 2
  brz        $SCRATCH, _out_j_1_remainder.\@

  call       $SCRATCH2, _Reduce_zero_and_load

_start_num_partials_loop_2_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_2_rem.\@:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  ld32 $VALUES_1, $IN_j_PTR, $SCRATCH, 1
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f32v4\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  {
    brnz       $SCRATCH2, _in_j_loop_start_2_rem.\@
    // move ACC0 and ACC2 to VALUES0_1 and shift the remaining results to
    // the odd accumulators. This does not shift the accumulator delay line.
    f16v4stacc $VALUES_0:1, 0
  }
  {
    brnzdec    $NUM_PART, _start_num_partials_loop_2_rem.\@
    // scale up in higher precision
    f32v2mul $VALUES_0:1, $SCALE:B, $VALUES_0:1
  }
// ************************************************* //
// end of 2 vector accumulating, scale and store
// ************************************************* //

.ifc "\UPDATE","true"
  {
    ld32       $VALUES_2, $OUT_j_PTR, $mzero, 0
    f32v2tof16 $VALUES_0, $VALUES_0:1
  }
  {
    add        $IN_j_DELTA, $IN_j_DELTA, 8
    f16v2add   $VALUES_0, $VALUES_0, $VALUES_2
  }
.else
  {
    add        $IN_j_DELTA, $IN_j_DELTA, 8
    f32v2tof16 $VALUES_0, $VALUES_0:1
  }
.endif
  {
    st32step   $VALUES_0, $mzero, $OUT_j_PTR+=, 1
    // The accumulator state is dirty and must be cleared for the next pass
    uput       $FP_CLR, $ZAACC
  }

// ************************************************* //
// 1 vector remainder accumulate, scale and store
// ************************************************* //
_out_j_1_remainder.\@:
  and        $SCRATCH, $OUT_j_SIZE, 1
  brz        $SCRATCH, _out_j_size_end.\@
  call       $SCRATCH2, _Reduce_zero_and_load

_start_num_partials_loop_1_rem.\@:
  call       $SCRATCH2, _Reduce_ptr_fetch
  mul        $IN_j_SIZE, $IN_j_SIZE, SIZE_OF_IN_TYPE

_in_j_loop_start_1_rem.\@:
  ld32 $VALUES_0, $IN_j_PTR, $SCRATCH, 0
  {
    add        $SCRATCH, $SCRATCH, $NUM_ELEM  // need to keep track of j delta
    f32v4\INSTRUCTION   $VALUES_0:3
  }
  cmpult     $SCRATCH2, $SCRATCH, $IN_j_SIZE
  brnz       $SCRATCH2, _in_j_loop_start_1_rem.\@
  {
    brnzdec    $NUM_PART, _start_num_partials_loop_1_rem.\@
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
.ifc "\UPDATE","true"
  {
    ldb16      $VALUES_2, $OUT_j_PTR, $mzero, 0
    f32v2tof16 $VALUES_0, $VALUES_0:1
  }
  {
    ldb16    $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
    f16v2add $VALUES_0, $VALUES_0, $VALUES_2
  }
.else
  {
    ldb16       $ASCRATCH_0, $OUT_j_PTR, $mzero, 1
    f32v2tof16  $VALUES_0, $VALUES_0:1
  }
.endif

  sort4x16lo $VALUES_0, $VALUES_0, $ASCRATCH_0
  {
    st32       $VALUES_0, $OUT_j_PTR, $mzero, 0
    // Clear accumulators for the next pass as they are dirty
    uput       $FP_CLR, $ZAACC
  }

_out_j_size_end.\@:
  // add num_partials to IN_i_ptr and store
  st32       $IN_i_PTR, $mworker_base, $mzero, IN_PTR_SCRATCH
  brnzdec    $OUT_i_SIZE, _loop_over_reductions.\@

// ************************************************* //
// End of loops
// ************************************************* //
_exit.\@:
  exitz      $mzero

.size REDUCE_FLOAT_HALF(Reduce,common),\
            .-REDUCE_FLOAT_HALF(Reduce,common)
.endm

//******************************************************************************
// Use macros to instantiate each vertex variant

INSTANTIATE_HALF_FLOAT true acc ReduceAdd
INSTANTIATE_HALF_FLOAT true sqacc ReduceSquareAdd
INSTANTIATE_HALF_FLOAT false acc ReduceAdd
INSTANTIATE_HALF_FLOAT false sqacc ReduceSquareAdd

INSTANTIATE_HALF_HALF true acc ReduceAdd
INSTANTIATE_HALF_HALF true sqacc ReduceSquareAdd
INSTANTIATE_HALF_HALF false acc ReduceAdd
INSTANTIATE_HALF_HALF false sqacc ReduceSquareAdd

INSTANTIATE_FLOAT_HALF true acc ReduceAdd
INSTANTIATE_FLOAT_HALF true sqacc ReduceSquareAdd
INSTANTIATE_FLOAT_HALF false acc ReduceAdd
INSTANTIATE_FLOAT_HALF false sqacc ReduceSquareAdd

INSTANTIATE_FLOAT_FLOAT true acc ReduceAdd
INSTANTIATE_FLOAT_FLOAT true sqacc ReduceSquareAdd
INSTANTIATE_FLOAT_FLOAT false acc ReduceAdd
INSTANTIATE_FLOAT_FLOAT false sqacc ReduceSquareAdd

//******************************************************************************
// Use macros to create various common functions
CREATE_COMMON_FUNCTIONS

#endif
