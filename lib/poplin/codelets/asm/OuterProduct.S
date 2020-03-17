// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Outer product assembler overview:
//
// This vertex calculates the outer product of 2 vectors, input and weights.
// When producing multiple outputs, multiple weights vectors are provided but
// the same input vector is used.
// The length of a single weights vector is given by the CHANS_PG parameter,
// meaning that the passed pointer to WEIGHTS contains CHANS_PG * OUTPUTS items.
//
// So the inputs are:
// IN[INPUT_SIZE]
// WEIGHTS[0][CHANS_PG]
// WEIGHTS[1][CHANS_PG] etc
//
// Outputs will be:
// OUT 0 =outer product(IN,WEIGHTS[0])
// OUT 1 =outer product(IN,WEIGHTS[1]) etc
//
// Each output is of size INPUT_SIZE * CHANS_PG:
// INPUT_SIZE rows, CHANS_PG columns
//
// Outer product produces a matrix where every item in IN and WEIGHTS is
// multiplied with every other.
/* -------------------------------------------------------------------------- */

#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// Register aliases

#define RPT_COUNT   m0
#define mSCRATCH    m1
#define WEIGHTS     m2
#define OUT_PTR     m3
#define mSCRATCH2   m4
#define CHANS_PG    m5

#define IN          m6
#define OUT         m7

#define IN_OUT_PTR  m8:9
#define OUT_ROWS_LOOPS m10
#define MATRIX_LOOP_COUNT m11

#define ROW_COUNT_UP m8
#define WEIGHTS_REWIND m9


#define INVAL a0
#define INVAL2 a1
#define INVAL_PAIR a0:1

#define PRODUCT_ONE  a2
#define PRODUCT_PAIR a2:3

#define WEIGHT_ONE    a4
#define WEIGHT_TWO    a5
#define WEIGHT_PAIR a4:5


// Constants

#define FLOATS_PER_64BITS 2
#define LOG2_FLOATS_PER_64BITS 1

#define HALVES_PER_64BITS 4
#define LOG2_HALVES_PER_64BITS 2

#define LOG2_SIZEOF_FLOAT 2
#define LOG2_SIZEOF_HALF 1

//****************************************************************************
// The input structure is always the same so a macro can be used to fetch the
// parameters:
// input start  pointer to input start
// input end    pointer to input end
// Weights      pointer to the weights array
// output start pointer to an array of output pointers
// output end   pointer to the end of the output - gives number of outputs
// chansPerGroup number of weights per output group
//****************************************************************************
.macro GET_PARAMS VOFF_IN VOFF_IN_END VOFF_W VOFF_OUT VOFF_OUT_END VOFF_CHANS_PG
    ld32     $IN,        $mzero, $mvertex_base, \VOFF_IN
    ld32     $mSCRATCH,  $mzero, $mvertex_base, \VOFF_IN_END
    ld32     $WEIGHTS,   $mzero, $mvertex_base, \VOFF_W
    ld32     $OUT_PTR,   $mzero, $mvertex_base, \VOFF_OUT
    ld32     $mSCRATCH2, $mzero, $mvertex_base, \VOFF_OUT_END
    ld32     $CHANS_PG,  $mzero, $mvertex_base, \VOFF_CHANS_PG
.endm

//********************************************************************
// Macro to calculate outer product quickly.
// when used, the parameters configure it for float or half processing
//
// Assumes that the weights and output are both 64 bit aligned,
// and that the number of weights (and therefore also the length of
// an output matix row) is also either a multiple of 2 (float) or 4 (half).
//
// Due to the loop structure there need to be at least
// 3 "groups" of weights :3*2=6 (float) or 3*4=12 (half).
//********************************************************************

.macro OUTER_PRODUCT_FAST ldininstr mulinstr
    // This macro contains a repeat block, which needs to be aligned to an 8
    // byte boundary. Aligning here ensures that it is aligned each time it's
    // used, with the expense that a potential inserted nop is executed once
.align 8
.Lmatrix_loop\@:
    // load pointers to input and output
    // input is the same each time so we fetch it fetch from the vertex state
    ld32     $IN, $mzero, $mvertex_base, 0
    ld32step $OUT, $mzero, $OUT_PTR+=, 1

    // load an input value, constant during the inner loop
    \ldininstr  $INVAL, $mzero, $IN+=, 1

    // prime the loop pipeline by fetching and calculating the 1st group of
    // products to store, and the second group of weights
    ld64step  $WEIGHT_PAIR, $mzero, $WEIGHTS+=, 1

    {ld64step  $WEIGHT_PAIR, $mzero, $WEIGHTS+=, 1
     \mulinstr $PRODUCT_PAIR, $INVAL:B, $WEIGHT_PAIR}
    // Pack the resulting weights and out pointers ready to use ldst64
    tapack    $IN_OUT_PTR, $WEIGHTS, $mzero, $OUT

    mov        $mSCRATCH2, $OUT_ROWS_LOOPS     //loop count for remaining rows
    // The first loop is different, as we've pre loaded and processed one
    // group of weights. So skip an equivalent load/store
    bri       .Lrow_loop_first\@

.Lrow_loop\@:
    // load the next input, ready for the next column,
    // multiplied by the 1st weight pair.
    \ldininstr   $INVAL, $mzero, $IN+=, 1

    {ldst64pace  $WEIGHT_PAIR, $PRODUCT_PAIR, $IN_OUT_PTR+=, $mzero, 0
      \mulinstr  $PRODUCT_PAIR, $INVAL:B, $WEIGHT_PAIR}


.Lrow_loop_first\@:

     {rpt  $RPT_COUNT, ((2f-1f)/8)-1
      fnop}
1:
    {ldst64pace  $WEIGHT_PAIR, $PRODUCT_PAIR, $IN_OUT_PTR+=, $mzero, 0
      \mulinstr  $PRODUCT_PAIR, $INVAL:B, $WEIGHT_PAIR}
2:
    // After the loop - fetch the last weights and
    // stride back to point at the 1st weight again
    {ldst64pace  $WEIGHT_PAIR, $PRODUCT_PAIR, $IN_OUT_PTR+=, $CHANS_PG, 1
      \mulinstr  $PRODUCT_PAIR, $INVAL:B, $WEIGHT_PAIR}

    // multiply the last weights, fetch the 1st weights all before we fetch the
    // next input
    {ldst64pace  $WEIGHT_PAIR, $PRODUCT_PAIR, $IN_OUT_PTR+=, $mzero, 0
     \mulinstr   $PRODUCT_PAIR, $INVAL:B, $WEIGHT_PAIR}


    brnzdec  $mSCRATCH2, .Lrow_loop\@

    // store the last result for this matrix
     st64pace  $PRODUCT_PAIR, $IN_OUT_PTR+=, $mzero, 0

    // adjust $WEIGHTS pointer to point at the next group to make the next output
     add      $WEIGHTS, $WEIGHTS, $mSCRATCH

     brnzdec  $MATRIX_LOOP_COUNT, .Lmatrix_loop\@
.endm


//******************************************************************************
// Slower version.
// Not conforming to the constraints under which the faster option will work.
// Approach is to try to keep writing a whole matrix output 2 items at once
// until the potential last single item. This avoids read modify write for
// halves, and allows 2 word writes for floats, which can simply be done
// by parameterising the same macro.

     .macro OUTER_PRODUCT_SLOWER ldinstr half

  mov      $mSCRATCH2, $OUT_ROWS_LOOPS
.Lmatrix_loop\@:
  // Prepare to loop each matrix - fetch IN pointer from the vertex state,
  // as it's the same each time
  // Weights will point to the first of a group of weights already
  ld32     $IN, $mzero, $mvertex_base, 0
  ld32step $OUT, $mzero, $OUT_PTR+=, 1
  mov      $OUT_ROWS_LOOPS, $mSCRATCH2

  // Initialise a counter that counts up per row, used when the number
  // of weights is odd so that we do some cleanup to finish the row.
  // Necessary as we write pairs of items.
  setzi   $ROW_COUNT_UP,1
  // load an input which is constant over the inner loop
  \ldinstr $INVAL, $mzero, $IN+=, 1

  bri       .Lweights_loop_late_entry\@

.Lweights_loop\@:
  // point back to the start of the weights (not on the first pass)
  sub       $WEIGHTS, $WEIGHTS, $WEIGHTS_REWIND
  // load an input which is constant over the inner loop
  \ldinstr  $INVAL, $mzero, $IN+=, 1
.Lweights_loop_late_entry\@:

  // Inner loop count: half the weights count, rounding down
  // Keeping it simpler - not assuming weights are aligned to 2x input size
  shr      $RPT_COUNT,$CHANS_PG,1

  // load the 1st weight so we can bundle in the loop. Means we overread by 1
  // in the loop body.
  // For the half case there is potential to bundle more but at the
  // expense of more complexity - an overread of 2 items.
  \ldinstr  $WEIGHT_ONE, $mzero, $WEIGHTS+=,1

  //Branching to the loop end provides a decrement as we need one to
  //use brnzdec, plus copes with edge cases where the loop count=0
  bri       2f
1:
  \ldinstr  $WEIGHT_TWO, $mzero, $WEIGHTS+=,1

.if \half

  roll16    $WEIGHT_ONE, $WEIGHT_ONE, $WEIGHT_TWO
 { \ldinstr  $WEIGHT_ONE, $mzero, $WEIGHTS+=,1
   f16v2mul  $PRODUCT_ONE, $INVAL:BU, $WEIGHT_ONE}
  st32step  $PRODUCT_ONE, $mzero, $OUT+=,1

.else

  {\ldinstr  $WEIGHT_ONE, $mzero, $WEIGHTS+=,1
    f32v2mul  $PRODUCT_PAIR, $INVAL:B, $WEIGHT_PAIR}

  st64step  $PRODUCT_PAIR, $mzero, $OUT+=,1
.endif
2:
  brnzdec   $RPT_COUNT,1b

  // Is there an item left at the end of this output row?
  // If so then some cleanup is needed to continue.
  // This happens on alternate rows when the number of weights is odd.
  and       $mSCRATCH, $CHANS_PG, $ROW_COUNT_UP
  and       $mSCRATCH, $mSCRATCH,1
  add       $ROW_COUNT_UP, $ROW_COUNT_UP,1
  brz       $mSCRATCH, .Lweights_even\@

.Lweights_odd\@:
  // We got to the end of a row with 1 unaligned item left to process
  // There are 2 situations - the end of a matrix output or the
  // case where we are just ending output of a row.

  brz      $OUT_ROWS_LOOPS,.Lweights_odd_matrix_end\@

  //************************
  // Not the end of the current matrix.
  // load a new input and the 1st weight, combining with the old input
  // and the last weight.
  \ldinstr  $INVAL2,$mzero,$IN+=,1

  sub       $WEIGHTS,$WEIGHTS,$WEIGHTS_REWIND
  \ldinstr  $WEIGHT_TWO, $mzero, $WEIGHTS+=,1

.if \half
  roll16    $INVAL, $INVAL, $INVAL2
  roll16    $WEIGHT_ONE, $WEIGHT_ONE, $WEIGHT_TWO

  // This multiply and store will output the last of a row, and the first
  // of the next row
  f16v2mul  $PRODUCT_ONE, $INVAL, $WEIGHT_ONE
  st32step  $PRODUCT_ONE, $mzero, $OUT+=,1
.else
  f32v2mul  $PRODUCT_PAIR, $INVAL_PAIR, $WEIGHT_PAIR
  st64step  $PRODUCT_PAIR, $mzero, $OUT+=,1
  mov       $INVAL, $INVAL2
.endif

  // This will always branch, as we confirmed it was non zero
  // before getting here. Branch to late entry point as the
  // weights pointer has already been adjusted and the next input fetched.
  brnzdec   $OUT_ROWS_LOOPS, .Lweights_loop_late_entry\@

//************************
//end of the current matrix, tidy up the end of the output
.Lweights_odd_matrix_end\@:

.if \half
  // Need to read- modify write the last output item in the case of halves
  f16v2mul  $PRODUCT_ONE, $INVAL:BL, $WEIGHT_ONE

  ldb16     $WEIGHT_TWO, $mzero, $OUT, 0
  roll16    $PRODUCT_ONE, $PRODUCT_ONE, $WEIGHT_TWO
  st32      $PRODUCT_ONE, $mzero, $OUT, 0
.else
  // Just multiply and store the last one in the case of floats
  f32mul    $PRODUCT_ONE, $INVAL, $WEIGHT_ONE
  st32      $PRODUCT_ONE, $mzero, $OUT, 0
.endif

  // By doing a dummy read we can just fall through to
  // the looping code below.
  \ldinstr $WEIGHT_ONE, $mzero, $WEIGHTS+=,1

.Lweights_even\@:
  // Ending a row or matrix having written and reached a 2 item boundary means
  // no clean up other than a dummy read so we adjust the weight pointer
  // back correctly
  \ldinstr  $WEIGHT_ONE, $mzero, $WEIGHTS+=,-1
  brnzdec   $OUT_ROWS_LOOPS,.Lweights_loop\@

  brnzdec   $MATRIX_LOOP_COUNT, .Lmatrix_loop\@

  exitz     $mzero

    .endm
//******************************************************************************
// Float outer product
//
// Reads the passed parameters/pointers and decides if the fast version can be
// run. If not, select the slower version. Calculates many of the input
// pointers and counts for the macros to use.
//******************************************************************************
DEF_STACK_USAGE 0 __runCodelet_poplin__OuterProduct___float
.section .text.__runCodelet_poplin__OuterProduct___float

.globl __runCodelet_poplin__OuterProduct___float
.type __runCodelet_poplin__OuterProduct___float, @function
.align 4

__runCodelet_poplin__OuterProduct___float:

    GET_PARAMS  0 1 2 3 4 5

    add  $OUT_ROWS_LOOPS, $mSCRATCH, -1

    // Outermost loop count = number of output matrices-1 (to use brnzdec)
    add  $MATRIX_LOOP_COUNT, $mSCRATCH2, -1

    //Can we use the fast version? Only if CHANS_PG (weight count) even and >=6
    and   $mSCRATCH, $CHANS_PG,1
    brnz  $mSCRATCH, .Lfloat_slower

    add   $mSCRATCH, $CHANS_PG, -6
    brneg $mSCRATCH, .Lfloat_slower

    // use $mSCRATCH to adjust the weights pointer to the next group at the loop
    // end in the fast version. It's in units of bytes
    // The net increment would need to be
    // (sizeof float) * CHANS_PG, but it was incremented by 2 64 bit reads
    // at the start of the loop
    // so the adjusted increment is 4 * (CHANS_PG - 4)
    add  $mSCRATCH, $CHANS_PG, -(2 * FLOATS_PER_64BITS)
    shl  $mSCRATCH, $mSCRATCH, 2

    // stride back for weights addressing
    // = -(weights per group/weights per 64 bits -1)
    // $RPT_COUNT=weights per group/weights per 64 bits - 3
    // - there are 3 groups of 64 bits processed outside the rpt loop
    shr      $CHANS_PG, $CHANS_PG, LOG2_FLOATS_PER_64BITS
    add      $RPT_COUNT, $CHANS_PG, -(6 / FLOATS_PER_64BITS)
    sub      $CHANS_PG, 1, $CHANS_PG

    //If stride back or repeat count are too big then we need the slow version
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
    cmpult    $mSCRATCH2, $RPT_COUNT, (CSR_W_REPEAT_COUNT__VALUE__MASK + 1)
#else
    // RPT_COUNT can be greater than 16 bits so we need to load into a register
    ldconst   $mSCRATCH2, CSR_W_REPEAT_COUNT__VALUE__MASK + 1
    cmpult    $mSCRATCH2, $RPT_COUNT, $mSCRATCH2
#endif
    brz       $mSCRATCH2, .Lfloat_slower_restore_chans_pg

    cmpslt    $mSCRATCH2, $CHANS_PG, -512
    brnz      $mSCRATCH2, .Lfloat_slower_restore_chans_pg

    OUTER_PRODUCT_FAST ld32step f32v2mul
    exitz   $mzero

.Lfloat_slower_restore_chans_pg:
    sub      $CHANS_PG, 1, $CHANS_PG
    shl      $CHANS_PG, $CHANS_PG, LOG2_FLOATS_PER_64BITS
.Lfloat_slower:
     shl      $WEIGHTS_REWIND, $CHANS_PG,LOG2_SIZEOF_FLOAT
     OUTER_PRODUCT_SLOWER ld32step 0
     exitz   $mzero

.size __runCodelet_poplin__OuterProduct___float, .-__runCodelet_poplin__OuterProduct___float

//******************************************************************************
// Half outer product
//
// Reads the passed parameters/pointers and decides if the fast version can be
// run. If not, select the slower version. Calculates many of the input
// pointers and counts for the macros to use.
//******************************************************************************

DEF_STACK_USAGE 0 __runCodelet_poplin__OuterProduct___half
.section .text.__runCodelet_poplin__OuterProduct___half
.global __runCodelet_poplin__OuterProduct___half
.type __runCodelet_poplin__OuterProduct___half,   @function


.align 4

__runCodelet_poplin__OuterProduct___half:

     GET_PARAMS  0 1 2 3 4 5
     add  $OUT_ROWS_LOOPS, $mSCRATCH, -1

    // Outermost loop count = number of output matrices-1 (to use brnzdec)
    add  $MATRIX_LOOP_COUNT, $mSCRATCH2, -1

    // Can we use the fast version? Only if a multiple of 4 and >=12
    and   $mSCRATCH, $CHANS_PG, 3
    brnz  $mSCRATCH, .Lhalf_slower

    cmpult   $mSCRATCH, $CHANS_PG, 12
    brnz     $mSCRATCH, .Lhalf_slower

    // use $mSCRATCH to adjust the weights pointer to the next group at the loop
    // end in the fast version. The net increment would need to be
    // (sizeof half) * CHANS_PG, but it was incremented by 2 64 bit reads
    // at the start of the loop
    // so the adjusted increment is 2 * (CHANS_PG - 8)
    add  $mSCRATCH, $CHANS_PG, -(2 * HALVES_PER_64BITS)
    shl  $mSCRATCH, $mSCRATCH, 1

    // stride back for weights addressing
    // =-(weights per group/weights per 64 bits -1)
    // $RPT_COUNT=weights per group/weights per 64 bits -3
    // - there are 3 groups of 64 bits processed outside the rpt loop
    shr      $CHANS_PG, $CHANS_PG, LOG2_HALVES_PER_64BITS
    add      $RPT_COUNT, $CHANS_PG, -(12 / HALVES_PER_64BITS)
    sub      $CHANS_PG, 1, $CHANS_PG


    //If stride back or repeat count are too big then we need the slow version
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
    cmpult    $mSCRATCH2, $RPT_COUNT, (CSR_W_REPEAT_COUNT__VALUE__MASK + 1)
#else
    // RPT_COUNT can be greater than 16 bits so we need to load into a register
    ldconst   $mSCRATCH2, CSR_W_REPEAT_COUNT__VALUE__MASK + 1
    cmpult    $mSCRATCH2, $RPT_COUNT, $mSCRATCH2
#endif
    brz       $mSCRATCH2, .Lhalf_slower_restore_chans_pg

    cmpslt    $mSCRATCH2, $CHANS_PG, -512
    brnz      $mSCRATCH2, .Lhalf_slower_restore_chans_pg

    OUTER_PRODUCT_FAST ldb16step f16v4mul
    exitz   $mzero

    // Shift back to restore CHANS_PG and enter the slower path
.Lhalf_slower_restore_chans_pg:
    sub      $CHANS_PG, 1, $CHANS_PG
    shl      $CHANS_PG, $CHANS_PG, LOG2_HALVES_PER_64BITS
.Lhalf_slower:
    shl      $WEIGHTS_REWIND, $CHANS_PG,LOG2_SIZEOF_HALF
    OUTER_PRODUCT_SLOWER ldb16step 1
    exitz   $mzero
.size __runCodelet_poplin__OuterProduct___half, .-__runCodelet_poplin__OuterProduct___half

#endif
/* -------------------------------------------------------------------------- */
