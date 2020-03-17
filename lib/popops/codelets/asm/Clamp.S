// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
//***************************************************************************************************
// Clamp: float and half type versions
//***************************************************************************************************
#ifdef __IPU__

#include "poplar/StackSizeDefs.hpp"

.globl __runCodelet_popops__Clamp___float
.type __runCodelet_popops__Clamp___float, @function

.globl __runCodelet_popops__BroadcastClamp___float
.type __runCodelet_popops__BroadcastClamp___float, @function

.globl __runCodelet_popops__Clamp___half
.type __runCodelet_popops__Clamp___half, @function

.globl __runCodelet_popops__BroadcastClamp___half
.type __runCodelet_popops__BroadcastClamp___half, @function


/* Input vertex structure offsets */
#define VOFF_IN1_PTR                0
#define VOFF_IN2_PTR                1
#define VOFF_IN3_PTR                2
#define VOFF_OUT_START_PTR          3
#define VOFF_OUT_SIZE_PTR           4

// Register aliases
// Registers for each of the passed parameters
#define IN1_PTR                    m1
#define IN2_PTR                    m2
#define IN3_PTR                    m3
#define OUT_START_PTR              m4

#define mSCRATCH m0

#define CLAMP_LOW     a0       //intended register pair for clamp range input
#define CLAMP_HIGH    a1
#define CLAMP_PAIR    a0:1     //MUST BE CONSISTENT WITH CLAMP_LOW AND CLAMP_HIGH
#define CLAMP_IN      a2
#define CLAMP_IN2     a3
#define CLAMP_RESULT  a4
#define CLAMP_RESULT2 a5       //result of 2nd clamp in a loop which processed 2 items
#define aSCRATCH      a6

#define IN1 m7                 //inner loop pointers
#define IN2 m8
#define IN3 m9
#define OUT m6

#define OUTER_LOOP_COUNT m11
#define INNER_LOOP_COUNT m10
//***************************************************************************************************
// Float version, using clamp instruction to do all calculation.
// Due to there being 3 input arrays and 1 output array dealt with in the inner optimised loop,
// alignment of data can be complex if we try to optimise further to process 2 items at once.
// In other words, in1,in2,in3 or out may or may not be 64 bit aligned.  An optimal loop could be written
// if they were all aligned, but dealing with the cases where some are, some aren't is complex.
// Maybe an "all aligned" version could be an option?
//
// Benchmark: 8+ vectors*(12+2) +(total_inputs_in_all_vectors-vectors)*4
//***************************************************************************************************

DEF_STACK_USAGE  0  __runCodelet_popops__Clamp___float
.section .text.__runCodelet_popops__Clamp___float
.align 8

clamp_prepad:
  nop // rpt alignment
__runCodelet_popops__Clamp___float:
    // load vertex state
    ld32     $IN1_PTR,       $mzero, $mvertex_base, VOFF_IN1_PTR
    ld32     $IN2_PTR,       $mzero, $mvertex_base, VOFF_IN2_PTR
    ld32     $IN3_PTR,       $mzero, $mvertex_base, VOFF_IN3_PTR
    ld32     $OUT_START_PTR, $mzero, $mvertex_base, VOFF_OUT_START_PTR
    ld32     $OUTER_LOOP_COUNT,   $mzero, $mvertex_base, VOFF_OUT_SIZE_PTR
    add      $OUTER_LOOP_COUNT,$OUTER_LOOP_COUNT,-1

clamp_loop_outer:
    // load all input data pointers
    ld32step    $IN1,$mzero,$IN1_PTR+=,1
    ld32step    $IN2,$mzero,$IN2_PTR+=,1
    ld32step    $IN3,$mzero,$IN3_PTR+=,1

    ld32step    $OUT,$mzero,$OUT_START_PTR+=,1
    ld32step    $INNER_LOOP_COUNT,$mzero,$OUT_START_PTR+=,1
    // one less loop as unrolled for one pass
    add         $INNER_LOOP_COUNT,$INNER_LOOP_COUNT,-1

    //unrolled loop portion: fetch and clamp the first entry
    ld32step    $CLAMP_IN,$mzero,$IN1+=,1
    ld32step    $CLAMP_LOW,$mzero,$IN2+=,1
    ld32step    $CLAMP_HIGH,$mzero,$IN3+=,1

    // Inner loop - note that there are vectorised clamp instructions, but they apply
    // the same max,min to input[0],input[1] etc, which is not what this function is
    // required to do.
    {rpt $INNER_LOOP_COUNT,((clamp_loop_inner_end-clamp_loop_inner)/8) -1
     f32clamp    $CLAMP_RESULT,$CLAMP_IN,$CLAMP_PAIR }
clamp_loop_inner:
        {ld32step    $CLAMP_IN,$mzero,$IN1+=,1
         fnop}
        {ld32step    $CLAMP_LOW,$mzero,$IN2+=,1
         fnop}
        {ld32step    $CLAMP_HIGH,$mzero,$IN3+=,1
         fnop}
        // Write previous output
        {st32step    $CLAMP_RESULT,$mzero,$OUT+=,1
         f32clamp    $CLAMP_RESULT,$CLAMP_IN,$CLAMP_PAIR }

clamp_loop_inner_end:
    //unrolled loop portion: store the last one
    st32step    $CLAMP_RESULT,$mzero,$OUT+=,1
    brnzdec     $OUTER_LOOP_COUNT,clamp_loop_outer

    exitz    $mzero

.size __runCodelet_popops__Clamp___float, .-__runCodelet_popops__Clamp___float

//***************************************************************************************************
// Broadcast float version, using clamp instruction to do all calculation.
//
// Benchmark: 10 + vectors*(10+2) + (total_inputs_in_all_vectors-vectors)*2
//***************************************************************************************************

DEF_STACK_USAGE  0  __runCodelet_popops__BroadcastClamp___float
.section .text.__runCodelet_popops__BroadcastClamp___float
.align 8

bclamp_prepad:

  nop // rpt alignment
__runCodelet_popops__BroadcastClamp___float:
    // Load vertex state
    ld32     $IN1_PTR,       $mzero, $mvertex_base, VOFF_IN1_PTR
    ld32     $IN2_PTR,       $mzero, $mvertex_base, VOFF_IN2_PTR
    ld32     $IN3_PTR,       $mzero, $mvertex_base, VOFF_IN3_PTR
    ld32     $OUT_START_PTR, $mzero, $mvertex_base, VOFF_OUT_START_PTR
    ld32     $OUTER_LOOP_COUNT,   $mzero, $mvertex_base, VOFF_OUT_SIZE_PTR
    //decrement, to work with the brnzdec instruction post decrement
    add      $OUTER_LOOP_COUNT,$OUTER_LOOP_COUNT,-1

    // Read the scalar low and high values, which are constant throughout the loop below
    ld32     $CLAMP_LOW,$mzero,$IN2_PTR,0
    ld32     $CLAMP_HIGH,$mzero,$IN3_PTR,0

bclamp_loop_outer:
    // Load in/out data ptrs, loop count, pointed to by the vertex state
    ld32step    $IN1,$mzero,$IN1_PTR+=,1

    ld32step    $OUT,$mzero,$OUT_START_PTR+=,1
    ld32step    $INNER_LOOP_COUNT,$mzero,$OUT_START_PTR+=,1
    add         $INNER_LOOP_COUNT,$INNER_LOOP_COUNT,-1

    //unrolled loop portion: fetch and clamp the first entry
    ld32step    $CLAMP_IN,$mzero,$IN1+=,1

    //Inner loop - takes one input element and use clamp instruction to
    // to bound it to connent of $CLAMP_PAIR(LOWER:UPPER) registers.
    {rpt $INNER_LOOP_COUNT,((bclamp_loop_inner_end-bclamp_loop_inner)/8) -1
     f32clamp    $CLAMP_RESULT,$CLAMP_IN,$CLAMP_PAIR }
bclamp_loop_inner:
        {ld32step    $CLAMP_IN,$mzero,$IN1+=,1
         fnop}
        //write PREVIOUS output
        {st32step    $CLAMP_RESULT,$mzero,$OUT+=,1
         f32clamp    $CLAMP_RESULT,$CLAMP_IN,$CLAMP_PAIR }

bclamp_loop_inner_end:
    //unrolled loop portion: store the last one
    st32step    $CLAMP_RESULT,$mzero,$OUT+=,1
    brnzdec     $OUTER_LOOP_COUNT,bclamp_loop_outer

    exitz    $mzero

.size __runCodelet_popops__BroadcastClamp___float, .-__runCodelet_popops__BroadcastClamp___float

//***************************************************************************************************
// Half version, using clamp instruction to do all calculation.
// Due to there being 3 input arrays and 1 output array dealt with in the inner optimised loop,
// alignment of data can be complex if we try to optimise further to process 2 items at once.
// In other words, in1,in2,in3 or out may or may not be 32 bit aligned.  An optimal loop could be written
// if they were all aligned, but dealing with the cases where some are, some aren't is complex.
// Maybe an "all aligned" version could be an option?
//
// Benckmark (aligned)= 8 + vectors*(14+2) + total_inputs_in_all_vectors * 7/2
//***************************************************************************************************

DEF_STACK_USAGE  0  __runCodelet_popops__Clamp___half
.section .text.__runCodelet_popops__Clamp___half
.align 8

clamp16_prepad:
  nop // pad for rpt loop alignment
__runCodelet_popops__Clamp___half:
    // load vertex state
    ld32     $IN1_PTR,       $mzero, $mvertex_base, VOFF_IN1_PTR
    ld32     $IN2_PTR,       $mzero, $mvertex_base, VOFF_IN2_PTR
    ld32     $IN3_PTR,       $mzero, $mvertex_base, VOFF_IN3_PTR
    ld32     $OUT_START_PTR, $mzero, $mvertex_base, VOFF_OUT_START_PTR
    ld32     $OUTER_LOOP_COUNT,   $mzero, $mvertex_base, VOFF_OUT_SIZE_PTR
    // decrement, to work with the brnzdec instruction post decrement
    add         $OUTER_LOOP_COUNT,$OUTER_LOOP_COUNT,-1

clamp16_loop_outer:
    // Load input vector pointers
    ld32step    $IN1,$mzero,$IN1_PTR+=,1
    ld32step    $IN2,$mzero,$IN2_PTR+=,1
    ld32step    $IN3,$mzero,$IN3_PTR+=,1
    // And output pointer/count
    ld32step    $OUT,$mzero,$OUT_START_PTR+=,1
    ld32step    $INNER_LOOP_COUNT,$mzero,$OUT_START_PTR+=,1

    //Note that we have only a 32 bit (no 16 bit) write instruction which we use to write the output.
    //Therefore we have to deal with 3 things based on output array alignment and data length:
    //Choose to write an inner loop that processes pairs of 16bit words for efficiency.
    //If the 1st word is not 32 bit aligned we need to treat that as a special case
    //If the last word is not part of a 32bit aligned pair it needs treating as a special case
    // We need to use the loop count and start address to determine if we have these special cases,
    // not forgetting the case of no loops

    //test the bit which will indicate a non alignment in the address
    and         $mSCRATCH,$OUT,0x2
    brz         $mSCRATCH,clamp16_start_aligned
    //Deal with a single first word which isn't 32 bit aligned
clamp_do_misaligned:
        andc         $OUT,$OUT,0x3
        // Get inputs, combine
        ldb16step   $CLAMP_LOW,$mzero,$IN2+=,1
        ldb16step   $CLAMP_HIGH,$mzero,$IN3+=,1
        {ldb16step  $CLAMP_IN,$mzero,$IN1+=,1
         sort4x16lo $CLAMP_HIGH,$CLAMP_LOW,$CLAMP_HIGH }
         //load the output word we are about to overwrite to combine with the half just clamped
        {ld32       $aSCRATCH,$mzero,$OUT,0
         f16v2clamp $CLAMP_RESULT,$CLAMP_IN,$CLAMP_HIGH }
        //adjust count:one less word to process
        {add         $INNER_LOOP_COUNT,$INNER_LOOP_COUNT,-1
         sort4x16lo  $CLAMP_RESULT,$aSCRATCH,$CLAMP_RESULT}
        st32step    $CLAMP_RESULT,$mzero,$OUT+=,1
clamp16_start_aligned:
    //test the bit which will indicate an odd number of items left to process
    and         $mSCRATCH,$INNER_LOOP_COUNT,0x1
    //2 items per inner loop
    shr         $INNER_LOOP_COUNT,$INNER_LOOP_COUNT,1
    // pre-fetch inputs
    ldb16step   $CLAMP_LOW,$mzero,$IN2+=,1
    ldb16step   $CLAMP_HIGH,$mzero,$IN3+=,1

    //Inner loop: Processes 2 inputs at once, which are combined in a 32 bit word before being written to avoid memory read modify write
    //(there is no 16 bit write)
    //Note that we are using a vectorised f16v2clamp instruction but the instruction applies the same max,min to input[0] and input[1],
    //which is not what this function is required to do.  So we only use one of the two results generated.
    {rpt $INNER_LOOP_COUNT,((clamp16_loop_inner_end-clamp16_loop_inner)/8) -1
     fnop }
clamp16_loop_inner:
        {ldb16step    $CLAMP_IN,$mzero,$IN1+=,1
         sort4x16lo   $CLAMP_HIGH,$CLAMP_LOW,$CLAMP_HIGH }//Combine lower word of in2, in3
        {ldb16step    $CLAMP_LOW,$mzero,$IN2+=,1
         f16v2clamp   $CLAMP_RESULT,$CLAMP_IN,$CLAMP_HIGH }

        {ldb16step    $CLAMP_HIGH,$mzero,$IN3+=,1
         fnop}
        {ldb16step    $CLAMP_IN,$mzero,$IN1+=,1
         sort4x16lo   $CLAMP_HIGH,$CLAMP_LOW,$CLAMP_HIGH }//Combine lower word of in2,in3
        {ldb16step    $CLAMP_LOW,$mzero,$IN2+=,1
         f16v2clamp   $CLAMP_RESULT2,$CLAMP_IN,$CLAMP_HIGH }
        //Fetch in3 for 1st clamp in the loop
        {ldb16step    $CLAMP_HIGH,$mzero,$IN3+=,1
         sort4x16lo   $CLAMP_RESULT,$CLAMP_RESULT,$CLAMP_RESULT2}//combine output words
        {st32step     $CLAMP_RESULT,$mzero,$OUT+=,1
         fnop}
clamp16_loop_inner_end:

      //Deal with a single last word, noting that low and high bounds are already fetched but input isn't
      //condition for odd number of items left after alignment
      brz          $mSCRATCH,clamp16_loop_outer_end

        {ldb16step    $CLAMP_IN,$mzero,$IN1+=,1
         sort4x16lo   $CLAMP_HIGH,$CLAMP_LOW,$CLAMP_HIGH }
         //load the output word we are about to overwrite
        {ld32         $aSCRATCH,$mzero,$OUT,0
         f16v2clamp    $CLAMP_RESULT,$CLAMP_IN,$CLAMP_HIGH }

        sort4x16hi   $CLAMP_RESULT,$CLAMP_RESULT,$aSCRATCH
        //write output, 1 word, preserve next word
        st32step     $CLAMP_RESULT,$mzero,$OUT+=,1
clamp16_loop_outer_end:
    brnzdec     $OUTER_LOOP_COUNT,clamp16_loop_outer

    exitz    $mzero

.size __runCodelet_popops__Clamp___half, .-__runCodelet_popops__Clamp___half

//***************************************************************************************************
// Broadcast half version, using clamp instruction to do all calculation.
//
// Benckmark (aligned)= 11 + vectors*(15+2) + total_inputs_in_all_vectors * 3
//***************************************************************************************************

DEF_STACK_USAGE  0  __runCodelet_popops__BroadcastClamp___half
.section .text.__runCodelet_popops__BroadcastClamp___half
.align 8

bclamp16_prepad:
__runCodelet_popops__BroadcastClamp___half:
    // load vertex state
    ld32     $IN1_PTR,       $mzero, $mvertex_base, VOFF_IN1_PTR
    ld32     $IN2_PTR,       $mzero, $mvertex_base, VOFF_IN2_PTR
    ld32     $IN3_PTR,       $mzero, $mvertex_base, VOFF_IN3_PTR
    ld32     $OUT_START_PTR, $mzero, $mvertex_base, VOFF_OUT_START_PTR
    ld32     $OUTER_LOOP_COUNT,   $mzero, $mvertex_base, VOFF_OUT_SIZE_PTR
    //decrement, to work with the brnzdec instruction post decrement
    add      $OUTER_LOOP_COUNT,$OUTER_LOOP_COUNT,-1

    // Read the scalar low and high values, which are constant throughout the loop below
    ldb16       $CLAMP_LOW,$mzero,$IN2_PTR,0
    ldb16       $CLAMP_HIGH,$mzero,$IN3_PTR,0
    //Combine lower word of in2, in3
    sort4x16lo  $CLAMP_HIGH,$CLAMP_LOW,$CLAMP_HIGH

bclamp16_loop_outer:
    // load data pointers using vertex state
    ld32step    $IN1,$mzero,$IN1_PTR+=,1
    ld32step    $OUT,$mzero,$OUT_START_PTR+=,1
    ld32step    $INNER_LOOP_COUNT,$mzero,$OUT_START_PTR+=,1

    //Note that we have only a 32 bit (no 16 bit) write instruction which we use to write the output.
    //Therefore we have to deal with 3 things based on output array alignment and data length:
    //Choose to write an inner loop that processes pairs of 16bit words for efficiency.
    //If the 1st word is not 32 bit aligned we need to treat that as a special case
    //If the last word is not part of a 32bit aligned pair it needs treating as a special case
    // We need to use the loop count and start address to determine if we have these special cases,
    // not forgetting the case of no loops

    //test the bit which will indicate a non alignment in the address
    and         $mSCRATCH,$OUT,0x2
    brz         $mSCRATCH,bclamp16_start_aligned
    //Deal with a single first word which isn't 32 bit aligned
bclamp_do_misaligned:
    andc         $OUT,$OUT,0x3
    ldb16step    $CLAMP_IN,$mzero,$IN1+=,1
    //load the output word we are about to overwrite
    {ld32        $aSCRATCH,$mzero,$OUT,0
     f16v2clamp  $CLAMP_RESULT,$CLAMP_IN,$CLAMP_HIGH }
    //adjust count:one less word to process
    {add         $INNER_LOOP_COUNT,$INNER_LOOP_COUNT,-1
     sort4x16lo  $CLAMP_RESULT,$aSCRATCH,$CLAMP_RESULT}
    st32step     $CLAMP_RESULT,$mzero,$OUT+=,1

bclamp16_start_aligned:
//test the bit which will indicate an odd number of items left to process
    and         $mSCRATCH,$INNER_LOOP_COUNT,0x1
    //2 items per inner loop, -2 as two values will be processed outside inner loop
    shr         $INNER_LOOP_COUNT,$INNER_LOOP_COUNT,1
    brz         $INNER_LOOP_COUNT, bclamp16_loop_outer_tail
    sub         $INNER_LOOP_COUNT,$INNER_LOOP_COUNT,1

//Inner loop: Processes 2 inputs at once, which are combined in a 32 bit word
// before being written to avoid memory read modify write
// (there is no 16 bit write)
//NOTE: We are using a vectorised f16v2clamp instruction which applies min-max
// to both halves of input. That is fine as input has replicated value to both
// halves. Although before store it we need combine two processed inputs together
// to store them as 32bit word. That's done by sort4x16lo command before
// calling st32step

    // preload, as loop unrolled
    ldb16step     $CLAMP_IN,$mzero,$IN1+=,1
    {ldb16step    $CLAMP_IN2,$mzero,$IN1+=,1
     f16v2clamp   $CLAMP_RESULT,$CLAMP_IN,$CLAMP_HIGH}
    {rpt $INNER_LOOP_COUNT,((bclamp16_loop_inner_end-bclamp16_loop_inner)/8) -1
     fnop }

bclamp16_loop_inner:
        {ldb16step    $CLAMP_IN,$mzero,$IN1+=,1              // Load 1st value
         f16v2clamp   $CLAMP_RESULT2,$CLAMP_IN2,$CLAMP_HIGH}  // Clamp 2nd value

        {ldb16step    $CLAMP_IN2,$mzero,$IN1+=,1             // Load 2nd value
         sort4x16lo   $CLAMP_RESULT,$CLAMP_RESULT,$CLAMP_RESULT2 } // Combine 1st and 2nd

        {st32step     $CLAMP_RESULT,$mzero,$OUT+=,1          // Save combined values
         f16v2clamp   $CLAMP_RESULT,$CLAMP_IN,$CLAMP_HIGH}    // Clamp 1st value
bclamp16_loop_inner_end:

    // Process remaining two values
    f16v2clamp   $CLAMP_RESULT2,$CLAMP_IN2,$CLAMP_HIGH      // Clamp 2nd value
    sort4x16lo   $CLAMP_RESULT,$CLAMP_RESULT,$CLAMP_RESULT2 // Combine 1st and 2nd
    st32step     $CLAMP_RESULT,$mzero,$OUT+=,1             // Save combined values

    // Deal with a single last word, noting that low and high bounds and
    // input is clamped so just need to save back to memory
bclamp16_loop_outer_tail:
    brz          $mSCRATCH,bclamp16_loop_outer_end
    // Last single half
    ldb16step    $CLAMP_IN,$mzero,$IN1+=,1
    //load the output word we are about to overwrite
    {ld32        $aSCRATCH,$mzero,$OUT,0
     f16v2clamp  $CLAMP_RESULT,$CLAMP_IN,$CLAMP_HIGH}
    sort4x16hi   $CLAMP_RESULT,$CLAMP_RESULT,$aSCRATCH
    //write output, 1 word, preserve next word
    st32step     $CLAMP_RESULT,$mzero,$OUT+=,1
bclamp16_loop_outer_end:
    brnzdec     $OUTER_LOOP_COUNT,bclamp16_loop_outer

    exitz    $mzero

.size __runCodelet_popops__BroadcastClamp___half, .-__runCodelet_popops__BroadcastClamp___half

#endif
