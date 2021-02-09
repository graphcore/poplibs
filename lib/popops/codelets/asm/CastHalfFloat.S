// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Contains functions to cast matrix content:
// From float to half
// From half to float
/* -------------------------------------------------------------------------- */
#include "CastCommon.inc"


// Register aliases

#define INOUT_PTR                   m6:7
#define INOUT_PTR_LOWER             m6
#define TOTAL_COUNT                 m5

#define mSCRATCH      m0
#define mSCRATCH2     m9
#define STRIDE        m8
#define LOOP_COUNT    m11

#define RESULTv2    a4:5
#define RESULT1     a4
#define RESULT2     a5

#define INPUTv4     a0:3
#define INPUT1v2    a0:1
#define INPUT3v2    a2:3
#define INPUT1      a0
#define INPUT2      a1
#define INPUT3      a2



//******************************************************************************
// Cast float to half core
//
// Assuming input alignment to 8 bytes, output aligned to 4 bytes.
// This means we can make an optimal inner loop which deals with 4 items in 2
// cycles, relying on alignment to 8 byte/ 64 bit boundaries.  As there is
// just 1 input, one output and due to the size of the operands it is
// fairly simple to cope with an output that is only aligned to 32 bits, so this
// is done to provide a little more flexibility.
//******************************************************************************
.section .text.cast_float_half_core
EXPORT_FN cast_float_half_core
.align 8
.worker
cast_float_half_core:
    // If output not 8 byte aligned, process 2 items, input will still be
    // 8 byte aligned (2 floats), output will become 8 byte aligned.
    and         $mSCRATCH, $OUT_PTR, 4
    brz         $mSCRATCH, 1f
    // Initial stage to align
    // Either deal with a pair of items and fall through if > 2 to do, else
    // process 1 or 2 and return.
    // Compare: if <2 to process
    cmpult      $mSCRATCH2, $N_ELEMS, 2
    brz         $mSCRATCH2, 2f
    // 1 to process - set up registers and exit via the path that processes 1
    ld32        $INPUT1,  $IN_PTR, 0
    mov         $mSCRATCH, $OUT_PTR
    bri         .Lrem1
2:
    ld64step    $INPUT1v2, $mzero, $IN_PTR+=,1
    {
      // update $N_ELEMS to reflect the two elements we are processing now.
      sub         $N_ELEMS, $N_ELEMS, 2
      f32v2tof16  $RESULT1, $INPUT1v2
    }
    st32step    $RESULT1, $mzero, $OUT_PTR+=,1

    // Compare: if only 2 to process (note, N_ELEMS has already been reduced
    // by 2).
    cmpult      $mSCRATCH2, $N_ELEMS, 1
    brz         $mSCRATCH2, 1f
    br          $lr
1:
    // calculate loop count - number of items to process (after possible 1st 2)
    shl         $LOOP_COUNT, $N_ELEMS, 1

    // 4 items per loop, so a div by 4
    shr         $TOTAL_COUNT, $LOOP_COUNT, 3
    brnz        $TOTAL_COUNT,4f

    // < 4 floats to convert so prepare registers
    // as they would be after the loop when completing the last 3
    // and jump into that code
    ld64step    $INPUT1v2,$mzero, $IN_PTR+=,1
    add         $IN_PTR, $IN_PTR, 16
    bri         .Lless_than_4_total
4:
    // Get 1st input pair, make a new pointer to the 3rd pair
    ld64step    $INPUT1v2,$mzero, $IN_PTR+=,1
    add         $mSCRATCH, $IN_PTR, 8

    // load the 2nd input pair and point to the 4th input pair
    ld64step    $INPUT3v2,$mzero, $IN_PTR+=,2
    // Pack pointers - to use for all outputs, and input pairs 3, 5, 7 ...
    // IN_PTR will be used to read input pairs 4, 6, 8 ..
    tapack      $INOUT_PTR, $mSCRATCH,$mzero, $OUT_PTR
    // Small stride so that the offset in pointers can be maintained
    setzi       $STRIDE, 2

    // Process 4 items per loop, but it's using rpt so check the size
    // Note - use a size smaller than CSR_W_REPEAT_COUNT__VALUE__MASK to test?
3:
    // TOTAL_COUNT derives from LOOP_COUNT, from N_ELEMS which is 32 bits.
    // The interim operations don't currently put it in the range of 16 bits
    // so we need to make sure we don't exceed loop count for RPT
    min        $LOOP_COUNT, $TOTAL_COUNT, CSR_W_REPEAT_COUNT__VALUE__MASK
    sub        $TOTAL_COUNT, $TOTAL_COUNT, $LOOP_COUNT

    rpt $LOOP_COUNT,((2f-1f)/8)-1
1:
    {ld64step   $INPUT3v2, $mzero, $IN_PTR+=,2
     f32v4tof16 $RESULTv2, $INPUTv4}
     // Stride: 1= 0001 bin.  00: result_ptr++.  01: source_ptr+=$STRIDE
    {ldst64pace $INPUT1v2, $RESULTv2, $INOUT_PTR+=,$STRIDE, 1
     fnop}
2:
    brnz        $TOTAL_COUNT, 3b

    // Done all groups of 4 - how many are left?
    // Pointers are untouched, so recalculate total words to process
    // Gather TOTAL_COUNT - recreated loop count from above
    shl         $LOOP_COUNT, $N_ELEMS, 1
    shr         $TOTAL_COUNT, $LOOP_COUNT, 3

.Lless_than_4_total:
    shr         $LOOP_COUNT, $LOOP_COUNT, 1
    and         $LOOP_COUNT, $LOOP_COUNT, 3

    // Generate a non- packed address for storage of trailing 1-3 items
    // Using TOTAL_COUNT - number of 4s dealt with above
    // Note TOTAL_COUNT intentionally shifted right then left!
    shl         $TOTAL_COUNT, $TOTAL_COUNT, 3
    add         $mSCRATCH, $TOTAL_COUNT, $OUT_PTR

    // decision on the last 0-3 items
    brnzdec     $LOOP_COUNT, .Lremn0

    // 0 - return
    br          $lr

.Lremn0:
    brnzdec     $LOOP_COUNT, .Lremn1

    // 1 left to do
    bri          .Lrem1
.Lremn1:
    brnzdec     $LOOP_COUNT, .Lremn2

    // 2 left, they are already loaded
    f32v2tof16   $RESULT1, $INPUT1v2
    st32step     $RESULT1, $mzero, $mSCRATCH+=, 1
    br           $lr
.Lremn2:

    // 3 left to do - loaded 2 already
    // Now, to cover every case we need to read again
    // Adjust the read pointer back to fetch the last one
    { add         $IN_PTR, $IN_PTR, -4*4
      f32v2tof16  $RESULT1, $INPUT1v2}
    st32step      $RESULT1, $mzero, $mSCRATCH+=, 1

    // and the last of 3
    ld32          $INPUT1, $mzero, $IN_PTR,0
.Lrem1:
    // Common code: last 1 and last of 3
    { ldb16       $INPUT1, $mSCRATCH, +1
      f32tof16    $RESULT1, $INPUT1}
    roll16        $RESULT1,$RESULT1, $INPUT1
    st32          $RESULT1, $mzero, $mSCRATCH,0
    br            $lr

FN_SIZE cast_float_half_core


//******************************************************************************
// Cast float to half core
//
// Assuming Input and Output alignment are both 8 bytes this function
// has an optimal inner loop.  No particular critical use cases have been
// identified so this constraint has not been seen to have any consequence.
//******************************************************************************
.section .text.cast_half_float_core
EXPORT_FN cast_half_float_core
.align 8
cast_half_float_core:
    mov         $TOTAL_COUNT, $N_ELEMS

    // Total number of items to process, is it <4 ?
    cmpult      $mSCRATCH, $TOTAL_COUNT, 4
    brz         $mSCRATCH,1f

    // Setup pointers and load input to suit the code that deals with the
    // last 3 items below, and branch there, so we deal with up to 3 items the
    // same way as the trailing 3.
    add         $OUT_PTR, $OUT_PTR, 8
    ld64step    $INPUT1v2,$mzero, $IN_PTR+=,1
    bri         .Lless_than_4_total_hf
1:
    // Remember the output start pointer for later
    mov         $mSCRATCH, $OUT_PTR
    // total number of inner loops, each processing 4
    shr         $TOTAL_COUNT, $TOTAL_COUNT, 2

    // Ready the loop pipeline by loading one
    ld64step    $INPUT1v2,$mzero, $IN_PTR+=,1

    // Pack addresses and set a small stride -
    // Alternate instructions write alternate outputs.
    tapack      $INOUT_PTR, $IN_PTR,$mzero, $OUT_PTR
    setzi       $STRIDE, 2
    add         $OUT_PTR, $OUT_PTR, 8

3:
    min         $LOOP_COUNT,$TOTAL_COUNT, CSR_W_REPEAT_COUNT__VALUE__MASK
    sub         $TOTAL_COUNT, $TOTAL_COUNT, $LOOP_COUNT
    // 1 less loop to avoid any over processing
    {add         $LOOP_COUNT, $LOOP_COUNT, -1
     f16v2tof32  $RESULTv2, $INPUT1}

    // Ready the loop pipeline by converting one bundled with repeat
    {rpt         $LOOP_COUNT,((2f-1f)/8)-1
     fnop}
1:
    // Load/store: Stride the input pointer by 1(64 bit step), and the output by
    // 2(64 bit steps)
    {ldst64pace  $INPUT1v2, $RESULTv2, $INOUT_PTR+=,$STRIDE, 4
     f16v2tof32  $RESULTv2, $INPUT2}
    {st64step    $RESULTv2,$mzero, $OUT_PTR+=,2
     f16v2tof32  $RESULTv2, $INPUT1}
2:
    // Store the last 2 that were converted, avoiding over-processing
    // Stride the input pointer by 1(64 bit step), and the output by
    // 2(64 bit steps)
    {ldst64pace  $INPUT1v2, $RESULTv2, $INOUT_PTR+=,$STRIDE, 4
     f16v2tof32  $RESULTv2, $INPUT2}
    st64step     $RESULTv2, $mzero, $OUT_PTR+=,2

    brnz         $TOTAL_COUNT, 3b

    // use ptr to start/end of the output array to check how many are left
    and         $TOTAL_COUNT, $N_ELEMS, 3

.Lless_than_4_total_hf:
    brnzdec     $TOTAL_COUNT, .Lhf_nrem0

    // 0 left: nothing to do
    br          $lr
.Lhf_nrem0:
    add         $OUT_PTR, $OUT_PTR, -8
    brnzdec     $TOTAL_COUNT, .Lhf_nrem1

    // 1 left, convert just 1 and store
    f16tof32    $RESULT1, $INPUT1
    st32        $RESULT1, $mzero, $OUT_PTR, 0
     br         $lr
.Lhf_nrem1:
   brnzdec     $TOTAL_COUNT, .Lhf_nrem2

    // 2 left - convert 2 and store
    f16v2tof32 $RESULTv2, $INPUT1
    st64       $RESULTv2, $mzero, $OUT_PTR, 0
    br         $lr
.Lhf_nrem2:

    // 3 left
    // Convert 3 of the 4 we already read and store them
    f16v2tof32  $RESULTv2, $INPUT1
    { st64step  $RESULTv2, $mzero, $OUT_PTR+=, 1
    f16tof32    $RESULT1, $INPUT2}
    st32        $RESULT1, $mzero, $OUT_PTR, 0

     br         $lr
FN_SIZE cast_half_float_core

#endif
/* -------------------------------------------------------------------------- */
