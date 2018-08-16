#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Contains functions to cast matrix content:
// From float to half
// Other casts dealt with elsewhere (in C++)
/* -------------------------------------------------------------------------- */
#include "tilearch.h"

// Register aliases

#define INOUT_PTR                   m6:7
#define INOUT_PTR_LOWER             m6
#define IN_PTR                      m3
#define OUT_START_PTR               m2
#define OUT_END_PTR                 m4
#define TOTAL_COUNT                 m5

#define mSCRATCH    m0
#define mSCRATCH2   m9
#define STRIDE      m8
#define LOOP_COUNT  m10

#define RESULTv2    a4:5
#define RESULT1     a4
#define RESULT2     a5

#define INPUTv4     a0:3
#define INPUT1v2    a0:1
#define INPUT3v2    a2:3
#define INPUT1      a0
#define INPUT2      a1
#define INPUT3      a2
#define INPUT4      a3
#define aSCRATCH    a6


//******************************************************************************
// The input structure is always the same so a macro can be used to 
// fetch the parameters.
//******************************************************************************

// Defines added for clarity later in the code, and in case the vertex state is
// re-read later 
#define VOFF_IN         0
#define VOFF_OUT_START  1
#define VOFF_OUT_END    2

.macro GET_PARAMS p1 p2 p3
    ld32     $IN_PTR,        $mzero, $mvertex_base, \p1          
    ld32     $OUT_START_PTR, $mzero, $mvertex_base, \p2   
    ld32     $OUT_END_PTR,   $mzero, $mvertex_base, \p3      
.endm


//******************************************************************************
// Cast float to half.
//
// Assuming input alignment to 8 bytes, output aligned to 4 bytes. 
// This means we can make an optimal inner loop which deals with 4 items in 2 
// cycles, relying on alignment to 8 byte/ 64 bit boundaries.  As there is
// just 1 input, one output and due to the size of the operands it is
// fairly simple to cope with an output that is only aligned to 32 bits, so this
// is done to provide a little more flexibility.
//******************************************************************************
#define FLOAT_HALF_FUNC __runCodelet_popops__Cast___float_half
.globl FLOAT_HALF_FUNC
.type FLOAT_HALF_FUNC, @function
.section .text.FLOAT_HALF_FUNC
.align 8
FLOAT_HALF_FUNC:

    GET_PARAMS VOFF_IN VOFF_OUT_START VOFF_OUT_END
    
    // If output not 8 byte aligned, process 2 items, input will still be
    // 8 byte aligned (2 floats), output will become 8 byte aligned.
    and         $mSCRATCH, $OUT_START_PTR, 4
    brz         $mSCRATCH, 1f
    // Initial stage to align
    // Either deal with a pair of items and fall through if > 2 to do, else
    // process 1 or 2 and return.
    // calculate loop count - number of items to process
    sub         $mSCRATCH, $OUT_END_PTR, $OUT_START_PTR
    shr         $mSCRATCH, $mSCRATCH, 1
    // Compare: if <2 to process
    cmpult      $mSCRATCH2, $mSCRATCH, 2
    brz         $mSCRATCH2, 2f
    // 1 to process - set up registers and exit via the path that processes 1
    ld32        $INPUT1,  $IN_PTR, 0
    mov         $mSCRATCH, $OUT_START_PTR
    bri         .Lrem1
2:
    ld64step    $INPUT1v2, $mzero, $IN_PTR+=,1
    f32v2tof16  $RESULT1, $INPUT1v2
    st32step    $RESULT1, $mzero, $OUT_START_PTR+=,1

    // Compare: if only 2 to process
    cmpult      $mSCRATCH2, $mSCRATCH, 3
    brz         $mSCRATCH2, 1f
    exitz       $mzero
1:
    // calculate loop count - number of items to process (after possible 1st 2)
    sub         $LOOP_COUNT, $OUT_END_PTR, $OUT_START_PTR

    // 2 bytes per word: div by 2, gives total items to process
    // 4 items per loop, so a further div by 4
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
    tapack      $INOUT_PTR, $mSCRATCH,$mzero, $OUT_START_PTR
    // Small stride so that the offset in pointers can be maintained
    setzi       $STRIDE, 2

    // Process 4 items per loop, but it's using rpt so check the size
    // Note - use a size smaller than CSR_W_REPEAT_COUNT__VALUE__MASK to test?
3:
    mov        $LOOP_COUNT, $TOTAL_COUNT
    cmpult     $mSCRATCH, $LOOP_COUNT, (CSR_W_REPEAT_COUNT__VALUE__MASK +1)
    brnz       $mSCRATCH, 5f
    setzi      $LOOP_COUNT, (CSR_W_REPEAT_COUNT__VALUE__MASK +1)
5:
    sub        $TOTAL_COUNT, $TOTAL_COUNT, $LOOP_COUNT

    {rpt $LOOP_COUNT,((2f-1f)/8)-1
        fnop}
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
    sub         $LOOP_COUNT, $OUT_END_PTR, $OUT_START_PTR
    shr         $TOTAL_COUNT, $LOOP_COUNT, 3

.Lless_than_4_total:
    shr         $LOOP_COUNT, $LOOP_COUNT, 1
    and         $LOOP_COUNT, $LOOP_COUNT, 3

    // Generate a non- packed address for storage of trailing 1-3 items
    // Using TOTAL_COUNT - number of 4s dealt with above
    // Note TOTAL_COUNT intentionally shifted right then left!
    shl         $TOTAL_COUNT, $TOTAL_COUNT, 3
    add         $mSCRATCH, $TOTAL_COUNT, $OUT_START_PTR

    // decision on the last 0-3 items
    brnzdec     $LOOP_COUNT, .Lremn0

    // 0 - return
    exitz   $mzero

.Lremn0:
    brnzdec     $LOOP_COUNT, .Lremn1

    // 1 left to do
    bri          .Lrem1
.Lremn1:
    brnzdec     $LOOP_COUNT, .Lremn2

    // 2 left, they are already loaded
    f32v2tof16   $RESULT1, $INPUT1v2
    st32step     $RESULT1, $mzero, $mSCRATCH+=, 1
    exitz        $mzero
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
    exitz         $mzero

.size FLOAT_HALF_FUNC, .-FLOAT_HALF_FUNC
#endif
/* -------------------------------------------------------------------------- */
