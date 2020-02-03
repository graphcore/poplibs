// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
// Contains functions to cast matrix content:
// From float to half
// From half to float
// Other casts dealt with elsewhere (in C++)
/* -------------------------------------------------------------------------- */
#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"

// Register aliases

#define INOUT_PTR                   m6:7
#define INOUT_PTR_LOWER             m6
#define IN_PTR                      m3
#define OUT_START_PTR               m2
#define N_ELEMS                     m4
#define TOTAL_COUNT                 m5

#define mSCRATCH      m0
#define mSCRATCH2     m9
#define STRIDE        m8
#define LOOP_COUNT    m10
#define LINK          m11
#define DELTA_LAST    m0
#define WORKER_LAST   m1
#define WORKER_COUNT  m5
#define WORKER_ID     m6
#define OFFS          m7

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
// re-read later. All offsets are in bytes
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
#define VOFF_IN         0
#define VOFF_OUT_START  2
#define VOFF_N_ELEMS    4
#else
#define VOFF_IN         0
#define VOFF_OUT_START  4
#define VOFF_N_ELEMS    8
#endif

.macro GET_PARAMS p1 p2 p3
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16    $IN_PTR,        $mzero, $mvertex_base, \p1 / 2
    ldz16    $OUT_START_PTR, $mzero, $mvertex_base, \p2 / 2
#else
    ld32    $IN_PTR,        $mzero, $mvertex_base, \p1 / 4
    ld32    $OUT_START_PTR, $mzero, $mvertex_base, \p2 / 4
#endif
    ld32     $N_ELEMS,   $mzero, $mvertex_base, \p3 / 4
.endm



//******************************************************************************
// Common code to be run at the beginning of the workers *started by the
// supervisor vertices* (FLOAT->HALF and HALF->FLOAT).
// At the end of this macro the following registers will be populated:
//
//    $IN_PTR  $OUT_START_PTR  -  input and output BASE data pointers: These
//                                need to adjusted by adding $OFFS (see below)
//
//    $N_ELEMS   - number of elements to be processed by this worker.
//
//    $OFFS      - offset (in number of elements) to add to IN_PTR and
//                 $OUT_START_PTR to get the correct start points for this
//                 worker.
// The macro parameter EXIT_LABEL is the label to jump to to terminate the
// codelet.
//******************************************************************************
.macro WORKER_FROM_SUPERVISOR_START EXIT_LABEL
    // get fields from vertex state
    GET_PARAMS VOFF_IN VOFF_OUT_START VOFF_N_ELEMS

    // Extract bit fields from the 'partitionParams' vertex state word (read
    // as $N_ELEMS by GET_PARAMS above)
    and       $DELTA_LAST, $N_ELEMS, 0x7
    shr       $N_ELEMS, $N_ELEMS, 3
    and       $WORKER_LAST, $N_ELEMS, 0x7
    shr       $N_ELEMS, $N_ELEMS, 3
    and       $WORKER_COUNT, $N_ELEMS, 0x7
    shr       $N_ELEMS, $N_ELEMS, 3
    // Get worker ID
    get       $WORKER_ID, $WSR
    and       $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    mul       $OFFS, $WORKER_ID, $N_ELEMS

    // If ID < WORKER_COUNT, N_ELEMS and OFFS are already ok.
    // WORKER_ID is 'rebased' on WORKER_COUNT
    sub        $mSCRATCH2, $WORKER_ID, $WORKER_COUNT
    brneg      $mSCRATCH2, 1f

    // If ID >= WORKER_COUNT, re-compute N_ELEMS, OFFS
    // If N_ELEMS==4, exit (we have to do N_ELEMS-4, i.e. 0, elements)
    add       $N_ELEMS, $N_ELEMS, -4
    brz       $N_ELEMS, \EXIT_LABEL
    shl       $mSCRATCH2, $mSCRATCH2, 2
    sub       $OFFS, $OFFS, $mSCRATCH2
1:
    // Worker with ID $WORKER_LAST need to process $DELTA_LAST element less
    cmpne     $WORKER_ID, $WORKER_ID, $WORKER_LAST
    brnz      $WORKER_ID, 2f
    sub       $N_ELEMS, $N_ELEMS, $DELTA_LAST
2:
.endm

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
.section .text.Float_Half_core
.align 8
.worker
Float_Half_core:
    // If output not 8 byte aligned, process 2 items, input will still be
    // 8 byte aligned (2 floats), output will become 8 byte aligned.
    and         $mSCRATCH, $OUT_START_PTR, 4
    brz         $mSCRATCH, 1f
    // Initial stage to align
    // Either deal with a pair of items and fall through if > 2 to do, else
    // process 1 or 2 and return.
    // Compare: if <2 to process
    cmpult      $mSCRATCH2, $N_ELEMS, 2
    brz         $mSCRATCH2, 2f
    // 1 to process - set up registers and exit via the path that processes 1
    ld32        $INPUT1,  $IN_PTR, 0
    mov         $mSCRATCH, $OUT_START_PTR
    bri         .Lrem1
2:
    ld64step    $INPUT1v2, $mzero, $IN_PTR+=,1
    {
      // update $N_ELEMS to reflect the two elements we are processing now.
      sub         $N_ELEMS, $N_ELEMS, 2
      f32v2tof16  $RESULT1, $INPUT1v2
    }
    st32step    $RESULT1, $mzero, $OUT_START_PTR+=,1

    // Compare: if only 2 to process (note, N_ELEMS has already been reduced
    // by 2).
    cmpult      $mSCRATCH2, $N_ELEMS, 1
    brz         $mSCRATCH2, 1f
    br          $LINK
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
    tapack      $INOUT_PTR, $mSCRATCH,$mzero, $OUT_START_PTR
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
    add         $mSCRATCH, $TOTAL_COUNT, $OUT_START_PTR

    // decision on the last 0-3 items
    brnzdec     $LOOP_COUNT, .Lremn0

    // 0 - return
    br          $LINK

.Lremn0:
    brnzdec     $LOOP_COUNT, .Lremn1

    // 1 left to do
    bri          .Lrem1
.Lremn1:
    brnzdec     $LOOP_COUNT, .Lremn2

    // 2 left, they are already loaded
    f32v2tof16   $RESULT1, $INPUT1v2
    st32step     $RESULT1, $mzero, $mSCRATCH+=, 1
    br           $LINK
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
    br            $LINK

.size Float_Half_core, .-Float_Half_core

//------------------------------------------------------------------------------

#define FLOAT_HALF_FUNC __runCodelet_popops__Cast___float_half

.globl FLOAT_HALF_FUNC
.type FLOAT_HALF_FUNC, @function
.section .text.FLOAT_HALF_FUNC
.align 4
FLOAT_HALF_FUNC:

// get fields from vertex state and call core function
    GET_PARAMS VOFF_IN VOFF_OUT_START VOFF_N_ELEMS

#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
    // Expand pointers:
    // Input pointer is SCALED64
    shl           $IN_PTR, $IN_PTR, 3
    // Output pointer is SCALED32
    shl           $OUT_START_PTR, $OUT_START_PTR, 2
    setzi         $mSCRATCH, TMEM_REGION0_BASE_ADDR
    add           $OUT_START_PTR, $OUT_START_PTR, $mSCRATCH
#endif

    call          $LINK, Float_Half_core
    exitz         $mzero

.size FLOAT_HALF_FUNC, .-FLOAT_HALF_FUNC

//------------------------------------------------------------------------------

.section .text.cast_float_half_from_supervisor
.worker
.align 4
cast_float_half_from_supervisor:
    WORKER_FROM_SUPERVISOR_START .Lexit_fh

#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
    // Expand pointers:
    // Input pointer is SCALED64
    shl       $IN_PTR, $IN_PTR, 3
    // Output pointer is SCALED32
    shl       $OUT_START_PTR, $OUT_START_PTR, 2
    setzi     $mSCRATCH, TMEM_REGION0_BASE_ADDR
    add       $OUT_START_PTR, $OUT_START_PTR, $mSCRATCH
#endif
    // OFFS is in units of number of elements. We add it to input and output
    // pointers, scaled appropriately (2 bytes/4 bytes) using dummy loads:
    // Increment $IN_PTR by 4xOFFS and $OUT_START_PTR by 2xOFFS
    ld32step  $mzero, $mzero, $IN_PTR+=, $OFFS
    ldz16step $mzero, $mzero, $OUT_START_PTR+=, $OFFS

    call      $LINK, Float_Half_core
.Lexit_fh:
    exitz     $mzero
.size cast_float_half_from_supervisor, .-cast_float_half_from_supervisor

//------------------------------------------------------------------------------

#define FLOAT_HALF_FUNC_SUPERVISOR __runCodelet_popops__CastSupervisor___float_half

.globl FLOAT_HALF_FUNC_SUPERVISOR
.type FLOAT_HALF_FUNC_SUPERVISOR, @function
.section .text.FLOAT_HALF_FUNC_SUPERVISOR
.align 4
.supervisor
FLOAT_HALF_FUNC_SUPERVISOR:

  setzi       $m1, cast_float_half_from_supervisor
  runall      $m1, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr


.size FLOAT_HALF_FUNC_SUPERVISOR, .-FLOAT_HALF_FUNC_SUPERVISOR

.worker
//------------------------------------------------------------------------------

// vertex offsets
#define VOFF_SRC_PTR_2D           0
#define VOFF_DST_PTR_BEGIN_2D     4
#define VOFF_DST_SIZE_2D          8

// register aliases
#define src_ptr_2d                m10
#define dst_ptr_2d                m9
#define num_vecs_2d               m8

// Stack offsets
#define STACK_OFF_2D_SRC_PTR      0
#define STACK_OFF_2D_DST_PTR      4
#define STACK_OFF_2D_DST_COUNT    8

#define FLOAT_HALF_FUNC_2D __runCodelet_popops__Cast2d___float_half


.globl FLOAT_HALF_FUNC_2D
.type FLOAT_HALF_FUNC_2D, @function
.section .text.FLOAT_HALF_FUNC_2D
.align 4
.worker
FLOAT_HALF_FUNC_2D:

    ld32          $src_ptr_2d, $mvertex_base, $mzero, VOFF_SRC_PTR_2D/4
    ld32          $dst_ptr_2d, $mvertex_base, $mzero, VOFF_DST_PTR_BEGIN_2D/4
    ld32          $num_vecs_2d, $mvertex_base, $mzero, VOFF_DST_SIZE_2D/4
    brnzdec       $num_vecs_2d, Lloop_2d_vectors_f_h
    exitz         $mzero

Lloop_2d_vectors_f_h:

  ld32step   $IN_PTR, $mzero, $src_ptr_2d+=, 1
  ld32step   $OUT_START_PTR, $mzero, $dst_ptr_2d+=, 1
  ld32step   $N_ELEMS, $mzero, $dst_ptr_2d+=, 1
  st32       $src_ptr_2d, $mworker_base, STACK_OFF_2D_SRC_PTR/4
  st32       $dst_ptr_2d, $mworker_base, STACK_OFF_2D_DST_PTR/4
  st32       $num_vecs_2d, $mworker_base, STACK_OFF_2D_DST_COUNT/4

  call       $LINK, Float_Half_core

  ld32       $src_ptr_2d, $mworker_base, STACK_OFF_2D_SRC_PTR/4
  ld32       $dst_ptr_2d, $mworker_base, STACK_OFF_2D_DST_PTR/4
  ld32       $num_vecs_2d, $mworker_base, STACK_OFF_2D_DST_COUNT/4
  brnzdec    $num_vecs_2d, Lloop_2d_vectors_f_h

LEnd_Float_Half_2D:
exitz         $mzero

.size FLOAT_HALF_FUNC_2D, .-FLOAT_HALF_FUNC_2D


//******************************************************************************
// Cast float to half core
//
// Assuming Input and Output alignment are both 8 bytes this function
// has an optimal inner loop.  No particular critical use cases have been
// identified so this constraint has not been seen to have any consequence.
//******************************************************************************
.section .text.Half_Float_core
.align 8
Half_Float_core:
    mov         $TOTAL_COUNT, $N_ELEMS

    // Total number of items to process, is it <4 ?
    cmpult      $mSCRATCH, $TOTAL_COUNT, 4
    brz         $mSCRATCH,1f

    // Setup pointers and load input to suit the code that deals with the
    // last 3 items below, and branch there, so we deal with up to 3 items the
    // same way as the trailing 3.
    add         $OUT_START_PTR, $OUT_START_PTR, 8
    ld64step    $INPUT1v2,$mzero, $IN_PTR+=,1
    bri         .Lless_than_4_total_hf
1:
    // Remember the output start pointer for later
    mov         $mSCRATCH, $OUT_START_PTR
    // total number of inner loops, each processing 4
    shr         $TOTAL_COUNT, $TOTAL_COUNT, 2

    // Ready the loop pipeline by loading one
    ld64step    $INPUT1v2,$mzero, $IN_PTR+=,1

    // Pack addresses and set a small stride -
    // Alternate instructions write alternate outputs.
    tapack      $INOUT_PTR, $IN_PTR,$mzero, $OUT_START_PTR
    setzi       $STRIDE, 2
    add         $OUT_START_PTR, $OUT_START_PTR, 8

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
    {st64step    $RESULTv2,$mzero, $OUT_START_PTR+=,2
     f16v2tof32  $RESULTv2, $INPUT1}
2:
    // Store the last 2 that were converted, avoiding over-processing
    // Stride the input pointer by 1(64 bit step), and the output by
    // 2(64 bit steps)
    {ldst64pace  $INPUT1v2, $RESULTv2, $INOUT_PTR+=,$STRIDE, 4
     f16v2tof32  $RESULTv2, $INPUT2}
    st64step     $RESULTv2, $mzero, $OUT_START_PTR+=,2

    brnz         $TOTAL_COUNT, 3b

    // use ptr to start/end of the output array to check how many are left
    and         $TOTAL_COUNT, $N_ELEMS, 3

.Lless_than_4_total_hf:
    brnzdec     $TOTAL_COUNT, .Lhf_nrem0

    // 0 left: nothing to do
    br          $LINK
.Lhf_nrem0:
    add         $OUT_START_PTR, $OUT_START_PTR, -8
    brnzdec     $TOTAL_COUNT, .Lhf_nrem1

    // 1 left, convert just 1 and store
    f16tof32    $RESULT1, $INPUT1
    st32        $RESULT1, $mzero, $OUT_START_PTR, 0
     br         $LINK
.Lhf_nrem1:
   brnzdec     $TOTAL_COUNT, .Lhf_nrem2

    // 2 left - convert 2 and store
    f16v2tof32 $RESULTv2, $INPUT1
    st64       $RESULTv2, $mzero, $OUT_START_PTR, 0
    br         $LINK
.Lhf_nrem2:

    // 3 left
    // Convert 3 of the 4 we already read and store them
    f16v2tof32  $RESULTv2, $INPUT1
    { st64step  $RESULTv2, $mzero, $OUT_START_PTR+=, 1
    f16tof32    $RESULT1, $INPUT2}
    st32        $RESULT1, $mzero, $OUT_START_PTR, 0

     br         $LINK
.size Half_Float_core, .-Half_Float_core

//******************************************************************************
// Cast half to float
// Fetches the vertex state and calls the half to float core function
//******************************************************************************
#define HALF_FLOAT_FUNC __runCodelet_popops__Cast___half_float

.globl HALF_FLOAT_FUNC
.type HALF_FLOAT_FUNC, @function
.section .text.HALF_FLOAT_FUNC
.align 4
HALF_FLOAT_FUNC:

    // get fields from vertex state and call core function
    GET_PARAMS VOFF_IN VOFF_OUT_START VOFF_N_ELEMS
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
    // Expand pointers:
    // Input pointer is SCALED64
    shl           $IN_PTR, $IN_PTR, 3
    // Output pointer is SCALED64
    shl           $OUT_START_PTR, $OUT_START_PTR, 3
#endif
    call          $LINK, Half_Float_core
    exitz         $mzero

.size HALF_FLOAT_FUNC, .-HALF_FLOAT_FUNC

//------------------------------------------------------------------------------

.section .text.cast_half_float_from_supervisor
.worker
.align 4
cast_half_float_from_supervisor:

    WORKER_FROM_SUPERVISOR_START .Lexit_hf
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
    // Expand pointers:
    // Input pointer is SCALED64
    shl           $IN_PTR, $IN_PTR, 3
    // Output pointer is SCALED64
    shl           $OUT_START_PTR, $OUT_START_PTR, 3
#endif
    // OFFS is in units of number of elements. We add it to input and output
    // pointers, scaled appropriately (2 bytes/4 bytes) using dummy loads:
    // Increment $IN_PTR by 2xOFFS and $OUT_START_PTR by 4xOFFS
    ldz16step $mzero, $mzero, $IN_PTR+=, $OFFS
    ld32step  $mzero, $mzero, $OUT_START_PTR+=, $OFFS

    call      $LINK, Half_Float_core
.Lexit_hf:
    exitz     $mzero

.size cast_half_float_from_supervisor, .-cast_half_float_from_supervisor

//------------------------------------------------------------------------------

#define HALF_FLOAT_FUNC_SUPERVISOR __runCodelet_popops__CastSupervisor___half_float

.globl HALF_FLOAT_FUNC_SUPERVISOR
.type HALF_FLOAT_FUNC_SUPERVISOR, @function
.section .text.HALF_FLOAT_FUNC_SUPERVISOR
.align 4
.supervisor
HALF_FLOAT_FUNC_SUPERVISOR:

  setzi       $m1, cast_half_float_from_supervisor
  runall      $m1, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr


.size HALF_FLOAT_FUNC_SUPERVISOR, .-HALF_FLOAT_FUNC_SUPERVISOR

.worker

//******************************************************************************
// Cast half to float 2d
// Fetches the vertex state and calls the half to float core function within
// its own outer loop
//******************************************************************************

#define HALF_FLOAT_FUNC_2D __runCodelet_popops__Cast2d___half_float

.globl HALF_FLOAT_FUNC_2D
.type HALF_FLOAT_FUNC_2D, @function
.section .text.HALF_FLOAT_FUNC_2D
.align 4
.worker
HALF_FLOAT_FUNC_2D:

    ld32          $src_ptr_2d, $mvertex_base, $mzero, VOFF_SRC_PTR_2D/4
    ld32          $dst_ptr_2d, $mvertex_base, $mzero, VOFF_DST_PTR_BEGIN_2D/4
    ld32          $num_vecs_2d, $mvertex_base, $mzero, VOFF_DST_SIZE_2D/4
    brnzdec       $num_vecs_2d, Lloop_2d_vectors_h_f
    exitz         $mzero

Lloop_2d_vectors_h_f:

  ld32step   $IN_PTR, $mzero, $src_ptr_2d+=, 1
  ld32step   $OUT_START_PTR, $mzero, $dst_ptr_2d+=, 1
  ld32step   $N_ELEMS, $mzero, $dst_ptr_2d+=, 1
  st32       $src_ptr_2d, $mworker_base, STACK_OFF_2D_SRC_PTR/4
  st32       $dst_ptr_2d, $mworker_base, STACK_OFF_2D_DST_PTR/4
  st32       $num_vecs_2d, $mworker_base, STACK_OFF_2D_DST_COUNT/4

  call       $LINK, Half_Float_core

  ld32       $src_ptr_2d, $mworker_base, STACK_OFF_2D_SRC_PTR/4
  ld32       $dst_ptr_2d, $mworker_base, STACK_OFF_2D_DST_PTR/4
  ld32       $num_vecs_2d, $mworker_base, STACK_OFF_2D_DST_COUNT/4
  brnzdec    $num_vecs_2d, Lloop_2d_vectors_h_f

LEnd_Half_Float_2D:
  exitz         $mzero

.size HALF_FLOAT_FUNC_2D, .-HALF_FLOAT_FUNC_2D



//------------------------------------------------------------------------------
// We define here the supervisor vertices for all other needed type conversions,
// apart from HALF<->FLOAT (which are defined above, both supervisors and
// workers).
// The worker vertices for these below are defined in C++

// Macro to create one supervisor vertex
.macro INSTANTIATE_SUPERVISOR  INTYPE  OUTTYPE
.globl __runCodelet_popops__CastSupervisor___\INTYPE\()_\OUTTYPE
.type __runCodelet_popops__CastSupervisor___\INTYPE\()_\OUTTYPE, @function
.section .text.__runCodelet_popops__CastSupervisor___\INTYPE\()_\OUTTYPE
.align 4
.supervisor
__runCodelet_popops__CastSupervisor___\INTYPE\()_\OUTTYPE:
  setzi       $m1, __runCodelet_popops__CastWorker___\INTYPE\()_\OUTTYPE
  runall      $m1, $m0, 0
  sync        TEXCH_SYNCZONE_LOCAL
  br          $lr
.size __runCodelet_popops__CastSupervisor___\INTYPE\()_\OUTTYPE, .-__runCodelet_popops__CastSupervisor___\INTYPE\()_\OUTTYPE
.endm

// Note that we don't define here the following:
//    1. FLOAT<->HALF conversion (defined above)
//    2. Identity conversions (XXX->XXX) and INT<->UNSIGNED as these will be
//       replaced with Copy() in popops::cast()
INSTANTIATE_SUPERVISOR float int
INSTANTIATE_SUPERVISOR float unsigned_int
INSTANTIATE_SUPERVISOR float bool

INSTANTIATE_SUPERVISOR half int
INSTANTIATE_SUPERVISOR half unsigned_int
INSTANTIATE_SUPERVISOR half bool

INSTANTIATE_SUPERVISOR int float
INSTANTIATE_SUPERVISOR int half
INSTANTIATE_SUPERVISOR int bool

INSTANTIATE_SUPERVISOR unsigned_int float
INSTANTIATE_SUPERVISOR unsigned_int half
INSTANTIATE_SUPERVISOR unsigned_int bool

INSTANTIATE_SUPERVISOR bool float
INSTANTIATE_SUPERVISOR bool half
INSTANTIATE_SUPERVISOR bool int
INSTANTIATE_SUPERVISOR bool unsigned_int

#endif
/* -------------------------------------------------------------------------- */
