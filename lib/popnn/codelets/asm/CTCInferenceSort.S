// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Vertices to support sorting in CTC Inference

#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// Name mangling

#define RANK_VERTEX_NAME __runCodelet_popnn__CTCRankCandidates___\P_TYPE\()_\SYM_TYPE
#define REDUCE_VERTEX_NAME __runCodelet_popnn__CTCReduceCandidates___\P_TYPE\()_\SYM_TYPE

// Vertex state (Common between rank, reduce vertices)
#define VERTEX_STATE_OFFSET_IN_PARENT 0
#define VERTEX_STATE_OFFSET_IN_ADDEND 4
#define VERTEX_STATE_OFFSET_IN_PNB 8
#define VERTEX_STATE_OFFSET_IN_PB 12
#define VERTEX_STATE_OFFSET_IN_PTOTAL 16

// The reduce vertex uses the way that both VERTEX_STATE_OFFSET_IN_X and
// VERTEX_STATE_OFFSET_OUT_x are arranged to pick up pointers to the input and
// output of the data to reduce
#define NUM_ITEMS_TO_REDUCE (1+(VERTEX_STATE_OFFSET_IN_PTOTAL-VERTEX_STATE_OFFSET_IN_PARENT)/4)

#define VERTEX_STATE_OFFSET_OUT_PARENT 20
#define VERTEX_STATE_OFFSET_OUT_ADDEND 24
#define VERTEX_STATE_OFFSET_OUT_PNB 28
#define VERTEX_STATE_OFFSET_OUT_PB 32
#define VERTEX_STATE_OFFSET_OUT_PTOTAL 36
#define VERTEX_STATE_OFFSET_TOTAL_CANDIDATES 40

// Vertex state (Rank vertex only)
#define VERTEX_STATE_OFFSET_COMPLETE 44
#define VERTEX_STATE_OFFSET_BEAMWIDTH 48
#define VERTEX_STATE_FIRST_CANDIDATE 52
#define VERTEX_STATE_LAST_CANDIDATE 56

// Register defines (Supervisor)
#define msupervisor_vertex_base m0
#define mworker_entry m1
#define mcomplete m7

// Common between Worker and supervisor
#define mcount m1
#define mparent_ptr m2
#define maddend_ptr m3
#define mpnb_ptr m4
#define mpb_ptr m5
#define mptotal_ptr m6

// Register defines (Worker)
#define mfirst m0
#define mlast m7
#define mworker_id m8
#define mscratch m8
#define mtotal m9
#define mptr m10
#define mrank m11   // Note: not using the worker stack pointer so register can be used

#define ato_rank0 a0
#define ato_rank1 a1
#define ato_rank01 a0:1
#define atotal0 a2
#define atotal1 a3
#define atotal01 a2:3
#define aconst0 a4
#define aconst1 a5
#define aconst01 a4:5

// Constants
#define FLOAT_1_0 0x3f800000

// Macro to use to copy, given an input pointer in a register and an output
// pointer in the vertex state
.macro COPY IN OUT
  ld32 $mscratch, $\IN\(), $mzero, $mfirst
  ld32 $mptr, $mvertex_base, $mzero, \OUT\()/4
  stm32 $mscratch, $mptr, $mrank
.endm

// *****************************************************************************
// Supervisor entry point
// The supervisor initialises the output area to zero and exits if
// processing is complete.  Otherwise it calls the workers.

.macro INSTANTIATE_RANK P_TYPE SYM_TYPE
.globl RANK_VERTEX_NAME
.type RANK_VERTEX_NAME, @function
.section .text.RANK_VERTEX_NAME, "ax"
.align 4
.supervisor
RANK_VERTEX_NAME:
  // Exit if this input is all processed (compare, branch are later to avoid pipline hits)
  ld32 $mcomplete, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_COMPLETE/4
  ld32 $mcount, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4

  ld32 $mparent_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PARENT/4
  ld32 $maddend_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_ADDEND/4
  ld32 $mpnb_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PNB/4
  ld32 $mpb_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PB/4
  ld32 $mptotal_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PTOTAL/4

  ld32 $mcomplete, $mzero, $mcomplete, 0
  sub  $mcount,$mcount,1

  // Zero the results, ensuring they are zero before any worker runs, and should
  // be faster in the supervisor context
1:
  st32step $mzero, $mzero, $mparent_ptr+=,1
  st32step $mzero, $mzero, $maddend_ptr+=,1
  st32step $mzero, $mzero, $mpb_ptr+=,1
  st32step $mzero, $mzero, $mpnb_ptr+=,1
  st32step $mzero, $mzero, $mptotal_ptr+=,1
  brnzdec $mcount, 1b

  // Exit if all processed
  setzi $mworker_entry, .Lworker_\P_TYPE\()\@
  brnz  $mcomplete, 2f
  // Run the workers
  runall  $mworker_entry, $msupervisor_vertex_base, 0
2:
  sync    TEXCH_SYNCZONE_LOCAL
  br      $lr

// Worker code:
// Workers use their worker ID to choose which item to rank, by comparing against
// all other items.  If found to be in the top 'beamwidth' ranked items the
// candidate data is copied into the output.
.align 8
.Lworker_float\@:
.worker
  // Get worker ID and find the index of this workers 1st item
  get $mworker_id, $WSR
  and $mworker_id, $mworker_id, CSR_W_WSR__CTXTID_M1__MASK
  ld32 $mfirst, $mvertex_base, $mzero, VERTEX_STATE_FIRST_CANDIDATE/4
  add  $mfirst, $mfirst, $mworker_id

  ld32 $mlast, $mvertex_base, $mzero, VERTEX_STATE_LAST_CANDIDATE/4
  cmpult $mscratch, $mfirst, $mlast
  // Exit if this worker has nothing to do
  brz   $mscratch, .Lreturn_f\@

  // Load vertex state
  ld32 $mtotal, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_TOTAL_CANDIDATES/4
  // Enough registers free to keep these pointers which are used only when
  // storing ranked results.
  // Also set the constant 2 registers 1.0f, 1.0f to `and` with later.
  // This is because the result of a compare is:
  // TFPU_FP32_TRUE   0xffffffff
  // TFPU_FP32_FALSE  0x00000000
  // So TFPU_FP32_TRUE & 1.0f = 1.0f
  //    TFPU_FP32_FALSE & 1.0f = 0.0f
  // Which is then easy to accumulate and use later
  ld32 $mparent_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_IN_PARENT/4
  ld32 $maddend_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_IN_ADDEND/4
  ld32 $mptotal_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_IN_PTOTAL/4
  {ld32 $mpnb_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_IN_PNB/4
   or    $aconst0, $azero, FLOAT_1_0}
  {ld32 $mpb_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_IN_PB/4
   mov   $aconst1, $aconst0}

.Lloop\@:
  // Zero the accumulator of items > the current
  // Load and broadcast the one to rank
  {ld32 $mptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_IN_PTOTAL/4
   mov $atotal01, $azeros}

  ld32 $ato_rank0, $mzero, $mptr, $mfirst

   // Loop over those stored before the one to rank, if any
   // 2 items per loop
  {shr  $mscratch, $mfirst, 1
   mov  $ato_rank1, $ato_rank0}

  // Load the first pair to compare to
  ld64step $a6:7, $mzero, $mptr+=, 1

  rpt $mscratch, (2f - 1f)/8 - 1
1:
  // For anything before the one to rank use >= as the inequality
  {nop
  f32v2cmpge $a6:7, $a6:7, $ato_rank01}
  {nop
   and64 $a6:7, $a6:7, $aconst01}
  {ld64step $a6:7, $mzero, $mptr+=, 1
   f32v2add $atotal01, $atotal01, $a6:7}
2:
  // Find the number of items to process in the second loop if we ran it.
  // 2ndLoopElements = totalElements - 1stLoopElements - 2
  // (The code between the loops would deal with 2 elements if run).
  // However if there would be < 0 elements to process then we can exit now:
  // < 0 : The one we are ranking is the last item, we processed all
  //       those before it in the first loop, so we are done
  // = 0 : The one we are ranking is one of the 2 to process in the code between
  //       loops, then we are done. So we continue (And will run 0 passes of the
  //       second loop)
  // > 0 : Valid number to process in the 2nd loop and/or as the odd one at the
  //       end
  shl  $mscratch, $mscratch, 1
  sub  $mtotal, $mtotal, $mscratch
  sub  $mtotal, $mtotal, 2
  brneg $mtotal, .Lcheck_ranking\@

  // The last input loaded contains the one we compare to, and another.
  // Avoid using a ld32 to ensure input data alignment for the next loop.

  // For anything before the one to rank continue to use >= as the inequality
  // For anything after use > to ensure a unique ranking
  {and $mscratch, $mfirst, 1
   f32cmpge $a6, $a6, $ato_rank0}
  {brz $mscratch,3f
   f32cmpgt $a7, $a7, $ato_rank0}
  {bri 4f
  // This is used when taking the input BEFORE the one we are ranking, held in $a6
   and $a6, $a6, $aconst0}
3:
  // This is used when taking the input AFTER the one we are ranking, held in $a7
  and $a6, $a7, $aconst0
4:
  // Remaining loops, 2 items per loop
  {shr $mscratch, $mtotal, 1
   f32add $atotal0, $atotal0, $a6}
  // Load the next pair to process
  {ld64step $a6:7, $mzero, $mptr+=, 1
   fnop} // Rpt alignment

  rpt $mscratch, (2f - 1f)/8 - 1
1:
  // For anything after the one to rank use > as the inequality
  {nop
   f32v2cmpgt $a6:7, $a6:7, $ato_rank01}
  {nop
   and64 $a6:7, $a6:7, $aconst01}
  {ld64step $a6:7, $mzero, $mptr+=, 1
   f32v2add $atotal01, $atotal01, $a6:7}
2:
  // Subtract the number processed by the second loop
  shl  $mscratch, $mscratch, 1
  sub  $mscratch, $mtotal, $mscratch
  brz $mscratch, .Lcheck_ranking\@
  // Last one to process, already read at the end of the last loop.
  // But don't process it unless we need to in case of invalid inputs.
  // For anything after the one to rank use > as the inequality
  f32cmpgt $a6, $a6, $ato_rank0
  and      $a6, $a6, $aconst0
  f32add   $atotal0, $atotal0, $a6

.Lcheck_ranking\@:
  // Combine the 2 halves of the accumulators to find the ranking and decide
  // if ranked well enough to store
  // Reload the total for the next pass
  {ld32 $mtotal, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_TOTAL_CANDIDATES/4
   f32add $atotal0, $atotal0, $atotal1}
  {ld32 $mscratch, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4
   f32toi32 $atotal0, $atotal0}
  atom $mrank, $atotal0
  cmpult $mscratch, $mrank, $mscratch
  brz $mscratch, 6f

  // Ranked in the top beamwidth so store.

  COPY mptotal_ptr VERTEX_STATE_OFFSET_OUT_PTOTAL
  COPY mpb_ptr VERTEX_STATE_OFFSET_OUT_PB
  COPY mpnb_ptr VERTEX_STATE_OFFSET_OUT_PNB
  COPY mparent_ptr VERTEX_STATE_OFFSET_OUT_PARENT
  COPY maddend_ptr VERTEX_STATE_OFFSET_OUT_ADDEND
6:
  // Step the number of the candidate to rank - skipping those ranked by the
  // other workers and loop if this worker has more to do.
  add $mfirst, $mfirst, CTXT_WORKERS
  cmpult $mscratch, $mfirst, $mlast
  brnz $mscratch, .Lloop\@

.Lreturn_f\@:
    exitz $mzero

.size RANK_VERTEX_NAME, .-RANK_VERTEX_NAME

DEF_STACK_USAGE 0 RANK_VERTEX_NAME

.endm

// *****************************************************************************
// Supervisor entry point

.macro INSTANTIATE_REDUCE P_TYPE SYM_TYPE
.globl REDUCE_VERTEX_NAME
.type REDUCE_VERTEX_NAME, @function
.section .text.REDUCE_VERTEX_NAME, "ax"
.align 4
.supervisor
REDUCE_VERTEX_NAME:
  setzi   $mworker_entry, .Lworker_\P_TYPE\()\@
  // Run the workers
  runall  $mworker_entry, $msupervisor_vertex_base, 0
2:
  sync    TEXCH_SYNCZONE_LOCAL
  br      $lr

// Worker code:
// Workers use their worker ID to choose which item to reduce
.align 8
.Lworker_float\@:
.worker
  // Get worker ID and find the index of this workers 1st item, and output pointer
  get $mworker_id, $WSR
  and $mworker_id, $mworker_id, CSR_W_WSR__CTXTID_M1__MASK
  // Workers with an ID >= NUM_ITEMS_TO_REDUCE-1 will just exit
  cmpult $mptr, $mworker_id, NUM_ITEMS_TO_REDUCE
  brz   $mptr, .Lexit\@

  ld32 $mfirst, $mvertex_base, $mzero, $mworker_id
  add  $mworker_id, $mworker_id, NUM_ITEMS_TO_REDUCE
  ld32 $mptr, $mvertex_base, $mzero, $mworker_id

  // Find the number of loops.  2 items per loop, and 2 items are processed
  // before the loop begins
  ld32 $mtotal, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_TOTAL_CANDIDATES/4
  shr  $mcount, $mtotal, 1
  sub  $mcount, $mcount, 1

  // Load 1st pair, which initialises the result
  ld64step $atotal01, $mzero, $mfirst+=, 1
  // Exit if 1 item only
  brneg  $mcount, .Lwrite_and_exit\@
  // Load 2nd pair to begin combining, loop and use or64 to combine:
  // All inputs are zero except for one, but some data is float, some unsigned_int
  // Using or64 allows the same code to combine the inputs regardless of type,
  // and is an instruction with 64 bit operands which we can bundle
  ld64step $a0:1, $mzero, $mfirst+=, 1
  rpt $mcount, (2f - 1f)/8 -1
1:
  {ld64step $a0:1, $mzero, $mfirst+=, 1
   or64 $atotal01, $atotal01, $a0:1}
2:
  // Check if there was an odd number to combine
  and   $mscratch, $mtotal, 1
  {brz   $mscratch, 3f
   or    $atotal0, $atotal0, $atotal1} // Combine 2 halves of the result so far
  or    $atotal0, $atotal0, $a0        // Combine a last odd numbered one
3:
.Lwrite_and_exit\@:
  st32  $atotal0, $mzero, $mptr, 0
.Lexit\@:
  exitz $mzero

.size REDUCE_VERTEX_NAME, .-REDUCE_VERTEX_NAME

DEF_STACK_USAGE 0 REDUCE_VERTEX_NAME

.endm
//******************************************************************************
// Use the macros to instantiate the vertices

INSTANTIATE_RANK float unsigned_int
INSTANTIATE_REDUCE float unsigned_int


#endif
