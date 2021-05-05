// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Vertices to support sorting in CTC Inference

#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Name mangling

#define VERTEX_NAME __runCodelet_popnn__CTCRankCandidates___\P_TYPE\()_\SYM_TYPE

// Vertex state
#define VERTEX_STATE_OFFSET_IN_PARENT 0
#define VERTEX_STATE_OFFSET_IN_ADDEND 4
#define VERTEX_STATE_OFFSET_IN_PNB 8
#define VERTEX_STATE_OFFSET_IN_PB 12
#define VERTEX_STATE_OFFSET_IN_PTOTAL 16

#define VERTEX_STATE_OFFSET_OUT_PARENT 20
#define VERTEX_STATE_OFFSET_OUT_ADDEND 24
#define VERTEX_STATE_OFFSET_OUT_PNB 28
#define VERTEX_STATE_OFFSET_OUT_PB 32
#define VERTEX_STATE_OFFSET_OUT_PTOTAL 36

#define VERTEX_STATE_OFFSET_CURRENT_TIMESTEP 40
#define VERTEX_STATE_OFFSET_DATA_LENGTH 44

#define VERTEX_STATE_OFFSET_BEAMWIDTH 48
#define VERTEX_STATE_OFFSET_TOTAL_CANDIDATES 52
#define VERTEX_STATE_FIRST_CANDIDATE 56
#define VERTEX_STATE_LAST_CANDIDATE 60

// Register defines (Supervisor)
#define msupervisor_vertex_base m0
#define mworker_entry m1
#define mdata_length m7
#define mtimestep m8

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
// The supervisor initialises the output area to zero and exits if the loop counter
// indicates that processing is complete.  Otherwise it calls the workers.

.macro INSTANTIATE P_TYPE SYM_TYPE
.globl VERTEX_NAME
.type VERTEX_NAME, @function
.section .text.VERTEX_NAME, "ax"
.align 4
.supervisor
VERTEX_NAME:
  // Exit if this input is all processed (compare, branch are later to avoid pipline hits)
  ld32 $mtimestep, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_CURRENT_TIMESTEP/4
  ld32 $mdata_length, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_DATA_LENGTH/4
  ld32 $mcount, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4

  ld32 $mparent_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PARENT/4
  ld32 $maddend_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_ADDEND/4
  ld32 $mpnb_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PNB/4

  ld32 $mtimestep, $mzero, $mtimestep, 0
  ld32 $mdata_length, $mzero, $mdata_length, 0

  ld32 $mpb_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PB/4
  ld32 $mptotal_ptr, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_OUT_PTOTAL/4

  sub  $mcount,$mcount,1
  nop     // Avoid pipeline hit for mdata_length
  nop
  cmpult  $mdata_length, $mdata_length, $mtimestep
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
  setzi   $mworker_entry, .Lworker_\P_TYPE\()\@
  brnz  $mdata_length, 2f
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
  // If we are ranking the last one, we are done
  sub  $mscratch, $mtotal, $mfirst
  cmpeq $mscratch, $mscratch, 1
  brnz $mscratch, .Lcheck_ranking\@
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
  // Remaining loops, avoid processing a last one that isn't part of a pair
  {sub $mscratch, $mtotal, $mfirst
   f32add $atotal0, $atotal0, $a6}
  shr $mscratch, $mscratch, 1
  // Load the next pair to process
  {ld64step $a6:7, $mzero, $mptr+=, 1
   fnop} // Rpt alignment
  brz  $mscratch, 2f
  sub $mscratch, $mscratch, 1

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
  // Deal with a last one if the total candidates is odd
  and  $mscratch, $mtotal, 1
  brz  $mscratch, .Lcheck_ranking\@
  // Last one to process, already read at the end of the last loop.
  // But don't process it unless we need to in case of invalid inputs
  // For anything after the one to rank > as the inequality
  f32cmpgt $a6, $a6, $ato_rank0
  and      $a6, $a6, $aconst0
  f32add   $atotal0, $atotal0, $a6

.Lcheck_ranking\@:
  // Combine the 2 halves of the accumulators to find the ranking and decide
  // if ranked well enough to store
  f32add $atotal0, $atotal0, $atotal1
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

.size VERTEX_NAME, .-VERTEX_NAME

DEF_STACK_USAGE 0 VERTEX_NAME

.endm
//******************************************************************************
// Use the macro to instantiate the vertex

INSTANTIATE float unsigned_int

#endif
