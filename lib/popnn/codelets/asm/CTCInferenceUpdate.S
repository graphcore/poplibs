// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Vertex for CTC Inference update

#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#define INCLUDE_IN_ASSEMBLER
#include "poplibs_support/CTCInferenceDefs.hpp"
#include "poplibs_support/LogArithmetic.hpp"

// Name mangling
#define UPDATE_VERTEX_NAME __runCodelet_popnn__CTCUpdate___\P_TYPE\()_\SYM_TYPE

// Vertex state
#define VERTEX_STATE_OFFSET_CANDIDATE_PARENT 0
#define VERTEX_STATE_OFFSET_CANDIDATE_ADDEND 4
#define VERTEX_STATE_OFFSET_CANDIDATE_PNB 8
#define VERTEX_STATE_OFFSET_CANDIDATE_PB 12
#define VERTEX_STATE_OFFSET_CANDIDATE_PTOTAL 16

#define VERTEX_STATE_OFFSET_BEAM_PNB 20
#define VERTEX_STATE_OFFSET_BEAM_PB 24
#define VERTEX_STATE_OFFSET_BEAM_PTOTAL 28
#define VERTEX_STATE_OFFSET_BEAM_ADDEND 32
#define VERTEX_STATE_OFFSET_BEAM_PARENT 36

#define VERTEX_STATE_OFFSET_BEAM_LENGTH 40
#define VERTEX_STATE_OFFSET_LAST_BEAM_OUT 44

#define VERTEX_STATE_OFFSET_CURRENT_TIMESTEP 48
#define VERTEX_STATE_OFFSET_DATA_LENGTH 52
#define VERTEX_STATE_OFFSET_BEAMWIDTH 56


// Register defines (Supervisor)
#define msupervisor_vertex_base m0
#define mworker_entry0 m1
#define mworker_entry1 m2
#define mworker_entry2 m3
#define mworker_entry3 m4
#define mworker_entry4 m5

#define mworker_entry5 m6
#define mregion_base m6

#define mdata_length m7
#define mtimestep m8

// Worker
#define mworker_id m0
#define malternate_result m0

#define mcount m1
#define mscratch m2

#define min_ptr m3
#define mparent_ptr m3

#define mout_ptr m4

#define mvoid_symbol m5
#define minvalid_symbol m5

#define mcandidate_addend m6
#define mcandidate_parent m7
#define mptr m8
#define maddend_ptr m9
#define mbaseM1 m10
#define mbase m11

.macro LOAD_COMMON
  // All paths but for the simple copy paths use these.
  ldconst $mvoid_symbol, VOID_SYMBOL
  ld32 $maddend_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CANDIDATE_ADDEND/4
  ld32 $mparent_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CANDIDATE_PARENT/4
  ld32 $mtimestep, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CURRENT_TIMESTEP/4
  ld32 $mtimestep, $mtimestep, $mzero, 0

  ld32 $mcount, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4
.endm

// *****************************************************************************
// Supervisor entry point

.macro INSTANTIATE_UPDATE P_TYPE SYM_TYPE
.globl UPDATE_VERTEX_NAME
.type UPDATE_VERTEX_NAME, @function
.section .text.UPDATE_VERTEX_NAME, "ax"
.align 4
.supervisor
UPDATE_VERTEX_NAME:
  // Exit if this input is all processed,
  // Subtract TMEM_REGION0_BASE_ADDR from the supervisor base to provide a
  // worker base.
  // Load function pointers to an invividual stub per worker
  // Jumbled up to avoid pipeline bubbles as much as possible
  ld32 $mtimestep, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_CURRENT_TIMESTEP/4
  ld32 $mdata_length, $msupervisor_vertex_base, $mzero, VERTEX_STATE_OFFSET_DATA_LENGTH/4

  setzi   $mregion_base, TMEM_REGION0_BASE_ADDR
  setzi   $mworker_entry0, .Laddend\@
  setzi   $mworker_entry1, .Lparent\@
  setzi   $mworker_entry2, .Lbeam_length\@
  //----- Pipeline length
  ld32 $mtimestep, $mzero, $mtimestep, 0
  ld32 $mdata_length, $mzero, $mdata_length, 0

  sub  $msupervisor_vertex_base, $msupervisor_vertex_base, $mregion_base
  setzi   $mworker_entry3, .Lprob_pnb\@
  setzi   $mworker_entry4, .Lprob_pb\@
  setzi   $mworker_entry5, .Lprob_ptotal\@
  //----- Pipeline length
  nop
  cmpult  $mdata_length, $mdata_length, $mtimestep
  // No avoiding pipline bubbles now
  // Exit if all processed
  brnz  $mdata_length, 2f
  // Run the workers
  run $mworker_entry0, $msupervisor_vertex_base, 0
  run $mworker_entry1, $msupervisor_vertex_base, 0
  run $mworker_entry2, $msupervisor_vertex_base, 0
  run $mworker_entry3, $msupervisor_vertex_base, 0
  run $mworker_entry4, $msupervisor_vertex_base, 0
  run $mworker_entry5, $msupervisor_vertex_base, 0
2:
  sync    TEXCH_SYNCZONE_LOCAL
  br      $lr

.align 8
// Worker code:
.worker

// Update the addend. Equivalent to:
// addend[i] = (candidateAddend[i] == voidSymbol)
//      ? previousAddend[candidateParent[i]] : candidateAddend[i];
// lastBeamOutputs[i] = addend[i];
//
.Laddend\@:
  LOAD_COMMON
  // Pointing to the addend and parent in the beam history requires a base
  // offset for this timestep
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4
   // Offset into either an addend or parent array given by:
  // beamwidth * timestep
  mul  $mbase, $mscratch, $mtimestep
  // And the previous timestep by subtracting beamwidth
  sub  $mbaseM1, $mbase, $mscratch
  // Convert to bytes for use as an address
  shl  $mbaseM1, $mbaseM1, 2

  ld32 $mout_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAM_ADDEND/4
  // mbaseM1 points to the start of the addend at the previous timestep
  add  $mbaseM1, $mbaseM1, $mout_ptr
  ld32 $mptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_LAST_BEAM_OUT/4
  // use mbase to increment mout_ptr to the current timestep
  ld32step $mscratch, $mzero, $mout_ptr+=, $mbase

  rpt $mcount, (2f - 1f)/ 8 - 1
1:
  {ld32step $mcandidate_addend, $mzero, $maddend_ptr+=, 1 ;fnop}
  {ld32step $mcandidate_parent, $mzero, $mparent_ptr+=, 1 ;fnop}
  // malternate_result = previousAddend[candidateParent[i]]
  {ld32     $malternate_result, $mbaseM1, $mzero, $mcandidate_parent ;fnop}
  // if(candidateAddend[i] == voidSymbol) {result = malternate_result}
  {cmpeq    $mscratch, $mcandidate_addend, $mvoid_symbol  ;fnop}
  {movz     $mcandidate_addend, $mscratch, $malternate_result ;fnop}
  // (mbase + mout_ptr) point to the current timestep and increment with the
  // beam being updated as we write the output
  {st32step $mcandidate_addend, $mzero, $mout_ptr+=, 1 ;fnop}
  {st32step $mcandidate_addend, $mzero, $mptr+=, 1 ;fnop}
2:
  exitz $mzero


// Update the parent. Equivalent to:
// parent[i] = (candidateAddend[i] == voidSymbol)
//     ? previousParent[candidateParent[i]] : candidateParent[i] + previousBaseOffset;
//
.Lparent\@:
  LOAD_COMMON
  // Pointing to the parent in the beam history requires anoffset for this timestep
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4
   // Offset into either an addend or parent array given by:
  // beamwidth * timestep
  mul  $mbase, $mscratch, $mtimestep
  // And the previous timestep by subtracting beamwidth
  sub  $mbaseM1, $mbase, $mscratch

  ld32 $mout_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAM_PARENT/4
  mov  $mptr, $mout_ptr
  // Offset both pointers with a dummy load
  ld32step $mscratch, $mzero, $mptr+=, $mbaseM1
  ld32step $mscratch, $mzero, $mout_ptr+=, $mbase

  rpt $mcount, (2f - 1f)/ 8 - 1
1:
  {ld32step $mcandidate_addend, $mzero, $maddend_ptr+=, 1  ;fnop}
  {ld32step $mcandidate_parent, $mzero, $mparent_ptr+=, 1  ;fnop}
  // malternate_result = previousParent[candidateParent[i]]
  {ld32     $malternate_result, $mzero, $mptr, $mcandidate_parent  ;fnop}
  // mcandidate_parent = candidateParent + previousBaseOffset
  {add      $mcandidate_parent, $mcandidate_parent, $mbaseM1  ;fnop}
  // if(candidateAddend[i] == voidSymbol) {result = malternate_result}
  {cmpeq    $mscratch, $mcandidate_addend, $mvoid_symbol  ;fnop}
  {movz     $mcandidate_parent, $mscratch, $malternate_result  ;fnop}
  {st32step $mcandidate_parent, $mzero, $mout_ptr+=, 1  ;fnop}
 2:
  exitz $mzero

// Update the beam length. Equivalent to:
// newBeamLength[i] =  previousBeamLength[candidateParent[i]] + (candidateAddend[i] != voidSymbol);
//
.Lbeam_length\@:
  LOAD_COMMON
  // (timestep & 1) used to determine newBeamLength or previousBeamLength
  // as 2 halves of a ping-pong buffer, with each half of size beamwidth
  ld32  $mbaseM1, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CURRENT_TIMESTEP/4
  ld32  $mbaseM1, $mzero, $mbaseM1, 0
  and   $mbaseM1, $mbaseM1, 1

  // beamwidth size in bytes
  ld32  $mscratch, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4
  shl   $mscratch, $mscratch, 2
  // mbase offsets to be the newBeamLength
  // mbaseM1 offsets to be the previousBeamLength
  mul   $mbase, $mbaseM1, $mscratch
  xor   $mbaseM1, $mbase, $mscratch

  ld32  $mout_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAM_LENGTH/4
  mov   $mptr, $mout_ptr

  rpt $mcount, (2f - 1f)/ 8 - 1
1:
  {ld32step $mcandidate_addend, $mzero, $maddend_ptr+=, 1  ;fnop}
  {ld32step $mcandidate_parent, $mzero, $mparent_ptr+=, 1  ;fnop}
  // mcandidate_parent = previousBeamLength[candidateParent[i]]
  {ld32     $mcandidate_parent, $mbaseM1, $mptr, $mcandidate_parent  ;fnop}
  // mcandidate_parent += (candidateAddend[i] != voidSymbol)
  {cmpne    $mscratch, $mcandidate_addend, $mvoid_symbol  ;fnop}
  {add      $mcandidate_parent, $mcandidate_parent, $mscratch  ;fnop}
  {st32step $mcandidate_parent, $mbase, $mout_ptr+=,1  ;fnop}
2:
  exitz $mzero

// Updating the probabilities is just a copy.  It need not be efficient
// as speed will be limited by update of the other parameters.
.Lprob_pnb\@:
  ld32 $min_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CANDIDATE_PNB/4
  ld32 $mout_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAM_PNB/4
  bri  1f

.Lprob_pb\@:
  ld32 $min_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CANDIDATE_PB/4
  ld32 $mout_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAM_PB/4
  bri  1f

.Lprob_ptotal\@:
  ld32 $min_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CANDIDATE_PTOTAL/4
  ld32 $mout_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAM_PTOTAL/4
  // Fall through
1:
  ld32 $mcount, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_BEAMWIDTH/4
  ldconst $minvalid_symbol, INVALID_SYMBOL
  ldconst $a1, LOG_PROBABILITY_ZERO_FLOAT
  ld32 $maddend_ptr, $mvertex_base, $mzero, VERTEX_STATE_OFFSET_CANDIDATE_ADDEND/4
  // Decrement as using brnzdec
  sub  $mcount, $mcount, 1
2:
  // Load the addend and check if its an invalid symbol
  // (Candidate was invalid as it was merged) - store probability zero if so
  ld32step $mscratch, $mzero, $maddend_ptr+=, 1
  cmpeq    $mscratch, $mscratch, $minvalid_symbol
  ld32step $a0, $mzero, $min_ptr+=, 1
  brz     $mscratch, 3f
  // addend == invalidSymbol
  mov      $a0, $a1
3:
  st32step $a0,  $mzero, $mout_ptr+=, 1
  brnzdec $mcount, 2b
  exitz $mzero

.size UPDATE_VERTEX_NAME, .-UPDATE_VERTEX_NAME

DEF_STACK_USAGE 0 UPDATE_VERTEX_NAME

.endm
//******************************************************************************
// Use the macro to instantiate the vertex

INSTANTIATE_UPDATE float unsigned_int

#endif
