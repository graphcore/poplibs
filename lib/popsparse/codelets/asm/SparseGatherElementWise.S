// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of  SparseGatherElementWise vertex template variations.

// Restrictions
//
//  * The input/output to be gathered from/to must be 64-bit aligned

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Symbols
#define CODELET_HALF __runCodelet_popsparse__SparseGatherElementWise___half
#define CODELET_FLOAT __runCodelet_popsparse__SparseGatherElementWise___float


// vertex state
#define s_rInPtr                   0
#define s_rOutPtr                  4
#define s_IndicesPtr               8
#define s_NumIndices               12
#define s_Offsets                  14

// worker register aliases
#define numIndices                 m0
#define numVectors                 m1
#define remainder                  m2
#define wId                        m3
#define addOffset                  m4
#define shift                      m5
#define addOffset1                 m6
#define addOffset2                 m7
#define addVectors                 m8
#define workerOffset               m9
#define indicesPtr                 m10
#define rInPtr                     m11
#define rOutPtr                    m0
#define delta                      m5
#define multiplex2                 m6
#define revWId                     m6

//------------------------------------------------------------------------------

// common macro to extract worker offsets and additional work. This can be done
// because the code uses the same registers.
.macro WORKER_DIVISION_AND_LOAD_STATE NUM_REMAINDER_BITS
ldz16          $numIndices, $mvertex_base, s_NumIndices/2
shr            $numVectors, $numIndices, \NUM_REMAINDER_BITS
and            $remainder, $numIndices, (1 << \NUM_REMAINDER_BITS) - 1

get            $wId, $WSR
and            $wId, $wId, CSR_W_WSR__CTXTID_M1__MASK

// Extract additional vector if any for this worker
ldz16          $addOffset, $mvertex_base, s_Offsets/2
shr            $addOffset1, $addOffset, $wId
and            $addVectors, $addOffset1, 0x1

// Find additional offset by summing up work done by previous workers
shr            $addOffset1, $addOffset1, 1
popc           $addOffset2, $addOffset1

// To position worker pointers
sub            $revWId, 5, $wId
mul            $workerOffset, $numVectors, $revWId
add            $workerOffset, $workerOffset, $addOffset2

// These are the total number of vectors to process
add            $numVectors, $numVectors, $addVectors

// Load pointers
ld32           $indicesPtr, $mvertex_base, s_IndicesPtr/4
ld32           $rInPtr, $mvertex_base, s_rInPtr/4
ld32           $rOutPtr, $mvertex_base, s_rOutPtr/4

.endm

//------------------------------------------------------------------------------

DEF_STACK_USAGE  0  CODELET_HALF

.section .text.CODELET_HALF
.align 8
.globl CODELET_HALF
.type CODELET_HALF, @function
.worker
CODELET_HALF:

WORKER_DIVISION_AND_LOAD_STATE 2

// Move to offset for worker
ld64step       $azeros, $mzero, $indicesPtr+=, $workerOffset
ld64step       $azeros, $mzero, $rOutPtr+=, $workerOffset
ldz16step      $delta, $mzero, $indicesPtr+=, 1

brz            $numVectors, LCheckRemainderHalf
add            $numVectors, $numVectors, -1

ldd16b16       $a0, $indicesPtr++, $rInPtr, $delta@
ldd16b16       $a1, $indicesPtr++, $rInPtr, $delta@
{
  ldd16b16       $a0, $indicesPtr++, $rInPtr, $delta@
  sort4x16lo     $a2, $a0, $a1
}
ldd16b16       $a1, $indicesPtr++, $rInPtr, $delta@

rpt $numVectors, (LLoopEndHalf - LLoopStartHalf)/8 -1
LLoopStartHalf:
  {
    ldd16b16       $a0, $indicesPtr++, $rInPtr, $delta@
    sort4x16lo     $a3, $a0, $a1
  }
  {
    st64step       $a2:3, $mzero, $rOutPtr+=, 1
    fnop
  }
  {
    ldd16b16       $a1, $indicesPtr++, $rInPtr, $delta@
    fnop
  }
  {
    ldd16b16       $a0, $indicesPtr++, $rInPtr, $delta@
    sort4x16lo     $a2, $a0, $a1
  }
  {
    ldd16b16       $a1, $indicesPtr++, $rInPtr, $delta@
    fnop
  }
LLoopEndHalf:
  sort4x16lo     $a3, $a0, $a1
  st64step       $a2:3, $mzero, $rOutPtr+=, 1

LCheckRemainderHalf:
  // Only worker zero does extra work

  brnz           $wId, LExitHalf
  brz            $remainder, LExitHalf
  and            $multiplex2, $remainder, 0x2
  brz            $multiplex2, LFinalElementHalf

  // process two samples
  ldd16b16       $a0, $indicesPtr++, $rInPtr, $delta@
  ldd16b16       $a1, $indicesPtr++, $rInPtr, $delta@
  sort4x16lo     $a2, $a0, $a1

  st32step       $a2, $mzero, $rOutPtr+=, 1

LFinalElementHalf:

  and            $remainder, $remainder, 0x1
  brz            $remainder, LExitHalf
  ldb16          $a0, $delta, $rInPtr, 0
  ldb16          $a1, $rOutPtr, 0
  sort4x16lo     $a2, $a0, $a1
  st32           $a2, $rOutPtr, 0
LExitHalf:
  exitz          $mzero

.size CODELET_HALF, . - CODELET_HALF

//------------------------------------------------------------------------------

DEF_STACK_USAGE  0  CODELET_FLOAT

.section .text.CODELET_FLOAT
.align 8
.globl CODELET_FLOAT
.type CODELET_FLOAT, @function
.worker
CODELET_FLOAT:

WORKER_DIVISION_AND_LOAD_STATE 1

// Move to offset for worker: dummy reads
ld32step       $azero, $mzero, $indicesPtr+=, $workerOffset
ld64step       $azeros, $mzero, $rOutPtr+=, $workerOffset
ldz16step      $delta, $mzero, $indicesPtr+=, 1

brz            $numVectors, LCheckRemainderFloat
add            $numVectors, $numVectors, -1

ldd16a32       $a0, $indicesPtr++, $rInPtr, $delta@
ldd16a32       $a1, $indicesPtr++, $rInPtr, $delta@

{
  rpt $numVectors, (LLoopEndFloat - LLoopStartFloat)/8 -1
  fnop
}
LLoopStartFloat:
  {
    st64step     $a0:1, $mzero, $rOutPtr+=, 1
    fnop
  }
  {
    ldd16a32       $a0, $indicesPtr++, $rInPtr, $delta@
    fnop
  }
  {
    ldd16a32       $a1, $indicesPtr++, $rInPtr, $delta@
    fnop
  }
LLoopEndFloat:
  st64step       $a0:1, $mzero, $rOutPtr+=, 1

LCheckRemainderFloat:
  // Only worker zero does extra work

  brnz           $wId, LExitFloat
  brz            $remainder, LExitFloat

  ld32           $a0, $rInPtr, $delta, 0
  st32           $a0, $rOutPtr, 0

LExitFloat:
  exitz          $mzero

//------------------------------------------------------------------------------


#endif // __IPU__
