// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Given a flat input tensor of shape [Z][XY], transpose into a tensor of
// shape [XY/Bxy][Z/Bz][[Bxy][Bz]

#ifdef __IPU__
#include "BlockTransposeGradW.h.S"

// =============================================================================

#define WORKER_FN BlockTransposeGradWF

// =============================================================================

DEF_STACK_USAGE 0 WORKER_FN
.section .text.WORKER_FN, FUNCTION_IS_WORKER
.type WORKER_FN, @function
.align 8
.worker
// worker code
WORKER_FN:

// Set up registers for the loop below
SETUP_FOR_MAIN_LOOP float

LXYBlockLoop:
  mov           $w_zBlockCnt, $w_numZBlocks
LZBlockLoop:
    rpt             $w_numXYSubblocks, (LXYSubblockLoopEnd - LXYSubblockLoopStart)/8 - 1
LXYSubblockLoopStart:
      // dummy load from the write address
      {
        ld2x64pace      $a0:1, $azeros, $w_tripack+=, $w_numXY, 0b0111
        fnop
      }
      {
        ld2x64pace      $a2:3, $azeros, $w_tripack+=, $w_numXY, 0b0111
        fnop
      }
      {
        ld2x64pace      $a4:5, $azeros, $w_tripack+=, $w_numXY, 0b0111
        sort4x32lo      $a6:7, $a0:1, $a2:3
      }
      {
        ldst64pace      $a0:1, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1011
        sort4x32hi      $a6:7, $a0:1, $a2:3
      }
      {
        ldst64pace      $a2:3, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b0111
        sort4x32lo      $a6:7, $a4:5, $a0:1
      }
      {
        ldst64pace      $a4:5, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1011
        sort4x32hi      $a6:7, $a4:5, $a0:1
      }
      {
        ldst64pace      $a0:1, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b0111
        sort4x32lo      $a6:7, $a2:3, $a4:5
      }
      {
        ldst64pace      $a2:3, $a6:7, $w_tripack+=, $w_inStrideSubblockXY, 0b0110
        sort4x32hi      $a6:7, $a2:3, $a4:5
      }

      {
        st64pace        $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b01
        sort4x32lo      $a6:7, $a0:1, $a2:3
      }
      {
        st64pace        $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b10
        sort4x32hi      $a6:7, $a0:1, $a2:3
      }
      {
        st64pace        $a6:7, $w_tripack+=, $mzero, 0b00
        fnop
      }
LXYSubblockLoopEnd:
    add             $w_in, $w_in, $w_inStrideBlockZ
    brnzdec         $w_zBlockCnt, LZBlockLoop
  add             $w_in, $w_in, $w_inStrideBlockXY
  brnzdec         $w_numBlocksXYThisWorker, LXYBlockLoop
exitz           $mzero

.size WORKER_FN, . - WORKER_FN

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__BlockTransposeGradW___float

DEF_STACK_USAGE 0 CODELET_NAME

.section .text.CODELET_NAME
.align 4
.globl CODELET_NAME
.type CODELET_NAME, @function
CODELET_NAME:
.supervisor

setzi           $m1, WORKER_FN
runall          $m1, $m0, 0
br              $lr
.size CODELET_NAME, . - CODELET_NAME

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
