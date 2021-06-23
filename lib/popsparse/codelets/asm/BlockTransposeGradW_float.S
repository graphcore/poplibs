// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Given a flat input tensor of shape [Z][XY], transpose into a tensor of
// shape [XY/Bxy][Z/Bz][[Bxy][Bz]

#ifdef __IPU__
#include "BlockTransposeGradW.h.S"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__BlockTransposeGradW___float

// =============================================================================

DEF_STACK_USAGE 0 CODELET_NAME
.section .text.CODELET_NAME
.globl CODELET_NAME
.type CODELET_NAME, @function
.align 8
.worker

CODELET_NAME:

// Set up registers for the loop below
SETUP_FOR_MAIN_LOOP float

LXYBlockLoop:
  mov           $w_zBlockCnt, $w_numZBlocks
LZBlockLoop:
    {
      rpt             $w_numXYSubblocks, (LXYSubblockLoopEnd - LXYSubblockLoopStart)/8 - 1
      fnop
    }
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
        ldst64pace      $a2:3, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1000
        sort4x32hi      $a6:7, $a2:3, $a4:5
      }
      {
        // Done to extend the range for XY
        add             $w_in, $w_in, $w_inStrideSubblockXY
        fnop
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

.size CODELET_NAME, . - CODELET_NAME

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
