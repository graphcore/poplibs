// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Given a input tensor of shape [Z][XY] that is flattened, transpose into a tensor of
// shape [XY/Bxy][Z/Bz][[Bxy][Bz]

#ifdef __IPU__

#include "BlockTransposeGradW.h.S"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__BlockTransposeGradW___half

DEF_STACK_USAGE 0 CODELET_NAME
.section .text.CODELET_NAME
.globl CODELET_NAME
.type CODELET_NAME, @function
.align 8
.worker

CODELET_NAME:

// Set up registers for the loop below
SETUP_FOR_MAIN_LOOP half

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
        fnop
      }
      {
        ld2x64pace      $a6:7, $a0:1, $w_tripack+=, $w_numXY, 0b0111
        f16v4istacc     $azeros, $a0:1, $a2:3, TISTACC_P0
      }
      {
        ld2x64pace      $a0:1, $a2:3, $w_tripack+=, $w_numXY, 0b0111
        f16v4istacc     $azeros, $a4:5, $a6:7, TISTACC_P1
      }
      {
        ld2x64pace      $a2:3, $azeros, $w_tripack+=, $w_numXY, 0b0111
        f16v4stacc      $a6:7, TSTACC_P0
      }
      // Store first 4x4 subblock
      {
        ldst64pace      $a4:5, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1011
        f16v4stacc      $a6:7, TSTACC_P1
      }
      {
        ldst64pace      $a6:7, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1011
        f16v4istacc     $a2:3, $a0:1, $a2:3, TISTACC_P0
      }
      {
        ldst64pace      $a0:1, $a2:3, $w_tripack+=, $w_outStridesNumXY, 0b1011
        f16v4istacc     $a2:3, $a4:5, $a6:7, TISTACC_P1
      }
      {
        ldst64pace      $a2:3, $a2:3, $w_tripack+=, $w_outStridesNumXY, 0b0111
        f16v4stacc      $a6:7, TSTACC_P0
      }
      // Store second 4x4 subblock
      {
        ldst64pace      $a4:5, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1011
        f16v4stacc      $a6:7, TSTACC_P1
      }
      {
        ldst64pace      $a6:7, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1011
        f16v4istacc     $a2:3, $a0:1, $a2:3, TISTACC_P0
      }
      {
        ldst64pace      $a0:1, $a2:3, $w_tripack+=, $w_outStridesNumXY, 0b1011
        f16v4istacc     $a2:3, $a4:5, $a6:7, TISTACC_P1
      }
      {
        ldst64pace      $a2:3, $a2:3, $w_tripack+=, $w_outStridesNumXY, 0b0111
        f16v4stacc      $a6:7, TSTACC_P0
      }
      // Third 4x4 subblock
      {
        ldst64pace      $a4:5, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1011
        f16v4stacc      $a6:7, TSTACC_P1
      }
      {
        ldst64pace      $a6:7, $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b1000
        f16v4istacc     $a2:3, $a0:1, $a2:3, TISTACC_P0
      }
      {
        add             $w_in, $w_in, $w_inStrideSubblockXY
        fnop
      }
      {
        st64pace        $a2:3, $w_tripack+=, $w_outStridesNumXY, 0b10
        f16v4istacc     $a2:3, $a4:5, $a6:7, TISTACC_P1
      }
      {
        st64pace        $a2:3, $w_tripack+=, $w_outStridesNumXY, 0b01
        f16v4stacc      $a6:7, TSTACC_P0
      }

      // Fourth 4x4 subblock
      {
        st64pace        $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b10
        f16v4stacc      $a6:7, TSTACC_P1
      }
      {
        st64pace        $a6:7, $w_tripack+=, $w_outStridesNumXY, 0b10
        f16v4istacc     $a2:3, $azeros, $azeros, TISTACC_P0
      }
      {
        st64pace        $a2:3, $w_tripack+=, $w_outStridesNumXY, 0b10
        f16v4istacc     $a2:3, $azeros, $azeros, TISTACC_P1
      }
      {
        st64pace        $a2:3, $w_tripack+=, $mzero, 0b00
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
