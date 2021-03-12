// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Given a input tensor of shape [Z][XY] that is flattened, transpose into a tensor of
// shape [XY/Bxy][Z/Bz][[Bxy][Bz]


#ifdef __IPU__
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"


// =============================================================================

#define STRIDE_BITS           10
#define STRIDE_BITS_MASK      ((1 << STRIDE_BITS) - 1)

// =============================================================================

#define VBase_in               0
#define VBase_out              4
#define VBase_blockSizeXY      8
#define VBase_numXYBlocks      10
#define VBase_numZ             12
#define VBase_maxBlocksXY      14

// worker registers
#define w_in                   m0
#define w_out                  m1
#define w_blockSizeXY          m2
#define w_numXYBlocks          m3
#define w_numZ                 m4
#define w_id                   m5
#define w_startBlockOff        m5
#define w_numBlocksXYThisWorker m3
#define w_maxBlocksXY          m6
#define w_temp                 m6

#define w_outStridesNumXY      m5
#define w_numXY                m7
#define w_inStrideBlockZ       m6
#define w_inStrideSubblockXY   m8
#define w_inStrideBlockXY      m9
#define w_tripack              m0:1
#define w_numXYSubblocks       m10
#define w_numZBlocks           m11
#define w_zBlockCnt            m2
#define w_xyBlockCnt           m4


// Macro to setup registers for the main loop
.macro SETUP_FOR_MAIN_LOOP TYPE

.ifc \TYPE, float

.equ BLOCK_SIZE_Z,          8
.equ SUBBLOCK_SIZE_XY,      2
.equ ELEMS_PER_LD_ST,       2
.equ LOG2_BLOCK_SIZE_Z,     3
.equ LOG2_ELEMS_PER_LD_ST,  1
.equ SIZEOF_DATA,           4
.equ LOG2_SUBBLOCK_SIZE_XY, 1

.else

.equ BLOCK_SIZE_Z,          16
.equ SUBBLOCK_SIZE_XY,      4
.equ ELEMS_PER_LD_ST,       4
.equ LOG2_BLOCK_SIZE_Z,     4
.equ LOG2_ELEMS_PER_LD_ST,  2
.equ SIZEOF_DATA,           2
.equ LOG2_SUBBLOCK_SIZE_XY, 2

.endif

get             $w_id, $WSR
and             $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

// Each BlockSizeXY is divided into subblocks and processed in the innermost loop
ld32            $w_in, $mvertex_base, VBase_in/4
ld32            $w_out, $mvertex_base, VBase_out/4
ldz16           $w_blockSizeXY, $mvertex_base, VBase_blockSizeXY/2
ldz16           $w_numXYBlocks, $mvertex_base, VBase_numXYBlocks/2
ldz16           $w_numZ, $mvertex_base, VBase_numZ/2
mul             $w_numXY, $w_blockSizeXY, $w_numXYBlocks
ldz16           $w_maxBlocksXY, $mvertex_base, VBase_maxBlocksXY/2

mul             $w_startBlockOff, $w_id, $w_maxBlocksXY
sub             $w_numBlocksXYThisWorker, $w_numXYBlocks, $w_startBlockOff
min             $w_numBlocksXYThisWorker, $w_numBlocksXYThisWorker, $w_maxBlocksXY
brpos           $w_numBlocksXYThisWorker, LCheckZero

LExit:
exitz           $mzero

LCheckZero:
brnzdec         $w_numBlocksXYThisWorker, LProcessBlocks
exitz           $mzero

LProcessBlocks:
mul             $w_startBlockOff, $w_startBlockOff, $w_blockSizeXY
.ifc \TYPE, float
ld32step        $mzero, $mzero, $w_in+=, $w_startBlockOff
.else
ldz16step       $mzero, $mzero, $w_in+=, $w_startBlockOff
.endif

mul             $w_temp, $w_startBlockOff, $w_numZ
.ifc \TYPE, float
ld32step        $mzero, $mzero, $w_out+=, $w_temp
.else
ldz16step       $mzero, $mzero, $w_out+=, $w_temp
.endif

// write strides:
.equ ST_STRIDE_1, (BLOCK_SIZE_Z/ELEMS_PER_LD_ST)
.equ LD_STRIDE_1, ((-BLOCK_SIZE_Z * (SUBBLOCK_SIZE_XY - 1) + SUBBLOCK_SIZE_XY)/ELEMS_PER_LD_ST)
ldconst         $w_outStridesNumXY, (ST_STRIDE_1 << STRIDE_BITS) + (LD_STRIDE_1 & STRIDE_BITS_MASK)
// read strides
// This is the increment to be applied before moving to the next Z Block
mul             $w_inStrideBlockZ, $w_numXY, BLOCK_SIZE_Z
sub             $w_inStrideBlockZ, $w_inStrideBlockZ, $w_blockSizeXY
mul             $w_inStrideBlockZ, $w_inStrideBlockZ, SIZEOF_DATA

mul             $w_inStrideSubblockXY, $w_numXY, -(BLOCK_SIZE_Z - 1) * SIZEOF_DATA
add             $w_inStrideSubblockXY, $w_inStrideSubblockXY, (SUBBLOCK_SIZE_XY - ELEMS_PER_LD_ST) * SIZEOF_DATA

// This is the increment applied to move to the next XY block
// -numXY * numZ + blockSizeXY
mul             $w_inStrideBlockXY, $w_numXY, $w_numZ
sub             $w_inStrideBlockXY, $w_blockSizeXY, $w_inStrideBlockXY
mul             $w_inStrideBlockXY, $w_inStrideBlockXY, SIZEOF_DATA

// Number of XY subblocks (rpt used)
shr             $w_numXYSubblocks, $w_blockSizeXY, LOG2_SUBBLOCK_SIZE_XY

// Max count for number of Z blocks: -1 because brnzdec is used
shr             $w_numZBlocks, $w_numZ, LOG2_BLOCK_SIZE_Z
add             $w_numZBlocks, $w_numZBlocks, -1

shl             $w_numXY, $w_numXY, (2 * STRIDE_BITS - LOG2_ELEMS_PER_LD_ST)
or              $w_outStridesNumXY, $w_outStridesNumXY, $w_numXY

// compact representation of read/write
tapack          $w_tripack, $m0, $m1, $m1

.endm

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
