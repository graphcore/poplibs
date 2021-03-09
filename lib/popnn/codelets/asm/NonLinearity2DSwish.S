// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of SWISH for popnn::NonLinearity2D vertex.

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "NonLinearitySwishCommon.S"

// Symbols
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearity2D___half_popnn__NonLinearityType__SWISH
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearity2D___float_popnn__NonLinearityType__SWISH

// Constants
#define BASE_AND_N0_VOFFSET 0
#define DELTAN_PTR_VOFFSET 4

#define PTR16_SHL_BITS 1
#define PTR32_SHL_BITS 2

#if defined(VECTORLIST_AVAIL_DELTAN)
#define DELTAN_HALF_OFFSET_BITS 18
#define DELTAN_HALF_OFFSET_MASK ((1 << DELTAN_HALF_OFFSET_BITS) - 1)
#define DELTAN_FLOAT_OFFSET_BITS 18
#define DELTAN_FLOAT_OFFSET_MASK ((1 << DELTAN_FLOAT_OFFSET_BITS) - 1)
#else
#define DELTAN_HALF_OFFSET_BITS 20
#define DELTAN_HALF_OFFSET_MASK ((1 << DELTAN_HALF_OFFSET_BITS) - 1)
#define DELTAN_FLOAT_OFFSET_BITS 19
#define DELTAN_FLOAT_OFFSET_MASK ((1 << DELTAN_FLOAT_OFFSET_BITS) - 1)
#endif


// Worker register aliases
#define MASK m0
#if defined(VECTORLIST_AVAIL_DELTAN)
#define MEMORY_BASE m1
#else
#define MEMORY_BASE mzero
#endif
#define BASE_PTR m2
#define N0 m3
#define N0_B m6
#define DELTAN_PTR m4
#define DATA_PTR m5
#define N1 m6
#define N1_64BIT m7

// Equivalent to $lr
#define MSCRATCH m10
#define MSCRATCH2 m11

DEF_STACK_USAGE 0 HALF_SYMBOL

.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

.align 8
HALF_SYMBOL:
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET/4
#if defined(VECTORLIST_AVAIL_DELTAN)
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/2
#else
    ld32 $MSCRATCH2, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/4
#endif

    // Unpack base pointer and n0
#if defined(VECTORLIST_AVAIL_DELTAN)
    setzi $MASK, DELTAN_BASE_PTR_MASK
#else
    ldconst $MASK, DELTAN_BASE_PTR_MASK
#endif
    and $BASE_PTR, $MSCRATCH, $MASK
    shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS
 
#if defined(VECTORLIST_AVAIL_DELTAN)
    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, PTR32_SHL_BITS
#else
    // DeltaN table pointer contains a 24 bit absolute address
    // followed by the upper 8 bits of N0. Combine with the
    // lower N0 bits which has alraedy been loaded from the
    // upper 8 bits of the Base pointer
    and $DELTAN_PTR, $MSCRATCH2, $MASK
    shr $N0_B, $MSCRATCH2, DELTAN_BASE_PTR_BITS
    shl $N0, $N0, 8
    or $N0, $N0, $N0_B
#endif

    setzi $MASK, DELTAN_HALF_OFFSET_MASK

    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.Lhalf_n0_loop:
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $MASK
#if !defined(VECTORLIST_AVAIL_DELTAN)
    shl $DATA_PTR, $DATA_PTR, PTR16_SHL_BITS
#endif
    shr $N1, $MSCRATCH, DELTAN_HALF_OFFSET_BITS
    // Actually offset DATA_PTR so that below alignment checks
    // take BASE_PTR alignment into account
    add $DATA_PTR, $BASE_PTR, $DATA_PTR

    and $MSCRATCH, $DATA_PTR, 0x3
    brz $MSCRATCH, .Lhalf_32_bit_aligned

    // Handle the first 16-bit element. We'll always have
    // at least 1 element here.
    andc $DATA_PTR, $DATA_PTR, 0x3
    ldb16 $ACTS_0, $DATA_PTR, $mzero, 1
    // Could be bundled better to use slots in the macro
    SwishActivationHalfV2 RESULT_0 ACTS_0
    ldb16 $RESULT_1, $DATA_PTR, $mzero, 0
    {
      add $N1, $N1, -1
      roll16 $RESULT_0, $RESULT_1, $RESULT_0
    }
    st32step $RESULT_0, $mzero, $DATA_PTR+=, 1
    brz $N1, .Lhalf_n0_loop_cond

.Lhalf_32_bit_aligned:
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lhalf_64_bit_aligned

    // Special case for a single 16-bit element at 32-bit
    // aligned address.
    cmpult $MSCRATCH, $N1, 2
    brnz $MSCRATCH, .Lhalf_16_bit_remainder

    ld32 $ACTS_0, $DATA_PTR, $mzero, 0     
    SwishActivationHalfV2 RESULT_0 ACTS_0
    add $N1, $N1, -2
    st32step $RESULT_0, $mzero, $DATA_PTR+=, 1

.Lhalf_64_bit_aligned:
    shr $N1_64BIT, $N1, 2
    brz $N1_64BIT, .Lhalf_32_bit_remainder
    {
      add $N1_64BIT, $N1_64BIT, -1
      fnop
    }
    SwishActivationLoopHalfV4 DATA_PTR mzero N1_64BIT 1

.Lhalf_32_bit_remainder:
    and $MSCRATCH, $N1, 0x2
    brz $MSCRATCH, .Lhalf_16_bit_remainder

    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    SwishActivationHalfV2 RESULT_0 ACTS_0
    st32step $RESULT_0, $mzero, $DATA_PTR+=, 1

.Lhalf_16_bit_remainder:
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .Lhalf_n0_loop_cond

    ldb16 $ACTS_0, $DATA_PTR, $mzero, 0
    SwishActivationHalfV2 RESULT_0 ACTS_0
    ldb16 $RESULT_1, $DATA_PTR, $mzero, 1
    roll16 $RESULT_0, $RESULT_0, $RESULT_1
    st32step $RESULT_0, $mzero, $DATA_PTR+=, 1

.Lhalf_n0_loop_cond:
    brnzdec $N0, .Lhalf_n0_loop
    exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL

#-------------------------------------------------------------------------------

DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

.align 8
FLOAT_SYMBOL:
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET/4
#if defined(VECTORLIST_AVAIL_DELTAN)
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/2
    setzi $MASK, DELTAN_BASE_PTR_MASK
#else
    ld32 $MSCRATCH2, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET/4
    ldconst $MASK, DELTAN_BASE_PTR_MASK
#endif

    // Unpack base pointer and n0
    and $BASE_PTR, $MSCRATCH, $MASK
    shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

#if defined(VECTORLIST_AVAIL_DELTAN)
    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, PTR32_SHL_BITS
#else
    // DeltaN table pointer contains a 24 bit absolute address
    // followed by the upper 8 bits of N0. Combine with the
    // lower N0 bits which has alraedy been loaded from the
    // upper 8 bits of the Base pointer
    and $DELTAN_PTR, $MSCRATCH2, $MASK
    shr $N0_B, $MSCRATCH2, DELTAN_BASE_PTR_BITS
    shl $N0, $N0, 8
    or $N0, $N0, $N0_B
#endif

    setzi $MASK, DELTAN_FLOAT_OFFSET_MASK

    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.Lfloat_n0_loop:
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $MASK
#if !defined(VECTORLIST_AVAIL_DELTAN)
    shl $DATA_PTR, $DATA_PTR, PTR32_SHL_BITS
#endif
    shr $N1, $MSCRATCH, DELTAN_FLOAT_OFFSET_BITS
    // Actually offset DATA_PTR so that below alignment checks
    // take BASE_PTR alignment into account
    add $DATA_PTR, $BASE_PTR, $DATA_PTR

    // DATA_PTR and N1 give us the regions to actually loop
.Lfloat_32_bit_aligned:
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lfloat_64_bit_aligned

    // Handle the first 32-bit element. We'll always have
    // at least 1 element here.
    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    SwishActivationFloatV1 RESULT_0 ACTS_0
    st32step $RESULT_0, $mzero, $DATA_PTR+=, 1
    add $N1, $N1, -1

.Lfloat_64_bit_aligned:
    shr $N1_64BIT, $N1, 1
    ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 0
    brz $N1_64BIT, .Lfloat_32_bit_remainder
    {
      add $N1_64BIT, $N1_64BIT, -1
      fnop
    }
    SwishActivationLoopFloatV2 DATA_PTR mzero N1_64BIT 1

.Lfloat_32_bit_remainder:
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .Lfloat_n0_loop_cond
    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    SwishActivationFloatV1 RESULT_0 ACTS_0
    st32step $RESULT_0, $mzero, $DATA_PTR+=, 1

.Lfloat_n0_loop_cond:
    brnzdec $N0, .Lfloat_n0_loop
    exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
