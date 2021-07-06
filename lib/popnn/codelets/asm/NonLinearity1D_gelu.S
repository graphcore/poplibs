// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of popnn::NonLinearity1D vertex template variations.

// Restrictions
//
//  * At least 32-bit aligned source/destination address.

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"

// Symbols
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearity1D___half_popnn__NonLinearityType__GELU
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearity1D___float_popnn__NonLinearityType__GELU

// Force all accumulators to zero
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// Constants
#if defined(VECTOR_AVAIL_SCALED_PTR32)
#define DATA_PTR_VOFFSET 0
#define SIZE_VOFFSET 2
#else
#define DATA_PTR_VOFFSET 0
#define SIZE_VOFFSET 4
#endif

#define RECIPROCAL_3_SHL17 ((((1 << 17) - 1) / 3) + 1)
#define LOG2_24_OVER_3 3
#define LOG2_12_OVER_3 2

#define HALF_1_0 0x3C00
#define HALF_0_5 0x3800
// The actual values used her for alpha and beta are chosen to
// reduce the error in the gradient and hence are slightly off from
// the actual.
#define HALF_ALPHA 0x3A6C  // 0.8027
#define HALF_BETA 0x29C8   // 4.526E-2
#define HALF_ALPHA_TIMES_BETA 0x28A6 // 3.632E-2
#define HALF_6_0 0x4600
#define HALF_MINUS_6_0 0xC600

#define FLOAT_12_0 0x41400000
#define FLOAT_MINUS_12_0 0xC1400000

#define FLOAT_1_0 0x3F800000
#define FLOAT_0_5 0x3F000000
#define FLOAT_ALPHA 0x3F4C422A  // 0.7978845608f
#define FLOAT_BETA 0x3D372713   // 0.044715

// all scratch offsets given in words
#define SCRATCH_OFFSET_CONST_ALPHA           0
#define SCRATCH_OFFSET_CONST_BETA            1
#define SCRATCH_OFFSET_CONST_1_0             2
#define SCRATCH_OFFSET_CONST_0_5             3

// Worker register aliases
#define WORKER_ID m0
#ifdef VECTOR_AVAIL_SCALED_PTR32
#define BASE m1
#else
#define BASE mzero
#endif
#define DATA_PTR m2
#define SIZE m3
#define REM m4
#define REM_32BIT m5
#define REM_16BIT m6

// Equivalent to $lr
#define MSCRATCH m10
#define MSCRATCH2 m11

#define ACTS_0 a0
#define ACTS_1 a1
#define ACTS_PAIR a0:1
#define RESULTS_0 a4
#define RESULTS_1 a5
#define RESULTS_PAIR a4:5
#define CONST_HI_1_0_LO_0_5 a3
#define CONST_SCRATCH_0 a0
#define CONST_SCRATCH_1 a1
#define ASCRATCH_0 a6
#define ASCRATCH_1 a7
#define ASCRATCH_PAIR a6:7
#define fpctl_cur a2

#define FLOAT_CLAMP_0 a2
#define FLOAT_CLAMP_1 a3
#define FLOAT_CLAMP_PAIR a2:3

// NOTE: Register definitions as well as Loop implementation are included here
#include "NonLinearityLoop_gelu.S"

DEF_STACK_USAGE 0 HALF_SYMBOL
.section .text.HALF_SYMBOL

.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

// All inputs must be separate registers
// Splits 64-bit chunks of n elements between workers.
// The result we want is n / (no. of worker contexts * elements per-64-bits).
// We achieve this by dividing by 3 first, by multiplying n by the reciprocal
// of 3 shifted left. This value is then shifted right by the same amount + any
// further division by powers of 2 to get the actual divisor we want.
// As an example, in this half case there are 4 halves per-64-bits and
// 6 worker contexts so the divisor we want is 24.
// (n / 3) / 8 = n / 24 so the extra divisor is 8, meaning an extra shift of 3.
.macro HALF_SPLIT_BETWEEN_WORKERS n size rem
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, (17 + LOG2_24_OVER_3)
    mul \rem, \size, 24
    sub \rem, \n, \rem
.endm

.align 8

    // For rpt alignment below.
    nop
HALF_SYMBOL:
.worker
    { 
      ldz16 $MSCRATCH, $mvertex_base, $mzero, SIZE_VOFFSET/2
      // calls to NonLinearityGELU_half use $ACTS_1 for mix. clear it here
      // so that arithmetic doesn't use rubbish.
      mov $ACTS_1, $azero
    }

    // $SIZE = No. of 64-bit elements each worker should process
    // $REM = No. of remaining elements between workers
    HALF_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

#if defined(VECTOR_AVAIL_SCALED_PTR32)
    // Scaled pointer gives offset in 32-bit units from
    // TMEM_REGION0_BASE_ADDR
    //
    ldz16 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/2
    shl $DATA_PTR, $DATA_PTR, 2
    setzi $BASE, TMEM_REGION0_BASE_ADDR
#else
    ld32 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
#endif

    // Clear overflow and NANOO to allow saturation. We can afford this as
    // all the terms in the approximation have the same sign (i.e. x, x^3)
    uget $fpctl_cur, CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK
    ldconst $ASCRATCH_0, (CSR_S_FP_ICTL__NANOO__MASK << CSR_S_FP_ICTL__NANOO__SHIFT) + (CSR_S_FP_ICTL__OFLO__MASK << CSR_S_FP_ICTL__OFLO__SHIFT)
    andc $ASCRATCH_0, $fpctl_cur, $ASCRATCH_0
    uput CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK, $ASCRATCH_0

    // Get worker ID
    {
      get $WORKER_ID, $WSR
      setzi $ASCRATCH_0, ZAACC_BITMASK
    }

    {
      and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK
      uput $FP_CLR, $ASCRATCH_0
    }

    // Load constants
    ldconst $CONST_HI_1_0_LO_0_5, (HALF_0_5) | (HALF_1_0 << 16)
    ldconst $ASCRATCH_0, (HALF_ALPHA_TIMES_BETA) | (HALF_ALPHA << 16)

    // Check if address is 64-bit aligned
    {
      and $MSCRATCH, $DATA_PTR, 0x7
      uput $TAS, $ASCRATCH_0
    }

    {
      brz $MSCRATCH, .Lhalf_64_bit_aligned
      fnop
    }
.Lhalf_32_bit_aligned:
    // Catch special case for just 1 or 2 elements at a 32-bit aligned address.
    setzi $MSCRATCH, 2
    cmpult $MSCRATCH, $MSCRATCH, $REM
    or $MSCRATCH, $MSCRATCH, $SIZE
    brnz $MSCRATCH, .Lhalf_32_bit_lead

    shr $REM_32BIT, $REM, 1
    and $REM_16BIT, $REM, 0x1
    shr $REM_32BIT, $REM_32BIT, $WORKER_ID
    shr $REM_16BIT, $REM_16BIT, $WORKER_ID

    bri .Lhalf_32_bit_remainder

.Lhalf_32_bit_lead:
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .Lhalf_skip_32_bit_lead

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    call $MSCRATCH, NonLinearityGELU_half
    st32 $RESULTS_0, $DATA_PTR, $BASE, 0

.Lhalf_skip_32_bit_lead:
    ld32step $ASCRATCH_0, $BASE, $DATA_PTR+=, 1

    // Decrement remaining element count
    add $REM, $REM, -2
    brpos $REM, .Lhalf_64_bit_aligned
    add $REM, $REM, (CTXT_WORKERS * 4)
    add $SIZE, $SIZE, -1

.Lhalf_64_bit_aligned:
    // $REM_32BIT = Non-zero if a remaining 32-bit load
    // $REM_16BIT = Non-zero if a remaining 16-bit load
    // $REM = No. of remaining 64-bit loads
    and $REM_32BIT, $REM, 0x2
    and $REM_16BIT, $REM, 0x1
    shr $REM, $REM, 2

    // Add any remaining 64-bit loads/stores possible to relevant
    // workers
    cmpult $MSCRATCH, $WORKER_ID, $REM
    add $SIZE, $SIZE, $MSCRATCH

    // Offset each worker's pointer into the data to interleave them.
    ld64step $azeros, $BASE, $DATA_PTR+=, $WORKER_ID

    brz $SIZE, .Lhalf_64_bit_loop_exit

    NONLINEARITY_GELU_HALF $SIZE $BASE CTXT_WORKERS

.Lhalf_64_bit_loop_exit:

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    brz $MSCRATCH, .Lhalf_end

.Lhalf_32_bit_remainder:
    brz $REM_32BIT, .Lhalf_16_bit_remainder

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    call $MSCRATCH, NonLinearityGELU_half
    st32step $RESULTS_0, $BASE, $DATA_PTR+=, 1

.Lhalf_16_bit_remainder:
    brz $REM_16BIT, .Lhalf_end

    // Load the first and second half in the word to store along
    // with the remaining
    ldb16 $ACTS_0, $DATA_PTR, $BASE, 0

    call $MSCRATCH, NonLinearityGELU_half

    ldb16 $RESULTS_1, $DATA_PTR, $BASE, 1
    roll16 $RESULTS_0, $RESULTS_0, $RESULTS_1
    st32 $RESULTS_0, $DATA_PTR, $BASE, 0

.Lhalf_end:
    uput CSR_W_FP_CTL__INDEX & CSR_W_WSR__CTXTID_M1__MASK, $fpctl_cur
    exitz $mzero
  
.size HALF_SYMBOL, .-HALF_SYMBOL

DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL

.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

// All inputs must be separate registers
// As described above in HALF_SPLIT_BETWEEN_WORKERS with different
// divisor.
.macro FLOAT_SPLIT_BETWEEN_WORKERS n size rem
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, (17 + LOG2_12_OVER_3)
    mul \rem, \size, 12
    sub \rem, \n, \rem
.endm

.align 8

    // For rpt alignment below.
    nop
FLOAT_SYMBOL:
.worker
    ldz16 $MSCRATCH, $mvertex_base, $mzero, SIZE_VOFFSET/2

    // $SIZE = No. of 64-bit elements each worker should process
    // $REM = No. of remaining elements between workers
    FLOAT_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

#if defined(VECTOR_AVAIL_SCALED_PTR32)
    // Scaled pointer gives offset in 32-bit units from
    // TMEM_REGION0_BASE_ADDR
    //
    ldz16 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/2
    shl $DATA_PTR, $DATA_PTR, 2
    setzi $BASE, TMEM_REGION0_BASE_ADDR
#else
    ld32 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
#endif

    // Get worker ID
    get $WORKER_ID, $WSR
    and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    // Load clamp values
    ldconst $FLOAT_CLAMP_0, FLOAT_MINUS_12_0
    ldconst $FLOAT_CLAMP_1, FLOAT_12_0
    ldconst $ASCRATCH_0, FLOAT_ALPHA
    st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
    ldconst $ASCRATCH_0, FLOAT_BETA
    st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_BETA
    or      $ASCRATCH_0, $azero, FLOAT_1_0
    st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
    or      $ASCRATCH_0, $azero, FLOAT_0_5
    st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_0_5

    // Check if address is 64-bit aligned
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lfloat_64_bit_aligned

.Lfloat_32_bit_aligned:
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .Lfloat_skip_32_bit_lead

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    call $MSCRATCH, NonLinearityGELU_float
    st32 $RESULTS_0, $DATA_PTR, $BASE, 0

.Lfloat_skip_32_bit_lead:
    ld32step $ASCRATCH_0, $BASE, $DATA_PTR+=, 1

    // Decrement remaining element count
    add $REM, $REM, -1
    brpos $REM, .Lfloat_64_bit_aligned
    add $REM, $REM, (CTXT_WORKERS * 2)
    add $SIZE, $SIZE, -1

.Lfloat_64_bit_aligned:
    // $SIZE = No. of 64-bit loads/stores possible
    // $REM_32BIT = No. of remaining 32-bit loads
    // $REM = No. of remaining 64-bit loads
    and $REM_32BIT, $REM, 0x1
    shr $REM, $REM, 1

    // Add any remaining 64-bit loads/stores possible to relevant
    // workers
    cmpult $MSCRATCH, $WORKER_ID, $REM
    add $SIZE, $SIZE, $MSCRATCH

    // Offset each worker's pointer into the data to interleave them.
    ld64step $ASCRATCH_PAIR, $BASE, $DATA_PTR+=, $WORKER_ID

    // Overlap 64-bit loads/stores with float calculations
    rpt $SIZE, (2f - 1f) / 8 - 1

1:
    {
      ld64 $ACTS_PAIR, $DATA_PTR, $BASE, 0
      fnop
    }

    {
      nop
      f32v2clamp $ASCRATCH_PAIR, $ACTS_PAIR, $FLOAT_CLAMP_PAIR
    }

    {
      ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_BETA
      f32v2mul $RESULTS_PAIR, $ASCRATCH_PAIR, $ASCRATCH_PAIR
    }

    {
      ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
      f32v2mul $RESULTS_PAIR, $CONST_SCRATCH_0:B, $RESULTS_PAIR
    }

    {
      ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
      f32v2add $RESULTS_PAIR, $CONST_SCRATCH_0:B, $RESULTS_PAIR
    }

    {
      nop
      f32v2mul $RESULTS_PAIR, $CONST_SCRATCH_0:B, $RESULTS_PAIR
    }

    {
      nop
      f32v2mul $RESULTS_PAIR, $RESULTS_PAIR, $ASCRATCH_PAIR
    }

    {
      ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
      f32tanh $RESULTS_0, $RESULTS_0
    }

    {
      ld32   $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_0_5
      f32tanh $RESULTS_1, $RESULTS_1
    }

    {
      ld64 $ACTS_PAIR, $DATA_PTR, $BASE, 0
      f32v2add $RESULTS_PAIR, $CONST_SCRATCH_0:B, $RESULTS_PAIR
    }

    {
      nop
      f32v2mul $RESULTS_PAIR, $ASCRATCH_0:B, $RESULTS_PAIR
    }

    {
      nop
      f32v2mul $RESULTS_PAIR, $RESULTS_PAIR, $ACTS_PAIR
    }

    {
      st64step $RESULTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
      fnop
    }
2:


.Lfloat_64_bit_loop_exit:

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    and $MSCRATCH, $MSCRATCH, $REM_32BIT
    brz $MSCRATCH, .Lfloat_end

.Lfloat_32_bit_remainder:
    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    call $MSCRATCH, NonLinearityGELU_float
    st32step $RESULTS_0, $BASE, $DATA_PTR+=, 1

.Lfloat_end:
    exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
