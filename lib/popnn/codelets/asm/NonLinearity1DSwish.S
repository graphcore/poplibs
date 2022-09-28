// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of SWISH nonlinearity for popnn::NonLinearity1D vertex.

// Restrictions
//
//  * At least 32-bit aligned source/destination address.

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "NonLinearitySwishCommon.S"

// Symbols
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearity1D___half_popnn__NonLinearityType__SWISH
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearity1D___float_popnn__NonLinearityType__SWISH
#define HALF_SYMBOL_INPLACE \
  __runCodelet_popnn__NonLinearity1DInPlace___half_popnn__NonLinearityType__SWISH
#define FLOAT_SYMBOL_INPLACE \
  __runCodelet_popnn__NonLinearity1DInPlace___float_popnn__NonLinearityType__SWISH

// Constants
#define DATA_PTR_VOFFSET 0
#define INPLACE_SIZE_VOFFSET 4
// Non inplace
#define OUT_PTR_VOFFSET 4
#define SIZE_VOFFSET 8

// Worker register aliases
#define WORKER_ID m0
#define BASE mzero
#define DATA_PTR m2
#define SIZE m3
#define REM m4
#define REM_32BIT m5
#define REM_16BIT m6
#define OUT_PTR m7

#define MSCRATCH m10

.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
DEF_STACK_USAGE 0 HALF_SYMBOL
    .align 4
HALF_SYMBOL:
.worker
    ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
    ldz16 $MSCRATCH, $mvertex_base, $mzero, SIZE_VOFFSET/2
    bri HALF_SYMBOL_COMMON
.size HALF_SYMBOL, .-HALF_SYMBOL


.section .text.HALF_SYMBOL_INPLACE
DEF_STACK_USAGE 0 HALF_SYMBOL_INPLACE


.globl HALF_SYMBOL_INPLACE
    .align 8
HALF_SYMBOL_INPLACE:
.worker
    ld32 $OUT_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
    ldz16 $MSCRATCH, $mvertex_base, $mzero, INPLACE_SIZE_VOFFSET/2

HALF_SYMBOL_COMMON:
    // $SIZE = No. of 64-bit elements each worker should process
    // $REM = No. of remaining elements between workers
    HALF_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

    {ld32 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
     fnop  } // Repeat alignment

    // Get worker ID
    get $WORKER_ID, $WSR
    and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    // Check if address is 64-bit aligned
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lhalf_64_bit_aligned

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
    SwishActivationHalfV2 RESULT_0 ACTS_0
    st32 $RESULT_0, $OUT_PTR, $BASE, 0

.Lhalf_skip_32_bit_lead:
    ld32step $azero, $BASE, $DATA_PTR+=, 1
    ld32step $azero, $BASE, $OUT_PTR+=, 1

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
    ld64step $azeros, $BASE, $OUT_PTR+=, $WORKER_ID
    brz $SIZE, .Lhalf_64_bit_loop_exit
    add $SIZE, $SIZE, -1
    // Do the inner loop
    SwishActivationLoopHalfV4 DATA_PTR OUT_PTR BASE SIZE CTXT_WORKERS

.Lhalf_64_bit_loop_exit:

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    brz $MSCRATCH, .Lhalf_end

.Lhalf_32_bit_remainder:
    brz $REM_32BIT, .Lhalf_16_bit_remainder

    ld32step $ACTS_0, $BASE, $DATA_PTR+=, 1
    SwishActivationHalfV2 RESULT_0 ACTS_0
    st32step $RESULT_0, $BASE, $OUT_PTR+=, 1

.Lhalf_16_bit_remainder:
    brz $REM_16BIT, .Lhalf_end

    // Load the first and second half in the word to store along
    // with the remaining
    ldb16 $ACTS_0, $DATA_PTR, $BASE, 0
    SwishActivationHalfV2 RESULT_0 ACTS_0
    ldb16 $RESULT_1, $OUT_PTR, $BASE, 1
    roll16 $RESULT_0, $RESULT_0, $RESULT_1
    st32 $RESULT_0, $OUT_PTR, $BASE, 0

.Lhalf_end:
    exitz $mzero

.size HALF_SYMBOL_INPLACE, .-HALF_SYMBOL_INPLACE

DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL

.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

.align 4
FLOAT_SYMBOL:
    .worker
    ld32 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET/4
    ldz16 $MSCRATCH, $mvertex_base, $mzero, SIZE_VOFFSET/2
    bri FLOAT_SYMBOL_COMMON
 .size FLOAT_SYMBOL, .-FLOAT_SYMBOL

.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

DEF_STACK_USAGE 0 FLOAT_SYMBOL_INPLACE
.section .text.FLOAT_SYMBOL_INPLACE

.globl FLOAT_SYMBOL_INPLACE
.type FLOAT_SYMBOL_INPLACE, @function

.align 8
FLOAT_SYMBOL_INPLACE:
.worker
    ld32 $OUT_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
    ldz16 $MSCRATCH, $mvertex_base, $mzero, INPLACE_SIZE_VOFFSET/2

FLOAT_SYMBOL_COMMON:

    // $SIZE = No. of 64-bit elements each worker should process
    // $REM = No. of remaining elements between workers
    FLOAT_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

    {ld32 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET/4
    fnop} // Rpt alignment

    // Get worker ID
    get $WORKER_ID, $WSR
    and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    // Check if address is 64-bit aligned
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lfloat_64_bit_aligned

.Lfloat_32_bit_aligned:
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .Lfloat_skip_32_bit_lead

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    SwishActivationFloatV1 RESULT_0 ACTS_0
    st32 $RESULT_0, $OUT_PTR, $BASE, 0

.Lfloat_skip_32_bit_lead:
    ld32step $azero, $BASE, $DATA_PTR+=, 1
    ld32step $azero, $BASE, $OUT_PTR+=, 1

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
    ld64step $azeros, $BASE, $DATA_PTR+=, $WORKER_ID
    ld64step $azeros, $BASE, $OUT_PTR+=, $WORKER_ID
    brz $SIZE, .Lfloat_64_bit_loop_exit
    add $SIZE, $SIZE, -1
    // Do the loop processing V2 floats at a time
    SwishActivationLoopFloatV2 DATA_PTR OUT_PTR BASE SIZE CTXT_WORKERS

.Lfloat_64_bit_loop_exit:

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    and $MSCRATCH, $MSCRATCH, $REM_32BIT
    brz $MSCRATCH, .Lfloat_end

.Lfloat_32_bit_remainder:
    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    SwishActivationFloatV1 RESULT_0 ACTS_0
    st32step $RESULT_0, $BASE, $OUT_PTR+=, 1

.Lfloat_end:
    exitz $mzero

.size FLOAT_SYMBOL_INPLACE, .-FLOAT_SYMBOL_INPLACE

#endif // __IPU__
