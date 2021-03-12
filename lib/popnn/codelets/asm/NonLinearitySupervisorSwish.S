// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Assembly implementation of SWISH nonlinearity for popnn::NonLinearitySupervisor vertex.

// Restrictions
//
//  * At least 32-bit aligned source/destination address.

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "NonLinearitySwishCommon.S"

// Symbols
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearitySupervisor___half_popnn__NonLinearityType__SWISH
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearitySupervisor___float_popnn__NonLinearityType__SWISH

// Constants
#if defined(VECTOR_AVAIL_SCALED_PTR32)
#define DATA_PTR_VOFFSET 0
#define SIZE_VOFFSET 2
#else
#define DATA_PTR_VOFFSET 0
#define SIZE_VOFFSET 4
#endif

// Supervisor register aliases
#define SUPER_BASE m0
#define WORKER_ENTRY m1

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

#define MSCRATCH m10

DEF_STACK_USAGE 0 HALF_SYMBOL

.align 4
.globl HALF_SYMBOL
.supervisor
HALF_SYMBOL:
    setzi $WORKER_ENTRY, .Lhalf_worker
    runall $WORKER_ENTRY, $SUPER_BASE, 0
    sync TEXCH_SYNCZONE_LOCAL
    br $MSCRATCH
.align 8
.Lhalf_worker:
.worker
    ldz16 $MSCRATCH, $mvertex_base, $mzero, SIZE_VOFFSET/2

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
    st32 $RESULT_0, $DATA_PTR, $BASE, 0

.Lhalf_skip_32_bit_lead:
    ld32step $azero, $BASE, $DATA_PTR+=, 1

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
    add $SIZE, $SIZE, -1
    // Do the inner loop
    SwishActivationLoopHalfV4 DATA_PTR BASE SIZE CTXT_WORKERS

.Lhalf_64_bit_loop_exit:

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    brz $MSCRATCH, .Lhalf_end

.Lhalf_32_bit_remainder:
    brz $REM_32BIT, .Lhalf_16_bit_remainder

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    SwishActivationHalfV2 RESULT_0 ACTS_0
    st32step $RESULT_0, $BASE, $DATA_PTR+=, 1

.Lhalf_16_bit_remainder:
    brz $REM_16BIT, .Lhalf_end

    // Load the first and second half in the word to store along
    // with the remaining
    ldb16 $ACTS_0, $DATA_PTR, $BASE, 0
    SwishActivationHalfV2 RESULT_0 ACTS_0
    ldb16 $RESULT_1, $DATA_PTR, $BASE, 1
    roll16 $RESULT_0, $RESULT_0, $RESULT_1
    st32 $RESULT_0, $DATA_PTR, $BASE, 0

.Lhalf_end:
    exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL

DEF_STACK_USAGE 0 FLOAT_SYMBOL
.section .text.FLOAT_SYMBOL

.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

.supervisor
.globl FLOAT_SYMBOL
FLOAT_SYMBOL:
    setzi $WORKER_ENTRY, .Lfloat_worker
    runall $WORKER_ENTRY, $SUPER_BASE, 0
    sync TEXCH_SYNCZONE_LOCAL
    br $MSCRATCH

.align 8
.Lfloat_worker:
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

    // Check if address is 64-bit aligned
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lfloat_64_bit_aligned

.Lfloat_32_bit_aligned:
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .Lfloat_skip_32_bit_lead

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    SwishActivationFloatV1 RESULT_0 ACTS_0
    st32 $RESULT_0, $DATA_PTR, $BASE, 0

.Lfloat_skip_32_bit_lead:
    ld32step $azero, $BASE, $DATA_PTR+=, 1

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
    brz $SIZE, .Lfloat_64_bit_loop_exit
    add $SIZE, $SIZE, -1
    // Do the loop processing V2 floats at a time
    SwishActivationLoopFloatV2 DATA_PTR BASE SIZE CTXT_WORKERS

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
    st32step $RESULT_0, $BASE, $DATA_PTR+=, 1

.Lfloat_end:
    exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
