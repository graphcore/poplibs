// Required defines:
//
// HALF_SYMBOL_NAME
// FLOAT_SYMBOL_NAME
// OP_UNIQUE_PREFIX
// CALC_HALF(dst, src)
// CALC_FLOAT(dst, src)

// Constants
#define DATA_PTR_VOFFSET 0
#define SIZE_VOFFSET 1

// Supervisor register aliases
#define SUPER_BASE m0
#define WORKER_ENTRY m1

// Worker register aliases
#define WORKER_ID m0
#define BASE m1
#define DATA_PTR m2
#define SIZE m3
#define REM m4
#define REM_32BIT m5
#define REM_16BIT m6
#define MSCRATCH m10

#define ACTS_0 a0
#define ACTS_1 a1
#define ACTS_PAIR a0:1
#define RESULTS_0 a4
#define RESULTS_1 a5
#define RESULTS_PAIR a4:5
#define ASCRATCH a6
#define ASCRATCH_PAIR a6:7

#define PREFIX_PASTE(p, s) p ## s
#define PREFIX_EVAL(p, s) PREFIX_PASTE(p, s)
#define PREFIX(s) PREFIX_EVAL(OP_UNIQUE_PREFIX, s)

.globl HALF_SYMBOL_NAME
.type HALF_SYMBOL_NAME, @function

.align 8
PREFIX(_half_worker):
    ldz16 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET
    ldz16 $SIZE, $mvertex_base, $mzero, SIZE_VOFFSET

    // Scaled pointer gives offset in 32-bit units from
    // TMEM_REGION0_BASE_ADDR
    shl $DATA_PTR, $DATA_PTR, 2
    setzi $BASE, TMEM_REGION0_BASE_ADDR

    // Get worker ID
    get $WORKER_ID, $WSR
    and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    // Extract packed size + remainder
    // $SIZE = No. of elements / (CTXT_WORKERS * elements per 64-bits)
    // $REM = No. of remaining elements
    and $REM, $SIZE, 0x1F
    shr $SIZE, $SIZE, 5

    // Check if address is 64-bit aligned
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .PREFIX(_half_64_bit_aligned)

.PREFIX(_half_32_bit_aligned):
    // Catch special case for just 1 or 2 elements at a 32-bit aligned address.
    setzi $MSCRATCH, 2
    cmpult $MSCRATCH, $MSCRATCH, $REM
    or $MSCRATCH, $MSCRATCH, $SIZE
    brnz $MSCRATCH, .PREFIX(_half_32_bit_lead)

    // Just 1 or 2 elements remain. Skip to remainder handling, only calculate
    // with a single worker (worker 0).
    shr $REM_32BIT, $REM, 1
    and $REM_16BIT, $REM, 0x1
    shr $REM_32BIT, $REM_32BIT, $WORKER_ID
    shr $REM_16BIT, $REM_16BIT, $WORKER_ID
    bri .PREFIX(_half_32_bit_remainder)

.PREFIX(_half_32_bit_lead):
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .PREFIX(_half_skip_32_bit_lead)

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    CALC_HALF($RESULTS_0, $ACTS_0)
    st32 $RESULTS_0, $DATA_PTR, $BASE, 0

.PREFIX(_half_skip_32_bit_lead):
    ld32step $ASCRATCH, $BASE, $DATA_PTR+=, 1

    // Decrement remaining element count
    add $REM, $REM, -2
    brpos $REM, .PREFIX(_half_64_bit_aligned)
    add $REM, $REM, (CTXT_WORKERS * 4)
    add $SIZE, $SIZE, -1

.PREFIX(_half_64_bit_aligned):
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
    ld64step $ASCRATCH_PAIR, $BASE, $DATA_PTR+=, $WORKER_ID

    // Overlap 64-bit loads/stores with vector 2 half calculations
    brz $SIZE, .PREFIX(_half_64_bit_loop_exit)
    add $SIZE, $SIZE, -1
    ld64 $ACTS_PAIR, $DATA_PTR, $BASE, 0
    {
      rpt $SIZE, (.PREFIX(_half_rpt_end) - .PREFIX(_half_rpt_start)) / 8 - 1
      CALC_HALF($RESULTS_0, $ACTS_0)
    }
    .PREFIX(_half_rpt_start):
    {
      ld64 $ACTS_PAIR, $DATA_PTR, $BASE, CTXT_WORKERS
      CALC_HALF($RESULTS_1, $ACTS_1)
    }
    {
      st64step $RESULTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
      CALC_HALF($RESULTS_0, $ACTS_0)
    }
    .PREFIX(_half_rpt_end):
    CALC_HALF($RESULTS_1, $ACTS_1)
    st64step $RESULTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
.PREFIX(_half_64_bit_loop_exit):

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    brz $MSCRATCH, .PREFIX(_half_end)

.PREFIX(_half_32_bit_remainder):
    brz $REM_32BIT, .PREFIX(_half_16_bit_remainder)

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    CALC_HALF($RESULTS_0, $ACTS_0)
    st32step $RESULTS_0, $BASE, $DATA_PTR+=, 1

.PREFIX(_half_16_bit_remainder):
    brz $REM_16BIT, .PREFIX(_half_end)

    // Load the first and second half in the word to store along
    // with the remaining
    ldb16 $ACTS_0, $DATA_PTR, $BASE, 0
    {
      ldb16 $ASCRATCH, $DATA_PTR, $BASE, 1
      CALC_HALF($RESULTS_0, $ACTS_0)
    }
    roll16 $RESULTS_0, $RESULTS_0, $ASCRATCH
    st32 $RESULTS_0, $DATA_PTR, $BASE, 0

.PREFIX(_half_end):
    exitz $mzero

HALF_SYMBOL_NAME:
    setzi $WORKER_ENTRY, PREFIX(_half_worker)
    runall $WORKER_ENTRY, $SUPER_BASE, 0
    sync TEXCH_SYNCZONE_LOCAL
    br $lr

.size HALF_SYMBOL_NAME, .-HALF_SYMBOL_NAME

.globl FLOAT_SYMBOL_NAME
.type FLOAT_SYMBOL_NAME, @function

.align 8
PREFIX(_float_worker):
    ldz16 $DATA_PTR, $mvertex_base, $mzero, DATA_PTR_VOFFSET
    ldz16 $SIZE, $mvertex_base, $mzero, SIZE_VOFFSET

    // Scaled pointer gives offset in 32-bit units from
    // TMEM_REGION0_BASE_ADDR
    shl $DATA_PTR, $DATA_PTR, 2
    setzi $BASE, TMEM_REGION0_BASE_ADDR

    // Get worker ID
    get $WORKER_ID, $WSR
    and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    // Extract packed size + remainder
    // $SIZE = No. of elements / (CTXT_WORKERS * elements per 64-bits)
    // $REM = No. of remaining elements
    and $REM, $SIZE, 0x0F
    shr $SIZE, $SIZE, 4

    // Check if address is 64-bit aligned
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .PREFIX(_float_64_bit_aligned)

.PREFIX(_float_32_bit_aligned):
    // Select a single worker to do this
    cmpeq $MSCRATCH, $WORKER_ID, 0
    brz $MSCRATCH, .PREFIX(_float_skip_32_bit_lead)

    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    CALC_FLOAT($RESULTS_0, $ACTS_0)
    st32 $RESULTS_0, $DATA_PTR, $BASE, 0

.PREFIX(_float_skip_32_bit_lead):
    ld32step $ASCRATCH, $BASE, $DATA_PTR+=, 1

    // Decrement remaining element count
    add $REM, $REM, -1
    brpos $REM, .PREFIX(_float_64_bit_aligned)
    add $REM, $REM, (CTXT_WORKERS * 2)
    add $SIZE, $SIZE, -1

.PREFIX(_float_64_bit_aligned):
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
    brz $SIZE, .PREFIX(_float_64_bit_loop_exit)
    add $SIZE, $SIZE, -1
    ld64 $ACTS_PAIR, $DATA_PTR, $BASE, 0
    {
      rpt $SIZE, (.PREFIX(_float_rpt_end) - .PREFIX(_float_rpt_start)) / 8 - 1
      CALC_FLOAT($RESULTS_0, $ACTS_0)
    }
    .PREFIX(_float_rpt_start):
    {
      ld64 $ACTS_PAIR, $DATA_PTR, $BASE, CTXT_WORKERS
      CALC_FLOAT($RESULTS_1, $ACTS_1)
    }
    {
      st64step $RESULTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
      CALC_FLOAT($RESULTS_0, $ACTS_0)
    }
    .PREFIX(_float_rpt_end):
    CALC_FLOAT($RESULTS_1, $ACTS_1)
    st64step $RESULTS_PAIR, $BASE, $DATA_PTR+=, CTXT_WORKERS
.PREFIX(_float_64_bit_loop_exit):

    // Handle remaining elements with the worker with the correct $DATA_PTR.
    // $REM = Num of remaining 64-bit loads possible = index to first worker
    // for which 64-bit load isn't possible
    cmpeq $MSCRATCH, $WORKER_ID, $REM
    and $MSCRATCH, $MSCRATCH, $REM_32BIT
    brz $MSCRATCH, .PREFIX(_float_end)

.PREFIX(_float_32_bit_remainder):
    ld32 $ACTS_0, $DATA_PTR, $BASE, 0
    CALC_FLOAT($RESULTS_0, $ACTS_0)
    st32step $RESULTS_0, $BASE, $DATA_PTR+=, 1

.PREFIX(_float_end):
    exitz $mzero

FLOAT_SYMBOL_NAME:
    setzi $WORKER_ENTRY, PREFIX(_float_worker)
    runall $WORKER_ENTRY, $SUPER_BASE, 0
    sync TEXCH_SYNCZONE_LOCAL
    br $lr

.size FLOAT_SYMBOL_NAME, .-FLOAT_SYMBOL_NAME
