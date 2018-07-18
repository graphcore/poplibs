// Required defines:
//
// HALF_SYMBOL_NAME
// FLOAT_SYMBOL_NAME
// OP_UNIQUE_PREFIX
// CALC_HALF(dst, src)
// CALC_FLOAT(dst, src)

#define BASE_AND_N0_VOFFSET 0
#define DELTAN_PTR_VOFFSET 2

// Worker register aliases
#define MEMORY_BASE m0
#define BASE_PTR m1
#define N0 m2
#define DELTAN_PTR m3
#define DELTAN_OFFSET_MASK m4
#define DATA_PTR m5
#define N1 m6
#define N1_64BIT m7
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
    nop
HALF_SYMBOL_NAME:
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET

    // Unpack base pointer and n0
    shl $BASE_PTR, $MSCRATCH, 12
    shr $BASE_PTR, $BASE_PTR, 12
    shr $N0, $MSCRATCH, 20

    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, 2

    setzi $DELTAN_OFFSET_MASK, 0x3FFFF

    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.PREFIX(_half_n0_loop):
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $DELTAN_OFFSET_MASK
    shr $N1, $MSCRATCH, 18
    // Actually offset DATA_PTR so that below alignment checks
    // take BASE_PTR alignment into account
    add $DATA_PTR, $BASE_PTR, $DATA_PTR

    and $MSCRATCH, $DATA_PTR, 0x3
    brz $MSCRATCH, .PREFIX(_half_32_bit_aligned)

    // Handle the first 16-bit element. We'll always have
    // at least 1 element here.
    andc $DATA_PTR, $DATA_PTR, 0x3
    ldb16 $ACTS_0, $DATA_PTR, $mzero, 1
    {
      ldb16 $ASCRATCH, $DATA_PTR, $mzero, 0
      CALC_HALF($RESULTS_0, $ACTS_0)
    }
    roll16 $RESULTS_0, $ASCRATCH, $RESULTS_0
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1
    add $N1, $N1, -1
    brz $N1, .PREFIX(_half_n0_loop_cond)

.PREFIX(_half_32_bit_aligned):
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .PREFIX(_half_64_bit_aligned)

    // Special case for a single 16-bit element at 32-bit
    // aligned address.
    cmpult $MSCRATCH, $N1, 2
    brnz $MSCRATCH, .PREFIX(_half_16_bit_remainder)

    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    CALC_HALF($RESULTS_0, $ACTS_0)
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1
    add $N1, $N1, -2

.PREFIX(_half_64_bit_aligned):
    shr $N1_64BIT, $N1, 2

    brz $N1_64BIT, .PREFIX(_half_32_bit_remainder)
    add $N1_64BIT, $N1_64BIT, -1
    ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 0
    {
      rpt $N1_64BIT, (.PREFIX(_half_rpt_end) - .PREFIX(_half_rpt_start)) / 8 - 1
      CALC_HALF($RESULTS_0, $ACTS_0)
    }
.PREFIX(_half_rpt_start):
    {
      ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 1
      CALC_HALF($RESULTS_1, $ACTS_1)
    }
    {
      st64step $RESULTS_PAIR, $mzero, $DATA_PTR+=, 1
      CALC_HALF($RESULTS_0, $ACTS_0)
    }
.PREFIX(_half_rpt_end):
    CALC_HALF($RESULTS_1, $ACTS_1)
    st64step $RESULTS_PAIR, $mzero, $DATA_PTR+=, 1

.PREFIX(_half_32_bit_remainder):
    and $MSCRATCH, $N1, 0x2
    brz $MSCRATCH, .PREFIX(_half_16_bit_remainder)

    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    CALC_HALF($RESULTS_0, $ACTS_0)
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.PREFIX(_half_16_bit_remainder):
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .PREFIX(_half_n0_loop_cond)

    ldb16 $ACTS_0, $DATA_PTR, $mzero, 0
    {
      ldb16 $ASCRATCH, $DATA_PTR, $mzero, 1
      CALC_HALF($RESULTS_0, $ACTS_0)
    }
    roll16 $RESULTS_0, $RESULTS_0, $ASCRATCH
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.PREFIX(_half_n0_loop_cond):
    brnzdec $N0, .PREFIX(_half_n0_loop)
    exitz $mzero

.size HALF_SYMBOL_NAME, .-HALF_SYMBOL_NAME

.globl FLOAT_SYMBOL_NAME
.type FLOAT_SYMBOL_NAME, @function

.align 8
    nop
FLOAT_SYMBOL_NAME:
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET

    // Unpack base pointer and n0
    shl $BASE_PTR, $MSCRATCH, 12
    shr $BASE_PTR, $BASE_PTR, 12
    shr $N0, $MSCRATCH, 20

    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, 2

    setzi $DELTAN_OFFSET_MASK, 0x3FFFF

    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.PREFIX(_float_n0_loop):
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $DELTAN_OFFSET_MASK
    shr $N1, $MSCRATCH, 18
    // Actually offset DATA_PTR so that below alignment checks
    // take BASE_PTR alignment into account
    add $DATA_PTR, $BASE_PTR, $DATA_PTR

    // DATA_PTR and N1 give us the regions to actually loop
.PREFIX(_float_32_bit_aligned):
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .PREFIX(_float_64_bit_aligned)

    // Handle the first 32-bit element. We'll always have
    // at least 1 element here.
    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    CALC_FLOAT($RESULTS_0, $ACTS_0)
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1
    add $N1, $N1, -1

.PREFIX(_float_64_bit_aligned):
    shr $N1_64BIT, $N1, 1

    brz $N1_64BIT, .PREFIX(_float_32_bit_remainder)
    add $N1_64BIT, $N1_64BIT, -1
    ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 0
    {
      rpt $N1_64BIT, (.PREFIX(_float_rpt_end) - .PREFIX(_float_rpt_start)) / 8 - 1
      CALC_FLOAT($RESULTS_0, $ACTS_0)
    }
.PREFIX(_float_rpt_start):
    {
      ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 1
      CALC_FLOAT($RESULTS_1, $ACTS_1)
    }
    {
      st64step $RESULTS_PAIR, $mzero, $DATA_PTR+=, 1
      CALC_FLOAT($RESULTS_0, $ACTS_0)
    }
.PREFIX(_float_rpt_end):
    CALC_FLOAT($RESULTS_1, $ACTS_1)
    st64step $RESULTS_PAIR, $mzero, $DATA_PTR+=, 1

.PREFIX(_float_32_bit_remainder):
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .PREFIX(_float_n0_loop_cond)
    
    ld32 $ACTS_0, $DATA_PTR, $mzero, 0
    CALC_FLOAT($RESULTS_0, $ACTS_0)
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.PREFIX(_float_n0_loop_cond):
    brnzdec $N0, .PREFIX(_float_n0_loop)
    exitz $mzero

.size FLOAT_SYMBOL_NAME, .-FLOAT_SYMBOL_NAME
