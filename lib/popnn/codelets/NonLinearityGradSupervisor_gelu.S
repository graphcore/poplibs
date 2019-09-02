#ifdef __IPU__

// Assembly implementation of popnn::NonLinearityGrad2D vertex template instantiations.

// Restrictions
//
//  * All input/output regions 8-byte aligned.
//  * Load up to 64-bits past the end of outGrad and out regions without exceptions.

#include "poplibs_support/TileConstants.hpp"

// Symbol names
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearityGradSupervisor___half_popnn__NonLinearityType__GELU
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearityGradSupervisor___float_popnn__NonLinearityType__GELU

// Constants
#define OUTGRAD_PTR_VOFFSET 0
#define OUT_PTR_VOFFSET 1
#define INGRAD_PTR_VOFFSET 2
#define N_VOFFSET 3

#define RECIPROCAL_3_SHL17 ((((1 << 17) - 1) / 3) + 1)
#define LOG2_24_OVER_3 3
#define LOG2_12_OVER_3 2

#define HALF_3_0 0x4200
#define HALF_1_0 0x3C00
#define HALF_0_5 0x3800
#define HALF_ALPHA 0x3A62  // 0.7978845608f
#define HALF_BETA 0x29B9   // 0.044715
#define HALF_6_0 0x4600
#define HALF_MINUS_6_0 0xC600

#define FLOAT_12_0 0x41400000
#define FLOAT_MINUS_12_0 0xC1400000
    
#define FLOAT_0_5 0x3F000000
#define FLOAT_3_0 0x40400000
#define FLOAT_ALPHA 0x3F4C422A  // 0.7978845608f
#define FLOAT_BETA 0x3D372713   // 0.044715

// Scratch Offsets in 1xFloats
#define SCRATCH_OFFSET_CONST_ALPHA_BETA      0
#define SCRATCH_OFFSET_CONST_HI_1_0_LO_0_5   1
    
#define SCRATCH_OFFSET_CONST_ALPHA           2    
#define SCRATCH_OFFSET_CONST_BETA            3
#define SCRATCH_OFFSET_CONST_0_5             4
#define SCRATCH_OFFSET_CONST_1_0             5
#define SCRATCH_OFFSET_CONST_3_0             6
#define SCRATCH_OFFSET_HALF_CLAMP_LIMITS     7

// Scratch Offsets in 2xFloats
#define SCRATCH_OFFSET_ONE_PLUS_PHI          (8 / 2)
#define SCRATCH_OFFSET_ONE_MINUS_PHI_TIMES_X (10 / 2)
#define SCRATCH_OFFSET_FLOAT_CLAMP           (12 / 2)
#define SCRATCH_OFFSET_BETA_X_SQR            (14 / 2)

// Supervisor register aliases
#define SUPER_BASE m0
#define WORKER_ENTRY m1

// Worker register aliases
#define WORKER_ID m0
#define MEMORY_BASE m1
#define OUTGRAD_PTR m2
#define OUT_PTR m3
#define INGRAD_PTR m4
#define SIZE m5
#define REM m6
#define REM_64BIT m7
#define MSCRATCH m10

// NOTE: Register definitions as well as Loop implementation are included here
#include "NonLinearityGradLoop_gelu.S"
    
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
.supervisor

HALF_SYMBOL:
  setzi $WORKER_ENTRY, .Lhalf_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

  // For rpt alignment below.
.Lhalf_worker:
.worker
  ldz16 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET
  ldz16 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET
  ldz16 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET
  ldz16 $MSCRATCH, $mvertex_base, $mzero, N_VOFFSET
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 2
  shl $OUT_PTR, $OUT_PTR, 2
  shl $INGRAD_PTR, $INGRAD_PTR, 2
  setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR

  // $SIZE = No. of 64-bit elements each worker should process
  // $REM = No. of remaining elements between workers
  HALF_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

  // Get worker ID
  get $WORKER_ID, $WSR
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Add remaining 64-bit loads/stores to relevant workers
  shr $REM_64BIT, $REM, 2
  cmpult $MSCRATCH, $WORKER_ID, $REM_64BIT
  add $SIZE, $SIZE, $MSCRATCH

  // Use dummy loads to offset each worker's pointers into the data to
  // interleave them
  ld64step $azeros, $MEMORY_BASE, $OUTGRAD_PTR+=, $WORKER_ID
  ld64step $azeros, $MEMORY_BASE, $OUT_PTR+=, $WORKER_ID
  ld64step $azeros, $MEMORY_BASE, $INGRAD_PTR+=, $WORKER_ID

  // Load clamp values
  ldconst $HALF_CLAMP_LIMITS, (HALF_MINUS_6_0 | (HALF_6_0 << 16))
  st32    $HALF_CLAMP_LIMITS, $mworker_base, $mzero, SCRATCH_OFFSET_HALF_CLAMP_LIMITS
  ldconst $ASCRATCH_1, (HALF_0_5) | (HALF_1_0 << 16)
  ldconst $ASCRATCH_0, (HALF_BETA) | (HALF_ALPHA << 16)

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, (SCRATCH_OFFSET_CONST_ALPHA_BETA/2)
    or      $ASCRATCH_0, $azero, (HALF_3_0 << 16)
  }
    
  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0

  rpt $SIZE, (2f - 1f) / 8 - 1
  
1:
  NONLINEARITY_GELU_HALF $MEMORY_BASE $MEMORY_BASE $MEMORY_BASE CTXT_WORKERS
2:

    
.Lhalf_32_bit_remainder:
  // Handle remaining element with a single worker. We pick the first
  // worker which didn't handle a remainder element.
  // $REM_64BIT = No. of remaining 64-bit loads possible = index to first
  // worker for which 64-bit load isn't possible.
  cmpeq $MSCRATCH, $WORKER_ID, $REM_64BIT
  brz $MSCRATCH, .Lhalf_end

  and $MSCRATCH, $REM, 0x2
  brz $MSCRATCH, .Lhalf_16_bit_remainder

  // Handle remaining 32-bit value
  ld32step $OUT_0, $MEMORY_BASE, $OUT_PTR+=, 1
  ld32step $OUTGRAD_PAR, $MEMORY_BASE, $OUTGRAD_PTR+=, 1
  call $MSCRATCH, calc_GELU_half
  st32step $INGRAD_0, $MEMORY_BASE, $INGRAD_PTR+=, 1

.Lhalf_16_bit_remainder:
  and $MSCRATCH, $REM, 0x1
  brz $MSCRATCH, .Lhalf_end

  ldb16 $OUT_0, $MEMORY_BASE, $OUT_PTR, 0
    
  // Handle remaining 16-bit value
  // Broadcasting lower 16-bits of remaining input words to
  // ensure no exceptions when calculating last gradient.
  ldb16 $OUTGRAD_PAR, $MEMORY_BASE, $OUTGRAD_PTR, 0
  call $MSCRATCH, calc_GELU_half
  ldb16 $INGRAD_1, $MEMORY_BASE, $INGRAD_PTR, 1
  sort4x16lo $INGRAD_0, $INGRAD_0, $INGRAD_1
  st32 $INGRAD_0, $MEMORY_BASE, $INGRAD_PTR, 0

.Lhalf_end:
  exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL

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
.supervisor
FLOAT_SYMBOL:
  setzi $WORKER_ENTRY, .Lfloat_worker
  runall $WORKER_ENTRY, $SUPER_BASE, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

  // For rpt alignment below.
  nop
.Lfloat_worker:
.worker
  ldz16 $OUTGRAD_PTR, $mvertex_base, $mzero, OUTGRAD_PTR_VOFFSET
  ldz16 $OUT_PTR, $mvertex_base, $mzero, OUT_PTR_VOFFSET
  ldz16 $INGRAD_PTR, $mvertex_base, $mzero, INGRAD_PTR_VOFFSET
  ldz16 $MSCRATCH, $mvertex_base, $mzero, N_VOFFSET
  shl $OUTGRAD_PTR, $OUTGRAD_PTR, 2
  shl $OUT_PTR, $OUT_PTR, 2
  shl $INGRAD_PTR, $INGRAD_PTR, 2
  setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR

  // $SIZE = No. of 64-bit elements each worker should process
  // $REM = No. of remaining elements between workers
  FLOAT_SPLIT_BETWEEN_WORKERS $MSCRATCH $SIZE $REM

  // Get worker ID
  get $WORKER_ID, $WSR
  and $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // Add remaining 64-bit loads/stores to relevant workers
  shr $REM_64BIT, $REM, 1
  cmpult $MSCRATCH, $WORKER_ID, $REM_64BIT

  // Use dummy loads to offset each worker's pointers into the data to
  // interleave them
  ld64step $azeros, $MEMORY_BASE, $OUTGRAD_PTR+=, $WORKER_ID
  ld64step $azeros, $MEMORY_BASE, $OUT_PTR+=, $WORKER_ID
  ld64step $azeros, $MEMORY_BASE, $INGRAD_PTR+=, $WORKER_ID

  // Load clamp values
  ldconst $ASCRATCH_0, FLOAT_ALPHA
  ldconst $ASCRATCH_1, FLOAT_BETA

  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, (SCRATCH_OFFSET_CONST_ALPHA/2)
    or      $ASCRATCH_0, $azero, FLOAT_0_5
  }

  {
    add $SIZE, $SIZE, $MSCRATCH
    f32exp  $ASCRATCH_1, $azero
  }
    
  {
    st64    $ASCRATCH_PAIR, $mworker_base, $mzero, (SCRATCH_OFFSET_CONST_0_5/2)
    or      $ASCRATCH_0, $azero, FLOAT_3_0
  }

  st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_3_0
  ldconst $ASCRATCH_0, FLOAT_MINUS_12_0
  ldconst $ASCRATCH_1, FLOAT_12_0
  st64    $ASCRATCH_PAIR, $mworker_base, $mzero, SCRATCH_OFFSET_FLOAT_CLAMP

  rpt $SIZE, (2f - 1f) / 8 - 1
  
1:
  NONLINEARITY_GELU_FLOAT $MEMORY_BASE $MEMORY_BASE $MEMORY_BASE CTXT_WORKERS
2:
    
.Lfloat_32_bit_remainder:
  // Handle remaining element with a single worker. We pick the first
  // worker which didn't handle a remainder element.
  // $REM_64BIT = No. of remaining 64-bit loads possible = index to first
  // worker for which 64-bit load isn't possible.
  cmpeq $MSCRATCH, $WORKER_ID, $REM_64BIT
  brz $MSCRATCH, .Lfloat_end

  and $MSCRATCH, $REM, 0x1
  brz $MSCRATCH, .Lfloat_end

  // Handle remaining 32-bit value
  ld32 $OUT_0, $MEMORY_BASE, $OUT_PTR, 0
  ld32 $OUTGRAD_PAR, $MEMORY_BASE, $OUTGRAD_PTR, 0
  call $MSCRATCH, calc_GELU_float
  st32 $INGRAD_0, $MEMORY_BASE, $INGRAD_PTR, 0

.Lfloat_end:
  exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
