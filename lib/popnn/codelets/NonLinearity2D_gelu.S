#ifdef __IPU__

// Assembly implementation of popnn::NonLinearity2D vertex template instantiations.

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.

#include "poplibs_support/TileConstants.hpp"

// Symbols
#define HALF_SYMBOL \
  __runCodelet_popnn__NonLinearity2D___half_popnn__NonLinearityType__GELU
#define FLOAT_SYMBOL \
  __runCodelet_popnn__NonLinearity2D___float_popnn__NonLinearityType__GELU

// Constants
#define BASE_AND_N0_VOFFSET 0
#define DELTAN_PTR_VOFFSET 2

#define DELTAN_BASE_PTR_BITS 20
#define DELTAN_BASE_PTR_MASK ((1 << DELTAN_BASE_PTR_BITS) - 1)
#define DELTAN_OFFSET_BITS 18
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)

#define HALF_1_0 0x3C00
#define HALF_0_5 0x3800
#define HALF_ALPHA 0x3A62  // 0.7978845608f
#define HALF_BETA 0x29B9   // 0.044715
#define HALF_6_0 0x4600
#define HALF_MINUS_6_0 0xC600

#define FLOAT_12_0 0x41400000
#define FLOAT_MINUS_12_0 0xC1400000
    
#define FLOAT_1_0 0x3F800000
#define FLOAT_0_5 0x3F000000
#define FLOAT_ALPHA 0x3F4C422A  // 0.7978845608f
#define FLOAT_BETA 0x3D372713   // 0.044715

// all scratch offsets given in words
#define SCRATCH_OFFSET_CONST_ALPHA_BETA      0
    
#define SCRATCH_OFFSET_CONST_ALPHA           0    
#define SCRATCH_OFFSET_CONST_BETA            1
#define SCRATCH_OFFSET_CONST_1_0             2
#define SCRATCH_OFFSET_CONST_0_5             3
    
// Worker register aliases
#define MASK m0
#define MEMORY_BASE m1
#define BASE_PTR m2
#define N0 m3
#define DELTAN_PTR m4
#define DATA_PTR m5
#define N1 m6
#define N1_64BIT m7

// Equivalent to $lr    
#define MSCRATCH m10

#define ACTS_0 a0
#define ACTS_1 a1
#define ACTS_PAIR a0:1
#define RESULTS_0 a4
#define RESULTS_1 a5
#define RESULTS_PAIR a4:5

#define HALF_CLAMP a2
#define CONST_HI_1_0_LO_0_5 a3
    
#define CONST_SCRATCH_0 a0
#define CONST_SCRATCH_1 a1

#define ASCRATCH_0 a6
#define ASCRATCH_1 a7
#define ASCRATCH_PAIR a6:7
    
#define FLOAT_CLAMP_0 a2
#define FLOAT_CLAMP_1 a3
#define FLOAT_CLAMP_PAIR a2:3
    
// Subroutine: Calculate GELU non-linearity for two halves
//   ACTS_0 should contain the 16x2 operands
.section .text.NonLinearityGELU_half

.globl NonLinearityGELU_half
.type NonLinearityGELU_half, @function

.align 4
NonLinearityGELU_half:
    f16v2clamp $ASCRATCH_0, $ACTS_0, $HALF_CLAMP

    {
      ld32     $ASCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_BETA
      f16v2mul $RESULTS_0, $ASCRATCH_0, $ASCRATCH_0
    }
    
    f16v2mul $RESULTS_0, $ASCRATCH_1:BL, $RESULTS_0
    f16v2add $RESULTS_0, $CONST_HI_1_0_LO_0_5:BU, $RESULTS_0
    f16v2mul $RESULTS_0, $ASCRATCH_1:BU, $RESULTS_0
    f16v2mul $RESULTS_0, $RESULTS_0, $ASCRATCH_0
    f16v2tanh $RESULTS_0, $RESULTS_0
    f16v2add $RESULTS_0, $CONST_HI_1_0_LO_0_5:BU, $RESULTS_0
    f16v2mul $RESULTS_0, $RESULTS_0, $ACTS_0

    {
      br $MSCRATCH
      f16v2mul $RESULTS_0, $CONST_HI_1_0_LO_0_5:BL, $RESULTS_0
    }


// Subroutine: Calculate GELU non-linearity for a float
//   ACTS_0 should contain the float operand
    
.section .text.NonLinearityGELU_float

.globl NonLinearityGELU_float
.type NonLinearityGELU_float, @function

.align 4
NonLinearityGELU_float:
    f32clamp $ASCRATCH_0, $ACTS_0, $FLOAT_CLAMP_PAIR

    {
      ld32   $ASCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_BETA
      f32mul $RESULTS_0, $ASCRATCH_0, $ASCRATCH_0
    }

    {
      ld32   $ASCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
      f32mul $RESULTS_0, $ASCRATCH_1, $RESULTS_0
    }

    {
      ld32   $ASCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA
      f32add $RESULTS_0, $ASCRATCH_1, $RESULTS_0
    }

    f32mul $RESULTS_0, $ASCRATCH_1, $RESULTS_0
    f32mul $RESULTS_0, $RESULTS_0, $ASCRATCH_0

    {
      ld32   $ASCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_1_0
      f32tanh $RESULTS_0, $RESULTS_0
    }

    f32add $RESULTS_0, $ASCRATCH_1, $RESULTS_0

    {
      ld32   $ASCRATCH_1, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_0_5
      f32mul $RESULTS_0, $RESULTS_0, $ACTS_0
    }

    {
      br $MSCRATCH
      f32mul $RESULTS_0, $ASCRATCH_1, $RESULTS_0
    }    

    
.section .text.HALF_SYMBOL
.globl HALF_SYMBOL
.type HALF_SYMBOL, @function

.align 8
    nop
HALF_SYMBOL:
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET

    // Unpack base pointer and n0
    setzi $MASK, DELTAN_BASE_PTR_MASK
    and $BASE_PTR, $MSCRATCH, $MASK
    shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, 2

    setzi $MASK, DELTAN_OFFSET_MASK

    // Load clamp values
    ldconst $HALF_CLAMP, (HALF_MINUS_6_0 | (HALF_6_0 << 16))
    ldconst $CONST_HI_1_0_LO_0_5, (HALF_0_5) | (HALF_1_0 << 16)
    ldconst $ASCRATCH_0, (HALF_BETA) | (HALF_ALPHA << 16)
    st32    $ASCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_BETA
    
    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.Lhalf_n0_loop:
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $MASK
    shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
    // Actually offset DATA_PTR so that below alignment checks
    // take BASE_PTR alignment into account
    add $DATA_PTR, $BASE_PTR, $DATA_PTR

    and $MSCRATCH, $DATA_PTR, 0x3
    brz $MSCRATCH, .Lhalf_32_bit_aligned

    // Handle the first 16-bit element. We'll always have
    // at least 1 element here.
    andc $DATA_PTR, $DATA_PTR, 0x3
    ldb16 $ACTS_0, $DATA_PTR, $mzero, 1

    call $MSCRATCH, NonLinearityGELU_half
    
    ldb16 $RESULTS_1, $DATA_PTR, $mzero, 0
    
    {
      add $N1, $N1, -1
      roll16 $RESULTS_0, $RESULTS_1, $RESULTS_0
    }
    
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1
    brz $N1, .Lhalf_n0_loop_cond

.Lhalf_32_bit_aligned:
    and $MSCRATCH, $DATA_PTR, 0x7
    brz $MSCRATCH, .Lhalf_64_bit_aligned

    // Special case for a single 16-bit element at 32-bit
    // aligned address.
    cmpult $MSCRATCH, $N1, 2
    brnz $MSCRATCH, .Lhalf_16_bit_remainder

    ld32 $ACTS_0, $DATA_PTR, $mzero, 0

    call $MSCRATCH, NonLinearityGELU_half
    
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1
    add $N1, $N1, -2

.Lhalf_64_bit_aligned:
    shr $N1_64BIT, $N1, 2

    ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 0
    
    rpt $N1_64BIT, (2f - 1f) / 8 - 1

1:
    // Perform the following calculations on the 2 lower halves:
    //   - Clamp input x to +/-6.0 in order to accomodater intermediate results
    //   - 0.5 * x * (1 + tanh(x * alpha * (1 + (beta * x * x))))
    
    {
      nop
      f16v4clamp $ASCRATCH_PAIR, $ACTS_PAIR, $HALF_CLAMP
    }
    
    {
      ld32     $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_ALPHA_BETA
      f16v4mul $RESULTS_PAIR, $ASCRATCH_PAIR, $ASCRATCH_PAIR
    }

    {
      nop
      f16v4mul $RESULTS_PAIR, $CONST_SCRATCH_0:BL, $RESULTS_PAIR
    }
    
    {
      nop
      f16v4add $RESULTS_PAIR, $CONST_HI_1_0_LO_0_5:BU, $RESULTS_PAIR
    }
    
    {
      nop
      f16v4mul $RESULTS_PAIR, $CONST_SCRATCH_0:BU, $RESULTS_PAIR
    }
    
    {
      nop
      f16v4mul $RESULTS_PAIR, $RESULTS_PAIR, $ASCRATCH_PAIR
    }
    
    {
      nop
      f16v2tanh $RESULTS_0, $RESULTS_0
    }
    
    {
      nop
      f16v2tanh $RESULTS_1, $RESULTS_1
    }
    
    {
      ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 0
      f16v4add $RESULTS_PAIR, $CONST_HI_1_0_LO_0_5:BU, $RESULTS_PAIR
    }
    
    {
      nop
      f16v4mul $RESULTS_PAIR, $RESULTS_PAIR, $ACTS_PAIR
    }
    
    {
      ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 1
      f16v4mul $RESULTS_PAIR, $CONST_HI_1_0_LO_0_5:BL, $RESULTS_PAIR
    }
    
    {
      st64step $RESULTS_PAIR, $mzero, $DATA_PTR+=, 1
      fnop
    }
2:

.Lhalf_32_bit_remainder:
    and $MSCRATCH, $N1, 0x2
    brz $MSCRATCH, .Lhalf_16_bit_remainder

    // ACTS_0 should already be loaded at this point

    call $MSCRATCH, NonLinearityGELU_half

    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.Lhalf_16_bit_remainder:
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .Lhalf_n0_loop_cond

    ldb16 $ACTS_0, $DATA_PTR, $mzero, 0

    call $MSCRATCH, NonLinearityGELU_half
    
    ldb16 $RESULTS_1, $DATA_PTR, $mzero, 1
    roll16 $RESULTS_0, $RESULTS_0, $RESULTS_1
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.Lhalf_n0_loop_cond:
    brnzdec $N0, .Lhalf_n0_loop
    exitz $mzero

.size HALF_SYMBOL, .-HALF_SYMBOL

.section .text.FLOAT_SYMBOL
.globl FLOAT_SYMBOL
.type FLOAT_SYMBOL, @function

.align 8
FLOAT_SYMBOL:
    ld32 $MSCRATCH, $mvertex_base, $mzero, BASE_AND_N0_VOFFSET
    ldz16 $DELTAN_PTR, $mvertex_base, $mzero, DELTAN_PTR_VOFFSET

    // Unpack base pointer and n0
    setzi $MASK, DELTAN_BASE_PTR_MASK
    and $BASE_PTR, $MSCRATCH, $MASK
    shr $N0, $MSCRATCH, DELTAN_BASE_PTR_BITS

    // DeltaN table pointer is a ScaledPtr32, gives offset in
    // 32-bit units from TMEM_REGION0_BASE_ADDR
    setzi $MEMORY_BASE, TMEM_REGION0_BASE_ADDR
    shl $DELTAN_PTR, $DELTAN_PTR, 2

    setzi $MASK, DELTAN_OFFSET_MASK

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
    
    // Top-level loop through each DeltaN
    add $N0, $N0, -1
.Lfloat_n0_loop:
    ld32step $MSCRATCH, $MEMORY_BASE, $DELTAN_PTR+=, 1
    and $DATA_PTR, $MSCRATCH, $MASK
    shr $N1, $MSCRATCH, DELTAN_OFFSET_BITS
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

    call $MSCRATCH, NonLinearityGELU_float
    
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1
    add $N1, $N1, -1

.Lfloat_64_bit_aligned:
    shr $N1_64BIT, $N1, 1

    ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 0
    rpt $N1_64BIT, (2f - 1f) / 8 - 1
    
1:
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
      nop
      f32tanh $RESULTS_1, $RESULTS_1
    }

    {
      ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 0
      f32v2add $RESULTS_PAIR, $CONST_SCRATCH_0:B, $RESULTS_PAIR
    }

    {
      ld32   $CONST_SCRATCH_0, $mworker_base, $mzero, SCRATCH_OFFSET_CONST_0_5
      f32v2mul $RESULTS_PAIR, $RESULTS_PAIR, $ACTS_PAIR
    }

    {
      ld64 $ACTS_PAIR, $DATA_PTR, $mzero, 1
      f32v2mul $RESULTS_PAIR, $CONST_SCRATCH_0:B, $RESULTS_PAIR
    }
    
    {
      st64step $RESULTS_PAIR, $mzero, $DATA_PTR+=, 1
      fnop
    }
2:

.Lfloat_32_bit_remainder:
    and $MSCRATCH, $N1, 0x1
    brz $MSCRATCH, .Lfloat_n0_loop_cond

    // ACTS_0 should already be loaded at this point
      
    call $MSCRATCH, NonLinearityGELU_float
    
    st32step $RESULTS_0, $mzero, $DATA_PTR+=, 1

.Lfloat_n0_loop_cond:
    brnzdec $N0, .Lfloat_n0_loop
    exitz $mzero

.size FLOAT_SYMBOL, .-FLOAT_SYMBOL

#endif // __IPU__
