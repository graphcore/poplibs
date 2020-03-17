// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// See ContinuousReductionAcc.S for more information

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "MathConstants.S"
#include "poplar/StackSizeDefs.hpp"

// Offsets into vertex base
#ifdef VECTOR_AVAIL_SCALED_PTR32
    #define PARTIALS_PTR_OFFSET 0
    #define OUTPUT_PTR_OFFSET   2
    #define NUM_OUTPUTS_OFFSET  4
    #define NUM_PARTIALS_OFFSET 6
    #define SCALE_OFFSET        8
#else
    #define PARTIALS_PTR_OFFSET 0
    #define OUTPUT_PTR_OFFSET   4
    #define NUM_OUTPUTS_OFFSET  8
    #define NUM_PARTIALS_OFFSET 10
    #define SCALE_OFFSET        12
#endif

#define PARTIAL_PTR         m0
#define OUTPUT_PTR          m1
#define NUM_PARTIALS        m2
#define NUM_OUTPUTS         m3
#define PARTIALS_COUNTER    m4
#define MSCRATCH            m5
#define MSCRATCH2           m6
#ifdef VECTOR_AVAIL_SCALED_PTR32
    #define BASE            m8
#else
    #define BASE            mzero
#endif

#define VALUES0             a0
#define VALUES1             a1
#define VALUES2             a2
#define VALUES3             a3
#define SCALE               a4
#define ASCRATCH            a5

// Define parameters:
/*
    REDUCTION_NAME - {ReduceMax, ReduceMin}
    TYPE           - {half, float}
    UPDATE         - {true, false}
*/
#define M_VERTEX_NAME __runCodelet_popops__ContinuousReduce___popops__\REDUCTION_NAME\()_\TYPE\()_\TYPE\()_\UPDATE

#define M_SCALED_VERTEX_HALF_NAME __runCodelet_popops__ScaledContinuousReduce___popops__\REDUCTION_NAME\()_half_half_\UPDATE
#define M_SCALED_VERTEX_FLOAT_NAME __runCodelet_popops__ScaledContinuousReduce___popops__\REDUCTION_NAME\()_float_float_\UPDATE

// Define parameters:
/*
    FTYPE          - {f16, f32}
    OP             - {min, max}
*/
#define OP_V1 \FTYPE\()\OP
#define OP_V2 \FTYPE\()v2\OP
#define OP_V4 \FTYPE\()v4\OP

.macro DEFINE_ENTRY_POINT VERTEX_NAME ALIGNMENT OPTIONAL_NOP
    DEF_STACK_USAGE  0  \VERTEX_NAME
    .section .text.\VERTEX_NAME, "ax"
    .global \VERTEX_NAME
    .type \VERTEX_NAME, @function
    .align \ALIGNMENT
\VERTEX_NAME\()_START:
\OPTIONAL_NOP
\VERTEX_NAME:
.endm

.macro FN_SIZE VERTEX_NAME
    .size \VERTEX_NAME, . - \VERTEX_NAME\()_START
.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

.macro SCALED_VERSION_HALF REDUCTION_NAME FTYPE OP UPDATE
#ifdef VECTOR_AVAIL_SCALED_PTR32
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_HALF_NAME 8
#else
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_HALF_NAME 8 nop
#endif

    ld32            $MSCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4
    ld32            $SCALE, $MSCRATCH, $mzero, 0

__non_scaled_entry_\REDUCTION_NAME\()_half_\UPDATE\():
    ldz16           $NUM_PARTIALS, $mvertex_base, $mzero, NUM_PARTIALS_OFFSET/2
    ldz16           $NUM_OUTPUTS , $mvertex_base, $mzero, NUM_OUTPUTS_OFFSET/2

    // Unpack scaled ptrs
    // --------------------------------------------------------------------------
#ifdef VECTOR_AVAIL_SCALED_PTR32
    ldz16           $PARTIAL_PTR , $mvertex_base, $mzero, PARTIALS_PTR_OFFSET/2
    ldz16           $OUTPUT_PTR  , $mvertex_base, $mzero, OUTPUT_PTR_OFFSET/2

    setzi           $BASE, TMEM_REGION0_BASE_ADDR
    shl             $PARTIAL_PTR, $PARTIAL_PTR, 2
    shl             $OUTPUT_PTR, $OUTPUT_PTR, 2
#else
    ld32            $PARTIAL_PTR, $mvertex_base, $mzero, PARTIALS_PTR_OFFSET/4
    ld32            $OUTPUT_PTR, $mvertex_base, $mzero, OUTPUT_PTR_OFFSET/4
#endif

.ifc "\REDUCTION_NAME", "ReduceMax"
    .equ INITIAL_VALUE, MIN_HALF_BROADCAST
.endif
.ifc "\REDUCTION_NAME", "ReduceMin"
    .equ INITIAL_VALUE, MAX_HALF_BROADCAST
.endif

_output_loop_start\@:

    // Initialize values
    // ---------------------------------------------------------------------
    mov                $PARTIALS_COUNTER, $NUM_PARTIALS

    // Deal with leading misalignment
    // check if start is misaligned or size 1
    // ---------------------------------------------------------------------
    {and                $MSCRATCH, $PARTIAL_PTR, 0x2
     setzi              $VALUES0, INITIAL_VALUE & ((1 << 20) - 1)} // Set upper half of const
    {cmpeq              $MSCRATCH2, $PARTIALS_COUNTER, 1
     or                 $VALUES0, $VALUES0, INITIAL_VALUE & ~((1 << 20) - 1)} // Set lower half of const
    {or                 $MSCRATCH , $MSCRATCH, $MSCRATCH2
     mov                $VALUES1, $VALUES0} // Init VALUES1
    {brz                $MSCRATCH, _if_misaligned_one_half_end_\@
     or64               $VALUES2:3, $azeros, $VALUES0:1} // Init VALUES2:3

_if_misaligned_one_half_start_\@:
    ldb16step      $VALUES0, $BASE, $PARTIAL_PTR+=, 1
    add            $PARTIALS_COUNTER, $PARTIALS_COUNTER, -1
    // We can't have two halves which of which the first was misaligned
    // because the whole edge is aligned to at least 8 bytes and
    // if every partial is two bytes in length, at least 4 byte alignment
    // will be maintained each output loop.
    brz            $PARTIALS_COUNTER, __retrieve_outputs\@ // None left!
_if_misaligned_one_half_end_\@:

    // Deal with another 2 if the misalignment is still off
    // ---------------------------------------------------------------------
    and             $MSCRATCH, $PARTIAL_PTR, 0x4
    brz             $MSCRATCH, _if_misaligned_two_half_end_\@

_if_misaligned_two_half_start_\@:
    ld32step      $VALUES1, $BASE, $PARTIAL_PTR+=, 1
    add           $PARTIALS_COUNTER, $PARTIALS_COUNTER, -2
    brz           $PARTIALS_COUNTER,  __retrieve_outputs\@ // None left!
_if_misaligned_two_half_end_\@:

    // PRECOND : VALUES0:1 holds reduced value or defaults, VALUES2:3 defaults
    // Aligned data
    // ---------------------------------------------------------------------
__aligned_start_\@:
    shr                $MSCRATCH, $PARTIALS_COUNTER, 2 // We are processing 4 at a time
    rpt                $MSCRATCH, ((2f - 1f)/8) - 1
1:
    {ld64step        $VALUES2:3, $BASE, $PARTIAL_PTR+=, 1
     OP_V4           $VALUES0:1, $VALUES0:1, $VALUES2:3}
2:
__aligned_end_\@:

    // PRECOND : VALUES0:1 holds reduced value or defaults, VALUES2:3 defaults
    // Deal with trailing misaligned
    // ---------------------------------------------------------------------
    // The reduce op in the bundle reads the data a cycle too early so do last here
    {and            $MSCRATCH, $PARTIALS_COUNTER, 0x3
     OP_V4          $VALUES0:1, $VALUES0:1, $VALUES2:3}

    // Not misaligned
    brz            $MSCRATCH, __retrieve_outputs\@
    rpt            $MSCRATCH, ((2f - 1f)/8) - 1
1:
    {ldb16step     $VALUES2, $BASE, $PARTIAL_PTR+=, 1
     OP_V2         $VALUES1, $VALUES1, $VALUES2}
2:
    OP_V2          $VALUES1, $VALUES1, $VALUES2

    // PRECOND VALUES0:1 holds the reduction values
__retrieve_outputs\@:
    OP_V2        $VALUES0, $VALUES0, $VALUES1
    // VALUES0 holds two of the reduced values

    swap16     $VALUES1, $VALUES0
    OP_V2      $VALUES0, $VALUES0, $VALUES1
    // VALUES0 holds duplicates of the single reduced value

    {and                $MSCRATCH2, $OUTPUT_PTR, 0x2 // Figure out which half we want within the pair
     f16tof32           $VALUES0, $VALUES0} // Cast up our output
.ifc "\UPDATE","true"
    {ldb16              $VALUES1, $BASE, $OUTPUT_PTR, 0 // Load existing output
     f32mul             $VALUES0, $VALUES0, $SCALE} // Apply scale

    {andc               $MSCRATCH , $OUTPUT_PTR, 0x3 // Get the pair of halves we want
     f16tof32           $VALUES1, $VALUES1} // Cast up existing output to add to
    f32add              $VALUES0, $VALUES0, $VALUES1
.else
    {andc               $MSCRATCH , $OUTPUT_PTR, 0x3 // Get the pair of halves we want
    f32mul              $VALUES0, $VALUES0, $SCALE} // Apply scale
.endif
    // VALUES0 now holds f32 of scaled reduction to write back

    {ld32               $ASCRATCH , $BASE, $MSCRATCH, 0 // Load the two halves into ASCRATCH
     f32tof16           $VALUES0, $VALUES0} // Convert back to half

    // If it's 0, then we are looking at aligned case (first half)
    // otherwise we are looking at the misaligned case (second half)
    brnz               $MSCRATCH2, __mislaigned_case\@

__aligned_case\@:
    {add            $OUTPUT_PTR, $OUTPUT_PTR, 0x2 // Read a half
     roll16         $ASCRATCH, $ASCRATCH, $VALUES0}
    {bri            __retrieve_outputs_end\@
     swap16         $ASCRATCH, $ASCRATCH}
__mislaigned_case\@:
    {add            $OUTPUT_PTR, $OUTPUT_PTR, 0x2 // Read a half
     sort4x16lo     $ASCRATCH, $ASCRATCH, $VALUES0}

__retrieve_outputs_end\@:
    st32           $ASCRATCH, $BASE, $MSCRATCH, 0

    brnzdec         $NUM_OUTPUTS, _output_loop_start\@
    _output_loop_end\@:

    exitz         $mzero
    FN_SIZE M_SCALED_VERTEX_HALF_NAME
.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

.macro SCALED_VERSION_FLOAT REDUCTION_NAME FTYPE OP UPDATE
#ifdef VECTOR_AVAIL_SCALED_PTR32
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_FLOAT_NAME 8
#else
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_FLOAT_NAME 8 nop
#endif

    ld32            $MSCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4
    ld32            $SCALE, $MSCRATCH, $mzero, 0

__non_scaled_entry_\REDUCTION_NAME\()_float_\UPDATE\():
    ldz16           $NUM_PARTIALS, $mvertex_base, $mzero, NUM_PARTIALS_OFFSET/2
    ldz16           $NUM_OUTPUTS , $mvertex_base, $mzero, NUM_OUTPUTS_OFFSET/2

    // Unpack scaled ptrs
    // --------------------------------------------------------------------------
#ifdef VECTOR_AVAIL_SCALED_PTR32
    ldz16           $PARTIAL_PTR , $mvertex_base, $mzero, PARTIALS_PTR_OFFSET/2
    ldz16           $OUTPUT_PTR  , $mvertex_base, $mzero, OUTPUT_PTR_OFFSET/2

    setzi           $BASE, TMEM_REGION0_BASE_ADDR
    shl             $PARTIAL_PTR, $PARTIAL_PTR, 2
    shl             $OUTPUT_PTR, $OUTPUT_PTR, 2
#else
    ld32            $PARTIAL_PTR, $mvertex_base, $mzero, PARTIALS_PTR_OFFSET/4
    ld32            $OUTPUT_PTR, $mvertex_base, $mzero, OUTPUT_PTR_OFFSET/4
#endif

.ifc "\REDUCTION_NAME", "ReduceMax"
    .equ INITIAL_VALUE, MIN_FLOAT
.endif
.ifc "\REDUCTION_NAME", "ReduceMin"
    .equ INITIAL_VALUE, MAX_FLOAT
.endif

_output_loop_start\@:

    // Initialize values
    // ---------------------------------------------------------------------
    {mov                $PARTIALS_COUNTER, $NUM_PARTIALS
     or                 $VALUES0, $azero, INITIAL_VALUE}

    // Deal with leading misalignment
    {and                $MSCRATCH, $PARTIAL_PTR, 0x7
     mov                $VALUES1, $VALUES0}
    {brz                $MSCRATCH, _if_misaligned_one_float_end_\@ // no misalignment
     or64               $VALUES2:3, $azeros, $VALUES0:1}

_if_misaligned_one_float_start_\@:
    ld32step            $VALUES0, $BASE, $PARTIAL_PTR+=, 1
    brnzdec             $PARTIALS_COUNTER, _if_misaligned_one_float_end_\@
    bri                 __retrieve_outputs\@ // None left!
_if_misaligned_one_float_end_\@:

    // PRECOND : VALUES0:1 holds reduced value or defaults, VALUES2:3 defaults
    // Aligned data
    // ---------------------------------------------------------------------
__aligned_start_\@:
    shr                 $MSCRATCH, $PARTIALS_COUNTER, 1 // We are processing 2 at a time
    rpt                 $MSCRATCH, ((2f - 1f)/8) - 1
1:
    {ld64step        $VALUES2:3, $BASE, $PARTIAL_PTR+=, 1
     OP_V2           $VALUES0:1, $VALUES0:1, $VALUES2:3}
2:
__aligned_end_\@:

    // PRECOND : VALUES0:1 holds reduced value or defaults, VALUES2:3 defaults
    // Deal with trailing misaligned
    // ---------------------------------------------------------------------
    // The reduce op in the bundle reads the data a cycle too early so do last here
    {and             $MSCRATCH, $PARTIALS_COUNTER, 0x1
     OP_V2           $VALUES0:1, $VALUES0:1, $VALUES2:3}

    brz             $MSCRATCH, __retrieve_outputs\@ // No trailing
    ld32step        $VALUES2, $BASE, $PARTIAL_PTR+=, 1
    OP_V1           $VALUES1, $VALUES1, $VALUES2

    // PRECOND VALUES0:1 holds the reduction values
__retrieve_outputs\@:
    OP_V1          $VALUES0, $VALUES0, $VALUES1
    // VALUES0 now holds the reduced value
.ifc "\UPDATE","true"
    {ld32             $VALUES1, $BASE, $OUTPUT_PTR, 0
     f32mul           $VALUES0, $VALUES0, $SCALE}
    f32add            $VALUES0, $VALUES0, $VALUES1
.else
    f32mul            $VALUES0, $VALUES0, $SCALE
.endif
    // VALUES0 holds the scaled reduced value (updated)
    st32step        $VALUES0, $BASE, $OUTPUT_PTR+=, 1

    brnzdec         $NUM_OUTPUTS, _output_loop_start\@
_output_loop_end\@:

    exitz         $mzero
    FN_SIZE M_SCALED_VERTEX_FLOAT_NAME
.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

// Jumps to the Scaled equivalent after setting scale = 1
.macro NON_SCALED_VERSION REDUCTION_NAME FTYPE OP TYPE UPDATE
    DEFINE_ENTRY_POINT M_VERTEX_NAME 4

    {bri __non_scaled_entry_\REDUCTION_NAME\()_\TYPE\()_\UPDATE\()
     f32exp      $SCALE, $azero} // using e^0 = 1.0

    FN_SIZE M_VERTEX_NAME
.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

.macro MAKE_VERTEX REDUCTION_NAME FTYPE OP TYPE UPDATE
.ifc "\TYPE", "half"
    SCALED_VERSION_HALF \REDUCTION_NAME \FTYPE \OP \UPDATE
.endif
.ifc "\TYPE", "float"
    SCALED_VERSION_FLOAT \REDUCTION_NAME \FTYPE \OP \UPDATE
.endif
    NON_SCALED_VERSION \REDUCTION_NAME \FTYPE \OP \TYPE \UPDATE
.endm

MAKE_VERTEX ReduceMax f16 max half  false
MAKE_VERTEX ReduceMax f16 max half  true
MAKE_VERTEX ReduceMax f32 max float false
MAKE_VERTEX ReduceMax f32 max float true

MAKE_VERTEX ReduceMin f16 min half  false
MAKE_VERTEX ReduceMin f16 min half  true
MAKE_VERTEX ReduceMin f32 min float false
MAKE_VERTEX ReduceMin f32 min float true

#endif
