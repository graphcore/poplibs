// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Continuous Reduce (ALL_REGIONS_CONTINUOUS) - Overview:
// `partials` is a single edge
// `out` is a single edge
// The vertex treats partials (a 1D array) as a 2D array, size {`numOutputs`, `numPartials`}
// Eg, for numOutputs = 3, (numOutputsM1 = 2), numPartials = 12
// A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12
// B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12
// C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12
//
// We will sum across each row, producing 3 outputs:
// SUM(As), SUM(Bs), SUM(Cs)
//
// If "isUpdate", then the result is added to the existing output instead of overwritten
// E.g.
// if (isUpdate)
//  out[0] += SUM(As)
// else
//  out[0] = SUM(As)
//
// Constraints:
// - Although the edges require alignment we have a flexible `numPartials` allowing
//   for any row length, and therefore also any alignment on the next row.
//   Therefore the alignment constraint just allows for a small vertex state.
// - There is no output constraint.
// - Sizes of numOutputsM1, numPartials as 16-bit.
//
// Operation:
// Outer loop operates per row, dealing with non-vectorwidth elements
// before/after a main loop. The inner loop deals with 64 bits (2 floats, 4 halves)
// in 1 cycle in all cases that have assembler.

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
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

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

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
    REDUCTION_NAME - {ReduceAdd, ReduceSquareAdd}
    PARTIALS_TYPE  - {half, float}
    OUT_TYPE       - {half, float}
    UPDATE         - {true, false}
*/
#define M_VERTEX_NAME              __runCodelet_popops__ContinuousReduce___popops__\REDUCTION_NAME\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_\UPDATE

#define M_SCALED_VERTEX_HALF_NAME __runCodelet_popops__ScaledContinuousReduce___popops__\REDUCTION_NAME\()_half_\OUT_TYPE\()_\UPDATE
#define M_SCALED_VERTEX_FLOAT_NAME __runCodelet_popops__ScaledContinuousReduce___popops__\REDUCTION_NAME\()_float_\OUT_TYPE\()_\UPDATE

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

.macro STORE_HALF_OUTPUTS PARTIALS_TYPE UPDATE
    {andc               $MSCRATCH , $OUTPUT_PTR, 0x3
     f32v2gina          $VALUES0:1, $azeros, 0}

.ifc "\PARTIALS_TYPE", "half"
    f32v2gina          $VALUES2:3, $azeros, 0
    f32v2add           $VALUES0:1, $VALUES0:1, $VALUES2:3
.endif
    {ld32               $ASCRATCH , $BASE, $MSCRATCH, 0
     f32add             $VALUES0  , $VALUES0, $VALUES1}

.ifc "\UPDATE", "true"
    {ldb16          $VALUES2, $BASE, $OUTPUT_PTR, 0
     f32mul         $VALUES0, $VALUES0, $SCALE}
    f16tof32        $VALUES2, $VALUES2
    {and            $MSCRATCH2, $OUTPUT_PTR, 0x2
     f32add         $VALUES0, $VALUES0, $VALUES2}
.else
    {and            $MSCRATCH2, $OUTPUT_PTR, 0x2
     f32mul         $VALUES0, $VALUES0, $SCALE}
.endif

    {brnz               $MSCRATCH2, __mislaigned_case\@
     f32tof16           $VALUES0, $VALUES0}
__aligned_case\@:
    {add            $OUTPUT_PTR, $OUTPUT_PTR, 2
     roll16         $ASCRATCH, $ASCRATCH, $VALUES0}
    {bri            __retrieve_outputs_end\@
     swap16         $ASCRATCH, $ASCRATCH}

__mislaigned_case\@:
    {add            $OUTPUT_PTR, $OUTPUT_PTR, 2
     sort4x16lo     $ASCRATCH, $ASCRATCH, $VALUES0}

__retrieve_outputs_end\@:
    st32        $ASCRATCH, $BASE, $MSCRATCH, 0

.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

.macro STORE_FLOAT_OUTPUTS PARTIALS_TYPE UPDATE
    f32v2gina           $VALUES0:1, $azeros, 0
.ifc "\PARTIALS_TYPE", "half"
    f32v2gina           $VALUES2:3, $azeros, 0
    f32v2add            $VALUES0:1, $VALUES0:1, $VALUES2:3
.endif
    f32add              $VALUES0, $VALUES0, $VALUES1

.ifc "\UPDATE", "true"
    {ld32             $VALUES1, $BASE, $OUTPUT_PTR, 0
     f32mul           $VALUES0, $VALUES0, $SCALE}
    f32add            $VALUES0, $VALUES0, $VALUES1
.else
    f32mul            $VALUES0, $VALUES0, $SCALE
.endif

    st32step          $VALUES0, $BASE, $OUTPUT_PTR+=, 1

.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

.macro SCALED_VERSION_HALF_PARTIALS REDUCTION_NAME OP OUT_TYPE UPDATE
#ifdef VECTOR_AVAIL_SCALED_PTR32
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_HALF_NAME 8 nop
#else
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_HALF_NAME 8
#endif

    ld32            $MSCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4
    ld32            $SCALE, $MSCRATCH, $mzero, 0

__non_scaled_entry_\REDUCTION_NAME\()_half_\OUT_TYPE\()_\UPDATE\():
    ldz16           $NUM_PARTIALS, $mvertex_base, $mzero, NUM_PARTIALS_OFFSET/2
    ldz16           $NUM_OUTPUTS , $mvertex_base, $mzero, NUM_OUTPUTS_OFFSET/2

    // Unpack scaled ptrs
    // --------------------------------------------------------------------------
#ifdef VECTOR_AVAIL_SCALED_PTR32
    ldz16           $PARTIAL_PTR , $mvertex_base, $mzero, PARTIALS_PTR_OFFSET/2
    ldz16           $OUTPUT_PTR  , $mvertex_base, $mzero, OUTPUT_PTR_OFFSET/2

    setzi           $BASE, TMEM_REGION0_BASE_ADDR
    shl             $PARTIAL_PTR, $PARTIAL_PTR, 2
    shl            $OUTPUT_PTR, $OUTPUT_PTR, 2
#else
    ld32            $PARTIAL_PTR, $mvertex_base, $mzero, PARTIALS_PTR_OFFSET/4
    ld32            $OUTPUT_PTR, $mvertex_base, $mzero, OUTPUT_PTR_OFFSET/4
#endif

_output_loop_start\@:
    // Deal with leading misaligned
    // ---------------------------------------------------------------------
    {mov        $PARTIALS_COUNTER, $NUM_PARTIALS
     mov        $VALUES2:3, $azeros} // We never touch $VALUES2:3 intentionally!
    // Later we use the f16v8sqacc instruction, as there is no f16v4sqacc
    // (which we'd prefer to use as we only load $VALUES0:1).
    // We need to zero $VALUES2:3 to avoid processing undefined data.
    // We are also limited by loading 64-bit at once, while in non-interleaved memory.
    // If we could load more in a single cycle, we could fully utilise the vectorised methods.

    // zero accumulators and check if start is misaligned or size 1
    // ---------------------------------------------------------------------
    {and        $MSCRATCH , $PARTIAL_PTR, 0x2
     setzi      $ASCRATCH , ZAACC_BITMASK}
    {cmpeq      $MSCRATCH2, $PARTIALS_COUNTER, 1
     uput       $FP_CLR   , $ASCRATCH}
    or          $MSCRATCH , $MSCRATCH, $MSCRATCH2
    brz         $MSCRATCH , _if_misaligned_one_half_end_\@

    // Deal with single half if misaligned or there is only a single half
    // ---------------------------------------------------------------------
_if_misaligned_one_half_start_\@:
    {ldb16step     $VALUES0, $BASE, $PARTIAL_PTR+=, 1
     mov           $VALUES1, $azero}
    {add           $PARTIALS_COUNTER, $PARTIALS_COUNTER, -1
     sort4x16lo    $VALUES0, $VALUES0, $azero}
    {cmpeq         $MSCRATCH, $PARTIALS_COUNTER, 1 // Is there only one left to do
     \OP           $VALUES0:3}
    brnz           $MSCRATCH, _if_misaligned_one_half_start_\@
    brz            $PARTIALS_COUNTER, __retrieve_outputs\@
_if_misaligned_one_half_end_\@:

    // Deal with another 2 if the misalignment is still off
    // ---------------------------------------------------------------------
    and             $MSCRATCH, $PARTIAL_PTR, 0x4
    brz             $MSCRATCH, _if_misaligned_two_end_\@

_if_misaligned_two_half_start_\@:
    {ld32step      $VALUES0, $BASE, $PARTIAL_PTR+=, 1
     mov           $VALUES1, $azero}
    add            $PARTIALS_COUNTER, $PARTIALS_COUNTER, -2
    {brz           $PARTIALS_COUNTER,  __retrieve_outputs\@
     \OP           $VALUES0:3}
_if_misaligned_two_end_\@:

    // Alignment is correct, deal with everything which is aligned
    // ---------------------------------------------------------------------
    shr             $MSCRATCH, $PARTIALS_COUNTER, 2
    {rpt            $MSCRATCH, ((2f - 1f)/ 8) - 1
     mov            $VALUES0:1, $azeros }
1:
    {ld64step      $VALUES0:1, $BASE, $PARTIAL_PTR+=, 1
     \OP           $VALUES0:3}
2:

    // Deal with trailing misaligned
    // ---------------------------------------------------------------------
    // The reduce op in the bundle reads the data a cycle too early so do last here
    {and            $MSCRATCH, $PARTIALS_COUNTER, 0x3
     \OP            $VALUES0:3}

    {rpt $MSCRATCH, ((2f - 1f)/8) - 1
     fnop}
1:
    {ldb16step         $VALUES0, $BASE, $PARTIAL_PTR+=, 1
     mov               $VALUES1, $azero}
    {add               $PARTIALS_COUNTER, $PARTIALS_COUNTER, -1
     sort4x16lo        $VALUES0, $VALUES0, $azero}
    {nop
     \OP               $VALUES0:3}
2:

__retrieve_outputs\@:
.ifc "\OUT_TYPE", "half"
    STORE_HALF_OUTPUTS half \UPDATE
.endif
.ifc "\OUT_TYPE", "float"
    STORE_FLOAT_OUTPUTS half \UPDATE
.endif

    brnzdec           $NUM_OUTPUTS, _output_loop_start\@
_output_loop_end\@:

    exitz         $mzero
    FN_SIZE M_SCALED_VERTEX_HALF_NAME
.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

.macro SCALED_VERSION_FLOAT_PARTIALS REDUCTION_NAME OP OUT_TYPE UPDATE

#ifdef VECTOR_AVAIL_SCALED_PTR32
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_FLOAT_NAME 8 nop
#else
    DEFINE_ENTRY_POINT M_SCALED_VERTEX_FLOAT_NAME 8
#endif

    ld32            $MSCRATCH, $mvertex_base, $mzero, SCALE_OFFSET/4
    ld32            $SCALE, $MSCRATCH, $mzero, 0

__non_scaled_entry_\REDUCTION_NAME\()_float_\OUT_TYPE\()_\UPDATE\():
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

_output_loop_start\@:
    // Deal with leading misaligned
    // ---------------------------------------------------------------------
    {mov        $PARTIALS_COUNTER, $NUM_PARTIALS
     mov        $VALUES2:3, $azeros} // We never touch $VALUES2:3 intentionally!

    // zero accumulators and deal with misaligned start
    // ---------------------------------------------------------------------
    {and            $MSCRATCH, $PARTIAL_PTR, 0x7
     setzi          $ASCRATCH, ZAACC_BITMASK}
    {brz            $MSCRATCH, _if_misaligned_one_float_end_\@
     uput           $FP_CLR, $ASCRATCH}

_if_misaligned_one_float_start_\@:
    {ld32step      $VALUES0, $BASE, $PARTIAL_PTR+=, 1
     mov           $VALUES1, $azero}
    {brnzdec       $PARTIALS_COUNTER, _if_misaligned_one_float_end_\@
     \OP           $VALUES0:3}
    bri            __retrieve_outputs\@
_if_misaligned_one_float_end_\@:

    // Alignment is correct, deal with everything which is aligned
    // ---------------------------------------------------------------------
    shr             $MSCRATCH, $PARTIALS_COUNTER, 1
    {rpt            $MSCRATCH, ((2f - 1f)/ 8) - 1
     mov            $VALUES0:1, $azeros }
1:
    {ld64step      $VALUES0:1, $BASE, $PARTIAL_PTR+=, 1
     \OP           $VALUES0:3}
2:

    // Deal with trailing misaligned
    // ---------------------------------------------------------------------
    // The reduce op in the bundle reads the data a cycle too early so do last here
    {and            $MSCRATCH, $PARTIALS_COUNTER, 0x1
     \OP            $VALUES0:3}

    // if the size is odd must accumulate final value
    brz              $MSCRATCH, __retrieve_outputs\@
    {ld32step        $VALUES0, $BASE, $PARTIAL_PTR+=, 1
     mov             $VALUES1, $azero}
    \OP              $VALUES0:3

__retrieve_outputs\@:
.ifc "\OUT_TYPE", "half"
    STORE_HALF_OUTPUTS float \UPDATE
.endif
.ifc "\OUT_TYPE", "float"
    STORE_FLOAT_OUTPUTS float \UPDATE
.endif

    brnzdec           $NUM_OUTPUTS, _output_loop_start\@
_output_loop_end\@:

    exitz         $mzero
    FN_SIZE M_SCALED_VERTEX_FLOAT_NAME
.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

// Jumps to the Scaled equivalent after setting scale = 1
.macro NON_SCALED_VERSION REDUCTION_NAME OP PARTIALS_TYPE OUT_TYPE UPDATE
    DEFINE_ENTRY_POINT M_VERTEX_NAME 4

    {bri __non_scaled_entry_\REDUCTION_NAME\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_\UPDATE\()
     f32exp      $SCALE, $azero} // using e^0 = 1.0

    FN_SIZE M_VERTEX_NAME
.endm

// ===========================================================================
// ===========================================================================
// ===========================================================================

.macro MAKE_VERTEX REDUCTION_NAME OP PARTIALS_TYPE OUT_TYPE UPDATE
.ifc "\PARTIALS_TYPE", "half"
    SCALED_VERSION_HALF_PARTIALS \REDUCTION_NAME \OP \OUT_TYPE \UPDATE
.endif
.ifc "\PARTIALS_TYPE", "float"
    SCALED_VERSION_FLOAT_PARTIALS \REDUCTION_NAME \OP \OUT_TYPE \UPDATE
.endif
    NON_SCALED_VERSION \REDUCTION_NAME \OP \PARTIALS_TYPE \OUT_TYPE \UPDATE
.endm

MAKE_VERTEX ReduceAdd f16v8acc half  half  false
MAKE_VERTEX ReduceAdd f16v8acc half  half  true
MAKE_VERTEX ReduceAdd f16v8acc half  float false
MAKE_VERTEX ReduceAdd f16v8acc half  float true

MAKE_VERTEX ReduceAdd f32v4acc float half  false
MAKE_VERTEX ReduceAdd f32v4acc float half  true
MAKE_VERTEX ReduceAdd f32v4acc float float false
MAKE_VERTEX ReduceAdd f32v4acc float float true

MAKE_VERTEX ReduceSquareAdd f16v8sqacc half  half  false
MAKE_VERTEX ReduceSquareAdd f16v8sqacc half  half  true
MAKE_VERTEX ReduceSquareAdd f16v8sqacc half  float false
MAKE_VERTEX ReduceSquareAdd f16v8sqacc half  float true

MAKE_VERTEX ReduceSquareAdd f32v4sqacc float half  false
MAKE_VERTEX ReduceSquareAdd f32v4sqacc float half  true
MAKE_VERTEX ReduceSquareAdd f32v4sqacc float float false
MAKE_VERTEX ReduceSquareAdd f32v4sqacc float float true

#endif
