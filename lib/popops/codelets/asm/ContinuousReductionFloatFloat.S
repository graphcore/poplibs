// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
// Continuous Reduce (ALL_REGIONS_CONTINUOUS) - Overview:
// `partials` is a single edge
// `out` is a single edge
// The vertex treats partials as a 2D array, size {`numOutputs`, `numPartials`}
// Eg, for numOutputs = 3, numPartials = 12
// 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
// 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12,
// 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12, 13,
//
// We will sum across each row, producing 3 outputs:
// SUM(0..11), SUM(1..12), SUM(2..13)
//
// Constraints:
// Although the edges require alignment we have a flexible `numPartials` allowing
// for any row length, and therefore also any alignment on the next row.  therefore
// the alignment constraint just allows for a small vertex state.
// There is no output constraint.
// Sizes of numOutputs, numPartials as 16-bit.
//
// Operation:
// Outer loop operates per row, dealing with non-vectorwidth elements before/
// after a main loop. The inner loop deals with 64 bits (2 floats, 4 halves) in 1 cycle in
// all cases that have assembler.

#include "poplibs_support/TileConstants.hpp"

.macro MANGLED_NAMES name isUpdate vertex align optionalNop

.section .text.__runCodelet_popops__\vertex\()___\name\()_float_float_\isUpdate\(), "ax"
.global __runCodelet_popops__\vertex\()___\name\()_float_float_\isUpdate\()
.type __runCodelet_popops__\vertex\()___\name\()_float_float_\isUpdate\(), @function
.align \align
  \optionalNop
__runCodelet_popops__\vertex\()___\name\()_float_float_\isUpdate\():

.endm

.macro SECTION_SIZE name isUpdate vertex
.size __runCodelet_popops__\vertex\()___\name\()_float_float_\isUpdate\(),  \
      . - __runCodelet_popops__\vertex\()___\name\()_float_float_\isUpdate\()
.endm

#define P_PTR_OFF 0
#define O_PTR_OFF 2
#define NUM_OUT_OFF 4
#define NUM_PART_OFF 6
#define SCALE_OFF 8

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)



#define PARTIAL_PTR         m0
#define OUTPUT_PTR          m1
#define NUM_PARTIALS        m2
#define NUM_OUTPUTS         m3
#define PARTIALS_COUNTER    m4
#define MSCRATCH            m5
#define SCALE_PTR           m6
#define BASE                m7


#define VALUES0             a0
#define VALUES1             a1
#define VALUES2             a2
#define VALUES3             a3
#define SCALE               a4
#define ASCRATCH            a5

.macro NON_SCALED_VERSION reduce_op_name isUpdate vertex scaled_vertex update_flag

MANGLED_NAMES \reduce_op_name \isUpdate \vertex 4


    // SCALE = 1.0 using e^0 = 1.0
     {bri            __non_scaled_entry_point_\scaled_vertex\()_\reduce_op_name\()_\update_flag\()
     f32exp            $SCALE, $azero}

    SECTION_SIZE \reduce_op_name \isUpdate \vertex
.endm

.macro CONTINUOUS_FLOAT_REDUCE reduce_op reduce_op_name isUpdate vertex update_flag
// Macro begin
//-----------------------------------------------------------------------------------
MANGLED_NAMES \reduce_op_name \isUpdate \vertex 8 nop

    // load vertex state
    // ------------------------------------------------------------------------
    ld32           $SCALE_PTR, $mvertex_base, $mzero, SCALE_OFF/4
    ld32           $SCALE, $SCALE_PTR, $mzero, 0

__non_scaled_entry_point_\vertex\()_\reduce_op_name\()_\update_flag\():
    ldz16           $PARTIAL_PTR, $mvertex_base, $mzero, P_PTR_OFF/2
    ldz16           $OUTPUT_PTR, $mvertex_base, $mzero, O_PTR_OFF/2
    ldz16           $NUM_PARTIALS, $mvertex_base, $mzero, NUM_PART_OFF/2
    ldz16           $NUM_OUTPUTS, $mvertex_base, $mzero, NUM_OUT_OFF/2

    // unpack scaled ptrs
    //--------------------------------------------------------------------------
    setzi    $BASE, TMEM_REGION0_BASE_ADDR
    shl      $PARTIAL_PTR, $PARTIAL_PTR, 2
    shl      $OUTPUT_PTR, $OUTPUT_PTR, 2

    // output loop begin
    //--------------------------------------------------------------------------
_output_loop_start\@:
        {mov            $PARTIALS_COUNTER, $NUM_PARTIALS
         mov            $VALUES2:3, $azeros}
        // zero accumulators and deal with misaligned start
        // ---------------------------------------------------------------------
        {and            $MSCRATCH, $PARTIAL_PTR, 0x7
         setzi          $ASCRATCH, ZAACC_BITMASK}
        {brz            $MSCRATCH, _if_misaligned_end_\@
         uput           $FP_CLR, $ASCRATCH}
_if_misaligned_start_\@:
            {ld32step      $VALUES0, $BASE, $PARTIAL_PTR+=, 1
             mov           $VALUES1, $azero}
            {brnzdec       $PARTIALS_COUNTER, _if_misaligned_end_\@
             \reduce_op     $VALUES0:3}
            bri            __retrieve_outputs\@
_if_misaligned_end_\@:

        // accumulate continuous region into accumulators
        //--------------------------------------------------------------------------
        shr             $MSCRATCH, $PARTIALS_COUNTER, 1

        {rpt            $MSCRATCH, ((2f - 1f)/ 8) - 1
         mov            $VALUES0:1, $azeros }
1:
            {ld64step      $VALUES0:1, $BASE, $PARTIAL_PTR+=, 1
             \reduce_op    $VALUES0:3}
2:
        // The reduce op in the bundle reads the data a cycle too early so do last here
        {and            $MSCRATCH, $PARTIALS_COUNTER, 0x1
         \reduce_op      $VALUES0:3}

        // if the size is odd must accumulate final value
        //----------------------------------------------------------------------------
        brz              $MSCRATCH, __retrieve_outputs\@
_if_odd_size_start\@:
            {ld32step         $VALUES0, $BASE, $PARTIAL_PTR+=, 1
             mov              $VALUES1, $azero}
            \reduce_op        $VALUES0:3
_if_odd_size_end\@:
        // retrieve outputs
        //---------------------------------------------------------------------------
__retrieve_outputs\@:
        f32v2gina           $VALUES0:1, $azeros, 0
        f32add              $VALUES0, $VALUES0, $VALUES1
        // update version
.if \update_flag
            {ld32             $VALUES1, $BASE, $OUTPUT_PTR, 0
            f32mul              $VALUES0, $VALUES0, $SCALE}
            f32add           $VALUES0, $VALUES0, $VALUES1
.else
        f32mul              $VALUES0, $VALUES0, $SCALE
.endif
        st32step            $VALUES0, $BASE, $OUTPUT_PTR+=, 1

    // output loop end
    //-------------------------------------------------------------------------
    brnzdec       $NUM_OUTPUTS, _output_loop_start\@
    exitz         $mzero

// Macro end
//------------------------------------------------------------------------------------
    SECTION_SIZE \reduce_op_name \isUpdate \vertex
.endm


CONTINUOUS_FLOAT_REDUCE f32v4acc popops__ReduceAdd false ScaledContinuousReduce 0
CONTINUOUS_FLOAT_REDUCE f32v4sqacc popops__ReduceSquareAdd false ScaledContinuousReduce 0
CONTINUOUS_FLOAT_REDUCE f32v4acc popops__ReduceAdd true ScaledContinuousReduce 1
CONTINUOUS_FLOAT_REDUCE f32v4sqacc popops__ReduceSquareAdd true ScaledContinuousReduce 1

NON_SCALED_VERSION  popops__ReduceAdd false ContinuousReduce ScaledContinuousReduce 0
NON_SCALED_VERSION  popops__ReduceSquareAdd false ContinuousReduce ScaledContinuousReduce 0
NON_SCALED_VERSION  popops__ReduceAdd true ContinuousReduce ScaledContinuousReduce 1
NON_SCALED_VERSION  popops__ReduceSquareAdd true ContinuousReduce ScaledContinuousReduce 1

#endif
