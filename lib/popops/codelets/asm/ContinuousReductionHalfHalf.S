// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
// Continuous Reduce (ALL_REGIONS_CONTINUOUS) - Overview:
// See ContinuousReductionFloatFloat.S

#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"

.macro MANGLED_NAMES name isUpdate vertex align optionalNop

.section .text.__runCodelet_popops__\vertex\()___\name\()_half_half_\isUpdate\(), "ax"
.global __runCodelet_popops__\vertex\()___\name\()_half_half_\isUpdate\()
.type __runCodelet_popops__\vertex\()___\name\()_half_half_\isUpdate\(), @function
.align \align
  \optionalNop
__runCodelet_popops__\vertex\()___\name\()_half_half_\isUpdate\():

.endm

.macro SECTION_SIZE name isUpdate vertex
.size __runCodelet_popops__\vertex\()___\name\()_half_half_\isUpdate\(),  \
      . - __runCodelet_popops__\vertex\()___\name\()_half_half_\isUpdate\()
.endm

#ifdef VECTOR_AVAIL_SCALED_PTR32
#define P_PTR_OFF 0
#define O_PTR_OFF 2
#define NUM_OUT_OFF 4
#define NUM_PART_OFF 6
#define SCALE_OFF 8
#else
#define P_PTR_OFF 0
#define O_PTR_OFF 4
#define NUM_OUT_OFF 8
#define NUM_PART_OFF 10
#define SCALE_OFF 12
#endif

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)



#define PARTIAL_PTR         m0
#define OUTPUT_PTR          m1
#define NUM_PARTIALS        m2
#define NUM_OUTPUTS         m3
#define PARTIALS_COUNTER    m4
#define MSCRATCH            m5
#define SCALE_PTR           m6
#define MSCRATCH2           m7
#ifdef VECTOR_AVAIL_SCALED_PTR32
#define BASE                m8
#else
#define BASE                mzero
#endif


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
    ld32            $SCALE_PTR, $mvertex_base, $mzero, SCALE_OFF/4
    ld32            $SCALE, $SCALE_PTR, $mzero, 0

__non_scaled_entry_point_\vertex\()_\reduce_op_name\()_\update_flag\():
    ldz16           $NUM_PARTIALS, $mvertex_base, $mzero, NUM_PART_OFF/2
    ldz16           $NUM_OUTPUTS, $mvertex_base, $mzero, NUM_OUT_OFF/2
#ifdef VECTOR_AVAIL_SCALED_PTR32
    ldz16           $PARTIAL_PTR, $mvertex_base, $mzero, P_PTR_OFF/2
    ldz16           $OUTPUT_PTR, $mvertex_base, $mzero, O_PTR_OFF/2

    // unpack scaled ptrs
    //--------------------------------------------------------------------------
    setzi    $BASE, TMEM_REGION0_BASE_ADDR
    shl      $PARTIAL_PTR, $PARTIAL_PTR, 2
    shl      $OUTPUT_PTR, $OUTPUT_PTR, 2
#else
    ld32            $PARTIAL_PTR, $mvertex_base, $mzero, P_PTR_OFF/4
    {ld32           $OUTPUT_PTR, $mvertex_base, $mzero, O_PTR_OFF/4
     fnop}
#endif

    // output loop begin
    //--------------------------------------------------------------------------
_output_loop_start\@:
        {mov            $PARTIALS_COUNTER, $NUM_PARTIALS
         mov            $VALUES2:3, $azeros}
        // zero accumulators and check if start is misaligned or size 1
        // ---------------------------------------------------------------------
        {and            $MSCRATCH, $PARTIAL_PTR, 0x2
         setzi          $ASCRATCH, ZAACC_BITMASK}
        {cmpeq           $MSCRATCH2, $PARTIALS_COUNTER, 1
         uput           $FP_CLR, $ASCRATCH}
        or              $MSCRATCH, $MSCRATCH, $MSCRATCH2
        brz            $MSCRATCH, _if_misaligned_half_end_\@

        // Deal with single half if misaligned or there is only a single half
        // ---------------------------------------------------------------------
_if_misaligned_half_start_\@:
            {ldb16step     $VALUES0, $BASE, $PARTIAL_PTR+=, 1
             mov           $VALUES1, $azero}
            {add           $PARTIALS_COUNTER, $PARTIALS_COUNTER, -1
             sort4x16lo     $VALUES0, $VALUES0, $azero}
            {cmpeq         $MSCRATCH, $PARTIALS_COUNTER, 1
             \reduce_op     $VALUES0:3}
            brnz          $MSCRATCH, _if_misaligned_half_start_\@
            brz           $PARTIALS_COUNTER, __retrieve_outputs\@
_if_misaligned_half_end_\@:

        // Deal with another 2 if the misalignment is still off
        // ---------------------------------------------------------------------
        and             $MSCRATCH, $PARTIAL_PTR, 0x4
        brz             $MSCRATCH, _if_misaligned_end_\@

_if_misaligned_start_\@:
            {ld32step      $VALUES0, $BASE, $PARTIAL_PTR+=, 1
             mov           $VALUES1, $azero}
            add            $PARTIALS_COUNTER, $PARTIALS_COUNTER, -2
            {brz       $PARTIALS_COUNTER,  __retrieve_outputs\@
             \reduce_op     $VALUES0:3}
_if_misaligned_end_\@:

        // accumlate continuous region into accumulators
        //--------------------------------------------------------------------------
        shr             $MSCRATCH, $PARTIALS_COUNTER, 2

        {rpt            $MSCRATCH, ((2f - 1f)/ 8) - 1
         mov            $VALUES0:1, $azeros }
1:
            {ld64step      $VALUES0:1, $BASE, $PARTIAL_PTR+=, 1
             \reduce_op    $VALUES0:3}
2:
        // The reduce op in the bundle reads the data a cycle too early so do last here
        {and            $MSCRATCH, $PARTIALS_COUNTER, 0x3
         \reduce_op      $VALUES0:3}

        // if the size is not a multiple of 4 have to do last bits
        //----------------------------------------------------------------------------
        {rpt $MSCRATCH, ((2f - 1f)/8) - 1
         fnop}
1:
            {ldb16step     $VALUES0, $BASE, $PARTIAL_PTR+=, 1
             mov           $VALUES1, $azero}
            {add           $PARTIALS_COUNTER, $PARTIALS_COUNTER, -1
             sort4x16lo     $VALUES0, $VALUES0, $azero}
            {nop
             \reduce_op    $VALUES0:3}
2:
        // retrieve outputs (requires the read modify write)
        //---------------------------------------------------------------------------
__retrieve_outputs\@:
        {andc               $MSCRATCH, $OUTPUT_PTR, 0x3
         f32v2gina           $VALUES0:1, $azeros, 0}
        {ld32               $ASCRATCH, $BASE, $MSCRATCH, 0
         f32v2gina           $VALUES2:3, $azeros, 0}
        {and                $MSCRATCH2, $OUTPUT_PTR, 0x2
         f32v2add            $VALUES0:1, $VALUES0:1, $VALUES2:3}
        f32add              $VALUES0, $VALUES0, $VALUES1
         // update version
.if \update_flag
            {ldb16            $VALUES2, $BASE, $OUTPUT_PTR, 0
            f32mul              $VALUES0, $VALUES0, $SCALE}
            f16tof32         $VALUES2, $VALUES2
            f32add           $VALUES0, $VALUES0, $VALUES2
.else
            f32mul              $VALUES0, $VALUES0, $SCALE
.endif
        {brnz                $MSCRATCH2, __mislaigned_case\@
         f32tof16            $VALUES0, $VALUES0}
__aligned_case\@:
            {add            $OUTPUT_PTR, $OUTPUT_PTR, 2
             roll16          $ASCRATCH, $ASCRATCH, $VALUES0}
            {bri             __output_loop_end_\@
            swap16          $ASCRATCH, $ASCRATCH}

__mislaigned_case\@:
            {add            $OUTPUT_PTR, $OUTPUT_PTR, 2
             sort4x16lo     $ASCRATCH, $ASCRATCH, $VALUES0}

    // output loop end
    //-------------------------------------------------------------------------
__output_loop_end_\@:
    st32            $ASCRATCH, $BASE, $MSCRATCH, 0
    brnzdec       $NUM_OUTPUTS, _output_loop_start\@
    exitz         $mzero

// Macro end
//------------------------------------------------------------------------------------
    SECTION_SIZE \reduce_op_name \isUpdate \vertex
.endm


CONTINUOUS_FLOAT_REDUCE f16v8acc popops__ReduceAdd false ScaledContinuousReduce 0
CONTINUOUS_FLOAT_REDUCE f16v8sqacc popops__ReduceSquareAdd false ScaledContinuousReduce 0
CONTINUOUS_FLOAT_REDUCE f16v8acc popops__ReduceAdd true ScaledContinuousReduce 1
CONTINUOUS_FLOAT_REDUCE f16v8sqacc popops__ReduceSquareAdd true ScaledContinuousReduce 1

NON_SCALED_VERSION  popops__ReduceAdd false ContinuousReduce ScaledContinuousReduce 0
NON_SCALED_VERSION  popops__ReduceSquareAdd false ContinuousReduce ScaledContinuousReduce 0
NON_SCALED_VERSION  popops__ReduceAdd true ContinuousReduce ScaledContinuousReduce 1
NON_SCALED_VERSION  popops__ReduceSquareAdd true ContinuousReduce ScaledContinuousReduce 1

#endif
