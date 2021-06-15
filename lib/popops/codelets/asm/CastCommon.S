// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// Common code for Cast vertices.
// Here we define the entry points for cast vertices:
//
//    Cast1DSingleWorker<INTYPE, OUTTYPE>        (1D, single worker)
//
//    Cast1D<INTYPE, OUTTYPE>                    (1D, MultiWorker)
//
//    Cast2D<INTYPE, OUTTYPE>                    (2D, single worker)
//
// This will contain all the 'setup' code before calling the specific
// processing functions ('cast_<INTYPE>_<OUTTYPE>_core'), which are
// defined in separate files.

#include "CastCommon.inc"

// Offsets for fields both in the 1D MultiVertex and single worker state.
// All offsets are in bytes
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
#define VOFF_IN         0
#define VOFF_OUT_START  2
#define VOFF_N_ELEMS    4
#else
#define VOFF_IN         0
#define VOFF_OUT_START  4
#define VOFF_N_ELEMS    8
#endif

// Offsets for fields in the 2D vertex state. Offsets are in bytes
#define VOFF_SRC_PTR_2D           0
#define VOFF_DST_PTR_BEGIN_2D     4
#define VOFF_DST_SIZE_2D          8

// Offsets in the scratch area for saving data. Offsets are in bytes
#define SCRATCH_OFF_2D_SRC_PTR      0
#define SCRATCH_OFF_2D_DST_PTR      4
#define SCRATCH_OFF_2D_DST_COUNT    8



#define mSCRATCH      m0
#define DELTA_LAST    m0
#define WORKER_LAST   m1
#define WORKER_COUNT  m5
#define WORKER_ID     m6
#define OFFS          m7
#define mSCRATCH2     m9


// Increment the PTR register (with a dummy load) by N elements of length
// BIT_SIZE (8, 16 or 32)
.macro INCREMENT_PTR  BIT_SIZE  PTR  N
  .ifeqs "\BIT_SIZE", "8"
      ldz8step   $mzero, $mzero, $\PTR+=, $\N
  .else
    .ifeqs "\BIT_SIZE", "16"
      ldz16step   $mzero, $mzero, $\PTR+=, $\N
    .else
      .ifeqs "\BIT_SIZE", "32"
        ld32step   $mzero, $mzero, $\PTR+=, $\N
      .else
        .error "Unhandled bit size \BIT_SIZE in INCREMENT_PTR"
      .endif
    .endif
  .endif
.endm

//******************************************************************************
//
// Invoking this macro creates the 3 different entry points required to
// implement all vertices required for a cast from INTYPE to OUTTYPE:
// (Cast<INTYPE,OUTTYPE>, CastWorker<INTYPE,OUTTYPE>, Cast2d<INTYPE,OUTTYPE>).
//
// TYPES        INTYPE and OUTTYPE separated by underscore (i.e.:
//              'float_signed_char' or 'char_half'
//
// ALIAS_TYPES  If 'NONE', it's ignored. Otherwise is a pair of types like
//              TYPES, for which another entry point label will be setup (no
//              duplication of code, just the labels).
//              For instance, 'char_float' is the same as 'signed_char_float'
//
// INTYPE_BITS  8, 16, 32 depending on the size of INTYPE
//
// OUTTYPE_BITS 8, 16, 32 depending on the size of OUTTYPE
//
// OUT_SCALED32 1 => use SCALED_PTR32 for the output pointer in the vertex state
//              (Cast1DSingleWorker and Cast1D verticews); 0 => use SCALED_PTR64
//              (only if scaled ptrs are enabled at all, otherwise, this has no
//              effect).
//
// SETUP        An instruction or macro to call at the start to do any setup
//              (for instance setup the registers for clamping). Can be empty
//
//
//******************************************************************************
.macro INSTANTIATE_CAST TYPES ALIAS_TYPES INTYPE_BITS OUTTYPE_BITS OUT_SCALED32 SETUP

//------------------------------------------------------------------------------
// Entry point for the 1D worker vertex which is directly instantianted by
// poplibs (in the 'cast()' function).
// Just calls the core function on the 1D vector specified in the vertex state
//------------------------------------------------------------------------------
#define FUNC(TYPES) __runCodelet_popops__Cast1DSingleWorker___##TYPES

DEF_STACK_USAGE 0 .text.FUNC(\TYPES)
.section .text.FUNC(\TYPES)
.align 4
EXPORT_FN FUNC(\TYPES)
.ifnc \ALIAS_TYPES, NONE
  EXPORT_FN FUNC(\ALIAS_TYPES)
  FUNC(\ALIAS_TYPES):
.endif
FUNC(\TYPES):
// get fields from vertex state
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16   $IN_PTR,        $mzero, $mvertex_base, VOFF_IN / 2
    ldz16   $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_START / 2
    // Expand pointers:
    // Input pointer is SCALED64
    shl     $IN_PTR, $IN_PTR, 3
    .if \OUT_SCALED32
      // Output pointer is SCALED32
      shl     $OUT_PTR, $OUT_PTR, 2
      setzi   $mSCRATCH, TMEM_REGION0_BASE_ADDR
      add     $OUT_PTR, $OUT_PTR, $mSCRATCH
    .else
      // Output pointer is SCALED64
      shl     $OUT_PTR, $OUT_PTR, 3
    .endif
#else
    ld32    $IN_PTR,        $mzero, $mvertex_base, VOFF_IN / 4
    ld32    $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_START / 4
#endif
    ld32    $N_ELEMS,   $mzero, $mvertex_base, VOFF_N_ELEMS / 4

    \SETUP
    call    $lr, cast_\TYPES\()_core
    exitz   $mzero
FN_SIZE FUNC(\TYPES)

//------------------------------------------------------------------------------
// Entry point for the 1D MultiVertex.
// Read the vertex state and finds out which segment of the total 1D vector
// this has to process, then calls the core function on it.
//------------------------------------------------------------------------------
#define FUNC_1D(TYPES) __runCodelet_popops__Cast1D___##TYPES

DEF_STACK_USAGE 0 .text.FUNC_1D(\TYPES)
.section .text.FUNC_1D(\TYPES)
.align 4
EXPORT_FN FUNC_1D(\TYPES)
.ifnc \ALIAS_TYPES, NONE
  EXPORT_FN FUNC_1D(\ALIAS_TYPES)
  FUNC_1D(\ALIAS_TYPES):
.endif
FUNC_1D(\TYPES):

   // get fields from vertex state
#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16   $IN_PTR,        $mzero, $mvertex_base, VOFF_IN / 2
    ldz16   $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_START / 2
    // Expand pointers:
    // Input pointer is SCALED64
    shl     $IN_PTR, $IN_PTR, 3
    .if \OUT_SCALED32
      // Output pointer is SCALED32
      shl     $OUT_PTR, $OUT_PTR, 2
      setzi   $mSCRATCH, TMEM_REGION0_BASE_ADDR
      add     $OUT_PTR, $OUT_PTR, $mSCRATCH
    .else
      // Output pointer is SCALED64
      shl     $OUT_PTR, $OUT_PTR, 3
    .endif

#else
    ld32    $IN_PTR,  $mzero, $mvertex_base, VOFF_IN / 4
    ld32    $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_START / 4
#endif
    ld32    $N_ELEMS,   $mzero, $mvertex_base, VOFF_N_ELEMS / 4

    // 'divide work' part to find which segment of the full 1D vector we need
    // to process
    // Now $N_ELEMS contains the 'partitionParams' vertex state word (see the
    // C++ vertex definition for details).
    // We need to convert that into:
    //   $N_ELEMS: number of elements to be processed by this worker.
    //
    //   $OFFS   : Offset (in number of elements) to add to $IN_PTR and
    //             $OUT_PTR to get the correct start points for this worker
    and      $DELTA_LAST, $N_ELEMS, 0x7
    shr      $N_ELEMS, $N_ELEMS, 3
    and      $WORKER_LAST, $N_ELEMS, 0x7
    shr      $N_ELEMS, $N_ELEMS, 3
    and      $WORKER_COUNT, $N_ELEMS, 0x7
    shr      $N_ELEMS, $N_ELEMS, 3
    // Get worker ID
    get      $WORKER_ID, $WSR
    and      $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

    mul      $OFFS, $WORKER_ID, $N_ELEMS

    // If ID < WORKER_COUNT, N_ELEMS and OFFS are already ok.
    // WORKER_ID is 'rebased' on WORKER_COUNT
    sub      $mSCRATCH2, $WORKER_ID, $WORKER_COUNT
    brneg    $mSCRATCH2, 1f

    // If ID >= WORKER_COUNT, re-compute N_ELEMS, OFFS
    // If N_ELEMS==4, exit (we have to do N_ELEMS-4, i.e. 0, elements)
    add      $N_ELEMS, $N_ELEMS, -4
    brz      $N_ELEMS, .Lexit\@
    shl      $mSCRATCH2, $mSCRATCH2, 2
    sub      $OFFS, $OFFS, $mSCRATCH2
1:
    // Worker with ID $WORKER_LAST need to process $DELTA_LAST element less
    cmpne    $WORKER_ID, $WORKER_ID, $WORKER_LAST
    brnz     $WORKER_ID, 2f
    sub      $N_ELEMS, $N_ELEMS, $DELTA_LAST
2:

    // OFFS is in units of number of elements. We add it to input and output
    // pointers, scaled appropriately (2 bytes/4 bytes) using dummy loads
    // (this avoid having to multiply OFFS by the sizes)
    INCREMENT_PTR   \INTYPE_BITS  IN_PTR  OFFS
    INCREMENT_PTR   \OUTTYPE_BITS OUT_PTR OFFS

    \SETUP
    call     $lr, cast_\TYPES\()_core
.Lexit\@:
    exitz    $mzero
FN_SIZE FUNC_1D(\TYPES)



//------------------------------------------------------------------------------
// Entry point for the 2D worker vertex.
// Just process all the 1D subvectors one at a time.
//------------------------------------------------------------------------------
#define IN_ITERATOR                m5
#define OUT_ITERATOR               m6
#define NUM_VECTORS                m7

#define FUNC_2D(TYPES) __runCodelet_popops__Cast2D___##TYPES

DEF_STACK_USAGE 0 .text.FUNC_2D(\TYPES)
.section .text.FUNC_2D(\TYPES)
.align 4
EXPORT_FN FUNC_2D(\TYPES)
.ifnc \ALIAS_TYPES, NONE
  EXPORT_FN FUNC_2D(\ALIAS_TYPES)
  FUNC_2D(\ALIAS_TYPES):
.endif
FUNC_2D(\TYPES):
  ld32       $IN_ITERATOR, $mvertex_base, $mzero, VOFF_SRC_PTR_2D/4
  ld32       $OUT_ITERATOR, $mvertex_base, $mzero, VOFF_DST_PTR_BEGIN_2D/4
  ld32       $NUM_VECTORS, $mvertex_base, $mzero, VOFF_DST_SIZE_2D/4
  brnzdec    $NUM_VECTORS, 1f
  exitz      $mzero
1:
  \SETUP

.Lloop_2d_vectors\@:

  ld32step   $IN_PTR, $mzero, $IN_ITERATOR+=, 1
  ld32step   $OUT_PTR, $mzero, $OUT_ITERATOR+=, 1
  ld32step   $N_ELEMS, $mzero, $OUT_ITERATOR+=, 1
  st32       $IN_ITERATOR, $mworker_base, SCRATCH_OFF_2D_SRC_PTR/4
  st32       $OUT_ITERATOR, $mworker_base, SCRATCH_OFF_2D_DST_PTR/4
  st32       $NUM_VECTORS, $mworker_base, SCRATCH_OFF_2D_DST_COUNT/4

  call       $lr, cast_\TYPES\()_core

  ld32       $IN_ITERATOR, $mworker_base, SCRATCH_OFF_2D_SRC_PTR/4
  ld32       $OUT_ITERATOR, $mworker_base, SCRATCH_OFF_2D_DST_PTR/4
  ld32       $NUM_VECTORS, $mworker_base, SCRATCH_OFF_2D_DST_COUNT/4
  brnzdec    $NUM_VECTORS, .Lloop_2d_vectors\@

.L2D_end\@:
  exitz         $mzero
FN_SIZE FUNC_2D(\TYPES)

.endm // INSTANTIATE_CAST

INSTANTIATE_CAST half_float          NONE       16 32  0
INSTANTIATE_CAST float_half          NONE       32 16  1

INSTANTIATE_CAST unsigned_char_half  NONE        8 16  1
INSTANTIATE_CAST unsigned_char_float NONE        8 32  1
INSTANTIATE_CAST signed_char_half    char_half   8 16  1
INSTANTIATE_CAST signed_char_float   char_float  8 32  1

INSTANTIATE_CAST half_unsigned_char  NONE       16  8  1  SETUP_UCHAR_OVERFLOW_CHECK
INSTANTIATE_CAST float_unsigned_char NONE       32  8  1  SETUP_UCHAR_OVERFLOW_CHECK
INSTANTIATE_CAST half_signed_char    half_char  16  8  1  SETUP_SCHAR_OVERFLOW_CHECK
INSTANTIATE_CAST float_signed_char   float_char 32  8  1  SETUP_SCHAR_OVERFLOW_CHECK
#endif
