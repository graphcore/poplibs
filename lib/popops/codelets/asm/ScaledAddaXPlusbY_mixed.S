// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// Vertices for the computation of:   a*X[] + b*Y[]     where:
//
//       X[]      : HALF
//       Y[]      : HALF
//       'a', 'b' : FLOAT
//
// Operation is in-place (results stored back in X[])
//
// The operation is to be performed in single precision, converting X[] and Y[]
// to single float. The result is converted back to half before storing.
//
// Before executing the computation, the accuracy of 'a' and 'b' is verified.
// If they can be converted to HALF with "enough" accuracy (based on a
// 'tolerance' parameter), then the code in this file is not used,; instead
// we call the 'full HALF' code, passing 'a' and 'b' converted to HALF prec.
//
// Note the different naming of the vertex fields:
//         In this file        C++ code
//              'a'       =>    scaleA
//              'b'       =>    scaleB
//              'X'       =>      A
//              'Y'       =>      B
//
// Vertices are provided for the 'Supervisor' and 2D' cases, and for the scaling
// values 'a' and 'b' being constants and being single-element tensors.
//
// Memory constraints are not relevant for the worker code in this file but
// the entry points for 'a' and 'b' being Tensors exist both Memory Constraints
// = both False and True.
// This is needed if  we end up calling the 'full HALF' code that has both cases
// For the 'a' & 'b' = Const this is not applicable because the decision to use
// the vertex is taken at graph compilation time, where Memory Constraints is
// forced to False.

#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "ScaledAddSupervisor.inc"
#include "CommonPoplibsMacros.h.S"

#define VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS __runCodelet_popops__aXPlusbYSupervisor___half_float_false_false
#define VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS_MEMCONSTR __runCodelet_popops__aXPlusbYSupervisor___half_float_false_true
#define VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_CONST __runCodelet_popops__aXPlusbYSupervisor___half_float_true_false
#define VERTEX_AXPLUSBY_2D_HALF_FLOAT_TENS __runCodelet_popops__aXPlusbY2D___half_float_false_false
#define VERTEX_AXPLUSBY_2D_HALF_FLOAT_TENS_MEMCONSTR __runCodelet_popops__aXPlusbY2D___half_float_false_true
#define VERTEX_AXPLUSBY_2D_HALF_FLOAT_CONST __runCodelet_popops__aXPlusbY2D___half_float_true_false

#define VERTEX_AXMINUSBY_SUPERV_HALF_FLOAT_TENS __runCodelet_popops__aXMinusbYSupervisor___half_float_false_false
#define VERTEX_AXMINUSBY_SUPERV_HALF_FLOAT_TENS_MEMCONSTR __runCodelet_popops__aXMinusbYSupervisor___half_float_false_true
#define VERTEX_AXMINUSBY_2D_HALF_FLOAT_TENS __runCodelet_popops__aXMinusbY2D___half_float_false_false
#define VERTEX_AXMINUSBY_2D_HALF_FLOAT_TENS_MEMCONSTR __runCodelet_popops__aXMinusbY2D___half_float_false_true

#if defined(VECTOR_AVAIL_SCALED_PTR64)
  #define SCALED_PTR64_SHIFTS       3
#endif
  #define SCALED_PTR128_SHIFTS       4

#if defined(VECTOR_AVAIL_SHORT_SPAN)
  #define SHORT_SPAN_PTR_SIZE      20
  #define SHORT_SPAN_LENGTH_SHIFTS 12
#endif

/*
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

 The exact vertex state format depends on the type of the vertex
 (2D/Supervisor), on the availabilty of short pointers (IPU Mk0/Mk1),
 and on the type of the scale values (constant/tensor)

-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
                           +------------------+
                           | 2D worker vertex |
                           +------------------+

  Most                                Least
  Signif.                            Signif.
  /--------------- 32 bits ---------------\
                                            byte   word
                                            offs   offs
  +---------------------------------------+              \
  |           'A' ('X') row PTR           |   0      0   |
  +---------------------------------------+              +-- SPAN type
  |               Num Rows.               |   4      1   |
  +---------------------------------------+              /
  |           'B' ('Y') row PTR           |   8      2   ONE_PTR type
  +---------------------------------------+
  |        Scale A float [value/PTR]      |  12      3   \   float value or
  +---------------------------------------+              |-- PTR depending
  |        Scale B float [value/PTR]      |  16      4   /   if CONSTANT/TENSOR
  +---------------------------------------+
  |               tolerance               |  20      5   float value (TENS only)
  +---------------------------------------+

       ================= SHORT_SPAN available =================

'X' row PTR points to an array with 'Num Rows' words, each having the format:
  /- 11 bits-\ /------- 20 bits --------\
  +-+-----------+-------------------------+
  |.|   Count   |      Row N data PTR     | 4xN      N   SHORT SPAN type
  +-+-----------+-------------------------+

       ============== SHORT_SPAN *NOT* available ==============

'X' row PTR points to an array with 2*'Num Rows' words, each having the
format (where M = N/2):
  +---------------------------------------+              \
  |             Row M data PTR            | 4xN      N   |
  +---------------------------------------+              +-- SPAN type
  |                 Count                 |4xN+4   N+1   |
  +---------------------------------------+              /


       ================= SCALED_PTR_64 available =================

'Y' row PTR points to an array with 'Num Rows'/2 words, each having the
format (where M = 2*N):
  /----- 16 bits -----\
  +-------------------+-------------------+
  |  Row M+1 data PTR |   Row M data PTR  | 4xN      N
  +-------------------+-------------------+

       ============== SCALED_PTR_64 *NOT* available ==============

'Y' row PTR points to an array with 'Num Rows' words, each having the format:
  +---------------------------------------+
  |             Row N data PTR            | 4xN      N   ONE_PTR type
  +---------------------------------------+


-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
                          +-------------------+
                          | Supervisor vertex |
                          +-------------------+

       ================= SCALED_PTR_64 available =================

scaleA & scaleB values are constant:
                      /----- 16 bits -----\ byte   word
                                            offs   offs
  +-------------------+-------------------+
  | 'B' ('Y') data PTR| 'A' ('X') data PTR|   0      0   SCALED_PTR64
  +-------------------+-------------------+
  |           Scale A float value         |   4      1   float
  +---------------------------------------+
  | . . . . . . . . . | Count             |   8      2   unsigned short
  +-------------------+-------------------+
  |           Scale B float value         |  12      3   float
  +---------------------------------------+


scaleA & scaleB values are contained in a 1-value Tensor:

                      /----- 16 bits -----\ byte   word
                                            offs   offs
  +-------------------+-------------------+
  | 'B' ('Y') data PTR| 'A' ('X') data PTR|   0      0   SCALED_PTR64
  +-------------------+-------------------+
  |    Count          |    scale A PTR    |   4      1   SCALED_PTR128
  +-------------------+-------------------+
  | . . . . . . . . . |    scale B PTR    |   8      2   SCALED_PTR128
  +-------------------+-------------------+
  |               tolerance               |  12      3   float
  +---------------------------------------+


       ============== SCALED_PTR_64 *NOT* available ==============

scaleA & scaleB values are constant:

  +---------------------------------------+
  |          'A' ('X') data PTR           |   0      0   ONE_PTR type
  +-------------------+-------------------+
  |          'B' ('Y') data PTR           |   4      1   ONE_PTR type
  +---------------------------------------+
  |        Scale A float value            |   8      2   float
  +---------------------------------------+
  | . . . . . . . . . |       Count       |  12      3   unsigned short
  +-------------------+-------------------+
  |        Scale B float value            |  16      4   float
  +---------------------------------------+
  |               tolerance               |  20      5   float
  +---------------------------------------+


scaleA & scaleB values are contained in a 1-value Tensor:

  +---------------------------------------+
  |          'A' ('X') data PTR           |   0      0   ONE_PTR type
  +-------------------+-------------------+
  |          'B' ('Y') data PTR           |   4      1   ONE_PTR type
  +---------------------------------------+
  |    Count          | Scale A PTR       |   8      2   SCALED_PTR128
  +---------------------------------------+
  | . . . . . . . . . | Scale B PTR       |  12      3   SCALED_PTR128
  +---------------------------------------+
  |               tolerance               |  16      4   float
  +---------------------------------------+

-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
*/

// Offsets in bytes for the 2D vertex state (as described above)
#define VERTEX_2D_DATA_A_OFFSET     0
#define VERTEX_2D_NUM_ROWS_OFFSET   4
#define VERTEX_2D_DATA_B_OFFSET     8
#define VERTEX_2D_SCALE_A_OFFSET    12
#define VERTEX_2D_SCALE_B_OFFSET    16
#define VERTEX_2D_TOLERANCE_OFFSET  20

// Offsets in bytes for the supervisor vertex state (as described above) ...
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  // ======== Short pointers =======
  #define VERTEX_SV_DATA_A_OFFSET           0
  #define VERTEX_SV_DATA_B_OFFSET           2
  #define VERTEX_SV_SCALE_A_OFFSET          4

  // ... constant scale coefficients
  #define VERTEX_SV_COUNT_CONST_OFFSET      8
  #define VERTEX_SV_SCALE_B_CONST_OFFSET    12

  // ... or tensor scale coefficients
  #define VERTEX_SV_COUNT_OFFSET            6
  #define VERTEX_SV_SCALE_B_TENS_OFFSET     8
  #define VERTEX_SV_TOLERANCE_OFFSET       12
#else
  // ======== No short pointers =======
  #define VERTEX_SV_DATA_A_OFFSET     0
  #define VERTEX_SV_DATA_B_OFFSET     4
  #define VERTEX_SV_SCALE_A_OFFSET    8

  #define VERTEX_SV_COUNT_CONST_OFFSET      12
  #define VERTEX_SV_SCALE_B_CONST_OFFSET    16

  #define VERTEX_SV_COUNT_OFFSET      10
  #define VERTEX_SV_SCALE_B_OFFSET    12
  #define VERTEX_SV_TOLERANCE_OFFSET  16
#endif


// --------------------------------------------------------------------
//                        Supervisor code
// --------------------------------------------------------------------

// 'VERTEX(supervisor)' and 'VERTEX(axplusby_half).kernel' are defined in
// ScaledAddSupervisor_fp.S as the common supervisor code to partitions
// works among workers, and the worker for the HALF/HALF worker.
// We use the same macro name for consistency (and ease of text search)
#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty


// The checkAccuracyWorker worker thread uses a record on the supervisor stack
// with this format for in/out parameters
#define CHECK_ACCURACY_SCALE_A_OFFSET    0
#define CHECK_ACCURACY_SCALE_B_OFFSET    4
#define CHECK_ACCURACY_TOLERANCE_OFFSET  8

#define CHECK_ACCURACY_RESULT_OFFSET     0 // Result
#define CHECK_ACCURACY_SCALE_AB_OFFSET   4 // Scale values converted to HALF

// Note that this size must be less than STACK_SIZE (in ScaledAddSupervisor.inc)
#define CHECK_STATE_SIZE                12



#define vertexPtr       m0 // Standard vertex state pointer

//The scaleB value is not handled by 'VERTEX(supervisor)'
#define scaleB          m3


#define memBase          m7
#define workerVertexBase m7
#define isAccurate       m3
#define tolerance        m5

#define mscratch m8


// ---------------------------------------------------------------
// Supervisor entry point (constant scale values).
// Load parameters and jumps to the worker partitioning function
FN_SUPERVISOR_ENTRY_POINT   VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_CONST 4 "" STACK_SIZE
 #if defined(VECTOR_AVAIL_SCALED_PTR64)
  ld32    $sv_scale, $vertexPtr, $mzero, VERTEX_SV_SCALE_A_OFFSET/4
  ld32    $scaleB, $vertexPtr, $mzero, VERTEX_SV_SCALE_B_CONST_OFFSET/4
  ldz16   $sv_count, $vertexPtr, $mzero, VERTEX_SV_COUNT_CONST_OFFSET/2
#else
  ld32    $sv_scale, $vertexPtr, $mzero, VERTEX_SV_SCALE_A_OFFSET/4
  ld32    $scaleB, $vertexPtr, $mzero, VERTEX_SV_SCALE_B_CONST_OFFSET/4
  ldz16   $sv_count, $vertexPtr, $mzero, VERTEX_SV_COUNT_CONST_OFFSET/2
#endif
  add     $sp, $sp, -STACK_SIZE
  setzi   $sv_log2AtomSize, 1
  setzi   $sv_atomSizeMask, 1
  setzi   $sv_workerFunction, aXplusbY_half_float_worker
  // Store the scaleB value, which is not handled by VERTEX(supervisor), in the
  // record on the stack for the worker
  st32    $scaleB, $sp, $mzero, SV_STATE_SCALEB_OFFSET/4
  bri     VERTEX(supervisor)

FN_SIZE VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_CONST

// ---------------------------------------------------------------
// Common setup code used by both the AXMinusBY and AXPlusBY supervisors.
.macro AXBY_MIXED_SUPERVISOR_SETUP
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16   $sv_scale, $vertexPtr, $mzero, VERTEX_SV_SCALE_A_OFFSET/2
  ldz16   $scaleB, $vertexPtr, $mzero, VERTEX_SV_SCALE_B_TENS_OFFSET/2
  ld32    $tolerance, $vertexPtr, $mzero, VERTEX_SV_TOLERANCE_OFFSET/4
  ldz16   $sv_count, $vertexPtr, $mzero, VERTEX_SV_COUNT_OFFSET/2
  add     $sp, $sp, -STACK_SIZE
  setzi   $memBase, TMEM_REGION0_BASE_ADDR
#else
  ldz16   $sv_scale, $vertexPtr, $mzero, VERTEX_SV_SCALE_A_OFFSET/2
  ldz16   $scaleB, $vertexPtr, $mzero, VERTEX_SV_SCALE_B_OFFSET/2
  ld32    $tolerance, $vertexPtr, $mzero, VERTEX_SV_TOLERANCE_OFFSET/4
  add     $sp, $sp, -STACK_SIZE
  ldz16   $sv_count, $vertexPtr, $mzero, VERTEX_SV_COUNT_OFFSET/2
  setzi   $memBase, TMEM_REGION0_BASE_ADDR
#endif
  shl     $sv_scale, $sv_scale, SCALED_PTR128_SHIFTS
  shl     $scaleB, $scaleB, SCALED_PTR128_SHIFTS
.endm

// ---------------------------------------------------------------
// aXMinusbY Supervisor entry point (tensor scale values) with and without
// memory constraints. These just negate the scaleB value and then lets the
// aXPlusbY Supervisor do the rest.
FN_SUPERVISOR_ENTRY_POINT VERTEX_AXMINUSBY_SUPERV_HALF_FLOAT_TENS_MEMCONSTR 4 "" STACK_SIZE
  setzi   $sv_memConstraints, MEM_CONSTRAINTS_MASK
  bri     1f
FN_SIZE VERTEX_AXMINUSBY_SUPERV_HALF_FLOAT_TENS_MEMCONSTR

FN_SUPERVISOR_ENTRY_POINT   VERTEX_AXMINUSBY_SUPERV_HALF_FLOAT_TENS 4 "" STACK_SIZE
  setzi   $sv_memConstraints, 0
1:
  AXBY_MIXED_SUPERVISOR_SETUP

  or      $mscratch, $mzero, NEGATE_FLOAT_BY_XOR
  ld32    $scaleB, $mzero, $scaleB, 0
  ld32    $sv_scale, $mzero, $sv_scale, 0
  setzi   $sv_workerFunction, checkAccuracyWhenCastFloatV2ToHalfWorker
  // Need to get the worker vertex base as offset from base of memory for 'run'
  sub     $workerVertexBase, $sp, $memBase

  st32    $tolerance, $sp, $mzero, CHECK_ACCURACY_TOLERANCE_OFFSET/4
  nop
  // Negate scaleB
  xor     $scaleB, $scaleB, $mscratch
  st32    $sv_scale, $sp, $mzero, CHECK_ACCURACY_SCALE_A_OFFSET/4

  nop
  nop
  nop
  nop
  st32    $scaleB, $sp, $mzero, CHECK_ACCURACY_SCALE_B_OFFSET/4

  bri     VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS_RUN_WORKERS
FN_SIZE VERTEX_AXMINUSBY_SUPERV_HALF_FLOAT_TENS

// ---------------------------------------------------------------
// Supervisor entry point (tensor scale values) with and without
// memory constraints.
// Load parameters and jumps to the worker partitioning function.
// Instructions are moved around and nops added to avoid pipeline stalls
FN_SUPERVISOR_ENTRY_POINT VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS_MEMCONSTR 4 "" STACK_SIZE
  setzi   $sv_memConstraints, MEM_CONSTRAINTS_MASK
  bri     1f
FN_SIZE VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS_MEMCONSTR

FN_SUPERVISOR_ENTRY_POINT VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS 4 "" STACK_SIZE
  setzi   $sv_memConstraints, 0
1:
  AXBY_MIXED_SUPERVISOR_SETUP

  ld32    $sv_scale, $mzero, $sv_scale, 0
  ld32    $scaleB, $mzero, $scaleB, 0
  setzi   $sv_workerFunction, checkAccuracyWhenCastFloatV2ToHalfWorker
  // Need to get the worker vertex base as offset from base of memory for 'run'
  sub     $workerVertexBase, $sp, $memBase

  st32    $tolerance, $sp, $mzero, CHECK_ACCURACY_TOLERANCE_OFFSET/4
  nop
  st32    $sv_scale, $sp, $mzero, CHECK_ACCURACY_SCALE_A_OFFSET/4
  st32    $scaleB, $sp, $mzero, CHECK_ACCURACY_SCALE_B_OFFSET/4

VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS_RUN_WORKERS:
  // Store the scaleB value for the structure on stack used by the worker
  // called by VERTEX(supervisor); VERTEX(supervisor) itself is not aware of it.
  st32    $scaleB, $sp, $mzero, SV_STATE_SCALEB_OFFSET/4

  run     $sv_workerFunction, $workerVertexBase, 0
  // wait for worker to terminate before checking result. We cannot do anything
  // while the worker runs as we have to wait for the result.
  sync    TEXCH_SYNCZONE_LOCAL
  ld32    $isAccurate, $sp, $mzero, CHECK_ACCURACY_RESULT_OFFSET/4

  setzi   $sv_workerFunction, aXplusbY_half_float_worker
  setzi   $sv_log2AtomSize, 1
  setzi   $sv_atomSizeMask, 1
  nop
  nop

  brnz    $isAccurate, 1f
  bri     VERTEX(supervisor)
1:
  // If the accuracy check returned non-zero (i.e. HALF are accurate enough),
  // run the faster data=HALF/scales=HALF code

  // Get the combined scale A & B values, returned by the checkAccuracy thread
  ld32    $sv_scale, $sp, $mzero, CHECK_ACCURACY_SCALE_AB_OFFSET/4
  setzi   $sv_log2AtomSize, 2
  setzi   $sv_atomSizeMask, 0x3

  setzi   $sv_workerFunction, VERTEX(axplusby_half).kernel
  bri     VERTEX(supervisor)

FN_SIZE VERTEX_AXPLUSBY_SUPERV_HALF_FLOAT_TENS

#undef vertexPtr
#undef scaleB
#undef memBase
#undef workerVertexBase
#undef isAccurate
#undef tolerance
#undef mscratch


// --------------------------------------------------------------------
//                           Worker code
// --------------------------------------------------------------------


// ****************************************************************************
// A macro to check if two FLOAT values, when converted to HALF, are both
// within the desired accuracy.
// Exceptions needs to be masked before using the macro, to avoid getting one
// if the FLOAT value is out of range of HALF
//
// Parameters:
//   $a0 (value1)        First float value (unmodified on exit)
//   $a1 (value2)        Second float value (unmodified on exit)
//   $a2 (tolerance)     The tolerance, multiplied by each float value, gives
//                       the greatest error that is acceptable in the conversion
//                       from float to half for that value
// On exit:
//   $m0 (bothAccurate)  Set to 'TFPU_FP32_TRUE' (0xffffffff) if both values
//                       can be converted to half with the desired accuracy;
//                       set to '0' otherwise.
//
//   $a2 (value1_2_half) The first and second value converted to halves.
//                       First value ($a0) converted is in the low half word,
//                       second one ($a1) is the high half word.
// ****************************************************************************
#define value1          a0
#define value2          a1
#define value1_2        a0:1
#define tolerance       a2
#define value1_2_half   a2
#define saveFpCtl       a3
#define maxErr1         a6
#define maxErr2         a7
#define maxErr1_2       a6:7
#define diff1           a4
#define diff2           a5
#define diff1_2         a4:5
#define lessThan1       a6
#define lessThan2       a7
#define lessThan1_2     a6:7
#define bothAccurate_a  a6
#define bothAccurate    m0

.macro CHECK_ACCURACY_FLOAT_V2_TO_HALF
  // Compute the maximum admissible error
  f32v2mul    $maxErr1_2, $tolerance:B, $value1_2// multiply inputs by tolerance
  f32v2absadd $maxErr1_2, $maxErr1_2, $azeros// abs value: max admissible errors

  // Compute the diffs between the inputs and the inputs converted to halves
  f32v2tof16  $value1_2_half, $value1_2     // convert 2 input singles to halves
  f16v2tof32  $diff1_2, $value1_2_half      // and back to single
  f32v2sub    $diff1_2, $diff1_2, $value1_2 // subtract original input values
  f32v2absadd $diff1_2, $diff1_2, $azeros   // get absolute value of diffs

  // Return TFPU_FP32_TRUE if *both* differences are less than their errors
  f32v2cmplt  $lessThan1_2, $diff1_2, $maxErr1_2
  and         $bothAccurate_a, $lessThan1, $lessThan2
  mov         $bothAccurate, $bothAccurate_a  // transfer to MRF register
.endm

// ****************************************************************************
// The above code for accuracy check of 2 single floats, as a worker thread.
// The return values ($m0 and $a2 from the macro) are stored in the vertex
// state itself.
// ****************************************************************************
FN_SECTION checkAccuracyWhenCastFloatV2ToHalfWorker
checkAccuracyWhenCastFloatV2ToHalfWorker:
  ld32   $value1, $mvertex_base, $mzero, CHECK_ACCURACY_SCALE_A_OFFSET/4
  ld32   $value2, $mvertex_base, $mzero, CHECK_ACCURACY_SCALE_B_OFFSET/4
{
  ld32   $tolerance, $mvertex_base, $mzero, CHECK_ACCURACY_TOLERANCE_OFFSET/4
  uput   $FP_CTL, $azero   // disable FP exceptions for macro
}
  CHECK_ACCURACY_FLOAT_V2_TO_HALF
  st32   $bothAccurate, $mvertex_base, $mzero, CHECK_ACCURACY_RESULT_OFFSET/4
  st32   $value1_2_half, $mvertex_base, $mzero, CHECK_ACCURACY_SCALE_AB_OFFSET/4
  exitz  $mzero
FN_SIZE checkAccuracyWhenCastFloatV2ToHalfWorker

#undef value1
#undef value2
#undef value1_2
#undef saveFpCtl
#undef maxErr1
#undef maxErr2
#undef maxErr1_2
#undef diff1
#undef diff2
#undef diff1_2
#undef lessThan1
#undef lessThan2
#undef lessThan1_2
#undef bothAccurate_a

// ------------------------------------------------------------------
// REGISTER NAMING FOR THE WORKER THREADS
#define Xptr           m0  // Will read X values with this ptr
#define Yptr           m1  // Will read Y values with this ptr

#define XptrStore      m2  // Will store X values with this ptr
#define size           m3  // number of half values to process
#define rptCount       m4  // counter for rpt loop
#define remainder      m5  // 0/1 to indicate even/odd size
#define scratch        m4  // same as rptCount
#define Xrow           m6  // for the 2D vertex, pointer to the 2d vector
#define Yrow           m7  // for the 2D vertex, pointer to the 2d vector
#define numRows        m8  // for the 2D vertex
#define stride         m9  // 1 or 6, process contiguous or strided elements
#define memConstraints m11 // passed to the _half_half entry point


#define a             a0   // used briefly, same as $result
#define b             a1   // scaleB for 'Y' values
#define result        a0   // two result values, as half floats
#define Xhalf         a2   // Where we read the X input (2 halves)
#define Yhalf         a3   // Where we read the Y input (2 halves)
#define X1            a4   // first X value, expanded as single float
#define X2            a5   // second X value, expanded as single float
#define X             a4:5 // two X values, as single floats
#define Y1            a6   // first Y value, expanded as single float
#define Y2            a7   // second Y value, expanded as single float
#define Y             a6:7 // two Y values, as single floats
#define ascratch      a4

// ---------------------------------------------------------------
// The supervisor vertices will start 6 of these worker threads.
// Each will processes elements from the 1D vector, with a stride of
// 12 halves [CTXT_WORKERS (6) x 2 halves] = 6 words.
// This will take its parameters from the state record created on the
// (supervisor) stack by the supevisor vertex.
// It will adjust the parameters (start ptrs, count of elements) based
// on its worker id.
#define workerIdM1  m4
#define remM1       m6 // workers [0..'remM1'] will process 2 extra halves
#define final       m7 // workers 'remM1' will 'final' (0 or 1) extra half
#define mscratch    m8

FN_SECTION aXplusbY_half_float_worker
aXplusbY_half_float_worker:
  // Get the worker ID (0..5)
  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
  // Get scaleA coefficient
  ld32          $a, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
{
  // ScaleB value is stored in the 'MEM_CONSTRAINTS' word (no memory
  // constraints for this vertex)
  ld32          $b, $mvertex_base, $mzero, SV_STATE_SCALEB_OFFSET/4
  uput          $TAS, $a
}

  ld32          $Xptr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET/4
  ld32          $Yptr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4
  ld32          $size, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
  ld32          $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
  ld32          $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4
  setzi         $stride, CTXT_WORKERS

  // If worker id is less than the remainder this worker processes an extra 2.
  cmpslt        $mscratch, $workerIdM1, $remM1
  shl           $mscratch, $mscratch, 1
  add           $size, $size, $mscratch
  // If this is the "last" worker, it may process one extra single value
  cmpeq         $mscratch, $workerIdM1, $remM1
  brz           $mscratch, 1f
  add           $size, $size, $final
1:
  // Increment X,Y pointers by the 0-based worker Id for the strided access
  ld32step      $mscratch, $mzero, $Xptr+=, $workerIdM1
  ld32step      $mscratch, $mzero, $Yptr+=, $workerIdM1

  call          $lr, aXplusbY_half_float_kernel
  exitz         $mzero
FN_SIZE aXplusbY_half_float_worker

// ---------------------------------------------------------------
// Common worker code used by both AXPlusBy2D and AXMinusBy2D. The optional
// argument is intended to allow the scale B factor to be negated when called by
// AXMinusBy2D
.macro VERTEX_AXBY_2D_HALF_FLOAT_TENS_COMMON OPTIONAL_SCALEB_MODIFIER
  ld32          $tolerance, $mvertex_base, $mzero, VERTEX_2D_TOLERANCE_OFFSET/4

  // The 'a' scale will go in the special $TAS register, for the f32v2axpy instr
  ld32          $scratch, $mvertex_base, $mzero, VERTEX_2D_SCALE_A_OFFSET/4
  ld32          $a, $scratch, $mzero, 0
{
  // The 'b' scale.
  ld32          $scratch, $mvertex_base, $mzero, VERTEX_2D_SCALE_B_OFFSET/4
  // save exception register to restore later
  uget          $a3, $FP_CTL
}
{
  ld32          $b, $scratch, $mzero, 0
  uput          $FP_CTL, $azero
} // disable exceptions for the following macro

  CHECK_ACCURACY_FLOAT_V2_TO_HALF

  \OPTIONAL_SCALEB_MODIFIER

  // if one (or both) not accurate enough, use the mixed code in this file
{
  brz           $bothAccurate, .LcontinueMixedCode
  uput          $FP_CTL, $a3 // restore FP exception register
}
  // We can use the faster _half_half version (in ScaledAdd2D_half.S).
  // Load registers as required there:
  //    $outData        ($m0)  [Would be $Xrow here]
  //    $outDataB       ($m2)  [Would be $Yrow here]
  //    $outDataSize    ($m1)  [Would be $numRows here]
  //    $memConstraints ($m11)
  //    $TAS                    'a'  and 'b'
  ld32          $m1,  $mvertex_base, $mzero, VERTEX_2D_NUM_ROWS_OFFSET/4
  brz           $m1, .Lend
  ld32          $m0,  $mvertex_base, $mzero, VERTEX_2D_DATA_A_OFFSET/4
  ld32          $m2,  $mvertex_base, $mzero, VERTEX_2D_DATA_B_OFFSET/4
{
  bri           axplusby_half_half_common
  uput          $TAS, $value1_2_half // set by CHECK_ACCURACY_FLOAT_V2_TO_HALF
}
.endm

// ---------------------------------------------------------------
// 2D scaled minus (worker) entry point: tensor scale values, WITH memory constraints
FN_WORKER_ENTRY_POINT   VERTEX_AXMINUSBY_2D_HALF_FLOAT_TENS_MEMCONSTR
  setzi         $memConstraints, 1
  bri           1f
FN_SIZE VERTEX_AXMINUSBY_2D_HALF_FLOAT_TENS_MEMCONSTR

// ---------------------------------------------------------------
// For the 'Minus' vertex we need to invert the 'b' scale both in $b (in case we
// are about to jump to ''.LcontinueMixedCode') and in $value1_2_half (in case
// we will jump to 'axplusby_half_half_common')
.macro INVERT_SCALE_B
  f32sub     $b, $azero, $b
  f32tof16   $ascratch, $b
  sort4x16lo $value1_2_half, $value1_2_half, $ascratch
.endm

// ---------------------------------------------------------------
// 2D scaled minus (worker) entry point: tensor scale values, NO memory constraints
FN_WORKER_ENTRY_POINT  VERTEX_AXMINUSBY_2D_HALF_FLOAT_TENS
  setzi   $memConstraints,0
1:
  VERTEX_AXBY_2D_HALF_FLOAT_TENS_COMMON INVERT_SCALE_B

FN_SIZE VERTEX_AXMINUSBY_2D_HALF_FLOAT_TENS

// ---------------------------------------------------------------
// 2D (worker) entry point: tensor scale values, WITH memory constraints
// Note that memory constraints are not relevant to the code in this file
// but need to be passed on if the code in ScaledAdd2D_half.S is called
FN_WORKER_ENTRY_POINT  VERTEX_AXPLUSBY_2D_HALF_FLOAT_TENS_MEMCONSTR
  setzi         $memConstraints, 1
  bri           1f
FN_SIZE VERTEX_AXPLUSBY_2D_HALF_FLOAT_TENS_MEMCONSTR

// ---------------------------------------------------------------
// 2D (worker) entry point: tensor scale values, NO memory constraints
FN_WORKER_ENTRY_POINT  VERTEX_AXPLUSBY_2D_HALF_FLOAT_TENS
  setzi         $memConstraints,0
1:
  VERTEX_AXBY_2D_HALF_FLOAT_TENS_COMMON

FN_SIZE VERTEX_AXPLUSBY_2D_HALF_FLOAT_TENS_FAST

// ---------------------------------------------------------------
// 2D (worker) entry point (constant scale values).
FN_WORKER_ENTRY_POINT  VERTEX_AXPLUSBY_2D_HALF_FLOAT_CONST

  // The 'a' scale will go in the special $TAS register, for the f32v2axpy instr
  ld32          $a, $mvertex_base, $mzero, VERTEX_2D_SCALE_A_OFFSET/4
  // The 'b' scale.
  ld32          $b, $mvertex_base, $mzero, VERTEX_2D_SCALE_B_OFFSET/4
.LcontinueMixedCode:
  ld32          $numRows,  $mvertex_base, $mzero, VERTEX_2D_NUM_ROWS_OFFSET/4
  brz           $numRows, .Lend

{
  ld32          $Xrow,  $mvertex_base, $mzero, VERTEX_2D_DATA_A_OFFSET/4
  uput          $TAS, $a
}
  ld32          $Yrow,  $mvertex_base, $mzero, VERTEX_2D_DATA_B_OFFSET/4

  setzi         $stride, 1
  // Loop over all rows
  add           $numRows, $numRows, -1
.Lrow_loop:
  // Get X data pointer for this row based on ptr availiability
  #if defined(VECTOR_AVAIL_SHORT_SPAN)
    ld32step      $Xptr, $mzero, $Xrow+=, 1
    shr           $size, $Xptr, SHORT_SPAN_PTR_SIZE
    shl           $Xptr, $Xptr, SHORT_SPAN_LENGTH_SHIFTS
    shr           $Xptr, $Xptr, SHORT_SPAN_LENGTH_SHIFTS
  #else
    ld32step      $Xptr, $mzero, $Xrow+=, 1
    ld32step      $size, $mzero, $Xrow+=, 1
  #endif
  // Get Y data pointer for this row based on ptr availiability
  #if defined(VECTOR_AVAIL_SCALED_PTR64)
    ldz16step     $Yptr, $mzero, $Yrow+=, 1
    shl           $Yptr, $Yptr, SCALED_PTR64_SHIFTS
  #else
    ld32step      $Yptr, $mzero, $Yrow+=, 1
  #endif

  call          $lr, aXplusbY_half_float_kernel
  brnzdec       $numRows, .Lrow_loop
.Lend:
  exitz         $mzero

FN_SIZE VERTEX_AXPLUSBY_2D_HALF_FLOAT_CONST

// *********************************************************************
// Main worker computation code for all the vertices. Processes one
// vector of data, with successive processed elements being either
// contiguous, or CTXT_WORKERS (i.e. 6) elements apart ($stride = 1 or 6),
// for the 2D or Supervisor vertex.
//
// These registers must be set on entry in this function:
//    $Xptr, $Yptr    pointers to the first values to process.
//    $size           how many half values to process. Can be 0, odd or even.
//    $stride         The stride between each value to process  (1 or 6)
//    $b              The scaleB value.
//    TAS             The TAS special register must be loaded with the
//                    'scaleA' value.
// This will also use the $XptrStore, $rptCount/$scratch and $remainder MRF
// registers, and all the ARF registers.
// The operation is done with two floating point instructions:
//   32 bit (x2) multiply:   Y' <- b*Y
//   32 bit (x2) axpy:       a*X  + Y'
// *********************************************************************
FN_SECTION aXplusbY_half_float_kernel 8
aXplusbY_half_float_kernel:
  mov           $XptrStore, $Xptr  // A second 'X' pointer for stores
  // $remainder is 1 if there is a lone last value to process ($size is odd)
  and           $remainder, $size, 1  // is $size odd?
  // But, if less than 4 values, cannot process in the main prologue+loop.
  cmpult        $scratch, $size, 4
  brnz          $scratch, .Lprocess_less_than_4

  // $rptCount will be the counter for the rpt instruction
  shr           $rptCount, $size, 1   // $rptCount = number of pairs

  // Prologue to start the pipeline for the loop.
  // The loop will process one pair of half values per iteration.
  // This prologue will process the first two pairs of half float result values
  // into '$result' and AACC.
  // The two-step prologue is needed, because the f32v2axpy instruction has an
  // internal pipleine stage (using the AACC register)
  ld32step      $Yhalf, $mzero, $Yptr+=, $stride // Load [Y0,1] [halves]
{
  ld32step      $Xhalf, $mzero, $Xptr+=, $stride // Load [X0,X1] [halves]
  f16v2tof32    $Y, $Yhalf                       // Convert [Y0,Y1] into singles
}
  f32v2mul      $Y, $b:B, $Y                     // [Y0,Y1]  <- b*[Y0,Y1]
{
  ld32step      $Yhalf, $mzero, $Yptr+=, $stride // Load [Y2,Y3] [halves]
  f16v2tof32    $X, $Xhalf                       // Convert [X0,X1] into singles
}
{
  ld32step      $Xhalf, $mzero, $Xptr+=, $stride // Load [X2,X3] [halves]
  f32v2axpy     $azeros, $X, $Y                 // AACC <- a*[X0,X1] + b*[Y0,Y1]
}
{
  ld32step      $Yhalf, $mzero, $Yptr+=, $stride // Load [Y4,Y5] [halves]
  f16v2tof32    $Y, $Yhalf                       // Convert [Y2,Y3] into singles
}
  f32v2mul      $Y, $b:B, $Y                     // [Y2,Y3]  <-  b*[Y2,Y3]
{
  ld32step      $Xhalf, $mzero, $Xptr+=, $stride // Load [X4,X5] [halves]
  f16v2tof32    $X, $Xhalf                       // Convert [X2,X3] into singles
}
{
  add           $rptCount, $rptCount, -2 // two pairs are done in this prologue
  f32v2axpy     $X, $X, $Y   // AACC <- a*[X2,X3] + b*[Y2,Y3]; [X0,X1]<-AACC
}
{
  rpt           $rptCount, (2f-1f)/8-1
  f32v2tof16    $result, $X             // convert [X0,X1] (result) to 2 halves
}
// ===== Loop on two halves each iteration (2.5 cycles/1 half value) ====
// 1st iteration: store [X0,X1], convert [X2,X3], process [X4,X5], load [X6,X7]
// 2nd iteration: store [X2,X3], convert [X4,X5], process [X6,X7], load [X8,X9]
//   . . .
// The epilogue after the loop will need to convert to half the penultimate
// pair and store both the penultimate and the last pair.
// The loop will over-read one word from X and Y, if $size is even. When started
// from supervisor the overread is 12 halves = 24 bytes which is the max allowed
1:
{
  ld32step      $Yhalf, $mzero, $Yptr+=, $stride// Load [Y6,Y7]. Might over-read
  f16v2tof32    $Y, $Yhalf                      // Convert [Y4,Y5] to singles
}
{
  st32step      $result, $mzero, $XptrStore +=, $stride // Store [X0,X1] result
  f32v2mul      $Y, $b:B, $Y                            // [Y4,Y5] <- b*[Y4,Y5]
}
{
  ld32step      $Xhalf, $mzero, $Xptr+=, $stride//Load [X6,X7]. Might over-read
  f16v2tof32    $X, $Xhalf                      // Convert [X4,X5] to singles
}
{
  nop
  f32v2axpy     $X, $X, $Y  // AACC <- a*[X4,X5] + b*[Y4,Y5]; [X2,X3]<-AACC
}
{
  nop
  f32v2tof16    $result, $X    // convert [X2,X3] (result) to 2 halves
}
// ==== end  of loop ====
2:
{
  st32step      $result, $mzero, $XptrStore +=, $stride // store [X2,X3] result
  f32v2axpy     $X, $azeros, $azeros                   // [X4,X5] <- AACC
}
  f32v2tof16    $result, $X              // convert [X4,X5] (result) to 2 halves
  st32step      $result, $mzero, $XptrStore +=, $stride// store [X4,X5] (result)

// Is there a single half value left to process ($remainder), or none?
.Lprocess_zero_one:
  // when we reach here, the last X and Y half value are already loaded in
  // $Xhalf,$Yhalf. The value to write (if any) is the least significant
  // half word. The top half word of $Xhalf also will need to be written back
  // to memory
{
  brz           $remainder, .Ldone      // If none, bail out
  mov           $Y2, $azero             // zero for the following f32v2axpy
}
  mov           $X2, $azero             // zero for the following f32v2axpy
  f16tof32      $X1, $Xhalf             // Convert 'X' into single
  f16tof32      $Y1, $Yhalf             // Convert 'Y' into single
  f32mul        $Y1, $b, $Y1            // Y  <-  b*Y

  // This does two values, when only one is needed. If $size was exactly 1, we
  // could end up processing NANs in the unused part (because X2, Y2 would have
  // never been initialized to valid float values). This is why we earlier
  // zeroed X2, Y2
  f32v2axpy     $azeros, $X, $Y              // AACC <- a*X + b*Y
  f32v2axpy     $X, $azeros, $azeros         // X <- AACC
  f32tof16      $result, $X1                 // convert result to half

  // Mix 16-bit result with pre-existing top half word (top half word of $Xhalf)
  // and write back
  sort4x16hi    $X1, $result, $Xhalf
  st32          $X1, $mzero, $XptrStore, 0
.Ldone:
  br            $lr

// This will handle the case for $size = 0, 1, 2 or 3
.Lprocess_less_than_4:
  brz           $size, .Ldone    // if none, need to bail out before the loads

  ld32step      $Xhalf, $mzero, $Xptr+=, $stride // this might over-read (1 word)
  ld32step      $Yhalf, $mzero, $Yptr+=, $stride
  cmpeq         $scratch, $size, 1
  brnz          $scratch, .Lprocess_zero_one   // if only one to do, jump out
  // Note: cannot bundle the floating point instructions below with the
  // loads, compare branch above, as the second half word might be any pattern
  // in memory (if $size==1), including NANs, which can cause an exception
  f16v2tof32    $X, $Xhalf                 // Convert X into singles
  f16v2tof32    $Y, $Yhalf                 // Convert Y into singles
  f32v2mul      $Y, $b:B, $Y               // Y  <-  b*Y
  f32v2axpy     $azeros, $X, $Y            // AACC <- a*X+b*Y
{
  ld32step      $Xhalf, $mzero, $Xptr+=, $stride// this might over-read (1 word)
  f32v2axpy     $X, $azeros, $azeros       // X <- AACC
}
{
  ld32step      $Yhalf, $mzero, $Yptr+=, $stride// this might over-read (1 word)
  f32v2tof16    $result, $X                // convert X (result) to 2 halves
}
  st32step      $result, $mzero, $XptrStore +=, $stride // store X result
  bri           .Lprocess_zero_one

FN_SIZE aXplusbY_half_float_kernel

#endif  // IPU
