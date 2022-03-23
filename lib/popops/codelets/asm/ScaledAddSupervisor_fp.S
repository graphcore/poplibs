// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "ScaledAddSupervisor.inc"
#include "CommonPoplibsMacros.h.S"
#include "workDivision.h.S"
#include "MathConstants.S"

#define VERTEX(ty) __runCodelet_popops__ScaledAddSupervisor___ ## ty
#define SUBTRACT_VERTEX(ty) __runCodelet_popops__ScaledSubtractSupervisor___ ## ty
#define AXPLUSBY_VERTEX(ty) __runCodelet_popops__aXPlusbYSupervisor___ ## ty
#define XMINUSAXPLUSBY_VERTEX(ty) __runCodelet_popops__XMinusaXPlusbYSupervisor___ ## ty
#define AX_MINUS_BY_VERTEX(ty) __runCodelet_popops__aXMinusbYSupervisor___ ## ty


// vertex state (offsets in bytes)
//
// Vertex state
//
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_B_OFFSET 4
#define VERTEX_SCALE_OFFSET 8
#define VERTEX_PACKED_COUNT_OFFSET 10
#define VERTEX_SCALE_B_OFFSET 12

// constants
#define SCALED_PTR128_SHL_BITS 4

// supervisor variables
#define vertexPtr m0
#define final m2

#define remM1 m3
#define mscratch  m4
#define mscratch2 m5
#define mscratch3 m6


//******************************************************************************
// Float variant entry points
//******************************************************************************
FN_SUPERVISOR_ENTRY_POINT VERTEX(float_float_float_false_false)  4 ""
  setzi $sv_workerFunction, VERTEX(float).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles

FN_EXPORT VERTEX(float_float_float_false_true)
  setzi $sv_workerFunction, VERTEX(float_true).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE VERTEX(float_float_float_false_false)


FN_SUPERVISOR_ENTRY_POINT SUBTRACT_VERTEX(float_float_float_false) 4 ""
  setzi $sv_workerFunction, VERTEX(sub_float_false).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_EXPORT SUBTRACT_VERTEX(float_float_float_true)
  setzi $sv_workerFunction, VERTEX(sub_float_true).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE SUBTRACT_VERTEX(float_float_float_false)






FN_SUPERVISOR_ENTRY_POINT AXPLUSBY_VERTEX(float_float_false_false) 4 ""
  setzi $sv_workerFunction, AXPLUSBY_VERTEX(float).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE AXPLUSBY_VERTEX(float_float_false_false)


FN_SUPERVISOR_ENTRY_POINT AX_MINUS_BY_VERTEX(float_float_false_false) 4 ""
  setzi $sv_workerFunction, AXPLUSBY_VERTEX(sub_float).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE AX_MINUS_BY_VERTEX(float_float_false_false)

//******************************************************************************
// Half variant entry points
//******************************************************************************

DEF_STACK_SIZE_OWN STACK_SIZE .text.VERTEX(half_half_half_false_false)
FN_SUPERVISOR_ENTRY_POINT VERTEX(half_half_half_false_false) 4 "" NONE
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  setzi $sv_memConstraints, 0
  bri   1f

FN_EXPORT VERTEX(half_half_half_false_true)
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  setzi $sv_memConstraints, MEM_CONSTRAINTS_MASK
1:
  shl    $mscratch, $mscratch, SCALED_PTR128_SHL_BITS

  add   $sp, $sp, -STACK_SIZE
  setzi $sv_log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $sv_atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX(half).kernel
  ldz16  $sv_count, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ldz16 $sv_scale, $mzero, $mscratch, 0
  bri   VERTEX(supervisor) // 6 cycles
FN_SIZE VERTEX(half_half_half_false_false)


DEF_STACK_SIZE_OWN STACK_SIZE .text.SUBTRACT_VERTEX(half_half_half_false)
FN_SUPERVISOR_ENTRY_POINT SUBTRACT_VERTEX(half_half_half_false) 4 "" NONE
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  setzi $sv_memConstraints, 0
  bri   1f
FN_EXPORT SUBTRACT_VERTEX(half_half_half_true)
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  setzi $sv_memConstraints, MEM_CONSTRAINTS_MASK
1:
  shl    $mscratch, $mscratch, SCALED_PTR128_SHL_BITS

  setzi $mscratch2, NEGATE_HALF_BY_XOR
  add   $sp, $sp, -STACK_SIZE
  setzi $sv_log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $sv_atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX(half).kernel
  ldz16  $sv_count, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // load factor using its pointer - here to avoid pipeline hit
  ldz16 $sv_scale, $mzero, $mscratch, 0
  xor   $sv_scale, $mscratch, $mscratch2 // 6 cycles
  bri   VERTEX(supervisor) // 6 cycles
FN_SIZE SUBTRACT_VERTEX(half_half_half_false)


DEF_STACK_SIZE_OWN STACK_SIZE .text.AXPLUSBY_VERTEX(half_half_false_false)
FN_SUPERVISOR_ENTRY_POINT AXPLUSBY_VERTEX(half_half_false_false) 4 "" NONE
  setzi $sv_memConstraints, 0
  bri   1f
FN_EXPORT AXPLUSBY_VERTEX(half_half_false_true)
  setzi $sv_memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  ldz16  $mscratch2, $vertexPtr, $mzero, VERTEX_SCALE_B_OFFSET/2

  // The call allows some code sharing and as we would suffer pipe hits
  // by using $mscratch, $mscratch2 frequently it doesn't cost many cycles.
  call $remM1, AX_MINUS_PLUS_BY_COMMON  // 6 cycles

  sort4x16lo $sv_scale, $mscratch, $mscratch2

  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX(axplusby_half).kernel
  bri   VERTEX(supervisor) // 6 cycles
FN_SIZE AXPLUSBY_VERTEX(half_half_false_false)


DEF_STACK_SIZE_OWN STACK_SIZE .text.XMINUSAXPLUSBY_VERTEX(half_false_false)
  // Implement (1 - a) X + b * Y as follows:
  //     -aX + bY + X
  //
  // The implementation is to be performed in 32-bit precision in the
  // accumulators.
FN_SUPERVISOR_ENTRY_POINT XMINUSAXPLUSBY_VERTEX(half_false_false) 4 "" NONE
  setzi $sv_memConstraints, 0
  bri   1f
FN_EXPORT XMINUSAXPLUSBY_VERTEX(half_false_true)
  setzi $sv_memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  ldz16  $mscratch2, $vertexPtr, $mzero, VERTEX_SCALE_B_OFFSET/2
  // The call allows some code sharing and as we would suffer pipe hits
  // by using $mscratch, $mscratch2 frequently it doesn't cost many cycles.
  setzi  $mscratch3, NEGATE_HALF_BY_XOR
  call $remM1, AX_MINUS_PLUS_BY_COMMON  // 6 cycles

  //negate scale_a
  xor $mscratch, $mscratch, $mscratch3

  sort4x16lo $sv_scale, $mscratch, $mscratch2

  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX(xminusaxplusby_half).kernel
  bri   VERTEX(supervisor) // 6 cycles
FN_SIZE XMINUSAXPLUSBY_VERTEX(half_false_false)



DEF_STACK_SIZE_OWN STACK_SIZE .text.AX_MINUS_BY_VERTEX(half_half_false_false)
FN_SUPERVISOR_ENTRY_POINT AX_MINUS_BY_VERTEX(half_half_false_false) 4 "" NONE
  setzi $sv_memConstraints, 0
  bri   1f
FN_EXPORT AX_MINUS_BY_VERTEX(half_half_false_true)
  setzi $sv_memConstraints, MEM_CONSTRAINTS_MASK
1:
  ldz16  $mscratch, $vertexPtr, $mzero, VERTEX_SCALE_OFFSET/2
  ldz16  $mscratch2, $vertexPtr, $mzero, VERTEX_SCALE_B_OFFSET/2

  // The call allows some code sharing and as we would suffer pipe hits
  // by using $mscratch, $mscratch2 frequently it doesn't cost many cycles.
  setzi  $mscratch3, NEGATE_HALF_BY_XOR
  call $remM1, AX_MINUS_PLUS_BY_COMMON  // 6 cycles

  //negate scale_b
  xor $mscratch2, $mscratch2, $mscratch3

  sort4x16lo $sv_scale, $mscratch, $mscratch2

  // pointer to the worker code to run
  setzi $sv_workerFunction, VERTEX(axplusby_half).kernel
  bri   VERTEX(supervisor) // 6 cycles
FN_SIZE AX_MINUS_BY_VERTEX(half_half_false_false)

FN_SUPERVISOR_SECTION AX_MINUS_PLUS_BY_COMMON 4 "" STACK_SIZE
AX_MINUS_PLUS_BY_COMMON:
  shl    $mscratch, $mscratch, SCALED_PTR128_SHL_BITS
  shl    $mscratch2, $mscratch2, SCALED_PTR128_SHL_BITS
  add   $sp, $sp, -STACK_SIZE
  setzi $sv_log2AtomSize, LOG2_HALF_ATOM_SIZE
  setzi $sv_atomSizeMask, (1 << LOG2_HALF_ATOM_SIZE) - 1

  ldz16  $sv_count, $vertexPtr, $mzero, VERTEX_PACKED_COUNT_OFFSET/2

  // load factors using pointers - here to avoid pipeline hit
  ldz16  $mscratch, $mzero, $mscratch, 0
  ldz16  $mscratch2, $mzero, $mscratch2, 0
  br $remM1
FN_SIZE AX_MINUS_PLUS_BY_COMMON

//******************************************************************************
// Mixed precision, i.e. dataA (X) = half, dataB (Y) = float, scaleB = half
// (tensor) entry point
// The leaf code has no constraints.
//******************************************************************************

FN_SUPERVISOR_ENTRY_POINT VERTEX(half_float_half_false_false) 4 ""
  setzi $sv_workerFunction, VERTEX(half_float_half).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles

FN_SIZE VERTEX(half_float_half_false_false)

//******************************************************************************
// Mixed precision, i.e. dataA (X) = half, dataB (Y) = float, scaleB = float
// (tensor) entry point
// The leaf code has no constraints.
//******************************************************************************
FN_SUPERVISOR_ENTRY_POINT VERTEX(half_float_float_false_false) 4 ""
  setzi $sv_workerFunction, VERTEX(half_float_float).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles
FN_SIZE VERTEX(half_float_float_false_false)

//*****************************************************************************

FN_SUPERVISOR_ENTRY_POINT SUBTRACT_VERTEX(half_float_half_false) 4 ""
FN_EXPORT SUBTRACT_VERTEX(half_float_half_true)
// vertex has no memory constraints
  setzi $sv_workerFunction, VERTEX(sub_half_float_half).kernel
  runall $sv_workerFunction, $vertexPtr, 0 // 6 cycles
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles

FN_SIZE SUBTRACT_VERTEX(half_float_half_false)

//*****************************************************************************
// Common code for half, float and mixed precision, supervisor vertices
// (ScaledAdd, ScaledSubtract, aXplusbY, aXMinusbY, etc.)
// Note that this code is called from other files as well.
//
// Divides the supervisor work among all workers and runs them.
//
// Each worker will be assigned to do a number 'N' of elements (with N a
// multiple of the 'atom' size, 2 or 4). The workers with ID = [0..remM1]
// will process an additional 'atom' of elements, and the worker ID = remM1
// will also do an extra 'final' number of elements (either 0,1 if atom=2, or
// 0,1,2,3 if atom=4).
// The partitioning parameters are stored on the supervisor stack in a state
// record which is common for all workers. Each worker will read the record,
// adjust its parameter base on its worker ID and process the element with a
// stride of 6.
//
// This code expects the following registers to have been set on entry:
//
// $m1 ($sv_count)          Total count (number of elements to process, among
//                          all workers).
//
// $m7 ($sv_log2AtomSize)   These two defines if we are processing 2 or 4
// $m8 ($sv_atomSizeMask)   elements at a time in the main worker loop:
//                          If atom=2 elements:
//                              $log2AtomSize = 1,  $atomSizeMask= 0x1
//                          If atom=4 elements:
//                              $log2AtomSize = 2,  $atomSizeMask= 0x3
//
// $m6 ($sv_workerFunction) Pointer to the worker function to run.
//
// $m2 ($sv_memConstraints) 1 or 0 to indicate if the A and B data is guranteed
//                          to be in different memory regions or not. It's not
//                          used here, just passed through to the workers.
//
// $m4 ($sv_scale)          The scale value scaleA (half or float) for the
//                          scaledAdd/Subtract vertices or both the scaleA and
//                          scaleB for the aXplusbY/aXminusbY/XminuxaXplusbY
//                          variants. It's not used here, just passed through
//                          to the workers.
//
// In addition to the above inputs, this code will read the pointers to the
// A and B data, using the macros VERTEX_DATA_A_OFFSET/VERTEX_DATA_B_OFFSET
//
//*****************************************************************************

DEF_STACK_SIZE_OWN  0  VERTEX(supervisor)
FN_SUPERVISOR_ENTRY_POINT VERTEX(supervisor) 4 "" NONE
  ld32  $mscratch2, $vertexPtr, $mzero, VERTEX_DATA_A_OFFSET/4
  ld32  $remM1, $vertexPtr, $mzero, VERTEX_DATA_B_OFFSET/4

  st32 $sv_memConstraints, $sp, $mzero, SV_STATE_MEM_CONSTRAINTS/4
  st32 $sv_scale, $sp, $mzero, SV_STATE_SCALES_OFFSET/4

  // transform the total count into remM1, final and count/6:
  //  where remM1 is the amount of workers (minus 1) that are required to
  //  process an extra atom size of elements, final is the non atom size
  //  remainder the final worker must process (when N is not divisible by the
  //  atoms size) and count is how many elements every worker processes

  // for the rest calculate n / 6 and n % 6 by reciprocal multiplcation
  //   n/6 = (n * 0xAAAB) >> 18
  //   n%6 = n - (n/6)*6
  // where n = count/atomSize
  // see recipe #1 for how these constants were derived:
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  setzi $mscratch, 0xAAAB
  // final = count % atomSize
  and   $final, $sv_count, $sv_atomSizeMask
  shr $sv_count, $sv_count, $sv_log2AtomSize

  // mscratch = n/6
  mul $mscratch, $sv_count, $mscratch // 6 cycles
  shr $mscratch, $mscratch, 18 // 6 cycles

  st32  $mscratch2, $sp, $mzero, SV_STATE_DATA_OFFSET/4
  st32  $remM1, $sp, $mzero, SV_STATE_DATA_B_OFFSET/4

  // rem = (count / atomSize) % numWorkers + ceil(final, atomSize)
  //  where ceil(x, y) = x / y + (x % y > 0);
  shr $remM1, $final, $sv_log2AtomSize

  // Avoid mscratch register bubble
  nop
  nop
  // mscratch2 = n%6
  mul $mscratch2, $mscratch, 6
  sub $mscratch2, $sv_count, $mscratch2 // 6 cycles

  // countPerWorker = (count / atomSize) / numWorkers * atomSize
  shl $sv_count, $mscratch, $sv_log2AtomSize

  and   $mscratch, $final, $sv_atomSizeMask
  cmpne $mscratch, $mscratch, $mzero // 6 cycles

  add $remM1, $remM1, $mscratch2
  add $remM1, $remM1, $mscratch // 6 cycles

  // when final is zero that means that the final worker can process an entire
  // block of elements. the easiest way to represent this is to add one to
  // remM1 (or just don't decrement it) in that case.
  //  cycles: 6 if final is zero, 7 if not.
  brz $final, 1f
  add $remM1, $remM1, -1 // 6 cycles
1:

  st32 $remM1, $sp, $mzero, SV_STATE_REMM1_OFFSET/4 // 6 cycles if final != 0
  st32 $final, $sp, $mzero, SV_STATE_FINAL_OFFSET/4
  st32 $sv_count, $sp, $mzero, SV_STATE_COUNT_OFFSET/4

  runall $sv_workerFunction, $sp, 0 // 6 cycles
  // restore the stack pointer that was changed in the supervisor common code.
  add  $sp, $sp, STACK_SIZE
  sync TEXCH_SYNCZONE_LOCAL // max(worker cycles) * 6

  br $lr // 6 cycles

FN_SIZE VERTEX(supervisor)

// clear supervisor variables
#undef vertexPtr
#undef dataPtr
#undef final
#undef remM1
#undef mscratch
#undef mscratch2

// worker variables

// integer variables
#define memConstraints m0
#define dataPtr m1
#define remM1 m2
#define final m3
#define countD2 m4
#define countD4 m4
#define dataBPtr m5
#define triPtr m6:7
#define triPtri0 m6
#define triPtri1 m7
#define workerIdM1 m8
#define stride m9

#define data a0:1

#define datab a0:1
#define dataa a2:3
#define databa a0:3
#define datab1 a4:5
#define dataa1 a6:7
#define datab1a1 a4:7

#define datai0 a0
#define datai1 a1
#define dataBHiLo a4:7
#define dataB a4:5
#define dataBHi a6:7
#define dataBi0 a4
#define dataBi1 a5
#define result a2:3
#define k a6

// scratch variables
#define mscratch m10
#define ascratch a7

# This macro checks if fast path can be executed. It can:
# - if Poplar requested constrained memory, or
# - Poplar did not request constrained memory, but
#   vectors are layed out in memory as if memory was constrained
# Here the memory is constrained if vectors A and B are in 
# different banks. This is necessary for ld2x64pace.
.macro CHOOSE_FAST_OR_SLOW_PATH SLOW_PATH_LABEL
  // The fast path is always OK if constraints were applied
  brnz $memConstraints, 1f
  // Or if the data start is far enough apart.  It could be ok in some other
  // circumstances but this is time consuming to check correctly.
  sub $memConstraints, $dataPtr, $dataBPtr
  abs $memConstraints, $memConstraints
  // +8 is to account for really wanting a <= instruction
  cmpult $memConstraints, $memConstraints, (2 * TMEM_ELEMSIZE) + 8
  brnz $memConstraints, \SLOW_PATH_LABEL
1:
.endm

# This macro checks if fast path can be executed (as above).
# Here the constrained memory means that vectors A and B are in 
# separate memory banks as well as vector A is in interleaved
# memory. This is necessary for ld2xst64pace.
.macro CHOOSE_FAST_OR_SLOW_PATH_INTERLEAVED SLOW_PATH_LABEL
  // The fast path is always OK if constraints were applied
  brnz $memConstraints, 1f

  // The 1st part of the test is identical to CHOOSE_FAST_OR_SLOW_PATH

  // Or if the data start is far enough apart.  It could be ok in some other
  // circumstances but this is time consuming to check correctly.
  sub $memConstraints, $dataPtr, $dataBPtr
  abs $memConstraints, $memConstraints
  // the pipeline is three loads deep and the offset between load/store
  // in ld2xst64pace is either 0x68 or 0xb8. We add 0xb8 which is the max of
  // the two offsets. We want to avoid a situation when the first store A 
  // would address the same segment as load B, which is possible if B is
  // placed at the end of segment N and A is placed at the end of segment N+1
  // +8 is to account for really wanting a <= instruction
  cmpult $memConstraints, $memConstraints, (2 * TMEM_ELEMSIZE) + 0xb8 + 8
  brnz $memConstraints, \SLOW_PATH_LABEL
  
  // so far we know A and B are far enough
  // now we need to check if A is also in interleaved memory   
  #if (__IPU_ARCH_VERSION__ > 1)
    shr $memConstraints, $dataPtr, 19 // address check from TMem_RegionId
  #else
    bri \SLOW_PATH_LABEL // for older architectures just do the slow path
  #endif
  brz $memConstraints, \SLOW_PATH_LABEL
1:
.endm


FN_WORKER_ENTRY_POINT VERTEX(float).kernel 8 nop
  {setzi $memConstraints, 0
   or    $datai0,$azero,FLOAT_1_0}
  bri 1f
VERTEX(float_true).kernel:
  {setzi $memConstraints, 1
   or    $datai0,$azero,FLOAT_1_0}
  bri 1f
VERTEX(sub_float_true).kernel:
  {setzi $memConstraints, 1
   or    $datai0,$azero,FLOAT_NEG_1_0}
  bri 1f

VERTEX(sub_float_false).kernel:
  {setzi $memConstraints, 0
   or    $datai0,$azero,FLOAT_NEG_1_0}
1:
  // load vertex state
  ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2

  // Fetch scale : a PTR128 that poitns to a float
  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHL_BITS
  ld32  $k, $dataPtr, $mzero, 0
3:
  {
    ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
    uput $FP_CLR, $ascratch
  }

  {
    get $workerIdM1, $WSR
    f32mul $k,$k,$datai0}
  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  DIVIDE_BY_WORKER $remM1, $workerIdM1, $mscratch, $countD2, LOG2_FLOAT_ATOM_SIZE

  // pack out pointers (in is never used).
  tapack $triPtr, $dataPtr, $dataBPtr, $mzero
  CHOOSE_FAST_OR_SLOW_PATH .Lfloat_slow_path
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1


  // offset each worker's pointer into the data to interleave them.
  // use $data as a temporary scratch register as we can't write to $azeros
  // twice in the same instruction.
  ld2x64pace $azeros, $data, $triPtr+=, $workerIdM1, 0b0101

  brz $countD2, .Lfloat_loop_epilogue

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // preload 4 values and fill the accumulators.
  ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD2, -1
    f32v2axpy $azeros, $dataB, $data
  }
  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {
    ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
    f32v2axpy $result, $azeros, $azeros
  }
  {
    st64step $result, $mzero, $dataPtr+=, $stride
    f32v2axpy $azeros, $dataB, $data
  }
2:
  // store the final 2 processed values.
  f32v2axpy $result, $azeros, $azeros
  st64step $result, $mzero, $dataPtr+=, $stride

.Lfloat_loop_epilogue:
  // Exit if no remainder
  and $mscratch, $remM1, 1
  brz $mscratch, .Lfloat_epilogue

  // unpack the data and dataB pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and $dataPtr, $triPtri0, $mscratch
  and $dataBPtr, $triPtri1, $mscratch
.Lfloat_epilogue_common:
  // There is a remainder to process if we get here.
  // Use the worker which has already got the same address as the remainder
  // to process it
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  andc $remM1, $remM1, 1
  ld32step $azero, $mzero, $mscratch+=, $remM1
  cmpeq $mscratch, $mscratch, $dataPtr
  brz $mscratch, .Lfloat_epilogue

  // scalar.
  {ld32 $datai0, $dataPtr, $mzero, 0
    // zero the top half of data and dataB so we can safely accumulate them
    zero $datai1}
  {ld32step $dataBi0, $mzero, $dataBPtr+=, 1
    zero $dataBi1}

  f32v2axpy $azeros, $dataB, $data
  f32v2axpy $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

.Lfloat_epilogue:
  exitz $mzero

.align 8
  nop // rpt align
// No assumptions made about operands being in different memory segments
.Lfloat_slow_path:
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  brz $countD2, .Lfloat_epilogue_slow

  mov $triPtri0, $dataPtr
  ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   // minus 1 because we pipeline the first value.
  {add $mscratch, $countD2, -1
   f32v2axpy $azeros, $dataB, $data}

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
   f32v2axpy $result, $azeros, $azeros}
  {ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   fnop}
  {st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS
   f32v2axpy $azeros, $dataB, $data}
2:
  f32v2axpy $result, $azeros, $azeros
  st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS
.Lfloat_epilogue_slow:
  // Exit if no remainder
  and  $mscratch, $remM1, 1
  brnz $mscratch, .Lfloat_epilogue_common
  exitz $mzero

FN_SIZE VERTEX(float).kernel

//******************************************************************************

FN_WORKER_ENTRY_POINT AXPLUSBY_VERTEX(float).kernel 8 nop

  // load vertex state
  {ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
   or    $datai0,$azero,FLOAT_1_0}
   bri   1f
AXPLUSBY_VERTEX(sub_float).kernel:
  // load vertex state
  {ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
   or    $datai0,$azero,FLOAT_NEG_1_0}
1:

  // Fetch scale : a PTR128 that points to a float
  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHL_BITS
  ld32  $k, $mzero, $dataPtr, 0

  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHL_BITS
  ld32  $ascratch, $mzero, $dataPtr, 0
  {ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
   f32mul $ascratch, $ascratch, $datai0}
  ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4

  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
  DIVIDE_BY_WORKER $remM1, $workerIdM1, $mscratch, $countD2, LOG2_FLOAT_ATOM_SIZE

  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  brz $countD2, .Lfloat_loop_epilogue_axpby

  ld64 $data, $mzero, $dataPtr, 0
  {ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   f32v2mul $data, $k:B, $data}
   // minus 1 because we pipeline the first value.
  {add $mscratch, $countD2, -1
   f32v2mul $dataB, $ascratch:B, $dataB}

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {ld64 $data, $mzero, $dataPtr, CTXT_WORKERS
   f32v2add $result, $data, $dataB}
  {ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   f32v2mul $data, $k:B, $data}
  {st64step $result, $mzero, $dataPtr+=, CTXT_WORKERS
   f32v2mul $dataB, $ascratch:B, $dataB}
2:
  f32v2add $result, $data, $dataB
  st64step $result, $mzero, $dataPtr+=, CTXT_WORKERS

.Lfloat_loop_epilogue_axpby:
  // Exit if no remainder
  and $mscratch, $remM1, 1
  brz $mscratch, .Lfloat_epilogue

  // Use the worker that points to the remainder to process it
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  andc $remM1, $remM1, 1
  ld32step $azero, $mzero, $mscratch+=, $remM1
  cmpeq $mscratch, $mscratch, $dataPtr
  brz $mscratch, .Lfloat_epilogue_axpby

  ld32    $datai0, $mzero, $dataPtr, 0
  {ld32   $dataBi0, $mzero, $dataBPtr, 0
   f32mul $datai0, $datai0, $k}
  f32mul  $dataBi0, $dataBi0, $ascratch
  f32add  $datai0, $dataBi0, $datai0
  st32    $datai0, $mzero, $dataPtr, 0
.Lfloat_epilogue_axpby:
  exitz $mzero

FN_SIZE AXPLUSBY_VERTEX(float).kernel
//******************************************************************************

FN_WORKER_ENTRY_POINT VERTEX(half_half_float_continue) 4
  // load k with a float scale, cast it and continue as normal
  {ld32  $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
   f16v2exp $ascratch, $azero}
   f32tof16 $k, $k
  {
    bri not_axplusby_continue
    // $k should have the form of {1, k}
    sort4x16lo $k, $ascratch, $k
  }
FN_SIZE VERTEX(half_half_float_continue)

FN_SECTION VERTEX(half).kernel
VERTEX(half).kernel:
  {
    ldb16 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/2
    f16v2exp $ascratch, $azero
  }

  {
    bri not_axplusby_continue
    // $k should have the form of {1, k}
    sort4x16lo $k, $ascratch, $k
  }
FN_SIZE VERTEX(half).kernel

FN_WORKER_ENTRY_POINT VERTEX(axplusby_half).kernel 8 nop
  ld32 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
not_axplusby_continue:
  ld32 $memConstraints, $mvertex_base, $mzero, SV_STATE_MEM_CONSTRAINTS/4

  get $workerIdM1, $WSR
   // load vertex state
  {
    ld32 $countD4, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
    uput $FP_CLR, $ascratch
  }
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4

  ld32 $dataPtr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET/4
  ld32 $dataBPtr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4

  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  // pack pointers
  // $dataBPtr is in EA[0] as it cannot be in EA[1] due to:
  // if (TMem_AddressIsExecutable(EA[1])) { 
  // EXCEPT(TEXCPT_INVALID_ADDR) }
  tapack $triPtr, $dataBPtr, $dataPtr, $dataPtr
  
  // process 4 at a time first as this is the optimal scenario
  shr $countD4, $countD4, 2

  CHOOSE_FAST_OR_SLOW_PATH_INTERLEAVED .Lhalf_slow_path


  // set strides
  // CTXT_WORKERS + 5 = 11 - used in fast loop (ld2xst64pace)
  // CTXT_WORKERS = 6 - used to process the tail
  setzi $stride, (CTXT_WORKERS + 5) * 1024 + CTXT_WORKERS

  // workers positioning - 64 bit apart (sufficient for tail processing)
  // using $dataa as temporary scratch as we cannot use $azeros twice
  ld2x64pace $azeros, $dataa, $triPtr+=, $workerIdM1, 0b0101
  tapack $triPtr, $triPtri0, $triPtri1, $triPtri1

  cmpult $mscratch, $countD4, 16
  brnz $mscratch, .Lcount_less_than_16

  // workers positining - 128 bits apart (extend from 64bits)
  // using $dataa as temporary scratch as we cannot use $azeros twice
  ld2x64pace $azeros, $dataa, $triPtr+=, $workerIdM1, 0b0101
  tapack $triPtr, $triPtri0, $triPtri1, $triPtri1

  add  $countD4, $countD4, -4
  shr $mscratch, $countD4, 1
  and $countD4, $countD4, 1 
 
  // the pipeline is 3 loads deep
  ld2x64pace $datab, $dataa, $triPtr+=, $stride, 0
  {
    ld2x64pace $datab1, $dataa1, $triPtr+=, $stride, 0b1010
    f16v4mix $azeros, $dataa, $datab
  }
  {
    ld2x64pace $datab, $dataa, $triPtr+=, $stride, 0b0
    f16v4mix $dataa1, $dataa1, $datab1
  }

  rpt $mscratch, (2f - 1f)/8 - 1
1:
  {
    ld2xst64pace $datab1a1, $dataa1, $triPtr+=, $stride, 0b001010
    f16v4mix $dataa, $dataa, $datab
  }
  {
    ld2xst64pace $databa, $dataa, $triPtr+=, $stride, 0b100000
    f16v4mix $dataa1, $dataa1, $datab1
  }

2:
  {
    ld2xst64pace $datab1a1, $dataa1, $triPtr+=, $stride, 0b001010
    f16v4mix $dataa, $dataa, $datab
  }
  {
    st64pace $dataa, $triPtr+=, $stride, 0b10
    f16v4mix $dataa1, $dataa1, $datab1
  }
  {
    st64pace $dataa1, $triPtr+=, $stride, 0b0
    f16v4mix $dataa, $azeros, $azeros
  }
  st64pace $dataa, $triPtr+=, $stride, 0b10

  // workers are 128 bits apart,
  // we need to bring them back (64bit apart)
  mul $mscratch, $workerIdM1, -1
  ld2x64pace $datab, $dataa, $triPtr+=, $mscratch, 0b0101
  tapack $triPtr, $triPtri0, $triPtri1, $triPtri1
 
.Lcount_less_than_16:
  // less than 64 if fast loop above has not been executed
  // less than 8 if fast loop executed
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD4, $countD4, $mscratch

  brz $countD4, .Lhalf_loop_epilogue

  // preload 4 values and fill the accumulators.
  ld2x64pace $datab, $dataa, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD4, -1
    f16v4mix $azeros, $dataa, $datab
  }
  {
    rpt $mscratch, (2f - 1f) / 8 - 1
    fnop
  }
1:
  {
    ld2x64pace $datab, $dataa, $triPtr+=, $stride, 0b0101
    f16v4mix $dataa1, $azeros, $azeros
  }
  {
    st64pace $dataa1, $triPtr+=, $stride, 0b01
    f16v4mix $azeros, $dataa, $datab
  }
2:
  // store the final 4 processed values.
  f16v4mix $result, $azeros, $azeros
  st64pace $result, $triPtr+=, $stride, 0b01

.Lhalf_loop_epilogue:
  // at most one of our workers will have to do the remaining elements. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 3 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_epilogue
  brz $final, .Lhalf_epilogue

  // unpack the data and dataB pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and $dataBPtr, $triPtri0, $mscratch
  and $dataPtr, $triPtri1, $mscratch
.Lhalf_epilogue_common:
  {
    // is there at least 2 left?
    cmpult $mscratch, $final, 2
    // zero the top half of data and dataB so we can safely accumulate them
    // for the x2 and x1 cases.
    zero $datai1
  }
  {
    brnz $mscratch, .Lhalf_scalar
    zero $dataBi1
  }

  // remainder 2
  ld32 $datai0, $dataPtr, $mzero, 0
  ld32step $dataBi0, $mzero, $dataBPtr+=, 1

  f16v4mix $azeros, $data, $dataB
  f16v4mix $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

  // finish now if that's all.
  cmpeq $mscratch, $final, 2
  brnz $mscratch, .Lhalf_epilogue

.Lhalf_scalar:
  ldb16 $datai0, $dataPtr, $mzero, 0
  ldb16 $dataBi0, $dataBPtr, $mzero, 0

  f16v4mix $azeros, $data, $dataB

  {
    // load the last word and perform a read/modify/write.
    ld32 $ascratch, $dataPtr, $mzero, 0
    f16v4mix $data, $azeros, $azeros
  }

  sort4x16hi $ascratch, $datai0, $ascratch
  st32 $ascratch, $dataPtr, $mzero, 0

.Lhalf_epilogue:
  exitz $mzero
.align 8
  nop // rpt align
// No assumptions made about operands being in different memory segments
.Lhalf_slow_path:
  // if worker id is less than the remainder, this worker will process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD2, $countD2, $mscratch

 // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  brz $countD2, .Lhalf_loop_epilogue_slow

  mov $triPtri0, $dataPtr
  ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   // minus 1 because we pipeline the first value.
  {add $mscratch, $countD2, -1
   f16v4mix $azeros, $data, $dataB}

  rpt $mscratch, (2f - 1f) / 8 - 1
1:
  {ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
   f16v4mix $result, $azeros, $azeros}
  {ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   fnop}
  {st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS
   f16v4mix $azeros, $data, $dataB}
2:
  f16v4mix $result, $azeros, $azeros
  st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS

.Lhalf_loop_epilogue_slow:
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_epilogue
  brnz $final, .Lhalf_epilogue_common
  exitz $mzero

FN_SIZE VERTEX(axplusby_half).kernel

//******************************************************************************
FN_WORKER_ENTRY_POINT VERTEX(xminusaxplusby_half).kernel 8
  ld32 $k, $mvertex_base, $mzero, SV_STATE_SCALES_OFFSET/4
not_xminusaxplusby_continue:
  ld32 $memConstraints, $mvertex_base, $mzero, SV_STATE_MEM_CONSTRAINTS/4

  get $workerIdM1, $WSR
   // load vertex state
  {
    ld32 $countD4, $mvertex_base, $mzero, SV_STATE_COUNT_OFFSET/4
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $remM1, $mvertex_base, $mzero, SV_STATE_REMM1_OFFSET/4
    uput $FP_CLR, $ascratch
  }
  ld32 $final, $mvertex_base, $mzero, SV_STATE_FINAL_OFFSET/4

  ld32 $dataPtr, $mvertex_base, $mzero, SV_STATE_DATA_OFFSET/4
  ld32 $dataBPtr, $mvertex_base, $mzero, SV_STATE_DATA_B_OFFSET/4

  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $k
  }
  // pack out points (in is never used).
  tapack $triPtr, $dataPtr, $dataBPtr, $mzero

  // process 4 at a time first as this is the optimal scenario
  shr $countD4, $countD4, 2

  // if worker id is less than the remainder this worker can process an extra 4.
  cmpslt $mscratch, $workerIdM1, $remM1
  add $countD4, $countD4, $mscratch

  CHOOSE_FAST_OR_SLOW_PATH .Lhalf_xminusaxplusby_slow_path
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1

  // offset each worker's pointer into the data to interleave them.
  // use $data as a temporary scratch register as we can't write to $azeros
  // twice in the same instruction.
  ld2x64pace $azeros, $data, $triPtr+=, $workerIdM1, 0b0101

  brz $countD4, .Lhalf_xminusaxplusby_loop_epilogue

  // each worker's data is interleaved so set a stride of how many workers
  // we have.
  setzi $stride, CTXT_WORKERS

  // load the first values and push them into the accumulators.
  ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
  {
    // minus 1 from our count because of the preloading above.
    add $mscratch, $countD4, -1
    f16v4mix $azeros, $data, $dataB
  }

  brz $mscratch, .LFast_one_remaining

  {
    // Load second pair of inputs X and Y
    ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
    // Add previous X input to the accumulator
    f16v4acc $data
  }
  {
    // Decrement loop count due to depth-2 pipelining
    add $mscratch, $mscratch, -1
    // Obtain first result, process previous inputs
    f16v4mix $result, $data, $dataB
  }
  rpt $mscratch, (2f-1f)/8-1
1:
  {
    // Load the next inputs
    ld2x64pace $data, $dataB, $triPtr+=, $stride, 0b0101
    // Add previous X input to the accumulator
    f16v4acc $data
  }
  {
    // Store the current result
    st64step $result, $mzero, $dataPtr+=, $stride
    // Obtain result for previous inputs and process the current inputs
    f16v4mix $result, $data, $dataB
  }
2:
  // Store the last-but-one result
  st64step $result, $mzero, $dataPtr+=, $stride
.LFast_one_remaining:
  // Finish processing and store the final result
  f16v4acc $data
  f16v4mix $result, $azeros, $azeros
  st64step $result, $mzero, $dataPtr+=, $stride

.Lhalf_xminusaxplusby_loop_epilogue:
  // at most one of our workers will have to do the remaining elements. this
  // worker id is equal to the $rem value in the vertex state. the amount
  // of elements remaining is the $final value. $final will be 3 at most.
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_xminusaxplusby_epilogue
  brz $final, .Lhalf_xminusaxplusby_epilogue

  // unpack the data and dataB pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and $dataPtr, $triPtri0, $mscratch
  and $dataBPtr, $triPtri1, $mscratch
.Lhalf_xminusaxplusby_epilogue_common:
  {
    // is there at least 2 left?
    cmpult $mscratch, $final, 2
    // zero the top half of data and dataB so we can safely accumulate them
    // for the x2 and x1 cases.
    zero $datai1
  }
  {
    brnz $mscratch, .Lhalf_xminusaxplusby_scalar
    zero $dataBi1
  }

  // remainder 2
  ld32 $datai0, $dataPtr, $mzero, 0
  ld32step $dataBi0, $mzero, $dataBPtr+=, 1

  f16v4mix $azeros, $data, $dataB
  f16v4acc $data
  f16v4mix $data, $azeros, $azeros

  st32step $datai0, $mzero, $dataPtr+=, 1

  // finish now if that's all.
  cmpeq $mscratch, $final, 2
  brnz $mscratch, .Lhalf_xminusaxplusby_epilogue

.Lhalf_xminusaxplusby_scalar:
  ldb16 $datai0, $dataPtr, $mzero, 0
  ldb16 $dataBi0, $dataBPtr, $mzero, 0

  f16v4mix $azeros, $data, $dataB
  f16v4acc $data
  {
    // load the last word and perform a read/modify/write.
    ld32 $ascratch, $dataPtr, $mzero, 0
    f16v4mix $data, $azeros, $azeros
  }

  sort4x16hi $ascratch, $datai0, $ascratch
  st32 $ascratch, $dataPtr, $mzero, 0

.Lhalf_xminusaxplusby_epilogue:
  exitz $mzero
.align 8
  nop // rpt align
// No assumptions made about operands being in different memory segments
.Lhalf_xminusaxplusby_slow_path:
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  brz $countD2, .Lhalf_xminusaxplusby_loop_epilogue_slow

  mov $triPtri0, $dataPtr
  ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS

  // minus 1 because we pipeline the first value.
  {
    add $mscratch, $countD2, -1
    f16v4mix $azeros, $data, $dataB
  }
  rpt $mscratch, (2f-1f)/8-1
1:
  {
    // Load new X input
    ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
    // Add previous X input to the accumulator
    f16v4acc $data
  }
  {
    // Load new Y input
    ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
    // Obtain the result from the accumulator for the previous input
    f16v4mix $result, $azeros, $azeros
  }
  {
    // Store the result for the previous input
    st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS
    // Perform -aX + bY for current inputs
    f16v4mix $result, $data, $dataB
  }
2:
  // Process and store final result
  f16v4acc $data
  f16v4mix $result, $azeros, $azeros
  st64step $result, $mzero, $triPtri0+=, CTXT_WORKERS

.Lhalf_xminusaxplusby_loop_epilogue_slow:
  cmpeq $mscratch, $workerIdM1, $remM1
  brz $mscratch, .Lhalf_xminusaxplusby_epilogue
  brnz $final, .Lhalf_xminusaxplusby_epilogue_common
  exitz $mzero

FN_SIZE VERTEX(xminusaxplusby_half).kernel

//******************************************************************************
// variant that accepts half data and float dataB.
// The implementation is common for float and half BScale; there are no memory
// constraints.
//******************************************************************************

#define aA0123  a0:1  //f16v4
#define aA01    a0    //f16v2
#define aBScale a2    //f32v1
#define aTmp    a3
#define aB0123  a4:7  //f32v4
#define aB01    a4:5  //f32v2
#define aB23    a6:7  //f32v2, also used as f16v4
#define aB0     a4    //f32v1
#define aB1     a5    //f32v1
#define aB01h   a4

FN_WORKER_ENTRY_POINT VERTEX(half_float_half).kernel 8 nop
  ldz16  $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl    $mscratch, $mscratch, SCALED_PTR128_SHL_BITS
  {ldb16 $aBScale, $mzero, $mscratch, 0
   or    $aTmp,$azero,FLOAT_1_0}
  {bri 1f
   f16tof32 $aBScale, $aBScale
  }
FN_EXPORT VERTEX(sub_half_float_half).kernel
  ldz16  $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl    $mscratch, $mscratch, SCALED_PTR128_SHL_BITS
  {ldb16 $aBScale, $mzero, $mscratch, 0
   or    $aTmp,$azero,FLOAT_NEG_1_0}
  {bri 1f
   f16tof32 $aBScale, $aBScale
  }

// This core handles a = a+b*bScale, where a is half and b and bScale are float.
// The multiplies are done in fp32 precision to retain accuracy, the add is
// done in fp16 precision. So a slightly more accurate version is possible which
// uses fp16 for the addition - but that hasn't been deemed necessary so far.
// This is a 4 cycle inner loop, so in this case there is no benefit from
// adding any constraints to the Tensors.
FN_EXPORT VERTEX(half_float_float).kernel
  // load vertex state
  ldz16 $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $mscratch, $mscratch, SCALED_PTR128_SHL_BITS
  {ld32  $aBScale, $mzero, $mscratch, 0
   or    $aTmp,$azero,FLOAT_1_0}
1:
  {ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
    f32mul $aBScale, $aBScale, $aTmp}

  ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4

  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  DIVIDE_BY_WORKER $remM1, $workerIdM1, $mscratch, $countD4, LOG2_HALF_ATOM_SIZE

  // advance dataptr(halves) by workerId * (workers * 8 bytes)
  // advance dataBptr(floats) by workerId * (workers * 16 bytes)
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1

  brz $countD4, .Lhalf_loop_epilogue4

  // Unrolling
  ld64        $aB23,    $mzero,     $dataBPtr, 1
  {ld64step   $aB01,    $mzero,     $dataBPtr+=, CTXT_WORKERS*2
   f32v2mul   $aB23,    $aBScale:B, $aB23
  }
  // minus 1 because of the pipeline

  add $countD4, $countD4, -1
  rpt $countD4, (2f-1f)/8-1
1:
  {ld64       $aA0123,  $mzero,     $dataPtr, 0
   f32v2mul   $aB01,    $aBScale:B, $aB01
  }
  {ld64       $aB23,    $mzero,     $dataBPtr, 1
   f32v4tof16 $aB01,    $aB0123
  }
  {ld64step   $aB01,    $mzero,     $dataBPtr+=, CTXT_WORKERS*2
   f16v4add   $aA0123,  $aA0123,    $aB01
  }
  {st64step   $aA0123,  $mzero,     $dataPtr+=, CTXT_WORKERS
   f32v2mul   $aB23,    $aBScale:B, $aB23
  }
2:
  {ld64       $aA0123,  $mzero,     $dataPtr, 0
   f32v2mul   $aB01,    $aBScale:B, $aB01
  }
  f32v4tof16  $aB01,    $aB0123
  f16v4add    $aA0123,  $aA0123,    $aB01
  st64step    $aA0123,  $mzero,     $dataPtr+=, CTXT_WORKERS
  // All full/4 vectors have now been processed and stored.


.Lhalf_loop_epilogue4:
  // Use the worker which poitns to the remainder to process it
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  andc $final, $remM1, 3
  ldb16step $azero, $mzero, $mscratch+=, $final
  cmpeq $mscratch, $mscratch, $dataPtr
  brz $mscratch, .Lhalf_float_float_exit

  // are there at least 2 left?
  and $mscratch, $remM1, 2
  brz $mscratch, .Lhalf_final_element

  // process next 32bit of result
  ld64step    $aB01,    $mzero,     $dataBPtr+=, 1
  {ld32       $aA01,    $mzero,     $dataPtr, 0
   f32v2mul   $aB01,    $aBScale:B, $aB01
  }
  f32v2tof16  $aB01h,   $aB01
  f16v2add    $aA01,    $aA01,      $aB01h
  st32step    $aA01,    $mzero,     $dataPtr+=, 1

.Lhalf_final_element:
  // how many left do we have? maximum of 1.
  and $remM1, $remM1,   0x1
  brz $remM1, .Lhalf_float_float_exit
  ld32        $aB0,     $mzero,     $dataBPtr, 0 // [ MShalf  | LShalf]
  {ld32       $aA01,    $mzero,     $dataPtr, 0  // [ overrun | src]
   f32mul     $aB0,     $aBScale,   $aB0
  }
  f32tof16   $aB01h,   $aB0                      // [ addend | addend ]
  sort4x16lo  $aTmp,    $azero,     $aA01        // [ value   | 0]
  f16v2add    $aTmp,    $aTmp,      $aB01h       // [ result  | addend]
  sort4x16hi  $aTmp,    $aTmp,      $aA01        // [ overrun | result]
  st32        $aTmp,    $mzero,     $dataPtr, 0

.Lhalf_float_float_exit:
  exitz $mzero

FN_SIZE VERTEX(half_float_half).kernel

#endif // __IPU__
