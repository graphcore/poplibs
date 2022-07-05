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

// worker variables

// integer variables
#define dataPtr m1
#define nElem m2
#define remM1 m2
#define final m3
#define rem   m3
#define countD2 m4
#define countD4 m4
#define dataBPtr m5
#define triPtr m6:7
#define triPtri0 m6
#define triPtri1 m7
#define workerIdM1 m8
#define stride m9
//#define memConstraints m11 // Note memConstraints - defined/shared between files

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


FN_WORKER_ENTRY_POINT VERTEX(float_float_float_false) 8 nop
  {setzi $memConstraints, 0
   or    $datai0,$azero,FLOAT_1_0}
  bri 1f
FN_EXPORT VERTEX(float_float_float_true)
  {setzi $memConstraints, 1
   or    $datai0,$azero,FLOAT_1_0}
  bri 1f
FN_EXPORT SUBTRACT_VERTEX(float_float_float_true)
  {setzi $memConstraints, 1
   or    $datai0,$azero,FLOAT_NEG_1_0}
  bri 1f

FN_EXPORT SUBTRACT_VERTEX(float_float_float_false)
  {setzi $memConstraints, 0
   or    $datai0,$azero,FLOAT_NEG_1_0}
1:
  // load vertex state
  ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2

  // Fetch scale : a PTR128 that points to a float
  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHIFTS
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

FN_WORKER_ENTRY_POINT AXPLUSBY_VERTEX(float_float_false) 8 nop

  // load vertex state
  {ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
   or    $datai0,$azero,FLOAT_1_0}
   bri   1f
FN_EXPORT AX_MINUS_BY_VERTEX(float_float_false)
  // load vertex state
  {ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
   or    $datai0,$azero,FLOAT_NEG_1_0}
1:

  // Fetch scale : a PTR128 that points to a float
  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHIFTS
  ld32  $k, $mzero, $dataPtr, 0

  ldz16 $dataPtr, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET/2
  shl   $dataPtr, $dataPtr, SCALED_PTR128_SHIFTS
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

FN_SECTION VERTEX(half).kernel
FN_EXPORT SUBTRACT_VERTEX(half_half_half_true)
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
  {bri 1f
   or $datai0, $azero, HALF_NEG_1_0<<16}
FN_EXPORT SUBTRACT_VERTEX(half_half_half_false)
  setzi $memConstraints, 0
  {bri 1f
   or $datai0, $azero, HALF_NEG_1_0<<16}
FN_EXPORT VERTEX(half_half_half_true)
  setzi $memConstraints, MEM_CONSTRAINTS_MASK
  {bri 1f
  or $datai0, $azero, HALF_1_0<<16}
FN_EXPORT VERTEX(half_half_half_false)
  {setzi $memConstraints, 0
   or $datai0, $azero, HALF_1_0<<16}
1:
  {
    ldz16 $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
    f16v2exp $ascratch, $azero
  }
  shl    $mscratch, $mscratch, SCALED_PTR128_SHIFTS
  ldb16  $k, $mscratch, $mzero, 0
  f16v2mul $k, $datai0:BU, $k
  // $k should have the form of {1, k}
  {bri half_half_half_scales_in_k_continue
  sort4x16lo $k, $ascratch, $k}

FN_SIZE VERTEX(half).kernel

FN_WORKER_ENTRY_POINT AXPLUSBY_VERTEX(half_half_false) 8
  {setzi $memConstraints, 0
   or $datai0, $azero, HALF_1_0<<16}
  bri   1f

FN_EXPORT AXPLUSBY_VERTEX(half_half_true)
  {setzi $memConstraints, MEM_CONSTRAINTS_MASK
   or $datai0, $azero, HALF_1_0<<16}
  bri 1f
FN_EXPORT AX_MINUS_BY_VERTEX(half_half_false)
  {setzi $memConstraints, 0
   or $datai0, $azero, HALF_NEG_1_0<<16}
  bri   1f
FN_EXPORT AX_MINUS_BY_VERTEX(half_half_true)
  {setzi $memConstraints, MEM_CONSTRAINTS_MASK
   or $datai0, $azero, HALF_NEG_1_0<<16}
1:
  ldz16  $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET/2
  shl    $mscratch, $mscratch, SCALED_PTR128_SHIFTS
  ldb16  $k, $mscratch, $mzero, 0

  {ldz16  $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
   f16v2mul $k, $datai0:BU, $k}

  shl    $mscratch, $mscratch, SCALED_PTR128_SHIFTS
  ldb16  $ascratch, $mscratch, $mzero, 0
  sort4x16lo $k, $ascratch, $k

// Entry point, where a float scale is found to be accurately represented in
// half and so execution is faster
// Either {1, scaleB} or {scaleA, scaleB} as required must be in $TAS
FN_EXPORT half_half_half_scales_in_k_continue

  get $workerIdM1, $WSR
   // load vertex state
  {
    ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
    uput $FP_CLR, $ascratch
  }

  ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4

  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f16v4axpby instructions below.
    uput $TAS, $k
  }
  // pack pointers
  // $dataBPtr is in EA[0] as it cannot be in EA[1] due to:
  // if (TMem_AddressIsExecutable(EA[1])) {
  // EXCEPT(TEXCPT_INVALID_ADDR) }
  tapack $triPtr, $dataBPtr, $dataPtr, $dataPtr

  CHOOSE_FAST_OR_SLOW_PATH_INTERLEAVED .Lhalf_slow_path

  // Divide by 24: 4 elements per worker per loop pass, where the 1st
  // loop processes 8 elements. Ignore the remainder for now as we need each
  // worker to do the same work so that the worker spacing by 128 bits and then
  // back to 64 bits works correctly
  SPLIT_BETWEEN_WORKERS $remM1 $countD4 $final 24
  // set strides
  // CTXT_WORKERS + 5 = 11 - used in fast loop (ld2xst64pace)
  // CTXT_WORKERS = 6 - used to process the tail
  setzi $stride, (CTXT_WORKERS + 5) * 1024 + CTXT_WORKERS

  // workers positioning - 64 bit apart (sufficient for tail processing)
  // using $dataa as temporary scratch as we cannot use $azeros twice
  ld2x64pace $azeros, $dataa, $triPtr+=, $workerIdM1, 0b0101
  tapack $triPtr, $triPtri0, $triPtri1, $triPtri1

  cmpult $mscratch, $countD4, 4
  brnz $mscratch, .Lcount_less_than_16

  // workers positioning - 128 bits apart (extend from 64bits)
  // using $dataa as temporary scratch as we cannot use $azeros twice
  ld2x64pace $azeros, $dataa, $triPtr+=, $workerIdM1, 0b0101
  tapack $triPtr, $triPtri0, $triPtri1, $triPtri1

   add  $countD4, $countD4, -4
   shr  $mscratch, $countD4, 1
   and  $countD4, $countD4, 1

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
  // less than 16 if fast loop above has not been executed
  // less than 8 if fast loop executed
  shr    $mscratch, $final, 2
  cmpslt $mscratch, $workerIdM1, $mscratch
  add $countD4, $countD4, $mscratch

  brz $countD4, .Lhalf_loop_epilogue

  // preload 4 values and fill the accumulators.
  ld2x64pace $datab, $dataa, $triPtr+=, $stride, 0b0101
  {
    // minus 1 because we pipeline the first value.
    add $mscratch, $countD4, -1
    f16v4mix $azeros, $dataa, $datab
  }
  rpt $mscratch, (2f - 1f) / 8 - 1
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
  // unpack the data and dataB pointers from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  and $dataBPtr, $triPtri0, $mscratch
  and $dataPtr, $triPtri1, $mscratch
.Lhalf_epilogue_common:
  andc $final, $remM1, 3

  and  $remM1, $remM1, 3
  brz $remM1, .Lhalf_epilogue


  // Use the worker which is pointing to the remainder to process it
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  ldb16step $azero, $mzero, $mscratch+=, $final
  cmpeq $mscratch, $mscratch, $dataPtr
  brz   $mscratch, .Lhalf_epilogue
  {
    // is there at least 2 left?
    cmpult $mscratch, $remM1, 2
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
  cmpeq $mscratch, $remM1, 2
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
  DIVIDE_BY_WORKER $remM1, $workerIdM1, $mscratch, $countD4, LOG2_HALF_ATOM_SIZE

 // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  brz $countD4, .Lhalf_epilogue_common

  mov $triPtri0, $dataPtr
  ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS
   // minus 1 because we pipeline the first value.
  {add $mscratch, $countD4, -1
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
  bri .Lhalf_epilogue_common

FN_SIZE VERTEX(axplusby_half).kernel

//******************************************************************************
FN_WORKER_ENTRY_POINT XMINUSAXPLUSBY_VERTEX(half_false) 8 nop
  setzi   $memConstraints, 0
  bri     1f
FN_EXPORT XMINUSAXPLUSBY_VERTEX(half_true)
  setzi   $memConstraints, MEM_CONSTRAINTS_MASK
1:
  // -1 * scale, scaleB
  ldz16 $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  {shl   $mscratch, $mscratch, SCALED_PTR128_SHIFTS
   or $ascratch, $azero, HALF_NEG_1_0<<16}
  ldb16 $k, $mzero, $mscratch, 0

  {ldz16 $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET/2
   f16v2mul $k, $ascratch:BU, $k}
  shl   $mscratch, $mscratch, SCALED_PTR128_SHIFTS
  ldb16 $ascratch, $mzero, $mscratch, 0
  {get $workerIdM1, $WSR
   sort4x16lo $k, $k, $ascratch}

   // load vertex state
  {
    ldz16 $remM1, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
    setzi $ascratch, ZAACC_BITMASK
  }
  {
    ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
    uput $FP_CLR, $ascratch
  }
  ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4

  {
    and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK
    // setup $TAS for the f16v4mix instructions below.
    uput $TAS, $k
  }

  DIVIDE_BY_WORKER $remM1, $workerIdM1, $mscratch, $countD4, LOG2_HALF_ATOM_SIZE

  CHOOSE_FAST_OR_SLOW_PATH .Lhalf_xminusaxplusby_slow_path
  // offset each worker's pointer into the data to interleave them.
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1

  brz $countD4, .Lhalf_xminusaxplusby_loop_epilogue

  // pack out points (in is never used).
  tapack $triPtr, $dataPtr, $dataBPtr, $mzero

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
  // unpack the dataB pointer from our triPtr.
  ldconst $mscratch, TMEM_FULL_ADDRESS_MASK
  {and $dataBPtr, $triPtri1, $mscratch
   f16v4acc $data}
  f16v4mix $result, $azeros, $azeros
  st64step $result, $mzero, $dataPtr+=, $stride

.Lhalf_xminusaxplusby_loop_epilogue:
  // Use the worker which is pointing to the remainder to process it.
  // Up to 3 elements remain
  andc $final, $remM1, 3
  and $remM1, $remM1, 3
  brz $remM1, .Lhalf_xminusaxplusby_epilogue

  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  ldb16step $azero, $mzero, $mscratch+=, $final
  cmpeq $mscratch, $dataPtr, $mscratch
  brz $mscratch, .Lhalf_xminusaxplusby_epilogue

.Lhalf_xminusaxplusby_epilogue_common:
  {
    // is there at least 2 left?
    cmpult $mscratch, $remM1, 2
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
  cmpeq $mscratch, $remM1, 2
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
  brz $countD4, .Lhalf_xminusaxplusby_loop_epilogue

  mov $triPtri0, $dataPtr
  ld64step $data, $mzero, $dataPtr+=, CTXT_WORKERS
  ld64step $dataB, $mzero, $dataBPtr+=, CTXT_WORKERS

  // minus 1 because we pipeline the first value.
  {
    add $mscratch, $countD4, -1
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
  bri .Lhalf_xminusaxplusby_loop_epilogue

FN_SIZE VERTEX(xminusaxplusby_half).kernel

//******************************************************************************
// variant that accepts half data and float dataB.
// The implementation is common for float and half BScale; there are no memory
// constraints.
//******************************************************************************

#define aA0123  a0:1  //f16v4
#define aA01    a0    //f16v2
#define aBScale a2    //f32v1
#define aB0123  a4:7  //f32v4
#define aB01    a4:5  //f32v2
#define aB23    a6:7  //f32v2, also used as f16v4
#define aB0     a4    //f32v1
#define aB1     a5    //f32v1
#define aB01h   a4

FN_WORKER_ENTRY_POINT VERTEX(half_float_half_false) 8
  ldz16  $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl    $mscratch, $mscratch, SCALED_PTR128_SHIFTS
  ldb16 $aBScale, $mzero, $mscratch, 0
  {bri 1f
   f16tof32 $aBScale, $aBScale
  }
FN_EXPORT SUBTRACT_VERTEX(half_float_half_false)
FN_EXPORT SUBTRACT_VERTEX(half_float_half_true)
// vertex has no memory constraints

  ldz16  $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl    $mscratch, $mscratch, SCALED_PTR128_SHIFTS
  ldb16 $aBScale, $mzero, $mscratch, 0
   f16tof32 $aBScale, $aBScale
  {bri 1f
   f32sub $aBScale, $azero, $aBScale  
  }
  
// This core handles a = a+b*bScale, where a is half and b and bScale are float.
// The multiplies are done in fp32 precision to retain accuracy, the add is
// done in fp16 precision. So a slightly more accurate version is possible which
// uses fp16 for the addition - but that hasn't been deemed necessary so far.
// This is a 4 cycle inner loop, so in this case there is no benefit from
// adding any constraints to the Tensors.
FN_EXPORT VERTEX(half_float_float_false)
  // load vertex state
  ldz16 $mscratch, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET/2
  shl   $mscratch, $mscratch, SCALED_PTR128_SHIFTS
  ld32  $aBScale, $mzero, $mscratch, 0

1:

  // Clear accumulators
 {ldz16 $nElem, $mvertex_base, $mzero, VERTEX_PACKED_COUNT_OFFSET/2
  // Load scale ready for f32v2axpy
  uput $TAS, $aBScale}
  
 {ld32 $dataPtr, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  setzi $a0, ZAACC_BITMASK}
 {ld32 $dataBPtr, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET/4
  uput $FP_CLR, $a0}

  get $workerIdM1, $WSR
  and $workerIdM1, $workerIdM1, CSR_W_WSR__CTXTID_M1__MASK

  DIVIDE_BY_WORKER $nElem, $workerIdM1, $mscratch, $countD4, LOG2_HALF_ATOM_SIZE

  // advance dataptr(halves) by workerId * (workers * 8 bytes)
  // advance dataBptr(floats) by workerId * (workers * 16 bytes)
  ld64step $azeros, $mzero, $dataPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1
  ld64step $azeros, $mzero, $dataBPtr+=, $workerIdM1

  brz $countD4, .Lhalf_loop_epilogue4

// aux register aliases
#define aA01f   a0:1
#define aA0123h             a6:7
#define aA01h               a6
#define aA23h                 a7

#define aB01f       a2:3
#define aB0f        a2
#define aB1f          a3
#define aB23f           a4:5

#define aTmpfA          a4:5
#define aTmpf       a2:3

#define aR01f   a0:1
#define aR0f    a0
#define aR0123h a0:1
#define aR01h   a0 
#define aR23f       a2:3
#define aR0123f a0:3
#define aR23h     a1

  // The loop is unrolled by 4 elements.
 add $countD4, $countD4, -1
  
  // warmup
  # A registers                                                                 // a0 a1 a2 a3 a4 a5 a6 a7
  ld64            $aA0123h, $mzero,   $dataPtr, 0                               //                   ^^ ^^ # 
 {ld64step        $aB01f,   $mzero,   $dataBPtr+=, 1                            //       ^^ ^^             # 
    f16v2tof32    $aTmpfA,  $aA01h}                                             //             ^^ ^^ vv    # 
 {ld32            $aA01h,   $mzero,   $dataPtr, 2*CTXT_WORKERS                  //                   ^^    #                      
    f32v2axpy     $azeros,  $aB01f,   $aTmpfA}                                  //       vv vv vv vv       #
 {ld64step        $aB23f,   $mzero,   $dataBPtr+=, 2*CTXT_WORKERS-1             //             ^^ ^^       # 
    f16v2tof32    $aTmpf,   $aA23h}                                             //       ^^ ^^          vv # 

 // loop
 {rpt $countD4, (2f-1f)/8-1
    f32v2axpy     $aR01f,     $aB23f,   $aTmpf}                                 // ^^ ^^ vv vv vv vv       #
1:
  # A registers                                                                 // a0 a1 a2 a3 a4 a5 a6 a7 

 {ld64step        $aB01f,   $mzero,   $dataBPtr+=, 1                            //       ^^ ^^             #
    f16v2tof32    $aTmpfA,  $aA01h}                                             //             ^^ ^^ vv    #
 {ld32            $aA23h,   $mzero,   $dataPtr, 2*CTXT_WORKERS+1                //                      ^^ #                                           
    f32v2axpy     $aR23f,   $aB01f,   $aTmpfA}                                  //       ++ ++ vv vv       #
 {ld64step        $aB23f,   $mzero,   $dataBPtr+=, 2*CTXT_WORKERS-1             //             ^^ ^^       #
    f32v4tof16    $aR0123h, $aR0123f}                                           // ++ ++ vv vv             # 
 {st64step        $aR0123h, $mzero,   $dataPtr+=, CTXT_WORKERS                  // vv vv                   #  
    f16v2tof32    $aTmpf,   $aA23h}                                             //       ^^ ^^          vv #   
 {ld32            $aA01h,   $mzero,   $dataPtr, 2*CTXT_WORKERS                    //                   ^^    #       
    f32v2axpy     $aR01f,   $aB23f,   $aTmpf}                                   // ^^ ^^ vv vv vv vv       #            
2:
  // flush
    f32v2tof16    $aR01h,   $aR01f                                             // vv vv             ^^                   
 {and             $rem,     $nElem,   3 // any remainder?
    f16v2gina     $aR23h,   $azero,   0}                                        //                      ^^             
  st64step        $aR0123h, $mzero,   $dataPtr+=, CTXT_WORKERS                  //                   vv vv
  // Early finish for the common case of no remainder
  brz $rem, .Lhalf_float_float_exit

.Lhalf_loop_epilogue4:
  // Only here if there are 1, 2 or 3 remaining elements
  // Use the worker which points to the remainder to process it
  ld32 $mscratch, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET/4
  ldb16step $azero, $mzero, $mscratch+=, $nElem
  andc $mscratch, $mscratch, 0x7 // back to 8B boundary
  cmpeq $mscratch, $mscratch, $dataPtr
  brz $mscratch, .Lhalf_float_float_exit

  and $mscratch, $nElem, 2
  brz $mscratch, .Lhalf_final_element // If we branch there is exactly 1 element left

  ld32            $aA01h,   $mzero,   $dataPtr, 0                               //             ^^
 {ld64step        $aB01f,   $mzero,   $dataBPtr+=, 1                            //       ^^ ^^    
    f16v2tof32    $aA01f,   $aA01h}                                             // ^^ ^^       vv
    f32v2axpy     $azeros,  $aB01f,   $aA01f                                    // vv vv       vv vv
    f16v2gina     $aR01h,   $azero,   0                                         // ^^
  st32step        $aR01h,   $mzero,   $dataPtr+=, 1                             // vv

  // how many left do we have? maximum of 1.
.Lhalf_final_element:
  and $nElem, $nElem,   0x1
  brz $nElem, .Lhalf_float_float_exit

 // Note the subword off-end value may be a NaN so we mustn't do fp operations
 // on it.
 {ldb16           $aA01h,   $mzero,   $dataPtr, 0                               //             ^^
    setzi         $aB1f,    0}                                                  //          ^^    
 {ld32            $aB0f,    $mzero,   $dataBPtr, 0                              //       ^^
    f16v2tof32    $aA01f,   $aA01h}                                             // ^^ ^^       vv
    f32v2axpy     $azeros,  $aB01f,   $aA01f                                    // vv vv vv vv
 {ldb16           $aA01h,   $mzero,   $dataPtr, 1    // [ pastend | pastend ]   //             ^^
    f16v2gina     $aR01h,   $azero,   0}             // [result | result ]      // ^^

  sort4x16lo      $aR01h,    $aR01h,  $aA01h     // [ value   | 0]              // ++          vv         
  st32            $aR01h,    $mzero,  $dataPtr, 0                               // vv
.Lhalf_float_float_exit:
  exitz $mzero

FN_SIZE VERTEX(half_float_half).kernel

#endif // __IPU__
