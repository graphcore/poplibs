// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "poplar/StackSizeDefs.hpp"
#include "MathConstants.S"
#include "CommonPoplibsMacros.h.S"

#define VERTEX_ADD_FAST __runCodelet_popops__ScaledAdd2D___float_float_float_true
#define VERTEX_SUBTRACT_FAST __runCodelet_popops__ScaledSubtract2D___float_float_true

#define VERTEX_ADD_SLOW __runCodelet_popops__ScaledAdd2D___float_float_float_false
#define VERTEX_SUBTRACT_SLOW __runCodelet_popops__ScaledSubtract2D___float_float_false

#define VERTEX_AXPBY __runCodelet_popops__aXPlusbY2D___float_float_false
#define VERTEX_AX_MINUS_BY __runCodelet_popops__aXMinusbY2D___float_float_false

#define VERTEX_COMMON __ScaledAdd2D___float_common
#define VERTEX_AXPBY_COMMON __aXPlusbY2D___float_common

// constants
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 1
#define VERTEX_DATA_B_OFFSET 2
#define VERTEX_SCALE_OFFSET 3
// For ax+by we have a second scale value
#define VERTEX_SCALE_B_OFFSET 4

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define data m3
#define dataSize m4
#define dataSizeD2 m5
#define dataB m6
#define origDataSize m7
#define triPtr m8:9
#define triPtrData m8
#define triPtrDataB m9
#define offset m10
#define memConstraints m11

// float variables
#define data0 a0:1
#define data0i0 a0
#define dataB0 a2:3
#define dataB0i0 a2
#define data1 a4:5
#define data1i0 a4
#define data1i1 a5
#define dataB1 a6:7
#define dataB1i0 a6
#define dataB1i1 a7
#define ascaleB  a7

// scratch variables
#define mscratch m8
#define ascratch a6

#ifdef VECTOR_AVAIL_SHORT_SPAN
#define SHORT_SPAN_PTR_SIZE 20
#define SHORT_SPAN_LENGTH_SIZE 12
#endif

.macro CHOOSE_FAST_OR_SLOW_PATH FAST_PATH_LABEL
  // The fast path is always OK if constraints were applied
  brnz $memConstraints, \FAST_PATH_LABEL
  // Or if the data start is far enough apart.  It could be ok in some other
  // circumstances but this is time consuming to check correctly.
  sub $mscratch, $data, $dataB
  abs $mscratch, $mscratch
  // +8 is to account for really wanting a <= instruction
  cmpult $mscratch, $mscratch, (2 * TMEM_ELEMSIZE) + 8
  brz $mscratch, \FAST_PATH_LABEL
1:
.endm


FN_WORKER_ENTRY_POINT VERTEX_ADD_SLOW
  setzi $memConstraints, 0
  bri 1f
FN_EXPORT VERTEX_ADD_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  ld32  $ascratch, $mzero, $data, 0
  bri   VERTEX_COMMON
FN_SIZE VERTEX_ADD_SLOW

FN_WORKER_ENTRY_POINT VERTEX_SUBTRACT_SLOW
  setzi $memConstraints, 0
  bri   1f
FN_EXPORT VERTEX_SUBTRACT_FAST
  setzi $memConstraints, 1
1:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  {ld32  $ascratch, $mzero, $data, 0
   or    $data1i0, $azero, FLOAT_NEG_1_0}
  {bri  VERTEX_COMMON
   f32mul $ascratch, $ascratch, $data1i0}
FN_SIZE VERTEX_SUBTRACT_SLOW

FN_SECTION VERTEX_COMMON 8 nop
VERTEX_COMMON:
  // load vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
  ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
  {
    // minus 1 for the brnzdec
    add $outDataSize, $outDataSize, -1
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $ascratch
  }
.Louter_loop:
#ifdef VECTOR_AVAIL_SHORT_SPAN
  ld32step $data, $mzero, $outData+=, 1
  shr $origDataSize, $data, SHORT_SPAN_PTR_SIZE
  shl $data, $data, SHORT_SPAN_LENGTH_SIZE
  shr $data, $data, SHORT_SPAN_LENGTH_SIZE
#else
  ld32step $data, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
#endif

  ld32step $dataB, $mzero, $outDataB+=, 1

  // process 2 at a time first as this is the optimal scenario
  shr $dataSizeD2, $origDataSize, 1
  brz $dataSizeD2, .Lvector2_loop_end

  // Choose the fast or slow path, based on flag set at the entry point
  CHOOSE_FAST_OR_SLOW_PATH .Lfast_path

  // Use tapack to copy the 2 addresses into working registers for the loop
  tapack $triPtr, $data, $dataB, $mzero

  ld64 $data0, $mzero, $triPtrData, 0
  ld64step $dataB0, $mzero, $triPtrDataB+=, 1
  {add $dataSizeD2, $dataSizeD2, -1
   f32v2axpy $azeros, $dataB0, $data0}

  rpt $dataSizeD2, (2f-1f)/8-1
1:
  {ld64 $data0, $mzero, $triPtrData, 1
   f32v2axpy $data1, $azeros, $azeros}

  {ld64step $dataB0, $mzero, $triPtrDataB+=, 1
   fnop}

  {st64step $data1, $mzero, $triPtrData+=, 1
   f32v2axpy $azeros, $dataB0, $data0}
2:
  f32v2axpy $data1, $azeros, $azeros
  st64step $data1, $mzero, $triPtrData+=, 1
  bri .Lvector2_loop_end

.Lfast_path:
  // pack out/in pointers
  tapack $triPtr, $data, $dataB, $data
  // load the first values and push them into the accumulators.
  ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
  {
    // minus 1 from our count because of the preloading above.
    add $dataSizeD2, $dataSizeD2, -1
    f32v2axpy $azeros, $dataB0, $data0
  }

  rpt $dataSizeD2, (2f-1f)/8-1
1:
  {
    // load the next values and retrieve the current from the accumulators.
    ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
    f32v2axpy $data1, $azeros, $azeros
  }
  {
    // store the current result and process the next ones.
    st64pace $data1, $triPtr+=, $mzero, 0
    f32v2axpy $azeros, $dataB0, $data0
  }
2:
  // process and store the final values.
  f32v2axpy $data1, $azeros, $azeros
  st64pace $data1, $triPtr+=, $mzero, 0

.Lvector2_loop_end:
  // how many left do we have? maximum of 1.
  and $dataSize, $origDataSize, 0x1
  brz $dataSize, .Lend

  // we need to calculate what our out pointer is because the value is hidden
  // inside the $triPtr with no easy way of extracting it. we do this by using
  // how many elements we have processed (origDataSize-currentDataSize), then
  // times 4 as we do one 32-bit load for every float and we want the offset
  // to be number of bytes, not items.
  sub $offset, $origDataSize, $dataSize
  shl $offset, $offset, 2

.Lscalar:
  // zero the second half of the $data1 and $dataB1 registers because we will
  // only be loading into the first half from now on but processing them using
  // a v2 instruction.
  {
    ld32 $data1i0, $data, $offset, 0
    zero $data1i1
  }
  {
    ld32 $dataB1i0, $dataB, $offset, 0
    zero $dataB1i1
  }
  f32v2axpy $azeros, $dataB1, $data1
  f32v2axpy $data1, $azeros, $azeros
  st32step $data1i0, $data, $offset+=, 1

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

FN_SIZE VERTEX_COMMON

//------------------------------------------------------------------------------
FN_WORKER_ENTRY_POINT VERTEX_AX_MINUS_BY
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  ld32  $ascratch, $mzero, $data, 0
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  {ld32  $ascaleB, $mzero, $data, 0
  or $data0i0, $azero, FLOAT_NEG_1_0}
  {bri   VERTEX_AXPBY_COMMON
  f32mul $ascaleB, $ascaleB, $data0i0}
FN_SIZE VERTEX_AX_MINUS_BY


FN_WORKER_ENTRY_POINT VERTEX_AXPBY
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  ld32  $ascratch, $mzero, $data, 0
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  ld32  $ascaleB, $mzero, $data, 0
  bri   VERTEX_AXPBY_COMMON
FN_SIZE VERTEX_AXPBY


FN_SECTION VERTEX_AXPBY_COMMON 8 nop
VERTEX_AXPBY_COMMON:
  // load vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
  ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1

.Louter_loop_axpby:
#ifdef VECTOR_AVAIL_SHORT_SPAN
  ld32step $data, $mzero, $outData+=, 1
  shr $origDataSize, $data, SHORT_SPAN_PTR_SIZE
  shl $data, $data, SHORT_SPAN_LENGTH_SIZE
  shr $data, $data, SHORT_SPAN_LENGTH_SIZE
#else
  ld32step $data, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
#endif

  ld32step $dataB, $mzero, $outDataB+=, 1
  // process 2 at a time first as this is the optimal scenario
  shr $dataSizeD2, $origDataSize, 1
  brz $dataSizeD2, .Lvector2_loop_end_axpby

  ld64 $data0, $mzero, $data, 0
  {ld64step $dataB0, $mzero, $dataB+=, 1
   f32v2mul $data0, $ascratch:B, $data0}
  {add $dataSizeD2, $dataSizeD2, -1
   f32v2mul $dataB0, $ascaleB:B, $dataB0}

  rpt $dataSizeD2, (2f-1f)/8-1
1:
  {ld64 $data0, $mzero, $data, 1
   f32v2add $data1, $data0, $dataB0}

  {ld64step $dataB0, $mzero, $dataB+=, 1
   f32v2mul $data0, $ascratch:B, $data0}

  {st64step $data1, $mzero, $data+=, 1
   f32v2mul $dataB0, $ascaleB:B, $dataB0}
2:
  f32v2add $data1, $data0, $dataB0
  st64step $data1, $mzero, $data+=, 1

 .Lvector2_loop_end_axpby:
  // how many left? maximum of 1.
  and $dataSize, $origDataSize, 0x1
  brz $dataSize, .Lend_axpby

.Lscalar_axpby:
  ld32 $data0i0, $mzero, $data, 0
  {ld32 $dataB0i0, $mzero, $dataB, 0
   f32mul $data0i0, $data0i0, $ascratch}
  f32mul $dataB0i0, $dataB0i0, $ascaleB
  f32add $data0i0, $dataB0i0, $data0i0
  st32 $data0i0, $mzero, $data, 0
.Lend_axpby:
  brnzdec $outDataSize, .Louter_loop_axpby

  exitz $mzero

FN_SIZE VERTEX_AXPBY_COMMON

#endif // __IPU__
