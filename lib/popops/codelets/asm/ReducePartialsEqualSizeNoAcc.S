// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__
// Partials Equal Size Reduce Overview:
// See ReducePartialsEqualSizeAcc.S
#include "poplibs_support/TileConstants.hpp"
#include "poplar/AvailableVTypes.h"
#include "MathConstants.S"

#if defined(VECTOR_AVAIL_SCALED_PTR32) && defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
#define COMPACT_VECTOR_TYPES_AVAILABLE 1
#else
#define COMPACT_VECTOR_TYPES_AVAILABLE 0
#endif

// vertex state, all offsets are 16-bit
#if COMPACT_VECTOR_TYPES_AVAILABLE
#define VERTEX_OUT_PTR_OFFSET 0
#define VERTEX_OUT_COUNT_OFFSET 2
#define VERTEX_PARTIALS_BASE_OFFSET 4
#define VERTEX_PARTIALS_DELTA_PTR_OFFSET 8
#define VERTEX_PARTIALS_SIZE_OFFSET 10
#define VERTEX_SCALE_OFFSET 12
#else
#define VERTEX_OUT_PTR_OFFSET 0
#define VERTEX_OUT_COUNT_OFFSET 4
#define VERTEX_PARTIALS_BASE_OFFSET 8
#define VERTEX_PARTIALS_DELTA_PTR_OFFSET 12
#define VERTEX_PARTIALS_SIZE_OFFSET 16
#define VERTEX_SCALE_OFFSET 20
#endif

// stack state, all offsets are 32-bit
#define STACK_PARTIALS_DELTA_PTR_OFFSET 0

// constants
#define LDCONST_MASK ((1<<20)-1)
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

#if COMPACT_VECTOR_TYPES_AVAILABLE
#define SCALED_PTR32_SHL_BITS 2
#define SCALED_PTR64_SHL_BITS 3
#define VECTOR_LIST_BASE_BITS 20
#define VECTOR_LIST_COUNT_BITS 12
#define DELTAN_ADDRESS_BITS 18
#define DELTAN_ADDRESS_MASK ((1 << DELTAN_ADDRESS_BITS) - 1)
#else
#define PTR64_SHL_BITS 3
#define VECTOR_LIST_BASE_BITS 24
#define VECTOR_LIST_BASE_MASK ((1 << VECTOR_LIST_BASE_BITS) - 1)
#define VECTOR_LIST_COUNT_BITS 8
#define DELTAN_ADDRESS_BITS (21-PTR64_SHL_BITS)
#define DELTAN_ADDRESS_MASK ((1 << DELTAN_ADDRESS_BITS) - 1)
#endif

#define FLOAT_CONST_ONE 0x3f800000
#define FLOAT_GRAIN_SIZE 4
#define HALF_GRAIN_SIZE 8
#define BYTES_PER_GRAIN 16

// integer variables
#define outPtr m0
#define outCount m1
#define partialsSizeCounter m2
#define partialsBase m3
#define partialsDeltaPtr m4
#define numPartials m5
#define partialsSize m6
#define partialsElementPtr m7
#define partialsBaseWorking m8
#define mscratch m9
#define partialsStride m10
#if COMPACT_VECTOR_TYPES_AVAILABLE
#define base m11
#else
#define base mzero
#define numPartialsB0 m10
#define mask m9
#endif

// floating point variables
#define scale a7

#define VERTEX __runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_\UPDATE
#define VERTEX_SCALED __runCodelet_popops__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_\UPDATE

//******************************************************************************
.macro OUTPUT_HALFS_NO_ACCUMULATORS UPDATE
.ifc "\UPDATE","true"

    {add $partialsBase, $partialsBase, BYTES_PER_GRAIN
     f16v4mul  $a2:3, $scale:B, $a2:3}
    {ld64  $a0:1, $mzero, $outPtr, 0
     f16v4mul  $a4:5, $scale:B, $a4:5}
    {ld64  $a0:1, $mzero, $outPtr, 0
     f16v4add  $a2:3, $a2:3, $a0:1}

    {st64step  $a2:3, $mzero, $outPtr+=, 1
     f16v4add  $a4:5, $a4:5, $a0:1}
    st64step  $a4:5, $mzero, $outPtr+=, 1

.else
    {add $partialsBase, $partialsBase, BYTES_PER_GRAIN
     f16v4mul  $a2:3, $scale:B, $a2:3}
    {st64step  $a2:3, $mzero, $outPtr+=, 1
     f16v4mul  $a4:5, $scale:B, $a4:5}
    st64step  $a4:5, $mzero, $outPtr+=, 1
.endif
.endm
//******************************************************************************
.macro OUTPUT_FLOATS_NO_ACCUMULATORS UPDATE
.ifc "\UPDATE","true"

    {add $partialsBase, $partialsBase, BYTES_PER_GRAIN
     f32v2mul  $a2:3, $scale:B, $a2:3}
    {ld64  $a0:1, $mzero, $outPtr, 0
     f32v2mul  $a4:5, $scale:B, $a4:5}
    {ld64  $a0:1, $mzero, $outPtr, 0
     f32v2add  $a2:3, $a2:3, $a0:1}

    {st64step  $a2:3, $mzero, $outPtr+=, 1
     f32v2add  $a4:5, $a4:5, $a0:1}
    st64step  $a4:5, $mzero, $outPtr+=, 1

.else
    {add $partialsBase, $partialsBase, BYTES_PER_GRAIN
     f32v2mul  $a2:3, $scale:B, $a2:3}
    {st64step  $a2:3, $mzero, $outPtr+=, 1
     f32v2mul  $a4:5, $scale:B, $a4:5}
    st64step  $a4:5, $mzero, $outPtr+=, 1
.endif
.endm
//******************************************************************************
.macro MAKE_REDUCE_VERTEX OP PARTIALS_TYPE OUT_TYPE UPDATE ACC

// Setup some stuff depending on the operation selected

.ifc "\OP", "ReduceMax"
  .ifc "\PARTIALS_TYPE", "float"
    .equ INITIAL_VALUE, MIN_FLOAT
  .else
    .equ INITIAL_VALUE, MIN_HALF_BROADCAST
  .endif
.endif

.ifc "\OP", "ReduceMin"
  .ifc "\PARTIALS_TYPE", "float"
    .equ INITIAL_VALUE, MAX_FLOAT
  .else
    .equ INITIAL_VALUE, MAX_HALF_BROADCAST
  .endif
.endif

.equ SCALE_CAST_REQUIRED, 0
.ifc "\PARTIALS_TYPE", "half"
  .equ SCALE_CAST_REQUIRED, 1
.endif

// ************* Macro code ****************

.globl VERTEX
.type VERTEX @function

.section .text.VERTEX
.align 4
VERTEX:
#if COMPACT_VECTOR_TYPES_AVAILABLE
  setzi $base, TMEM_REGION0_BASE_ADDR
#endif
  {
    bri common_\@
    or $scale, $azero, FLOAT_CONST_ONE & ~LDCONST_MASK
  }

.size VERTEX, .-VERTEX

.globl VERTEX_SCALED
.type VERTEX_SCALED @function

.section .text.VERTEX_SCALED
.align 8
.if SCALE_CAST_REQUIRED==0
  nop     // Rpt alignment
.endif

VERTEX_SCALED:
#if COMPACT_VECTOR_TYPES_AVAILABLE
  ldz16 $mscratch, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET/2
  {
    setzi $base, TMEM_REGION0_BASE_ADDR
    fnop
  }
  ld32  $scale, $base, $mzero, $mscratch
#else
  ld32 $mscratch, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET/4
  ld32  $scale, $base, $mscratch, 0
#endif

common_\@:
  ldz16 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET/2
#if COMPACT_VECTOR_TYPES_AVAILABLE
  setzi $base, TMEM_REGION0_BASE_ADDR
  ldz16 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/2
  shl $outPtr, $outPtr, SCALED_PTR64_SHL_BITS
  ld32  $partialsBase, $mzero, $mvertex_base, VERTEX_PARTIALS_BASE_OFFSET/4
  ldz16 $partialsDeltaPtr, $mzero, $mvertex_base, VERTEX_PARTIALS_DELTA_PTR_OFFSET/2
  shr $numPartials, $partialsBase, VECTOR_LIST_BASE_BITS
  shl $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS
  shr $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS
  shl $partialsDeltaPtr, $partialsDeltaPtr, SCALED_PTR32_SHL_BITS
#else
  ld32 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET/4

  // Extract upper 8 bits of count followed by base address
  ld32 $partialsBase, $mzero, $mvertex_base, VERTEX_PARTIALS_BASE_OFFSET/4
  shr $numPartials, $partialsBase, VECTOR_LIST_BASE_BITS
  shl $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS
  shr $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS

  // Extract lower 8 bits of count followed by deltan pointer address
  ld32 $partialsDeltaPtr, $mzero, $mvertex_base, VERTEX_PARTIALS_DELTA_PTR_OFFSET/4
  shr $numPartialsB0, $partialsDeltaPtr, VECTOR_LIST_BASE_BITS
  shl $partialsDeltaPtr, $partialsDeltaPtr, VECTOR_LIST_COUNT_BITS
  shr $partialsDeltaPtr, $partialsDeltaPtr, VECTOR_LIST_COUNT_BITS

  // Combine lower and upper 8-bits parts to form count
  shl $numPartials, $numPartials, 8
  or $numPartials, $numPartials, $numPartialsB0
#endif

  // store the partialsDeltaPtr onto the stack as we reset it each iteration of
  // the middle loop.
  st32 $partialsDeltaPtr, $mzero, $mworker_base, STACK_PARTIALS_DELTA_PTR_OFFSET
  // Setup for addressing partials
  shl $partialsStride, $outCount, 4
  setzi $mscratch, DELTAN_ADDRESS_MASK

.if SCALE_CAST_REQUIRED
  {sub $outCount, $outCount, 1
   f32tof16  $scale, $scale}
.else
  sub $outCount, $outCount, 1
.endif

out_loop_\@:
  ldconst $a2, INITIAL_VALUE
  {
    ldz16 $partialsSizeCounter, $mzero, $mvertex_base, VERTEX_PARTIALS_SIZE_OFFSET/2
    mov $a3,$a2
  }
  {
    mov $partialsBaseWorking, $partialsBase
    mov $a4:5, $a2:3
  }
partial_column_loop_\@:

  // Loop though all the partials, reducing our grain size columns each loop
  ld32 $partialsDeltaPtr, $mzero, $mworker_base, STACK_PARTIALS_DELTA_PTR_OFFSET
  ld32step $partialsElementPtr, $base, $partialsDeltaPtr+=, 1

   rpt $numPartials, (2f-1f)/8-1
1:
  {
    and $partialsElementPtr, $partialsElementPtr, $mscratch
    fnop
  }
#if (COMPACT_VECTOR_TYPES_AVAILABLE == 0)
  // TODO T15192: On Mk2 the use of DELTANELEMENTS results in the need of an
  // additional shift depending on the alignment. This reduces the efficiency
  // of the loop. Update the cycle estimates accordingly.
  {
    shl $partialsElementPtr, $partialsElementPtr, PTR64_SHL_BITS
    fnop
  }
#endif
  {
    ld64 $a0:1, $partialsBaseWorking, $partialsElementPtr, 0
    fnop
  }
  {
    ld64 $a0:1, $partialsBaseWorking, $partialsElementPtr, 1
    \ACC $a2:3, $a2:3, $a0:1
  }
  {
    // Over read of a pointer
    ld32step $partialsElementPtr, $base, $partialsDeltaPtr+=, 1
    \ACC $a4:5, $a4:5, $a0:1
  }
2:
  // Stride to the next grain size column group that needs to be reduced into
  // the current set of outputs
  add $partialsBaseWorking, $partialsBaseWorking, $partialsStride
  brnzdec $partialsSizeCounter, partial_column_loop_\@

// Pick an output macro based on output type and on if the operation used the
// accumulators
.ifc "\OUT_TYPE", "float"
  OUTPUT_FLOATS_NO_ACCUMULATORS \UPDATE
.endif
.ifc "\OUT_TYPE", "half"
  OUTPUT_HALFS_NO_ACCUMULATORS \UPDATE
.endif

  brnzdec $outCount, out_loop_\@

  exitz $mzero

.size VERTEX_SCALED, .-VERTEX_SCALED

.endm

MAKE_REDUCE_VERTEX ReduceMax half half true f16v4max
MAKE_REDUCE_VERTEX ReduceMax float float true f32v2max
MAKE_REDUCE_VERTEX ReduceMax half half false f16v4max
MAKE_REDUCE_VERTEX ReduceMax float float false f32v2max

MAKE_REDUCE_VERTEX ReduceMin half half true f16v4min
MAKE_REDUCE_VERTEX ReduceMin float float true f32v2min
MAKE_REDUCE_VERTEX ReduceMin half half false f16v4min
MAKE_REDUCE_VERTEX ReduceMin float float false f32v2min

#endif // __IPU__
