#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

// vertex state, all offsets are 16-bit
#define VERTEX_OUT_PTR_OFFSET 0
#define VERTEX_OUT_COUNT_OFFSET 1
#define VERTEX_PARTIALS_BASE_OFFSET 2
#define VERTEX_PARTIALS_DELTA_PTR_OFFSET 4
#define VERTEX_PARTIALS_SIZE_OFFSET 5
#define VERTEX_SCALE_OFFSET 6

// stack state, all offsets are 32-bit
#define STACK_PARTIALS_DELTA_PTR_OFFSET 0

// constants
#define LDCONST_MASK ((1<<20)-1)
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)
#define SCALED_PTR32_SHL_BITS 2
#define SCALED_PTR64_SHL_BITS 3
#define VECTOR_LIST_BASE_BITS 20
#define VECTOR_LIST_COUNT_BITS 12
#define DELTAN_LENGTH_BITS 14
#define DELTAN_ADDRESS_MASK 0x3ffff

#define FLOAT_CONST_ONE 0x3f800000
#define FLOAT_GRAIN_SIZE 4
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
#define base m11

// floating point variables
#define scale a7

.macro MAKE_REDUCE_VERTEX OP PARTIALS_TYPE OUT_TYPE ACC GRAIN_SIZE

.globl __runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false
.type __runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false @function

.section .text.__runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false
.align 4
__runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false:
  setzi $base, TMEM_REGION0_BASE_ADDR
  {
    bri common_\@
    or $scale, $azero, FLOAT_CONST_ONE & ~LDCONST_MASK
  }

.size __runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false,\
  .-__runCodelet_popops__ReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false

.globl __runCodelet_popops__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false
.type __runCodelet_popops__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false @function

.section .text.__runCodelet_popops__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false
.align 8

__runCodelet_popops__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false:
  ldz16 $mscratch, $mzero, $mvertex_base, VERTEX_SCALE_OFFSET
  setzi $base, TMEM_REGION0_BASE_ADDR
  ld32  $scale, $base, $mzero, $mscratch

common_\@:
  ldz16 $outPtr, $mzero, $mvertex_base, VERTEX_OUT_PTR_OFFSET
  ldz16 $outCount, $mzero, $mvertex_base, VERTEX_OUT_COUNT_OFFSET
  ld32  $partialsBase, $mzero, $mvertex_base, VERTEX_PARTIALS_BASE_OFFSET/2
  ldz16 $partialsDeltaPtr, $mzero, $mvertex_base, VERTEX_PARTIALS_DELTA_PTR_OFFSET

  // unpack pointers and sizes from the vertex state.
  shl $outPtr, $outPtr, SCALED_PTR64_SHL_BITS
  shr $numPartials, $partialsBase, VECTOR_LIST_BASE_BITS
  shl $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS
  shr $partialsBase, $partialsBase, VECTOR_LIST_COUNT_BITS
  shl $partialsDeltaPtr, $partialsDeltaPtr, SCALED_PTR32_SHL_BITS

  // store the partialsDeltaPtr onto the stack as we reset it each iteration of
  // the middle loop.
  st32 $partialsDeltaPtr, $mzero, $mworker_base, STACK_PARTIALS_DELTA_PTR_OFFSET
  // Setup for addressing partials
  shl $partialsStride, $outCount, 4
  setzi $mscratch, DELTAN_ADDRESS_MASK

  sub $outCount, $outCount, 1

out_loop_\@:
  // reload the partials base pointer and reset numPartialsM1
  {
    ldz16 $partialsSizeCounter, $mzero, $mvertex_base, VERTEX_PARTIALS_SIZE_OFFSET
    // zero the accumulators.
    setzi $a0, ZAACC_BITMASK
  }
  {
    mov $partialsBaseWorking, $partialsBase
    uput $FP_CLR, $a0
  }

partial_column_loop_\@:

  // Loop though all the partials, reducing our grain size columns each loop
  ld32 $partialsDeltaPtr, $mzero, $mworker_base, STACK_PARTIALS_DELTA_PTR_OFFSET
  ld32step $partialsElementPtr, $base, $partialsDeltaPtr+=, 1
  {
    rpt $numPartials, (2f-1f)/8-1
    fnop
  }
1:
  {
    and $partialsElementPtr, $partialsElementPtr, $mscratch
    fnop
  }
  {
    ld64 $a0:1, $partialsBaseWorking, $partialsElementPtr, 0
    fnop
  }
  {
    ld64 $a2:3, $partialsBaseWorking, $partialsElementPtr, 1
    fnop
  }
  {
    // Over read of a pointer
    ld32step $partialsElementPtr, $base, $partialsDeltaPtr+=, 1
    \ACC $a0:3
  }
2:
  // Stride to the next grain size column group that needs to be reduced into
  // the current set of outputs
  add $partialsBaseWorking, $partialsBaseWorking, $partialsStride
  brnzdec $partialsSizeCounter, partial_column_loop_\@

  // Read, scale and store the output
  // offset by grain size 128 bit words for the next output group
  {
    add $partialsBase, $partialsBase, BYTES_PER_GRAIN
    f32v2gina $a0:1, $azeros, 0
  }
  {
    rpt \GRAIN_SIZE/2-1, (2f-1f)/8-1
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
1:
  {
    st64step $a0:1, $mzero, $outPtr+=, 1
    f32v2gina $a0:1, $azeros, 0
  }
  {
    nop
    f32v2mul  $a0:1, $scale:B, $a0:1
  }
2:
  st64step $a0:1, $mzero, $outPtr+=, 1

  brnzdec $outCount, out_loop_\@

  exitz $mzero

.size __runCodelet_popops__v2__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false,\
  .-__runCodelet_popops__v2__ScaledReducePartialsEqualSize___popops__\OP\()_\PARTIALS_TYPE\()_\OUT_TYPE\()_false

.endm

MAKE_REDUCE_VERTEX ReduceAdd half float f16v8acc 8
MAKE_REDUCE_VERTEX ReduceAdd float float f32v4acc 4
MAKE_REDUCE_VERTEX ReduceSquareAdd half float f16v8sqacc 8
MAKE_REDUCE_VERTEX ReduceSquareAdd float float f32v4sqacc 4

#endif // __IPU__
