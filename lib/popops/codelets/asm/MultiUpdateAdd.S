#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

#define VERTEX(type) __runCodelet_popops__MultiUpdateAdd___##type##_false

// vertex state, all offsets are 16-bit
#define VERTEX_STATE_STATE_OFFSET 0
#define VERTEX_STATE_POINTER_OFFSET 2
#define VERTEX_STATE_SIZE_OFFSET 4
#define VERTEX_STATE_SUB_T_OFFSET 6
#define VERTEX_STATE_BASE_T_OFFSET 8
#define VERTEX_STATE_REGION_SIZE_OFFSET 9
#define VERTEX_STATE_BASE_OFFSET_OFFSET 10
#define VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET 12

// constants
#define SCALED_PTR32_SHL_BITS 2
#define SIZEOF_HALF 2
#define SIZEOF_FLOAT 4
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

// integer variables
#define offsetPtr m0
#define offsetSize m1
#define baseTPtr m2
#define subTPtr m3
#define baseOffset m4
#define numBaseElements m5
#define regionSize m6
#define regionSizeM1 m7
#define regionBytesOffset m8
#define baseIdx m9
#define subTOffset m10

// floating point variables
#define scale a7

// scratch variables
#define mscratch m11

.globl VERTEX(float)
.type VERTEX(float), @function

.section .text.VERTEX(float)
.align 8
  nop
VERTEX(float):
  // load vertex state, do a size check on offsets
  ld32 $offsetSize, $mzero, $mvertex_base, VERTEX_STATE_SIZE_OFFSET/2
  brz $offsetSize, .Lfloat_epilogue

  ld32 $offsetPtr, $mzero, $mvertex_base, VERTEX_STATE_POINTER_OFFSET/2
  ld32 $subTPtr, $mzero, $mvertex_base, VERTEX_STATE_SUB_T_OFFSET/2
  ldz16 $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET
  ldz16 $regionSize, $mzero, $mvertex_base, VERTEX_STATE_REGION_SIZE_OFFSET
  ld32 $baseOffset, $mzero, $mvertex_base, VERTEX_STATE_BASE_OFFSET_OFFSET/2
  ld32 $numBaseElements, $mzero, $mvertex_base, VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET/2

  // load scale
  ld32 $mscratch, $mzero, $mvertex_base, VERTEX_STATE_STATE_OFFSET/2
  ld32 $scale, $mzero, $mscratch, 0

  // expand the SCALED_PTR32 pointer
  shl $baseTPtr, $baseTPtr, SCALED_PTR32_SHL_BITS
  setzi $mscratch, TMEM_REGION0_BASE_ADDR
  add $baseTPtr, $baseTPtr, $mscratch

  // minus 1 from the region size because we pipeline it.
  sub $regionSizeM1, $regionSize, 1

  // we offset both baseT and subT by regionSize * sizeof(T) so precalculate
  // that outside of the main loop.
  mul $regionBytesOffset, $regionSize, SIZEOF_FLOAT

  sub $offsetSize, $offsetSize, 1
.Lfloat_offset_loop:
  ld32 $baseIdx, $offsetPtr, $mzero, $offsetSize

  sub $baseIdx, $baseIdx, $baseOffset
  // check baseIdx is within the range of the values in baseT by doing:
  //  if (baseIdx > numBaseElements) continue;
  // note: this overflow relies on baseIdx and numBaseElements being smaller
  // then 2^31.
  cmpult $mscratch, $baseIdx, $numBaseElements
  brz $mscratch, .Lfloat_offset_loop_epilogue

  // correct baseIdx to the current offset and move it onto the correct region
  mul $baseIdx, $baseIdx, $regionBytesOffset

  // move subT on onto the correct region
  mul $subTOffset, $offsetSize, $regionBytesOffset

  // load from the first two pointers.
  ld32step $a1, $subTPtr, $subTOffset+=, 1
  {
    ld32 $a0, $baseTPtr, $baseIdx, 0
    f32mul $a1, $a1, $scale
  }
  {
    rpt $regionSizeM1, (2f-1f)/8-1
    f32add $a3, $a0, $a1
  }
1:
  {
    ld32step $a1, $subTPtr, $subTOffset+=, 1
    f32add $a2, $a0, $a1
  }
  {
    ld32 $a0, $baseTPtr, $baseIdx, 1
    f32mul $a1, $a1, $scale
  }
  {
    st32step $a2, $baseTPtr, $baseIdx+=, 1
    f32add $a3, $a0, $a1
  }
2:
  // process the final element
  st32 $a3, $baseTPtr, $baseIdx, 0

.Lfloat_offset_loop_epilogue:
  brnzdec $offsetSize, .Lfloat_offset_loop

.Lfloat_epilogue:
  exitz $mzero

.size VERTEX(float), . - VERTEX(float)

.globl VERTEX(half)
.type VERTEX(half), @function

.section .text.VERTEX(half)
.align 8
VERTEX(half):
  // load vertex state, do a size check on offsets
  ld32 $offsetSize, $mzero, $mvertex_base, VERTEX_STATE_SIZE_OFFSET/2
  brz $offsetSize, .Lhalf_epilogue

  ld32 $offsetPtr, $mzero, $mvertex_base, VERTEX_STATE_POINTER_OFFSET/2
  ld32 $subTPtr, $mzero, $mvertex_base, VERTEX_STATE_SUB_T_OFFSET/2
  ldz16 $baseTPtr, $mzero, $mvertex_base, VERTEX_STATE_BASE_T_OFFSET
  ldz16 $regionSize, $mzero, $mvertex_base, VERTEX_STATE_REGION_SIZE_OFFSET
  ld32 $baseOffset, $mzero, $mvertex_base, VERTEX_STATE_BASE_OFFSET_OFFSET/2
  ld32 $numBaseElements, $mzero, $mvertex_base, VERTEX_STATE_NUM_BASE_ELEMENTS_OFFSET/2

  // clear the accumulators incase there is anything nefarious in there for the
  // first call to f16v4mix.
  {
    ld32 $mscratch, $mzero, $mvertex_base, VERTEX_STATE_STATE_OFFSET/2
    setzi $a0, ZAACC_BITMASK
  }
  {
    ldb16 $scale, $mzero, $mscratch, 0
    uput $FP_CLR, $a0
  }
  // load scale and place {1, scale} into the $TAS CSR
  {
    setzi $mscratch, TMEM_REGION0_BASE_ADDR
    f16v2exp $a0, $azero
  }

  // expand the SCALED_PTR32 pointer
  {
    shl $baseTPtr, $baseTPtr, SCALED_PTR32_SHL_BITS
    sort4x16lo $scale, $a0, $scale
  }
  {
    add $baseTPtr, $baseTPtr, $mscratch
    uput $TAS, $scale
  }

  // we process 32-bits at a time so halve the region size. the host code must
  // enforce this. finally minus 1 from the result because we pipeline it.
  // also as we don't have an f16v2mix instruction need to zero the odd
  // registers in each pair that we plan to use.
  {
    shr $regionSizeM1, $regionSize, 1
    zero $a1
  }
  {
    sub $regionSizeM1, $regionSizeM1, 1
    zero $a3
  }

  // we offset both baseT and subT by regionSize * sizeof(T) so precalculate
  // that outside of the main loop.
  mul $regionBytesOffset, $regionSize, SIZEOF_HALF

  sub $offsetSize, $offsetSize, 1
.Lhalf_offset_loop:
  ld32 $baseIdx, $offsetPtr, $mzero, $offsetSize

  sub $baseIdx, $baseIdx, $baseOffset
  // check baseIdx is within the range of the values in baseT by doing:
  //  if (baseIdx > numBaseElements) continue;
  // note: this overflow relies on baseIdx and numBaseElements being smaller
  // then 2^31.
  cmpult $mscratch, $baseIdx, $numBaseElements
  brz $mscratch, .Lhalf_offset_loop_epilogue

  // correct baseIdx to the current offset and move it onto the correct region
  mul $baseIdx, $baseIdx, $regionBytesOffset

  // move subT on onto the correct region
  mul $subTOffset, $offsetSize, $regionBytesOffset

  // load from the first two pointers.
  ld32 $a0, $baseTPtr, $baseIdx, 0
  ld32step $a2, $subTPtr, $subTOffset+=, 1

  {
    rpt $regionSizeM1, (2f-1f)/8-1
    f16v4mix $azeros, $a0:1, $a2:3
  }
1:
  {
    ld32step $a2, $subTPtr, $subTOffset+=, 1
    f16v4mix $a4:5, $azeros, $azeros
  }
  {
    ld32 $a0, $baseTPtr, $baseIdx, 1
    fnop
  }
  {
    st32step $a4, $baseTPtr, $baseIdx+=, 1
    f16v4mix $azeros, $a0:1, $a2:3
  }
2:
  // process the final element
  f16v4mix $a4:5, $azeros, $azeros
  st32 $a4, $baseTPtr, $baseIdx, 0

.Lhalf_offset_loop_epilogue:
  brnzdec $offsetSize, .Lhalf_offset_loop

.Lhalf_epilogue:
  exitz $mzero

.size VERTEX(half), . - VERTEX(half)

#endif // __IPU__
