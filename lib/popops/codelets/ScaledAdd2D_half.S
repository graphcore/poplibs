#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

#define VERTEX_AXPBY_TRUE __runCodelet_popops__aXPlusbY2D___half_true
#define VERTEX_AXPBY_FALSE __runCodelet_popops__aXPlusbY2D___half_false
#define VERTEX_TRUE __runCodelet_popops__ScaledAdd2D___half_true
#define VERTEX_FALSE __runCodelet_popops__ScaledAdd2D___half_false
#define VERTEX_SUBTRACT __runCodelet_popops__ScaledSubtract2D___half
#define VERTEX_COMMON __ScaledAdd2D___half_common

// constants
#define VERTEX_DATA_A_OFFSET 0
#define VERTEX_DATA_A_SIZE_OFFSET 1
#define VERTEX_DATA_B_OFFSET 2
// 2 versions: one has a constant, which is a half
// the other a pointer to a tensor half
#define VERTEX_SCALE_CONST_OFFSET 6
#define VERTEX_SCALE_OFFSET 3
#define VERTEX_SCALE_B_OFFSET 4
// The AXPBY variants have 2 16-bit constants stored at VERTEX_SCALE_AB_OFFSET
// which can be accessed as one 32 bit word
#define VERTEX_SCALE_AB_OFFSET 3

// integer variables
#define outData m0
#define outDataSize m1
#define outDataB m2
#define data m3
#define dataSize m4
#define dataSizeD4 m5
#define dataB m6
#define origDataSize m7
#define triPtr m8:9
#define offset m10

// float variables
#define data0 a0:1
#define dataB0 a2:3
#define data1 a4:5
#define data1i0 a4
#define data1i1 a5
#define dataB1 a6:7
#define dataB1i0 a6
#define dataB1i1 a7

// scratch variables
#define mscratch m11
#define ascratch a6
#define ascratch2 a7

.globl VERTEX_AXPBY_TRUE
.type VERTEX_AXPBY_TRUE, @function

.globl VERTEX_AXPBY_FALSE
.type VERTEX_AXPBY_FALSE, @function

.globl VERTEX_TRUE
.type VERTEX_TRUE, @function

.globl VERTEX_FALSE
.type VERTEX_FALSE, @function

.globl VERTEX_SUBTRACT
.type VERTEX_SUBTRACT, @function


.section .text.VERTEX_FALSE
.align 4
VERTEX_FALSE:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET

  ldb16 $ascratch2, $mzero, $data, 0
  bri   VERTEX_COMMON
.size VERTEX_FALSE, .-VERTEX_FALSE

.section .text.VERTEX_SUBTRACT
.align 4
VERTEX_SUBTRACT:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  ldb16 $ascratch2, $mzero, $data, 0
  setzi $ascratch, -1.0h
  {
    bri   VERTEX_COMMON
    f16v2mul $ascratch2, $ascratch:BL, $ascratch2
  }
.size VERTEX_SUBTRACT, .-VERTEX_SUBTRACT

.section .text.VERTEX_AXPBY_TRUE
.align 4
VERTEX_AXPBY_TRUE:
  // load vertex state specific to this version of the vertex :
  // constant A,B pair of halves.  32 bit aligned
  ld32  $ascratch, $mvertex_base, $mzero, VERTEX_SCALE_AB_OFFSET
  ld32  $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  ld32  $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
  bri   axplusby_true_continue
.size VERTEX_AXPBY_TRUE, .-VERTEX_AXPBY_TRUE

.section .text.VERTEX_AXPBY_FALSE
.align 4
VERTEX_AXPBY_FALSE:
  // load vertex state specific to this version of the vertex : Tensors A,B
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_OFFSET
  ldb16 $ascratch2, $mzero, $data, 0
  ld32  $data, $mvertex_base, $mzero, VERTEX_SCALE_B_OFFSET
  ldb16 $ascratch, $mzero, $data, 0
  ld32  $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
  bri   axplusby_false_continue
.size VERTEX_AXPBY_FALSE, .-VERTEX_AXPBY_FALSE


.section .text.VERTEX_COMMON
.align 8
VERTEX_TRUE:
  // load vertex state specific to this version of the vertex : k, constant
  // set ascratch2 to {k, k}
  ldb16 $ascratch2, $mvertex_base, $mzero, VERTEX_SCALE_CONST_OFFSET

VERTEX_COMMON:
  // load common vertex state
  {
    ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_A_OFFSET
    // set ascratch to {1, 1}
    f16v2exp $ascratch, $azero
  }
axplusby_false_continue:
  {
    ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_A_SIZE_OFFSET
    // vertex->k should have the form of {1, k} for scaled add/sub
    // or {A, B} for axpby variants
    sort4x16lo $ascratch, $ascratch, $ascratch2
  }
axplusby_true_continue:
  {
    ld32 $outDataB, $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
    // setup $TAS for the f16v4mix instructions below.
    uput $TAS, $ascratch
  }

  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1
.Louter_loop:
  // load inner pointers
  ld32step $data, $mzero, $outData+=, 1
  {
    ld32step $origDataSize, $mzero, $outData+=, 1
    setzi $a0, CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT
  }
  {
    ld32step $dataB, $mzero, $outDataB+=, 1
    uput $FP_CLR, $a0
  }

  // pack out/in pointers
  tapack $triPtr, $data, $dataB, $data

  // process 4 at a time first as this is the optimal scenario
  shr $dataSizeD4, $origDataSize, 2
  brz $dataSizeD4, .Lvector4_loop_end

  // load the first values and push them into the accumulators.
  ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
  {
    // minus 1 from our count because of the preloading above.
    add $dataSizeD4, $dataSizeD4, -1
    f16v4mix $azeros, $data0, $dataB0
  }

    rpt $dataSizeD4, (2f-1f)/8-1
1:
  {
    // load the next values and retrieve the current from the accumulators.
    ld2x64pace $data0, $dataB0, $triPtr+=, $mzero, 0
    f16v4mix $data1, $azeros, $azeros
  }
  {
    // store the current result and process the next ones.
    st64pace $data1, $triPtr+=, $mzero, 0
    f16v4mix $azeros, $data0, $dataB0
  }
2:
  // process and store the final values.
  f16v4mix $data1, $azeros, $azeros
  st64pace $data1, $triPtr+=, $mzero, 0

.Lvector4_loop_end:
  // how many left do we have? maximum of 3.
  and $dataSize, $origDataSize, 0x3
  brz $dataSize, .Lend

  // we need to calculate what our out pointer is because the value is hidden
  // inside the $triPtr with no easy way of extracting it. we do this by using
  // how many elements we have processed (origDataSize-currentDataSize), then
  // doubled as we do one 32-bit load for every 2 halfs and we want the offset
  // to be number of bytes, not items.
  sub $offset, $origDataSize, $dataSize
  shl $offset, $offset, 1

  // zero the second half of the $data1 and $dataB1 registers because we will
  // only be loading into the first half from now on but processing them using
  // a v4 instruction.
  {
    // if we have at least 2 left we can use a st32 variation for at least some
    // of the remaining values.
    cmpult $mscratch, $dataSize, 2
    zero $data1i1
  }
  {
    brnz $mscratch, .Lscalar
    zero $dataB1i1
  }
.Lvector2:
  ld32 $data1i0, $data, $offset, 0
  ld32 $dataB1i0, $dataB, $offset, 0
  f16v4mix $azeros, $data1, $dataB1
  {
    add $dataSize, $dataSize, -2
    f16v4mix $data1, $azeros, $azeros
  }
  st32step $data1i0, $data, $offset+=, 1
  brz $dataSize, .Lend

.Lscalar:
  // there is one more element that needs to be stored, do a read/modify/write
  // so we do not trash anything else may be stored in the same word.
  ldb16 $data1i0, $data, $offset, 0
  ldb16 $dataB1i0, $dataB, $offset, 0

  f16v4mix $azeros, $data1, $dataB1
  {
    ldb16 $ascratch, $data, $offset, 1
    f16v4mix $data1, $azeros, $azeros
  }
  roll16 $data1i0, $data1i0, $ascratch

  st32 $data1i0, $data, $offset, 0

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

.size VERTEX_COMMON, .-VERTEX_COMMON

#endif // __IPU__
