#ifdef __IPU__

#define VERTEX_TRUE __runCodelet_popops__ScaledAdd2D___half_true
#define VERTEX_FALSE __runCodelet_popops__ScaledAdd2D___half_false
#define VERTEX_COMMON __ScaledAdd2D___half_common

// constants
#define VERTEX_DATA_OFFSET 0
#define VERTEX_DATA_SIZE_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
// 2 versions: one has a constant(K), which is a half
// the other a pointer to a tensor half (factor)
#define VERTEX_K_OFFSET 6
#define VERTEX_FACTOR_OFFSET 3

// integer variables
#define outData m0
#define outDataSize m1
#define outDeltas m2
#define data m3
#define dataSize m4
#define dataSizeD4 m5
#define deltas m6
#define origDataSize m7
#define triPtr m8:9
#define offset m10

// float variables
#define data0 a0:1
#define delta0 a2:3
#define data1 a4:5
#define data1i0 a4
#define data1i1 a5
#define delta1 a6:7
#define delta1i0 a6
#define delta1i1 a7

// scratch variables
#define mscratch m11
#define ascratch a6
#define ascratch2 a7

.globl VERTEX_TRUE
.type VERTEX_TRUE, @function

.globl VERTEX_FALSE
.type VERTEX_FALSE, @function

.section .text.VERTEX_TRUE
.align 4

VERTEX_TRUE:
  // load vertex state specific to this version of the vertex : k, constant
  ldb16 $ascratch2, $mvertex_base, $mzero, VERTEX_K_OFFSET
  bri   VERTEX_COMMON
.size VERTEX_TRUE, .-VERTEX_TRUE

.section .text.VERTEX_FALSE
.align 4

VERTEX_FALSE:
  // load vertex state specific to this version of the vertex : Tensor: via a pointer
  ld32  $data, $mvertex_base, $mzero, VERTEX_FACTOR_OFFSET
  ldb16 $ascratch2, $mzero, $data, 0
  bri   VERTEX_COMMON
.size VERTEX_FALSE, .-VERTEX_FALSE


.section .text.VERTEX_COMMON
.align 8
  nop     // To force alignment for the rpt below

VERTEX_COMMON:
  // load common vertex state
  {
    ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_OFFSET
    // vertex->k should have the form of {1, k}
    f16v2exp $ascratch, $azero
  }
  {
    ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_SIZE_OFFSET
    sort4x16lo $ascratch, $ascratch, $ascratch2
  }
  {
    ld32 $outDeltas, $mvertex_base, $mzero, VERTEX_DELTAS_OFFSET
    // setup $TAS for the f16v4mix instructions below.
    uput $TAS, $ascratch
  }

  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1
.Louter_loop:
  // load inner pointers
  ld32step $data, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
  ld32step $deltas, $mzero, $outDeltas+=, 1

  // pack out/in pointers
  tapack $triPtr, $data, $deltas, $data

  // process 4 at a time first as this is the optimal scenario
  shr $dataSizeD4, $origDataSize, 2
  brz $dataSizeD4, .Lvector4_loop_end

  // load the first values and push them into the accumulators.
  ld2x64pace $data0, $delta0, $triPtr+=, $mzero, 0
  {
    // minus 1 from our count because of the preloading above.
    add $dataSizeD4, $dataSizeD4, -1
    f16v4mix $azeros, $data0, $delta0
  }

    rpt $dataSizeD4, (2f-1f)/8-1
1:
  {
    // load the next values and retrieve the current from the accumulators.
    ld2x64pace $data0, $delta0, $triPtr+=, $mzero, 0
    f16v4mix $data1, $azeros, $azeros
  }
  {
    // store the current result and process the next ones.
    st64pace $data1, $triPtr+=, $mzero, 0
    f16v4mix $azeros, $data0, $delta0
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

  // zero the second half of the $data1 and $delta1 registers because we will
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
    zero $delta1i1
  }
.Lvector2:
  ld32 $data1i0, $data, $offset, 0
  ld32 $delta1i0, $deltas, $offset, 0
  f16v4mix $azeros, $data1, $delta1
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
  ldb16 $delta1i0, $deltas, $offset, 0

  f16v4mix $azeros, $data1, $delta1
  f16v4mix $data1, $azeros, $azeros

  roll16 $data1i0, $ascratch, $data1i0
  swap16 $data1i0, $data1i0
  st32 $data1i0, $data, $offset, 0

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

.size VERTEX_COMMON, .-VERTEX_COMMON

#endif // __IPU__
