#ifdef __IPU__

#define VERTEX __runCodelet_popops__ScaledAdd2D___float

.globl VERTEX
.type VERTEX, @function

// constants
#define VERTEX_DATA_OFFSET 0
#define VERTEX_DATA_SIZE_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
#define VERTEX_K_OFFSET 3

// integer variables
#define outData m0
#define outDataSize m1
#define outDeltas m2
#define data m3
#define dataSize m4
#define dataSizeD2 m5
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

.section .text.VERTEX
.align 8

VERTEX:
  // load vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_OFFSET
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_SIZE_OFFSET
  ld32 $outDeltas, $mvertex_base, $mzero, VERTEX_DELTAS_OFFSET
  ld32 $ascratch, $mvertex_base, $mzero, VERTEX_K_OFFSET

  {
    // minus 1 for the brnzdec
    add $outDataSize, $outDataSize, -1
    // setup $TAS for the f32v2axpy instructions below.
    uput $TAS, $ascratch
  }
.Louter_loop:
  // load inner pointers
  ld32step $data, $mzero, $outData+=, 1
  ld32step $origDataSize, $mzero, $outData+=, 1
  ld32step $deltas, $mzero, $outDeltas+=, 1

  // pack out/in pointers
  tapack $triPtr, $data, $deltas, $data

  // process 2 at a time first as this is the optimal scenario
  shr $dataSizeD2, $origDataSize, 1
  brz $dataSizeD2, .Lvector2_loop_end

  // load the first values and push them into the accumulators.
  ld2x64pace $data0, $delta0, $triPtr+=, $mzero, 0
  {
    // minus 1 from our count because of the preloading above.
    add $dataSizeD2, $dataSizeD2, -1
    f32v2axpy $azeros, $delta0, $data0
  }

  rpt $dataSizeD2, (2f-1f)/8-1
1:
  {
    // load the next values and retrieve the current from the accumulators.
    ld2x64pace $data0, $delta0, $triPtr+=, $mzero, 0
    f32v2axpy $data1, $azeros, $azeros
  }
  {
    // store the current result and process the next ones.
    st64pace $data1, $triPtr+=, $mzero, 0
    f32v2axpy $azeros, $delta0, $data0
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
  // zero the second half of the $data1 and $delta1 registers because we will
  // only be loading into the first half from now on but processing them using
  // a v2 instruction.
  {
    ld32 $data1i0, $data, $offset, 0
    zero $data1i1
  }
  {
    ld32 $delta1i0, $deltas, $offset, 0
    zero $delta1i1
  }
  f32v2axpy $azeros, $delta1, $data1
  f32v2axpy $data1, $azeros, $azeros
  st32step $data1i0, $data, $offset+=, 1

.Lend:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

.size VERTEX, .-VERTEX

#endif // __IPU__
