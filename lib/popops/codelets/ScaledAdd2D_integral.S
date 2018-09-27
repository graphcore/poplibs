#ifdef __IPU__

#define VERTEX(ty) __runCodelet_popops__ScaledAdd2D___ ## ty

.globl VERTEX(int)
.type VERTEX(int), @function

.globl VERTEX(unsigned_int)
.type VERTEX(unsigned_int), @function

// constants
#define VERTEX_DATA_OFFSET 0
#define VERTEX_DATA_SIZE_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
#define VERTEX_K_OFFSET 3

// integer variables
#define outData m0
#define outDataSize m1
#define outDeltas m2
#define k m3
#define data m4
#define dataSize m5
#define deltas m6
#define datum m7
#define delta m8

.section .text.VERTEX(int)
.align 8

  nop
VERTEX(int):
VERTEX(unsigned_int):
  // load vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_OFFSET
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_SIZE_OFFSET
  ld32 $outDeltas, $mvertex_base, $mzero, VERTEX_DELTAS_OFFSET
  ld32 $k, $mvertex_base, $mzero, VERTEX_K_OFFSET

  // minus 1 for the brnzdec
  add $outDataSize, $outDataSize, -1
.Louter_loop:
  // load inner pointers
  ld32step $data, $mzero, $outData+=, 1
  ld32step $dataSize, $mzero, $outData+=, 1
  ld32step $deltas, $mzero, $outDeltas+=, 1

  rpt $dataSize, (.Linner_loop_end - .Linner_loop_begin)/8-1
.Linner_loop_begin:
  {
    ld32 $datum, $mzero, $data, 0
    fnop
  }
  {
    ld32step $delta, $mzero, $deltas+=, 1
    fnop
  }
  {
    mul $delta, $delta, $k
    fnop
  }
  {
    add $datum, $datum, $delta
    fnop
  }
  {
    st32step $datum, $mzero, $data+=, 1
    fnop
  }
.Linner_loop_end:
  brnzdec $outDataSize, .Louter_loop
  exitz $mzero

.size VERTEX(int), .-VERTEX(int)

#endif // __IPU__
