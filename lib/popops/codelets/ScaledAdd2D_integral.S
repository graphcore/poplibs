#ifdef __IPU__

#define VERTEX(ty) __runCodelet_popops__ScaledAdd2D___ ## ty
#define VERTEX_COMMON __ScaledAdd2D___
.globl VERTEX(int_true)
.type VERTEX(int_true), @function

.globl VERTEX(unsigned_int_true)
.type VERTEX(unsigned_int_true), @function


.globl VERTEX(int_false)
.type VERTEX(int_false), @function

.globl VERTEX(unsigned_int_false)
.type VERTEX(unsigned_int_false), @function

// constants
#define VERTEX_DATA_OFFSET 0
#define VERTEX_DATA_SIZE_OFFSET 1
#define VERTEX_DELTAS_OFFSET 2
// 2 versions - one with K as a constant, the other using a tensor "factor"
#define VERTEX_K_OFFSET 3
#define VERTEX_FACTOR_OFFSET 3

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

.section .text.VERTEX(int_true)
.align 4

VERTEX(int_true):
VERTEX(unsigned_int_true):
  // load vertex state specific to the case where we multiply by a constant, k
  ld32 $k, $mvertex_base, $mzero, VERTEX_K_OFFSET
  bri  VERTEX_COMMON
.size VERTEX(int_true), .-VERTEX(int_true)

.section .text.VERTEX(int_false)
.align 4

VERTEX(int_false):
VERTEX(unsigned_int_false):
  // load vertex state specific to the case where we multiply by a tensor, factor
  ld32 $k, $mvertex_base, $mzero, VERTEX_FACTOR_OFFSET
  ld32 $k, $mzero, $k, 0
  bri  VERTEX_COMMON
.size VERTEX(int_false), .-VERTEX(int_false)


.section .text.VERTEX_COMMON
.align 8
VERTEX_COMMON:
  // load common vertex state
  ld32 $outData, $mvertex_base, $mzero, VERTEX_DATA_OFFSET
  ld32 $outDataSize, $mvertex_base, $mzero, VERTEX_DATA_SIZE_OFFSET
  ld32 $outDeltas, $mvertex_base, $mzero, VERTEX_DELTAS_OFFSET

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

.size VERTEX_COMMON, .-VERTEX_COMMON

#endif // __IPU__
