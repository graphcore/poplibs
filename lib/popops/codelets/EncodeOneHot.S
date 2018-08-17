#ifdef __IPU__

#include "tileimplconsts_tommy.h"

#define VERTEX __runCodelet_popops__EncodeOneHot___unsigned_int_half

.globl VERTEX
.type VERTEX, @function

// constants
#define VERTEX_INDEX_BEGIN_OFFSET 0
#define VERTEX_INDEX_END_OFFSET 1
#define VERTEX_OUT_OFFSET 2
#define VERTEX_SLICE_OFFSET 3

#define MEMSET_VERTEX_DST_OFFSET 0
#define MEMSET_VERTEX_SIZE_OFFSET 1

#define STACK_OUT_BEGIN_OFFSET 2
#define STACK_INDEX_BEGIN_OFFSET 3
#define STACK_INDEX_SIZE_OFFSET 4
#define STACK_SLICE_OFFSET 5
#define STACK_LR_OFFSET 6

#define MEMSET_VERTEX_SIZE (2*4)
#define STACK_SIZE (5*4)

#define LOG2_SIZEOF_HALF 1
#define LOG2_SIZEOF_UNSIGNED 2

// supervisor variables
#define vertexPtr m0
#define remainder m1
#define wordsPerWorker m2
#define trailingBytes m3
#define dst m4
#define size m5
#define outBegin m6
#define index m7
#define slice m8

#define mscratch m0

.section .text.VERTEX
.align 4

VERTEX:
  // one hot encoding is:
  //
  //  memset(out.begin(), 0, indices.size() * slice * sizeof(OutType));
  //  for (unsigned i = 0; i < indices.size(); ++i) {
  //    out[i * slice + indices[i]] = 1;
  //  }
  //
  // therefore we delegate to the MemsetZeroSupervisor codelet to do the bulk
  // of the work and then do the encoding ourselves on the supervisor after.

  // the MemsetZeroSupervisor vertex has the following vertex state:
  //  ScaledPtr dst;
  //  unsigned short alignmentInfo;
  //  unsigned size;
  // where
  //  alignmentInfo is a bitfield:
  //   [0:2] numBytesPreAlignment
  //   [3:5] alignment (dst & 0x7)
  //   [6:]  trailingBytes
  //  size is a bitfield:
  //   [0:2] remainder
  //   [3:]  wordsPerWorker
  //  because the encode codelet is 8-byte aligned most of these are pretty
  //  easy to derive.

  // TODO: optimise instruction ordering to avoid stalls.

  ld32 $outBegin, $vertexPtr, $mzero, VERTEX_OUT_OFFSET
  ld32 $size, $vertexPtr, $mzero, VERTEX_INDEX_END_OFFSET

  // load the index from our vertex state.
  ld32 $index, $vertexPtr, $mzero, VERTEX_INDEX_BEGIN_OFFSET

  ld32 $slice, $vertexPtr, $mzero, VERTEX_SLICE_OFFSET

  // initialise some constants used below now while waiting for loads.
  setzi $wordsPerWorker, 0xAAAB

  // make some space on our stack
  add $sp, $sp, -(MEMSET_VERTEX_SIZE + STACK_SIZE)

  // calculate size in bytes by doing:
  //  (end - begin)/sizeof(IndexType) * slice * sizeof(OutType)
  sub $size, $size, $index
  shr $size, $size, LOG2_SIZEOF_UNSIGNED

  // store number of indices now before we turn it into number of bytes in
  // the output tensor.
  st32 $size, $sp, $mzero, STACK_INDEX_SIZE_OFFSET

  mul $size, $size, $slice
  shl $size, $size, LOG2_SIZEOF_HALF

  // save the index before we run the memset vertex.
  st32 $index, $sp, $mzero, STACK_INDEX_BEGIN_OFFSET

  // begin saving our vertex state to the stack because this will get trashed by
  // the memset. we'll save $index later to avoid the load latency penalty.
  st32 $lr, $sp, $mzero, STACK_LR_OFFSET

  st32 $outBegin, $sp, $mzero, STACK_OUT_BEGIN_OFFSET

  st32 $slice, $sp, $mzero, STACK_SLICE_OFFSET

  // wordsPerWorker = size / (8*6) = (size * 0xAAAB) >> 21
  // see here for how these constants were derived: 
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  // NOTE: the maximum size that can be handled is ~96000 because at that point
  // log2(n * 0xAAAB) > 32 and we lose bits before we shift the answer in.
  mul $wordsPerWorker, $size, $wordsPerWorker
  shr $wordsPerWorker, $wordsPerWorker, 21

  // remainder = (size/8) - (wordsPerWorker*6)
  shr $remainder, $size, 3
  mul $mscratch, $wordsPerWorker, 6
  sub $remainder, $remainder, $mscratch

  // numBytesPreAlignment and alignment are both zero therefore alignmentInfo
  // is just trailingBytes << 6.
  // trailingBytes = size & 0x7
  and $trailingBytes, $size, 0x7

  shl $size, $wordsPerWorker, 3
  or $size, $size, $remainder
  shl $trailingBytes, $trailingBytes, 6

  // turn $outBegin into a scaled pointer
  setzi $dst, TMEM_REGION0_BASE_ADDR
  sub $dst, $outBegin, $dst
  shr $dst, $dst, 2

  // pack $dst and $alignmentInfo into 32-bits to store in the vertex state.
  sort4x16lo $dst, $dst, $trailingBytes

  // create the memset zero state.
  st32 $dst, $sp, $mzero, MEMSET_VERTEX_DST_OFFSET
  st32 $size, $sp, $mzero, MEMSET_VERTEX_SIZE_OFFSET

  // run the vertex ($m0 has already been set).
  // prep $m0 as the vertex state parameter that will be passed to the memset.
  mov $m0, $sp
  call $lr, __runCodelet_poplar_rt__MemsetZeroSupervisor

  // retrieve our vertex state and perform the one hot encoding.
  ld32 $size, $sp, $mzero, STACK_INDEX_SIZE_OFFSET
  ld32 $index, $sp, $mzero, STACK_INDEX_BEGIN_OFFSET
  ld32 $slice, $sp, $mzero, STACK_SLICE_OFFSET
  ld32 $lr, $sp, $mzero, STACK_LR_OFFSET
  ld32 $outBegin, $sp, $mzero, STACK_OUT_BEGIN_OFFSET

  // registers m0, m1, m2, m3 and m4 are available to use for encoding.

  // minus 1 for brnzdec
  add $size, $size, -1

  // keep a register with slice/2 because when we do the 32-bit store that is
  // offset by slice we must treat it like a 32-bit offset rather than 16-bit.
  shr $m4, $slice, 1

.Lencode_loop:
  // load the next index.
  ld32 $m3, $index, $mzero, $size

  setzi $m0, 0x00003c00
  shl $m1, $m0, 16

  // there is a 6 cycle penalty when index is larger than out.size().
  cmpult $m2, $m3, $slice
  brz $m2, .Lskip_index

  // if index is even we want to write [1, 0], otherwise we want to write [0, 1]
  // so, firstly check to see if index is even or odd
  and $m2, $m3, 0x1

  // when index is odd we want to swap the values round in our register.
  movz $m0, $m2, $m1

  // store it to index/2 because index refers to every 16-bits.
  shr $m3, $m3, 1
  mul $m2, $m4, $size
  add $m3, $m3, $m2
  stm32 $m0, $outBegin, $m3

  // 6 cycle penalty each iteration.
  brnzdec $size, .Lencode_loop

.Lepilogue:
  // restore stack pointer.
  add $sp, $sp, MEMSET_VERTEX_SIZE + STACK_SIZE
  br $lr

.Lskip_index:
  brnzdec $size, .Lencode_loop

#endif // __IPU__
