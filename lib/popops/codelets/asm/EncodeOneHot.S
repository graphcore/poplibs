// Copyright (c) Graphcore Ltd, All rights reserved.
#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplibs_support/TileConstants.hpp"
#include "popops/EncodingConstants.hpp"

#define VERTEX __runCodelet_popops__EncodeOneHot___unsigned_int_half

.globl VERTEX
.type VERTEX, @function

// Field offsets in vertex
#define VERTEX_INDEX_BEGIN_OFFSET 0
#define VERTEX_INDEX_SIZE_OFFSET 1
#define VERTEX_OUT_OFFSET 2
#define VERTEX_SLICE_OFFSET 3
#define VERTEX_OFFSETS_OFFSET 4
#define VERTEX_OUT_LENGTH_OFFSET 5

// Memset vertex fields (bytes)
// Offsets must match those in MemsetSupervisorTemplate.S
#ifdef VECTOR_AVAIL_SCALED_PTR64
#define MEMSET_VERTEX_DST_OFFSET 0
// Aligninfo is explicitly merged with dst info
#define MEMSET_VERTEX_SIZE_OFFSET 4
#define MEMSET_VERTEX_SIZE (2*4)
#else
#define MEMSET_VERTEX_DST_OFFSET 0
#define MEMSET_VERTEX_ALIGNINFO_OFFSET 4
#define MEMSET_VERTEX_SIZE_OFFSET 8
#define MEMSET_VERTEX_SIZE (4*4) //rounded up to a multiple of 8B
#endif

// Stack offsets
#define STACK_MEMSET_WORDS (MEMSET_VERTEX_SIZE/4)
#define STACK_OUT_BEGIN_OFFSET (STACK_MEMSET_WORDS+0)
#define STACK_INDEX_BEGIN_OFFSET (STACK_MEMSET_WORDS+1)
#define STACK_INDEX_SIZE_OFFSET (STACK_MEMSET_WORDS+2)
#define STACK_SLICE_OFFSET (STACK_MEMSET_WORDS+3)
#define STACK_LR_OFFSET (STACK_MEMSET_WORDS+4)
#define STACK_OFFSETS (STACK_MEMSET_WORDS+5)
#define STACK_CALLER_SAVE_OFFSET (STACK_MEMSET_WORDS+6)

// Total stack must be a multiple of 8 bytes
#define STACK_SIZE (8*4)

// constants
#define LOG2_SIZEOF_HALF 1
#define LOG2_SIZEOF_UNSIGNED 2

// supervisor variables
#define vertexPtr m0
#define remainder m1
#define ignore_code m2
#define wordsPerWorker m2
#define trailingBytes m3
#define dst m4
#define size m5
#define outBegin m6
#define index m7
#define slice m8
#define offsets_ptr m1
#define outBegin_cpy m9
#define mscratch m0

.section .text.VERTEX
.align 4

VERTEX:
  // one hot encoding is:
  //
  //  memset(out.begin(), 0, outLength * sizeof(OutType));
  //  for (unsigned i = 0; i < indices.size(); ++i) {
  //    if (indices[i] >= offsets[i] &&
  //      (offsets[i] < indices[i] + sliceLength[i])) {
  //      out[idx + indices[i] - offsets[i]] = 1;
  //    }
  //    idx += sliceLength[i];
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

  // TODO: T12918 Optimise instruction ordering to avoid stalls.

  ld32 $size, $vertexPtr, $mzero, VERTEX_INDEX_SIZE_OFFSET
  ld32 $index, $vertexPtr, $mzero, VERTEX_INDEX_BEGIN_OFFSET
  ld32 $outBegin, $vertexPtr, $mzero, VERTEX_OUT_OFFSET
  ld32 $offsets_ptr, $vertexPtr, $mzero, VERTEX_OFFSETS_OFFSET

  // load the index from our vertex state.

  ld32 $slice, $vertexPtr, $mzero, VERTEX_SLICE_OFFSET

  // initialise some constants used below now while waiting for loads.
  setzi $wordsPerWorker, 0xAAAB

  // make some space on our stack
  add $sp, $sp, -(MEMSET_VERTEX_SIZE + STACK_SIZE)

  // store number of indices now before we turn it into number of bytes in
  // the output tensor.
  st32 $size, $sp, $mzero, STACK_INDEX_SIZE_OFFSET // 6 cycles
  ld32 $size, $vertexPtr, $mzero, VERTEX_OUT_LENGTH_OFFSET
  shl $size, $size, LOG2_SIZEOF_HALF // 6 cycles

  // save the index before we run the memset vertex.
  st32 $index, $sp, $mzero, STACK_INDEX_BEGIN_OFFSET

  // begin saving our vertex state to the stack because this will get trashed by
  // the memset. we'll save $index later to avoid the load latency penalty.
  st32 $lr, $sp, $mzero, STACK_LR_OFFSET
  st32 $m9, $sp, $mzero, STACK_CALLER_SAVE_OFFSET

  st32 $outBegin, $sp, $mzero, STACK_OUT_BEGIN_OFFSET

  st32 $slice, $sp, $mzero, STACK_SLICE_OFFSET

  st32 $offsets_ptr, $sp, $mzero, STACK_OFFSETS

  // wordsPerWorker = size / (8*6) = (size * 0xAAAB) >> 21
  // see here for how these constants were derived:
  //   https://embeddedgurus.com/stack-overflow/2009/06/division-of-integers-by-constants/
  // NOTE: the maximum size that can be handled is ~96000 because at that point
  // log2(n * 0xAAAB) > 32 and we lose bits before we shift the answer in.
  mul $wordsPerWorker, $size, $wordsPerWorker
  shr $wordsPerWorker, $wordsPerWorker, 21 // 6 cycles

  // remainder = (size/8) - (wordsPerWorker*6)
  shr $remainder, $size, 3
  mul $mscratch, $wordsPerWorker, 6
  sub $remainder, $remainder, $mscratch // 6 cycles

  // numBytesPreAlignment and alignment are both zero therefore alignmentInfo
  // is just trailingBytes << 6.
  // trailingBytes = size & 0x7
  and $trailingBytes, $size, 0x7

  shl $size, $wordsPerWorker, 3
  or $size, $size, $remainder // 6 cycles
  shl $trailingBytes, $trailingBytes, 6

#ifdef VECTOR_AVAIL_SCALED_PTR64
  // turn $outBegin into a scaled pointer
  shr $dst, $outBegin, 3 // 1 cycle

  // pack $dst and $alignmentInfo into 32-bits to store in the vertex state.
  sort4x16lo $dst, $dst, $trailingBytes // 6 cycles

  // create the memset zero state.
  st32 $dst, $sp, $mzero, MEMSET_VERTEX_DST_OFFSET/4 // 6 cycles
  st32 $size, $sp, $mzero, MEMSET_VERTEX_SIZE_OFFSET/4
#else
  st32 $outBegin, $sp, $mzero, MEMSET_VERTEX_DST_OFFSET/4
  st32 $trailingBytes, $sp, $mzero, MEMSET_VERTEX_ALIGNINFO_OFFSET/4 // 6 cycles
  st32 $size, $sp, $mzero, MEMSET_VERTEX_SIZE_OFFSET/4
#endif

  // run the vertex ($m0 has already been set).
  // prep $m0 as the vertex state parameter that will be passed to the memset.
  mov $m0, $sp
  call $lr, __runCodelet_poplar_rt__MemsetZeroSupervisor // 6 cycles

  // retrieve our vertex state and perform the one hot encoding.
  ld32 $size, $sp, $mzero, STACK_INDEX_SIZE_OFFSET
  ld32 $offsets_ptr, $sp, $mzero, STACK_OFFSETS
  ld32 $index, $sp, $mzero, STACK_INDEX_BEGIN_OFFSET
  ld32 $slice, $sp, $mzero, STACK_SLICE_OFFSET
  ld32 $lr, $sp, $mzero, STACK_LR_OFFSET
  ld32 $outBegin, $sp, $mzero, STACK_OUT_BEGIN_OFFSET

  // avoid register bubble
  nop

  // registers m0, m2, m3 and m4 are available to use for encoding.

  // minus 1 for brnzdec
  add $size, $size, -1

.Lencode_loop:
    ld32step $m4, $mzero, $slice+=, 1 // 6 cycles on first iteration.

    // use a separate copy as we need to check alignment of write address
    mov $outBegin_cpy, $outBegin

    // move out pointer to next slice before check on whether this index
    // lies within this slice
    ldz16step $mzero, $mzero, $outBegin+=, $m4 // 6 cycles

    ld32step $m3, $mzero, $index+=, 1
 #if MASKED_LABEL_CODE == 0xFFFFFFFFU
    // in the case when the index is 0xFFFFFFFF(-1), we can check it by
    // directly adding 1
    add      $ignore_code, $m3, 1
 #else
    // This is not handled optimally as we expect this not to be typically used.
    // Ideally, make a spare register for this outside the loop
    ldconst  $ignore_code, MASKED_LABEL_CODE
    cmpne    $ignore_code, $ignore_code, $m3

 #endif
    ld32step $m0, $mzero, $offsets_ptr+=, 1

    // ignore if invalid code
    brz      $ignore_code, .Lskip_index


    // Need to write only if index satisfies:
    //   offset <= index < offset + slice

    // index - offset
    sub $m3, $m3, $m0

    brneg $m3, .Lskip_index // take this branch: 27

    // there is a 6 cycle penalty when index is larger than out.size().
    cmpult $m2, $m3, $m4
    brz $m2, .Lskip_index // take this branch: 29

    // dummy load to move to index to write to
    ldz16step $mzero, $mzero, $outBegin_cpy+=, $m3
    and $m2, $outBegin_cpy, 0x2 // 6 cycles
    // Pointer is 32-bit aligned
    sub $outBegin_cpy, $outBegin_cpy, $m2 // 6 cycles
    setzi $m0, 0x3c00

    // TODO: T12918 Reduce bubbles.
    brz $m2, .LEven
    ldz16 $m4, $mzero, $outBegin_cpy, 0 // 6 cycles
    sort4x16lo $m4, $m4, $m0 // 6 cycles
    bri .LStore // 6 cycles
.LEven:
    ldz16 $m4, $mzero, $outBegin_cpy, 1
    sort4x16lo $m4, $m0, $m4 // 6 cycles
.LStore:
    stm32 $m4, $outBegin_cpy, $mzero // 6 cycles if $outBegin_cpy is even.

.Lskip_index:
    // 6 cycle penalty each iteration.
    brnzdec $size, .Lencode_loop

.Lepilogue:
  // restore stack pointer.
  ld32 $m9, $sp, $mzero, STACK_CALLER_SAVE_OFFSET
  add $sp, $sp, MEMSET_VERTEX_SIZE + STACK_SIZE
  br $lr // 6 cycles

// size is used as a register name above.
#undef size

.size VERTEX, . - VERTEX

#endif // __IPU__
