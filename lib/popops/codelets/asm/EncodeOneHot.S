// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
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

// constants
#define HALF_1_0 0x3C00

// supervisor variables
#define vertexPtr m0
#define indexPtr m1
#define numIndices m2
#define outPtr m3
#define slicePtr m4
#define offsetsPtr m5

#define outPtr_cpy m0
#define offset m6
#define slice m7
#define mscratch m8
#define mscratch2 m9

.section .text.VERTEX
.align 4

VERTEX:
  // one hot encoding is:
  //
  //  memset(out.begin(), 0, outLength * sizeof(OutType));
  //
  // followed by:
  //
  //  for (unsigned i = 0; i < indices.size(); ++i) {
  //    if ((indices[i] >= offsets[i]) &&
  //         (offsets[i] < indices[i] + sliceLength[i])) {
  //      out[idx + indices[i] - offsets[i]] = 1;
  //    }
  //    idx += sliceLength[i];
  //  }
  //
  // We memset in the compute set before this one and offsets is
  // In/Out allowing partial writes to the output vector

  // Load Vertex state
  ld32 $indexPtr, $vertexPtr, $mzero, VERTEX_INDEX_BEGIN_OFFSET
  ld32 $numIndices, $vertexPtr, $mzero, VERTEX_INDEX_SIZE_OFFSET
  ld32 $outPtr, $vertexPtr, $mzero, VERTEX_OUT_OFFSET
  ld32 $slicePtr, $vertexPtr, $mzero, VERTEX_SLICE_OFFSET
  ld32 $offsetsPtr, $vertexPtr, $mzero, VERTEX_OFFSETS_OFFSET

  // minus 1 for brnzdec
  add $numIndices, $numIndices, -1

.Lencode_loop:
    ld32step $slice, $mzero, $slicePtr+=, 1 // 6 cycles on first iteration.

    // use a separate copy as we need to check alignment of write address
    mov $outPtr_cpy, $outPtr

    // move out pointer to next slice before check on whether this index
    // lies within this slice
    ldz16step $mzero, $mzero, $outPtr+=, $slice // 6 cycles

    ld32step $mscratch, $mzero, $indexPtr+=, 1

 #if MASKED_LABEL_CODE == 0xFFFFFFFFU
    // in the case when the index is 0xFFFFFFFF(-1), we can check it by
    // directly adding 1
    add      $mscratch2, $mscratch, 1
 #else
    // This is not handled optimally as we expect this not to be typically used.
    // Ideally, make a spare register for this outside the loop
    ldconst  $mscratch2, MASKED_LABEL_CODE
    cmpne    $mscratch2, $mscratch2, $mscratch
 #endif

    ld32step $offset, $mzero, $offsetsPtr+=, 1

    // ignore if invalid code
    brz      $mscratch2, .Lskip_index

    // Need to write only if index satisfies:
    //   offset <= index < offset + slice

    // index - offset
    sub $mscratch, $mscratch, $offset

    brneg $mscratch, .Lskip_index // take this branch: 27

    // there is a 6 cycle penalty when index is larger than out.size().
    cmpult $mscratch2, $mscratch, $slice
    brz $mscratch2, .Lskip_index // take this branch: 29

    // dummy load to move to index to write to
    ldz16step $mzero, $mzero, $outPtr_cpy+=, $mscratch
    and $mscratch, $outPtr_cpy, 0x2 // 6 cycles
    // Pointer is 32-bit aligned
    sub $outPtr_cpy, $outPtr_cpy, $mscratch // 6 cycles

    setzi $mscratch2, HALF_1_0

    brz $mscratch, .LEven
    ldz16 $mscratch, $mzero, $outPtr_cpy, 0 // 6 cycles
    sort4x16lo $mscratch, $mscratch, $mscratch2 // 6 cycles
    bri .LStore // 6 cycles
.LEven:
    ldz16 $mscratch, $mzero, $outPtr_cpy, 1
    sort4x16lo $mscratch, $mscratch2, $mscratch // 6 cycles
.LStore:
    stm32 $mscratch, $outPtr_cpy, $mzero // 6 cycles if $outPtr_cpy is even.

.Lskip_index:
    // 6 cycle penalty each iteration.
    brnzdec $numIndices, .Lencode_loop

.Lepilogue:
  br $lr // 6 cycles

.size VERTEX, . - VERTEX

#endif // __IPU__
