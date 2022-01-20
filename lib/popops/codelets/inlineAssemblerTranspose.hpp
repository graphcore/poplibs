// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

template <typename T>
static inline void transpose1(const T *srcPtr, T *dstPtr, unsigned numSrcRows,
                              unsigned numSrcColumns) {
  for (unsigned x = 0; x != numSrcColumns; ++x) {
    for (unsigned y = 0; y != numSrcRows; ++y) {
      dstPtr[x * numSrcRows + y] = srcPtr[y * numSrcColumns + x];
    }
  }
}

template <typename T>
static void transposeRowsColumnsFast(const T *srcPtr, T *dstPtr,
                                     unsigned numSrcRowsD4Or8,
                                     unsigned numSrcColumnsD4Or8) {
  constexpr unsigned subTransposeSize = 4;
  auto numSrcRows = numSrcRowsD4Or8 * subTransposeSize;
  auto numSrcColumns = numSrcColumnsD4Or8 * subTransposeSize;
  transpose1(srcPtr, dstPtr, numSrcRows, numSrcColumns);
}

template <typename T>
static void transposeRowsColumnsFast(const T *srcPtr, T *dstPtr,
                                     unsigned numSrcRowsD4Or8,
                                     unsigned numSrcColumnsD4Or8,
                                     unsigned allocatedRowsD4Or8,
                                     unsigned allocatedColumnsD4Or8) {
  constexpr unsigned subTransposeSize = 4;
  auto numSrcRows = numSrcRowsD4Or8 * subTransposeSize;
  auto numSrcColumns = numSrcColumnsD4Or8 * subTransposeSize;
  auto allocatedRows = allocatedRowsD4Or8 * subTransposeSize;
  auto allocatedColumns = allocatedColumnsD4Or8 * subTransposeSize;
  for (unsigned x = 0; x != allocatedRows; ++x) {
    for (unsigned y = 0; y != allocatedColumns; ++y) {
      dstPtr[y * numSrcRows + x] = srcPtr[x * numSrcColumns + y];
    }
  }
}

template <typename T>
static void transposeRowsColumns(const T *srcPtr, T *dstPtr,
                                 unsigned numSrcRows, unsigned numSrcColumns) {
  transpose1(srcPtr, dstPtr, numSrcRows, numSrcColumns);
}

#if __IPU_ARCH_VERSION__ == 21

static void transpose8(const quarter *srcPtr, quarter *dstPtr,
                       unsigned numSrcRowsD8, unsigned numSrcColumnsD8,
                       unsigned allocatedRowsD8, unsigned allocatedColumnsD8) {
  // Doing an 8x8 transpose.  1 per loop
  constexpr unsigned innerLoopTransposeSize = 8;

  const int inRowStride = 2 * numSrcColumnsD8;
  const int inBackStride = 1 - (innerLoopTransposeSize - 1) * inRowStride;

  const int outRowStride = numSrcRowsD8;
  const int outBackStride = 1 - (innerLoopTransposeSize / 2 - 1) * outRowStride;

  for (unsigned i = 0; i < allocatedColumnsD8; i++) {
    auto srcPtrInner = &srcPtr[i * innerLoopTransposeSize];
    auto dstPtrInner = &dstPtr[innerLoopTransposeSize * numSrcRowsD8 *
                               (i * innerLoopTransposeSize + 2)];

    // Inner loop - Input - 8x8 bits by 8x8 bits
    // Unrolled, so we see 2 transpositions
    //
    // a0 b0 c0 d0 e0 f0 g0 h0  ... multiple of 8x 8 bits to stride over
    // a1 b1 c1 d1 e1 f1 g1 h1  ...
    // a2 b2 c2 d2 e2 f2 g2 h2  ...
    // a3 b3 c3 d3 e3 f3 g3 h3  ...
    // a4 b4 c4 d4 e4 f4 g4 h4  ...
    // a5 b5 c5 d5 e5 f5 g5 h5  ...
    // a6 b6 c6 d6 e6 f6 g6 h6  ...
    // a7 b7 c7 d7 e7 f7 g7 h7  ...
    //  <---------------------- Input continues here for next loop pass
    ///
    // Read order is [a0 b0 c0 d0], [a1 b1 c1 d1] in 2 32 bit reads
    // ...
    // [a6 b6 c6 d6], [a7 b7 c7 d7]
    // [e0 f0 g0 h0], [e1 f1 g1 h1]
    // Each pair of reads is permuted (8 bit rearrangement)
    // and then passed into the accumulators for a 4x4 16 bit transpose
    //
    // The output is
    // a0 a1 a2 a3 a4 a5 a6 a7  ... multiple of 8x 8 bits to stride over
    // b0 b1 b2 b3 b4 b5 b6 b7  ...
    // c0 c1 c2 c3 c4 c5 c6 c7  <---Output continues here for next loop pass
    // d0 d1 d2 d3 d4 d5 d6 d7  ...
    // e0 e1 e2 e3 e4 e5 e6 e7  ...
    // f0 f1 f2 f3 f4 f5 f6 f7  ...
    // g0 g1 g2 g3 g4 g5 g6 g7  ...
    // h0 h1 h2 h3 h4 h5 h6 h7  ...
    //
    // Write order is [c0 c1 c2 c3 c4 c5 c6 c7] in 1 64 bit write
    // [d0 d1 d2 d3 d4 d5 d6 d7],
    // [a0 a1 a2 a3 a4 a5 a6 a7],
    // [b0 b1 b2 b3 b4 b5 b6 b7],
    // Followed by the rows with letters g, h, e, f
    //
    // We can refer to the inputs as row-0 row-1 ...
    // The outputs as row-a row-b ...

    asm volatile(
        R"l(
              // Preamble: prime the accumulators (No output yet)

              // Load [a0 b0 c0 d0], [a1 b1 c1 d1]
              ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride]
              ld32step    $a5, $mzero, %[srcPtr]+=, %[inRowStride]

              // Load [a2 b2 c2 d2], [a3 b3 c3 d3]
              // $a0, $a5 permute into $a4:5
              {ld32step   $a2, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8hi  $a4, $a0, $a5}
              {ld32step   $a3, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8lo  $a5, $a0, $a5}
              
              // Load [a4 b4 c4 d4], [a5 b5 c5 d5]
              // $a2, $a3 permute into $a6:7
              {ld32step   $a0, $mzero, %[srcPtr]+=, %[inRowStride] 
               shuf8x8hi  $a6, $a2, $a3}
              {ld32step   $a1, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8lo  $a7, $a2, $a3}

              // First 2*64 bit inputs to the accumulators - no result yet
              // These are the permuted results from the 1st 32bits of rows:
              // [row-0, row-1, row-2, row-3]
              f16v4istacc $azeros, $a4:5, $a6:7, %[TISTACC_P0]

              // Load [a6 b6 c6 d6], [a7 b7 c7 d7]
              // $a0, $a1 permute into $a4:5
              {ld32step   $a2, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8hi  $a4, $a0, $a1}
              {ld32step   $a3, $mzero, %[srcPtr]+=, %[inBackStride]
               shuf8x8lo  $a5, $a0, $a1}

              // $a2, $a3 permute into $a6:7
              shuf8x8hi   $a6, $a2, $a3
              shuf8x8lo   $a7, $a2, $a3
               
              // Load [e0 f0 g0 h0] 
              // Second 2*64 bit inputs to the accumulators - no result yet
              // These are the permuted results from the 1st 32bits of rows:
              // [row-4, row-5, row-6, row-7]
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride] 
               f16v4istacc $azeros, $a4:5, $a6:7, %[TISTACC_P1]}
              
            .align 8   
              // Repeat and get row-c result [c0 c1 c2 c3 c4 c5 c6 c7]      
              {rpt %[loops], (2f - 1f) / 8 - 1
               f16v4stacc  $a6:7, %[TSTACC_P0]}
            1:
              // Load [e1 f1 g1 h1]
              // Get row-d output
              {ld32step    $a5, $mzero, %[srcPtr]+=, %[inRowStride] 
               f16v4stacc  $a2:3, %[TSTACC_P1]}

              // Load [e2 f2 g2 h2], [e3 f3 g3 h3]
              // $a0, $a5 permute into $a:5
              {ld32step    $a1, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8hi   $a4, $a0, $a5}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8lo   $a5, $a0, $a5}
              
              // Load [e4 f4 g4 h4]
              // Store row-c
              // $a1, $a0 permute into $a6:7
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStride]
               shuf8x8hi   $a6, $a1, $a0}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride] 
               shuf8x8lo   $a7, $a1, $a0}


              // Load [e5 f5 g5 h5]
              // Accumulators - write the 2nd 32bits of rows:
              // [row-0, row-1, row-2, row-3]     
              // Get row-a output
              {ld32step    $a5, $mzero, %[srcPtr]+=, %[inRowStride]
               f16v4istacc $a6:7, $a4:5, $a6:7, %[TISTACC_P0]}
 
              // Load [e6 f6 g6 h6], [e7 f7 g7 h7] 
              // $a0, $a5 permute into $a4:5
              {ld32step    $a1, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8hi   $a4, $a0, $a5}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStridem1]
               shuf8x8lo   $a5, $a0, $a5}
 
              // Store row-d, row-a
              // $a1, $a0 permute into $a2:3
              {st64step    $a2:3, $mzero, %[dstPtr]+=, %[outRowStridexm3]
               shuf8x8hi   $a2, $a1, $a0}
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStride]
               shuf8x8lo   $a3, $a1, $a0}
            
    
              // Load [a8 b8 c8 d8] (For next transposition)
              // Accumulators - write the 2nd 32bits of rows:
              // [row-4, row-5, row-6, row-7]     
              // Get row-b output
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride]
               f16v4istacc $a6:7, $a4:5, $a2:3, %[TISTACC_P1]}

              // Load [a9 b8 c9 d9]
              // Store row-b
              // Get row-g, row-h outputs
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStridex5]
               f16v4stacc  $a6:7, %[TSTACC_P0]}
              {ld32step    $a5, $mzero, %[srcPtr]+=, %[inRowStride]
               f16v4stacc  $a2:3, %[TSTACC_P1]}

              // Load [a10 b10 c10 d10] [a11 b11 c11 d11] 
              // $a0, $a5 permute into $a4:5
              {ld32step    $a1, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8hi   $a4, $a0, $a5}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8lo   $a5, $a0, $a5}
         
              // Load [a12 b12 c12 d12]
              // $a1, $a0 permute into $a6:7
              // Store row-g
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStride] 
               shuf8x8hi   $a6, $a1, $a0}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride] 
               shuf8x8lo   $a7, $a1, $a0}

              // Accumulators - write for the next transposition:
              // [row-8, row-9, row-10, row-11]     
              // Get row-e output
              // Load [a13 b13 c13 d13]
              {ld32step    $a5, $mzero, %[srcPtr]+=, %[inRowStride]
               f16v4istacc $a6:7, $a4:5, $a6:7, %[TISTACC_P0]}     
 
              // Load [a14 b14 c14 d14], [a15 b15 c15 d15]
              // $a0, $a5 permute into $a4:5
              {ld32step    $a1, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8hi   $a4, $a0, $a5}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inBackStride]
               shuf8x8lo   $a5, $a0, $a5}


              // $a1, $a0 permute into $a2:3
              // Store row-h and row-e
              {st64step    $a2:3, $mzero, %[dstPtr]+=, %[outRowStridexm3]
               shuf8x8hi   $a2, $a1, $a0}
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStride]
               shuf8x8lo   $a3, $a1, $a0}

              // Load [e8 f8 g8 h8]
              // Store row-f
              // Accumulators - write for the next transposition:
              // [row-12, row-13, row-14, row-15]     
              // Get row-f, row-k outputs
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride] 
               f16v4istacc $a2:3, $a4:5, $a2:3, %[TISTACC_P1]}
              {st64step    $a2:3, $mzero, %[dstPtr]+=, %[outBackStride]
               f16v4stacc  $a6:7, %[TSTACC_P0]}
            2:
              // Load [e9 f9 g9 h9]            
              // Get row-l output
              {ld32step    $a5, $mzero, %[srcPtr]+=, %[inRowStride]    
               f16v4stacc  $a2:3, %[TSTACC_P1]}
              
              // Load [e10 f10 g10 h10], [e11 f11 g11 h11]
              // $a0, $a5 permute into $a4:5
              {ld32step    $a1, $mzero, %[srcPtr]+=, %[inRowStride] 
               shuf8x8hi   $a4, $a0, $a5}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride] 
               shuf8x8lo   $a5, $a0, $a5}

              // Load [e12 f12 g12 h12]
              // Store row-k (2nd transposition)
              // $a1, $a0 permute into $a6:7
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStride] 
               shuf8x8hi   $a6, $a1, $a0}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8lo   $a7, $a1, $a0}

              // Load [e13 f13 g13 h13]
              // Accumulators - write for the next transposition:
              // [row-8, row-9, row-10, row-11] 2nd half     
              // Get row-i output
              {ld32step    $a5, $mzero, %[srcPtr]+=, %[inRowStride]
               f16v4istacc $a6:7, $a4:5, $a6:7, %[TISTACC_P0]}

              // Load [e14 f14 g14 h14], [e15 f15 g15 h15]
              // $a0, $a5 permute into $a4:5
              {ld32step    $a1, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8hi   $a4, $a0, $a5}
              {ld32step    $a0, $mzero, %[srcPtr]+=, %[inRowStride]
               shuf8x8lo   $a5, $a0, $a5}

              // Store row-l, row-i (2nd transposition)
              // $a1, $a0 permute into $a2:3
              {st64step    $a2:3, $mzero, %[dstPtr]+=, %[outRowStridexm3]
               shuf8x8hi   $a2, $a1, $a0}
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStride]
               shuf8x8lo   $a3, $a1, $a0}

              // Accumulators - write for the next transposition:
              // [row-12, row-13, row-14, row-15] 2nd half     
              // Get row-j output
              f16v4istacc  $a6:7, $a4:5, $a2:3, %[TISTACC_P1]
              
              // Flush the last 4*64 bits through with no more inputs
              // Store row-j
              // Get row-o
              {st64step    $a6:7, $mzero, %[dstPtr]+=, %[outRowStridex5]
               f16v4stacc  $a0:1, %[TSTACC_P0]}

              // Store row-o
              // Get row-p
              {st64step    $a0:1, $mzero, %[dstPtr]+=, %[outRowStride] 
               f16v4stacc  $a2:3, %[TSTACC_P1]}

              // Store row-p
              // Get row-m
              {st64step    $a2:3, $mzero, %[dstPtr]+=, %[outRowStridexm3]
               f16v4istacc $a4:5, $azeros, $azeros, %[TISTACC_P0]}

              // Store row-m
              // Get row-n
              {st64step    $a4:5, $mzero, %[dstPtr]+=, %[outRowStride]
               f16v4istacc $a6:7, $azeros, $azeros, %[TISTACC_P1]}

               // Store row-n
               st64         $a6:7, $mzero, %[dstPtr], 0
            )l"
        : [srcPtr] "+r"(srcPtrInner), [dstPtr] "+r"(dstPtrInner)
        : [loops] "r"(allocatedRowsD8 - 1), [inBackStride] "r"(inBackStride),
          [inRowStride] "r"(inRowStride), [inRowStridem1] "r"(inRowStride - 1),
          [outRowStride] "r"(outRowStride),
          [outRowStridexm3] "r"(outRowStride * -3),
          [outRowStridex5] "r"(outRowStride * 5),
          [outBackStride] "r"(outBackStride), [TISTACC_P0] "i"(TISTACC_P0),
          [TISTACC_P1] "i"(TISTACC_P1), [TSTACC_P0] "i"(TSTACC_P0),
          [TSTACC_P1] "i"(TSTACC_P1)
        : "memory", "$a0:1", "$a2:3", "$a4:5", "$a6:7");
  }
}
static void transpose4(const quarter *srcPtr, quarter *dstPtr,
                       unsigned numSrcRowsD4, unsigned numSrcColumnsD4) {

  constexpr unsigned innerLoopTransposeSize = 4;
  const int outBackStride = 1 - (innerLoopTransposeSize - 1) * numSrcRowsD4;

  for (unsigned i = 0; i < numSrcColumnsD4; i++) {
    auto srcPtrInner = &srcPtr[i * innerLoopTransposeSize];
    auto dstPtrInner = &dstPtr[i * numSrcRowsD4 * innerLoopTransposeSize *
                               innerLoopTransposeSize];

    float in0, in1, in2, out0, out1, out2, out3, out4;
    // 4x4 transpose.  Reading and wrting 32 bits at a time we can
    // perform:
    // Stage1: 4 8-bit permutations (shuf8x8hi and lo)
    // Stage2: 4 16 bit permutations on the result of stage1 (sort4x16hi and lo)
    // which completes a transposition
    //
    // With the input:
    // a0 b0 c0 d0  ... Multiple of 4 elements in columns that we stride over
    // a1 b1 c1 d1
    // a2 b2 c2 d2
    // a3 b3 c4 d3
    // <------------Next input begins here
    // Reading simply progresses reading the first 4 columns.
    //
    // Output:
    // a0 a1 a2 a3 <------------Next output begins here [a4 a5 a6 a7]
    // b0 b1 b2 b3
    // c0 c1 c2 c3
    // d0 d1 d2 d3
    // Writing folows the 1st 4 output columns down, then strides back to the
    // location marked (A stride of back 3 rows less 4 elements)
    asm volatile(
        R"l(
          // Preamble - load the 1st 4x4 inputs: row-0, row-1, row-2, row-3
          // Do the 8bit permutations
          ld32step %[in0], $mzero, %[srcPtr]+=, %[inRowStride]  
          ld32step %[in1], $mzero, %[srcPtr]+=, %[inRowStride]    
          {ld32step %[in2], $mzero, %[srcPtr]+=, %[inRowStride]    
           shuf8x8hi %[out0], %[in0], %[in1]}
          {ld32step %[in0], $mzero, %[srcPtr]+=, %[inRowStride] 
           shuf8x8lo %[out2], %[in0], %[in1]}
          shuf8x8hi %[out1], %[in2], %[in0]

          // 8 cycles to transpose 4x4
          .align 8
          {rpt %[loops], (2f - 1f) / 8 -1 
           shuf8x8lo %[out3], %[in2], %[in0]}
          1:
            // Load row-4, row-5 (for the second transposition)
            // Do 2 of the second stage 16-bit permutations
            {ld32step   %[in0], $mzero, %[srcPtr]+=, %[inRowStride]
             sort4x16hi %[out4], %[out2], %[out3]}
            {ld32step   %[in1], $mzero, %[srcPtr]+=, %[inRowStride]  
             sort4x16lo %[out3], %[out2], %[out3]}

            // Write the first 2 outputs (row-a, row-b)
            // Complete the second stage permutations 
            {st32step   %[out3], $mzero, %[dstPtr]+=, %[outRowStride] 
             sort4x16hi %[out3], %[out0], %[out1]}
            {st32step   %[out4], $mzero, %[dstPtr]+=, %[outRowStride] 
             sort4x16lo %[out4], %[out0], %[out1]}
 
            // Load row-6, row-7
            // begin the first stage permutation of row-4, row-5
            {ld32step   %[in2], $mzero, %[srcPtr]+=, %[inRowStride]    
             shuf8x8hi  %[out0], %[in0], %[in1]}
            {ld32step   %[in0], $mzero, %[srcPtr]+=, %[inRowStride] 
             shuf8x8lo  %[out2], %[in0], %[in1]}

            // Write row-c, row-d and stride back to row-a
            // Complete the first stage permutation: row-6 and row-7 
            {st32step   %[out4], $mzero, %[dstPtr]+=, %[outRowStride] 
             shuf8x8hi %[out1], %[in2], %[in0]}
            {st32step   %[out3], $mzero, %[dstPtr]+=, %[outBackStride]
             shuf8x8lo %[out3], %[in2], %[in0]}

          2:
            // Stage 2 (4x16) permutations of row-4, row-5, row-6, row-7
            // And store, into row-a,b,c,d (Columns 4-7 [a4 a5 a6 a7] etc) 
            sort4x16lo %[out4], %[out2], %[out3]

            {st32step %[out4], $mzero, %[dstPtr]+=, %[outRowStride] 
             sort4x16hi %[out3], %[out2], %[out3]}

            {st32step %[out3], $mzero, %[dstPtr]+=, %[outRowStride] 
             sort4x16lo %[out3], %[out0], %[out1]}
            {st32step %[out3], $mzero, %[dstPtr]+=, %[outRowStride] 
             sort4x16hi %[out3], %[out0], %[out1]}
            st32step %[out3], $mzero, %[dstPtr]+=, %[outBackStride] 
          )l"
        : [srcPtr] "+r"(srcPtrInner), [dstPtr] "+r"(dstPtrInner),
          [in0] "=r"(in0), [in1] "=r"(in1), [in2] "=r"(in2), [out0] "=r"(out0),
          [out1] "=r"(out1), [out2] "=r"(out2), [out3] "=r"(out3),
          [out4] "=r"(out4)

        : [loops] "r"(numSrcRowsD4 - 1), [outBackStride] "r"(outBackStride),
          [outRowStride] "r"(numSrcRowsD4), [inRowStride] "r"(numSrcColumnsD4)

        : "memory");
  }
}

template <>
void transposeRowsColumnsFast<quarter>(const quarter *srcPtr, quarter *dstPtr,
                                       unsigned numSrcRowsD4Or8,
                                       unsigned numSrcColumnsD4Or8) {
  transpose8(srcPtr, dstPtr, numSrcRowsD4Or8, numSrcColumnsD4Or8,
             numSrcRowsD4Or8, numSrcColumnsD4Or8);
}

template <>
void transposeRowsColumnsFast<quarter>(const quarter *srcPtr, quarter *dstPtr,
                                       unsigned numSrcRowsD4Or8,
                                       unsigned numSrcColumnsD4Or8,
                                       unsigned allocatedRows,
                                       unsigned allocatedColumns) {
  transpose8(srcPtr, dstPtr, numSrcRowsD4Or8, numSrcColumnsD4Or8, allocatedRows,
             allocatedColumns);
}

template <>
void transpose1<quarter>(const quarter *srcPtr, quarter *dstPtr,
                         unsigned numSrcRows, unsigned numSrcColumns) {
  // Subword write of 8 bits is slow.  We write to sequential addresses anyway,
  // and begin at an 8 byte aligned location. So gather 4 bytes into a 32-bit
  // result and write that when full.  Flush any last 1,2 or 3 bytes only
  // once at the end.
  // Faster than sub word writes and avoids inclusion of a function to do that,
  // but still quite slow
  constexpr unsigned bytesToGather = 4;
  unsigned outWord = 0;
  unsigned outCount = bytesToGather;
  unsigned *unsignedDstPtr = reinterpret_cast<unsigned *>(dstPtr);
  for (unsigned x = 0; x != numSrcColumns; ++x) {
    for (unsigned y = 0; y != numSrcRows; ++y) {
      unsigned in;
      // Load the quarter type data into an unsigned type for bit manipulation
      // now and when flushing.  roll8r is an efficient may to shift the next
      // byte in, so do that in assembler.
      asm volatile(" ldz8 %[in], $mzero, %[srcPtr], 0\n"
                   " roll8r %[outWord], %[outWord], %[in]\n"
                   : [in] "=r"(in), [outWord] "+r"(outWord)
                   : [srcPtr] "r"(&srcPtr[y * numSrcColumns + x])
                   :);
      outCount--;
      if (outCount == 0) {
        *unsignedDstPtr++ = outWord;
        outCount = bytesToGather;
      }
    }
  }
  if (outCount != bytesToGather) {
    // Flush at the end
    const unsigned prevWordMask = (0xffffffff << (8 * (4 - outCount)));
    *unsignedDstPtr =
        (outWord >> (8 * outCount)) | (*unsignedDstPtr & prevWordMask);
  }
}

template <>
void transposeRowsColumns<quarter>(const quarter *srcPtr, quarter *dstPtr,
                                   unsigned numSrcRows,
                                   unsigned numSrcColumns) {
  constexpr unsigned rptLimit = CSR_W_REPEAT_COUNT__VALUE__MASK;
  if ((numSrcRows & 0x7) == 0 && (numSrcColumns & 0x7) == 0 &&
      (numSrcRows / 8) < rptLimit) {
    transpose8(srcPtr, dstPtr, numSrcRows / 8, numSrcColumns / 8,
               numSrcRows / 8, numSrcColumns / 8);
  } else if ((numSrcRows & 0x3) == 0 && (numSrcColumns & 0x3) == 0 &&
             (numSrcRows / 4) < rptLimit) {
    transpose4(srcPtr, dstPtr, numSrcRows / 4, numSrcColumns / 4);
  } else {
    transpose1(srcPtr, dstPtr, numSrcRows, numSrcColumns);
  }
}

#endif
