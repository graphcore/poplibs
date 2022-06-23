// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#if __IPU_ARCH_VERSION__ >= 21

template <bool use128BitLoad, unsigned convUnits>
static __attribute__((always_inline)) void slicLoadWeights(void) {}

template <> void slicLoadWeights<false, 16>(void) {
  asm volatile(
      // Last processing block (closest to the output)
      // Select using TSLIC_F16V4_1x4_W0
      R"l(
             ld64putcs   24     // ch=0
             ld64putcs   28     // ch=1
             ld64putcs   24+32  // ch=2
             ld64putcs   28+32  // ch=3
      )l"
      // Select using TSLIC_F16V4_1x4_W1
      R"l(
             ld64putcs   25     // ch=4
             ld64putcs   29     // ch=5
             ld64putcs   25+32  // ch=6
             ld64putcs   29+32  // ch=7
      )l"
      // 2nd last processing block
      // Select using TSLIC_F16V4_1x4_W0
      R"l(
             ld64putcs   16     // ch=0
             ld64putcs   20     // ch=1
             ld64putcs   16+32  // ch=2
             ld64putcs   20+32  // ch=3
      )l"
      // Select using TSLIC_F16V4_1x4_W1
      R"l(
             ld64putcs   17     // ch=4
             ld64putcs   21     // ch=5
             ld64putcs   17+32  // ch=6
             ld64putcs   21+32  // ch=7
      )l"
      // 3rd last processing block
      // Select using TSLIC_F16V4_1x4_W0
      R"l(
             ld64putcs   8      // ch=0
             ld64putcs   12     // ch=1
             ld64putcs   8+32   // ch=2
             ld64putcs   12+32  // ch=3
      )l"
      // Select using TSLIC_F16V4_1x4_W1
      R"l(
             ld64putcs   9      // ch=4
             ld64putcs   13     // ch=5
             ld64putcs   9+32   // ch=6
             ld64putcs   13+32  // ch=7
      )l"
      // 1st processing block (closest to the input)
      // Select using TSLIC_F16V4_1x4_W0
      R"l(
             ld64putcs   0     // out ch=0
             ld64putcs   4     // ch=1
             ld64putcs   0+32  // ch=2
             ld64putcs   4+32  // ch=3
      )l"
      // Select using TSLIC_F16V4_1x4_W1
      R"l(
             ld64putcs   1     // ch=4
             ld64putcs   5     // ch=5
             ld64putcs   1+32  // ch=6
             ld64putcs   5+32  // ch=7
      )l"
      :
      :
      :);
}

template <unsigned weightSelection>
static __attribute__((always_inline)) void
f8v8hihoSLICImplicitZero(uint2 triAddr, unsigned strides, unsigned loops) {
  asm volatile(/* Prime with 1st 3 sets of inputs, there are no outputs yet*/
               R"l(
        brnzdec %[loops], 1f
        // Nothing to do
        bri 9f
      1:
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1000
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1000
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        add $m1, %[loops], 1-5
        // Branch to the loop
        brpos $m1, 3f
        brnzdec %[loops], 1f
      )l"
               /* 1 output - no further need to load */
               R"l(
        f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]
        {bri      4f
         f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      1:
        brnzdec %[loops], 1f
      )l"
               /* 2 outputs, load once, don't advance the pointer*/
               /* (dummy loads later) */
               R"l(
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0101
        f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {bri      5f
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      1:
        brnzdec %[loops], 1f
      )l"
               /* 3 outputs, load twice, only advance the pointer once */
               /* (dummy loads later) */
               R"l(
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $azeros, %[triAddr]+=, %[strides], 0b0101
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
        bri      6f
      )l"
               /* 4 outputs, load three times, only advance the pointer twice */
               /* (dummy loads later)*/
               R"l(
      1:
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $azeros, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
        bri      7f
       )l"
               /* The first real output is available AFTER this */
               R"l(
       .align 8
        nop // Repeat alignment
      3:
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

       {ld2x64pace $a0:1, $azeros, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}

        rpt $m1, (2f - 1f) / 8 - 1
      1:
        {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      2:
         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      7: {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      6: {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      5: {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      4: st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
      9:
      )l"
               : [triAddr] "+r"(triAddr), [loops] "+r"(loops)
               : [strides] "r"(strides), [SLIC_FLAGS] "i"(weightSelection)
               : "$a0:1", "$a2:3", "$a4:5", "$m1");
}
template <unsigned weightSelection>
static __attribute__((always_inline)) void
f8v8hihoSLICLoop(uint2 triAddr, unsigned strides, unsigned loops) {
  asm volatile(/* Prime with 1st 3 sets of inputs, there are no outputs yet*/
               R"l(
        .align 8
          ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides],0b1100
           f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
         /* The first real output is available AFTER this*/
         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $a2:3, %[SLIC_FLAGS]}

         rpt %[loops], (2f - 1f) / 8 - 1
       1:
         {ld2xst64pace $a0:3, $a4:5, %[triAddr]+=, %[strides], 0b111100
          f8v8hihov4slic $a4:5, $a0:1, $a2:3, %[SLIC_FLAGS]}
       2:
         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
         {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
         {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
         st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
      )l"
               : [triAddr] "+r"(triAddr)
               : [strides] "r"(strides), [loops] "r"(loops),
                 [SLIC_FLAGS] "i"(weightSelection)
               : "$a0:1", "$a2:3", "$a4:5");
}
template <unsigned weightSelection>
static __attribute__((always_inline)) void
f8v8hihoSLICLessThan5(uint2 triAddr, unsigned strides, unsigned loops) {
  asm volatile(
      R"l(
        brnzdec %[loops], 1f
        // Nothing to do
        bri 9f
      1:
        brnzdec %[loops], 1f
      )l" /* Path for 1 output. Load 1 set of partials, 4 inputs only. No
             increment to partialsPtr */
      R"l(
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]
        {bri 3f
          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      1:
        brnzdec %[loops], 1f
      )l" /* Path for 2 outputs. Load 2 set2 of partials, 5 inputs. One
             increment to partialsPtr*/
      R"l(
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {bri 4f
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      1:
        brnzdec %[loops], 1f
      )l" /* Path for 3 outputs. Load 3 sets of partials, 6 inputs. Two
             increments to partialsPtr */
      R"l(
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $a4:5, $a0:1, $a2:3, %[SLIC_FLAGS]}
        bri 5f
      )l" /* Path for 4 outputs. Load 4 sets of partials, 6 inputs. Two
             increments to partialsPtr */
      R"l(
      1:
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $a4:5, $a0:1, $a2:3, %[SLIC_FLAGS]}
      )l" /* Start storing. Fall through for the 4 of case, jump in for
             others*/
      R"l(
        {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      5:
        {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      4:
         {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      3:
        st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
      9:
      )l"
      : [triAddr] "+r"(triAddr), [loops] "+r"(loops)
      : [strides] "r"(strides), [SLIC_FLAGS] "i"(weightSelection)

      : "$a0:1", "$a2:3", "$a4:5", "$m1");
}
#endif // __IPU__ARCH_VERSION
