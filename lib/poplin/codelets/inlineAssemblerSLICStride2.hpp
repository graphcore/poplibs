// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#if __IPU_ARCH_VERSION__ >= 21

template <unsigned weightSelection>
static __attribute__((always_inline)) void
f8v8hihoSLICImplicitZeroStride2(uint2 triAddr, unsigned strides,
                                unsigned loops) {
  asm volatile(/* Prime with 1st 3 sets of inputs, there are no outputs yet*/
               R"l(
        brnzdec %[loops], 1f
        // Nothing to do
        bri 9f
      1:
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        brnzdec %[loops], 1f
      )l"
               /* 1 output - no further need to load */
               R"l(
        f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]
        {bri      4f
         f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      1:
        brnzdec %[loops], 3f
      )l"
               /* 2 outputs, load once*/
               R"l(
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
        f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $azeros, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
        bri      5f
       .align 8
        nop // Repeat alignment
      )l"
               /* The first real output is available AFTER this */
               R"l(
      3:
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $azeros, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}

        rpt %[loops], (2f - 1f) / 8 - 1
      1:
        {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $azeros, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      2:
         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
      5: {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
         f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]
      4: st64pace $a4:5, %[triAddr]+=, %[strides], 0b01
      9:
      )l"
               : [triAddr] "+r"(triAddr), [loops] "+r"(loops)
               : [strides] "r"(strides), [SLIC_FLAGS] "i"(weightSelection)
               : "$a0:1", "$a2:3", "$a4:5", "$m1");
}
template <unsigned weightSelection>
static __attribute__((always_inline)) void
f8v8hihoSLICStride2(uint2 triAddr, unsigned strides, unsigned loops) {
  asm volatile(
      R"l(
        brnzdec   %[loops], 1f
        // Nothing to do
        bri 9f
      1:
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
        brnzdec   %[loops], 1f
      )l"  /* Load 1 set of partials, 4 inputs only. No increment to
              partialsPtr */
      R"l(

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0101
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]
        {bri 3f
          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      1:
       brnzdec   %[loops], 4f
       )l" /* 2 outputs: Load 2 sets of partials, 6 inputs are needed */
      R"l(
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        bri 5f
         .align 8
         nop  // Rpt align
      )l"  /* Prime with 1st 3 sets of inputs, there are no outputs yet*/
      R"l(
        4:
         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides],0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
         /* The first real output is available AFTER this*/
         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $a2:3, %[SLIC_FLAGS]}

         rpt %[loops], (2f - 1f) / 8 - 1
       1:
         {ld2xst64pace $a0:3, $a4:5, %[triAddr]+=, %[strides], 0b110100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
         {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $a2:3, %[SLIC_FLAGS]}
       2:

         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1100
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
        5:
         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b0101
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
         {st64pace $a4:5, %[triAddr]+=, %[strides], 0b11
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}

          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]
        3:
          st64pace $a4:5, %[triAddr]+=, %[strides], 0b01
        9:
      )l"
      : [triAddr] "+r"(triAddr), [loops] "+r"(loops)
      : [strides] "r"(strides), [SLIC_FLAGS] "i"(weightSelection)
      : "$a0:1", "$a2:3", "$a4:5", "$m1");
}
#endif // __IPU__ARCH_VERSION
