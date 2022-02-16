// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#if __IPU_ARCH_VERSION__ >= 21

#define F8v8HIHO_SLIC_IMPLICIT_ZERO_STRIDE2(SLIC_WEIGHT_SELECTION)             \
  asm volatile( /* Prime with 1st 3 sets of inputs, there are no outputs yet*/ \
      R"l(
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
      )l" \
        /* In each case, subtracting 3 as that's what was subtracted*/ \
        /* from the count already*/ \
        /* If >= 3 outputs use the loop*/  \
      R"l(
        brpos   %[loops], 3f
        cmpeq   $m1, %[loops], 1-3
        brz     $m1, 1f
      )l" \
        /* 1 output - no further need to load */ \
      R"l(
        f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]
        {bri      4f
         f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      1:
        cmpeq   $m1, %[loops], 2-3
        brz     $m1, 1f
      )l" \
        /* 2 outputs, load once*/ \
      R"l(
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
        f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $azeros, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}
        bri      5f
       .align 8
        nop // Repeat alignment
      3:
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
         f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
      )l" \
         /* The first real output is available AFTER this */ \
      R"l(
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
      5: {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1101
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
         f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]
      4: st64pace $a4:5, %[triAddr]+=, %[strides], 0b01
      )l"   \
      : [triAddr] "+r"(triAddr)                                                \
      : [strides] "r"(strides), [loops] "r"(loops),                            \
        [SLIC_FLAGS] "i"(SLIC_WEIGHT_SELECTION)                                \
      : "$a0:1", "$a2:3", "$a4:5", "$m1");

#define F8v8HIHO_SLIC_STRIDE2(SLIC_WEIGHT_SELECTION)                           \
  asm volatile(                                                                \
      R"l(
        ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
        brpos   %[loops], 4f
        cmpeq $m1, %[loops],1-3
        brz $m1, 1f /* Path for 1 output */
      )l" /* Load 1 set of partials, 4 inputs only. No increment to            \
             partialsPtr */                                                    \
      R"l(

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $a2:3, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0100
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}

        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0101
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0101
          f8v8hihov4slic $azeros, $a0:1, $azeros, %[SLIC_FLAGS]}
        {bri 3f
          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
      1:
       )l" /* 2 outputs: Load 2 sets of partials, */ /* 6 inputs are needed */ \
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
      )l" /* Prime with 1st 3 sets of inputs, there are no outputs yet*/       \
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
         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b1101
          f8v8hihov4slic $a4:5, $a0:1, $azeros, %[SLIC_FLAGS]}

         {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b0101
          f8v8hihov4slic $a4:5, $azeros, $azeros, %[SLIC_FLAGS]}
        3:
         st64pace $a4:5, %[triAddr]+=, %[strides], 0b01
      )l"                                                                     \
      : [triAddr] "+r"(triAddr)                                                \
      : [strides] "r"(strides), [loops] "r"(loops),                            \
        [SLIC_FLAGS] "i"(SLIC_WEIGHT_SELECTION)                                \
      : "$a0:1", "$a2:3", "$a4:5", "$m1");

#endif // __IPU__ARCH_VERSION
