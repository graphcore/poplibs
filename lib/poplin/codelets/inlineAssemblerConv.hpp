// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#if __IPU_ARCH_VERSION__ >= 21

// Bit to force all accumulators to zero when written in FP_CLR register
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

#define DELTAN_OFFSET_BITS 20
#define DELTAN_OFFSET_MASK ((1 << DELTAN_OFFSET_BITS) - 1)

template <typename T>
static __attribute__((always_inline)) T *ld64StepToIncPtr(T *ptr,
                                                          unsigned stride) {
  asm volatile(" ld64step $azeros,$mzero, %[ptr]+=,%[stride]"
               : [ptr] "+r"(ptr)
               : [stride] "r"(stride)
               :);
  return ptr;
}

static __attribute__((always_inline)) void
setFp8Format(const MetadataType weightsMetadata,
             const MetadataType inMetadata) {
  auto format = (0x2 & (inMetadata >> 6)) | (weightsMetadata >> 7);
  __builtin_ipu_put(format, CSR_S_FP_INFMT__INDEX);
}

static __attribute__((always_inline)) void
setFp8Scale(const MetadataType weightsMetadata, const MetadataType inMetadata) {
  // Scale is the sum of scales, as
  // we compute: half(input*weights) * 2^(scaleIn+scaleWeights)
  __builtin_ipu_put(weightsMetadata + inMetadata, CSR_S_FP_ISCL__INDEX);
}

#define WEIGHT_LOAD_4(C1, C2, C3, C4)                                          \
  WEIGHT_LOAD(C1) /* 0th conv unit,  1st out channel*/                         \
  WEIGHT_LOAD(C2) /* 0th conv unit, 2nd out channel*/                          \
  WEIGHT_LOAD(C3) /* 0th conv unit, 3rd out channel*/                          \
  WEIGHT_LOAD(C4) /* 0th conv unit, 3rd out channel*/

template <unsigned kernelHeightM1, bool use128BitLoad, unsigned convUnits>
static __attribute__((always_inline)) __attribute__((target("supervisor"))) void
ampLoadWeights(unsigned weightPtr = 0, unsigned stride = 0) {
#define WEIGHT_LOAD(BASE)                                                      \
  __builtin_ipu_ld64putcs(BASE);      /* Phase 0*/                             \
  __builtin_ipu_ld64putcs(BASE + 4);  /* Phase 1*/                             \
  __builtin_ipu_ld64putcs(BASE + 32); /* Phase 2*/                             \
  __builtin_ipu_ld64putcs(BASE + 36); /* Phase 3*/

  weightPtr += stride;
  WEIGHT_LOAD_4(0, 8, 16, 24)
  __builtin_ipu_put(weightPtr, CSR_S_CCCSLOAD__INDEX);
  weightPtr += stride;
  WEIGHT_LOAD_4(1, 9, 17, 25)
  __builtin_ipu_put(weightPtr, CSR_S_CCCSLOAD__INDEX);
  weightPtr += stride;
  WEIGHT_LOAD_4(2, 10, 18, 26)
  __builtin_ipu_put(weightPtr, CSR_S_CCCSLOAD__INDEX);
  WEIGHT_LOAD_4(3, 11, 19, 27)
#undef WEIGHT_LOAD
}

template <>
void ampLoadWeights<0, false, 16>(unsigned weightPtr, unsigned stride) {
#define WEIGHT_LOAD(BASE)                                                      \
  __builtin_ipu_ld64putcs(BASE);     /* Phase 0*/                              \
  __builtin_ipu_ld64putcs(BASE + 1); /* Phase 1*/                              \
  __builtin_ipu_ld64putcs(BASE + 2); /* Phase 2*/                              \
  __builtin_ipu_ld64putcs(BASE + 3); /* Phase 3*/

  WEIGHT_LOAD_4(0, 4, 32, 36)
  WEIGHT_LOAD_4(8, 12, 40, 44)
  WEIGHT_LOAD_4(16, 20, 48, 52)
  WEIGHT_LOAD_4(24, 28, 56, 60)
#undef WEIGHT_LOAD
}

template <>
void ampLoadWeights<1, false, 16>(unsigned weightPtr, unsigned stride) {
#define WEIGHT_LOAD(BASE)                                                      \
  __builtin_ipu_ld64putcs(BASE);     /* Phase 0*/                              \
  __builtin_ipu_ld64putcs(BASE + 1); /* Phase 1*/                              \
  __builtin_ipu_ld64putcs(BASE + 4); /* Phase 2*/                              \
  __builtin_ipu_ld64putcs(BASE + 5); /* Phase 3*/

  weightPtr += stride;
  WEIGHT_LOAD_4(0, 32, 8, 40)
  WEIGHT_LOAD_4(16, 48, 24, 56)
  __builtin_ipu_put(weightPtr, CSR_S_CCCSLOAD__INDEX);
  WEIGHT_LOAD_4(2, 34, 10, 42)
  WEIGHT_LOAD_4(18, 50, 26, 58)
#undef WEIGHT_LOAD
}

template <>
void ampLoadWeights<0, true, 16>(unsigned weightPtr, unsigned stride) {
#define WEIGHT_LOAD(BASE)                                                      \
  __builtin_ipu_ld128putcs(BASE);     /* Phase 0,1*/                           \
  __builtin_ipu_ld128putcs(BASE + 2); /* Phase 2,3*/

  WEIGHT_LOAD_4(0, 4, 32, 36)
  WEIGHT_LOAD_4(8, 12, 40, 44)
  WEIGHT_LOAD_4(16, 20, 48, 52)
  WEIGHT_LOAD_4(24, 28, 56, 60)
#undef WEIGHT_LOAD
}

template <>
void ampLoadWeights<1, true, 16>(unsigned weightPtr, unsigned stride) {
#define WEIGHT_LOAD(BASE)                                                      \
  __builtin_ipu_ld128putcs(BASE);     /* Phase 0,1*/                           \
  __builtin_ipu_ld128putcs(BASE + 4); /* Phase 2,3*/

  weightPtr += stride;
  WEIGHT_LOAD_4(0, 32, 8, 40)
  WEIGHT_LOAD_4(16, 48, 24, 56)
  __builtin_ipu_put(weightPtr, CSR_S_CCCSLOAD__INDEX);
  WEIGHT_LOAD_4(2, 34, 10, 42)
  WEIGHT_LOAD_4(18, 50, 26, 58)
#undef WEIGHT_LOAD
}

template <bool ZeroPartials>
static __attribute__((always_inline)) void
convQuarterHalfLoop(const quarter *inPtr, half *outPtr, unsigned loops,
                    unsigned strides) {
  // Packed strides in `strides`:
  // b[0,10) = inputsStride  (inputs read only, select with 01)
  // b[10,20) = partialsInOutStride (partials in/out only, select with 10)
  // b[20,30) = 0 for no stride (reading dummy partials,inputs select with 11)
  auto triAddr = __builtin_ipu_tapack(inPtr, outPtr, outPtr);
  if constexpr (ZeroPartials == false) {
    asm volatile(
        R"l(
              .macro amp OP1 OP2 OP3 OP4
                f8v8hihov4amp \OP1 , \OP2 , \OP3 , \OP4
              .endm
              .equ ZERO_PARTIALS, 0
    )l" ::
            :);
  } else {
    asm volatile(
        R"l(
              .macro amp OP1 OP2 OP3 OP4
                f8v8hihov4amp \OP1 , \OP2 , $azeros, \OP4
              .endm
              .equ ZERO_PARTIALS, 1
    )l" ::
            :);
  }

  asm volatile(
      R"l(
             // loops = outs -3.  Fast forward to the >=3 loops fast path
             {brpos %[loops], 5f
              setzi $a0, %[ZAACC_MASK]}
             // -3 -2 -1  => 0 1 2
             {add %[loops], %[loops], 3
              uput $FP_CLR, $a0}
             brnzdec %[loops], 1f
             // Nothing to do for this worker so exit
             bri 8f

            // General addressing pattern for partials, outputs:
            // Forward 1 (3 times), back inOutStride
            //  8 9 a b, 4 5 6 7,  0 1 2 3

            // Prime with partials - Each is a read of the partials,
            // a dummy read of the input with no pointer increment,
            // and a call to the amp instruction with phase!=0
            // Loads to $a0:1 are dummy loads as we can't write
            // twice to $azeros in one bundle
            // ld2x64pace: 0bxxyy stride select:
            // xx=partialsInPtr, yy=inPtr
          1:
          .if ZERO_PARTIALS == 0
            ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0011

            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3,  %[triAddr]+=, %[strides], 0b0011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}
            // Check for the case of 1 output
            brnzdec %[loops], 1f
            // There is only 1 output - avoid the stride in the partials load
            // to avoid overreads when we fetch unused partials
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1111
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            // This is the first genuine load of the input, and increments the
            // pointer
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}
          .else
            // Check for the case of 1 output
            brnzdec %[loops], 2f

            // This is the first genuine load of the input, and increments the
            // pointer
            ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
          .endif

            // ***** One output *****
            // Push in a genuine input (and next set of partials)
            // Phase 0..3
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P0]}

            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P1]}

            // For 1 output avoid striding the partials pointer and then
            // skip the loop body
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1101
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P2]}

             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P3]

            {bri 7f
             amp $a4:5, $azeros, $azeros, %[TAMP_F16V4_E4_P0]}

          // ***** Two outputs *****
          1:
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}
          2:
            // This is the first genuine load of the input, and increments the
            // pointer
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

             // Push in a genuine input (and next set of partials)
             // Phase 0..3
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P0]}

            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P1]}

            // There are 2 outputs - avoid the stride in the partials load
            // to avoid overreads when we fetch unused partials
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1101
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P2]}

            // $a0:1 read, $a2:3 dummy read (Can't write $azeros twice)
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P3]}

            {ld2x64pace $a4:5, $a6:7, %[triAddr]+=, %[strides], 0b0000
             amp $a0:1, $a0:1, $a4:5, %[TAMP_F16V4_E4_P0]}
            bri 6f
            //  Always branches to tail of >=3 path
         .align 8
            nop // Repeat alignment

          // ***** Fast path to fall through if >=3 "loops" *****
          5:
          .if ZERO_PARTIALS == 0
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0011
              uput $FP_CLR, $a0}
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3,  %[triAddr]+=, %[strides], 0b0011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}
          .else
            // This is the first genuine load of the input, and increments the
            // pointer
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             uput $FP_CLR, $a0}
          .endif

             // Push in a genuine input (and next set of partials)
             // Phase 0..3
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P0]}

            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b0000
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1001
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P2]}

            // $a0:1 read, $a2:3 dummy read (Can't write $azeros twice)
            {ld2x64pace $a0:1, $a2:3, %[triAddr]+=, %[strides], 0b1100
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P3]}

            ld2x64pace $azeros, $a4:5, %[triAddr]+=, %[strides], 0b0011

            // One more partials read to move to an alternate memory segment
            // to the writes so we can use ld2xst64pace in the inner loop
            ld2x64pace $azeros, $a2:3, %[triAddr]+=, %[strides], 0b0011

            {ld2x64pace $a4:5, $a6:7, %[triAddr]+=, %[strides], 0b0000
             amp $a0:1, $a0:1, $a4:5, %[TAMP_F16V4_E4_P0]}

            // Loop is the first point the output is actually stored,
            // Continue loading inputs and partials and striding pointers
            rpt %[loops], (2f - 1f) / 8 - 1
          1:
            // ld2xst64pace: 0bxxyyzz stride select:
            // xx=outPtr, yy=partialsInPtr, zz=inPtr
            {ld2xst64pace $a0:3, $a0:1, %[triAddr]+=, %[strides], 0b001000
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2xst64pace $a4:7, $a4:5, %[triAddr]+=, %[strides], 0b000001
             amp $a0:1, $a0:1, $a6:7, %[TAMP_F16V4_E4_P2]}

            {ld2xst64pace $a0:3, $a0:1, %[triAddr]+=, %[strides], 0b000000
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P3]}

            {ld2xst64pace $a4:7, $a4:5, %[triAddr]+=, %[strides], 0b100000
             amp $a0:1, $a0:1, $a6:7, %[TAMP_F16V4_E4_P0]}
          2:

            {ld2xst64pace $a0:3, $a0:1, %[triAddr]+=, %[strides], 0b001000
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P1]}

            // Now we have read all the partials that are needed so
            // don't overread (Different to loop body)
            // ldst64pace: 0bxxyy stride select:
            // xx=inPtr, yy=outPtr
            {ldst64pace $a4:5, $a4:5, %[triAddr]+=, %[strides], 0b0001
             amp $a0:1, $a0:1, $a6:7, %[TAMP_F16V4_E4_P2]}

            {ldst64pace $a0:1, $a0:1, %[triAddr]+=, %[strides], 0b0000
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P3]}

            {ldst64pace $a4:5, $a4:5, %[triAddr]+=, %[strides], 0b1000
             amp $a0:1, $a0:1, $azeros, %[TAMP_F16V4_E4_P0]}

          6:
            {ldst64pace $a0:1, $a0:1, %[triAddr]+=, %[strides], 0b0000
             amp $a4:5,  $a4:5, $azeros, %[TAMP_F16V4_E4_P1]}
            {ldst64pace $a0:1, $a4:5, %[triAddr]+=, %[strides], 0b0011
              amp $a4:5,  $a0:1, $azeros, %[TAMP_F16V4_E4_P2]}
            // Use the last input, no more need to load
            {st64pace $a4:5, %[triAddr]+=, %[strides], 0b00
              amp $a4:5,  $a0:1, $azeros, %[TAMP_F16V4_E4_P3]}

            // Result output only
            {st64pace $a4:5, %[triAddr]+=, %[strides], 0b10
             amp $a4:5, $azeros, $azeros, %[TAMP_F16V4_E4_P0]}
         7:
            {st64pace $a4:5, %[triAddr]+=, %[strides], 0b00
              amp $a4:5,  $azeros, $azeros, %[TAMP_F16V4_E4_P1]}
            {st64pace $a4:5, %[triAddr]+=, %[strides], 0b00
              amp $a4:5,  $azeros, $azeros, %[TAMP_F16V4_E4_P2]}
            {st64pace $a4:5, %[triAddr]+=, %[strides], 0b00
              amp $a4:5,  $azeros, $azeros, %[TAMP_F16V4_E4_P3]}
            st64pace $a4:5, %[triAddr]+=, %[strides], 0b00
          8:

          // Remove macro definition to avoid later re-definition issues
          .purgem amp
        )l"
      : [loops] "+r"(loops), [strides] "+r"(strides), [triAddr] "+r"(triAddr)
      : [TAMP_F16V4_E4_P0] "i"(TAMP_F16V4_E4_P0),
        [TAMP_F16V4_E4_P1] "i"(TAMP_F16V4_E4_P1),
        [TAMP_F16V4_E4_P2] "i"(TAMP_F16V4_E4_P2),
        [TAMP_F16V4_E4_P3] "i"(TAMP_F16V4_E4_P3),
        [ZAACC_MASK] "i"(ZAACC_BITMASK)
      // As we want to access a group of 4 registers and also
      // the first/second 2 of the group we can't use C variables of types
      // float4, float2 or similar.  So just clobber all a-registers
      : "memory", "$a0:1", "$a2:3", "$a4:5", "$a6:7");
}

static __attribute__((always_inline)) void
convQuarterHalfLoopnx1(const quarter *inPtr, half *outPtr, int loops,
                       unsigned strideAC, unsigned stridesZ_OutM3_AB,
                       unsigned stridesZ_Out_BA, unsigned stridesOutM1_3_BA,
                       unsigned stridesZ_OutM1_X) {
  // There are various strides pre prepared by the supervisor
  //
  // Name                  b[20,30)       b[10,20)     b[0,10)
  // stridesZ_OutM3_AB     Zero           outStride-3  inStrideAB
  // stridesZ_Out_BA       Zero           outStride    inStrideBA'
  // stridesOutM1_3_BA     outStride-1    3            inStrideBA'
  // stridesZ_OutM1_X      Zero           outStride-1  X:Unused
  //
  // The reason for many strides is that for the nx1 vertex there are 3 possible
  // kernel shapes.  Each involves reading 4 vectors of input elements:
  // A,B,C,D and then  we move to the next group 4: A',B',C',D' and so on
  // The 3 shapes result in ( Within [] : contiguous in memory):
  // 1x1 kernel:
  // [ABCD]       [A'B'C'D]        //This is the same as the 1x1Out vertex
  // 2x1  kernel:
  // [AB]         [A'B']
  // [CD]         [C'D']
  // 4x1 kernel:
  // [A]          [A']
  // [B]          [B']
  // [C]          [C']
  // [D]          [D']
  //
  // In each case to read the inputs we stride A->B->C->D->A'->B ...
  // In every memory layout pattern the stride A->B and C->D is the same so
  // 1 register (A->B) represents both A->B and C->D
  // Similarly for B->A' and D->C'
  //
  // To avoid a large backward stride we pre-offset and begin with a pointer
  // to A and one to C and therefore have 2 tri-packed addresses

  auto outPtrC = ld64StepToIncPtr(outPtr, 2);
  auto inPtrC = ld64StepToIncPtr(inPtr, strideAC);
  // Addresses listed are input, partials, output
  auto triAddrAB = __builtin_ipu_tapack(inPtr, outPtr, outPtrC);
  auto triAddrCD = __builtin_ipu_tapack(inPtrC, outPtrC, outPtr);

  // Declare the instruction for future flexibility
  asm volatile(
      R"l(
              .macro amp OP1 OP2 OP3 OP4
                f8v8hihov4amp \OP1 , \OP2 , \OP3, \OP4
              .endm
    )l");

  asm volatile(
      R"l(
             // loops = outs -3.  Fast forward to the >=3 loops fast path
             {brpos %[loops], 5f
              setzi $a0, %[ZAACC_MASK]}
             // -3 -2 -1  => 0 1 2
             {add %[loops], %[loops], 3
              uput $FP_CLR, $a0}
             brnzdec %[loops], 1f
             // Nothing to do for this worker so exit
             bri 8f

            // Prime with partials - Each is a read of the partials,
            // a dummy read of the input with no pointer increment,
            // and a call to the amp instruction with phase!=0
            // Loads to $a0:1 are dummy loads as we can't write
            // twice to $azeros in one bundle
            // ld2x64pace: 0bxxyy stride select:
            // xx=partialsInPtr, yy=inPtr
          1:
            ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, $mzero, 0b0011
            // Check for the case of 1 output
            brnzdec %[loops], 1f

            // ***** One output *****
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, $mzero, 0b1111
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3,  %[triAddrCD]+=, $mzero, 0b0011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}


            // There is only 1 output - avoid the stride in the partials load
            // to avoid overreads when we fetch unused partials
            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, $mzero, 0b1111
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            // This is the first genuine load of the input, and increments the
            // pointer
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesZ_OM3_AB], 0b1101
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            // ***** One output *****
            // Push in a genuine input (and dummy load of partials)
            // Phase 0..3
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesOM1_3_BA], 0b1101
             amp $azeros, $a0:1, $azeros, %[TAMP_F16V4_E4_P0]}

            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, %[stridesZ_OM3_AB], 0b1101
             amp $azeros, $a0:1, $azeros, %[TAMP_F16V4_E4_P1]}

            // For 1 output avoid striding the partials pointer and then
            // skip the loop body
            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, $mzero, 0b1111
             amp $azeros, $a0:1, $azeros, %[TAMP_F16V4_E4_P2]}

             amp $azeros, $a0:1, $azeros, %[TAMP_F16V4_E4_P3]

            {bri 7f
             amp $a4:5, $azeros, $azeros, %[TAMP_F16V4_E4_P0]}

          // ***** Two outputs *****
          1:
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesZ_OM1_X], 0b1011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3,  %[triAddrCD]+=, %[stridesZ_O_BA], 0b0011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, %[stridesZ_OM1_X], 0b1011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            // This is the first genuine load of the input, and increments the
            // pointer
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesZ_OM3_AB], 0b0001
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

             // Push in a genuine input (and next set of partials)
             // Phase 0..3
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesOM1_3_BA], 0b0001
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P0]}

            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, %[stridesZ_OM3_AB], 0b0001
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P1]}

            // There are 2 outputs - avoid the stride in the partials load
            // to avoid overreads when we fetch unused partials
            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, %[stridesZ_O_BA], 0b1101
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P2]}

            // $a0:1 read, $a2:3 dummy read (Can't write $azeros twice)
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesZ_OM3_AB], 0b1101
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P3]}

            {ld2x64pace $a4:5, $a6:7, %[triAddrAB]+=, %[stridesOM1_3_BA], 0b0001
             amp $a0:1, $a0:1, $a4:5, %[TAMP_F16V4_E4_P0]}
            bri 6f
            //  Always branches to tail of >=3 path
         .align 8

          // ***** Fast path to fall through if >=3 "loops" *****
          5:
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesZ_O_BA], 0b0011
              uput $FP_CLR, $a0}
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesZ_OM1_X], 0b1011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3,  %[triAddrCD]+=, %[stridesZ_O_BA], 0b0011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, %[stridesZ_OM1_X], 0b1011
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesZ_OM3_AB], 0b0001
             amp $azeros, $azeros, $a2:3, %[TAMP_F16V4_E4_P1]}

             // Push in a genuine input (and next set of partials)
             // Phase 0..3
            {ld2x64pace $a0:1, $a2:3, %[triAddrAB]+=, %[stridesOM1_3_BA], 0b1101
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P0]}

            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, %[stridesZ_OM3_AB], 0b0001
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2x64pace $a0:1, $a2:3, %[triAddrCD]+=, %[stridesZ_O_BA], 0b1001
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P2]}

            {ld2x64pace $a0:1, $a4:5, %[triAddrAB]+=, %[stridesZ_OM3_AB], 0b0001
             amp $azeros, $a0:1, $a2:3, %[TAMP_F16V4_E4_P3]}

            // One more partials read to move to an alternate memory segment
            // to the writes so we can use ld2xst64pace in the inner loop
            ld2x64pace $azeros, $a2:3, %[triAddrAB]+=, $mzero, 0b0011

            {ld2x64pace $a4:5, $a6:7, %[triAddrAB]+=, %[stridesOM1_3_BA], 0b1101
             amp $a0:1, $a0:1, $a4:5, %[TAMP_F16V4_E4_P0]}

            // Loop is the first point the output is actually stored,
            // Continue loading inputs and partials and striding pointers
            rpt %[loops], (2f - 1f) / 8 - 1
          1:
            // ld2xst64pace: 0bxxyyzz stride select:
            // xx=outPtr, yy=partialsInPtr, zz=inPtr
            {ld2xst64pace $a0:3, $a0:1, %[triAddrCD]+=, %[stridesZ_OM3_AB], 0b001001
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P1]}

            {ld2xst64pace $a4:7, $a4:5, %[triAddrCD]+=, %[stridesOM1_3_BA], 0b111001
             amp $a0:1, $a0:1, $a6:7, %[TAMP_F16V4_E4_P2]}

            {ld2xst64pace $a0:3, $a0:1, %[triAddrAB]+=, %[stridesZ_OM3_AB], 0b000001
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P3]}

            {ld2xst64pace $a4:7, $a4:5, %[triAddrAB]+=, %[stridesOM1_3_BA], 0b111101
             amp $a0:1, $a0:1, $a6:7, %[TAMP_F16V4_E4_P0]}
          2:

            {ld2xst64pace $a0:3, $a0:1, %[triAddrCD]+=, %[stridesZ_OM3_AB], 0b001001
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P1]}

            // Now we have read all the partials that are needed so
            // don't overread (Different to loop body)
            // ldst64pace: 0bxxyy stride select:
            // xx=inPtr, yy=outPtr
            {ldst64pace $a4:5, $a4:5, %[triAddrCD]+=, %[stridesOM1_3_BA], 0b1101
             amp $a0:1, $a0:1, $a6:7, %[TAMP_F16V4_E4_P2]}

            {ldst64pace $a0:1, $a0:1, %[triAddrAB]+=, %[stridesZ_OM3_AB], 0b0001
             amp $a4:5, $a4:5, $a2:3, %[TAMP_F16V4_E4_P3]}

            {ldst64pace $a4:5, $a4:5, %[triAddrAB]+=, %[stridesOM1_3_BA], 0b1101
             amp $a0:1, $a0:1, $azeros, %[TAMP_F16V4_E4_P0]}

          6:
            {ldst64pace $a0:1, $a0:1, %[triAddrCD]+=, %[stridesZ_OM3_AB], 0b0001
             amp $a4:5,  $a4:5, $azeros, %[TAMP_F16V4_E4_P1]}

            {ldst64pace $a0:1, $a4:5, %[triAddrCD]+=, $mzero, 0b0011
              amp $a4:5,  $a0:1, $azeros, %[TAMP_F16V4_E4_P2]}

            // Use the last input, no more need to load
            {st64pace $a4:5, %[triAddrCD]+=, $mzero, 0b00
              amp $a4:5,  $a0:1, $azeros, %[TAMP_F16V4_E4_P3]}

            // Result output only
            {st64pace $a4:5, %[triAddrCD]+=, %[stridesZ_OM3_AB], 0b10
             amp $a4:5, $azeros, $azeros, %[TAMP_F16V4_E4_P0]}
         7:
            {st64pace $a4:5, %[triAddrCD]+=, $mzero, 0b00
              amp $a4:5,  $azeros, $azeros, %[TAMP_F16V4_E4_P1]}
            {st64pace $a4:5, %[triAddrCD]+=, $mzero, 0b00
              amp $a4:5,  $azeros, $azeros, %[TAMP_F16V4_E4_P2]}
            {st64pace $a4:5, %[triAddrCD]+=, $mzero, 0b00
              amp $a4:5,  $azeros, $azeros, %[TAMP_F16V4_E4_P3]}
            st64pace $a4:5, %[triAddrCD]+=, $mzero, 0b11
          8:

          // Remove macro definition to avoid later re-definition issues
          .purgem amp
        )l"
      : [loops] "+r"(loops), [triAddrAB] "+r"(triAddrAB),
        [triAddrCD] "+r"(triAddrCD)
      : [TAMP_F16V4_E4_P0] "i"(TAMP_F16V4_E4_P0),
        [TAMP_F16V4_E4_P1] "i"(TAMP_F16V4_E4_P1),
        [TAMP_F16V4_E4_P2] "i"(TAMP_F16V4_E4_P2),
        [TAMP_F16V4_E4_P3] "i"(TAMP_F16V4_E4_P3),
        [ZAACC_MASK] "i"(ZAACC_BITMASK),
        [stridesZ_OM3_AB] "r"(stridesZ_OutM3_AB),
        [stridesZ_O_BA] "r"(stridesZ_Out_BA),
        [stridesOM1_3_BA] "r"(stridesOutM1_3_BA),
        [stridesZ_OM1_X] "r"(stridesZ_OutM1_X)
      // As we want to access a group of 4 registers and also
      // the first/second 2 of the group we can't use C variables of types
      // float4, float2 or similar.  So just clobber all a-registers
      : "memory", "$a0:1", "$a2:3", "$a4:5", "$a6:7");
}

template <typename WorkerStateType>
static __attribute__((always_inline)) WorkerStateType *workerState(void) {
  WorkerStateType *state;
  asm volatile(" mov %[state], $mvertex_base\n" : [state] "=r"(state) : :);
  return state;
}

#define RUN_ALL(NAME_STR, STATE)                                               \
  {                                                                            \
    unsigned workerAddress;                                                    \
    asm volatile(" setzi  %[workerAddress], " NAME_STR "\n"                    \
                 " runall %[workerAddress], %[state], 0\n"                     \
                 : [workerAddress] "=&r"(workerAddress)                        \
                 : [state] "r"(STATE)                                          \
                 :);                                                           \
  }

#define SET_ADDR(RESULT, NAME_STR)                                             \
  asm volatile(" setzi  %[workerAddress], " NAME_STR "\n"                      \
               : [workerAddress] "=r"(RESULT)                                  \
               :                                                               \
               :);

template <typename T>
static __attribute__((always_inline)) void runAll(const unsigned *workerAddress,
                                                  const T *state) {
  asm volatile(" runall %[workerAddress], %[state], 0\n"
               :
               : [workerAddress] "r"(workerAddress), [state] "r"(state)
               :);
}

static __attribute__((always_inline)) void syncWorkers(void) {
  asm volatile(
      " sync   %[sync_zone]\n" ::[sync_zone] "i"(TEXCH_SYNCZONE_LOCAL));
}

static __attribute__((always_inline)) unsigned getWid(void) {
  unsigned result;
  return __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
}

static __attribute__((always_inline)) unsigned long long
packStrides(unsigned stride0, unsigned stride1) {
  constexpr unsigned numStrideBits = NUM_STRIDE_BITS;
  return stride0 | (stride1 << numStrideBits);
}

static __attribute__((always_inline)) unsigned long long
packStrides(unsigned stride0, unsigned stride1, unsigned stride2) {
  constexpr unsigned numStrideBits = NUM_STRIDE_BITS;
  return stride0 | (stride1 << numStrideBits) |
         (stride2 << (numStrideBits * 2));
}

static __attribute__((always_inline)) int
extractSignExtendedStride(unsigned strides, unsigned idx) {
  // idx 0 = b[0,10), idx 1 = b[10,20), idx 2 = b[20,30)
  constexpr unsigned numStrideBits = NUM_STRIDE_BITS;
  return int(strides << (32 - numStrideBits - idx * numStrideBits)) >>
         (32 - numStrideBits);
}

constexpr unsigned stocasticRoundingMask =
    ~(CSR_S_FP_ICTL__ESR__MASK << CSR_S_FP_ICTL__ESR__SHIFT);

static __attribute__((always_inline)) unsigned getFPICTL(void) {
  return __builtin_ipu_get(CSR_S_FP_ICTL__INDEX);
}

static __attribute__((always_inline)) void putFPICTL(unsigned value) {
  __builtin_ipu_put(value, CSR_S_FP_ICTL__INDEX);
}

#endif
