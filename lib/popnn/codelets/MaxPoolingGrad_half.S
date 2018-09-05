#ifdef __IPU__

#include "tileimplconsts_tommy.h"

#define VERTEX __runCodelet_popnn__MaxPoolingGrad___half

.globl VERTEX
.type VERTEX, @function

// vertex state
#define VERTEX_OUT_OFFSET 0
#define VERTEX_OUT_GRAD_OFFSET 1
#define VERTEX_IN_BASE_OFFSET 1 // ld32
#define VERTEX_IN_PTR_OFFSET 4
#define VERTEX_WINDOW_SIZES_OFFSET 5
#define VERTEX_IN_GRAD_BASE_OFFSET 3 // ld32
#define VERTEX_IN_GRAD_PTR_OFFSET 8

// stack state
#define STACK_OUT_GRAD_PTR_PTR_OFFSET 0
#define STACK_OUT_PTR_PTR_OFFSET 1
#define STACK_CHAN_SIZE_OFFSET 2
#define STACK_IN_BASE_OFFSET 3
#define STACK_IN_GRAD_BASE_OFFSET 4

// constants
#define SCALED_PTR32_SHL_BITS 2
#define VECTOR_LIST_BASE_BITS 20
#define VECTOR_LIST_LENGTH_BITS 12
#define DELTAN_OFFSET_BITS 18
#define DELTAN_LENGTH_BITS 14

#define LOG2_VECTOR_WIDTH 2

// integer variables
#define outPtrPtr m0
#define outGradPtrPtr m1
#define inBase m2
#define inDeltaPtr m3
#define inGradBase m4
#define inGradSize m5
#define inGradDeltaPtr m6
#define windowSizes m7

#define inOffset m8
#define inGradOffset m9
#define windowSize m10
#define chanSize m11
#define outPtr m2 // same as inBase
#define outGradPtr m4 // same as inGradBase

// floating point variables
#define outGrad a0:1
#define outGradi0 a0
#define in a2:3
#define ini0 a2
#define out a4:5
#define outi0 a4
#define value a6:7
#define valuei0 a6

.section .text.VERTEX
.align 8
  nop
VERTEX:
  ldz16 $outPtrPtr, $mvertex_base, $mzero, VERTEX_OUT_OFFSET
  ldz16 $outGradPtrPtr, $mvertex_base, $mzero, VERTEX_OUT_GRAD_OFFSET
  ld32 $inBase, $mvertex_base, $mzero, VERTEX_IN_BASE_OFFSET
  ldz16 $inDeltaPtr, $mvertex_base, $mzero, VERTEX_IN_PTR_OFFSET
  ldz16 $windowSizes, $mvertex_base, $mzero, VERTEX_WINDOW_SIZES_OFFSET
  ld32 $inGradBase, $mvertex_base, $mzero, VERTEX_IN_GRAD_BASE_OFFSET
  ldz16 $inGradDeltaPtr, $mvertex_base, $mzero, VERTEX_IN_GRAD_PTR_OFFSET

  // unpack scaled pointers.
  setzi $m11, TMEM_REGION0_BASE_ADDR
  shl $outPtrPtr, $outPtrPtr, SCALED_PTR32_SHL_BITS
  add $outPtrPtr, $outPtrPtr, $m11
  shl $outGradPtrPtr, $outGradPtrPtr, SCALED_PTR32_SHL_BITS
  add $outGradPtrPtr, $outGradPtrPtr, $m11
  shl $inDeltaPtr, $inDeltaPtr, SCALED_PTR32_SHL_BITS
  add $inDeltaPtr, $inDeltaPtr, $m11
  shl $inGradDeltaPtr, $inGradDeltaPtr, SCALED_PTR32_SHL_BITS
  add $inGradDeltaPtr, $inGradDeltaPtr, $m11
  shl $windowSizes, $windowSizes, SCALED_PTR32_SHL_BITS
  add $windowSizes, $windowSizes, $m11

  // unpack base for $in
  shl $inBase, $inBase, VECTOR_LIST_LENGTH_BITS
  shr $inBase, $inBase, VECTOR_LIST_LENGTH_BITS

  // unpack base and length for $inGrad
  shr $inGradSize, $inGradBase, VECTOR_LIST_BASE_BITS
  shl $inGradBase, $inGradBase, VECTOR_LIST_LENGTH_BITS
  shr $inGradBase, $inGradBase, VECTOR_LIST_LENGTH_BITS

  // save $inBase and $inGradBase to the stack so we can use those registers
  // for something else in the channel loops.
  st32 $inBase, $mzero, $mworker_base, STACK_IN_BASE_OFFSET
  st32 $inGradBase, $mzero, $mworker_base, STACK_IN_GRAD_BASE_OFFSET

  // minus 1 for the brnzdec at the bottom of the loop. additionally skip
  // adjusting the pointers and loading inBase and inGradBase which are already
  // in the correct registers on the first iteration.
  brnzdec $inGradSize, 1f

.Louter_loop:
  // outPtrPtr and outGradPtrPtr over-read in the chan loop so we need to move
  // them back one to get them to the start of the next window for all but the
  // first iteration.
  add $outPtrPtr, $outPtrPtr, -4
  add $outGradPtrPtr, $outGradPtrPtr, -4

  // pull inBase and inGradBase off the stack to create complete pointers for
  // the next DeltaN.
  ld32 $inBase, $mzero, $mworker_base, STACK_IN_BASE_OFFSET
  ld32 $inGradBase, $mzero, $mworker_base, STACK_IN_GRAD_BASE_OFFSET

1:
  ld32step $inOffset, $mzero, $inDeltaPtr+=, 1
  ld32step $inGradOffset, $mzero, $inGradDeltaPtr+=, 1
  ldz16step $windowSize, $mzero, $windowSizes+=, 1

  // unpack offset from inOffset and size and offset inGradOffset
  shl $inOffset, $inOffset, DELTAN_LENGTH_BITS
  shr $inOffset, $inOffset, DELTAN_LENGTH_BITS
  shr $chanSize, $inGradOffset, DELTAN_OFFSET_BITS
  shl $inGradOffset, $inGradOffset, DELTAN_LENGTH_BITS
  shr $inGradOffset, $inGradOffset, DELTAN_LENGTH_BITS

  // add base to the offset now because we don't have enough registers to keep
  // them around (they share with $outPtr and $outGradPtr).
  add $inOffset, $inBase, $inOffset
  add $inGradOffset, $inGradBase, $inGradOffset

  // save chan size so we can reuse it later (it gets trashed by the brnzdec).
  st32 $chanSize, $mzero, $mworker_base, STACK_CHAN_SIZE_OFFSET

  // save $outGradPtrPtr and $outPtrPtr to the stack because we need to go back
  // each iteration.
  st32 $outGradPtrPtr, $mzero, $mworker_base, STACK_OUT_GRAD_PTR_PTR_OFFSET
  st32 $outPtrPtr, $mzero, $mworker_base, STACK_OUT_PTR_PTR_OFFSET

  // process multiple channels at a time.
  shr $chanSize, $chanSize, LOG2_VECTOR_WIDTH
  // and minus 1 for the brnzdec at the bottom of the loop. additionally, first
  // time we can skip loading our outer pointers to save a cycle. take advantage
  // of brnzdec to do both of these things.
  brnzdec $chanSize, 1f

  // less channels than a vector width case
  bri .Lchan2_loop

.Lchan4_loop:
  // reload outGradPtrPtr and outPtrPtr.
  ld32 $outGradPtrPtr, $mzero, $mworker_base, STACK_OUT_GRAD_PTR_PTR_OFFSET
  ld32 $outPtrPtr, $mzero, $mworker_base, STACK_OUT_PTR_PTR_OFFSET

1:
  ld64 $in, $mzero, $inOffset, $chanSize

  // pre-load the output pointers for the first iteration.
  ld32step $outPtr, $mzero, $outPtrPtr+=, 1
  ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1

  {
    rpt $windowSize, (2f-1f)/8-1
    zero $value
  }
1:
  {
    ld64 $out, $mzero, $outPtr, $chanSize
    fnop
  }
  {
    ld64 $outGrad, $mzero, $outGradPtr, $chanSize
    // floating point cmpeq returns either all 1's or all 0's, which we can use
    // to mask the outGrad and sum the ones that match.
    f16v4cmpeq $out, $out, $in
  }
  {
    // speculatively load pointers for the next iteration.
    ld32step $outPtr, $mzero, $outPtrPtr+=, 1
    and64 $outGrad, $out, $outGrad
  }
  {
    // load outGradPtr too
    ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1
    f16v4add $value, $value, $outGrad
  }
2:
  // store the final result
  st64 $value, $mzero, $inGradOffset, $chanSize
  brnzdec $chanSize, .Lchan4_loop

.Lchan2_loop:
  // reload chanSize, see if we have to do some more (max of 3), do that here.
  ld32 $chanSize, $mzero, $mworker_base, STACK_CHAN_SIZE_OFFSET

  // use outPtr as a scratch register.
  and $outPtr, $chanSize, 0x3
  // is there at least 2 left?
  cmpult $outPtr, $outPtr, 2
  brnz $outPtr, .Lchan1_loop

  // reload outGradPtrPtr and outPtrPtr.
  ld32 $outGradPtrPtr, $mzero, $mworker_base, STACK_OUT_GRAD_PTR_PTR_OFFSET
  ld32 $outPtrPtr, $mzero, $mworker_base, STACK_OUT_PTR_PTR_OFFSET

  // divide 2 and minus 1 as we use chanSize as a zero-based index.
  shr $chanSize, $chanSize, 1
  add $chanSize, $chanSize, -1

  {
    ld32 $ini0, $mzero, $inOffset, $chanSize
    zero $valuei0
  }

  ld32step $outPtr, $mzero, $outPtrPtr+=, 1
  ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1

  {
    rpt $windowSize, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld32 $outi0, $mzero, $outPtr, $chanSize
    fnop
  }
  {
    ld32 $outGradi0, $mzero, $outGradPtr, $chanSize
    f16v2cmpeq $outi0, $outi0, $ini0
  }
  {
    ld32step $outPtr, $mzero, $outPtrPtr+=, 1
    and $outGradi0, $outi0, $outGradi0
  }
  {
    ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1
    f16v2add $valuei0, $valuei0, $outGradi0
  }
2:
  // store the final result
  st32 $valuei0, $mzero, $inGradOffset, $chanSize

.Lchan1_loop:
  // reload chanSize, see if we have to do one more, do that here.
  ld32 $chanSize, $mzero, $mworker_base, STACK_CHAN_SIZE_OFFSET

  // use outPtr as a scratch register.
  and $outPtr, $chanSize, 0x1
  brz $outPtr, .Louter_loop_epilogue

  // reload outGradPtrPtr and outPtrPtr.
  ld32 $outGradPtrPtr, $mzero, $mworker_base, STACK_OUT_GRAD_PTR_PTR_OFFSET
  ld32 $outPtrPtr, $mzero, $mworker_base, STACK_OUT_PTR_PTR_OFFSET

  // minus 1 as we use chanSize as a zero-based index.
  add $chanSize, $chanSize, -1

  ldb16 $ini0, $mzero, $inOffset, $chanSize

  ld32step $outPtr, $mzero, $outPtrPtr+=, 1
  ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1

  {
    rpt $windowSize, (2f-1f)/8-1
    zero $valuei0
  }
1:
  {
    ldb16 $outi0, $mzero, $outPtr, $chanSize
    fnop
  }
  {
    ldb16 $outGradi0, $mzero, $outGradPtr, $chanSize
    f16v2cmpeq $outi0, $outi0, $ini0
  }
  {
    ld32step $outPtr, $mzero, $outPtrPtr+=, 1
    and $outGradi0, $outi0, $outGradi0
  }
  {
    ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1
    f16v2add $valuei0, $valuei0, $outGradi0
  }
2:
  // read/modify/write to store the final result, storing is done at a 32-bit
  // granularity so halve the offset first.
  shr $chanSize, $chanSize, 1
  ld32 $outi0, $mzero, $inGradOffset, $chanSize
  sort4x16hi $valuei0, $valuei0, $outi0
  st32 $valuei0, $mzero, $inGradOffset, $chanSize

.Louter_loop_epilogue:
  brnzdec $inGradSize, .Louter_loop
  exitz $mzero

.size VERTEX, .-VERTEX

#endif // __IPU__
