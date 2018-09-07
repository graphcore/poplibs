#ifdef __IPU__

#include "tileimplconsts_tommy.h"

#define VERTEX __runCodelet_popnn__SumPoolingGrad___float

.globl VERTEX
.type VERTEX, @function

// vertex state
#define VERTEX_OUT_GRAD_OFFSET 0
#define VERTEX_IN_GRAD_BEGIN_OFFSET 1
#define VERTEX_IN_GRAD_END_OFFSET 2
#define VERTEX_WINDOW_SIZES_OFFSET 3

// stack state
#define STACK_OUT_GRAD_PTR_PTR_OFFSET 0

// constants
#define LOG2_SIZEOF_TWO_PTR 3
#define LOG2_SIZEOF_FLOAT 2

// integer variables
#define outGradPtrPtr m0
#define inGradPtrPtr m1
#define inGradSize m2
#define windowSizes m3

#define outGradPtr m4
#define inGradPtr m5
#define chanSize m6
#define chanSizeD2 m7
#define windowSize m8

#define mscratch m11

// floating point variables
#define value a0:1
#define valuei0 a0
#define outGrad a2:3
#define outGradi0 a2

.section .text.VERTEX
.align 8
  nop
VERTEX:
  ld32 $outGradPtrPtr, $mzero, $mvertex_base, VERTEX_OUT_GRAD_OFFSET
  ld32 $inGradPtrPtr, $mzero, $mvertex_base, VERTEX_IN_GRAD_BEGIN_OFFSET
  ld32 $inGradSize, $mzero, $mvertex_base, VERTEX_IN_GRAD_END_OFFSET
  ld32 $windowSizes, $mzero, $mvertex_base, VERTEX_WINDOW_SIZES_OFFSET

  // calculate inGradSize by doing (end-begin)/sizeof(TWO_PTR)
  sub $inGradSize, $inGradSize, $inGradPtrPtr
  shr $inGradSize, $inGradSize, LOG2_SIZEOF_TWO_PTR

  // minus 1 for the outer_loop brnzdec, additionally skip the readjustment
  // of outGradPtrPtr on the first iteration.
  brnzdec $inGradSize, 1f

outer_loop:
  // outGradPtrPtr is over-read in the channel loop so move it back here.
  add $outGradPtrPtr, $outGradPtrPtr, -4

1:
  ld32step $inGradPtr, $mzero, $inGradPtrPtr+=, 1
  ld32step $chanSize, $mzero, $inGradPtrPtr+=, 1
  ldz16step $windowSize, $mzero, $windowSizes+=, 1

  // save outGradPtrPtr to the stack as we need to reset it each iteration of
  // the chan_loop. we could use a register here but there is no cycle gain
  // so may as well use the stack.
  st32 $outGradPtrPtr, $mzero, $mworker_base, STACK_OUT_GRAD_PTR_PTR_OFFSET

  // calculate chanSize by doing (end-begin)/sizeof(float)
  sub $chanSize, $chanSize, $inGradPtr
  shr $chanSize, $chanSize, LOG2_SIZEOF_FLOAT

  // do 2 at a time first.
  shr $chanSizeD2, $chanSize, 1

  // minus 1 for the chan2_loop brnzdec but also check to see if chanSizeD2 is
  // zero and skip the vectorised loop if so. additionally skip the first
  // loading of outGradPtrPtr off the stack as it is currently correct
  brnzdec $chanSizeD2, 1f
  bri chan1_loop

chan2_loop:
  // reload outGradPtrPtr
  ld32 $outGradPtrPtr, $mzero, $mworker_base, STACK_OUT_GRAD_PTR_PTR_OFFSET

1:
  {
    // load the first outGradPtr
    ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1
    zero $value
  }

  rpt $windowSize, (2f-1f)/8-1
1:
  {
    ld64 $outGrad, $mzero, $outGradPtr, $chanSizeD2
    fnop
  }
  {
    // specutively load the next outGradPtr
    ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1
    f32v2add $value, $value, $outGrad
  }
2:
  st64 $value, $mzero, $inGradPtr, $chanSizeD2

  brnzdec $chanSizeD2, chan2_loop

chan1_loop:
  and $mscratch, $chanSize, 0x1
  brz $mscratch, outer_loop_epilogue

  // reload outGradPtrPtr
  ld32 $outGradPtrPtr, $mzero, $mworker_base, STACK_OUT_GRAD_PTR_PTR_OFFSET

  // minus 1 as we use chanSize as a zero-based index.
  add $chanSize, $chanSize, -1

  {
    ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1
    zero $valuei0
  }

  {
    rpt $windowSize, (2f-1f)/8-1
    fnop
  }
1:
  {
    ld32 $outGradi0, $mzero, $outGradPtr, $chanSize
    fnop
  }
  {
    // specutively load the next outGradPtr
    ld32step $outGradPtr, $mzero, $outGradPtrPtr+=, 1
    f32add $valuei0, $valuei0, $outGradi0
  }
2:
  st32 $valuei0, $mzero, $inGradPtr, $chanSize

outer_loop_epilogue:
  brnzdec $inGradSize, outer_loop
  exitz $mzero

.size VERTEX, .-VERTEX

#endif // __IPU__
