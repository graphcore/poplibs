// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__


//******************************************************************************
// MACRO to divide work between 6 workers
//
// C equivalent of
// (((size >> vectorWidthShifts) + CTXT_WORKERS - 1 - worker) * 0xaaab) >> 18;
// When dividing number of elements by 24 (4 halves per worker)
// then grainsize = 4, SHIFTS_FOR_GRAINSIZE = 2
// When dividing number of elements by 12 (2 floats per worker)
// then grainsize = 2, SHIFTS_FOR_GRAINSIZE = 1
//
// With SHIFTS_FOR_GRAINSIZE = 2 this is correct up to and including 393195
// With SHIFTS_FOR_GRAINSIZE = 1 this is correct up to and including 196597
// With SHIFTS_FOR_GRAINSIZE = 0 this is correct up to and including 98299
//
// Reason for failure is for example:
// (((196597 + 1) >> 1) + 5) * 0xAAAB = 0x100008000
// and is greater than the 32 bit register size
//******************************************************************************

// For use as SHIFTS_FOR_GRAINSIZE
#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE 2
#define RECIPROCAL_3_SHL17 ((((1 << 17) - 1) / 3) + 1)
#define DIV_6_SHIFT 18

.macro DIVIDE_BY_WORKER mIN mWORKER mSCRATCH mOUT SHIFTS_FOR_GRAINSIZE
  shr \mOUT, \mIN, \SHIFTS_FOR_GRAINSIZE
  add \mOUT, \mOUT, CTXT_WORKERS-1
  sub \mOUT, \mOUT, \mWORKER
  setzi \mSCRATCH, RECIPROCAL_3_SHL17
  mul \mOUT, \mOUT, \mSCRATCH
  shr \mOUT, \mOUT, DIV_6_SHIFT
.endm

// As above but omit the 1st shift, as some use cases need the shifted result
// to be preserved or to divide by 6 instead of 12 or 24
.macro DIVIDE_BY_WORKER_PRE_SHIFTED mIN mWORKER mSCRATCH mOUT
  add \mOUT, \mIN, CTXT_WORKERS-1
  sub \mOUT, \mOUT, \mWORKER
  setzi \mSCRATCH, RECIPROCAL_3_SHL17
  mul \mOUT, \mOUT, \mSCRATCH
  shr \mOUT, \mOUT, DIV_6_SHIFT
.endm

//******************************************************************************
// MACRO to Divide by 12 or 24 without using a worker ID to do rounding
// Also provides a remainder (Which can be used later, to compare to the workerId
// and deal with the remainder, or for other purposes)
// Working on the same principle as above, but merging the 2 shifts.  This saves
// a cycle, but there is no shift before the multiply therefore overflow of the
// multiplication happens earlier :
// 98303 is the largest input where the multiplication result does not overflow
//******************************************************************************
.macro SPLIT_BETWEEN_WORKERS n size rem divisor
.ifc "\divisor","12"
  .equ SHIFT, (DIV_6_SHIFT + LOG2_FLOAT_ATOM_SIZE)
.else
  .equ SHIFT, (DIV_6_SHIFT + LOG2_HALF_ATOM_SIZE)
.endif
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, SHIFT
    mul \rem, \size, \divisor
    sub \rem, \n, \rem
.endm

#endif // __IPU__
