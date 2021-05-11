// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#ifndef __DYNAMIC_SLICE_INC__
#define __DYNAMIC_SLICE_INC__

// Common definitions for the dynamicSlice/MultiSlice code


// Parameters for the copy macros/functions defined in
// dynamicSliceSupervisor_8bit.S, MultiSlice.S and SliceCopyFunction.S
#define mReturnAddress  m7
#define mSrcPtr         m8
#define mDstPtr         m9
#define mRegionSize     m10
#define mScratch        m11

// ARF registers used to load/store values during copy
#define VAL12   a0:1
#define VAL1    a0
#define VAL2    a1
#define VAL3    a2


//****************************************************************************
// Offset in scratch area ('worker_base') where we save registers
//****************************************************************************

// Save area for DynamicSlice2d_common
#define WOFF_REGIONS         0
#define WOFF_BASET_BASE      1
#define WOFF_SUBT_BASE       2

// Save area for 'Slice_copy_function_8bit'
#define WOFF_SAVE_REGS1      3
#define WOFF_SAVE_REGS2      4
#define WOFF_SAVE_REGS3      5
#define WOFF_SAVE_REGS4      6

#endif // __DYNAMIC_SLICE_INC__
#endif // __IPU__
