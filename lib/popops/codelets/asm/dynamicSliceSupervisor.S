// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Dynamic Slice and Dynamic Update Slice Supervisor
// vertex code for int, half and float variants
/* -------------------------------------------------------------------------- */

// Supervisor registers
#define SUP_VERTEX_BASE m0
#define WORKER_ENTRY    m4

// Worker Registers
#define WKR_ID      m0
#define SRC         m1

#define DST         m2
#define BASE_ELEM   m3
#define SUB_ELEM    m4
#define ELEMENTS_PER_WORKER m5

#define mSCRATCH    m6
#define mSCRATCH2   m7

#define BASE_SLICE  m8
#define mSCRATCH3   m9
#define SUB_REGION  m10
#define REGION_SIZE m11

#define VAL12   a0:1
#define VAL1    a0
#define VAL2    a1
#define VAL3    a2


#define UPDATE  a5

//****************************************************************************
// The input structure parameters:
// 32 bit offset
// 32 ptr baseT
// 32 ptr subT
// 32 bit numBaseElements
// 16 bit numSubElements
// 16 bit regionSize
//****************************************************************************
#define VOFF_OFFSET      0
#define VOFF_BASET       1
#define VOFF_SUBT        2
#define VOFF_BASE_ELEM   3
// 16 bit - offsets for use with 16 bit loads
#define VOFF_SUB_ELEM    (4*2)
#define VOFF_REGION_SIZE (4*2+1)

//****************************************************************************
// Worker base local variable storage
//****************************************************************************
#define WOFF_ELEMENTS 0
#define WOFF_ELEMENTS_DEC 1


#include "poplibs_support/TileConstants.hpp"

#define DSS_FLOAT_FUNC __runCodelet_popops__DynamicSlice1d___float
#define DUSS_FLOAT_FUNC __runCodelet_popops__DynamicUpdateSlice1d___float

#define DSS_INT_FUNC __runCodelet_popops__DynamicSlice1d___int
#define DUSS_INT_FUNC __runCodelet_popops__DynamicUpdateSlice1d___int

#define DSS_UNSIGNED_FUNC __runCodelet_popops__DynamicSlice1d___unsigned_int
#define DUSS_UNSIGNED_FUNC __runCodelet_popops__DynamicUpdateSlice1d___unsigned_int

#define DSS_HALF_FUNC __runCodelet_popops__DynamicSlice1d___half
#define DUSS_HALF_FUNC __runCodelet_popops__DynamicUpdateSlice1d___half

.global DSS_FLOAT_FUNC
.type DSS_FLOAT_FUNC, @function

.global DUSS_FLOAT_FUNC
.type DUSS_FLOAT_FUNC, @function

.global DSS_INT_FUNC
.type DSS_INT_FUNC, @function

.global DUSS_INT_FUNC
.type DUSS_INT_FUNC, @function

.global DSS_UNSIGNED_FUNC
.type DSS_UNSIGNED_FUNC, @function

.global DUSS_UNSIGNED_FUNC
.type DUSS_UNSIGNED_FUNC, @function

.global DSS_HALF_FUNC
.type DSS_HALF_FUNC, @function

.global DUSS_HALF_FUNC
.type DUSS_HALF_FUNC, @function

// ************************************************* //
// Supervisor vertex - entry point, run workers
// Float variants
// ************************************************* //
.section .text.DynamicSlice1d_float
.align 8
.supervisor

DSS_INT_FUNC:
DSS_UNSIGNED_FUNC:
DSS_FLOAT_FUNC:
  setzi        $WORKER_ENTRY, .Lds_super_worker_32
  runall       $WORKER_ENTRY, $SUP_VERTEX_BASE, 0
  sync         TEXCH_SYNCZONE_LOCAL
  br           $lr

DUSS_INT_FUNC:
DUSS_UNSIGNED_FUNC:
DUSS_FLOAT_FUNC:
  setzi        $WORKER_ENTRY, .Ldus_super_worker_32
  runall       $WORKER_ENTRY, $SUP_VERTEX_BASE, 0
  sync         TEXCH_SYNCZONE_LOCAL
  br           $lr
// ************************************************* //
// Worker thread execution for supervisor vertex
// Update version: set a flag for update,
// load base, sub array pointers
// Src = Sub
// Dst = Base
// ************************************************* //
.Ldus_super_worker_32:
.worker
  setzi     $UPDATE,1
  ld32      $DST, $mzero, $mvertex_base, VOFF_BASET
  ld32      $SRC, $mzero, $mvertex_base, VOFF_SUBT
  bri       .Lds_super_common_32
// ************************************************* //
// Worker thread execution for supervisor vertex
// Non Update version: clear a flag for no update,
// load base, sub array pointers
// Src = Base
// Dst = Sub
// ************************************************* //
.Lds_super_worker_32:
  setzi     $UPDATE,0
  ld32      $SRC, $mzero, $mvertex_base, VOFF_BASET
  ld32      $DST, $mzero, $mvertex_base, VOFF_SUBT

.Lds_super_common_32:
  // Call pre-loop calculation, common between all variants
  call      $mSCRATCH2, load_and_preprocess

  // Region size converted into bytes
  ldz16     $REGION_SIZE, $mzero, $mvertex_base, VOFF_REGION_SIZE
  shl       $REGION_SIZE, $REGION_SIZE, 2

.Lsub_loop:
  // Calculate temporary incrementing pointers for the copy
  mov   $mSCRATCH2, $SUB_REGION
  mul   $mSCRATCH, $REGION_SIZE, $BASE_SLICE
  // Pointers swapped for the update version
  atom  $mSCRATCH3, $UPDATE
  brz   $mSCRATCH3, 1f
  mov   $mSCRATCH, $SUB_REGION
  mul   $mSCRATCH2, $REGION_SIZE, $BASE_SLICE
1:
  // Is a 32 bit copy needed to align the output to 64 bits?
  ld32      $ELEMENTS_PER_WORKER, $mzero, $mworker_base, WOFF_ELEMENTS
  add       $mSCRATCH3, $mSCRATCH2, $DST
  and       $mSCRATCH3, $mSCRATCH3, 4
  brz       $mSCRATCH3, .Lout_aligned_f

  // Misaligned for 64  bit copies.  Worker 0 copies 1x32 bits. All workers
  // need the base source and destination addresses to be incremented
  // The remaining code needs to know that 1 less item is now to be copied,
  // which is done by selecting the decremented elements per worker
  // value which was pre-computed.
  ld32      $ELEMENTS_PER_WORKER, $mzero, $mworker_base, WOFF_ELEMENTS_DEC
  ld32step  $VAL1, $SRC, $mSCRATCH+=, 1
  brnz      $WKR_ID, 5f
  st32      $VAL1, $DST, $mSCRATCH2, 0
5:
  ld32step  $VAL1, $DST, $mSCRATCH2+=, 1
.Lout_aligned_f:
  // Output is now aligned, is the input aligned?
  add       $mSCRATCH3, $mSCRATCH, $SRC
  and       $mSCRATCH3, $mSCRATCH3, 4
  brz       $mSCRATCH3, .Lin_out_aligned
  // Dummy load, advancing in ptr and out ptr by wkr_id * 64 bits
  // knowing that the destination is misaligned.
  // Now all workers have a different pointer, offset by 64 bits
  // Repeat loop maintains this difference
  ld32step   $VAL1, $SRC, $mSCRATCH+=, $WKR_ID
  ld32step   $VAL1, $SRC, $mSCRATCH+=, $WKR_ID
  ld64step   $VAL12, $DST, $mSCRATCH2+=, $WKR_ID

  rpt        $ELEMENTS_PER_WORKER, (2f-1f)/8 -1;
1:
  {ld32step  $VAL1, $SRC , $mSCRATCH+=, 1; fnop}
  {ld32step  $VAL2, $SRC , $mSCRATCH+=, (2 * CTXT_WORKERS -1); fnop}
  {st64step  $VAL12, $DST, $mSCRATCH2+=, CTXT_WORKERS; fnop}
2:
  bri   .Lcontinue

.Lin_out_aligned:
  // Dummy load, advancing in ptr and out ptr by wkr_id * 64 bits
  // Now all workers have a different pointer, offset by 64 bits
  // Repeat loop maintains this difference
  ld64step   $VAL12, $SRC, $mSCRATCH+=, $WKR_ID
  ld64step   $VAL12, $DST, $mSCRATCH2+=, $WKR_ID

  rpt       $ELEMENTS_PER_WORKER, (2f-1f)/8 -1
1:
  {ld64step  $VAL12, $SRC , $mSCRATCH+=, CTXT_WORKERS; fnop}
  {st64step  $VAL12, $DST, $mSCRATCH2+=, CTXT_WORKERS; fnop}
2:
.Lcontinue:
  // Is there a last item to copy?
  // If so, select the worker that is pointing to it to copy it.  This is OK
  // in the case of float, but not half due to the half word writes being
  // non atomic.
  atom      $mSCRATCH3, $UPDATE
  brz       $mSCRATCH3, 1f
  // Update version of the code
  add       $mSCRATCH3, $REGION_SIZE, $SUB_REGION
  sub       $mSCRATCH3, $mSCRATCH3, $mSCRATCH
  add       $mSCRATCH3, $mSCRATCH3, -4
  brnz      $mSCRATCH3, 3f
  bri       2f
1:
  // Non update version of the code
  add       $mSCRATCH3, $REGION_SIZE, $SUB_REGION
  sub       $mSCRATCH3, $mSCRATCH3, $mSCRATCH2
  add       $mSCRATCH3, $mSCRATCH3, -4
  brnz      $mSCRATCH3, 3f
2:
  // Do the last copy
  ld32      $VAL1, $SRC, $mSCRATCH, 0
  st32      $VAL1, $DST, $mSCRATCH2, 0
3:
  // Increment base slice and wrap to start if required
  add        $BASE_SLICE, $BASE_SLICE, 1
  cmpult     $mSCRATCH, $BASE_SLICE, $BASE_ELEM
  brnz       $mSCRATCH, 4f
  zero       $BASE_SLICE
4:
  // The next sub region is just region size bytes afterwards
  add        $SUB_REGION, $SUB_REGION, $REGION_SIZE
  brnzdec    $SUB_ELEM, .Lsub_loop

  exitz      $mzero
.size DynamicSlice1d_float, .-DSS_FLOAT_FUNC

// ************************************************* //
// Supervisor vertex - entry point, run workers
// Half variants
// ************************************************* //
.section .text.DynamicSlice1d_half
.align 8
.supervisor

DSS_HALF_FUNC:
  setzi        $WORKER_ENTRY, .Lds_super_worker_16
  runall       $WORKER_ENTRY, $SUP_VERTEX_BASE, 0
  sync         TEXCH_SYNCZONE_LOCAL
  br           $lr

DUSS_HALF_FUNC:
  setzi        $WORKER_ENTRY, .Ldus_super_worker_16
  runall       $WORKER_ENTRY, $SUP_VERTEX_BASE, 0
  sync         TEXCH_SYNCZONE_LOCAL
  br           $lr
// ************************************************* //
// Worker thread execution for supervisor vertex
// Update version: set a flag for update,
// load base, sub array pointers
// Src = Sub
// Dst = Base
// ************************************************* //
.Ldus_super_worker_16:
.worker
  setzi     $UPDATE,1
  ld32      $DST, $mzero, $mvertex_base, VOFF_BASET
  ld32      $SRC, $mzero, $mvertex_base, VOFF_SUBT
  bri       .Lds_super_common_16
// ************************************************* //
// Worker thread execution for supervisor vertex
// Non Update version: clear a flag for update,
// load base, sub array pointers
// Src = Base
// Dst = Sub
// ************************************************* //
.Lds_super_worker_16:
  setzi     $UPDATE,0
  ld32      $SRC, $mzero, $mvertex_base, VOFF_BASET
  ld32      $DST, $mzero, $mvertex_base, VOFF_SUBT

.Lds_super_common_16:
  // Call pre-loop calculation, common between all variants
  call      $mSCRATCH2, load_and_preprocess

  // Region size converted into bytes
  ldz16     $REGION_SIZE, $mzero, $mvertex_base, VOFF_REGION_SIZE
  shl       $REGION_SIZE, $REGION_SIZE, 1
.Lsub_loop2:
  // Calculate temporary incrementing pointers for the copy
  mov   $mSCRATCH2, $SUB_REGION
  mul   $mSCRATCH, $REGION_SIZE, $BASE_SLICE

  // Update/ non update versions require source/dest to be swapped
  atom  $mSCRATCH3, $UPDATE
  brz   $mSCRATCH3, 1f
  mov   $mSCRATCH, $SUB_REGION
  mul   $mSCRATCH2, $REGION_SIZE, $BASE_SLICE
1:

  // Is a 16 bit copy needed to align the output to 32 bits?
  ld32      $ELEMENTS_PER_WORKER, $mzero, $mworker_base, WOFF_ELEMENTS
  add       $mSCRATCH3, $mSCRATCH2, $DST
  and       $mSCRATCH3, $mSCRATCH3, 2
  brz       $mSCRATCH3, .Lout_aligned_h

  // Misaligned for 32  bit copies.  Worker 5 copies 1x16 bit. All workers
  // need the base source and destination addresses to be incremented.
  // Choose the last worker as it may have slightly less to copy than others.
  // The remaining code needs to know that 1 less item is now to be copied,
  // which is done by selecting the decremented elements per worker
  // value which was pre-computed.
  ld32       $ELEMENTS_PER_WORKER, $mzero, $mworker_base, WOFF_ELEMENTS_DEC
  ldb16step  $VAL1, $SRC, $mSCRATCH+=, 1
  cmpeq      $mSCRATCH3, $WKR_ID, (CTXT_WORKERS-1)
  brz        $mSCRATCH3, 5f

  add        $mSCRATCH2, $mSCRATCH2, -2
  ldb16      $VAL2, $DST, $mSCRATCH2,0
  roll16     $VAL1, $VAL2, $VAL1
  st32step   $VAL1, $DST, $mSCRATCH2+=, 1
  bri        .Lout_aligned_h
5:
  ldb16step  $VAL1, $DST, $mSCRATCH2+=, 1
.Lout_aligned_h:
  // Output is now aligned, is the input aligned?
  add       $mSCRATCH3, $mSCRATCH, $SRC
  and       $mSCRATCH3, $mSCRATCH3, 2
  brz       $mSCRATCH3, .Lin_out_aligned2
  // Dummy load, advancing in ptr and out ptr by wkr_id * 32 bits
  // knowing that the destination is misaligned
  // Now all workers have a different pointer, offset by 32 bits
  // Repeat loop maintains this difference
  ldb16step $VAL1, $SRC, $mSCRATCH+=, $WKR_ID
  ldb16step $VAL1, $SRC, $mSCRATCH+=, $WKR_ID
  ld32step  $VAL1, $DST, $mSCRATCH2+=, $WKR_ID

  // Prepare the 1st value to store so that we can use a bundle in the loop
  ldb16step  $VAL1, $SRC , $mSCRATCH+=, 1
  ldb16step  $VAL2, $SRC , $mSCRATCH+=, (2 * CTXT_WORKERS -1)

  {ldb16step  $VAL1, $SRC , $mSCRATCH+=, 1
   roll16     $VAL3, $VAL1, $VAL2}
  ldb16step   $VAL2, $SRC , $mSCRATCH+=, (2 * CTXT_WORKERS -1)

  rpt        $ELEMENTS_PER_WORKER, (2f-1f)/8 -1
1:
  {st32step   $VAL3, $DST, $mSCRATCH2+=, CTXT_WORKERS
   roll16     $VAL3, $VAL1, $VAL2}
  {ldb16step  $VAL1, $SRC , $mSCRATCH+=, 1; fnop}
  {ldb16step  $VAL2, $SRC , $mSCRATCH+=, (2 * CTXT_WORKERS -1); fnop}
2:
  bri   .Lcontinue2

.Lin_out_aligned2:
  // Dummy load, advancing in ptr and out ptr by wkr_id * 32 bits
  // Now all workers have a different pointer, offset by 32 bits
  // Repeat loop maintains this difference
  ld32step   $VAL1, $SRC, $mSCRATCH+=, $WKR_ID
  ld32step   $VAL1, $DST, $mSCRATCH2+=, $WKR_ID

  rpt        $ELEMENTS_PER_WORKER, (2f-1f)/8 -1
1:
  {ld32step  $VAL1, $SRC , $mSCRATCH+=, CTXT_WORKERS; fnop}
  {st32step  $VAL1, $DST, $mSCRATCH2+=, CTXT_WORKERS; fnop}
2:
.Lcontinue2:
  // Is there a last item to copy?
  // If so, we could select the worker that is pointing to it to copy it
  // BUT if worker 5 is dealing with the read modify write of a 1st
  // misaligned item the accesses could clash.  So instead make worker
  // 5 do this despite having to calculate the pointers
  cmpeq     $mSCRATCH3, $WKR_ID, (CTXT_WORKERS -1)
  brz       $mSCRATCH3, 3f
  atom      $mSCRATCH3, $UPDATE
  brnz      $mSCRATCH3, 1f
  // non-update version
  add       $mSCRATCH2, $REGION_SIZE, $SUB_REGION
  add       $mSCRATCH, $BASE_SLICE, 1
  mul       $mSCRATCH, $REGION_SIZE, $mSCRATCH
  bri       2f
  // update version
1:
  add       $mSCRATCH, $REGION_SIZE, $SUB_REGION
  add       $mSCRATCH2, $BASE_SLICE, 1
  mul       $mSCRATCH2, $REGION_SIZE, $mSCRATCH2
2:
  add       $mSCRATCH2, $mSCRATCH2, $DST
  // There will be no last one to copy if the store pointer generated is
  // 32 bit aligned
  add       $mSCRATCH2, $mSCRATCH2, -2
  and       $mSCRATCH3, $mSCRATCH2, 2
  brnz      $mSCRATCH3, 3f
  // do the last copy
  add        $mSCRATCH, $mSCRATCH, -2
  ldb16step  $VAL1, $SRC, $mSCRATCH+=, 1
  ldb16      $VAL2, $mzero, $mSCRATCH2, 1
  roll16     $VAL1, $VAL1, $VAL2
  st32       $VAL1, $mzero, $mSCRATCH2, 0
3:
  // Increment base slice and wrap to start if required
  add        $BASE_SLICE, $BASE_SLICE, 1
  cmpult     $mSCRATCH, $BASE_SLICE, $BASE_ELEM
  brnz       $mSCRATCH, 4f
  zero       $BASE_SLICE
4:
  // The next sub region is just region size bytes afterwards
  add        $SUB_REGION, $SUB_REGION, $REGION_SIZE
  brnzdec    $SUB_ELEM, .Lsub_loop2

  exitz      $mzero
.size DynamicSlice1d_half, .-DSS_HALF_FUNC

//******************************************************************************
// Subroutine to deal with loading vertex state and calculating the
// number of loop passes which we will carry out.  The number of loop
// passes is a function of:
// REGION_SIZE
// This workers WKR_ID
// Output alignment (to 2*floats or 2*halves)
// Output alignment can change on each pass of the loop it we are copying
// an odd number of items
//******************************************************************************
.section .text.DynamicSlice1d_common
.align 4

load_and_preprocess:
  // Base slice is initially *offset
  ld32      $BASE_SLICE, $mzero, $mvertex_base, VOFF_OFFSET
  ld32      $BASE_SLICE, $mzero, $BASE_SLICE, 0

  ld32      $BASE_ELEM, $mzero, $mvertex_base, VOFF_BASE_ELEM

  // Keep base slice within range of base elem
  cmpult    $mSCRATCH, $BASE_SLICE, $BASE_ELEM
  brnz      $mSCRATCH, 1f
  zero      $BASE_SLICE
1:
  // Load sub elem, decrement for use as a loop counter
  ldz16     $SUB_ELEM, $mzero, $mvertex_base, VOFF_SUB_ELEM
  add       $SUB_ELEM, $SUB_ELEM, -1

  // Fetch worker ID, masking it out: 0..5
  get        $WKR_ID, $WSR
  and        $WKR_ID, $WKR_ID, CSR_W_WSR__CTXTID_M1__MASK

  zero       $SUB_REGION

  // Get region size, use to calculate the elements for this worker for the
  // case where there is no pre loop copy to align
  ldz16     $REGION_SIZE, $mzero, $mvertex_base, VOFF_REGION_SIZE
  call      $mSCRATCH, divide_work
  st32      $ELEMENTS_PER_WORKER, $mzero, $mworker_base, WOFF_ELEMENTS

  // and the case where there is a loop copy to align
  add       $REGION_SIZE, $REGION_SIZE, -1
  call      $mSCRATCH, divide_work
  st32      $ELEMENTS_PER_WORKER, $mzero, $mworker_base, WOFF_ELEMENTS_DEC

  br        $mSCRATCH2

//******************************************************************************
// Macro and constants to divide by 12 (12 is the number of items
// copied in 1 loop pass by all workers combined)
//******************************************************************************
#define RECIPROCAL_3_SHL17 ((((1 << 17) - 1) / 3) + 1)
#define LOG2_24_OVER_3 3
#define LOG2_12_OVER_3 2
.macro SPLIT_BETWEEN_WORKERS n size rem
    setzi \size, RECIPROCAL_3_SHL17
    mul \size, \n, \size
    shr \size, \size, (17 + LOG2_12_OVER_3)
    mul \rem, \size, 12
    sub \rem, \n, \rem
.endm

//******************************************************************************
// Divide work between 6 workers.
// This work division, and the whole copy function is based on each worker
// copying pairs of items and then striding over the 5 pairs of items that
// the other workers will have copied.
//
// So sequential items will be copied by workers with numbers below:
// 00 11 22 33 44 55 00 11 22 33 ...
//
// This means that to copy 14 items, worker 0 will need to loop twice but
// all other workers will only have to loop once.
// Needing to copy a 1st, misaligned item results in the same calculation
// but using REGION_SIZE -1.
// A single last item is dealt with after the loop completes.
//******************************************************************************
divide_work:

  SPLIT_BETWEEN_WORKERS $REGION_SIZE $ELEMENTS_PER_WORKER $mSCRATCH3

  // Use the (remainder/2) - worker id to check if we need to copy an
  // extra element compared to the (rounded down) division result
  shr       $mSCRATCH3, $mSCRATCH3, 1
  cmpult    $mSCRATCH3, $WKR_ID, $mSCRATCH3
  add       $ELEMENTS_PER_WORKER, $ELEMENTS_PER_WORKER, $mSCRATCH3

  br        $mSCRATCH

.size DynamicSlice1d_common, .-load_and_preprocess
#endif
