// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

/* -------------------------------------------------------------------------- */
// Dynamic Slice and Dynamic Update Slice 1D MultiWorker
// vertex code for 16bit and 32bit variants
/* -------------------------------------------------------------------------- */

// Worker Registers
#define WKR_ID      m0
#define mSCRATCH2   m0
#define SRC         m1  // base of src tensor
#define DST         m2  // base of dst tensor

#define BASE_ELEM   m3  // number of slices in base tensor
#define SUB_ELEM    m4  // current slice index in base tensor
#define WORDS_PER_WORKER m5 //64bit words

#define SRC_OFFS    m6
#define DST_OFFS    m7

#define TEMP_LOG2_HALFS_PER_ATOM m6

#define BASE_SLICE  m8
#define mSCRATCH    m9
#define SUB_REGION  m10 // offset of current subslice, in bytes
#define REGION_SIZE m11 // in bytes after initial processing

#define UPDATE       a7

//****************************************************************************
// The input structure parameters:
// 32 bit offset
// 32 ptr baseT
// 32 ptr subT
// 32 bit numBaseElements
// 32 bit numSubElements
// 32 bit regionSize
//****************************************************************************
#define VOFF_OFFSET      0
#define VOFF_BASET       1
#define VOFF_SUBT        2
#define VOFF_BASE_ELEM   3
#define VOFF_SUB_ELEM    4
#define VOFF_REGION_SIZE 5 // in atoms

//****************************************************************************
// Worker base local variable storage
//****************************************************************************
// Offset of this worker's processing from each slice's start following 64bit
// alignment
#define WOFF_REGION_START64 0 // offset of this 64bit aligned region in bytes
// Number of 64bit words to copy
#define WOFF_REGION_SIZE64 1
// Number of halves in head and tail
#define WOFF_ENDS_ATOMS 2
// True for the last worker - which copies the head and tail
#define WOFF_LAST_WORKER 3
// regions size IN HALVES minus three. When negative separate copy code is used
#define WOFF_REGION_SIZE_M3 4

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "workDivision.h.S"
#include "CommonPoplibsMacros.h.S"

#define DSS_FLOAT_FUNC __runCodelet_popops__DynamicSlice1D___float
#define DUSS_FLOAT_FUNC __runCodelet_popops__DynamicUpdateSlice1D___float

#define DSS_INT_FUNC __runCodelet_popops__DynamicSlice1D___int
#define DUSS_INT_FUNC __runCodelet_popops__DynamicUpdateSlice1D___int

#define DSS_UNSIGNED_FUNC __runCodelet_popops__DynamicSlice1D___unsigned_int
#define DUSS_UNSIGNED_FUNC __runCodelet_popops__DynamicUpdateSlice1D___unsigned_int

#define DSS_HALF_FUNC __runCodelet_popops__DynamicSlice1D___half
#define DUSS_HALF_FUNC __runCodelet_popops__DynamicUpdateSlice1D___half

#define DSS_ULONGLONG_FUNC __runCodelet_popops__DynamicSlice1D___unsigned_long_long
#define DUSS_ULONGLONG_FUNC __runCodelet_popops__DynamicUpdateSlice1D___unsigned_long_long

#define DSS_LONGLONG_FUNC __runCodelet_popops__DynamicSlice1D___long_long
#define DUSS_LONGLONG_FUNC __runCodelet_popops__DynamicUpdateSlice1D___long_long

// ************************************************* //
// Worker thread execution for supervisor vertex (32-bit types)
// Update version: set a flag for update,
// load base, sub array pointers
// Src = Sub
// Dst = Base
// ************************************************* //
FN_WORKER_ENTRY_POINT DUSS_LONGLONG_FUNC
FN_EXPORT DUSS_ULONGLONG_FUNC
  ld32     $DST, $mzero, $mvertex_base, VOFF_BASET
  ld32      $SRC, $mzero, $mvertex_base, VOFF_SUBT
  {bri       .Lds_super_common_64
   setzi    $UPDATE,1}
FN_SIZE DUSS_LONGLONG_FUNC
// ************************************************* //
// Worker thread execution for supervisor vertex (32-bit types)
// Non Update version: clear a flag for no update,
// load base, sub array pointers
// Src = Base
// Dst = Sub
// ************************************************* //
FN_WORKER_ENTRY_POINT DSS_LONGLONG_FUNC
FN_EXPORT DSS_ULONGLONG_FUNC
  ld32      $SRC, $mzero, $mvertex_base, VOFF_BASET
  {ld32      $DST, $mzero, $mvertex_base, VOFF_SUBT
  setzi     $UPDATE,0}

.Lds_super_common_64:
  // Call pre-loop calculation, common between all variants
  ld32      $REGION_SIZE, $mzero, $mvertex_base, VOFF_REGION_SIZE
  shl       $REGION_SIZE, $REGION_SIZE, 2 // need REGION_SIZE in halves
  bri  load_vertex_fields_and_preprocess
FN_SIZE DSS_LONGLONG_FUNC

// ************************************************* //
// Worker thread execution for supervisor vertex (32-bit types)
// Update version: set a flag for update,
// load base, sub array pointers
// Src = Sub
// Dst = Base
// ************************************************* //
FN_WORKER_ENTRY_POINT DUSS_INT_FUNC
FN_EXPORT DUSS_UNSIGNED_FUNC
FN_EXPORT DUSS_FLOAT_FUNC
  ld32     $DST, $mzero, $mvertex_base, VOFF_BASET
  ld32      $SRC, $mzero, $mvertex_base, VOFF_SUBT
  {bri       .Lds_super_common_32
   setzi    $UPDATE,1}
FN_SIZE DUSS_INT_FUNC
// ************************************************* //
// Worker thread execution for supervisor vertex (32-bit types)
// Non Update version: clear a flag for no update,
// load base, sub array pointers
// Src = Base
// Dst = Sub
// ************************************************* //
FN_WORKER_ENTRY_POINT DSS_INT_FUNC
FN_EXPORT DSS_UNSIGNED_FUNC
FN_EXPORT DSS_FLOAT_FUNC
  ld32      $SRC, $mzero, $mvertex_base, VOFF_BASET
  {ld32      $DST, $mzero, $mvertex_base, VOFF_SUBT
  setzi     $UPDATE,0}

.Lds_super_common_32:
  // Call pre-loop calculation, common between all variants
  ld32      $REGION_SIZE, $mzero, $mvertex_base, VOFF_REGION_SIZE
  shl       $REGION_SIZE, $REGION_SIZE, 1 // need REGION_SIZE in halves
  bri  load_vertex_fields_and_preprocess
FN_SIZE DSS_INT_FUNC

// ************************************************* //
// Update version: set a flag for update,
// load base, sub array pointers
// Src = Sub
// Dst = Base
// ************************************************* //
FN_WORKER_ENTRY_POINT DUSS_HALF_FUNC 
  ld32      $DST, $mzero, $mvertex_base, VOFF_BASET
  ld32      $SRC, $mzero, $mvertex_base, VOFF_SUBT
  {bri      .Lds_super_common_16
   setzi    $UPDATE,1}
FN_SIZE DUSS_HALF_FUNC
// ************************************************* //
// Non Update version: clear a flag for update,
// load base, sub array pointers
// Src = Base
// Dst = Sub
// ************************************************* //
FN_WORKER_ENTRY_POINT DSS_HALF_FUNC 8
  ld32      $SRC, $mzero, $mvertex_base, VOFF_BASET
  {ld32     $DST, $mzero, $mvertex_base, VOFF_SUBT
   setzi    $UPDATE,0}

.Lds_super_common_16:
  // Call pre-loop calculation, common between all variants
  ld32      $REGION_SIZE, $mzero, $mvertex_base, VOFF_REGION_SIZE

load_vertex_fields_and_preprocess:
  // At this point $REGION_SIZE must be the size of each slice IN HALVES.

  // The bulk of the copy is performed using 64bit accesses by all workers.
  // The initial unaligned head and final aligned tail are copied by a single
  // worker; head+tail is fixed by the region size, but the split depends on
  // the alignment of each output slice. The will always be some head/tail to
  // ensure that we can handle a misaligned output.
  // The central copy is always 64bit aligned and is split between workers; it
  // will total (REGION_SIZE-3) & ~0x3
  // One worker always does the initial elements to align the start of the
  // region to to 64bits, and the tail.
  // The head+tail total a constant of [3:6) atoms; the head is [0:3) depending
  // on alignment.
  // A separate code path is used when REGION_SIZE<3halfs
  // When fully aligned we could avoid the handling of the initial 3 halves.

  // Fetch worker ID, masking it out: 0..5
  get        $WKR_ID, $WSR
  and        $WKR_ID, $WKR_ID, CSR_W_WSR__CTXTID_M1__MASK

  cmpeq     $mSCRATCH,    $WKR_ID,      CTXT_WORKERS-1
  st32      $mSCRATCH,    $mzero,       $mworker_base, WOFF_LAST_WORKER
  // Up to 3 initial halves are handled separately to give 64bit alignment for
  // the centre of the slice.
  sub       $mSCRATCH, $REGION_SIZE, 3
  // When REGION_SIZE_M3 is negative different code is invoked to simplify the
  // common code path. ("M3" for minus three halves)
  // There will be a similar number of halves in the final 64bit word, so end
  // atoms will be [3:6] for half, [2:4] halves for float (cf floats are
  // processed as halves but with REGION_SIZE doubled)
  st32      $mSCRATCH, $mzero,       $mworker_base, WOFF_REGION_SIZE_M3
  and       $mSCRATCH, $mSCRATCH,    0x3
  add       $mSCRATCH, $mSCRATCH,    3
  st32      $mSCRATCH, $mzero,       $mworker_base, WOFF_ENDS_ATOMS

  // Now distribute the main 64bit copies between workers
  sub       $mSCRATCH, $REGION_SIZE, $mSCRATCH // number of aligned 64bit copies
  shr       $mSCRATCH, $mSCRATCH,    2 // halves -> 64bit words
#define TEMP_64_LOWER SUB_REGION
#define TEMP_64_UPPER BASE_ELEM
  GET_WORKER_RANGE $mSCRATCH, $WKR_ID,  $TEMP_64_LOWER,   $TEMP_64_UPPER
  sub       $mSCRATCH, $TEMP_64_UPPER, $TEMP_64_LOWER
  st32      $mSCRATCH, $mzero,         $mworker_base, WOFF_REGION_SIZE64
  shl       $TEMP_64_LOWER,   $TEMP_64_LOWER,  3 // word64->bytes
  st32      $TEMP_64_LOWER,   $mzero,  $mworker_base, WOFF_REGION_START64

    // Base slice is initially *offset
  ld32      $BASE_SLICE, $mzero,       $mvertex_base, VOFF_OFFSET
  ld32      $BASE_SLICE, $mzero,       $BASE_SLICE, 0

  ld32      $BASE_ELEM,  $mzero,       $mvertex_base, VOFF_BASE_ELEM
  {
    brneg     $BASE_ELEM, .LClearMsb
    fnop
  }
  // exit if starting offset is not within base_slice
  cmpult    $mSCRATCH, $BASE_SLICE, $BASE_ELEM
  brz       $mSCRATCH, .Lexit

.LClearMsb:
  shl       $BASE_ELEM, $BASE_ELEM, 1
  shr       $BASE_ELEM, $BASE_ELEM, 1

    // Load sub elem, decrement for use as a loop counter
  ld32      $SUB_ELEM, $mzero, $mvertex_base, VOFF_SUB_ELEM
  add       $SUB_ELEM, $SUB_ELEM, -1

  shl       $REGION_SIZE, $REGION_SIZE, 1 // convert from halves to bytes
  zero      $SUB_REGION
  bri .Lslice_loop_entry

.Lslice_loop:
  // The next sub region is just region size bytes afterwards
  add        $SUB_REGION, $SUB_REGION, $REGION_SIZE
  // Increment base slice
  add        $BASE_SLICE, $BASE_SLICE, 1

.Lslice_loop_entry:
  // Wrap base slice to start if required
  cmpult     $mSCRATCH,   $BASE_SLICE, $BASE_ELEM
  mul        $BASE_SLICE, $BASE_SLICE, $mSCRATCH

   // Calculate temporary incrementing offsets for the copy
  mov   $DST_OFFS, $SUB_REGION
  mul   $SRC_OFFS, $REGION_SIZE, $BASE_SLICE

  // Update/ non update versions require source/dest offs to be swapped
  atom  $mSCRATCH, $UPDATE
  brz   $mSCRATCH, 1f
  mov   $SRC_OFFS, $SUB_REGION
  mul   $DST_OFFS, $REGION_SIZE, $BASE_SLICE
1:
  ld32  $mSCRATCH, $mzero, $mworker_base, WOFF_REGION_SIZE_M3
  brneg $mSCRATCH, ShortRegionHandler

  ld32      $mSCRATCH2, $mzero, $mworker_base, WOFF_ENDS_ATOMS
  // Is an initial copy needed to align the output to 64 bits?
  add       $mSCRATCH, $DST_OFFS, $DST
  and       $mSCRATCH, $mSCRATCH, 0x6
  brz       $mSCRATCH, .L_out_aligned_64_core
  // All workers perform most of this head calculation, but only one does the
  // store. This ensures all have incremented their pointers correctly, and the
  // execution time is limited by the worker handling the head/tail anyway.
  sub       $DST_OFFS, $DST_OFFS, $mSCRATCH // DST+DST_OFFSETS now 64bit aligned
  shr       $mSCRATCH, $mSCRATCH, 1 // 0/2/4/6 -> 0/1/2/3
  // Adjust number of tail words for later; we copy (4-alignment) halves here,
  // so the tail will be correspondingly shorter.
  add       $mSCRATCH2, $mSCRATCH2, $mSCRATCH
  sub       $mSCRATCH2, $mSCRATCH2, 4

  // Misaligned for 64 bit copies.  ALL workers load 1-3x16 bits as all workers
  // need the base source and destination addresses to be incremented, but only
  // one stores the result.
  // Optimised for the worker doing the head/tail as it will be the last to
  // finish. Hence all workers do the loading here which aligns their src to
  // 64bits, but only one worker stores the result.
  // If all workers were to store then there's a race between the partial write
  // at the end of one slice and the partial write at the start of the following
  // slice which could lead to some of those elements not being updated.
  // We choose the last worker as it may have slightly less to copy than others.

  ld64        $a0:1, $DST, $DST_OFFS,   0 // the trailing 3/2/1 halves will be overwritten

  // Calculate the number of halves to read from the src, less 1
  sub $mSCRATCH, 2, $mSCRATCH // 1/2/3->1/0/-1

  // Merge 1, 2 or 3 halves onto the end of this partial word.
  ldb16step   $a2,   $SRC, $SRC_OFFS+=, 1
  {brneg      $mSCRATCH, 1f;               sort4x16lo $a1, $a1, $a2}
  ldb16step   $a3,   $SRC, $SRC_OFFS+=, 1
  {brz        $mSCRATCH, 1f;               sort4x16lo $a1, $a2, $a3}
  {ldb16step  $a4,   $SRC, $SRC_OFFS+=, 1; sort4x16lo $a0, $a0, $a2}
  sort4x16lo $a1,   $a3,  $a4
1:
  // Only one worker can do the store to avoid races corrupting the output.
  ld32      $mSCRATCH, $mzero, $mworker_base, WOFF_LAST_WORKER
  brz       $mSCRATCH, 1f
  // Store the first partial output 64bits.
  st64      $a0:1, $DST, $DST_OFFS, 0
  1:
  add       $DST_OFFS, $DST_OFFS, 8

.L_out_aligned_64_core:
  // Copy the aligned 64bit words for this worker.
  ld32       $mSCRATCH, $mzero, $mworker_base, WOFF_REGION_START64
  add        $SRC_OFFS, $SRC_OFFS, $mSCRATCH
  add        $DST_OFFS, $DST_OFFS, $mSCRATCH
  ld32       $WORDS_PER_WORKER, $mzero, $mworker_base, WOFF_REGION_SIZE64

  // Copy WORDS_PER_WORKER*64bits.
  add       $mSCRATCH, $SRC, $SRC_OFFS
  and       $mSCRATCH, $mSCRATCH, 0x6
  shr       $mSCRATCH, $mSCRATCH, 1
  brnzdec   $mSCRATCH, .L_in_aligned123
.L_in_aligned0:
  {rpt   $WORDS_PER_WORKER, (2f-1f)/8-1; fnop}
1:
  {ld64step $a0:1, $SRC, $SRC_OFFS+=, 1; fnop}
  {st64step $a0:1, $DST, $DST_OFFS+=, 1; fnop}
2:
  // no overread
  bri .Lcontinue2
.L_in_aligned123:
  brnzdec   $mSCRATCH, .L_in_aligned23
.L_in_aligned1:
  ldb16step $a0,   $SRC, $SRC_OFFS+=, 1
  ld32step  $a1,   $SRC, $SRC_OFFS+=, 1

  {ld64step  $a2:3, $SRC, $SRC_OFFS+=, 1; roll16 $a4, $a0, $a1}
  shr $mSCRATCH, $WORDS_PER_WORKER, 1 // 2 words per loop
  rpt   $mSCRATCH, (2f-1f)/8-1
1:
  {ld64step $a0:1, $SRC, $SRC_OFFS+=, 1; roll16 $a5, $a1, $a2}
  {st64step $a4:5, $DST, $DST_OFFS+=, 1; roll16 $a4, $a2, $a3}
  {ld64step $a2:3, $SRC, $SRC_OFFS+=, 1; roll16 $a5, $a3, $a0}
  {st64step $a4:5, $DST, $DST_OFFS+=, 1; roll16 $a4, $a0, $a1}
2:
  // overread by 7*16bits
  sub  $SRC_OFFS, $SRC_OFFS,         2*7
  and  $mSCRATCH, $WORDS_PER_WORKER, 1
  {brz $mSCRATCH, .Lcontinue2; roll16 $a5, $a1, $a2}
  bri .LstoreOdd64

.L_in_aligned23:
  brnzdec   $mSCRATCH, .L_in_aligned3

  ld32step $a1, $SRC, $SRC_OFFS+=, 1
  ld64step $a2:3,      $SRC, $SRC_OFFS+=, 1
  shr      $mSCRATCH,  $WORDS_PER_WORKER, 1 // 2 words per loop
  // need to handle odd 64bits
  rpt  $mSCRATCH, (2f-1f)/8-1
1:
  {ld64step $a0:1, $SRC, $SRC_OFFS+=, 1; roll32 $a4:5, $a0:1, $a2:3}
  {st64step $a4:5, $DST, $DST_OFFS+=, 1; fnop}
  {ld64step $a2:3, $SRC, $SRC_OFFS+=, 1; roll32 $a4:5, $a2:3, $a0:1}
  {st64step $a4:5, $DST, $DST_OFFS+=, 1; fnop}
2:
  // overread by 6*16bits
  sub $SRC_OFFS,   $SRC_OFFS, 2*6

  and $mSCRATCH,   $WORDS_PER_WORKER, 1
  {brz $mSCRATCH, .Lcontinue2; roll32 $a4:5, $a0:1, $a2:3}
  bri .LstoreOdd64
.L_in_aligned3:
  ldb16step $a1,   $SRC, $SRC_OFFS+=, 1
  ld64step  $a2:3, $SRC, $SRC_OFFS+=, 1
  shr       $mSCRATCH, $WORDS_PER_WORKER, 1 // 2 words per loop
  {rpt   $mSCRATCH, (2f-1f)/8-1;         roll16 $a4, $a1, $a2}
1:
  {ld64step $a0:1, $SRC, $SRC_OFFS+=, 1; roll16 $a5, $a2, $a3}
  {st64step $a4:5, $DST, $DST_OFFS+=, 1; roll16 $a4, $a3, $a0}
  {ld64step $a2:3, $SRC, $SRC_OFFS+=, 1; roll16 $a5, $a0, $a1}
  {st64step $a4:5, $DST, $DST_OFFS+=, 1; roll16 $a4, $a1, $a2}
2:
  // overread by 5*16bits
  sub $SRC_OFFS, $SRC_OFFS, 2*5

  and $mSCRATCH, $WORDS_PER_WORKER, 1
  {brz $mSCRATCH, .Lcontinue2;           roll16 $a5, $a2, $a3}
  // fall through to store final 64bits
.LstoreOdd64:
  st64step   $a4:5,      $DST,      $DST_OFFS+=, 1
  add        $SRC_OFFS,  $SRC_OFFS, 2*4
.Lcontinue2:

  // DST+DST_OFFS now 64bit aligned
  // SRC+SRC_OFFS now point at the corresponding input
.Lout_aligned_64_handle_tail:
  // Now copy any tail - there will be [0:6] halves to copy
  brz        $mSCRATCH2, .Lend_inner
  // Only the last worker copies the tail bytes; it is already pointing at the
  // correct src and dst positions
  ld32       $mSCRATCH,  $mzero, $mworker_base, WOFF_LAST_WORKER
  brz        $mSCRATCH,  .Lend_inner

  // SRC+SRC_OFFS may be misaligned so always load single halves
  shr $mSCRATCH, $mSCRATCH2, 1
  ldb16step  $a0, $SRC, $SRC_OFFS+=, 1
  rpt $mSCRATCH, (2f-1f)/8-1 // up to 3*aligned words
1:
  {ldb16step $a1, $SRC, $SRC_OFFS+=, 1; fnop}
  {ldb16step $a0, $SRC, $SRC_OFFS+=, 1; roll16 $a2, $a0, $a1}
  {st32step  $a2, $DST, $DST_OFFS+=, 1; fnop}
2:
  // Handle final half. It will be stored in the first 16bits of the output
  // word.
  // When src, dst and region_size are aligned there will be a tail of 4 halves
  // so this code is optimised for the branch being taken.
  and        $mSCRATCH, $mSCRATCH2, 0x1
  brz        $mSCRATCH,             1f;
  ld32       $a2,       $DST,       $DST_OFFS, 0
  sort4x16hi $a2,       $a0,        $a2
  st32       $a2,       $DST,       $DST_OFFS,  0
1:
  // The last worker has now copied the final halves

.Lend_inner:
  brnzdec    $SUB_ELEM,   .Lslice_loop

.Lexit:
  exitz      $mzero

// Dedicated code for small regions (up to 3 halves) to avoid slowing the main
// path
ShortRegionHandler:
  // There willbe lots of subword writes so run a single worker
  ld32 $mSCRATCH, $mzero, $mworker_base, WOFF_LAST_WORKER
  brz $mSCRATCH, .Lexit

  shr $mSCRATCH2, $REGION_SIZE, 1 // need halves not bytes
  add $mSCRATCH,  $DST, $DST_OFFS
  and $mSCRATCH,  $mSCRATCH, 0x2
  brz $mSCRATCH,  1f
  // Copy the initial misaligned half
  sub            $DST_OFFS,    $DST_OFFS,    2
  ldb16          $a2,          $DST,         $DST_OFFS, 0
  ldb16step      $a0,          $SRC,         $SRC_OFFS+=, 1
  {add           $mSCRATCH2,   $mSCRATCH2,   -1
   sort4x16lo    $a2,          $a2,          $a0}
  st32step       $a2,          $DST,         $DST_OFFS+=, 1

1:
  // DST+DST_OFFS now 32bit aligned, only ever 16bit aligned for SRC+SRC_OFFS
  shr $mSCRATCH, $mSCRATCH2, 1
  // $mSCRATCH will be 0 or 1
  brz $mSCRATCH, 2f
  ldb16step $a0, $SRC, $SRC_OFFS+=, 1
  ldb16step $a1, $SRC, $SRC_OFFS+=, 1
  roll16 $a2, $a0, $a1
  st32step $a2, $DST, $DST_OFFS+=, 1
2:
  and $mSCRATCH2, $mSCRATCH2, 0x1
  brz $mSCRATCH2, 1f
  // final odd
  ldb16step $a0, $SRC, $SRC_OFFS+=, 1
  ldb16 $a2, $DST, $DST_OFFS, 1
  sort4x16lo $a2, $a0, $a2
  st32 $a2, $DST, $DST_OFFS, 0
1:
bri .Lend_inner
FN_SIZE DSS_HALF_FUNC

#endif
