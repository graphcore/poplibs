// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// # Overview
//
// Assembly for BroadcastVectorInnerXXX<DIVIDE,half> vertices
//
// Given two vectors:
//
// data:   [0 1 2 3 4 5 6 7 8 9 10 11]
// B:      [0 1 2]
//
// Divide 'data' by a repeated 'B'.
// Of course the length of 'B' must be a submultiple of the length of 'data'
//
// data:        [0 1 2  3 4 5  6 7 8  9 10 11]
//               - - -  - - -  - - -  - -- --   <== divide
// repeated B:  [0 1 2][0 1 2][0 1 2][0  1  2]
//
//
// The 2D Vertices repeat the above process for all the sub-vectors of the
// 2D 'data' and 'B'

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

// Let's create a macro to shorten the name for the entry points, which are
// very long, as required by C++ name mangling.
// TYPE needs to be '1D', '1DInPlace', '2D' or '2DInPlace'.
// OPERATION will be DIVIDE
#define CPP_FUNCNAME(TYPE, OPERATION) __runCodelet_popops__BroadcastVectorInner ## TYPE ## ___popops__expr__BinaryOpType__ ## OPERATION ## _half


// This is the main function that does the actual work.
// It will process a contiguous row of 'data_block_count' x 'B_size' elements
// pointed by 'data'.
//
// It takes the following register arguments:
#define B                m0 // pointer to (first elem of) B
#define B_size           m1 // length of B in elements (halves)
#define data             m2 // pointer to (first elem of) data
#define data_block_count m3 // How many times 'B_size' fits into 'data'
#define out              m4 // pointer to result (migth be the same as 'data')


# Some locally used registers. More are defined below
#define mscratch         m5


FN_SECTION VectorInnerDiv_core_half 8 nop

VectorInnerDiv_core_half:
  // If we have no blocks to do, exit.
  brz $data_block_count, .Lreturn

  // If B_size is even we can use the fast path which processes two elements
  // per loop.
  and     $mscratch, $B_size, 1
  brz     $mscratch, .L_even_B_size

  // if B_size is odd and greater than 1, we have a slighlty slower path
  cmpeq   $mscratch, $B_size, 1
  brz     $mscratch, .L_odd_B_size

// ----------------- B_size is 1 ---------------------
// This is a special case that needs to be handle separately, as the other
// two paths require 'B_size' to be at least 2, or 3.
// Note that this is the same as a BroadcastScalar operation. This vertex should
// never be selected for this case, but this path is here for robustness (i.e.
// if it get selected we don't crash)

#define rpt_counter                m1
#define pair_count                 m3
#define remainder                  m5
#define B_start                    m5

#define ascratch                   a0
#define ascratch1                  a1
#define ascratch2                  a2
#define current_B01                a4:5
#define current_B0                 a4
#define current_B1                 a5
#define current_data01             a6:7
#define current_data0              a6
#define current_data1              a7
#define result                     a6

  // If 'data' also is single element, need to jump to the end.
  // Also, load and convert the lone B element which is needed anyway
  cmpeq   $mscratch, $data_block_count, 1
  ldb16   $current_B0, $mzero, $B, 0
 {brnz    $mscratch, .L_last_odd_one
  f16tof32   $current_B0, $current_B0}

  and        $remainder, $data_block_count, 1  // is 'data' size odd?
  ld32step   $ascratch, $mzero, $data+=, 1     // load 1st pair of 'data'
 {shr        $pair_count, $data_block_count, 1 // how many pairs of elems
  f16v2tof32 $current_data01, $ascratch}       // 1st pair of 'data' to f32
 {add        $pair_count, $pair_count, -1      // 1 pair in prologue+epilogue
  f32div     $current_data0, $current_data0, $current_B0}

  // Inner loop over all pairs of elements
 {rpt        $pair_count, (2f - 1f)/8-1
  f32div     $current_data1, $current_data1, $current_B0}
1:
 {ld32step   $ascratch, $mzero, $data+=, 1
  f32v2tof16 $result, $current_data01}
 {st32step   $result, $mzero, $out+=, 1
  f16v2tof32 $current_data01, $ascratch}
 {nop
  f32div     $current_data0, $current_data0, $current_B0}
 {nop
  f32div     $current_data1, $current_data1, $current_B0}
2:
  f32v2tof16 $result, $current_data01    // convert last pair
  st32step   $result, $mzero, $out+=, 1  // Store last pair

  brz         $remainder, .Lreturn
  // We have the very last element to process and store, taking care of
  // leaving the other half word of 'out' as it is (read it and write it back)
.L_last_odd_one:
  ldb16       $current_data0, $mzero, $data, 0
  f16tof32    $current_data0, $current_data0
  f32div      $current_data0, $current_data0, $current_B0
 {ld32        $ascratch, $mzero, $out, 0
  f32tof16    $current_data0, $current_data0}
  sort4x16hi  $ascratch, $current_data0, $ascratch
  st32        $ascratch, $mzero, $out, 0
  br $lr

// -------------------------- B_size is odd ---------------------------
// Here we have an outer loop, containing two back-to-back inner loops that
// will process two 'B' blocks
//
//     data:  x x x x x x x x x x x x x x x x x x x .... x x x
//     B:     a b c d e f g
//                          a b c d e f g
//                          ^
//                          1st element of 2nd 'B' block is NOT 32-bit aligned
//
// Inner Loop 1: ('rpt'); Iterates over 'B', in pairs of element, processing
//             them with the corresponding pairs of 'data'. All arrays ('data',
//             'B' and 'out' are accessed 32-bit aligned.
//             The very last element of 'B' is not processed.
//             Takes 5 instruction bundles for 2 half values (9 cycles, as each
//             f32div takes 3 cycles)
//
// Inner Loop 2: ('rpt'); Iterates again over 'B', in pairs of element.
//             But now 'B' is misalingned (so it's read with 2 'ldb16') while
//             'data' and 'out' are still aligned.
//             The very last element from the previous 'innert loop 1'loop is
//             processed at the start of this.
//             Takes 6 bundles for 2 half values (10 cycles)
//
// Outer Loop: Iterates over all of 'data' (in pairs of 'data blocks').
.align 8
nop
.L_odd_B_size:
  mov       $B_start, $B

  add       $data_block_count, $data_block_count, -1  // for brnzdec

  // 'rpt_counter' is the same for Inner Loop 1 and 2, despite Inner Loop 2
  // processing 1 extra pair of elements. This is because Inner Loop 1 processes
  // 1 pair in the prologue+epilogue, while Inner Loop 1 processes 2 pairs.
  shr       $rpt_counter, $B_size, 1
  add       $rpt_counter, $rpt_counter, -1

// ====== Outer loop: processes two adjacent blocks of B_size elements
.L_odd_B_size_loop:

  // --- Process first block of (B_size - 1) elements, starting with the 'B'
  // --- pointer aligned on a 32-bit word, so that 'data', 'B' and 'out' can
  // --- all be read/written in whole words (i.e. process a pair of elements).
  // --- The very last element of 'B' is not processed
  //
  //  data:  x x x x x x x x x x x x x x x x x x x .... x x x
  //  B:     a b
  //             c d
  //                 e f
  //                    |g| <- Last element not processed

  ld32step   $ascratch, $mzero, $B+=, 1     // load 1st pair of 'B'
 {ld32step   $ascratch, $mzero, $data+=, 1  // load 1st pair of 'data'
  f16v2tof32 $current_B01, $ascratch}       // 1st pair of 'B' to f32
  f16v2tof32 $current_data01, $ascratch     // 1st pair of 'data' to f32
  f32div     $current_data0, $current_data0, $current_B0

  // Inner loop over all pairs of elements
 {rpt        $rpt_counter, (2f - 1f)/8-1
  f32div     $current_data1, $current_data1, $current_B1}
1:
 {ld32step   $ascratch, $mzero, $B+=, 1
  f32v2tof16 $result, $current_data01}
 {ld32step   $ascratch, $mzero, $data+=, 1
  f16v2tof32 $current_B01, $ascratch}
 {st32step   $result, $mzero, $out+=, 1
  f16v2tof32 $current_data01, $ascratch}
 {nop
  f32div     $current_data0, $current_data0, $current_B0}
 {nop
  f32div     $current_data1, $current_data1, $current_B1}
2:
 {ldb16step   $current_B0, $mzero, $B+=, 1   // Load last element of B
  f32v2tof16 $result, $current_data01}       // convert last pair of results
  st32step   $result, $mzero, $out+=, 1      // Store last pair

 {brnzdec     $data_block_count, .L_second_block
  f16tof32    $current_B0, $current_B0}
  // We have finished (data_block_count was originally odd)
  // We have one very last element to process and store, taking care of
  // reading and writing back the other half word of 'out'.
  ldb16       $current_data0, $mzero, $data, 0
  f16tof32    $current_data0, $current_data0
  f32div      $current_data0, $current_data0, $current_B0
 {ld32        $ascratch, $mzero, $out, 0
  f32tof16    $current_data0, $current_data0}
  sort4x16hi  $ascratch, $current_data0, $ascratch
  st32        $ascratch, $mzero, $out, 0
  br $lr

.align 8
.L_second_block:
  // --- Second inner loop: process second block of B_size elements (with added
  // --- the last element from 1st block.
  // --- Both 'data' and 'out' can be read/written in whole words, but 'B'
  // --- is read in half-words (because it is misaligned)
  // --- (This section processes two pairs in the prologue+epilogue)
  //
  // data:  x x x x x x x x x x x x x x x x x x x .... x x x
  // B:                |g|a
  //                    *   b c
  //                    |       d e
  //                    |           f g
  //                    |
  //                   last element of 1st block processed here

  mov         $B, $B_start
  ldb16step   $current_B1, $mzero, $B+=, 1   // Load first element of B
 {ld32step    $ascratch2, $mzero, $data+=, 1
  f16tof32    $current_B1, $current_B1}
  ldb16step   $ascratch, $mzero, $B+=, 1     // load next B for next time ...
 {ldb16step   $ascratch1, $mzero, $B+=, 1    // ... and load B again for next
  f16v2tof32  $current_data01, $ascratch2}
 {ld32step    $ascratch2, $mzero, $data+=, 1
  f32div      $current_data0, $current_data0, $current_B0}
  f32div      $current_data1, $current_data1, $current_B1

 {rpt $rpt_counter, (2f - 1f)/8-1
  f32v2tof16  $result, $current_data01}
1:
 {st32step    $result, $mzero, $out+=, 1
  sort4x16lo  $ascratch, $ascratch, $ascratch1}
 {ldb16step   $ascratch, $mzero, $B+=, 1
  f16v2tof32  $current_B01, $a0}
 {ldb16step   $ascratch1, $mzero, $B+=, 1
  f16v2tof32  $current_data01, $ascratch2}
 {ld32step    $ascratch2, $mzero, $data+=, 1
  f32div      $current_data0, $current_data0, $current_B0}
 {nop
  f32div      $current_data1, $current_data1, $current_B1}
 {nop
  f32v2tof16  $result, $current_data01}
2:
 {st32step    $result, $mzero, $out+=, 1
  sort4x16lo  $ascratch, $ascratch, $ascratch1}
  f16v2tof32  $current_B01, $ascratch
  f16v2tof32  $current_data01, $ascratch2
  f32div      $current_data0, $current_data0, $current_B0
  f32div      $current_data1, $current_data1, $current_B1
 {mov         $B, $B_start           // wrap around to start of B
  f32v2tof16  $result, $current_data01}
  st32step    $result, $mzero, $out+=, 1

  brnzdec     $data_block_count, .L_odd_B_size_loop
  br $lr
#undef B_start
#undef rpt_counter

#undef ascratch
#undef result
#undef current_B01
#undef current_B0
#undef current_B1
#undef current_data01
#undef current_data0
#undef current_data1

// ---------------------- B_size is even --------------------------
// There are two nested loop, with opposite iteration compared to the
// 'B_size odd' case (i.e. inner loop iterates over 'data, outer over 'B'):
//
// Inner Loop: ('rpt'); Iterates over 'data', with stride 'B_size', keeping one
//             pair of elems from B constant and processing all corresponding
//             pairs of 'data'.
//             Takes 4 instruction bundles for 2 half values (but note that
//             this will be up to 8 cycles, as f32div takes 1-3 cycles)
//
// Outer Loop: Iterates over 'B' (in pairs of elements).
#define outer_stride      m5
#define B_loop_count      m6

#define ascratch          a0
#define result            a1
#define current_B01       a2:3
#define current_B0        a2
#define current_B1        a3
#define current_data01    a4:5
#define current_data0     a4
#define current_data1     a5

.align 8
nop
.L_even_B_size:
  // Calculate the (back)stride for the outer loop (in bytes). This is
  // subtracted from 'data' and 'out' at the end of inner loop, to get them
  // back to the next pair to process.
  mul      $outer_stride, $data_block_count, $B_size
  add      $outer_stride, $outer_stride, -2
  shl      $outer_stride, $outer_stride, 1

  // One block is done in the prologue + epilogue to the rpt loop
  add       $data_block_count, $data_block_count, -1

  // for i = 0..B_size/2
  shr      $B_size, $B_size, 1 // B_size = B_size/2 (units of pair of halves)
  add      $B_loop_count, $B_size, -1 // for brnzdec
.Lfast_B_loop:
  // load pair of f16 values from 'B'
  ld32step   $ascratch, $mzero, $B+=, 1

  // load and process first pair of f16 values from 'data'
 {ld32step   $ascratch, $mzero, $data+=, $B_size // load 1st pair of 'data'
  f16v2tof32 $current_B01, $ascratch}            // 'B' pair to f32

  f16v2tof32 $current_data01, $ascratch
  f32div     $current_data0, $current_data0, $current_B0

  // Inner loop over all pairs of elements
 {rpt        $data_block_count, (2f - 1f)/8-1
  f32div     $current_data1, $current_data1, $current_B1}
1:
 {ld32step   $ascratch, $mzero, $data+=, $B_size
  f32v2tof16 $result, $current_data01}

 {st32step   $result, $mzero, $out+=, $B_size
  f16v2tof32 $current_data01, $ascratch}

 {nop
  f32div     $current_data0, $current_data0, $current_B0}

 {nop
  f32div     $current_data1, $current_data1, $current_B1}
2:

 {sub $data, $data, $outer_stride  // Move 'data' ptr back to the next element.
  f32v2tof16 $result, $current_data01}         // convert last pair
  st32step   $result, $mzero, $out+=, $B_size  // Store last pair

  sub $out, $out, $outer_stride    // Move 'out' ptr back to the next element.

  brnzdec $B_loop_count, .Lfast_B_loop
#undef outer_stride
#undef B_loop_count

#undef ascratch
#undef result
#undef current_B01
#undef current_B0
#undef current_B1
#undef current_data01
#undef current_data0
#undef current_data1

.Lreturn:
  br $lr

#undef mscratch

FN_SIZE VectorInnerAdd_core_half




/////////////////// 1D MultiVertex ///////////////////////

// Vertex state layout for VectorInner1D
#define VERTEX_DATA_B_OFFSET 0            // In 32-bits
#define VERTEX_DATA_B_LEN_OFFSET 1        // In 32-bits
#define VERTEX_DATA_DATA_OFFSET 2         // In 32-bits
#define VERTEX_DATA_BLOCK_COUNT_OFFSET 6  // In 16-bits

// Additional state for non-in place, Scaled
#define VERTEX_DATA_OUT_OFFSET  4               // In 32-bits

// Worker code

.worker

#define blocks_per_worker m11
#define worker_id m5
#define block_begin m6
#define remaining_blocks m7
#define mscratch0 m8


FN_WORKER_ENTRY_POINT CPP_FUNCNAME(1D,DIVIDE)
  ld32 $data, $mvertex_base, $mzero, VERTEX_DATA_DATA_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_DATA_OUT_OFFSET
  // Jump to the shared worker code
  bri .Lworker

FN_SIZE CPP_FUNCNAME(1D,DIVIDE)

FN_WORKER_ENTRY_POINT CPP_FUNCNAME(1DInPlace,DIVIDE)
  ld32 $data, $mvertex_base, $mzero, VERTEX_DATA_DATA_OFFSET
  mov $out, $data
  // Fall through.

.Lworker:

  // Load rest of vertex state.
  ld32 $B,                 $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
  ld32 $B_size,            $mvertex_base, $mzero, VERTEX_DATA_B_LEN_OFFSET
  ldz16 $data_block_count, $mvertex_base, $mzero, VERTEX_DATA_BLOCK_COUNT_OFFSET

  // Get the worker ID.
  get $worker_id, $WSR
  and $worker_id, $worker_id, CSR_W_WSR__CTXTID_M1__MASK

  // The division of work among 6 workers has been done when creating the vertex
  // (contrary to other types of vertices that do that in the device code).
  //
  // The amount of work to do is expressed by:
  //        dataBlockCount = data.size() / B.size();
  // i.e. how many times the 'B' vector fits inside 'data'
  // This has been divided by 6; the quotient and remainder of this division
  // has been packed into 'dataBlockCountPacked'
  //
  //                         15 14 13 12 11 10            4  3  2  1  0
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  // dataBlockCountPacked:  |           13 bits               | 3 bits |
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  //
  //                        |                                 |        |
  //                        +---------------+-----------------+----+---+
  //                                        |                      |
  //                            floor(dataBlockCount/6)    dataBlockCount % 6
  //                              (blocks_per_worker)      (remaining_blocks)
  //
  // Each worker will process at least 'blocks_per_worker' blocks, but the
  // first few (up to remaining_blocks-1) workers will process 1 extra block.
  //
  // Example:   'data' is 350 elements
  //            'B'    is   5 elements
  //
  //          350/5 = 70 (dataBlockCount)
  //
  //          70/6 => quotient 11, remainder 4
  //
  //   dataBlockCountPacked = (11 << 3) | 4
  //
  // Division of work among workers:
  //          worker id:      0     1     2     3     4     5
  //  blocks to process:     12     12   12    12    11    11

  // Get blocks per worker and remainder.
  shr $blocks_per_worker, $data_block_count, 3
  and $remaining_blocks, $data_block_count, 0x7

  // Work out block begin, accounting for remainders
  mul $block_begin, $blocks_per_worker, $worker_id
  min $mscratch0, $worker_id, $remaining_blocks
  add $block_begin, $block_begin, $mscratch0

  // Add an extra block to workers with IDs less than the remainder.
  cmpult $mscratch0, $worker_id, $remaining_blocks
  add $data_block_count, $blocks_per_worker, $mscratch0

  // All workers except the last one must process an even number of elements to
  // avoid the issue of subword writes done by separate worker threads, so we
  // migth need to adjust 'blocks_per_worker'

  // If B_size is even, we have an even number, so no need to adjust
  and $mscratch0, $B_size, 0x1
  brz $mscratch0, update_data_ptrs

  // If block_begin is odd, round it down and increment data_block_count.
  and $mscratch0, $block_begin, 1
  add $data_block_count, $data_block_count, $mscratch0
  andc $block_begin, $block_begin, 1
  // If we aren't the last worker with blocks, round $data_block_count down to
  // an even number. The last worker with blocks is 5 if $blocks_per_worker is
  // not 0, or $remaining_blocks otherwise.
  brz $blocks_per_worker, 1f
  setzi $remaining_blocks, 5
1:
  // $mscratch0 is the id of the last worker with blocks.
  cmpeq $mscratch0, $worker_id, $remaining_blocks
  // Don't alter data_block_count if we are the last worker.
  brnz $mscratch0, 1f
  // Round data_block_count down to the next even number.
  andc $data_block_count, $data_block_count, 1
1:

update_data_ptrs:
  // How many elements to advance $data.
  mul $mscratch0, $block_begin, $B_size
  // Advance $data by 2*$mscratch0 bytes to the $block_begin'th block using
  // a dummy load.
  ldb16step $azero, $mzero, $data+=, $mscratch0
  ldb16step $azero, $mzero, $out+=, $mscratch0

  call $lr, VectorInnerDiv_core_half

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(1DInPlace,DIVIDE)

#undef blocks_per_worker
#undef worker_id
#undef block_begin
#undef remaining_blocks
#undef mscratch0



///////////////////// VectorInner2D Worker Vertices ////////////////////////

// Vertex state layout for BroadcastVectorInner2D
#define VERTEX2D_DATA_B_OFFSET 0                    // In 32-bits
#define VERTEX2D_DATA_WORKLIST_OFFSET 1             // In 32-bits
#define VERTEX2D_DATA_DATA_OFFSET 2                 // In 32-bits
// Additional state for non-inplace variant
#define VERTEX2D_DATA_OUT_OFFSET 3                  // In 32-bits


#define B_iterator                m5
#define workList_iterator         m8
#define data_iterator             m7
#define nM1                       m9
#define out_iterator              m11

#define SCRATCH_OFFSET_B_ITERATOR               0

FN_WORKER_ENTRY_POINT CPP_FUNCNAME(2D,DIVIDE)
  ld32 $data_iterator,  $mvertex_base, $mzero, VERTEX2D_DATA_DATA_OFFSET
  ld32 $out_iterator,   $mvertex_base, $mzero, VERTEX2D_DATA_OUT_OFFSET
  // Jump to the shared worker code + set the scale to 1.0.
  bri .Lworker2d

FN_SIZE CPP_FUNCNAME(2D,DIVIDE)


FN_WORKER_ENTRY_POINT CPP_FUNCNAME(2DInPlace,DIVIDE)
  ld32 $data_iterator,  $mvertex_base, $mzero, VERTEX2D_DATA_DATA_OFFSET
  mov $out_iterator, $data_iterator
  // Fall through.

.Lworker2d:
  ld32 $workList_iterator,         $mvertex_base, $mzero, VERTEX2D_DATA_WORKLIST_OFFSET
  ldz16step $nM1, $mzero, $workList_iterator+=, 1
  ld32 $B_iterator,                $mvertex_base, $mzero, VERTEX2D_DATA_B_OFFSET

.Louter_loop:
  // Advance all the iterators.
  ld32step  $B,                $mzero, $B_iterator+=, 1
  ldz16step $B_size,           $mzero, $workList_iterator+=, 1
  ld32step  $data,             $mzero, $data_iterator+=, 1
  ldz16step $data_block_count, $mzero, $workList_iterator+=, 1
  ld32step  $out,              $mzero, $out_iterator+=, 1

  // We need to save & restore these as they are clobbered by the function call.
  st32   $B_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_B_ITERATOR

  call $lr, VectorInnerDiv_core_half

  ld32   $B_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_B_ITERATOR
  brnzdec $nM1, .Louter_loop

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(2DInPlace,DIVIDE)

#undef B_iterator
#undef B_workList_iterator
#undef data_iterator
#undef nM1

#endif // __IPU__
