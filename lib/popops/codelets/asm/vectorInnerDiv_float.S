// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

// # Overview
//
// Assembly for BroadcastVectorInnerXXX<DIVIDE,float> vertices
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
// 2D 'data' and 'B'.
//
//

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

// Let's create a macro to shorten the name for the entry points, which are
// very long, as required by C++ name mangling.
// TYPE needs to be '', 'InPlace', '2D' or '2DInPlace'.
// OPERATION will be DIVIDE
#define CPP_FUNCNAME(TYPE, OPERATION) __runCodelet_popops__BroadcastVectorInner ## TYPE ## ___popops__expr__BinaryOpType__ ## OPERATION ## _float

#define SIZEOF_FLOAT        4
#define LOG2_SIZEOF_FLOAT   2

// This is the main function that does the actual work.
// It will process a continuous row of 'data_block_count' x 'B_size' elements.
// There are two nested loop: the internal one keeps the 'B' element constant
// and processes all corresponding elements of 'data' (i.e. it has a stride of
// 'B_size' along 'data').
// The outside loop iterates over all elements of 'B'.
//
// It takes the following register arguments:
#define B                m0  // pointer to (first elem of) B
#define B_size           m1  // length of B in elements (floats)
#define data             m2  // pointer to (first elem of) data
#define data_block_count m3  // How many times 'B' fits into 'data'
#define out              m11 // pointer to result (migth be the same as 'data')
//
// ALl parameters are modified. It also uses the following scratch registers:
//
#define outer_stride   m4
#define B_loop_count   m5
#define mscratch       m5

#define current_B      a0
#define current_B0     a0
#define current_B1     a1
#define current_B01    a0:1
#define current_data   a2
#define current_data01 a2:3
#define current_data0  a2
#define current_data1  a3
#define result         a4
#define result01       a4:5
#define result0        a4
#define result1        a5

// Main processing.
FN_SECTION VectorInnerDiv_core_float 8 nop
VectorInnerDiv_core_float:
  // If we have no blocks to do, return.
  brz $data_block_count, .Lreturn

  // We use `rpt` which has a limited loop count, but this is taken care of
  // in the popops host code.

  // Start to calculate the stride for the outer loop. This will be subtracted
  // from 'data' & 'out' ptrs, after the inner loop ends, to get those pointers
  // back to where they started, plus one (or two) float further.
  // This is computed in bytes and the subtraction will be done with 'sub'
  // instead of a dummy load to avoid overreads.
  mul $outer_stride, $data_block_count, $B_size
  shl $outer_stride, $outer_stride, LOG2_SIZEOF_FLOAT

  // One block is done in the prologue + epilogue to the rpt loop
  add $data_block_count, $data_block_count, -1

  // Id B_size is even we can use the fast path which processes two floats per
  // loop instead of one. Note that it is NOT TWICE as fast, becuse the f32div
  // takes up to 3 cycles
  and $mscratch, $B_size, 1
  brz $mscratch, .Lfast_path

  // -------------- Slower loop: B_size is odd ------------------
  add $outer_stride, $outer_stride, -SIZEOF_FLOAT
  add $B_loop_count, $B_size, -1  // Subtract 1 for for brnzdec

// for i = 0..B_size
.L_B_loop:
  // Get the next element of B.
  ld32step $current_B, $mzero, $B+=, 1

  // Load the first value.
  ld32step $current_data, $mzero, $data+=, $B_size

  // Loop through data.
  rpt $data_block_count, (2f - 1f)/8-1
1:
 {ld32step $current_data, $mzero, $data+=, $B_size
  f32div $result, $current_data, $current_B}
 {st32step $result, $mzero, $out+=, $B_size
  fnop}
2:

 {sub $data, $data, $outer_stride  // Move 'data' ptr back to the next element.
  f32div $result, $current_data, $current_B} // Divide last value.
  st32step $result, $mzero, $out+=, $B_size  // Storew last value.

  sub $out, $out, $outer_stride   // Move 'out' ptr back to the next element.

  brnzdec $B_loop_count, .L_B_loop

  br $lr

// -------------- Fast loop: B_size is even ------------------
nop
.Lfast_path:
  add      $outer_stride, $outer_stride, -2*SIZEOF_FLOAT
  shr      $B_size, $B_size, 1   // in units of pairs of floats

  add      $B_loop_count, $B_size, -1 // for brnzdec
  // for i = 0..B_size/2
.Lfast_B_loop:
  // Get the next 2 elements of B.
  ld64step $current_B01, $mzero, $B+=, 1

  // Load the first 2 data values.
  ld64step $current_data01, $mzero, $data+=, $B_size

  // Loop through data.
 {rpt $data_block_count, (2f - 1f)/8-1
  f32div $result0, $current_data0, $current_B0}
1:
 {ld64step $current_data01, $mzero, $data+=, $B_size
  f32div $result1, $current_data1, $current_B1}
 {st64step $result01, $mzero, $out+=, $B_size
  f32div $result0, $current_data0, $current_B0}
2:

 {sub $data, $data, $outer_stride  // Move 'data' ptr back to the next element.
  f32div $result1, $current_data1, $current_B1}  // Divide last pair.
  st64step $result01, $mzero, $out+=, $B_size    // Store last pair.

  sub $out, $out, $outer_stride    // Move 'out' ptr back to the next element.

  // If B_size != 0 decrement it and loop.
  brnzdec $B_loop_count, .Lfast_B_loop


.Lreturn:
  br $lr

FN_SIZE VectorInnerDiv_core_float

// Undefine 'local' registers. Arguments are left defined for the code below.
#undef outer_stride
#undef B_loop_count

#undef current_B
#undef current_B0
#undef current_B1
#undef current_data
#undef current_data01
#undef current_data0
#undef current_data1
#undef result
#undef result01
#undef result0
#undef result1



/////////////////// 1D  MultiVertex ///////////////////////

// Vertex state layout
#define VERTEX_DATA_B_OFFSET 0            // In 32-bits
#define VERTEX_DATA_B_SIZE_OFFSET 1       // In 32-bits
#define VERTEX_DATA_DATA_OFFSET 2         // In 32-bits
#define VERTEX_DATA_BLOCK_COUNT_OFFSET 6  // In 16-bits

// Additional state for non-in place
#define VERTEX_DATA_OUT_OFFSET  4         // In 32-bits



.worker

#define blocks_per_worker m4
#define worker_id m5
#define block_begin m6
#define remaining_blocks m7
#define mscratch0 m8


FN_WORKER_ENTRY_POINT CPP_FUNCNAME(1D,DIVIDE)

  ld32 $data,  $mvertex_base, $mzero, VERTEX_DATA_DATA_OFFSET
  ld32 $out,   $mvertex_base, $mzero, VERTEX_DATA_OUT_OFFSET
  // Jump to the shared worker code
  bri .Lworker

FN_SIZE CPP_FUNCNAME(1D,DIVIDE)

FN_WORKER_ENTRY_POINT CPP_FUNCNAME(1DInPlace,DIVIDE)

  ld32 $data, $mvertex_base, $mzero, VERTEX_DATA_DATA_OFFSET
  mov  $out, $data

.Lworker:
  // Load rest of vertex state.
  ld32 $B,                 $mvertex_base, $mzero, VERTEX_DATA_B_OFFSET
  ld32 $B_size,            $mvertex_base, $mzero, VERTEX_DATA_B_SIZE_OFFSET
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

  // Work out block begin, accounting for remainders.
  mul $block_begin, $blocks_per_worker, $worker_id
  min $mscratch0, $worker_id, $remaining_blocks
  add $block_begin, $block_begin, $mscratch0

  // Add remainder to workers with IDs less than the remainder.
  cmpult $mscratch0, $worker_id, $remaining_blocks
  add $data_block_count, $blocks_per_worker, $mscratch0

  // How many elements to advance $data.
  mul $mscratch0, $block_begin, $B_size
  // Advance $data and $out by 4*$mscratch0 bytes to the $block_begin'th block,
  // using a dummy load.
  ld32step $azero, $mzero, $data+=, $mscratch0
  ld32step $azero, $mzero, $out+=, $mscratch0

  call $lr, VectorInnerDiv_core_float

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
// Additional state for non-inplace & scaled variants
#define VERTEX2D_DATA_OUT_OFFSET 3                  // In 32-bits


#define B_iterator m6
#define workList_iterator m7
#define data_iterator m8
#define nM1 m9
#define out_iterator m4

#define SCRATCH_OFFSET_OUT_ITERATOR   0

FN_WORKER_ENTRY_POINT CPP_FUNCNAME(2D,DIVIDE)

  ld32 $data_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_DATA_OFFSET
  ld32 $out_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_OUT_OFFSET
  // Jump to the shared worker code
  bri .Lworker2d

FN_SIZE CPP_FUNCNAME(2D,DIVIDE)


FN_WORKER_ENTRY_POINT CPP_FUNCNAME(2DInPlace,DIVIDE)

  ld32 $data_iterator, $mvertex_base, $mzero, VERTEX2D_DATA_DATA_OFFSET
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
  st32 $out_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_OUT_ITERATOR

  call $lr, VectorInnerDiv_core_float

  ld32 $out_iterator, $mworker_base, $mzero, SCRATCH_OFFSET_OUT_ITERATOR
  brnzdec $nM1, .Louter_loop

  exitnz $mzero

FN_SIZE CPP_FUNCNAME(2DInPlace,DIVIDE)

#undef B_iterator
#undef workList_iterator
#undef data_iterator
#undef n

#endif // __IPU__
