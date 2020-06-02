// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// Computes a 1x4 convolution using SLIC. A contiguous field is
// partitioned between workers for each position of each 1x4
// sub-kernel.
//
#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define CODELET_SYMBOL_ENTRY(partials_type, stride, conv_units) \
        __runCodelet_poplin__ConvPartial1x4SLIC___half_ ## partials_type ##_## stride ## _true_ ## conv_units

#define CODELET_SYMBOL(suffix) __runCodelet_poplin__ConvPartial1x4SLIC___half_ ## suffix

//=============
#define SECOND_WEIGHT_BANK 1

#define NUM_WEIGHTS_PER_WORKER_LOOP 64
#define INPUT_ELEM_SIZE 2
#define WEIGHTS_ELEM_SIZE 2

#define HALF_PARTIAL_ELEM_SIZE 2

#define BYTES_PER_DELTAN 4
#define WORKLIST_ALIGN_LOG2 1
#define WORKLIST_DELTAN_OFFSET_BITS (21 - WORKLIST_ALIGN_LOG2)
//=============

//=============
// The vertex is provided with a buffer containing weights, a pointer and then
// space to hold a copy of the output.  The copy of the output is used to store
// half of the output, the actual output buffer is used to store the other half.
// The two are combined at the end of the process when the last sub-kernel is run.
// This allows us to use the ld2xst64pace instruction in the inner loop
// making it single cycle (although only processing 2 halves not 4).  The outputs are:
// Out    = c0, c1, x, x, c0, c1, x, x .....
// Buffer = c2, c3, x, x, c2, c3, x, x .....
// Combined at the end to produce:
// Out    = c0, c1, c2, c3, c0, c1, c2,c3 ......
//
// Note that due to the fact that c0,c1 require one set of weights and c2,c3 require a
// different set of weights we can't ever usefully process and write any contiguous 64 bit
// piece of memory within the same loop.  That makes the mechanism described above necessary.
//
// Unlike other variants of this vertex, buffer alignment (128bit/ not 128bit aligned) is not
// relevant as we always read/write the same buffer in any loop.  (Of course it is 64 bit aligned)
#define BUFFER_OFFSET 200

//=============

//=============
#define WORKITEM_OFFSET_out_offset 0
#define WORKITEM_OFFSET_num_field_elems 2
#define WORKITEM_OFFSET_in_offset 4
//=============

//=============
#define SUPERVISOR_STATE_OFFSET_in 0
#define SUPERVISOR_STATE_OFFSET_weights 4
#define SUPERVISOR_STATE_OFFSET_out 8
#define SUPERVISOR_STATE_OFFSET_outFieldBuffer 12
#define SUPERVISOR_STATE_OFFSET_worklists 16
#define SUPERVISOR_STATE_OFFSET_mode 24
#define SUPERVISOR_STATE_OFFSET_outPtrLoadOffset 25
#define SUPERVISOR_STATE_OFFSET_numSubKernelsM1 26
#define SUPERVISOR_STATE_OFFSET_numConvGroupGroupsM1 28
//=============

//=============
// NOTE: It's very important that the base offset from the stack pointer
// for the weight loading routines is 0. This is assumed in those routines.
#define LOAD_WEIGHTS_WORKER_STATE_OVERREAD (2 * 16 * WEIGHTS_ELEM_SIZE)
#define LOAD_WEIGHTS_WORKER_STATE_STORAGE_SIZE (NUM_WEIGHTS_PER_WORKER_LOOP * WEIGHTS_ELEM_SIZE + LOAD_WEIGHTS_WORKER_STATE_OVERREAD)
#define LOAD_WEIGHTS_WORKER_STATE_OFFSET_storage_mem 0
#define LOAD_WEIGHTS_WORKER_STATE_OFFSET_weights_ptr LOAD_WEIGHTS_WORKER_STATE_STORAGE_SIZE
#define LOAD_WEIGHTS_WORKER_STATE_SIZE (8 + LOAD_WEIGHTS_WORKER_STATE_STORAGE_SIZE)
//=============

//=============
#define PROCESS_GROUP_WORKER_STATE_BASE_OFFSET (0)
#define PROCESS_GROUP_WORKER_STATE_OFFSET_in_ptr (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + 0)
// Tell the worker how many sub-kernels are remaining so that on the last one it can combine the buffer
// and actual outputs together
#define PROCESS_GROUP_WORKER_STATE_OFFSET_sub_kernels_remaining (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + 4)
#define PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + 8)
// out_ptrs storage provides space for 3 output pointers,
// The first points to a temporary buffer
// The second to the actual output
// The third is unused (leaving the worker state with a common structure to the other versions of this vertex)
#define PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + 12)
#define PROCESS_GROUP_WORKER_STATE_OFFSET_worklist (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + 24)
#define PROCESS_GROUP_WORKER_STATE_SIZE (32)
//=============

//=============
#define SUPERVISOR_STACK_BASE_OFFSET (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + PROCESS_GROUP_WORKER_STATE_SIZE)
#define SUPERVISOR_STACK_OFFSET_m9 (SUPERVISOR_STACK_BASE_OFFSET + 0)
#define SUPERVISOR_STACK_OFFSET_m10 (SUPERVISOR_STACK_BASE_OFFSET + 4)
#define SUPERVISOR_STACK_OFFSET_worklists_deltan_ptr (SUPERVISOR_STACK_BASE_OFFSET + 8)
#define SUPERVISOR_STACK_OFFSET_num_sub_kernels_m1 (SUPERVISOR_STACK_BASE_OFFSET + 12)
// This is unused, but kept, in common with stack allocation for other similar vertices
#define SUPERVISOR_STACK_OFFSET_swap_out_ptrs_on_load (SUPERVISOR_STACK_BASE_OFFSET + 16)
#define SUPERVISOR_STACK_OFFSET_worker_fn_ptr (SUPERVISOR_STACK_BASE_OFFSET + 20)
#define SUPERVISOR_STACK_SIZE (24) // (aligns to 8 byte boundary)
//=============

//=============
#define MAX_STACK_SIZE (SUPERVISOR_STACK_BASE_OFFSET + SUPERVISOR_STACK_SIZE)
//=============

//=============
#define msupervisor_vertex_base m0
#define s_worklists_base_ptr m1
#define s_mode m4
#define s_weights_ptr_iterator m5
#define s_worker_function m6
#define s_worklists_deltan_ptr m7
#define s_out_field_buffer_ptr m8
//=============

////////////////////////////////////////////////////////////////////////////////
// Supervisor function, entry point macro definition
.macro supervisor_fn PARTIALS_TYPE CONV_UNITS

// Supervisor uses MAX_STACK_SIZE, workers  don't use any stack
DEF_STACK_USAGE  MAX_STACK_SIZE   CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_UNITS)
DEF_STACK_USAGE  MAX_STACK_SIZE   CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_UNITS)

.global CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_UNITS)
.type CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_UNITS), @function

.global CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_UNITS)
.type CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_UNITS), @function

.section .text.CODELET_SYMBOL(\PARTIALS_TYPE\()_entry_stride2_\CONV_UNITS), "ax"
.align 4
.supervisor

CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_UNITS):
  // The only differences in codelet execution come in the worker function body,
  // much much later on.  So put a pointer to the required function onto the stack
  // to pick up later.  This will mean that we will link just the functions(s)
  // required whereas using a flag would reference both labels
//=============
#define s_worker_fn_ptr m3
//=============
   setzi  $s_worker_fn_ptr, CODELET_SYMBOL(\PARTIALS_TYPE\()_stride2_\CONV_UNITS)
   bri CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_UNITS\()_common_entry)

.section .text.CODELET_SYMBOL(\PARTIALS_TYPE\()_entry_stride1_\CONV_UNITS), "ax"
.align 4
.supervisor

CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_UNITS):

  setzi  $s_worker_fn_ptr, CODELET_SYMBOL(\PARTIALS_TYPE\()_stride1_\CONV_UNITS)
  bri CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_UNITS\()_common_entry)

.section .text.CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_UNITS), "ax"
.align 4
.supervisor
CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_UNITS\()_common_entry):
  // Calculation of $s_worklists_base_ptr is essentially the critical path as
  // it requires 4 instructions to load, mask and store it for a total of 19 cycles minimum
  // (6 cycles * 3 instructions for load, shl, shr + 1 cycle to issue the store)
  ld32 $s_worklists_base_ptr, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_worklists/4
  ld32 $s_worklists_deltan_ptr, $msupervisor_vertex_base, $mzero, (SUPERVISOR_STATE_OFFSET_worklists + 4)/4
  // Setup space on the stack to store m9/m10 and worker state.
  add $sp, $sp, -MAX_STACK_SIZE

  // The load weights jump table (and routines pointed to) is shared with the half_half_16 and half_float_8 vertices
  setzi $s_worker_function, CODELET_SYMBOL(worker_load_weights_jump_table)
  ldz8 $s_mode, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_mode/1
  ld32 $s_out_field_buffer_ptr, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_outFieldBuffer/4
  ld32 $s_weights_ptr_iterator, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_weights/4
  shl $s_worklists_base_ptr, $s_worklists_base_ptr, 8
  shl $s_worklists_deltan_ptr, $s_worklists_deltan_ptr, 8
  // Store m9 and m10 (lr) away on stack as we want to use these registers.
  st32 $m10, $sp, $mzero, SUPERVISOR_STACK_OFFSET_m10/4
  st32 $s_worker_fn_ptr, $sp, $mzero, SUPERVISOR_STACK_OFFSET_worker_fn_ptr/4
//=============
#undef s_worker_fn_ptr
//=============
//=============
#define s_out_ptr_iterator m3
//=============
  ld32 $s_out_ptr_iterator, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_out/4
  ld32 $s_worker_function, $s_worker_function, $mzero, $s_mode
//=============
#undef s_mode
//=============
#define s_in_ptr_iterator m4
//=============
  add $s_out_field_buffer_ptr, $s_out_field_buffer_ptr, BUFFER_OFFSET
  shr $s_worklists_base_ptr, $s_worklists_base_ptr, 8
  shr $s_worklists_deltan_ptr, $s_worklists_deltan_ptr, 8
  st32 $m9, $sp, $mzero, SUPERVISOR_STACK_OFFSET_m9/4
//=============
#define s_num_sub_kernels_m1 m10
#define s_num_conv_group_groups_m1 m2
//=============
  ldz16 $s_num_sub_kernels_m1, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_numSubKernelsM1/2
  ldz16 $s_num_conv_group_groups_m1, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_numConvGroupGroupsM1/2
  ld32 $s_in_ptr_iterator, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_in/4
  st32 $s_out_field_buffer_ptr, $sp, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs + 0)/4
  st32 $s_worklists_base_ptr, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_worklist/4
  st32 $s_worklists_deltan_ptr, $sp, $mzero, SUPERVISOR_STACK_OFFSET_worklists_deltan_ptr/4
  st32 $s_num_sub_kernels_m1, $sp, $mzero, SUPERVISOR_STACK_OFFSET_num_sub_kernels_m1/4
//=============
#undef s_num_sub_kernels_m1
#undef s_out_field_buffer_ptr
#undef s_worklists_base_ptr
#undef s_worklists_deltan_ptr
//=============
#define s_implicit_zero m7
#define s_in_ptr m8
#define s_out_ptr m9
//=============
#define s_load_weights_stack_ptr m0
//=============
  ld32 $s_load_weights_stack_ptr, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_outFieldBuffer/4
//=============
#undef msupervisor_vertex_base // Done with the vertex state from here
//=============
  // Process all sub-kernels and groups
.LConvGroupGroupsLoop\@:
    // Top bit is used to indicate whether or not to take implicit zeroing path
    ld32step $s_in_ptr, $mzero, $s_in_ptr_iterator+=, 1
    ld32step $s_out_ptr, $mzero, $s_out_ptr_iterator+=, 1
    or $s_implicit_zero, $mzero, (1 << 31)
//=============
#define s_num_sub_kernels_m1 m1
#define s_worker_function2 m10
//=============
    ld32 $s_worker_function2, $sp, $mzero, SUPERVISOR_STACK_OFFSET_worker_fn_ptr/4
    ld32 $s_num_sub_kernels_m1, $sp, $mzero, SUPERVISOR_STACK_OFFSET_num_sub_kernels_m1/4
    nop
    st32 $s_in_ptr, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_in_ptr/4
    st32 $s_out_ptr, $sp, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs + 4)/4
//=============
#undef s_in_ptr
#undef s_out_ptr
//=============
#define s_weights_ptr m8
#define s_worklists_deltan_ptr m9
//=============
    ld32step $s_weights_ptr, $mzero, $s_weights_ptr_iterator+=, 1
    ld32 $s_worklists_deltan_ptr, $sp, $mzero, SUPERVISOR_STACK_OFFSET_worklists_deltan_ptr/4
    st32 $s_implicit_zero, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero/4
//=============
#undef s_implicit_zero
//=============
.LSubKernelLoop\@:
      // If we don't need to do any special rearrangement of the weights, the
      // weight loading function will be null and we will directly load the
      // weights instead.
      brz $s_worker_function, .LSetupDirectWeightLoad\@

      // Prepare ldput pointer
      put $CCCSLOAD, $s_load_weights_stack_ptr

      st32 $s_weights_ptr, $s_load_weights_stack_ptr, $mzero, LOAD_WEIGHTS_WORKER_STATE_OFFSET_weights_ptr/4
      runall $s_worker_function, $s_load_weights_stack_ptr, 0
      sync TEXCH_SYNCZONE_LOCAL

.LSubKernelLoopLoadWeightsToCWEI\@:
//=============
#undef s_weights_ptr
//=============
      ld64putcs (6 * 4)
      ld64putcs (7 * 4)
      ld64putcs (6 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (7 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (4 * 4)
      ld64putcs (5 * 4)
      ld64putcs (4 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (5 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (2 * 4)
      ld64putcs (3 * 4)
      ld64putcs (2 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (3 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (0 * 4)
      ld64putcs (1 * 4)
      ld64putcs (0 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (1 * 4 + SECOND_WEIGHT_BANK)
      // Pass the workers the number of remaining sub kernels, used by the worker to trigger a
      // copy to merge the output buffer with the output after using the last sub kernel
      st32 $s_num_sub_kernels_m1,$sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_sub_kernels_remaining/4
      st32 $s_worklists_deltan_ptr, $sp, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_worklist + 4)/4
      add $s_worklists_deltan_ptr, $s_worklists_deltan_ptr, BYTES_PER_DELTAN * CTXT_WORKERS

      runall $s_worker_function2, $sp, 0
      sync TEXCH_SYNCZONE_LOCAL
      st32 $mzero, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero/4

//=============
#define s_weights_ptr m8
//=============
      ld32step $s_weights_ptr, $mzero, $s_weights_ptr_iterator+=, 1
      brnzdec $s_num_sub_kernels_m1, .LSubKernelLoop\@
    // We over-incremented the pointer above because we're going to
    // increment it in the warmup for the next loop over sub-kernels
    // as well.
    add $s_weights_ptr_iterator, $s_weights_ptr_iterator, -4
//=============
#undef s_weights_ptr
//=============
  brnzdec $s_num_conv_group_groups_m1, .LConvGroupGroupsLoop\@
  ld32 $m9, $sp, $mzero, SUPERVISOR_STACK_OFFSET_m9/4
  ld32 $m10, $sp, $mzero, SUPERVISOR_STACK_OFFSET_m10/4
  add $sp, $sp, MAX_STACK_SIZE
  br $lr

// Note this define must match that of where we branched from...
//=============
#define s_weights_ptr m8
//=============
.LSetupDirectWeightLoad\@:
  put $CCCSLOAD, $s_weights_ptr
  bri .LSubKernelLoopLoadWeightsToCWEI\@
//=============
#undef s_weights_ptr
//=============
.endm
////////////////////////////////////////////////////////////////////////////////
// Instantiate supervisor entry
supervisor_fn half 8


////////////////////////////////////////////////////////////////////////////////
//=============
#define w_in_base_ptr m2
#define w_curr_out_base_ptr m3
#define w_buffer_out_base_ptr m4
// Registers above must be preserved between runs for the same conv group group
#define w_inoutout_triptr m0:1
#define w_work_items m5
#define w_worklist_ptr m6
#define w_id m10
#define w_implicit_zero m11

#define w_input_pair a0:1
#define w_partials_pair a2:3
#define w_partials_0 a2
#define w_partials1 a3
#define w_input_and_partials_pairs a0:3
#define w_output_pair a4:5
#define w_output_0 a4
// We need a second register in order to delay
// output stores so that we don't cause a conflict with
// partial loads.
#define w_alt_output_pair a6:7
#define w_alt_output0 a6

////////////////////////////////////////////////////////////////////////////////
// Worker routine to process a single group entry point in macro definition

.macro worker_fn WORK_MACRO_POSTFIX PARTIALS_TYPE
.worker
.align 8
  .equ PARTIAL_ELEM_SIZE, HALF_PARTIAL_ELEM_SIZE


.global CODELET_SYMBOL(\PARTIALS_TYPE\()_\WORK_MACRO_POSTFIX)
.type CODELET_SYMBOL(\PARTIALS_TYPE\()_\WORK_MACRO_POSTFIX), @function

CODELET_SYMBOL(\PARTIALS_TYPE\()_\WORK_MACRO_POSTFIX):
  { get $w_id, $WSR
    setzi $a0, (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT) }
  { and $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
    uput $FP_CLR, $a0 }

  ld32 $w_buffer_out_base_ptr, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs/4
  ld32 $w_curr_out_base_ptr, $mvertex_base, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs + 4)/4
  ld32 $w_in_base_ptr, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_in_ptr/4

  ld32 $w_worklist_ptr, $mvertex_base, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_worklist + 4)/4
  // Each worker has a separate delta N entry for this sub-kernel.
  // The supervisor loop surrounding this worker advances to the
  // next kernel position.
  ld32 $w_worklist_ptr, $w_worklist_ptr, $mzero, $w_id

  // Extract number of entries in worklist and worklist pointer
  shr $w_work_items, $w_worklist_ptr, WORKLIST_DELTAN_OFFSET_BITS
  brz $w_work_items, 0f
  shl $w_worklist_ptr, $w_worklist_ptr, (32 - WORKLIST_DELTAN_OFFSET_BITS)
  shr $w_worklist_ptr, $w_worklist_ptr, (32 - WORKLIST_DELTAN_OFFSET_BITS - WORKLIST_ALIGN_LOG2)

//=============
#define w_worklist_base_ptr m7
//=============
  ld32 $w_worklist_base_ptr, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_worklist/4
  add $w_worklist_ptr, $w_worklist_ptr, $w_worklist_base_ptr
//=============
#undef w_worklist_base_ptr
//=============

  // Divide worklist length by 3 (awkward, is there a good use for a 4th entry
  // in the worklist that would justify making them multiples of 4 elements?).
  //
  // The following approximates num_partitions/3 - 1 for values
  // [3:3:2^14-1]. The case of zero is handled above
  mul $w_work_items, $w_work_items, 21845
  shr $w_work_items, $w_work_items, 16

  ld32 $w_implicit_zero, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero/4
  {brpos $w_implicit_zero, .Lworker_process_group_worklist_loop\@
  fnop}
//=============
#undef w_implicit_zero
//=============

// Loop over work items for implicit zero path
.Lworker_process_group_worklist_loop_implicit_zero\@:
//=============
#define w_in_offset m7
//=============
    ldz16step $w_in_offset, $mzero, $w_worklist_ptr+=, 1
    // The offset for both input/output is in terms of field elements, so we
    // must multiply this by:
    // (convGroupsPerGroup * outChansPerGroup * sizeof(type))
    // to get a byte offset.
    mul $w_in_offset, $w_in_offset, 4 * INPUT_ELEM_SIZE
//=============
#define w_in_ptr m7
//=============
    add $w_in_ptr, $w_in_base_ptr, $w_in_offset
//=============
#undef w_in_offset
//=============
#define w_out_offset m8
//=============
    ldz16step $w_out_offset, $mzero, $w_worklist_ptr+=, 1
    mul $w_out_offset, $w_out_offset, 4 * PARTIAL_ELEM_SIZE
//=============
#define w_curr_out_ptr m8
#define w_buffer_out_ptr m9
//=============
    add $w_buffer_out_ptr, $w_buffer_out_base_ptr, $w_out_offset
    add $w_curr_out_ptr, $w_curr_out_base_ptr, $w_out_offset
//=============
#undef w_out_offset
#undef w_id
//=============
#define w_num_field_elems m10
#define w_num_field_elems_const m11
//=============

    ldz16step $w_num_field_elems_const, $mzero, $w_worklist_ptr+=, 1
    // If there's nothing to do, move on to the next work item.
    brz $w_num_field_elems_const, .LLoopEndZero\@

    // Two loops required, accounting for 2 half partials each.  These go in place into the actual output,
    // with each 64 bits padded with 2 don't care values
    worker_process_group_field_row_implicit_zero_\WORK_MACRO_POSTFIX W0_C8 TSLIC_F16V4_1x4_W0 w_curr_out_ptr

    // These go into the temporary buffer, with each 64 bits padded with 2 don't care values
    worker_process_group_field_row_implicit_zero_\WORK_MACRO_POSTFIX W1_C8 TSLIC_F16V4_1x4_W1 w_buffer_out_ptr

    // The 2 outputs remain in their buffers until the case of the last sub kernel - so now we
    // have written the outputs for the last time and combine them, into the real output
//=============
#define w_sub_kernels_remaining m10
//=============
    ld32 $w_sub_kernels_remaining, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_sub_kernels_remaining/4
    brnz $w_sub_kernels_remaining, .LLoopEndZero\@
//=============
#undef w_sub_kernels_remaining
//=============
    // Offset the "write" ptr to write into the 2nd half of each 64 bits of the
    // actual output
    add  $w_curr_out_ptr, $w_curr_out_ptr, 4
    {rpt $w_num_field_elems_const, (2f - 1f)/8 -1;
     fnop}
1:
    {ld32step $w_output_0, $mzero, $w_buffer_out_ptr+=,2
     fnop}
    {st32step $w_output_0, $mzero, $w_curr_out_ptr+=,2
     fnop}
2:
.LLoopEndZero\@:
    brnzdec $w_work_items, .Lworker_process_group_worklist_loop_implicit_zero\@
    exitz $mzero
//=============
#undef w_num_field_elems
#undef w_num_field_elems_const
#undef w_curr_out_ptr
#undef w_buffer_out_ptr
#undef w_in_ptr
//=============

// Loop over work items - no implicit zeros
.align 8
 nop
.Lworker_process_group_worklist_loop\@:
//=============
#define w_in_offset m7
//=============
    ldz16step $w_in_offset, $mzero, $w_worklist_ptr+=, 1
    // The offset for both input/output is in terms of field elements, so we
    // must multiply this by:
    // (convGroupsPerGroup * outChansPerGroup * sizeof(type))
    // to get a byte offset.
    mul $w_in_offset, $w_in_offset, 4 * INPUT_ELEM_SIZE
//=============
#define w_in_ptr m7
//=============
    add $w_in_ptr, $w_in_base_ptr, $w_in_offset
//=============
#undef w_in_offset
//=============
#define w_out_offset m8
//=============
    ldz16step $w_out_offset, $mzero, $w_worklist_ptr+=, 1
    mul $w_out_offset, $w_out_offset, 4 * PARTIAL_ELEM_SIZE
//=============
#define w_curr_out_ptr m8
#define w_buffer_out_ptr m9
//=============
    add $w_buffer_out_ptr, $w_buffer_out_base_ptr, $w_out_offset
    add $w_curr_out_ptr, $w_curr_out_base_ptr, $w_out_offset
//=============
#undef w_out_offset
#undef w_id
//=============
#define w_num_field_elems m10
#define w_num_field_elems_const m11
//=============

    ldz16step $w_num_field_elems_const, $mzero, $w_worklist_ptr+=, 1
    // If there's nothing to do, move on to the next work item.
    brz $w_num_field_elems_const, .LLoopEnd\@

    // Two loops required, accounting for 2 half partials each.  These go in place into the actual output,
    // with each 64 bits padded with 2 don't care values
    worker_process_group_field_row_\WORK_MACRO_POSTFIX W0_C8 TSLIC_F16V4_1x4_W0 w_curr_out_ptr

    // These go into the temporary buffer, with each 64 bits padded with 2 don't care values
    worker_process_group_field_row_\WORK_MACRO_POSTFIX W1_C8 TSLIC_F16V4_1x4_W1 w_buffer_out_ptr

    // The 2 outputs remain in their buffers until the case of the last sub kernel - so now we
    // have written the outputs for the last time and combine them, into the real output
//=============
#define w_sub_kernels_remaining m10
//=============
    ld32 $w_sub_kernels_remaining, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_sub_kernels_remaining/4
    brnz $w_sub_kernels_remaining, .LLoopEnd\@
//=============
#undef w_sub_kernels_remaining
//=============
    // Offset the "write" ptr to write into the 2nd half of each 64 bits of the
    // actual output
    add  $w_curr_out_ptr, $w_curr_out_ptr, 4
    rpt $w_num_field_elems_const, (2f - 1f)/8 -1
1:
    {ld32step $w_output_0, $mzero, $w_buffer_out_ptr+=,2
     fnop}
    {st32step $w_output_0, $mzero, $w_curr_out_ptr+=,2
     fnop}
2:
.LLoopEnd\@:
    brnzdec $w_work_items, .Lworker_process_group_worklist_loop\@
0:
  exitz $mzero
  .align 8 // Maintain alignmnet for the next macro and its repeat loop
.endm

////////////////////////////////////////////////////////////////////////////////
// Macro definitions to process data using SLIC instrutions
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// >= 5 items, output stride = 1 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride1_8 ID SLIC_FLAGS PAR_OUT_PTR
    tapack $w_inoutout_triptr, $w_in_ptr, $\PAR_OUT_PTR, $\PAR_OUT_PTR

    add $w_num_field_elems, $w_num_field_elems_const, -5
    brneg $w_num_field_elems, .Lworker_process_group_field_row_implicit_zero_stride1_lt5_elems_\ID
    // The 1st load has no stride to ensure the load of (unused) partials,
    // and the output store don't clash in the loop
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1000
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    rpt $w_num_field_elems, (2f - 1f) / 8 - 1;
1:
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b000000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
2:
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
.Lworker_process_group_field_row_implicit_zero_write_and_end_\ID:
    st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b11

.endm // worker_process_group_field_row_implicit_zero

////////////////////////////////////////////////////////////////////////////////
// < 5 items, output stride = 1 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride1_lt5_elems ID SLIC_FLAGS
.Lworker_process_group_field_row_implicit_zero_stride1_lt5_elems_\ID:
    // + 5 (back to original num field elems) - 2
    add $w_num_field_elems, $w_num_field_elems, (5 - 2)
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    // Handle 1 element separately
    brneg $w_num_field_elems, 3f

    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    rpt $w_num_field_elems, (2f - 1f) / 8 - 1
1:
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
2:
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
    bri .Lworker_process_group_field_row_implicit_zero_write_and_end_\ID

3:
    { bri .Lworker_process_group_field_row_implicit_zero_write_and_end_\ID
      f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
.align 8 // Maintain alignment for the next macro and its repeat loop
.endm // worker_process_group_field_row_implicit_zero_lt5_elems


////////////////////////////////////////////////////////////////////////////////
// >= 6 items, output stride = 1 with no implicit zero
.macro worker_process_group_field_row_stride1_8 ID SLIC_FLAGS PAR_OUT_PTR
    tapack $w_inoutout_triptr, $w_in_ptr, $\PAR_OUT_PTR, $\PAR_OUT_PTR

    add $w_num_field_elems, $w_num_field_elems_const, -6
    brneg $w_num_field_elems, .Lworker_process_group_field_row_stride1_lt6_elems_\ID
    shr $w_num_field_elems, $w_num_field_elems, 1

    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    // Now w_output_0 (in w_output_pair) has the 1st valid output, but we need to force 1 more cycle delay
    // to ensure that there is no memory element clash when storing out, reading partials.  Use a second output
    // register to do this, requiring 2 different instructions in the inner loop.

    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_alt_output0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }

    rpt $w_num_field_elems, (2f - 1f) / 8 - 1
1:
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b000000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2xst64pace $w_input_and_partials_pairs, $w_alt_output_pair, $w_inoutout_triptr+=, $mzero, 0b000000
      f16v4hihoslic $w_alt_output0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
2:
    // Check if there were an odd number of items to process (loop always processes an even number)
    and $w_num_field_elems, $w_num_field_elems_const,1
    brz $w_num_field_elems, 3f
    // Path for 1 extra item left, process it, then store alt_output_pair, output_pair, alt_output_pair, output_pair.....
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b000000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }

    { ldst64pace $w_input_pair, $w_alt_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_alt_output0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_alt_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_alt_output0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
    st64pace $w_alt_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
    bri .Lworker_process_group_field_row_write_and_end_\ID

3:
    // Path for nothing left but storing the pipelined SLIC result: store output_pair, alt_output_pair, output_pair.....
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_alt_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_alt_output0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_alt_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_alt_output0, $azeros, $azero, \SLIC_FLAGS }
    st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
    st64pace $w_alt_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
    bri 4f

.Lworker_process_group_field_row_write_and_end_\ID:
    // Careful - shared instruction
    st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b11
4:
.endm // worker_process_group_field_row

////////////////////////////////////////////////////////////////////////////////
// < 6 items, output stride = 1 with no implicit zero
.macro worker_process_group_field_row_stride1_lt6_elems ID SLIC_FLAGS
.Lworker_process_group_field_row_stride1_lt6_elems_\ID:
    // + 6 (back to original num field elems) - 1
    add $w_num_field_elems, $w_num_field_elems, (6 - 1)
    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    // Handle remainder of 1 field element case separately
    brz $w_num_field_elems, 3f
    // 1st loop - inputs and partials used.  Max 4 passes so with the instruction above
    // 5 slic instructions and (in that case) the output is ready
    rpt $w_num_field_elems, (2f - 1f) / 8 - 1
1:
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
2:
    // 2nd loop - input only, to get to the point the output is ready. 4 - (elems -1) passes
    // so no loop in case of 5 items, 1 if 4 items etc....
    sub $w_num_field_elems, (5 - 1), $w_num_field_elems
    rpt $w_num_field_elems, (2f - 1f) / 8 - 1
1:
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
2:
    // 3rd loop - write the output, 3 - ( 4 -(elems-1)) =  elems - 2 passes,
    // there are 2 different writes after the loop
    sub $w_num_field_elems, (5 - 2), $w_num_field_elems
    rpt $w_num_field_elems, (2f - 1f) / 8 - 1
1:
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
2:
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
    bri .Lworker_process_group_field_row_write_and_end_\ID
3:
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS

    { bri .Lworker_process_group_field_row_write_and_end_\ID
     f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
.align 8 // Maintain alignment for the next macro and its repeat loop
.endm // worker_process_group_field_row_lt6_elems

////////////////////////////////////////////////////////////////////////////////
// >= 3 items, output stride = 2 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride2_8 ID SLIC_FLAGS PAR_OUT_PTR
    tapack $w_inoutout_triptr, $w_in_ptr, $\PAR_OUT_PTR, $\PAR_OUT_PTR

    add $w_num_field_elems, $w_num_field_elems_const, -3
    brneg $w_num_field_elems, .Lworker_process_group_field_row_implicit_zero_stride2_lt3_elems_\ID
    // The 1st load has no stride to ensure the load of (unused) partials,
    // and the output store don't clash in the loop
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }

    rpt $w_num_field_elems, (2f - 1f) / 8 - 1;
1:
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b000000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
2:
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { nop // Maintain alignment for the next macro and its repeat loop
      f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
.Lworker_process_group_field_row_implicit_zero_stride2_write_and_end_\ID:
    st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b11

.endm // worker_process_group_field_row_implicit_zero_stride2

////////////////////////////////////////////////////////////////////////////////
// < 3 items, output stride = 2 with no implicit zero
.macro worker_process_group_field_row_stride2_lt3_elems ID SLIC_FLAGS
.Lworker_process_group_field_row_stride2_lt3_elems_\ID:
    // + 3 (back to original num field elems) - 1
    add $w_num_field_elems, $w_num_field_elems, (3 - 1)

    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    // Branch to handle remainder of 1 field element case
    brz $w_num_field_elems, 3f

    // 2 field elements
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    // Write result
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }

    // Overwrite dummy result after branch
    {bri .Lworker_process_group_field_row_stride2_write_and_end_\ID
     f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS}

// One field element case, don't push the 2nd partials that were read in, as they can be an
// over-read and therefore if not valid inputs could cause an exception
3:
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS

    { bri .Lworker_process_group_field_row_stride2_write_and_end_\ID
     f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS }
.endm // worker_process_group_field_row_stride2_lt3_elems

////////////////////////////////////////////////////////////////////////////////
// >= 3 items, output stride = 2 with no implicit zero
.macro worker_process_group_field_row_stride2_8 ID SLIC_FLAGS PAR_OUT_PTR
    tapack $w_inoutout_triptr, $w_in_ptr, $\PAR_OUT_PTR, $\PAR_OUT_PTR

    add $w_num_field_elems, $w_num_field_elems_const, -3
    brneg $w_num_field_elems, .Lworker_process_group_field_row_stride2_lt3_elems_\ID

    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $azero, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $azero, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $azero, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $azero, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }

    { rpt $w_num_field_elems, (2f - 1f) / 8 - 1; fnop }
1:
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b000000
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $w_partials_0, \SLIC_FLAGS }
2:
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS
.Lworker_process_group_field_row_stride2_write_and_end_\ID:
    st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b11
.endm // worker_process_group_field_row_stride2

////////////////////////////////////////////////////////////////////////////////
// < 3 items, output stride = 2 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride2_lt3_elems ID SLIC_FLAGS
.Lworker_process_group_field_row_implicit_zero_stride2_lt3_elems_\ID:
    // + 3 (back to original num field elems) - 2
    add $w_num_field_elems, $w_num_field_elems, (3 - 2)
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0000
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    // Branch to handle 1 element
    brneg $w_num_field_elems, 3f
    // Must be 2 elements (store, dummy, store)
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b1100
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }

    { st64pace $w_output_pair, $w_inoutout_triptr+=, $mzero, 0b00
      f16v4hihoslic $w_output_0, $w_input_pair, $azero, \SLIC_FLAGS }
    f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS
    bri .Lworker_process_group_field_row_implicit_zero_stride2_write_and_end_\ID
3:
    {bri .Lworker_process_group_field_row_implicit_zero_stride2_write_and_end_\ID
    f16v4hihoslic $w_output_0, $azeros, $azero, \SLIC_FLAGS}
.endm // worker_process_group_field_row_implicit_zero_lt3_elems


////////////////////////////////////////////////////////////////////////////////
// Use the worker function and macros above to build the code for the stride = 1
// case
.section .text.CODELET_SYMBOL(half_worker_fn_stride1_8), "ax"
  // Instantiate the worker entry point and loop body
  worker_fn stride1_8  half
  // Instantiate routines used in the loop body
  worker_process_group_field_row_implicit_zero_stride1_lt5_elems W0_C8 TSLIC_F16V4_1x4_W0
  worker_process_group_field_row_implicit_zero_stride1_lt5_elems W1_C8 TSLIC_F16V4_1x4_W1
  worker_process_group_field_row_stride1_lt6_elems W0_C8 TSLIC_F16V4_1x4_W0
  worker_process_group_field_row_stride1_lt6_elems W1_C8 TSLIC_F16V4_1x4_W1


////////////////////////////////////////////////////////////////////////////////
// Use the worker function and macros above to build the code for the stride = 2
// case with half partials
 .section .text.CODELET_SYMBOL(half_worker_fn_stride2_8), "ax"
  // Instantiate the worker entry point and loop body
  worker_fn stride2_8  half
  // Instantiate routines used in the loop body for stride = 2
  worker_process_group_field_row_implicit_zero_stride2_lt3_elems W0_C8 TSLIC_F16V4_1x4_W0
  worker_process_group_field_row_implicit_zero_stride2_lt3_elems W1_C8 TSLIC_F16V4_1x4_W1
  worker_process_group_field_row_stride2_lt3_elems W0_C8 TSLIC_F16V4_1x4_W0
  worker_process_group_field_row_stride2_lt3_elems W1_C8 TSLIC_F16V4_1x4_W1

//=============
#undef w_rem
#undef w_out_ptr
#undef w_id
//=============

#endif // __IPU__
