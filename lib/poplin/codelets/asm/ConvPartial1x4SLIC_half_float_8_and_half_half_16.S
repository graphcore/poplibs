// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// Computes a 1x4 convolution using SLIC. A contiguous field is
// partitioned between workers for each position of each 1x4
// sub-kernel.
//
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define CODELET_SYMBOL_ENTRY(partials_type, stride, conv_chains) \
        __runCodelet_poplin__ConvPartial1xNSLIC___half_ ## partials_type ##_## stride ## _true_4_ ## conv_chains

#define CODELET_SYMBOL(suffix) __runCodelet_poplin__ConvPartial1x4SLIC___half_ ## suffix

//=============
#define HALF_PARTIALS_OUTPUT_STRIDE 1
#define FLOAT_PARTIALS_OUTPUT_STRIDE 2

#define NUM_WEIGHTS_PER_WORKER_LOOP 64
#define INPUT_ELEM_SIZE 2
#define WEIGHTS_ELEM_SIZE 2

#define HALF_PARTIAL_ELEM_SIZE 2
#define FLOAT_PARTIAL_ELEM_SIZE 4

#define BYTES_PER_DELTAN 4
#define WORKLIST_ALIGN_LOG2 1
#define WORKLIST_DELTAN_OFFSET_BITS (21 - WORKLIST_ALIGN_LOG2)
//=============

//=============
// The vertex is provided with a buffer containing weights, a pointer and then
// space to hold a copy of the output.  This is used to read partials while writing
// the actual output (or vica versa). This is because we need to force the alignment
// so that the ld2xst64pace can be used (Alternate memory segments in interleaved memory).
// In some cases we need an aligned (16 byte boundary) pointer, in other cases
// misaligned (16 byte boundary plus 8 bytes).
// The buffer is aligned to 16 bytes so an offset of 200 bytes gives: 200%16 = 8 bytes offset
// 208 bytes gives: 208%16 = 0 bytes offset
//
// Note: When partials are float the buffer must always be misaligned as each
//       loop writes 8 bytes, but strides 16 - so misalignment is maintained with
//       each read/write instrudtion.
//
//       When partials are half 8 bytes are written each loop and the stride is 8 -
//       so alignment/misalignment changes on each read/write instruction.
//       As the lead in to the stride=1 case is 6 reads we require misaligned pointers
//       at the start.  However the lead in the stride=2 case is 3 instructions so
//       we require aligned pointers at the start so that the 3 reads produce a misalignment
//       once we start using ld2x64pace.
//       Technically all this means that a buffer is not needed for the stride=2 half case
//       but having no buffer is a fairly small optimisation so this isn't implemented at present
#define BUFFER_MISALIGNED_OFFSET 200
#define BUFFER_ALIGNED_OFFSET 208
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
// Provides an offset into the below output pointer storage to load from
#define PROCESS_GROUP_WORKER_STATE_OFFSET_swap_out_ptrs_on_load (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + 4)
// out_ptrs storage provides 3 output pointers, the first and last of
// which are the same. This allows us to flip the pointers when loading
// by just loading at an offset.
#define PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero_and_strides (PROCESS_GROUP_WORKER_STATE_BASE_OFFSET + 8)
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
// Jump table for weight loading routines
.global CODELET_SYMBOL(worker_load_weights_jump_table)
.section .data.CODELET_SYMBOL(worker_load_weights_jump_table), "a", @progbits
.align 4
CODELET_SYMBOL(worker_load_weights_jump_table):
.int CODELET_SYMBOL(worker_load_weights_4x1x1)
.int CODELET_SYMBOL(worker_load_weights_2x2x2)
.int 0


////////////////////////////////////////////////////////////////////////////////
// Supervisor function, entry point macro definition
.macro supervisor_fn PARTIALS_TYPE CONV_CHAINS

// Supervisor uses MAX_STACK_SIZE, workers  don't use any stack
DEF_STACK_USAGE  MAX_STACK_SIZE   CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_CHAINS)
DEF_STACK_USAGE  MAX_STACK_SIZE   CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_CHAINS)

.global CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_CHAINS)
.type CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_CHAINS), @function

.global CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_CHAINS)
.type CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_CHAINS), @function

// Parameters we can deduce directly from the macro params
.ifc "\PARTIALS_TYPE", "float"
  .equ  OUTPUT_STRIDE, FLOAT_PARTIALS_OUTPUT_STRIDE
  .equ  SECOND_WEIGHT_BANK, 1
.else
  .equ  OUTPUT_STRIDE, HALF_PARTIALS_OUTPUT_STRIDE
  .equ  SECOND_WEIGHT_BANK, 32
  .equ  STRIDE1_BUFFER_OFFSET, BUFFER_MISALIGNED_OFFSET
  // The case where we have half partials, stride2 is special, we need the buffer pointer
  // to be aligned as strides result in 64 bit steps not 128.
  .equ  STRIDE2_BUFFER_OFFSET, BUFFER_ALIGNED_OFFSET
.endif

.section .text.CODELET_SYMBOL(\PARTIALS_TYPE\()_entry_stride2), "ax"
.align 4
.supervisor

CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),2,\CONV_CHAINS):
  // The only differences in codelet execution come in the worker function body,
  // much much later on.  So put a pointer to the required function onto the stack
  // to pick up later.  This will mean that we will link just the functions(s)
  // required whereas using a flag would reference both labels
//=============
#define s_out_field_buffer_offset m2
#define s_worker_fn_ptr m3
//=============
  setzi  $s_worker_fn_ptr, CODELET_SYMBOL(\PARTIALS_TYPE\()_stride2)
.ifc "\PARTIALS_TYPE", "half"
  setzi  $s_out_field_buffer_offset, STRIDE2_BUFFER_OFFSET
.endif
  bri CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_CHAINS\()_common_entry)

.section .text.CODELET_SYMBOL(\PARTIALS_TYPE\()_entry_stride1), "ax"
.align 4
.supervisor

CODELET_SYMBOL_ENTRY(\PARTIALS_TYPE\(),1,\CONV_CHAINS):

  setzi  $s_worker_fn_ptr, CODELET_SYMBOL(\PARTIALS_TYPE\()_stride1)
.ifc "\PARTIALS_TYPE", "half"
  setzi  $s_out_field_buffer_offset, STRIDE1_BUFFER_OFFSET
.endif
  bri CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_CHAINS\()_common_entry)

.section .text.CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_CHAINS), "ax"
.align 4
.supervisor
CODELET_SYMBOL(\PARTIALS_TYPE\()\CONV_CHAINS\()_common_entry):
  // Calculation of $s_worklists_base_ptr is essentially the critical path as
  // it requires 4 instructions to load, mask and store it for a total of 19 cycles minimum
  // (6 cycles * 3 instructions for load, shl, shr + 1 cycle to issue the store)
  ld32 $s_worklists_base_ptr, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_worklists/4
  ld32 $s_worklists_deltan_ptr, $msupervisor_vertex_base, $mzero, (SUPERVISOR_STATE_OFFSET_worklists + 4)/4
  // Setup space on the stack to store m9/m10 and worker state.
  add $sp, $sp, -MAX_STACK_SIZE

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
#define s_swap_out_ptrs_on_load m10
//=============
  ldz8 $s_swap_out_ptrs_on_load, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_outPtrLoadOffset/1
  ld32 $s_out_ptr_iterator, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_out/4
  ld32 $s_worker_function, $s_worker_function, $mzero, $s_mode
//=============
#undef s_mode
//=============
#define s_in_ptr_iterator m4
//=============
.ifc "\PARTIALS_TYPE", "float"
  add $s_out_field_buffer_ptr, $s_out_field_buffer_ptr, BUFFER_MISALIGNED_OFFSET
.else
  add $s_out_field_buffer_ptr, $s_out_field_buffer_ptr, $s_out_field_buffer_offset
.endif
//=============
#undef s_out_field_buffer_offset
//=============
#define s_num_conv_group_groups_m1 m2
//=============
  st32 $m9, $sp, $mzero, SUPERVISOR_STACK_OFFSET_m9/4
  shr $s_worklists_base_ptr, $s_worklists_base_ptr, 8
  shr $s_worklists_deltan_ptr, $s_worklists_deltan_ptr, 8

  ldz16 $s_num_conv_group_groups_m1, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_numConvGroupGroupsM1/2
  st32 $s_swap_out_ptrs_on_load, $sp, $mzero, SUPERVISOR_STACK_OFFSET_swap_out_ptrs_on_load/4
//=============
#undef s_swap_out_ptrs_on_load
//=============
#define s_num_sub_kernels_m1 m10
//=============
  ldz16 $s_num_sub_kernels_m1, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_numSubKernelsM1/2
  ld32 $s_in_ptr_iterator, $msupervisor_vertex_base, $mzero, SUPERVISOR_STATE_OFFSET_in/4
  st32 $s_out_field_buffer_ptr, $sp, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs + 0)/4
  st32 $s_out_field_buffer_ptr, $sp, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs + 8)/4
  st32 $s_worklists_base_ptr, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_worklist/4
  st32 $s_worklists_deltan_ptr, $sp, $mzero, SUPERVISOR_STACK_OFFSET_worklists_deltan_ptr/4
  st32 $s_num_sub_kernels_m1, $sp, $mzero, SUPERVISOR_STACK_OFFSET_num_sub_kernels_m1/4
//=============
#undef s_num_sub_kernels_m1
//=============
#define s_worker_function2 m10
//=============
  ld32 $s_worker_function2, $sp, $mzero, SUPERVISOR_STACK_OFFSET_worker_fn_ptr/4
//=============
#undef s_out_field_buffer_ptr
#undef s_worklists_base_ptr
#undef s_worklists_deltan_ptr
//=============
#define s_implicit_zero_or_strides m1
#define s_swap_out_ptrs_on_load m7
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
    or $s_implicit_zero_or_strides, $mzero, (1 << 31)
    ldz8 $s_swap_out_ptrs_on_load, $sp, $mzero, SUPERVISOR_STACK_OFFSET_swap_out_ptrs_on_load/1
    ld32step $s_in_ptr, $mzero, $s_in_ptr_iterator+=, 1
    ld32step $s_out_ptr, $mzero, $s_out_ptr_iterator+=, 1
    nop
    nop
    or $s_implicit_zero_or_strides, $s_implicit_zero_or_strides, OUTPUT_STRIDE
    st32 $s_swap_out_ptrs_on_load, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_swap_out_ptrs_on_load/4
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
    st32 $s_implicit_zero_or_strides, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero_and_strides/4
//=============
#undef s_implicit_zero_or_strides
//=============
#define s_num_sub_kernels_m1 m1
//=============
    ld32 $s_num_sub_kernels_m1, $sp, $mzero, SUPERVISOR_STACK_OFFSET_num_sub_kernels_m1/4
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
#define s_implicit_zero_or_strides m8
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
      sub       $s_swap_out_ptrs_on_load, 4, $s_swap_out_ptrs_on_load
      ld64putcs (2 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (3 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (0 * 4)
      ld64putcs (1 * 4)
      ld64putcs (0 * 4 + SECOND_WEIGHT_BANK)
      ld64putcs (1 * 4 + SECOND_WEIGHT_BANK)

      st32 $s_worklists_deltan_ptr, $sp, $mzero, (PROCESS_GROUP_WORKER_STATE_OFFSET_worklist + 4)/4
      add $s_worklists_deltan_ptr, $s_worklists_deltan_ptr, BYTES_PER_DELTAN * CTXT_WORKERS

      or $s_implicit_zero_or_strides, $mzero, OUTPUT_STRIDE
      runall $s_worker_function2, $sp, 0
      sync TEXCH_SYNCZONE_LOCAL
      st32 $s_implicit_zero_or_strides, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero_and_strides/4
      st32 $s_swap_out_ptrs_on_load, $sp, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_swap_out_ptrs_on_load/4
//=============
#undef s_implicit_zero_or_strides
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
  supervisor_fn float 2

#if __IPU_ARCH_VERSION__ >=2
  supervisor_fn half 4
#endif

////////////////////////////////////////////////////////////////////////////////
//=============
#define w_in_base_ptr m2
#define w_curr_out_base_ptr m3
#define w_last_out_base_ptr m4
// Registers above must be preserved between runs for the same conv group group
#define w_inoutout_triptr m0:1
#define w_work_items m5
#define w_worklist_ptr m6
#define w_id m10
#define w_implicit_zero_and_strides m11

#define w_input_pair a0:1
#define w_partials_pair a2:3
#define w_input_and_partials_pairs a0:3
#define w_dummy_a64 a0:1
#define w_output_pair a4:5
// We need a second register in order to delay
// output stores so that we don't cause a conflict with
// partial loads.
#define w_output_pair2 a6:7
//=============
////////////////////////////////////////////////////////////////////////////////
// Macros to ensure repeat alignment
// Rely on careful, exclusive use of the 2 labels used
////////////////////////////////////////////////////////////////////////////////
.macro ALIGNED_RPT_START LOOP_COUNT
  // The condition means:
  // If the delta (which can be computed at assemble time) between a ".align 8 aligned
  // label" and the current point is not a multiple of 8
  .if ( . - 7b ) & 0x7
    // If misaligned vs the last aligned label, this will cause alignment
    rpt \LOOP_COUNT, (7f - 6f) / 8 - 1;
  .else
    // Otherwise maintain alignment
    {rpt \LOOP_COUNT, (7f - 6f) / 8 - 1; fnop}
  .endif
6:
.endm

.macro ALIGNED_RPT_END
7:
.endm

.macro ALIGNED_RPT_BASE
.align 8
7:
.endm


////////////////////////////////////////////////////////////////////////////////
// Worker routine to process a single group entry point in macro definition

.macro worker_fn WORK_MACRO_POSTFIX SLIC_INSTR PARTIALS_TYPE
.worker
ALIGNED_RPT_BASE
.ifc "\PARTIALS_TYPE", "float"
  .equ PARTIAL_ELEM_SIZE, FLOAT_PARTIAL_ELEM_SIZE
.else
  .equ PARTIAL_ELEM_SIZE, HALF_PARTIAL_ELEM_SIZE
.endif

.global CODELET_SYMBOL(\PARTIALS_TYPE\()_\WORK_MACRO_POSTFIX)
.type CODELET_SYMBOL(\PARTIALS_TYPE\()_\WORK_MACRO_POSTFIX), @function

CODELET_SYMBOL(\PARTIALS_TYPE\()_\WORK_MACRO_POSTFIX):
  { get $w_id, $WSR
    setzi $a0, (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT) }
  { and $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK
    uput $FP_CLR, $a0 }

//=============
#undef w_in_base_ptr
//=============
#define w_swap_out_ptrs_on_load m2
//=============
  ld32 $w_swap_out_ptrs_on_load, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_swap_out_ptrs_on_load/4
  ld32 $w_last_out_base_ptr, $mvertex_base, $w_swap_out_ptrs_on_load, PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs/4
  ld32 $w_curr_out_base_ptr, $mvertex_base, $w_swap_out_ptrs_on_load, (PROCESS_GROUP_WORKER_STATE_OFFSET_out_ptrs + 4)/4
//=============
#undef w_swap_out_ptrs_on_load
//=============
#define w_in_base_ptr m2
//=============
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

  ld32 $w_implicit_zero_and_strides, $mvertex_base, $mzero, PROCESS_GROUP_WORKER_STATE_OFFSET_implicit_zero_and_strides/4

// Loop over work items
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
#define w_last_out_ptr m9
#define w_out_ptr_pair m8:9
//=============
    add $w_last_out_ptr, $w_last_out_base_ptr, $w_out_offset
    add $w_curr_out_ptr, $w_curr_out_base_ptr, $w_out_offset
//=============
#undef w_out_offset
#undef w_id
//=============
#define w_num_field_elems m10
//=============
.ifc "\PARTIALS_TYPE", "float"
    ldz16 $w_num_field_elems, $w_worklist_ptr, $mzero, 0
    // If there's nothing to do, move on to the next work item.
    brz $w_num_field_elems, 5f

    brpos $w_implicit_zero_and_strides, 4f

    // Implicit zeroing path.
    // Two loops required, accounting for 2 float partials each
    worker_process_group_field_row_implicit_zero_\WORK_MACRO_POSTFIX W0 TSLIC_F16V4_1x4_W0 \SLIC_INSTR
    ldz16step $w_num_field_elems, $mzero, $w_worklist_ptr+=, 1

    // Dummy loads
    ld2x64pace $w_dummy_a64, $azeros, $w_out_ptr_pair+=, $mzero, 0b0000

    worker_process_group_field_row_implicit_zero_\WORK_MACRO_POSTFIX W1 TSLIC_F16V4_1x4_W1 \SLIC_INSTR
    brnzdec $w_work_items, .Lworker_process_group_worklist_loop\@
    exitz $mzero

4:
    // No implicit zeroing path.
    // Two loops required, accounting for 2 float partials each
    worker_process_group_field_row_\WORK_MACRO_POSTFIX W0 TSLIC_F16V4_1x4_W0 \SLIC_INSTR
    ldz16step $w_num_field_elems, $mzero, $w_worklist_ptr+=, 1
    // Dummy loads
    ld2x64pace $w_dummy_a64, $azeros, $w_out_ptr_pair+=, $mzero, 0b0000
    worker_process_group_field_row_\WORK_MACRO_POSTFIX W1 TSLIC_F16V4_1x4_W1 \SLIC_INSTR
.endif

.ifc "\PARTIALS_TYPE", "half"
    ldz16step $w_num_field_elems, $mzero, $w_worklist_ptr+=,1
    // If there's nothing to do, move on to the next work item.
    brz $w_num_field_elems, 5f

    brpos $w_implicit_zero_and_strides, 4f

    // Implicit zeroing path
    // One loop required, accounting for 4 half partials
    worker_process_group_field_row_implicit_zero_\WORK_MACRO_POSTFIX W0_32 TSLIC_F16V4_1x4_W0 \SLIC_INSTR
    brnzdec $w_work_items, .Lworker_process_group_worklist_loop\@
    exitz $mzero

4:
    // No implicit zeroing path
    // One loop required, accounting for 4 half partials
    worker_process_group_field_row_\WORK_MACRO_POSTFIX W0_32 TSLIC_F16V4_1x4_W0 \SLIC_INSTR
.endif

5:
    brnzdec $w_work_items, .Lworker_process_group_worklist_loop\@
0:
  exitz $mzero
.endm

////////////////////////////////////////////////////////////////////////////////
// Macro definitions to process data using SLIC instrutions
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// >= 5 items, output stride = 1 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride1 ID SLIC_FLAGS SLIC_INSTR
    tapack $w_inoutout_triptr, $w_in_ptr, $w_last_out_ptr, $w_curr_out_ptr
    add $w_num_field_elems, $w_num_field_elems, -5
    brneg $w_num_field_elems, .Lworker_process_group_field_row_implicit_zero_stride1_lt5_elems_\ID
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }

ALIGNED_RPT_START $w_num_field_elems
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b010100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
ALIGNED_RPT_END

    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS }
.Lworker_process_group_field_row_implicit_zero_write_and_end_\ID:
    st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b11
.endm // worker_process_group_field_row_implicit_zero

////////////////////////////////////////////////////////////////////////////////
// < 5 items, output stride = 1 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride1_lt5_elems ID SLIC_FLAGS SLIC_INSTR
.Lworker_process_group_field_row_implicit_zero_stride1_lt5_elems_\ID:
    // + 5 (back to original num field elems) - 2
    add $w_num_field_elems, $w_num_field_elems, (5 - 2)
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    // Handle 1 element separately
    brneg $w_num_field_elems, 3f

    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
ALIGNED_RPT_START $w_num_field_elems
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
ALIGNED_RPT_END
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS }
    bri .Lworker_process_group_field_row_implicit_zero_write_and_end_\ID

3:
    { bri .Lworker_process_group_field_row_implicit_zero_write_and_end_\ID
      \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS }
.endm // worker_process_group_field_row_implicit_zero_lt5_elems


////////////////////////////////////////////////////////////////////////////////
// >= 5 items, output stride = 1 with no implicit zero
.macro worker_process_group_field_row_stride1 ID SLIC_FLAGS SLIC_INSTR
    tapack $w_inoutout_triptr, $w_in_ptr, $w_last_out_ptr, $w_curr_out_ptr

    add $w_num_field_elems, $w_num_field_elems, -5
    brneg $w_num_field_elems, .Lworker_process_group_field_row_stride1_lt5_elems_\ID

    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }

ALIGNED_RPT_START $w_num_field_elems
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b010100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
ALIGNED_RPT_END

    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS }
.Lworker_process_group_field_row_write_and_end_\ID:
    st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
.endm // worker_process_group_field_row

////////////////////////////////////////////////////////////////////////////////
// < 5 items, output stride = 1 with no implicit zero
.macro worker_process_group_field_row_stride1_lt5_elems ID SLIC_FLAGS SLIC_INSTR
.Lworker_process_group_field_row_stride1_lt5_elems_\ID:
    // + 5 (back to original num field elems) - 1
    add $w_num_field_elems, $w_num_field_elems, (5 - 1)
    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    // Handle remainder of 1 field element case separately
    brz $w_num_field_elems, 3f

ALIGNED_RPT_START $w_num_field_elems
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
ALIGNED_RPT_END

    sub $w_num_field_elems, (5 - 1), $w_num_field_elems

ALIGNED_RPT_START $w_num_field_elems
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
ALIGNED_RPT_END

    sub $w_num_field_elems, (5 - 2), $w_num_field_elems

ALIGNED_RPT_START $w_num_field_elems
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
ALIGNED_RPT_END

    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0111
      \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS }
    bri .Lworker_process_group_field_row_write_and_end_\ID
3:
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS

    { bri .Lworker_process_group_field_row_write_and_end_\ID
     \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS }
.endm // worker_process_group_field_row_lt5_elems

////////////////////////////////////////////////////////////////////////////////
// >= 3 items, output stride = 2 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride2 ID SLIC_FLAGS SLIC_INSTR
    tapack $w_inoutout_triptr, $w_in_ptr, $w_last_out_ptr, $w_curr_out_ptr

    add $w_num_field_elems, $w_num_field_elems, -3
    brneg $w_num_field_elems, .Lworker_process_group_field_row_implicit_zero_stride2_lt3_elems_\ID
// We don't need to stride the partials ptr to get the correct result (as we don't use the value read)
// but we do need to maintain an alignment such that there is no conflict when we use the ld2xst64pace
// instruction in the inner loop.  Given starting alignments and the stride parameter the use of 0b1100
// here and in the loop produces:
// float case: pointers misaligned to start with, partials += (5 * STRIDE) pre the loop. = 10 (even so (mis)alignment maintained)
//             In loop - partialsPtr+=STRIDE, outPtr+=STRIDE.  STRIDE is 2 so both remain misaligned
// half case: pointers aligned to start with, partials += (5 * STRIDE) pre the loop. = 5 (so become misaligned)
//             In loop - partialsPtr+=STRIDE, outPtr+=STRIDE.  STRIDE is 1 so both remain misaligned
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }

ALIGNED_RPT_START $w_num_field_elems
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b010100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
ALIGNED_RPT_END

    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }

      \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS
.Lworker_process_group_field_row_implicit_zero_stride2_write_and_end_\ID:
    st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b11
.endm // worker_process_group_field_row_implicit_zero_stride2

////////////////////////////////////////////////////////////////////////////////
// < 3 items, output stride = 2 with no implicit zero
.macro worker_process_group_field_row_stride2_lt3_elems ID SLIC_FLAGS SLIC_INSTR
.Lworker_process_group_field_row_stride2_lt3_elems_\ID:
    // + 3 (back to original num field elems) - 1
    add $w_num_field_elems, $w_num_field_elems, (3 - 1)

    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    // Branch to handle remainder of 1 field element case
    brz $w_num_field_elems, 3f

    // 2 field elements
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    // Write result
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }

    // Overwrite dummy result after branch
    {bri .Lworker_process_group_field_row_stride2_write_and_end_\ID
     \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS}

// One field element case, don't push the 2nd partials that were read in, as they can be an
// over-read and therefore if not valid inputs could cause an exception
3:
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS

    { bri .Lworker_process_group_field_row_stride2_write_and_end_\ID
     \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS }
.endm // worker_process_group_field_row_stride2_lt3_elems

////////////////////////////////////////////////////////////////////////////////
// >= 3 items, output stride = 2 with no implicit zero
.macro worker_process_group_field_row_stride2 ID SLIC_FLAGS SLIC_INSTR
    tapack $w_inoutout_triptr, $w_in_ptr, $w_last_out_ptr, $w_curr_out_ptr

    add $w_num_field_elems, $w_num_field_elems, -3
    brneg $w_num_field_elems, .Lworker_process_group_field_row_stride2_lt3_elems_\ID

    ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $azeros, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $azeros, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $azeros, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $azeros, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $w_partials_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }

ALIGNED_RPT_START $w_num_field_elems
    { ld2xst64pace $w_input_and_partials_pairs, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b010100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $w_partials_pair, \SLIC_FLAGS }
ALIGNED_RPT_END

    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ldst64pace $w_input_pair, $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b1100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }

      \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS
.Lworker_process_group_field_row_stride2_write_and_end_\ID:
    st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b11
.endm // worker_process_group_field_row_stride2

////////////////////////////////////////////////////////////////////////////////
// < 3 items, output stride = 2 with implicit zero
.macro worker_process_group_field_row_implicit_zero_stride2_lt3_elems ID SLIC_FLAGS SLIC_INSTR
.Lworker_process_group_field_row_implicit_zero_stride2_lt3_elems_\ID:
    // + 3 (back to original num field elems) - 2
    add $w_num_field_elems, $w_num_field_elems, (3 - 2)
    ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    // Branch to handle 1 element
    brneg $w_num_field_elems, 3f
    // Must be 2 elements (store, dummy store, store)
    { ld2x64pace $w_input_pair, $azeros, $w_inoutout_triptr+=, $mzero, 0b0100
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }

    { st64pace $w_output_pair, $w_inoutout_triptr+=, $w_implicit_zero_and_strides, 0b01
      \SLIC_INSTR $w_output_pair, $w_input_pair, $azeros, \SLIC_FLAGS }
    \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS
    bri .Lworker_process_group_field_row_implicit_zero_stride2_write_and_end_\ID
3:
    {bri .Lworker_process_group_field_row_implicit_zero_stride2_write_and_end_\ID
    \SLIC_INSTR $w_output_pair, $azeros, $azeros, \SLIC_FLAGS}
.endm // worker_process_group_field_row_implicit_zero_lt3_elems

////////////////////////////////////////////////////////////////////////////////
// Use the worker function and macros above to build the code for the stride = 1
// case with float partials
.section .text.CODELET_SYMBOL(float_worker_fn_stride1), "ax"
  // Instantiate the worker entry point and loop body
  worker_fn stride1 f16v4sisoslic float
  // Instantiate routines used in the loop body
  worker_process_group_field_row_implicit_zero_stride1_lt5_elems W0 TSLIC_F16V4_1x4_W0 f16v4sisoslic
  worker_process_group_field_row_implicit_zero_stride1_lt5_elems W1 TSLIC_F16V4_1x4_W1 f16v4sisoslic
  worker_process_group_field_row_stride1_lt5_elems W0 TSLIC_F16V4_1x4_W0 f16v4sisoslic
  worker_process_group_field_row_stride1_lt5_elems W1 TSLIC_F16V4_1x4_W1 f16v4sisoslic


////////////////////////////////////////////////////////////////////////////////
// Use the worker function and macros above to build the code for the stride = 2
// case with float partials
.section .text.CODELET_SYMBOL(float_worker_fn_stride2), "ax"
  // Instantiate the worker entry point and loop body
  worker_fn stride2 f16v4sisoslic float
  // Instantiate routines used in the loop body for stride = 2
  worker_process_group_field_row_implicit_zero_stride2_lt3_elems W0 TSLIC_F16V4_1x4_W0 f16v4sisoslic
  worker_process_group_field_row_implicit_zero_stride2_lt3_elems W1 TSLIC_F16V4_1x4_W1 f16v4sisoslic
  worker_process_group_field_row_stride2_lt3_elems W0 TSLIC_F16V4_1x4_W0 f16v4sisoslic
  worker_process_group_field_row_stride2_lt3_elems W1 TSLIC_F16V4_1x4_W1 f16v4sisoslic


////////////////////////////////////////////////////////////////////////////////
// Use the worker function and macros above to build the code for the stride = 1
// case with half partials

#if __IPU_ARCH_VERSION__ >=2
.section .text.CODELET_SYMBOL(half_worker_fn_stride1), "ax"
  // Instantiate the worker entry point and loop body
  worker_fn stride1 f16v4hihov4slic half
  // Instantiate routines used in the loop body
  worker_process_group_field_row_implicit_zero_stride1_lt5_elems W0_32 TSLIC_F16V4_1x4_W0 f16v4hihov4slic
  worker_process_group_field_row_stride1_lt5_elems W0_32 TSLIC_F16V4_1x4_W0 f16v4hihov4slic
#endif

////////////////////////////////////////////////////////////////////////////////
// Use the worker function and macros above to build the code for the stride = 2
// case with half partials
#if __IPU_ARCH_VERSION__ >=2
.section .text.CODELET_SYMBOL(half_worker_fn_stride2), "ax"
  // Instantiate the worker entry point and loop body
  worker_fn stride2 f16v4hihov4slic half
  // Instantiate routines used in the loop body for stride = 2
  worker_process_group_field_row_implicit_zero_stride2_lt3_elems W0_32 TSLIC_F16V4_1x4_W0 f16v4hihov4slic
  worker_process_group_field_row_stride2_lt3_elems W0_32 TSLIC_F16V4_1x4_W0 f16v4hihov4slic
#endif
//=============
#undef w_rem
#undef w_out_ptr
#undef w_id
//=============


////////////////////////////////////////////////////////////////////////////////
// Worker weight loading routines
//=============
#define w_id m0
#define w_weights_ptr m1
#define w_weights_out_offset m2
#define w_weight_pair0 a0
#define w_weight_pair1 a1
#define w_weight_quad a0:1
//=============
.global CODELET_SYMBOL(worker_load_weights_4x1x1)
.type CODELET_SYMBOL(worker_load_weights_4x1x1), @function
.section .text.CODELET_SYMBOL(worker_load_weights_4x1x1), "ax"
CODELET_SYMBOL(worker_load_weights_4x1x1):
  get $w_id, $WSR
  and $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

  // Each worker loads and arranges the weights for 4 conv groups.
  // 4 workers collectively load the weights for the 4 kernel elements.
  //
  // NOTE: We do not actually stop the other 2 workers from doing anything
  // and instead allocate extra stack space for the garbage writes and
  // use the fact that we can overread by up to 24 bytes safely from the
  // weights to avoid the runtime overhead. We need to overread
  // 2 workers * 8 bytes, and overwrite 2 workers * 32 bytes.
  { ld32 $w_weights_ptr, $mvertex_base, $mzero, LOAD_WEIGHTS_WORKER_STATE_OFFSET_weights_ptr/4
    or $a3, $azero, $azero }
  { ld64 $w_weight_quad, $w_weights_ptr, $mzero, $w_id
    or $a4, $azero, $azero }

  // We need to offset each worker's stores correctly
  { mul $w_weights_out_offset, $w_id, 32
    sort4x16lo $a2, $w_weight_pair0, $azero }
  // Store [cg0, 0, 0, 0]
  { st64 $a2:3, $mvertex_base, $w_weights_out_offset, 0
    sort4x16hi $a2, $azero, $w_weight_pair0 }
  // Store [0, cg1, 0, 0]
  { st64 $a2:3, $mvertex_base, $w_weights_out_offset, 1
    sort4x16lo $a5, $w_weight_pair1, $azero }
  // Store [0, 0, cg2, 0]
  { st64 $a4:5, $mvertex_base, $w_weights_out_offset, 2
    sort4x16hi $a5, $azero, $w_weight_pair1 }
  // Store [0, 0, 0, cg3]
  st64 $a4:5, $mvertex_base, $w_weights_out_offset, 3
  exitz $mzero
//=============
#undef w_weight_quad
#undef w_weight_pair1
#undef w_weight_pair0
#undef w_weights_out_offset
#undef w_weights_ptr
#undef w_id
//=============

//=============
#define w_id m0
#define w_weights_ptr m1
#define w_weights_in_offset m2
#define w_weights_out_offset m3
#define w_is_odd m4
//=============
.global CODELET_SYMBOL(worker_load_weights_2x2x2)
.type CODELET_SYMBOL(worker_load_weights_2x2x2), @function
CODELET_SYMBOL(worker_load_weights_2x2x2):
  get $w_id, $WSR
  and $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

  ld32 $w_weights_ptr, $mvertex_base, $mzero, LOAD_WEIGHTS_WORKER_STATE_OFFSET_weights_ptr/4

  // Each worker handles 2 input channels 2 output channels and 2 kernel positions
  // for a total of 8 actual weights.
  //
  // NOTE: We do not actually stop the other 2 workers from doing anything
  // and instead allocate extra stack space for the garbage writes and
  // use the fact that we can overread by up to 24 bytes safely from the
  // weights to avoid the runtime overhead. We need to overread
  // 2 workers * 16 bytes, and overwrite 2 workers * 32 bytes.
  mul $w_weights_in_offset, $w_id, 16
  mul $w_weights_out_offset, $w_id, 32

  ld64 $a0:1, $w_weights_ptr, $w_weights_in_offset, 0
  { ld64 $a2:3, $w_weights_ptr, $w_weights_in_offset, 1
    sort4x32lo $a4:5, $a0:1, $azeros }
  { st64 $a4:5, $mvertex_base, $w_weights_out_offset, 0
    sort4x32hi $a4:5, $a0:1, $azeros }
  { st64 $a4:5, $mvertex_base, $w_weights_out_offset, 1
    sort4x32lo $a4:5, $azeros, $a2:3 }
  { st64 $a4:5, $mvertex_base, $w_weights_out_offset, 2
    sort4x32hi $a4:5, $azeros, $a2:3 }
  st64 $a4:5, $mvertex_base, $w_weights_out_offset, 3
  exitz $mzero

//=============
#undef w_weights_out_offset
#undef w_weights_in_offset
#undef w_weights_ptr
#undef w_id
//=============

#endif // __IPU__
