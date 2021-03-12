// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifdef __IPU__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define CODELET_SYMBOL __runCodelet_popops__CompareAndSwapAtDistanceKeyVal___float_unsigned_int

#define STATE_OFFSET_keys 0
#define STATE_OFFSET_values 4
#define STATE_OFFSET_worklists_outer_ptr 8
#define STATE_OFFSET_worklists_outer_count 12
#define STATE_OFFSET_distanceToChangeOrder 16

.section .text.CODELET_SYMBOL, "ax"
.align 8
.worker
.Lfloat_unsigned_worker:
//=================
#define w_id m0
#define w_numWorkers m1
#define w_haveWork m2
//=================
  get $w_id, $WSR
  and $w_id, $w_id, CSR_W_WSR__CTXTID_M1__MASK

  ld32 $w_numWorkers, $mvertex_base, STATE_OFFSET_worklists_outer_count/4
  cmpult $w_haveWork, $w_id, $w_numWorkers
  brz $w_haveWork, .LnumEntriesLoopExit

//=================
#undef w_numWorkers
#undef w_haveWork
//=================
#define w_worklist m11
//=================
  ld32 $w_worklist, $mvertex_base, STATE_OFFSET_worklists_outer_ptr/4
  ld32 $w_worklist, $w_worklist, $mzero, $w_id
//=================
#undef w_id
//=================
#define w_innerElemCount m0
#define w_numEntriesM1 m1
#define w_singleElemOffsetBytes m3
#define w_keys_ptr m4
#define w_values_ptr m5
#define w_distance m6
#define w_elemCount m7
#define w_initialOffset m8
//=================
  ldz16step $w_numEntriesM1, $mzero, $w_worklist+=, 1
  // Load up pointers and advance
  ld32 $w_keys_ptr, $mvertex_base, $mzero, STATE_OFFSET_keys/4
  ld32 $w_values_ptr, $mvertex_base, $mzero, STATE_OFFSET_values/4
  ldz16step $w_initialOffset, $mzero, $w_worklist+=, 1
  ld32step $mzero, $mzero, $w_keys_ptr+=, $w_initialOffset
  ld32step $mzero, $mzero, $w_values_ptr+=, $w_initialOffset

//=================
#undef initialOffset
//=================
#define w_changeOrderCounter m8
#define w_order m9
#define w_packedOrderAndCount m10
//=================
  ld32step $w_packedOrderAndCount, $mzero, $w_worklist+=, 1
  and $w_order, $w_packedOrderAndCount, 0x1
  shr $w_changeOrderCounter, $w_packedOrderAndCount, 1
//=================
#undef w_packedOrderAndCount
//=================
#define w_distanceToChangeOrder m10
//=================
  ldz16step $w_innerElemCount, $mzero, $w_worklist+=, 1
  ld32 $w_distanceToChangeOrder, $mvertex_base, STATE_OFFSET_distanceToChangeOrder/4
  sub $w_changeOrderCounter, $w_distanceToChangeOrder, $w_changeOrderCounter

  setzi $w_singleElemOffsetBytes, 4

  // Start loading the first entry
  ldz16step $w_distance, $mzero, $w_worklist+=, 1
  ldz16step $w_elemCount, $mzero, $w_worklist+=, 1

.LnumEntriesLoopBegin:
.LelemCountLoopBegin:
      sub $w_elemCount, $w_elemCount, $w_innerElemCount
      sub $w_changeOrderCounter, $w_changeOrderCounter, $w_innerElemCount

      brnz $w_order, .LinnerElemCount_ascending

.macro INNER_LOOP COMP_INSTRUCTION

//=================
#define w_mside_pred m2
#define w_single_elem_offset m3
#define w_key_a a0
#define w_key_b a1
#define w_value_a a2
#define w_value_b a3
#define w_temp1 a4
#define w_temp2 a5
#define w_aside_pred a6
//=================

      ld32 $w_key_a, $w_keys_ptr, $mzero, 0
      ld32 $w_key_b, $w_keys_ptr, $mzero, $w_distance
      // Note that $w_innerElemCount is never 0
      add $w_innerElemCount, $w_innerElemCount, -2
      { brneg $w_innerElemCount, 3f
        \COMP_INSTRUCTION $w_aside_pred, $w_key_a, $w_key_b }
1:
      atom $w_mside_pred, $w_aside_pred
      brnz $w_mside_pred, 2f

      // Don't swap
      ld32step $w_key_a, $w_singleElemOffsetBytes, $w_keys_ptr+=, 1
      ld32 $w_key_b, $w_keys_ptr, $mzero, $w_distance
      { ld32step $mzero, $mzero, $w_values_ptr+=, 1
        \COMP_INSTRUCTION $w_aside_pred, $w_key_a, $w_key_b }
      brnzdec $w_innerElemCount, 1b
      bri 3f
2:
      // Swap
      st32 $w_key_a, $w_keys_ptr, $mzero, $w_distance
      st32step $w_key_b, $mzero, $w_keys_ptr+=, 1
      ld32 $w_key_a, $w_keys_ptr, $mzero, 0
      ld32 $w_key_b, $w_keys_ptr, $mzero, $w_distance
      { ld32 $w_value_a, $w_values_ptr, $mzero, 0
        \COMP_INSTRUCTION $w_aside_pred, $w_key_a, $w_key_b }
      ld32 $w_value_b, $w_values_ptr, $mzero, $w_distance
      st32 $w_value_a, $w_values_ptr, $mzero, $w_distance
      st32step $w_value_b, $mzero, $w_values_ptr+=, 1
      brnzdec $w_innerElemCount, 1b
3:
      atom $w_mside_pred, $w_aside_pred
      brnz $w_mside_pred, 2f

      // Don't swap
      //
      // Note that we advance the pointers like this because this
      // is the last iteration of the inner loop and reading at
      // an offset of $w_distance may cause clashes if this is the
      // last inner loop for this worker
      ld32step $mzero, $mzero, $w_keys_ptr+=, 1
      ld32step $mzero, $mzero, $w_keys_ptr+=, $w_distance
      ld32step $mzero, $mzero, $w_values_ptr+=, 1
      ld32step $mzero, $mzero, $w_values_ptr+=, $w_distance
      bri 3f
2:
      // Swap
      st32step $w_key_b, $mzero, $w_keys_ptr+=, $w_distance
      st32step $w_key_a, $mzero, $w_keys_ptr+=, 1
      ld32 $w_value_a, $w_values_ptr, $mzero, 0
      ld32 $w_value_b, $w_values_ptr, $mzero, $w_distance
      st32step $w_value_b, $mzero, $w_values_ptr+=, $w_distance
      st32step $w_value_a, $mzero, $w_values_ptr+=, 1
3:
//=================
#undef w_mside_pred
#undef w_key_a
#undef w_key_b
#undef w_value_a
#undef w_value_b
#undef w_temp1
#undef w_temp2
#undef w_aside_pred
//=================
      min $w_innerElemCount, $w_elemCount, $w_distance
.endm
.LinnerElemCount_descending:
      INNER_LOOP f32cmplt

      brz $w_changeOrderCounter, .LchangeOrder
      bri .LinnerElemCountEnd
.LinnerElemCount_ascending:
      INNER_LOOP f32cmpgt

      brnz $w_changeOrderCounter, .LinnerElemCountEnd
.LchangeOrder:
      sub $w_order, 1, $w_order
      mov $w_changeOrderCounter, $w_distanceToChangeOrder
.LinnerElemCountEnd:
      brnz $w_elemCount, .LelemCountLoopBegin
    ldz16step $w_distance, $mzero, $w_worklist+=, 1
    ldz16step $w_elemCount, $mzero, $w_worklist+=, 1
    min $w_innerElemCount, $w_elemCount, $w_distance
    brnzdec $w_numEntriesM1, .LnumEntriesLoopBegin
.LnumEntriesLoopExit:
  exitz $mzero

//======================================
// Supervisor
//======================================

DEF_STACK_USAGE 0 CODELET_SYMBOL
.globl CODELET_SYMBOL, @function
.supervisor
CODELET_SYMBOL:
//=================
#define s_vertexBase m0
#define s_workerEntry m1
//=================
  setzi $s_workerEntry, .Lfloat_unsigned_worker
  runall $s_workerEntry, $s_vertexBase, 0
  sync TEXCH_SYNCZONE_LOCAL
  br $lr

#endif // __IPU__
