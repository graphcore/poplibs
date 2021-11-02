// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"

// This file contains the assembly for 'BroadcastSelect' and
// 'BroadcastSelectorSelect[InPlace]' codelets (for all input types)
// implementing the ternary 'Select' operator (like the C conditional operator):
//
//    Select(in1, in2, in3)  ==>   in3? in1 : in2
//
// for the cases where input 'in1' and 'in2' are scalar tensors, or where input
// 'in3' is a scalar tensor.

#define BCAST_SELECT_FLOAT __runCodelet_popops__BroadcastSelect___float
#define BCAST_SELECT_INT __runCodelet_popops__BroadcastSelect___int
#define BCAST_SELECT_UNSIGNED_INT __runCodelet_popops__BroadcastSelect___unsigned_int
#define BCAST_SELECT_HALF __runCodelet_popops__BroadcastSelect___half
#define BCAST_SELECT_BOOL __runCodelet_popops__BroadcastSelect___bool

#define BCAST_SELECTOR_SELECT_HALF __runCodelet_popops__BroadcastSelectorSelect___half
#define BCAST_SELECTOR_SELECT_FLOAT __runCodelet_popops__BroadcastSelectorSelect___float
#define BCAST_SELECTOR_SELECT_INT __runCodelet_popops__BroadcastSelectorSelect___int
#define BCAST_SELECTOR_SELECT_UNSIGNED_INT __runCodelet_popops__BroadcastSelectorSelect___unsigned_int
#define BCAST_SELECTOR_SELECT_BOOL __runCodelet_popops__BroadcastSelectorSelect___bool

#define BCAST_SELECTOR_SELECT_INPLACE_HALF __runCodelet_popops__BroadcastSelectorSelectInPlace___half
#define BCAST_SELECTOR_SELECT_INPLACE_FLOAT __runCodelet_popops__BroadcastSelectorSelectInPlace___float
#define BCAST_SELECTOR_SELECT_INPLACE_INT __runCodelet_popops__BroadcastSelectorSelectInPlace___int
#define BCAST_SELECTOR_SELECT_INPLACE_UNSIGNED_INT __runCodelet_popops__BroadcastSelectorSelectInPlace___unsigned_int
#define BCAST_SELECTOR_SELECT_INPLACE_BOOL __runCodelet_popops__BroadcastSelectorSelectInPlace___bool


// Offset in the vertex data is the same for all types of codelets, except
// 'InPlace'
#define VERTEX_IN1_OFFSET        0
#define VERTEX_IN2_OFFSET        1
#define VERTEX_IN3_OFFSET        2
#define VERTEX_OUT_START_OFFSET  3
#define VERTEX_OUT_ROWS_OFFSET   4

// Offset in the vertex data for 'InPlace' variants
#define VERTEX_INPLACE_IN1_OFFSET        0
#define VERTEX_INPLACE_ROWS_OFFSET       1
#define VERTEX_INPLACE_IN2_OFFSET        2
#define VERTEX_INPLACE_IN3_OFFSET        3



//-------------------------------------------------------------
// BroadcastSelect for FLOAT and INT (4 bytes):
// 'in1' and 'in2' are scalars, 'in3' is a vector of vectors

#define in1             m0
#define in2             m1
#define in3             m2
#define out             m3
#define nRows           m4
#define in3_row         m5
#define out_row         m6
#define nElems          m7
#define nElemsRpt       m8
#define selector        m9
#define value           m10

FN_WORKER_ENTRY_POINT BCAST_SELECT_FLOAT 8
FN_EXPORT BCAST_SELECT_INT
FN_EXPORT BCAST_SELECT_UNSIGNED_INT
  // Load the vertex state. Bail out straight away if no rows to do
  ld32 $nRows, $mvertex_base, $mzero, VERTEX_OUT_ROWS_OFFSET
  brz $nRows, word_end
  add $nRows, $nRows, -1  // subtract for brnzdec
  ld32 $in1, $mvertex_base, $mzero, VERTEX_IN1_OFFSET
  ld32 $in2, $mvertex_base, $mzero, VERTEX_IN2_OFFSET
  ld32 $in3, $mvertex_base, $mzero, VERTEX_IN3_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_OUT_START_OFFSET

  // load values for 'in1' and 'in2'
  ld32 $in1, $mzero, $in1, 0
  ld32 $in2, $mzero, $in2, 0

word_row_loop_start:
  // Load in-rows
  ld32step $in3_row, $mzero, $in3+=, 1

  // Load next out pointer and number of row elements
  ld32step $out_row, $mzero, $out+=, 1
  ld32step $nElems, $mzero, $out+=, 1

word_column_loop_start:
  // nElems comes from type SPAN, which can be 32 bit, so we need to
  // make sure we don't exceed loop count for RPT
  min $nElemsRpt, $nElems, CSR_W_REPEAT_COUNT__VALUE__MASK
  rpt $nElemsRpt, ((word_rpt_end - word_rpt_start)/8)-1
word_rpt_start:
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop}
  {mov $value, $in2; fnop}
  {movz $value, $selector, $in1; fnop}
  {st32step $value, $mzero, $out_row+=, 1; fnop}
word_rpt_end:
  sub $nElems, $nElems, $nElemsRpt
  brnz $nElems, word_column_loop_start

  brnzdec $nRows, word_row_loop_start

word_end:
  exitz $mzero

.size BCAST_SELECT_FLOAT, . - BCAST_SELECT_FLOAT

#undef in1
#undef in2
#undef in3
#undef out
#undef nRows
#undef in3_row
#undef out_row
#undef nElems
#undef nElemsRpt
#undef selector
#undef value



//-------------------------------------------------------------
// BroadcastSelect for HALF (2 bytes):
// 'in1' and 'in2' are scalars, 'in3' is a vector of vectors (of bool)
// 'out' is guaranteed only 2-byte aligned (not 4-byte aligned)
#define in1         m0
#define in2         m1
#define in3         m2
#define out         m3
#define in3_row     m4
#define out_row     m5
#define nRows       m6
#define nElems      m7
#define selector    m8
#define values2     m9
#define value       m10
#define scratch     m10
#define trail       m11
#define nElemsRpt   m11

// Offsets in scratch area for saving values (not enough registers!)
#define TRAIL_SAVE  0

FN_WORKER_ENTRY_POINT BCAST_SELECT_HALF 8
  // load vertex state. Bail out straight away if no rows to do
  ld32 $nRows, $mvertex_base, $mzero, VERTEX_OUT_ROWS_OFFSET
  brz $nRows, half_end
  add $nRows, $nRows, -1  // subtract for brnzdec
  ld32 $in1, $mvertex_base, $mzero, VERTEX_IN1_OFFSET
  ld32 $in2, $mvertex_base, $mzero, VERTEX_IN2_OFFSET
  ld32 $in3, $mvertex_base, $mzero, VERTEX_IN3_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_OUT_START_OFFSET

  // load values for 'in1' and 'in2'
  ldz16 $in1, $mzero, $in1, 0
  ldz16 $in2, $mzero, $in2, 0

half_row_loop_start:
  // Load pointer for in3 row
  ld32step $in3_row, $mzero, $in3+=, 1

  // Load next out pointer and number of elems in row
  ld32step $out_row, $mzero, $out+=, 1
  ld32step $nElems, $mzero, $out+=, 1

  brz $nElems, half_load_next_row  // if row is empty, go to next one

  // If 'out_row' is not 4-byte aligned, process the first element, so that
  // in the main loop we can work on 4-byte aligned (2 halves at a time)
  // We need to read the previous half word (whatever is in memory) and write
  // it back together with the new 'out' value.
  and $scratch, $out_row, 3
  brz $scratch, 1f
  add $out_row, $out_row, -2
  ldz16 $values2, $mzero, $out_row, 0  // read prev half word

  mov $value, $in2                          // Put in 'value' the correct
  ldz8step $selector, $mzero, $in3_row+=, 1 // value, based on the boolean
  movz $value, $selector, $in1              // read from in3_row[]

  sort4x16lo $values2, $values2, $value  // mix prev half word and 'value'
  st32step $values2, $mzero, $out_row+=, 1
  add $nElems, $nElems, -1
1:
  // Do we have to do a 'trail' element (i.e. is nElems odd now?)
  and $trail, $nElems, 1 // Number of elements mod 2
  st32 $trail, $mworker_base, $mzero, TRAIL_SAVE

  // Main loop, processing two half words per cycle
  shr $nElems, $nElems, 1
half_column_loop_start:
  // make sure we don't exceed loop count for RPT
  min $nElemsRpt, $nElems, CSR_W_REPEAT_COUNT__VALUE__MASK
  rpt $nElemsRpt, ((half_rpt_end - half_rpt_start)/8)-1
half_rpt_start:
  {mov $values2, $in2; fnop}                        // Put in 'values2' the correct
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop} // value, based on the boolean
  {movz $values2, $selector, $in1; fnop}            // read from in3_row[]

  {mov $value, $in2; fnop}                          // Put in 'value' the correct
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop} // value, based on the boolean
  {movz $value, $selector, $in1; fnop}              // read from in3_row[]

  {sort4x16lo $values2, $values2, $value; fnop}     // mix value and values2
  {st32step $values2, $mzero, $out_row+=, 1; fnop}
half_rpt_end:
  sub $nElems, $nElems, $nElemsRpt
  brnz $nElems, half_column_loop_start

  // If there is a 'trail' element we process it.
  // We need to read the half word in memory following where we will write the
  // 'out' value so that we can write back the two together.
  ld32 $trail, $mworker_base, $mzero, TRAIL_SAVE
  brz $trail, half_load_next_row
  ldz16 $values2, $mzero, $out_row, 1  // read the following half word

  mov $value, $in2                    // Put in 'value' the correct
  ldz8 $selector, $mzero, $in3_row, 0 // value, based on the boolean
  movz $value, $selector, $in1        // read from in3_row[]

  sort4x16lo $values2, $value, $values2  // mix 'value' with half from mem
  st32 $values2, $mzero, $out_row, 0

half_load_next_row:
  brnzdec $nRows, half_row_loop_start

half_end:
  exitz $mzero

.size BCAST_SELECT_HALF, . - BCAST_SELECT_HALF

#undef in1
#undef in2
#undef in3
#undef out
#undef in3_row
#undef out_row
#undef nRows
#undef nElems
#undef selector
#undef values2
#undef value
#undef scratch
#undef trail
#undef nElemsRpt
#undef TRAIL_SAVE




//-------------------------------------------------------------
// BroadcastSelect for BOOLEAN (1 byte):
// 'in1' and 'in2' are scalars, 'in3' is a vector of vectors
//
// 'out' can have any alignment and length. This complicates the code as we
// need to take into account leading bytes (before the first word aligned byte)
// and trailing as well.

#define in1                  m0
#define in2                  m1
#define in3                  m2
#define out                  m3
#define in3_row              m4
#define out_row              m5
#define nRows                m6
#define nElems               m7
#define selector             m8
#define memSave              m8
#define values4              m9
#define value                m10
#define scratch              m10
#define leadingOffs          m10
#define nLeading             m10
#define nTrail               m11
#define nElemsRpt            m11
#define leftovers            m11
#define leadingBits          m11

// Offsets in scratch area for saving values (not enough registers!)
#define LEFTOVER_SHIFT_SAVE 0
#define LEFTOVER_SAVE       1
#define TRAIL_SAVE          2

FN_WORKER_ENTRY_POINT BCAST_SELECT_BOOL 8 nop
  // load vertex state. Bail out straight away if no rows to do
  ld32 $nRows, $mvertex_base, $mzero, VERTEX_OUT_ROWS_OFFSET
  brz $nRows, bool_end
  add $nRows, $nRows, -1  // subtract for brnzdec
  ld32 $in1, $mvertex_base, $mzero, VERTEX_IN1_OFFSET
  ld32 $in2, $mvertex_base, $mzero, VERTEX_IN2_OFFSET
  ld32 $in3, $mvertex_base, $mzero, VERTEX_IN3_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_OUT_START_OFFSET

  // load values for 'in1' and 'in2'
  ldz8 $in1, $mzero, $in1, 0
  ldz8 $in2, $mzero, $in2, 0

bool_row_loop_start:
  // Load pointer for in3 row
  ld32step $in3_row, $mzero, $in3+=, 1

  // Load next out pointer and number of elems in row
  ld32step $out_row, $mzero, $out+=, 1
  ld32step $nElems, $mzero, $out+=, 1

  brz $nElems, bool_load_next_row  // if row is empty, go to next one

  and $leadingOffs, $out_row, 3  // is row 4 bytes aligned?
  brz $leadingOffs, bool_row_is_aligned
  // If 'out_row' is not 4-byte aligned, process the first (1-2-3) leading
  // element(s), so that in the main loop we can work on 4-byte aligned
  // (4 bytes at a time).
  // We need to read the previous 3-2-1 byte(s) (whatever is in memory) and
  // write them back together with the new 'out' value(s).
  // The extra complication is if the row has only 1 or 2 bytes and is also
  // not aligned, in which case it means the whole row can be 'in the middle'
  // of a word in memory and we need to restore bytes at the end as well.
  //
  // Example:  nElems == 2, leadingOffs == 1:
  //
  // MEMORY ADDR:    1000     1001     1002     1003
  //                   |   ($out_row)    |        |
  //                   |        |        |        |
  //                   +--------+--------+--------+--------+
  //                   |    x   |    A   |    B   |    y   |
  //                   +--------+--------+--------+--------+
  //                     1 byte
  //
  // The whole row is bytes [A, B]. The whole word [x, A, B, y] needs to be
  // read from memory so that we can write back 'x' and 'y'.
  // Note that when read in a register this will be (MSB to LSB): [y,B,A,x]
  // The [x] byte (could be 1, 2 or 3 bytes) is the 'leading' part to write
  // back in any case.
  // The [y] byte (could be two bytes if nElems == 1, leadingOffs == 1) is
  // the part called 'leftovers' below.
  sub $out_row, $out_row, $leadingOffs  // point back to word boundary
  ld32 $values4, $mzero, $out_row, 0    // read whole 4-byte word from memory
  // Do we have to track the last 1-2 bytes of the word from memory
  // ('leftovers')? If yes we save in the scratch area. In any case
  // first write zero in scratch area to mark 'no leftovers'
  sub $leftovers, 4, $leadingOffs
  sub $leftovers, $leftovers, $nElems
  st32 $mzero, $mworker_base, $mzero, LEFTOVER_SHIFT_SAVE
  // compute how many bit shift will be needed in the row bytes to align for
  // leftovers (if any) and save in scratch area
  brneg $leftovers, .Lno_leftovers
  shl  $leftovers, $leftovers, 3  // times 8, convert byte offset to bit offs
  st32 $leftovers, $mworker_base, $mzero, LEFTOVER_SHIFT_SAVE
  // The leftovers are in the MSB part of the register.
  // Zero away everything except the leftovers (shift right and then left)
  // and store that word in scracth area
  sub $leftovers, 32, $leftovers
  shr $memSave, $values4, $leftovers
  shl $memSave, $memSave, $leftovers
  st32 $memSave, $mworker_base, $mzero, LEFTOVER_SAVE
.Lno_leftovers:
  // Prepare $values4 with the previous elements in memory ([x] in the above
  // example) at the most significant end, so that roll8r's in the rpt below
  // work correctly
  sub $nLeading, 4, $leadingOffs
  shl $leadingBits, $nLeading, 3  // times 8, convert bits to bytes
  shl $values4, $values4, $leadingBits
  min $nLeading, $nLeading, $nElems

  sub $nElems, $nElems, $nLeading

  // Accumulate the $nLeading new 'out' values in $values4
  rpt $nLeading, ((2f - 1f)/8)-1
1:
  {mov $value, $in2;fnop}                         // Put in 'value' the correct
  {ldz8step $selector, $mzero, $in3_row+=, 1;fnop}// value, based on the boolean
  {movz $value, $selector, $in1;fnop}             // read from in3_row[]
  {roll8r $values4, $values4, $value; fnop}
2:

  // If we need to restore the leftovers in $values4 word before writing, do it
  ld32 $leftovers, $mworker_base, $mzero, LEFTOVER_SHIFT_SAVE
  brz $leftovers, .Lno_leftovers_restore
  ld32 $memSave, $mworker_base, $mzero, LEFTOVER_SAVE
  shr $values4, $values4, $leftovers
  or $values4, $values4, $memSave
.Lno_leftovers_restore:

  st32step $values4, $mzero, $out_row+=, 1
bool_row_is_aligned:
  // Do we have to do 1-2-3 'trail' element (i.e. nElems not a multiple of 4?)
  and $nTrail, $nElems, 3 // Number of elements mod 4
  st32 $nTrail, $mworker_base, $mzero, TRAIL_SAVE

  // ---  Main loop, processing 4 bytes per cycle
  shr $nElems, $nElems, 2
bool_column_loop_start:
  // make sure we don't exceed loop count for RPT
  min $nElemsRpt, $nElems, CSR_W_REPEAT_COUNT__VALUE__MASK
 {rpt $nElemsRpt, ((bool_rpt_end - bool_rpt_start)/8)-1
  fnop}
bool_rpt_start:
  // Do 4 times:
  // Put in 'value' the correct value, based on the boolean read from in3_row[]
  // Shift it in 'values4' (from the 'left')
  {mov $value, $in2; fnop}
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop}
  {movz $value, $selector, $in1; fnop}
  {roll8r $values4, $values4, $value; fnop}

  {mov $value, $in2; fnop}
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop}
  {movz $value, $selector, $in1; fnop}
  {roll8r $values4, $values4, $value; fnop}

  {mov $value, $in2; fnop}
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop}
  {movz $value, $selector, $in1; fnop}
  {roll8r $values4, $values4, $value; fnop}

  {mov $value, $in2; fnop}
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop}
  {movz $value, $selector, $in1; fnop}
  {roll8r $values4, $values4, $value; fnop}

  {st32step $values4, $mzero, $out_row+=, 1; fnop}
bool_rpt_end:
  sub $nElems, $nElems, $nElemsRpt
  brnz $nElems, bool_column_loop_start

  // If there are 'trail' elements we process them. This is similar to the
  // processing of the leading elements
  ld32 $nTrail, $mworker_base, $mzero, TRAIL_SAVE
  brz $nTrail, bool_load_next_row

  // Accumulate the $nTrail new 'out' values in $values4
 {rpt $nTrail, ((20f - 10f)/8)-1
  fnop}
10:
  {mov $value, $in2; fnop}
  {ldz8step $selector, $mzero, $in3_row+=, 1; fnop}
  {movz $value, $selector, $in1; fnop}
  {roll8r $values4, $values4, $value; fnop}
20:
  ld32 $scratch, $mzero, $out_row, 0  // read whole 4-byte word
  shl $nTrail, $nTrail, 3 // times 8, i.e. $nTrail <= number of bits
  // zero the least significant $nTrail bits by shifting right, then left
  shr $scratch, $scratch, $nTrail
  shl $scratch, $scratch, $nTrail
  // Align $values4 to 32-$nTrail and combine with $scratch
  sub $nTrail, 32, $nTrail
  shr $values4, $values4, $nTrail
  or $values4, $values4, $scratch

  st32 $values4, $mzero, $out_row, 0

bool_load_next_row:
  brnzdec $nRows, bool_row_loop_start

bool_end:
  exitz $mzero

.size BCAST_SELECT_BOOL, . - BCAST_SELECT_BOOL

#undef in1
#undef in2
#undef in3
#undef out
#undef in3_row
#undef out_row
#undef nRows
#undef nElems
#undef selector
#undef memSave
#undef values4
#undef value
#undef scratch
#undef leadingOffs
#undef nLeading
#undef nTrail
#undef nElemsRpt
#undef leftovers
#undef leadingBits
#undef LEFTOVER_SHIFT_SAVE
#undef LEFTOVER_SAVE
#undef TRAIL_SAVE


// -------------------------------------------------------------
// BroadcastSelectorSelect (for all types): 'in1' and 'in2' are vectors of
// vectors, 'in3' is a scalar.
// In this case we just fully copy, row by row, 'in1' or 'in2' (depending on
// the scalar value of 'in3') into 'out'
// For the 'InPlace' variant, 'out' and 'in1' are the same, so if the selector
// 'in3' is true, there is nothing to do.
// For copying we just call '_Longmemcpy_entry' which is a general purpose
// memcpy which might not have the best performance for small sizes but
// gives very small code size here and has very good performance for aligned
// vectors/big sizes.

// These three registers are as requested by _Longmemcpy_entry
#define out_row         m1
#define in_row          m0
#define nElems          m8

#define BYTES_PER_WORD  m3
#define in3             m1
#define nRows           m7
#define in              m2
#define out             m9

// ---- InPlace entry points
DEF_STACK_SIZE_OWN 0 .text.BCAST_SELECTOR_SELECT_INPLACE_FLOAT

FN_WORKER_ENTRY_POINT BCAST_SELECTOR_SELECT_INPLACE_FLOAT 8 "" NONE
FN_EXPORT BCAST_SELECTOR_SELECT_INPLACE_INT
FN_EXPORT BCAST_SELECTOR_SELECT_INPLACE_UNSIGNED_INT
  setzi  $BYTES_PER_WORD, 4
  bri    1f
FN_EXPORT BCAST_SELECTOR_SELECT_INPLACE_BOOL
  setzi  $BYTES_PER_WORD, 1
  bri    1f
FN_EXPORT BCAST_SELECTOR_SELECT_INPLACE_HALF
  setzi  $BYTES_PER_WORD, 2
1:
  // If the selector is TRUE, nothing to do ('in1Out' stays the same) ...
  ld32 $in3, $mvertex_base, $mzero, VERTEX_INPLACE_IN3_OFFSET
  ldz8 $in3, $in3, $mzero, 0
  brnz $in3, .Lbroadcast_selector_end
  // ... else we will copy 'in2' into 'out'
  ld32 $in, $mvertex_base, $mzero, VERTEX_INPLACE_IN2_OFFSET
  ld32 $out, $mvertex_base, $mzero, VERTEX_INPLACE_IN1_OFFSET
  ld32 $nRows, $mvertex_base, $mzero, VERTEX_INPLACE_ROWS_OFFSET
  bri 4f

// --- Non-inplace entry points.
FN_EXPORT BCAST_SELECTOR_SELECT_FLOAT
FN_EXPORT BCAST_SELECTOR_SELECT_INT
FN_EXPORT BCAST_SELECTOR_SELECT_UNSIGNED_INT
  setzi  $BYTES_PER_WORD, 4
  bri    2f
FN_EXPORT BCAST_SELECTOR_SELECT_BOOL
  setzi  $BYTES_PER_WORD, 1
  bri    2f
FN_EXPORT BCAST_SELECTOR_SELECT_HALF
  setzi  $BYTES_PER_WORD, 2
2:
  // Read the base 'in' pointer from 'in1' or 'in2', based on the selector 'in3'
  ld32 $in3, $mvertex_base, $mzero, VERTEX_IN3_OFFSET
  ldz8 $in3, $in3, $mzero, 0
  ld32 $in, $mvertex_base, $mzero, VERTEX_IN1_OFFSET
  brnz $in3, 3f
  ld32 $in, $mvertex_base, $mzero, VERTEX_IN2_OFFSET
3:
  ld32 $out, $mvertex_base, $mzero, VERTEX_OUT_START_OFFSET
  ld32 $nRows, $mvertex_base, $mzero, VERTEX_OUT_ROWS_OFFSET

4:
  // if no rows, terminate; else decrement row counter for brnzdec
  brz $nRows, .Lbroadcast_selector_end
  add $nRows, $nRows, -1
.Lrow_loop:
  // starting pointers to input and output rows
  ld32step $in_row, $mzero, $in+=, 1
  ld32step $out_row, $mzero, $out+=, 1
  // load number of elements in this row
  ld32step $nElems, $mzero, $out+=, 1

  // save in scratch area this function's state to preserve them when
  // calling _Longmemcpy_entry
  st32 $in, $mworker_base, $mzero, 0
  st32 $out, $mworker_base, $mzero, 1
  st32 $BYTES_PER_WORD, $mworker_base, $mzero, 2
  st32 $nRows, $mworker_base, $mzero, 3

  // Now just copy nElems, each of BYTES_PER_WORD length, from in to out
  mul $nElems, $nElems, $BYTES_PER_WORD
  call $lr, _Longmemcpy_entry

  // restore
  ld32 $nRows, $mworker_base, $mzero, 3
  ld32 $BYTES_PER_WORD, $mworker_base, $mzero, 2
  ld32 $out, $mworker_base, $mzero, 1
  ld32 $in, $mworker_base, $mzero, 0

  brnzdec $nRows, .Lrow_loop
.Lbroadcast_selector_end:
  exitz $mzero

.size BCAST_SELECTOR_SELECT_HALF, . - BCAST_SELECTOR_SELECT_HALF

#undef out_row
#undef in_row
#undef nElems

#undef BYTES_PER_WORD
#undef in3
#undef nRows
#undef in
#undef out


#endif
