// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
/* -------------------------------------------------------------------------- */
//
// Here we define the transpose vertices:
//
//    Transpose1DSingleWorker  (1D, single worker)    half, short, ushort
//
//    Transpose1D              (1D, MultiWorker)      half, short, ushort
//
//    Transpose2D              (2D, single worker)    half, short, ushort
//                                                    float, int, uint
//
// Transposing assembler overview:
//
// Each input is a series of vectors, each of the same size.
// The start of each is 8 byte aligned.
// Each vector is to be treated as a matrix with size (rows,columns).
//
// The above means that for certain cases we can use 64 bit loads/stores
// when processing the whole of a vector if the number of rows and columns are a
// multiple of 2 (floats) or 4(halves).
// Alternatively we can read individual values (down a column)
// and store pairs of values (across a row).  This is not as fast but works
// for other sized arrays.
/* -------------------------------------------------------------------------- */
#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "CommonPoplibsMacros.h.S"
// Register aliases

#define MATRIX_LOOP_COUNT   m10

// These may be used as a pair and hence should be kept as a $m pair
// Note: Needed if we want to use a dummy multi-load instruction to increment
//       pointers
#define IN_START_PTR        m2
#define OUT_PTR             m3

#define IN_ROWS             m5
#define IN_COLUMNS          m4
#define STRIDE              m0

#define mSCRATCH3   m8
#define mSCRATCH2   m9
#define mSCRATCH4   m6
#define INOUT_PTR   m6:7
#define INOUT_PTR2  m8:9

#define LOOP_COUNT m11
#define mSCRATCH   m1

#define VAL12 a0:1
#define VAL1  a0
#define VAL2  a1
#define VAL34 a2:3
#define VAL3  a2
#define VAL4  a3
#define VAL56 a4:5


#define SHORT_SPAN_WORKLIST_SIZE_BITS 12
#define SHORT_SPAN_ADDR_SIZE_BITS     (32 - 12)

//****************************************************************************
// The input structure is always the same so a macro can be used to fetch the
// parameters:
// input start (pointer to an array of pointers)
// input end   (pointer to the end of the array of pointers)
// output start (pointer to an array of pointers)
// Input rows   (integer - row count)
// Input columns (integer - column count)
//****************************************************************************
#define VOFF_IN_START_PTR_2D   0  // words
#define VOFF_IN_SIZE_2D        1  // words
#define VOFF_OUT_PTR_2D        2  // words
#define VOFF_SRC_ROWS_2D       6  // half
#define VOFF_SRC_COLUMNS_2D    7  // half

.macro GET_PARAMS_2D p1 p2 p3 p4 p5
    ld32     $IN_START_PTR, $mzero, $mvertex_base, \p1
    ld32     $mSCRATCH4,    $mzero, $mvertex_base, \p2
    ld32     $OUT_PTR,      $mzero, $mvertex_base, \p3
    ldz16    $IN_ROWS,      $mzero, $mvertex_base, \p4
    ldz16    $IN_COLUMNS,   $mzero, $mvertex_base, \p5
.endm


//********************************************************************
//A slower option, where alignment is not assumed thoughout the whole
//vector/matrix.
//
// The approach is to read single values and write pairs of values.
// This means we can do 64 bit writes when processing floats and avoid
// read-modify write of memory when processing halves.  (Until the last item
// of an array with an odd number of total values).
// Written as a parameterised macro to avoid repetition in creating float,half
// variants.
// Input registers:
// $IN_ROWS
// $IN_COLUMNS
// $MATRIX_LOOP_COUNT
// $IN_START_PTR
// $OUT_PTR

.macro TRANSPOSE_SLOWER ldinstr st2instr st2operand wordinc half
//Force alignment to 64 bit boundary here so the loop body is aligned.
//putting it here means (if it inserted a nop) it only gets executed once,
//not every loop pass.
 .align 8
    // check if we have an odd or even number of rows- different loop in each case
    // $STRIDE is repurposed as a flag in the loop body below
    and    $STRIDE, $IN_ROWS, 1
    shr    $IN_ROWS, $IN_ROWS, 1

.Los_matrix_loop\@:
    ld32step  $mSCRATCH, $mzero, $IN_START_PTR+=, 1
    ld32step  $mSCRATCH2, $mzero, $OUT_PTR+=, 1
    add       $LOOP_COUNT, $IN_COLUMNS, -1

.Los_loop_start\@:
    // point back to the column start,  next input column
    mov       $mSCRATCH3, $mSCRATCH
    add       $mSCRATCH, $mSCRATCH, \wordinc

    // 1st inner loop - for output rows that are aligned to 2 x(input size)
    // Also the only inner loop if there were an even number of rows
    // equivalent to   rpt      $IN_ROWS, ((2f-1f)/8)-1 but no
    // loop count size limitation
    mov       $mSCRATCH4, $IN_ROWS
    bri       2f
1:
    \ldinstr $VAL1, $mzero, $mSCRATCH3+=, $IN_COLUMNS
    \ldinstr $VAL2, $mzero, $mSCRATCH3+=, $IN_COLUMNS
.if \half
     roll16 $VAL1, $VAL1, $VAL2          //Combine 2 halves
.endif
     \st2instr \st2operand, $mzero, $mSCRATCH2+=, 1
2:
    brnzdec     $mSCRATCH4,1b

    // To save code space, deal with an even number of rows by conditionally
    // executing the end of a loop body for that case here
    brnz        $STRIDE,3f
    brnzdec     $LOOP_COUNT, .Los_loop_start\@
    brnzdec     $MATRIX_LOOP_COUNT, .Los_matrix_loop\@
    exitz       $mzero
3:

   // There will be 1 item left as we have an odd number of rows
   \ldinstr  $VAL1, $mzero, $mSCRATCH3+=, $IN_COLUMNS


    // Decrement the loop count in the middle of the loop so we exit when there
    // are an odd number of columns
    brnzdec  $LOOP_COUNT, 1f
    bri      .Los_loop_exit\@
1:
    // point back to the column start, next input column
    mov     $mSCRATCH3, $mSCRATCH
    add     $mSCRATCH, $mSCRATCH, \wordinc

    // 2nd inner loop - for output rows that are not aligned to 2 x(input size)
    // equivalent to rpt    $IN_ROWS, ((2f-1f)/8)-1 but no loop count size
    // limitation
    mov       $mSCRATCH4, $IN_ROWS
    bri       2f
1:
    \ldinstr $VAL2, $mzero, $mSCRATCH3+=, $IN_COLUMNS
.if \half
     roll16 $VAL1, $VAL1, $VAL2           //Combine 2 halves
.endif
    \st2instr \st2operand, $mzero, $mSCRATCH2+=, 1
    \ldinstr $VAL1, $mzero, $mSCRATCH3+=, $IN_COLUMNS
2:
    brnzdec     $mSCRATCH4,1b

    \ldinstr  $VAL2, $mzero, $mSCRATCH3+=, $IN_COLUMNS
.if \half
     roll16 $VAL1, $VAL1, $VAL2           //Combine 2 halves
.endif
    \st2instr   \st2operand, $mzero, $mSCRATCH2+=, 1

    brnzdec     $LOOP_COUNT, .Los_loop_start\@

    brnzdec     $MATRIX_LOOP_COUNT, .Los_matrix_loop\@
    exitz $mzero

    // Break -in the case of odd rows, odd columns - store the last one
.Los_loop_exit\@:
.if \half
    // Case where we do need to read-modify write to write the (16bit) last word
    ldb16       $VAL2, $mzero, $mSCRATCH2, 1
    roll16      $VAL1, $VAL1, $VAL2
    st32        $VAL1, $mzero, $mSCRATCH2, 0
.else
    // float - just store the last single word
    st32        $VAL1, $mzero, $mSCRATCH2, 0
.endif
    brnzdec     $MATRIX_LOOP_COUNT, .Los_matrix_loop\@
    exitz $mzero
.endm

//******************************************************************************
// Float transpose
// Take advantage of 64 bit read and write by transposing 2x2 squares.
// Need to be aligned to 64 bit boundaries and have an even number of
// rows and columns.
// Due to lack of a "swap" instruction it takes 3 cycles for 4 values: 0.75
// cycles per value.  If the swap could be done in 1 cycle this could be 0.5.
// Alternatively the inner loop could process more data but at the loss
// of flexibility
//******************************************************************************

#define TRANSPOSE_2D_FLOAT \
  __runCodelet_popops__Transpose2D___float
#define TRANSPOSE_2D_UNSIGNED_INT \
  __runCodelet_popops__Transpose2D___unsigned_int
#define TRANSPOSE_2D_INT \
  __runCodelet_popops__Transpose2D___int

FN_WORKER_ENTRY_POINT TRANSPOSE_2D_FLOAT 8
FN_EXPORT TRANSPOSE_2D_UNSIGNED_INT
FN_EXPORT TRANSPOSE_2D_INT

     GET_PARAMS_2D  VOFF_IN_START_PTR_2D VOFF_IN_SIZE_2D VOFF_OUT_PTR_2D VOFF_SRC_ROWS_2D VOFF_SRC_COLUMNS_2D

    add      $MATRIX_LOOP_COUNT, $mSCRATCH4, -1

    // The fast algorithm works for even rows, even columns only - test & branch
    // if we can't use it.
    or       $mSCRATCH, $IN_ROWS, $IN_COLUMNS
    and      $mSCRATCH, $mSCRATCH, 1
    brnz     $mSCRATCH, TransposeAT4SlowPath

    // As all matrices to process have the same dimensions, calculate parameters
    // that don't vary with input/output address:

    // loop count for pairs of input columns: constant throughout and
    // unaffected by rpt
    shr   $LOOP_COUNT, $IN_COLUMNS, 1
    add   $LOOP_COUNT, $LOOP_COUNT, -1

   //check that the repeat count is within range of its register value
#if (CSR_W_REPEAT_COUNT__VALUE__MASK < 0xFFFF)
    // Max value of LOOP_COUNT is 16 bit, this check must be here for
    // CSR_W_REPEAT_COUNT__VALUE__MASK < 16 bit. In Mk1, it is 12 bits (0x0FFF)
    cmpult    $STRIDE, $LOOP_COUNT, CSR_W_REPEAT_COUNT__VALUE__MASK + 1
    brz       $STRIDE, TransposeAT4SlowPath
#endif

    // calculate a write stride for the end of the inner loop=
    // 1-((cols-2)* rows)/2
    // Keeping only 10 bits of the (negative) result so we can pack it in
    add    $STRIDE, $IN_COLUMNS, -2
    mul    $STRIDE, $STRIDE, $IN_ROWS
    shr    $STRIDE, $STRIDE, 1
    sub    $STRIDE, 1, $STRIDE

    // This should be the largest stride calculated so check it:
    // Is this negative stride going to exceed a signed 10 bit range?
    cmpslt  $mSCRATCH2, $STRIDE, -512
    brnz    $mSCRATCH2, TransposeAT4SlowPath
    // Rows are used as a positive stride that must be within the same 10-bit
    // signed range, so check these too.
    cmpult  $mSCRATCH2, $IN_ROWS, 512
    brz     $mSCRATCH2, TransposeAT4SlowPath

    //Mask off so we can pack it
    and    $STRIDE, $STRIDE, 0x3ff

    // calculate a read stride for the end of the inner loop
    // =number of input pairs/column, read stride stored in bit field<<10
    shl    $mSCRATCH2, $IN_COLUMNS, (10-1)
    // combine the 2 strides to use after the inner loop
    or     $STRIDE, $STRIDE, $mSCRATCH2
    // columns becomes 4x columns, as that's all it's used for from now on
    shl    $IN_COLUMNS, $IN_COLUMNS, 2

    // Loop per matrix processed
.Lmatrix_loop_at4:
    // Get input address and calculate address of start of 2nd input row
    ld32step  $mSCRATCH3, $mzero, $IN_START_PTR+=, 1
    add       $mSCRATCH2, $IN_COLUMNS, $mSCRATCH3
    // Get output address and calculate address of start of 2nd output row
    ld32step  $mSCRATCH, $mzero, $OUT_PTR+=, 1

    tapack    $INOUT_PTR, $mSCRATCH3, $mzero, $mSCRATCH
    shl       $mSCRATCH3, $IN_ROWS, 2
    add       $mSCRATCH3, $mSCRATCH3, $mSCRATCH

    // Now $INOUT_PTR and $INOUT_PTR2 are 2 pairs of packed addresses
    // to use in outer and inner loops below as addresses to process a whole
    // vector
    tapack  $INOUT_PTR2, $mSCRATCH2, $mzero, $mSCRATCH3

    // loop count for pairs of input rows
    shr   $mSCRATCH, $IN_ROWS, 1
    add   $mSCRATCH, $mSCRATCH, -1

    // oloop - process all pairs of input rows over the whole matrix
    // Put the last 2 load/stores at the beginning of the loop, but skip
    // them the first time around.  This avoid overreads on the last loop pass
    bri      .Loloop_first_pass

.Loloop_start:
    // Store the last 2 results (From the end of the loop),
    // use the strides computed above to adjust the pointers for the next
    // pair of input rows, output columns
    {ldst64pace $VAL12, $VAL56, $INOUT_PTR+=, $STRIDE, 6
     sort4x32hi $VAL34, $VAL12, $VAL34}
    ldst64pace $VAL34, $VAL34, $INOUT_PTR2+=, $STRIDE, 6
.Loloop_first_pass:
    // load first 2 input pairs, dummy store. Could possibly optimise
    // by combining with the 2 stores/dummy loads at the end of the loop,
    // but doing so would reduce flexibility.
    ldst64pace $VAL12, $VAL56, $INOUT_PTR+=, $mzero, 4   //out=no inc,  in=+=2xfloat
    ldst64pace $VAL34, $VAL56, $INOUT_PTR2+=, $mzero, 4

    // iloop - each pass processes a 2x2 matrix and transposes/stores it.
    // the loop continues, processing 2 input row-> 2 output columns
    // Potential to optimise more but only by processing 2 sets of 2x2 floats per
    // loop pass at the expense of flexibility.

    { rpt        $LOOP_COUNT, ((2f-1f)/8)-1
      sort4x32lo $VAL56, $VAL12, $VAL34}
1:
    { ldst64pace  $VAL12, $VAL56, $INOUT_PTR+=, $IN_ROWS, 4 //in+=2xfloat, out+=2 rows
      sort4x32hi $VAL34, $VAL12, $VAL34 }
    { ldst64pace $VAL34, $VAL34, $INOUT_PTR2+=, $IN_ROWS, 4
      fnop}
    { nop
      sort4x32lo $VAL56, $VAL12, $VAL34}
2:

    brnzdec     $mSCRATCH, .Loloop_start

    // Avoid over read - making sure we store, with no load
    { st64pace   $VAL56, $INOUT_PTR+=, $STRIDE, 1
      sort4x32hi $VAL34, $VAL12, $VAL34}
    st64pace $VAL34, $INOUT_PTR2+=, $STRIDE, 1


   brnzdec    $MATRIX_LOOP_COUNT, .Lmatrix_loop_at4

   exitz   $mzero
//*****************************************************************************
// Slower version for non-multiples of 2 rows,columns - uses the macro above
TransposeAT4SlowPath:
    TRANSPOSE_SLOWER ld32step st64step $VAL12 4 0

FN_SIZE TRANSPOSE_2D_FLOAT

//*****************************************************************************
// Half transpose:
//
// Transpose 4x4 squares and store, to take advantage of the 64 bit load/stores.
// Uses the AACC registers as a pipleine of 16bit values to do the transposition
// of the 4x4 squares.  A similar principle to both the transpose16x16.S
// microbenchmark and the "f16v4stacc example" in the processor manual.
// However this is made more flexible to cope with sizes and shapes of matrix
// other than 16 x 16.
//******************************************************************************

// Register aliases

// Input pointer
#define LD_PTR m6
//Output pointer
#define  ST_PTR m7

// Link register used for call to common code
#define LINK_REG m1

//Input/output pointers are packed into a register pair
#define TRI_PTR m6:7

#define STRIDES  m0

//Definitions to select which stride field to increment by
//w,x,y,z referenced below
#define INC_LD1_STw  0x4
#define INC_LDMz_STw 0x7
#define INC_LDx_STw  0x6
#define OUT_GRP_INC_LDx_STw  0xd
#define INC_LDx_STM0 0xd
//#define INC_LDx_STMz 0xe

#define COUNTER m4

#define IN_GROUPS  m8
#define OUT_GROUPS m9
#define IN_REWIND  m10

// Common code for fast half
FN_SECTION TransposeAT2FastPath4ColumnSetup
//*******************************
// Setup for the nx4 matrix cases
TransposeAT2FastPath4ColumnSetup:
  add       $LOOP_COUNT, $OUT_GROUPS, -2
  // In the 4 columns case there is a stride needed for the output
  // -3 x (output columns/4 -1)  equivalent to -3*(output columns/4) +1
  mul       $IN_REWIND, $OUT_GROUPS, -3
  add       $IN_REWIND, $IN_REWIND, 1
  // start to build the stride fields, this field differs between the 2 methods
  shl       $STRIDES, $IN_GROUPS, 10
  // Out groups is repurposed for more choice in strides (st_ptr is unused here)
  shl       $ST_PTR, $IN_REWIND, 20
  bri       .LTransposeAT2FastPathSetupCommon

//*******************************
// Setup for the nxm matrix cases where m > 4
TransposeAT2FastPathSetup:
  add       $LOOP_COUNT, $IN_GROUPS, -2
  // find input rewind stride = -3*input_columns/4 +1
  mul       $IN_REWIND, $IN_GROUPS, -3
  add       $IN_REWIND, $IN_REWIND, 1
 // start to build the stride fields, this field differs between the 2 methods
  shl       $STRIDES, $IN_REWIND, 10
  // Out groups is repurposed for more choice in strides (st_ptr is unused here)
  shl       $ST_PTR, $OUT_GROUPS, 20

.LTransposeAT2FastPathSetupCommon:
  // Combine to form stride fields
  // 4 columns  $STRIDES = 1 << 20         | (1== in_groups) << 10 | out_groups
  // >4 columns $STRIDES = in_rewind << 20 | in_groups << 10       | out_groups
  or        $STRIDES, $STRIDES, $IN_GROUPS
  shl       $STRIDES, $STRIDES, 10
  or        $STRIDES, $STRIDES, $OUT_GROUPS
  // Out groups is repurposed for more choice in strides
  // 4 columns:  out_rewind << 20 | (unused) | (1 == IN_GROUPS)
  // >4 columns: out_groups << 20 | (unused) | IN_GROUPS
  or        $OUT_GROUPS, $ST_PTR, $IN_GROUPS
  br        $LINK_REG
//****************************************************
// Common code for all half, multiple of 4 rows/ 4 columns matrices.
// Detect 4x4 matrix as a special case, uses configurable
// strides and loop counts to process different shaped matrices.
// The outer loop is not used in the case of nx4 matrices
.align 8
TransposeAT2FastPathCommon:
    // Warmup - preload data to feed into AACC
    ld64step $a0:1, $m15, $LD_PTR+=, $IN_GROUPS
    ld64step $a2:3, $m15, $LD_PTR+=, $IN_GROUPS
    ld64step $a4:5, $m15, $LD_PTR+=, $IN_GROUPS

    // Continue to read, feed into AACC but there is no output yet
    // In rewind here, and as referenced in INC_LDMz below
    // results in moving the input ptr back to the next block of 4x4 on the same row
    { ld64step $a6:7, $m15, $LD_PTR+=, $IN_REWIND
      f16v4istacc $a14:15, $a0:1, $a2:3, TISTACC_P0 }
    brpos   $LOOP_COUNT, 1f
    // Special case for a single 4x4 matrix, detected if the loop count is <0
    // From now on use packed addresses to reuse common exit code
    {tapack      $TRI_PTR, $LD_PTR, $ST_PTR, $ST_PTR
     f16v4istacc $a14:15, $a4:5, $a6:7, TISTACC_P1}
    {bri         .Lat2_4_last_writes
     f16v4stacc  $a6:7, TSTACC_P0}

1:
  { ld64step $a0:1, $m15, $LD_PTR+=, $IN_GROUPS
    f16v4istacc $a14:15, $a4:5, $a6:7, TISTACC_P1 }
  { ld64step $a2:3, $m15, $LD_PTR+=, $IN_GROUPS
    f16v4stacc  $a6:7, TSTACC_P0 }
    // Outer loop to deal with an input slice, 4 rows x all columns

    bri  .Louter_loop_first_pass
.Louter_loop_start:
   // This is the end of the loop body, which stores and loads.  It
   // would read 6x64 bits from 4 non-existant input rows so put loop ending here
   // and skip it the 1st time around
  { ldst64pace $a0:1, $a2:3, $TRI_PTR+=, $STRIDES, INC_LDx_STw
   f16v4istacc $a2:3, $a4:5, $a6:7, TISTACC_P1 }
  { ldst64pace $a2:3, $a2:3, $TRI_PTR+=, $STRIDES, INC_LDx_STw
   f16v4stacc  $a6:7, TSTACC_P0 }

  { ldst64pace $a4:5, $a6:7, $TRI_PTR+=, $STRIDES, INC_LDx_STw
   f16v4stacc  $a6:7, TSTACC_P1 }
  { ldst64pace $a6:7, $a6:7, $TRI_PTR+=, $STRIDES,  INC_LDMz_STw
   f16v4istacc $a2:3, $a0:1, $a2:3, TISTACC_P0 }
  { ldst64pace $a0:1, $a2:3, $TRI_PTR+=, $STRIDES, INC_LDx_STw
   f16v4istacc $a2:3, $a4:5, $a6:7, TISTACC_P1 }
  // The last store will modify the store pointer and point back to the
  // next group of 4 output columns, could be done with a
  // stride but that limits the size of array to around 40 x 48
  { ldst64pace $a2:3, $a2:3, $TRI_PTR+=, $IN_GROUPS, INC_LDx_STM0
   f16v4stacc  $a6:7, TSTACC_P0 }

  // Stride the unmodified ST_PTR, (stored in TRI_PTR) by 4 halves, to
  // point at the top of the next group of 4 columns.  Re-pack but
  // twice so we have a copy to use for the next group of 4 columns.
  add    $ST_PTR, $ST_PTR, 4*2

.Louter_loop_first_pass:
  // From now on we'll use load/store so pack into tri ptr
  // The first reference to store pointer is to store an copy
  // of the pointer which won't get modified by the stores ldst64pace
  // instructions.  However it will still be available in ST_PTR (as
  // LD_PTR, ST_PTR overlay with TRI_PTR) to modify manually below to
  // stride to the next 4 column group
  tapack $TRI_PTR, $LD_PTR, $ST_PTR, $ST_PTR
  rpt $LOOP_COUNT, ((2f - 1f) / 8) - 1
1:
  { ldst64pace $a4:5, $a6:7, $TRI_PTR+=, $STRIDES, INC_LDx_STw
    f16v4stacc  $a6:7, TSTACC_P1 }
  { ldst64pace $a6:7, $a6:7, $TRI_PTR+=, $STRIDES,  INC_LDMz_STw
    f16v4istacc $a2:3, $a0:1, $a2:3, TISTACC_P0 }
  { ldst64pace $a0:1, $a2:3, $TRI_PTR+=, $STRIDES, INC_LDx_STw
    f16v4istacc $a2:3, $a4:5, $a6:7, TISTACC_P1 }
  { ldst64pace $a2:3, $a2:3, $TRI_PTR+=, $OUT_GROUPS, OUT_GRP_INC_LDx_STw
    f16v4stacc  $a6:7, TSTACC_P0 }
2:

  { ldst64pace $a4:5, $a6:7, $TRI_PTR+=, $STRIDES, INC_LDx_STw
   f16v4stacc  $a6:7, TSTACC_P1 }
  // Different increment of the load pointer to just move to the next row
  { ldst64pace $a6:7, $a6:7, $TRI_PTR+=, $STRIDES,  INC_LD1_STw
   f16v4istacc $a2:3, $a0:1, $a2:3, TISTACC_P0 }

   brnzdec $COUNTER, .Louter_loop_start

   // Just stores to avoid a lot of overreading, at the end of a matrix

  { st64pace  $a2:3, $TRI_PTR+=, $STRIDES, 1
   f16v4istacc $a2:3, $a4:5, $a6:7, TISTACC_P1 }

  { st64pace  $a2:3, $TRI_PTR+=, $OUT_GROUPS, 3
   f16v4stacc  $a6:7, TSTACC_P0 }
.Lat2_4_last_writes:
  { st64pace  $a6:7, $TRI_PTR+=, $STRIDES, 1
   f16v4stacc  $a6:7, TSTACC_P1 }
  { st64pace  $a6:7, $TRI_PTR+=, $STRIDES, 1
   f16v4istacc $a2:3, $a0:1, $a2:3, TISTACC_P0 }
  { st64pace  $a2:3, $TRI_PTR+=, $STRIDES, 1
   f16v4istacc $a2:3, $a4:5, $a6:7, TISTACC_P1 }
   st64pace  $a2:3, $TRI_PTR+=, $STRIDES, 1
   br        $LINK_REG

FN_SIZE TransposeAT2FastPath4ColumnSetup

/*----------------------------------------------------------------------------*/

// Vertex state for the HALF, fast, version
// All offsets in bytes.
// The vertex state has the same fields if the workers is called directly or
// if it is a MultiVertex, except for the NUM_MATRICES and WORKER_COUNT fields:
//
// -) When the worker is called directly.'NUM_MATRICES' will be: (number of
//    matrices to transpose minus 1). WORKER_COUNT is not used
//
// -) When the worker is a MultiVertex:
//    The first WORKER_COUNT workers will transpose NUM_MATRICES matrices,
//    The second (6-WORKER_COUNT) will transpose (NUM_MATRICES-1) matrices.
//    Note that (6-WORKER_COUNT) and/or (NUM_MATRICES-1) could be zero.
#if defined(VECTOR_AVAIL_SCALED_PTR64)
#define VOFF_IN_PTR_1D             0     // Scaled 64 source pointer
#define VOFF_OUT_PTR_1D            2     // Scaled 64 dst pointer
#define VOFF_SRC_ROWS_1D           4     // Number of rows divided by 4
#define VOFF_SRC_COLUMNS_1D        6     // Number of columns divided by 4
#define VOFF_NUM_MATRICES_1D       8     // Number of matrices (or num - 1)
#define VOFF_WORKER_COUNT_1D      10     // Only when MultiVertex

#else
#define VOFF_IN_PTR_1D             0     // one_ptr source pointer
#define VOFF_OUT_PTR_1D            4     // one_ptr dst pointer
#define VOFF_SRC_ROWS_1D           8     // Number of rows divided by 4
#define VOFF_SRC_COLUMNS_1D       10     // Number of columns divided by 4
#define VOFF_NUM_MATRICES_1D      12     // Number of matrices (or num - 1)
#define VOFF_WORKER_COUNT_1D      14     // Only when MultiVertex
#endif




//----------------------------------------------------------------------------
// Entry point for MultiVertex code where multiple transpositions are 
// processed and work split is done at runtime.
// This computes the offsets from IN and OUT data ptrs for this worker and
// (if needed) decrements NUM_MATRICES.
#define TRANSPOSE_1D_HALF            __runCodelet_popops__Transpose1D___half
#define TRANSPOSE_1D_UNSIGNED_SHORT  __runCodelet_popops__Transpose1D___unsigned_short
#define TRANSPOSE_1D_SHORT           __runCodelet_popops__Transpose1D___short


FN_WORKER_ENTRY_POINT TRANSPOSE_1D_HALF
FN_EXPORT TRANSPOSE_1D_UNSIGNED_SHORT
FN_EXPORT TRANSPOSE_1D_SHORT
#define PTR_INCR        m1
#define NUM_MATRICES    m5
#define WORKER_ID       m7
#define WORKER_COUNT    m6
#define OFFS            m0
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16     $IN_START_PTR, $mzero, $mvertex_base, VOFF_IN_PTR_1D / 2
  ldz16     $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR_1D / 2
#else
  ld32      $IN_START_PTR, $mzero, $mvertex_base, VOFF_IN_PTR_1D / 4
  ld32      $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR_1D / 4
#endif
  ldz16     $OUT_GROUPS, $mzero, $mvertex_base, VOFF_SRC_ROWS_1D / 2
  ldz16     $IN_GROUPS, $mzero, $mvertex_base, VOFF_SRC_COLUMNS_1D / 2
  ldz16     $NUM_MATRICES, $mzero, $mvertex_base, VOFF_NUM_MATRICES_1D / 2
  ldz16     $WORKER_COUNT, $mzero, $mvertex_base, VOFF_WORKER_COUNT_1D / 2

  // Get worker ID
  get       $WORKER_ID, $WSR
  and       $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  mul       $OFFS, $WORKER_ID, $NUM_MATRICES

  // If ID < WORKER_COUNT, NUM_MATRICES and OFFS are already ok.
  // WORKER_ID is 'rebased' on WORKER_COUNT
  sub        $WORKER_ID, $WORKER_ID, $WORKER_COUNT
  brneg      $WORKER_ID, 1f

  // If ID >= WORKER_COUNT, re-compute NUM_MATRICES, OFFS
  // If NUM_MATRICES==1, exit (we have to do NUM_MATRICES-1, i.e. 0, transposes)
  add       $NUM_MATRICES, $NUM_MATRICES, -1
  brz       $NUM_MATRICES, TransposeAT2Exit
  sub       $OFFS, $OFFS, $WORKER_ID
1:
  // OFFS is now in units of number of matrices. We want it in units of double
  // words (because IN_START_PTR, OUT_PTR are PTR64, so shifted right by 3 bits)
  // So we need to multiply OFFS by:
  mul       $PTR_INCR, $OUT_GROUPS, $IN_GROUPS
#if defined(VECTOR_AVAIL_SCALED_PTR64)
  //     ((OUT_GROUPS x 4) x (IN_GROUPS x 4) x sizeof(half)) / 8 =
  //     OUT_GROUPS x IN_GROUPS x 4
  shl       $PTR_INCR, $PTR_INCR, 2   // multiply by 4
  mul       $OFFS, $OFFS, $PTR_INCR
  add       $IN_START_PTR, $IN_START_PTR, $OFFS
  add       $OUT_PTR, $OUT_PTR, $OFFS
  // shift up because rows and columns are shifted down by 2. Additional shift
  // because data is half
  shl       $PTR_INCR, $PTR_INCR, (2 + 1)
#else
  //     ((OUT_GROUPS x 4) x (IN_GROUPS x 4) x sizeof(half))
  shl       $PTR_INCR, $PTR_INCR, 5   // multiply by 32 (4 * 4 * 2)
  mul       $OFFS, $OFFS, $PTR_INCR
  add       $IN_START_PTR, $IN_START_PTR, $OFFS
  add       $OUT_PTR, $OUT_PTR, $OFFS
#endif
  // Rest of the code wants NUM_MATRICES-1 (for the loop instructions) so we
  // decrement and jump to main code (NUM_MATRICES is for sure > 0 here).
  brnzdec   $NUM_MATRICES, TransposeAT2EnterMultiVertex

FN_SIZE TRANSPOSE_1D_HALF


//----------------------------------------------------------------------------
// Entry point when worker codelet is called directly. The parameter in the
// vertex state are already correct.

#define TRANSPOSE_HALF \
  __runCodelet_popops__Transpose1DSingleWorker___half
#define TRANSPOSE_UNSIGNED_SHORT \
  __runCodelet_popops__Transpose1DSingleWorker___unsigned_short
#define TRANSPOSE_SHORT \
  __runCodelet_popops__Transpose1DSingleWorker___short

FN_WORKER_ENTRY_POINT TRANSPOSE_HALF
FN_EXPORT TRANSPOSE_UNSIGNED_SHORT
FN_EXPORT TRANSPOSE_SHORT

// kept on stack
#define SOFF_NUM_PTR_INC           0

#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16     $IN_START_PTR, $mzero, $mvertex_base, VOFF_IN_PTR_1D / 2
  ldz16     $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR_1D / 2
#else
  ld32      $IN_START_PTR, $mzero, $mvertex_base, VOFF_IN_PTR_1D / 4
  ld32      $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR_1D / 4
#endif
  ldz16     $OUT_GROUPS, $mzero, $mvertex_base, VOFF_SRC_ROWS_1D / 2
  ldz16     $IN_GROUPS, $mzero, $mvertex_base, VOFF_SRC_COLUMNS_1D / 2
  ldz16     $NUM_MATRICES, $mzero, $mvertex_base, VOFF_NUM_MATRICES_1D / 2

  // increment for src/dst pointers saved on stack
  mul       $PTR_INCR, $IN_GROUPS, $OUT_GROUPS
  // shift up because rows and columns are shifted down by 2. Additional shift
  // because data is half
  shl       $PTR_INCR, $PTR_INCR, (2 + 2 + 1)
TransposeAT2EnterMultiVertex:

  st32      $PTR_INCR, $mzero, $mworker_base, SOFF_NUM_PTR_INC / 4

#if defined(VECTOR_AVAIL_SCALED_PTR64)
  // expand pointers as they are scaled64
  shl       $IN_START_PTR, $IN_START_PTR, 3
  shl       $OUT_PTR, $OUT_PTR, 3
#endif

  // Special case 4x4 matrices, check based on computed size (PTR_INCR)
  cmpeq     $mSCRATCH, $PTR_INCR, (4 * 4 * 2)
  brnz      $mSCRATCH, TransposeAT2FastPathNx4x4Setup
  // Special case nx4 matrices, check based on IN_GROUPS (IN_GROUPS = columns/4)
  cmpeq     $mSCRATCH, $IN_GROUPS, 1
  brnz      $mSCRATCH, .Lat2_nx4x4

  call      $LINK_REG, TransposeAT2FastPathSetup
.Lmatrix_loop_half_1d:
  tapack     $INOUT_PTR, $IN_START_PTR, $OUT_PTR, $mzero
  // calculate outer loop passes
  ldz16      $COUNTER, $mzero, $mvertex_base, VOFF_SRC_ROWS_1D / 2
  add        $COUNTER, $COUNTER, -1
  call       $LINK_REG, TransposeAT2FastPathCommon
  ld32       $PTR_INCR, $mzero, $mworker_base, SOFF_NUM_PTR_INC / 4
  add        $IN_START_PTR, $IN_START_PTR, $PTR_INCR
  add        $OUT_PTR, $OUT_PTR, $PTR_INCR
  brnzdec    $NUM_MATRICES, .Lmatrix_loop_half_1d
TransposeAT2Exit:
  exitz      $mzero




//***************************************
// the TransposeAT2FastPathCommon function can be used to
// transpose multiple 4x4 matrices instead of a single
// nx4 matrix if we let it loop per matrix and make
// all the strides that would index by a row etc just increment
TransposeAT2FastPathNx4x4Setup:

  // Inner loop deals with each matrix - no other loops required
  add        $LOOP_COUNT, $NUM_MATRICES, -1

  ldconst    $STRIDES, (1<<20) | (1<<10) | (1)
  // Re-purpose out groups to deal with more strides, but in this case all = 1
  mov        $OUT_GROUPS, $STRIDES
  // A little slower, but smaller code to share the loop end below (with no loop passes)
  setzi      $NUM_MATRICES, 0
  bri        .Lat2_nx4x4_common

//***************************************
.Lat2_nx4x4:
  call       $LINK_REG, TransposeAT2FastPath4ColumnSetup
.Lat2_nx4x4_common:
  // Input will step in a simple linear way throughout
  mov      $IN_REWIND, $IN_GROUPS
  // No outer loop passes
  setzi      $COUNTER, 0
.Lat2_nx4x4_loop:
  tapack     $INOUT_PTR, $IN_START_PTR, $OUT_PTR, $mzero
  call       $LINK_REG, TransposeAT2FastPathCommon
  ld32       $PTR_INCR, $mzero, $mworker_base, SOFF_NUM_PTR_INC / 4
  add        $IN_START_PTR, $IN_START_PTR, $PTR_INCR
  add        $OUT_PTR, $OUT_PTR, $PTR_INCR
  brnzdec    $NUM_MATRICES, .Lat2_nx4x4_loop

  exitz      $mzero

FN_SIZE TRANSPOSE_HALF

/* -------------------------------------------------------------------------- */

#define TRANSPOSE_2D_HALF \
  __runCodelet_popops__Transpose2D___half
#define TRANSPOSE_2D_UNSIGNED_SHORT \
  __runCodelet_popops__Transpose2D___unsigned_short
#define TRANSPOSE_2D_SHORT \
  __runCodelet_popops__Transpose2D___short

FN_WORKER_ENTRY_POINT TRANSPOSE_2D_HALF
FN_EXPORT TRANSPOSE_2D_UNSIGNED_SHORT
FN_EXPORT TRANSPOSE_2D_SHORT

    GET_PARAMS_2D  VOFF_IN_START_PTR_2D VOFF_IN_SIZE_2D VOFF_OUT_PTR_2D VOFF_SRC_ROWS_2D VOFF_SRC_COLUMNS_2D

    // The fast algorithm works for rows,cols a multiple of 4 only - test &
    // branch if we can't use it.
    or       $mSCRATCH2, $IN_ROWS, $IN_COLUMNS
    and      $mSCRATCH2, $mSCRATCH2, 3
    brnz     $mSCRATCH2, TransposeAT2SlowPath
    // we need various strides forward and backward through the data to
    // address input and output. Constant as every matrix has the same rows,cols
    // Here groups are COLUMNS/4 or ROWS/4: the number of 64 bit fetches/writes
    shr       $OUT_GROUPS, $IN_ROWS, 2
    shr       $IN_GROUPS, $IN_COLUMNS, 2
    // OUT_GROUPS must fit into a signed 10bit field to use the fast vertex.
    cmpult   $mSCRATCH2, $OUT_GROUPS, 0x400
    brnz     $mSCRATCH2, TransposeAT2SlowPath
    // matrix(outermost) loop count
    add       $NUM_MATRICES, $mSCRATCH4, -1

    cmpeq     $mSCRATCH, $IN_COLUMNS, 4
    brnz      $mSCRATCH, .Lat2_4_columns

    call      $LINK_REG, TransposeAT2FastPathSetup
    // This should be the largest stride calculated so check it:
    // Is this -ve stride going to be bigger than a signed 10 bit range?
    cmpslt    $mSCRATCH, $IN_REWIND, -512
    brnz      $mSCRATCH, TransposeAT2SlowPath
    //repeat count will be  within range of its register value
    // as the stride constraint will always happen 1st.
    // -3x +1 < -512 happens for smaller x than x-2 > 4096

1:
    // loop per matrix starts here - fetch input/output pointers
    ld32step   $LD_PTR, $mzero, $IN_START_PTR+=, 1
    ld32step   $ST_PTR, $mzero, $OUT_PTR+=, 1
    // calculate outer loop passes
    ldz16      $COUNTER, $mzero, $mvertex_base, VOFF_SRC_ROWS_2D
    shr        $COUNTER, $COUNTER, 2
    add        $COUNTER, $COUNTER, -1
    call       $LINK_REG, TransposeAT2FastPathCommon

    brnzdec    $NUM_MATRICES, 1b
    exitz      $mzero

//***************************
.Lat2_4_columns:
     call      $LINK_REG, TransposeAT2FastPath4ColumnSetup
   // This should be the largest stride calculated so check it:
    // Is this -ve stride going to be bigger than a signed 10 bit range?
    cmpslt    $mSCRATCH, $IN_REWIND, -512
    brnz      $mSCRATCH, TransposeAT2SlowPath

    //repeat count will be  within range of its register value
    // as the stride constraint will always happen 1st.
    // -3x +1 < -512 happens for smaller x than x-2 > 4096

    // Input will step in a simple linear way throughout
    setzi     $IN_REWIND, 1
    // No outer loop passes within the called function
    setzi     $COUNTER, 0
.Lat2_4_columns_loop:
    // loop per matrix starts here - fetch input/output pointers
    ld32step   $LD_PTR, $mzero, $IN_START_PTR+=, 1
    ld32step   $ST_PTR, $mzero, $OUT_PTR+=, 1
    call       $LINK_REG, TransposeAT2FastPathCommon

    brnzdec    $NUM_MATRICES, .Lat2_4_columns_loop
    exitz      $mzero


//***************************************************************
// Slower version for half processing - using macro above

TransposeAT2SlowPath:
   ldz16    $IN_ROWS, $mzero, $mvertex_base, VOFF_SRC_ROWS_2D
   add      $MATRIX_LOOP_COUNT, $mSCRATCH4, -1

  TRANSPOSE_SLOWER ldb16step st32step $VAL1 2 1

FN_SIZE TRANSPOSE_2D_HALF


//----------------------------------------------------------------------------
// Entry point for MultiVertex code where multiple transpositions are 
// processed and work split is done at runtime.
// This computes the offsets from IN and OUT data ptrs for this worker and
// (if needed) decrements NUM_MATRICES.
#define SPLIT_TRANSPOSE_1D_HALF            __runCodelet_popops__SplitTranspose1D___half
#define SPLIT_TRANSPOSE_1D_UNSIGNED_SHORT  __runCodelet_popops__SplitTranspose1D___unsigned_short
#define SPLIT_TRANSPOSE_1D_SHORT           __runCodelet_popops__SplitTranspose1D___short

//------------------------------------------------------------------------------

// MultiVertex state for the HALF, fast, version with split transposes
// All offsets in bytes.
//
#if defined(VECTOR_AVAIL_SCALED_PTR64)
#define VOFF_IN_PTR_1D_SPLIT             0     // Scaled 64 source pointer
#define VOFF_OUT_PTR_1D_SPLIT            2     // Scaled 64 dst pointer
#define VOFF_SRC_ROWS_1D_SPLIT           4     // Number of rows divided by 4
#define VOFF_SRC_COLUMNS_1D_SPLIT        6     // Number of columns divided by 4
#define VOFF_WORKLIST_1D_SPLIT           8     // Worklist pointer (points to 4 * NUM_WORKERS ushorts)
#else
#define VOFF_IN_PTR_1D_SPLIT             0     // one_ptr source pointer
#define VOFF_OUT_PTR_1D_SPLIT            4     // one_ptr dst pointer
#define VOFF_SRC_ROWS_1D_SPLIT           8     // Number of rows divided by 4
#define VOFF_SRC_COLUMNS_1D_SPLIT       10     // Number of columns divided by 4
#define VOFF_WORKLIST_1D_SPLIT          12     // Compact worklist pointer (points to ushorts of size 4 * workers used)
#endif


// WORKIST ENTRY: offsets in ushorts
#define WORKLIST_ENTRY_IN_OFFSET      0   // multiple of 4 elements to cover larger range
#define WORKLIST_ENTRY_OUT_OFFSET     1   // multiple of 4 elements to cover larger range
#define WORKLIST_ENTRY_NUM_ALLOC_ROWS 2   // multiples of 4
#define WORKLIST_ENTRY_NUM_ALLOC_COLS 3   // multiples of 4

FN_WORKER_ENTRY_POINT SPLIT_TRANSPOSE_1D_HALF
FN_EXPORT SPLIT_TRANSPOSE_1D_UNSIGNED_SHORT
FN_EXPORT SPLIT_TRANSPOSE_1D_SHORT
#define WORKLIST_PTR   m5
#define WORKLIST_ENTRY m6
#define WORKERS_USED   m6
#define NUM_ALLOC_COLS m4
#define NUM_ALLOC_ROWS m5
#define SCRATCH_SPLIT  m6

#if defined(VECTOR_AVAIL_SCALED_PTR64)
  ldz16     $IN_START_PTR, $mzero, $mvertex_base, VOFF_IN_PTR_1D_SPLIT / 2
  ldz16     $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR_1D_SPLIT / 2
  shl       $IN_START_PTR, $IN_START_PTR, 3
  shl       $OUT_PTR, $OUT_PTR, 3
#else
  ld32      $IN_START_PTR, $mzero, $mvertex_base, VOFF_IN_PTR_1D_SPLIT / 4
  ld32      $OUT_PTR, $mzero, $mvertex_base, VOFF_OUT_PTR_1D_SPLIT / 4
#endif
  ldz16     $OUT_GROUPS, $mzero, $mvertex_base, VOFF_SRC_ROWS_1D_SPLIT / 2
  ldz16     $IN_GROUPS, $mzero, $mvertex_base, VOFF_SRC_COLUMNS_1D_SPLIT / 2
  ld32      $WORKLIST_PTR, $mvertex_base, VOFF_WORKLIST_1D_SPLIT / 4
  
  // extract number of workers used from worklist: divide by 4
  shr       $WORKERS_USED, $WORKLIST_PTR, (SHORT_SPAN_ADDR_SIZE_BITS + 2)

  // Get worker ID
  get       $WORKER_ID, $WSR
  and       $WORKER_ID, $WORKER_ID, CSR_W_WSR__CTXTID_M1__MASK

  // check if any work is assigned worker
  cmpult    $SCRATCH_SPLIT, $WORKER_ID, $WORKERS_USED
  brz       $SCRATCH_SPLIT, .LSplitTranspose1DHalfExit

  // extract pointer
  shl       $WORKLIST_PTR, $WORKLIST_PTR, SHORT_SPAN_WORKLIST_SIZE_BITS
  shr       $WORKLIST_PTR, $WORKLIST_PTR, SHORT_SPAN_WORKLIST_SIZE_BITS

  mul       $OFFS, $WORKER_ID, 8  // each worklist entry is 4 ushorts

  // dummy load to increment pointer to correct offset
  ldz16     $WORKLIST_ENTRY, $OFFS, $WORKLIST_PTR, WORKLIST_ENTRY_IN_OFFSET
  ld64step  $azeros, $mzero, $IN_START_PTR+=, $WORKLIST_ENTRY

  // dummy load to increment pointer to correct offset
  ldz16     $WORKLIST_ENTRY, $OFFS, $WORKLIST_PTR, WORKLIST_ENTRY_OUT_OFFSET
  ld64step  $azeros, $mzero, $OUT_PTR+=, $WORKLIST_ENTRY

  ldz16     $NUM_ALLOC_COLS, $OFFS, $WORKLIST_PTR, WORKLIST_ENTRY_NUM_ALLOC_COLS
  ldz16     $NUM_ALLOC_ROWS, $OFFS, $WORKLIST_PTR, WORKLIST_ENTRY_NUM_ALLOC_ROWS
  
  cmpeq     $SCRATCH_SPLIT, $NUM_ALLOC_COLS, 1
  brz       $SCRATCH_SPLIT, .LSplitTranpose1DHalfMxN

  // Do setup for 4 columns if number of allocated columns are 4.
  call      $LINK_REG, TransposeAT2FastPath4ColumnSetup
  // The setup uses the transposition rows to set up loop count. Over-write with
  // allocated rows
  add       $LOOP_COUNT, $NUM_ALLOC_ROWS, -2
  // Input will step in a simple linear way throughout
  mov        $IN_REWIND, $IN_GROUPS
  // No outer loop passes
  setzi      $COUNTER, 0
  bri        .LSplitDoTranspose

.LSplitTranpose1DHalfMxN:
  // Do setup of columns > 4
  call       $LINK_REG, TransposeAT2FastPathSetup
  // The setup uses the transposition columns to set up loop count. Over-write 
  ///with allocated columns
  add        $LOOP_COUNT, $NUM_ALLOC_COLS, -2
  // calculate outer loop passes
  add        $COUNTER, $NUM_ALLOC_ROWS, -1  

.LSplitDoTranspose:
  tapack     $INOUT_PTR, $IN_START_PTR, $OUT_PTR, $mzero
  call       $LINK_REG, TransposeAT2FastPathCommon

.LSplitTranspose1DHalfExit:
  exitz $mzero

FN_SIZE SPLIT_TRANSPOSE_1D_HALF

#endif
