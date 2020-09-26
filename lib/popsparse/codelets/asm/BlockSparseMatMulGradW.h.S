// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Header for Sparse-Dense matrix multiplication for GradW asm codelets

#ifndef _BlockSparseMatMulGradW_h_
#define _BlockSparseMatMulGradW_h_

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)
#define LOG2_SIZEOF_OUT_ATOM 2

// =============================================================================

// =============================================================================

//// Supervisor vertex state for non-AMP codelet
#define SUP_VBASE_QGRAD_BASE         0    // one pointer
#define SUP_VBASE_RGRAD_BASE         4    // one pointer 
#define SUP_VBASE_META_BASE          8    // one pointer
#define SUP_VBASE_S_BASE             12   // one pointer
#define SUP_VBASE_PN_SUBGROUP_ID     16   // pointer to ushort
#define SUP_VBASE_ZERO_INFO          20   // ushort
#define SUP_VBASE_NUM_Z              22   // ushort
#define SUP_VBASE_ZSTRIDE_IN_Q       24   // ushort
#define SUP_VBASE_ZSTRIDE_IN_S       26   // ushort

//// Supervisor vertex state for AMP codelet
#define SUP_VBASE_AMP_QGRAD_BASE     0    // one pointer
#define SUP_VBASE_AMP_RGRAD_BASE     4    // one pointer 
#define SUP_VBASE_AMP_META_BASE      8    // one pointer
#define SUP_VBASE_AMP_S_BASE         12   // one pointer
#define SUP_VBASE_AMP_PN_SUBGROUP_ID 16   // pointer to ushort
#define SUP_VBASE_AMP_ZERO_INFO      20   // ushort
#define SUP_VBASE_AMP_NUM_Z          22   // ushort

// =============================================================================

//// Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////
#define W_S_BASE                        0
#define W_QGRAD_BASE                    4
#define W_RGRAD_BASE                    8
#define W_METAINFO                      12
#define W_NUM_Z                         16
#define W_ZSTRIDE_IN_Q                  20
#define W_ZSTRIDE_IN_S                  24
#define STACK_SIZE                      (W_ZSTRIDE_IN_S + 4)

//// AMP Vertex state shared between workers (Worker vertex state is allocated
//// on supervisor stack and along with stack space used by supervisor must be
//// a multiple of 8 bytes)
////
#define W_AMP_S_BASE                    0
#define W_AMP_QGRAD_BASE                4
#define W_AMP_RGRAD_BASE                8
#define W_AMP_RGRAD_BASE_BLOCK          12
#define W_AMP_METAINFO                  16
#define W_AMP_PN_SUBGROUP_ID            20
#define W_AMP_NUM_X16_BATCHES           24
#define W_AMP_OFFSET_Z                  28
#define STACK_SIZE_AMP                  (W_AMP_OFFSET_Z + 4)

// =============================================================================

// Macro load with post increment given the type of data to load from.
.macro LOAD_POSTINC, TYPE, EXPR1, EXPR2, EXPR3, EXPR4
.ifc \TYPE, float
    ld32step \EXPR1, \EXPR2, \EXPR3+=, \EXPR4
.else
    ldz16step  \EXPR1, \EXPR2, \EXPR3+=, \EXPR4
.endif
.endm

// Increment pointer given the type and increment
.macro INCR_PTR, TYPE, PTR, INCR
.ifc \TYPE, float
add  \PTR, \PTR, \INCR * 4
.else
add  \PTR, \PTR, \INCR * 2
.endif
.endm

// Decrement pointer given the type and increment
.macro DECR_PTR, TYPE, PTR, DECR
.ifc \TYPE, float
add  \PTR, \PTR, \DECR * -4
.else
add  \PTR, \PTR, \DECR * -2
.endif
.endm


// =============================================================================
#endif // #define _BlockSparseMatMulGradW_h_
// =============================================================================
