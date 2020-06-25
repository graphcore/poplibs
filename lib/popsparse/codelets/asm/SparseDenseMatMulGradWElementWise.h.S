// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Header for Sparse-Dense matrix multiplication for GradW asm codelets

#ifndef _SparseDenseMatMulGradWElementWise_h_
#define _SparseDenseMatMulGradWElementWise_h_

#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)
#define LOG2_SIZEOF_OUT_ATOM 2

// =============================================================================

// =============================================================================

//// Supervisor vertex state
#define SUP_VBASE_QGRAD_BASE         0    // one pointer
#define SUP_VBASE_RGRAD_BASE         4    // one pointer 
#define SUP_VBASE_META_BASE          8    // one pointer
#define SUP_VBASE_S_BASE             12   // one pointer
#define SUP_VBASE_PN_SUBGROUP_ID     16   // pointer to ushort
#define SUP_VBASE_ZERO_INFO          20   // ushort
#define SUP_VBASE_NUM_Z              22   // ushort

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
#define STACK_SIZE                      (W_NUM_Z + 4)

// =============================================================================
#endif // #define _SparseDenseMatMulGradWElementWise_h_
// =============================================================================
