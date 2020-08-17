// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 8x8
//

#ifdef __IPU__
#include "BlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradA___half_half_8_8

// =============================================================================

.extern blockSparseDenseMultiply_hh8x8
.extern blockSparseDenseMultiply_hh8x8_retained

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME half half hh8x8 8 1

// =============================================================================
#endif // #ifdef __IPU__
// =============================================================================
