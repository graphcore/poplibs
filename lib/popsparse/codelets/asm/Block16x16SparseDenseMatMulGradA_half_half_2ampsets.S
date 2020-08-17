// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Performs sparse matrix multiplication Q = R * S Where
// Q and S are dense matrices and R is a sparse matrix
// with block size of 16x16

#if defined(__IPU__)
#if __IPU_ARCH_VERSION__ >=2
#include "BlockSparseDenseMatMul.h.S"
#include "poplar/AvailableVTypes.h"

// =============================================================================

#define CODELET_NAME __runCodelet_popsparse__SparseDenseMatMulBlockGradA___half_half_16_16

// =============================================================================

.extern blockSparseDenseMultiply_hh16x16
.extern blockSparseDenseMultiply_hh16x16_retained

// =============================================================================
// Supervisor codelet which launches the zeroing of the output Q matrix and
// then parses the meta information buckets. Each bucket is walked through to
// match the PNs subgroup id. 

// Instantiate supervisor codelet
BLOCK_SPARSE_MATMUL CODELET_NAME half half hh16x16 16 1

#endif // __IPU_ARCH_VERSION__ >=2
#endif // defined(__IPU__)
