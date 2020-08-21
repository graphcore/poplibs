// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Header for Block Sparse-Dense matrix multiplication for GradW asm codelets

#ifndef _BlockSparseDenseMatMulStructs_h_
#define _BlockSparseDenseMatMulStructs_h_

// Bucket meta information field offsets and sizes (in bytes)
#define sizeof_MetaInfoSubGroupEntry                            10
#define MetaInfoSubGroupEntry_id                                0
#define MetaInfoSubGroupEntry_offsetToNextSubGroupSparseEntries 2
#define MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo      4
#define MetaInfoSubGroupEntry_numXm1                            6
#define MetaInfoSubGroupEntry_numGradWWorkers                   8

// =============================================================================
#endif // #define _BlockSparseDenseMatMulStructs_h_
// =============================================================================
