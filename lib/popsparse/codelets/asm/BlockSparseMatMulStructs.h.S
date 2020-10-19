// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Header for Block Sparse-Dense matrix multiplication for GradW asm codelets

#ifndef _BlockSparseDenseMatMulStructs_h_
#define _BlockSparseDenseMatMulStructs_h_

// Bucket meta information field offsets and sizes (in bytes)
#define sizeof_MetaInfoSubGroupEntry                            14
#define MetaInfoSubGroupEntry_id                                0
#define MetaInfoSubGroupEntry_xPartition                        2
#define MetaInfoSubGroupEntry_yPartition                        4
#define MetaInfoSubGroupEntry_offsetToNextSubGroupSparseEntries 6
#define MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo      8
#define MetaInfoSubGroupEntry_numXm1                            10
#define MetaInfoSubGroupEntry_numGradWWorkers                   12

#define sizeof_MetaInfoGradWWorkerEntry                             8
#define MetaInfoGradWWorkerEntry_sparseOffset                       0
#define MetaInfoGradWWorkerEntry_metaInfoOffsetOutputEntry          2
#define MetaInfoGradWWorkerEntry_metaInfoOffsetToOffsetsYInSFirst   4
#define MetaInfoGradWWorkerEntry_totalNumY                          6

// =============================================================================
#endif // #define _BlockSparseDenseMatMulStructs_h_
// =============================================================================
