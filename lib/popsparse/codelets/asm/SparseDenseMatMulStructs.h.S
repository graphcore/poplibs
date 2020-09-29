// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Header for Sparse-Dense matrix multiplication for GradW asm codelets

#ifndef _SparseDenseMatMulStructs_h_
#define _SparseDenseMatMulStructs_h_

// MetaInfo field offsets and sizes
#define Sizeof_MIFwdWorkerEntry                            10
#define Sizeof_MIGradWWorkerEntry                          8

// !!! Assumption in the code that sparse offset is the first entry in the table
#define MIGradWorkerEntry_sparseOffset                     0
#define MIGradWorkerEntry_metaInfoOffsetOutputEntry        2
#define MIGradWorkerEntry_metaInfoOffsetToOffsetsYInSFirst 4
#define MIGradWorkerEntry_totalNumY                        6

// Bucket meta information field offsets and sizes (in bytes)
#define sizeof_MetaInfoSubGroupEntry                       18
#define MetaInfoSubGroupEntry_id                           0
#define MetaInfoSubGroupEntry_xPartition                   2
#define MetaInfoSubGroupEntry_yPartition                   4
#define MetaInfoSubGroupEntry_sparseElementCount           6
#define MetaInfoSubGroupEntry_offsetToNextSubGroupMetaInfo 8
#define MetaInfoSubGroupEntry_numZ                         10
#define MetaInfoSubGroupEntry_numXm1                       12
#define MetaInfoSubGroupEntry_offsetToFirstOutputEntry     14
#define MetaInfoSubGroupEntry_numWorkers                   16
#define sizeof_metaInfoEntry                               2


// MetaInfo field offsets
#define Sizeof_MetaInfoWorkerEntry                        10
#define MetaInfoWorkerEntry_numZ                          2
#define MetaInfoWorkerEntry_sparseOffset                  0
#define MetaInfoWorkerEntry_offsetZ                       4
#define MetaInfoWorkerEntry_numXm1                        6
#define MetaInfoWorkerEntry_metaInfoOffset                8

// =============================================================================
#endif // #define _SparseDenseMatMulStructs_h_
// =============================================================================
