// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

namespace popsparse {

// Y offset is scaled by this given input type
static inline unsigned getYOffsetTypeScaleFactor(bool floatInput) {
  return floatInput ? 4 : 2;
}

template <typename T> struct MetaInfo {
  constexpr static T endSubGroupId = 0;
  struct SubGroupEntry {
    // ID of the sub-group: must be the first entry as it is checked for
    // in deciding to exit processing of a bucket.
    T id;
    // The partition in the X dimension that this sub-group's information
    // belongs to
    T xPartition;
    // The partition in the Y dimension that this sub-group's information
    // belongs to
    T yPartition;
    // Number of sparse values for this sub-group in this bucket.
    T sparseElementCount;
    // Offset to next sub-group's entry in this bucket.
    T offsetToNextSubGroupMetaInfo;
    // The total Z dimension used by all the workers
    T numZ;
    // The total number of rows processed by all the workers on this tile
    T numXm1;
    // The offset of the first output entry on this tile
    T offsetToFirstOutputEntryMetaInfo;
    // Number of workers to utilise for this sub-group's work in this bucket.
    T numWorkers;
  };
  struct WorkerEntry {
    // Offset in elements to sparse values for this worker to process.
    T sparseOffset;
    // Number of elements of dimension Z to process.
    T numZ;
    // Offset in elements to first element of dimension Z to process. This is
    // the same for both operands Q and S as Z is expected to be the inner-most
    // dimension in memory for both.
    T offsetZ;
    // Number of elements of dimension X to process - 1.
    T numXm1;
    // Offset in elements from the beginning of this struct to
    // annotation describing elements of dimensions X/Y to process.
    T metaInfoOffset;
  };
  struct GradWWorkerEntry {
    // Offset in elements to sparse values for this worker to process;
    T sparseOffset;
    // Offset in elements from the beginning of this struct to first output
    // entry this worker will process.
    T metaInfoOffsetOutputEntry;
    // Offset into the output entry's columns for the first output entry
    // this worker processes.
    T metaInfoOffsetToOffsetsYInSFirst;
    // Total number of elements of Y to process over all rows.
    T totalNumY;
  };
  struct OutputEntry {
    // Offset in 8-bytes to first element of dimension X to process in the
    // operand Q. Note due to expected layout this will essentially be
    // offsetX * Z * bytesPerElements / 8
    T offsetXInQ;
    // Number of elements of dimension Y to process.
    T numY;
  };
};

// Meta-info for block-sparse codelets only
template <typename T> struct BlockMetaInfo {
  constexpr static T endSubGroupId = 0;
  struct SubGroupEntry {
    // ID of the sub-group
    T id;
    // The partition in the X dimension that this sub-group's information
    // belongs to
    T xPartition;
    // The partition in the Y dimension that this sub-group's information
    // belongs to
    T yPartition;
    // Offset in elements to next sub-group's non-zero values from
    // the end of the last sub-groups' non-zero values.
    T offsetToNextSubGroupSparseEntries;
    // Offset to next sub-group's entry in this bucket from the beginning
    // of this structure.
    T offsetToNextSubGroupMetaInfo;
    // Number of blocks in X dimension in this sub-group (minus 1).
    T numXm1;
    // Total number of GradW workers
    T numGradWWorkers;
  };

  struct GradWWorkerEntry {
    // Offset in elements to sparse values for this worker to process;
    T sparseOffset;
    // Offset in elements from the beginning of this struct to first output
    // entry this worker will process.
    T metaInfoOffsetOutputEntry;
    // Offset into the output entry's columns for the first output entry
    // this worker processes.
    T metaInfoOffsetToOffsetsYInSFirst;
    // Total number of elements of Y to process over all rows.
    T totalNumY;
  };

  struct OutputEntry {
    // Offset to X index in Q in elements of Q
    // Q has layout {Z,X} in row major order.
    T offsetXInQ;
    // Number of blocks for this index in X (minus 1).
    T numYm1;
  };
  struct InputEntry {
    // Offset to Y index in elements of S.
    // S has layout {Z,Y} in row major order.
    T offsetYInS;
  };
};

} // namespace popsparse
