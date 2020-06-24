// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

namespace popsparse {

// X offset is scaled by this given input type
static inline unsigned getXOffsetTypeFactor(bool floatInput) {
  return floatInput ? 1 : 2;
}

// Y offset is scaled by this given input type
static inline unsigned getYOffsetTypeFactor(bool floatInput) {
  return floatInput ? 4 : 2;
}

template <typename T> struct MetaInfo {
  constexpr static T endSubGroupId = 0;
  struct SubGroupEntry {
    // ID of the sub-group: must be the first entry as it is checked for
    // in deciding to exit processing of a bucket.
    T id;
    // Number of sparse values for this sub-group in this bucket.
    T sparseElementCount;
    // Offset to next sub-group's entry in this bucket.
    T offsetToNextSubGroupMetaInfo;
    // The total Z dimension used by all the workers
    T numZ;
    // The total number of rows processed by all the workers on this tile
    T numXm1;
    // The offset of the first output entry on this tile
    T offsetToFirstOutputEntry;
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
    // Offset in bytes to first element of dimension X to process in the
    // operand Q. Note due to expected layout this will essentially be
    // offsetX * Z * bytesPerElement.
    T offsetXInQ;
    // Number of elements of dimension Y to process.
    T numY;
  };
};

} // namespace popsparse
