// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef ScalarInFromRowsCommon_hpp__
#define ScalarInFromRowsCommon_hpp__

// Common test functions for UpdateScalarInRowsTest and SelectScalarFromRowsTest
// rearrangeTensor, rearrangeInput: We want to test the identification of
// a layout where we have a rank 2 tensor and want a specific memory layout,
// for example
// | A     | D |
// | B     | E |
// | C     | F |
//
// Suppose dimensions are {3, 16}, and the width of A,B and  C = 10, the width
// of D,E,F = 6.  Data is to be stored in memory in the order A B C D E F:
// tIn: | A     | B     | C     | D | E | F |
//
// tIn is a flattened tensor stored linearly in memory as above.  tOut needs to
// be a view into the tensor which represents the data.  So we concatenate
// slices of tIn to form tOut:
// tOut: | A     | D | B      | E | C     | F |
//
// When reshaped back to {3, 16} this has the layout above.
//
// slicedWidthGroups denotes that gaps will be enforced so that groups of
// columns are in a different memory region. Eg:
// | A     | D | G  |
// | B     | E | H  |
// | C     | F | I  |
//
// Can be laid out with slicedWidthGroups {1, 2}:
// Region 1: | A     | B    | C    | (One element gap between regions)
// Region 2: | D | E | F | G  | H  | I  |
//
// If laid out in just one region we would have:
// | A     | B     | C     | D | E | F | G  | H  | I  |
//
// rearrangeInput moves the data to match.

namespace {
void rearrangeTensor(const poplar::Tensor &tIn,
                     const std::vector<unsigned> sliceWidths,
                     const std::vector<unsigned> sliceWidthGroups,
                     poplar::Tensor &tOut, unsigned rows) {
  std::vector<poplar::Tensor> slices;
  for (unsigned i = 0; i < rows; i++) {
    // Used to provide the accumulation of all column's data so far.
    // Eg 0, (A+B+C), (A+B+C+D+E+F), (A+B+C+D+E+F+G+H+I)
    unsigned sliceAcc = 0;
    // Used to produce a gap between regions
    unsigned slicedRegion = 0;
    unsigned slicedRegionAcc = 0;
    for (unsigned j = 0; j < sliceWidths.size(); j++) {
      const unsigned sliceStart = sliceAcc + i * sliceWidths[j];
      if (j - slicedRegionAcc == sliceWidthGroups[slicedRegion]) {
        slicedRegion++;
        slicedRegionAcc += j - slicedRegionAcc;
      }
      slices.push_back(tIn.slice(sliceStart + slicedRegion,
                                 sliceStart + sliceWidths[j] + slicedRegion));
      sliceAcc += rows * sliceWidths[j];
    }
  }
  tOut = concat(slices);
}

void rearrangeInput(const std::vector<float> &in, std::vector<float> &out,
                    const std::vector<unsigned> sliceWidths,
                    const std::vector<unsigned> sliceWidthGroups,
                    unsigned rows) {
  out.resize(in.size() + sliceWidthGroups.size() - 1);
  // This mirrors the effect of concatanating tensors in rearrangeTensor
  unsigned inIndex = 0;
  for (unsigned i = 0; i < rows; i++) {
    // Used to provide the accumulation of all column's data so far.
    // Eg 0, (A+B+C), (A+B+C+D+E+F), (A+B+C+D+E+F+G+H+I)
    unsigned sliceAcc = 0;
    // Used to produce a gap between regions
    unsigned slicedRegion = 0;
    unsigned slicedRegionAcc = 0;
    for (unsigned j = 0; j < sliceWidths.size(); j++) {
      const unsigned sliceStart = sliceAcc + i * sliceWidths[j];
      if (j - slicedRegionAcc == sliceWidthGroups[slicedRegion]) {
        slicedRegion++;
        slicedRegionAcc += j - slicedRegionAcc;
      }
      std::copy(&in[inIndex], &in[inIndex + sliceWidths[j]],
                &out[sliceStart + slicedRegion]);
      inIndex += sliceWidths[j];
      sliceAcc += rows * sliceWidths[j];
    }
  }
}

} // namespace
#endif // __ScalarInFromRowsCommon__hpp
