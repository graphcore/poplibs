// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "SelectScalarFromRows.hpp"

using namespace poplar;

namespace popops {

// Assume that one Vector of params is a region with the following layout:
//
// Parent tensor
// |
// |
// |            Start of region (column x)            paramsWidth
// |            +                                     +
// |            |                                     |
// v            v          v       v      v           v
// +------------+----------+-------+------+-----------+
// |            |    A     |   E   |  I   |           |
// |            +-------------------------+           |
// |            |    B     |   F   |  J   |           |
// |            +-------------------------+           |
// |            |    C     |   G   |  K   |           |
// |            +-------------------------+           |
// |            |    D     |   H   |  L   |           |
// |            +-------------------------+           |
// |                                                  |
// +--------------------------------------------------+
//
// This is a typical layout used by matrix multiplication.
// The values in interval A precede B, C, D, etc consecutively in memory.
// In this example the values of the input Vector<unsigned> are (assuming a
// single region) columnWidths: {A.size} regionHeights: {4} regionWidths: {3}
// firstColumns {x}
//
// indices has size 3.

template <typename T> class UpdateColumnsDEC : public Vertex {
  static_assert(std::is_same<T, float>() || std::is_same<T, half>(),
                "T must be a either float or half");

public:
  // List of contiguous regions of the params tensor.
  Vector<InOut<Vector<T, ONE_PTR>>> params;
  // List of vectors of indices, one for each contiguous region of params.
  Vector<Input<Vector<unsigned, ONE_PTR>>, ONE_PTR> indices;

  // Width of the columns that make up the params regions.
  // Ie. As above A.size, E.size, I.size
  // If we have another 2 regions each with 2 columns that becomes
  // A1.size, E1.size, I1.size, A2.size, E2.size, A3.size, E3.size
  Vector<unsigned, ONE_PTR> columnWidths;
  // Number of rows for each region.
  Vector<unsigned, ONE_PTR> regionHeights;
  // Number of columns for each region.
  Vector<unsigned, ONE_PTR> regionWidths;
  // Index of the first column, for each region.
  Vector<unsigned, ONE_PTR> firstColumns;
  // The width of the original 2D param matrix. Used for in-bounds checks.
  unsigned paramsWidth;

  bool compute() {
    // For each contiguous region.
    unsigned columnWidthOffset = 0;
    for (unsigned region = 0; region != params.size(); ++region) {
      const unsigned regionWidth = regionWidths[region];
      const unsigned regionHeight = regionHeights[region];

      unsigned columnStart = 0;
      unsigned columnOffset = 0;
      // For each column.
      for (unsigned c = 0; c != regionWidth; ++c) {
        const unsigned columnWidth = columnWidths[columnWidthOffset + c];
        // For each row.
        for (unsigned r = 0; r != regionHeight; ++r) {
          decrementParams(
              &params[region][columnStart + r * columnWidth],
              indices[region][r], firstColumns[region] + columnOffset,
              firstColumns[region] + columnOffset + columnWidth, paramsWidth);
        }
        columnStart += columnWidth * regionHeight;
        columnOffset += columnWidth;
      }
      columnWidthOffset += regionWidth;
    }

    return true;
  }
};

template class UpdateColumnsDEC<float>;
template class UpdateColumnsDEC<half>;

} // namespace popops
