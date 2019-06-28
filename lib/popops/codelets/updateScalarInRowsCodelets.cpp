#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include "popops/EncodingConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

namespace {
// The type half does not have the -- operator.
template <typename T>
inline void decrementParams(T *params, unsigned index, unsigned startCol,
                            unsigned endCol, unsigned paramsWidth) {
  if (__builtin_expect(index < paramsWidth, 1)) {
    if (__builtin_expect(startCol <= index && index < endCol, 0)) {
      params[index - startCol] = params[index - startCol] - static_cast<T>(1.f);
    }
  } else {
    if (index != MASKED_LABEL_CODE) {
      for (unsigned col = startCol; col != endCol; ++col) {
        params[col - startCol] = static_cast<T>(__builtin_nanf(""));
      }
    }
  }
}

} // namespace

namespace popops {

template <typename T>
class UpdateIntervalDEC : public Vertex {
  static_assert(std::is_same<T, float>() || std::is_same<T, half>(),
                "T must be a either float or half");

public:
  // Params slice.
  InOut<Vector<T, ONE_PTR>> params;
  // For each row spanned by the params, list the indices of the columns that
  // need to be updated.
  Input<Vector<unsigned, ONE_PTR>> indices;

  // For each row spanned by the intervals, report the starting index within the
  // interval.
  Vector<unsigned> rowsStart;
  // The width of the original 2D param matrix. Used for in-bounds checks.
  unsigned paramsWidth;
  // For the first row spanned by the intervals, report the starting column.
  // All other row segments are assumed to start at column 0.
  unsigned firstStartCol;
  // For the last row spanned by the intervals, report the end column.
  // All other row segments are assumed to end at column `paramsWidth`.
  unsigned lastEndCol;
  unsigned rowCount;

  bool compute() {
    // For each row spanned by the interval.
    for (unsigned i = 0; i < rowCount; ++i) {
      unsigned startCol = i == 0 ? firstStartCol : 0;
      unsigned endCol = (i == rowCount - 1) ? lastEndCol : paramsWidth;
      decrementParams(&params[rowsStart[i]], indices[i], startCol, endCol,
                      paramsWidth);
    }
    return true;
  }
};

template class UpdateIntervalDEC<float>;
template class UpdateIntervalDEC<half>;

template <typename T>
class UpdateIntervalsDEC : public Vertex {
  static_assert(std::is_same<T, float>() || std::is_same<T, half>(),
                "T must be a either float or half");

public:
  Vector<InOut<Vector<T, ONE_PTR>>> params;
  // For each row spanned by the params, list the indices of the columns that
  // need to be updated.
  Vector<Input<Vector<unsigned, ONE_PTR>>> indices;
  // For each row spanned by the intervals, report the starting index within the
  // interval.
  Vector<unsigned, ONE_PTR> rowsStart;
  // For the first row spanned by the intervals, report the starting column.
  // All other row segments are assumed to start at column 0.
  Vector<unsigned, ONE_PTR> firstStartCol;
  // For the last row spanned by the intervals, report the end column.
  // All other row segments are assumed to end at column `paramsWidth`.
  Vector<unsigned, ONE_PTR> lastEndCol;
  // For each interval report how many rows it spans.
  Vector<unsigned, ONE_PTR> rowCounts;
  // The width of the original 2D param matrix. Used for in-bounds checks.
  unsigned paramsWidth;

  bool compute() {
    unsigned counter = 0;
    // For each param chunk.
    for (unsigned p = 0; p < params.size(); ++p) {
      unsigned rowCount = rowCounts[p];
      // For each row spanned by the interval.
      for (unsigned i = 0; i < rowCount; ++i) {
        unsigned startCol = i == 0 ? firstStartCol[p] : 0;
        unsigned endCol = (i == rowCount - 1) ? lastEndCol[p] : paramsWidth;
        decrementParams(&params[p][rowsStart[counter]], indices[p][i], startCol,
                        endCol, paramsWidth);
        ++counter;
      }
    }
    return true;
  }
};

template class UpdateIntervalsDEC<float>;
template class UpdateIntervalsDEC<half>;

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

template <typename T> class
UpdateColumnsDEC : public Vertex {
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
