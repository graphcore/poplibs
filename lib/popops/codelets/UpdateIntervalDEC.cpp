// Copyright (c) Graphcore Ltd, All rights reserved.
#include "SelectScalarFromRows.hpp"

using namespace poplar;

namespace popops {

template <typename T> class UpdateIntervalDEC : public Vertex {
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

} // namespace popops
