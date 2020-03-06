// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "SelectScalarFromRows.hpp"

using namespace poplar;

namespace popops {

template <typename T> class SelectFromInterval : public Vertex {
  static_assert(std::is_same<T, float>() || std::is_same<T, half>(),
                "T must be a either float or half");

public:
  Input<Vector<T, ONE_PTR>> params;
  // For each row spanned by the params, list the indices of the columns that
  // need to be selected.
  Input<Vector<unsigned, ONE_PTR>> indices;
  // For each row spanned by the params, have an output value.
  Output<Vector<T, ONE_PTR>> output;

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
  // Number of rows spanned by the interval.
  unsigned rowCount;

  bool compute() {
    // For each row spanned by the interval.
    for (unsigned r = 0; r < rowCount; ++r) {
      unsigned startCol = r == 0 ? firstStartCol : 0;
      unsigned endCol = (r == rowCount - 1) ? lastEndCol : paramsWidth;
      output[r] = getParam(&params[rowsStart[r]], indices[r], startCol, endCol,
                           paramsWidth);
    }
    return true;
  }
};

template class SelectFromInterval<float>;
template class SelectFromInterval<half>;

} // namespace popops
