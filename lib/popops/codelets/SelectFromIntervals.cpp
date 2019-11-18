#include "SelectScalarFromRows.hpp"

namespace popops {

template <typename T> class SelectFromIntervals : public Vertex {
  static_assert(std::is_same<T, float>() || std::is_same<T, half>(),
                "T must be a either float or half");

public:
  Vector<Input<Vector<T, ONE_PTR>>> params;
  // For each row spanned by the params, list the indices of the columns that
  // need to be updated.
  Vector<Input<Vector<unsigned, ONE_PTR>>, ONE_PTR> indices;
  // For each row spanned by the params, have an output value.
  Vector<Output<Vector<T, ONE_PTR>>, ONE_PTR> output;
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
      for (unsigned r = 0; r < rowCount; ++r) {
        unsigned startCol = r == 0 ? firstStartCol[p] : 0;
        unsigned endCol = (r == rowCount - 1) ? lastEndCol[p] : paramsWidth;
        output[p][r] = getParam(&params[p][rowsStart[counter]], indices[p][r],
                                startCol, endCol, paramsWidth);
        ++counter;
      }
    }
    return true;
  }
};

template class SelectFromIntervals<float>;
template class SelectFromIntervals<half>;

} // namespace popops
