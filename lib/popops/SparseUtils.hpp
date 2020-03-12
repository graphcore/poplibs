// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include <poplar/Interval.hpp>

#include <vector>

namespace popops {

// Throw an exception if condition is not true.
void expect(bool condition, std::string message);

// Assuming that `interval` spans a single row of a 2D tensor of width `width`,
// return an Interval of the form: [first column, last column).
poplar::Interval
getSingleRowIntervalColumnIndices(const poplar::Interval &interval,
                                  std::size_t width);

// Assuming that `interval` spans a single row of a 2D tensor of width `width`,
// return the row which the interval belongs to.
std::size_t getSingleRowIntervalRowIndex(const poplar::Interval &interval,
                                         std::size_t width);

// Assuming that `interval` spans mulitple rows of a 2D tensor of width `width`,
// Return the first row covered by the interval and append the bounds of the
// rows spanned the interval to `columnBounds`.
// Consider the matrix in the figure and the interval within the matrix.
//
//                    0           'x'     'y'    'width'
//                    +            +       +     +
//                    |            |       |     |
//                    v            v       v     v
//
//                    +------------+-------+-----+
//                    |                          |
//                    |            +-------------+
//  Row 'r'  +---->   |            |             |
//                    +------------+-------------+
//                    |                          |
//                    +--------------------+-----+
//                    |                    |     |
//                    +--------------------+     |
//                    |                          |
//                    |                          |
//                    +--------------------------+
//
// The function will return 'r' and append {[x, width), [0, width), [0, y)} to
// `columnBounds`.
std::size_t getBounds(const poplar::Interval &interval, std::size_t width,
                      std::vector<poplar::Interval> &columnBounds);

// Utility aliases, a region is a sequence of intervals that span a contiguous
// chunk of tile memory.
using Region = std::vector<poplar::Interval>;
using Regions = std::vector<Region>;

// Return a 2D description of `region`, this assumes that `region` belongs to a
// 2D tensor of width `width`.
// Suppose we are given a region made of the following intervals (each letter is
// an interval) {A, B, C, D, E, F, G, H, I, J, K, L}. The matrix multiplication
// algorithm often lays out memory like so.
//
// + Parent tensor
// |
// |
// |            Start of region                       width
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
// This function will return the height and width of the region in terms of
// intervals. {4, 3} in this case. Also, columnWidths will be filled with the
// width of each of the columns of the region: {A.size(), E.size(), I.size()}.

std::pair<unsigned, unsigned>
getRegionBounds(const Region &region, std::size_t width,
                std::vector<unsigned> &columnWidths);

// Check that the intervals within a region follow the required pattern to
// allow the createColumnsVertex to be used

bool checkRegionShapes(const Regions &tileRegions, std::size_t width);
} // namespace popops
