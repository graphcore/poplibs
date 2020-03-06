// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include "SparseUtils.hpp"

#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include <vector>

using namespace poplar;
using namespace poputil;

namespace {

std::size_t nextLineEnd(std::size_t point, std::size_t width) {
  return point + (width - point % width);
}

} // namespace

namespace popops {

void expect(bool condition, std::string message) {
  if (!condition) {
    throw poputil::poplibs_error(message);
  }
}

Interval getSingleRowIntervalColumnIndices(const Interval &interval,
                                           std::size_t width) {
  assert(interval.begin() / width == (interval.end() - 1) / width);
  std::size_t begin = interval.begin() % width;
  std::size_t end = interval.end() % width;
  if (end == 0) {
    end = width;
  }
  return Interval(begin, end);
}

std::size_t getSingleRowIntervalRowIndex(const Interval &interval,
                                         std::size_t width) {
  assert(interval.begin() / width == (interval.end() - 1) / width);
  return interval.begin() / width;
}

std::size_t getBounds(const Interval &interval, std::size_t width,
                      std::vector<Interval> &bounds) {
  assert(interval.size() > 0 && "Interval must be non-empty");
  std::size_t startRow = interval.begin() / width;
  std::size_t endRow = (interval.end() - 1) / width;
  long int spannedRows = endRow - startRow + 1;
  assert(spannedRows >= 1);
  std::size_t startCol = interval.begin() % width;
  std::size_t endCol = interval.end() % width;
  if (endCol == 0) {
    endCol = width;
  }
  if (spannedRows == 1) {
    bounds.emplace_back(startCol, endCol);
    return startRow;
  }
  for (int i = 0; i < spannedRows; ++i) {
    if (i == spannedRows - 1) {
      bounds.emplace_back(startCol, endCol);
    } else {
      bounds.emplace_back(startCol, nextLineEnd(startCol, width));
    }
    startCol = 0;
  }

  return startRow;
}

std::pair<unsigned, unsigned>
getRegionBounds(const Region &region, std::size_t width,
                std::vector<unsigned> &columnWidths) {
  if (region.empty()) {
    return {0, 0};
  }

  int regionWidth = 1;

  Interval columnBounds =
      getSingleRowIntervalColumnIndices(region.front(), width);
  columnWidths.push_back(columnBounds.size());

  for (const Interval &interval : region) {
    Interval newBounds = getSingleRowIntervalColumnIndices(interval, width);
    if (columnBounds != newBounds) {
      regionWidth++;
      columnBounds = newBounds;
      columnWidths.push_back(newBounds.size());
    }
  }

  assert(region.size() % regionWidth == 0);

  return {region.size() / regionWidth, regionWidth};
}
// checkRegionShapes is given a vector of tensor regions which we expect to be
// in order A,B,C,D,E... belonging to rows and columns as shown below:
// +----------+-------+------+
// |    A     |   E   |  I   |
// +-------------------------+
// |    B     |   F   |  J   |
// +-------------------------+
// |    C     |   G   |  K   |
// +-------------------------+
// |    D     |   H   |  L   |
// +-------------------------+
//
// In other words the start and end of column A,B,C,D is the same,
// the start and end of E,F,G,H is the same etc.
// Also that A,E,I belong to the same row of the tensor, B,F,J
// belong to the same row of the tensor etc.  A,B,C,D do not have to be on
// consecutive rows.
//
// We go through the intervals in order A,B,C,D,E, ....
// The algorithm identifies the head of each column (A is a head of column so
// B,C,D must have the same start, end).  Likewise E is the head
// of a column which needs to have the same start, end as F,G,H.
//
// For the first column A,B,C,D we gather a vector containing the row that A is
// on, and the same for B,C,D.  Then E must be on the same row as A, F on the
// same row as B etc.

bool checkRegionShapes(const Regions &tileRegions, std::size_t width) {
  // Multiple regions to check - outer loop
  for (auto I = tileRegions.cbegin(), E = tileRegions.cend(); I != E; ++I) {
    // *I is a single contiguous region containing the intervals A,B... as above
    const auto &intervals = *I;
    bool firstColumn = true;
    unsigned rowIndex = 0;
    std::vector<std::size_t> firstColumnRows;
    Interval currentColumnHead;
    // Loop over the intervals in a region
    for (const Interval &interval : intervals) {
      std::size_t startRow = interval.begin() / width;
      std::size_t endRow = (interval.end() - 1) / width;
      if (endRow != startRow) {
        return false;
      } else {
        Interval thisIntervalColumn(interval.begin() % width,
                                    (interval.end() - 1) % width);
        // Check against the head of each column
        if (rowIndex != 0) {
          if (currentColumnHead != thisIntervalColumn) {
            if (firstColumn) {
              // Completed gathering 1st column and checking it
              firstColumn = false;
              rowIndex = 0;
            } else {
              if (rowIndex == firstColumnRows.size()) {
                // Another column complete
                rowIndex = 0;
              } else {
                // Inconsistent shape detected
                return false;
              }
            }
          }
        }
        if (rowIndex == 0) {
          // Gather detail of the head of each column
          currentColumnHead = thisIntervalColumn;
        }
        if (firstColumn) {
          // Gather detail of the first column
          firstColumnRows.push_back(startRow);
        }
        if (!firstColumn) {
          // Check for row match against the first column
          if (firstColumnRows[rowIndex] != startRow) {
            return false;
          }
        }
        rowIndex++;
      }
    }
    // If there is more than one column, check that the last one is complete
    if ((!firstColumn) && rowIndex != firstColumnRows.size()) {
      return false;
    }
  }
  return true;
}
} // namespace popops
