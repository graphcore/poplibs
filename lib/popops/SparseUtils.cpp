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

std::pair<unsigned, unsigned> getRegionBounds(
                              const Region &region,
                              std::size_t width,
                              std::vector<unsigned> &columnWidths) {
  if (region.empty()) {
    return {0, 0};
  }

  int regionWidth = 1;

  Interval columnBounds = getSingleRowIntervalColumnIndices(region.front(),
                                                            width);
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

} // namespace popops
