// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef _popops_SortOrder_hpp_
#define _popops_SortOrder_hpp_

#include <iosfwd>

namespace popops {

/// Defines a required order for sorting operations.
enum class SortOrder {
  /// No ordering is required.
  NONE,
  /// Sort in ascending order.
  ASCENDING,
  /// Sort in descending order.
  DESCENDING
};

std::ostream &operator<<(std::ostream &os, const SortOrder &o);

} // end namespace popops

#endif // _popops_SortOrder_hpp_
