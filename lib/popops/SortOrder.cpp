// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popops/SortOrder.hpp>
#include <poputil/exceptions.hpp>

#include <ostream>

namespace popops {

std::ostream &operator<<(std::ostream &os, const SortOrder &o) {
  switch (o) {
  case SortOrder::NONE:
    os << "none";
    break;
  case SortOrder::ASCENDING:
    os << "ascending";
    break;
  case SortOrder::DESCENDING:
    os << "descending";
    break;
  default:
    throw poputil::poplibs_error("Unhandled sort order");
  }
  return os;
}

} // end namespace popops
