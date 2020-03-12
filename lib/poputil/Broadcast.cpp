// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poputil/Broadcast.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;

namespace poputil {

void expandToMatchRanks(Tensor &a, Tensor &b) {
  if (a.rank() < b.rank()) {
    auto difference = b.rank() - a.rank();
    a = a.expand(std::vector<std::size_t>(difference, 0));
  } else if (b.rank() < a.rank()) {
    auto difference = a.rank() - b.rank();
    b = b.expand(std::vector<std::size_t>(difference, 0));
  }
}

void broadcastToMatch(Tensor &a, const std::vector<std::size_t> &shape) {
  auto rank = shape.size();

  if (rank < a.rank())
    throw poputil::poplibs_error("Cannot broadcast tensor to match shape");

  // First expand with singleton dimensions to match rank.
  if (a.rank() < rank) {
    const auto N = a.rank();
    for (unsigned i = 0; i < rank - N; ++i)
      a = a.expand({0});
  }

  for (unsigned i = 0; i < rank; ++i) {
    if (a.dim(i) == shape[i])
      continue;

    if (a.dim(i) == 1) {
      a = a.broadcast(shape[i], i);
    } else {
      throw poputil::poplibs_error(
          "Cannot broadcast tensors to match dimension " + std::to_string(i));
    }
  }
}

void broadcastToMatch(Tensor &a, Tensor &b) {
  expandToMatchRanks(a, b);

  auto rank = a.rank();
  for (unsigned i = 0; i < rank; ++i) {
    if (a.dim(i) == b.dim(i))
      continue;

    if (a.dim(i) == 1) {
      a = a.broadcast(b.dim(i), i);
    } else if (b.dim(i) == 1) {
      b = b.broadcast(a.dim(i), i);
    } else {
      throw poputil::poplibs_error(
          "Cannot broadcast tensors to match dimension " + std::to_string(i));
    }
  }
}

void broadcastToMatch(Tensor &a, Tensor &b, Tensor &c) {
  broadcastToMatch(a, b);
  broadcastToMatch(b, c);
  broadcastToMatch(c, a);
}

bool canBroadcastToMatch(const Tensor &a, const Tensor &b) {
  std::size_t aRankDefecit = a.rank() < b.rank() ? b.rank() - a.rank() : 0;
  std::size_t bRankDefecit = b.rank() < a.rank() ? a.rank() - b.rank() : 0;
  auto rank = a.rank() + aRankDefecit;
  for (std::size_t d = 0; d < rank; ++d) {
    auto aDim = d < aRankDefecit ? 1 : a.dim(d - aRankDefecit);
    auto bDim = d < bRankDefecit ? 1 : b.dim(d - bRankDefecit);
    if (aDim == bDim ||              // Dimensions match or
        aDim * bDim < aDim + bDim) { // one or both dimensions are singular.
      continue;
    }
    return false;
  }
  return true;
}

} // end namespace poputil
