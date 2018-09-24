#include <poputil/Broadcast.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;

void poputil::broadcastToMatch(Tensor &a,
                               const std::vector<std::size_t> &shape) {
  auto rank = shape.size();

  if (rank < a.rank())
      throw poputil::poplib_error(
             "Cannot broadcast tensor to match shape"
            );

  // First expand with singleton dimensions to match rank.
  if (a.rank() < rank){
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
      throw poputil::poplib_error(
             "Cannot broadcast tensors to match dimension " + std::to_string(i)
            );
    }
  }
}


void poputil::broadcastToMatch(Tensor &a, Tensor &b) {
  // First expand with singleton dimensions to match ranks.
  if (a.rank() < b.rank()) {
    const auto N = b.rank() - a.rank();
    for (unsigned i = 0; i < N; ++i)
      a = a.expand({0});
  }

  if (b.rank() < a.rank()) {
    const auto N = a.rank() - b.rank();
    for (unsigned i = 0; i < N; ++i)
      b = b.expand({0});
  }

  auto rank = a.rank();

  for (unsigned i = 0; i < rank; ++i) {
    if (a.dim(i) == b.dim(i))
      continue;

    if (a.dim(i) == 1) {
      a = a.broadcast(b.dim(i), i);
    } else if (b.dim(i) == 1) {
      b = b.broadcast(a.dim(i), i);
    } else {
      throw poputil::poplib_error(
             "Cannot broadcast tensors to match dimension " + std::to_string(i)
            );
    }
  }
}
