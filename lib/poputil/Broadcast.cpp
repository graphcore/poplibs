#include <poputil/Broadcast.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;

void poputil::broadcastToMatch(Tensor &a,
                               const std::vector<std::size_t> &shape) {
  auto rank = shape.size();

  if (rank < a.rank())
      throw poputil::poplibs_error(
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
      throw poputil::poplibs_error(
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
      throw poputil::poplibs_error(
             "Cannot broadcast tensors to match dimension " + std::to_string(i)
            );
    }
  }
}


Tensor poputil::extendDimensionsToMatch(Tensor in1, Tensor in2) {
  if (in1.rank() < in2.rank())
      throw poputil::poplibs_error(
             "Cannot extend tensor dimensions to match"
            );

  if(in1.rank() != in2.rank()) {
    std::vector<std::size_t> extraDims(in1.rank() - in2.rank(), 0);
    return in2.expand(extraDims);
  }
  else {
    return in2;
  }
}


bool poputil::detectVectorBroadcastOperands(Tensor in1, Tensor in2) {
 if (in1.rank() < in2.rank())
      throw poputil::poplibs_error(
             "Tensor ranks are incompatible with vector broadcast operations"
            );
  auto in2Extend = extendDimensionsToMatch(in1, in2);
  unsigned count = 0;
  for(unsigned i = 0; i < in1.rank(); i++) {
    if(in2Extend.dim(i) != 1 ) {
      if(in1.dim(i) == in2Extend.dim(i)) {
        count++;
      }
      else {
        return false;
      }
    }
  }
  return count == 1;
}
