#include "ConvReuse.hpp"
#include <tuple>
#include <cassert>

bool ConvImplSpec::operator<(const ConvImplSpec &other) const {
  auto t1 = std::tie(tensorDims, kernelSize, stride, padding, nonLinearityType,
                     resMethod);
  auto t2 = std::tie(other.tensorDims, other.kernelSize, other.stride,
                     other.padding, other.nonLinearityType, other.resMethod);
  return t1 < t2;
}
