#include "ConvReuse.hpp"
#include <tuple>

bool ConvImplSpec::operator<(const ConvImplSpec &other) const {
  auto t1 = std::make_tuple(inNumChans, inNumChanGroups, inDimX, inDimY,
                            outNumChans, outNumChanGroups, outDimX, outDimY,
                            resNumChans, resNumChanGroups, resDimX, resDimY,
                            kernelSize, stride, padding, nonLinearityType,
                            resMethod);
  auto t2 = std::make_tuple(other.inNumChans, other.inNumChanGroups,
                            other.inDimX, other.inDimY,
                            other.outNumChans, other.outNumChanGroups,
                            other.outDimX, other.outDimY,
                            other.resNumChans, other.resNumChanGroups,
                            other.resDimX, other.resDimY,
                            other.kernelSize, other.stride, other.padding,
                            other.nonLinearityType, other.resMethod);
  return t1 < t2;
}
