#include "Regroup.hpp"
#include "gcd.hpp"
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

Tensor
regroup(Tensor in, unsigned outChansPerGroup) {
  const auto inNumChanGroups = in.dim(0);
  const auto dimY = in.dim(1);
  const auto dimX = in.dim(2);
  const auto inChansPerGroup = in.dim(3);
  const auto numChans = inNumChanGroups * inChansPerGroup;
  assert(numChans % outChansPerGroup == 0);
  const auto outNumChanGroups = numChans / outChansPerGroup;
  return in.dimShuffle({1, 2, 0, 3})
           .reshape({dimY, dimX, outNumChanGroups, outChansPerGroup})
           .dimShuffle({2, 0, 1, 3});
}
