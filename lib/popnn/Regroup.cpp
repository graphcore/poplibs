#include "Regroup.hpp"
#include "gcd.hpp"
#include "VertexTemplates.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;

Tensor
regroup(Tensor in, unsigned outChansPerGroup) {
  auto rank = in.rank();
  assert(rank == 4 || rank == 5);
  if (rank == 4)
    in = in.reshape({1, in.dim(0), in.dim(1), in.dim(2), in.dim(3)});

  const auto batchSize = in.dim(0);
  const auto inNumChanGroups = in.dim(1);
  const auto dimY = in.dim(2);
  const auto dimX = in.dim(3);
  const auto inChansPerGroup = in.dim(4);
  const auto numChans = inNumChanGroups * inChansPerGroup;
  assert(numChans % outChansPerGroup == 0);
  const auto outNumChanGroups = numChans / outChansPerGroup;
  Tensor regrouped = in.dimShuffle({0, 2, 3, 1, 4})
           .reshape({batchSize, dimY, dimX, outNumChanGroups, outChansPerGroup})
           .dimShuffle({0, 3, 1, 2, 4});
  if (rank == 5)
    return regrouped;
  else //rank == 4 so unbatched
    return regrouped.reshape({regrouped.dim(1), regrouped.dim(2),
                              regrouped.dim(3), regrouped.dim(4)});
}
