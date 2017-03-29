#ifndef __popstd_Regroup_hpp__
#define __popstd_Regroup_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

namespace popstd {

poplar::Tensor regroup(poplar::Tensor t, unsigned outerDim, unsigned innerDim,
                       unsigned newGroupSize);

poplar::Tensor regroup(poplar::Tensor in, unsigned outChansPerGroup);

} // end namespace popstd

#endif //__popstd_Regroup_hpp__
