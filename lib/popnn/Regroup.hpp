#ifndef __Regroup_hpp__
#define __Regroup_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

poplar::Tensor regroup(poplar::Tensor in, unsigned outChansPerGroup);
#endif //__Regroup_hpp__
