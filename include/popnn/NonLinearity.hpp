#ifndef __NonLinearity_hpp__
#define __NonLinearity_hpp__
#include "popnn/NonLinearityDef.hpp"
#include "poplar/Program.hpp"

poplar::program::Program
bwdNonLinearity(poplar::Graph &graph,
                poplar::Tensor activations, poplar::Tensor deltasIn,
                poplar::Tensor zDeltas,
                NonLinearityType nonLinearityType);

#endif // __NonLinearity_hpp__
