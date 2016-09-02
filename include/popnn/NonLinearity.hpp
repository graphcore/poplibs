#ifndef __NonLinearity_hpp__
#define __NonLinearity_hpp__
#include "popnn/NonLinearityDef.hpp"
#include "poplar/Program.hpp"

poplar::program::Program
bwdNonLinearity(poplar::Graph &graph,
                poplar::Tensor activations, poplar::Tensor deltasIn,
                poplar::Tensor zDeltas,
                NonLinearityType nonLinearityType);

extern float nonlinearity(NonLinearityType t, float x);
extern float nonlinearity_derivative(NonLinearityType t, float activation);



#endif // __NonLinearity_hpp__
