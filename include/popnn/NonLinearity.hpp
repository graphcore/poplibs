#ifndef __NonLinearity_hpp__
#define __NonLinearity_hpp__
#include "popnn/NonLinearityDef.hpp"
#include "poplar/Program.hpp"

// Update tensor t in place by applying a non-linearity
void nonLinearity(poplar::Graph &graph, NonLinearityType nonLinearityType,
                  poplar::Tensor t, poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");

poplar::Tensor
nonLinearityInputGradient(poplar::Graph &graph,
                          NonLinearityType nonLinearityType,
                          poplar::Tensor out, poplar::Tensor outGradient,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "");

#endif // __NonLinearity_hpp__
