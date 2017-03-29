#ifndef __popnn_NonLinearity_hpp__
#define __popnn_NonLinearity_hpp__
#include "poplar/Program.hpp"

namespace popnn {

enum NonLinearityType {
  NON_LINEARITY_SIGMOID,
  NON_LINEARITY_RELU,
  NON_LINEARITY_TANH
};

#ifndef __POPC__

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

#endif // !__POPC__

}

#endif // __popnn_NonLinearity_hpp__
