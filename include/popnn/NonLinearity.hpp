// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_NonLinearity_hpp
#define popnn_NonLinearity_hpp

namespace popnn {

enum NonLinearityType {
  NON_LINEARITY_SIGMOID,
  NON_LINEARITY_RELU,
  NON_LINEARITY_TANH,
  NON_LINEARITY_SOFTMAX
};

} // end namespace popnn

#ifndef __POPC__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {

// Update tensor t in place by applying a non-linearity
// For SOFTMAX nonlinearity type, the soft max is done over the innermost
// dimension
void nonLinearity(poplar::Graph &graph, NonLinearityType nonLinearityType,
                  poplar::Tensor t, poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");

void nonLinearity(poplar::Graph &graph, NonLinearityType nonLinearityType,
                  poplar::Tensor t, poplar::ComputeSet &cs,
                  const std::string &debugPrefix = "");

inline void sigmoid(poplar::Graph &graph,
                    poplar::Tensor t, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "") {
  nonLinearity(graph, NON_LINEARITY_SIGMOID, t, prog, debugPrefix);
}

inline void relu(poplar::Graph &graph,
                 poplar::Tensor t, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "") {
  nonLinearity(graph, NON_LINEARITY_RELU, t, prog, debugPrefix);
}

inline void tanh(poplar::Graph &graph,
                 poplar::Tensor t, poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "") {
  nonLinearity(graph, NON_LINEARITY_TANH, t, prog, debugPrefix);
}

// For softmax, the softmax is done over the innermost dimension
inline void softmax(poplar::Graph &graph,
                    poplar::Tensor t, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "") {
  nonLinearity(graph, NON_LINEARITY_SOFTMAX, t, prog, debugPrefix);
}

poplar::Tensor
nonLinearityInputGradient(poplar::Graph &graph,
                          NonLinearityType nonLinearityType,
                          poplar::Tensor out, poplar::Tensor outGradient,
                          poplar::ComputeSet &cs,
                          const std::string &debugPrefix = "");

poplar::Tensor
nonLinearityInputGradient(poplar::Graph &graph,
                          NonLinearityType nonLinearityType,
                          poplar::Tensor out, poplar::Tensor outGradient,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "");

} // end namespace popnn

#endif // !__POPC__

#endif // popnn_NonLinearity_hpp
