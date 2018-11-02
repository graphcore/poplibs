// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_NonLinearity_hpp
#define popnn_NonLinearity_hpp

#include <popnn/NonLinearityDef.hpp>

#ifndef __POPC__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {

#define DEF_NONLINEARITY_INPLACE(fn, nlType)                                  \
  inline void fn ## InPlace(poplar::Graph &graph,                             \
                                      poplar::Tensor t,                       \
                                      poplar::program::Sequence &prog,        \
                                      const std::string &debugPrefix = "") {  \
    nonLinearityInPlace(graph, nlType, t, prog, debugPrefix);                 \
  }

#define DEF_NONLINEARITY_(fn, nlType)                                         \
  inline poplar::Tensor fn(poplar::Graph &graph,                              \
                           poplar::Tensor t,                                  \
                           poplar::program::Sequence &prog,                   \
                           const std::string &debugPrefix = "") {             \
    return nonLinearity(graph, nlType, t, prog, debugPrefix);                 \
  }



#define DEF_NONLINEARITY(fn, nlType) \
  DEF_NONLINEARITY_INPLACE(fn, nlType) \
  DEF_NONLINEARITY_(fn, nlType)

// Update tensor t in place by applying a non-linearity
// For SOFTMAX nonlinearity type, the soft max is done over the innermost
// dimension
void
nonLinearityInPlace(poplar::Graph &graph, NonLinearityType nonLinearityType,
                    poplar::Tensor t, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

void
nonLinearityInPlace(poplar::Graph &graph, NonLinearityType nonLinearityType,
                    poplar::Tensor t, poplar::ComputeSet &cs,
                    const std::string &debugPrefix = "");

poplar::Tensor
nonLinearity(poplar::Graph &graph, NonLinearityType nonLinearityType,
             poplar::Tensor t, poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

DEF_NONLINEARITY(sigmoid, NonLinearityType::SIGMOID);
DEF_NONLINEARITY(relu, NonLinearityType::RELU);
DEF_NONLINEARITY(tanh, NonLinearityType::TANH);
DEF_NONLINEARITY(softmax, NonLinearityType::SOFTMAX);
DEF_NONLINEARITY(softmaxStable, NonLinearityType::SOFTMAX_STABLE);

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
