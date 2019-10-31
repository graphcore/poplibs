// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_NonLinearityDef_hpp
#define popnn_NonLinearityDef_hpp

namespace popnn {

enum class NonLinearityType {
  SIGMOID,
  RELU,
  TANH,
  GELU,
  SOFTMAX,
  SOFTMAX_STABLE, ///< Slower but more numerically stable algorithm.
  SOFTMAX_SCALED  ///< Stable and scaled by SOFTMAX_SCALING for better accuracy.
};

} // end namespace popnn

#endif // popnn_NonLinearityDef_hpp
