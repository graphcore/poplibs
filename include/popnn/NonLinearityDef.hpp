// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_NonLinearityDef_hpp
#define popnn_NonLinearityDef_hpp

namespace popnn {

enum class NonLinearityType {
  SIGMOID,
  RELU,
  TANH,
  SOFTMAX
};

} // end namespace popnn

#endif // popnn_NonLinearityDef_hpp
