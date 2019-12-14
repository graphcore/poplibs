// Copyright (c) Graphcore Ltd, All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
namespace popops {

class CircBufIncrIndex : public Vertex {
public:
  CircBufIncrIndex();

  InOut<unsigned> index;
  const unsigned hSize;
  bool compute() {
    *index = (*index + 1) % hSize;
    return true;
  }
};

} // namespace popops
