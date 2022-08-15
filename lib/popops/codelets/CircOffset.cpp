// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;
namespace popops {

class CircOffset : public Vertex {
public:
  CircOffset();

  Input<unsigned> indexIn;
  Output<unsigned> indexOut;
  const unsigned hSize;
  const unsigned offset;
  void compute() {
    auto updated = *indexIn + offset;
    if (updated >= hSize) {
      updated -= hSize;
    }
    *indexOut = updated;
  }
};

} // namespace popops
