// Copyright (c) Graphcore Ltd, All rights reserved.
#include "NonLinearity.hpp"

namespace popnn {

template <typename FPType, NonLinearityType nlType>
class NonLinearity2D : public Vertex {
public:
  NonLinearity2D();

  InOut<VectorList<FPType, VectorListLayout::DELTAN>> data;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      for (unsigned j = 0; j < data[i].size(); ++j) {
        data[i][j] = FPType(nonlinearity(nlType, float(data[i][j])));
      }
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearity2D)

} // namespace popnn
