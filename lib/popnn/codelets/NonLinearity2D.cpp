// Copyright (c) Graphcore Ltd, All rights reserved.
#include "NonLinearity.hpp"

using namespace poplar;

namespace popnn {

template <typename FPType, NonLinearityType nlType>
class NonLinearity2D : public Vertex {
public:
  NonLinearity2D();

#if defined(VECTORLIST_AVAIL_DELTAN)
  InOut<VectorList<FPType, DELTAN>> data;
#else
  InOut<VectorList<FPType, DELTANELEMENTS>> data;
#endif

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
