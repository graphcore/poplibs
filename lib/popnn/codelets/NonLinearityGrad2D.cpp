// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "NonLinearity.hpp"

using namespace poplar;

namespace popnn {

template <typename FPType, NonLinearityType nlType>
class NonLinearityGrad2D : public Vertex {
public:
  NonLinearityGrad2D();

#if defined(VECTOR_AVAIL_SCALED_PTR64) && defined(VECTORLIST_AVAIL_DELTAN)
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, ONE_PTR> outGrad;
  Vector<Input<Vector<FPType, SCALED_PTR64, 8>>, ONE_PTR> out;
  Output<VectorList<FPType, DELTAN, 8>> inGrad;
#else
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> outGrad;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> out;
  Output<VectorList<FPType, DELTANELEMENTS, 8>> inGrad;
#endif

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < inGrad.size(); ++i) {
      for (unsigned j = 0; j < inGrad[i].size(); ++j) {
        const auto derivative =
            nonlinearity_derivative(nlType, float(out[i][j]));
        inGrad[i][j] = outGrad[i][j] * FPType(derivative);
      }
    }
    return true;
  }
};

INSTANTIATE_NL_GRAD(NonLinearityGrad2D)

} // namespace popnn
