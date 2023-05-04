// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "NonLinearity.hpp"

using namespace poplar;

namespace popnn {

template <typename FPType, NonLinearityType nlType>
class NonLinearityGrad1D : public MultiVertex {
public:
  NonLinearityGrad1D();

  Input<Vector<FPType, ONE_PTR, 8>> outGrad;
  Input<Vector<FPType, ONE_PTR, 8>> out;
  Output<Vector<FPType, ONE_PTR, 8>> inGrad;
  const unsigned short n;

  IS_EXTERNAL_CODELET(nlType != NonLinearityType::GELU_ERF);
  void compute(unsigned wid) {
    if (wid == 0) {
      for (unsigned i = 0; i < n; ++i) {
        const auto derivative = nonlinearity_derivative(nlType, float(out[i]));
        inGrad[i] = outGrad[i] * FPType(derivative);
      }
    }
  }
};

INSTANTIATE_NL_GRAD(NonLinearityGrad1D)

} // namespace popnn
