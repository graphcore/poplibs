// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "NonLinearity.hpp"

#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

namespace popnn {

template <typename FPType, NonLinearityType nlType>
class NonLinearity1DInPlace : public MultiVertex {
public:
  NonLinearity1DInPlace();

  InOut<Vector<FPType, ONE_PTR>> data;
  const unsigned short n;

  IS_EXTERNAL_CODELET(true);
  void compute(unsigned wid) {
    if (wid == 0) {
      for (unsigned i = 0; i < n; ++i) {
        data[i] = nonlinearity(nlType, float(data[i]));
      }
    }
  }
};

INSTANTIATE_NL(NonLinearity1DInPlace)

template <typename FPType, NonLinearityType nlType>
class NonLinearity1D : public MultiVertex {
public:
  NonLinearity1D();

  Input<Vector<FPType, ONE_PTR, 8>> data;
  Output<Vector<FPType, ONE_PTR, 8>> out;
  const unsigned short n;

  IS_EXTERNAL_CODELET(true);
  void compute(unsigned wid) {
    if (wid == 0) {
      for (unsigned i = 0; i < n; ++i) {
        out[i] = nonlinearity(nlType, float(data[i]));
      }
    }
  }
};

template class NonLinearity1D<float, popnn::NonLinearityType::SWISH>;
template class NonLinearity1D<half, popnn::NonLinearityType::SWISH>;

} // namespace popnn
