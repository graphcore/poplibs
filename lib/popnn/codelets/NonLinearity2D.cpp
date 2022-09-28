// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "NonLinearity.hpp"

using namespace poplar;

namespace popnn {

template <typename FPType, NonLinearityType nlType>
class NonLinearity2DInPlace : public Vertex {
public:
  NonLinearity2DInPlace();

  InOut<VectorList<FPType, DELTANELEMENTS>> data;

  IS_EXTERNAL_CODELET(true);
  void compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      for (unsigned j = 0; j < data[i].size(); ++j) {
        data[i][j] = FPType(nonlinearity(nlType, float(data[i][j])));
      }
    }
  }
};

INSTANTIATE_NL(NonLinearity2DInPlace)

template <typename FPType, NonLinearityType nlType>
class NonLinearity2D : public Vertex {
public:
  NonLinearity2D();

  Input<VectorList<FPType, DELTANELEMENTS, 8>> data;
  Vector<Output<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> out;

  IS_EXTERNAL_CODELET(true);
  void compute() {
    for (unsigned i = 0; i < data.size(); ++i) {
      for (unsigned j = 0; j < data[i].size(); ++j) {
        out[i][j] = FPType(nonlinearity(nlType, float(data[i][j])));
      }
    }
  }
};

template class NonLinearity2D<float, popnn::NonLinearityType::SWISH>;
template class NonLinearity2D<half, popnn::NonLinearityType::SWISH>;

} // namespace popnn
