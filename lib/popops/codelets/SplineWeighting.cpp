// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <poplar/Vertex.hpp>
using namespace poplar;

namespace popops {

template <typename FPType> class SplineWeighting : public Vertex {
public:
  SplineWeighting();
  Input<Vector<FPType, VectorLayout::ONE_PTR>> input;
  Input<Vector<FPType, VectorLayout::ONE_PTR>> weight;
  Input<Vector<FPType, VectorLayout::ONE_PTR>> basis;
  Input<Vector<int, VectorLayout::ONE_PTR>> weightIndex;
  Vector<Output<Vector<FPType>>> output;

  const unsigned numInCh;
  const unsigned numOutCh;
  const unsigned numSplines;
  const Vector<unsigned> offsets;
  const unsigned edgeOffset;

  void compute() {
    for (unsigned i = 0; i < output.size(); ++i) {
      const auto offset = offsets[i];
      for (unsigned j = 0; j < output[i].size(); ++j) {
        const unsigned idx = j + offset;
        const unsigned e = idx / numOutCh - edgeOffset;
        const unsigned oc = idx % numOutCh;
        FPType v = static_cast<FPType>(0.);

        for (unsigned s = 0; s < numSplines; s++) {
          FPType b = basis[e * numSplines + s];
          unsigned wi = weightIndex[e * numSplines + s];
          for (unsigned ic = 0; ic < numInCh; ic++) {
            FPType tmp = weight[wi * numInCh * numOutCh + ic * numOutCh + oc];
            tmp *= b * input[e * numInCh + ic];
            v += tmp;
          }
        }

        output[i][j] = v;
      }
    }
  }
};

template class SplineWeighting<float>;
template class SplineWeighting<half>;
} // namespace popops
