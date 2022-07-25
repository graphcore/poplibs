// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cmath>
#include <cstdlib>

using namespace poplar;

static constexpr auto SPAN = VectorLayout::SPAN;

namespace popops {
static bool check(float data, bool checkNaNAndInf) {
  return std::isnan(data) || (checkNaNAndInf && std::isinf(data));
}
// The second template parameter
// nanOrInf = true  : Detect if either NaN or Inf is present
// nanOrInf = false : Detect only NaN
template <typename FPType, bool nanOrInf> class HasNaNOrInf2D : public Vertex {
public:
  HasNaNOrInf2D();
  IS_EXTERNAL_CODELET(true);
  Vector<Input<Vector<FPType, SPAN, 8>>> in;
  Output<float> out;

  bool compute() {
    constexpr bool checkNaNAndInf = nanOrInf;
    for (unsigned i = 0; i < in.size(); ++i) {
      for (unsigned j = 0; j < in[i].size(); ++j) {
        if (check(float(in[i][j]), checkNaNAndInf)) {
          *out = 1.0f;
          return true;
        }
      }
    }
    *out = 0.0f;
    return true;
  }
};

template class HasNaNOrInf2D<float, true>;
template class HasNaNOrInf2D<float, false>;
template class HasNaNOrInf2D<half, true>;
template class HasNaNOrInf2D<half, false>;

// The second template parameter
// nanOrInf = true  : Detect if either NaN or Inf is present
// nanOrInf = false : Detect only NaN
// This multivertex only writes '1.0' to `outSetIfFound` when NaN/Inf is
// detected; that Tensor must be zeroed before this vertex is called.
template <typename FPType, bool nanOrInf>
class HasNaNOrInf1D : public MultiVertex {
public:
  HasNaNOrInf1D();
  IS_EXTERNAL_CODELET(true);
  Input<Vector<FPType, VectorLayout::ONE_PTR, 8>> in;
  unsigned short sizeIn8BytesPerWorker;
  unsigned char remWorkerId;
  unsigned char remWorkerExtras;
  InOut<float> outSetIfFound;
  bool compute(unsigned wid) {
    if (wid == 0) {
      unsigned size = sizeIn8BytesPerWorker * numWorkers() + remWorkerId;
      size = size * (std::is_same<FPType, half>() ? 4 : 2) + remWorkerExtras;
      constexpr bool checkNaNAndInf = nanOrInf;
      for (unsigned i = 0; i < size; ++i) {
        if (check(float(in[i]), checkNaNAndInf)) {
          *outSetIfFound = 1.0f;
          return true;
        }
      }
    }
    return true;
  }
};

template class HasNaNOrInf1D<float, true>;
template class HasNaNOrInf1D<float, false>;
template class HasNaNOrInf1D<half, true>;
template class HasNaNOrInf1D<half, false>;

} // namespace popops
