// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
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
template <typename FPType, bool nanOrInf> class HasNaNOrInf : public Vertex {
public:
  HasNaNOrInf();
  IS_EXTERNAL_CODELET(true);
  Vector<Input<Vector<FPType, SPAN, 8>>> in;

  bool compute() {
    constexpr bool checkNaNAndInf = nanOrInf;
    for (unsigned i = 0; i < in.size(); ++i) {
      for (unsigned j = 0; j < in[i].size(); ++j) {
        if (check(float(in[i][j]), checkNaNAndInf)) {
          return false;
        }
      }
    }
    return true;
  }
};

template class HasNaNOrInf<float, true>;
template class HasNaNOrInf<float, false>;
template class HasNaNOrInf<half, true>;
template class HasNaNOrInf<half, false>;

// The second template parameter
// nanOrInf = true  : Detect if either NaN or Inf is present
// nanOrInf = false : Detect only NaN
template <typename FPType, bool nanOrInf>
class HasNaNOrInfSupervisor : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  HasNaNOrInfSupervisor();
  IS_EXTERNAL_CODELET(true);
  Input<Vector<FPType, VectorLayout::ONE_PTR, 8>> in;
  unsigned short sizeIn8BytesPerWorker;
  unsigned char remWorkerId;
  unsigned char remWorkerExtras;
  bool compute() {
    unsigned size = sizeIn8BytesPerWorker * NUM_WORKERS + remWorkerId;
    size = size * (std::is_same<FPType, half>() ? 4 : 2) + remWorkerExtras;
    constexpr bool checkNaNAndInf = nanOrInf;
    for (unsigned i = 0; i < size; ++i) {
      if (check(float(in[i]), checkNaNAndInf)) {
        return false;
      }
    }
    return true;
  }
};

template class HasNaNOrInfSupervisor<float, true>;
template class HasNaNOrInfSupervisor<float, false>;
template class HasNaNOrInfSupervisor<half, true>;
template class HasNaNOrInfSupervisor<half, false>;

} // namespace popops
