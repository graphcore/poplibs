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

template <typename FPType> class HasNaN : public Vertex {
public:
  HasNaN();
  IS_EXTERNAL_CODELET(true);
  Vector<Input<Vector<FPType, SPAN, 8>>> in;

  bool compute() {
    for (unsigned i = 0; i < in.size(); ++i) {
      for (unsigned j = 0; j < in[i].size(); ++j) {
        if (std::isnan(float(in[i][j]))) {
          return false;
        }
      }
    }
    return true;
  }
};

template class HasNaN<float>;
template class HasNaN<half>;

template <typename FPType>
class HasNaNSupervisor : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  HasNaNSupervisor();
  IS_EXTERNAL_CODELET(true);
  Input<Vector<FPType, VectorLayout::ONE_PTR, 8>> in;
  unsigned short sizeIn8BytesPerWorker;
  unsigned char remWorkerId;
  unsigned char remWorkerExtras;
  bool compute() {
    unsigned size = sizeIn8BytesPerWorker * NUM_WORKERS + remWorkerId;
    size = size * (std::is_same<FPType, half>() ? 4 : 2) + remWorkerExtras;

    for (unsigned i = 0; i < size; ++i) {
      if (std::isnan(float(in[i]))) {
        return false;
      }
    }
    return true;
  }
};

template class HasNaNSupervisor<float>;
template class HasNaNSupervisor<half>;

} // namespace popops
