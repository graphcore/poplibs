// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/Vertex.hpp>

static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

class CastFloatToGf8Supervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  CastFloatToGf8Supervisor();

  Input<Vector<int, COMPACT_PTR, 8>> param;
  Input<Vector<float, COMPACT_PTR, 8>> in;
  Output<Vector<signed char, COMPACT_PTR, 4>> out;
  unsigned short elementsPerWorker;
  unsigned short lastWorkerParams;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  void compute() {}
};

} // end namespace experimental
} // end namespace popfloat
