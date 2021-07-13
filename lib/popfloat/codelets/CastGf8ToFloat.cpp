// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popfloatCodelets.hpp"
#include "popfloatUtils.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include <array>
#include <cmath>
#include <ipudef.h>
#include <popfloat/experimental/GfloatExpr.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>

static constexpr auto COMPACT_PTR = poplar::VectorLayout::COMPACT_PTR;

using namespace poplar;

namespace popfloat {
namespace experimental {

class CastGf8ToFloatSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  CastGf8ToFloatSupervisor();

  Input<Vector<int, COMPACT_PTR, 8>> param;
  Input<Vector<signed char, COMPACT_PTR, 4>> in;
  Output<Vector<float, COMPACT_PTR, 8>> out;
  unsigned short elementsPerWorker;
  unsigned short lastWorkerParams;

  IS_EXTERNAL_CODELET(EXTERNAL_CODELET);

  bool compute() { return true; }
};

} // end namespace experimental
} // end namespace popfloat
