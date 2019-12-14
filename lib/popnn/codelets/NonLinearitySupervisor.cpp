// Copyright (c) Graphcore Ltd, All rights reserved.
#include "NonLinearity.hpp"

#include "poplibs_support/ExternalCodelet.hpp"

namespace popnn {

template <typename FPType, NonLinearityType nlType>
class WORKER_ALIGN NonLinearitySupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  NonLinearitySupervisor();

  InOut<Vector<FPType, SCALED_PTR32>> data;
  const unsigned short n;

  IS_EXTERNAL_CODELET(true);
  bool compute() {
    for (unsigned i = 0; i < n; ++i) {
      data[i] = nonlinearity(nlType, float(data[i]));
    }
    return true;
  }
};

INSTANTIATE_NL(NonLinearitySupervisor)

} // namespace popnn
