// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "poplibs_support/forceInterleavedEstimates.hpp"
#include <cstdlib>

namespace poplibs_support {

bool getForceInterleavedEstimates() {
  static bool forceInterleavedEstimates =
      std::getenv("POPLIBS_FORCE_INTERLEAVED_ESTIMATES");
  return forceInterleavedEstimates;
}

} // namespace poplibs_support
