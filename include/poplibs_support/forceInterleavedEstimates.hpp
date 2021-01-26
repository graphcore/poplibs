// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef poplibs_support_forceInterleavedEstimates_hpp
#define poplibs_support_forceInterleavedEstimates_hpp

// This file contains the function that is called to check the environment
// variable that is set when we want the cycle estimators to be forced to
// account for interleaved memory

namespace poplibs_support {

bool getForceInterleavedEstimates();

} // namespace poplibs_support

#endif // poplibs_support_forceInterleavedEstimates_hpp
