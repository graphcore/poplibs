// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "poplibs_support/TraceChannels.hpp"

namespace poplibs_support {

pvti::TraceChannel tracePoplin{"graphConstruction/poplin"};
pvti::TraceChannel tracePopnn{"graphConstruction/popnn"};
pvti::TraceChannel tracePopops{"graphConstruction/popops"};
pvti::TraceChannel tracePoprand{"graphConstruction/poprand"};
pvti::TraceChannel tracePopsparse{"graphConstruction/popsparse"};
pvti::TraceChannel tracePoputil{"graphConstruction/poputil"};

} // end namespace poplibs_support
