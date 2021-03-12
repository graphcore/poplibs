// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_TraceChannels_hpp
#define poplibs_support_TraceChannels_hpp

#include <pvti/pvti.hpp>

namespace poplibs_support {

extern pvti::TraceChannel tracePoplin;
extern pvti::TraceChannel tracePopnn;
extern pvti::TraceChannel tracePopops;
extern pvti::TraceChannel tracePoprand;
extern pvti::TraceChannel tracePopsparse;
extern pvti::TraceChannel tracePoputil;

} // end namespace poplibs_support

#endif // poplibs_support_TraceChannels_hpp
