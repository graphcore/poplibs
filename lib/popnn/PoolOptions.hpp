// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_PoolOptions_hpp
#define popnn_PoolOptions_hpp

namespace popnn {
namespace pooling {

// Options to control the implementation of pooling.
struct PoolOptions {
  // Use tile introspective mapping.
  // If disabled a linear tile mapping is used based on planner split
  bool poolUseIntrospectiveMapping = true;
};

} // namespace pooling
} // namespace popnn

#endif // #ifndef popnn_PoolOptions_hpp
