// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef popnn_PoolOptions_hpp
#define popnn_PoolOptions_hpp

namespace popnn {
namespace pooling {

// Options to control the implementation of pooling.
struct PoolOptions {
  // Use tile introspective mapping.
  // If disabled a linear tile mapping is used based on planner split
  bool poolUseIntrospectiveMapping = true;
  // The pooling implementation defaults to being optimised to aid memory
  // allocation.  To optimise for speed instead, set this option to true
  bool optimizeForSpeed = false;
  // Select the data type to use for intermediate results during pooling
  // calculation, selecting float values where that is beneficial to accuracy
  bool useFloatPartialsWhereBeneficial = false;
};

} // namespace pooling
} // namespace popnn

#endif // #ifndef popnn_PoolOptions_hpp
