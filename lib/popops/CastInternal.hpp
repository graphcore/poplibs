// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_CastInternal_hpp
#define popops_CastInternal_hpp

namespace popops {
namespace internal {

struct Cast1DPartition {
  unsigned workerElems;
  unsigned workerCount;
  unsigned workerLast;
  unsigned deltaLast;
  unsigned pack() const;
};

Cast1DPartition getCast1DPartition(unsigned numWorkerContexts,
                                   unsigned numElems);

} // end namespace internal
} // end namespace popops

#endif // popops_CastInternal_hpp
