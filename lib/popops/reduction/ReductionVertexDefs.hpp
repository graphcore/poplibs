// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popops_reduction_ReductionVertexDefs_hpp_
#define popops_reduction_ReductionVertexDefs_hpp_

namespace popops {

enum class ReductionSpecialisation {
  // TODO: T12965 Swap 2&3 so that higher specialisations are cheaper.

  DEFAULT,
  SCALAR_OUTPUT_REGIONS,
  SCALAR_OUTPUT_SINGLE_INPUT,
  SINGLE_OUTPUT_REGION,
  ALL_REGIONS_CONTINUOUS,
  PARTIALS_EQUAL_SIZE
};

} // end namespace popops
#endif // popops_reduction_ReductionVertexDefs_hpp_
