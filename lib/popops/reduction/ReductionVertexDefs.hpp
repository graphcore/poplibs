// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popops_reduction_ReductionVertexDefs_hpp_
#define popops_reduction_ReductionVertexDefs_hpp_

namespace popops {

enum class ReductionSpecialisation {
  // TODO: T12965 Swap 2&3 so that higher specialisations are cheaper.

  DEFAULT,
  SCALAR_OUTPUT_REGIONS,
  SCALAR_OUTPUT_SINGLE_INPUT,
  STRIDED_REDUCE,
  STRIDED_REDUCE_OUTER,
  ALL_REGIONS_CONTINUOUS
};

// For use with STRIDED_REDUCE
template <typename T> struct CountsAndStrides {
  T numOutputsM1;
  T numPartialsM1;
  T partialsWidth;
  T numOuterStridesM1;
  T outerStride;
};

} // end namespace popops
#endif // popops_reduction_ReductionVertexDefs_hpp_
