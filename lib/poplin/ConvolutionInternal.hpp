// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _ConvolutionInternal_hpp_
#define _ConvolutionInternal_hpp_

#include "ConvOptions.hpp"
#include <poplin/Convolution.hpp>

namespace poplin {

poplar::Tensor
convolution(poplar::Graph &graph, const poplar::Tensor &in,
            const poplar::Tensor &weights, const ConvParams &params,
            bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix, const ConvOptions &options,
            PlanningCache *cache = nullptr);

} // namespace poplin

#endif // _ConvolutionInternal_hpp_
