// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _ConvolutionInternal_hpp_
#define _ConvolutionInternal_hpp_

#include "CanonicalConvParams.hpp"
#include "ConvOptions.hpp"
#include <poplin/Convolution.hpp>

namespace poplin {

poplar::Tensor createInput(poplar::Graph &graph, const Plan &plan,
                           const CanonicalConvParams &params,
                           const std::string &name, const ConvOptions &options);

poplar::Tensor createWeights(poplar::Graph &graph, const Plan &plan,
                             const CanonicalConvParams &params,
                             const std::string &name,
                             const ConvOptions &options);

poplar::Tensor convolution(poplar::Graph &graph, const poplar::Tensor &in,
                           const poplar::Tensor &weights, const Plan &plan,
                           const CanonicalConvParams &params,
                           bool transposeAndFlipWeights,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix,
                           const ConvOptions &options);

poplar::Tensor calculateWeightDeltas(
    poplar::Graph &graph, const poplar::Tensor &zDeltas_,
    const poplar::Tensor &activations_, const Plan &wuPlan,
    const CanonicalConvParams &wuParams, poplar::program::Sequence &prog,
    const std::string &debugPrefix, const ConvOptions &wuOptions);

void convolutionWeightUpdate(
    poplar::Graph &graph, const poplar::Tensor &zDeltas,
    const poplar::Tensor &weights, const poplar::Tensor &activations,
    const Plan &plan, CanonicalConvParams params, const poplar::Tensor &scale,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    const ConvOptions &options);

void convolutionWeightUpdate(poplar::Graph &graph,
                             const poplar::Tensor &zDeltas,
                             const poplar::Tensor &weights,
                             const poplar::Tensor &activations,
                             const Plan &plan, CanonicalConvParams params,
                             float scale, poplar::program::Sequence &prog,
                             const std::string &debugPrefix,
                             const ConvOptions &options);

} // namespace poplin

#endif // _ConvolutionInternal_hpp_
