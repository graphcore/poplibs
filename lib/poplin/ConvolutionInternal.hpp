// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _ConvolutionInternal_hpp_
#define _ConvolutionInternal_hpp_

#include "CanonicalConvParams.hpp"
#include "ConvOptions.hpp"
#include <poplin/Convolution.hpp>

namespace poplin {

poplar::Tensor createInput(poplar::Graph &graph,
                           const CanonicalConvParams &params,
                           const std::string &name, const ConvOptions &options,
                           PlanningCache *cache = nullptr);

poplar::Tensor createWeights(poplar::Graph &graph,
                             const CanonicalConvParams &params,
                             const std::string &name,
                             const ConvOptions &options,
                             PlanningCache *cache = nullptr);

poplar::Tensor
convolution(poplar::Graph &graph, const poplar::Tensor &in,
            const poplar::Tensor &weights, const CanonicalConvParams &params,
            bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix, const ConvOptions &options,
            PlanningCache *cache = nullptr);

poplar::Tensor calculateWeightDeltas(
    poplar::Graph &graph, const poplar::Tensor &zDeltas_,
    const poplar::Tensor &activations_, const CanonicalConvParams &fwdParams_,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    const ConvOptions &options, PlanningCache *cache);

void convolutionWeightUpdate(
    poplar::Graph &graph, const poplar::Tensor &zDeltas,
    const poplar::Tensor &weights, const poplar::Tensor &activations,
    CanonicalConvParams params, const poplar::Tensor &scale,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    const ConvOptions &options, PlanningCache *cache);

void convolutionWeightUpdate(poplar::Graph &graph,
                             const poplar::Tensor &zDeltas,
                             const poplar::Tensor &weights,
                             const poplar::Tensor &activations,
                             CanonicalConvParams params, float scale,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix,
                             const ConvOptions &options, PlanningCache *cache);

} // namespace poplin

#endif // _ConvolutionInternal_hpp_
