// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplin/MultiConvolution.hpp"

#include "ConvUtilInternal.hpp"
#include "poplin/Convolution.hpp"
#include "poputil/exceptions.hpp"

namespace poplin {

namespace multiconv {

std::vector<poplar::Tensor>
createWeights(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args,
              PlanningCache *cache) {
  std::vector<poplar::Tensor> weights;
  for (const auto &arg : args) {
    weights.push_back(
        poplin::createWeights(graph, arg.params, arg.name, arg.options, cache));
  }
  return weights;
}

std::vector<poplar::Tensor>
createInput(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args,
            PlanningCache *cache) {
  std::vector<poplar::Tensor> inputs;
  for (const auto &arg : args) {
    inputs.push_back(
        poplin::createInput(graph, arg.params, arg.name, arg.options, cache));
  }
  return inputs;
}

std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args,
            const bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix, PlanningCache *cache) {
            poplar::program::Sequence &prog, const std::string &debugPrefix,
            PlanningCache *cache) {
  // Optimisation: try combining similar-size convolutions
  const auto groups = groupCombinables(args);
  const auto combined = combine(groups);

  std::vector<poplar::Tensor> outs;
  for (const auto &arg : combined) {
    outs.push_back(poplin::convolution(graph, arg.inputs, arg.weights,
                                       arg.params, transposeAndFlipWeights,
                                       prog, debugPrefix, arg.options, cache));
  }

  return split(groups, outs);
}

std::vector<poplar::Tensor>
calculateWeightDeltas(poplar::Graph &graph,
                      const std::vector<CalculateWeightDeltasArgs> &args,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix, PlanningCache *cache) {
  std::vector<poplar::Tensor> weightDeltas;
  for (const auto &arg : args) {
    weightDeltas.push_back(poplin::calculateWeightDeltas(
        graph, arg.zDeltas, arg.activations, arg.params, prog, debugPrefix,
        arg.options, cache));
  }
  return weightDeltas;
}

void convolutionWeightUpdate(poplar::Graph &graph,
                             const std::vector<ConvWeightUpdateArgs> &args,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix,
                             PlanningCache *cache) {
  for (const auto &arg : args) {
    poplin::convolutionWeightUpdate(graph, arg.zDeltas, arg.weights,
                                    arg.activations, arg.params, arg.scale,
                                    prog, debugPrefix, arg.options, cache);
  }
}

void convolutionWeightUpdate(
    poplar::Graph &graph, const std::vector<ConvWeightUpdateArgsScalar> &args,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    PlanningCache *cache) {
  for (const auto &arg : args) {
    poplin::convolutionWeightUpdate(graph, arg.zDeltas, arg.weights,
                                    arg.activations, arg.params, arg.scale,
                                    prog, debugPrefix, arg.options, cache);
  }
}

} // namespace multiconv

} // namespace poplin
