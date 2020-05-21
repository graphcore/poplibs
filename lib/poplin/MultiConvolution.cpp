// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplin/MultiConvolution.hpp"
#include "ConvolutionInternal.hpp"

#include "ConvUtilInternal.hpp"
#include "MultiConvolutionInternal.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/Convolution.hpp"
#include "poputil/exceptions.hpp"

namespace poplin {
namespace multiconv {

namespace logging = poplibs_support::logging;

static void log(const char *name, const std::vector<ConvolutionArgs> &args) {
  if (logging::shouldLog(logging::Level::Info)) {
    logging::info(name);
    for (unsigned i = 0; i < args.size(); ++i) {
      logging::info("  conv {}:", i);
      log(4, args[i].params);
    }
  }
}

// optimisation pipeline
template <typename T> T preProcess(const T &args) {
  // Optimisation: try combining similar-size convolutions
  const auto groups = groupCombinables(args);
  const auto combined = combine(groups);

  return combined;
}

// apply any post-processing required to the output tensors as a result of the
// optimisations applied before the operation.
template <typename T>
std::vector<poplar::Tensor>
postProcessOutput(const T &args, const std::vector<poplar::Tensor> &out) {
  // split the output apart from previously combined convolutions.
  const auto groups = groupCombinables(args);
  return split(groups, out, splitOutput);
}

// apply any post-processing required to the input tensors as a result of the
// optimisations applied before the operation.
static std::vector<poplar::Tensor>
postProcessInput(const std::vector<multiconv::internal::CreateTensorArgs> &args,
                 const std::vector<poplar::Tensor> &in) {
  // split the input apart from previously combined convolutions.
  const auto groups = groupCombinables(args);
  return split(groups, in, splitInput);
}

// apply any post-processing required to the weights tensors as a result of the
// optimisations applied before the operation.
static std::vector<poplar::Tensor> postProcessWeights(
    const std::vector<multiconv::internal::CreateTensorArgs> &args,
    const std::vector<poplar::Tensor> &weights) {
  // split the weights apart from previously combined convolutions.
  const auto groups = groupCombinables(args);
  return split(groups, weights, splitWeights);
}

std::vector<poplar::Tensor>
createWeights(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args_,
              PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  std::vector<poplar::Tensor> weights;
  for (const auto &arg : args) {
    weights.push_back(
        poplin::createWeights(graph, arg.params, arg.name, arg.options, cache));
  }

  return postProcessWeights(argsWithConvOptions, weights);
}

std::vector<poplar::Tensor>
createInput(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args_,
            PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  std::vector<poplar::Tensor> inputs;
  for (const auto &arg : args) {
    inputs.push_back(
        poplin::createInput(graph, arg.params, arg.name, arg.options, cache));
  }

  return postProcessInput(argsWithConvOptions, inputs);
}

std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args_,
            const bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix, PlanningCache *cache) {
  log("multiconv::convolution", args_);

  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  std::vector<poplar::Tensor> outs;
  for (unsigned i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    const auto name = debugPrefix + "/conv" + std::to_string(i);

    outs.push_back(poplin::convolution(graph, arg.inputs, arg.weights,
                                       arg.params, transposeAndFlipWeights,
                                       prog, name, arg.options, cache));
  }

  return postProcessOutput(argsWithConvOptions, outs);
}

std::vector<poplar::Tensor>
calculateWeightDeltas(poplar::Graph &graph,
                      const std::vector<CalculateWeightDeltasArgs> &args_,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix, PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  std::vector<poplar::Tensor> weightDeltas;
  for (const auto &arg : args) {
    weightDeltas.push_back(poplin::calculateWeightDeltas(
        graph, arg.zDeltas, arg.activations, arg.params, prog, debugPrefix,
        arg.options, cache));
  }

  return postProcessOutput(argsWithConvOptions, weightDeltas);
}

void convolutionWeightUpdate(poplar::Graph &graph,
                             const std::vector<ConvWeightUpdateArgs> &args_,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix,
                             PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  for (const auto &arg : args) {
    // TODO: convolutionWeightUpdate expects inputType == outputType, handle
    // when that is not the case.
    if (arg.params->inputType != arg.params->outputType) {
      throw poputil::poplibs_error(
          "multiconv::convolutionWeightUpdate does not support having a "
          "different input and output type.");
    }

    poplin::convolutionWeightUpdate(graph, arg.zDeltas, arg.weights,
                                    arg.activations, arg.params, arg.scale,
                                    prog, debugPrefix, arg.options, cache);
  }
}

void convolutionWeightUpdate(
    poplar::Graph &graph, const std::vector<ConvWeightUpdateArgsScalar> &args_,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  for (const auto &arg : args) {
    poplin::convolutionWeightUpdate(graph, arg.zDeltas, arg.weights,
                                    arg.activations, arg.params, arg.scale,
                                    prog, debugPrefix, arg.options, cache);
  }
}

} // namespace multiconv
} // namespace poplin
