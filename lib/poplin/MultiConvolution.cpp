// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplin/MultiConvolution.hpp"

#include "ConvPlan.hpp"
#include "ConvUtilInternal.hpp"
#include "ConvolutionInternal.hpp"
#include "MultiConvolutionInternal.hpp"
#include "poplibs_support/Visitor.hpp"
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

template <typename T>
static MultiPlan getMultiPlan(const poplar::Target &target,
                              const std::vector<T> &args,
                              PlanningCache *cache) {
  std::vector<CanonicalConvParams> params;
  std::vector<ConvOptions> options;
  for (const auto &arg : args) {
    params.push_back(arg.params);
    options.push_back(arg.options);
  }

  return getMultiPlan(target, params, options, cache);
}

std::vector<poplar::Tensor>
createWeights(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args_,
              PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        ResultType weights;
        weights.reserve(args.size());

        assert(args.size() == serial.plans.size());
        for (unsigned i = 0; i < args.size(); ++i) {
          const auto &arg = args[i];

          weights.push_back(poplin::createWeights(
              graph, serial.plans[i], arg.params, arg.name, arg.options));
        }

        return weights;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  const auto weights = boost::apply_visitor(visitor, multiPlan);

  return postProcessWeights(argsWithConvOptions, weights);
}

std::vector<poplar::Tensor>
createInput(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args_,
            PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        ResultType inputs;
        inputs.reserve(args.size());

        assert(args.size() == serial.plans.size());
        for (unsigned i = 0; i < args.size(); ++i) {
          const auto &arg = args[i];

          inputs.push_back(poplin::createInput(
              graph, serial.plans[i], arg.params, arg.name, arg.options));
        }

        return inputs;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  const auto inputs = boost::apply_visitor(visitor, multiPlan);

  return postProcessInput(argsWithConvOptions, inputs);
}

std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args_,
            const bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix, PlanningCache *cache) {
  log("multiconv::convolution", args_);

  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  const auto args = preProcess(argsWithConvOptions);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        ResultType outs;
        outs.reserve(args.size());

        assert(serial.plans.size() == args.size());
        for (unsigned i = 0; i < args.size(); ++i) {
          const auto &arg = args[i];
          const auto name = debugPrefix + "/conv" + std::to_string(i);

          outs.push_back(poplin::convolution(
              graph, arg.inputs, arg.weights, serial.plans[i], arg.params,
              transposeAndFlipWeights, prog, name, arg.options));
        }
        return outs;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  const auto outs = boost::apply_visitor(visitor, multiPlan);

  return postProcessOutput(argsWithConvOptions, outs);
}

template <typename T>
static std::vector<T> getWeightUpdateArgs(std::vector<T> args) {
  for (auto &arg : args) {
    arg.params = getWeightUpdateParams(arg.params.releaseParams());
    arg.options = getWeightUpdateOptions(std::move(arg.options));
  }

  return args;
}

std::vector<poplar::Tensor>
calculateWeightDeltas(poplar::Graph &graph,
                      const std::vector<CalculateWeightDeltasArgs> &args_,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix, PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  auto args = preProcess(argsWithConvOptions);
  args = getWeightUpdateArgs(std::move(args));

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        ResultType weightDeltas;
        weightDeltas.reserve(args.size());

        assert(serial.plans.size() == args.size());
        for (unsigned i = 0; i < args.size(); ++i) {
          const auto &arg = args[i];

          weightDeltas.push_back(poplin::calculateWeightDeltas(
              graph, arg.zDeltas, arg.activations, serial.plans[i], arg.params,
              prog, debugPrefix, arg.options));
        }

        return weightDeltas;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  const auto weightDeltas = boost::apply_visitor(visitor, multiPlan);

  return postProcessOutput(argsWithConvOptions, weightDeltas);
}

template <typename ArgType>
void convolutionWeightUpdateImpl(poplar::Graph &graph,
                                 const std::vector<ArgType> &args_,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix,
                                 PlanningCache *cache) {
  const auto argsWithConvOptions = convertToConvOptions(graph, args_);
  auto args = preProcess(argsWithConvOptions);
  args = getWeightUpdateArgs(std::move(args));

  const auto visitor = poplibs_support::make_visitor<void>(
      [&](const SerialPlan &serial) {
        assert(serial.plans.size() == args.size());
        for (unsigned i = 0; i < args.size(); ++i) {
          const auto &arg = args[i];

          // TODO: convolutionWeightUpdate expects inputType == outputType,
          // handle when that is not the case.
          if (arg.params->inputType != arg.params->outputType) {
            throw poputil::poplibs_error(
                "multiconv::convolutionWeightUpdate does not support having a "
                "different input and output type.");
          }

          poplin::convolutionWeightUpdate(
              graph, arg.zDeltas, arg.weights, arg.activations, serial.plans[i],
              arg.params, arg.scale, prog, debugPrefix, arg.options);
        }
      },
      [](const ParallelPlan &) {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  boost::apply_visitor(visitor, multiPlan);
}

void convolutionWeightUpdate(poplar::Graph &graph,
                             const std::vector<ConvWeightUpdateArgs> &args,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix,
                             PlanningCache *cache) {
  convolutionWeightUpdateImpl(graph, args, prog, debugPrefix, cache);
}

void convolutionWeightUpdate(
    poplar::Graph &graph, const std::vector<ConvWeightUpdateArgsScalar> &args,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    PlanningCache *cache) {
  convolutionWeightUpdateImpl(graph, args, prog, debugPrefix, cache);
}

} // namespace multiconv
} // namespace poplin
