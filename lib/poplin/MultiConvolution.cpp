// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplin/MultiConvolution.hpp"

#include "ConvPlan.hpp"
#include "ConvProgramTree.hpp"
#include "ConvUtilInternal.hpp"
#include "ConvolutionInternal.hpp"
#include "MultiConvolutionInternal.hpp"
#include "poplibs_support/Visitor.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/Convolution.hpp"
#include "poputil/exceptions.hpp"
#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>

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

template <typename T>
static MultiPlan getMultiPlan(const poplar::Target &target,
                              const std::vector<T> &args, PlanningCache *cache,
                              const poplar::OptionFlags &options) {
  std::vector<CanonicalConvParams> params;
  std::vector<ConvOptions> convOptions;
  for (const auto &arg : args) {
    params.push_back(arg.params);
    convOptions.push_back(arg.options);
  }

  return getMultiPlan(target, params, convOptions, cache, options);
}

template <typename T>
static std::string getLayerName(const std::string &debugPrefix,
                                const std::vector<T> &args) {
  const auto suffixes = boost::adaptors::transform(
      args, [](const auto &arg) { return convSuffix(arg.params); });

  return debugPrefix + "/MultiConv_{" + boost::algorithm::join(suffixes, ",") +
         "}";
}

// serial plans are implemented by just performing each convolution in it's
// own control program, one after the other.
template <typename T, typename F>
static void applyMultiPlan(poplar::Graph &graph, const SerialPlan &serial,
                           const std::vector<T> &args,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix, const F &fn) {
  assert(serial.plans.size() == args.size());
  for (unsigned i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    const auto &plan = serial.plans[i];

    const auto name = debugPrefix + "/" + std::to_string(i);
    const auto numLevels = plan.partitions.size() + 1;
    ConvProgramTree cpt(numLevels, plan.totalSerialSplit(),
                        graph.addVariable(arg.params->inputType, {0}),
                        graph.addComputeSet(name + "/Convolve"));

    fn(plan, arg, cpt, name);
    cpt.lower(prog);
  }
}

// parallel plans are implemented by merging each ConvProgramTree into a single
// tree and lowering only that. therefore each conv share programs. each plan
// will be allocated to a different tile range to guarantee that these convs
// won't overlap.
template <typename T, typename F>
static void applyMultiPlan(poplar::Graph &graph, const ParallelPlan &para,
                           const std::vector<T> &args,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix, const F &fn) {
  assert(para.plans.size() == args.size());
  std::vector<ConvProgramTree> cpts;
  cpts.reserve(args.size());

  for (unsigned i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    const auto &plan = para.plans[i];

    const auto name = debugPrefix + "/" + std::to_string(i);
    const auto numLevels = plan.partitions.size() + 1;
    // TODO: assert(numLevels && serialSplits);
    // TODO: inputType
    cpts.emplace_back(numLevels, plan.totalSerialSplit(),
                      graph.addVariable(arg.params->inputType, {0}),
                      graph.addComputeSet(name + "/Convolve"));

    fn(plan, arg, cpts.back(), name);
  }

  auto cpt = merge(cpts);
  cpt.lower(prog);
}

poplar::Tensor createWeights(poplar::Graph &graph,
                             const std::vector<CreateTensorArgs> &args_,
                             unsigned weightsIndex,
                             const poplar::OptionFlags &options,
                             poplin::PlanningCache *cache) {
  const auto args = convertToConvOptions(graph, args_);

  using ResultType = poplar::Tensor;
  const auto visitor =
      poplibs_support::make_visitor<ResultType>([&](const auto &plan) {
        return poplin::createWeights(
            graph, plan.plans[weightsIndex], args[weightsIndex].params,
            args[weightsIndex].name, args[weightsIndex].options);
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  return boost::apply_visitor(visitor, multiPlan);
}

poplar::Tensor createInput(poplar::Graph &graph,
                           const std::vector<CreateTensorArgs> &args_,
                           unsigned inputIndex,
                           const poplar::OptionFlags &options,
                           poplin::PlanningCache *cache) {
  const auto args = convertToConvOptions(graph, args_);

  using ResultType = poplar::Tensor;
  const auto visitor =
      poplibs_support::make_visitor<ResultType>([&](const auto &plan) {
        return poplin::createInput(
            graph, plan.plans[inputIndex], args[inputIndex].params,
            args[inputIndex].name, args[inputIndex].options);
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  return boost::apply_visitor(visitor, multiPlan);
}

std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args_,
            const bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix, const poplar::OptionFlags &options,
            PlanningCache *cache) {
  log("multiconv::convolution", args_);

  const auto args = convertToConvOptions(graph, args_);

  const auto layerName = getLayerName(debugPrefix, args);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor =
      poplibs_support::make_visitor<ResultType>([&](const auto &multiPlan) {
        ResultType outs;
        outs.reserve(args.size());

        applyMultiPlan(
            graph, multiPlan, args, prog, layerName,
            [&](const Plan &plan, const auto &arg, ConvProgramTree &cpt,
                const std::string &debugPrefix) {
              outs.push_back(poplin::convolution(
                  graph, arg.inputs, arg.weights, plan, arg.params,
                  transposeAndFlipWeights, cpt, debugPrefix, arg.options));
            });

        return outs;
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  return boost::apply_visitor(visitor, multiPlan);
}

template <typename T>
static std::vector<T> getWeightUpdateArgs(std::vector<T> args) {
  for (auto &arg : args) {
    arg.params = getWeightUpdateParams(arg.params.releaseParams());
    arg.options = getWeightUpdateOptions(std::move(arg.options));
  }

  return args;
}

std::vector<poplar::Tensor> calculateWeightDeltas(
    poplar::Graph &graph, const std::vector<CalculateWeightDeltasArgs> &args_,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  const auto args = getWeightUpdateArgs(convertToConvOptions(graph, args_));

  const auto layerName = getLayerName(debugPrefix, args);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const auto &multiPlan) {
        ResultType weightDeltas;
        weightDeltas.reserve(args.size());

        applyMultiPlan(graph, multiPlan, args, prog, layerName,
                       [&](const Plan &plan, const auto &arg,
                           ConvProgramTree &cpt,
                           const std::string &debugPrefix) {
                         weightDeltas.push_back(poplin::calculateWeightDeltas(
                             graph, arg.zDeltas, arg.activations, plan,
                             arg.params, cpt, debugPrefix, arg.options));
                       });

        return weightDeltas;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  return boost::apply_visitor(visitor, multiPlan);
}

template <typename ArgType>
void convolutionWeightUpdateImpl(poplar::Graph &graph,
                                 const std::vector<ArgType> &args_,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix,
                                 const poplar::OptionFlags &options,
                                 PlanningCache *cache) {
  const auto args = getWeightUpdateArgs(convertToConvOptions(graph, args_));

  const auto layerName = getLayerName(debugPrefix, args);

  const auto visitor =
      poplibs_support::make_visitor<void>([&](const auto &multiPlan) {
        applyMultiPlan(
            graph, multiPlan, args, prog, layerName,
            [&](const Plan &plan, const auto &arg, ConvProgramTree &cpt,
                const std::string &debugPrefix) {
              // TODO: convolutionWeightUpdate expects inputType == outputType,
              // handle when that is not the case.
              if (arg.params->inputType != arg.params->outputType) {
                throw poputil::poplibs_error(
                    "multiconv::convolutionWeightUpdate does not support "
                    "having a different input and output type.");
              }

              poplin::convolutionWeightUpdate(
                  graph, arg.zDeltas, arg.weights, arg.activations, plan,
                  arg.params, arg.scale, cpt, debugPrefix, arg.options);
            });
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  boost::apply_visitor(visitor, multiPlan);
}

void convolutionWeightUpdate(poplar::Graph &graph,
                             const std::vector<ConvWeightUpdateArgs> &args,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix,
                             const poplar::OptionFlags &options,
                             PlanningCache *cache) {
  convolutionWeightUpdateImpl(graph, args, prog, debugPrefix, options, cache);
}

void convolutionWeightUpdate(
    poplar::Graph &graph, const std::vector<ConvWeightUpdateArgsScalar> &args,
    poplar::program::Sequence &prog, const std::string &debugPrefix,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  convolutionWeightUpdateImpl(graph, args, prog, debugPrefix, options, cache);
}

} // namespace multiconv
} // namespace poplin
