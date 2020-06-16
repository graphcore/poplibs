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

template <typename T>
static std::string getLayerName(const std::string &debugPrefix,
                                const std::vector<T> &args) {
  const auto suffixes = boost::adaptors::transform(
      args, [](const auto &arg) { return convSuffix(arg.params); });

  return debugPrefix + "/MultiConv" + boost::algorithm::join(suffixes, ",");
}

// serial plans are implemented by just performing each convolution in it's
// own control program, one after the other.
template <typename T, typename F>
static void forEachSerialPlan(poplar::Graph &graph, const SerialPlan &serial,
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

// overload that doens't require a control program
template <typename T, typename F>
static void forEachSerialPlan(const SerialPlan &serial,
                              const std::vector<T> &args, const F &fn) {
  assert(serial.plans.size() == args.size());
  for (unsigned i = 0; i < args.size(); ++i) {
    fn(serial.plans[i], args[i]);
  }
}

poplar::Tensor createWeights(poplar::Graph &graph,
                             const std::vector<CreateTensorArgs> &args_,
                             unsigned weightsIndex,
                             poplin::PlanningCache *cache) {
  const auto args = convertToConvOptions(graph, args_);

  using ResultType = poplar::Tensor;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        return poplin::createWeights(
            graph, serial.plans[weightsIndex], args[weightsIndex].params,
            args[weightsIndex].name, args[weightsIndex].options);
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  return boost::apply_visitor(visitor, multiPlan);
}

poplar::Tensor createInput(poplar::Graph &graph,
                           const std::vector<CreateTensorArgs> &args_,
                           unsigned inputIndex, poplin::PlanningCache *cache) {
  const auto args = convertToConvOptions(graph, args_);

  using ResultType = poplar::Tensor;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        return poplin::createInput(
            graph, serial.plans[inputIndex], args[inputIndex].params,
            args[inputIndex].name, args[inputIndex].options);
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  return boost::apply_visitor(visitor, multiPlan);
}

std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args_,
            const bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix, PlanningCache *cache) {
  log("multiconv::convolution", args_);

  const auto args = convertToConvOptions(graph, args_);

  const auto layerName = getLayerName(debugPrefix, args);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        ResultType outs;
        outs.reserve(args.size());

        forEachSerialPlan(
            graph, serial, args, prog, layerName,
            [&](const Plan &plan, const auto &arg, ConvProgramTree &cpt,
                const std::string &debugPrefix) {
              outs.push_back(poplin::convolution(
                  graph, arg.inputs, arg.weights, plan, arg.params,
                  transposeAndFlipWeights, cpt, debugPrefix, arg.options));
            });

        return outs;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
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

std::vector<poplar::Tensor>
calculateWeightDeltas(poplar::Graph &graph,
                      const std::vector<CalculateWeightDeltasArgs> &args_,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix, PlanningCache *cache) {
  const auto args = getWeightUpdateArgs(convertToConvOptions(graph, args_));

  const auto layerName = getLayerName(debugPrefix, args);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const SerialPlan &serial) {
        ResultType weightDeltas;
        weightDeltas.reserve(args.size());

        forEachSerialPlan(
            graph, serial, args, prog, layerName,
            [&](const Plan &plan, const auto &arg, ConvProgramTree &cpt,
                const std::string &debugPrefix) {
              weightDeltas.push_back(poplin::calculateWeightDeltas(
                  graph, arg.zDeltas, arg.activations, plan, arg.params, cpt,
                  debugPrefix, arg.options));
            });

        return weightDeltas;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache);
  return boost::apply_visitor(visitor, multiPlan);
}

template <typename ArgType>
void convolutionWeightUpdateImpl(poplar::Graph &graph,
                                 const std::vector<ArgType> &args_,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix,
                                 PlanningCache *cache) {
  const auto args = getWeightUpdateArgs(convertToConvOptions(graph, args_));

  const auto layerName = getLayerName(debugPrefix, args);

  const auto visitor = poplibs_support::make_visitor<void>(
      [&](const SerialPlan &serial) {
        forEachSerialPlan(
            graph, serial, args, prog, layerName,
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
