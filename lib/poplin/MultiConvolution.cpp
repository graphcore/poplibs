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
#include "poputil/DebugInfo.hpp"
#include "poputil/exceptions.hpp"
#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>

namespace poputil {
template <>
poplar::ProfileValue
toProfileValue(const poplin::multiconv::ConvolutionArgs &t) {
  poplar::ProfileValue::Map v;
  v.insert({"inputs", toProfileValue(t.inputs)});
  v.insert({"weights", toProfileValue(t.weights)});
  v.insert({"params", toProfileValue(t.params)});
  v.insert({"options", toProfileValue(t.options)});
  return v;
}

template <>
poplar::ProfileValue
toProfileValue(const poplin::multiconv::ConvWeightUpdateArgs &t) {
  poplar::ProfileValue::Map v;
  v.insert({"zDeltas", toProfileValue(t.zDeltas)});
  v.insert({"weights", toProfileValue(t.weights)});
  v.insert({"activations", toProfileValue(t.activations)});
  v.insert({"scale", toProfileValue(t.scale)});
  v.insert({"params", toProfileValue(t.params)});
  v.insert({"options", toProfileValue(t.options)});
  return v;
}

template <>
poplar::ProfileValue
toProfileValue(const poplin::multiconv::ConvWeightUpdateArgsScalar &t) {
  poplar::ProfileValue::Map v;
  v.insert({"zDeltas", toProfileValue(t.zDeltas)});
  v.insert({"weights", toProfileValue(t.weights)});
  v.insert({"activations", toProfileValue(t.activations)});
  v.insert({"scale", toProfileValue(t.scale)});
  v.insert({"params", toProfileValue(t.params)});
  v.insert({"options", toProfileValue(t.options)});
  return v;
}

template <>
poplar::ProfileValue
toProfileValue(const poplin::multiconv::CalculateWeightDeltasArgs &t) {
  poplar::ProfileValue::Map v;
  v.insert({"zDeltas", toProfileValue(t.zDeltas)});
  v.insert({"activations", toProfileValue(t.activations)});
  v.insert({"params", toProfileValue(t.params)});
  v.insert({"options", toProfileValue(t.options)});
  return v;
}
} // namespace poputil

namespace poplin {
namespace multiconv {

namespace logging = poplibs_support::logging;

static void log(const char *name, const std::vector<ConvolutionArgs> &args) {
  if (logging::poplin::shouldLog(logging::Level::Info)) {
    logging::poplin::info(name);
    for (unsigned i = 0; i < args.size(); ++i) {
      logging::poplin::info("  conv {}:", i);
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

  params.reserve(args.size());
  convOptions.reserve(args.size());

  for (const auto &arg : args) {
    params.push_back(arg.params);
    convOptions.push_back(arg.options);
  }

  return getMultiPlan(target, params, convOptions, cache, options);
}

template <typename T>
static std::string getLayerName(const std::vector<T> &args) {
  const auto suffixes = boost::adaptors::transform(
      args, [](const auto &arg) { return convSuffix(arg.params); });

  return "MultiConv_{" + boost::algorithm::join(suffixes, ",") + "}";
}

// serial plans are implemented by just performing each convolution in it's
// own control program, one after the other.
template <typename T, typename F>
static void applyMultiPlan(poplar::Graph &graph, const SerialPlan &serial,
                           const std::vector<T> &args,
                           poplar::program::Sequence &prog,
                           const poplar::DebugNameAndId &dnai, const F &fn) {
  assert(serial.plans.size() == args.size());
  logging::poplin::info("Implementing multi-convs using a serial plan: {}",
                        dnai.getPathName());

  for (unsigned i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    const auto &plan = serial.plans[i];

    const auto name = std::to_string(i);
    ConvProgramTree cpt(graph, plan, {dnai, name});

    fn(plan, arg, cpt, i, {dnai, name});

    const ConvOptions options(arg.options);
    cpt.lower(graph, prog, options.insertTransformsCycleCountProgs, {dnai});
  }
}

// parallel plans are implemented by re-using the same  ConvProgramTree for each
// convolution. each plan will be allocated to a different tile range to
// guarantee that these convs don't overlap.
template <typename T, typename F>
static void applyMultiPlan(poplar::Graph &graph, const ParallelPlan &para,
                           const std::vector<T> &args,
                           poplar::program::Sequence &prog,
                           const poplar::DebugNameAndId &dnai, const F &fn) {
  assert(para.plans.size() == args.size());
  logging::poplin::info("Implementing multi-convs using a parallel plan: {}",
                        dnai.getPathName());

  for (unsigned i = 1; i < para.plans.size(); ++i) {
    assert(para.plans[0].numLevels() == para.plans[i].numLevels());
    assert(para.plans[0].totalSerialSplit() ==
           para.plans[i].totalSerialSplit());
  }

  ConvProgramTree cpt(graph, para.plans.front(), {dnai});
  const ConvOptions options(args[0].options);
  const bool insertCycleCounts = options.insertTransformsCycleCountProgs;

  for (unsigned i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    const auto &plan = para.plans[i];

    const auto name = std::to_string(i);

    fn(plan, arg, cpt, i, {dnai, name});

    const ConvOptions optionsNextPlan(arg.options);
    if (insertCycleCounts != optionsNextPlan.insertTransformsCycleCountProgs) {
      throw poputil::poplibs_error(
          "Parallel multi-plans require to have same value for "
          "<insertTransformsCycleCountProgs> option");
    }
  }

  cpt.lower(graph, prog, insertCycleCounts, {dnai});
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

void weightsTransposeChansFlipXY(poplar::Graph &graph,
                                 std::vector<ConvolutionArgs> &args,
                                 const std::vector<poplar::Tensor> &weightsIn,
                                 poplar::program::Sequence &prog,
                                 const poplar::OptionFlags &options,
                                 const poplar::DebugContext &debugContext,
                                 poplin::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(args, weightsIn, options));

  const auto visitor =
      poplibs_support::make_visitor<void>([&](const auto &multiPlan) {
        logging::poplin::info("multiconv::weightsTransposeChansFlipXY");
        applyMultiPlan(
            graph, multiPlan, args, prog, {di, "weightsTranspose"},
            [&](const Plan &plan, const auto &arg, ConvProgramTree &cpt,
                unsigned index, const poplar::DebugNameAndId &dnai) {
              poplin::weightsTransposeChansFlipXY(graph, weightsIn[index],
                                                  arg.weights, cpt, {dnai});
            });
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  boost::apply_visitor(visitor, multiPlan);
}

std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args_,
            const bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const poplar::DebugContext &debugContext,
            const poplar::OptionFlags &options, PlanningCache *cache) {

  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(args_, transposeAndFlipWeights, options, cache));

  log("multiconv::convolution", args_);

  const auto args = convertToConvOptions(graph, args_);

  const auto layerName = getLayerName(args);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor =
      poplibs_support::make_visitor<ResultType>([&](const auto &multiPlan) {
        ResultType outs;
        outs.reserve(args.size());

        applyMultiPlan(
            graph, multiPlan, args, prog, {di, layerName},
            [&](const Plan &plan, const auto &arg, ConvProgramTree &cpt,
                unsigned index, const poplar::DebugNameAndId &dnai) {
              outs.push_back(poplin::convolution(
                  graph, arg.inputs, arg.weights, plan, arg.params,
                  transposeAndFlipWeights, cpt, {dnai}, arg.options));
            });

        return outs;
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  auto output = boost::apply_visitor(visitor, multiPlan);
  di.addOutputs(DI_ARGS(output));
  return output;
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
    poplar::program::Sequence &prog, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(args_, options, cache));

  const auto args = getWeightUpdateArgs(convertToConvOptions(graph, args_));

  const auto layerName = getLayerName(args);

  using ResultType = std::vector<poplar::Tensor>;
  const auto visitor = poplibs_support::make_visitor<ResultType>(
      [&](const auto &multiPlan) {
        ResultType weightDeltas;
        weightDeltas.reserve(args.size());

        applyMultiPlan(graph, multiPlan, args, prog, {di, layerName},
                       [&](const Plan &plan, const auto &arg,
                           ConvProgramTree &cpt, unsigned index,
                           const poplar::DebugNameAndId &dnai) {
                         weightDeltas.push_back(poplin::calculateWeightDeltas(
                             graph, arg.zDeltas, arg.activations, plan,
                             arg.params, cpt, {dnai}, arg.options));
                       });

        return weightDeltas;
      },
      [](const ParallelPlan &) -> ResultType {
        throw poputil::poplibs_error("Parallel multi-plans not yet supported");
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  auto output = boost::apply_visitor(visitor, multiPlan);
  di.addOutputs(DI_ARGS(output));
  return output;
}

template <typename ArgType>
void convolutionWeightUpdateImpl(poplar::Graph &graph,
                                 const std::vector<ArgType> &args_,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugNameAndId &dnai,
                                 const poplar::OptionFlags &options,
                                 PlanningCache *cache) {
  const auto args = getWeightUpdateArgs(convertToConvOptions(graph, args_));

  const auto layerName = getLayerName(args);

  const auto visitor =
      poplibs_support::make_visitor<void>([&](const auto &multiPlan) {
        applyMultiPlan(
            graph, multiPlan, args, prog, {dnai, layerName},
            [&](const Plan &plan, const auto &arg, ConvProgramTree &cpt,
                unsigned index, const poplar::DebugNameAndId &dnai) {
              // TODO: convolutionWeightUpdate expects inputType == outputType,
              // handle when that is not the case.
              if (arg.params->inputType != arg.params->outputType) {
                throw poputil::poplibs_error(
                    "multiconv::convolutionWeightUpdate does not support "
                    "having a different input and output type.");
              }

              poplin::convolutionWeightUpdate(
                  graph, arg.zDeltas, arg.weights, arg.activations, plan,
                  arg.params, arg.scale, cpt, {dnai}, arg.options);
            });
      });

  const auto &target = graph.getTarget();
  const auto multiPlan = getMultiPlan(target, args, cache, options);
  boost::apply_visitor(visitor, multiPlan);
}

void convolutionWeightUpdate(poplar::Graph &graph,
                             const std::vector<ConvWeightUpdateArgs> &args,
                             poplar::program::Sequence &prog,
                             const poplar::DebugContext &debugContext,
                             const poplar::OptionFlags &options,
                             PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(args, options, cache));
  convolutionWeightUpdateImpl(graph, args, prog, {di}, options, cache);
}

void convolutionWeightUpdate(
    poplar::Graph &graph, const std::vector<ConvWeightUpdateArgsScalar> &args,
    poplar::program::Sequence &prog, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options, PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(args, options, cache));
  convolutionWeightUpdateImpl(graph, args, prog, {di}, options, cache);
}

} // namespace multiconv
} // namespace poplin
