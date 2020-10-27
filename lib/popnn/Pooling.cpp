// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#include "popnn/Pooling.hpp"
#include "PoolOptions.hpp"
#include "PoolPlan.hpp"
#include "PoolVertices.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_support/print.hpp"
#include "poplin/ConvUtil.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Pad.hpp"
#include "popops/Reduce.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <boost/icl/interval_map.hpp>
#include <cassert>
#include <map>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using std::tie;
using namespace poputil;
using namespace poplibs_support;

namespace popnn {
namespace pooling {

std::ostream &operator<<(std::ostream &o, const PoolParams &params) {
  o << "{ poolingType=" << asString(params.poolingType) << "\n"
    << "  inputFieldShape=";
  printContainer(params.inputFieldShape, o);
  o << "\n  kernelShape=";
  printContainer(params.kernelShape, o);
  o << "\n  stride=";
  printContainer(params.stride, o);
  o << "\n  inputTruncationOrPaddingLower=";
  printContainer(params.inputTruncationOrPaddingLower, o);
  o << "\n  inputTruncationOrPaddingUpper=";
  printContainer(params.inputTruncationOrPaddingUpper, o);
  o << "\n  numChannels=" << params.numChannels
    << "\n  batchSize=" << params.batchSize
    << "\n  dType=" << params.dType.toString() << " }\n";
  return o;
}

static PoolOptions parsePoolOptions(const poplar::OptionFlags &options) {
  PoolOptions poolOptions;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec poolSpec{
      {"poolUseIntrospectiveMapping",
       OptionHandler::createWithBool(poolOptions.poolUseIntrospectiveMapping)},
      {"optimizeForSpeed",
       OptionHandler::createWithBool(poolOptions.optimizeForSpeed)},
  };
  for (const auto &option : options) {
    poolSpec.parse(option.first, option.second);
  }
  return poolOptions;
}

static std::string kernelShapeAsString(const std::vector<std::size_t> &shape) {
  std::string kString;
  for (std::size_t i = 0; i != shape.size() - 1; ++i) {
    kString += std::to_string(shape[i]) + "x";
  }
  kString += std::to_string(shape.back());
  return kString;
}

const char *asString(const PoolingType &pType) {
  switch (pType) {
  case PoolingType::MAX:
    return "max";
  case PoolingType::AVG:
    return "avg";
  case PoolingType::SUM:
    return "sum";
  }
  POPLIB_UNREACHABLE();
}

static void checkWindowParameters(const PoolParams &params) {
  if (params.inputFieldShape.size() != params.kernelShape.size() ||
      params.kernelShape.size() != params.stride.size() ||
      params.stride.size() != params.inputTruncationOrPaddingLower.size() ||
      params.inputTruncationOrPaddingLower.size() !=
          params.inputTruncationOrPaddingUpper.size()) {
    throw poputil::poplibs_error("Mismatched window dimensions on poplibs "
                                 "pool operation");
  }
}

// Create a convolution with parameters with same special characteristics
// as a pooling operation.
ConvParams makeConvParams(const PoolParams &poolParams) {
  const auto numFieldDims = poolParams.inputFieldShape.size();
  std::vector<unsigned> inputTruncationLower(numFieldDims),
      inputPaddingLower(numFieldDims), inputTruncationUpper(numFieldDims),
      inputPaddingUpper(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (poolParams.inputTruncationOrPaddingLower[dim] < 0) {
      inputTruncationLower[dim] =
          static_cast<unsigned>(-poolParams.inputTruncationOrPaddingLower[dim]);
    } else {
      inputPaddingLower[dim] =
          static_cast<unsigned>(poolParams.inputTruncationOrPaddingLower[dim]);
    }
    if (poolParams.inputTruncationOrPaddingUpper[dim] < 0) {
      inputTruncationUpper[dim] =
          static_cast<unsigned>(-poolParams.inputTruncationOrPaddingUpper[dim]);
    } else {
      inputPaddingUpper[dim] =
          static_cast<unsigned>(poolParams.inputTruncationOrPaddingUpper[dim]);
    }
  }

  ConvParams params{
      poolParams.dType,     // Data type
      poolParams.batchSize, // batch size
      poolParams
          .inputFieldShape,   // input field shape for each channel and batch
      poolParams.kernelShape, // kernel shape for each input & output channel
      poolParams.numChannels, // input channels
      poolParams.numChannels, // output channels
      1 // conv groups: for pooling, conv group is merged with channels/group
  };
  params.inputTransform.truncationLower = inputTruncationLower;
  params.inputTransform.truncationUpper = inputTruncationUpper;
  params.inputTransform.paddingLower = inputPaddingLower;
  params.inputTransform.paddingUpper = inputPaddingUpper;
  params.outputTransform.stride = poolParams.stride;

  return params;
}

std::vector<std::size_t> PoolParams::getOutputFieldShape() const {
  checkWindowParameters(*this);
  auto params = makeConvParams(*this);
  return params.getOutputFieldShape();
}

static std::vector<boost::icl::interval_map<std::size_t, std::size_t>>
getNumKernelPositionsUsedForOutputs(const ConvParams &params) {
  const auto numFieldDims = params.inputFieldShape.size();
  const auto outputShape = params.getOutputFieldShape();
  const auto kernelShape = params.kernelShape;

  std::vector<boost::icl::interval_map<std::size_t, std::size_t>> usedMap(
      numFieldDims);

  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    for (unsigned k = 0; k != kernelShape[dim]; ++k) {
      auto range =
          getOutputRangeForKernelIndex(dim, {0, outputShape[dim]}, k, params);
      const auto region = boost::icl::interval<std::size_t>::right_open(
          range.first, range.second);
      usedMap[dim].add(std::make_pair(region, 1));
    }
  }
  return usedMap;
}

// Scale gradient with scale factor determined by the number of samples
// used in the averaging of a pooled sample in the forward pass
static Tensor scaleGradient(Graph &graph, const ConvParams &fwdParams,
                            const Tensor &grad, Sequence &prog,
                            const std::string &debugPrefix) {
  const auto numFieldDims = fwdParams.inputFieldShape.size();
  const auto outputShape = fwdParams.getOutputFieldShape();

  const auto scaleFactorMap = getNumKernelPositionsUsedForOutputs(fwdParams);

  const auto numFieldElems = product(outputShape);
  std::vector<float> scaleOut(numFieldElems, 0.0);
  // use cartesian product from independent field dimensions
  for (std::size_t elem = 0; elem != numFieldElems; ++elem) {
    const auto index = unflattenIndex(outputShape, elem);
    float scaleThisElem = 1.0;
    for (std::size_t dim = 0; dim != numFieldDims; ++dim) {
      const auto region = boost::icl::interval<std::size_t>::right_open(
          index[dim], index[dim] + 1);
      const auto it = scaleFactorMap[dim].find(region);
      float scaleThisDim = it != scaleFactorMap[dim].end() ? it->second : 0;
      scaleThisElem *= scaleThisDim;
    }
    if (scaleThisElem != 0.0f)
      scaleOut[elem] = 1.0f / scaleThisElem;
  }

  // create constant tensor and broadcast
  const auto batchSize = grad.dim(0);
  const auto channels = grad.dim(grad.rank() - 1);
  std::vector<std::size_t> shape = {batchSize, channels};
  shape.insert(shape.end(), outputShape.begin(), outputShape.end());
  auto scaleTensor =
      graph.addConstant(grad.elementType(), {numFieldElems}, scaleOut.data(),
                        debugPrefix + "/scaleTensor");
  graph.setTileMapping(scaleTensor, 0);
  auto bScaleTensor = scaleTensor.broadcast(batchSize * channels, 0)
                          .reshape(shape)
                          .dimShufflePartial({1}, {grad.rank() - 1});
  return popops::mul(graph, grad, bScaleTensor, prog,
                     debugPrefix + "/preScale");
}

static uint64_t getFlops(const ConvParams &params, PoolingType poolingType) {
  const auto usedMap = getNumKernelPositionsUsedForOutputs(params);
  const auto outputShape = params.getOutputFieldShape();

  std::uint64_t numFlops = 0;
  unsigned addCost = poolingType == PoolingType::AVG ? 1 : 0;
  for (std::size_t f = 0; f != product(outputShape); ++f) {
    const auto index = unflattenIndex(outputShape, f);
    unsigned numKernelPosUsed = 1;
    for (std::size_t dim = 0; dim != outputShape.size(); ++dim) {
      const auto region = boost::icl::interval<std::size_t>::right_open(
          index[dim], index[dim] + 1);
      const auto it = usedMap[dim].find(region);
      numKernelPosUsed *= it != usedMap[dim].end() ? it->second : 0;
    }
    if (numKernelPosUsed)
      numFlops += numKernelPosUsed + addCost;
  }

  return params.batchSize * numFlops * params.getNumInputChans();
}

std::uint64_t getFwdFlops(const PoolParams &poolParams) {
  checkWindowParameters(poolParams);
  const auto params = makeConvParams(poolParams);
  return getFlops(params, poolParams.poolingType);
}

std::uint64_t getBwdFlops(const PoolParams &poolParams) {
  checkWindowParameters(poolParams);
  const auto params = getGradientParams(makeConvParams(poolParams));
  return getFlops(params, poolParams.poolingType);
}

double getFwdPerfectCycleCount(const Graph &graph, const PoolParams &params) {
  checkWindowParameters(params);
  const auto &target = graph.getTarget();
  unsigned dTypeSize = target.getTypeSize(params.dType);
  const auto numTiles = target.getNumTiles();
  const auto numFLOPs = getFwdFlops(params);
  const auto vectorWidth = target.getDataPathWidth() / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}

double getBwdPerfectCycleCount(const Graph &graph, const PoolParams &params) {
  checkWindowParameters(params);
  return getFwdPerfectCycleCount(graph, params) * 2;
}

// Reshape the activations tensor from [N][C][H][W] shape to [N][H][W][C]
// shape.
static Tensor actsToInternalShape(const Tensor &act) {
  return act.dimShufflePartial({1}, {act.rank() - 1});
}
// Reshape the activations tensor from  [N][H][W][C] shape to [N][C][H][W]
// shape.
static Tensor actsToExternalShape(const Tensor &act) {
  return act.dimShufflePartial({act.rank() - 1}, {1});
}

// Get the shape of the field in a tensor at the interface of pooling functions.
// The shape of the tensor at the interface is [N][C][...]
// where [...] is a general ND shape.
static std::vector<std::size_t> getInputFieldShape(const Tensor &in) {
  if (in.rank() < 2) {
    throw poputil::poplibs_error("Pooling input tensor has fewer than two "
                                 "dimensions");
  }
  const auto numFieldDims = in.rank() - 2;
  std::vector<std::size_t> inputFieldShape(numFieldDims);
  for (unsigned i = 0; i != numFieldDims; ++i) {
    inputFieldShape[i] = in.dim(i + 2);
  }
  return inputFieldShape;
}
// The activations tensor has a shape [N][H][W][C]
// We benefit most by increasing the [W] dimension as the vertex inner loop
// operates over this dimension. Here we try to do that and also combine the
// [N] and [C] to simplify work division
TransformedInput gatherDimsTransformIn(const poplar::Tensor &in_,
                                       const ConvParams &params_,
                                       unsigned minChannelGrouping) {

  // In implementing pooling, as every channel is independent there is no reason
  // internally to ever treat batches any differently to "just more channels"
  // so combine them which simplifies work division
  auto params = params_;
  const auto allChannels = params.batchSize * params.inputChannelsPerConvGroup;

  // Reshape so that we have [N][C][H][W] as we will start by manipulating
  // batches, channels
  auto in = actsToExternalShape(in_);
  in = in.reshapePartial(0, 2, {1, allChannels});

  params.batchSize = 1;
  params.inputChannelsPerConvGroup = allChannels;
  const auto spatialDims = params.inputFieldShape;

  // Under some conditions we can treat every row the same internally.
  // This is possible if each field dim can be cleanly divided into
  // independent pooling calculations.  "Cleanly" implies that each field dim is
  // exactly divisible by the stride, and that the stride >= kernel
  // This includes (but isn't limited to) the frequent use case of an
  // {n,n} kernel with a stride of {n, n} and a
  // field with dimension {m * n, m * n}.
  const bool canCombineIntoSpatialDims = [&]() {
    for (unsigned i = 0; i < spatialDims.size(); i++) {
      if (params.inputTransform.truncationLower[i] != 0 ||
          params.inputTransform.truncationUpper[i] != 0 ||
          params.inputTransform.paddingLower[i] != 0 ||
          params.inputTransform.paddingUpper[i] != 0) {
        return false;
      }
      if (params.outputTransform.stride[i] < params.kernelShape[i]) {
        return false;
      }
      if (spatialDims[i] % params.outputTransform.stride[i] != 0) {
        return false;
      }
    }
    return true;
  }();

  const bool canCombineChannelsIntoSpatialDims =
      !(allChannels % minChannelGrouping);

  if (canCombineIntoSpatialDims) {
    // Combine other spatial dimensions (if they exist) into the innermost dim
    // Create [B][Ch][heightStride][W *(H / HeightStride)]
    auto inRank = in.rank();
    if (inRank >= 4) {
      auto strideHeight = params.outputTransform
                              .stride[params.outputTransform.stride.size() - 2];
      auto transformedRows = in.shape()[in.rank() - 2] / strideHeight;
      auto innerDim = in.shape().back() * transformedRows;

      in = in.reshapePartial(inRank - 2, inRank - 1,
                             {transformedRows, strideHeight});
      in = in.dimShufflePartial({inRank - 1}, {inRank - 2});
      in =
          in.reshapePartial(in.rank() - 3, in.rank(), {strideHeight, innerDim});
      params.inputFieldShape[params.inputFieldShape.size() - 2] = strideHeight;
      params.inputFieldShape.back() = innerDim;
    }

    // Shuffle and combine channels into the innermost dimension giving
    // the opportunity to have a larger inner loop in the vertex execution
    // Create: [B=1][Ch=minChannelGrouping][H][W*channelGroups]
    if (canCombineChannelsIntoSpatialDims) {
      const auto channelGroups = allChannels / minChannelGrouping;
      in = in.reshapePartial(1, 2, {channelGroups, minChannelGrouping});
      in = in.dimShufflePartial({1}, {in.rank() - 2});
      in = in.flatten(in.rank() - 2, in.rank());
      params.inputFieldShape.back() *= channelGroups;
      params.inputChannelsPerConvGroup = minChannelGrouping;
    }
  }
  params.outputChannelsPerConvGroup = params.inputChannelsPerConvGroup;

  // Return the shape to [N][C][H][W]
  in = actsToInternalShape(in);
  return {in, params, minChannelGrouping,
          (canCombineIntoSpatialDims && canCombineChannelsIntoSpatialDims)};
}

// Reverse the transformation created by transformInput.
Tensor gatherDimsTransformOut(const Tensor &result_, const ConvParams &params,
                              const TransformedInput &transform) {
  auto result = result_;
  if (transform.channelsWereTransformed) {
    auto allChannels = params.batchSize * params.inputChannelsPerConvGroup;
    const auto channelGroups = allChannels / transform.channelGrouping;
    // Reverse the transform that combined channels into the innerMost dim
    result = result.reshapePartial(
        result.rank() - 1, result.rank(),
        {channelGroups, result.shape().back() / channelGroups});
    result = result.dimShufflePartial({result.rank() - 2}, {1});
  }

  // A simple reshape can split channels, batches back out and reverse the
  // transformation that combined the 2 innermost dimensions
  auto resultShape = params.getOutputFieldShape();
  resultShape.insert(resultShape.begin(),
                     {params.batchSize, params.inputChannelsPerConvGroup});

  return result.reshape(resultShape);
}
// Creates an output tensor and pre-process the input tensor and parameters
// according to the plan for implementing this pooling operation.
// The input and output tensors may be padded to match the planning
// parameters. The parameters given may be modified if there is some
// transformation.
static Tensor createOutputAndPreprocess(Graph &graph, ConvParams &params,
                                        const Plan &plan, Tensor &in,
                                        Tensor *fwdInputActs,
                                        Tensor *fwdOutputActs,
                                        const std::string &debugPrefix) {
  // Check if the params match the input tensor
  assert(params.batchSize == in.dim(0));
  assert(params.getNumInputChans() == in.dim(in.rank() - 1));
  assert(params.getNumInputChans() == params.getNumOutputChans());

  params = applyTransform(std::move(params), plan.transform,
                          {&in, fwdInputActs, fwdOutputActs});

  const auto numInChans = params.getNumInputChans();
  const auto numChanGroups = (numInChans + plan.partition.chansPerGroup - 1) /
                             plan.partition.chansPerGroup;

  // this should already include padding if required
  std::size_t paddedChans = numChanGroups * plan.partition.chansPerGroup;

  // padded channels should be a multiple of the number of elements that can
  // be stored in 64-bits.
  if (paddedChans % (params.outputType == HALF ? 4 : 2) != 0) {
    throw poputil::poplibs_error("Expected padded channels (" +
                                 std::to_string(paddedChans) +
                                 ")"
                                 " to be a multiple of 64-bits.");
  }

  // create shape for output tensor
  std::vector<std::size_t> outTensorShape = {numChanGroups, in.dim(0)};
  const auto &outputShape = params.getOutputFieldShape();
  outTensorShape.insert(outTensorShape.end(), outputShape.begin(),
                        outputShape.end());
  outTensorShape.push_back(plan.partition.chansPerGroup);
  auto out = graph.addVariable(params.outputType, outTensorShape,
                               debugPrefix + "/out");
  // Default mapping in case there are padding elements
  // TODO: T12915 Improve the handling of padding elements.
  mapTensorLinearly(graph, out);

  // pad input channels if they don't match the planner
  if (numInChans != paddedChans) {
    assert(numInChans <= paddedChans);
    in = popops::pad(graph, in, 0, static_cast<int>(paddedChans - numInChans),
                     in.rank() - 1);
    if (fwdInputActs) {
      *fwdInputActs = popops::pad(graph, *fwdInputActs, 0,
                                  static_cast<int>(paddedChans - numInChans),
                                  fwdInputActs->rank() - 1);
    }
    if (fwdOutputActs) {
      *fwdOutputActs = popops::pad(graph, *fwdOutputActs, 0,
                                   static_cast<int>(paddedChans - numInChans),
                                   fwdOutputActs->rank() - 1);
    }
  }

  // This has format [G][B][...][CPG]
  in = in.reshapePartial(in.rank() - 1, in.rank(),
                         {numChanGroups, plan.partition.chansPerGroup})
           .dimShufflePartial({0, in.rank() - 1}, {1, 0});
  if (fwdInputActs) {
    // This has format [G][B][...][CPG]
    const auto rank = fwdInputActs->rank();
    *fwdInputActs =
        fwdInputActs
            ->reshapePartial(rank - 1, rank,
                             {numChanGroups, plan.partition.chansPerGroup})
            .dimShufflePartial({0, rank - 1}, {1, 0});
  }
  if (fwdOutputActs) {
    // This has format [G][B][...][CPG]
    const auto rank = fwdOutputActs->rank();
    *fwdOutputActs =
        fwdOutputActs
            ->reshapePartial(rank - 1, rank,
                             {numChanGroups, plan.partition.chansPerGroup})
            .dimShufflePartial({0, rank - 1}, {1, 0});
  }
  return out;
}

// Post process output. Remove any padding that may have been added and reshape
// to match desired output shape
static void postProcess(const ConvParams &params, const Plan &plan,
                        Tensor &out) {
  const auto transformedParams = applyTransform(params, plan.transform);
  const auto numOutChans = transformedParams.getNumOutputChans();
  // flatten channel groups
  out = out.dimRoll(0, out.rank() - 2).flatten(out.rank() - 2, out.rank());
  // remove any padding
  if (numOutChans != out.dim(out.rank() - 1)) {
    out = out.slice(0, numOutChans, out.rank() - 1);
  }
  // undo any transforms
  applyTransformInverse(params, plan.transform, {&out});
  // dim shuffle back to expected output shape
  out = out.dimRoll(out.rank() - 1, 1);
}
static Tensor poolingImpl(Graph &graph, const PoolConfig &poolCfg,
                          const Tensor &in_, const Tensor *fwdInputActs_,
                          const Tensor *fwdOutputActs_,
                          const ConvParams &params, Sequence &prog,
                          const std::string &debugPrefix,
                          const PoolOptions &poolOptions) {
  if (poolCfg.pass == PoolPass::POOL_FWD ||
      (poolCfg.pass == PoolPass::POOL_BWD &&
       (poolCfg.type == PoolingType::AVG ||
        poolCfg.type == PoolingType::SUM))) {
    assert(fwdInputActs_ == nullptr);
    if (poolCfg.type != PoolingType::MAX || !poolCfg.scaledGradient)
      assert(fwdOutputActs_ == nullptr);
  } else {
    assert(in_.shape() == fwdOutputActs_->shape());
  }

  // Attempt to transform the input to achieve a faster implementation.
  // Do this twice, once leaving a minimum number of channels = the vector
  // width of the data type, and also leaving a preferred channel grouping.
  // The preferred channel grouping can be chosen by the planner if the cost
  // of implementation is not much more than the optimum
  TransformedInput transformVectorWidth;
  TransformedInput transformGroupedWidth;
  bool isFwdPass =
      (poolCfg.pass == PoolPass::POOL_FWD) && !poolCfg.scaledGradient;
  if (isFwdPass) {
    const unsigned vectorWidth = (in_.elementType() == HALF ? 4 : 2);
    const auto groupedWidth = getPreferredChannelGrouping(in_.elementType());
    transformVectorWidth = gatherDimsTransformIn(in_, params, vectorWidth);
    transformGroupedWidth = gatherDimsTransformIn(in_, params, groupedWidth);
  } else {
    transformVectorWidth.in = in_;
    transformVectorWidth.params = params;
    transformGroupedWidth = transformVectorWidth;
  }
  auto poolMethodResult =
      getPlan(graph, poolCfg, transformVectorWidth, transformGroupedWidth);
  auto &chosenTransform = poolMethodResult.useGroupedWidth
                              ? transformGroupedWidth
                              : transformVectorWidth;

  const auto poolingHasConvolutionEquivalent = [&poolCfg, &params]() {
    if (poolCfg.type == PoolingType::MAX) {
      // Max pooling is not equivalent to convolution
      return false;
    }
    if (poolCfg.type == PoolingType::AVG &&
        poolCfg.pass == PoolPass::POOL_FWD) {
      // Average pooling is equivalent to convolution in general.
      // Scale is given by:
      // scale = 1/(Number of non-padding elements in the kernel's operating
      //         region)
      // So where we use scale !=1.0 and there is input padding, any part of the
      // pooling that covers a padded input would have a different scale.
      // As we plan to use a simple convolution kernel with all
      // elements = scale, this is not so elegant so don't attempt to use
      // convolution to implement pooling
      for (unsigned i = 0; i < params.inputTransform.paddingLower.size(); i++) {
        if (params.inputTransform.paddingLower[i] != 0 ||
            params.inputTransform.paddingUpper[i] != 0) {
          return false;
        }
      }
    }
    // SUM and remaining AVG pool cases are equivalent to convolution
    return true;
  }();

  if (poolOptions.optimizeForSpeed && poolingHasConvolutionEquivalent) {
    // Describe the operation slightly differently for a convolution that is
    // equivalent to pooling.  For pooling, each channel is independent.
    auto paramsForConv = params;
    paramsForConv.numConvGroups = params.inputChannelsPerConvGroup;
    paramsForConv.inputChannelsPerConvGroup = 1;
    paramsForConv.outputChannelsPerConvGroup = 1;

    const poplar::OptionFlags convOptions = {
        {"partialsType", in_.elementType() == HALF ? "half" : "float"}};
    poplin::PlanningCache cache;
    auto convMethodCosts =
        reportPlanEstimatedCosts(graph, paramsForConv, convOptions, &cache);

    logging::popnn::debug(
        "Choosing method between Pool {} cycles or Conv {} cycles ",
        poolMethodResult.cycles, convMethodCosts.cycles);
    if (convMethodCosts.cycles < poolMethodResult.cycles) {
      // Create a scaling tensor, which will become the convolution weights
      // kernel when broadcast
      auto kernelElements =
          std::accumulate(params.kernelShape.begin(), params.kernelShape.end(),
                          1.0, std::multiplies<double>());

      const auto scaleToFindMean = 1.0 / kernelElements;
      const auto scaleValue = (poolCfg.pass == PoolPass::POOL_FWD &&
                               poolCfg.type == PoolingType::AVG)
                                  ? scaleToFindMean
                                  : 1.0;
      auto weights = graph.addConstant(in_.elementType(), {1}, scaleValue);

      // Convolution weights: [convGroups, outCPG, inCPG, H, W]
      // For pooling we have: [convGroups, 1, 1, H, W]
      // So broadcast the scalar `weights` to match
      std::vector<std::size_t> zeros(params.kernelShape.size() + 2, 0);
      weights = weights.expand(zeros);
      weights = weights.broadcast(paramsForConv.numConvGroups, 0);
      for (unsigned i = 0; i < params.kernelShape.size(); i++) {
        weights =
            weights.broadcast(params.kernelShape[i],
                              weights.rank() - params.kernelShape.size() + i);
      }
      mapTensorLinearly(graph, weights);

      // Shuffle to match convolution input: [batches, channels, H, W]
      auto in = in_.dimShufflePartial({in_.rank() - 1}, {1});

      return poplin::convolution(graph, in, weights, paramsForConv, false, prog,
                                 debugPrefix, convOptions, &cache);
    }
  }
  // Implement as pooling
  logging::popnn::debug("Plan cost: {} cycles", poolMethodResult.cycles);
  logging::popnn::debug("Pooling plan:\n{}", poolMethodResult.plan);
  Tensor fwdInputActs, fwdOutputActs;
  if (fwdInputActs_) {
    fwdInputActs = *fwdInputActs_;
  }
  if (fwdOutputActs_) {
    fwdOutputActs = *fwdOutputActs_;
  }
  auto preprocessedParams = chosenTransform.params;
  // preprocessing may create new tensors
  auto out = createOutputAndPreprocess(
      graph, preprocessedParams, poolMethodResult.plan, chosenTransform.in,
      fwdInputActs_ ? &fwdInputActs : nullptr,
      fwdOutputActs_ ? &fwdOutputActs : nullptr, debugPrefix);
  tilePartitions(graph, poolCfg, chosenTransform.in, out,
                 fwdInputActs_ ? &fwdInputActs : nullptr,
                 fwdOutputActs_ ? &fwdOutputActs : nullptr, preprocessedParams,
                 prog, poolMethodResult.plan, debugPrefix, poolOptions);
  postProcess(chosenTransform.params, poolMethodResult.plan, out);
  if (isFwdPass) {
    return gatherDimsTransformOut(out, params, chosenTransform);
  } else {
    return out;
  }
}

static Tensor poolingFwd(Graph &graph, const Tensor &in_,
                         const ConvParams &fwdParams, PoolingType poolingType,
                         Sequence &prog, const std::string &debugPrefix,
                         const PoolOptions &poolOptions) {
  return poolingImpl(graph, {poolingType, PoolPass::POOL_FWD, false}, in_,
                     nullptr, nullptr, fwdParams, prog, debugPrefix,
                     poolOptions);
}

static Tensor poolingMaxScale(Graph &graph, const Tensor &in_,
                              const Tensor &fwdOut, const ConvParams &fwdParams,
                              Sequence &prog, const std::string &debugPrefix,
                              const PoolOptions &poolOptions) {
  const auto output =
      poolingImpl(graph, {PoolingType::MAX, PoolPass::POOL_FWD, true}, in_,
                  nullptr, &fwdOut, fwdParams, prog, debugPrefix, poolOptions);
  // poolingImpl shapes output to be as required at the API interface. Reshape
  // back to internal shape
  return actsToInternalShape(output);
}

static Tensor poolingBwd(Graph &graph, const Tensor &in_,
                         const ConvParams &bwdParams, PoolingType poolingType,
                         Sequence &prog, const std::string &debugPrefix,
                         const PoolOptions &poolOptions) {
  return poolingImpl(graph, {poolingType, PoolPass::POOL_BWD, false}, in_,
                     nullptr, nullptr, bwdParams, prog, debugPrefix,
                     poolOptions);
}

static Tensor poolingBwd(Graph &graph, const Tensor &in_,
                         const Tensor &fwdInputActs,
                         const Tensor &fwdOutputActs,
                         const ConvParams &bwdParams, PoolingType poolingType,
                         Sequence &prog, const std::string &debugPrefix,
                         const PoolOptions &poolOptions) {
  return poolingImpl(graph, {poolingType, PoolPass::POOL_BWD, false}, in_,
                     &fwdInputActs, &fwdOutputActs, bwdParams, prog,
                     debugPrefix, poolOptions);
}

static bool detectMatchingFieldAndKernel(const ConvParams &params) {
  auto allZeros = [](const std::vector<unsigned> &vec) {
    return std::all_of(vec.begin(), vec.end(),
                       [](unsigned e) { return e == 0; });
  };
  return params.kernelShape == params.inputFieldShape &&
         allZeros(params.inputTransform.paddingLower) &&
         allZeros(params.inputTransform.paddingUpper) &&
         allZeros(params.inputTransform.truncationLower) &&
         allZeros(params.inputTransform.truncationUpper);
}

// When the kernal and input field shapes match and there is no padding,
// the gradient operation is the same as a broadcast operation for AVG and
// SUM pooling
static bool substPoolingGradientWithBroadcast(const ConvParams &params) {
  return detectMatchingFieldAndKernel(params);
}

// When the kernal and input field shapes match and there is no padding,
// the fwd operation is the same as a reduction operation for AVG and
// SUM pooling
static bool substPoolingWithReduction(const ConvParams &params) {
  return detectMatchingFieldAndKernel(params);
}

Tensor pool(Graph &graph, const PoolParams &poolParams, const Tensor &in_,
            Sequence &prog, const poplar::DebugContext &debugContext,
            const poplar::OptionFlags &options) {
  const auto debugPrefix = debugContext.getPathName();
  const auto poolOptions = parsePoolOptions(options);
  checkWindowParameters(poolParams);

  logging::popnn::info("pool");
  logging::popnn::info("  method={} type={} name=\"{}\"",
                       asString(poolParams.poolingType), poolParams.dType,
                       debugPrefix);
  logging::popnn::info("  input={}x({}x{}) padding={}/{}",
                       poolParams.inputFieldShape, poolParams.batchSize,
                       poolParams.numChannels,
                       poolParams.inputTruncationOrPaddingLower,
                       poolParams.inputTruncationOrPaddingUpper);
  logging::popnn::info("  kernel={}", poolParams.kernelShape);
  logging::popnn::info("  output={}x({}x{}) stride={}",
                       poolParams.getOutputFieldShape(), poolParams.batchSize,
                       poolParams.numChannels, poolParams.stride);

  const auto poolingType = poolParams.poolingType;
  if (in_.dim(0) != poolParams.batchSize ||
      in_.dim(1) != poolParams.numChannels) {
    throw poputil::poplibs_error(
        "First 2 dimensions of pooling input Tensor don't "
        "match poolParams.batchSize and poolParams.numChannels. "
        "Expected " +
        std::to_string(poolParams.batchSize) + ", " +
        std::to_string(poolParams.numChannels) + "and got " +
        std::to_string(in_.dim(0)) + ", " + std::to_string(in_.dim(1)));
  }

  const auto dType = in_.elementType();
  ConvParams convParams = makeConvParams(poolParams);

  const auto layerName = debugPrefix + "/" + asString(poolingType) + "Pool" +
                         kernelShapeAsString(poolParams.kernelShape) + "/Fwd";
  // Special handling when pooling can be represented as a reduction operation.
  // This is done because average pooling is slower because codelets handle
  // only a multiple of 4 channels and kernel is not split.
  if (substPoolingWithReduction(convParams)) {
    std::vector<std::size_t> reduceDims(in_.rank() - 2);
    std::iota(reduceDims.begin(), reduceDims.end(), 2);
    const auto kernelElems = std::accumulate(poolParams.kernelShape.begin(),
                                             poolParams.kernelShape.end(), 1U,
                                             std::multiplies<unsigned>());
    popops::ReduceParams params;
    if (poolingType == PoolingType::MAX) {
      params = {popops::Operation::MAX};
    } else {
      if (poolingType == PoolingType::AVG) {
        auto scale = graph.addConstant(FLOAT, {}, 1.0f / kernelElems,
                                       debugPrefix + "/scale");
        graph.setTileMapping(scale, 0);
        params = {popops::Operation::ADD, false, scale};
      } else {
        params = {popops::Operation::ADD};
      }
    }
    auto t =
        popops::reduce(graph, in_, dType, reduceDims, params, prog, layerName);
    for (auto i = 0U; i != reduceDims.size(); ++i) {
      t = t.expand({2 + i});
    }
    return t;
  }
  // Convert activations to internal shape
  auto in = actsToInternalShape(in_);
  return poolingFwd(graph, in, convParams, poolingType, prog, layerName,
                    poolOptions);
}

void poolInputGradientImpl(Graph &graph, const PoolParams &poolParams,
                           const Tensor &in_, const Tensor &pooled_,
                           const Tensor &pooledGradient_, Tensor &output,
                           const bool useScaledGradForMaxPool, Sequence &prog,
                           const std::string &debugPrefix,
                           const poplar::OptionFlags &options) {
  checkWindowParameters(poolParams);
  const auto poolOptions = parsePoolOptions(options);
  const auto poolingType = poolParams.poolingType;
  const bool maxPooling = poolingType == PoolingType::MAX;
  const auto inputFieldShape = getInputFieldShape(output);
  const auto batchSize = output.dim(0);
  const auto numChannels = output.dim(1);
  assert(poolParams.batchSize == batchSize);
  assert(poolParams.numChannels == numChannels);

  logging::popnn::debug("PoolGrad({}x({}x{}), kernel {}, name='{}'",
                        poolParams.inputFieldShape, poolParams.batchSize,
                        poolParams.numChannels, poolParams.kernelShape,
                        debugPrefix);
  logging::popnn::debug("PoolGrad params:\n{}", poolParams);

  Tensor in, pooled;
  if (maxPooling) {
    in = actsToInternalShape(in_);
    pooled = actsToInternalShape(pooled_);
  }
  const auto dType = pooledGradient_.elementType();
  ConvParams fwdParams = makeConvParams(poolParams);

  auto pooledGradient = actsToInternalShape(pooledGradient_);
  const auto layerName = debugPrefix + "/" + asString(poolingType) + "Pool" +
                         kernelShapeAsString(poolParams.kernelShape) + "/Bwd";
  const auto numFieldDims = inputFieldShape.size();

  if (maxPooling) {
    if (pooledGradient.dim(0) != batchSize || pooled.dim(0) != batchSize) {
      throw poputil::poplibs_error("Forward pass batch size does not match "
                                   "gradient calculation pass");
    }
    assert(pooled.rank() == pooledGradient.rank());
    const auto channelDim = pooled.rank() - 1;
    if (pooledGradient.dim(channelDim) != numChannels ||
        pooled.dim(channelDim) != numChannels) {
      throw poputil::poplibs_error("Forward pass number of channels does not "
                                   "match gradient calculation pass");
    }
    if (pooled.rank() != numFieldDims + 2) {
      throw poputil::poplibs_error("Number of output field dimensions do not "
                                   "match the input activation dimensions");
    }
    // Check if gradient field shapes match
    for (std::size_t dim = 0; dim != numFieldDims; ++dim) {
      if (pooled.dim(1 + dim) != pooledGradient.dim(1 + dim)) {
        throw poputil::poplibs_error("Forward pass output and gradient "
                                     "calculation size for dim " +
                                     std::to_string(dim) + " do not match");
      }
    }
  }
  if (pooledGradient.rank() != numFieldDims + 2) {
    throw poputil::poplibs_error("Gradient calculation pass output field size "
                                 "does not match input activations size");
  }
  auto bwdParams = getGradientParams(fwdParams);

  if (poolingType == PoolingType::SUM || poolingType == PoolingType::AVG) {
    // For certain pooling parameters the gradient operation can be cast as a
    // scaled broadcast operation
    if (substPoolingGradientWithBroadcast(fwdParams)) {
      auto scaledPooledGradient = pooledGradient_;
      if (poolingType == PoolingType::AVG) {
        float scale = 1.0f / product(poolParams.kernelShape);
        auto scaleTensor =
            graph.addConstant(dType, pooledGradient_.shape(), scale,
                              debugPrefix + "/scaleTensor");
        graph.setTileMapping(scaleTensor, 0);
        scaledPooledGradient =
            popops::mul(graph, pooledGradient_, scaleTensor, prog, layerName);
      }
      // do an explicit copy instead of returning a view
      prog.add(
          Copy(scaledPooledGradient.broadcast(product(fwdParams.kernelShape), 2)
                   .reshape(output.shape()),
               output));
      return;
    }

    if (poolingType == PoolingType::AVG) {
      pooledGradient =
          scaleGradient(graph, fwdParams, pooledGradient, prog, layerName);
    }
    output = poolingBwd(graph, pooledGradient, bwdParams, poolingType, prog,
                        layerName, poolOptions);
    return;
  } else if (poolingType == PoolingType::MAX) {
    Tensor gradient;
    if (useScaledGradForMaxPool) {
      auto scale = poolingMaxScale(graph, in, pooled, fwdParams, prog,
                                   layerName + "/Scale", poolOptions);
      gradient = popops::mul(graph, pooledGradient, scale, prog,
                             layerName + "/ScaleGrad");
    } else {
      gradient = pooledGradient;
    }

    // Do an explicit copy of the gradients to match the pooled forward tensor
    // This reduces exchange code.
    auto gradsRearranged = graph.clone(pooled, layerName + "/gradsRearranged");
    prog.add(Copy(gradient, gradsRearranged));
    output = poolingBwd(graph, gradsRearranged, in, pooled, bwdParams,
                        poolingType, prog, layerName, poolOptions);
    return;
  } else {
    throw poputil::poplibs_error("Unexpected pooling type");
  }
}

Tensor poolInputGradient(Graph &graph, const PoolParams &poolParams,
                         const Tensor &in_, const Tensor &pooled_,
                         const Tensor &pooledGradient_, bool useScaledGradient,
                         Sequence &prog,
                         const poplar::DebugContext &debugContext,
                         const poplar::OptionFlags &options) {
  const auto debugPrefix = debugContext.getPathName();
  // create the output tensor, based on the input
  auto output = graph.clone(in_);
  poolInputGradientImpl(graph, poolParams, in_, pooled_, pooledGradient_,
                        output, useScaledGradient, prog, debugPrefix, options);
  return output;
}

Tensor poolInputGradient(Graph &graph, const PoolParams &poolParams,
                         const unsigned fwdChansPerGroup,
                         const Tensor &pooledGradient_, Sequence &prog,
                         const poplar::DebugContext &debugContext,
                         const poplar::OptionFlags &options) {
  const auto debugPrefix = debugContext.getPathName();
  assert(poolParams.poolingType != PoolingType::MAX);

  // Create the output tensor, based on the parameters provided
  std::vector<std::size_t> shape;
  shape.reserve(2 + poolParams.inputFieldShape.size() + 1);
  shape.push_back(poolParams.numChannels / fwdChansPerGroup);
  shape.push_back(poolParams.batchSize);
  shape.insert(std::end(shape), std::begin(poolParams.inputFieldShape),
               std::end(poolParams.inputFieldShape));
  shape.push_back(fwdChansPerGroup);

  const auto elementType = pooledGradient_.elementType();
  Tensor output = graph.addVariable(elementType, std::move(shape));
  mapTensorLinearly(graph, output);
  output = output.dimShufflePartial({0, output.rank() - 1}, {1, 2})
               .reshapePartial(1, 3, {poolParams.numChannels});
  poolInputGradientImpl(graph, poolParams, {}, {}, pooledGradient_, output,
                        false, prog, debugPrefix, options);
  return output;
}

} // namespace pooling
} // namespace popnn
