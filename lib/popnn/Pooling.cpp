// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include "PoolPlan.hpp"
#include "PoolVertices.hpp"
#include "PoolOptions.hpp"
#include "popnn/Pooling.hpp"
#include "popops/Pad.hpp"
#include "popops/Reduce.hpp"
#include "poputil/TileMapping.hpp"
#include "poplin/ConvUtil.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poplibs_support/Compiler.hpp"
#include "popops/ElementWise.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include <cassert>
#include <map>
#include <boost/icl/interval_map.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using std::tie;
using namespace poputil;

namespace popnn {
namespace pooling {

static PoolOptions parsePoolOptions(const poplar::OptionFlags &options) {
  PoolOptions poolOptions;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec poolSpec{
    { "poolUseIntrospectiveMapping", OptionHandler::createWithBool(
      poolOptions.poolUseIntrospectiveMapping)},
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
  case PoolingType::MAX: return "max";
  case PoolingType::AVG: return "avg";
  case PoolingType::SUM: return "sum";
  }
  POPLIB_UNREACHABLE();
}

static void
checkWindowParameters(const PoolParams &params) {
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
static ConvParams
makeConvParams(const PoolParams &poolParams) {
  const auto numFieldDims = poolParams.inputFieldShape.size();
  std::vector<unsigned> inputTruncationLower(numFieldDims),
                        inputPaddingLower(numFieldDims),
                        inputTruncationUpper(numFieldDims),
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

  const std::vector<bool> flip(numFieldDims, false);
  const std::vector<unsigned> ones(numFieldDims, 1);
  const std::vector<unsigned> zeros(numFieldDims, 0);
  return  {poolParams.dType,
           // batch size
           poolParams.batchSize,
           // input field shape for each channel and batch
           poolParams.inputFieldShape,
           // kernel shape for each input and output channel
           poolParams.kernelShape,
           // input channels
           poolParams.numChannels,
           // output channels
           poolParams.numChannels,
           // conv groups: for pooling, conv group is merged with channels/group
           1,
           // input truncation lower
           inputTruncationLower,
           // input truncation upper
           inputTruncationUpper,
           // input dilation
           ones,
           inputPaddingLower,
           inputPaddingUpper,
           // flip input
           flip,
           // kernel truncation lower
           zeros,
           // kernel truncation upper
           zeros,
           // kernel dilation
           ones,
           // kernel padding lower
           zeros,
           // kernel padding upper
           zeros,
           // flip kernel
           flip,
           // output truncation lower
           zeros,
           // output truncation upper
           zeros,
           poolParams.stride,
           // output padding lower
           zeros,
           // output padding upper
           zeros};
}


std::vector<std::size_t>
PoolParams::getOutputFieldShape() const {
  checkWindowParameters(*this);
  auto params = makeConvParams(*this);
  return params.getOutputFieldShape();
}

static std::vector<boost::icl::interval_map<std::size_t, std::size_t>>
getNumKernelPositionsUsedForOutputs(const ConvParams &params) {
  const auto numFieldDims = params.inputFieldShape.size();
  const auto outputShape = params.getOutputFieldShape();
  const auto kernelShape = params.kernelShape;

  std::vector<boost::icl::interval_map<std::size_t, std::size_t>>
        usedMap(numFieldDims);

  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    for (unsigned k = 0; k != kernelShape[dim]; ++k) {
      auto range = getOutputRangeForKernelIndex(dim, {0, outputShape[dim]},
                                                k, params);
      const auto region =
          boost::icl::interval<std::size_t>::right_open(range.first,
                                                          range.second);
      usedMap[dim].add(std::make_pair(region, 1));
    }
  }
  return usedMap;
}

// Scale gradient with scale factor determined by the number of samples
// used in the averaging of a pooled sample in the forward pass
static Tensor
scaleGradient(Graph &graph,
              const ConvParams &fwdParams,
              const Tensor &grad,
              Sequence &prog,
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
      const auto region =
            boost::icl::interval<std::size_t>::right_open(index[dim],
                                                          index[dim] + 1);
      const auto it = scaleFactorMap[dim].find(region);
      float scaleThisDim = it != scaleFactorMap[dim].end() ? it->second : 0;
      scaleThisElem *= scaleThisDim;
    }
    if (scaleThisElem != 0.0f)
      scaleOut[elem] = 1.0f / scaleThisElem;
  }

  // create constant tensor and broadcast
  const auto batchSize = grad.dim(0);
  const auto channels = grad.dim(3);
  std::vector<std::size_t> shape = {batchSize, channels};
  shape.insert(shape.end(), outputShape.begin(), outputShape.end());
  auto scaleTensor =
      graph.addConstant(grad.elementType(), { numFieldElems }, scaleOut.data());
  graph.setTileMapping(scaleTensor, 0);
  auto bScaleTensor =
    scaleTensor.broadcast(batchSize * channels, 0)
               .reshape(shape)
               .dimShufflePartial({1}, {3});
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
      const auto region =
            boost::icl::interval<std::size_t>::right_open(index[dim],
                                                          index[dim] + 1);
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

double getFwdPerfectCycleCount(const Graph &graph,
                              const PoolParams &params) {
  checkWindowParameters(params);
  const auto &target = graph.getTarget();
  unsigned dTypeSize = target.getTypeSize(params.dType);
  const auto numTiles = target.getNumTiles();
  const auto numFLOPs = getFwdFlops(params);
  const auto vectorWidth = target.getDataPathWidth() / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}

double getBwdPerfectCycleCount(const Graph &graph,
                               const PoolParams &params) {
  checkWindowParameters(params);
  return getFwdPerfectCycleCount(graph, params) * 2;
}

// Reshape the activations tensor from [N][C][H][W] shape to [N][H][W][C]
// shape.
static Tensor
actsToInternalShape(const Tensor &act) {
  return act.dimShufflePartial({1}, {act.rank() - 1});
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

// Creates an output tensor and pre-process the input tensor such that they
// have the shape required by the pooling operation. The input and
// output tensors may be padded to match the planning parameters.
static Tensor
createOutputAndPreprocess(Graph &graph,
                          const ConvParams &params,
                          const Partition &plan,
                          Tensor &in,
                          Tensor *fwdInputActs,
                          Tensor *fwdOutputActs,
                          const std::string &debugPrefix) {
  // Check if the params match the input tensor
  const auto numInChans = params.getNumInputChans();
  assert(params.batchSize == in.dim(0));
  assert(numInChans == in.dim(in.rank() - 1));
  assert(numInChans == params.getNumOutputChans());

  const auto numChanGroups = (numInChans + plan.chansPerGroup - 1) /
                              plan.chansPerGroup;

  // this should already include padding if required
  std::size_t paddedChans = numChanGroups * plan.chansPerGroup;

  // padded channels should be a multiple of the number of elements that can
  // be stored in 64-bits.
  if (paddedChans % (params.dType == HALF ? 4 : 2) != 0) {
    throw poputil::poplibs_error(
      "Expected padded channels (" + std::to_string(paddedChans) + ")"
        " to be a multiple of 64-bits.");
  }

  // create shape for output tensor
  std::vector<std::size_t> outTensorShape = {numChanGroups, in.dim(0)};
  const auto &outputShape = params.getOutputFieldShape();
  outTensorShape.insert(outTensorShape.end(),
                        outputShape.begin(),
                        outputShape.end());
  outTensorShape.push_back(plan.chansPerGroup);
  auto out =
      graph.addVariable(params.dType, outTensorShape, debugPrefix + "/out");
  // default mapping in case there are padding elements
  // TODO: handle padding elements properly
  mapTensorLinearly(graph, out);

  // pad input channels if they don't match the planner
  if (numInChans != paddedChans) {
    assert(numInChans <= paddedChans);
    in = popops::pad(graph, in, 0, static_cast<int>(paddedChans - numInChans),
                     in.rank() - 1);
    if (fwdInputActs) {
      *fwdInputActs =
            popops::pad(graph, *fwdInputActs, 0,
                        static_cast<int>(paddedChans - numInChans),
                        fwdInputActs->rank() - 1);
    }
    if (fwdOutputActs) {
      *fwdOutputActs =
              popops::pad(graph, *fwdOutputActs, 0,
                          static_cast<int>(paddedChans - numInChans),
                          fwdOutputActs->rank() - 1);
    }
  }

  // This has format [G][B][...][CPG]
  in = in.reshapePartial(in.rank() - 1, in.rank(),
                         {numChanGroups, plan.chansPerGroup})
         .dimShufflePartial({0, in.rank() - 1}, {1, 0});
  if (fwdInputActs) {
    // This has format [G][B][...][CPG]
    const auto rank = fwdInputActs->rank();
    *fwdInputActs = fwdInputActs->reshapePartial(rank - 1, rank,
                                                 {numChanGroups,
                                                  plan.chansPerGroup})
                                  .dimShufflePartial({0, rank - 1}, {1, 0});
  }
  if (fwdOutputActs) {
    // This has format [G][B][...][CPG]
    const auto rank = fwdOutputActs->rank();
    *fwdOutputActs = fwdOutputActs->reshapePartial(rank - 1, rank,
                                                  {numChanGroups,
                                                   plan.chansPerGroup})
                                   .dimShufflePartial({0, rank - 1}, {1, 0});
  }
  return out;
}

// Post process output. Remove any padding that may have been added and reshape
// to match desired output shape
static void
postProcess(const ConvParams &params, Tensor &out) {
  const auto numOutChans = params.getNumOutputChans();
  const auto numPaddedChans = out.dim(0) * out.dim(out.rank() - 1);
  // reshape output to match output dimension ordering. i.e. [B][C][...]
  out =  out.dimShufflePartial({0, out.rank() - 1}, {1, 2})
            .reshapePartial(1, 3, {numPaddedChans});
  // remove any padding
  if (numOutChans != numPaddedChans)
    out = out.slice(0, numOutChans, 1);
}

static Tensor
poolingImpl(Graph &graph,
            const PoolConfig &poolCfg,
            const Tensor &in_,
            const Tensor *fwdInputActs_,
            const Tensor *fwdOutputActs_,
            const ConvParams &params,
            Sequence &prog,
            const std::string &debugPrefix,
            const PoolOptions &poolOptions) {
  if (poolCfg.pass == PoolPass::POOL_FWD ||
      (poolCfg.pass == PoolPass::POOL_BWD &&
       (poolCfg.type == PoolingType::AVG ||
        poolCfg.type == PoolingType::SUM))) {
    assert(fwdInputActs_ == nullptr);
    if (poolCfg.type != PoolingType::MAX ||
        !poolCfg.scaledGradient)
      assert(fwdOutputActs_ == nullptr);
  } else {
    assert(in_.shape() == fwdOutputActs_->shape());
  }

  const auto plan = getPlan(graph,
                            poolCfg,
                            params,
                            in_.shape(),
                            detectChannelGrouping(graph, in_));
  Tensor fwdInputActs, fwdOutputActs;
  if (fwdInputActs_) {
    fwdInputActs = *fwdInputActs_;
  }
  if (fwdOutputActs_) {
    fwdOutputActs = *fwdOutputActs_;
  }
  auto in = in_;
  // preprocessing may create new tensors
  auto out =
      createOutputAndPreprocess(graph,
                                params,
                                plan,
                                in,
                                fwdInputActs_ ? &fwdInputActs : nullptr,
                                fwdOutputActs_ ? &fwdOutputActs : nullptr,
                                debugPrefix);
  tilePartitions(graph,
                 poolCfg,
                 in,
                 out,
                 fwdInputActs_ ? &fwdInputActs : nullptr,
                 fwdOutputActs_ ? &fwdOutputActs : nullptr,
                 params,
                 prog,
                 plan,
                 debugPrefix,
                 poolOptions);
  postProcess(params, out);
  return out;
}

static Tensor poolingFwd(Graph &graph,
                         const Tensor &in_,
                         const ConvParams &fwdParams,
                         PoolingType poolingType,
                         Sequence &prog,
                         const std::string &debugPrefix,
                         const PoolOptions &poolOptions) {
  return poolingImpl(graph, {poolingType, PoolPass::POOL_FWD, false}, in_,
                     nullptr, nullptr, fwdParams, prog, debugPrefix,
                     poolOptions);
}

static Tensor poolingMaxScale(Graph &graph,
                              const Tensor &in_,
                              const Tensor &fwdOut,
                              const ConvParams &fwdParams,
                              Sequence &prog,
                              const std::string &debugPrefix,
                              const PoolOptions &poolOptions) {
  const auto output =
      poolingImpl(graph, {PoolingType::MAX, PoolPass::POOL_FWD, true},
                  in_, nullptr, &fwdOut, fwdParams, prog, debugPrefix,
                  poolOptions);
  // poolingImpl shapes output to be as required at the API interface. Reshape
  // back to internal shape
  return actsToInternalShape(output);
}

static Tensor poolingBwd(Graph &graph,
                         const Tensor &in_,
                         const ConvParams &bwdParams,
                         PoolingType poolingType,
                         Sequence &prog,
                         const std::string &debugPrefix,
                         const PoolOptions &poolOptions) {
  return poolingImpl(graph, {poolingType, PoolPass::POOL_BWD, false}, in_,
                     nullptr, nullptr, bwdParams, prog, debugPrefix,
                     poolOptions);
}

static Tensor poolingBwd(Graph &graph,
                         const Tensor &in_,
                         const Tensor &fwdInputActs,
                         const Tensor &fwdOutputActs,
                         const ConvParams &bwdParams,
                         PoolingType poolingType,
                         Sequence &prog,
                         const std::string &debugPrefix,
                         const PoolOptions &poolOptions) {
  return poolingImpl(graph, {poolingType, PoolPass::POOL_BWD, false}, in_,
                     &fwdInputActs, &fwdOutputActs, bwdParams, prog,
                     debugPrefix, poolOptions);
}


static bool detectMatchingFieldAndKernel(const ConvParams &params) {
  auto allZeros = [](const std::vector<unsigned> &vec) {
    return std::all_of(vec.begin(), vec.end(), [](unsigned e) {
                      return e == 0;
                     });
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

Tensor pool(Graph &graph,
            const PoolParams &poolParams,
            const Tensor &in_, Sequence &prog,
            const std::string &debugPrefix,
            const poplar::OptionFlags &options) {
  const auto poolOptions = parsePoolOptions(options);
  checkWindowParameters(poolParams);

  const auto inputFieldShape = getInputFieldShape(in_);

  // TODO: remove this once reference nD is supported
  if (in_.rank() != 4) {
    throw poputil::poplibs_error("Only 2D pooling supported");
  }

  const auto poolingType = poolParams.poolingType;
  assert(in_.dim(0) == poolParams.batchSize);
  assert(in_.dim(1) == poolParams.numChannels);

  // convert activations to internal shape
  auto in = actsToInternalShape(in_);
  const auto dType = in_.elementType();
  ConvParams convParams = makeConvParams(poolParams);

  const auto layerName = debugPrefix + "/" + asString(poolingType) + "Pool"
                         + kernelShapeAsString(poolParams.kernelShape) + "/Fwd";
  // Special handling when pooling can be represented as a reduction operation.
  // This is done because average pooling is slower because codelets handle
  // only a multiple of 4 channels and kernel is not split.
  if (substPoolingWithReduction(convParams)) {
    std::vector<std::size_t> reduceDims(in_.rank() - 2);
    std::iota(reduceDims.begin(), reduceDims.end(), 2);
    const auto kernelElems =
        std::accumulate(poolParams.kernelShape.begin(),
                        poolParams.kernelShape.end(),
                        1U,
                        std::multiplies<unsigned>());
    auto t =  popops::reduce(graph, in_, dType, reduceDims,
                             {poolingType == PoolingType::MAX ?
                                popops::Operation::MAX :
                                popops::Operation::ADD,
                              poolingType == PoolingType::AVG ?
                                1.0f / kernelElems : 1.0f},
                             prog, layerName);
    for (auto i = 0U; i != reduceDims.size(); ++i) {
        t = t.expand({2 + i});
    }
    return t;
  }

  return poolingFwd(graph, in, convParams, poolingType, prog, layerName,
                    poolOptions);
}

void
poolInputGradientImpl(Graph &graph,
                  const PoolParams &poolParams,
                  const Tensor &in_, const Tensor &pooled_,
                  const Tensor &pooledGradient_, Tensor &output,
                  const bool useScaledGradForMaxPool,
                  Sequence &prog,
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

  Tensor in, pooled;
  if(maxPooling) {
    in = actsToInternalShape(in_);
    pooled = actsToInternalShape(pooled_);
  }
  const auto dType = pooledGradient_.elementType();
  ConvParams fwdParams = makeConvParams(poolParams);

  auto pooledGradient = actsToInternalShape(pooledGradient_);
  const auto layerName = debugPrefix + "/" + asString(poolingType) + "Pool"
                         + kernelShapeAsString(poolParams.kernelShape) + "/Bwd";
  const auto numFieldDims = inputFieldShape.size();
  if(maxPooling) {
    if (pooledGradient.dim(0) != batchSize || pooled.dim(0) != batchSize)
      throw poputil::poplibs_error("Forward pass batch size does not match "
                                   "gradient calculation pass");
    if (pooledGradient.dim(3) != numChannels || pooled.dim(3) != numChannels)
      throw poputil::poplibs_error("Forward pass number of channels does not "
                                   "match gradient calculation pass");
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
  if (pooledGradient_.rank() != numFieldDims + 2) {
    throw poputil::poplibs_error("Gradient calculation pass output field size "
                                 "does not match input activations size");
  }
  auto bwdParams = canonicalizeParams(getGradientParams(fwdParams));

  if (poolingType == PoolingType::SUM || poolingType == PoolingType::AVG) {
    // For certain pooling parameters the gradient operation can be cast as a
    // scaled broadcast operation
    if (substPoolingGradientWithBroadcast(fwdParams)) {
      auto scaledPooledGradient = pooledGradient_;
      if (poolingType == PoolingType::AVG) {
        float scale = 1.0f / product(poolParams.kernelShape);
        auto scaleTensor =
            graph.addConstant(dType, pooledGradient_.shape(), scale);
        graph.setTileMapping(scaleTensor, 0);
        scaledPooledGradient =
            popops::mul(graph, pooledGradient_, scaleTensor, prog, layerName);
      }
       // do an explicit copy instead of returning a view
      prog.add(Copy(scaledPooledGradient
                            .broadcast(product(fwdParams.kernelShape), 2)
                            .reshape(output.shape()), output));
      return;
    }

    if (poolingType == PoolingType::AVG) {
      pooledGradient =
          scaleGradient(graph, fwdParams, pooledGradient, prog, layerName);
    }
    output = poolingBwd(graph, pooledGradient, bwdParams, poolingType, prog,
                      layerName, poolOptions);
    return;
  } else if (poolingType == PoolingType::MAX){
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

Tensor
poolInputGradient(Graph &graph,
                  const PoolParams &poolParams,
                  const Tensor &in_, const Tensor &pooled_,
                  const Tensor &pooledGradient_,
                  bool useScaledGradient,
                  Sequence &prog,
                  const std::string &debugPrefix,
                  const poplar::OptionFlags &options) {
  // TODO: remove this once reference nD is supported
  if (in_.rank() != 4) {
    throw poputil::poplibs_error("Only 2D pooling supported");
  }

  // create the output tensor, based on the input
  auto output = graph.clone(in_);
  poolInputGradientImpl(graph,
                        poolParams,
                        in_,
                        pooled_,
                        pooledGradient_,
                        output,
                        useScaledGradient,
                        prog,
                        debugPrefix,
                        options);
  return output;
}

Tensor
poolInputGradient(Graph &graph,
                  const PoolParams &poolParams,
                  const unsigned fwdChansPerGroup,
                  const Tensor &pooledGradient_, Sequence &prog,
                  const std::string &debugPrefix,
                  const poplar::OptionFlags &options) {
  assert(poolParams.poolingType != PoolingType::MAX);

  // TODO: remove this once reference nD is supported
  if (pooledGradient_.rank() != 4) {
    throw poputil::poplibs_error("Only 2D pooling supported");
  }

  // Create the output tensor, based on the parameters provided
  auto shape = poolParams.inputFieldShape;
  Tensor output = graph.addVariable(pooledGradient_.elementType(),
                                      {poolParams.numChannels/fwdChansPerGroup,
                                      poolParams.batchSize,
                                      shape[0],
                                      shape[1],
                                      fwdChansPerGroup});
  mapTensorLinearly(graph, output);
  output = output.dimShufflePartial({0, output.rank() - 1}, {1, 2})
                   .reshapePartial(1, 3, {poolParams.numChannels});
  poolInputGradientImpl(graph,
                        poolParams,
                        {},
                        {},
                        pooledGradient_,
                        output,
                        false,
                        prog,
                        debugPrefix,
                        options);
  return output;
}
} // namespace pooling
} // namespace poplibs
