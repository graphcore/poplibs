// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "ConvTransforms.hpp"
#include "ConvUtilInternal.hpp"
#include "popops/Pad.hpp"
#include "popops/Rearrange.hpp"
#include <poplin/ConvUtil.hpp>

#include <gccs/Algorithm.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;

namespace poplin {

static std::vector<unsigned>
inversePermutation(const std::vector<unsigned> &permutation) {
  const auto rank = permutation.size();
  std::vector<unsigned> inverse(rank);
  for (unsigned i = 0; i != rank; ++i) {
    inverse[permutation[i]] = i;
  }
  return inverse;
}

static Tensor flattenDimsMultiStage(Tensor t, unsigned from, unsigned to,
                                    unsigned kernelFactor) {
  const auto rank = t.rank();
  // Permute the dimensions so the dimension we want to flatten are at the
  // front.
  std::vector<unsigned> bringToFront = {from, to};
  bringToFront.reserve(rank);
  for (unsigned dim = 0; dim != rank; ++dim) {
    if (dim == from || dim == to)
      continue;
    bringToFront.push_back(dim);
  }
  t = t.dimShuffle(bringToFront);
  // Flatten from dimension into to dimension.
  auto flattenedShape = t.shape();
  flattenedShape[1] *= kernelFactor;
  if (kernelFactor != 0) {
    assert((flattenedShape[0] % kernelFactor) == 0);
    flattenedShape[0] /= kernelFactor;
  } else {
    flattenedShape[0] = 1;
  }
  t = t.reshape(flattenedShape);
  // Undo the previous permutation.
  t = t.dimShuffle(inversePermutation(bringToFront));
  return t;
}

Tensor flattenDims(Tensor t, unsigned from, unsigned to) {
  unsigned factor = t.dim(from);
  return flattenDimsMultiStage(t, from, to, factor);
}

Tensor unflattenDims(Tensor t, unsigned from, unsigned to, unsigned fromSize) {
  const auto rank = t.rank();
  // Permute the dimensions so the dimension we want to flatten are at the
  // front.
  std::vector<unsigned> bringToFront = {from, to};
  bringToFront.reserve(rank);
  for (unsigned dim = 0; dim != rank; ++dim) {
    if (dim == from || dim == to)
      continue;
    bringToFront.push_back(dim);
  }
  t = t.dimShuffle(bringToFront);
  // Reshape the dimensions.
  auto flattenedShape = t.shape();
  assert(flattenedShape[1] % fromSize == 0);
  assert(flattenedShape[0] == 1);
  flattenedShape[1] /= fromSize;
  flattenedShape[0] = fromSize;
  t = t.reshape(flattenedShape);
  // Undo the previous permutation.
  t = t.dimShuffle(inversePermutation(bringToFront));
  return t;
}

Tensor dilate(Graph &graph, const Tensor &t, unsigned dilationFactor,
              unsigned dim, const DebugNameAndId &dnai) {
  const auto oldSize = t.dim(dim);
  const auto newSize = getDilatedSize(oldSize, dilationFactor);
  if (newSize == oldSize)
    return t;
  auto expandedT = t.expand({dim + 1});
  const auto dType = expandedT.elementType();
  auto zeroShape = expandedT.shape();
  zeroShape[dim + 1] = dilationFactor - 1;
  auto zero = graph.addConstant(dType, expandedT.getMetadata(), zeroShape, 0,
                                {dnai, "zero"});
  graph.setTileMapping(zero, 0);
  return concat(expandedT, zero, dim + 1)
      .flatten(dim, dim + 2)
      .slice(0, newSize, dim);
}

// Dilate a tensor but instead of padding with zeros duplicate the nearest
// neighbouring element.
Tensor dilateWithNearestNeighbour(const Tensor &t, unsigned dilationFactor,
                                  unsigned dim) {
  const auto oldSize = t.dim(dim);
  const auto newSize = getDilatedSize(oldSize, dilationFactor);
  if (newSize == oldSize)
    return t;
  return t.expand({dim + 1})
      .broadcast(dilationFactor, dim + 1)
      .flatten(dim, dim + 2)
      .slice(dilationFactor / 2, newSize + dilationFactor / 2, dim);
}

/** Apply an (unpacked) input transform to an input tensor. */
void expandSpatialDimDoInputTransform(
    unsigned dim, size_t &size, unsigned &truncationLower,
    unsigned &truncationUpper, unsigned &dilation, unsigned &paddingLower,
    unsigned &paddingUpper, std::vector<bool>::reference flip,
    boost::optional<Graph &> &graph, boost::optional<Tensor> &tensor,
    const DebugNameAndId &dnai) {
  if (tensor) {
    assert(graph);
    // Explicitly truncate.
    tensor = pad(*graph, *tensor, -static_cast<int>(truncationLower),
                 -static_cast<int>(truncationUpper), dim);
    // Explicitly dilate.
    tensor = dilate(*graph, *tensor, dilation, dim, {dnai});
    // Explicitly pad.
    tensor = pad(*graph, *tensor, paddingLower, paddingUpper, dim);
    // Explicitly flip.
    if (flip) {
      *tensor = tensor->reverse(dim);
    }
  }

  size -= (truncationLower + truncationUpper);
  size = getDilatedSize(size, dilation);
  size += paddingLower + paddingUpper;

  dilation = 1;
  truncationLower = 0;
  truncationUpper = 0;
  paddingLower = 0;
  paddingUpper = 0;
  flip = false;
}

/**
 * Expand a spatial dimension of activations/weights to increase
 * the number of input channels, potentially in multiple stages.
 *
 * \param params  Convolutional parameters, these will be modified
 *                to represent a convolution with the dimension
 *                expanded (by the given factor).
 * \param dim     The spatial dimension to expand.
 * \param kernelFactor  The factor by which to expand the kernel.
 *                      This need not evenly divide the kernel
 *                      dimension, but if it does not padding will
 *                      be added.
 * \param graph   Graph in which tensors etc. are contained.
 * \param acts    Optional tensor which will be manipulated to
 *                perform the expansion.
 * \param weights Optional tensor which will be manipulated to
 *                perform the expansion.
 */
static void expandSpatialDimMultiStageImpl(ConvParams &params, unsigned dim,
                                           unsigned kernelFactor,
                                           boost::optional<Graph &> &graph,
                                           boost::optional<Tensor> &acts,
                                           boost::optional<Tensor> &weights,
                                           const DebugNameAndId &dnai) {
  unsigned actsDimIndex = dim + 2;
  unsigned weightsDimIndex = dim + 1;
  auto &actsSize = params.inputFieldShape[dim];
  auto &weightsSize = params.kernelShape[dim];
  auto &actsTruncationLower = params.inputTransform.truncationLower[dim];
  auto &actsTruncationUpper = params.inputTransform.truncationUpper[dim];
  auto &actsDilation = params.inputTransform.dilation[dim];
  auto &actsPaddingLower = params.inputTransform.paddingLower[dim];
  auto &actsPaddingUpper = params.inputTransform.paddingUpper[dim];
  std::vector<bool>::reference actsFlip = params.inputTransform.flip[dim];
  std::vector<bool>::reference weightsFlip = params.kernelTransform.flip[dim];
  auto &weightsTruncationLower = params.kernelTransform.truncationLower[dim];
  auto &weightsTruncationUpper = params.kernelTransform.truncationUpper[dim];
  auto &weightsDilation = params.kernelTransform.dilation[dim];
  auto &weightsPaddingLower = params.kernelTransform.paddingLower[dim];
  auto &weightsPaddingUpper = params.kernelTransform.paddingUpper[dim];
  auto &outputTruncationLower = params.outputTransform.truncationLower[dim];
  auto &outputTruncationUpper = params.outputTransform.truncationUpper[dim];
  auto &stride = params.outputTransform.stride[dim];
  auto &outputPaddingLower = params.outputTransform.paddingLower[dim];
  auto &outputPaddingUpper = params.outputTransform.paddingUpper[dim];

  // Apply the input transform to acts.
  expandSpatialDimDoInputTransform(actsDimIndex, actsSize, actsTruncationLower,
                                   actsTruncationUpper, actsDilation,
                                   actsPaddingLower, actsPaddingUpper, actsFlip,
                                   graph, acts, dnai);

  // Note weights is handled separately due to partial expansion.

  if (weights) {
    // Explicitly truncate.
    *weights = pad(*graph, *weights, -static_cast<int>(weightsTruncationLower),
                   -static_cast<int>(weightsTruncationUpper), weightsDimIndex);
  }
  weightsSize = params.getTruncatedKernelSize(dim);
  weightsTruncationLower = 0;
  weightsTruncationUpper = 0;

  // A partial expansion entails splitting the input dimension
  // into another dimension with size equal to the given kernelFactor
  // which is then folded into the input channels for the activations,
  // or flattened from the expanded dimension to the input channels
  // for the weights.
  //
  // In order to ensure that new input channels created by this
  // expansion can be expanded without interleaving with the original
  // input channels (allowing us to group by input channels after a partial
  // expansion) the kernel is split by uniformly subsampling kernel elements by
  // the kernel factor.

  // First pad weights/input if needed to allow expansion by the kernel
  // factor.
  bool emptyWeights = weightsSize == 0;
  const auto paddedKernelSize =
      emptyWeights ? 0 : gccs::alignNext(weightsSize, kernelFactor);
  const auto kernelPadding = paddedKernelSize - weightsSize;
  const auto actsPaddedSize =
      actsSize + getDilatedSize(paddedKernelSize, weightsDilation) -
      getDilatedSize(weightsSize, weightsDilation);
  const auto actsPadding = actsPaddedSize - actsSize;
  if (acts) {
    // If the kernel will be flipped, then pad the input such that padding
    // is at the lower end of the spatial dimension to match where the
    // kernel padding ends up.
    *acts = pad(*graph, *acts, actsPadding * weightsFlip,
                actsPadding * !weightsFlip, actsDimIndex);
  }
  if (weights) {
    *weights = pad(*graph, *weights, 0, kernelPadding, weightsDimIndex);
  }
  actsSize = actsPaddedSize;
  weightsSize = paddedKernelSize;

  bool fullExpansion = kernelFactor == weightsSize;
  assert(emptyWeights || (weightsSize % kernelFactor) == 0);
  const auto weightsFactoredSize =
      emptyWeights ? 1 : weightsSize / kernelFactor;
  auto weightsFactoredDilation =
      emptyWeights ? 1 : weightsDilation * kernelFactor;
  auto weightsFactoredDilatedSize =
      getDilatedSize(weightsFactoredSize, weightsFactoredDilation);
  auto actsFactoredSize = actsSize -
                          (getDilatedSize(weightsSize, weightsDilation) +
                           weightsPaddingLower + weightsPaddingUpper) +
                          weightsFactoredDilatedSize;
  actsFactoredSize -= outputTruncationLower + outputTruncationUpper;
  if (fullExpansion) {
    actsFactoredSize = (actsFactoredSize + stride - 1) / stride;
    actsFactoredSize += outputPaddingLower + outputPaddingUpper;
  }
  assert(!fullExpansion || actsFactoredSize == params.getOutputSize(dim));
  if (acts) {
    // Expand the acts tensor.
    auto dType = acts->elementType();
    if (weightsSize == 0) {
      auto newActsShape = acts->shape();
      newActsShape[actsDimIndex] = actsFactoredSize;
      newActsShape.back() = 0;
      *acts = (*graph).addConstant(dType, acts->getMetadata(), newActsShape, 0,
                                   {dnai, "acts"});
      (*graph).setTileMapping(*acts, 0);
    } else {
      std::vector<Tensor> slices;
      slices.reserve(kernelFactor);
      auto subsampledKernelElements =
          getDilatedSize(weightsFactoredSize, kernelFactor);
      for (unsigned k = 0; k < kernelFactor; ++k) {
        auto weightOutRange = getOutputRangeForKernelRange(
            dim, {0, params.getOutputSize(dim)},
            {k, k + subsampledKernelElements}, params);
        assert(weightOutRange.first != weightOutRange.second);
        auto weightInRange =
            getInputRange(dim, {0, params.getOutputSize(dim)},
                          {k, k + subsampledKernelElements}, params);
        auto slice = acts->slice(weightInRange.first, weightInRange.second,
                                 actsDimIndex);
        if (fullExpansion) {
          slice = slice.subSample(stride, actsDimIndex);
          const auto slicePaddingLower = weightOutRange.first;
          const auto slicePaddingUpper =
              params.getOutputSize(dim) - weightOutRange.second;
          slice = pad(*graph, slice, slicePaddingLower, slicePaddingUpper,
                      actsDimIndex);
        }
        assert(slice.dim(actsDimIndex) == actsFactoredSize);
        slices.push_back(slice);
      }
      auto expanded = concat(slices, acts->rank() - 1);
      *acts = expanded;
    }
  }
  if (weights) {
    // Flatten the spatial dimension of the weights tensor into the input
    // channels.
    *weights = flattenDimsMultiStage(*weights, weightsDimIndex,
                                     weights->rank() - 1, kernelFactor);
  }
  actsSize = actsFactoredSize;
  params.inputChannelsPerConvGroup *= kernelFactor;
  weightsPaddingLower = 0;
  weightsPaddingUpper = 0;
  outputTruncationLower = 0;
  outputTruncationUpper = 0;
  // These transformations of the kernel cannot be applied until we
  // are fully expanding the kernel
  if (fullExpansion) {
    weightsFlip = false;
    weightsSize = 1;
    weightsDilation = 1;
    stride = 1;
    outputPaddingLower = 0;
    outputPaddingUpper = 0;
  } else {
    weightsSize /= kernelFactor;
    weightsDilation *= kernelFactor;
  }
}

// Modify acts and weights from graph
static void expandSpatialDimMultiStage(ConvParams &params, unsigned dim,
                                       unsigned kernelFactor, Graph &graph,
                                       boost::optional<Tensor> &acts,
                                       boost::optional<Tensor> &weights,
                                       const DebugNameAndId &dnai) {
  boost::optional<Graph &> g = graph;
  expandSpatialDimMultiStageImpl(params, dim, kernelFactor, g, acts, weights,
                                 dnai);
}

// Planning only, no modification to acts or weights
static void expandSpatialDimMultiStage(ConvParams &params, unsigned dim,
                                       unsigned kernelFactor) {
  boost::optional<Graph &> graph;
  boost::optional<Tensor> acts;
  boost::optional<Tensor> weights;
  expandSpatialDimMultiStageImpl(params, dim, kernelFactor, graph, acts,
                                 weights, "");
}

// Modify acts and weights from graph
void expandSpatialDim(ConvParams &params, unsigned dim, Graph &graph,
                      boost::optional<Tensor> &acts,
                      boost::optional<Tensor> &weights,
                      const poplar::DebugNameAndId &dnai) {
  const auto factor = params.getTruncatedKernelSize(dim);
  expandSpatialDimMultiStage(params, dim, factor, graph, acts, weights, {dnai});
}

// Planning only, no modification to acts or weights
void expandSpatialDim(ConvParams &params, unsigned dim) {
  const auto factor = params.getTruncatedKernelSize(dim);
  expandSpatialDimMultiStage(params, dim, factor);
}

void swapOperands(ConvParams &params, boost::optional<Tensor> &acts,
                  boost::optional<Tensor> &weights) {
  swapOperands(params);
  std::swap(acts, weights);
  if (acts) {
    *acts = acts->dimRoll(acts->rank() - 2, 1);
  }
  if (weights) {
    *weights = weights->dimRoll(1, weights->rank() - 2);
  }
}

// A plan for how to perform a dimension expansion for the
// best possible memory usage/cycles.
// Plan is unified for activations and weights as we may need
// to modify the convolution parameters as we go and these
// must agree for both.
struct ExpandDimsPlan {
  // A dimension and a factor by which to divide the kernel in order to
  // partially expand the dimension.
  // This allows a transpose on a partially expanded tensor. This is not
  // advantageous for the weights hence this applies to activations only.
  std::pair<unsigned, unsigned> partialExpansion = std::make_pair(0U, 1U);
  // Grouping info before/after for both activations and weights
  // used to regroup either before or after the expansion.
  // Same grouped dimension before and after means no regroup
  // will occur.
  // [0] = weights, [1] = activations
  std::array<std::pair<GroupingInfo, GroupingInfo>, 2> regroup;
  // Whether to perform any present regroup operation before or
  // after the expansion/flattening of the activations/weights.
  // [0] = weights, [1] = activations
  std::array<bool, 2> regroupPost;
};

std::ostream &operator<<(std::ostream &o, const ExpandDimsPlan &p) {
  if (p.partialExpansion.second > 1) {
    o << "Partially expand spatial dimension " << p.partialExpansion.first
      << " by factor " << p.partialExpansion.second << "\n";
  }
  for (bool isActs : {false, true}) {
    if (p.regroup[isActs].first.first != p.regroup[isActs].second.first) {
      o << "regroup " << (isActs ? "activations " : "weights ")
        << (p.regroupPost[isActs] ? "after" : "before") << " expanding: {"
        << p.regroup[isActs].first.first << ","
        << p.regroup[isActs].first.second << "} -> {"
        << p.regroup[isActs].second.first << ","
        << p.regroup[isActs].second.second << "}\n";
    }
  }
  return o;
}

std::vector<GroupingInfo>
determinePreprocessedGroupingFromPlan(const ConvParams &params,
                                      const Plan &plan, unsigned level) {
  std::vector<GroupingInfo> grouping(1);

  // Total dimensions are spatial dimensions + 3 for either
  // acts or weights. Input channels are always the last dimension.
  auto inChanDim = params.kernelShape.size() + 3 - 1;
  auto inChanGrainSize = plan.partitions[level].inChanGrainSize;

  // The final grouping will have to be some multiple of inChanGrainSize
  grouping[0] = std::make_pair(inChanDim, inChanGrainSize);
  return grouping;
}

// Get a description for how to perform a transform for efficiency in memory
// and cycles.
static ExpandDimsPlan getExpandDimsPlan(/*TODO: const*/ Graph &graph,
                                        const ConvParams &params,
                                        const Plan &convPlan, unsigned level,
                                        const Tensor &in, bool isActs) {
  ExpandDimsPlan plan;
  const auto &expandDimsSpatial = convPlan.transforms[level].expandDims;
  const auto expandDims = dimsFromSpatialDims(expandDimsSpatial, isActs);
  auto grouping = detectDimGroupings(graph, in);
  // We can simply use the fully preprocessed grouping as further transforms
  // won't interleave the input channels we get from expanding and our regroup
  // operation is mapped linearly without regard for the ordering as defined
  // by the other dimensions (which may be shuffled around by other
  // transforms).
  auto destGrouping =
      determinePreprocessedGroupingFromPlan(params, convPlan, level);

  // If there was no detected grouping we've got nothing to go on unfortunately
  // so there's no special ops to help this expansion.
  if (!grouping.empty() && !destGrouping.empty() &&
      grouping[0].first != destGrouping[0].first && !expandDims.empty() &&
      (std::find(expandDims.begin(), expandDims.end(), grouping[0].first) ==
       expandDims.end()) &&
      (std::find(expandDims.begin(), expandDims.end(), destGrouping[0].first) ==
       expandDims.end())) {
    // TODO: T10360 - Consider avoiding regrouping of float inputs.
    auto grainSize =
        popops::rearrange::getMinimumRegroupGrainSize(params.inputType);
    unsigned dimElems = in.dim(destGrouping[0].first);
    auto maxGroupSize = std::gcd(dimElems, destGrouping[0].second);
    auto nextGrouping = destGrouping[0];
    nextGrouping.second = maxGroupSize;

    // If the dimension to transpose to doesn't meet our minimum grain size
    // then we can do a special multi-stage expansion where
    // we partially expand a dimension, then transpose, then
    // perform the rest of the expansion. This is only advantageous for
    // activations where the elements are broadcasted increasing the amount
    // of data that needs transposing. For weights we can always do a
    // transpose after completely flattening the tensor.
    if (isActs && nextGrouping.first == in.rank() - 1 &&
        (maxGroupSize % grainSize) != 0) {
      unsigned expandedDimElems = dimElems;
      for (unsigned i = 0; i != expandDimsSpatial.size(); ++i) {
        const unsigned roundedElems = std::lcm(expandedDimElems, grainSize);
        const unsigned factor = roundedElems / expandedDimElems;
        const auto dim = expandDimsSpatial[i];
        const auto truncatedKernelSize = params.getTruncatedKernelSize(dim);
        // We're all padding anyway
        if (truncatedKernelSize == 0)
          break;

        poplin::ConvParams noPaddingParams;
        poplin::ConvParams paddingParams;
        { // Without padding
          ConvOptions options{};
          std::vector<Split<ConvIndices>> indices;
          auto noPaddingPlan = convPlan;
          noPaddingParams =
              convolutionPreprocess(graph, params, options, noPaddingPlan,
                                    level, indices, false)
                  .getParams();
        }
        { // With padding
          ConvOptions options{};
          std::vector<Split<ConvIndices>> indices;
          auto paddingPlan = convPlan;
          auto tmp = params;
          // Fully expand up to the dimension to partially expand.
          size_t nextToExpand = 0;
          for (; nextToExpand != expandDimsSpatial.size(); ++nextToExpand) {
            if (expandDimsSpatial[nextToExpand] == dim) {
              break;
            }
            expandSpatialDim(tmp, expandDimsSpatial[nextToExpand]);
          }
          // Partially expand
          expandSpatialDimMultiStage(tmp, dim, factor);
          paddingParams =
              convolutionPreprocess(graph, tmp, options, paddingPlan, level,
                                    indices, false)
                  .getParams();
        }
        const auto partiallyExpandingModifiedParams =
            noPaddingParams != paddingParams;

        // We can partially expand at this point either if:
        if (
            // * the kernel is evenly divisible by the desired factor (no
            //   padding required)
            (truncatedKernelSize % factor) == 0 ||
            // * the padding after fully expanding is no more than would be
            //   added as a result of rounding up to the input channel grain
            //   size and other convolution preprocessing operations.
            (truncatedKernelSize > 1 && !partiallyExpandingModifiedParams) ||
            // * this is the last dimension to be expanded (padding can be
            //   safely added as it will end up in the last input channels and
            //   can be easily stripped.
            (truncatedKernelSize > 1 && i == expandDimsSpatial.size() - 1)) {
          plan.partialExpansion.first = dim;
          plan.partialExpansion.second = factor;
          maxGroupSize = std::gcd(roundedElems, destGrouping[0].second);
          nextGrouping.second = maxGroupSize;
          plan.regroupPost[isActs] = false;
          break;
        }
        expandedDimElems *= truncatedKernelSize;
      }
    }

    if ((maxGroupSize % grainSize) == 0 &&
        (grouping[0].second % grainSize) == 0) {
      plan.regroup[isActs].first = grouping[0];
      plan.regroup[isActs].second = nextGrouping;
      plan.regroupPost[isActs] = false;
    }
  }

  return plan;
}

static ExpandDimsPlan mergeExpandDimsPlans(ExpandDimsPlan planActs,
                                           const ExpandDimsPlan &planWeights) {
  planActs.regroup[false] = planWeights.regroup[false];
  planActs.regroupPost[false] = planWeights.regroupPost[false];
  return planActs;
}

void expandSpatialDims(ConvParams &params, Plan &plan, unsigned level,
                       Graph &graph, boost::optional<Tensor> &acts,
                       boost::optional<Tensor> &weights,
                       const ExpandDimsPlan &expandDimsPlan,
                       ConvProgramTree::TransformPreProgram *rearrangeProg,
                       bool rearrangeActs, bool rearrangeWeights,
                       const poplar::DebugNameAndId &dnai) {
  const auto &expandDimsSpatial = plan.transforms[level].expandDims;

  const auto &partialExpansion = expandDimsPlan.partialExpansion;
  bool hasPartialExpansion = partialExpansion.second > 1;
  unsigned inputChanPaddingLower = 0, inputChanPaddingUpper = 0;
  bool stripPadding = false;
  std::size_t nextToExpand = 0;
  if (!expandDimsSpatial.empty() && hasPartialExpansion) {
    // Fully expand up to the dimension to partially expand.
    for (; nextToExpand != expandDimsSpatial.size(); ++nextToExpand) {
      if (expandDimsSpatial[nextToExpand] == partialExpansion.first) {
        break;
      }
      expandSpatialDim(params, expandDimsSpatial[nextToExpand], graph, acts,
                       weights, {dnai});
    }
    unsigned kernelSizeBefore =
        params.getTruncatedKernelSize(partialExpansion.first);
    unsigned inputChansBefore = params.inputChannelsPerConvGroup;

    // Partially expand
    expandSpatialDimMultiStage(params, partialExpansion.first,
                               partialExpansion.second, graph, acts, weights,
                               {dnai});

    // Check for any padding added for the partial expansion
    unsigned kernelSizeAfter =
        params.getTruncatedKernelSize(partialExpansion.first);
    unsigned padding =
        (kernelSizeAfter * partialExpansion.second) - kernelSizeBefore;
    inputChanPaddingLower = 0;
    inputChanPaddingUpper = padding * inputChansBefore;
    // We can only strip the padding if it does not become interleaved
    // in the input channels as a result of further expanded dimensions.
    stripPadding = (inputChanPaddingUpper + inputChanPaddingLower > 0 &&
                    partialExpansion.first == expandDimsSpatial.back());
  }

  // Pre-expansion regroup.
  for (bool isActs : {false, true}) {
    const auto &regroup = expandDimsPlan.regroup[isActs];
    bool regroupPreExpansion = !expandDimsPlan.regroupPost[isActs];
    if (regroup.first.first != regroup.second.first && regroupPreExpansion) {
      if ((acts && isActs && rearrangeActs) ||
          (weights && !isActs && rearrangeWeights)) {
        auto &t = isActs ? *acts : *weights;
        assert(rearrangeProg);
        auto &preTranspose = isActs ? rearrangeProg->preTransposeActs
                                    : rearrangeProg->preTransposeWeights;
        auto &transposeCS = isActs ? rearrangeProg->transposeCSActs.back()
                                   : rearrangeProg->transposeCSWeights.back();
        t = popops::rearrange::regroupTensor(graph, t, preTranspose,
                                             transposeCS, regroup.first,
                                             regroup.second, {dnai});
      }
    }
  }

  // Fully expand remaining dimensions now
  for (; nextToExpand != expandDimsSpatial.size(); ++nextToExpand) {
    const auto factor =
        params.getTruncatedKernelSize(expandDimsSpatial[nextToExpand]);
    expandSpatialDimMultiStage(params, expandDimsSpatial[nextToExpand], factor,
                               graph, acts, weights, {dnai});
    // Trim any padding added by a potential earlier partial expansion
    if ((inputChanPaddingLower || inputChanPaddingUpper) && stripPadding) {
      if (acts) {
        *acts = pad(graph, *acts, -static_cast<int>(inputChanPaddingLower),
                    -static_cast<int>(inputChanPaddingUpper), acts->rank() - 1);
      }
      if (weights) {
        *weights =
            pad(graph, *weights, -static_cast<int>(inputChanPaddingLower),
                -static_cast<int>(inputChanPaddingUpper), weights->rank() - 1);
      }
      params.inputChannelsPerConvGroup -=
          inputChanPaddingUpper + inputChanPaddingLower;
      inputChanPaddingLower = inputChanPaddingUpper = 0;
    }
  }

  // Post-expansion regroup.
  for (bool isActs : {false, true}) {
    const auto &regroup = expandDimsPlan.regroup[isActs];
    bool regroupPostExpansion = expandDimsPlan.regroupPost[isActs];
    if (regroup.first.first != regroup.second.first && regroupPostExpansion) {
      if ((acts && isActs && rearrangeActs) ||
          (weights && !isActs && rearrangeWeights)) {
        auto &t = isActs ? *acts : *weights;
        assert(rearrangeProg);
        auto &preTranspose = isActs ? rearrangeProg->preTransposeActs
                                    : rearrangeProg->preTransposeWeights;
        auto &transposeCS = isActs ? rearrangeProg->transposeCSActs.back()
                                   : rearrangeProg->transposeCSWeights.back();
        t = popops::rearrange::regroupTensor(graph, t, preTranspose,
                                             transposeCS, regroup.first,
                                             regroup.second, {dnai});
      }
    }
  }
}

void expandSpatialDims(ConvParams &params, Plan &plan, unsigned level,
                       Graph &graph, boost::optional<Tensor> &acts,
                       boost::optional<Tensor> &weights,
                       ConvProgramTree::TransformPreProgram *rearrangeProg,
                       bool rearrangeActs, bool rearrangeWeights,
                       const DebugNameAndId &dnai) {
  const auto &expandDimsSpatial = plan.transforms[level].expandDims;
  if (expandDimsSpatial.empty()) {
    return;
  }

  ExpandDimsPlan actsTransformPlan, weightsTransformPlan;
  Tensor actsExpanded, weightsExpanded;
  if (acts && rearrangeActs) {
    actsTransformPlan =
        getExpandDimsPlan(graph, params, plan, level, *acts, true);
  }

  if (weights && rearrangeWeights) {
    weightsTransformPlan =
        getExpandDimsPlan(graph, params, plan, level, *weights, false);
  }

  ExpandDimsPlan mergedTransformPlan =
      mergeExpandDimsPlans(actsTransformPlan, weightsTransformPlan);

  // Now do a series of transforms as appropriate.
  expandSpatialDims(params, plan, level, graph, acts, weights,
                    mergedTransformPlan, rearrangeProg, rearrangeActs,
                    rearrangeWeights, dnai);
}

// Flatten dimensions according to dimsToFlatten. If available, the tensor acts
// is reshaped, the spatial dimensions spatialDims and batchSize are updated
// to represented the flattened shape, with batchSize beeing the outermost
// dimension.
void doFlatten(const std::vector<unsigned> &dimsToFlatten,
               boost::optional<Tensor> &acts,
               std::vector<std::size_t> &spatialDims, std::size_t &batchSize) {
  for (auto it = std::next(dimsToFlatten.rbegin()), end = dimsToFlatten.rend();
       it != end; ++it) {
    const auto fromDimIndex = *it;
    const auto toDimIndex = dimsToFlatten.back();
    assert(fromDimIndex != toDimIndex);
    if (acts) {
      *acts = flattenDims(*acts, fromDimIndex + 1, toDimIndex + 1);
    }
    auto &fromDimSize =
        fromDimIndex ? spatialDims[fromDimIndex - 1] : batchSize;
    auto &toDimSize = toDimIndex ? spatialDims[toDimIndex - 1] : batchSize;
    toDimSize *= fromDimSize;
    fromDimSize = 1;
  }
}

bool expandDimTransformIsViewOnly(const ConvParams &params, unsigned dim) {
  if (params.inputTransform.truncationLower[dim] == 0 &&
      params.inputTransform.truncationUpper[dim] == 0 &&
      params.inputTransform.paddingLower[dim] == 0 &&
      params.inputTransform.paddingUpper[dim] == 0 &&
      params.inputTransform.dilation[dim] == 1 &&
      params.kernelTransform.truncationLower[dim] == 0 &&
      params.kernelTransform.truncationUpper[dim] == 0 &&
      params.kernelTransform.paddingLower[dim] == 0 &&
      params.kernelTransform.paddingUpper[dim] == 0 &&
      params.kernelTransform.dilation[dim] == 1 &&
      params.outputTransform.truncationLower[dim] == 0 &&
      params.outputTransform.truncationUpper[dim] == 0 &&
      params.outputTransform.paddingLower[dim] == 0 &&
      params.outputTransform.paddingUpper[dim] == 0 &&
      params.outputTransform.stride[dim] == 1 &&
      params.kernelShape[dim] == params.inputFieldShape[dim]) {
    return true;
  }
  return false;
}

} // namespace poplin
