// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "poplin/Convolution.hpp"

#include "CanonicalConvParams.hpp"
#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include "ConvProgramTree.hpp"
#include "ConvReduce.hpp"
#include "ConvUtilInternal.hpp"
#include "ConvolutionInternal.hpp"
#include "CreateConvPartialVertex.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/Algorithms.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/TileHierarchy.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/ConvParams.hpp"
#include "poplin/ConvUtil.hpp"
#include "popops/Cast.hpp"
#include "popops/DynamicSlice.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Pad.hpp"
#include "popops/Rearrange.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "popops/Zero.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <boost/optional.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <cassert>
#include <cmath>
#include <limits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;

namespace poplin {

namespace {
struct ConvIndices {
  unsigned cg;
  unsigned b;
  std::vector<unsigned> out;
  unsigned oc;
  unsigned ic;
  std::vector<unsigned> kernel;
};

struct ConvSlice {
  unsigned cgBegin, cgEnd;
  unsigned batchBegin, batchEnd;
  std::vector<unsigned> outFieldBegin, outFieldEnd;
  unsigned outChanBegin, outChanEnd;
  unsigned inChanBegin, inChanEnd;
  std::vector<unsigned> kernelBegin, kernelEnd;

  unsigned getNumFieldDims() const { return outFieldBegin.size(); }
  unsigned getNumConvGroups() const { return cgEnd - cgBegin; }
  unsigned getBatchSize() const { return batchEnd - batchBegin; }
  unsigned getNumOutputChans() const { return outChanEnd - outChanBegin; }
  unsigned getNumInputChans() const { return inChanEnd - inChanBegin; }
  unsigned getOutputSize(unsigned dim) const {
    return outFieldEnd[dim] - outFieldBegin[dim];
  }
  unsigned getKernelSize(unsigned dim) const {
    return kernelEnd[dim] - kernelBegin[dim];
  }
};

} // End anonymous namespace

static Tensor createInputImpl(Graph &graph, const CanonicalConvParams &params,
                              unsigned level, bool serial,
                              const std::vector<Split<ConvIndices>> &indices,
                              const std::string &name, const Plan &plan,
                              const ConvOptions &options);
static Tensor createWeightsImpl(Graph &graph, const CanonicalConvParams &params,
                                unsigned level, bool serial,
                                const std::vector<Split<ConvIndices>> &indices,
                                const std::string &name, const Plan &plan,
                                const ConvOptions &options);

static unsigned getNumElementsInSlice(const std::vector<unsigned> &sliceBegin,
                                      const std::vector<unsigned> &sliceEnd) {
  const auto rank = sliceBegin.size();
  assert(sliceEnd.size() == rank);
  unsigned numElements = 1;
  for (unsigned dim = 0; dim != rank; ++dim) {
    numElements *= sliceEnd[dim] - sliceBegin[dim];
  }
  return numElements;
}

static unsigned getTruncatedSize(std::size_t size, unsigned truncationLower,
                                 unsigned truncationUpper) {
  assert(size >= truncationLower + truncationUpper);
  return size - (truncationLower + truncationUpper);
}

static unsigned getTransformedSize(const std::vector<std::size_t> &size,
                                   const ConvParams::InputTransform &transform,
                                   unsigned dim) {
  const auto truncatedSize =
      getTruncatedSize(size[dim], transform.truncationLower[dim],
                       transform.truncationUpper[dim]);
  const auto truncatedDilatedSize =
      getDilatedSize(truncatedSize, transform.dilation[dim]);
  int truncatedDilatedPaddedSize = transform.paddingLower[dim] +
                                   truncatedDilatedSize +
                                   transform.paddingUpper[dim];
  return truncatedDilatedPaddedSize;
}

// Reorder the underlying memory regions of a tensor to make sliced regions
// contiguous on each tile.
static Tensor
remapTensorToContiguousTileRegions(Graph &graph, Tensor t,
                                   const std::vector<std::size_t> &shape,
                                   const unsigned grouping) {
  // TODO: T12871 Implement poplibs_expensive_assert like in poplar.
  // This could be used to check, for example:
  // poplibs_expensive_assert(t.getContiguousRegions().size() == 1);
  auto mapping = graph.getTileMapping(t);
  auto inverseMapping = poplibs::getInverseMapping(mapping);
  t = t.flatten();
  std::vector<Tensor> toConcat;
  toConcat.reserve(inverseMapping.size());
  for (const auto &i : inverseMapping) {
    assert(i.begin() % grouping == 0 && i.end() % grouping == 0);
    toConcat.push_back(t.slice(i.begin(), i.end()));
  }
  t = concat(toConcat).reshape(shape);
  graph.setTileMapping(t, mapping);
  return t;
}

unsigned ConvParams::getTruncatedInputSize(unsigned dim) const {
  return getTruncatedSize(inputFieldShape[dim],
                          inputTransform.truncationLower[dim],
                          inputTransform.truncationUpper[dim]);
}

unsigned ConvParams::getTruncatedKernelSize(unsigned dim) const {
  return getTruncatedSize(kernelShape[dim],
                          kernelTransform.truncationLower[dim],
                          kernelTransform.truncationUpper[dim]);
}

unsigned ConvParams::getTransformedInputSize(unsigned dim) const {
  return getTransformedSize(inputFieldShape, inputTransform, dim);
}
unsigned ConvParams::getTransformedKernelSize(unsigned dim) const {
  return getTransformedSize(kernelShape, kernelTransform, dim);
}

std::size_t ConvParams::getUntransformedOutputSize(unsigned dim) const {
  auto transformedInputSize = getTransformedInputSize(dim);
  auto transformedKernelSize = getTransformedKernelSize(dim);
  assert(transformedInputSize >= transformedKernelSize);
  return transformedInputSize + 1 - transformedKernelSize;
}

std::size_t ConvParams::getOutputSize(unsigned dim) const {
  auto convOutSize = getUntransformedOutputSize(dim);
  assert(convOutSize >= outputTransform.truncationLower[dim] +
                            outputTransform.truncationUpper[dim]);
  auto truncatedSize = convOutSize - (outputTransform.truncationLower[dim] +
                                      outputTransform.truncationUpper[dim]);
  auto stride = outputTransform.stride[dim];
  auto truncatedStridedSize = (truncatedSize + stride - 1) / stride;
  auto truncatedStridedPaddedSize = outputTransform.paddingLower[dim] +
                                    truncatedStridedSize +
                                    outputTransform.paddingUpper[dim];
  return truncatedStridedPaddedSize;
}

std::vector<std::size_t> ConvParams::getOutputFieldShape() const {
  std::vector<std::size_t> outputFieldShape;
  for (auto dim = 0U; dim != inputFieldShape.size(); ++dim) {
    outputFieldShape.push_back(getOutputSize(dim));
  }
  return outputFieldShape;
}

static std::string getCapitalizedFieldDimName(unsigned dim,
                                              unsigned numFieldDims) {
  assert(dim < numFieldDims);
  if (numFieldDims > 3) {
    return "Field dimension " + std::to_string(dim);
  }
  // Dimensions are named from the innermost dimension outwards.
  switch (numFieldDims - dim) {
  case 1:
    return "Width";
  case 2:
    return "Height";
  case 3:
    return "Depth";
  }
  POPLIB_UNREACHABLE();
}

static void verifyInputShapes(const CanonicalConvParams &params,
                              const Tensor &in, const Tensor &weights) {
  const auto numFieldDims = params->getNumFieldDims();
  if (in.rank() != 3 + numFieldDims) {
    throw poputil::poplibs_error(
        "Input tensor does not have the expected rank");
  }
  if (weights.rank() != 3 + numFieldDims) {
    throw poputil::poplibs_error("Weight tensor does not have the expected "
                                 "rank");
  }
  for (unsigned i = 0; i != numFieldDims; ++i) {
    if (params->inputFieldShape[i] != in.dim(2 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw poputil::poplibs_error(dimName + " of input tensor does not match "
                                             "convolution parameters");
    }
    if (params->kernelShape[i] != weights.dim(1 + i)) {
      const auto dimName = getCapitalizedFieldDimName(i, numFieldDims);
      throw poputil::poplibs_error(dimName + " of kernel does not match "
                                             "convolution parameters");
    }
  }
  if (params->numConvGroups != in.dim(0)) {
    throw poputil::poplibs_error("Number of convolution groups of input tensor "
                                 "does not match convolution parameters");
  }
  if (params->getBatchSize() != in.dim(1)) {
    throw poputil::poplibs_error("Batchsize of input tensor does not match "
                                 "convolution parameters");
  }
  if (in.dim(1) == 0) {
    throw poputil::poplibs_error("Batch size of input tensor equal to zero "
                                 "is not supported");
  }
  if (params->getNumInputChansPerConvGroup() != in.dim(in.rank() - 1)) {
    throw poputil::poplibs_error("Number of channels per convolution group of "
                                 "input tensor does not match convolution "
                                 "parameters");
  }
  if (params->numConvGroups != weights.dim(0)) {
    throw poputil::poplibs_error(
        "Number of convolution groups of weights "
        "tensor does not match convolution parameters");
  }
  if (params->getNumOutputChansPerConvGroup() !=
      weights.dim(weights.rank() - 2)) {
    throw poputil::poplibs_error("Kernel output channel size does not match "
                                 "convolution parameters");
  }
  if (params->getNumInputChansPerConvGroup() !=
      weights.dim(weights.rank() - 1)) {
    throw poputil::poplibs_error("Kernel input channel size does not match "
                                 "convolution parameters");
  }
}

static unsigned getConvGroupsPerGroup(const Plan &plan,
                                      unsigned numConvGroups) {
  return gcd(plan.convGroupsPerGroup, numConvGroups);
}

static unsigned getInChansPerGroup(const Plan &plan, unsigned numInChans) {
  return gcd(plan.inChansPerGroup, numInChans);
}

static unsigned getOutChansPerGroup(const Plan &plan, unsigned numOutChans) {
  return gcd(plan.partialChansPerGroup, numOutChans);
}

static unsigned linearizeConvIndices(const std::vector<unsigned> &outIndices,
                                     const std::vector<unsigned> &kernelIndices,
                                     unsigned ic, unsigned b, unsigned oc,
                                     unsigned cg,
                                     const std::vector<unsigned> &fieldSplit,
                                     const std::vector<unsigned> &kernelSplit,
                                     unsigned inChanSplit, unsigned batchSplit,
                                     unsigned outChanSplit) {
  const auto numFieldDims = outIndices.size();
  // Use ozg as the innermost dimension to increase the chance that
  // tiles in a supertile both read the same activations. This reduces
  // exchange time when supertile send / receive is used.
  auto tile = cg;
  tile = tile * inChanSplit + ic;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    tile = tile * kernelSplit[dim] + kernelIndices[dim];
  }
  tile = tile * batchSplit + b;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    tile = tile * fieldSplit[dim] + outIndices[dim];
  }
  tile = tile * outChanSplit + oc;
  return tile;
}

static unsigned
linearizeTileIndices(const Target &target, const ConvOptions &opts,
                     const std::vector<Split<ConvIndices>> &indices,
                     const Plan &plan) {
  const auto hierarchy =
      poplibs::getTileHierarchy(opts.numIPUs, opts.tilesPerIPU);
  const auto numLevels = hierarchy.size();
  assert(indices.size() == numLevels);
  assert(plan.partitions.size() == numLevels);
  unsigned tile = 0;
  for (unsigned i = 0; i != numLevels; ++i) {
    const auto &levelIndices = indices[i].parallel;
    const auto &levelPartition = plan.partitions[i];
    auto fwdOutIndices = levelIndices.out;
    const auto &fwdKernelIndices = levelIndices.kernel;
    auto fwdic = levelIndices.ic;
    const auto fwdb = levelIndices.b;
    auto fwdoc = levelIndices.oc;
    const auto fwdcg = levelIndices.cg;
    auto fwdFieldSplit = levelPartition.fieldSplit;
    const auto &fwdKernelSplit = levelPartition.kernelSplit;
    auto fwdInChanSplit = levelPartition.inChanSplit.parallel;
    const auto &fwdBatchSplit = levelPartition.batchSplit;
    auto fwdOutChanSplit = levelPartition.outChanSplit.parallel;
    switch (plan.linearizeTileOrder) {
    case Plan::LinearizeTileOrder::FC_WU:
      // For the fully connected weight update the in group and out group are
      // swapped compared to the forward pass.
      std::swap(fwdInChanSplit, fwdOutChanSplit);
      std::swap(fwdic, fwdoc);
      break;
    case Plan::LinearizeTileOrder::FC_BWD_AS_CONV:
      // For the fully connected backward pass the width and the input channels
      // are swapped compared to the forward pass.
      {
        std::swap(fwdFieldSplit.back(), fwdInChanSplit);
        std::swap(fwdOutIndices.back(), fwdic);
      }
      break;
    case Plan::LinearizeTileOrder::STANDARD:
      break;
    }
    const auto linearizedIndex =
        linearizeConvIndices(fwdOutIndices, fwdKernelIndices, fwdic, fwdb,
                             fwdoc, fwdcg, fwdFieldSplit, fwdKernelSplit,
                             fwdInChanSplit, fwdBatchSplit, fwdOutChanSplit);
    tile = tile * hierarchy[i] + linearizedIndex;
  }

  // split into a per-IPU tile here so that any wrap around from the dithering
  // stays on the same IPU.
  assert(tile < target.getNumTiles());
  const auto tilesPerIpu = target.getTilesPerIPU();
  const auto ipu = tile / tilesPerIpu;

  // make sure that we utilise 64-bit exchange if it is available.
  assert(plan.startTile % target.getTilesPerSharedExchangeBus() == 0);

  // dither
  assert(plan.startTile < tilesPerIpu);
  tile = (tile + plan.startTile) % tilesPerIpu;

  // direction
  switch (plan.linearizeTileDirection) {
  case Plan::LinearizeTileDirection::ASCENDING:
    break;
  case Plan::LinearizeTileDirection::DESCENDING:
    tile = tilesPerIpu - tile - 1;
  }

  assert(tile < tilesPerIpu);
  return ipu * tilesPerIpu + tile % tilesPerIpu;
}

static std::pair<unsigned, unsigned>
getTileOutRange(const CanonicalConvParams &params, const Partition &partition,
                unsigned tileIndex, unsigned dim) {
  const auto outSize = params->getOutputSize(dim);
  const auto grainSize = partition.fieldAxisGrainSize[dim];
  const auto numGrains = ceildiv(outSize, grainSize);
  const auto split = partition.fieldSplit[dim];

  const auto outGrainBegin = (tileIndex * numGrains) / split;
  const auto outGrainEnd = ((tileIndex + 1) * numGrains) / split;

  const auto outBegin = outGrainBegin * grainSize;
  const auto outEnd = std::min(outGrainEnd * grainSize, outSize);

  return {outBegin, outEnd};
}

/// Compute the sub-convolution corresponding to the specified slice of a larger
/// convolution. The parameters and tensors are updated in place to
/// the parameters and tensors for the sub-convolution.
static CanonicalConvParams
getSubConvolution(const ConvSlice &slice,
                  const CanonicalConvParams &originalParams, Tensor *in,
                  Tensor *weights) {
  auto params = originalParams.getParams();
  const auto numFieldDims = params.getNumFieldDims();
  // Explicitly truncate the convGroup, channel and batch axes.
  params.numConvGroups = slice.getNumConvGroups();
  params.batchSize = slice.getBatchSize();
  params.inputChannelsPerConvGroup = slice.getNumInputChans();
  params.outputChannelsPerConvGroup = slice.getNumOutputChans();
  if (in) {
    *in = in->slice({slice.cgBegin, slice.batchBegin},
                    {slice.cgEnd, slice.batchEnd})
              .slice(slice.inChanBegin, slice.inChanEnd, in->rank() - 1);
  }
  if (weights) {
    *weights =
        weights->slice(slice.cgBegin, slice.cgEnd)
            .slice(slice.outChanBegin, slice.outChanEnd, weights->rank() - 2)
            .slice(slice.inChanBegin, slice.inChanEnd, weights->rank() - 1);
  }

  // Explicitly truncate the spatial dimensions.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto extraTruncationLower = slice.outFieldBegin[dim];
    auto extraTruncationUpper =
        static_cast<unsigned>(params.getOutputSize(dim)) -
        slice.outFieldEnd[dim];

    // Ensure the truncation at either end is less than or equal to the padding
    // at that end plus the size of the downsampled convolution output. If the
    // truncation exceeds this amount the final output is zero and it is
    // therefore equivalent to any other convolution with inputs of the same
    // size that results in a zero output of the same size. We choose to
    // transform it into a convolution where the truncation at one end equals
    // the padding at that end plus the size of the downsampled convolution
    // output (ensuring the output remains zero) and the truncation at the other
    // end is adjusted to keep the same output size.
    if (extraTruncationLower >
        params.getOutputSize(dim) - params.outputTransform.paddingUpper[dim]) {
      auto excess =
          extraTruncationLower - (params.getOutputSize(dim) -
                                  params.outputTransform.paddingUpper[dim]);
      extraTruncationUpper += excess;
      extraTruncationLower -= excess;
    }
    if (extraTruncationUpper >
        params.getOutputSize(dim) - params.outputTransform.paddingLower[dim]) {
      auto excess =
          extraTruncationUpper - (params.getOutputSize(dim) -
                                  params.outputTransform.paddingLower[dim]);
      extraTruncationLower += excess;
      extraTruncationUpper -= excess;
    }
    auto &outputPaddingLower = params.outputTransform.paddingLower[dim];
    auto &outputPaddingUpper = params.outputTransform.paddingUpper[dim];
    const auto &stride = params.outputTransform.stride[dim];
    auto &outputTruncationLower = params.outputTransform.truncationLower[dim];
    auto &outputTruncationUpper = params.outputTransform.truncationUpper[dim];
    const auto excessPaddingLower =
        std::min(outputPaddingLower, extraTruncationLower);
    outputPaddingLower -= excessPaddingLower;
    extraTruncationLower -= excessPaddingLower;
    if (extraTruncationLower ==
        params.getOutputSize(dim) - outputPaddingUpper) {
      outputTruncationLower += 1 + (extraTruncationLower - 1) * stride;
    } else {
      outputTruncationLower += extraTruncationLower * stride;
    }
    extraTruncationLower = 0;
    const auto excessPaddingUpper =
        std::min(outputPaddingUpper, extraTruncationUpper);
    outputPaddingUpper -= excessPaddingUpper;
    extraTruncationUpper -= excessPaddingUpper;
    if (extraTruncationUpper ==
        params.getOutputSize(dim) - outputPaddingLower) {
      outputTruncationUpper += 1 + (extraTruncationUpper - 1) * stride;
    } else {
      outputTruncationUpper += extraTruncationUpper * stride;
    }
    extraTruncationUpper = 0;
  }
  // Replace unused kernel elements with zero padding.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto sliceBegin = std::max(slice.kernelBegin[dim],
                               params.kernelTransform.truncationLower[dim]);
    auto sliceEnd = std::min(slice.kernelEnd[dim],
                             static_cast<unsigned>(params.kernelShape[dim]) -
                                 params.kernelTransform.truncationUpper[dim]);
    const auto transformedKernelSize = params.getTransformedKernelSize(dim);
    if (sliceBegin >= sliceEnd) {
      sliceBegin = 0;
      sliceEnd = 0;
      params.kernelTransform.truncationLower[dim] = 0;
      params.kernelTransform.truncationUpper[dim] = params.kernelShape[dim];
      params.kernelTransform.paddingLower[dim] = 0;
      params.kernelTransform.paddingUpper[dim] = transformedKernelSize;
      continue;
    }
    params.kernelTransform.truncationLower[dim] = sliceBegin;
    params.kernelTransform.paddingLower[dim] +=
        transformedKernelSize - params.getTransformedKernelSize(dim);
    params.kernelTransform.truncationUpper[dim] =
        params.kernelShape[dim] - sliceEnd;
    params.kernelTransform.paddingUpper[dim] +=
        transformedKernelSize - params.getTransformedKernelSize(dim);
  }

  // Canonicalize parameters. This may move truncation from the output to
  // the input or the kernel.
  params = params.canonicalize();

  // Explicitly truncate the input.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto &inputTruncationLower = params.inputTransform.truncationLower[dim];
    auto &inputTruncationUpper = params.inputTransform.truncationUpper[dim];
    if (in) {
      *in = in->slice(inputTruncationLower,
                      params.inputFieldShape[dim] - inputTruncationUpper,
                      2 + dim);
    }
    params.inputFieldShape[dim] -= inputTruncationLower + inputTruncationUpper;
    inputTruncationLower = 0;
    inputTruncationUpper = 0;
  }

  // Explicitly truncate the kernel.
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    auto &kernelTruncationLower = params.kernelTransform.truncationLower[dim];
    auto &kernelTruncationUpper = params.kernelTransform.truncationUpper[dim];
    if (weights) {
      *weights = weights->slice(kernelTruncationLower,
                                params.kernelShape[dim] - kernelTruncationUpper,
                                1 + dim);
    }
    params.kernelShape[dim] -= kernelTruncationLower + kernelTruncationUpper;
    kernelTruncationLower = 0;
    kernelTruncationUpper = 0;
  }
  assert(params == params.canonicalize());
  return params;
}

static void iteratePartitionParallel(
    const CanonicalConvParams &params, const Partition &partition,
    const std::function<void(const ConvIndices &, const ConvSlice &)> &f) {
  const auto numFieldDims = params->getNumFieldDims();

  const unsigned numOutChans = params->getNumOutputChansPerConvGroup();
  const auto outChanGrainSize = partition.outChanGrainSize;
  const auto outChanNumGrains = ceildiv(numOutChans, outChanGrainSize);
  const auto outChanSplit = partition.outChanSplit;

  const auto batchSplit = partition.batchSplit;
  const unsigned batchSize = params->getBatchSize();

  const unsigned numInChans = params->getNumInputChansPerConvGroup();
  const auto inChanGrainSize = partition.inChanGrainSize;
  const auto inChanNumGrains = ceildiv(numInChans, inChanGrainSize);
  const auto inChanSplit = partition.inChanSplit;

  const unsigned numConvGroups = params->getNumConvGroups();
  const auto convGroupGrainSize = partition.convGroupGrainSize;
  const auto convGroupNumGrains = ceildiv(numConvGroups, convGroupGrainSize);
  const auto convGroupSplit = partition.convGroupSplit;

  const auto totalFieldSplit = product(partition.fieldSplit);
  const auto totalKernelSplit = product(partition.kernelSplit);

  for (unsigned cg = 0; cg != convGroupSplit; ++cg) {
    const auto convGroupGrainBegin = (cg * convGroupNumGrains) / convGroupSplit;
    const auto convGroupGrainEnd =
        ((cg + 1) * convGroupNumGrains) / convGroupSplit;
    const auto cgBegin = convGroupGrainBegin * convGroupGrainSize;
    const auto cgEnd =
        std::min(convGroupGrainEnd * convGroupGrainSize, numConvGroups);

    for (unsigned b = 0; b != batchSplit; ++b) {
      const auto batchBegin = (b * batchSize) / batchSplit;
      const auto batchEnd = ((b + 1) * batchSize) / batchSplit;
      for (unsigned ic = 0; ic != inChanSplit.parallel; ++ic) {
        const auto inChanGrainBegin =
            (ic * inChanNumGrains) / inChanSplit.parallel;
        const auto inChanGrainEnd =
            ((ic + 1) * inChanNumGrains) / inChanSplit.parallel;
        const auto inChanBegin = inChanGrainBegin * inChanGrainSize;
        const auto inChanEnd =
            std::min(inChanGrainEnd * inChanGrainSize, numInChans);

        for (unsigned k = 0; k != totalKernelSplit; ++k) {
          auto kernelIndices = unflattenIndex(partition.kernelSplit, k);
          std::vector<unsigned> kernelBegin(numFieldDims),
              kernelEnd(numFieldDims);
          for (unsigned dim = 0; dim != numFieldDims; ++dim) {
            const auto kernelSize = params->kernelShape[dim];
            kernelBegin[dim] =
                (kernelIndices[dim] * kernelSize) / partition.kernelSplit[dim];
            kernelEnd[dim] = ((kernelIndices[dim] + 1) * kernelSize) /
                             partition.kernelSplit[dim];
          }

          for (unsigned oc = 0; oc != outChanSplit.parallel; ++oc) {
            const auto outChanGrainBegin =
                (oc * outChanNumGrains) / outChanSplit.parallel;
            const auto outChanGrainEnd =
                ((oc + 1) * outChanNumGrains) / outChanSplit.parallel;
            const auto outChanBegin = outChanGrainBegin * outChanGrainSize;
            const auto outChanEnd =
                std::min(outChanGrainEnd * outChanGrainSize, numOutChans);

            for (unsigned of = 0; of != totalFieldSplit; ++of) {
              auto outIndices = unflattenIndex(partition.fieldSplit, of);
              std::vector<unsigned> outFieldBegin(numFieldDims),
                  outFieldEnd(numFieldDims);
              for (unsigned dim = 0; dim != numFieldDims; ++dim) {
                std::tie(outFieldBegin[dim], outFieldEnd[dim]) =
                    getTileOutRange(params, partition, outIndices[dim], dim);
              }

              f({cg, b, outIndices, oc, ic, kernelIndices},
                {cgBegin, cgEnd, batchBegin, batchEnd, outFieldBegin,
                 outFieldEnd, outChanBegin, outChanEnd, inChanBegin, inChanEnd,
                 kernelBegin, kernelEnd});
            }
          }
        }
      }
    }
  }
}

static void iteratePartitionSerial(
    const CanonicalConvParams &params, const Partition &partition,
    const std::function<void(const ConvIndices &, const ConvSlice &)> &f) {
  const auto numFieldDims = params->getNumFieldDims();
  const unsigned numConvGroups = params->getNumConvGroups();

  const unsigned batchSize = params->getBatchSize();
  const unsigned numOutChans = params->getNumOutputChansPerConvGroup();
  const unsigned numInChans = params->getNumInputChansPerConvGroup();
  const auto outChanSplit = partition.outChanSplit;
  const auto inChanSplit = partition.inChanSplit;

  std::vector<unsigned> zeroSpatialIndices(numFieldDims, 0);
  std::vector<unsigned> outFieldEnd(numFieldDims);
  std::vector<unsigned> kernelEnd(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outFieldEnd[dim] = params->getOutputSize(dim);
    kernelEnd[dim] = params->kernelShape[dim];
  }

  for (unsigned ic = 0; ic != inChanSplit.serial; ++ic) {
    // Since serial splits use the same vertex instance, numInChans must be an
    // integer multiple of inChanSplit.serial
    const auto inChanBegin = (ic * numInChans) / inChanSplit.serial;
    const auto inChanEnd = ((ic + 1) * numInChans) / inChanSplit.serial;
    for (unsigned oc = 0; oc != outChanSplit.serial; ++oc) {
      const auto outChanBegin = (oc * numOutChans) / outChanSplit.serial;
      const auto outChanEnd = ((oc + 1) * numOutChans) / outChanSplit.serial;
      f({0, 0, zeroSpatialIndices, oc, ic, zeroSpatialIndices},
        {0, numConvGroups, 0, batchSize, zeroSpatialIndices, outFieldEnd,
         outChanBegin, outChanEnd, inChanBegin, inChanEnd, zeroSpatialIndices,
         kernelEnd});
    }
  }
}

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

static Tensor flattenDims(Tensor t, unsigned from, unsigned to) {
  unsigned factor = t.dim(from);
  return flattenDimsMultiStage(t, from, to, factor);
}

static Tensor unflattenDims(Tensor t, unsigned from, unsigned to,
                            unsigned fromSize) {
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

static Tensor dilate(Graph &graph, const Tensor &t, unsigned dilationFactor,
                     unsigned dim, const std::string &debugPrefix) {
  const auto oldSize = t.dim(dim);
  const auto newSize = getDilatedSize(oldSize, dilationFactor);
  if (newSize == oldSize)
    return t;
  auto expandedT = t.expand({dim + 1});
  const auto dType = expandedT.elementType();
  auto zeroShape = expandedT.shape();
  zeroShape[dim + 1] = dilationFactor - 1;
  Tensor zero = graph.addConstant(dType, zeroShape, 0, debugPrefix + "/zero");
  graph.setTileMapping(zero, 0);
  return concat(expandedT, zero, dim + 1)
      .flatten(dim, dim + 2)
      .slice(0, newSize, dim);
}

// Dilate a tensor but instead of padding with zeros duplicate the nearest
// neighbouring element.
static Tensor dilateWithNearestNeighbour(const Tensor &t,
                                         unsigned dilationFactor,
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
                                           const std::string &debugPrefix) {
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
  if (acts) {
    // Explicitly truncate.
    *acts = pad(*graph, *acts, -static_cast<int>(actsTruncationLower),
                -static_cast<int>(actsTruncationUpper), actsDimIndex);
    // Explicitly dilate.
    *acts = dilate(*graph, *acts, actsDilation, actsDimIndex, debugPrefix);
    // Explicitly pad.
    *acts =
        pad(*graph, *acts, actsPaddingLower, actsPaddingUpper, actsDimIndex);
    // Explicitly flip.
    if (actsFlip) {
      *acts = acts->reverse(actsDimIndex);
    }
  }
  actsSize -= (actsTruncationLower + actsTruncationUpper);
  actsSize = getDilatedSize(actsSize, actsDilation);
  actsSize += actsPaddingLower + actsPaddingUpper;
  actsDilation = 1;
  actsTruncationLower = 0;
  actsTruncationUpper = 0;
  actsPaddingLower = 0;
  actsPaddingUpper = 0;
  actsFlip = false;
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
      emptyWeights ? 0 : roundUp(weightsSize, kernelFactor);
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
      *acts =
          (*graph).addConstant(dType, newActsShape, 0, debugPrefix + "/acts");
      (*graph).setTileMapping(*acts, 0);
    } else {
      std::vector<Tensor> slices;
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
                                       const std::string &debugPrefix) {
  boost::optional<Graph &> g = graph;
  expandSpatialDimMultiStageImpl(params, dim, kernelFactor, g, acts, weights,
                                 debugPrefix);
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
static void expandSpatialDim(ConvParams &params, unsigned dim, Graph &graph,
                             boost::optional<Tensor> &acts,
                             boost::optional<Tensor> &weights,
                             const std::string &debugPrefix) {
  const auto factor = params.getTruncatedKernelSize(dim);
  expandSpatialDimMultiStage(params, dim, factor, graph, acts, weights,
                             debugPrefix);
}

// Planning only, no modification to acts or weights
static void expandSpatialDim(ConvParams &params, unsigned dim) {
  const auto factor = params.getTruncatedKernelSize(dim);
  expandSpatialDimMultiStage(params, dim, factor);
}

static void swapOperands(ConvParams &params, boost::optional<Tensor> &acts,
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

static std::vector<GroupingInfo>
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

// Required for expand dims planning, we don't intend to use graph as mutable,
// but it is deemed safe because it only adds extra compute sets. (we don't
// provide rearrange prog to add them to). We also are careful to pass a copy of
// the plan to this function.
// TODO: make version of this without requiring a mutable graph/plan
static CanonicalConvParams
convolutionPreprocess(Graph &graph, const ConvParams &params,
                      const ConvOptions &options, Plan &plan, unsigned level,
                      const std::vector<Split<ConvIndices>> &indices,
                      bool serial);

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
    auto maxGroupSize = gcd(dimElems, destGrouping[0].second);
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
        const unsigned roundedElems = lcm(expandedDimElems, grainSize);
        const unsigned factor = roundedElems / expandedDimElems;
        const auto dim = expandDimsSpatial[i];
        const auto truncatedKernelSize = params.getTruncatedKernelSize(dim);
        // We're all padding anyway
        if (truncatedKernelSize == 0)
          break;

        poplin::ConvParams noPaddingParams;
        poplin::ConvParams paddingParams;
        { // Without padding
          ConvOptions options{graph.getTarget()};
          std::vector<Split<ConvIndices>> indices;
          auto noPaddingPlan = convPlan;
          noPaddingParams =
              convolutionPreprocess(graph, params, options, noPaddingPlan,
                                    level, indices, false)
                  .getParams();
        }
        { // With padding
          ConvOptions options{graph.getTarget()};
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
          maxGroupSize = gcd(roundedElems, destGrouping[0].second);
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

static void expandSpatialDims(
    ConvParams &params, Plan &plan, unsigned level, Graph &graph,
    boost::optional<Tensor> &acts, boost::optional<Tensor> &weights,
    const ExpandDimsPlan &expandDimsPlan,
    ConvProgramTree::TransformPreProgram *rearrangeProg, bool rearrangeActs,
    bool rearrangeWeights, const std::string &debugPrefix) {
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
                       weights, debugPrefix);
    }
    unsigned kernelSizeBefore =
        params.getTruncatedKernelSize(partialExpansion.first);
    unsigned inputChansBefore = params.inputChannelsPerConvGroup;

    // Partially expand
    expandSpatialDimMultiStage(params, partialExpansion.first,
                               partialExpansion.second, graph, acts, weights,
                               debugPrefix);

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
        t = popops::rearrange::regroupTensor(
            graph, t, rearrangeProg->preTranspose, rearrangeProg->transposeCS,
            regroup.first, regroup.second, debugPrefix);
      }
    }
  }

  // Fully expand remaining dimensions now
  for (; nextToExpand != expandDimsSpatial.size(); ++nextToExpand) {
    const auto factor =
        params.getTruncatedKernelSize(expandDimsSpatial[nextToExpand]);
    expandSpatialDimMultiStage(params, expandDimsSpatial[nextToExpand], factor,
                               graph, acts, weights, debugPrefix);
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
        t = popops::rearrange::regroupTensor(
            graph, t, rearrangeProg->preTranspose, rearrangeProg->transposeCS,
            regroup.first, regroup.second, debugPrefix);
      }
    }
  }
}

static void
expandSpatialDims(ConvParams &params, Plan &plan, unsigned level, Graph &graph,
                  boost::optional<Tensor> &acts,
                  boost::optional<Tensor> &weights,
                  ConvProgramTree::TransformPreProgram *rearrangeProg,
                  bool rearrangeActs = false, bool rearrangeWeights = false,
                  const std::string &debugPrefix = "") {
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
                    rearrangeWeights, debugPrefix);
}

static void
regroupIfBeneficialForPlan(Graph &graph, const ConvParams &params,
                           const Plan &plan, unsigned level, Tensor &in,
                           ConvProgramTree::TransformPreProgram *rearrangeProg,
                           const std::string &debugPrefix = "") {
  auto grouping = detectDimGroupings(graph, in);
  auto destGrouping =
      determinePreprocessedGroupingFromPlan(params, plan, level);
  auto grainSize =
      popops::rearrange::getMinimumRegroupGrainSize(params.inputType);
  if (!grouping.empty() && !destGrouping.empty() &&
      grouping[0].first != destGrouping[0].first &&
      (grouping[0].second % grainSize) == 0 &&
      (destGrouping[0].second % grainSize) == 0) {
    assert(rearrangeProg);
    in = popops::rearrange::regroupTensor(
        graph, in, rearrangeProg->preTranspose, rearrangeProg->transposeCS,
        grouping[0], destGrouping[0], debugPrefix);
  }
}

/// Return whether the specified convolution produces an output that is known
/// to be all zeros. A convolution that produces an empty output is trivially
/// a zero convolution.
static bool isZeroConvolution(const CanonicalConvParams &params) {
  if (params->inputChannelsPerConvGroup == 0 ||
      params->outputChannelsPerConvGroup == 0 || params->batchSize == 0 ||
      params->numConvGroups == 0)
    return true;
  const auto numFieldDims = params->getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (params->outputTransform.paddingLower[dim] +
            params->outputTransform.paddingUpper[dim] ==
        params->getOutputSize(dim)) {
      return true;
    }
  }
  return false;
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

static bool expandDimTransformIsViewOnly(const ConvParams &params,
                                         unsigned dim) {
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

/// Apply any pre-convolution transformations implied by the plan. The
/// plan and the parameters are updated to describe the convolution operation
/// performed on the transformed input. If the \a acts or \ weights pointers are
/// not null they are updated to be views of the original tensors with
/// dimensions that match the shape expected by the convolution operation.
static CanonicalConvParams convolutionPreprocess(
    Graph &graph, ConvParams params, const ConvOptions &options, Plan &plan,
    unsigned level, const std::vector<Split<ConvIndices>> &indices,
    boost::optional<Tensor> &acts, boost::optional<Tensor> &weights,
    bool serial, ConvProgramTree::TransformPreProgram *rearrangeProg = nullptr,
    std::map<Type, Tensor> *rearrangeWritten = nullptr,
    bool rearrangeActs = false, bool rearrangeWeights = false,
    const std::string &debugPrefix = "") {
  if (rearrangeActs) {
    logging::debug("'{}': forcing rearrangement of activations", debugPrefix);
  }

  if (rearrangeWeights) {
    logging::debug("'{}': forcing rearrangement of weights", debugPrefix);
  }

  ConvTransform &transform = plan.transforms[level];
  const auto convGroupGrainSize =
      level < plan.partitions.size() ? plan.partitions[level].convGroupGrainSize
                                     : plan.convGroupsPerGroup;
  const auto inChanGrainSize = level < plan.partitions.size()
                                   ? plan.partitions[level].inChanGrainSize
                                   : plan.inChansPerGroup;
  const auto outChanGrainSize = level < plan.partitions.size()
                                    ? plan.partitions[level].outChanGrainSize
                                    : plan.partialChansPerGroup;

  // transformations that are applied before serially splitting (which is only
  // the top level view transforms).
  if (serial) {
    // implement the extraFieldDims transformation.
    if (transform.extraFieldDims) {
      addExtraDims(params, transform.extraFieldDims);
      if (acts) {
        *acts =
            acts->expand(std::vector<std::size_t>(transform.extraFieldDims, 2));
      }
      if (weights) {
        *weights = weights->expand(
            std::vector<std::size_t>(transform.extraFieldDims, 1));
      }
      transform.extraFieldDims = 0;
    }

    // implement the dilatePostConv transformation.
    params =
        calculateParamsWithDeferredDilation(params, transform.dilatePostConv);
    transform.dilatePostConv.clear();

    // implement the swapOperands transformation.
    if (transform.swapOperands) {
      swapOperands(params, acts, weights);
      transform.swapOperands = false;
    }

    // implement the expandDims transformation for those dimensions which
    // can be expanded without adding/removing elements.
    for (auto it = transform.expandDims.begin();
         it != transform.expandDims.end();) {
      if (expandDimTransformIsViewOnly(params, *it)) {
        expandSpatialDim(params, *it, graph, acts, weights, debugPrefix);
        it = transform.expandDims.erase(it);
      } else {
        ++it;
      }
    }
  } else {
    // implement the expandDims transformation.
    expandSpatialDims(params, plan, level, graph, acts, weights, rearrangeProg,
                      rearrangeActs, rearrangeWeights, debugPrefix);
    transform.expandDims.clear();

    // implement the outChanFlattenDims transformation.
    if (!transform.outChanFlattenDims.empty()) {
      boost::optional<Tensor> maybeActs, maybeWeights;
      if (acts) {
        maybeActs.reset(*acts);
      }

      if (weights) {
        maybeWeights.reset(*weights);
      }

      swapOperands(params, maybeActs, maybeWeights);
      for (auto dim : transform.outChanFlattenDims) {
        expandSpatialDim(params, dim, graph, maybeActs, maybeWeights,
                         debugPrefix);
        if (maybeActs) {
          *maybeActs = flattenDims(*maybeActs, dim + 2, 1);
        }
        params.batchSize *= params.inputFieldShape[dim];
        params.inputFieldShape[dim] = 1;
      }
      swapOperands(params, maybeActs, maybeWeights);
      if (acts) {
        *acts = *maybeActs;
      }

      if (weights) {
        *weights = *maybeWeights;
      }

      transform.outChanFlattenDims.clear();
    }

    // Flatten dimensions.
    if (!transform.flattenDims.empty()) {
      // Zero convolutions introduce truncation. If the truncation is not
      // flattened with the inputFieldShape, we can end up with a negative
      // outputFieldShape, which is invalid. To fix that, for zero
      // convolutions, we flatten the inputFieldShape, reset
      // the truncation so it match the flattened inputFieldShape and flatten
      // the outputPadding, which determines the outputFieldShape.
      bool isZeroConv = isZeroConvolution(params);
      auto preFlattenoutFieldShape = params.getOutputFieldShape();
      auto preFlattenbatchSize = params.batchSize;
      if (isZeroConv) {
        params = getZeroConv(params);
      }

      // Flatten the input field shape.
      doFlatten(transform.flattenDims, acts, params.inputFieldShape,
                params.batchSize);

      if (isZeroConv) {
        // Flatten the truncation and output shape padding, preserving the
        // output field shape.
        params.inputTransform.truncationUpper =
            vectorConvert<unsigned>(params.inputFieldShape);
        params.kernelTransform.truncationUpper =
            vectorConvert<unsigned>(params.kernelShape);

        boost::optional<Tensor> nothing;
        doFlatten(transform.flattenDims, nothing, preFlattenoutFieldShape,
                  preFlattenbatchSize);

        params.outputTransform.paddingUpper =
            vectorConvert<unsigned>(preFlattenoutFieldShape);
      }
    }
    transform.flattenDims.clear();

    // implement the combineConvGroups transformation.
    if (transform.combineConvGroupsFactor != 1) {
      const auto factor = transform.combineConvGroupsFactor;
      const auto numConvGroups = params.numConvGroups;
      const auto paddedNumConvGroups =
          roundUp(params.numConvGroups, std::size_t(factor));
      const auto extraConvGroups = paddedNumConvGroups - numConvGroups;

      // pad conv groups if necessary.
      if (extraConvGroups != 0) {
        if (acts) {
          *acts = pad(graph, *acts, 0, extraConvGroups, 0);
        }

        if (weights) {
          *weights = pad(graph, *weights, 0, extraConvGroups, 0);
        }
      }

      // reshape activations.
      if (acts) {
        // split the group dimension up into two dimensions: [G/f][f]
        *acts =
            acts->reshapePartial(0, 1, {paddedNumConvGroups / factor, factor});

        // move the newly created dimension so that is next to the channel dim.
        *acts = acts->dimRoll(1, acts->rank() - 2);

        // combine the factor dim and the channel dim together.
        *acts = acts->flatten(acts->rank() - 2, acts->rank());
      }

      // reshape and pad weights.
      if (weights) {
        // for this transformation the weights need to be padded in the input
        // and output channel dimensions and the padding needs to wrap the
        // original weights in a one-hot way. for example if you had the
        // following 4 groups of weights (each of which can be any number of
        // spatial dimensions):
        //   A B C D
        // following the transformation you would expect one group of weights:
        //   A 0 0 0
        //   0 B 0 0
        //   0 0 C 0
        //   0 0 0 D
        // where the x and y axes are the input and output channels respectively
        const auto shape = weights->shape();

        const auto N = shape.size();
        const unsigned ciDim = N - 1;
        const unsigned coDim = N - 2;
        const auto numInChans = shape[ciDim];
        const auto numOutChans = shape[coDim];

        // split the group dimension up into two dimensions: [G/f][f]
        *weights = weights->reshapePartial(0, 1, {shape[0] / factor, factor});

        // move the newly created dim so that it is next to the output channel
        // dim.
        *weights = weights->dimRoll(1, weights->rank() - 3);

        // combine the factor dim and the output channel dim together.
        *weights = weights->flatten(weights->rank() - 3, weights->rank() - 1);

        // place the output channels as the first dim, then input channels and
        // then everything else.
        *weights = weights->dimShufflePartial({coDim, ciDim}, {0, 1});

        // need to build up a new tensor with the output channels padded.
        std::vector<Tensor> paddedCi;

        for (unsigned co = 0; co < weights->dim(0); ++co) {
          auto x = (co / numOutChans) % factor;
          auto paddingLower = x * numInChans;
          auto paddingUpper = (factor - 1 - x) * numInChans;

          // pad the input channel dim, which is currently dim 0 if we index by
          // output channel.
          paddedCi.push_back(
              pad(graph, (*weights)[co], paddingLower, paddingUpper, 0, 0));

          // add an extra dim to the front that we can concatenate on.
          paddedCi.back() = paddedCi.back().expand({0});
        }

        *weights = concat(paddedCi, 0);

        // place the input channels and output channels back as the inner-most
        // dims.
        *weights = weights->dimShufflePartial({0, 1}, {coDim, ciDim});
      }

      combineConvGroups(transform.combineConvGroupsFactor, params);
      transform.combineConvGroupsFactor = 1;
    }

    // Zero pad the input / weights.
    const auto paddedConvGroups =
        roundUp(params.getNumConvGroups(), convGroupGrainSize);
    const auto paddedInChans =
        roundUp(params.getNumInputChansPerConvGroup(), inChanGrainSize);
    const auto paddedOutChans =
        roundUp(params.getNumOutputChansPerConvGroup(), outChanGrainSize);

    if (acts) {
      const unsigned gDim = 0;
      const unsigned ciDim = acts->rank() - 1;

      *acts = pad(graph, *acts, 0, paddedConvGroups - params.getNumConvGroups(),
                  gDim);
      *acts = pad(graph, *acts, 0,
                  paddedInChans - params.getNumInputChansPerConvGroup(), ciDim);
    }

    if (weights) {
      const unsigned gDim = 0;
      const unsigned coDim = weights->rank() - 2;
      const unsigned ciDim = weights->rank() - 1;

      *weights = pad(graph, *weights, 0,
                     paddedConvGroups - params.getNumConvGroups(), gDim);
      *weights =
          pad(graph, *weights, 0,
              paddedInChans - params.getNumInputChansPerConvGroup(), ciDim);
      *weights =
          pad(graph, *weights, 0,
              paddedOutChans - params.getNumOutputChansPerConvGroup(), coDim);
    }

    params.numConvGroups = paddedConvGroups;
    params.inputChannelsPerConvGroup = paddedInChans;
    params.outputChannelsPerConvGroup = paddedOutChans;
  }

  if (acts && rearrangeActs) {
    regroupIfBeneficialForPlan(graph, params, plan, level, *acts, rearrangeProg,
                               debugPrefix);
    auto actsRearranged =
        createInputImpl(graph, params, level, serial, indices,
                        debugPrefix + "/actsRearranged", plan, options);

    assert(rearrangeProg);
    rearrangeProg->postTranspose.emplace_back(*acts, actsRearranged);
    auto actsType = actsRearranged.elementType();
    if (rearrangeWritten->count(actsType) == 0) {
      rearrangeWritten->insert(
          std::make_pair(actsType, graph.addVariable(actsType, {0})));
    }
    (*rearrangeWritten)[actsType] =
        concat((*rearrangeWritten)[actsType], actsRearranged.flatten());
    *acts = actsRearranged;
  }

  if (weights && rearrangeWeights) {
    regroupIfBeneficialForPlan(graph, params, plan, level, *weights,
                               rearrangeProg, debugPrefix);
    auto weightsRearranged =
        createWeightsImpl(graph, params, level, serial, indices,
                          debugPrefix + "/weightsRearranged", plan, options);

    assert(rearrangeProg);
    rearrangeProg->postTranspose.emplace_back(*weights, weightsRearranged);
    auto weightsType = weightsRearranged.elementType();
    if (rearrangeWritten->count(weightsType) == 0) {
      rearrangeWritten->insert(
          std::make_pair(weightsType, graph.addVariable(weightsType, {0})));
    }
    (*rearrangeWritten)[weightsType] =
        concat((*rearrangeWritten)[weightsType], weightsRearranged.flatten());
    *weights = weightsRearranged;
  }

  return params;
}

static CanonicalConvParams convolutionPreprocess(
    Graph &graph, const ConvParams &params, const ConvOptions &options,
    Plan &plan, unsigned level, const std::vector<Split<ConvIndices>> &indices,
    Tensor &acts, Tensor &weights, bool serial,
    ConvProgramTree::TransformPreProgram *rearrangeProg = nullptr,
    std::map<Type, Tensor> *rearrangeWritten = nullptr,
    bool rearrangeActs = false, bool rearrangeWeights = false,
    const std::string &debugPrefix = "") {
  auto actsOptional = boost::make_optional(acts);
  auto weightsOptional = boost::make_optional(weights);
  const auto newParams = convolutionPreprocess(
      graph, params, options, plan, level, indices, actsOptional,
      weightsOptional, serial, rearrangeProg, rearrangeWritten, rearrangeActs,
      rearrangeWeights, debugPrefix);
  acts = *actsOptional;
  weights = *weightsOptional;
  return newParams;
}

static CanonicalConvParams
convolutionPreprocess(Graph &graph, const ConvParams &params,
                      const ConvOptions &options, Plan &plan, unsigned level,
                      const std::vector<Split<ConvIndices>> &indices,
                      bool serial) {
  boost::optional<Tensor> acts, weights;
  return convolutionPreprocess(graph, params, options, plan, level, indices,
                               acts, weights, serial);
}

static Tensor convolutionPreprocessInverse(
    const Graph &graph, const ConvParams &originalParams,
    const ConvOptions &options, const Plan &originalPlan, unsigned level,
    const std::vector<Split<ConvIndices>> &indices, Tensor t, bool isActs,
    bool serial) {
  // We only handle applying the inverse of pre-serial split preprocessing
  // currently.
  assert(serial);

  if (serial) {
    const auto &transform = originalPlan.transforms[level];

    auto postExtraFieldDimsParams = originalParams;
    if (transform.extraFieldDims) {
      addExtraDims(postExtraFieldDimsParams, transform.extraFieldDims);
    }
    auto postDeferDilationParams = calculateParamsWithDeferredDilation(
        postExtraFieldDimsParams, transform.dilatePostConv);
    auto postSwapParams = postDeferDilationParams;
    if (transform.swapOperands) {
      swapOperands(postSwapParams);
    }

    for (const auto d : boost::adaptors::reverse(transform.expandDims)) {
      if (expandDimTransformIsViewOnly(postSwapParams, d)) {
        // Undo expand dims transform.
        const unsigned inputChans = t.dim(t.rank() - 1);
        const unsigned n = postSwapParams.kernelShape[d];
        const std::size_t spatialDim =
            d + ((transform.swapOperands ^ isActs) ? 2 : 1);
        assert(t.dim(spatialDim) == 1);
        t = t.reshapePartial(t.rank() - 1, t.rank(), {n, inputChans / n})
                .dimRoll(t.rank() - 1, spatialDim)
                .flatten(spatialDim, spatialDim + 2);
      }
    }

    if (transform.swapOperands) {
      if (isActs) {
        // Output channels become batch size.
        t = t.dimRoll(t.rank() - 2, 1);
      } else {
        // Batch size becomes output channels.
        t = t.dimRoll(1, t.rank() - 2);
      }
    }

    if (transform.extraFieldDims) {
      const unsigned outerSpatialDim = isActs ? 2 : 1;
      t = t.squeeze(
          std::vector<std::size_t>(transform.extraFieldDims, outerSpatialDim));
    }
  }

  return t;
}

// Postprocess results of convolution
// - undo any flattening of the field
// - undo any padding
// shape of output/activations is the internal shape: [G][N]...[Co]
static Tensor convolutionPostprocess(Graph &graph,
                                     const CanonicalConvParams &params,
                                     const ConvTransform &transform,
                                     Tensor activations, bool serial,
                                     std::vector<Copy> &transformPost,
                                     const std::string &debugPrefix) {
  if (serial) {
    auto postAddExtraDimsParams = params.getParams();
    if (transform.extraFieldDims) {
      addExtraDims(postAddExtraDimsParams, transform.extraFieldDims);
    }
    auto postDeferDilationParams = calculateParamsWithDeferredDilation(
        postAddExtraDimsParams, transform.dilatePostConv);
    auto postExpandParams = postDeferDilationParams;
    if (transform.swapOperands) {
      swapOperands(postExpandParams);
    }
    // Undo the swapping of operands.
    if (transform.swapOperands) {
      activations = activations.dimShufflePartial({1, activations.rank() - 1},
                                                  {activations.rank() - 1, 1});
    }
    // Perform any dilations that were deferred until after the convolution.
    if (!transform.dilatePostConv.empty()) {
      // Create a dilated padded view of the activations and copy it to a
      // new variable. It is not valid to return the view as the result as the
      // convolution function is expected to be a writable tensor.

      // the two innermost dimensions of the activations output tensor are the
      // conv group grouping and the output channel grouping. use tensor
      // introspection to find out what these groupings are.
      const auto detectedGroupings = detectChannelGrouping(graph, activations);

      const auto convGroupsPerGroup = detectedGroupings.convGroupsPerGroup;
      const auto outChansPerGroup = detectedGroupings.chansPerGroup;

      auto activationsView = activations;
      // View that matches the activations view except each zero element is
      // replaced with the nearest non zero element. This is used to
      // determine the tile mapping of the variable we create.
      auto mappingView = activations;
      for (const auto spatialDim : transform.dilatePostConv) {
        const auto dilation =
            postAddExtraDimsParams.inputTransform.dilation[spatialDim];
        const auto paddingLower =
            postAddExtraDimsParams.outputTransform.paddingLower[spatialDim];
        const auto paddingUpper =
            postAddExtraDimsParams.outputTransform.paddingUpper[spatialDim];
        const auto dim = 2 + spatialDim;
        activationsView =
            dilate(graph, activationsView, dilation, dim, debugPrefix);
        mappingView = dilateWithNearestNeighbour(mappingView, dilation, dim);
        activationsView =
            pad(graph, activationsView, paddingLower, paddingUpper, dim);
        // pad with nearest neighbour.
        mappingView = pad(mappingView, paddingLower, paddingUpper, dim,
                          popops::padding::Type::EDGE);
      }
      assert(activationsView.shape() == mappingView.shape());
      activationsView = splitActivationIntoGroups(
          activationsView, convGroupsPerGroup, outChansPerGroup);
      mappingView = splitActivationIntoGroups(mappingView, convGroupsPerGroup,
                                              outChansPerGroup);
      activations = graph.addVariable(activationsView.elementType(),
                                      activationsView.shape(),
                                      debugPrefix + "/activationsPostDilate");
      graph.setTileMapping(activations, graph.getTileMapping(mappingView));
      transformPost.emplace_back(activationsView, activations);
      activations = unsplitActivationFromGroups(activations);
    }
    // Remove extra dimensions.
    if (transform.extraFieldDims) {
      std::vector<std::size_t> toSqueeze(transform.extraFieldDims);
      std::iota(toSqueeze.begin(), toSqueeze.end(), std::size_t(2));
      activations = activations.squeeze(toSqueeze);
    }
  } else {
    auto postExpandParams = params.getParams();
    for (auto dim : transform.expandDims) {
      expandSpatialDim(postExpandParams, dim);
    }
    auto postOutChanFlattenParams = postExpandParams;
    if (!transform.outChanFlattenDims.empty()) {
      swapOperands(postOutChanFlattenParams);
      for (auto dim : transform.outChanFlattenDims) {
        expandSpatialDim(postOutChanFlattenParams, dim);
        // Flatten into the batch axis (this will become the output channel
        // axis when we swap back).
        postOutChanFlattenParams.batchSize *=
            postOutChanFlattenParams.inputFieldShape[dim];
        postOutChanFlattenParams.inputFieldShape[dim] = 1;
      }
      swapOperands(postOutChanFlattenParams);
    }

    auto postCombineConvGroupsParams = postOutChanFlattenParams;
    if (transform.combineConvGroupsFactor != 1) {
      combineConvGroups(transform.combineConvGroupsFactor,
                        postCombineConvGroupsParams);
    }

    // Undo padding.
    assert(activations.dim(0) >= postCombineConvGroupsParams.numConvGroups);
    const auto convGroupPadding =
        activations.dim(0) - postCombineConvGroupsParams.numConvGroups;
    activations =
        pad(graph, activations, 0, -static_cast<int>(convGroupPadding), 0);

    assert(activations.dim(activations.rank() - 1) >=
           postCombineConvGroupsParams.outputChannelsPerConvGroup);
    const auto outChanPadding =
        activations.dim(activations.rank() - 1) -
        postCombineConvGroupsParams.outputChannelsPerConvGroup;
    activations = pad(graph, activations, 0, -static_cast<int>(outChanPadding),
                      activations.rank() - 1);

    // undo the combineConvGroups transformation.
    if (transform.combineConvGroupsFactor != 1) {
      // this is the inverse of the operation performed on the activations
      // during convolution preprocessing.
      const auto factor = transform.combineConvGroupsFactor;

      // split the channel dimension from [C*f] to [f][C]
      const auto co = activations.dim(activations.rank() - 1);
      activations = activations.reshapePartial(
          activations.rank() - 1, activations.rank(), {factor, co / factor});

      // move the newly created dim so that it is next to the group dimension.
      activations = activations.dimRoll(activations.rank() - 2, 1);

      // join the factor dimension and the group dimensions back together.
      activations = activations.flatten(0, 2);

      // if we padded the number of conv groups then undo that now.
      if (activations.dim(0) != postOutChanFlattenParams.numConvGroups) {
        const int convGroupPadding =
            activations.dim(0) - postOutChanFlattenParams.numConvGroups;
        activations = pad(graph, activations, 0, -convGroupPadding, 0);
      }
    }

    // Undo flattening of the batch / spatial fields.
    if (!transform.flattenDims.empty()) {
      for (auto it = transform.flattenDims.begin(),
                end = std::prev(transform.flattenDims.end());
           it != end; ++it) {

        if (isZeroConvolution(postCombineConvGroupsParams)) {
          // For zero convolutions, we may not be able to use unflattenDims (as
          // we cannot derive dimension sizes with a product of 0). Instead, we
          // obtain the unflattened shape from postCombineConvGroupsParams.
          const auto innerShape =
              postCombineConvGroupsParams.getOutputFieldShape();
          auto shape = activations.shape();
          shape[1] = postCombineConvGroupsParams.batchSize;
          std::copy(innerShape.begin(), innerShape.end(), shape.begin() + 2);
          activations = activations.reshape(shape);
        } else {
          const auto fromDimIndex = *it;
          const auto toDimIndex = transform.flattenDims.back();
          const auto fromSize = fromDimIndex
                                    ? postCombineConvGroupsParams
                                          .inputFieldShape[fromDimIndex - 1]
                                    : postCombineConvGroupsParams.batchSize;
          activations = unflattenDims(activations, 1 + fromDimIndex,
                                      1 + toDimIndex, fromSize);
        }
      }
    }

    // Undo flattening into output channels.
    for (auto it = transform.outChanFlattenDims.rbegin(),
              end = transform.outChanFlattenDims.rend();
         it != end; ++it) {
      const auto spatialDim = *it;
      const auto spatialDimSize = params->getOutputSize(spatialDim);
      activations = unflattenDims(activations, 2 + spatialDim,
                                  activations.rank() - 1, spatialDimSize);
    }
  }

  return activations;
}

/** Used to iterate usage of tensor elements by different partitions
 *  of the convolution as described by the plan. This is used to
 *  decide on an appropriate mapping for inputs/weights for
 *  a convolution.
 *
 *  \param graph          Poplar graph in which input/weights tensors were
 *                        created for inspection.
 *  \param params         Convolutional parameters for the convolution these
 *                        inputs/weights will be used for at this level.
 *  \param plan           Convolutional plan for this convolution.
 *  \param level          The level in the hierarchy at which to
 *                        calculate input/weights usage.
 *  \param serial         If we are calculating usage for elements in the
 *                        serial partition at this level or the parallel
 *                        partition.
 *  \param acts           Optional activations tensor for which to calculate
 *                        usage.
 *  \param weights        Optional weights tensor for which to calculate
 *                        usage.
 *  \param indices        Stack of convolutional indices for previous levels
 *                        in the hierarchy.
 *  \param options        Options for this convolution.
 */
static TensorUseTracker iterateUsageByPartition(
    Graph &graph, CanonicalConvParams params, Plan plan, unsigned level,
    bool serial, boost::optional<Tensor> acts, boost::optional<Tensor> weights,
    const std::vector<Split<ConvIndices>> &indices, unsigned grainSize,
    unsigned minElementsPerTile, const ConvOptions &options) {
  // Pre-process prior to the parallel partition at this level.
  params = convolutionPreprocess(graph, params.releaseParams(), options, plan,
                                 level, indices, acts, weights, serial);

  TensorUseTracker tracker(graph.getTarget().getNumTiles());

  // TODO: T12870 Where it is known that partitioning does not cause elements of
  // either the inputs or weights to be used on multiple tiles, this should
  // skip calculating the mapping for all but the first serial (and parallel?)
  // slice and reuse the mapping across each slice to save compile time.

  if (level == plan.partitions.size()) {
    const auto &target = graph.getTarget();
    const auto tile = linearizeTileIndices(target, options, indices, plan);
    assert(bool(acts) != bool(weights));
    tracker.add(graph, tile, acts ? *acts : *weights);
  } else {
    const auto &partition = plan.partitions[level];
    if (serial) {
      const auto totalSerialSplit = partition.totalSerialSplit();
      iteratePartitionSerial(
          params, partition,
          [&](const ConvIndices &serialIndices, const ConvSlice &slice) {
            auto subActs = acts;
            auto subWeights = weights;
            auto subIndices = indices;
            Split<ConvIndices> levelIndices;
            levelIndices.serial = serialIndices;
            subIndices.push_back(levelIndices);
            const auto subParams = getSubConvolution(
                slice, params, subActs.get_ptr(), subWeights.get_ptr());
            auto usage = iterateUsageByPartition(
                graph, subParams, plan, level, false, subActs, subWeights,
                subIndices, grainSize, minElementsPerTile, options);
            if (totalSerialSplit == 1) {
              // N.B. we do not resolve usage if there is no serial splitting.
              tracker = std::move(usage);
            } else {
              usage.resolve(
                  graph, grainSize, minElementsPerTile, false,
                  TensorUseTracker::MappingMethod::OptimizeHaloRegions);
              tracker.add(std::move(usage));
            }
          });
    } else {
      const auto totalParallelSplit = partition.totalParallelSplit();
      iteratePartitionParallel(
          params, partition,
          [&](const ConvIndices &parallelIndices, const ConvSlice &slice) {
            auto subActs = acts;
            auto subWeights = weights;
            auto subIndices = indices;
            assert(subIndices.size() == level + 1);
            Split<ConvIndices> &levelIndices = subIndices.back();
            levelIndices.parallel = parallelIndices;
            const auto subParams = getSubConvolution(
                slice, params, subActs.get_ptr(), subWeights.get_ptr());
            auto usage = iterateUsageByPartition(
                graph, subParams, plan, level + 1, true, subActs, subWeights,
                subIndices, grainSize, minElementsPerTile, options);
            if (totalParallelSplit == 1) {
              tracker = std::move(usage);
            } else {
              tracker.add(std::move(usage));
            }
          });
    }
  }
  return tracker;
}

static TensorUseTracker calculateActivationsOrWeightsUsage(
    Graph &graph, const CanonicalConvParams &params, Plan plan, unsigned level,
    bool serial, const Tensor *actsPtr, const Tensor *weightsPtr,
    const std::vector<Split<ConvIndices>> &indices, unsigned grainSize,
    unsigned minElementsPerTile, const ConvOptions &options) {
  boost::optional<Tensor> acts, weights;
  if (actsPtr) {
    acts = *actsPtr;
  }
  if (weightsPtr) {
    weights = *weightsPtr;
  }

  return iterateUsageByPartition(graph, params, plan, level, serial, acts,
                                 weights, indices, grainSize,
                                 minElementsPerTile, options);
}

/// Map the input tensor such that the exchange required during the
/// convolution operation is minimized. If \a isActs is true then the
/// tensor is mapped assuming it the activations operand in convolution
/// operation, otherwise it is mapped assuming it is the weights operand.
static void mapActivationsOrWeights(
    Graph &graph, const CanonicalConvParams &params, Plan plan, unsigned level,
    bool serial, const std::vector<Split<ConvIndices>> &indices,
    const Tensor &in, bool isActs, const ConvOptions &options) {
  // Limit the minimum number of bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was chosen
  // experimentally.
  const auto inType = params->inputType;
  const auto inTypeSize = graph.getTarget().getTypeSize(inType);
  const auto minBytesPerTile = isActs ? 128 : 256;
  const auto minElementsPerTile =
      (minBytesPerTile + inTypeSize - 1) / inTypeSize;
  const auto grainSize = isActs ? plan.inChansPerGroup * plan.convGroupsPerGroup
                                : plan.inChansPerGroup *
                                      plan.partialChansPerGroup *
                                      plan.convGroupsPerGroup;
  auto usage = calculateActivationsOrWeightsUsage(
      graph, params, plan, level, serial, isActs ? &in : nullptr,
      isActs ? nullptr : &in, indices, grainSize, minElementsPerTile, options);
  if (usage.empty()) {
    mapTensorLinearly(graph, in);
  } else {
    usage.mapTensorsByUse(graph, grainSize, minElementsPerTile, true,
                          TensorUseTracker::MappingMethod::OptimizeHaloRegions);
  }
}

static void mapActivations(Graph &graph, const CanonicalConvParams &params,
                           Plan plan, unsigned level, bool serial,
                           const std::vector<Split<ConvIndices>> &indices,
                           Tensor acts, const ConvOptions &options) {
  return mapActivationsOrWeights(graph, params, plan, level, serial, indices,
                                 acts, true, options);
}

static void mapWeights(Graph &graph, const CanonicalConvParams &params,
                       Plan plan, unsigned level, bool serial,
                       const std::vector<Split<ConvIndices>> &indices,
                       Tensor weights, const ConvOptions &options) {
  return mapActivationsOrWeights(graph, params, plan, level, serial, indices,
                                 weights, false, options);
}

static Tensor createInputImpl(Graph &graph, const CanonicalConvParams &params,
                              unsigned level, bool serial,
                              const std::vector<Split<ConvIndices>> &indices,
                              const std::string &name, const Plan &plan,
                              const ConvOptions &options) {
  // If an expensive view-only (just a view of the original operand)
  // transform is applied (i.e. swapOperands), then allocate with this
  // transformation already applied.
  const auto &transforms = plan.transforms[level];
  if (transforms.swapOperands) {
    auto originalParams = params.getParams();
    auto newPlan = plan;
    auto newParams = convolutionPreprocess(graph, params.getParams(), options,
                                           newPlan, level, indices, serial);
    auto t = createWeightsImpl(graph, newParams, level, serial, indices, name,
                               newPlan, options);
    return convolutionPreprocessInverse(graph, originalParams, options, plan,
                                        level, indices, t, true /* isActs */,
                                        serial);
  }

  if (std::any_of(transforms.expandDims.begin(), transforms.expandDims.end(),
                  [&](unsigned dim) {
                    return expandDimTransformIsViewOnly(params.getParams(),
                                                        dim);
                  })) {
    auto originalParams = params.getParams();
    auto newPlan = plan;
    auto newParams = convolutionPreprocess(graph, params.getParams(), options,
                                           newPlan, level, indices, serial);
    auto t = createInputImpl(graph, newParams, level, serial, indices, name,
                             newPlan, options);
    return convolutionPreprocessInverse(graph, originalParams, options, plan,
                                        level, indices, t, true /* isActs */,
                                        serial);
  }
  unsigned inChanSerialSplit = 1;
  if (serial && level < plan.partitions.size()) {
    inChanSerialSplit = plan.partitions[level].inChanSplit.serial;
  }
  const auto numConvGroups = params->getNumConvGroups();
  const auto convGroupsPerGroup = getConvGroupsPerGroup(plan, numConvGroups);
  assert(numConvGroups % convGroupsPerGroup == 0);

  const auto numInChans = params->getNumInputChansPerConvGroup();
  const auto inChansPerGroup =
      getInChansPerGroup(plan, numInChans / inChanSerialSplit);
  assert(numInChans % inChanSerialSplit == 0);
  assert(numInChans % inChansPerGroup == 0);

  std::vector<std::size_t> tensorShape = {
      inChanSerialSplit,
      numConvGroups / convGroupsPerGroup,
      numInChans / (inChansPerGroup * inChanSerialSplit),
      params->getBatchSize(),
  };
  tensorShape.insert(tensorShape.end(), params->inputFieldShape.begin(),
                     params->inputFieldShape.end());
  tensorShape.push_back(convGroupsPerGroup);
  tensorShape.push_back(inChansPerGroup);

  auto t = graph.addVariable(params->inputType, tensorShape, name);
  t = unsplitActivationFromGroups(t.dimRoll(0, 1).flatten(1, 3));
  mapActivations(graph, params, plan, level, serial, indices, t, options);

  // If we're splitting serially then reorder underlying memory regions
  // to make sliced regions contiguous on each tile respecting existing
  // grain size etc.
  if (inChanSerialSplit > 1) {
    // Recover original shape (that is contiguous in memory).
    t = splitActivationIntoGroups(t, convGroupsPerGroup, inChansPerGroup)
            .reshapePartial(
                1, 2,
                {inChanSerialSplit,
                 numInChans / (inChansPerGroup * inChanSerialSplit)})
            .dimRoll(1, 0);
    t = remapTensorToContiguousTileRegions(graph, t, tensorShape,
                                           inChansPerGroup);
    t = unsplitActivationFromGroups(t.dimRoll(0, 1).flatten(1, 3));
  }
  return t;
}

Tensor createInput(Graph &graph, const Plan &plan,
                   const CanonicalConvParams &params, const std::string &name,
                   const ConvOptions &options) {
  const unsigned level = 0;
  bool serial = true;
  const std::vector<Split<ConvIndices>> indices;
  auto input = createInputImpl(graph, params, level, serial, indices, name,
                               plan, options);
  input = actsToExternalShape(input);
  return input;
}

Tensor createInput(Graph &graph, const ConvParams &params_,
                   const std::string &name, const poplar::OptionFlags &options_,
                   PlanningCache *cache) {
  const CanonicalConvParams params(params_);
  const ConvOptions options(graph.getTarget(), options_);

  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  return createInput(graph, plan, params, name, options);
}

static Tensor createWeightsImpl(Graph &graph, const CanonicalConvParams &params,
                                unsigned level, bool serial,
                                const std::vector<Split<ConvIndices>> &indices,
                                const std::string &name, const Plan &plan,
                                const ConvOptions &options) {
  // If an expensive view-only (just a view of the original operand)
  // transform is applied (i.e. swapOperands), then allocate with this
  // transformation already applied.
  const auto &transforms = plan.transforms[level];
  if (transforms.swapOperands) {
    auto originalParams = params.getParams();
    auto newPlan = plan;
    auto newParams = convolutionPreprocess(graph, params.getParams(), options,
                                           newPlan, level, indices, serial);
    auto t = createInputImpl(graph, newParams, level, serial, indices, name,
                             newPlan, options);
    return convolutionPreprocessInverse(graph, originalParams, options, plan,
                                        level, indices, t, false /* isActs */,
                                        serial);
  }

  if (std::any_of(transforms.expandDims.begin(), transforms.expandDims.end(),
                  [&](unsigned dim) {
                    return expandDimTransformIsViewOnly(params.getParams(),
                                                        dim);
                  })) {
    auto originalParams = params.getParams();
    auto newPlan = plan;
    auto newParams = convolutionPreprocess(graph, params.getParams(), options,
                                           newPlan, level, indices, serial);
    auto t = createWeightsImpl(graph, newParams, level, serial, indices, name,
                               newPlan, options);
    return convolutionPreprocessInverse(graph, originalParams, options, plan,
                                        level, indices, t, false /* isActs */,
                                        serial);
  }

  unsigned inChanSerialSplit = 1;
  unsigned outChanSerialSplit = 1;
  if (serial && (level < plan.partitions.size())) {
    inChanSerialSplit = plan.partitions[level].inChanSplit.serial;
    outChanSerialSplit = plan.partitions[level].outChanSplit.serial;
  }
  auto serialSplitIndex = (inChanSerialSplit > 1) ? 2 : 1;
  const auto totalSerialSplit = inChanSerialSplit * outChanSerialSplit;
  const auto numConvGroups = params->getNumConvGroups();
  const auto weightConvGroupsPerGroup =
      getConvGroupsPerGroup(plan, numConvGroups);
  assert(numConvGroups % weightConvGroupsPerGroup == 0);
  const auto weightNumConvGroupGroups =
      numConvGroups / weightConvGroupsPerGroup;

  const auto outNumChans = params->getNumOutputChansPerConvGroup();
  assert(outNumChans % outChanSerialSplit == 0);
  const auto weightNumOutChansPerSerialSplit = outNumChans / outChanSerialSplit;
  const auto weightOutChansPerGroup =
      getOutChansPerGroup(plan, weightNumOutChansPerSerialSplit);
  assert(outNumChans % weightOutChansPerGroup == 0);
  const auto weightNumOutChanGroups = outNumChans / weightOutChansPerGroup;
  const auto weightNumOutChanGroupsPerSerialSplit =
      weightNumOutChanGroups / outChanSerialSplit;

  const auto inNumChans = params->getNumInputChansPerConvGroup();
  assert(inNumChans % inChanSerialSplit == 0);
  const auto weightNumInChansPerSerialSplit = inNumChans / inChanSerialSplit;
  const auto weightInChansPerGroup =
      getInChansPerGroup(plan, weightNumInChansPerSerialSplit);
  assert(inNumChans % weightInChansPerGroup == 0);
  const auto weightNumInChanGroups = inNumChans / weightInChansPerGroup;
  const auto weightNumInChanGroupsPerSerialSplit =
      weightNumInChanGroups / inChanSerialSplit;
  std::vector<std::size_t> weightsShape = {totalSerialSplit,
                                           weightNumConvGroupGroups,
                                           weightNumOutChanGroupsPerSerialSplit,
                                           weightNumInChanGroupsPerSerialSplit};
  weightsShape.insert(weightsShape.end(), params->kernelShape.begin(),
                      params->kernelShape.end());
  weightsShape.push_back(weightConvGroupsPerGroup);
  weightsShape.push_back(weightOutChansPerGroup);
  weightsShape.push_back(weightInChansPerGroup);
  auto weights = graph.addVariable(params->inputType, weightsShape, name);
  weights = unsplitWeightsFromGroups(
      weights.dimRoll(0, serialSplitIndex)
          .flatten(serialSplitIndex, serialSplitIndex + 2));
  mapWeights(graph, params, plan, level, serial, indices, weights, options);

  // If we're splitting serially then reorder underlying memory regions
  // to make sliced regions contiguous on each tile respecting existing
  // grain size etc.
  auto remapSplitTensorToContiguousTileRegions =
      [&](const Tensor &weightsTensor, const unsigned chanIndex,
          const unsigned splitFactor, const unsigned weightsPerSplit) {
        // Recover original shape (that is contiguous in memory).
        Tensor t = splitWeightsIntoGroups(
                       weightsTensor, weightConvGroupsPerGroup,
                       weightInChansPerGroup, weightOutChansPerGroup)
                       .reshapePartial(chanIndex, chanIndex + 1,
                                       {splitFactor, weightsPerSplit})
                       .dimRoll(chanIndex, 0);
        auto grouping = weightConvGroupsPerGroup * weightOutChansPerGroup *
                        weightInChansPerGroup;
        t = remapTensorToContiguousTileRegions(graph, t, weightsShape,
                                               grouping);
        t = unsplitWeightsFromGroups(
            t.dimRoll(0, chanIndex).flatten(chanIndex, chanIndex + 2));
        return t;
      };
  if (inChanSerialSplit > 1) {
    weights = remapSplitTensorToContiguousTileRegions(
        weights, serialSplitIndex, inChanSerialSplit,
        weightNumInChanGroupsPerSerialSplit);
  }
  if (outChanSerialSplit > 1) {
    weights = remapSplitTensorToContiguousTileRegions(
        weights, serialSplitIndex, outChanSerialSplit,
        weightNumOutChanGroupsPerSerialSplit);
  }
  return weights;
}

Tensor createWeights(Graph &graph, const Plan &plan,
                     const CanonicalConvParams &params, const std::string &name,
                     const ConvOptions &options) {
  const unsigned level = 0;
  bool serial = true;
  const std::vector<Split<ConvIndices>> indices;
  auto weights = createWeightsImpl(graph, params, level, serial, indices, name,
                                   plan, options);
  return weightsToExternalShape(weights);
}

Tensor createWeights(Graph &graph, const ConvParams &params_,
                     const std::string &name,
                     const poplar::OptionFlags &options_,
                     PlanningCache *cache) {
  const CanonicalConvParams params(params_);
  const ConvOptions options(graph.getTarget(), options_);

  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  return createWeights(graph, plan, params, name, options);
}

static void mapBiases(poplar::Graph &graph, const poplar::Tensor &biases,
                      const poplar::Tensor &out) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  TensorUseTracker useTracker(numTiles);
  // Create a view of the output where channels are the outermost dimension.
  auto outRegrouped = out.dimShufflePartial({out.rank() - 1}, {1})
                          .flatten(2, out.rank())
                          .flatten(0, 2);
  auto outMapping = graph.getTileMapping(outRegrouped);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    for (const auto &interval : outMapping[tile]) {
      unsigned chanBegin = interval.begin() / outRegrouped.dim(1);
      unsigned chanEnd =
          (interval.end() + outRegrouped.dim(1) - 1) / outRegrouped.dim(1);
      useTracker.add(graph, tile, biases.slice(chanBegin, chanEnd));
    }
  }
  const auto dType = out.elementType();
  const auto grainSize = target.getVectorWidth(dType);

  // Limit the minimum number of bias bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto dTypeSize = target.getTypeSize(dType);
  const auto minBytesPerTile = 8;
  const auto minElementsPerTile = (minBytesPerTile + dTypeSize - 1) / dTypeSize;

  useTracker.mapTensorsByUse(graph, grainSize, minElementsPerTile,
                             true /* extendPartialUsage */);
}

poplar::Tensor createBiases(poplar::Graph &graph, const Tensor &acts_,
                            const std::string &name) {
  const auto acts = actsToInternalShape(acts_, 1, acts_.dim(1));
  const auto numOutChans = acts.dim(acts.rank() - 1);
  const auto dType = acts.elementType();
  auto biases = graph.addVariable(dType, {numOutChans}, name);
  mapBiases(graph, biases, acts);
  return biases;
}

struct ConvOutputSlice {
  unsigned outXBegin;
  unsigned outXEnd;
  unsigned b;
  std::vector<unsigned> outerFieldIndices;
  unsigned outZGroup;
  unsigned cg;
  ConvOutputSlice(unsigned outXBegin, unsigned outXEnd, unsigned b,
                  std::vector<unsigned> outerFieldIndices, unsigned outZGroup,
                  unsigned cg)
      : outXBegin(outXBegin), outXEnd(outXEnd), b(b),
        outerFieldIndices(std::move(outerFieldIndices)), outZGroup(outZGroup),
        cg(cg) {}
};

static std::vector<std::vector<ConvOutputSlice>>
partitionConvOutputBetweenWorkers(const Graph &graph, unsigned batchBegin,
                                  unsigned batchEnd,
                                  const std::vector<unsigned> &outFieldBegin,
                                  const std::vector<unsigned> &outFieldEnd,
                                  unsigned outZGroupBegin,
                                  unsigned outZGroupEnd, unsigned cgBegin,
                                  unsigned cgEnd) {
  const auto numFieldDims = outFieldBegin.size();
  assert(outFieldEnd.size() == numFieldDims);
  std::vector<std::vector<ConvOutputSlice>> perWorkerConvOutputSlices;
  const auto &target = graph.getTarget();
  std::vector<unsigned> rowIterationSpace = {
      outZGroupEnd - outZGroupBegin, batchEnd - batchBegin, cgEnd - cgBegin};
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    rowIterationSpace.push_back(outFieldEnd[dim] - outFieldBegin[dim]);
  }
  const auto numRows = product(rowIterationSpace);
  const auto numWorkers = target.getNumWorkerContexts();
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numRows);
  rowIterationSpace.push_back(rowSplitFactor);
  const auto numPartRows = numRows * rowSplitFactor;
  const auto outXBegin = outFieldBegin.back();
  const auto outXEnd = outFieldEnd.back();
  const auto outWidth = outXEnd - outXBegin;
  for (unsigned worker = 0; worker != numWorkers; ++worker) {
    const auto begin = (worker * numPartRows) / numWorkers;
    const auto end = ((worker + 1) * numPartRows) / numWorkers;
    perWorkerConvOutputSlices.emplace_back();
    for (unsigned partRow = begin; partRow != end; ++partRow) {
      auto indices = unflattenIndex(rowIterationSpace, partRow);
      const auto ocg = outZGroupBegin + indices[0];
      const auto b = batchBegin + indices[1];
      const auto cg = cgBegin + indices[2];
      std::vector<unsigned> outerFieldIndices;
      for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
        outerFieldIndices.push_back(outFieldBegin[dim] + indices[dim + 3]);
      }
      const auto partInRow = indices.back();
      const auto workerOutXBegin =
          outXBegin + (partInRow * outWidth) / rowSplitFactor;
      const auto workerOutXEnd =
          outXBegin + ((partInRow + 1) * outWidth) / rowSplitFactor;
      if (workerOutXBegin == workerOutXEnd)
        continue;
      if (!perWorkerConvOutputSlices.back().empty() &&
          cg == perWorkerConvOutputSlices.back().back().cg &&
          b == perWorkerConvOutputSlices.back().back().b &&
          ocg == perWorkerConvOutputSlices.back().back().outZGroup &&
          outerFieldIndices ==
              perWorkerConvOutputSlices.back().back().outerFieldIndices) {
        perWorkerConvOutputSlices.back().back().outXEnd = workerOutXEnd;
      } else {
        perWorkerConvOutputSlices.back().emplace_back(
            workerOutXBegin, workerOutXEnd, b, outerFieldIndices, ocg, cg);
      }
    }
  }
  return perWorkerConvOutputSlices;
}

static bool fitsMachineStride(const Target &target, int stride) {
  int64_t maxLimit = (1 << target.getNumStrideBits()) / 2 - 1;
  int64_t minLimit = -(1 << target.getNumStrideBits()) / 2;
  return stride >= minLimit && stride <= maxLimit;
}

// Weights for output channel groups is reordered to be reverse order
static std::vector<Tensor> reorderWeightsTensor(std::vector<Tensor> &in,
                                                unsigned numInGroups,
                                                unsigned numOutGroups,
                                                unsigned numConvGroups) {
  assert(in.size() == numInGroups * numOutGroups * numConvGroups);
  std::vector<Tensor> reorderedIn;
  for (auto cg = 0U; cg != numConvGroups; ++cg) {
    for (auto ig = 0U; ig != numInGroups; ++ig) {
      for (auto ogp1 = numOutGroups; ogp1 > 0; --ogp1) {
        const auto og = ogp1 - 1;
        auto inIndex = cg * numOutGroups * numInGroups + og * numInGroups + ig;
        reorderedIn.push_back(in[inIndex]);
      }
    }
  }
  return reorderedIn;
}

struct ConvVertexSpatialPartition {
  std::vector<std::size_t> outBeginIndices;
  unsigned outXWidth;
  std::vector<std::size_t> inBeginIndices;
  unsigned inXWidth;
  unsigned context;
  unsigned subKernelPosition;
};

std::vector<ConvVertexSpatialPartition>
createPartitions(const CanonicalConvParams &params,
                 unsigned convUnitWeightHeight, unsigned convUnitWeightWidth,
                 unsigned contextsPerVertex) {
  auto numSubKernelSlices = params->kernelShape;
  assert(numSubKernelSlices[0] % convUnitWeightHeight == 0);
  assert(numSubKernelSlices.back() % convUnitWeightWidth == 0);
  numSubKernelSlices[0] /= convUnitWeightHeight;
  numSubKernelSlices.back() /= convUnitWeightWidth;
  const auto numSubKernelPositions = product(numSubKernelSlices);
  const auto numFieldDims = params->getNumFieldDims();

  std::vector<ConvVertexSpatialPartition> partitions;

  for (unsigned k = 0; k != numSubKernelPositions; ++k) {
    auto kernelBeginIndices = unflattenIndex(numSubKernelSlices, k);
    kernelBeginIndices[0] = kernelBeginIndices[0] * convUnitWeightHeight;
    kernelBeginIndices.back() = kernelBeginIndices.back() * convUnitWeightWidth;
    std::vector<unsigned> tileConvOutBegin;
    std::vector<unsigned> tileConvOutSize;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      const auto kernelBeginIndex = kernelBeginIndices[dim];
      const auto kernelEndIndex =
          kernelBeginIndex + (dim == 0 ? convUnitWeightHeight : 1);
      const auto outputSize = params->getOutputSize(dim);
      auto convOutRange = getOutputRangeForKernelRange(
          dim, {0, outputSize}, {kernelBeginIndex, kernelEndIndex},
          params.getParams());
      tileConvOutBegin.push_back(convOutRange.first);
      tileConvOutSize.push_back(convOutRange.second - convOutRange.first);
    }
    if (product(tileConvOutSize) == 0)
      continue;
    auto workerPartition = partitionConvPartialByWorker(
        params->getBatchSize(), tileConvOutSize, contextsPerVertex,
        params->inputTransform.dilation, params->outputTransform.stride);
    for (unsigned i = 0; i != contextsPerVertex; ++i) {
      for (const auto &partialRow : workerPartition[i]) {
        auto workerOutXBegin = tileConvOutBegin.back() + partialRow.xBegin;
        auto workerOutXEnd = tileConvOutBegin.back() + partialRow.xEnd;
        std::tie(workerOutXBegin, workerOutXEnd) = getOutputRangeForKernelIndex(
            numFieldDims - 1, {workerOutXBegin, workerOutXEnd},
            kernelBeginIndices.back(), params.getParams());
        const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
        if (workerOutWidth == 0)
          continue;
        std::vector<std::size_t> outBeginIndices = {partialRow.b};
        for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
          outBeginIndices.push_back(partialRow.outerFieldIndices[dim] +
                                    tileConvOutBegin[dim]);
        }
        outBeginIndices.push_back(partialRow.xBegin + tileConvOutBegin.back());
        std::vector<std::size_t> inBeginIndices = {partialRow.b};
        if (numFieldDims > 1) {
          const auto kOuterBegin = kernelBeginIndices[0];
          const auto kOuterEnd = kOuterBegin + convUnitWeightHeight;
          const auto outOuterIndex =
              tileConvOutBegin[0] + partialRow.outerFieldIndices[0];
          for (unsigned k = kOuterBegin; k != kOuterEnd; ++k) {
            auto inOuterIndex =
                getInputIndex(0, outOuterIndex, k, params.getParams());
            if (inOuterIndex != ~0U) {
              auto inOuterBeginIndex =
                  inOuterIndex + (params->inputTransform.flip.front() !=
                                          params->kernelTransform.flip.front()
                                      ? 1
                                      : -1) *
                                     (k - kOuterBegin) *
                                     params->kernelTransform.dilation.front();
              inBeginIndices.push_back(inOuterBeginIndex);
              break;
            }
          }
          if (inBeginIndices.size() < 2) {
            continue;
          }
        }
        for (unsigned dim = 1; dim + 1 < numFieldDims; ++dim) {
          auto inIndex = getInputIndex(
              dim, tileConvOutBegin[dim] + partialRow.outerFieldIndices[dim],
              kernelBeginIndices[dim], params.getParams());
          assert(inIndex != ~0U);
          inBeginIndices.push_back(inIndex);
        }
        auto workerInXRange =
            getInputRange(numFieldDims - 1, {workerOutXBegin, workerOutXEnd},
                          kernelBeginIndices.back(), params.getParams());
        assert(workerInXRange.first != ~0U);
        inBeginIndices.push_back(workerInXRange.first);
        partitions.push_back({std::move(outBeginIndices), workerOutWidth,
                              std::move(inBeginIndices),
                              workerInXRange.second - workerInXRange.first, i,
                              k});
      }
    }
  }
  return partitions;
}

static void createConvPartialAmpVertex(Graph &graph, const Plan &plan,
                                       unsigned tile,
                                       const CanonicalConvParams &params,
                                       ComputeSet fwdCS, Tensor in,
                                       Tensor weights, Tensor out,
                                       bool use128BitConvUnitLoad,
                                       const std::string &debugPrefix) {
  const auto &target = graph.getTarget();
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(in.elementType() == FLOAT);
  const auto convUnitWeightHeight = weightsPerConvUnit / plan.inChansPerGroup;
  if (convUnitWeightHeight != 1) {
    assert(weights.dim(3) % convUnitWeightHeight == 0);
    assert(params->inputTransform.truncationLower[0] == 0);
    assert(params->inputTransform.truncationUpper[0] == 0);
    assert(params->inputTransform.dilation[0] == 1);
    assert(params->inputTransform.paddingLower[0] == 0);
    assert(params->inputTransform.paddingUpper[0] == 0);
  }

  const auto numFieldDims = params->getNumFieldDims();
  const unsigned numConvGroupGroups = out.dim(0);
  const unsigned numOutChanGroups = out.dim(1);
  const unsigned numInChanGroups = in.dim(1);
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const unsigned inChansPerGroup = plan.inChansPerGroup;

  // AMP vertices only support having a single conv group per grouping.
  assert(plan.convGroupsPerGroup == 1);

  auto isNonZero = [](unsigned x) { return x != 0; };

  // If the number of input channels is zero, the output could still be only
  // padding. The 1x1 vertex requires the input channels to be non-zero to
  // write zero to the output. Hence we always use a nx1 vertex if number
  // of input channels is zero.
  bool nx1Vertex =
      numInChanGroups * inChansPerGroup == 0 ||
      product(params->kernelShape) != 1 ||
      params->inputTransform.dilation != params->outputTransform.stride ||
      std::any_of(params->outputTransform.paddingLower.begin(),
                  params->outputTransform.paddingLower.end(), isNonZero) ||
      std::any_of(params->outputTransform.paddingUpper.begin(),
                  params->outputTransform.paddingUpper.end(), isNonZero);
  bool flipOut = params->inputTransform.flip[numFieldDims - 1];

  std::vector<Tensor> weightsWindow;
  for (unsigned cg = 0; cg != numConvGroupGroups; ++cg) {
    for (unsigned ozg = 0; ozg < numOutChanGroups; ++ozg) {
      for (unsigned izg = 0; izg < numInChanGroups; ++izg) {
        auto window = weights[cg][ozg][izg].flatten();
        weightsWindow.push_back(window.flatten());
      }
    }
  }

  const auto contextsPerVertex = target.getNumWorkerContexts();
  // The number of n x 1 x ... 1 slices required to cover the kernel in each
  // dimension.
  auto numSubKernelSlices = params->kernelShape;
  assert(numSubKernelSlices[0] % convUnitWeightHeight == 0);
  numSubKernelSlices[0] /= convUnitWeightHeight;
  const auto numSubKernelPositions = product(numSubKernelSlices);

  auto kernelInnerElements =
      product(numSubKernelSlices) / numSubKernelSlices[0];
  auto inStrideX = params->outputTransform.stride.back();
  auto outStrideX = params->inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;
  outStrideX /= strideDivisor;

  const auto convInputLoadElems =
      target.getConvUnitInputLoadElemsPerCycle(in.elementType() == FLOAT);

  const auto convUnitWeightWidth = 1u;
  auto partitions = createPartitions(params, convUnitWeightHeight,
                                     convUnitWeightWidth, contextsPerVertex);

  assert(!partitions.empty());
  std::vector<std::size_t> inputBatchAndFieldShape = {params->getBatchSize()};
  std::vector<std::size_t> outputBatchAndFieldShape = {params->getBatchSize()};
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    inputBatchAndFieldShape.push_back(params->inputFieldShape[dim]);
    outputBatchAndFieldShape.push_back(params->getOutputSize(dim));
  }

  bool useConvPartial1x1OutVertex = !nx1Vertex;

  if (useConvPartial1x1OutVertex) {
    // In most common cases there should only be one partition per worker for
    // a 1x1 vertex. To avoid having two types of 1x1 vertices we just make it
    // a nx1 vertex if there's more than one partition.
    std::vector<unsigned> partitionsPerContext(contextsPerVertex);
    std::for_each(partitions.begin(), partitions.end(),
                  [&](const ConvVertexSpatialPartition &p) {
                    partitionsPerContext[p.context] += 1;
                  });

    // find if any of the contexts has more than one partition
    unsigned contextsHaveMoreThanOnePartition = false;
    for (auto v : partitionsPerContext) {
      if (v > 1) {
        contextsHaveMoreThanOnePartition = true;
        break;
      }
    }
    useConvPartial1x1OutVertex = !contextsHaveMoreThanOnePartition;
  }

  std::vector<std::vector<unsigned>> worklist(contextsPerVertex *
                                              numSubKernelPositions);

  // create worklist now that dimensions of all splits are known
  for (const auto &p : partitions) {
    const auto outBeginOffset =
        flattenIndex(outputBatchAndFieldShape, p.outBeginIndices);
    const auto inBeginOffset =
        flattenIndex(inputBatchAndFieldShape, p.inBeginIndices);
    const auto outOffset =
        flipOut ? outBeginOffset + p.outXWidth - 1 : outBeginOffset;
    const auto numFieldElems =
        useConvPartial1x1OutVertex
            ? p.outXWidth
            : (p.outXWidth + outStrideX - 1) / outStrideX;
    const auto wIndex = p.subKernelPosition * contextsPerVertex + p.context;

    worklist[wIndex].push_back(outOffset);
    worklist[wIndex].push_back(numFieldElems);
    worklist[wIndex].push_back(inBeginOffset);
  }

  // Encode worklist offsets
  [&](std::vector<std::vector<unsigned>> &worklists) {
    const auto outElementTypeSize = out.elementType() == HALF ? 2 : 4;
    const auto inElementTypeSize = in.elementType() == HALF ? 2 : 4;
    // We represent the the worklist offset as:
    // offset = field offset * chansPerGroup * size(element) / 8
    // This works because we know chansPerGroup * size(element) % 8 = 0, from
    // the constraints on the vertex. Which means in the vertex we just need to
    // multiply by 8 to get the offset relative to the base.
    assert((outChansPerGroup * outElementTypeSize) % 8 == 0);
    assert((inChansPerGroup * inElementTypeSize) % 8 == 0);

    for (auto &worklist : worklists) {
      for (unsigned i = 0; i < worklist.size(); i += 3) {
        worklist[i] = (worklist[i] * outChansPerGroup * outElementTypeSize) / 8;
      }
      for (unsigned i = 2; i < worklist.size(); i += 3) {
        worklist[i] = (worklist[i] * inChansPerGroup * inElementTypeSize) / 8;
      }
    }
  }(worklist);

  std::vector<Tensor> outWindow;
  std::vector<Tensor> inWindow;

  for (unsigned cg = 0; cg != numConvGroupGroups; ++cg) {
    for (unsigned ozg = 0; ozg != numOutChanGroups; ++ozg) {
      auto o = out[cg][ozg];
      outWindow.push_back(o.flatten());
    }
    // TODO: T12872 If the tile kernel size is 1 and the stride is greater than
    // one we could subsample the input instead of using input striding.
    for (unsigned izg = 0; izg != numInChanGroups; ++izg) {
      auto window = in[cg][izg];
      inWindow.push_back(window.flatten());
    }
  }
  // This stride is what's used to move down one element in the input field by
  // the vertex.
  int inRowStride = getInRowStride(
      params.getParams(),
      product(inputBatchAndFieldShape) /
          (inputBatchAndFieldShape[0] * inputBatchAndFieldShape[1]),
      useConvPartial1x1OutVertex, convUnitWeightHeight);

  int transformedInStride =
      (static_cast<int>(inStrideX) - 1 -
       static_cast<int>(convUnitWeightHeight - 1) * inRowStride) *
          static_cast<int>(inChansPerGroup / convInputLoadElems) +
      1;
  // fill in worklist
  unsigned outStrideToUse = useConvPartial1x1OutVertex ? 1 : outStrideX;
  int scaledOutStride = static_cast<int>(outStrideToUse * outChansPerGroup);
  int halfStrideAdj = -4;
  int floatStrideAdj = -6;
  // For dual AMP codelets need to offset output stride by extra 8 elements
  if (plan.numConvUnitsRequired > 8) {
    halfStrideAdj += -8;
    floatStrideAdj += -8;
  }

  int transformedOutStride =
      (plan.types.back().partialType == poplar::FLOAT ? floatStrideAdj
                                                      : halfStrideAdj) +
      (flipOut ? -scaledOutStride : scaledOutStride);

  int transformedInRowStride =
      (inRowStride - 1) *
          static_cast<int>(inChansPerGroup / convInputLoadElems) +
      1;

  // Limits for field and worklist elements
  const auto unsignedMax = std::numeric_limits<unsigned short>::max();
  const auto signedMax = std::numeric_limits<short>::max();
  const auto signedMin = std::numeric_limits<short>::min();

  bool useLimitedVer = true;
  const auto zerosInfo = outWindow[0].numElements();
  if (!fitsMachineStride(target, transformedOutStride / 2) ||
      !fitsMachineStride(target, transformedInStride) ||
      !fitsMachineStride(target, transformedInRowStride))
    useLimitedVer = false;

  if ((numConvGroupGroups - 1 > unsignedMax) ||
      (numOutChanGroups - 1 > unsignedMax) || (numInChanGroups > unsignedMax) ||
      (transformedInStride < signedMin) || (transformedInStride > signedMax) ||
      (outChansPerGroup > unsignedMax) || (transformedOutStride < signedMin) ||
      (transformedOutStride > signedMax))
    useLimitedVer = false;

  const auto doubleWordWrites =
      zerosInfo / (8 / target.getTypeSize(outWindow[0].elementType()));
  const auto doubleWordWritesPerWorker =
      (doubleWordWrites + contextsPerVertex - 1) / contextsPerVertex;

  if (!useConvPartial1x1OutVertex) {
    if ((kernelInnerElements - 1 > unsignedMax) ||
        (numSubKernelSlices[0] - 1 > unsignedMax) ||
        (convUnitWeightHeight - 1 > unsignedMax) ||
        (transformedInRowStride > signedMax) ||
        (transformedInRowStride < signedMin) ||
        (inChansPerGroup > unsignedMax) ||
        doubleWordWritesPerWorker > target.getRptCountMax())
      useLimitedVer = false;

    if (convUnitWeightHeight != 1 && convUnitWeightHeight != 2 &&
        convUnitWeightHeight != 4)
      useLimitedVer = false;
  }
  // check if all worklist items meet range constraints
  for (auto j = 0U; j != worklist.size() && useLimitedVer; ++j) {
    const auto &vec = worklist[j];
    for (auto i = 0U; i != vec.size(); ++i) {
      // worklist is a multiple of 3.
      // i % 3 == 0 : output offset
      // i % 3 == 1 : number of field elems
      // i % 3 == 2 : input offset
      if ((i % 3) == 1) {
        if (vec[i] > target.getRptCountMax()) {
          useLimitedVer = false;
          break;
        }
      } else {
        if (vec[i] > unsignedMax) {
          useLimitedVer = false;
          break;
        }
      }
    }
  }

  const auto worklistEntryType = useLimitedVer ? UNSIGNED_SHORT : UNSIGNED_INT;

  auto codeletName = useConvPartial1x1OutVertex ? "poplin::ConvPartial1x1Out"
                                                : "poplin::ConvPartialnx1";
  auto v = graph.addVertex(
      fwdCS, templateVertex(codeletName, in.elementType(),
                            plan.types.back().partialType,
                            useLimitedVer ? "true" : "false",
                            use128BitConvUnitLoad ? "true" : "false",
                            plan.numConvUnitsRequired));

  // The parameters are modified to what the vertex uses
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"],
                reorderWeightsTensor(weightsWindow, numInChanGroups,
                                     numOutChanGroups, numConvGroupGroups));
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroupsM1"], numOutChanGroups - 1);
  graph.setInitialValue(v["numInGroups"], numInChanGroups);
  assert(inChansPerGroup % convInputLoadElems == 0);

  graph.setInitialValue(v["transformedInStride"], transformedInStride);

  graph.setInitialValue(v["numConvGroupsM1"], numConvGroupGroups - 1);

  graph.setInitialValue(v["transformedOutStride"], transformedOutStride);

  // Subtract numFieldElems by 3 to avoid computing this within the vertex
  auto numFieldElemsLessThree = [worklistEntryType](
                                    std::vector<unsigned> &wlist) {
    for (unsigned i = 1; i < wlist.size(); i += 3) {
      auto numFieldElemsMinus3 = static_cast<int>(wlist[i]) - 3;
      if (worklistEntryType == UNSIGNED_SHORT) {
        wlist[i] = static_cast<unsigned short>(numFieldElemsMinus3 & 0xffff);
      } else {
        wlist[i] = static_cast<unsigned>(numFieldElemsMinus3);
      }
    }
  };

  // Worklists are 2D for nx1 and 1D for 1x1
  if (useConvPartial1x1OutVertex) {
    std::vector<unsigned> worklist1x1(contextsPerVertex * 3);
    for (unsigned i = 0; i < worklist.size(); ++i) {
      std::copy(std::begin(worklist[i]), std::end(worklist[i]),
                worklist1x1.begin() + 3 * i);
    }
    numFieldElemsLessThree(worklist1x1);
    auto t = graph.addConstant(worklistEntryType, {worklist1x1.size()},
                               worklist1x1.data(), debugPrefix + "/worklists");
    graph.setTileMapping(t, tile);
    graph.connect(v["worklists"], t);
  } else {
    graph.setFieldSize(v["worklists"], worklist.size());
    for (unsigned i = 0; i < worklist.size(); ++i) {
      numFieldElemsLessThree(worklist[i]);
      auto t =
          graph.addConstant(worklistEntryType, {worklist[i].size()},
                            worklist[i].data(), debugPrefix + "/worklists");
      graph.setTileMapping(t, 0);
      graph.connect(v["worklists"][i], t);
    }
  }

  if (!useConvPartial1x1OutVertex) {
    graph.setInitialValue(v["kernelInnerElementsM1"], kernelInnerElements - 1);
    graph.setInitialValue(v["kernelOuterSizeM1"], numSubKernelSlices[0] - 1);
    graph.setInitialValue(v["ampKernelHeightM1"], convUnitWeightHeight - 1);
    graph.setInitialValue(v["transformedInRowStride"], transformedInRowStride);
    graph.setInitialValue(v["zerosInfo"], zerosInfo);
  }
  graph.setTileMapping(v, tile);
}

static Tensor sliceOutput(const Tensor &out, const ConvSlice &slice,
                          const unsigned convGroupsPerGroup,
                          const unsigned outChansPerGroup) {
  // shape of out is [G1][OC1][N]...[G2][OC2]
  std::vector<std::size_t> begin, end;

  assert(slice.cgBegin % convGroupsPerGroup == 0);
  assert(slice.cgEnd % convGroupsPerGroup == 0);
  begin.push_back(slice.cgBegin / convGroupsPerGroup);
  end.push_back(slice.cgEnd / convGroupsPerGroup);

  assert(slice.outChanBegin % outChansPerGroup == 0);
  assert(slice.outChanEnd % outChansPerGroup == 0);
  begin.push_back(slice.outChanBegin / outChansPerGroup);
  end.push_back(slice.outChanEnd / outChansPerGroup);

  begin.push_back(slice.batchBegin);
  end.push_back(slice.batchEnd);

  const auto numFieldDims = slice.outFieldBegin.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    begin.push_back(slice.outFieldBegin[dim]);
    end.push_back(slice.outFieldEnd[dim]);
  }

  begin.push_back(0);
  end.push_back(convGroupsPerGroup);

  begin.push_back(0);
  end.push_back(outChansPerGroup);

  return out.slice(begin, end);
}

/// Return the tensor \a t with the specified amount of padding added to the
/// specified dimension. The padding elements are added as a new variable
/// which is concatenated to the \a padding tensors. It is the caller's
/// responsibility to initialize the padding.
static Tensor padWithVariable(Graph &graph, Tensor t, unsigned paddingLower,
                              unsigned paddingUpper, unsigned dim,
                              Tensor &padding, const std::string &debugPrefix) {
  auto paddingSize = paddingLower + paddingUpper;
  auto paddingShape = t.shape();
  paddingShape[dim] = paddingSize;
  auto paddingTensor = graph.addVariable(t.elementType(), paddingShape,
                                         debugPrefix + "/zeroPadding");
  auto paddingLowerTensor = paddingTensor.slice(0, paddingLower, dim);
  auto paddingUpperTensor = paddingTensor.slice(paddingLower, paddingSize, dim);
  padding = concat(padding, paddingTensor.flatten());
  return concat({paddingLowerTensor, t, paddingUpperTensor}, dim);
}

// Padding the input / weights using constants creates aliases of the zero
// constant which causes a rearrangement between the exchange and compute
// steps. This rearrangement can double amount of temporary memory required
// Workaround this by creating a padding variable that in used in place
// of the constant zero. The size of the variable is equal to the
// amount of padding required so we can avoid aliasing of elements.
// TODO: fixing T5913 means we should be able to remove this.
struct Padder {
  Padder(Graph &graph, const unsigned tile, std::vector<Copy> &transformPre,
         std::map<Type, Tensor> &copyWritten, const Type &type,
         const std::string &debugPrefix)
      : graph(graph), tile(tile), transformPre(transformPre), type(type),
        copyWritten(copyWritten), debugPrefix(debugPrefix) {
    paddingTensor =
        graph.addConstant(type, {0}, 0, debugPrefix + "/paddingTensor");
    graph.setTileMapping(paddingTensor, 0);
  }

  ~Padder() {
    if (paddingTensor.numElements() != 0) {
      auto c =
          graph.addConstant(paddingTensor.elementType(), paddingTensor.shape(),
                            0, debugPrefix + "/paddingTensor");
      graph.setTileMapping(c, 0);
      graph.setTileMapping(paddingTensor, tile);
      transformPre.emplace_back(c, paddingTensor);

      if (copyWritten.count(type) == 0) {
        copyWritten.insert(std::make_pair(type, graph.addVariable(type, {0})));
      }
      copyWritten[type] = concat(copyWritten[type], paddingTensor.flatten());
    }
  }

  Tensor operator()(Graph &graph, const Tensor &t, unsigned paddingLower,
                    unsigned paddingUpper, unsigned dim) {
    assert(t.elementType() == paddingTensor.elementType());
    return padWithVariable(graph, t, paddingLower, paddingUpper, dim,
                           paddingTensor, debugPrefix);
  }

private:
  Graph &graph;
  unsigned tile;
  std::vector<Copy> &transformPre;
  Type type;
  std::map<Type, Tensor> &copyWritten;
  const std::string &debugPrefix;

  Tensor paddingTensor;
};

// pad the specified kernel dimension by the specified amount. modifies the
// conv params in place to reflect this transformation.
static Tensor padKernelSpatialDim(Graph &graph, ConvParams &params,
                                  Tensor weights, const unsigned dim,
                                  const unsigned padToMultipleOf,
                                  Padder &padder) {
  // the weights are in the grouped internal shape so the spatial dimensions
  // begin at the third dimension (after G, Ci and Co).
  const auto tensorDim = dim + 3;

  const auto kernel = weights.dim(tensorDim);
  if (kernel % padToMultipleOf != 0) {
    const auto extraKernelPadding =
        (padToMultipleOf - kernel % padToMultipleOf);
    const auto extraInputPadding =
        extraKernelPadding * params.kernelTransform.dilation[dim];

    unsigned extraKernelPaddingLower = 0, extraKernelPaddingUpper = 0;
    auto &flippedExtraKernelPaddingUpper = params.kernelTransform.flip[dim]
                                               ? extraKernelPaddingLower
                                               : extraKernelPaddingUpper;

    auto &inputPaddingLower = params.inputTransform.paddingLower[dim];
    auto &inputPaddingUpper = params.inputTransform.paddingUpper[dim];
    auto &flippedInputPaddingUpper =
        params.inputTransform.flip[dim] ? inputPaddingLower : inputPaddingUpper;

    flippedExtraKernelPaddingUpper += extraKernelPadding;
    flippedInputPaddingUpper += extraInputPadding;

    weights = padder(graph, weights, extraKernelPaddingLower,
                     extraKernelPaddingUpper, tensorDim);
    params.kernelShape[dim] += extraKernelPadding;
  }

  return weights;
}

// Explicitly truncate, dilate and pad the specified spatial field of the
// input. modifies the conv params in place to reflect this transformation.
static Tensor truncateDilateAndPadInput(Graph &graph, ConvParams &params,
                                        Tensor in, const unsigned dim,
                                        Padder &padder,
                                        const std::string &debugPrefix) {
  // the input is in the grouped internal shape so the spatial dimensions
  // begin at the third dimension (after G, Ci and N).
  const auto tensorDim = dim + 3;

  const auto inputTruncationLower = params.inputTransform.truncationLower[dim];
  const auto inputTruncationUpper = params.inputTransform.truncationUpper[dim];
  in = pad(graph, in, -static_cast<int>(inputTruncationLower),
           -static_cast<int>(inputTruncationUpper), tensorDim);
  params.inputFieldShape[dim] -= inputTruncationLower + inputTruncationUpper;
  params.inputTransform.truncationLower[dim] = 0;
  params.inputTransform.truncationUpper[dim] = 0;

  const auto inputDilation = params.inputTransform.dilation[dim];
  in = dilate(graph, in, inputDilation, tensorDim, debugPrefix);
  params.inputTransform.dilation[dim] = 1;
  params.inputFieldShape[dim] =
      getDilatedSize(params.inputFieldShape[dim], inputDilation);

  const auto inputPaddingLower = params.inputTransform.paddingLower[dim];
  const auto inputPaddingUpper = params.inputTransform.paddingUpper[dim];
  in = padder(graph, in, inputPaddingLower, inputPaddingUpper, tensorDim);
  params.inputFieldShape[dim] += inputPaddingLower + inputPaddingUpper;
  params.inputTransform.paddingLower[dim] = 0;
  params.inputTransform.paddingUpper[dim] = 0;

  return in;
}

static void createConvPartialAmpVertices(
    Graph &graph, const Plan &plan, unsigned tile, ConvParams params,
    std::vector<Copy> &transformPre, std::map<Type, Tensor> &copyWritten,
    ComputeSet fwdCS, Tensor in, Tensor weights, Tensor out,
    bool use128BitConvUnitLoad, const std::string &debugPrefix) {
  assert(params == params.canonicalize());
  const auto &target = graph.getTarget();
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(in.elementType() == FLOAT);
  assert(weightsPerConvUnit % plan.inChansPerGroup == 0);
  const auto convUnitWeightHeight = weightsPerConvUnit / plan.inChansPerGroup;
  if (convUnitWeightHeight != 1) {
    assert(weights.elementType() == in.elementType());
    Padder padder(graph, tile, transformPre, copyWritten, weights.elementType(),
                  debugPrefix);

    // If we are doing an nx1 convolution we need to pad the weights to a
    // multiple of n.
    const auto kernelHeightDim = 0;
    weights = padKernelSpatialDim(graph, params, weights, kernelHeightDim,
                                  convUnitWeightHeight, padder);

    // Explicitly apply input transforms.
    in = truncateDilateAndPadInput(graph, params, in, 0, padder, debugPrefix);
  }

  const auto partialsType = out.elementType();
  auto isNonZero = [](unsigned x) { return x != 0; };
  bool nx1Vertex =
      product(params.kernelShape) != 1 ||
      params.inputTransform.dilation != params.outputTransform.stride ||
      std::any_of(params.outputTransform.paddingLower.begin(),
                  params.outputTransform.paddingLower.end(), isNonZero) ||
      std::any_of(params.outputTransform.paddingUpper.begin(),
                  params.outputTransform.paddingUpper.end(), isNonZero);
  bool useConvPartial1x1OutVertex = !nx1Vertex;
  const unsigned inChansPerGroup = plan.inChansPerGroup;

  auto inStrideX = params.outputTransform.stride.back();
  auto outStrideX = params.inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;

  const auto inRowStrideBeforeSplit = getInRowStride(
      params, product(params.inputFieldShape) / params.inputFieldShape[0],
      useConvPartial1x1OutVertex, convUnitWeightHeight);
  const auto convInputLoadElems =
      target.getConvUnitInputLoadElemsPerCycle(in.elementType() == FLOAT);
  int transformedInStrideBeforeSplit =
      (static_cast<int>(inStrideX) - 1 -
       static_cast<int>(convUnitWeightHeight - 1) * inRowStrideBeforeSplit) *
          static_cast<int>(inChansPerGroup / convInputLoadElems) +
      1;
  int transformedInRowStrideBeforeSplit =
      (inRowStrideBeforeSplit - 1) *
          static_cast<int>(inChansPerGroup / convInputLoadElems) +
      1;

  // Find field split that satisfies machine stride bit-widths
  // Only use input striding to decide to split field as it is most likely to
  // exceed machine strides
  const auto partition = splitConvIntoAmpVertices(
      params, target.getNumStrideBits(), transformedInStrideBeforeSplit,
      transformedInRowStrideBeforeSplit);

  iteratePartitionParallel(
      params, partition, [&](const ConvIndices &, const ConvSlice &slice) {
        // Get sub convolution
        Tensor subIn = unsplitActivationFromGroups(in);
        Tensor subWeights = unsplitWeightsFromGroups(weights);
        const auto subParams =
            getSubConvolution(slice, params, &subIn, &subWeights);
        subIn = splitActivationIntoGroups(subIn, plan.convGroupsPerGroup,
                                          plan.inChansPerGroup);
        subWeights = splitWeightsIntoGroups(subWeights, plan.convGroupsPerGroup,
                                            plan.inChansPerGroup,
                                            plan.partialChansPerGroup);
        Tensor subOut = sliceOutput(out, slice, plan.convGroupsPerGroup,
                                    plan.partialChansPerGroup);
        if (isZeroConvolution(subParams)) {
          zero(graph, subOut, tile, fwdCS);
        } else {
          createConvPartialAmpVertex(graph, plan, tile, subParams, fwdCS, subIn,
                                     subWeights, subOut, use128BitConvUnitLoad,
                                     debugPrefix);
        }
      });
}

static void createConvPartialHorizontalMacVertex(
    Graph &graph, const Plan &plan, unsigned tile, const ConvParams &params,
    ComputeSet fwdCS, const Tensor &in, const Tensor &weights,
    const Tensor &out, const std::string &debugPrefix) {
  const auto &target = graph.getTarget();
  const auto numFieldDims = params.getNumFieldDims();
  const auto xDimIndex = numFieldDims - 1;
  const unsigned numConvGroupGroups = out.dim(0);
  const unsigned numOutChanGroups = out.dim(1);
  const unsigned numInChanGroups = in.dim(1);
  const unsigned inChansPerGroup = plan.inChansPerGroup;
  const unsigned outChansPerGroup = plan.partialChansPerGroup;

  // MAC vertices only support having a single conv group per grouping.
  assert(plan.convGroupsPerGroup == 1);

  bool flipOut = params.inputTransform.flip[xDimIndex];

  if (plan.types.back().partialType == HALF) {
    assert(outChansPerGroup == 2);
  } else if (plan.types.back().partialType == FLOAT) {
    assert(outChansPerGroup == 1);
  }
  if (in.elementType() == HALF) {
    assert(inChansPerGroup % 2 == 0);
  }
  const auto outputFieldShape = params.getOutputFieldShape();
  const unsigned numOutFieldElems = product(outputFieldShape);
  if (numOutFieldElems == 0)
    return;

  std::vector<Tensor> outWindow;
  std::vector<Tensor> inWindow;
  std::vector<Tensor> weightsWindow;
  for (unsigned cg = 0; cg != numConvGroupGroups; ++cg) {
    // Output Tensor slices
    for (unsigned ozg = 0; ozg != numOutChanGroups; ++ozg) {
      auto o = out[cg][ozg].flatten();
      outWindow.push_back(o);
    }
    // Input tensor slices
    for (unsigned izg = 0; izg != numInChanGroups; ++izg) {
      auto i = in[cg][izg].flatten();
      inWindow.push_back(i);
    }
    // kernel tensor slices
    for (unsigned ozg = 0; ozg != numOutChanGroups; ++ozg) {
      for (unsigned izg = 0; izg != numInChanGroups; ++izg) {
        auto w = weights[cg][ozg][izg].flatten();
        weightsWindow.push_back(w);
      }
    }
  }

  auto inStrideX = params.outputTransform.stride.back();
  auto outStrideX = params.inputTransform.dilation.back();
  const auto strideDivisor = gcd(inStrideX, outStrideX);
  inStrideX /= strideDivisor;
  outStrideX /= strideDivisor;

  const unsigned numInFieldElems = product(params.inputFieldShape);
  const unsigned numKernelFieldElems = product(params.kernelShape);
  const unsigned kernelSizeX = params.kernelShape.back();
  const auto contextsPerVertex = target.getNumWorkerContexts();
  std::vector<std::vector<unsigned>> worklist(contextsPerVertex *
                                              numKernelFieldElems);
  for (unsigned k = 0; k != numKernelFieldElems / kernelSizeX; ++k) {
    // unflatten kernel index into a co-ordinate for the kernel
    auto kCoord = unflattenIndex(params.kernelShape, k * kernelSizeX);
    std::vector<unsigned> convOutBegin, convOutEnd;
    for (auto dim = 0U; dim + 1 != numFieldDims; ++dim) {
      unsigned begin, end;
      std::tie(begin, end) = getOutputRangeForKernelIndex(
          dim, {0, params.getOutputSize(dim)}, kCoord[dim], params);
      convOutBegin.push_back(begin);
      convOutEnd.push_back(end);
    }
    const auto convOutElems = getNumElementsInSlice(convOutBegin, convOutEnd);
    if (convOutElems == 0)
      continue;
    for (unsigned kx = 0; kx != params.kernelShape.back(); ++kx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) = getOutputRangeForKernelIndex(
          xDimIndex, {0, params.getOutputSize(xDimIndex)}, kx, params);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;

      auto outFieldBegin = convOutBegin;
      outFieldBegin.push_back(convOutXBegin);
      auto outFieldEnd = convOutEnd;
      outFieldEnd.push_back(convOutXEnd);
      auto workerPartition = partitionConvOutputBetweenWorkers(
          graph, 0, params.getBatchSize(), outFieldBegin, outFieldEnd, 0, 1, 0,
          1);
      for (unsigned i = 0; i != contextsPerVertex; ++i) {
        for (const auto &workerSlice : workerPartition[i]) {
          auto workerOutXBegin = workerSlice.outXBegin;
          auto workerOutXEnd = workerSlice.outXEnd;
          std::tie(workerOutXBegin, workerOutXEnd) =
              getOutputRangeForKernelIndex(
                  xDimIndex, {workerOutXBegin, workerOutXEnd}, kx, params);
          const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
          if (workerOutWidth == 0)
            continue;
          std::vector<std::size_t> workerIn;
          bool validRow = true;
          for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
            auto outIndex = workerSlice.outerFieldIndices[dim];
            auto inIndex = getInputIndex(dim, outIndex, kCoord[dim], params);
            if (inIndex == ~0U) {
              validRow = false;
              break;
            }
            workerIn.push_back(inIndex);
          }
          if (!validRow)
            continue;
          unsigned workerInXBegin, workerInXEnd;
          std::tie(workerInXBegin, workerInXEnd) = getInputRange(
              xDimIndex, {workerOutXBegin, workerOutXEnd}, kx, params);
          workerIn.push_back(workerInXBegin);

          auto workerOutFieldIndicesBegin =
              vectorConvert<std::size_t>(workerSlice.outerFieldIndices);
          workerOutFieldIndicesBegin.push_back(workerOutXBegin);
          const auto outBeginOffset =
              workerSlice.b * numOutFieldElems +
              flattenIndex(outputFieldShape, workerOutFieldIndicesBegin);

          const auto inBeginOffset =
              workerSlice.b * numInFieldElems +
              flattenIndex(params.inputFieldShape, workerIn);

          auto kIndex = k * kernelSizeX + kx;
          const auto numFieldElems =
              (workerOutWidth + outStrideX - 1) / outStrideX;

          const auto outOffset =
              flipOut ? outBeginOffset + workerOutWidth - 1 : outBeginOffset;

          worklist[kIndex * contextsPerVertex + i].push_back(outOffset);
          worklist[kIndex * contextsPerVertex + i].push_back(numFieldElems);
          worklist[kIndex * contextsPerVertex + i].push_back(inBeginOffset);
        }
      }
    }
  }

  int transformedOutStride = ((flipOut ? -static_cast<int>(outStrideX)
                                       : static_cast<int>(outStrideX)) -
                              1) *
                             outChansPerGroup;

  // Due to a fact that MAC codelet for half partials process 2 partials in one
  // loop iterration transformedOutStride need to be adjusted accordingly
  if (plan.types.back().partialType == poplar::HALF) {
    transformedOutStride /= 2;
  }

  const auto transformedInStride = inStrideX * inChansPerGroup;

  // Limits for field and worklist elements
  const auto unsignedMax = std::numeric_limits<unsigned short>::max();
  bool useLimitedVer = true;
  const auto zerosInfo = outWindow[0].numElements();

  const auto doubleWordWrites =
      zerosInfo / (8 / target.getTypeSize(outWindow[0].elementType()));
  const auto doubleWordWritesPerWorker =
      (doubleWordWrites + contextsPerVertex - 1) / contextsPerVertex;

  // check if field elements meet short representation
  if ((outChansPerGroup > unsignedMax) || (inChansPerGroup > unsignedMax) ||
      (numOutChanGroups - 1 > unsignedMax) || (numInChanGroups > unsignedMax) ||
      (numKernelFieldElems - 1 > unsignedMax) ||
      (numConvGroupGroups - 1 > unsignedMax) ||
      doubleWordWritesPerWorker > target.getRptCountMax())
    useLimitedVer = false;

  // check if all worklist items meet range constraints
  for (auto j = 0U; j != worklist.size() && useLimitedVer; ++j) {
    const auto &vec = worklist[j];
    for (auto entry : vec) {
      if (entry > unsignedMax) {
        useLimitedVer = false;
        break;
      }
    }
  }

  if (in.elementType() == HALF) {
    // Conv planner sets a grain size of 2 for input channels per group
    if (inChansPerGroup % 2)
      useLimitedVer = false;
    else {
      const auto maxRptCount =
          inChansPerGroup % 4 == 0 ? inChansPerGroup / 4 : inChansPerGroup / 2;
      if (maxRptCount > target.getRptCountMax())
        useLimitedVer = false;
    }
  } else if (in.elementType() == FLOAT) {
    const auto maxRptCount =
        inChansPerGroup % 2 == 0 ? inChansPerGroup / 2 : inChansPerGroup;
    if (maxRptCount > target.getRptCountMax())
      useLimitedVer = false;
  }

  const auto worklistEntryType = useLimitedVer ? UNSIGNED_SHORT : UNSIGNED_INT;
  auto v = graph.addVertex(
      fwdCS, templateVertex("poplin::ConvPartialHorizontalMac",
                            in.elementType(), plan.types.back().partialType,
                            useLimitedVer ? "true" : "false"));
  graph.connect(v["in"], inWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["weights"],
                reorderWeightsTensor(weightsWindow, numInChanGroups,
                                     numOutChanGroups, numConvGroupGroups));
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["numOutGroupsM1"], numOutChanGroups - 1);
  graph.setInitialValue(v["numInGroups"], numInChanGroups);
  graph.setInitialValue(v["kernelSizeM1"], numKernelFieldElems - 1);
  graph.setInitialValue(v["transformedInStride"], transformedInStride);
  graph.setInitialValue(v["transformedOutStride"], transformedOutStride);
  graph.setInitialValue(v["numConvGroupsM1"], numConvGroupGroups - 1);
  graph.setFieldSize(v["worklists"], worklist.size());
  for (unsigned i = 0; i < worklist.size(); ++i) {
    auto t = graph.addConstant(worklistEntryType, {worklist[i].size()},
                               worklist[i].data(), debugPrefix + "/worklist");
    graph.setTileMapping(t, 0);
    graph.connect(v["worklists"][i], t);
  }
  graph.setInitialValue(v["zerosInfo"], zerosInfo);
  graph.setTileMapping(v, tile);
}

// all shapes are in their grouped form:
//  in: [G1][CI1]...[G2][CI2]
//  out: [G1][CO1]...[G2][CO2]
//  weights: [G1][CO1][CI1]...[G2][CO2][CI2]
void createConvPartialSlicVertex(
    Graph &graph, unsigned slicWindowWidth, unsigned convGroupsPerGroup,
    unsigned chansPerGroup, unsigned convUnitsRequired, unsigned tile,
    ConvParams params, std::vector<Copy> &transformPre,
    std::map<Type, Tensor> &copyWritten, ComputeSet fwdCS,
    ConvProgramTree::PostProg &postConvProg, Tensor in, Tensor weights,
    Tensor out, const std::string &debugPrefix) {
  // We don't handle multiple input channel/output channel groups.
  assert(params.getNumInputChansPerConvGroup() == chansPerGroup &&
         params.getNumOutputChansPerConvGroup() == chansPerGroup);
  const auto outputStride = params.outputTransform.stride.back();

  const auto &target = graph.getTarget();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  const auto numFieldDims = params.getNumFieldDims();

  // TODO: Figure out how to specify this stuff in terms of
  // load elems per cycle etc. to deal with float/float and
  // half/half versions.
  assert(slicWindowWidth == 4u);

#ifndef NDEBUG
  const auto isAll = [](const auto k, const auto &c) {
    return std::all_of(std::begin(c), std::end(c),
                       [k](const auto x) { return x == k; });
  };
#endif
  assert(isAll(1u, params.kernelTransform.dilation));

  // TODO: unlike AMP, SLIC needs to apply field transforms before using them.
  // for now we just constrain against them in the planner.
  assert(isAll(false, params.inputTransform.flip));

  // We do not handle any kernel transforms at time of writing.
  assert(isAll(0u, params.kernelTransform.truncationLower));
  assert(isAll(0u, params.kernelTransform.truncationUpper));
  assert(isAll(1u, params.kernelTransform.dilation));
  assert(isAll(0u, params.kernelTransform.paddingLower));
  assert(isAll(0u, params.kernelTransform.paddingUpper));
  assert(isAll(false, params.kernelTransform.flip));

  // apply transformations (output padding is applied further down).
  {
    Padder padder(graph, tile, transformPre, copyWritten, weights.elementType(),
                  debugPrefix);

    // pad kernel width (aka the innermost dim) up to a multiple of 1xN if it is
    // not already.
    const auto kernelWidthDim = params.kernelShape.size() - 1;
    weights = padKernelSpatialDim(graph, params, weights, kernelWidthDim,
                                  slicWindowWidth, padder);

    // Explicitly apply input transforms.
    for (unsigned d = 0; d < numFieldDims; ++d) {
      in = truncateDilateAndPadInput(graph, params, in, d, padder, debugPrefix);
    }
  }

  const auto inType = in.elementType();
  const auto partialsType = out.elementType();
  assert(convGroupsPerGroup == 4u / chansPerGroup);

  // Indicates which of 3 modes we operate in:
  //
  //  =0 -> 4 conv groups, 1 input channel, 1 output channel
  //  =1 -> 2 conv groups, 2 input channels, 2 output channels
  //  =2 -> 1 conv group, 4 input channels, 4 output channels
  const unsigned char mode =
      convGroupsPerGroup == 4u ? 0 : convGroupsPerGroup == 2 ? 1 : 2;

  auto kernelGroups = params.kernelShape;
  assert(kernelGroups.back() % slicWindowWidth == 0);
  kernelGroups.back() /= slicWindowWidth;

  const auto outputSpatialShape = [&] {
    std::vector<unsigned> r;
    r.push_back(params.batchSize);
    for (unsigned d = 0; d < numFieldDims; ++d) {
      r.push_back(params.getUntransformedOutputSize(d));
    }
    return r;
  }();

  const auto paddedOutputSpatialShape = [&] {
    std::vector<unsigned> r;
    r.push_back(params.batchSize);
    for (unsigned d = 0; d < numFieldDims; ++d) {
      r.push_back(params.getOutputSize(d));
    }
    return r;
  }();

  const auto inputSpatialShape = [&] {
    auto r = vectorConvert<unsigned>(params.inputFieldShape);
    r.insert(r.begin(), params.batchSize);
    return r;
  }();

  const auto numSubKernels = product(kernelGroups);
  const auto numConvGroupGroups = out.dim(0);

  bool useShortTypes = true;
  const auto shortTypesVertexClass = templateVertex(
      "poplin::ConvPartial1x4SLIC", inType, partialsType, outputStride,
      /* useShortTypes */ true, convUnitsRequired);
  std::vector<std::vector<unsigned short>> worklists(numWorkerContexts *
                                                     numSubKernels);
  const auto slicWindowHeight = 1u;
  auto partitions = createPartitions(params, slicWindowHeight, slicWindowWidth,
                                     numWorkerContexts);

  std::vector<std::size_t> inputBatchAndFieldShape = {params.getBatchSize()};
  std::vector<std::size_t> outputBatchAndFieldShape = {params.getBatchSize()};
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    inputBatchAndFieldShape.push_back(params.inputFieldShape[dim]);
    outputBatchAndFieldShape.push_back(params.getOutputSize(dim));
  }

  // create worklist now that dimensions of all splits are known
  for (const auto &p : partitions) {
    if (p.inBeginIndices.size() == 0) {
      continue;
    }
    const auto outBeginOffset =
        flattenIndex(outputBatchAndFieldShape, p.outBeginIndices);
    const auto inBeginOffset =
        flattenIndex(inputBatchAndFieldShape, p.inBeginIndices);
    const auto wIndex = p.subKernelPosition * numWorkerContexts + p.context;
    worklists[wIndex].push_back(inBeginOffset);
    worklists[wIndex].push_back(outBeginOffset);
    worklists[wIndex].push_back(p.outXWidth);
  }
  // Determine whether or not we can use the assembly implementation
  // with short types.
  if (numSubKernels - 1 >
      graph.getMaxVertexFieldValue(shortTypesVertexClass, "numSubKernelsM1")) {
    useShortTypes = false;
  }
  if (numConvGroupGroups - 1 >
      graph.getMaxVertexFieldValue(shortTypesVertexClass,
                                   "numConvGroupGroupsM1")) {
    useShortTypes = false;
  }

  std::vector<Tensor> inWindow, weightsWindow, outWindow;
  for (unsigned cg = 0; cg < numConvGroupGroups; ++cg) {
    for (unsigned kg = 0; kg < numSubKernels; ++kg) {
      auto kernelStart = unflattenIndex(kernelGroups, kg);
      auto kernelEnd = kernelStart;
      for (auto &s : kernelEnd) {
        s += 1;
      }

      kernelStart.back() *= slicWindowWidth;
      kernelEnd.back() *= slicWindowWidth;

      const auto window =
          weights[cg][0][0].slice(kernelStart, kernelEnd).flatten();
      weightsWindow.push_back(window);
    }
  }

  for (unsigned cg = 0; cg < numConvGroupGroups; ++cg) {
    outWindow.push_back(out[cg][0].flatten());
    inWindow.push_back(in[cg][0].flatten());
  }

  // explicitly apply output padding. TODO: this is not modelled in the planner
  {
    // dims before the spatial dims are G1, OC1 and B.
    unsigned spatialDimOffset = 3;

    std::vector<Tensor> elemsToPad;
    auto outWithPadding = out;
    const auto &ot = params.outputTransform;
    for (unsigned d = 0; d < numFieldDims; ++d) {
      const unsigned spatialDim = spatialDimOffset + d;

      const auto &shape = outWithPadding.shape();
      const unsigned N = shape[spatialDim];

      auto lower = outWithPadding.slice(0, ot.paddingLower[d], spatialDim);
      auto upper = outWithPadding.slice(N - ot.paddingUpper[d], N, spatialDim);

      elemsToPad.push_back(lower.flatten());
      elemsToPad.push_back(upper.flatten());

      // prune the padding off of the tensor so that we don't repad elements
      // when we pad the next dimension.
      outWithPadding = outWithPadding.slice(ot.paddingLower[d],
                                            N - ot.paddingUpper[d], spatialDim);
    }
    const auto outputPadding = concat(elemsToPad).flatten();
    const auto zeros =
        graph.addConstant(partialsType, outputPadding.shape(), 0);
    graph.setTileMapping(zeros, 0);

    // Collect copies to do a single one at the end.
    auto &copyPair = postConvProg[partialsType];
    copyPair.first.push_back(zeros.flatten());
    copyPair.second.push_back(outputPadding.flatten());
  }

  // We also need an extra buffer for our vertex, 16-byte aligned, with size
  // equal the number of output elements per conv group group, plus 8 bytes to
  // enforce (&out[i][0] - &outFieldBuffer[0]) % 16 == 8
  // (or == 0 for the case where output == HALF and stride = 2) so that we
  // can use ld2xst64pace in the codelet even when out and outFieldBuffer
  // reside in the same bank.
  //
  // Additionally, we need 192 bytes (maximum), to store rearranged
  // weights, plus 4 bytes to store a pointer.
  // This isn't actually true for the mode which doesn't
  // use the weight storage (1cgx4ocx4ic) but for now we'll keep it
  // simple and uniform.
  constexpr unsigned extraBytes = 200u;
  const auto bytesForAlignedBufferOffset =
      (out.elementType() == HALF && params.outputTransform.stride.back() == 2)
          ? 8
          : 0;
  assert(extraBytes % 16 == 8);
  assert((extraBytes + bytesForAlignedBufferOffset) %
             target.getTypeSize(partialsType) ==
         0);

  const auto extraOutputElems = (extraBytes + bytesForAlignedBufferOffset) /
                                target.getTypeSize(partialsType);
  const auto numFieldElemsIncludingPadding = product(paddedOutputSpatialShape);
  const auto outFieldBuffer = graph.addVariable(
      partialsType,
      {extraOutputElems +
       numFieldElemsIncludingPadding * convGroupsPerGroup * chansPerGroup},
      "outFieldBuffer");
  graph.setTileMapping(outFieldBuffer, tile);

  const auto vertexClass =
      templateVertex("poplin::ConvPartial1x4SLIC", inType, partialsType,
                     outputStride, useShortTypes, convUnitsRequired);
  auto v = graph.addVertex(fwdCS, vertexClass);
  graph.setTileMapping(v, tile);

  graph.connect(v["in"], inWindow);
  graph.connect(v["weights"], weightsWindow);
  graph.connect(v["out"], outWindow);
  graph.connect(v["outFieldBuffer"], outFieldBuffer);
  graph.setFieldSize(v["worklists"], worklists.size());

  for (unsigned i = 0; i < worklists.size(); ++i) {
    const auto t = graph.addConstant(UNSIGNED_SHORT, {worklists[i].size()},
                                     worklists[i].data(), "worklists");
    graph.setTileMapping(t, 0);
    graph.connect(v["worklists"][i], t);
  }
  graph.setInitialValue(v["mode"], mode);
  graph.setInitialValue(v["outPtrLoadOffset"], (numSubKernels % 2) ? 0 : 4);
  graph.setInitialValue(v["numSubKernelsM1"], numSubKernels - 1);
  graph.setInitialValue(v["numConvGroupGroupsM1"], numConvGroupGroups - 1);
}

void ConvProgramTree::ComputeSetsGroup::lower(Sequence &prog) {
  if (pre) {
    prog.add(Execute(pre.get()));
  }
  prog.add(Execute(convolveCS));

  for (auto &p : postProg) {
    logging::debug("#convolution post program bunch copies of type {} = {}",
                   p.first.toString(), p.second.first.size());
    assert(p.second.first.size());
    prog.add(Copy(concat(p.second.first), concat(p.second.second)));
  }
  if (post) {
    prog.add(Execute(post.get()));
  }
}

static void createOuterProductVertex(Graph &graph, unsigned tile,
                                     unsigned xBegin, unsigned xEnd,
                                     const ConvParams &params,
                                     ConvProgramTree::ComputeSetsGroup &fwdCS,
                                     Tensor in, Tensor weights,
                                     const Tensor &out,
                                     const std::string &debugPrefix) {
  const auto numFieldDims = params.getNumFieldDims();
  assert(product(params.outputTransform.stride) == 1);
  assert(product(params.inputTransform.dilation) == 1);

  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    in = pad(graph, in,
             -static_cast<int>(params.inputTransform.truncationLower[dim]),
             -static_cast<int>(params.inputTransform.truncationUpper[dim]),
             3 + dim);
    in = pad(
        graph, in, static_cast<int>(params.inputTransform.paddingLower[dim]),
        static_cast<int>(params.inputTransform.paddingUpper[dim]), 3 + dim);
    weights =
        pad(graph, weights,
            -static_cast<int>(params.kernelTransform.truncationLower[dim]),
            -static_cast<int>(params.kernelTransform.truncationUpper[dim]),
            3 + dim);
    weights = pad(graph, weights,
                  static_cast<int>(params.kernelTransform.paddingLower[dim]),
                  static_cast<int>(params.kernelTransform.paddingUpper[dim]),
                  3 + dim);
  }

  // outer product vertices only support a single conv group group...
  assert(in.dim(1) == 1);
  // ...a single batch...
  assert(in.dim(2) == 1);
  // outer product only supports a grouping conv groups into single groups.
  assert(in.dim(in.rank() - 2) == 1);

  // check all input field dimensions other than the innermost is 1
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    assert(in.dim(dim + 3) == 1);
  }
  assert(in.dim(in.rank() - 1) == 1);

  // check every field dimension of the weights tensor is 1
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    assert(weights.dim(dim + 3) == 1);
  }

  assert(weights.dim(2) == 1);
  assert(weights.dim(weights.rank() - 1) == 1);
  assert(out.dim(1) == weights.dim(1));
  assert(out.dim(2) == 1);
  assert(out.dim(out.rank() - 2) == 1);

  // check all output field dimensions other than the innermost is 1
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    assert(out.dim(dim + 3) == 1);
  }
  assert(out.dim(3 + numFieldDims - 1) == in.dim(3 + numFieldDims - 1));
  assert(out.dim(out.rank() - 1) == weights.dim(weights.rank() - 2));
  const auto chansPerGroup = weights.dim(weights.rank() - 2);
  const auto dType = in.elementType();

  const auto numConvGroups = params.getNumConvGroups();
  for (unsigned cg = 0; cg != numConvGroups; ++cg) {
    auto inWindow = in[cg].flatten().slice(xBegin, xEnd);
    auto outWindow =
        out.slice(cg, cg + 1, 0)
            .slice(xBegin, xEnd, out.rank() - 3)
            .reshape({out.dim(1), (xEnd - xBegin) * chansPerGroup});
    auto weightsWindow = weights[cg].flatten();

    // This does some casting here instead of using the convtypes in the plan
    // as this could change the type of other passes see T14149
    if (dType == HALF && out.elementType() == FLOAT) {
      if (!fwdCS.pre) {
        fwdCS.pre = graph.addComputeSet(debugPrefix + "/PreOuterProductCast");
      }
      inWindow = cast(graph, inWindow, FLOAT, fwdCS.pre.get());
      weightsWindow = cast(graph, weightsWindow, FLOAT, fwdCS.pre.get());
    }
    const auto outerProductType = (dType == out.elementType()) ? dType : FLOAT;
    auto v = graph.addVertex(
        fwdCS.convolveCS,
        templateVertex("poplin::OuterProduct", outerProductType),
        {{"in", inWindow}, {"weights", weightsWindow}, {"out", outWindow}});

    graph.setInitialValue(v["chansPerGroup"],
                          weightsWindow.numElements() / outWindow.dim(0));

    graph.setTileMapping(v, tile);

    if (dType == FLOAT && out.elementType() == HALF) {
      if (!fwdCS.post) {
        fwdCS.post = graph.addComputeSet(debugPrefix + "/PostOuterProductCast");
      }
      outWindow = cast(graph, outWindow, HALF, fwdCS.post.get());
    }
  }
}

static void calcPartialConvOutput(
    Graph &graph, const Plan &plan, unsigned tile, ConvParams params,
    std::vector<Copy> &transformPre, std::map<Type, Tensor> &copyWritten,
    ConvProgramTree::ComputeSetsGroup &convolveCS, Tensor in, Tensor weights,
    Tensor out, bool use128BitConvUnitLoad, const std::string &debugPrefix) {
  assert(params.getNumConvGroups() % plan.convGroupsPerGroup == 0);
  assert(params.getNumOutputChansPerConvGroup() % plan.partialChansPerGroup ==
         0);
  assert(params.getNumInputChansPerConvGroup() % plan.inChansPerGroup == 0);

  graph.setTileMapping(out, tile);
  in = splitActivationIntoGroups(in, plan.convGroupsPerGroup,
                                 plan.inChansPerGroup);
  weights =
      splitWeightsIntoGroups(weights, plan.convGroupsPerGroup,
                             plan.inChansPerGroup, plan.partialChansPerGroup);
  if (isZeroConvolution(params)) {
    zero(graph, out, tile, convolveCS.convolveCS);
    return;
  }
  switch (plan.method) {
  case Plan::Method::AMP:
    createConvPartialAmpVertices(graph, plan, tile, params, transformPre,
                                 copyWritten, convolveCS.convolveCS, in,
                                 weights, out, use128BitConvUnitLoad,
                                 debugPrefix);
    break;
  case Plan::Method::MAC:
    createConvPartialHorizontalMacVertex(graph, plan, tile, params,
                                         convolveCS.convolveCS, in, weights,
                                         out, debugPrefix);
    break;
  case Plan::Method::SLIC:
    assert(plan.inChansPerGroup == plan.partialChansPerGroup);
    createConvPartialSlicVertex(
        graph, plan.slicWindowWidth, plan.convGroupsPerGroup,
        plan.partialChansPerGroup, plan.numConvUnitsRequired, tile, params,
        transformPre, copyWritten, convolveCS.convolveCS, convolveCS.postProg,
        in, weights, out, debugPrefix);
    break;
  case Plan::Method::OUTER_PRODUCT: {
    const auto &target = graph.getTarget();
    const auto outputLength =
        params.getOutputSize(params.getNumFieldDims() - 1);
    const auto perWorkerRegions =
        splitRegionsBetweenWorkers(target, {{0, outputLength}}, 1);
    for (const auto &entry : perWorkerRegions) {
      assert(entry.size() == 1);
      createOuterProductVertex(graph, tile, entry[0].begin(), entry[0].end(),
                               params, convolveCS, in, weights, out,
                               debugPrefix);
    }
  } break;
  default: {
    std::stringstream ss;
    ss << "Unexpected convolution method <" << plan.method << ">";
    throw poputil::poplibs_error(ss.str());
  }
  }
}

static bool inputRearrangementIsExpensive(const ConvOptions &options) {
  // During the weight update pass we change the innermost dimension when the
  // activations / deltas are rearranged.
  return options.pass == Pass::TRAINING_WU ||
         options.pass == Pass::FC_TRAINING_WU;
}

static bool weightRearrangementIsExpensive(const ConvOptions &options) {
  // During the weight update pass we change the innermost dimension when the
  // activations / deltas are rearranged.
  return options.pass == Pass::TRAINING_WU ||
         options.pass == Pass::FC_TRAINING_WU;
}

static unsigned getPartialIndex(const ConvIndices &indices,
                                const Partition &partition) {
  const auto numFieldDims = indices.kernel.size();
  unsigned partialIndex = indices.ic;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    partialIndex =
        partialIndex * partition.kernelSplit[dim] + indices.kernel[dim];
  }
  return partialIndex;
}

static unsigned getOutputIndex(const ConvIndices &indices,
                               const Partition &partition) {
  assert(indices.cg < partition.convGroupSplit &&
         indices.b < partition.batchSplit &&
         indices.oc < partition.outChanSplit.parallel);
  unsigned outputIndex = indices.cg;
  outputIndex *= partition.batchSplit;
  outputIndex += indices.b;
  const auto numFieldDims = indices.out.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    assert(indices.out[dim] < partition.fieldSplit[dim]);
    outputIndex *= partition.fieldSplit[dim];
    outputIndex += indices.out[dim];
  }
  outputIndex *= partition.outChanSplit.parallel;
  outputIndex += indices.oc;
  return outputIndex;
}

static std::vector<unsigned> getOutputDimSplits(const Partition &partition) {
  std::vector<unsigned> splits = {partition.convGroupSplit,
                                  partition.batchSplit};
  splits.insert(splits.end(), partition.fieldSplit.begin(),
                partition.fieldSplit.end());
  splits.push_back(partition.outChanSplit.parallel);
  return splits;
}

/// Stitch each run of \a dimSplit partial result tensors together by
/// concatenating them in the specified dimension to form a new
/// list of results in-place in \a results.
static void stitchResultsImpl(std::vector<Tensor> &results, unsigned dim,
                              unsigned dimSplit) {
  if (dimSplit == 1)
    return;
  std::vector<Tensor> stitched;
  assert(results.size() % dimSplit == 0);
  stitched.reserve(results.size() / dimSplit);
  for (auto it = results.begin(), end = results.end(); it != end;
       it += dimSplit) {
    std::vector<Tensor> slice(it, it + dimSplit);
    stitched.push_back(concat(slice, dim));
  }
  std::swap(stitched, results);
}

static Tensor stitchResults(std::vector<Tensor> results,
                            std::vector<unsigned> dimSplits) {
  const auto numDims = dimSplits.size();
  for (int dim = numDims - 1; dim >= 0; --dim) {
    stitchResultsImpl(results, dim, dimSplits[dim]);
  }
  assert(results.size() == 1);
  return results.front();
}

/// Stitch together a number of partial result tensors to form a single tensor.
/// The 1st dimension of \a results represents the dimension to reduce over and
/// the 2nd dimension is a list of results that should be stitched together in
/// the output axes. The list of results is lexigraphically ordered by the
/// indices the partition associated with the output in the order the axes
/// have in the output tensor.
static Tensor stitchPartialResults(
    const std::vector<std::vector<boost::optional<Tensor>>> &results,
    const Partition &partition) {
  std::vector<Tensor> partials;
  partials.reserve(results.size());
  auto dimSplits = getOutputDimSplits(partition);
  for (const auto &entry : results) {
    std::vector<Tensor> r(entry.size());
    for (unsigned i = 0; i < entry.size(); ++i) {
      assert(entry[i]);
      r[i] = *entry[i];
    }
    partials.push_back(stitchResults(r, dimSplits));
    partials.back() = partials.back().expand({0});
  }
  return concat(partials, 0);
}

static std::vector<std::size_t>
getPartialOutputShape(const ConvParams &params,
                      const unsigned convGroupsPerGroup,
                      const unsigned outChansPerGroup) {
  auto numConvGroups = params.getNumConvGroups();
  auto numOutChans = params.getNumOutputChansPerConvGroup();
  std::vector<std::size_t> outShape = {numConvGroups / convGroupsPerGroup,
                                       numOutChans / outChansPerGroup,
                                       params.getBatchSize()};
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outShape.push_back(params.getOutputSize(dim));
  }
  outShape.push_back(convGroupsPerGroup);
  outShape.push_back(outChansPerGroup);
  return outShape;
}

static std::size_t getSerialSliceIndex(const Partition &partition,
                                       const ConvIndices &serialIndices,
                                       bool isActs) {
  // We only handle output channel splits currently.
  assert(partition.totalSerialSplit() ==
         partition.inChanSplit.serial * partition.outChanSplit.serial);
  assert((partition.inChanSplit.serial == 1) ||
         (partition.outChanSplit.serial == 1));
  if (isActs) {
    return serialIndices.ic;
  } else if (partition.inChanSplit.serial > 1) {
    return serialIndices.ic;
  }
  return serialIndices.oc;
}

static Tensor stitchSerialSlices(const std::vector<Tensor> &slices, bool isActs,
                                 const Partition &partition) {
  // We only handle output channel splits currently.
  assert(!slices.empty());
  assert(partition.totalSerialSplit() ==
         partition.inChanSplit.serial * partition.outChanSplit.serial);
  assert((partition.inChanSplit.serial == 1) ||
         (partition.outChanSplit.serial == 1));
  if (isActs) {
    return concat(slices, slices[0].rank() - 1);
  } else if (partition.inChanSplit.serial > 1) {
    return concat(slices, slices[0].rank() - 1);
  }
  return concat(slices, slices[0].rank() - 2);
}

static std::tuple<CanonicalConvParams, ConvIndices>
preprocessForSerialSlice(Tensor *input, Tensor *weights,
                         const CanonicalConvParams &params,
                         const Partition &partition) {
  const unsigned numInputSlices = partition.inChanSplit.serial;
  const unsigned numWeightsSlices =
      partition.inChanSplit.serial * partition.outChanSplit.serial;
  assert(partition.inChanSplit.serial * partition.outChanSplit.serial ==
         partition.totalSerialSplit());
  assert(partition.totalSerialSplit() > 1);
  std::vector<Tensor> inputSlices;
  std::vector<Tensor> weightsSlices;
  if (input) {
    assert(input->dim(input->rank() - 1) % partition.inChanSplit.serial == 0);
    inputSlices.resize(numInputSlices);
  }
  if (weights) {
    // Check that a serial input channel split precisely divides the
    // input channels available at this point.
    assert(weights->dim(weights->rank() - 1) % partition.inChanSplit.serial ==
           0);
    // Check that a serial output channel split precisely divides the
    // output channels available at this point.
    assert(weights->dim(weights->rank() - 2) % partition.outChanSplit.serial ==
           0);
    weightsSlices.resize(numWeightsSlices);
  }
  auto parallelParams = params;
  ConvIndices indices;
  bool firstSerialSlice = true;
  iteratePartitionSerial(
      params, partition,
      [&](const ConvIndices &serialIndices, const ConvSlice &slice) {
        Tensor subInput = input ? *input : Tensor();
        auto subWeights = weights ? *weights : Tensor();
        const auto subParams =
            getSubConvolution(slice, params, input ? &subInput : nullptr,
                              weights ? &subWeights : nullptr);

        auto inputIndex = getSerialSliceIndex(partition, serialIndices, true);
        auto weightsIndex =
            getSerialSliceIndex(partition, serialIndices, false);
        if (input) {
          inputSlices[inputIndex] = subInput;
        }
        if (weights) {
          weightsSlices[weightsIndex] = subWeights;
        }

        if (firstSerialSlice) {
          indices = serialIndices;
          parallelParams = subParams;
          firstSerialSlice = false;
        } else {
          assert(subParams == parallelParams);
        }
      });

  auto refactorSplitDimToOutermost = [](Tensor t, unsigned dim,
                                        unsigned factor) {
    const auto chans = t.dim(dim);
    assert(chans % factor == 0);
    const auto chansPerSlice = chans / factor;
    return t.reshapePartial(dim, dim + 1, {factor, chansPerSlice})
        .dimRoll(dim, 0);
  };

  if (input) {
    *input = stitchSerialSlices(inputSlices, true /* isActs */, partition);
    if (partition.inChanSplit.serial > 1) {
      *input = refactorSplitDimToOutermost(*input, input->rank() - 1,
                                           partition.inChanSplit.serial);
    }
  }
  if (weights) {
    if (partition.inChanSplit.serial > 1) {
      *weights =
          stitchSerialSlices(weightsSlices, false /* !isActs */, partition);
      *weights = refactorSplitDimToOutermost(*weights, weights->rank() - 1,
                                             partition.inChanSplit.serial);
    }
    if (partition.outChanSplit.serial > 1) {
      *weights =
          stitchSerialSlices(weightsSlices, false /* !isActs */, partition);
      *weights = refactorSplitDimToOutermost(*weights, weights->rank() - 2,
                                             partition.outChanSplit.serial);
    }
  }
  return std::make_tuple(parallelParams, indices);
}

static void add(Sequence &prog, const std::vector<Copy> &copies) {
  for (const auto &copy : copies) {
    prog.add(copy);
  }
}

ConvProgramTree::TransformPreProgram::TransformPreProgram(
    Graph &graph, const std::string &debugPrefix)
    : transposeCS(graph.addComputeSet(debugPrefix + "/Transpose")) {}

void ConvProgramTree::TransformPreProgram::lower(
    poplar::program::Sequence &prog) {
  // TODO: make a map of type and concat same types together.
  for (const auto &t : writeUndef) {
    prog.add(WriteUndef(t));
  }

  add(prog, preTranspose);
  prog.add(Execute(transposeCS));
  add(prog, postTranspose);
}

ConvProgramTree::TransformPostSerialProgram::TransformPostSerialProgram(
    Graph &graph, const std::string &debugPrefix)
    : castCS(graph.addComputeSet(debugPrefix + "/CastSerialOut")) {}

void ConvProgramTree::TransformPostSerialProgram::lower(
    poplar::program::Sequence &prog) {
  prog.add(Execute(castCS));
  add(prog, copies);
}

ConvProgramTree::ConvProgramTree(Graph &graph, const Plan &plan,
                                 const std::string &debugPrefix)
    : transformPre(), transformPost(plan.numLevels()),
      transformPreSerial(graph, debugPrefix),
      transformPostSerial(graph, debugPrefix),
      loopCount(plan.totalSerialSplit()),
      convolveCSGroup(graph.addComputeSet(debugPrefix + "/Convolve")),
      reduceComputeSets(plan.numLevels() - 1) {
  for (unsigned i = 0; i < plan.numLevels(); ++i) {
    transformPre.emplace_back(graph, debugPrefix);
  }
}

void ConvProgramTree::lower(Sequence &prog) {
  for (const auto &c : copyWritten) {
    prog.add(WriteUndef(c.second));
  }

  prog.add(initProg);

  assert(transformPre.size() == transformPost.size());
  const unsigned numLevels = transformPre.size();
  assert(numLevels > 1);
  assert(numLevels - 1 == reduceComputeSets.size());

  Sequence body;

  // lower the transforms in ascending order as we climb the hierarchy.
  for (unsigned level = 0; level < numLevels; ++level) {
    transformPre[level].lower(body);
  }

  convolveCSGroup.lower(body);
  // lower the remaining reductions and inverse transforms in reverse order
  // as we descend the hierarchy.
  add(body, transformPost.back());
  for (int level = numLevels - 2; level >= 0; --level) {
    // there is a reduction for each level in the partition hierarchy, which is
    // one less than the transforms. therefore there is no reduction in the
    // outermost level which is why we start at numLevels - 2 and add the
    // innermost transform outside of this loop.
    for (const auto &reduceCS : reduceComputeSets[level]) {
      body.add(Execute(reduceCS));
    }

    add(body, transformPost[level]);
  }

  transformPreSerial.lower(prog);
  if (loopCount == 1) {
    prog.add(body);
  } else {
    assert(loopCount != 0);
    prog.add(Repeat(loopCount, Sequence(slice, body, update, loopPost)));
  }
  transformPostSerial.lower(prog);

  prog.add(finalizeProg);
}

static boost::optional<Tensor>
convolutionImpl(Graph &graph, const CanonicalConvParams &originalParams,
                Plan plan, unsigned level, Tensor in, Tensor weights,
                ConvProgramTree &cpt,
                const std::vector<Split<ConvIndices>> &indices, Tensor partials,
                unsigned createPartialsLevel, const std::string &debugPrefix,
                const ConvOptions &options) {
  // Slice.
  Tensor loopCounter;
  Tensor inSlice = in;
  Tensor weightsSlice = weights;
  auto serialParams = originalParams;

  const auto originalTransform = plan.transforms[level];
  serialParams =
      convolutionPreprocess(graph, serialParams.releaseParams(), options, plan,
                            level, indices, inSlice, weightsSlice, true);

  const auto ipuLevel = plan.transforms.size() - 2;

  auto levelIndices = indices;
  levelIndices.emplace_back();
  auto parallelParams = serialParams;
  const std::string levelSuffix = "[" + std::to_string(level) + "]";
  // we only support serial splits on the ipu level.
  if (level == ipuLevel) {
    const auto &partition = plan.partitions[level];

    if (partition.totalSerialSplit() > 1) {
      std::tie(parallelParams, levelIndices.back().serial) =
          preprocessForSerialSlice(&inSlice, &weightsSlice, serialParams,
                                   partition);
      // We check if the given tensor is such that a slice will cause exchange
      // and if so, rearrange before the loop rather than after as this will be
      // very expensive. this happens as a pre transform of the previous level
      // in the hierarchy.
      auto rearrangeIfSplitOverTiles = [&](Tensor &slice, bool isActs) {
        auto sliceKind = isActs ? "input" : "weights";
        logging::debug("'{}': forcing rearrangement of {} before slice "
                       "because sliced dimension is split over tiles",
                       debugPrefix, sliceKind);

        auto createSliceMethod = isActs ? createInputImpl : createWeightsImpl;
        auto sliceRearranged = createSliceMethod(
            graph, serialParams, level, true, indices,
            debugPrefix + "/" + sliceKind + "Rearranged", plan, options);

        auto inSliceRearranged = isActs ? &sliceRearranged : nullptr;
        auto weightsSliceRearranged = isActs ? nullptr : &sliceRearranged;
        preprocessForSerialSlice(inSliceRearranged, weightsSliceRearranged,
                                 serialParams, partition);

        slice = popops::rearrange::regroupIfBeneficial(
            graph, slice, sliceRearranged, cpt.transformPreSerial.preTranspose,
            cpt.transformPreSerial.transposeCS,
            debugPrefix + "/" + sliceKind + "RegroupBeforeSlice");
        cpt.transformPreSerial.postTranspose.emplace_back(slice,
                                                          sliceRearranged);

        return sliceRearranged;
      };

      if ((partition.inChanSplit.serial > 1) &&
          dimIsSplitOverTiles(graph, inSlice, 0)) {
        inSlice = rearrangeIfSplitOverTiles(inSlice, true);
      }

      if (((partition.inChanSplit.serial > 1) ||
           (partition.outChanSplit.serial > 1)) &&
          dimIsSplitOverTiles(graph, weightsSlice, 0)) {
        weightsSlice = rearrangeIfSplitOverTiles(weightsSlice, false);
      }

      // create and zero initialise loop counter.
      loopCounter = graph.addVariable(
          UNSIGNED_INT, {1}, debugPrefix + "/loopCounter" + levelSuffix);
      graph.setTileMapping(loopCounter, 0);
      const auto zeroConstant = graph.addConstant(
          UNSIGNED_INT, {1}, 0, debugPrefix + "/zero" + levelSuffix);
      graph.setTileMapping(zeroConstant, 0);
      cpt.transformPreSerial.postTranspose.emplace_back(zeroConstant,
                                                        loopCounter);

      // per iteration slices of input.
      if (partition.inChanSplit.serial > 1) {
        inSlice = dynamicSlice(graph, inSlice, loopCounter, {0}, {1}, cpt.slice,
                               debugPrefix + "/inputSerialSlice" + levelSuffix)
                      .squeeze({0});
        weightsSlice =
            dynamicSlice(graph, weightsSlice, loopCounter, {0}, {1}, cpt.slice,
                         debugPrefix + "/weightsSerialSlice" + levelSuffix)
                .squeeze({0});
      }
      if (partition.outChanSplit.serial > 1) {
        weightsSlice =
            dynamicSlice(graph, weightsSlice, loopCounter, {0}, {1}, cpt.slice,
                         debugPrefix + "/weightsSerialSlice" + levelSuffix)
                .squeeze({0});
      }
    }
  }

  const auto preTransformParams = parallelParams;

  // Transform.
  bool rearrangeActs = false;
  bool rearrangeWeights = false;
  if (level == ipuLevel) {
    // If the input tensors have a different memory layout to the one expected
    // by the vertices poplar will rearrange the data using exchange code or
    // copy pointers. If the data is broadcast this rearrangement happens on
    // every tile that receives the data. We can reduce the amount of exchange
    // code / number of copy pointers required by rearranging the data once and
    // broadcasting the rearranged data. This trades increased execution time
    // for reduced memory usage. The biggest reductions in memory usage come
    // when data is broadcast to many tiles. inViewMaxBroadcastDests and
    // weightViewMaxBroadcastDests specify the maximum number of broadcast
    // destinations a tensor can have before we insert a copy to rearrange it.
    // Note these copies will be elided if the inputs already use the expected
    // memory layout and tile mapping.
    const auto inViewMaxBroadcastDests = 7U;
    const auto weightViewMaxBroadcastDests = 7U;
    const auto inNumDests =
        std::accumulate(plan.partitions.back().kernelSplit.begin(),
                        plan.partitions.back().kernelSplit.end(), 1U,
                        std::multiplies<unsigned>()) *
        plan.partitions.back().outChanSplit.parallel;
    auto weightsNumDests = plan.partitions.back().batchSplit;
    for (const auto split : plan.partitions.back().fieldSplit) {
      weightsNumDests *= split;
    }
    rearrangeActs = inputRearrangementIsExpensive(options) ||
                    (inNumDests > inViewMaxBroadcastDests) ||
                    !plan.transforms[ipuLevel].expandDims.empty() ||
                    !plan.transforms[ipuLevel].outChanFlattenDims.empty();
    rearrangeWeights = weightRearrangementIsExpensive(options) ||
                       (weightsNumDests > weightViewMaxBroadcastDests) ||
                       !plan.transforms[ipuLevel].expandDims.empty() ||
                       !plan.transforms[ipuLevel].outChanFlattenDims.empty();
    // Check if the input/weights respect the desired grainSize at this level
    // in the correct dimension. If not we should probably rearrange prior to
    // the exchange.
    auto expectedGrouping = determinePreprocessedGroupingFromPlan(
        parallelParams.getParams(), plan, level);
    const auto &innermostGrouping = expectedGrouping.at(0);
    if (innermostGrouping.second > 1) {
      rearrangeActs |= (innermostGrouping.first != inSlice.rank() - 1);
      rearrangeWeights |= (innermostGrouping.first != weightsSlice.rank() - 1);
      if (!rearrangeActs) {
        auto actualGrouping = detectInnermostGrouping(graph, inSlice);
        rearrangeActs |= (actualGrouping == 1);
      }
      if (!rearrangeWeights) {
        auto actualGrouping = detectInnermostGrouping(graph, weightsSlice);
        rearrangeWeights |= (actualGrouping == 1);
      }
    }
  }
  parallelParams = convolutionPreprocess(
      graph, parallelParams.releaseParams(), options, plan, level, levelIndices,
      inSlice, weightsSlice, false, &cpt.transformPre[level], &cpt.copyWritten,
      rearrangeActs, rearrangeWeights, debugPrefix);

  // We create partials at as high a level in the hierarchy as possible so
  // as to reduce the complexity of the tensor expression that represents
  // the partials at the higher levels (a concatenation of multiple
  // consecutive slices of the same variable can be simplified into one
  // simpler expression). This is a graph construction-time optimisation.
  if (level == createPartialsLevel) {
    // shape is [G/cgpg][Co/cpg][N]...[cgpg][cpg]
    auto partialsShape = getPartialOutputShape(parallelParams.getParams(),
                                               plan.convGroupsPerGroup,
                                               plan.partialChansPerGroup);

    if (level != plan.partitions.size()) {
      // add an extra dimension at the beginning that we can reduce over.
      const auto &partition = plan.partitions[level];
      const auto numPartials =
          partition.inChanSplit.parallel * product(partition.kernelSplit);
      partialsShape.insert(partialsShape.begin(), numPartials);
    }

    partials = graph.addVariable(plan.types.back().partialType, partialsShape,
                                 debugPrefix + "/partials");
  }

  // Convolve.
  Tensor out;
  const auto resultType = plan.types[level].resultType;
  const auto tileLevel = plan.transforms.size() - 1;
  if (level == tileLevel) {
    const auto &target = graph.getTarget();
    const auto tile = linearizeTileIndices(target, options, indices, plan);
    assert(cpt.transformPre.size() - 1 == level);
    calcPartialConvOutput(graph, plan, tile, parallelParams.getParams(),
                          cpt.transformPre[level].postTranspose,
                          cpt.copyWritten, cpt.convolveCSGroup, inSlice,
                          weightsSlice, partials, options.use128BitConvUnitLoad,
                          debugPrefix);
    out = partials;

    if (level == createPartialsLevel) {
      out = unsplitActivationFromGroups(out);
    }
    if (level > createPartialsLevel) {
      // The explicit output of the partial convolution is never used.
      return {};
    }
  } else {
    const auto &partition = plan.partitions[level];
    const auto numPartials =
        partition.inChanSplit.parallel * product(partition.kernelSplit);
    const auto outputSplit = partition.convGroupSplit * partition.batchSplit *
                             product(partition.fieldSplit) *
                             partition.outChanSplit.parallel;
    std::vector<std::vector<boost::optional<Tensor>>> results(
        numPartials, std::vector<boost::optional<Tensor>>(outputSplit));
    iteratePartitionParallel(
        parallelParams, partition,
        [&](const ConvIndices &parallelIndices, const ConvSlice &slice) {
          // Get sub convolution
          Tensor subIn = inSlice;
          Tensor subWeights = weightsSlice;
          assert(levelIndices.size() == level + 1);
          auto subIndices = levelIndices;
          subIndices.back().parallel = parallelIndices;
          const auto subParams =
              getSubConvolution(slice, parallelParams, &subIn, &subWeights);
          auto partialIndex = getPartialIndex(parallelIndices, partition);
          Tensor nextLevelPartials;
          if (level >= createPartialsLevel) {
            nextLevelPartials = partials;
            if (level == createPartialsLevel) {
              nextLevelPartials = nextLevelPartials[partialIndex];
            }
            nextLevelPartials =
                sliceOutput(nextLevelPartials, slice, plan.convGroupsPerGroup,
                            plan.partialChansPerGroup);
          }
          auto subOut =
              convolutionImpl(graph, subParams, plan, level + 1, subIn,
                              subWeights, cpt, subIndices, nextLevelPartials,
                              createPartialsLevel, debugPrefix, options);
          auto outputIndex = getOutputIndex(parallelIndices, partition);

          // the shape of each result is the ungrouped internal shape.
          results[partialIndex][outputIndex] = subOut;
        });
    // Stitch together results.
    if (level < createPartialsLevel) {
      partials = stitchPartialResults(results, partition);
    } else {
      if (level != createPartialsLevel) {
        // The explicit output of the partial convolution is never used.
        return {};
      }

      // the partials at this point are in the grouped internal shape with an
      // extra dimension at the beginning to reduce over. transform these
      // partials into the ungrouped internal shape. temporarily reinterpret the
      // reduction dimension as an extra spatial dimension so we can use
      // the common grouping utility functions.

      // [R][G1][CO1][N]...[G2][CO2] -> [G1][CO1][N][R]...[G2][CO2]
      partials = partials.dimRoll(0, 3);

      // [G1][CO1][N][R]...[G2][CO2] -> [G][N][R]...[CO]
      partials = unsplitActivationFromGroups(partials);

      // [G][N][R]...[CO] -> [R][G][N]...[CO]
      partials = partials.dimRoll(2, 0);
    }

    // at this point the shape of the partials must be internal shape plus the
    // reduction dimension.

    // Reduce
    const auto partialType = partials.elementType();
    // Perform the reduction of partial sums.
    if (partials.dim(0) == 1) {
      out = partials.squeeze({0});
    } else {
      // the partials shape here is the ungrouped internal shape with a
      // reduction dimension. we want to change it to the grouped version so,
      // as above, dimroll the reduction dimension into the spatial dimensions
      // and then reshape the partials.
      const auto convGroupGrainSize = partition.convGroupGrainSize;
      const auto outChanGrainSize = partition.outChanGrainSize;

      // [R][G][N]...[CO] -> [G][N][R]...[CO]
      partials = partials.dimRoll(0, 2);

      // [G][N][R]...[CO] -> [G][N[G1][CO1][N][R]...[G2][CO2]
      partials = splitActivationIntoGroups(partials, convGroupGrainSize,
                                           outChanGrainSize);

      // [G1][CO1][N][R]...[G2][CO2] -> [R][G1][CO1][N]...[G2][CO2]
      partials = partials.dimRoll(3, 0);

      // Avoid reducing to the result type for serial input channel splitting
      // until inPlace addition of all serial splits have completed.
      auto reducedType =
          (partition.inChanSplit.serial > 1) ? partialType : resultType;
      out = multiStageGroupedReduce(graph, partials, reducedType,
                                    cpt.reduceComputeSets[level], options,
                                    debugPrefix);
      out = unsplitActivationFromGroups(out);
    }
  }
  const auto &partition = plan.partitions[level];
  if ((out.elementType() != resultType) &&
      (partition.inChanSplit.serial == 1)) {
    if (cpt.reduceComputeSets[level].empty()) {
      cpt.reduceComputeSets[level].push_back(
          graph.addComputeSet(debugPrefix + "/Cast"));
    }
    out = cast(graph, out, resultType, cpt.reduceComputeSets[level][0],
               debugPrefix);
  }

  // Inverse transform.
  out = convolutionPostprocess(graph, preTransformParams, originalTransform,
                               out, false /* serial */,
                               cpt.transformPost[level], debugPrefix);

  // Update.
  if (level == ipuLevel) {
    const bool inChansAreSeriallySplit = partition.inChanSplit.serial > 1;
    const bool outChansAreSeriallySplit = partition.outChanSplit.serial > 1;

    if (inChansAreSeriallySplit) {
      auto serialOut = graph.clone(out, debugPrefix + "/serialOut_clone");

      Sequence loopCounterGtZero;
      // Accumulate the results into the destination tensor serialOut
      addInPlace(graph, serialOut, out, loopCounterGtZero,
                 debugPrefix + "/serialOut" + levelSuffix);

      // Just save output on first iteration so it can be used to accumulate
      // following results without explicitly zeroing acc storage
      cpt.update.add(Switch(loopCounter.reshape({}),
                            {{0, Copy(out, serialOut)}}, loopCounterGtZero));
      out = serialOut;
    }

    if (outChansAreSeriallySplit) {
      // Make this tensor view suitable as a slice of the full output.
      out = out.expand({0});

      // Create an output tensor for the partials.
      auto serialOut = createSliceableTensorFromSlice(
          graph, out, {0}, {partition.outChanSplit.serial},
          debugPrefix + "/serialOut" + levelSuffix);

      // WriteUndef the output as it is Read/Write in each iteration but in the
      // course of the entire loop is completely written.
      cpt.transformPreSerial.writeUndef.emplace_back(serialOut);
      dynamicUpdate(graph, serialOut, out, loopCounter, {0}, {1}, cpt.update,
                    debugPrefix + "/serialUpdate" + levelSuffix);

      // Flatten serial output channel split back into output channels
      out = serialOut.dimRoll(0, serialOut.rank() - 2)
                .flatten(serialOut.rank() - 2, serialOut.rank());
    }

    // common code for either serial splits.
    if (inChansAreSeriallySplit || outChansAreSeriallySplit) {
      // Increment counter
      auto loopIncrement = graph.addConstant(UNSIGNED_INT, {}, 1);
      graph.setTileMapping(loopIncrement, 0);
      addInPlace(graph, loopCounter, loopIncrement, cpt.loopPost,
                 debugPrefix + "/loopIncrement" + levelSuffix);
    }
  }

  // Casting to the final result type (i.e., the result type of the outermost
  // level) should be deferred until all the serial splits have executed.
  if ((out.elementType() != resultType) && (partition.inChanSplit.serial > 1)) {
    out = cast(graph, out, resultType, cpt.transformPostSerial.castCS,
               debugPrefix);
  }

  // Inverse transform.
  out = convolutionPostprocess(graph, originalParams, originalTransform, out,
                               true /* serial */,
                               cpt.transformPostSerial.copies, debugPrefix);
  return out;
}

template <typename T>
static std::string getShapeAsString(const std::vector<T> &shape) {
  return shape.empty() ? std::string()
                       : std::accumulate(std::next(shape.begin()), shape.end(),
                                         std::to_string(shape[0]),
                                         [](std::string a, unsigned b) {
                                           return a + "x" + std::to_string(b);
                                         });
}

std::string convSuffix(const CanonicalConvParams &params) {
  std::string s = getShapeAsString(params->kernelShape);
  if (std::any_of(params->outputTransform.stride.begin(),
                  params->outputTransform.stride.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_stride" + getShapeAsString(params->outputTransform.stride);
  }
  if (std::any_of(params->inputTransform.dilation.begin(),
                  params->inputTransform.dilation.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_inDilation" + getShapeAsString(params->inputTransform.dilation);
  }
  return s;
}

static bool requiresReduction(const Partition &partition) {
  if (partition.inChanSplit.parallel != 1)
    return true;
  for (const auto &split : partition.kernelSplit) {
    if (split != 1)
      return true;
  }
  return false;
}

// Get the lowest level at which we can create the partials tensor.
static unsigned getCreatePartialsLevel(const Plan &plan) {
  const auto numLevels = plan.partitions.size();
  unsigned level = numLevels;
  const auto &partialType = plan.types.back().partialType;

  while (level > 0) {
    const auto &transform = plan.transforms[level];
    // If this level transforms the input in anyway then stop since
    // creating partials earlier may not be the right shape.
    if (transform.swapOperands || !transform.outChanFlattenDims.empty() ||
        !transform.flattenDims.empty() || !transform.expandDims.empty() ||
        !transform.dilatePostConv.empty() ||
        (transform.combineConvGroupsFactor != 1))
      break;
    // If this level casts the partials to a different type then stop.
    if (partialType != plan.types[level].resultType)
      break;
    if (level < plan.partitions.size()) {
      // If this level is earlier than the tile level, there may be a post
      // transformation of the partials (a regrouping or reduction) in which
      // we must also stop.
      const auto &partition = plan.partitions[level];
      if (partition.outChanGrainSize != plan.partialChansPerGroup)
        break;
      if (requiresReduction(partition))
        break;
      // If there is a serial split at this level there will be some kind of
      // buffering preventing us from passing the partials down any further.
      if (partition.totalSerialSplit() > 1)
        break;
    }
    level--;
  }
  return level;
}

void preplanConvolutions(const std::set<ConvPlanParams> &convs,
                         PlanningCache &cache) {
  std::set<ConvPlanKey> convsImpl;

  if (convs.empty())
    return;

  for (auto &conv : convs) {
    const ConvOptions options(*std::get<0>(conv), *std::get<2>(conv));
    convsImpl.emplace(std::get<1>(conv), options);
  }
  auto &commonTarget = *std::get<0>(*convs.cbegin());
  preplanConvolutionsImpl(commonTarget, convsImpl, cache);
}

static Tensor remapOutputTensor(Graph &graph, const poplar::Tensor &output,
                                Sequence &prog, const ConvParams &params,
                                const std::string &debugPrefix);

static Tensor convolutionInternal(
    Graph &graph, const poplar::Tensor &in_, const poplar::Tensor &weights_,
    const Plan &plan, const CanonicalConvParams &params,
    bool transposeAndFlipWeights, ConvProgramTree &cpt,
    const std::string &debugPrefix, const ConvOptions &options) {
  auto weights = weights_;
  if (weights.rank() == params->getNumFieldDims() + 2) {
    weights = weights.expand({0});
  }
  if (transposeAndFlipWeights) {
    // Create transposed/flipped weights
    auto bwdWeights =
        createWeights(graph, plan, params.getParams(), "bwdWeights", options);
    if (bwdWeights.dim(1) && bwdWeights.dim(2)) {
      weightsTransposeChansFlipXY(graph, weights, bwdWeights, cpt.initProg,
                                  debugPrefix);
    }
    weights = bwdWeights;
  }
  auto fullWeights = weights;

  auto weightsIn = weights;
  weights = weightsToInternalShape(weights);
  auto in = actsToInternalShape(in_, params->numConvGroups,
                                params->inputChannelsPerConvGroup);
  verifyInputShapes(params, in, weights);

  const auto createPartialsLevel = getCreatePartialsLevel(plan);
  auto activations = *convolutionImpl(
      graph, params, plan, 0, in, weights, cpt, {} /* indices */,
      {} /* partials */, createPartialsLevel, debugPrefix, options);

  assert(activations.elementType() == params->outputType);
  auto output = actsToExternalShape(activations);

  // Introspect output tensor to check if it has a decent layout as a bad layout
  // impacts operations using the tensor in both memory and cycles. This is a
  // conservative check as we only check if there's a grouping.
  if (options.remapOutputTensor) {
    const auto dimGroupings = detectDimGroupings(graph, output);
    if (dimGroupings.empty()) {
      return remapOutputTensor(graph, output, cpt.finalizeProg, *params,
                               debugPrefix);
    }
  }

  return output;
}

Tensor remapOutputTensor(Graph &graph, const poplar::Tensor &output,
                         Sequence &prog, const ConvParams &params,
                         const std::string &debugPrefix) {
  const auto grainSize = 8U;
  const auto minElementsPerTile = 8U;
  std::size_t chansPerGroup, numChanGroups;
  if (params.getNumOutputChansPerConvGroup() % grainSize) {
    // do not remap if the output channels is not a multiple of grain size.
    // We could find a grain size in other dimensions and map but keep it
    // simple for now.
    return output;
  } else {
    chansPerGroup = grainSize;
    numChanGroups = params.getNumOutputChansPerConvGroup() / chansPerGroup;
  }
  std::vector<std::size_t> remapShape = {params.getNumConvGroups(),
                                         numChanGroups, params.getBatchSize()};
  for (const auto &e : params.getOutputFieldShape())
    remapShape.push_back(e);
  remapShape.push_back(chansPerGroup);

  // Keep the created tensor contiguous in the channel dimension. We
  // could also create a grouping for the channels if possible
  auto remappedOutput = graph.addVariable(output.elementType(), remapShape,
                                          debugPrefix + "/remappedOutput");
  mapTensorLinearly(graph, remappedOutput, minElementsPerTile, grainSize);
  remappedOutput = remappedOutput.dimRoll(2, 0)
                       .dimRoll(remapShape.size() - 1, 3)
                       .flatten(1, 4);
  // Explicity copy to remapped tensor with a benign layout
  prog.add(Copy(output, remappedOutput));
  logging::debug("  convolution output tensor remapped linearly");
  return remappedOutput;
}

Tensor convolution(Graph &graph, const poplar::Tensor &in,
                   const poplar::Tensor &weights, const Plan &plan,
                   const CanonicalConvParams &params,
                   bool transposeAndFlipWeights, ConvProgramTree &cpt,
                   const std::string &debugPrefix, const ConvOptions &options) {
  logging::info("convolution");
  logging::info("  pass={}, name=\"{}\"", options.pass, debugPrefix);
  log(2, *params);

  auto output =
      convolutionInternal(graph, in, weights, plan, params,
                          transposeAndFlipWeights, cpt, debugPrefix, options);

  return output;
}

Tensor convolution(Graph &graph, const poplar::Tensor &in,
                   const poplar::Tensor &weights, const ConvParams &params_,
                   bool transposeAndFlipWeights, Sequence &prog,
                   const std::string &debugPrefix,
                   const poplar::OptionFlags &options_, PlanningCache *cache) {
  const CanonicalConvParams params(params_);
  const ConvOptions options(graph.getTarget(), options_);

  const auto layerName = debugPrefix + "/Conv_" + convSuffix(params);
  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  ConvProgramTree cpt(graph, plan, layerName);

  auto out = convolution(graph, in, weights, plan, params,
                         transposeAndFlipWeights, cpt, layerName, options);

  cpt.lower(prog);
  return out;
}

static uint64_t getFlops(const ConvParams &params) {
  return (2 * getNumberOfMACs(params));
}

uint64_t getFwdFlops(const ConvParams &params) { return getFlops(params); }

uint64_t getBwdFlops(const ConvParams &params) { return getFlops(params); }

uint64_t getWuFlops(const ConvParams &params) { return getFlops(params); }

static double getPerfectCycleCount(const Graph &graph,
                                   const ConvParams &params) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  auto numMacs = getNumberOfMACs(params);
  if (params.inputType == FLOAT) {
    const auto floatVectorWidth = target.getFloatVectorWidth();
    auto macCycles =
        static_cast<double>(numMacs) / (floatVectorWidth * numTiles);
    return macCycles;
  }
  assert(params.inputType == HALF);
  assert(params.outputType == HALF || params.outputType == FLOAT);
  const auto convUnitsPerTile =
      std::max(std::max(target.getFp16InFp16OutConvUnitsPerTile(),
                        target.getFp32InFp32OutConvUnitsPerTile()),
               target.getFp16InFp32OutConvUnitsPerTile());
  const auto halfVectorWidth = target.getHalfVectorWidth();
  auto macsPerCycle = convUnitsPerTile * halfVectorWidth;
  auto macCycles = static_cast<double>(numMacs) / (macsPerCycle * numTiles);
  return macCycles;
}

double getFwdPerfectCycleCount(const Graph &graph, const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

double getBwdPerfectCycleCount(const Graph &graph, const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

double getWuPerfectCycleCount(const Graph &graph, const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

void weightsTransposeChansFlipXY(Graph &graph, const Tensor &weightsInUnGrouped,
                                 const Tensor &weightsOutUnGrouped,
                                 Sequence &prog,
                                 const std::string &debugPrefix) {
  assert(weightsInUnGrouped.rank() >= 3);
  const auto numFieldDims = weightsInUnGrouped.rank() - 3;

  const auto weightsIn =
      splitWeightsFromGroups(graph, weightsToInternalShape(weightsInUnGrouped));
  const auto weightsOut = splitWeightsFromGroups(
      graph, weightsToInternalShape(weightsOutUnGrouped));

  // weightsIn = [GC1][O/G1][I/G2]...[GC2][G1][G2]
  // weightsOut = [GC1][I/G3][O/G4]...[GC2][G3][G4]
  const auto G1 = weightsIn.dim(weightsIn.rank() - 2);
  const auto G2 = weightsIn.dim(weightsIn.rank() - 1);
  const auto G3 = weightsOut.dim(weightsOut.rank() - 2);
  const auto G4 = weightsOut.dim(weightsOut.rank() - 1);
  const auto I = weightsOut.dim(1) * G3;
  const auto O = weightsOut.dim(2) * G4;
  if (weightsIn.dim(1) != O / G1 || weightsIn.dim(2) != I / G2) {
    throw poplibs_error("The sizes of the input and output channels of the two "
                        "weight tensors must be opposite each other.");
  }

  // Express the rearrangement as a composition of two rearrangements such
  // that the first rearrangement avoids exchange and maximises the size of the
  // block that is rearranged in the second step. This reduces exchange code
  // since the second step involves fewer, larger messages.
  // G5 is the size of the innermost dimension after the partial transposition.
  // To avoid exchange it must divide G1. If G4 divides G1 then set G5 to G4 -
  // this results in the block size of G1 * gcd(G2, G3) elements in the
  // second step. Otherwise set G5 to G1 for a block size of gcd(G1, G4)
  // elements.
  const auto G5 = (G1 % G4 == 0) ? G4 : G1;
  Tensor partiallyTransposed;
  if (G5 == 1) {
    // [GC1][O/G1][I/G2]...[GC2][G1][G2] -> [GC1][O/G1][I/G2]...[GC2][G1][G2][1]
    partiallyTransposed = weightsIn.expand({weightsIn.rank()});
  } else {
    // [GC1][O/G1][I/G2]...[GC2][G1][G2]
    //    -> [GC1][O/G1][I/G2]...[GC2][G1/G5][G5][G2]
    partiallyTransposed = weightsIn.reshapePartial(
        weightsIn.rank() - 2, weightsIn.rank(), {G1 / G5, G5, G2});

    auto cs = graph.addComputeSet(debugPrefix + "/WeightTranspose");

    // [GC1][O/G1][I/G2]...[GC2][G1/G5][G5][G2]
    //    -> [GC1][O/G1][I/G2]...[GC2][G1/G5][G2][G5]
    partiallyTransposed = popops::rearrange::partialTranspose(
        graph, partiallyTransposed, cs, debugPrefix);
    prog.add(Execute(cs));
  }

  auto flipped = partiallyTransposed;
  // reverse the data for each of the spatial dimensions
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    flipped = flipped.reverse(3 + dim);
  }

  // number of "other" dimensions, ie. conv groups (inc. grouping) and spatial
  const auto F = numFieldDims + 2;

  // clang-format off
  // [GC1][O/G1][I/G2]...[GC2][G1/G5][G2][G5]
  //    -> [GC1]...[GC2][O/G1][G1/G5][G5][I/G2][G2]
  // (assuming we took the G5 != 1 branch above)
  flipped = flipped.dimShufflePartial({1, F+2, F+4, 2,   F+3},
                                      {F, F+1, F+2, F+3, F+4});

  // [GC]...[O/G1][G1/G5][G5][I/G2][G2] -> [GC]...[O/G4][G4][I/G3][G3]
  flipped = flipped.reshapePartial(flipped.rank() - 5, flipped.rank(),
                                   {O / G4, G4, I / G3, G3});

  // [GC1]...[GC2][O/G4][G4][I/G3][G3]
  //    -> [GC1][I/G3][O/G4]...[GC2][G3][G4] (weightsOut)
  flipped = flipped.dimShufflePartial({F+2, F, F+3, F+1},
                                      {1,   2, F+2, F+3});
  // clang-format on

  // convert back to the external shape before doing the copy. we do this
  // because there is a limitation to determining the grouping through
  // introspection: when there is, for eg. [CO1][CI1][H][W][CO2][CI2] and
  // the kernel and num of input channels (CI1) are all 1. then we cannot
  // differentiate between [CO1][1]...[1][CO2][CI2] and [CO1 * CO2][CI2],
  // therefore the grouping of the output channels will be incorrect and the
  // memory layout of the input and output weight tensors will be different.
  // this is quite rare so we handle it by switching back to the internal shapes
  // which will still be consistent and then Poplar will add a copy in for us.
  // we may need to revisit this if it comes up in a real example rather than
  // a random test.
  flipped = weightsToExternalShape(unsplitWeightsFromGroups(flipped));

  // Before the flipped weights are copied, attempt to regroup source tensor
  // only if it's innermost dimension doesn't match the destination tensor. If
  // a partial transpose is done, then the regrouping will not be done.
  auto maybeRegroupedFlipped = popops::rearrange::regroupIfBeneficial(
      graph, flipped, weightsOutUnGrouped, prog,
      debugPrefix + "/attemptRegroup");

  prog.add(Copy(maybeRegroupedFlipped, weightsOutUnGrouped));
}

ConvParams getWeightUpdateParams(const ConvParams &fwdParams_) {
  const CanonicalConvParams fwdParams(fwdParams_);
  const auto numFieldDims = fwdParams->getNumFieldDims();
  auto wuFlipInput = fwdParams->inputTransform.flip;
  std::vector<bool> wuFlipKernel(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (fwdParams->kernelTransform.flip[dim]) {
      // If the kernel is flipped in the forward pass we must flip the output
      // in the weight update pass. This is equivalent to flipping both the
      // activations and the deltas in the weight update pass.
      wuFlipInput[dim] = !wuFlipInput[dim];
      wuFlipKernel[dim] = !wuFlipKernel[dim];
    }
  }

  auto newInputTransform = fwdParams->inputTransform;
  newInputTransform.flip = wuFlipInput;
  const auto &kernelTransform = fwdParams->kernelTransform;
  const auto &outputTransform = fwdParams->outputTransform;
  ConvParams::InputTransform newKernelTransform{
      outputTransform.paddingLower,    // kernelTruncationLower
      outputTransform.paddingUpper,    // kernelTruncationUpper
      outputTransform.stride,          // kernelDilation
      outputTransform.truncationLower, // kernelPaddingLower
      outputTransform.truncationUpper, // kernelPaddingUpper
      wuFlipKernel                     // flipKernel
  };
  ConvParams::OutputTransform newOutputTransform{
      kernelTransform.paddingLower,    // outputTruncationLower
      kernelTransform.paddingUpper,    // outputTruncationUpper
      kernelTransform.dilation,        // stride
      kernelTransform.truncationLower, // outputPaddingLower
      kernelTransform.truncationUpper  // outputPaddingUpper
  };
  ConvParams wuParams(
      fwdParams->inputType, fwdParams->outputType,
      fwdParams->getNumInputChansPerConvGroup(),  // batchSize
      fwdParams->inputFieldShape,                 // inputFieldShape
      fwdParams->getOutputFieldShape(),           // kernelShape
      fwdParams->getBatchSize(),                  // inputChannels
      fwdParams->getNumOutputChansPerConvGroup(), // outputChannels
      fwdParams->numConvGroups,                   // numConvGroups
      newInputTransform, newKernelTransform, newOutputTransform);
  return wuParams.canonicalize();
}

Tensor calculateWeightDeltas(Graph &graph, const Tensor &zDeltas_,
                             const Tensor &activations_, const Plan &wuPlan,
                             const CanonicalConvParams &wuParams,
                             ConvProgramTree &cpt,
                             const std::string &debugPrefix,
                             const ConvOptions &wuOptions) {
  const auto fwdNumConvGroups = wuParams->numConvGroups;
  const auto fwdOutChans = wuParams->outputChannelsPerConvGroup;
  const auto fwdInChans = wuParams->batchSize;

  // [G][N]...[Co]
  auto zDeltas = actsToInternalShape(zDeltas_, fwdNumConvGroups, fwdOutChans);

  // [G][N]...[Ci]
  auto activations =
      actsToInternalShape(activations_, fwdNumConvGroups, fwdInChans);

  // The weight update is equivalent to a convolution where:
  // - wu conv groups = fwd conv groups
  // - wu batch size = fwd input channels
  // - wu input channels = fwd batch size
  // - wu height = fwd height
  // - wu width = fwd width
  // - wu output channels = fwd output channels

  // [G][C]...[N]
  auto activationsRearranged = activations.dimShufflePartial(
      {1, activations.rank() - 1}, {activations.rank() - 1, 1});

  // Acts[G][C][N]... or Weights[G][OC][IC]...
  auto deltasRearranged = zDeltas.dimShufflePartial({zDeltas.rank() - 1}, {1});

  // [N][G * C]...
  auto weightDeltas = convolution(
      graph, actsToExternalShape(activationsRearranged), deltasRearranged,
      wuPlan, wuParams, false, cpt, debugPrefix, wuOptions);

  // [G][C]...[N]
  weightDeltas =
      actsToInternalShape(weightDeltas, fwdNumConvGroups, fwdOutChans);

  return weightsToExternalShape(
      weightDeltas.dimShufflePartial({1}, {weightDeltas.rank() - 1}));
}

Tensor calculateWeightDeltas(Graph &graph, const Tensor &zDeltas_,
                             const Tensor &activations_,
                             const ConvParams &fwdParams_, Sequence &prog,
                             const std::string &debugPrefix,
                             const poplar::OptionFlags &fwdOptions_,
                             PlanningCache *cache) {
  const CanonicalConvParams wuParams = getWeightUpdateParams(fwdParams_);
  const auto wuOptions =
      getWeightUpdateOptions({graph.getTarget(), fwdOptions_});
  const auto wuPlan = getPlan(graph.getTarget(), wuParams, wuOptions, cache);

  const auto layerName = debugPrefix + "/Conv_" + convSuffix(wuParams);
  ConvProgramTree cpt(graph, wuPlan, layerName);

  auto out = calculateWeightDeltas(graph, zDeltas_, activations_, wuPlan,
                                   wuParams, cpt, debugPrefix, wuOptions);

  cpt.lower(prog);
  return out;
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             const Plan &wuPlan, CanonicalConvParams wuParams,
                             const Tensor &scale, ConvProgramTree &cpt,
                             const std::string &debugPrefix,
                             const ConvOptions &wuOptions) {
  auto weightDeltas =
      calculateWeightDeltas(graph, zDeltas, activations, wuPlan, wuParams, cpt,
                            debugPrefix, wuOptions);

  // update weights
  assert(weightDeltas.shape() == weights.shape());
  scaledAddTo(graph, weights, weightDeltas, scale, cpt.finalizeProg,
              debugPrefix + "/UpdateWeights");
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             ConvParams fwdParams, const Tensor &scale,
                             Sequence &prog, const std::string &debugPrefix,
                             const poplar::OptionFlags &fwdOptions_,
                             PlanningCache *cache) {
  // Adjust params so that weightDelta is of inputType without needing to cast.
  fwdParams.outputType = fwdParams.inputType;

  const CanonicalConvParams wuParams =
      getWeightUpdateParams(std::move(fwdParams));
  const auto wuOptions =
      getWeightUpdateOptions({graph.getTarget(), fwdOptions_});
  const auto wuPlan = getPlan(graph.getTarget(), wuParams, wuOptions, cache);

  ConvProgramTree cpt(graph, wuPlan, debugPrefix);

  convolutionWeightUpdate(graph, zDeltas, weights, activations, wuPlan,
                          std::move(wuParams), scale, cpt, debugPrefix,
                          wuOptions);
  cpt.lower(prog);
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             const Plan &wuPlan, CanonicalConvParams wuParams,
                             float scale, ConvProgramTree &cpt,
                             const std::string &debugPrefix,
                             const ConvOptions &wuOptions) {
  auto weightDeltas =
      calculateWeightDeltas(graph, zDeltas, activations, wuPlan, wuParams, cpt,
                            debugPrefix, wuOptions);

  // Add the weight deltas to the weights.
  assert(weightDeltas.shape() == weights.shape());
  const auto maybeRegroupedWeightDeltas =
      popops::rearrange::regroupIfBeneficial(graph, weightDeltas, weights,
                                             cpt.finalizeProg,
                                             debugPrefix + "regroupGradds");

  scaledAddTo(graph, weights, maybeRegroupedWeightDeltas, scale,
              cpt.finalizeProg, debugPrefix + "/UpdateWeights");
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             ConvParams fwdParams, float scale, Sequence &prog,
                             const std::string &debugPrefix,
                             const poplar::OptionFlags &fwdOptions_,
                             PlanningCache *cache) {
  // Adjust params so that weightDelta is of inputType without needing to cast.
  fwdParams.outputType = fwdParams.inputType;

  const CanonicalConvParams wuParams =
      getWeightUpdateParams(std::move(fwdParams));
  const auto wuOptions =
      getWeightUpdateOptions({graph.getTarget(), fwdOptions_});
  const auto wuPlan = getPlan(graph.getTarget(), wuParams, wuOptions, cache);

  ConvProgramTree cpt(graph, wuPlan, debugPrefix);

  convolutionWeightUpdate(graph, zDeltas, weights, activations, wuPlan,
                          std::move(wuParams), scale, cpt, debugPrefix,
                          wuOptions);
  cpt.lower(prog);
}

// Add a program to update the biases tensor with the gradients derived
// from the zDeltas tensor
void convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                           const Tensor &biases, const Tensor &scale,
                           const poplar::OptionFlags &options_, Sequence &prog,
                           const std::string &debugPrefix) {
  const ConvOptions options(graph.getTarget(), options_);
  if (zDeltasUngrouped.rank() < 2)
    throw poplibs_error("convolutionBiasUpdate with rank " +
                        std::to_string(zDeltasUngrouped.rank()) +
                        "; must have at least channel and batch dimensions");

  std::vector<std::size_t> reduceDims(zDeltasUngrouped.rank() - 1);
  std::iota(reduceDims.begin() + 1, reduceDims.end(), 2);

  popops::reduceWithOutput(
      graph, zDeltasUngrouped, biases, reduceDims,
      {popops::Operation::ADD, true, scale}, prog, debugPrefix + "/BiasUpdate",
      {{"accumType.interTile", options.partialsType.toString()},
       {"accumType.inVertex", options.partialsType.toString()}});
}
void convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                           const Tensor &biases, float scale,
                           const poplar::OptionFlags &options_, Sequence &prog,
                           const std::string &debugPrefix) {
  auto scaleTensor =
      graph.addConstant(FLOAT, {}, scale, debugPrefix + "/scaleTensor");
  graph.setTileMapping(scaleTensor, 0);
  convolutionBiasUpdate(graph, zDeltasUngrouped, biases, scaleTensor, options_,
                        prog, debugPrefix + "/ConstLearning");
}

void addBias(Graph &graph, const Tensor &acts, const Tensor &biases,
             Sequence &prog, const std::string &debugPrefix) {
  if (acts.rank() < 2) {
    throw poplibs_error("Expected at least a batch size and channel dimension");
  }

  std::vector<std::size_t> broadcastBiases(acts.rank() - 2, 1);
  addInPlace(graph, acts, biases.expand(broadcastBiases), prog, debugPrefix);
}

static Plan getFullyConnectedFwdPlanFromBwdParams(
    const Target &target, const CanonicalConvParams &bwdParams,
    const ConvOptions &bwdOptions, PlanningCache *cache) {
  assert(bwdOptions.pass == Pass::FC_TRAINING_BWD);
  auto fwdParams = bwdParams.getParams();
  std::swap(fwdParams.inputFieldShape[0], fwdParams.inputChannelsPerConvGroup);
  if (fwdParams.inputFieldShape[0] == 0) {
    // Transformed input must be greater than or equal to the transformed kernel
    // size.
    fwdParams.inputTransform.paddingUpper[0] = 1;
    fwdParams.outputTransform.truncationUpper[0] = 1;
  }
  auto fwdOptions = bwdOptions;
  fwdOptions.pass = Pass::FC_TRAINING_FWD;
  return getPlan(target, fwdParams, fwdOptions, cache);
}

static bool planSwapsOperands(const Plan &plan) {
  for (const auto &entry : plan.transforms) {
    if (entry.swapOperands)
      return true;
  }
  return false;
}

struct FCWTGroupSizes {
  unsigned fwdGroupSize;
  unsigned bwdGroupSize;
  // It may be that the group sizes are such that there is no point in using
  // transpose vertices so fall back to copys
  bool useCopyImpl;
};

static std::vector<unsigned> commonDivisors(const unsigned A,
                                            const unsigned B) {
  const unsigned GCD = gcd(A, B);
  // All factors of GCD are common factors of A and B, and all common divisors
  // of A and B are factors of GCD
  std::vector<unsigned> result(1, GCD);
  // populated divisors ordered with largest first
  for (unsigned i = ceildiv(GCD, 2U); i > 0; --i) {
    if (GCD % i == 0) {
      result.push_back(i);
    }
  }
  return result;
}

static size_t accumSize(const std::vector<poplar::Interval> &vi) {
  return std::accumulate(
      vi.begin(), vi.end(), 0,
      [](size_t acc, const poplar::Interval &i) { return acc + i.size(); });
}

// If the output to the transpose vertex is not contiguous we will introduce a
// gather copy which ideally will be avoided. This function does not create the
// full output tensor but a part of it. This means it won't catch all cases but
// is faster than having to create the entire tensor and catches the case I care
// about today
static bool
transposeOutputIsContiguous(const Tensor &outTensor,
                            const std::vector<size_t> &firstInGroupShape,
                            const std::vector<poplar::Interval> &intervals) {

  const auto blockIndices =
      poputil::unflattenIndex(firstInGroupShape, intervals.front().begin());
  const auto outConnect = outTensor[blockIndices[0]][blockIndices[1]]
                                   [blockIndices[3]][blockIndices[2]]
                                       .flatten();
  return outConnect.isContiguous();
}

static Tensor getGroupedFCWeightsView(const Tensor &splitActivations,
                                      const unsigned inChansPerGroup,
                                      const unsigned fieldElementsPerGroup) {
  return splitActivations
      .reshape({splitActivations.dim(0), splitActivations.dim(1),
                splitActivations.dim(2) / inChansPerGroup, inChansPerGroup,
                splitActivations.dim(3) / fieldElementsPerGroup,
                fieldElementsPerGroup})
      .dimShufflePartial({3}, {4});
}

static Tensor getFirstInGroup(Tensor &splitActivations,
                              const unsigned bwdGroupSize,
                              const unsigned fwdGroupSize) {
  splitActivations =
      getGroupedFCWeightsView(splitActivations, bwdGroupSize, fwdGroupSize);

  return splitActivations
      .slice({0, 0, 0, 0, 0, 0},
             {splitActivations.dim(0), splitActivations.dim(1),
              splitActivations.dim(2), splitActivations.dim(3), 1, 1})
      .squeeze({4, 5});
}

// Returns a score for how fast the transpose compute set should be executed.
// Higher score means faster execution. This is an estimated score
static double blockScore(const unsigned fwdGroupSize,
                         const unsigned bwdGroupSize, const Graph &graph,
                         Tensor splitActivations,
                         const Tensor &splitTransposed) {
  // Call getGroupedFCWeightsView with fwd and bwd switched compared to
  // calcylation to get first in group
  auto outTensor =
      getGroupedFCWeightsView(splitTransposed, fwdGroupSize, bwdGroupSize);
  const auto firstInGroup =
      getFirstInGroup(splitActivations, bwdGroupSize, fwdGroupSize);
  const auto mapping = graph.getTileMapping(firstInGroup);

  unsigned spread = 0;
  double score = std::numeric_limits<double>::max();
  const auto firstInGroupShape = firstInGroup.shape();
  for (const auto &intervals : mapping) {
    if (!intervals.empty()) {
      ++spread;
      const unsigned numTileTranspositions = accumSize(intervals);
      const bool fastTrans = popops::rearrange::canUseFastTranspose(
          graph.getTarget(), splitActivations.elementType(), bwdGroupSize,
          fwdGroupSize, numTileTranspositions);

      const bool TOIC =
          transposeOutputIsContiguous(outTensor, firstInGroupShape, intervals);
      double tileScore = std::numeric_limits<double>::max();
      if (fastTrans) {
        // fast transpose transposes 64 bits per cycle in it's inner loop
        // slow transpose transfers 64 bits in 5 cycles
        tileScore = 5.0;
      } else {
        tileScore = 1.0;
      }
      if (!TOIC) {
        // If the transpose output is not contiguous then a post arrange copy
        // will be introduced. This number is fairly empiracle based on what
        // made the benchmarks pass
        tileScore = tileScore / 3;
      }
      score = std::min(score, tileScore); // update score with worst tile score
    }
  }
  score = score * spread;
  return score;
}

static FCWTGroupSizes pickGroupSizes(const std::vector<unsigned> &fwdChoices,
                                     const std::vector<unsigned> &bwdChoices,
                                     const Graph &graph,
                                     const Tensor &splitActivations,
                                     const Tensor &splitTranspose,
                                     const bool isJointPlan) {
  unsigned bestFwdIndex = 0;
  unsigned bestBwdIndex = 0;
  double bestScore = 0;
  // if is joint plan then planner will have accounted for this layer,
  // and so return the group sizes determined from the plan (which are
  // the first indices in each vector)
  if (isJointPlan) {
    return {fwdChoices[bestFwdIndex], bwdChoices[bestBwdIndex], false};
  }
  for (unsigned fwdIndex = 0; fwdIndex < fwdChoices.size(); ++fwdIndex) {
    for (unsigned bwdIndex = 0; bwdIndex < bwdChoices.size(); ++bwdIndex) {
      const auto score = blockScore(fwdChoices[fwdIndex], bwdChoices[bwdIndex],
                                    graph, splitActivations, splitTranspose);
      if (score > bestScore) {
        bestFwdIndex = fwdIndex;
        bestBwdIndex = bwdIndex;
        bestScore = score;
      }
    }
  }
  return {fwdChoices[bestFwdIndex], bwdChoices[bestBwdIndex], false};
}

static FCWTGroupSizes getGroupSizes(const Plan &fwdPlan, const Plan &bwdPlan,
                                    const Graph &graph,
                                    const Tensor &splitActivations,
                                    const Tensor &splitTranspose) {

  const auto possibleFwdGroupSizes = commonDivisors(
      fwdPlan.inChansPerGroup, static_cast<unsigned>(splitActivations.dim(3)));

  const auto possibleBwdGroupSizes = commonDivisors(
      bwdPlan.inChansPerGroup, static_cast<unsigned>(splitActivations.dim(2)));

  const auto isJointPlan = fwdPlan.isJointPlan && bwdPlan.isJointPlan;

  FCWTGroupSizes result =
      pickGroupSizes(possibleFwdGroupSizes, possibleBwdGroupSizes, graph,
                     splitActivations, splitTranspose, isJointPlan);

  result.useCopyImpl = result.fwdGroupSize == 1 || result.bwdGroupSize == 1 ||
                       planSwapsOperands(fwdPlan) || planSwapsOperands(bwdPlan);
  logging::trace("Transpose Group sizes fwd, bwd, useCopy = {} {} {}",
                 result.fwdGroupSize, result.bwdGroupSize, result.useCopyImpl);
  return result;
}

static Tensor fullyConnectedWeightTranspose(Graph &graph, Tensor activations,
                                            const CanonicalConvParams &params,
                                            Sequence &prog,
                                            const std::string &debugPrefix,
                                            const ConvOptions &options,
                                            PlanningCache *cache) {

  if (params->getNumFieldDims() != 1) {
    throw poputil::poplibs_error("fullyConnectedWeightTranspose() expects a 1-d"
                                 " convolution");
  }
  auto bwdPlan = getPlan(graph.getTarget(), params, options, cache);
  auto fwdPlan = getFullyConnectedFwdPlanFromBwdParams(graph.getTarget(),
                                                       params, options, cache);
  auto splitActivations = actsToInternalShape(
      activations, params->getNumConvGroups(), params->inputFieldShape.back());

  Tensor transposed =
      createInput(graph, bwdPlan, params, "transposed", options);
  auto splitTransposed =
      actsToInternalShape(transposed, params->getNumConvGroups(),
                          params->inputChannelsPerConvGroup);

  const auto groupSizes =
      getGroupSizes(fwdPlan, bwdPlan, graph, splitActivations, splitTransposed);
  const auto fwdGroupSize = groupSizes.fwdGroupSize;
  const auto bwdGroupSize = groupSizes.bwdGroupSize;

  if (groupSizes.useCopyImpl) {
    // In this case there is no benefit to using transpose vertices to
    // rearrange.
    return actsToExternalShape(splitActivations.dimShuffle({0, 1, 3, 2}));
  }

  auto splitTransposedUngroupedShape = splitTransposed.shape();
  const auto dType = activations.elementType();

  splitTransposed =
      getGroupedFCWeightsView(splitTransposed, fwdGroupSize, bwdGroupSize);

  auto firstInGroup =
      getFirstInGroup(splitActivations, bwdGroupSize, fwdGroupSize);

  auto blockTileMapping = graph.getTileMapping(firstInGroup);
  auto transposeCS = graph.addComputeSet(debugPrefix + "/Transpose");

  popops::rearrange::addTransposeVertices(
      graph, transposeCS, dType, bwdGroupSize, fwdGroupSize, blockTileMapping,
      [&](size_t index) {
        auto blockIndices =
            poputil::unflattenIndex(firstInGroup.shape(), index);
        return std::make_pair(splitActivations[blockIndices[0]][blockIndices[1]]
                                              [blockIndices[2]][blockIndices[3]]
                                                  .flatten(),
                              splitTransposed[blockIndices[0]][blockIndices[1]]
                                             [blockIndices[3]][blockIndices[2]]
                                                 .flatten());
      });
  prog.add(Execute(transposeCS));
  auto transposedWeights = splitTransposed.dimShufflePartial({3}, {4}).reshape(
      splitTransposedUngroupedShape);
  return actsToExternalShape(transposedWeights);
}

Tensor fullyConnectedWeightTranspose(Graph &graph, Tensor activations,
                                     const ConvParams &params_, Sequence &prog,
                                     const std::string &debugPrefix,
                                     const poplar::OptionFlags &options_,
                                     PlanningCache *cache) {
  const ConvOptions options(graph.getTarget(), options_);
  return fullyConnectedWeightTranspose(graph, activations, params_, prog,
                                       debugPrefix, options, cache);
}

void reportPlanInfo(std::ostream &out, const poplar::Graph &graph,
                    const ConvParams &params,
                    const poplar::OptionFlags &options_, PlanningCache *cache) {
  const ConvOptions options(graph.getTarget(), options_);
  auto plan = getPlan(graph.getTarget(), params, options, cache);
  if (options.pass != Pass::FC_TRAINING_WU &&
      options.pass != Pass::FC_TRAINING_BWD) {
    uint64_t cycles, memory;
    std::tie(cycles, memory) =
        estimateConvCost(graph.getTarget(), params, options, cache, plan);
    out << "  Estimated cost {cycles " << cycles << ", temporary bytes "
        << memory << "}\n";
  }
  out << plan;
}

void reportWeightUpdatePlanInfo(std::ostream &out, const Graph &graph,
                                const ConvParams &fwdParams,
                                const poplar::OptionFlags &fwdOptions,
                                PlanningCache *cache) {
  auto params = getWeightUpdateParams(fwdParams);
  auto options = fwdOptions;
  options.set("pass", "TRAINING_WU");
  // The weight update is equivalent to a convolution where:
  // - wu conv groups = fwd conv groups
  // - wu batch size = fwd input channels
  // - wu input channels = fwd batch size
  // - wu height = fwd height
  // - wu width = fwd width
  // - wu output channels = fwd output channels
  reportPlanInfo(out, graph, params, options, cache);
}

} // namespace poplin
