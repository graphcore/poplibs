// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "poplin/Convolution.hpp"
#include "poplin/ConvPreplan.hpp"

#include "CanonicalConvParams.hpp"
#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include "ConvProgramTree.hpp"
#include "ConvReduce.hpp"
#include "ConvTransforms.hpp"
#include "ConvUtilInternal.hpp"
#include "ConvValidation.hpp"
#include "ConvVertices.hpp"
#include "ConvolutionInternal.hpp"
#include "PerformanceEstimation.hpp"
#include "poplar/CycleCount.hpp"
#include "poplibs_support/Algorithms.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/VectorUtils.hpp"
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
#include "poputil/DebugInfo.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <boost/optional.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace poputil {
template <>
poplar::ProfileValue toProfileValue(const poplin::PlanningCache &t) {
  return poplar::ProfileValue("<PlanningCache>");
}
} // namespace poputil

namespace poplin {

static Tensor createInputImpl(Graph &graph, const CanonicalConvParams &params,
                              const Tensor &metadata, unsigned level,
                              bool serial,
                              const std::vector<Split<ConvIndices>> &indices,
                              const DebugNameAndId &dnai, const Plan &plan,
                              const ConvOptions &options);
static Tensor createWeightsImpl(Graph &graph, const CanonicalConvParams &params,
                                const Tensor &metadata, unsigned level,
                                bool serial,
                                const std::vector<Split<ConvIndices>> &indices,
                                const DebugNameAndId &dnai, const Plan &plan,
                                const ConvOptions &options);

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

template <typename T>
static std::string getShapeAsString(const std::vector<T> &shape) {
  return shape.empty() ? std::string()
                       : std::accumulate(std::next(shape.begin()), shape.end(),
                                         std::to_string(shape[0]),
                                         [](std::string a, unsigned b) {
                                           return a + "x" + std::to_string(b);
                                         });
}

static void verifyInputShapesAndTypes(const CanonicalConvParams &params,
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
  if (in.elementType() != weights.elementType()) {
    throw poputil::poplibs_error(
        "Convolution/MatMul input types must match.  Tensors were  "
        " provided with types " +
        in.elementType().toString() + " and " +
        weights.elementType().toString());
  }
}

static unsigned getConvGroupsPerGroup(const Plan &plan,
                                      unsigned numConvGroups) {
  return std::gcd(plan.convGroupsPerGroup, numConvGroups);
}

static unsigned getInChansPerGroup(const Plan &plan, unsigned level,
                                   unsigned numInChans) {
  auto grainSize = plan.inChansPerGroup;
  if (level < plan.partitions.size()) {
    grainSize = plan.partitions[level].inChanGrainSize;
  }

  if (numInChans < grainSize) {
    return std::gcd(numInChans, grainSize);
  }

  return grainSize;
}

static unsigned getInChansPerSerialSplit(const Partition &partition,
                                         unsigned numInChans) {
  if (partition.inChanSplit.serial == 1) {
    return numInChans;
  }

  // Align the split size with the grain size
  const auto grainSize =
      std::min(partition.inChanGrainSize, std::max(numInChans, 1u));
  const auto numGrains = gccs::ceildiv(numInChans, grainSize);
  const auto numGrainsPerSplit =
      gccs::ceildiv(numGrains, partition.inChanSplit.serial);
  return numGrainsPerSplit * grainSize;
}

static unsigned getOutChansPerGroup(const Plan &plan, unsigned level,
                                    unsigned numOutChans) {
  auto grainSize = plan.partialChansPerGroup;
  if (level < plan.partitions.size()) {
    grainSize = plan.partitions[level].outChanGrainSize;
  }

  if (numOutChans < grainSize) {
    return std::gcd(numOutChans, grainSize);
  }

  return grainSize;
}

static unsigned getOutChansPerSerialSplit(const Partition &partition,
                                          unsigned numOutChans) {
  if (partition.outChanSplit.serial == 1) {
    return numOutChans;
  }

  // Align the split size with the grain size
  const auto grainSize =
      std::min(partition.outChanGrainSize, std::max(numOutChans, 1u));
  const auto numGrains = gccs::ceildiv(numOutChans, grainSize);
  const auto numGrainsPerSplit =
      gccs::ceildiv(numGrains, partition.outChanSplit.serial);
  return numGrainsPerSplit * grainSize;
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
  assert(indices.size() == numLevels - 1);
  assert(plan.partitions.size() == numLevels - 1);
  unsigned tile = 0;
  const auto &levelIndices = indices[0].parallel;
  const auto &levelPartition = plan.partitions[0];
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
  const auto linearizedIndex = linearizeConvIndices(
      fwdOutIndices, fwdKernelIndices, fwdic, fwdb, fwdoc, fwdcg, fwdFieldSplit,
      fwdKernelSplit, fwdInChanSplit, fwdBatchSplit, fwdOutChanSplit);
  tile = tile * target.getNumTiles() + linearizedIndex;

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

static unsigned getParallelSplitBegin(unsigned index, unsigned dimSize,
                                      unsigned splitCount) {
  assert(index <= splitCount);
  const auto splitSize = dimSize / splitCount;
  const auto remainder = dimSize % splitCount;
  // Handle splits of dimensions that aren't multiples of the split count
  // by including an extra element in the last $remainder splits.
  // This ensures that if this is being used to calculate split positions
  // wrt grains rather than individual elements, and the dimension size also
  // isn't a multiple of the grain size, then we don't end up with an especially
  // short last split.
  return index * splitSize + (index - std::min(index, splitCount - remainder));
}

static std::pair<unsigned, unsigned>
getTileOutRange(const CanonicalConvParams &params, const Partition &partition,
                unsigned tileIndex, unsigned dim) {
  const auto outSize = params->getOutputSize(dim);
  const auto grainSize = partition.fieldAxisGrainSize[dim];
  const auto numGrains = gccs::ceildiv(outSize, grainSize);
  const auto split = partition.fieldSplit[dim];

  const auto outGrainBegin = getParallelSplitBegin(tileIndex, numGrains, split);
  const auto outGrainEnd =
      getParallelSplitBegin(tileIndex + 1, numGrains, split);

  const auto outBegin = outGrainBegin * grainSize;
  const auto outEnd =
      (tileIndex + 1 == split) ? outSize : (outGrainEnd * grainSize);

  return {outBegin, outEnd};
}

/// Compute the sub-convolution corresponding to the specified slice of a larger
/// convolution. The parameters and tensors are updated in place to
/// the parameters and tensors for the sub-convolution.
CanonicalConvParams getSubConvolution(const ConvSlice &slice,
                                      const CanonicalConvParams &originalParams,
                                      Tensor *in, Tensor *weights) {
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

void iteratePartitionParallel(
    const CanonicalConvParams &params, const Partition &partition,
    const std::function<void(const ConvIndices &, const ConvSlice &)> &f) {
  const auto numFieldDims = params->getNumFieldDims();

  const unsigned numOutChans = params->getNumOutputChansPerConvGroup();
  const auto outChanGrainSize = partition.outChanGrainSize;
  const auto outChanNumGrains = gccs::ceildiv(numOutChans, outChanGrainSize);
  const auto outChanSplit = partition.outChanSplit;

  const auto batchSplit = partition.batchSplit;
  const unsigned batchSize = params->getBatchSize();

  const unsigned numInChans = params->getNumInputChansPerConvGroup();
  const auto inChanGrainSize = partition.inChanGrainSize;
  const auto inChanNumGrains = gccs::ceildiv(numInChans, inChanGrainSize);
  const auto inChanSplit = partition.inChanSplit;

  const unsigned numConvGroups = params->getNumConvGroups();
  const auto convGroupGrainSize = partition.convGroupGrainSize;
  const auto convGroupNumGrains =
      gccs::ceildiv(numConvGroups, convGroupGrainSize);
  const auto convGroupSplit = partition.convGroupSplit;

  const auto totalFieldSplit = product(partition.fieldSplit);
  const auto totalKernelSplit = product(partition.kernelSplit);

  for (unsigned cg = 0; cg != convGroupSplit; ++cg) {
    const auto convGroupGrainBegin =
        getParallelSplitBegin(cg, convGroupNumGrains, convGroupSplit);
    const auto convGroupGrainEnd =
        getParallelSplitBegin(cg + 1, convGroupNumGrains, convGroupSplit);

    const auto cgBegin = convGroupGrainBegin * convGroupGrainSize;
    const auto cgEnd = (cg + 1 == convGroupSplit)
                           ? numConvGroups
                           : (convGroupGrainEnd * convGroupGrainSize);

    for (unsigned b = 0; b != batchSplit; ++b) {
      const auto batchBegin = getParallelSplitBegin(b, batchSize, batchSplit);
      const auto batchEnd = getParallelSplitBegin(b + 1, batchSize, batchSplit);

      for (unsigned ic = 0; ic != inChanSplit.parallel; ++ic) {
        const auto inChanGrainBegin =
            getParallelSplitBegin(ic, inChanNumGrains, inChanSplit.parallel);
        const auto inChanGrainEnd = getParallelSplitBegin(
            ic + 1, inChanNumGrains, inChanSplit.parallel);

        const auto inChanBegin = inChanGrainBegin * inChanGrainSize;
        const auto inChanEnd = (ic + 1 == inChanSplit.parallel)
                                   ? numInChans
                                   : (inChanGrainEnd * inChanGrainSize);

        for (unsigned k = 0; k != totalKernelSplit; ++k) {
          auto kernelIndices = unflattenIndex(partition.kernelSplit, k);
          std::vector<unsigned> kernelBegin(numFieldDims),
              kernelEnd(numFieldDims);
          for (unsigned dim = 0; dim != numFieldDims; ++dim) {
            const auto kernelSize = params->kernelShape[dim];
            kernelBegin[dim] = getParallelSplitBegin(
                kernelIndices[dim], kernelSize, partition.kernelSplit[dim]);
            kernelEnd[dim] = getParallelSplitBegin(
                kernelIndices[dim] + 1, kernelSize, partition.kernelSplit[dim]);
          }

          for (unsigned oc = 0; oc != outChanSplit.parallel; ++oc) {
            const auto outChanGrainBegin = getParallelSplitBegin(
                oc, outChanNumGrains, outChanSplit.parallel);
            const auto outChanGrainEnd = getParallelSplitBegin(
                oc + 1, outChanNumGrains, outChanSplit.parallel);

            const auto outChanBegin = outChanGrainBegin * outChanGrainSize;
            const auto outChanEnd = (oc + 1 == outChanSplit.parallel)
                                        ? numOutChans
                                        : (outChanGrainEnd * outChanGrainSize);

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
  const auto outChansPerSplit =
      getOutChansPerSerialSplit(partition, numOutChans);
  const auto inChansPerSplit = getInChansPerSerialSplit(partition, numInChans);

  std::vector<unsigned> zeroSpatialIndices(numFieldDims, 0);
  std::vector<unsigned> outFieldEnd(numFieldDims);
  std::vector<unsigned> kernelEnd(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    outFieldEnd[dim] = params->getOutputSize(dim);
    kernelEnd[dim] = params->kernelShape[dim];
  }

  for (unsigned ic = 0; ic != inChanSplit.serial; ++ic) {
    const auto inChanBegin = ic * inChansPerSplit;
    const auto inChanEnd = std::min(inChanBegin + inChansPerSplit, numInChans);
    for (unsigned oc = 0; oc != outChanSplit.serial; ++oc) {
      const auto outChanBegin = oc * outChansPerSplit;
      const auto outChanEnd =
          std::min(outChanBegin + outChansPerSplit, numOutChans);
      f({0, 0, zeroSpatialIndices, oc, ic, zeroSpatialIndices},
        {0, numConvGroups, 0, batchSize, zeroSpatialIndices, outFieldEnd,
         outChanBegin, outChanEnd, inChanBegin, inChanEnd, zeroSpatialIndices,
         kernelEnd});
    }
  }
}

static void
regroupIfBeneficialForPlan(Graph &graph, const ConvParams &params,
                           const Plan &plan, unsigned level, Tensor &in,
                           bool isActs,
                           ConvProgramTree::TransformPreProgram *rearrangeProg,
                           const DebugNameAndId &dnai) {
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
    auto &preTranspose = isActs ? rearrangeProg->preTransposeActs
                                : rearrangeProg->preTransposeWeights;
    auto &transposeCS = isActs ? rearrangeProg->transposeCSActs.back()
                               : rearrangeProg->transposeCSWeights.back();
    in = popops::rearrange::regroupTensor(graph, in, preTranspose, transposeCS,
                                          grouping[0], destGrouping[0], {dnai});
  } else {
    logging::poplin::debug("regroupIfBeneficialForPlan: No plan found for "
                           "debugStr={}, tensorStr={}, isActs={}",
                           dnai.getPathName(), in.getDebugStr(), isActs);
  }
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
    std::vector<Tensor> *rearrangeWritten = nullptr, bool rearrangeActs = false,
    bool rearrangeWeights = false, const DebugNameAndId &dnai = {}) {
  if (rearrangeActs) {
    logging::poplin::debug("'{}': forcing rearrangement of activations",
                           dnai.getPathName());
  }

  if (rearrangeWeights) {
    logging::poplin::debug("'{}': forcing rearrangement of weights",
                           dnai.getPathName());
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
        expandSpatialDim(params, *it, graph, acts, weights, {dnai});
        it = transform.expandDims.erase(it);
      } else {
        ++it;
      }
    }
  } else {
    // implement the expandDims transformation unless it's at tile level in
    // which case it should be deferred to vertex level.
    if (level != tileLevel) {
      expandSpatialDims(params, plan, level, graph, acts, weights,
                        rearrangeProg, rearrangeActs, rearrangeWeights, {dnai});
      transform.expandDims.clear();
    }

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
        expandSpatialDim(params, dim, graph, maybeActs, maybeWeights, {dnai});
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
          gccs::alignNext(params.numConvGroups, std::size_t(factor));
      const auto extraConvGroups = paddedNumConvGroups - numConvGroups;

      // pad conv groups if necessary.
      if (extraConvGroups != 0) {
        if (acts) {
          *acts = popops::pad(graph, *acts, 0, extraConvGroups, 0);
        }

        if (weights) {
          *weights = popops::pad(graph, *weights, 0, extraConvGroups, 0);
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
        paddedCi.reserve(weights->dim(0));

        for (unsigned co = 0; co < weights->dim(0); ++co) {
          auto x = (co / numOutChans) % factor;
          auto paddingLower = x * numInChans;
          auto paddingUpper = (factor - 1 - x) * numInChans;

          // pad the input channel dim, which is currently dim 0 if we index by
          // output channel.
          paddedCi.push_back(popops::pad(graph, (*weights)[co], paddingLower,
                                         paddingUpper, 0, 0));

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
        gccs::alignNext(params.getNumConvGroups(), convGroupGrainSize);
    const auto paddedInChans =
        gccs::alignNext(params.getNumInputChansPerConvGroup(), inChanGrainSize);
    const auto paddedOutChans = gccs::alignNext(
        params.getNumOutputChansPerConvGroup(), outChanGrainSize);

    if (acts) {
      const unsigned gDim = 0;
      const unsigned ciDim = acts->rank() - 1;

      *acts = popops::pad(graph, *acts, 0,
                          paddedConvGroups - params.getNumConvGroups(), gDim);
      *acts = popops::pad(graph, *acts, 0,
                          paddedInChans - params.getNumInputChansPerConvGroup(),
                          ciDim);
    }

    if (weights) {
      const unsigned gDim = 0;
      const unsigned coDim = weights->rank() - 2;
      const unsigned ciDim = weights->rank() - 1;

      *weights =
          popops::pad(graph, *weights, 0,
                      paddedConvGroups - params.getNumConvGroups(), gDim);
      *weights = popops::pad(
          graph, *weights, 0,
          paddedInChans - params.getNumInputChansPerConvGroup(), ciDim);
      *weights = popops::pad(
          graph, *weights, 0,
          paddedOutChans - params.getNumOutputChansPerConvGroup(), coDim);
    }

    params.numConvGroups = paddedConvGroups;
    params.inputChannelsPerConvGroup = paddedInChans;
    params.outputChannelsPerConvGroup = paddedOutChans;
  }

  if (acts && rearrangeActs) {
    regroupIfBeneficialForPlan(graph, params, plan, level, *acts, true,
                               rearrangeProg, {dnai});

    auto actsRearranged =
        createInputImpl(graph, params, acts.get().getMetadata(), level, serial,
                        indices, {dnai, "actsRearranged"}, plan, options);

    assert(rearrangeProg);
    rearrangeProg->postTransposeActs.emplace_back(*acts, actsRearranged, false,
                                                  dnai);
    rearrangeWritten->emplace_back(actsRearranged);
    *acts = actsRearranged;
  }

  if (weights && rearrangeWeights) {
    regroupIfBeneficialForPlan(graph, params, plan, level, *weights, false,
                               rearrangeProg, {dnai});

    auto weightsRearranged = createWeightsImpl(
        graph, params, weights.get().getMetadata(), level, serial, indices,
        {dnai, "weightsRearranged"}, plan, options);

    assert(rearrangeProg);
    rearrangeProg->postTransposeWeights.emplace_back(
        *weights, weightsRearranged, false, dnai);
    rearrangeWritten->emplace_back(weightsRearranged);
    *weights = weightsRearranged;
  }

  return params;
}

static CanonicalConvParams convolutionPreprocess(
    Graph &graph, const ConvParams &params, const ConvOptions &options,
    Plan &plan, unsigned level, const std::vector<Split<ConvIndices>> &indices,
    Tensor &acts, Tensor &weights, bool serial,
    ConvProgramTree::TransformPreProgram *rearrangeProg = nullptr,
    std::vector<Tensor> *rearrangeWritten = nullptr, bool rearrangeActs = false,
    bool rearrangeWeights = false, const DebugNameAndId &dnai = {}) {
  auto actsOptional = boost::make_optional(acts);
  auto weightsOptional = boost::make_optional(weights);
  const auto newParams = convolutionPreprocess(
      graph, params, options, plan, level, indices, actsOptional,
      weightsOptional, serial, rearrangeProg, rearrangeWritten, rearrangeActs,
      rearrangeWeights, dnai);
  acts = *actsOptional;
  weights = *weightsOptional;
  return newParams;
}

CanonicalConvParams
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
                                     const DebugNameAndId &dnai) {
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
        activationsView = dilate(graph, activationsView, dilation, dim, {dnai});
        mappingView = dilateWithNearestNeighbour(mappingView, dilation, dim);
        activationsView = popops::pad(graph, activationsView, paddingLower,
                                      paddingUpper, dim);
        // pad with nearest neighbour.
        mappingView = popops::pad(mappingView, paddingLower, paddingUpper, dim,
                                  popops::padding::Type::EDGE);
      }
      assert(activationsView.shape() == mappingView.shape());
      activationsView = splitActivationIntoGroups(
          activationsView, convGroupsPerGroup, outChansPerGroup);
      mappingView = splitActivationIntoGroups(mappingView, convGroupsPerGroup,
                                              outChansPerGroup);
      activations = graph.addVariable(activationsView.elementType(),
                                      activationsView.shape(),
                                      {dnai, "activationsPostDilate"});
      graph.setTileMapping(activations, graph.getTileMapping(mappingView));
      transformPost.emplace_back(activationsView, activations, false, dnai);
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
    activations = popops::pad(graph, activations, 0,
                              -static_cast<int>(convGroupPadding), 0);

    assert(activations.dim(activations.rank() - 1) >=
           postCombineConvGroupsParams.outputChannelsPerConvGroup);
    const auto outChanPadding =
        activations.dim(activations.rank() - 1) -
        postCombineConvGroupsParams.outputChannelsPerConvGroup;
    activations =
        popops::pad(graph, activations, 0, -static_cast<int>(outChanPadding),
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
        activations = popops::pad(graph, activations, 0, -convGroupPadding, 0);
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

static void iterateUsageByPartition(
    Graph &graph, CanonicalConvParams params, Plan plan, unsigned level,
    bool serial, boost::optional<Tensor> acts, boost::optional<Tensor> weights,
    const std::vector<Split<ConvIndices>> &indices, unsigned grainSize,
    unsigned minElementsPerTile, const ConvOptions &options, unsigned startTile,
    bool ascendingMapping, TensorUseTracker &tracker) {
  // Pre-process prior to the parallel partition at this level.
  params = convolutionPreprocess(graph, params.releaseParams(), options, plan,
                                 level, indices, acts, weights, serial);
  const auto numTiles = graph.getTarget().getNumTiles();

  // TODO: T12870 Where it is known that partitioning does not cause elements of
  // either the inputs or weights to be used on multiple tiles, this should
  // skip calculating the mapping for all but the first serial (and parallel?)
  // slice and reuse the mapping across each slice to save compile time.

  if (level == plan.partitions.size()) {
    const auto &target = graph.getTarget();
    auto tile = linearizeTileIndices(target, options, indices, plan);
    if (weights) {
      tile = transformTileIndex(tile, numTiles, startTile, ascendingMapping);
    }
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

            TensorUseTracker serialTracker(numTiles, startTile,
                                           ascendingMapping);
            iterateUsageByPartition(graph, subParams, plan, level, false,
                                    subActs, subWeights, subIndices, grainSize,
                                    minElementsPerTile, options, startTile,
                                    ascendingMapping, serialTracker);

            if (totalSerialSplit == 1) {
              // N.B. we do not resolve usage if there is no serial splitting.
              tracker = std::move(serialTracker);
            } else {
              serialTracker.resolve(
                  graph, grainSize, minElementsPerTile, false,
                  TensorUseTracker::MappingMethod::OptimizeHaloRegions);
              tracker.add(std::move(serialTracker));
            }
          });
    } else {
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
            iterateUsageByPartition(graph, subParams, plan, level + 1, true,
                                    subActs, subWeights, subIndices, grainSize,
                                    minElementsPerTile, options, startTile,
                                    ascendingMapping, tracker);
          });
    }
  }
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

  const auto [startTile, ascendingMapping] = [&]() {
    if (acts) {
      // Allow the planned dithering to be applied to the activations
      return std::make_pair(0u, true);
    }
    bool ascendingPlanMapping =
        plan.linearizeTileDirection == Plan::LinearizeTileDirection::ASCENDING;
    bool respectDithering = !(ascendingPlanMapping && plan.startTile == 0);
    if (respectDithering) {
      // The plan includes dithering so allocate weights consistently with that
      // dithering
      return std::make_pair(plan.startTile, ascendingPlanMapping);
    } else {
      // The plan contains no dithering - allocate from the end of the set of
      // tiles used by the plan
      return std::make_pair(
          graph.getTarget().getNumTiles() - (plan.totalTiles() - 1), false);
    }
  }();

  const auto numTiles = graph.getTarget().getNumTiles();
  TensorUseTracker tracker(numTiles, startTile, ascendingMapping);
  iterateUsageByPartition(graph, params, plan, level, serial, acts, weights,
                          indices, grainSize, minElementsPerTile, options,
                          startTile, ascendingMapping, tracker);

  return tracker;
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
                              const Tensor &metadata, unsigned level,
                              bool serial,
                              const std::vector<Split<ConvIndices>> &indices,
                              const DebugNameAndId &dnai, const Plan &plan,
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
    auto t = createWeightsImpl(graph, newParams, metadata, level, serial,
                               indices, {dnai}, newPlan, options);
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
    auto t = createInputImpl(graph, newParams, metadata, level, serial, indices,
                             {dnai}, newPlan, options);
    return convolutionPreprocessInverse(graph, originalParams, options, plan,
                                        level, indices, t, true /* isActs */,
                                        serial);
  }
  const auto numConvGroups = params->getNumConvGroups();
  const auto convGroupsPerGroup = getConvGroupsPerGroup(plan, numConvGroups);
  assert(numConvGroups % convGroupsPerGroup == 0);
  const auto numConvGroupGroups = numConvGroups / convGroupsPerGroup;

  const auto numInChans = params->getNumInputChansPerConvGroup();
  auto inChansPerSerialSplit = numInChans;

  unsigned inChanSerialSplit = 1;
  if (serial && level < plan.partitions.size()) {
    const auto &partition = plan.partitions[level];
    inChanSerialSplit = partition.inChanSplit.serial;
    inChansPerSerialSplit = getInChansPerSerialSplit(partition, numInChans);
  }

  const auto inChansPerGroup =
      getInChansPerGroup(plan, level, inChansPerSerialSplit);
  const auto inChanGroupsPerSerialSplit =
      gccs::ceildiv(inChansPerSerialSplit, inChansPerGroup);

  // Create an initial tensor with groupings set up using the dimensions
  // calculated above. Note that e.g. the number of input channel groups
  // may not be a factor of the requested number of input channels, and
  // so the extra channnels will need trimming.
  std::vector<std::size_t> tensorShape = {
      inChanSerialSplit,
      numConvGroupGroups,
      inChanGroupsPerSerialSplit,
      params->getBatchSize(),
  };
  tensorShape.insert(tensorShape.end(), params->inputFieldShape.begin(),
                     params->inputFieldShape.end());
  tensorShape.push_back(convGroupsPerGroup);
  tensorShape.push_back(inChansPerGroup);
  auto t = graph.addVariable(params->inputType, metadata, tensorShape, {dnai});

  // Clone to make sliced regions contiguous on each tile respecting existing
  // grain size etc.
  bool returnClone = inChanSerialSplit > 1;

  // Ensure the entire tensor is mapped to prevent incomplete mapping errors.
  // The parts that are actually used will be remapped to something more
  // sensible after the padding is trimmed.
  graph.setTileMapping(t, 0);

  // Flatten out the serial split dimension
  t = t.dimRoll(0, 1).flatten(1, 3);

  t = unsplitActivationFromGroups(t);

  if (t.dim(t.rank() - 1) != numInChans) {
    // Trim any channels added to bring channels per convolution group up to
    // a multiple of the grain sizes and/or split counts.
    t = t.slice(0, numInChans, t.rank() - 1);

    returnClone = true;
  }

  mapActivations(graph, params, plan, level, serial, indices, t, options);

  if (returnClone) {
    t = graph.clone(
        t.getMetadata(), t, {dnai},
        TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);
  }

  return t;
}

Tensor createInput(Graph &graph, const Plan &plan,
                   const CanonicalConvParams &params, const Tensor &metadata,
                   const DebugNameAndId &dnai, const ConvOptions &options) {
  const unsigned level = 0;
  bool serial = true;
  const std::vector<Split<ConvIndices>> indices;
  auto input = createInputImpl(graph, params, metadata, level, serial, indices,
                               {dnai}, plan, options);
  input = actsToExternalShape(input);
  return input;
}

Tensor createInput(Graph &graph, const ConvParams &params_,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options_, PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params_, options_, cache), "createInput");

  const CanonicalConvParams params(params_);
  const ConvOptions options(options_);

  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  auto output = createInput(graph, plan, params, {}, {di}, options);
  di.addOutput(output);
  return output;
}

static Tensor createWeightsImpl(Graph &graph, const CanonicalConvParams &params,
                                const Tensor &metadata, unsigned level,
                                bool serial,
                                const std::vector<Split<ConvIndices>> &indices,
                                const DebugNameAndId &dnai, const Plan &plan,
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
    auto t = createInputImpl(graph, newParams, metadata, level, serial, indices,
                             {dnai}, newPlan, options);
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
    auto t = createWeightsImpl(graph, newParams, metadata, level, serial,
                               indices, {dnai}, newPlan, options);
    return convolutionPreprocessInverse(graph, originalParams, options, plan,
                                        level, indices, t, false /* isActs */,
                                        serial);
  }

  const auto numConvGroups = params->getNumConvGroups();
  const auto convGroupsPerGroup = getConvGroupsPerGroup(plan, numConvGroups);
  assert(numConvGroups % convGroupsPerGroup == 0);
  const auto numConvGroupGroups = numConvGroups / convGroupsPerGroup;

  const auto numOutChans = params->getNumOutputChansPerConvGroup();
  unsigned outChanSerialSplit = 1;
  auto outChansPerSerialSplit = numOutChans;

  const auto numInChans = params->getNumInputChansPerConvGroup();
  unsigned inChanSerialSplit = 1;
  auto inChansPerSerialSplit = numInChans;

  if (serial && (level < plan.partitions.size())) {
    const auto &partition = plan.partitions[level];

    inChanSerialSplit = partition.inChanSplit.serial;
    inChansPerSerialSplit = getInChansPerSerialSplit(partition, numInChans);

    outChanSerialSplit = partition.outChanSplit.serial;
    outChansPerSerialSplit = getOutChansPerSerialSplit(partition, numOutChans);
  }
  const auto totalSerialSplit = outChanSerialSplit * inChanSerialSplit;

  const auto outChansPerGroup =
      getOutChansPerGroup(plan, level, outChansPerSerialSplit);
  const auto outChanGroupsPerSerialSplit =
      gccs::ceildiv(outChansPerSerialSplit, outChansPerGroup);

  const auto inChansPerGroup =
      getInChansPerGroup(plan, level, inChansPerSerialSplit);
  const auto inChanGroupsPerSerialSplit =
      gccs::ceildiv(inChansPerSerialSplit, inChansPerGroup);

  // Clone to make sliced regions contiguous on each tile respecting existing
  // grain size etc.
  bool returnClone = totalSerialSplit > 1;

  // Create an initial tensor with groupings set up using the dimensions
  // calculated above. Note that e.g. the number of input channel groups
  // may not be a factor of the requested number of input channels, and
  // so the extra channels will need trimming.
  std::vector<std::size_t> weightsShape = {
      outChanSerialSplit, inChanSerialSplit, numConvGroupGroups,
      outChanGroupsPerSerialSplit, inChanGroupsPerSerialSplit};
  weightsShape.insert(weightsShape.end(), params->kernelShape.begin(),
                      params->kernelShape.end());
  weightsShape.push_back(convGroupsPerGroup);
  weightsShape.push_back(outChansPerGroup);
  weightsShape.push_back(inChansPerGroup);
  auto weights =
      graph.addVariable(params->inputType, metadata, weightsShape, {dnai});

  // Ensure the entire tensor is mapped to prevent incomplete mapping errors.
  // The parts that are actually used will be remapped to something more
  // sensible after the padding is trimmed.
  graph.setTileMapping(weights, 0);

  // Flatten out the serial split dimension
  weights = weights.dimRoll(0, 2).flatten(2, 4);
  weights = weights.dimRoll(0, 2).flatten(2, 4);

  weights = unsplitWeightsFromGroups(weights);

  if (weights.dim(weights.rank() - 2) != numOutChans ||
      weights.dim(weights.rank() - 1) != numInChans) {
    // Trim any channels added to bring channels per convolution group up to
    // a multiple of the grain sizes and/or split counts.
    weights = weights.slice(0, numOutChans, weights.rank() - 2);
    weights = weights.slice(0, numInChans, weights.rank() - 1);

    returnClone = true;
  }

  mapWeights(graph, params, plan, level, serial, indices, weights, options);

  if (returnClone) {
    weights = graph.clone(
        weights.getMetadata(), weights, {dnai},
        TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);
  }

  return weights;
}

Tensor createWeights(Graph &graph, const Plan &plan,
                     const CanonicalConvParams &params, const Tensor &metadata,
                     const DebugNameAndId &dnai, const ConvOptions &options) {
  const unsigned level = 0;
  bool serial = true;
  const std::vector<Split<ConvIndices>> indices;
  auto weights = createWeightsImpl(graph, params, metadata, level, serial,
                                   indices, {dnai}, plan, options);
  return weightsToExternalShape(weights);
}

Tensor createWeights(Graph &graph, const ConvParams &params_,
                     const poplar::DebugContext &debugContext,
                     const poplar::OptionFlags &options_,
                     PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(params_, options_, cache), "createWeights");

  const CanonicalConvParams params(params_);
  const ConvOptions options(options_);

  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  auto output = createWeights(graph, plan, params, {}, {di}, options);
  di.addOutput(output);
  return output;
}

static void mapBiases(poplar::Graph &graph, const poplar::Tensor &biases,
                      const poplar::Tensor &out,
                      const boost::optional<Plan> &plan) {
  const auto &target = graph.getTarget();
  const auto dType = out.elementType();
  const auto grainSize = target.getVectorWidth(dType);
  // Limit the minimum number of bias bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto dTypeSize = target.getTypeSize(dType);
  const auto minBytesPerTile = 8;
  const auto minElementsPerTile = (minBytesPerTile + dTypeSize - 1) / dTypeSize;

  if (out.numElements() == 0) {
    mapTensorLinearly(graph, biases, minElementsPerTile, grainSize);
  }
  const auto numTiles = target.getNumTiles();
  // Create a view of the output where channels are the outermost dimension.
  auto outRegrouped = out.dimShufflePartial({out.rank() - 1}, {1})
                          .flatten(2, out.rank())
                          .flatten(0, 2);
  auto outMapping = graph.getTileMapping(outRegrouped);

  // If there is a plan with dithered tile allocation then respect that -
  // otherwise allocate the biases at the highest tile on which they will be
  // used
  auto highestTileWithDataAllocation =
      [](const std::vector<std::vector<Interval>> &mapping) {
        for (unsigned i = mapping.size() - 1; i != 0; i--) {
          if (mapping[i].size()) {
            return i;
          }
        }
        return 0u;
      };

  const bool respectDithering = [&]() {
    if (plan) {
      return plan.get().startTile != 0 ||
             plan.get().linearizeTileDirection ==
                 Plan::LinearizeTileDirection::DESCENDING;
    }
    return false;
  }();
  const unsigned startTile =
      respectDithering ? plan.get().startTile
                       : numTiles - highestTileWithDataAllocation(outMapping);
  const bool ascendingMapping =
      respectDithering ? plan.get().linearizeTileDirection ==
                             Plan::LinearizeTileDirection::ASCENDING
                       : false;
  TensorUseTracker useTracker(numTiles, startTile, ascendingMapping);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    for (const auto &interval : outMapping[tile]) {
      unsigned chanBegin = interval.begin() / outRegrouped.dim(1);
      unsigned chanEnd =
          (interval.end() + outRegrouped.dim(1) - 1) / outRegrouped.dim(1);

      auto tileIdx = transformTileIndex(tile, target.getNumTiles(), startTile,
                                        ascendingMapping);
      useTracker.add(graph, tileIdx, biases.slice(chanBegin, chanEnd));
    }
  }

  useTracker.mapTensorsByUse(graph, grainSize, minElementsPerTile,
                             true /* extendPartialUsage */);
}

static Tensor createBiases(Graph &graph, const Tensor &acts_,
                           const boost::optional<Plan> &plan,
                           const DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts_), "createBiases");
  const auto acts = actsToInternalShape(acts_, 1, acts_.dim(1));
  const auto numOutChans = acts.dim(acts.rank() - 1);
  const auto dType = acts.elementType();
  auto biases = graph.addVariable(dType, {numOutChans}, {di});
  mapBiases(graph, biases, acts, plan);
  di.addOutput(biases);
  return biases;
}

poplar::Tensor createBiases(poplar::Graph &graph, const Tensor &acts,
                            const poplar::DebugContext &debugContext) {
  return createBiases(graph, acts, boost::none, debugContext);
}

poplar::Tensor createBiases(poplar::Graph &graph, const Tensor &acts,
                            const ConvParams &params,
                            const poplar::DebugContext &debugContext,
                            const poplar::OptionFlags &options,
                            PlanningCache *cache) {
  const auto plan = getPlan(graph.getTarget(), params, options, cache);
  return createBiases(graph, acts, plan, debugContext);
}

Tensor sliceOutput(const Tensor &out, const ConvSlice &slice,
                   const unsigned convGroupsPerGroup,
                   const unsigned outChansPerGroup) {
  // shape of out is [G1][OC1][N]...[G2][OC2]
  std::vector<std::size_t> begin, end;

  const auto numFieldDims = slice.outFieldBegin.size();
  begin.reserve(5 + numFieldDims);
  end.reserve(5 + numFieldDims);

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

void ConvProgramTree::ComputeSetsGroup::lower(
    Sequence &prog, const poplar::DebugNameAndId &dnai) {
  if (pre) {
    prog.add(Execute(pre.get(), {dnai}));
  }
  prog.add(Execute(convolveCS, {dnai}));

  for (auto &p : postProg) {
    logging::poplin::debug(
        "#convolution post program bunch copies of type {} = {}",
        p.first.toString(), p.second.first.size());
    assert(p.second.first.size());
    prog.add(
        Copy(concat(p.second.first), concat(p.second.second), false, {dnai}));
  }
  if (post) {
    prog.add(Execute(post.get(), {dnai}));
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
  outShape.reserve(outShape.size() + numFieldDims + 2);
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
        auto subInput = input ? *input : Tensor();
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
    *input = refactorSplitDimToOutermost(*input, input->rank() - 1,
                                         partition.inChanSplit.serial);
  }
  if (weights) {
    *weights =
        stitchSerialSlices(weightsSlices, false /* !isActs */, partition);
    *weights = refactorSplitDimToOutermost(*weights, weights->rank() - 1,
                                           partition.inChanSplit.serial);
    *weights = refactorSplitDimToOutermost(*weights, weights->rank() - 2,
                                           partition.outChanSplit.serial);
  }
  return std::make_tuple(parallelParams, indices);
}

static void add(Sequence &prog, const std::vector<Copy> &copies) {
  for (const auto &copy : copies) {
    prog.add(copy);
  }
}

static void add(Sequence &prog, const std::vector<ComputeSet> &css,
                const poplar::DebugNameAndId &dnai) {
  for (const auto &cs : css) {
    prog.add(Execute(cs, {dnai}));
  }
}

ConvProgramTree::TransformPreProgram::TransformPreProgram(
    Graph &graph, const poplar::DebugNameAndId &dnai) {
  transposeCSActs.emplace_back(graph.addComputeSet({dnai, "Acts"}));
  transposeCSWeights.emplace_back(graph.addComputeSet({dnai, "Weights"}));
}

void ConvProgramTree::TransformPreProgram::lower(
    poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  // TODO: make a map of type and concat same types together.
  for (const auto &t : writeUndef) {
    prog.add(WriteUndef(t, {dnai}));
  }

  add(prog, preTransposeActs);
  add(prog, preTransposeWeights);
  add(prog, transposeCSActs, dnai);
  add(prog, transposeCSWeights, dnai);
  add(prog, postTransposeActs);
  add(prog, postTransposeWeights);
  add(prog, postTransposeCtrl);
}

ConvProgramTree::TransformPostSerialProgram::TransformPostSerialProgram(
    Graph &graph, const poplar::DebugNameAndId &dnai)
    : castCS(graph.addComputeSet({dnai})) {}

void ConvProgramTree::TransformPostSerialProgram::lower(
    poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  prog.add(Execute(castCS, {dnai}));
  add(prog, copies);
}

ConvProgramTree::ConvProgramTree(Graph &graph, const Plan &plan,
                                 const poplar::DebugNameAndId &dnai)
    : weightsTranspose(graph, {dnai, "WeightsTranspose"}), transformPre(),
      transformPost(plan.numLevels()),
      transformPreSerial(graph, {dnai, "PreTranspose"}),
      transformPostSerial(graph, {dnai, "CastSerialOut"}), loopPost{{dnai}},
      loopCount(plan.totalSerialSplit()), slice{{dnai}}, update{{dnai}},
      convolveCSGroup(graph.addComputeSet({dnai, "Convolve"})),
      reduceOrCastComputeSets(plan.numLevels()), finalizeProg{{dnai}} {
  transformPre.reserve(plan.numLevels());
  for (unsigned i = 0; i < plan.numLevels(); ++i) {
    transformPre.emplace_back(ConvProgramTree::TransformPreProgram(
        graph, {dnai, "Transpose_#" + std::to_string(i)}));
  }
}

ConvProgramTree::ConvProgramTree(Graph &graph,
                                 const poplar::DebugNameAndId &dnai)
    : weightsTranspose(graph, {dnai, "WeightsTranspose"}), transformPre(),
      transformPost(), transformPreSerial(graph, {dnai, "PreTranspose"}),
      transformPostSerial(graph, {dnai, "CastSerialOut"}), loopCount(1),
      convolveCSGroup(graph.addComputeSet({dnai, "Convolve"})) {}

template <typename T>
static void lowerAndAddCycleCount(Graph &graph, Sequence &prog,
                                  const bool insertCycleCount, T &tpp,
                                  const poplar::DebugNameAndId &dnai) {
  Sequence seq{{dnai}};
  tpp.lower(seq, {dnai});
  if (insertCycleCount == true) {
    cycleCount(graph, seq, 0, SyncType::INTERNAL, dnai.getPathName());
  }
  prog.add(seq);
}

template <typename T>
static void moveTransposeOps(std::vector<T> &dst, std::vector<T> &src) {
  if (src.size() > 0) {
    dst.insert(dst.end(), src.begin(), src.end());
    src.erase(src.begin(), src.end());
  }
}

void ConvProgramTree::lower(Graph &graph, Sequence &prog,
                            const boost::optional<Plan> &plan,
                            const bool insertCycleCount,
                            const poplar::DebugNameAndId &dnai) {
  for (const auto &c : copyWritten) {
    prog.add(WriteUndef(c, {dnai}));
  }

  // weightsTranspose
  lowerAndAddCycleCount(graph, prog, insertCycleCount, weightsTranspose,
                        {dnai, "weightsTransposeSeq"});

  assert(transformPre.size() == transformPost.size());
  const unsigned numLevels = transformPre.size();
  assert(numLevels == reduceOrCastComputeSets.size());

  Sequence body{{dnai}};

  // lower the transforms in ascending order as we climb the hierarchy.
  for (unsigned level = 0; level < numLevels; ++level) {
    if (plan.is_initialized()) {
      const auto systemLevel = plan->transforms.size() - 2;
      const auto tileLevel = plan->transforms.size() - 1;

      // if no serial split of acts then move
      // transformPre[systemLevel].postTransposeActs before a convolution loop
      if (plan->broadcastInputBeforeLoop &&
          (level == systemLevel || level == tileLevel)) {

        moveTransposeOps(transformPreSerial.preTransposeActs,
                         transformPre[level].preTransposeActs);
        moveTransposeOps(transformPreSerial.transposeCSActs,
                         transformPre[level].transposeCSActs);
        moveTransposeOps(transformPreSerial.postTransposeActs,
                         transformPre[level].postTransposeActs);
      }
    }

    // transformPre[level]
    lowerAndAddCycleCount(
        graph, body, insertCycleCount, transformPre[level],
        {dnai, std::string("transformPre") + std::to_string(level)});
  }

  convolveCSGroup.lower(body, {dnai});
  // lower the remaining reductions and inverse transforms in reverse order
  // as we descend the hierarchy.
  for (int level = numLevels - 1; level >= 0; --level) {
    for (const auto &reduceOrCastCS : reduceOrCastComputeSets[level]) {
      body.add(Execute(reduceOrCastCS, {dnai}));
    }

    // transformPost[level]
    Sequence reduceTransformPostSeq{{dnai}};
    add(reduceTransformPostSeq, transformPost[level]);
    if (insertCycleCount == true) {
      cycleCount(graph, reduceTransformPostSeq, 0, SyncType::INTERNAL,
                 dnai.getPathName() + "/transformPost" + std::to_string(level));
    }
    body.add(reduceTransformPostSeq);
  }

  // transformPreSerial
  lowerAndAddCycleCount(graph, prog, insertCycleCount, transformPreSerial,
                        {dnai, "transformPreSerialSeq"});

  if (loopCount == 1 && !lastSlice) {
    prog.add(body);
  } else {
    assert(loopCount != 0);
    prog.add(Repeat(loopCount,
                    Sequence{{slice, body, update, loopPost}, {dnai}}, {dnai}));
    if (lastSlice) {
      prog.add(Sequence{{*lastSlice, body, *lastUpdate}, {dnai}});
    }
  }

  // transformPostSerial
  lowerAndAddCycleCount(graph, prog, insertCycleCount, transformPostSerial,
                        {dnai, "transformPostSerialSeq"});

  prog.add(finalizeProg);
}

static boost::optional<Tensor>
convolutionImpl(Graph &graph, const CanonicalConvParams &originalParams,
                Plan plan, unsigned level, Tensor in, Tensor weights,
                ConvProgramTree &cpt,
                const std::vector<Split<ConvIndices>> &indices, Tensor partials,
                unsigned createPartialsLevel, const DebugNameAndId &dnai,
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

  const auto systemLevel = plan.transforms.size() - 2;

  // Make a note of the channel and serial split counts.
  // Non-factor numbers of serial splits are handled by effectively padding
  // the channel dimension up to the next multiple of the split count.
  const auto numInChans = weightsSlice.dim(weightsSlice.rank() - 1);
  auto inChansPerSerialSplit = numInChans;
  const auto numOutChans = weightsSlice.dim(weightsSlice.rank() - 2);
  auto outChansPerSerialSplit = numOutChans;

  auto levelIndices = indices;
  levelIndices.emplace_back();
  auto parallelParams = serialParams;
  const std::string levelSuffix = "[" + std::to_string(level) + "]";
  // we only support serial splits on the ipu level.
  if (level == systemLevel) {
    auto &partition = plan.partitions[level];

    if (partition.totalSerialSplit() > 1) {
      inChansPerSerialSplit = getInChansPerSerialSplit(partition, numInChans);
      outChansPerSerialSplit =
          getOutChansPerSerialSplit(partition, numOutChans);

      // If the number of output channel splits isn't a factor of the number
      // of channels, then extract the weights that would be in the last slice
      // if we padded them up to the nearest multiple of the split count.
      // These weights will then be handled as a special case outside of the
      // loop.
      std::optional<Tensor> lastInputSplit;
      std::optional<Tensor> lastWeightsSplit;

      if (numInChans % inChansPerSerialSplit != 0) {
        auto const lastSliceBegin =
            numInChans - (numInChans % inChansPerSerialSplit);

        lastInputSplit =
            inSlice.slice(lastSliceBegin, numInChans, inSlice.rank() - 1);
        inSlice = inSlice.slice(0, lastSliceBegin, inSlice.rank() - 1);

        lastWeightsSplit = weightsSlice.slice(lastSliceBegin, numInChans,
                                              weightsSlice.rank() - 1);
        weightsSlice =
            weightsSlice.slice(0, lastSliceBegin, weightsSlice.rank() - 1);

        // "Remove" this split from the params and plans used to set up
        // the loop over the rest of the splits.
        partition.inChanSplit.serial -= 1;
        auto trimmedSerialParams = serialParams.releaseParams();
        trimmedSerialParams.inputChannelsPerConvGroup = lastSliceBegin;
        serialParams = trimmedSerialParams;

        // The loop count should have been set to the total serial split on
        // initialisation, so it needs to be decremented as well since the
        // last split will be handled outside of the loop.
        // Note that this relies on the current restriction of only allowing
        // one set of channels to be serially split, meaning that the loop
        // counter will only be counting input channel splits.
        cpt.loopCount -= 1;
        assert(numOutChans == outChansPerSerialSplit);
      }

      if (numOutChans % outChansPerSerialSplit != 0) {
        auto const lastSliceBegin =
            numOutChans - (numOutChans % outChansPerSerialSplit);

        // Extract the last split, ready to be processed separately.
        lastWeightsSplit = weightsSlice.slice(lastSliceBegin, numOutChans,
                                              weightsSlice.rank() - 2);
        weightsSlice =
            weightsSlice.slice(0, lastSliceBegin, weightsSlice.rank() - 2);

        // "Remove" this split from the params and plans used to set up
        // the loop over the rest of the splits.
        partition.outChanSplit.serial -= 1;
        auto trimmedSerialParams = serialParams.releaseParams();
        trimmedSerialParams.outputChannelsPerConvGroup = lastSliceBegin;
        serialParams = trimmedSerialParams;

        // The loop count should have been set to the total serial split on
        // initialisation, so it needs to be decremented as well since the
        // last split will be handled outside of the loop.
        // Note that this relies on the current restriction of only allowing
        // one set of channels to be serially split, meaning that the loop
        // counter will only be counting output channel splits.
        cpt.loopCount -= 1;
        assert(numInChans == inChansPerSerialSplit);
      }

      std::tie(parallelParams, levelIndices.back().serial) =
          preprocessForSerialSlice(&inSlice, &weightsSlice, serialParams,
                                   partition);

      // We check if the given tensor is such that a slice will cause exchange
      // and if so, rearrange before the loop rather than after as this will be
      // very expensive. this happens as a pre transform of the previous level
      // in the hierarchy.
      auto rearrangeIfSplitOverTiles = [&](Tensor &slice, bool isActs) {
        std::string sliceKind = isActs ? "input" : "weights";
        logging::poplin::debug("'{}': forcing rearrangement of {} before slice "
                               "because sliced dimension is split over tiles",
                               dnai.getPathName(), sliceKind);
        auto createSliceMethod = isActs ? createInputImpl : createWeightsImpl;
        Tensor metadata;
        if (in.hasMetadata() && weights.hasMetadata()) {
          metadata = isActs ? in.getMetadata() : weights.getMetadata();
        }
        assert(in.hasMetadata() == weights.hasMetadata());
        auto sliceRearranged = createSliceMethod(
            graph, serialParams, metadata, level, true, indices,
            {dnai, sliceKind + "Rearranged"}, plan, options);
        if (cpt.createVertices) {
          // WriteUndef sliceRearranged as it may be partially written. See
          // T36794.
          cpt.transformPreSerial.writeUndef.push_back(sliceRearranged);
          auto inSliceRearranged = isActs ? &sliceRearranged : nullptr;
          auto weightsSliceRearranged = isActs ? nullptr : &sliceRearranged;
          preprocessForSerialSlice(inSliceRearranged, weightsSliceRearranged,
                                   serialParams, partition);
          auto &preTranspose = isActs
                                   ? cpt.transformPreSerial.preTransposeActs
                                   : cpt.transformPreSerial.preTransposeWeights;
          auto &transposeCS =
              isActs ? cpt.transformPreSerial.transposeCSActs.back()
                     : cpt.transformPreSerial.transposeCSWeights.back();
          slice = popops::rearrange::regroupIfBeneficial(
              graph, slice, sliceRearranged, preTranspose, transposeCS,
              {dnai, sliceKind + "RegroupBeforeSlice"});

          auto postTranspose =
              isActs ? &cpt.transformPreSerial.postTransposeActs
                     : &cpt.transformPreSerial.postTransposeWeights;
          postTranspose->emplace_back(slice, sliceRearranged, false, dnai);
        }
        return sliceRearranged;
      };

      if ((partition.inChanSplit.serial > 1) &&
          dimIsSplitOverTiles(graph, inSlice, 0)) {
        inSlice = rearrangeIfSplitOverTiles(inSlice, true);
      }

      if (((partition.inChanSplit.serial > 1) &&
           dimIsSplitOverTiles(graph, weightsSlice, 1)) ||
          ((partition.outChanSplit.serial > 1) &&
           dimIsSplitOverTiles(graph, weightsSlice, 0))) {
        weightsSlice = rearrangeIfSplitOverTiles(weightsSlice, false);
      }

      // create and zero initialise loop counter.
      loopCounter = graph.addVariable(UNSIGNED_INT, {1},
                                      {dnai, "loopCounter" + levelSuffix});
      graph.setTileMapping(loopCounter, 0);
      if (cpt.createVertices) {
        const auto zeroConstant = graph.addConstant(
            UNSIGNED_INT, {1}, 0, {dnai, "zero" + levelSuffix});
        graph.setTileMapping(zeroConstant, 0);
        cpt.transformPreSerial.postTransposeCtrl.emplace_back(
            zeroConstant, loopCounter, false, dnai);
      }
      // per iteration slices of output.
      if (numOutChans == outChansPerSerialSplit) {
        weightsSlice = weightsSlice.squeeze({0});
      } else {
        auto weightsSliceOut = popops::createDynamicSliceOutput(
            graph, weightsSlice, loopCounter, {0}, {1},
            {dnai, "weightsSerialSlice" + levelSuffix});

        if (cpt.createVertices) {
          popops::dynamicSliceWithOutput(
              graph, weightsSliceOut, weightsSlice, loopCounter, {0}, {1},
              cpt.slice, {dnai, "weightsSerialSlice" + levelSuffix});
          weightsSlice = weightsSliceOut.squeeze({0});

          // If there's output channels not covered by the dynamic slice then
          // add an extra iteration to the loop which manually copies them into
          // the slice tensor.
          if (lastWeightsSplit) {
            const auto remainingOutChans =
                lastWeightsSplit->dim(lastWeightsSplit->rank() - 2);
            const auto dest = weightsSlice.slice(0, remainingOutChans,
                                                 weightsSlice.rank() - 2);
            cpt.lastSlice =
                Sequence{{Copy(*lastWeightsSplit, dest, false,
                               {dnai, "weightsSerialLastSlice" + levelSuffix})},
                         {dnai}};
          }
        } else { // if(cpt.createVertices)
          weightsSlice = weightsSliceOut.squeeze({0});
        }
      }

      // per iteration slices of input.
      if (numInChans == inChansPerSerialSplit) {
        inSlice = inSlice.squeeze({0});
        weightsSlice = weightsSlice.squeeze({0});
      } else {
        auto inSliceOut = popops::createDynamicSliceOutput(
            graph, inSlice, loopCounter, {0}, {1},
            {dnai, "inputSerialSlice" + levelSuffix});
        auto weightsSliceOut = popops::createDynamicSliceOutput(
            graph, weightsSlice, loopCounter, {0}, {1},
            {dnai, "weightsSerialSlice" + levelSuffix});
        if (cpt.createVertices) {
          popops::dynamicSliceWithOutput(
              graph, inSliceOut, inSlice, loopCounter, {0}, {1}, cpt.slice,
              {dnai, "inputSerialSlice" + levelSuffix});
          popops::dynamicSliceWithOutput(
              graph, weightsSliceOut, weightsSlice, loopCounter, {0}, {1},
              cpt.slice, {dnai, "weightsSerialSlice" + levelSuffix});
          inSlice = inSliceOut.squeeze({0});
          weightsSlice = weightsSliceOut.squeeze({0});
          // If there's input channels not covered by the dynamic slice then add
          // an extra iteration to the loop which manually copies them into the
          // slice tensors.
          if (lastInputSplit) {
            assert(lastWeightsSplit);
            const auto remainingInChans =
                lastInputSplit->dim(lastInputSplit->rank() - 1);

            const auto inputDest =
                inSlice.slice(0, remainingInChans, inSlice.rank() - 1);
            const auto weightsDest = weightsSlice.slice(
                0, remainingInChans, weightsSlice.rank() - 1);

            cpt.lastSlice = Sequence{
                {
                    Copy(*lastInputSplit, inputDest, false,
                         {dnai, "inputSerialLastSlice" + levelSuffix}),
                    Copy(*lastWeightsSplit, weightsDest, false,
                         {dnai, "weightsSerialLastSlice" + levelSuffix}),
                },
                {dnai}};

            // The remaining channels in one of the slices will need clearing
            // to ensure they don't contribute anything to the output of the
            // convolution.
            const auto inPadding = inSlice.slice(
                remainingInChans, inChansPerSerialSplit, inSlice.rank() - 1);
            const auto weightsPadding =
                weightsSlice.slice(remainingInChans, inChansPerSerialSplit,
                                   weightsSlice.rank() - 1);

            // Assume clearing the smaller tensor is the cheapest option.
            if (product(inPadding.shape()) < product(weightsPadding.shape())) {
              popops::zero(graph, inPadding, *cpt.lastSlice,
                           {dnai, "inputSerialLastSlice" + levelSuffix});
            } else {
              popops::zero(graph, weightsPadding, *cpt.lastSlice,
                           {dnai, "weightsSerialLastSlice" + levelSuffix});
            }
          }
        } else { // if(cpt.createVertices)
          inSlice = inSliceOut.squeeze({0});
          weightsSlice = weightsSliceOut.squeeze({0});
        }
      }
    }
  }

  const auto preTransformParams = parallelParams;

  // Transform.
  bool rearrangeActs = false;
  bool rearrangeWeights = false;
  if (level == systemLevel) {
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
                    !plan.transforms[systemLevel].expandDims.empty() ||
                    !plan.transforms[systemLevel].outChanFlattenDims.empty();
    rearrangeWeights = weightRearrangementIsExpensive(options) ||
                       (weightsNumDests > weightViewMaxBroadcastDests) ||
                       !plan.transforms[systemLevel].expandDims.empty() ||
                       !plan.transforms[systemLevel].outChanFlattenDims.empty();
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
      rearrangeActs, rearrangeWeights, {dnai});

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
                                 {dnai, "partials"});
  }

  // Convolve.
  Tensor out;
  const auto resultType = plan.types[level].resultType;
  const auto tileLevel = plan.transforms.size() - 1;
  if (level == tileLevel) {
    const auto &target = graph.getTarget();
    const auto tile = linearizeTileIndices(target, options, indices, plan);
    assert(cpt.transformPre.size() - 1 == level);

    Tensor tileLevelActs = inSlice;
    if (plan.broadcastInputBeforeLoop) {
      auto &params = parallelParams.getParams();
      tileLevelActs = createInputImpl(
          graph, params, in.getMetadata(), tileLevel, false, indices,
          {dnai, "tileLevelActsRearranged"}, plan, options);

      const poplar::DebugContext debugContext = {dnai, "tileLevelActsCopy"};

      if (cpt.createVertices) {
        cpt.transformPre[tileLevel].postTransposeActs.emplace_back(
            inSlice, tileLevelActs, true, debugContext);
      }
    }
    graph.setTileMapping(partials, tile);
    if (cpt.createVertices) {
      calcPartialConvOutput(graph, plan, tile, parallelParams.getParams(),
                            cpt.transformPre[level].postTransposeWeights,
                            cpt.copyWritten, cpt.convolveCSGroup, tileLevelActs,
                            weightsSlice, partials,
                            options.use128BitConvUnitLoad,
                            options.disableSRForAMPVertices, {dnai});
    }
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
                              createPartialsLevel, {dnai}, options);
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
          (numInChans != inChansPerSerialSplit) ? partialType : resultType;
      bool ascendingMapping = plan.linearizeTileDirection ==
                              Plan::LinearizeTileDirection::ASCENDING;
      out = createMultiStageGroupedReduceOutput(graph, partials, reducedType,
                                                options, plan.startTile,
                                                ascendingMapping, {dnai});
      if (cpt.createVertices) {
        multiStageGroupedReduceWithOutput(graph, out, partials, reducedType,
                                          cpt.reduceOrCastComputeSets[level],
                                          options, plan.startTile,
                                          ascendingMapping, {dnai});
      }
      out = unsplitActivationFromGroups(out);
    }
  }

  if (out.elementType() != resultType &&
      (level == tileLevel || numInChans == inChansPerSerialSplit)) {
    if (cpt.createVertices) {
      if (cpt.reduceOrCastComputeSets[level].empty()) {
        cpt.reduceOrCastComputeSets[level].push_back(
            graph.addComputeSet({dnai, "Cast"}));
      }
      out = popops::cast(graph, out, resultType,
                         cpt.reduceOrCastComputeSets[level][0], {dnai});
    } else {
      out = graph.clone(resultType, out, {dnai});
    }
  }

  // Inverse transform.
  out = convolutionPostprocess(graph, preTransformParams, originalTransform,
                               out, false /* serial */,
                               cpt.transformPost[level], {dnai});

  // Update.
  if (level == systemLevel) {
    const auto &partition = plan.partitions[level];

    const auto inChansAreSeriallySplit = numInChans != inChansPerSerialSplit;
    const auto outChansAreSeriallySplit = numOutChans != outChansPerSerialSplit;

    if (inChansAreSeriallySplit) {
      auto serialOut = graph.clone(out, {dnai, "/serialOut" + levelSuffix});
      if (cpt.createVertices) {
        // In the first iteration of the loop over serial input channel splits,
        // just copy the output to a new clone. In future iterations sum the
        // result with the result of the first iteration.
        Sequence firstIterationBody, notFirstIterationBody;
        firstIterationBody.add(Copy(
            out, serialOut, false, {dnai, "/serialInChanAccum" + levelSuffix}));
        popops::addInPlace(graph, serialOut, out, notFirstIterationBody,
                           {dnai, "/serialInChanAccum" + levelSuffix});
        cpt.update.add(If(loopCounter.reshape({}), notFirstIterationBody,
                          firstIterationBody));

        // If the loop doesn't cover all serial splits then there will be an
        // extra slice to be processed afterwards.
        if (numInChans % inChansPerSerialSplit) {
          cpt.lastUpdate = notFirstIterationBody;
        }
      }
      out = serialOut;
    }

    if (outChansAreSeriallySplit) {
      const auto numOutChanSplits =
          gccs::ceildiv(numOutChans, outChansPerSerialSplit);
      const auto loopOutChanCount =
          numOutChans - (numOutChans % outChansPerSerialSplit);

      // Make this tensor view suitable as a slice of the full output.
      out = out.expand({0});

      // Create an output tensor for the partials to be computed as part of
      // the loop.
      auto serialOut = popops::createSliceableTensorFromSlice(
          graph, out, {0}, {numOutChanSplits},
          {dnai, "serialOut" + levelSuffix});

      // Flatten serial output channel split back into output channels.
      serialOut = serialOut.dimRoll(0, serialOut.rank() - 2)
                      .flatten(serialOut.rank() - 2, serialOut.rank());

      // Trim extra output channels added to accommodate a non-factor number
      // of splits.
      if (serialOut.dim(serialOut.rank() - 1) != numOutChans) {
        serialOut = serialOut.slice(0, numOutChans, serialOut.rank() - 1);

        // Use a clone to ensure the trimmed padding doesn't make it anywhere
        // and we end up with one contiguous tensor per tile.
        serialOut = graph.clone(
            serialOut, {dnai, "serialOut" + levelSuffix},
            TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);
      }

      // Copy partials computed by the loop to the output tensor.
      if (cpt.createVertices) {
        {
          auto loopOut =
              serialOut.slice(0, loopOutChanCount, serialOut.rank() - 1);

          loopOut = loopOut.reshapePartial(
              loopOut.rank() - 1, loopOut.rank(),
              {partition.outChanSplit.serial, out.dim(out.rank() - 1)});

          loopOut = loopOut.dimRoll(loopOut.rank() - 2, 0);

          popops::dynamicUpdate(graph, loopOut, out, loopCounter, {0}, {1},
                                cpt.update,
                                {dnai, "serialUpdate" + levelSuffix});
        }

        // Copy partials computed after the loop to the output tensor.
        if (numOutChans > loopOutChanCount) {
          const auto remainingOutChanCount = numOutChans - loopOutChanCount;

          const auto src =
              out[0].slice(0, remainingOutChanCount, out[0].rank() - 1);

          const auto dest = serialOut.slice(loopOutChanCount, numOutChans,
                                            serialOut.rank() - 1);

          cpt.lastUpdate =
              Sequence{{Copy(src, dest, false,
                             {dnai, "serialUpdateLastSlice" + levelSuffix})},
                       {dnai}};
        }
      }

      out = serialOut;

      // WriteUndef the output as it is Read/Write in each iteration but in the
      // course of the entire loop is completely written.
      if (cpt.createVertices) {
        cpt.transformPreSerial.writeUndef.emplace_back(out);
      }
    }

    // common code for either serial splits.
    if (cpt.createVertices) {
      if (inChansAreSeriallySplit || outChansAreSeriallySplit) {
        // Increment counter
        auto loopIncrement = graph.addConstant(UNSIGNED_INT, {}, 1, {dnai});
        graph.setTileMapping(loopIncrement, 0);
        popops::addInPlace(graph, loopCounter, loopIncrement, cpt.loopPost,
                           {dnai, "loopIncrement" + levelSuffix});
      }
    }
  }

  // Casting to the final result type (i.e., the result type of the outermost
  // level) should be deferred until all the serial splits have executed.
  if ((out.elementType() != resultType) && level != tileLevel &&
      numInChans != inChansPerSerialSplit) {
    if (cpt.createVertices) {
      out = popops::cast(graph, out, resultType, cpt.transformPostSerial.castCS,
                         {dnai});
    } else {
      out = graph.clone(resultType, out, {dnai});
    }
  }

  // Inverse transform.
  out = convolutionPostprocess(graph, originalParams, originalTransform, out,
                               true /* serial */,
                               cpt.transformPostSerial.copies, {dnai});
  return out;
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

static Tensor remapOutputTensor(Graph &graph, const poplar::Tensor &output,
                                const Plan &plan, Sequence &prog,
                                unsigned numConvGroups,
                                unsigned chansPerConvGroup,
                                const DebugNameAndId &dnai) {

  // prefer a grouping of 16 if possible, if not then fallback to either 8 or 4.
  const auto grainSize = [&] {
    if (chansPerConvGroup % 16u == 0) {
      return 16u;
    } else if (chansPerConvGroup % 8u == 0) {
      return 8u;
    } else {
      return 4u;
    }
  }();
  const auto minElementsPerTile = grainSize;

  if (chansPerConvGroup % grainSize) {
    // do not remap if the output channels is not a multiple of grain size.
    // We could find a grain size in other dimensions and map but keep it
    // simple for now.
    return output;
  }
  std::size_t chansPerGroup = grainSize;
  auto remapShape =
      splitActivationIntoGroups(
          actsToInternalShape(output, numConvGroups, chansPerConvGroup), 1,
          chansPerGroup)
          .shape();

  // Keep the created tensor contiguous in the channel dimension. We
  // could also create a grouping for the channels if possible
  auto remappedOutput = graph.addVariable(output.elementType(), remapShape,
                                          {dnai, "remappedOutput"});
  bool ascendingMapping =
      plan.linearizeTileDirection == Plan::LinearizeTileDirection::ASCENDING;
  mapTensorLinearlyWithOffset(graph, remappedOutput, minElementsPerTile,
                              grainSize, plan.startTile, ascendingMapping);
  remappedOutput =
      actsToExternalShape(unsplitActivationFromGroups(remappedOutput));
  // Explicitly copy to remapped tensor with a benign layout
  prog.add(Copy(output, remappedOutput, false, {dnai}));
  logging::poplin::debug("  convolution output tensor remapped linearly");
  return remappedOutput;
}

static Tensor
convolutionInternal(Graph &graph, const poplar::Tensor &in_,
                    const poplar::Tensor &weights_, const Plan &plan,
                    const CanonicalConvParams &params,
                    bool transposeAndFlipWeights, ConvProgramTree &cpt,
                    const DebugNameAndId &dnai, const ConvOptions &options) {
  auto weights = weights_;
  if (weights.rank() == params->getNumFieldDims() + 2) {
    weights = weights.expand({0});
  }
  if (transposeAndFlipWeights) {
    // Create transposed/flipped weights
    auto bwdWeights =
        createWeights(graph, plan, params.getParams(), weights.getMetadata(),
                      {dnai, "bwdWeights"}, options);
    if (bwdWeights.dim(1) && bwdWeights.dim(2)) {
      weightsTransposeChansFlipXY(graph, weights, bwdWeights, cpt, {dnai});
    }
    weights = bwdWeights;
  }
  weights = weightsToInternalShape(weights);
  auto in = actsToInternalShape(in_, params->numConvGroups,
                                params->inputChannelsPerConvGroup);
  verifyInputShapesAndTypes(params, in, weights);

  const auto createPartialsLevel = getCreatePartialsLevel(plan);
  auto activations = *convolutionImpl(graph, params, plan, 0, in, weights, cpt,
                                      {} /* indices */, {} /* partials */,
                                      createPartialsLevel, {dnai}, options);

  assert(activations.elementType() == params->outputType);
  auto output = actsToExternalShape(activations);

  // Introspect the output tensor to check if it has a decent layout as a bad
  // layout impacts operations using the tensor in both memory and cycles. This
  // is a conservative check as we only check if there's a grouping. Don't do
  // this if this is a weight update pass as the dimension we want to be
  // contiguous (the batch dimension) is not the innermost dimension of the
  // output.
  bool isWeightUpdatePass =
      options.pass == Pass::TRAINING_WU || options.pass == Pass::FC_TRAINING_WU;
  if (!isWeightUpdatePass && options.remapOutputTensor) {
    const auto dimGroupings = detectDimGroupings(graph, output);
    if (dimGroupings.empty()) {
      output = remapOutputTensor(graph, output, plan, cpt.finalizeProg,
                                 params->numConvGroups,
                                 params->outputChannelsPerConvGroup, {dnai});
    }
  }

  if (options.gatherConvOutput) {
    logging::poplin::debug("Gathering output for {}", dnai.getPathName());
    // If the output tensor is a slice of the output partials then this should
    // be a copy to a clone. Otherwise should be elided by poplar.
    output = poputil::duplicate(
        graph, output, cpt.finalizeProg, {dnai, "outputGather"},
        TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);
  }
  return output;
}

Tensor convolution(Graph &graph, const poplar::Tensor &in,
                   const poplar::Tensor &weights, const Plan &plan,
                   const CanonicalConvParams &params,
                   bool transposeAndFlipWeights, ConvProgramTree &cpt,
                   const DebugNameAndId &dnai, const ConvOptions &options) {
  logging::poplin::info("convolution");
  logging::poplin::info("  pass={}, compute set name=\"{}\"", options.pass,
                        dnai.getPathName());
  logging::poplin::info("  tensor names: input=\"{}\", kernel=\"{}\"",
                        in.getDebugStr(), weights.getDebugStr());

  log(2, *params);

  auto output =
      convolutionInternal(graph, in, weights, plan, params,
                          transposeAndFlipWeights, cpt, {dnai}, options);

  return output;
}

Tensor convolution(Graph &graph, const poplar::Tensor &in,
                   const poplar::Tensor &weights, const ConvParams &params_,
                   bool transposeAndFlipWeights, Sequence &prog,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options_, PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(in, weights, params_, transposeAndFlipWeights, options_, cache),
      "convolution");
  using namespace fmt::literals;
  POPLIN_TRACEPOINT_WITH_METADATA(
      pvti::createJsonMetadata(R"({{"name": "{}"}})"_format(di.getPathName())));

  const CanonicalConvParams params(params_);
  const ConvOptions options(options_);

  const std::string layerName = "Conv_" + convSuffix(params);
  poplar::ProfileValue::Map pv;
  const auto plan = getPlan(graph.getTarget(), params, options, cache, &pv);
  ConvProgramTree cpt(graph, plan, {di, layerName});
  di.add("planInfo", pv);

  auto out =
      convolution(graph, in, weights, plan, params, transposeAndFlipWeights,
                  cpt, {di, layerName}, options);

  cpt.lower(graph, prog, plan, options.insertTransformsCycleCountProgs, {di});
  di.addOutput(out);
  return out;
}

Tensor createConvOutput(poplar::Graph &graph, const ConvParams &params_,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options,
                        PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params_, options, cache),
                                 "createConvOutput");

  const CanonicalConvParams params(params_);
  const std::string layerName = "Conv_" + convSuffix(params);
  poplar::ProfileValue::Map pv;
  di.add("planInfo", pv);
  const auto plan = getPlan(graph.getTarget(), params, options, cache, &pv);

  auto weights = createWeights(graph, plan, params, {}, {di}, options);
  auto in = createInput(graph, plan, params, {}, {di}, options);

  ConvProgramTree cpt(graph, plan, {di, layerName});
  cpt.createVertices = false;
  auto out = convolutionInternal(graph, in, weights, plan, params, false, cpt,
                                 {di, layerName}, options);
  di.addOutput(out);
  return out;
}

void convolutionWithOutput(poplar::Graph &graph, const poplar::Tensor &in,
                           const poplar::Tensor &weights,
                           const poplar::Tensor &out, const ConvParams &params,
                           bool transposeAndFlipWeights,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext,
                           const poplar::OptionFlags &options,
                           PlanningCache *cache) {
  auto out_ = convolution(graph, in, weights, params, transposeAndFlipWeights,
                          prog, debugContext, options, cache);
  prog.add(Copy(out_, out));
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
                                 ConvProgramTree &cpt,
                                 const poplar::DebugNameAndId &dnai) {
  auto &preTranspose = cpt.weightsTranspose.preTransposeWeights;
  auto &transposeCS = cpt.weightsTranspose.transposeCSWeights.back();
  auto &postTranspose = cpt.weightsTranspose.postTransposeWeights;
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

    // [GC1][O/G1][I/G2]...[GC2][G1/G5][G5][G2]
    //    -> [GC1][O/G1][I/G2]...[GC2][G1/G5][G2][G5]
    auto preTrans = partiallyTransposed;
    partiallyTransposed = popops::rearrange::partialTranspose(
        graph, partiallyTransposed, transposeCS, {dnai});
    if (partiallyTransposed.hasMetadata()) {
      preTranspose.emplace_back(preTrans.getMetadata(),
                                partiallyTransposed.getMetadata(), false,
                                DebugContext(dnai, "metadata"));
    }
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
      graph, flipped, weightsOutUnGrouped, preTranspose, transposeCS,
      {dnai, "attemptRegroup"});
  postTranspose.emplace_back(maybeRegroupedFlipped, weightsOutUnGrouped, false,
                             dnai);
}

void weightsTransposeChansFlipXY(Graph &graph, const Tensor &weightsInUnGrouped,
                                 const Tensor &weightsOutUnGrouped,
                                 Sequence &prog,
                                 const poplar::DebugContext &debugContext,
                                 const poplar::OptionFlags &options_) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(weightsInUnGrouped, weightsOutUnGrouped, options_),
      "weightsTransposeChansFlipXY");

  const ConvOptions options(options_);
  ConvProgramTree cpt(graph, {di, "WeightsTranspose"});
  weightsTransposeChansFlipXY(graph, weightsInUnGrouped, weightsOutUnGrouped,
                              cpt, {di});
  cpt.lower(graph, prog, boost::none, options.insertTransformsCycleCountProgs,
            {di});
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
                             ConvProgramTree &cpt, const DebugNameAndId &dnai,
                             const ConvOptions &wuOptions) {
  POPLIN_TRACEPOINT();
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
      wuPlan, wuParams, false, cpt, {dnai}, wuOptions);

  // [G][C]...[N]
  weightDeltas =
      actsToInternalShape(weightDeltas, fwdNumConvGroups, fwdOutChans);

  return weightsToExternalShape(
      weightDeltas.dimShufflePartial({1}, {weightDeltas.rank() - 1}));
}

Tensor calculateWeightDeltas(Graph &graph, const Tensor &zDeltas_,
                             const Tensor &activations_,
                             const ConvParams &fwdParams_, Sequence &prog,
                             const poplar::DebugContext &debugContext,
                             const poplar::OptionFlags &fwdOptions_,
                             PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(zDeltas_, activations_, fwdParams_, fwdOptions_, cache),
      "calculateWeightDeltas");

  const CanonicalConvParams wuParams = getWeightUpdateParams(fwdParams_);
  const auto wuOptions = getWeightUpdateOptions({fwdOptions_});
  const auto wuPlan = getPlan(graph.getTarget(), wuParams, wuOptions, cache);

  const std::string layerName = "Conv_" + convSuffix(wuParams);
  ConvProgramTree cpt(graph, wuPlan, {di, layerName});

  auto out = calculateWeightDeltas(graph, zDeltas_, activations_, wuPlan,
                                   wuParams, cpt, {di}, wuOptions);

  cpt.lower(graph, prog, wuPlan, wuOptions.insertTransformsCycleCountProgs,
            {di});
  di.addOutput(out);
  return out;
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             const Plan &wuPlan, CanonicalConvParams wuParams,
                             const Tensor &scale, ConvProgramTree &cpt,
                             const DebugNameAndId &dnai,
                             const ConvOptions &wuOptions) {
  auto weightDeltas = calculateWeightDeltas(graph, zDeltas, activations, wuPlan,
                                            wuParams, cpt, dnai, wuOptions);

  // update weights
  assert(weightDeltas.shape() == weights.shape());
  popops::scaledAddTo(graph, weights, weightDeltas, scale, cpt.finalizeProg,
                      {dnai, "UpdateWeights"});
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             ConvParams fwdParams, const Tensor &scale,
                             Sequence &prog,
                             const poplar::DebugContext &debugContext,
                             const poplar::OptionFlags &fwdOptions_,
                             PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(zDeltas, weights, activations, scale,
                                         fwdParams, fwdOptions_, cache));

  // Adjust params so that weightDelta is of inputType without needing
  // to cast.
  fwdParams.outputType = fwdParams.inputType;

  const CanonicalConvParams wuParams =
      getWeightUpdateParams(std::move(fwdParams));
  const auto wuOptions = getWeightUpdateOptions({fwdOptions_});
  const auto wuPlan = getPlan(graph.getTarget(), wuParams, wuOptions, cache);

  ConvProgramTree cpt(graph, wuPlan, {di});

  convolutionWeightUpdate(graph, zDeltas, weights, activations, wuPlan,
                          std::move(wuParams), scale, cpt, {di}, wuOptions);
  cpt.lower(graph, prog, wuPlan, wuOptions.insertTransformsCycleCountProgs,
            {di});
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             const Plan &wuPlan, CanonicalConvParams wuParams,
                             float scale, ConvProgramTree &cpt,
                             const DebugNameAndId &dnai,
                             const ConvOptions &wuOptions) {
  auto weightDeltas = calculateWeightDeltas(graph, zDeltas, activations, wuPlan,
                                            wuParams, cpt, {dnai}, wuOptions);

  // Add the weight deltas to the weights.
  assert(weightDeltas.shape() == weights.shape());
  const auto maybeRegroupedWeightDeltas =
      popops::rearrange::regroupIfBeneficial(graph, weightDeltas, weights,
                                             cpt.finalizeProg,
                                             {dnai, "regroupGradds"});

  popops::scaledAddTo(graph, weights, maybeRegroupedWeightDeltas, scale,
                      cpt.finalizeProg, {dnai, "UpdateWeights"});
}

void convolutionWeightUpdate(Graph &graph, const Tensor &zDeltas,
                             const Tensor &weights, const Tensor &activations,
                             ConvParams fwdParams, float scale, Sequence &prog,
                             const poplar::DebugContext &debugContext,
                             const poplar::OptionFlags &fwdOptions_,
                             PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(zDeltas, weights, activations,
                                         fwdParams, scale, fwdOptions_, cache));

  // Adjust params so that weightDelta is of inputType without needing
  // to cast.
  fwdParams.outputType = fwdParams.inputType;

  const CanonicalConvParams wuParams =
      getWeightUpdateParams(std::move(fwdParams));
  const auto wuOptions = getWeightUpdateOptions({fwdOptions_});
  const auto wuPlan = getPlan(graph.getTarget(), wuParams, wuOptions, cache);

  ConvProgramTree cpt(graph, wuPlan, {di});

  convolutionWeightUpdate(graph, zDeltas, weights, activations, wuPlan,
                          std::move(wuParams), scale, cpt, {di}, wuOptions);
  cpt.lower(graph, prog, wuPlan, wuOptions.insertTransformsCycleCountProgs,
            {di});
}

// Add a program to update the biases tensor with the gradients derived
// from the zDeltas tensor
void convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                           const Tensor &biases, const Tensor &scale,
                           const poplar::OptionFlags &options_, Sequence &prog,
                           const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(zDeltasUngrouped, biases, scale, options_));

  const ConvOptions options(options_);
  if (zDeltasUngrouped.rank() < 2)
    throw poplibs_error("convolutionBiasUpdate with rank " +
                        std::to_string(zDeltasUngrouped.rank()) +
                        "; must have at least channel and batch dimensions");

  std::vector<std::size_t> reduceDims(zDeltasUngrouped.rank() - 1);
  std::iota(reduceDims.begin() + 1, reduceDims.end(), 2);

  popops::reduceWithOutput(
      graph, zDeltasUngrouped, biases, reduceDims,
      {popops::Operation::ADD, true, scale}, prog, {di, "BiasUpdate"},
      {{"accumType.interTile", options.partialsType.toString()},
       {"accumType.inVertex", options.partialsType.toString()}});
}

void convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                           const Tensor &biases, float scale,
                           const poplar::OptionFlags &options_, Sequence &prog,
                           const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(zDeltasUngrouped, biases, scale, options_));

  auto scaleTensor = graph.addConstant(FLOAT, {}, scale, {di, "scaleTensor"});
  graph.setTileMapping(scaleTensor, 0);
  convolutionBiasUpdate(graph, zDeltasUngrouped, biases, scaleTensor, options_,
                        prog, {di, "ConstLearning"});
}

void addBias(Graph &graph, const Tensor &acts, const Tensor &biases,
             Sequence &prog, const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  if (acts.rank() < 2) {
    throw poplibs_error("Expected at least a batch size and channel dimension");
  }

  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts, biases));

  std::vector<std::size_t> broadcastBiases(acts.rank() - 2, 1);
  popops::addInPlace(graph, acts, biases.expand(broadcastBiases), prog, {di});
}

static ConvParams
getFullyConnectedFwdParamsFromBwdParams(const CanonicalConvParams &bwdParams) {
  auto fwdParams = bwdParams.getParams();
  std::swap(fwdParams.inputChannelsPerConvGroup,
            fwdParams.outputChannelsPerConvGroup);
  return fwdParams;
}

static ConvOptions
getFullyConnectedFwdOptionsFromBwdOptions(const ConvOptions &bwdOptions) {
  assert(bwdOptions.pass == Pass::FC_TRAINING_BWD);
  auto fwdOptions = bwdOptions;
  fwdOptions.pass = Pass::FC_TRAINING_FWD;
  return fwdOptions;
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
  const unsigned GCD = std::gcd(A, B);
  // All factors of GCD are common factors of A and B, and all common divisors
  // of A and B are factors of GCD
  std::vector<unsigned> result(1, GCD);
  // populated divisors ordered with largest first
  for (unsigned i = gccs::ceildiv(GCD, 2U); i > 0; --i) {
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

static Tensor getGroupedFCWeightsView(const Tensor &splitWeights,
                                      const unsigned inChansPerGroup,
                                      const unsigned fieldElementsPerGroup) {
  return splitWeights
      .reshape({splitWeights.dim(0), splitWeights.dim(1),
                splitWeights.dim(2) / inChansPerGroup, inChansPerGroup,
                splitWeights.dim(3) / fieldElementsPerGroup,
                fieldElementsPerGroup})
      .dimShufflePartial({3}, {4});
}

static Tensor getFirstInGroup(Tensor &splitWeights, const unsigned bwdGroupSize,
                              const unsigned fwdGroupSize) {
  splitWeights =
      getGroupedFCWeightsView(splitWeights, bwdGroupSize, fwdGroupSize);

  return splitWeights
      .slice({0, 0, 0, 0, 0, 0},
             {splitWeights.dim(0), splitWeights.dim(1), splitWeights.dim(2),
              splitWeights.dim(3), 1, 1})
      .squeeze({4, 5});
}

// Returns a score for how fast the transpose compute set should be executed.
// Higher score means faster execution. This is an estimated score
static double blockScore(const unsigned fwdGroupSize,
                         const unsigned bwdGroupSize, const Graph &graph,
                         Tensor splitWeights, const Tensor &splitTransposed) {
  // Call getGroupedFCWeightsView with fwd and bwd switched compared to
  // calculation to get first in group
  auto outTensor =
      getGroupedFCWeightsView(splitTransposed, fwdGroupSize, bwdGroupSize);
  const auto firstInGroup =
      getFirstInGroup(splitWeights, bwdGroupSize, fwdGroupSize);
  const auto mapping = graph.getTileMapping(firstInGroup);

  unsigned spread = 0;
  double score = std::numeric_limits<double>::max();
  const auto firstInGroupShape = firstInGroup.shape();
  for (const auto &intervals : mapping) {
    if (!intervals.empty()) {
      ++spread;
      const unsigned numTileTranspositions = accumSize(intervals);
      const bool fastTrans = popops::rearrange::canUseFastTranspose(
          graph.getTarget(), splitWeights.elementType(), bwdGroupSize,
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
        // will be introduced. This number is fairly empirical based on what
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
                                     const Tensor &splitWeights,
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
                                    graph, splitWeights, splitTranspose);
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
                                    const Tensor &splitWeights,
                                    const Tensor &splitWeightsTranspose) {

  const auto possibleFwdGroupSizes = commonDivisors(
      fwdPlan.inChansPerGroup, static_cast<unsigned>(splitWeights.dim(3)));

  const auto possibleBwdGroupSizes = commonDivisors(
      bwdPlan.inChansPerGroup, static_cast<unsigned>(splitWeights.dim(2)));

  const auto isJointPlan = fwdPlan.isJointPlan && bwdPlan.isJointPlan;

  FCWTGroupSizes result =
      pickGroupSizes(possibleFwdGroupSizes, possibleBwdGroupSizes, graph,
                     splitWeights, splitWeightsTranspose, isJointPlan);

  result.useCopyImpl = result.fwdGroupSize == 1 || result.bwdGroupSize == 1 ||
                       !planSwapsOperands(fwdPlan) ||
                       !planSwapsOperands(bwdPlan);
  logging::poplin::trace("Transpose Group sizes fwd, bwd, useCopy = {} {} {}",
                         result.fwdGroupSize, result.bwdGroupSize,
                         result.useCopyImpl);
  return result;
}

static Tensor fullyConnectedWeightTranspose(
    Graph &graph, Tensor weights, const CanonicalConvParams &bwdParams,
    Sequence &prog, const DebugNameAndId &dnai, const ConvOptions &bwdOptions,
    PlanningCache *cache) {
  if (bwdParams->getNumFieldDims() != 1) {
    throw poputil::poplibs_error("fullyConnectedWeightTranspose() expects a 1-d"
                                 " convolution");
  }
  auto fwdParams = getFullyConnectedFwdParamsFromBwdParams(bwdParams);
  auto splitWeights = weightsToInternalShape(weights);
  auto bwdPlan = getPlan(graph.getTarget(), bwdParams, bwdOptions, cache);
  auto fwdOptions = getFullyConnectedFwdOptionsFromBwdOptions(bwdOptions);
  auto fwdPlan = getPlan(graph.getTarget(), fwdParams, fwdOptions, cache);

  Tensor transposed =
      createWeights(graph, bwdPlan, bwdParams, weights.getMetadata(),
                    {dnai, "transposed"}, bwdOptions);
  auto splitTransposed = weightsToInternalShape(transposed);

  const auto groupSizes =
      getGroupSizes(fwdPlan, bwdPlan, graph, splitWeights, splitTransposed);
  const auto fwdGroupSize = groupSizes.fwdGroupSize;
  const auto bwdGroupSize = groupSizes.bwdGroupSize;

  if (groupSizes.useCopyImpl) {
    // In this case there is no benefit to using transpose vertices to
    // rearrange.
    return weightsToExternalShape(splitWeights.dimShuffle({0, 1, 3, 2}));
  }

  auto splitTransposedUngroupedShape = splitTransposed.shape();
  const auto dType = weights.elementType();

  splitTransposed =
      getGroupedFCWeightsView(splitTransposed, fwdGroupSize, bwdGroupSize);

  auto firstInGroup = getFirstInGroup(splitWeights, bwdGroupSize, fwdGroupSize);

  auto blockTileMapping = graph.getTileMapping(firstInGroup);
  auto transposeCS = graph.addComputeSet({dnai, "Transpose"});

  popops::rearrange::addTransposeVertices(
      graph, transposeCS, dType, bwdGroupSize, fwdGroupSize, blockTileMapping,
      [&](size_t index) {
        auto blockIndices =
            poputil::unflattenIndex(firstInGroup.shape(), index);
        return std::make_pair(splitWeights[blockIndices[0]][blockIndices[1]]
                                          [blockIndices[2]][blockIndices[3]]
                                              .flatten(),
                              splitTransposed[blockIndices[0]][blockIndices[1]]
                                             [blockIndices[3]][blockIndices[2]]
                                                 .flatten());
      },
      {dnai});
  prog.add(Execute(transposeCS, {dnai}));
  auto transposedWeights = splitTransposed.dimShufflePartial({3}, {4}).reshape(
      splitTransposedUngroupedShape);
  return weightsToExternalShape(transposedWeights);
}

Tensor fullyConnectedWeightTranspose(Graph &graph, Tensor weights,
                                     const ConvParams &params_, Sequence &prog,
                                     const poplar::DebugContext &debugContext,
                                     const poplar::OptionFlags &options_,
                                     PlanningCache *cache) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(weights, params_, options_, cache),
                                 "fullyConnectedWeightTranspose");

  const ConvOptions options(options_);
  auto output = fullyConnectedWeightTranspose(graph, weights, params_, prog,
                                              {di}, options, cache);
  di.addOutput(output);
  return output;
}

PlanCosts reportPlanEstimatedCosts(const poplar::Graph &graph,
                                   const ConvParams &params,
                                   const poplar::OptionFlags &options_,
                                   PlanningCache *cache) {
  ConvOptions options(options_);
  // Note validateLayerParams may change the options.
  validateLayerParams(params, graph.getTarget(), options, "Convolution");
  auto plan = getPlan(graph.getTarget(), params, options, cache);
  std::size_t cycles, memory;
  std::tie(cycles, memory) =
      estimateConvCost(graph.getTarget(), params, options, cache, plan);

  return {cycles, memory};
}

namespace internal {

std::ostream &operator<<(std::ostream &os, DetailedPlanCosts const &c) {
  os << c.parallelSplit << " " << c.serialSplit << "\n";
  c.apply(
      [&os](const PlanCosts &c) { os << c.cycles << " " << c.memory << "\n"; });
  return os;
}

std::istream &operator>>(std::istream &is, DetailedPlanCosts &c) {
  is >> c.parallelSplit >> c.serialSplit;
  c.apply([&is](PlanCosts &c) { is >> c.cycles >> c.memory; });
  return is;
}

DetailedPlanCosts reportDetailedPlanEstimatedCosts(
    const poplar::Graph &graph, const ConvParams &params,
    const poplar::OptionFlags &options_, PlanningCache *cache) {
  ConvOptions options(options_);
  // Note validateLayerParams may change the options.
  validateLayerParams(params, graph.getTarget(), options, "Convolution");
  auto plan = getPlan(graph.getTarget(), params, options, cache);
  return estimateDetailedConvCost(graph.getTarget(), params, options, cache,
                                  plan);
}

} // namespace internal

void reportPlanInfo(std::ostream &out, const poplar::Graph &graph,
                    const ConvParams &params,
                    const poplar::OptionFlags &options_, PlanningCache *cache) {
  ConvOptions options(options_);
  // Note validateLayerParams may change the options.
  validateLayerParams(params, graph.getTarget(), options, "Convolution");
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

void convolutionValidateOptions(const poplar::OptionFlags &options) {
  ConvOptions validateOptions;
  validateOptions.parseConvOptions(options);
}

} // namespace poplin
