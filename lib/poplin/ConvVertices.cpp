// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ConvVertices.hpp"
#include "ConvTransforms.hpp"
#include "ConvUtilInternal.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "popops/Cast.hpp"
#include "popops/Pad.hpp"
#include "popops/Zero.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;
using namespace poplibs_support;

namespace poplin {

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

struct ConvVertexSpatialPartition {
  std::vector<std::size_t> outBeginIndices;
  unsigned outXWidth;
  std::vector<std::size_t> inBeginIndices;
  unsigned inXWidth;
  unsigned context;
  unsigned subKernelPosition;
};

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
  reorderedIn.reserve(in.size());
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
  perWorkerConvOutputSlices.reserve(numWorkers);
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
    tileConvOutBegin.reserve(numFieldDims);
    tileConvOutSize.reserve(numFieldDims);
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
        outBeginIndices.reserve(numFieldDims);
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
  const unsigned char mode = convGroupsPerGroup == 4u  ? 0
                             : convGroupsPerGroup == 2 ? 1
                                                       : 2;

  auto kernelGroups = params.kernelShape;
  assert(kernelGroups.back() % slicWindowWidth == 0);
  kernelGroups.back() /= slicWindowWidth;

  const auto outputSpatialShape = [&] {
    std::vector<unsigned> r;
    r.reserve(1 + numFieldDims);
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

  outWindow.reserve(numConvGroupGroups * numOutChanGroups);
  inWindow.reserve(numConvGroupGroups * numInChanGroups);
  weightsWindow.reserve(numConvGroupGroups * numOutChanGroups *
                        numInChanGroups);

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
  bool useLimitedVer = plan.useLimitedVersion;
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

void calcPartialConvOutput(Graph &graph, const Plan &plan, unsigned tile,
                           ConvParams params, std::vector<Copy> &transformPre,
                           std::map<Type, Tensor> &copyWritten,
                           ConvProgramTree::ComputeSetsGroup &convolveCS,
                           Tensor in, Tensor weights, Tensor out,
                           bool use128BitConvUnitLoad,
                           const std::string &debugPrefix) {
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

} // namespace poplin
