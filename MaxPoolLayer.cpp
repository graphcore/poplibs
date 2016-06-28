#include "MaxPoolLayer.hpp"
#include "VertexTemplates.hpp"
#include "ConvUtil.hpp"

MaxPoolLayerImpl::MaxPoolLayerImpl(const Net &net,
                 int index,
                 unsigned kernelSize,
                 unsigned stride,
                 unsigned padding)  :
  Layer(net, index),
  kernelSize(kernelSize),
  stride(stride),
  padding(padding) {
  layerName = "MaxPool" + std::to_string(kernelSize) + "x" +
    std::to_string(kernelSize);
}

void MaxPoolLayerImpl::describe(std::ostream &out) {
  out << "   -- Max pooling layer:\n"
      << "        Size: " << kernelSize << "x" << kernelSize << "\n"
      << "        Stride: " << stride << "\n"
      << "        Input: " << xDim << "x" << yDim
                   <<   "x" << numChannels << "\n"
      << "        Output: " << xDimOut << "x" << yDimOut
                   <<   "x" << numChannels << "\n"
      << "        FLOPs: " << getNumberOfFlops() << "\n";
}

std::uint64_t MaxPoolLayerImpl::getNumberOfFlops() {
  std::uint64_t numFlops = 0;
  for (unsigned y = 0; y < yDimOut; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                               padding, yDim);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < xDimOut; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 padding, xDim);
      const auto width = inXEnd - inXBegin;
      numFlops += width * height;
    }
  }
  return numFlops;
}

double MaxPoolLayerImpl::getPerfectCycleCount() {
  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  const auto &machineInfo = getNetOptions().ipuMachineInfo;
  const auto numFLOPs = getNumberOfFlops();
  const auto vectorWidth = machineInfo.dataPathWidth / (8 * getDTypeSize());
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}

size_t MaxPoolLayerImpl::getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                             size_t zPrev) const {
  const auto xDimOut = (xPrev + (2 * padding) - kernelSize) / stride + 1;
  const auto yDimOut = (yPrev + (2 * padding) - kernelSize) / stride + 1;
  Layer *next = getNextLayer();
  return next->getNumChannelGroupsIn(xDimOut, yDimOut, zPrev);
}

void MaxPoolLayerImpl::
init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
  const auto dType = getDType();
  Layer *prev = getPrevLayer();
  auto in = prev->getFwdActivations();
  xDim = in.dim(1);
  yDim = in.dim(2);
  numChannels = in.dim(0) * in.dim(3);
  xDimOut = (xDim + (2 * padding) - kernelSize) / stride + 1;
  yDimOut = (yDim + (2 * padding) - kernelSize) / stride + 1;
  Layer *next = getNextLayer();
  numChanGroups = next->getNumChannelGroupsIn(xDimOut, yDimOut, numChannels);
  if (!numChanGroups)
    numChanGroups = in.dim(0);
  size_t chansPerGroup = numChannels / numChanGroups;
  activations = graph.addTensor(dType, {numChanGroups, xDimOut, yDimOut,
                                        chansPerGroup});
  mapActivations(activations, mapping);
  if (getNetType() == TrainingNet) {
    errors = graph.addTensor(dType, prev->getFwdActivations().dims());
  }
}

Program MaxPoolLayerImpl::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
  Layer *prev = getPrevLayer();
  Tensor in = prev->getFwdActivations();
  unsigned prevChanGroups = in.dim(0);
  unsigned prevChansPerGroup = numChannels / prevChanGroups;
  unsigned chansPerGroup = numChannels / numChanGroups;
  ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
  const auto activationsMapping = computeActivationsMapping(activations);
  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileActivationsBegin = activationsMapping[tile];
    const auto tileActivationsEnd = activationsMapping[tile + 1];
    for (unsigned activation = tileActivationsBegin;
         activation != tileActivationsEnd; ++activation) {
      unsigned chanInGroup = activation % chansPerGroup;
      unsigned y = (activation / chansPerGroup) % yDimOut;
      unsigned x = (activation / (chansPerGroup * yDimOut)) % xDimOut;
      unsigned chanGroup = activation / (chansPerGroup * yDimOut * xDimOut);
      const auto chan = chanGroup * chansPerGroup + chanInGroup;
      unsigned prevChanGroup = chan / prevChansPerGroup;
      unsigned prevChanInGroup = chan % prevChansPerGroup;
      unsigned inYBegin, inYEnd;
      std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                                 padding, yDim);
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 padding, xDim);
      Tensor window =
          in[prevChanGroup].slice({inYBegin, inXBegin, prevChanInGroup},
                                  {inYEnd, inXEnd,
                                   prevChanInGroup + 1})
                           .flatten();
      auto v =
        graph.addVertex(fwd, templateVertex("MaxPooling", getDType()),
          { {"activationIn", window},
            {"activationOut", activations[chanGroup][y][x][chanInGroup]} });
      if (mapping) {
        mapping->setMapping(v, tile);
      }
    }
  }
  return Execute(fwd);
}

Program MaxPoolLayerImpl::
backward(Graph &graph) {
  const auto chansPerGroup = numChannels / numChanGroups;
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  auto errIn = getNextLayer()->getBwdErrors();
  Layer *prev = getPrevLayer();
  Tensor act = prev->getFwdActivations();
  const auto prevChanGroups = act.dim(0);
  const auto prevChansPerGroup = numChannels / prevChanGroups;
  assert(errIn.dim(0) == numChanGroups);
  assert(errIn.dim(1) == yDimOut);
  assert(errIn.dim(2) == xDimOut);
  assert(errIn.dim(3) == chansPerGroup);
  for (unsigned xIn = 0; xIn < xDim; ++xIn) {
    const auto xOut = xIn / stride;
    if (xOut >= xDimOut)
      continue;
    for (unsigned yIn = 0; yIn < yDim; ++yIn) {
      const auto yOut = yIn / stride;
      if (yOut >= xDimOut)
        continue;
      for (unsigned chan = 0; chan < numChannels; ++chan) {
        unsigned chanGroup = chan / chansPerGroup;
        unsigned chanInGroup = chan % chansPerGroup;
        unsigned prevChanGroup = chan / prevChansPerGroup;
        unsigned prevChanInGroup = chan % prevChansPerGroup;
        graph.addVertex(bwdCS, templateVertex("MaxPoolingBwd", getDType()),
          { {"actOut", activations[chanGroup][yOut][xOut][chanInGroup]},
            {"actIn", act[prevChanGroup][yIn][xIn][prevChanInGroup]},
            {"errIn", errIn[chanGroup][yOut][xOut][chanInGroup]},
            {"errOut", errors[prevChanGroup][yIn][xIn][prevChanInGroup]} });
      }
    }
  }
  return Execute(bwdCS);
}
