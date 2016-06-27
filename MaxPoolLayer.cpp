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
