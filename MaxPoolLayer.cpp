#include "MaxPoolLayer.hpp"
#include "VertexTemplates.hpp"

MaxPoolLayerImpl::MaxPoolLayerImpl(const Net &net,
                 int index,
                 unsigned kernelSize,
                 unsigned stride)  :
  Layer(net, index),
  kernelSize(kernelSize),
  stride(stride) {
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
  for (unsigned i = 0; i < xDimOut; ++i) {
    for (unsigned j = 0; j < yDimOut; ++j) {
      for (unsigned chan = 0; chan < numChannels; ++chan) {
        unsigned width =
          std::min(i * stride + kernelSize, xDim) - i * stride;
        unsigned height =
          std::min(j * stride + kernelSize, yDim) - j * stride;
        numFlops += width * height;
      }
    }
  }
  return numFlops;
}

double MaxPoolLayerImpl::getPerfectCycleCount() {
  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  // Can execute 4 f16 max or 2 f32 max per cycle.
  return static_cast<double>(getNumberOfFlops() / (getDTypeSize() * numTiles));
}

void MaxPoolLayerImpl::
init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
  const auto dType = getDType();
  Layer *prev = getPrevLayer();
  auto in = prev->getFwdActivations();
  xDim = in.dim(1);
  yDim = in.dim(2);
  numChannels = in.dim(0) * in.dim(3);
  xDimOut = (xDim - kernelSize) / stride + 1;
  yDimOut = (yDim - kernelSize) / stride + 1;
  Layer *next = getNextLayer();
  numChanGroups = next->getNumChannelGroupsIn(xDimOut, yDimOut, numChannels);
  if (!numChanGroups)
    numChanGroups = in.dim(0);
  size_t chansPerGroup = numChannels / numChanGroups;
  activations = graph.addTensor(dType, {numChanGroups, xDimOut, yDimOut,
                                        chansPerGroup});
  mapTensor(activations, mapping);
}

Program MaxPoolLayerImpl::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
  Layer *prev = getPrevLayer();
  Tensor in = prev->getFwdActivations();
  unsigned prevChanGroups = in.dim(0);
  unsigned prevChansPerGroup = numChannels / prevChanGroups;
  unsigned chansPerGroup = numChannels / numChanGroups;
  ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
  for (unsigned i = 0; i < xDimOut; ++i) {
    for (unsigned j = 0; j < yDimOut; ++j) {
      for (unsigned chan = 0; chan < numChannels; ++chan) {
        unsigned width = std::min(i * stride + kernelSize, xDim) - i * stride;
        unsigned height = std::min(j * stride + kernelSize, yDim) - j * stride;
        // Create window into previous layer
        unsigned prevChanGroup = chan / prevChansPerGroup;
        unsigned prevChanInGroup = chan % prevChansPerGroup;
        unsigned chanGroup = chan / chansPerGroup;
        unsigned chanInGroup = chan % chansPerGroup;
        Tensor window =
          in[prevChanGroup].slice({i * stride, j * stride, prevChanInGroup},
                                  {i * stride + width, j * stride + height,
                                    prevChanInGroup+1})
            .flatten();
        graph.addVertex(fwd, templateVertex("MaxPooling", getDType()),
          { {"activationIn", window},
            {"activationOut", activations[chanGroup][i][j][chanInGroup]} });
      }
    }
  }
  mapComputeSet(graph, fwd, mapping);
  return Execute(fwd);
}
