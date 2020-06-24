// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popsparse/FullyConnectedParams.hpp>
#include <tuple>

#include <poputil/exceptions.hpp>

namespace popsparse {
namespace dynamic {

FullyConnectedParams FullyConnectedParams::createWithNzRatio(
    const SparsityParams &sparsityParams, double nzRatio, std::size_t batchSize,
    std::size_t numGroups, std::size_t inputChannels,
    std::size_t outputChannels) {
  if (nzRatio < 0.0 || nzRatio > 1.0) {
    throw poputil::poplibs_error("Non-zero ratio (" + std::to_string(nzRatio) +
                                 ") must be in range [0.0,1.0] but is not");
  }
  FullyConnectedParams p;
  p.sparsityParams = sparsityParams;
  p.nzRatio = nzRatio;
  p.batchSize = batchSize;
  p.numGroups = numGroups;
  p.inputChannelsPerGroup = inputChannels;
  p.outputChannelsPerGroup = outputChannels;
  return p;
}

FullyConnectedParams FullyConnectedParams::createWithNumNonZeroValues(
    const SparsityParams &sparsityParams, std::size_t numNonZeroElems,
    std::size_t batchSize, std::size_t numGroups, std::size_t inputChannels,
    std::size_t outputChannels) {
  const std::size_t totalDenseElems =
      numGroups * inputChannels * outputChannels;
  if (numNonZeroElems > totalDenseElems) {
    throw poputil::poplibs_error(
        "Number of non-zero elements (" + std::to_string(numNonZeroElems) +
        ") exceeds maximum possible for given dense matrix dimensions (" +
        std::to_string(outputChannels) + "x" + std::to_string(inputChannels) +
        ")");
  }
  const double nzRatio = double(numNonZeroElems) / double(totalDenseElems);
  // Double check that we really can represent this number of non-zero elements
  // as a double without losing precision.
  assert(std::size_t(std::ceil(nzRatio * totalDenseElems)) == numNonZeroElems);
  return createWithNzRatio(sparsityParams, nzRatio, batchSize, numGroups,
                           inputChannels, outputChannels);
}

double FullyConnectedParams::getNzRatio() const { return nzRatio; }

std::size_t FullyConnectedParams::getNumNonZeroValues() const {
  const auto totalDenseElems =
      numGroups * inputChannelsPerGroup * outputChannelsPerGroup;
  return std::ceil(nzRatio * totalDenseElems);
}

std::ostream &operator<<(std::ostream &os, const FullyConnectedParams &p) {
  os << "{sparsity: " << p.getSparsityParams()
     << ",\n"
        " batch size: "
     << p.getBatchSize()
     << ",\n"
        " no. of groups: "
     << p.getNumGroups()
     << ",\n"
        " input channels: "
     << p.getInputChannelsPerGroup()
     << ",\n"
        " output channels: "
     << p.getOutputChannelsPerGroup() << "}";
  return os;
}

bool operator<(const FullyConnectedParams &a, const FullyConnectedParams &b) {
  return std::tie(a.batchSize, a.inputChannelsPerGroup, a.numGroups,
                  a.outputChannelsPerGroup, a.sparsityParams) <
         std::tie(b.batchSize, b.inputChannelsPerGroup, b.numGroups,
                  b.outputChannelsPerGroup, b.sparsityParams);
}

} // end namespace dynamic
} // end namespace popsparse
