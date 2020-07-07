// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popsparse/FullyConnectedParams.hpp>
#include <tuple>

#include <poputil/exceptions.hpp>

#include "poplibs_support/StructHelper.hpp"

#include "FullyConnectedUtils.hpp"

namespace popsparse {

using namespace fullyconnected;

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
        (numGroups > 1 ? std::to_string(numGroups) + "x" : "") +
        std::to_string(outputChannels) + "x" + std::to_string(inputChannels) +
        ")");
  }
  const auto nzRatio = convertAbsoluteNzElemsToRatio(
      numGroups, inputChannels, outputChannels, numNonZeroElems);
  return createWithNzRatio(sparsityParams, nzRatio, batchSize, numGroups,
                           inputChannels, outputChannels);
}

double FullyConnectedParams::getNzRatio() const { return nzRatio; }

std::size_t FullyConnectedParams::getNumNonZeroValues() const {
  return convertRatioNzElemsToAbsolute(numGroups, inputChannelsPerGroup,
                                       outputChannelsPerGroup, nzRatio);
}

std::ostream &operator<<(std::ostream &os, const FullyConnectedParams &p) {
  os << "{sparsity: " << p.getSparsityParams()
     << ",\n sparsity ratio: " << p.getNzRatio()
     << ",\n batch size: " << p.getBatchSize()
     << ",\n no. of groups: " << p.getNumGroups()
     << ",\n input channels: " << p.getInputChannelsPerGroup()
     << ",\n output channels: " << p.getOutputChannelsPerGroup() << "}";
  return os;
}

bool operator<(const FullyConnectedParams &a, const FullyConnectedParams &b) {
  // A bit awkward but this isn't a struct really and hence the members are not
  // public. Still shorter than writing out in full.
  static constexpr auto comparisonHelper = poplibs_support::makeStructHelper(
      &FullyConnectedParams::sparsityParams, &FullyConnectedParams::nzRatio,
      &FullyConnectedParams::batchSize,
      &FullyConnectedParams::inputChannelsPerGroup,
      &FullyConnectedParams::outputChannelsPerGroup);
  return comparisonHelper.lt(a, b);
}

bool operator==(const FullyConnectedParams &a, const FullyConnectedParams &b) {
  static constexpr auto comparisonHelper = poplibs_support::makeStructHelper(
      &FullyConnectedParams::sparsityParams, &FullyConnectedParams::nzRatio,
      &FullyConnectedParams::batchSize,
      &FullyConnectedParams::inputChannelsPerGroup,
      &FullyConnectedParams::outputChannelsPerGroup);
  return comparisonHelper.eq(a, b);
}

bool operator!=(const FullyConnectedParams &a, const FullyConnectedParams &b) {
  return !(a == b);
}

} // end namespace dynamic
} // end namespace popsparse
