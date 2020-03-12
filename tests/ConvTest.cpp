// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvTest
#include "ConvUtilInternal.hpp"
#include "TestDevice.hpp"
#include "poputil/TileMapping.hpp"
#include <boost/test/unit_test.hpp>
#include <poplin/Convolution.hpp>

using namespace poplar;
using namespace poplin;

BOOST_AUTO_TEST_CASE(MappingSplitOutChansSerially) {
  constexpr std::size_t numTiles = 16;
  constexpr std::size_t split = 4;
  constexpr std::size_t numOutChans = 16;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  static_assert(numOutChans % split == 0, "output channels must be evenly "
                                          "divisible by the serial split");
  const auto params =
      ConvParams(HALF, 4 /* batchSize */, {8, 8} /* inputFieldShape */,
                 {3, 3} /* kernelShape */, 16 /* inputChannels */,
                 numOutChans /* outputChannels */, 1 /* numConvGroups */);

  const OptionFlags options{
      {
          "planConstraints",
          R"({"0":{"partition":{"outChanSplit":{"serial":)" +
              std::to_string(split) + R"(}}}})",
      },
  };

  PlanningCache cache;
  const auto weights = createWeights(graph, params, "weights", options, &cache);

  std::stringstream ss;
  reportPlanInfo(ss, graph, params, options, &cache);
  BOOST_TEST_MESSAGE(ss.str());

  // Check each of the splits in the serial partition has the same mapping
  // for efficient dynamic slicing.
  const auto outChansPerSplit = numOutChans / split;
  const auto referenceMapping =
      graph.getTileMapping(weights.slice(0, outChansPerSplit, 1));
  for (std::size_t s = 1; s < split; ++s) {
    const auto slice =
        weights.slice(s * outChansPerSplit, (s + 1) * outChansPerSplit, 1);
    BOOST_CHECK(graph.getTileMapping(slice) == referenceMapping);
  }
}
