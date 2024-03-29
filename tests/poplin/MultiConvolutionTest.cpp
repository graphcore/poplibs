// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE MultiConvolution
#include <boost/test/unit_test.hpp>

#include "poplin/MultiConvolution.hpp"
#include "poplin/codelets.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/codelets.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace poplibs_support;
using namespace poputil;

BOOST_AUTO_TEST_CASE(DifferentDataTypes) {
  ConvParams convA(HALF, 4, {8, 8}, {4, 4}, 1, 1, 10);
  ConvParams convB(FLOAT, 4, {5, 5}, {1, 1}, 11, 5, 1);
  ConvParams convC(QUARTER, HALF, 4, {8, 8}, {4, 4}, 1, 1, 10);

  const auto testQuarter =
      TEST_TARGET == DeviceType::Sim21 || TEST_TARGET == DeviceType::IpuModel21;

  auto args = [&](const std::string &suffix) {
    auto result = std::vector<multiconv::CreateTensorArgs>{
        {convA, {}, "convA_" + suffix},
        {convB, {}, "convB_" + suffix},
    };
    if (testQuarter) {
      result.push_back({convC, {}, "convC_" + suffix});
    }
    return result;
  };

  auto device = createTestDevice(TEST_TARGET, 1, 64);
  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  poplin::PlanningCache cache;

  const poplar::OptionFlags multiConvOptions{
      {"planType", "parallel"},
      {"perConvReservedTiles", "16"},
  };

  const auto weights = args("w");
  const auto inputs = args("i");

  std::vector<Tensor> ws;
  std::vector<Tensor> is;
  for (unsigned i = 0; i < weights.size(); ++i) {
    is.push_back(
        multiconv::createInput(graph, inputs, i, multiConvOptions, &cache));
    ws.push_back(
        multiconv::createWeights(graph, weights, i, multiConvOptions, &cache));
  }
  const auto convArgs = [&]() {
    std::vector<multiconv::ConvolutionArgs> convArgs{
        {is[0], ws[0], convA, {}},
        {is[1], ws[1], convB, {}},
    };
    if (testQuarter) {
      convArgs.push_back({is[2], ws[2], convC, {}});
    }
    return convArgs;
  }();

  Sequence prog;
  std::vector<Tensor> outs;
  BOOST_CHECK_NO_THROW({
    outs = multiconv::convolution(graph, convArgs, false, prog, "hello",
                                  multiConvOptions, &cache);
  });
}

BOOST_AUTO_TEST_CASE(InvalidConvOptionsDiffInsertTransformsCycleCountProgs) {
  ConvParams convA(HALF, 4, {8, 8}, {4, 4}, 1, 1, 10);
  ConvParams convB(FLOAT, 4, {5, 5}, {1, 1}, 11, 5, 1);

  auto args = [&](const std::string &suffix) {
    return std::vector<multiconv::CreateTensorArgs>{
        {convA, {}, "convA_" + suffix},
        {convB, {}, "convB_" + suffix},
    };
  };

  auto device = createTestDevice(TEST_TARGET, 1, 64);
  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  poplin::PlanningCache cache;

  const poplar::OptionFlags multiConvOptions{
      {"planType", "parallel"},
      {"perConvReservedTiles", "16"},
  };

  const auto weights = args("w");
  const auto inputs = args("i");

  std::vector<Tensor> ws;
  std::vector<Tensor> is;
  for (unsigned i = 0; i < weights.size(); ++i) {
    is.push_back(
        multiconv::createInput(graph, inputs, i, multiConvOptions, &cache));
    ws.push_back(
        multiconv::createWeights(graph, weights, i, multiConvOptions, &cache));
  }

  // Setting different conv options will invalidate graph
  const std::vector<multiconv::ConvolutionArgs> convArgs{
      {is[0], ws[0], convA, {{"insertTransformsCycleCountProgs", "true"}}},
      {is[1], ws[1], convB, {{"insertTransformsCycleCountProgs", "false"}}},
  };

  Sequence prog;
  BOOST_CHECK_THROW(multiconv::convolution(graph, convArgs, false, prog,
                                           "hello", multiConvOptions, &cache),
                    poplibs_error);
}
