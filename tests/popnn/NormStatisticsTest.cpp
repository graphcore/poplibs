// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cfenv>
#include <iostream>
#include <limits>
#include <vector>

#define BOOST_TEST_MODULE NormStatisticsTest
#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplin/codelets.hpp>
#include <popnn/GroupNorm.hpp>
#include <popnn/codelets.hpp>
#include <popops/Fill.hpp>
#include <popops/codelets.hpp>

// Default FP exceptions do not include inexact FP exceptions which LLVM hits.
constexpr int defaultFPExceptions =
    (FE_DIVBYZERO | FE_UNDERFLOW | FE_OVERFLOW | FE_INVALID);

// Trap on the floating point exceptions for the lifetime of this object.
struct EnableFPSignalsInScope {
  EnableFPSignalsInScope(int _excepts = defaultFPExceptions)
      : excepts(_excepts), original(0) {
#ifdef __APPLE__
    (void)excepts;
    (void)original;
#else
    original = fegetexcept();
    if (feenableexcept(excepts))
      throw std::runtime_error("Failed to enable FP exception signals");
#endif
  }
  ~EnableFPSignalsInScope() noexcept(false) {
#ifndef __APPLE__
    if (fedisableexcept(excepts ^ original) == -1)
      throw std::runtime_error("Failed to disable FP exception signals");
#endif
  }

private:
  int excepts;
  int original;
};

struct Options {
  float epsilon = 1.0f;
  float fillValue = 1.0f;
  size_t numGroups = 1;
  bool unbiasedVarEstimate = false;
  bool stableAlgo = true;
};

void testGroupNormStatistics(const std::vector<size_t> &shape,
                             const std::vector<float> expectedMean,
                             const std::vector<float> expectedInvStdDev,
                             const Options &options = Options{}) {
  const size_t numChannels = shape[0];
  BOOST_REQUIRE_EQUAL(numChannels, expectedMean.size());
  BOOST_REQUIRE_EQUAL(numChannels, expectedInvStdDev.size());

  poplibs_support::TestDevice device = createTestDevice(TEST_TARGET, 1, 1);

  poplar::Graph graph(device.getTarget());
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);

  const poplar::Type dataType = poplar::equivalent_device_type<float>().value;
  poplar::Tensor input = graph.addVariable(dataType, shape);
  graph.setTileMapping(input, 0);

  poplar::program::Sequence prog;
  poplar::Tensor mean, invStdDev;
  {
    // The compiler should not raise any floating point exceptions.
    EnableFPSignalsInScope enableFPsignals;

    popops::fill(graph, input, prog, options.fillValue);

    std::tie(mean, invStdDev) = popnn::gn::groupNormStatistics(
        graph, input, options.epsilon, prog, options.numGroups,
        options.unbiasedVarEstimate, options.stableAlgo, dataType);
  }

  BOOST_REQUIRE_EQUAL(mean.rank(), 1);
  BOOST_REQUIRE_EQUAL(invStdDev.rank(), 1);

  BOOST_REQUIRE_EQUAL(mean.shape()[0], numChannels);
  BOOST_REQUIRE_EQUAL(invStdDev.shape()[0], numChannels);

  graph.createHostRead("mean", mean);
  graph.createHostRead("invStdDev", invStdDev);

  poplar::Engine engine(graph, prog);
  device.bind([&](const poplar::Device &d) {
    engine.load(d);
    engine.run();

    // Check that the mean is as expected.
    {
      std::vector<float> resultMean(numChannels, -1.0f);
      engine.readTensor("mean", resultMean.data(),
                        resultMean.data() + resultMean.size());
      BOOST_CHECK_EQUAL_COLLECTIONS(resultMean.begin(), resultMean.end(),
                                    expectedMean.begin(), expectedMean.end());
    }

    // Check that the inverse standard deviation is as expected.
    {
      std::vector<float> resultInvStdDev(numChannels, -1.0f);
      engine.readTensor("invStdDev", resultInvStdDev.data(),
                        resultInvStdDev.data() + resultInvStdDev.size());
      BOOST_CHECK_EQUAL_COLLECTIONS(
          resultInvStdDev.begin(), resultInvStdDev.end(),
          expectedInvStdDev.begin(), expectedInvStdDev.end());
    }
  });
}

// Zero channels should result in an empty output.
BOOST_AUTO_TEST_CASE(groupNormStatisticsZeroChannels) {
  testGroupNormStatistics({0, 3, 0}, {}, {});
}

// Data of all zeros should have a mean of zero.
BOOST_AUTO_TEST_CASE(groupNormStatisticsZeroData) {
  Options opts;
  opts.fillValue = 0.0f;
  testGroupNormStatistics({1, 3, 3}, {0.0f}, {1.0f}, opts);
}

// A zero spatial dimension should have a mean of 0.
BOOST_AUTO_TEST_CASE(groupNormStatisticsZeroSpatialDims) {
  testGroupNormStatistics({1, 3, 0}, {0.0f}, {1.0f});
  testGroupNormStatistics({1, 0, 3}, {0.0f}, {1.0f});
  testGroupNormStatistics({3, 3, 0}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
}

// Just one element should result in a mean of that element.
BOOST_AUTO_TEST_CASE(groupNormStatisticsSizeOne) {
  Options opts;
  opts.unbiasedVarEstimate = true;
  opts.fillValue = 7.0f;
  testGroupNormStatistics({2, 1, 1}, {7.0f, 7.0f}, {1.0f, 1.0f}, opts);
}

// This test case should not segfault.
BOOST_AUTO_TEST_CASE(groupNormStatisticsT28462) {
  Options opts;
  opts.epsilon = 0.1f;
  testGroupNormStatistics({0, 5}, {}, {}, opts);
}
