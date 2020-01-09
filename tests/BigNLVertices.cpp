// Copyright (c) Graphcore Ltd, All rights reserved.
// Check that we can handle large tensors on two tiles
//
#define BOOST_TEST_MODULE NonLinearityTest
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popnn;
using namespace poplibs_test::util;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

#include <iostream>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <vector>

void testReluWithTensorOfSize(size_t nElms) {
  auto device = createTestDevice(TEST_TARGET, 1, 2);

  poplar::Graph graph(device.getTarget());
  popnn::addCodelets(graph);

  std::vector<float> hPre(nElms);
  std::vector<float> hPost(nElms);
  for (unsigned i = 0; i < nElms; ++i) {
    // set all initial values to -1.0
    hPre[i] = -1.0;
    hPost[i] = 10.0;
  }
  poplar::Tensor dPre = graph.addVariable(poplar::FLOAT, {nElms});

  poputil::mapTensorLinearly(graph, dPre);

  poplar::Tensor dPost = graph.clone(dPre, "dPost");
  poplar::program::Sequence prog;
  poplar::program::Copy copyProg(dPre, dPost);
  prog.add(copyProg);
  popnn::nonLinearityInPlace(graph, popnn::NonLinearityType::RELU, dPost, prog,
                             "Relu");

  graph.createHostWrite("hPre", dPre);
  graph.createHostRead("hPost", dPost);

  poplar::Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("hPre", hPre.data(), hPre.data() + hPre.size());
    eng.run();
    eng.readTensor("hPost", hPost.data(), hPost.data() + hPost.size());
  });

  float minVal = hPost[0];
  for (auto &x : hPost) {
    if (x < minVal) {
      minVal = x;
    }
  }
  std::cout << "with " << nElms << " elements, min val : " << minVal
            << std::endl;
  // Test a float is exactly zero without getting a compiler warning:
  BOOST_CHECK(minVal >= -0.f && minVal <= 0.f);
}

BOOST_AUTO_TEST_CASE(BigVectorList) {

  // These may need to be updated for different arch versions or
  // if the code size radically changes:
  std::vector<size_t> sizesThatFit({100, 1000, 10000});
  std::vector<size_t> sizesThatDoNotFit({100000, 1000000});

  const bool everythingFits =
      TEST_TARGET == DeviceType::IpuModel || TEST_TARGET == DeviceType::Cpu;

  // Everything fits on CPU and IPU model:
  if (everythingFits) {
    for (const auto s : sizesThatDoNotFit) {
      sizesThatFit.push_back(s);
    }
  }

  for (const auto s : sizesThatFit) {
    std::cout << "Size " << s << " expected to fit\n";
    testReluWithTensorOfSize(s);
  }

  if (everythingFits == false) {
    for (const auto s : sizesThatDoNotFit) {
      std::cout << "Size " << s << " NOT expected to fit\n";
      BOOST_CHECK_THROW(testReluWithTensorOfSize(s),
                        graph_memory_allocation_error);
    }
  }
}
