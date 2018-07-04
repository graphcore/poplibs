// Check that we can handle large tensors on two tiles
//
#define BOOST_TEST_MODULE NonLinearityTest
#include <popnn/NonLinearity.hpp>
#include <boost/test/unit_test.hpp>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <iostream>
#include <poplibs_test/Util.hpp>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popnn;
using namespace poplibs_test::util;

const OptionFlags options {
  {"target.textSectionSizeInBytes", "0x9000"}
};

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(BigVectorList) {
#include <iostream>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <vector>

  // will perform Relu on tensors with this number of elements
  std::vector<size_t> sizes({100, 1000, 10000, 100000, 1000000});
  for (const size_t nElms : sizes) {
    auto device = createTestDevice(TEST_TARGET, 1, 2);

    poplar::Graph graph(device);
    popnn::addCodelets(graph);

    std::vector<float> hPre(nElms);
    std::vector<float> hPost(nElms);
    for (unsigned i = 0; i < nElms; ++i) {
      // set all initial values to -1.0
      hPre[i]  = -1.0;
      hPost[i] = 10.0;
    }
    poplar::Tensor dPre  = graph.addVariable(poplar::FLOAT, {nElms});

    poputil::mapTensorLinearly(graph, dPre);

    poplar::Tensor dPost = graph.clone(dPre, "dPost");
    poplar::program::Sequence prog;
    poplar::program::Copy copyProg(dPre, dPost);
    prog.add(copyProg);
    popnn::nonLinearity(graph,
                        popnn::NonLinearityType::NON_LINEARITY_RELU,
                        dPost,
                        prog,
                        "Relu");

    graph.createHostWrite("hPre", dPre);
    graph.createHostRead("hPost", dPost);



    poplar::Engine eng(device, graph, prog, options);
    eng.writeTensor("hPre", hPre.data());
    eng.run();
    eng.readTensor("hPost", hPost.data());

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
}
