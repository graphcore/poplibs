// Check that we can handle larget tensors on a single tile
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

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popnn;
using namespace poplibs_test::util;

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
  for (size_t nElms : {100, 1000, 10000, 100000, 1000000}) {
    poplar::IPUModel model;
    model.tilesPerIPU = 2;
    auto device = model.createDevice();

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

    //graph.setTileMapping(dPre, 0);
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



    poplar::Engine eng(device, graph, prog);
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
    BOOST_CHECK(minVal == 0);
  }
}
