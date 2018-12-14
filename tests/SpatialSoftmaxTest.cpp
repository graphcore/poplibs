#define BOOST_TEST_MODULE SpatialSoftmaxTest

#include <boost/test/unit_test.hpp>
#include <popops/codelets.hpp>
#include <popnn/SpatialSoftMax.hpp>
#include <popnn/codelets.hpp>
#include <poplin/codelets.hpp>
#include "TestDevice.hpp"
#include <poputil/TileMapping.hpp>

#include <vector>

BOOST_AUTO_TEST_CASE(SpatialSoftmax) {
  auto device = createTestDevice(TEST_TARGET);
  poplar::Graph g(device.getTarget());
  poplar::program::Sequence prog;
  popops::addCodelets(g);
  poplin::addCodelets(g);
  popnn::addCodelets(g);

  // Create some inputs:
  const auto width = 4u;
  const auto height = 3u;

  auto fields = g.addVariable(poplar::FLOAT, {1, height, width}, "input");
  poputil::mapTensorLinearly(g, fields);

  // Set the normalising factor to one so that the result is the mean
  // coordinate:
  const auto initialTemp = 1.f;

  // Disable adding the softmax operation itself so that computing the expected
  // values for the coordinates can be done by hand arithmetic below. (The
  // softmax is a vanilla call to Poplib's nonLinearity so independently
  // tested).
  auto ssm = popnn::spatialSoftMax2D(g, prog, fields, initialTemp, true, "ssm");
  // Check we will get a coordinate for every field down the rows:
  BOOST_CHECK_EQUAL(ssm.first.shape()[0], fields.shape()[0]);
  // Check the coords will be 2D (x,y) across columns
  BOOST_CHECK_EQUAL(ssm.first.shape()[1], 2u);

  g.createHostWrite("fields", fields);
  g.createHostRead("coords", ssm.first);
  g.createHostRead("temp", ssm.second);

  poplar::Engine e(g, prog);
  device.bind([&](const poplar::Device &d) {
    e.load(d);

    const std::vector<float> activations =
    {//x = -1, -1/3, 1/3,  1
           0.f, 0.f, 0.f, 0.f, // y = -1
           .5f, 0.f, 0.f, 0.f, // y =  0
           0.f, .5f, 0.f, 0.f  // y =  1
    };
    e.writeTensor("fields", activations.data());
    // Compute expected coord for the 'centre of mass' of the above activations.
    // (NOTE: we disabled the softmax normalisaiton above, otherwise the
    // expected result is painful to compute!)
    // The columns coordinates map from indeices [0,3] -> [-1,1]
    // The row coordinates map from indeices [0,2] -> [-1,1]
    // Hence, mean activated index gets mapped like this:
    // [ux, uy] -> [(ux/3)*2-1, (uy/2)*2-1]
    const std::vector<float> expected
      = {1.f/2, -2.f/3}; // {ey, ex} => {row, col}

    e.run();

    std::vector<float> actual(fields.shape()[0]*2u, -100.f);
    e.readTensor("coords", actual.data());
    for (auto i =0u; i < actual.size(); ++i) {
      BOOST_CHECK_CLOSE(actual.at(i), expected.at(i), 0.00001);
    }

    float checkTemp = 0.f;
    e.readTensor("temp", &checkTemp);
    BOOST_CHECK_EQUAL(checkTemp, initialTemp);
  });
}
