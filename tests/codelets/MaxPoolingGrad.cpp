#include "TestDevice.hpp"
#include <poplar/Engine.hpp>
#include "popnn/codelets.hpp"
#include "poplibs_test/Util.hpp"

#define BOOST_TEST_MODULE MaxPoolingGrad
#include <boost/test/included/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

#define TOL 0.1 //tolerance of 0.1%

using Vec2D = std::vector<std::vector<float>>;

const Vec2D out = {
  {20.9857, 9.1092, 36.3858, 13.0151, 69.7698, 15.6796, 59.7411, 35.6992},
  {92.3212, 11.9177, 2.3181, 13.0151, 54.7818, 78.6841, 42.8684, 70.2444},
  {25.5591, 55.0707, 92.8571, 1.5816, 40.3484, 86.3428, 82.9268, 10.0015},
  {61.6056, 83.7143, 43.8865, 53.8705, 46.1986, 25.1207, 72.9929, 62.0044},
  {38.2085, 55.8412, 54.7993, 48.7298, 30.9249, 29.35, 50.4223, 89.852},
};

const std::array<unsigned short, 3> windowSizes = {2, 0, 3};

const Vec2D in = {
  {92.3212, 11.9177, 36.3858, 13.0151, 69.7698, 78.6841, 59.7411, 70.2444},
  {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN},
  {61.6056, 83.7143, 92.8571, 53.8705, 46.1986, 86.3428, 22.2222, 89.852},
};

const Vec2D outGrad = {
  {20.2762, 53.4229, 13.4018, 7.1157, 85.3793, 53.3082, 2.7649, 84.5974},
  {97.1649, 34.7329, 26.4956, 42.8358, 12.8609, 87.8989, 23.2152, 79.5007},
  {46.714, 57.9405, 61.4473, 37.1417, 45.3669, 43.2174, 69.2814, 91.2615},
  {92.1645, 2.818, 21.6283, 27.331, 6.7908, 7.9717, 90.3777, 39.7396},
  {38.2157, 42.0879, 73.3562, 79.0739, 64.5654, 46.6707, 91.9358, 68.4146},
};

// dummy initial data in the output tensor.
const Vec2D inGrad = {
  {1, 2, 3, 4, 5, 6, 7, 8},
  {1, 2, 3, 4, 5, 6, 7, 8},
  {1, 2, 3, 4, 5, 6, 7, 8},
};

const Vec2D expected = {
  {97.1649, 34.7329, 13.4018, 49.9515, 85.3793, 87.8989, 2.7649, 79.5007},
  {0, 0, 0, 0, 0, 0, 0, 0},
  {92.1645, 2.818, 61.4473, 27.331, 6.7908, 43.2174, 0, 68.4146},
};

double atol(const Type &type) {
  return type == HALF ? 1e-7 : 1e-20;
}

void testMaxPoolingGrad(const char *vertex, const Type &type) {
  Device device = createTestDevice(TEST_TARGET);
  Graph graph(device);
  popnn::addCodelets(graph);

  const auto &target = device.getTarget();

  Sequence prog;

  // create a compute set for channel sizes 1 to 8.
  for (unsigned chan = 1; chan <= 8; ++chan) {
    auto cs = graph.addComputeSet("cs" + std::to_string(chan));
    auto v = graph.addVertex(cs, vertex);
    graph.setTileMapping(v, 0);
    auto ws = graph.addConstant(UNSIGNED_SHORT,
                                {windowSizes.size()}, windowSizes.data());
    graph.connect(v["windowSizes"], ws);

    // create tensors.
    auto addTensor = [&](const char *name, const Vec2D &data, bool hostRead) {
      graph.setFieldSize(v[name], data.size());
      for (unsigned i = 0; i < data.size(); ++i) {
        const auto tensorName = name + std::to_string(chan) + std::to_string(i);
        auto datumTensor = graph.addVariable(type, {chan});
        graph.setTileMapping(datumTensor, 0);
        graph.connect(v[name][i], datumTensor);
        graph.createHostWrite(tensorName, datumTensor);
        if (hostRead) {
          graph.createHostRead(tensorName, datumTensor);
        }
      }
    };

    addTensor("out", out, false);
    addTensor("in", in, false);
    addTensor("outGrad", outGrad, false);
    addTensor("inGrad", inGrad, true);

    prog.add(Execute(cs));
  }

  Engine e(graph, prog, {{"target.textSectionSizeInBytes", "0x4000" }});
  e.load(device);

  // write tensors to the device.
  for (unsigned chan = 1; chan <= 8; ++chan) {
    auto writeTensor = [&](const char *name, const Vec2D &data) {
      for (unsigned i = 0; i < data.size(); ++i) {
        const auto tensorName = name + std::to_string(chan) + std::to_string(i);
        std::unique_ptr<char[]> dst(new char[chan * target.getTypeSize(type)]);
        copy(target, data[i].data(), chan, type, dst.get());
        e.writeTensor(tensorName, dst.get());
      }
    };

    writeTensor("out", out);
    writeTensor("in", in);
    writeTensor("outGrad", outGrad);
    writeTensor("inGrad", inGrad);
  }

  e.run();

  // check results against the expected output.
  for (unsigned chan = 1; chan <= 8; ++chan) {
    for (unsigned i = 0; i < inGrad.size(); ++i) {
      std::unique_ptr<char[]> src(new char[chan * target.getTypeSize(type)]);
      e.readTensor("inGrad" + std::to_string(chan) + std::to_string(i),
                   src.get());

      std::vector<float> actual(chan);
      copy(target, type, src.get(), actual.data(), chan);

      BOOST_CHECK(checkIsClose("i=" + std::to_string(i), actual.data(), {chan},
                               expected[i].data(), chan, TOL, atol(type)));
    }
  }
}

BOOST_AUTO_TEST_CASE(MaxPoolingGradHalf) {
  testMaxPoolingGrad("popnn::MaxPoolingGrad<half>", HALF);
}

BOOST_AUTO_TEST_CASE(MaxPoolingGradFloat) {
  testMaxPoolingGrad("popnn::MaxPoolingGrad<float>", FLOAT);
}
