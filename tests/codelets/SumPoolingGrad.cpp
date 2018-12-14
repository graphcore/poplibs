#include "TestDevice.hpp"
#include <poplar/Engine.hpp>
#include "popnn/codelets.hpp"
#include "poplibs_test/Util.hpp"

#define BOOST_TEST_MODULE SumPoolingGrad
#include <boost/test/included/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

#define TOL 0.1 //tolerance of 0.1%

using Vec2D = std::vector<std::vector<float>>;

const std::array<unsigned short, 3> windowSizes = {2, 0, 3};

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
  {117.4411, 88.1558, 39.8974, 49.9515, 98.2402, 141.2071, 25.9801, 164.0981},
  {0, 0, 0, 0, 0, 0, 0, 0},
  {177.0942, 102.8464, 156.4318, 143.5466, 116.723, 97.8598, 251.594, 199.4157},
};

double atol(const Type &type) {
  return type == HALF ? 1e-7 : 1e-20;
}

void testSumPoolingGrad(const char *vertex, const Type &type) {
  auto device = createTestDevice(TEST_TARGET);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);

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

    addTensor("outGrad", outGrad, false);
    addTensor("inGrad", inGrad, true);

    prog.add(Execute(cs));
  }

  Engine e(graph, prog);
  device.bind([&](const Device &d) {
    e.load(d);

    // write tensors to the device.
    for (unsigned chan = 1; chan <= 8; ++chan) {
      auto writeTensor = [&](const char *name, const Vec2D &data) {
        for (unsigned i = 0; i < data.size(); ++i) {
          const auto tensorName =
            name + std::to_string(chan) + std::to_string(i);
          std::unique_ptr<char[]> dst(
            new char[chan * target.getTypeSize(type)]);
          copy(target, data[i].data(), chan, type, dst.get());
          e.writeTensor(tensorName, dst.get());
        }
      };

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

        BOOST_CHECK(checkIsClose("i=" + std::to_string(i), actual.data(),
                                 {chan}, expected[i].data(), chan, TOL,
                                 atol(type)));
      }
    }
  });
}

BOOST_AUTO_TEST_CASE(SumPoolingGradHalf) {
  testSumPoolingGrad("popnn::SumPoolingGrad<half>", HALF);
}

BOOST_AUTO_TEST_CASE(SumPoolingGradFloat) {
  testSumPoolingGrad("popnn::SumPoolingGrad<float>", FLOAT);
}
