#include "TestDevice.hpp"
#include <poplar/Engine.hpp>
#include "popops/codelets.hpp"
#include "poplibs_test/Util.hpp"

#define BOOST_TEST_MODULE ScaledAdd2D_fp
#include <boost/test/included/unit_test.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

#define TOL 0.1 //tolerance of 0.1%

// test data of lengths 1 to 16
const std::vector<std::vector<float>> data = {
  {36.8533},
  {23.5636, 23.7882},
  {24.417, 38.0108, 17.0022},
  {12.1692, 6.9111, 18.6011, 32.8726},
  {45.094, 24.5472, 37.5412, 3.4541, 6.9619},
  {10.4058, 20.8296, 33.4116, 16.1244, 25.7758, 48.9353},
  {6.7283, 46.5906, 42.5837, 1.6201, 47.1409, 35.1936, 40.1955},
  {47.5474, 36.7667, 30.1007, 15.7696, 10.2956, 36.4658, 28.5871, 29.7736},
  {29.6436, 35.528, 1.8762, 18.5469, 34.1792, 11.3447, 7.5079, 17.6522, 1.4099},
  {8.0924, 6.3349, 38.9663, 23.1158, 32.5319, 24.9817, 20.7309, 0.2784, 10.5053,
   9.5292},
  {45.321, 21.2337, 2.8621, 1.3115, 5.2678, 35.5605, 44.7426, 29.9398, 18.3015,
   7.652, 17.8568},
  {9.7856, 46.8125, 47.1037, 39.7729, 9.9586, 11.7717, 41.9851, 2.2573, 33.2076,
   3.7827, 1.2203, 12.4487},
  {34.2556, 39.2798, 24.5538, 30.5591, 12.5051, 15.4922, 25.2939, 27.9103,
   48.8992, 37.6403, 49.1898, 30.2812, 44.8177},
  {37.9318, 42.0591, 22.0478, 32.4315, 13.4697, 18.2585, 18.1887, 42.0544,
   13.2323, 39.8405, 0.9929, 16.7709, 6.0279, 27.7244},
  {37.5095, 29.3018, 42.4159, 41.1092, 15.3115, 8.1059, 49.794, 33.2661,
   12.0308, 32.1723, 20.4024, 33.2543, 45.788, 31.629, 10.0015},
  {15.4047, 20.302, 30.0201, 22.2119, 18.5737, 9.0296, 19.6283, 15.2062,
   29.6811, 26.4103, 3.3177, 37.487, 11.1615, 29.5318, 20.1781, 8.4898},
};

const std::vector<std::vector<float>> deltas = {
  {24.4383},
  {37.1046, 35.5357},
  {13.4149, 2.0323, 8.3695},
  {16.8299, 8.2711, 47.8028, 16.8223},
  {28.5599, 32.9726, 18.01, 49.5828, 26.3351},
  {43.593, 8.4078, 29.9879, 22.1314, 32.8828, 45.6865},
  {49.1341, 44.1327, 46.3776, 7.5378, 31.5884, 12.8373, 5.3876},
  {41.0354, 29.5542, 5.6224, 1.62, 23.3489, 42.2291, 18.367, 33.6943},
  {4.0742, 4.1536, 41.8209, 13.1041, 27.9982, 9.8072, 22.2375, 36.9369,
   35.2985},
  {34.6223, 13.4062, 23.1151, 28.9503, 25.0751, 5.6493, 26.9687, 36.45, 28.7066,
   22.3477},
  {11.8482, 38.347, 45.53, 40.5914, 22.6172, 5.8899, 49.6522, 10.6218, 5.4388,
   49.3297, 15.2486},
  {29.7998, 13.2817, 42.2754, 13.4615, 46.4793, 10.8529, 43.8179, 15.9517,
   14.1261, 46.1555, 24.9081, 13.8895},
  {4.6595, 46.7121, 16.9035, 41.9907, 24.343, 21.4885, 16.4146, 1.8442, 36.553,
   34.3669, 14.7804, 14.9641, 34.8731},
  {6.426, 29.7145, 25.934, 34.9078, 34.9429, 10.8451, 49.6866, 24.9291, 6.338,
   9.7048, 33.9664, 0.5189, 16.1818, 30.5154},
  {27.3815, 39.4755, 18.1972, 36.0831, 3.7732, 45.9714, 25.2575, 3.7553,
   47.3133, 5.6741, 5.8831, 20.8678, 1.2767, 20.6127, 37.955},
  {29.0532, 40.3651, 44.8964, 1.4079, 0.9379, 19.0102, 8.4806, 10.0201, 31.092,
   34.0013, 11.8073, 20.0071, 49.0702, 25.1766, 5.3527, 9.115},
};

float k = 2.5653;

double atol(const Type &type) {
  return type == HALF ? 1e-7 : 1e-20;
}

void testScaledAdd2D(const char *vertex, const Type &type,
                                          const bool &constantFactor,
                                          const bool &doSubtract) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

   // Generate the expected result
  std::vector<std::vector<float>> expected(data.size());

  for(unsigned i = 0; i < data.size(); i++) {
    expected[i].resize(data[i].size());
    for(unsigned j = 0; j < data[i].size(); j++) {
      if(doSubtract)
        expected[i][j] = data[i][j] - deltas[i][j] * k;
      else
        expected[i][j] = data[i][j] + deltas[i][j] * k;
    }
  }

  const auto &target = device.getTarget();

  auto cs = graph.addComputeSet("cs");
  auto v = graph.addVertex(cs, vertex);
  graph.setTileMapping(v, 0);
  graph.setFieldSize(v["data"], data.size());
  graph.setFieldSize(v["deltas"], deltas.size());

  if(constantFactor) {
    graph.setInitialValue(v["K"], k);
  }
  else {
    auto factorTensor = graph.addVariable(type, {});
    graph.setTileMapping(factorTensor, 0);
    graph.connect(v["factor"], factorTensor);
    graph.setInitialValue(factorTensor, k);
  }

  // create tensors for each of the input rows.
  assert(data.size() == deltas.size());
  for (unsigned i = 0; i < data.size(); ++i) {
    const auto size = data[i].size();
    assert(size == deltas[i].size());

    auto datumTensor = graph.addVariable(type, {size});
    graph.setTileMapping(datumTensor, 0);
    graph.connect(v["data"][i], datumTensor);
    graph.createHostRead("datum" + std::to_string(i), datumTensor);
    graph.createHostWrite("datum" + std::to_string(i), datumTensor);

    auto deltaTensor = graph.addVariable(type, {size});
    graph.setTileMapping(deltaTensor, 0);
    graph.connect(v["deltas"][i], deltaTensor);
    graph.createHostWrite("delta" + std::to_string(i), deltaTensor);
  }

  Execute prog(cs);
  Engine e(graph, prog);
  device.bind([&](const Device &d) {
    e.load(d);

    // write tensors to the device.
    for (unsigned i = 0; i < data.size(); ++i) {
      const auto &datum = data[i];
      const auto &delta = deltas[i];
      const auto size = datum.size();

      std::unique_ptr<char[]> dst(new char[size * target.getTypeSize(type)]);
      copy(target, datum.data(), size, type, dst.get());
      e.writeTensor("datum" + std::to_string(i), dst.get());

      copy(target, delta.data(), size, type, dst.get());
      e.writeTensor("delta" + std::to_string(i), dst.get());
    }

    e.run();

    // check results against the expected output.
    for (unsigned i = 0; i < data.size(); ++i) {
      const auto &datum = data[i];
      const auto size = datum.size();

      std::unique_ptr<char[]> src(new char[size * target.getTypeSize(type)]);
      e.readTensor("datum" + std::to_string(i), src.get());

      std::vector<float> actual(size);
      copy(target, type, src.get(), actual.data(), size);

      BOOST_CHECK(checkIsClose("i=" + std::to_string(i), actual.data(), {size},
                               expected[i].data(), size, TOL, atol(type)));
    }
  });
}

BOOST_AUTO_TEST_CASE(ScaledAdd2DHalfConst) {
  testScaledAdd2D("popops::ScaledAdd2D<half,true>", HALF, true, false);
}

BOOST_AUTO_TEST_CASE(ScaledAdd2DHalfTensor) {
  testScaledAdd2D("popops::ScaledAdd2D<half,false>", HALF, false, false);
}

BOOST_AUTO_TEST_CASE(ScaledSubtract2DHalfTensor) {
  testScaledAdd2D("popops::ScaledSubtract2D<half>", HALF, false, true);
}

BOOST_AUTO_TEST_CASE(ScaledAdd2DFloatConst) {
  testScaledAdd2D("popops::ScaledAdd2D<float,true>", FLOAT, true, false);
}
BOOST_AUTO_TEST_CASE(ScaledAdd2DFloatTensor) {
  testScaledAdd2D("popops::ScaledAdd2D<float,false>", FLOAT, false, false);
}
BOOST_AUTO_TEST_CASE(ScaledSubtract2DFloatTensor) {
  testScaledAdd2D("popops::ScaledSubtract2D<float>", FLOAT, false, true);
}
