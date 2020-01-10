// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifdef __POPC__

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

class Convert : public Vertex {
public:
  Input<float> fraction;
  Input<Vector<float>> vIn;
  Output<Vector<half>> vOut;
  bool compute() {
    for (unsigned i = 0; i != vOut.size(); ++i) {
      vOut[i] = vIn[i] + fraction;
      asm(""); // asm to prevent auto-vectorisation which can change the
               // rounding
    }
    return true;
  }
};

#else

#include "TestDevice.hpp"
#include <poplar/Engine.hpp>
#include <poplar/exceptions.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#define BOOST_TEST_MODULE StochasticRound
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <iostream>

using namespace poplar;

using namespace poplar::program;

void checkDistribution(const std::vector<unsigned> &targetDistribution,
                       const OptionFlags &engineOptions) {
  auto device = createTestDevice(TEST_TARGET);
  auto &target = device.getTarget();
  Graph graph(target);
  graph.addCodelets(__FILE__);
  poprand::addCodelets(graph);
  auto prog = Sequence();
  std::size_t N = 100;
  // longer tests on Hw give better stats
  if (TEST_TARGET == DeviceType::Hw)
    N = 10000;
  std::vector<char> dataBuf(target.getTypeSize(HALF) * N);
  std::vector<float> data_out(N);
  float hFraction;

  unsigned tile = 0;
  auto cs = graph.addComputeSet("cs");
  auto seed = graph.addVariable(UNSIGNED_INT, {2}, "seed");
  graph.setInitialValue(seed[0], 0x55555555);
  graph.setInitialValue(seed[1], 0x55555555);
  poprand::setSeed(graph, seed, 0, prog, "setSeed");
  graph.setTileMapping(seed, 0);
  auto v0 = graph.addVertex(cs, "Convert");
  graph.setCycleEstimate(v0, 1);
  auto fraction = graph.addVariable(FLOAT, {1}, "fraction");
  auto in = graph.addVariable(FLOAT, {N}, "dataIn");
  auto out = graph.addVariable(HALF, {N}, "dataOut");
  graph.setTileMapping(fraction, tile);
  graph.setTileMapping(in, tile);
  graph.setTileMapping(out, tile);
  graph.setTileMapping(v0, tile);
  for (auto i = 0u; i != N; ++i)
    graph.setInitialValue(in[i], float(1024.0));

  graph.connect(v0["fraction"], fraction[0]);
  graph.connect(v0["vIn"], in);
  graph.connect(v0["vOut"], out);
  graph.createHostWrite("fraction", fraction);
  graph.createHostRead("dataOut", out);

  prog.add(Execute(cs));
  Engine e(graph, prog, engineOptions);

  device.bind([&](const Device &d) {
    e.load(d);

    for (unsigned f = 0; f < 10; ++f) {
      hFraction = f * 0.1f;
      e.writeTensor("fraction", &hFraction, &hFraction + 1);
      e.run();
      e.readTensor("dataOut", dataBuf.data(), dataBuf.data() + dataBuf.size());
      copyDeviceHalfToFloat(target, dataBuf.data(), data_out.data(), N);

      unsigned n_1024 = 0, n_1025 = 0;
      for (auto i = 0u; i != N; ++i) {
        if (float(data_out[i]) == 1024.)
          n_1024++;
        else if (float(data_out[i]) == 1025.)
          n_1025++;
        else
          BOOST_FAIL("impossible_value " + std::to_string(data_out[i]) +
                     "received");
      }
      std::cerr << "conversion " << f << " gave " << n_1024 << " * 1024 and "
                << n_1025 << " * 1025\n";
      if (TEST_TARGET == DeviceType::Cpu || isIpuModel(TEST_TARGET)) {
        // no stochastic rounding on CPU
        BOOST_CHECK(n_1024 == N && n_1025 == 0);
      } else {
        if (f != 0)
          BOOST_CHECK(n_1024 != 0 && n_1025 != 0);
        // on average f/10 elements will round up
        // Quite wide checks are required when the number of samples is low
        unsigned tol = N < 1000 ? N * 0.16 : N * 0.06;
        auto max = f * N / 10 + tol;
        auto min = f * N / 10 > tol ? f * N / 10 - tol : 0;
        BOOST_CHECK(n_1025 >= min);
        BOOST_CHECK(n_1025 <= max);
      }
      if (!targetDistribution.empty())
        BOOST_CHECK(n_1025 == targetDistribution[f]);
      BOOST_CHECK(n_1024 + n_1025 == N);
    }
  });
}

BOOST_AUTO_TEST_CASE(Basic) {
  const auto engineOptions =
      OptionFlags({{"prng.enable", "true"}, {"prng.seed", "0x123"}});
  checkDistribution({}, engineOptions);
}

BOOST_AUTO_TEST_CASE(Deterministic) {
  if (TEST_TARGET == DeviceType::Hw || isSimulator(TEST_TARGET)) {
    // Only hardware is non-deterministic
    // Sim is deterministic anyway, tested to check the test itself hasn't
    // been broken.
    const auto engineOptions =
        OptionFlags({{"prng.enable", "true"},
                     {"prng.seed", "0x123"},
                     {"target.deterministicWorkers", "true"}});
    std::vector<unsigned> target;
    // Simulator has a different random generator and is slow(!)
    if (TEST_TARGET == DeviceType::Hw)
      target = {0, 997, 1984, 3007, 4051, 5091, 6072, 7047, 8080, 9028};
    else
      target = {0, 12, 19, 29, 36, 46, 54, 64, 72, 84};
    checkDistribution(target, engineOptions);
  }
}

#endif
