// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GraphReplicationTest
#include <TestDevice.hpp>
#include <boost/test/unit_test.hpp>
#include <math.h>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

using namespace poplar;
using namespace poplar::program;

BOOST_AUTO_TEST_CASE(InitialStateReplication) {
  auto device = createTestDevice(TEST_TARGET, 1, 16);
  poplar::Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Sequence sequence;

  auto const1 = graph.addConstant(INT, {1}, 0);
  graph.setTileMapping(const1, 0);

  auto const2 = graph.addConstant(FLOAT, {1}, 0.0);
  graph.setTileMapping(const2, 0);
  auto c2_bcast = const2.broadcast(300, 0).reshape({3, 10, 10});

  auto counter = poputil::duplicate(graph, const1, sequence);
  auto to_slice = graph.clone(c2_bcast, "blah");
  sequence.add(poplar::program::Copy(c2_bcast, to_slice));
  poputil::mapTensorLinearly(graph, to_slice);
  {
    Sequence repeat_sequence;
    auto counter_unsigned = counter.reinterpret(UNSIGNED_INT);
    auto slice = popops::dynamicSlice(graph, to_slice, counter_unsigned, {0},
                                      {1}, repeat_sequence);
    auto val = graph.addConstant(FLOAT, {1}, 1.0);
    graph.setTileMapping(val, 0);
    auto val_bcast = val.broadcast(100, 0).reshape({1, 10, 10});
    popops::scaledAddTo(graph, slice, val_bcast, 1.0f, repeat_sequence);
    popops::dynamicUpdate(graph, to_slice, slice, counter_unsigned, {0}, {1},
                          repeat_sequence);

    // increase counter.
    auto one = graph.addConstant(INT, {1}, 1);
    graph.setTileMapping(one, 0);
    popops::scaledAddTo(graph, counter, one, 1.0f, repeat_sequence);

    sequence.add(poplar::program::Repeat(3, repeat_sequence));
  }
  auto fifo_out =
      graph.addDeviceToHostFIFO("out_0.0", FLOAT, to_slice.numElements());

  sequence.add(poplar::program::Copy(to_slice, fifo_out, false));

  const std::string runtimeVerify =
      (isSimulator(TEST_TARGET)) ? "true" : "false";
  Engine engine(graph, sequence,
                {// T8477: Find why test fails if this is removed:
                 {"debug.runtimeVerify", runtimeVerify}});
  float result[3][10][10];
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.connectStream("out_0.0", &result);
    engine.run(0);
  });

  for (size_t i = 0; i != 3; ++i) {
    for (size_t j = 0; j != 10; ++j) {
      for (size_t k = 0; k != 10; ++k) {
        BOOST_CHECK_EQUAL(result[i][j][k], 1);
      }
    }
  }
}
