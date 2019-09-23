#include <poplar/Engine.hpp>
#define BOOST_TEST_MODULE FloatPointBehaviour
#include <boost/test/unit_test.hpp>
#include "TestDevice.hpp"

#include "poplar/CSRFunctions.hpp"
#include "popops/codelets.hpp"
#include "popops/ElementWise.hpp"

using namespace poplar;
using namespace poplar::program;

BOOST_AUTO_TEST_CASE(FloarPointBehaviourCheckExceptsFloat) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Sequence prog;

  FloatingPointBehaviour behaviour;
  setFloatingPointBehaviour(graph, prog, behaviour, "Set");

  auto init = graph.addConstant(FLOAT, {1}, 1e30f);
  auto t = graph.addVariable(FLOAT, {1}, "t");
  prog.add(Copy(init, t));
  graph.setTileMapping(init, 0);
  graph.setTileMapping(t, 0);

  popops::mulInPlace(graph, t, t, prog);

  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    BOOST_CHECK_THROW(eng.run(), poplar::poplar_error);
  });
}

BOOST_AUTO_TEST_CASE(FloarPointBehaviourCheckExceptsHalf) {
  auto device = createTestDevice(TEST_TARGET);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);

  Sequence prog;

  FloatingPointBehaviour behaviour;
  setFloatingPointBehaviour(graph, prog, behaviour, "Set");

  auto init = graph.addConstant(HALF, {1}, 60000);
  auto t = graph.addVariable(HALF, {1}, "t");
  prog.add(Copy(init, t));
  graph.setTileMapping(init, 0);
  graph.setTileMapping(t, 0);

  popops::mulInPlace(graph, t, t, prog);

  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    BOOST_CHECK_THROW(eng.run(), poplar::poplar_error);
  });
}
