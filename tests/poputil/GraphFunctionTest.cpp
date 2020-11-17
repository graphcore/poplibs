// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GraphFunctionTest
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/GraphFunction.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

BOOST_AUTO_TEST_CASE(VoidFunctionTest) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x1);
  graph.createHostRead("x1", x1);
  graph.createHostWrite("x1", x1);
  Tensor y1 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y1", y1);
  mapTensorLinearly(graph, y1);
  graphfn::VoidFunction f(graph, {graphfn::inout(x1), graphfn::input(y1)},
                          [&](std::vector<Tensor> &args, Sequence &prog) {
                            popops::addInPlace(graph, args[0], args[1], prog);
                          });
  Tensor x2 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x2);
  graph.createHostRead("x2", x2);
  graph.createHostWrite("x2", x2);
  Tensor y2 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y2", y2);
  mapTensorLinearly(graph, y2);
  Sequence prog;
  std::vector<Tensor> args1 = {x1, y1};
  f(args1, prog);
  std::vector<Tensor> args2 = {x2, y2};
  f(args2, prog);
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    std::vector<float> hx1 = {5, 3, 1, 7, 9};
    std::vector<float> hy1 = {55, 3, 2, 8, 4};
    std::vector<float> hx2 = {99, 2, 0, 3, 6};
    std::vector<float> hy2 = {23, 1, 66, 8, 22};
    eng.writeTensor("x1", hx1.data(), hx1.data() + hx1.size());
    eng.writeTensor("y1", hy1.data(), hy1.data() + hy1.size());
    eng.writeTensor("x2", hx2.data(), hx2.data() + hx2.size());
    eng.writeTensor("y2", hy2.data(), hy2.data() + hy2.size());
    eng.run();

    std::vector<float> result(5);
    eng.readTensor("x1", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx1.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx1[i] + hy1[i]);
    eng.readTensor("x2", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx2.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx2[i] + hy2[i]);
  });
}

BOOST_AUTO_TEST_CASE(VoidFunctionTest_WithDebugInfo) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x1);
  graph.createHostRead("x1", x1);
  graph.createHostWrite("x1", x1);
  Tensor y1 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y1", y1);
  mapTensorLinearly(graph, y1);
  graphfn::VoidFunction f(graph, {graphfn::inout(x1), graphfn::input(y1)},
                          [&](std::vector<Tensor> &args, Sequence &prog,
                              const poplar::DebugNameAndId &dnai) {
                            popops::addInPlace(graph, args[0], args[1], prog,
                                               dnai);
                          });
  Tensor x2 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x2);
  graph.createHostRead("x2", x2);
  graph.createHostWrite("x2", x2);
  Tensor y2 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y2", y2);
  mapTensorLinearly(graph, y2);
  Sequence prog;
  std::vector<Tensor> args1 = {x1, y1};
  f(args1, prog, "call1");
  std::vector<Tensor> args2 = {x2, y2};
  f(args2, prog, "call2");
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    std::vector<float> hx1 = {5, 3, 1, 7, 9};
    std::vector<float> hy1 = {55, 3, 2, 8, 4};
    std::vector<float> hx2 = {99, 2, 0, 3, 6};
    std::vector<float> hy2 = {23, 1, 66, 8, 22};
    eng.writeTensor("x1", hx1.data(), hx1.data() + hx1.size());
    eng.writeTensor("y1", hy1.data(), hy1.data() + hy1.size());
    eng.writeTensor("x2", hx2.data(), hx2.data() + hx2.size());
    eng.writeTensor("y2", hy2.data(), hy2.data() + hy2.size());
    eng.run();

    std::vector<float> result(5);
    eng.readTensor("x1", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx1.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx1[i] + hy1[i]);
    eng.readTensor("x2", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx2.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx2[i] + hy2[i]);
  });
}

BOOST_AUTO_TEST_CASE(ProgramFunctionTest) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x1);
  graph.createHostWrite("x1", x1);
  Tensor y1 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y1", y1);
  mapTensorLinearly(graph, y1);
  Tensor z1 = graph.addVariable(FLOAT, {5});
  graph.createHostRead("z1", z1);
  mapTensorLinearly(graph, z1);
  graphfn::ProgramFunction f(
      graph, {graphfn::input(x1), graphfn::input(y1), graphfn::output(z1)},
      [&](std::vector<Tensor> &args) {
        Sequence prog;
        prog.add(Copy(args[0], args[2]));
        popops::addInPlace(graph, args[2], args[1], prog);
        return prog;
      });
  Tensor x2 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x2);
  graph.createHostRead("x2", x2);
  graph.createHostWrite("x2", x2);
  Tensor y2 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y2", y2);
  mapTensorLinearly(graph, y2);
  Tensor z2 = graph.addVariable(FLOAT, {5});
  graph.createHostRead("z2", z2);
  mapTensorLinearly(graph, z2);
  Sequence prog;
  std::vector<Tensor> args1 = {x1, y1, z1};
  prog.add(f(args1));
  std::vector<Tensor> args2 = {x2, y2, z2};
  prog.add(f(args2));
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    std::vector<float> hx1 = {5, 3, 1, 7, 9};
    std::vector<float> hy1 = {55, 3, 2, 8, 4};
    std::vector<float> hx2 = {99, 2, 0, 3, 6};
    std::vector<float> hy2 = {23, 1, 66, 8, 22};
    eng.writeTensor("x1", hx1.data(), hx1.data() + hx1.size());
    eng.writeTensor("y1", hy1.data(), hy1.data() + hy1.size());
    eng.writeTensor("x2", hx2.data(), hx2.data() + hx2.size());
    eng.writeTensor("y2", hy2.data(), hy2.data() + hy2.size());
    eng.run();

    std::vector<float> result(5);
    eng.readTensor("z1", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx1.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx1[i] + hy1[i]);
    eng.readTensor("z2", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx2.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx2[i] + hy2[i]);
  });
}

BOOST_AUTO_TEST_CASE(CreatedTensorFunctionTest) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x1);
  graph.createHostWrite("x1", x1);
  Tensor y1 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y1", y1);
  mapTensorLinearly(graph, y1);
  graphfn::ProgramFunction f(
      graph, {graphfn::input(x1), graphfn::input(y1), graphfn::created()},
      [&](std::vector<Tensor> &args) {
        Sequence prog;
        args[2] = graph.addVariable(FLOAT, {5});
        mapTensorLinearly(graph, args[2]);
        prog.add(Copy(args[0], args[2]));
        popops::addInPlace(graph, args[2], args[1], prog);
        return prog;
      });
  Tensor x2 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x2);
  graph.createHostRead("x2", x2);
  graph.createHostWrite("x2", x2);
  Tensor y2 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y2", y2);
  mapTensorLinearly(graph, y2);
  Sequence prog;
  std::vector<Tensor> args1 = {x1, y1, Tensor()};
  prog.add(f(args1));
  std::vector<Tensor> args2 = {x2, y2, Tensor()};
  prog.add(f(args2));
  auto &z1 = args1[2];
  auto &z2 = args2[2];
  graph.createHostRead("z1", z1);
  graph.createHostRead("z2", z2);
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    std::vector<float> hx1 = {5, 3, 1, 7, 9};
    std::vector<float> hy1 = {55, 3, 2, 8, 4};
    std::vector<float> hx2 = {99, 2, 0, 3, 6};
    std::vector<float> hy2 = {23, 1, 66, 8, 22};
    eng.writeTensor("x1", hx1.data(), hx1.data() + hx1.size());
    eng.writeTensor("y1", hy1.data(), hy1.data() + hy1.size());
    eng.writeTensor("x2", hx2.data(), hx2.data() + hx2.size());
    eng.writeTensor("y2", hy2.data(), hy2.data() + hy2.size());
    eng.run();

    std::vector<float> result(5);
    eng.readTensor("z1", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx1.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx1[i] + hy1[i]);
    eng.readTensor("z2", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx2.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx2[i] + hy2[i]);
  });
}

BOOST_AUTO_TEST_CASE(CreatedTensorFunctionTest_WithDebugInfo) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());

  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x1);
  graph.createHostWrite("x1", x1);
  Tensor y1 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y1", y1);
  mapTensorLinearly(graph, y1);
  graphfn::ProgramFunction f(
      graph, {graphfn::input(x1), graphfn::input(y1), graphfn::created()},
      [&](std::vector<Tensor> &args, const poplar::DebugNameAndId &dnai) {
        Sequence prog;
        args[2] = graph.addVariable(FLOAT, {5}, {dnai, "v1"});
        mapTensorLinearly(graph, args[2]);
        prog.add(Copy(args[0], args[2], false, {dnai}));
        popops::addInPlace(graph, args[2], args[1], prog, {dnai});
        return prog;
      });
  Tensor x2 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x2);
  graph.createHostRead("x2", x2);
  graph.createHostWrite("x2", x2);
  Tensor y2 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y2", y2);
  mapTensorLinearly(graph, y2);
  Sequence prog;
  std::vector<Tensor> args1 = {x1, y1, Tensor()};
  prog.add(f(args1, "call1"));
  std::vector<Tensor> args2 = {x2, y2, Tensor()};
  prog.add(f(args2, "call2"));
  auto &z1 = args1[2];
  auto &z2 = args2[2];
  graph.createHostRead("z1", z1);
  graph.createHostRead("z2", z2);
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    std::vector<float> hx1 = {5, 3, 1, 7, 9};
    std::vector<float> hy1 = {55, 3, 2, 8, 4};
    std::vector<float> hx2 = {99, 2, 0, 3, 6};
    std::vector<float> hy2 = {23, 1, 66, 8, 22};
    eng.writeTensor("x1", hx1.data(), hx1.data() + hx1.size());
    eng.writeTensor("y1", hy1.data(), hy1.data() + hy1.size());
    eng.writeTensor("x2", hx2.data(), hx2.data() + hx2.size());
    eng.writeTensor("y2", hy2.data(), hy2.data() + hy2.size());
    eng.run();

    std::vector<float> result(5);
    eng.readTensor("z1", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx1.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx1[i] + hy1[i]);
    eng.readTensor("z2", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx2.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx2[i] + hy2[i]);
  });
}

BOOST_AUTO_TEST_CASE(TensorFunctionTest) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x1);
  graph.createHostRead("x1", x1);
  graph.createHostWrite("x1", x1);
  Tensor y1 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y1", y1);
  mapTensorLinearly(graph, y1);
  graphfn::TensorFunction f(graph, {graphfn::inout(x1), graphfn::input(y1)},
                            [&](std::vector<Tensor> &args, Sequence &prog) {
                              Tensor z = graph.addVariable(FLOAT, {5});
                              mapTensorLinearly(graph, z);
                              prog.add(Copy(args[0], z));
                              popops::addInPlace(graph, z, args[1], prog);
                              return z;
                            });
  Tensor x2 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x2);
  graph.createHostRead("x2", x2);
  graph.createHostWrite("x2", x2);
  Tensor y2 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y2", y2);
  mapTensorLinearly(graph, y2);
  Sequence prog;
  std::vector<Tensor> args1 = {x1, y1};
  auto z1 = f(args1, prog);
  std::vector<Tensor> args2 = {x2, y2};
  auto z2 = f(args2, prog);
  graph.createHostRead("z1", z1);
  graph.createHostRead("z2", z2);
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    std::vector<float> hx1 = {5, 3, 1, 7, 9};
    std::vector<float> hy1 = {55, 3, 2, 8, 4};
    std::vector<float> hx2 = {99, 2, 0, 3, 6};
    std::vector<float> hy2 = {23, 1, 66, 8, 22};
    eng.writeTensor("x1", hx1.data(), hx1.data() + hx1.size());
    eng.writeTensor("y1", hy1.data(), hy1.data() + hy1.size());
    eng.writeTensor("x2", hx2.data(), hx2.data() + hx2.size());
    eng.writeTensor("y2", hy2.data(), hy2.data() + hy2.size());
    eng.run();

    std::vector<float> result(5);
    eng.readTensor("z1", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx1.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx1[i] + hy1[i]);
    eng.readTensor("z2", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx2.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx2[i] + hy2[i]);
  });
}

BOOST_AUTO_TEST_CASE(TensorFunctionTest_WithDebugInfo) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Tensor x1 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x1);
  graph.createHostRead("x1", x1);
  graph.createHostWrite("x1", x1);
  Tensor y1 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y1", y1);
  mapTensorLinearly(graph, y1);
  graphfn::TensorFunction f(
      graph, {graphfn::inout(x1), graphfn::input(y1)},
      [&](std::vector<Tensor> &args, Sequence &prog,
          const poplar::DebugNameAndId &dnai) {
        Tensor z = graph.addVariable(FLOAT, {5}, {dnai, "z"});
        mapTensorLinearly(graph, z);
        prog.add(Copy(args[0], z, false, {dnai}));
        popops::addInPlace(graph, z, args[1], prog, {dnai, "y"});
        return z;
      });
  Tensor x2 = graph.addVariable(FLOAT, {5});
  mapTensorLinearly(graph, x2);
  graph.createHostRead("x2", x2);
  graph.createHostWrite("x2", x2);
  Tensor y2 = graph.addVariable(FLOAT, {5});
  graph.createHostWrite("y2", y2);
  mapTensorLinearly(graph, y2);
  Sequence prog;
  std::vector<Tensor> args1 = {x1, y1};
  auto z1 = f(args1, prog, "call1");
  std::vector<Tensor> args2 = {x2, y2};
  auto z2 = f(args2, prog, "call2");
  graph.createHostRead("z1", z1);
  graph.createHostRead("z2", z2);
  Engine eng(graph, prog);
  device.bind([&](const Device &d) {
    eng.load(d);
    std::vector<float> hx1 = {5, 3, 1, 7, 9};
    std::vector<float> hy1 = {55, 3, 2, 8, 4};
    std::vector<float> hx2 = {99, 2, 0, 3, 6};
    std::vector<float> hy2 = {23, 1, 66, 8, 22};
    eng.writeTensor("x1", hx1.data(), hx1.data() + hx1.size());
    eng.writeTensor("y1", hy1.data(), hy1.data() + hy1.size());
    eng.writeTensor("x2", hx2.data(), hx2.data() + hx2.size());
    eng.writeTensor("y2", hy2.data(), hy2.data() + hy2.size());
    eng.run();

    std::vector<float> result(5);
    eng.readTensor("z1", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx1.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx1[i] + hy1[i]);
    eng.readTensor("z2", result.data(), result.data() + result.size());
    for (unsigned i = 0; i < hx2.size(); ++i)
      BOOST_CHECK_EQUAL(result[i], hx2[i] + hy2[i]);
  });
}
