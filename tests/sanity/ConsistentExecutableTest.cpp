// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>

#define BOOST_TEST_MODULE ConsistentExecutableTest
#include <boost/test/unit_test.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;

// Check that the serialized executables are all identical.
void checkConsistency(const std::vector<Executable> &exes) {
  if (exes.size() < 2)
    throw std::runtime_error("test logic error");

  std::vector<std::string> binaries(exes.size());
  for (size_t i = 0; i < exes.size(); ++i) {
    std::stringstream os;
    exes[i].serialize(os);
    binaries[i] = os.str();
  }

  for (size_t i = 1; i < exes.size(); ++i)
    BOOST_CHECK_EQUAL(binaries[0], binaries[i]);
}

// Check that compiling a simple reduction is deterministic.
BOOST_AUTO_TEST_CASE(ConsistentExecutableReduction) {
  TestDevice device = createTestDevice(TEST_TARGET, 4, 1);

  constexpr bool rearrangeOnHost = true;
  constexpr size_t numExes = 2;
  constexpr float initialValue = 3;
  const std::vector<size_t> dims = {3, 3};
  const OptionFlags options = {};
  const Type partialsType = poplar::FLOAT;
  const Type outType = poplar::FLOAT;

  size_t numElements = !dims.empty();
  for (size_t dim : dims)
    numElements *= dim;

  std::vector<Executable> exes;
  exes.reserve(numExes);

  for (size_t i = 0; i < numExes; ++i) {
    Graph graph(device.getTarget(), replication_factor(2u));

    popops::addCodelets(graph);

    Tensor in = graph.addVariable(partialsType, dims, "in");
    graph.setTileMapping(in, 0);

    Tensor rate = graph.addVariable(FLOAT, {});
    graph.setTileMapping(rate, 0);
    graph.setInitialValue(rate, initialValue);

    DataStream h2d =
        graph.addHostToDeviceFIFO("h2d", partialsType, numElements);
    DataStream d2h = graph.addDeviceToHostFIFO("d2h", outType, dims[0]);

    Sequence prog;
    prog.add(Copy(h2d, in, rearrangeOnHost));
    Tensor out = popops::reduce(graph, in, outType, {0},
                                {popops::Operation::ADD, false, rate}, prog);
    prog.add(Copy(out, d2h, rearrangeOnHost));

    exes.push_back(poplar::compileGraph(graph, {prog}, options));
  }

  checkConsistency(exes);
}
