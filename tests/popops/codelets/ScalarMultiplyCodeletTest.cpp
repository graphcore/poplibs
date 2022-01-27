// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ScalarMultiplyTest

#include <algorithm>
#include <iostream>
#include <numeric>

#include "CodeletsTestsCommon.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Expr.hpp"
#include "popops/ScalarMultiply.hpp"
#include "popops/codelets.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include <boost/cstdfloat.hpp>
#include <boost/multi_array.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplibs_support/TestDevice.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace popops;
namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;
namespace tt = boost::test_tools;

using TensorTuple = std::tuple<Tensor, Tensor>;

static constexpr float halfMax = 65504;

static unsigned ceilDivide(unsigned a, unsigned b) { return (a + b - 1) / b; }

static unsigned roundUpToNextMultiple(unsigned n, unsigned m) {
  // n - the value to round up
  // m - the multiple
  return n % m == 0 ? n + m : ceilDivide(n, m) * m;
}

struct Fixture {
  Fixture() {
    namespace po = boost::program_options;
    po::options_description poDesc("Test the ScalarMultiply codelets.");

    size_t aSize;

    // clang-format off
    poDesc.add_options()
        ("a-size",
         po::value<size_t>(&aSize)->multitoken()->required(),
         "The size of `a` in `a * b` (e.g. `42`).")
        ("inplace",
         po::value<bool>(&inplace)->multitoken()->required(),
         "Perfoms the operation inplace if `true`.")
        ("device-type",
         po::value<DeviceType>(&dt)->required(),
         "Device type.")
        ("n-regions",
         po::value<unsigned>(&nRegions)->default_value(1),
         "Number of regions to split the data into.")
        ("tol",
         po::value<double>(&tol)->default_value(1e-6),
         "Float to half conversion accuracy tolerance.")
        ("extra-regions-size",
         po::value<unsigned>(&extraRegionsSize)->default_value(5),
         "The size of extra regions.")
        ;
    // clang-format on

    int argc = boost::unit_test::framework::master_test_suite().argc;
    char **argv = boost::unit_test::framework::master_test_suite().argv;
    parseOptions(argc, argv, poDesc);

    aShape = {aSize};
    aData = std::vector<float>(aSize);
    float step = 0.0001;
    for (unsigned i = 0; i < aSize; i++)
      aData[i] = step * i;
    aData[0] = 2; // Add one value that will clip the result at max(half).
    // Clip nRegions to the max value that can be satisfied by `aSize`.
    nRegions = std::min(nRegions, ceilDivide(aSize, extraRegionsSize));
  };

  std::vector<size_t> aShape;
  std::vector<size_t> bShape{1};
  Type aType = HALF;
  Type bType = FLOAT;
  std::vector<float> aData{};
  std::vector<float> bData{65536}; // Scalar is outside half range (2^16).
  bool inplace;
  DeviceType dt;
  unsigned nRegions;
  unsigned extraRegionsSize;
  double tol;
};

static TensorTuple addVariable(Graph &graph, const Type &type,
                               const std::vector<size_t> &shape,
                               const std::string &name, unsigned nRegions,
                               unsigned extraRegionsSize) {
  const auto elementsPerLoop = 4;
  const unsigned size = std::accumulate(shape.begin(), shape.end(), 0);

  BOOST_REQUIRE(nRegions > 0);
  BOOST_REQUIRE(extraRegionsSize * (nRegions - 1) <= size);

  const unsigned remainingRegionSize = size - (nRegions - 1) * extraRegionsSize;
  const unsigned maxRegionSize =
      std::max(remainingRegionSize, extraRegionsSize);

  Tensor tExt = graph.addVariable(
      type, {nRegions, roundUpToNextMultiple(maxRegionSize, elementsPerLoop)},
      name + "Ext");
  graph.setTileMapping(tExt, 0);

  Tensor t = tExt[0].slice(0, remainingRegionSize);

  for (unsigned i = 1; i < nRegions; i++) {
    t = concat(t, tExt[i].slice(0, extraRegionsSize));
  }

  t = t.reshape(shape);

  return {t, tExt};
}

static void buildGraph(Graph &graph, const Tensor &a, const Tensor &b,
                       const Tensor &c, Sequence &prog, bool inplace,
                       double tol) {
  BOOST_REQUIRE(graph.getTileMapping(a).size() == 1);
  BOOST_REQUIRE(graph.getTileMapping(b).size() == 1);
  BOOST_REQUIRE(a.shape().size() == 1);
  BOOST_REQUIRE(b.shape().size() == 1);
  BOOST_REQUIRE(inputsMatchMixedPrecisionScalarMultiplyPattern(a, b, !inplace));

  const auto cs = graph.addComputeSet({});

  if (a.getContiguousRegions().size() == 1) {
    std::string vertexName = "popops::ScalarMultiply1D";
    vertexName += inplace ? "Inplace" : "";
    vertexName = poputil::templateVertex(vertexName, HALF, FLOAT);
    const auto vertex = graph.addVertex(cs, vertexName);
    if (inplace) {
      graph.connect(vertex["in1Out"], a);
    } else {
      graph.connect(vertex["in1"], a);
      graph.connect(vertex["out"], c);
    }
    graph.connect(vertex["in2"], b);
    graph.setInitialValue(vertex["tolerance"], tol);
    graph.setTileMapping(vertex, 0);
  } else {
    std::string vertexName = "popops::ScalarMultiply2D";
    vertexName += inplace ? "Inplace" : "";
    vertexName = poputil::templateVertex(vertexName, HALF, FLOAT);
    const auto vertex = graph.addVertex(cs, vertexName);
    auto regions = a.getContiguousRegions();
    if (inplace) {
      graph.connect(vertex["in1Out"], a.slices(regions));
    } else {
      graph.connect(vertex["in1"], a.slices(regions));
      graph.connect(vertex["out"], c.slices(regions));
    }
    graph.connect(vertex["in2"], b);
    graph.setInitialValue(vertex["tolerance"], tol);
    graph.setTileMapping(vertex, 0);
  }

  prog.add(poplar::program::Execute(cs));
}

BOOST_FIXTURE_TEST_CASE(
    FunctionalTest, Fixture,
    *utf::tolerance<float>(fpc::percent_tolerance<float>(0.2f))) {
  auto device = createTestDevice(dt);
  const auto &target = device.getTarget();

  Graph graph(target);
  popops::addCodelets(graph);

  Sequence uploadProg, prog, downloadProg;

  setFloatingPointBehaviour(graph, prog,
                            {
                                false, // exceptOnInv
                                true,  // exceptOnDiv0
                                false, // exceptOnOflo
                                false, // enableStochasticRounding
                                false, // nanOnOverflow
                            },
                            "");

  Tensor a, b, c, cExt;

  if (inplace) {
    std::tie(a, cExt) =
        addVariable(graph, aType, aShape, "a", nRegions, extraRegionsSize);
  } else {
    std::tie(a, std::ignore) =
        addVariable(graph, aType, aShape, "a", nRegions, extraRegionsSize);
    std::tie(c, cExt) =
        addVariable(graph, aType, aShape, "c", nRegions, extraRegionsSize);
  }

  std::tie(b, std::ignore) = addVariable(graph, bType, bShape, "b", 1, 0);

  // Sanity check that `c` is not exactly `cExt`.
  if (inplace) {
    BOOST_REQUIRE(cExt.numElements() > a.numElements());
  } else {
    BOOST_REQUIRE(cExt.numElements() > c.numElements());
  }

  graph.createHostWrite("a", a);
  graph.createHostWrite("b", b);
  graph.createHostWrite("cExt", cExt);
  graph.createHostRead("cExtOut", cExt);

  buildGraph(graph, a, b, c, prog, inplace, tol);

  std::vector<float> aHost{aData};
  std::vector<float> bHost{bData};
  std::vector<float> cExtHost(cExt.numElements());
  std::vector<float> cExtHostOut(cExt.numElements());

  // Fill the extended `c` tensor with a range of negative values. These will be
  // used to test for overwriting later.
  std::iota(cExtHost.begin(), cExtHost.end(),
            -static_cast<float>(cExt.numElements()));

  std::vector<char> aRaw(target.getTypeSize(HALF) * a.numElements());
  std::vector<char> cExtRaw(target.getTypeSize(HALF) * cExt.numElements());

  poplar::copyFloatToDeviceHalf(target, aHost.data(), aRaw.data(),
                                a.numElements());
  poplar::copyFloatToDeviceHalf(target, cExtHost.data(), cExtRaw.data(),
                                cExt.numElements());

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, {});
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.writeTensor("cExt", cExtRaw.data(), cExtRaw.data() + cExtRaw.size());
    engine.writeTensor("a", aRaw.data(), aRaw.data() + aRaw.size());
    engine.writeTensor("b", bHost.data(), bHost.data() + bHost.size());
    engine.run(0);
    engine.readTensor("cExtOut", cExtRaw.data(),
                      cExtRaw.data() + cExtRaw.size());
  });

  poplar::copyDeviceHalfToFloat(target, cExtRaw.data(), cExtHostOut.data(),
                                cExt.numElements());

  if (!inplace) {
    BOOST_TEST(c.shape() == a.shape());
    BOOST_TEST(c.elementType() == a.elementType());
  }

  std::vector<float> cExpected(cExtHost);
  const auto stride = cExt.shape()[1];

  unsigned k = 0;
  const unsigned remainingRegionSize =
      aShape[0] - (nRegions - 1) * extraRegionsSize;
  for (; k < remainingRegionSize; k++) {
    cExpected[k] = std::min(aHost[k] * bHost[0], halfMax);
  }

  for (unsigned i = 1; i < cExt.shape()[0]; i++) {
    for (unsigned j = 0; j < extraRegionsSize; j++) {
      cExpected[i * stride + j] = std::min(aHost[k++] * bHost[0], halfMax);
    }
  }

  BOOST_TEST_CHECK(cExpected == cExtHostOut, tt::per_element());
}
