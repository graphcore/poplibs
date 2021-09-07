// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ScalarMultiplyTest

#include <algorithm>
#include <iostream>
#include <numeric>

#include "./codelets/CodeletsTestsCommon.hpp"
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
#include <boost/test/unit_test.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplibs_support/TestDevice.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace popops;

std::vector<float> clipData(const std::vector<float> &data,
                            const float &clipValue) {
  std::vector<float> clippedData(data.size());
  for (unsigned i = 0; i < clippedData.size(); i++) {
    clippedData[i] = std::min(data[i], clipValue);
  }
  return clippedData;
}

struct Tensor_ {
  Tensor_() = default;
  Tensor_(const std::vector<float> &data_,
          const std::vector<std::size_t> &shape_, const Type &poplar_type_)
      : data(data_), shape(shape_), poplar_type(poplar_type_) {}

  std::vector<float> data;
  std::vector<std::size_t> shape;
  Type poplar_type;

  static float getFP16ClipValue() { return 65504; }

  std::vector<float> getData(bool clip = true) const {
    if (clip && poplar_type == HALF) {
      return clipData(data, getFP16ClipValue());
    } else {
      return data;
    }
  }
};

Tensor addVariable(Graph &graph, const Type &type,
                   const std::vector<size_t> &shape, const std::string &name,
                   bool singleRegion) {
  unsigned size = std::accumulate(shape.begin(), shape.end(), 0);
  Tensor t;
  if (!singleRegion && size > 1) {
    Tensor t1 = graph.addVariable(type, {size / 2}, name + "1");
    Tensor t2 = graph.addVariable(type, {size - size / 2}, name + "2");
    poputil::mapTensorLinearly(graph, t1);
    poputil::mapTensorLinearly(graph, t2);
    t = concat(t1, t2);
    t = t.reshape(shape);
  } else {
    t = graph.addVariable(type, shape, name);
    poputil::mapTensorLinearly(graph, t);
  }
  return t;
}

struct TestResult {
  TestResult(poputil::poplibs_error error_) { errorMsg = error_.what(); }
  TestResult(Tensor_ c_) { c = c_; }

  std::string errorMsg;
  Tensor_ c;
};

struct Fixture {
  Fixture() {
    namespace po = boost::program_options;
    po::options_description poDesc("Test a specialisation of multiplication "
                                   "between a float scalar and a half tensor.");

    ShapeOption<size_t> aShape;
    ShapeOption<size_t> bShape;
    Type aType;
    Type bType;
    VectorOption<float> aData;
    VectorOption<float> bData;

    poDesc.add_options()                                                    //
        ("a-shape",                                                         //
         po::value<ShapeOption<size_t>>(&aShape)->multitoken()->required(), //
         "The shape of `a` in `a * b` (e.g. `{2,2}`).")                     //
        ("a-type",                                                          //
         po::value<Type>(&aType)->multitoken()->required(),                 //
         "The type of `a` in `a * b` (e.g. `half` or `float`).")            //
        ("a-data",                                                          //
         po::value<VectorOption<float>>(&aData)->multitoken()->required(),  //
         "The data of `a` in `a * b` (e.g. `{1,2,3,4}`).")                  //
        ("b-shape",                                                         //
         po::value<ShapeOption<size_t>>(&bShape)->multitoken()->required(), //
         "The shape of `b` in `a * b` (e.g. `{2,2}`).")                     //
        ("b-type",                                                          //
         po::value<Type>(&bType)->multitoken()->required(),                 //
         "The type of `b` in `a * b` (e.g. `half` or `float`).")            //
        ("b-data",                                                          //
         po::value<VectorOption<float>>(&bData)->multitoken()->required(),  //
         "The data of `b` in `a * b` (e.g. `{1,2,3,4}`).")                  //
        ("inplace",                                                         //
         po::value<bool>(&inplace)->multitoken()->required(),               //
         "Perfoms the operation inplace if `true`.")                        //
        ("device-type",                                                     //
         po::value<DeviceType>(&dt)->required(),                            //
         "Device type.")                                                    //
        ;

    int argc = boost::unit_test::framework::master_test_suite().argc;
    char **argv = boost::unit_test::framework::master_test_suite().argv;
    parseOptions(argc, argv, poDesc);

    a = Tensor_(aData.val, aShape.val, aType);
    b = Tensor_(bData.val, bShape.val, bType);
  };

  bool inplace;
  Tensor_ a;
  Tensor_ b;
  DeviceType dt;
};

enum class BuildGraphUsing {
  Map = 0,
  Codelet,
};

Tensor buildGraphUsingMap(Graph &graph, Tensor &a, Tensor &b, Sequence &prog,
                          bool inplace) {
  if (inplace) {
    mapInPlace(graph, expr::BinaryOpType::MULTIPLY, a, b, prog, "", {});
    return a;
  } else {
    Tensor c = map(graph, expr::BinaryOpType::MULTIPLY, a, b, prog, "", {});
    return c;
  }
}

Tensor buildGraphUsingCodelet(Graph &graph, Tensor &a, Tensor &b,
                              Sequence &prog, bool inplace) {
  poputil::PoplibsOpDebugInfo di("");
  if (inplace) {
    scalarMultiplyInplace(graph, a, b, prog, di);
    return a;
  } else {
    auto c = scalarMultiply(graph, a, b, prog, di);
    return c;
  }
}

TestResult runTest(BuildGraphUsing buildGraphMethod, const Tensor_ &a_,
                   const Tensor_ &b_, const DeviceType &dt, bool inplace) {
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

  Tensor a = addVariable(graph, a_.poplar_type, a_.shape, "a", true);
  Tensor b = addVariable(graph, b_.poplar_type, b_.shape, "b", true);

  Tensor c;
  try {
    switch (buildGraphMethod) {
    case BuildGraphUsing::Map:
      c = buildGraphUsingMap(graph, a, b, prog, inplace);
      break;
    case BuildGraphUsing::Codelet:
      c = buildGraphUsingCodelet(graph, a, b, prog, inplace);
      break;
    default:
      throw std::runtime_error("Invalid `buildGraphMethod`.");
      break;
    }
  } catch (const poputil::poplibs_error &e) {
    return TestResult(e);
  }

  std::vector<std::pair<std::string, char *>> tmap;
  std::vector<float> aHost{a_.getData()};
  std::vector<float> bHost{b_.getData()};
  std::vector<float> cHost(c.numElements());
  auto aRaw = allocateHostMemoryForTensor(a, "a", graph, uploadProg,
                                          downloadProg, tmap);
  auto bRaw = allocateHostMemoryForTensor(b, "b", graph, uploadProg,
                                          downloadProg, tmap);
  std::unique_ptr<char[]> cRaw;
  char *cRawPtr = nullptr;

  if (inplace) {
    cRawPtr = aRaw.get();
  } else {
    cRaw = allocateHostMemoryForTensor(c, "c", graph, uploadProg, downloadProg,
                                       tmap);
    cRawPtr = cRaw.get();
  }

  copy(target, aHost, a.elementType(), aRaw.get());
  copy(target, bHost, b.elementType(), bRaw.get());

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, {});
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0);
  });

  copy(target, c.elementType(), cRawPtr, cHost);

  return TestResult(Tensor_(cHost, c.shape(), c.elementType()));
}

void verifyResult(BuildGraphUsing buildGraphMethod, const Tensor_ &a,
                  const Tensor_ &b, bool inplace, const TestResult &result) {

  auto isScalar = [](const Tensor_ &t) {
    return t.shape == std::vector<size_t>{1};
  };

  auto isAnyScalar = [&isScalar](const Tensor_ &t0, const Tensor_ &t1) {
    return isScalar(t0) || isScalar(t1);
  };

  auto isFloatScalar = [&isScalar](const Tensor_ &t) {
    return isScalar(t) && t.poplar_type == FLOAT;
  };

  auto isAnyFloatScalar = [&isFloatScalar](const Tensor_ &t0,
                                           const Tensor_ &t1) {
    return isFloatScalar(t0) || isFloatScalar(t1);
  };

  // Can't have both a valid result and an exception.
  BOOST_TEST((result.errorMsg == "" || result.c.data.size() == 0));

  if (buildGraphMethod == BuildGraphUsing::Map) {
    if ((a.poplar_type != b.poplar_type) &&
        (!(inplace && isFloatScalar(b)) &&
         !(!inplace && isAnyFloatScalar(a, b)))) {
      BOOST_TEST(result.c.data.size() == 0);
      std::string expectedMsg = "Error inferring types in expression:";
      BOOST_TEST(boost::algorithm::starts_with(result.errorMsg, expectedMsg));
      return;
    }
  } else if (buildGraphMethod == BuildGraphUsing::Codelet) {
    auto argsMatchSignature = [&isFloatScalar](const Tensor_ &t0,
                                               const Tensor_ &t1) {
      return t0.poplar_type == poplar::HALF && isFloatScalar(t1);
    };
    if (!((inplace && argsMatchSignature(a, b)) ||
          (!inplace &&
           (argsMatchSignature(a, b) || argsMatchSignature(b, a))))) {
      BOOST_TEST(result.c.data.size() == 0);
      std::string expectedMsg = "Invalid operands of shape and type";
      BOOST_TEST(boost::algorithm::starts_with(result.errorMsg, expectedMsg));
      return;
    }
  }

  // Numerical test
  // --------------
  // Only cases without broadcasting are tested here.
  BOOST_REQUIRE(a.shape == b.shape || isAnyScalar(a, b));

  BOOST_TEST(result.errorMsg == "");

  std::vector<float> expectedData{};
  Type expectedType = a.poplar_type == b.poplar_type ? a.poplar_type : HALF;
  std::vector<std::size_t> expectedShape = isScalar(a) ? b.shape : a.shape;

  auto aData = a.getData();
  auto bData = b.getData();

  if (a.data.size() < b.data.size()) {
    std::swap(aData, bData);
  }

  if (bData.size() == 1) {
    for (unsigned i = 1; i < aData.size(); i++) {
      bData.push_back(bData[0]);
    }
  }

  for (unsigned i = 0; i < aData.size(); i++) {
    float value = aData[i] * bData[i];
    value = expectedType == HALF ? std::min(value, Tensor_::getFP16ClipValue())
                                 : value;
    expectedData.push_back(value);
  }

  BOOST_TEST(expectedData == result.c.data);
  BOOST_TEST(expectedType == result.c.poplar_type);
  BOOST_TEST(expectedShape == result.c.shape);
}

BOOST_FIXTURE_TEST_CASE(IntegrationTest, Fixture) {
  // TODO(T45374): Remove the short circuit below. In addition, cover the extra
  // cases inside verifyResult().
  if (inplace && a.shape == std::vector<std::size_t>{1} &&
      b.shape != std::vector<std::size_t>{1}) {
    return;
  }

  auto result = runTest(BuildGraphUsing::Map, a, b, dt, inplace);
  verifyResult(BuildGraphUsing::Map, a, b, inplace, result);
}

BOOST_FIXTURE_TEST_CASE(GraphBuilderTest, Fixture) {
  auto result = runTest(BuildGraphUsing::Codelet, a, b, dt, inplace);
  verifyResult(BuildGraphUsing::Codelet, a, b, inplace, result);
}
