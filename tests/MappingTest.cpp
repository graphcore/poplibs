#define BOOST_TEST_MODULE MappingTest
#include "TestDevice.hpp"
#include "poplar/Engine.hpp"
#include "poplar/IPUModel.hpp"
#include "poplibs_test/Util.hpp"
#include <boost/test/unit_test.hpp>

// codelets
#include "popnn/codelets.hpp"
#include "popops/codelets.hpp"

#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/random.hpp>
#include <cassert>
#include <limits>
#include <random>

#include <popops/ElementWise.hpp>
using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
namespace pe = popops::expr;

namespace {

static bool isDenormalOrZero(float a) {
  if (std::fabs(a) < std::numeric_limits<float>::min() || a == 0.0f)
    return true;
  return false;
}

// To get around the non-constexpr-ifs
template <typename T> struct abs_helper {
  static bool abs(T t) { return t; }
};

template <> struct abs_helper<float> {
  static bool abs(float t) { return fabs(t); }
};

template <> struct abs_helper<double> {
  static bool abs(double t) { return fabs(t); }
};

template <int Size, typename InType = float, typename OutType = InType,
          typename InTypeArg3 = InType>
static bool mapTest(const pe::Expr &expr, bool inPlace = true,
                    bool checkReport = true) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  auto target = device.getTarget();
  poplar::Graph graph(target);
  popops::addCodelets(graph);
  popnn::addCodelets(graph);

  poplar::Type inType = poplar::equivalent_device_type<InType>{}.value;

  poplar::Type inTypeArg3 = poplar::equivalent_device_type<InTypeArg3>{}.value;

  poplar::Tensor in1 =
      graph.addVariable(inType, {Size}, VariableMappingMethod::LINEAR, "in1");

  poplar::Tensor in1_gen = graph.addVariable(
      inType, {Size}, VariableMappingMethod::LINEAR, "in1_gen");

  poplar::Tensor in2 =
      graph.addVariable(inType, {Size}, VariableMappingMethod::LINEAR, "in2");

  poplar::Tensor in3 = graph.addVariable(inTypeArg3, {Size},
                                         VariableMappingMethod::LINEAR, "in3");

  graph.createHostRead("in1", in1);
  graph.createHostRead("in1_gen", in1_gen);

  OptionFlags generatedOptions{{"forceGenerateCodelet", "false"}};

  // Quickly dry run the example without host writes to test that it has
  // actually fused the operation into one codelet. We still have to read so the
  // codelet is connected to something
  if (checkReport && TEST_TARGET == DeviceType::IpuModel) {
    OutType tmp[Size];
    Sequence prog;
    if (inPlace) {
      popops::mapInPlace(graph, expr, {in1_gen, in2, in3}, prog, "",
                         generatedOptions);
    } else {
      poplar::Tensor testRead =
          popops::map(graph, expr, {in1, in2, in3}, prog, "", generatedOptions);
      graph.createHostRead("testRead", testRead);
    }

    Engine engine(graph, prog);

    // We expect the following to be in the graph:
    /*
      Graph
       Number of vertices:                4
       Number of compute sets:            1

       Cycles by vertex type:
         MapGeneratedVertex_%%%%          (4 instances)
    */

    ProfileValue profile = engine.getProfile();
    engine.printProfileSummary(std::cout);
    // Check the vertices.
    if (profile["graphProfile"]["graph"]["numVertices"].asInt() != 4) {
      BOOST_TEST_MESSAGE(
          "Number of vertices in the graph != 4 (is operation "
          "fused?): num vertices = "
          << profile["graphProfile"]["graph"]["numVertices"].asInt());
      return false;
    }

    // Check the compute sets.
    if (profile["graphProfile"]["graph"]["numComputeSets"].asInt() != 1) {
      BOOST_TEST_MESSAGE(
          "Compute sets != 1 (is operation fused?) compute sets = "
          << profile["graphProfile"]["graph"]["numComputeSets"].asInt());
      return false;
    }

    // Check the name of the vertex, first by checking that there is only one
    // vertex type.
    auto vertexTypes = profile["graphProfile"]["vertexTypes"]["names"];
    if (vertexTypes.size() != 1) {
      BOOST_TEST_MESSAGE(
          "Too many vertex types in graph (is operation fused?)");
      return false;
    }
    // Then check that the name is the generated name.
    std::string name = vertexTypes.asVector()[0].asString();
    if (std::string::npos == name.find("MapGeneratedVertex_")) {
      BOOST_TEST_MESSAGE("Name doesn't match MapGeneratedVertex_ (is operation "
                         "fused?) name="
                         << name);
      return false;
    }
  }

  graph.createHostWrite("in1", in1);
  graph.createHostWrite("in1_gen", in1_gen);

  graph.createHostWrite("in2", in2);
  graph.createHostWrite("in3", in3);

  InType hostIn1[Size];
  InType hostIn2[Size];
  InTypeArg3 hostIn3[Size];

  std::mt19937 randomEngine;

  boost::random::uniform_int_distribution<unsigned> randDist(
      std::numeric_limits<unsigned>::lowest(),
      std::numeric_limits<unsigned>::max());

  boost::random::uniform_real_distribution<float> randDistFloat(0.0f, 1.0f);

  boost::random::uniform_int_distribution<unsigned> randDistBool(0, 1);

  for (int i = 0; i < Size; ++i) {

    if (std::is_floating_point<InType>::value) {
      hostIn1[i] = randDistFloat(randomEngine);
      hostIn2[i] = randDistFloat(randomEngine);
      hostIn3[i] = randDistFloat(randomEngine);

    } else {

      unsigned tmp = randDist(randomEngine);
      if (std::is_same<bool, InType>::value) {
        hostIn1[i] = randDistBool(randomEngine);
        hostIn2[i] = randDistBool(randomEngine);
      } else {
        hostIn1[i] = *reinterpret_cast<InType *>(&tmp);
        tmp = randDist(randomEngine);
        hostIn2[i] = *reinterpret_cast<InType *>(&tmp);
        tmp = randDist(randomEngine);
      }

      // If the last argument should be a bool then generate in range 0, 1 else
      // generate in range MIN, MAX
      if (std::is_same<bool, InTypeArg3>::value) {
        hostIn3[i] = randDistBool(randomEngine);
      } else {
        hostIn3[i] = *reinterpret_cast<InType *>(&tmp);
      }
    }
  }

  // If we are dealing with a clamp we can't use completely random numbers as
  // in2 <= in3.
  if (expr.isA<pe::Clamp>()) {
    for (int i = 0; i < Size; ++i) {
      if (hostIn2[i] > hostIn3[i]) {

        InTypeArg3 tmp = hostIn3[i];
        hostIn3[i] = hostIn2[i];
        hostIn2[i] = tmp;
      }
    }
  }
  Sequence prog;

  poplar::Tensor generatedOut;
  poplar::Tensor originalOut;

  OptionFlags originalOptions{{"enableGenerateCodelet", "false"}};

  if (inPlace) {
    popops::mapInPlace(graph, expr, {in1_gen, in2, in3}, prog, "",
                       generatedOptions);
    popops::mapInPlace(graph, expr, {in1, in2, in3}, prog, "", originalOptions);
  } else {
    generatedOut =
        popops::map(graph, expr, {in1, in2, in3}, prog, "", generatedOptions);
    originalOut =
        popops::map(graph, expr, {in1, in2, in3}, prog, "", originalOptions);

    graph.createHostRead("originalOut", originalOut);
    graph.createHostRead("generatedOut", generatedOut);
  }

  OutType genOutHost[Size];
  OutType origOutHost[Size];

  Engine engine(graph, prog);
  device.bind([&](const Device &d) {
    engine.load(d);

    if (inPlace) {
      engine.writeTensor("in1_gen", hostIn1, &hostIn1[Size]);
    }
    engine.writeTensor("in1", hostIn1, &hostIn1[Size]);
    engine.writeTensor("in2", hostIn2, &hostIn2[Size]);
    engine.writeTensor("in3", hostIn3, &hostIn3[Size]);

    engine.run(0);

    if (!inPlace) {
      engine.readTensor("generatedOut", genOutHost, &genOutHost[Size]);
      engine.readTensor("originalOut", origOutHost, &origOutHost[Size]);
    } else {
      engine.readTensor("in1_gen", genOutHost, &genOutHost[Size]);
      engine.readTensor("in1", origOutHost, &origOutHost[Size]);
    }
  });

  bool matchesModel = true;

  for (int i = 0; i < Size; ++i) {

    // Report a missmatch at a specific index.
    bool match = genOutHost[i] == origOutHost[i];

    if (std::is_floating_point<OutType>::value) {

      match =
          abs_helper<OutType>::abs(genOutHost[i] - origOutHost[i]) < 0.000001f;
      if (std::isnan(genOutHost[i]) && std::isnan(origOutHost[i])) {
        match = true;
      }

      if (isDenormalOrZero(genOutHost[i]) && isDenormalOrZero(origOutHost[i])) {
        match = true;
      }
    }

    if (!match) {
      BOOST_TEST_MESSAGE("Values at index "
                         << i << " don't match: " << genOutHost[i] << " "
                         << origOutHost[i] << "\n");
    }

    // Fail the test.
    matchesModel &= match;
  }

  if (inPlace) {
    return mapTest<Size, InType, OutType, InTypeArg3>(expr, false, checkReport);
  }

  if (!matchesModel) {
    return false;
  }
  return true;
}

} // end anonymous namespace

//
// Unary operations
//
/* s
BOOST_AUTO_TEST_CASE(MappingAbs) {
  BOOST_CHECK((mapTest<10, float>(pe::Abs(pe::_1))));
  BOOST_CHECK((mapTest<10, int>(pe::Abs(pe::_1))));
}

BOOST_AUTO_TEST_CASE(MappingNeg) {
  BOOST_CHECK((mapTest<10, float>(pe::Neg(pe::_1))));
  BOOST_CHECK((mapTest<10, int>(pe::Neg(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingSignum) {
  BOOST_CHECK((mapTest<10, float>(pe::Signum(pe::_1))));
  BOOST_CHECK((mapTest<10, int>(pe::Signum(pe::_1))));
}

BOOST_AUTO_TEST_CASE(MappingSquare) {
  BOOST_CHECK((mapTest<10, float>(pe::Square(pe::_1))));
  BOOST_CHECK((mapTest<10, int>(pe::Square(pe::_1))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Square(pe::_1))));
}

BOOST_AUTO_TEST_CASE(MappingBitwiseNot) {
  BOOST_CHECK((mapTest<10, int>(pe::BitwiseNot(pe::_1))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::BitwiseNot(pe::_1))));
}

// FLOAT only
BOOST_AUTO_TEST_CASE(MappingFloor) {
  BOOST_CHECK((mapTest<10, float>(pe::Floor(pe::_1))));
}

BOOST_AUTO_TEST_CASE(MappingCeil) {
  BOOST_CHECK((mapTest<10, float>(pe::Ceil(pe::_1))));
}

// Is finite can generate memcopies as it uses booleans so we skip the report
stage of this. BOOST_AUTO_TEST_CASE(MappingIsFinite) { BOOST_CHECK((mapTest<10,
float, bool>(pe::IsFinite(pe::_1), false, false)));
}
BOOST_AUTO_TEST_CASE(MappingRound) {
  BOOST_CHECK((mapTest<10, float>(pe::Round(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingCos) {
  BOOST_CHECK((mapTest<10, float>(pe::Cos(pe::_1))));
}

BOOST_AUTO_TEST_CASE(MappingExp) {
  BOOST_CHECK((mapTest<10, float>(pe::Exp(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingExpm1) {
  BOOST_CHECK((mapTest<10, float>(pe::Expm1(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingInv) {
  BOOST_CHECK((mapTest<10, float>(pe::Inv(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingLog) {
  BOOST_CHECK((mapTest<10, float>(pe::Log(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingLog1p) {
  BOOST_CHECK((mapTest<10, float>(pe::Log1p(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingSqrt) {
  BOOST_CHECK((mapTest<10, float>(pe::Sqrt(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingRsqrt) {
  BOOST_CHECK((mapTest<10, float>(pe::Rsqrt(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingSigmoid) {
  BOOST_CHECK((mapTest<10, float>(pe::Sigmoid(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingSin) {
  BOOST_CHECK((mapTest<10, float>(pe::Sin(pe::_1))));
}
BOOST_AUTO_TEST_CASE(MappingTanh) {
  BOOST_CHECK((mapTest<10, float>(pe::Tanh(pe::_1))));
}

// Not can generate memcopies as it uses booleans so we skip the report stage of
// this.
BOOST_AUTO_TEST_CASE(MappingNot) {
  BOOST_CHECK((mapTest<10, bool>(pe::Not(pe::_1), true, false)));
}

//
// Binary Operations.
//

BOOST_AUTO_TEST_CASE(MappingAdd) {
  BOOST_CHECK((mapTest<10, float>(pe::Add(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, int>(pe::Add(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Add(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingDivide) {
  BOOST_CHECK((mapTest<10, float>(pe::Divide(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, int>(pe::Divide(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Divide(pe::_1, pe::_2))));
}

// All boolean operations can generate memcopies so will mess up the report
// reading stage of the test, hence why we skip it.
BOOST_AUTO_TEST_CASE(MappingEqual) {
  BOOST_CHECK(
      (mapTest<10, float, bool>(pe::Equal(pe::_1, pe::_2), false, false)));
  BOOST_CHECK(
      (mapTest<10, int, bool>(pe::Equal(pe::_1, pe::_2), false, false)));
  BOOST_CHECK(
      (mapTest<10, unsigned, bool>(pe::Equal(pe::_1, pe::_2), false, false)));
}
BOOST_AUTO_TEST_CASE(MappingGte) {
  BOOST_CHECK(
      (mapTest<10, float, bool>(pe::Gte(pe::_1, pe::_2), false, false)));
  BOOST_CHECK((mapTest<10, int, bool>(pe::Gte(pe::_1, pe::_2), false, false)));
  BOOST_CHECK(
      (mapTest<10, unsigned, bool>(pe::Gte(pe::_1, pe::_2), false, false)));
}
BOOST_AUTO_TEST_CASE(MappingGt) {
  BOOST_CHECK((mapTest<10, float, bool>(pe::Gt(pe::_1, pe::_2), false, false)));
  BOOST_CHECK((mapTest<10, int, bool>(pe::Gt(pe::_1, pe::_2), false, false)));
  BOOST_CHECK(
      (mapTest<10, unsigned, bool>(pe::Gt(pe::_1, pe::_2), false, false)));
}

BOOST_AUTO_TEST_CASE(MappingLte) {
  BOOST_CHECK(
      (mapTest<10, float, bool>(pe::Lte(pe::_1, pe::_2), false, false)));
  BOOST_CHECK((mapTest<10, int, bool>(pe::Lte(pe::_1, pe::_2), false, false)));
  BOOST_CHECK(
      (mapTest<10, unsigned, bool>(pe::Lte(pe::_1, pe::_2), false, false)));
}
BOOST_AUTO_TEST_CASE(MappingNotEqual) {
  BOOST_CHECK(
      (mapTest<10, float, bool>(pe::NotEqual(pe::_1, pe::_2), false, false)));
  BOOST_CHECK(
      (mapTest<10, int, bool>(pe::NotEqual(pe::_1, pe::_2), false, false)));
  BOOST_CHECK((
      mapTest<10, unsigned, bool>(pe::NotEqual(pe::_1, pe::_2), false, false)));
}
BOOST_AUTO_TEST_CASE(MappingLt) {
  BOOST_CHECK((mapTest<10, float, bool>(pe::Lt(pe::_1, pe::_2), false, false)));
  BOOST_CHECK((mapTest<10, int, bool>(pe::Lt(pe::_1, pe::_2), false, false)));
  BOOST_CHECK(
      (mapTest<10, unsigned, bool>(pe::Lt(pe::_1, pe::_2), false, false)));
}

BOOST_AUTO_TEST_CASE(MappingSub) {
  BOOST_CHECK((mapTest<10, float>(pe::Sub(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, int>(pe::Sub(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Sub(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingMax) {
  BOOST_CHECK((mapTest<10, float>(pe::Max(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, int>(pe::Max(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Max(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingMin) {
  BOOST_CHECK((mapTest<10, float>(pe::Min(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, int>(pe::Min(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Min(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingMul) {
  BOOST_CHECK((mapTest<10, float>(pe::Mul(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, int>(pe::Mul(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Mul(pe::_1, pe::_2))));
}

BOOST_AUTO_TEST_CASE(MappingBitwiseAnd) {
  BOOST_CHECK((mapTest<10, int>(pe::BitwiseAnd(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::BitwiseAnd(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingBitwiseOr) {
  BOOST_CHECK((mapTest<10, int>(pe::BitwiseOr(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::BitwiseOr(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingBitwiseXor) {
  BOOST_CHECK((mapTest<10, int>(pe::BitwiseXor(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::BitwiseXor(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingBitwiseXnor) {
  BOOST_CHECK((mapTest<10, int>(pe::BitwiseXnor(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::BitwiseXnor(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingRem) {
  BOOST_CHECK((mapTest<10, int>(pe::Rem(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Rem(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingShl) {
  BOOST_CHECK((mapTest<10, int>(pe::Shl(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Shl(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingShr) {
  BOOST_CHECK((mapTest<10, int>(pe::Shr(pe::_1, pe::_2))));
  BOOST_CHECK((mapTest<10, unsigned>(pe::Shr(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingShrSE) {
  BOOST_CHECK((mapTest<10, int>(pe::ShrSE(pe::_1, pe::_2))));
}

// Float only
BOOST_AUTO_TEST_CASE(MappingAtan2) {
  BOOST_CHECK((mapTest<10, float>(pe::Atan2(pe::_1, pe::_2))));
}
BOOST_AUTO_TEST_CASE(MappingPow) {
  BOOST_CHECK((mapTest<10, float>(pe::Pow(pe::_1, pe::_2))));
}


// And and OR can generate memcopies so will mess up the report reading stage of
the test. BOOST_AUTO_TEST_CASE(MappingAnd) { BOOST_CHECK((mapTest<10,
bool>(pe::And(pe::_1, pe::_2), true, false)));
}

BOOST_AUTO_TEST_CASE(MappingOr) {
  BOOST_CHECK((mapTest<10, bool>(pe::Or(pe::_1, pe::_2), true, false)));
}

//
// Ternary
//
BOOST_AUTO_TEST_CASE(MappingSelect) {
  BOOST_CHECK(
      (mapTest<10, float, float, bool>(pe::Select(pe::_1, pe::_2, pe::_3))));
}
BOOST_AUTO_TEST_CASE(MappingClamp) {
  BOOST_CHECK((mapTest<10, float>(pe::Clamp(pe::_1, pe::_2, pe::_3))));
}*/

BOOST_AUTO_TEST_CASE(MappingFusion) {
  BOOST_CHECK((mapTest<10, float>(
      pe::Square(pe::Divide(
          pe::Log(pe::Pow(pe::Mul(pe::Sub(pe::Add(pe::_1, pe::Const(5.0f)),
                                          pe::Const(3.0f)),
                                  pe::_2),
                          pe::Const(2.0f))),
          pe::_2)),
      true, true)));
}
