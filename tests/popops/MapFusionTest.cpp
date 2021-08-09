// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MapFusionTest
#include "poplar/Engine.hpp"
#include "poplar/IPUModel.hpp"
#include "poplibs_test/Util.hpp"
#include "poputil/exceptions.hpp"
#include <poplibs_support/TestDevice.hpp>
#include <pva/pva.hpp>

// codelets
#include "popnn/codelets.hpp"
#include "popops/codelets.hpp"

#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <cassert>
#include <limits>
#include <random>

#include <popops/ElementWise.hpp>
using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplibs_support;
namespace pe = popops::expr;

namespace {

static DeviceType deviceType;

static bool isDenormalOrZero(float a) {
  if (std::fabs(a) < std::numeric_limits<float>::min() || a == 0.0f)
    return true;
  return false;
}

// To get around the non-constexpr-ifs
template <typename T> struct abs_helper {
  static bool abs(T t) {
    throw poputil::poplibs_error(
        "Abs helper called on non floating point type");
  }
};

template <> struct abs_helper<float> {
  static bool abs(float t) { return fabs(t) < 0.000001f; }
};

template <> struct abs_helper<double> {
  static bool abs(double t) { return fabs(t) < 0.000001; }
};

template <int Size, typename InType = float, typename OutType = InType,
          typename InTypeArg3 = InType>
static bool mapTest(const pe::Expr &expr, bool inPlace = true,
                    bool checkReport = true) {
  auto device = createTestDevice(deviceType, 1, 4);
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

  OptionFlags generatedOptions{{"forceGenerateCodelet", "true"}};

  // Quickly dry run the example without host writes to test that it has
  // actually fused the operation into one codelet. We still have to read so the
  // codelet is connected to something
  if (checkReport && isIpuModel(deviceType)) {
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
       popops::map::%%%%                  (4 instances)
    */

    const auto report = engine.getReport(false);
    // Check the vertices.
    if (report.compilation().graph().numVertices() != 4) {
      std::cerr << "Number of vertices in the graph != 4 (is operation "
                   "fused?): num vertices = "
                << report.compilation().graph().numVertices();
      return false;
    }

    // Check the compute sets.
    if (report.compilation().graph().numComputeSets() != 1) {
      std::cerr << "Compute sets != 1 (is operation fused?) compute sets = "
                << report.compilation().graph().numComputeSets();
      return false;
    }

    // Check the name of the vertex, first by checking that there is only one
    // vertex type.
    std::string name;
    for (const auto &t : report.compilation().tiles()) {
      for (const auto &v : t.memory().vertices()) {
        if (name.empty()) {
          name = v.type().name();
        } else {
          if (name != v.type().name()) {
            std::cerr << "Too many vertex types in graph (is operation fused?)";
            return false;
          }
        }
      }
    }

    // Then check that the name is the generated name.
    constexpr char prefix[] = "popops::map::";
    if (std::string::npos == name.find(prefix)) {
      std::cerr << "Name doesn't match " << prefix
                << "(is operation fused?) name=" << name << "\n";
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
        std::memcpy(&hostIn1[i], &tmp, sizeof(InType));
        tmp = randDist(randomEngine);
        std::memcpy(&hostIn2[i], &tmp, sizeof(InType));
        tmp = randDist(randomEngine);
      }

      // If the last argument should be a bool then generate in range 0, 1 else
      // generate in range MIN, MAX
      if (std::is_same<bool, InTypeArg3>::value) {
        hostIn3[i] = randDistBool(randomEngine);
      } else {
        std::memcpy(&hostIn3[i], &tmp, sizeof(InType));
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
      match = abs_helper<OutType>::abs(genOutHost[i] - origOutHost[i]);
      if (std::isnan(genOutHost[i]) && std::isnan(origOutHost[i])) {
        match = true;
      }

      if (isDenormalOrZero(genOutHost[i]) && isDenormalOrZero(origOutHost[i])) {
        match = true;
      }
    }

    if (!match) {
      std::cerr << "Values at index " << i << " don't match: " << genOutHost[i]
                << " " << origOutHost[i] << "\n";
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

#define CHECK(test)                                                            \
  do {                                                                         \
    if (!test) {                                                               \
      std::cerr << "test \"" #test "\" failed." << std::endl;                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  std::string test;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("test",
     po::value<std::string>(&test)->required(),
     "The test to run");
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  //
  // Unary operations
  //
  if (test == "Abs") {
    CHECK((mapTest<10, float>(pe::Abs(pe::_1))));
    CHECK((mapTest<10, int>(pe::Abs(pe::_1))));
    // require 12 to avoid rearrangements for host reads
    CHECK((mapTest<12, long long>(pe::Abs(pe::_1))));
  } else if (test == "Neg") {
    CHECK((mapTest<10, float>(pe::Neg(pe::_1))));
    CHECK((mapTest<10, int>(pe::Neg(pe::_1))));
    CHECK((mapTest<12, long long>(pe::Neg(pe::_1))));
  } else if (test == "Signum") {
    CHECK((mapTest<10, int>(pe::Signum(pe::_1))));
    CHECK((mapTest<10, float>(pe::Signum(pe::_1))));
  } else if (test == "Square") {
    CHECK((mapTest<10, float>(pe::Square(pe::_1))));
    CHECK((mapTest<10, int>(pe::Square(pe::_1))));
    CHECK((mapTest<10, unsigned>(pe::Square(pe::_1))));
  } else if (test == "BitwiseNot") {
    CHECK((mapTest<10, int>(pe::BitwiseNot(pe::_1))));
    CHECK((mapTest<10, unsigned>(pe::BitwiseNot(pe::_1))));
    CHECK((mapTest<12, unsigned long long>(pe::BitwiseNot(pe::_1))));
  }

  // HALF and FLOAT
  else if (test == "IsFinite") {
    // Is finite can generate memcopies as it uses booleans so we skip the
    // report stage of this.
    CHECK((mapTest<10, float, bool>(pe::IsFinite(pe::Cast(pe::_1, HALF)), false,
                                    false)));
    CHECK((mapTest<10, float, bool>(pe::IsFinite(pe::_1), false, false)));
  }

  // FLOAT only
  else if (test == "Floor") {
    CHECK((mapTest<10, float>(pe::Floor(pe::_1))));
  } else if (test == "Cbrt") {
    CHECK((mapTest<10, float>(pe::Cbrt(pe::_1))));
  } else if (test == "Ceil") {
    CHECK((mapTest<10, float>(pe::Ceil(pe::_1))));
  } else if (test == "Round") {
    CHECK((mapTest<10, float>(pe::Round(pe::_1))));
  } else if (test == "Cos") {
    CHECK((mapTest<10, float>(pe::Cos(pe::_1))));
  } else if (test == "Erf") {
    CHECK((mapTest<10, float>(pe::Erf(pe::_1))));
  } else if (test == "Exp") {
    CHECK((mapTest<10, float>(pe::Exp(pe::_1))));
  } else if (test == "Expm1") {
    CHECK((mapTest<10, float>(pe::Expm1(pe::_1))));
  } else if (test == "Inv") {
    CHECK((mapTest<10, float>(pe::Inv(pe::_1))));
  } else if (test == "Log") {
    CHECK((mapTest<10, float>(pe::Log(pe::_1))));
  } else if (test == "Log1p") {
    CHECK((mapTest<10, float>(pe::Log1p(pe::_1))));
  } else if (test == "Sqrt") {
    CHECK((mapTest<10, float>(pe::Sqrt(pe::_1))));
  } else if (test == "Rsqrt") {
    CHECK((mapTest<10, float>(pe::Rsqrt(pe::_1))));
  } else if (test == "Sigmoid") {
    CHECK((mapTest<10, float>(pe::Sigmoid(pe::_1))));
  } else if (test == "Sin") {
    CHECK((mapTest<10, float>(pe::Sin(pe::_1))));
  } else if (test == "Tanh") {
    CHECK((mapTest<10, float>(pe::Tanh(pe::_1))));
  } else if (test == "Not") {
    // Not can generate memcopies as it uses booleans so we skip the report
    // stage of this.
    CHECK((mapTest<10, bool>(pe::Not(pe::_1), true, false)));
  }

  //
  // Binary Operations.
  // For bitwise operations and unsigned shift, long long types are not tested
  // as codegen is exactly the same.
  else if (test == "Add") {
    CHECK((mapTest<10, float>(pe::Add(pe::_1, pe::_2))));
    CHECK((mapTest<10, int>(pe::Add(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Add(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Add(pe::_1, pe::_2))));
    CHECK((mapTest<12, long long>(pe::Add(pe::_1, pe::_2))));
  } else if (test == "Divide") {
    CHECK((mapTest<10, float>(pe::Divide(pe::_1, pe::_2))));
    CHECK((mapTest<10, int>(pe::Divide(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Divide(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Divide(pe::_1, pe::_2))));
    CHECK((mapTest<12, long long>(pe::Divide(pe::_1, pe::_2))));
  }

  // All boolean operations can generate memcopies so will mess up the report
  // reading stage of the test, hence why we skip it.
  else if (test == "Equal") {
    CHECK((mapTest<10, float, bool>(pe::Equal(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, int, bool>(pe::Equal(pe::_1, pe::_2), false, false)));
    CHECK(
        (mapTest<10, unsigned, bool>(pe::Equal(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<12, unsigned long long, bool>(pe::Equal(pe::_1, pe::_2),
                                                 false, false)));
    CHECK((
        mapTest<12, long long, bool>(pe::Equal(pe::_1, pe::_2), false, false)));
  } else if (test == "Gte") {
    CHECK((mapTest<10, float, bool>(pe::Gte(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, int, bool>(pe::Gte(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, unsigned, bool>(pe::Gte(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<12, unsigned long long, bool>(pe::Gte(pe::_1, pe::_2), false,
                                                 false)));
    CHECK((mapTest<12, unsigned long long, bool>(pe::Gte(pe::_1, pe::_2), false,
                                                 false)));
    CHECK(
        (mapTest<12, long long, bool>(pe::Gte(pe::_1, pe::_2), false, false)));
  } else if (test == "Gt") {
    CHECK((mapTest<10, float, bool>(pe::Gt(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, int, bool>(pe::Gt(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, unsigned, bool>(pe::Gt(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<12, unsigned long long, bool>(pe::Gt(pe::_1, pe::_2), false,
                                                 false)));
    CHECK((mapTest<12, long long, bool>(pe::Gt(pe::_1, pe::_2), false, false)));
  } else if (test == "Lte") {
    CHECK((mapTest<10, float, bool>(pe::Lte(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, int, bool>(pe::Lte(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, unsigned, bool>(pe::Lte(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<12, unsigned long long, bool>(pe::Lte(pe::_1, pe::_2), false,
                                                 false)));
    CHECK(
        (mapTest<12, long long, bool>(pe::Lte(pe::_1, pe::_2), false, false)));
  } else if (test == "NotEqual") {
    CHECK(
        (mapTest<10, float, bool>(pe::NotEqual(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, int, bool>(pe::NotEqual(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, unsigned, bool>(pe::NotEqual(pe::_1, pe::_2), false,
                                       false)));
    CHECK((mapTest<12, unsigned long long, bool>(pe::NotEqual(pe::_1, pe::_2),
                                                 false, false)));
    CHECK((mapTest<12, long long, bool>(pe::NotEqual(pe::_1, pe::_2), false,
                                        false)));
  } else if (test == "Lt") {
    CHECK((mapTest<10, float, bool>(pe::Lt(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, int, bool>(pe::Lt(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<10, unsigned, bool>(pe::Lt(pe::_1, pe::_2), false, false)));
    CHECK((mapTest<12, unsigned long long, bool>(pe::Lt(pe::_1, pe::_2), false,
                                                 false)));
    CHECK((mapTest<12, long long, bool>(pe::Lt(pe::_1, pe::_2), false, false)));
  } else if (test == "Sub") {
    CHECK((mapTest<10, float>(pe::Sub(pe::_1, pe::_2))));
    CHECK((mapTest<10, int>(pe::Sub(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Sub(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Sub(pe::_1, pe::_2))));
    CHECK((mapTest<12, long long>(pe::Sub(pe::_1, pe::_2))));
  } else if (test == "Max") {
    CHECK((mapTest<10, float>(pe::Max(pe::_1, pe::_2))));
    CHECK((mapTest<10, int>(pe::Max(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Max(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Max(pe::_1, pe::_2))));
    CHECK((mapTest<12, long long>(pe::Max(pe::_1, pe::_2))));
  } else if (test == "Min") {
    CHECK((mapTest<10, float>(pe::Min(pe::_1, pe::_2))));
    CHECK((mapTest<10, int>(pe::Min(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Min(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Max(pe::_1, pe::_2))));
    CHECK((mapTest<12, long long>(pe::Max(pe::_1, pe::_2))));
  } else if (test == "Mul") {
    CHECK((mapTest<10, float>(pe::Mul(pe::_1, pe::_2))));
    CHECK((mapTest<10, int>(pe::Mul(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Mul(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Mul(pe::_1, pe::_2))));
    CHECK((mapTest<12, long long>(pe::Mul(pe::_1, pe::_2))));
  } else if (test == "BitwiseAnd") {
    CHECK((mapTest<10, int>(pe::BitwiseAnd(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::BitwiseAnd(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::BitwiseAnd(pe::_1, pe::_2))));
  } else if (test == "BitwiseOr") {
    CHECK((mapTest<10, int>(pe::BitwiseOr(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::BitwiseOr(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::BitwiseOr(pe::_1, pe::_2))));
  } else if (test == "BitwiseXor") {
    CHECK((mapTest<10, int>(pe::BitwiseXor(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::BitwiseXor(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::BitwiseXor(pe::_1, pe::_2))));
  } else if (test == "BitwiseXnor") {
    CHECK((mapTest<10, int>(pe::BitwiseXnor(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::BitwiseXnor(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::BitwiseXnor(pe::_1, pe::_2))));
  } else if (test == "Rem") {
    CHECK((mapTest<10, int>(pe::Rem(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Rem(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Rem(pe::_1, pe::_2))));
  } else if (test == "Shl") {
    CHECK((mapTest<10, int>(pe::Shl(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Shl(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Shl(pe::_1, pe::_2))));
  } else if (test == "Shr") {
    CHECK((mapTest<10, int>(pe::Shr(pe::_1, pe::_2))));
    CHECK((mapTest<10, unsigned>(pe::Shr(pe::_1, pe::_2))));
    CHECK((mapTest<12, unsigned long long>(pe::Shr(pe::_1, pe::_2))));
  } else if (test == "ShrSE") {
    CHECK((mapTest<10, int>(pe::ShrSE(pe::_1, pe::_2))));
    CHECK((mapTest<12, long long>(pe::ShrSE(pe::_1, pe::_2))));
  }

  // Float only
  else if (test == "Atan2") {
    CHECK((mapTest<10, float>(pe::Atan2(pe::_1, pe::_2))));
  } else if (test == "Pow") {
    CHECK((mapTest<10, float>(pe::Pow(pe::_1, pe::_2))));
  }

  // And and OR can generate memcopies so will mess up the report reading stage
  // of the test.
  else if (test == "And") {
    CHECK((mapTest<10, bool>(pe::And(pe::_1, pe::_2), true, false)));
  } else if (test == "Or") {
    CHECK((mapTest<10, bool>(pe::Or(pe::_1, pe::_2), true, false)));
  }

  //
  // Ternary
  //
  else if (test == "Select") {
    CHECK(
        (mapTest<10, float, float, bool>(pe::Select(pe::_1, pe::_2, pe::_3))));
  } else if (test == "Clamp") {
    CHECK((mapTest<10, float>(pe::Clamp(pe::_1, pe::_2, pe::_3))));
  }

  else if (test == "Fusion") {
    CHECK((mapTest<10, float>(
        pe::Square(pe::Divide(
            pe::Log(pe::Pow(pe::Mul(pe::Sub(pe::Add(pe::_1, pe::Const(5.0f)),
                                            pe::Const(3.0f)),
                                    pe::_2),
                            pe::Const(2.0f))),
            pe::_2)),
        true, true)));
    CHECK((mapTest<12, unsigned long long>(
        pe::BitwiseAnd(
            pe::Divide(pe::BitwiseNot(pe::Add(
                           pe::Mul(pe::Sub(pe::Add(pe::_1, pe::Const(5)),
                                           pe::Const(3)),
                                   pe::_2),
                           pe::Const(2))),
                       pe::Const(4)),
            pe::_2),
        true, true)));
  } else if (test == "MissingPlaceholder") {
    // Add an unused int argument.
    CHECK((mapTest<10, float, float, int>(pe::Add(pe::_1, pe::_2))));
  } else {
    std::cerr << "Unknown test: " << test << std::endl;
    return 1;
  }

  std::cerr << "Test passed." << std::endl;
  return 0;
}
