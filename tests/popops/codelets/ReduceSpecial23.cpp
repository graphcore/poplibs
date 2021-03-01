// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ReduceSpecial23
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
// codelets
#include "../../lib/popops/reduction/ReductionVertex.hpp"
#include "poplar/Target.hpp"
#include "poplibs_test/Check.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include <boost/program_options.hpp>
#include <poplibs_test/Reduce.hpp>
#include <popops/Reduce.hpp>
#include <stdexcept>
#include <string.h>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplibs_test::reduce;
using namespace poplibs_support;

const OptionFlags options;

static bool doTest(const DeviceType &deviceType, const Type &partialsType,
                   const Type &outType, const unsigned outerDim,
                   const unsigned innerDim, const unsigned outputDim,
                   const popops::Operation op, const float scale, bool isUpdate,
                   ReductionSpecialisation specialisation) {
  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // Using negative initial value is important for log-add operations and using
  // integers (in float type variables) is helpful for exact comparison
  // for other operations
  const float initialValue = -1.0;
  const auto partialsGrainSize = partialsType == poplar::HALF ? 4 : 2;

  // Check constraints:
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    if (outputDim != 1) {
      std::cerr
          << "Output dimension must be 1 for SCALAR_OUTPUT_SINGLE_INPUT\n";
      return false;
    }

  } else if (specialisation == ReductionSpecialisation::STRIDED_REDUCE) {
    if (innerDim % partialsGrainSize) {
      std::cerr << "Inner dimension must be a multiple of 4 (half) or 2 "
                   "(float) for STRIDED_REDUCE\n";
      return false;
    }
    if (outputDim % partialsGrainSize) {
      std::cerr << "Output dimension must be a multiple of 4 (partials type = "
                   "half) or 2 "
                   "(partials type = float) for STRIDED_REDUCE\n";
      return false;
    }
  } else {
    std::cerr << "Unsupported specialisation\n";
    return false;
  }

  // Claim enough space for floats
  std::vector<char> data(innerDim * outerDim * 4);
  std::vector<float> nums(innerDim * outerDim);
  std::vector<int> intData(innerDim * outerDim);

  // Using negative input data is important for log-add operations and using
  // integers (in float type variables) is helpful for exact comparison
  // for other operations
  for (unsigned i = 0; i < outerDim; ++i) {
    for (unsigned j = 0; j < innerDim; ++j) {
      nums[(i * innerDim) + j] = -1.0 * (i + j);
      intData[(i * innerDim + j)] = i + j;
    }
  }

  copy(target, nums.data(), innerDim * outerDim, partialsType, data.data());
  std::vector<float> answers(outputDim, initialValue);
  std::vector<char> ans_data((outputDim)*4);
  copy(target, answers.data(), outputDim, outType, ans_data.data());

  Sequence prog;

  auto cs = graph.addComputeSet("cs");

  Tensor partials;
  partials = graph.addVariable(partialsType, {outerDim, innerDim});
  Tensor out;
  out = graph.addVariable(outType, {outputDim});

  bool useScale = (op == popops::Operation::LOG_ADD && scale != 0.0f) ||
                  (op != popops::Operation::LOG_ADD && scale != 1.0f);
  const auto vertexClass =
      templateVertex(useScale ? "popops::ScaledReduce" : "popops::Reduce",
                     "popops::" + getReductionVertexOpName(op), partialsType,
                     outType, isUpdate, specialisation);

  auto v1 = graph.addVertex(cs, vertexClass);

  graph.connect(v1["partials"], partials.flatten());
  graph.connect(v1["out"], out);
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    graph.setInitialValue(v1["numPartials"], innerDim * outerDim);
  } else {
    graph.setInitialValue(v1["numOutputs"], outputDim);
    graph.setInitialValue(v1["numPartialsM1"], outerDim - 1);
    graph.setInitialValue(v1["partialsWidth"], innerDim);
  }
  auto scaleTensor = graph.addVariable(FLOAT, {});
  graph.setTileMapping(scaleTensor, 0);
  graph.setInitialValue(scaleTensor, scale);
  if (useScale) {
    graph.connect(v1["k"], scaleTensor.reshape({1}));
  }
  graph.setTileMapping(v1, 0);
  graph.setTileMapping(partials, 0);
  graph.setTileMapping(out, 0);

  graph.createHostWrite("partials", partials);
  graph.createHostWrite("outw", out);
  graph.createHostRead("out", out);

  prog.add(Execute(cs));

  Engine e(graph, prog, options);
  auto outSize = out.numElements() * target.getTypeSize(outType);

  device.bind([&](const Device &d) {
    e.load(d);
    if (outType == FLOAT || outType == HALF) {
      e.writeTensor("partials", data.data(),
                    data.data() + partials.numElements() *
                                      target.getTypeSize(partialsType));
      e.writeTensor("outw", ans_data.data(), ans_data.data() + outSize);
    } else if (outType == INT) {
      e.writeTensor("partials", intData.data(),
                    intData.data() + partials.numElements() *
                                         target.getTypeSize(partialsType));
      e.writeTensor("outw", ans_data.data(), ans_data.data() + outSize);
    }
    e.readTensor("out", ans_data.data(), ans_data.data() + outSize);

    e.run();

    e.readTensor("out", ans_data.data(), ans_data.data() + outSize);
  });

  copy(target, partialsType, data.data(), nums.data(), innerDim * outerDim);

  copy(target, outType, ans_data.data(), answers.data(), outputDim);
  copy(target, outType, ans_data.data(), intData.data(), outputDim);

  // Create / reduce the whole result, later we may only check part of it based
  // on outputDim.
  const auto arrayDim =
      (specialisation == ReductionSpecialisation::STRIDED_REDUCE) ? innerDim
                                                                  : outputDim;
  MultiArray<float> input{(outerDim * innerDim) / arrayDim, arrayDim};
  for (unsigned i = 0; i < input.numElements(); i++) {
    const unsigned row = i / (innerDim);
    const unsigned column = i % (innerDim);
    input.data()[i] = nums[row * innerDim + column];
  }

  auto result = reduce(input, {0}, op);

  std::vector<float> correct_answer(outputDim, initialValue);
  if (op == popops::Operation::LOG_ADD) {
    for (unsigned i = 0; i < outputDim; i++) {

      const auto scaledResult = log::mul<float>(result[i], scale);
      if (isUpdate) {
        correct_answer[i] = log::add(scaledResult, correct_answer[i]);
      } else {
        correct_answer[i] = scaledResult;
      }
    }
  } else {
    for (unsigned i = 0; i < outputDim; i++) {
      if (isUpdate) {
        correct_answer[i] += result[i] * scale;
      } else {
        correct_answer[i] = result[i] * scale;
      }
    }
  }

  bool success = true;
  if (outType == FLOAT || outType == HALF) {
    // When using half data the log-add result is slightly inaccurate
    // Other operations should be exact with integer value float/half inputs
    const double tolerance = (op == popops::Operation::LOG_ADD &&
                              (outType == HALF || partialsType == HALF))
                                 ? 0.001
                                 : 0.0;
    for (unsigned i = 0; i < outputDim; ++i) {
      success = checkIsClose(correct_answer[i], answers[i], tolerance);
      answers[i] = 0; // zero for next iteration
    }
    if (!success) {
      std::cerr << "Errors in result\n";
    }
  } else if (outType == INT) {
    for (unsigned i = 0; i < outputDim; ++i) {
      CHECK_IF(success, correct_answer[i] == intData[i]);
    }
  } else {
    success = false;
  }
  return success;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  popops::Operation op = popops::Operation::ADD;
  DeviceType deviceType;
  Type partialsType;
  Type outType;
  float scale = 2.0f;
  bool isUpdate = true;
  unsigned specialisation;
  unsigned outerDim, innerDim, outputDim;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("partials-type",
     po::value<Type>(&partialsType)->required(),
     "Partials Type")
    ("specialisation",
     po::value<unsigned>(&specialisation)->required(),
     "Specialisation: 2 = SCALAR_OUTPUT_SINGLE_INPUT or"
     " 3 = STRIDED_REDUCE")
    ("out-type",
     po::value<Type>(&outType)->required(),
     "Output type")
    ("operation",
     po::value(&op),
     "operation:ADD SQUARE_ADD LOG_ADD MAX MIN MUL LOGICAL_OR or LOGICAL_AND")
    ("update",
     po::value<bool>(&isUpdate)->default_value(isUpdate),
     "reduce with update")
    ("scale",
     po::value<float>(&scale)->default_value(scale),
     "scale")
    ("outer-dim",
     po::value<unsigned>(&outerDim)->required(),
     "Outer dimension")
    ("inner-dim",
     po::value<unsigned>(&innerDim)->required(),
     "Inner dimension")
    ("output-dim",
     po::value<unsigned>(&outputDim)->required(),
     "Output dimension");
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

  const auto specialisationType =
      specialisation == 2 ? ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT
                          : ReductionSpecialisation::STRIDED_REDUCE;
  if (!doTest(deviceType, partialsType, outType, outerDim, innerDim, outputDim,
              op, scale, isUpdate, specialisationType))
    return 1;
  return 0;
}
