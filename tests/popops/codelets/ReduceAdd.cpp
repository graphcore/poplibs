// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ReduceAdd
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
// codelets
#include "poplar/Target.hpp"
#include "poplibs_test/Check.hpp"
#include "poplibs_test/Util.hpp"
#include "poplin/Convolution.hpp"
#include "poplin/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include <optional>
#include <poplibs_test/TempDir.hpp>
#include <stdexcept>
#include <string.h>

#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

static bool doTest(const DeviceType &deviceType, const Type &partialsType,
                   const Type &outType, unsigned outerDim, unsigned innerDim,
                   bool singleInput, bool constrainPartials, bool profile) {
  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);

  // Claim enough space for floats
  std::vector<char> data(innerDim * outerDim * 4);
  std::vector<float> nums(innerDim * outerDim);
  for (unsigned i = 0; i < innerDim * outerDim; ++i) {
    nums[i] = 1.0 * (i % outerDim);
  }
  copy(target, nums.data(), innerDim * outerDim, partialsType, data.data());
  std::vector<float> answers(outerDim + 1);
  std::vector<char> ans_data((outerDim + 1) * 4);
  for (unsigned i = 0; i < outerDim + 1; ++i) {
    answers[i] = 0.0;
  }
  memcpy(ans_data.data(), answers.data(), (outerDim + 1) * 4);

  Sequence prog;

  auto cs = graph.addComputeSet("cs");

  Tensor partials;
  partials = graph.addVariable(partialsType, {innerDim, outerDim});

  auto out = graph.addVariable(outType, {outerDim + 1});

  const auto vertexClass =
      templateVertex("poplin::ReduceAdd", outType, partialsType, singleInput,
                     constrainPartials);
  auto v1 = graph.addVertex(cs, vertexClass);

  if (singleInput) {
    graph.connect(v1["partials"], partials.slice(0, 1, 0).flatten());
    graph.connect(v1["initialPartial"],
                  partials.slice(innerDim == 1 ? 0 : 1, innerDim, 0).flatten());
    graph.setInitialValue(v1["numPartials"], innerDim - 1);
  } else {
    for (unsigned i = 0; i < innerDim; ++i) {
      Tensor Row = partials.slice(i, i + 1, 0);
      graph.connect(v1["partials"][i], Row.reshape({outerDim}));
    }
    graph.setFieldSize(v1["partials"], innerDim);
    graph.setInitialValue(v1["numPartials"], innerDim);
  }
  graph.connect(v1["out"], out.slice(0, outerDim));
  graph.setInitialValue(v1["numElems"], outerDim);

  graph.setTileMapping(v1, 0);
  graph.setTileMapping(partials, 0);
  graph.setTileMapping(out, 0);

  graph.createHostWrite("partials", partials);
  graph.createHostWrite("outw", out);
  graph.createHostRead("out", out);

  prog.add(Execute(cs));

  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (profile) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }
  Engine e(graph, prog, engineOptions);
  auto outSize = out.numElements() * target.getTypeSize(outType);

  device.bind([&](const Device &d) {
    e.load(d);
    e.writeTensor("partials", data.data(),
                  data.data() + partials.numElements() *
                                    target.getTypeSize(partialsType));
    e.writeTensor("outw", ans_data.data(), ans_data.data() + outSize);
    e.readTensor("out", ans_data.data(), ans_data.data() + outSize);

    e.run();

    e.readTensor("out", ans_data.data(), ans_data.data() + outSize);
  });

  copy(target, outType, ans_data.data(), answers.data(), outerDim + 1);

  if (profile) {
    e.printProfileSummary(std::cout,
                          OptionFlags{{"showExecutionSteps", "true"}});
  }
  bool success = true;
  for (unsigned i = 0; i < outerDim; ++i) {
    CHECK_IF(success, innerDim * 1.0 * i == answers[i]);
    answers[i] = 0; // zero for next iteration
  }
  CHECK_IF(success, answers[outerDim] == 0.0);
  return success;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type partialsType;
  Type outType;
  unsigned outerDim, innerDim;
  bool singleInput = false;
  bool constrainPartials = false;
  bool profile = false;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("profile",
     po::value<bool>(&profile)->default_value(profile),
     "Show a profile report")
    ("partials-type",
     po::value<Type>(&partialsType)->required(),
     "Partials Type")
    ("out-type",
     po::value<Type>(&outType)->required(),
     "Output Type")
    ("outer-dim",
     po::value<unsigned>(&outerDim)->required(),
     "Outer dimension")
    ("inner-dim",
     po::value<unsigned>(&innerDim)->required(),
     "Inner dimension")
    ("single-input",
     po::value<bool>(&singleInput)->default_value(singleInput),
     "Use single input region variant of the vertex")
    ("constrain-partials",
     po::value<bool>(&constrainPartials)->default_value(constrainPartials),
     "Use variant with constrained partials memory allocation")
    ;
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
  if (!singleInput && constrainPartials) {
    std::cerr << "Error, vertex without singleInput but with constrained"
                 " partials is not supported\n";
  }
  if (!doTest(deviceType, partialsType, outType, outerDim, innerDim,
              singleInput, constrainPartials, profile))
    return 1;
  return 0;
}
