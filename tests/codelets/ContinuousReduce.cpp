#include <poplar/Engine.hpp>
#include "TestDevice.hpp"
// codelets
#include "popops/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/Util.hpp"
#include "poplibs_test/Util.hpp"
#include "poplar/Target.hpp"
#include <string.h>
#include <stdexcept>

#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;
using namespace poplibs_test::util;

#define CHECK_IF(result, cond) \
  do { \
    if (!(cond)) { \
      std::cerr << "Condition failed: " << #cond << '\n'; \
      result = false; \
    } \
  } while(false)

static bool doTest(const DeviceType &deviceType,
                   const Type &partialsType,
                   const Type &outType,
                   unsigned outerDim,
                   unsigned innerDim) {
  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  const unsigned scale = 1.2;
  const unsigned initialValue = 3.4;

  // Claim enough space for floats
  std::vector<char> data(innerDim * outerDim * 4);
  std::vector<float> nums(innerDim * outerDim);
  for (unsigned i = 0; i < outerDim; ++i) {
    for (unsigned j = 0; j < innerDim; ++j) {
      nums[(i * innerDim) + j] = j;
    }
  }
  copy(target, nums.data(), innerDim*outerDim, partialsType, data.data());
  std::vector<float> answers(outerDim);
  std::vector<char> ans_data((outerDim) * 4);
  for (unsigned i = 0; i < outerDim; ++i) {
    answers[i] = initialValue;
  }
  copy(target, answers.data(), outerDim, outType, ans_data.data());

  Sequence prog;

  auto cs = graph.addComputeSet("cs");

  Tensor partials;
  partials = graph.addVariable(partialsType, {outerDim, innerDim});
  Tensor out;
  out = graph.addVariable(outType, {outerDim});

  const auto vertexClass = templateVertex("popops::ScaledContinuousReduce",
                                          "popops::ReduceSquareAdd",
                                          partialsType, outType, true);

  auto v1 = graph.addVertex(cs, vertexClass);

  graph.connect(v1["partials"], partials.flatten());
  graph.connect(v1["out"], out);

  graph.setInitialValue(v1["numOutputs"], outerDim);
  graph.setInitialValue(v1["numPartials"], innerDim);

  auto scaleTensor = graph.addVariable(FLOAT, {});
  graph.setTileMapping(scaleTensor, 0);
  graph.setInitialValue(scaleTensor, scale);
  graph.connect(v1["k"], scaleTensor.reshape({1}));

  graph.setTileMapping(v1, 0);
  graph.setTileMapping(partials, 0);
  graph.setTileMapping(out, 0);

  graph.createHostWrite("partials", partials);
  graph.createHostWrite("outw", out);
  graph.createHostRead("out", out);

  prog.add(Execute(cs));

  Engine e(graph, prog);
  auto outSize = out.numElements() * target.getTypeSize(outType);

  device.bind([&](const Device &d) {
    e.load(d);
    e.writeTensor("partials", data.data(), data.data() +
                  partials.numElements() * target.getTypeSize(partialsType));
    e.writeTensor("outw", ans_data.data(), ans_data.data() + outSize);
    e.readTensor("out", ans_data.data(), ans_data.data() + outSize);

    e.run();

    e.readTensor("out", ans_data.data(), ans_data.data() + outSize);
  });

  copy(target, partialsType, data.data(), nums.data(), innerDim * outerDim);

  copy(target, outType, ans_data.data(), answers.data(), outerDim);

  bool success = true;

  float correct_answer = initialValue;
  for (unsigned i = 0; i < innerDim; ++i) {
    correct_answer += (i * i) * scale;
  }

  for(unsigned i = 0; i < outerDim; ++i){
    CHECK_IF(success, correct_answer == answers[i]);
    answers[i] = 0; // zero for next iteration
  }
  return success;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type partialsType;
  Type outType;
  unsigned outerDim, innerDim;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
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
     "Inner dimension");
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

  if (!doTest(deviceType, partialsType, outType, outerDim, innerDim))
    return 1;
  return 0;
}
