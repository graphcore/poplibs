#include <poplar/Engine.hpp>
#include "TestDevice.hpp"
// codelets
#include "poplin/codelets.hpp"
#include "poplin/Convolution.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/Util.hpp"
#include "poplibs_test/Util.hpp"
#include "poplar/Target.hpp"
#include <string.h>
#include <stdexcept>

#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
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
  poplin::addCodelets(graph);

  // Claim enough space for floats
  std::vector<char> data(innerDim * outerDim * 4);
  std::vector<float> nums(innerDim * outerDim);
  for (unsigned i = 0; i < innerDim * outerDim; ++i) {
    nums[i] = 1.0 * (i % outerDim);
  }
  copy(target, nums.data(), innerDim*outerDim, partialsType, data.data());
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
  Tensor out;
  out = graph.addVariable(outType, {outerDim+1});

  const auto vertexClass = templateVertex("poplin::ReduceAdd",
                                          outType, partialsType);
  auto v1 = graph.addVertex(cs,
                            vertexClass);

  for (int i = 0; i < innerDim; ++i) {
    Tensor Row = partials.slice(i, i+1, 0);
    graph.connect(v1["partials"][i], Row.reshape({outerDim}));
  }
  graph.setFieldSize(v1["partials"], innerDim);
  graph.connect(v1["out"], out.slice(0, outerDim));
  graph.setInitialValue(v1["numPartials"], innerDim);
  graph.setInitialValue(v1["numElems"], outerDim);

  graph.setTileMapping(v1, 0);
  graph.setTileMapping(partials, 0);
  graph.setTileMapping(out, 0);

  graph.createHostWrite("partials", partials);
  graph.createHostWrite("outw", out);
  graph.createHostRead("out", out);

  prog.add(Execute(cs));



  Engine e(graph, prog);

  device.bind([&](const Device &d) {
    e.load(d);
    e.writeTensor("partials", data.data());
    e.writeTensor("outw", ans_data.data());
    e.readTensor("out", ans_data.data());

    e.run();

    e.readTensor("out", ans_data.data());
  });

  copy(target, outType, ans_data.data(), answers.data(), outerDim+1);

  bool success = true;
  for(unsigned i = 0; i < outerDim; ++i){
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
