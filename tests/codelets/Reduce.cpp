#include <poplar/Engine.hpp>
#include "TestDevice.hpp"
// codelets
#include "popops/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/Util.hpp"
#include "poplibs_test/Util.hpp"
#include <string.h>
#include <stdexcept>

#include <boost/program_options.hpp>

#define INNER_DIM 11
#define PARTIALS_ARE 1
#define SCALE 2.0
#define UPDATE false

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;
using namespace poplibs_test::util;


static bool do_test(const DeviceType &deviceType,
                    const Type &inType,
                    const Type &outType,
                    unsigned outerDim) {
  static unsigned outer_dim = outerDim * (1 + PARTIALS_ARE);

  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();
  Graph graph(device);
  popops::addCodelets(graph);

  std::vector<unsigned short> has(2 * outerDim);
  std::fill(has.begin(), has.end(), 15360);
  std::vector<float> answers(2 * outerDim);
  std::fill(answers.begin(), answers.end(), 1.0);
  std::vector<char> ans_data(2 * outerDim*4);

  if (outType == FLOAT) {
    memcpy(ans_data.data(), answers.data(), outerDim*4*2);
  } else {
    memcpy(ans_data.data(), has.data(), outerDim*2*2);
  }

  std::vector<unsigned char> data(INNER_DIM * outer_dim * 4);
  std::vector<float> nums(INNER_DIM * outer_dim);
  for (unsigned i = 0; i < INNER_DIM * outer_dim; ++i) {
    nums[i] = 1.0 * (i % outerDim);
  }
  copy(target, nums.data(), INNER_DIM * outer_dim, inType, data.data());


  std::vector<unsigned> counts(2);
  counts[0] = INNER_DIM;
  counts[1] = INNER_DIM;

  Sequence prog;

  auto cs = graph.addComputeSet("cs");

  auto partials = graph.addVariable(inType, {INNER_DIM, outer_dim});
  auto partials_2 = graph.addVariable(inType, {INNER_DIM, outer_dim});
  auto out = graph.addVariable(outType, {2, outerDim});

  const auto vertexClass = templateVertex("popops::Reduce",
                              "popops::ReduceAdd",
                              inType, outType,
                              false, true, UPDATE);
  auto v1 = graph.addVertex(cs,
                            vertexClass);

  for (unsigned i = 0; i < INNER_DIM; ++i) {
    Tensor Row = partials.slice(i, i+1, 0);
    graph.connect(v1["partials"][i], Row.reshape({outer_dim}));
  }
  for (unsigned i = 0; i < INNER_DIM; ++i) {
    Tensor Row = partials_2.slice(i, i+1, 0);
    graph.connect(v1["partials"][i+INNER_DIM], Row.reshape({outer_dim}));
  }
  graph.setFieldSize(v1["partials"], 2*INNER_DIM);
  graph.connect(v1["out"], out);
  graph.setInitialValue(v1["k"], SCALE);
  graph.setInitialValue(v1["numPartials"], counts);

  graph.setTileMapping(v1, 0);
  graph.setTileMapping(partials, 0);
  graph.setTileMapping(partials_2, 0);
  graph.setTileMapping(out, 0);

  graph.createHostWrite("partials",
                        partials);
  graph.createHostWrite("partials_2",
                        partials_2);
  graph.createHostWrite("outw",
                        out);
  graph.createHostRead("out",
                        out);

  prog.add(Execute(cs));

  Engine e(graph, prog,
           OptionFlags{{"target.textSectionSizeInBytes", "0x9000"}});

  e.load(device);
  e.writeTensor("partials", data.data());
  e.writeTensor("partials_2", data.data());
  e.writeTensor("outw", ans_data.data());
  e.readTensor("out", ans_data.data());

  e.run();

  e.readTensor("out",
                ans_data.data());

  copy(target, outType, ans_data.data(), answers.data(), outerDim*2);

  bool sucess = true;
  for(int i =0; i < outerDim * 2; ++i){
    if ((INNER_DIM * 4.0 * (i % outerDim)) != answers[i]) {
      sucess = false;
      std::cerr << "Condition failed: index " << i
                << " expected " << (INNER_DIM * 4.0 * (i % outerDim))
                << " actual " << answers[i] << "\n";
    }
  }
  return sucess;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type inType;
  Type outType;
  unsigned outerDim;

  po::options_description desc("Options");
  desc.add_options() ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("in-type",
     po::value<Type>(&inType)->required(),
     "In Type")
    ("out-type",
     po::value<Type>(&outType)->required(),
     "Output Type")
    ("outer-dim",
     po::value<unsigned>(&outerDim)->required(),
     "Outer dimension");

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
  return !do_test(deviceType, inType, outType, outerDim);
}
