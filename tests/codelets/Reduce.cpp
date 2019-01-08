#include <poplar/Engine.hpp>

#include "TestDevice.hpp"
// codelets
#include "popops/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_test/Util.hpp"
#include "poplibs_test/Util.hpp"
#include <string.h>
#include <cmath>


#include <stdexcept>

#include <boost/program_options.hpp>

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
                    unsigned outerDim,
                    unsigned INNER_DIM) {
  static unsigned outer_dim = outerDim * (1 + PARTIALS_ARE);

  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();
  Graph graph(target);
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
                              inType, outType, UPDATE);
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
  auto t = graph.addConstant(UNSIGNED_SHORT, {counts.size()}, counts.data());
  graph.connect(v1["numPartials"], t);

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

  Engine e(graph, prog);

  device.bind([&](const Device &d) {
    e.load(d);

    e.writeTensor("partials", data.data());
    e.writeTensor("partials_2", data.data());
    e.writeTensor("outw", ans_data.data());
    e.readTensor("out", ans_data.data());

    e.run();

    e.readTensor("out",
                 ans_data.data());
  });

  copy(target, outType, ans_data.data(), answers.data(), outerDim*2);

  bool sucess = true;
  for(unsigned i =0; i < outerDim * 2; ++i){
    if ((INNER_DIM * 4.0 * (i % outerDim)) != answers[i]) {
      sucess = false;
      std::cerr << "Condition failed: index " << i
                << " expected " << (INNER_DIM * 4.0 * (i % outerDim))
                << " actual " << answers[i] << "\n";
    }
  }
  return sucess;
}


static bool do_test_multi(const DeviceType &deviceType,
                    const Type &inType,
                    const Type &outType,
                    unsigned outerDim) {
  static unsigned outer_dim = outerDim * (1 + PARTIALS_ARE);
  unsigned INNER_DIM = 2;
  if (outType == HALF) {
    INNER_DIM = 1;
  }

  std::vector<float> answers(4*2*outerDim);
  std::vector<char> ans_data(4*2*outerDim*4);
  std::fill(answers.begin(), answers.end(), 0.0);

  std::fill(ans_data.begin(), ans_data.end(), 0);
  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

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

  std::vector<Tensor> outs(4);
  for (unsigned i = 0; i < outs.size(); ++i) {
    outs[i] = graph.addVariable(outType, {2, outerDim});
  }

  const auto mul_vertex = templateVertex("popops::Reduce",
                              "popops::ReduceMul",
                              inType, outType, false);
  const auto max_vertex = templateVertex("popops::Reduce",
                              "popops::ReduceMax",
                              inType, outType, false);
  const auto min_vertex = templateVertex("popops::Reduce",
                              "popops::ReduceMin",
                              inType, outType, false);
  const auto sqadd_vertex = templateVertex("popops::Reduce",
                              "popops::ReduceSquareAdd",
                              inType, outType, false);
  auto v_mul = graph.addVertex(cs, mul_vertex);
  auto v_max = graph.addVertex(cs, max_vertex);
  auto v_min = graph.addVertex(cs, min_vertex);
  auto v_sqadd = graph.addVertex(cs, sqadd_vertex);

  for (unsigned i = 0; i < INNER_DIM; ++i) {
    Tensor Row = partials.slice(i, i+1, 0);
    graph.connect(v_mul["partials"][i], Row.reshape({outer_dim}));
    graph.connect(v_max["partials"][i], Row.reshape({outer_dim}));
    graph.connect(v_min["partials"][i], Row.reshape({outer_dim}));
    graph.connect(v_sqadd["partials"][i], Row.reshape({outer_dim}));
  }
  for (unsigned i = 0; i < INNER_DIM; ++i) {
    Tensor Row = partials_2.slice(i, i+1, 0);
    graph.connect(v_mul["partials"][i+INNER_DIM], Row.reshape({outer_dim}));
    graph.connect(v_max["partials"][i+INNER_DIM], Row.reshape({outer_dim}));
    graph.connect(v_min["partials"][i+INNER_DIM], Row.reshape({outer_dim}));
    graph.connect(v_sqadd["partials"][i+INNER_DIM], Row.reshape({outer_dim}));
  }
  graph.setFieldSize(v_mul["partials"], 2*INNER_DIM);
  graph.setFieldSize(v_max["partials"], 2*INNER_DIM);
  graph.setFieldSize(v_min["partials"], 2*INNER_DIM);
  graph.setFieldSize(v_sqadd["partials"], 2*INNER_DIM);
  graph.connect(v_mul["out"], outs[0]);
  graph.connect(v_max["out"], outs[1]);
  graph.connect(v_min["out"], outs[2]);
  graph.connect(v_sqadd["out"], outs[3]);

  auto t = graph.addConstant(UNSIGNED_SHORT, {counts.size()}, counts.data());
  graph.connect(v_mul["numPartials"], t);
  graph.setInitialValue(v_mul["k"], SCALE);
  graph.connect(v_max["numPartials"], t);
  graph.setInitialValue(v_max["k"], SCALE);
  graph.connect(v_min["numPartials"], t);
  graph.setInitialValue(v_min["k"], SCALE);
  graph.connect(v_sqadd["numPartials"], t);
  graph.setInitialValue(v_sqadd["k"], SCALE);

  graph.setTileMapping(v_mul, 0);
  graph.setTileMapping(v_max, 0);
  graph.setTileMapping(v_min, 0);
  graph.setTileMapping(v_sqadd, 0);
  graph.setTileMapping(partials, 0);
  graph.setTileMapping(partials_2, 0);
  for (unsigned i = 0; i < outs.size(); ++i) {
    graph.setTileMapping(outs[i], 0);
  }

  graph.createHostWrite("partials",
                        partials);
  graph.createHostWrite("partials_2",
                        partials_2);
  for (unsigned i = 0; i < outs.size(); ++i) {
    graph.createHostWrite("outw" + std::to_string(i), outs[i]);
    graph.createHostRead("out" + std::to_string(i), outs[i]);
  }

  prog.add(Execute(cs));

  Engine e(graph, prog);
  device.bind([&](const Device &d) {
    e.load(d);
    e.writeTensor("partials", data.data());
    e.writeTensor("partials_2", data.data());
    for (int k = 0; k < 4; ++k) {
      e.writeTensor("outw" + std::to_string(k), &ans_data[k*2*outerDim*4]);
      e.readTensor("out" + std::to_string(k), &ans_data[k*2*outerDim*4]);
    }

    e.run();
    for (int k = 0; k < 4; ++k) {
      e.readTensor("out" + std::to_string(k), &ans_data[k*2*outerDim*4]);
    }
  });

  for (int k = 0; k < 4; ++k) {
    unsigned size_of_out = (FLOAT == outType) ? 4 : 2;
    copy(target, outType,
            &ans_data[k*2*outerDim*4], &answers[k*2*outerDim], outerDim);
    copy(target, outType, &ans_data[k*2*outerDim*4+outerDim*size_of_out],
          &answers[k*2*outerDim+outerDim], outerDim);
  }

  for (int j = 0; j < 2; ++j) {
    for(unsigned i =0; i < outerDim; ++i) {
      if(pow(i, 2 * INNER_DIM) * 2.0 != answers[j*outerDim + i]) {
        std::cerr << "Condition failed: index " << i << " " << j
                << " expected " << pow(i, 2 * INNER_DIM) * 2.0
                << " actual " << answers[j*outerDim + i] << "\n";
        return false;
      }
    }
  }
  for (int j = 0; j < 2; ++j) {
    for(unsigned i =0; i < outerDim; ++i) {
      if (i * 2.0 != answers[2*outerDim + j*outerDim + i]) {
        std::cerr << "Condition failed: index " << i << " " << j
                << " expected " << i * 2.0
                << " actual " << answers[2*outerDim + j*outerDim + i] << "\n";
        return false;
      }
      if (i * 2.0 != answers[2*2*outerDim + j*outerDim + i]) {
        std::cerr << "Condition failed: index " << i << " " << j
                << " expected " << i * 2.0
                << " actual " << answers[2*outerDim + j*outerDim + i] << "\n";
        return false;
      }
    }
  }
  for (int j = 0; j < 2; ++j) {
    for(unsigned i =0; i < outerDim; ++i) {
      if (i*i*INNER_DIM*2*2.0 != answers[3*2*outerDim + j*outerDim + i]) {
        std::cerr << "Condition failed: index " << i << " " << j
                << " expected " << i*i*INNER_DIM*2*2.0
                << " actual " << answers[3*2*outerDim + j*outerDim + i] << "\n";
        return false;
      }
    }
  }
  return true;
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
  if (!do_test(deviceType, inType, outType, outerDim, 4)) {
    return 1;
  }
  if (inType == outType) {
    if (!do_test_multi(deviceType, inType, outType, outerDim)) {
      return 1;
    }
  }
  return 0;
}
