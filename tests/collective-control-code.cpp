// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#include <iostream>
#include <math.h>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

using namespace poplar;
using namespace poplar::program;
namespace pe = popops::expr;

Sequence AllReduce(Graph &graph, Tensor &input, const std::string &name) {
  Sequence seq;

  Tensor output = graph.clone(input, name + "/output");

  poplar::OptionFlags options;
  options.set("useReplicatedImplementation", "true");

  popops::replicatedAllReduceWithOutput(
      graph, input, output, popops::Operation::ADD, seq, name, options);

  return seq;
}

int main() {
  const int num_ipus = 8;

  std::cout << "Getting HW target" << std::endl;
  static poplar::DeviceManager device_mgr =
      poplar::DeviceManager::createDeviceManager();
  poplar::Device device;
  auto device_list = device_mgr.getDevices();
  bool have_device = false;
  for (auto &d : device_list) {
    if (d.getTarget().getTargetType() == poplar::TargetType::IPU &&
        d.getTarget().getNumIPUs() == num_ipus) {
      if (d.attach()) {
        device = std::move(d);
        std::cout << "Attached." << std::endl;
        have_device = true;
        break;
      }
    }
  }

  if (!have_device) {
    std::cerr << "Could not open the device";
    exit(-1);
  }
  Target target = device.getTarget();

  std::cout << "Number of IPUs: " << num_ipus << "\n";

  std::cout << "Creating graph\n";
  Graph graph(device, 0, poplar::replication_factor(num_ipus));
  popops::addCodelets(graph);

  Sequence sequence;
  auto t0 =
      graph.addVariable(HALF, {5365736}, VariableMappingMethod::LINEAR, "t0");
  auto t1 =
      graph.addVariable(HALF, {5345280}, VariableMappingMethod::LINEAR, "t1");
  auto t2 =
      graph.addVariable(HALF, {985856}, VariableMappingMethod::LINEAR, "t2");

  sequence.add(AllReduce(graph, t0, "t0"));
  sequence.add(AllReduce(graph, t1, "t1"));
  sequence.add(AllReduce(graph, t2, "t2"));

  std::cerr << "Create engine\n";
  Engine engine(graph, sequence);
  std::cerr << "Load engine\n";
  engine.load(device);
  std::cerr << "Before run\n";
  engine.run();
  std::cerr << "After run\n";
  engine.printProfileSummary(std::cout);

  return 0;
}
