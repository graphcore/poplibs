// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <math.h>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

using namespace poplar;
using namespace poplar::program;

int main() {
  std::cout << "Getting HW target" << std::endl;
  static poplar::DeviceManager device_mgr =
      poplar::DeviceManager::createDeviceManager();
  poplar::Device device;
  auto device_list = device_mgr.getDevices();
  bool have_device = false;
  for (auto &d : device_list) {
    if (d.getTarget().getTargetType() == poplar::TargetType::IPU &&
        d.getTarget().getNumIPUs() == 8) {
      if (d.attach()) {
        device = std::move(d);
        std::cout << "Attached. " << std::endl;
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

  Sequence sequence;

  std::cout << "Creating graph\n";

  Graph master_graph(device, poplar::replication_factor(2));
  popops::addCodelets(master_graph);

  std::vector<Graph> sharded_graphs;
  std::vector<Sequence> programs(8);
  std::vector<Sequence> copies(8);
  auto num_ipus = master_graph.getTarget().getNumIPUs();

  std::cout << "Number of IPUs: " << num_ipus << "\n";
  // Check that we have enough IPUs for this sharding configuration.
  auto tiles_per_ipu = master_graph.getTarget().getTilesPerIPU();
  for (unsigned ipu = 0; ipu < num_ipus; ++ipu) {
    sharded_graphs.emplace_back(master_graph.createVirtualGraph(
        ipu * tiles_per_ipu, (ipu + 1) * tiles_per_ipu));
  }

  // Stage 1.
  auto var0 = sharded_graphs[0].addVariable(HALF, {200});
  sharded_graphs[0].setTileMapping(var0, 0);
  auto var1 = sharded_graphs[0].addVariable(HALF, {200});
  sharded_graphs[0].setTileMapping(var1, 0);

  auto out = var0;
  popops::addInPlace(sharded_graphs[0], var0, var1, programs[0]);
  out = poputil::copyToIpu(master_graph, out, copies[0], 1);

  // Stage 2.
  auto var2 = sharded_graphs[1].addVariable(HALF, {200});
  sharded_graphs[1].setTileMapping(var2, 0);

  popops::addInPlace(sharded_graphs[1], out, var2, programs[1]);
  out = poputil::copyToIpu(master_graph, out, copies[1], 2);

  // Stage 3.
  auto var3 = sharded_graphs[2].addVariable(HALF, {200});
  sharded_graphs[2].setTileMapping(var3, 0);

  popops::addInPlace(sharded_graphs[2], out, var3, programs[2]);
  out = poputil::copyToIpu(master_graph, out, copies[2], 3);

  // Stage 4.
  auto var4 = sharded_graphs[3].addVariable(HALF, {200});
  sharded_graphs[3].setTileMapping(var4, 0);

  popops::addInPlace(sharded_graphs[3], out, var4, programs[3]);

  // Ramp up.
  sequence.add(programs[0]);
  sequence.add(copies[0]);

  sequence.add(programs[1]);
  sequence.add(copies[1]);

  // Repeat.
  sequence.add(programs[0]);
  sequence.add(copies[0]);
  sequence.add(programs[2]);
  sequence.add(copies[2]);

  sequence.add(programs[1]);
  sequence.add(copies[1]);
  sequence.add(programs[3]);
  sequence.add(copies[3]);

  sequence.add(programs[0]);
  sequence.add(copies[0]);
  sequence.add(programs[2]);
  sequence.add(copies[2]);

  sequence.add(programs[1]);
  sequence.add(copies[1]);
  sequence.add(programs[3]);
  sequence.add(copies[3]);

  // Ramp down.
  sequence.add(programs[2]);
  sequence.add(copies[2]);

  sequence.add(programs[3]);
  sequence.add(copies[3]);

  std::cerr << "Create engine\n";
  Engine engine(master_graph, {Sequence(), sequence, Sequence()});
  std::cerr << "Load engine\n";
  engine.load(device);
  std::cerr << "Before run0\n";
  engine.run(0);
  std::cerr << "Before run1\n";
  engine.run(1);
  std::cerr << "Before run2\n";
  engine.run(2);
  std::cerr << "After run\n";

  return 0;
}
