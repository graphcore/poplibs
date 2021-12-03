// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Create an executable to check for determinism issues;
// running this executable twice should produce the same
// executable binary.
#include <array>
#include <fstream>
#include <iostream>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>

#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>

using namespace poplar;
using namespace poplar::program;

int main(int argc, char *argv[]) try {
  // Parse command line argument(s).
  if (argc != 2) {
    std::cerr << "Error: Invalid options - an output path must be specified.\n"
              << "Usage: " << argv[0] << " OUTPUT_PATH\n";
    return EXIT_FAILURE;
  }
  const char *exeFilePath = argv[1];

  constexpr bool rearrangeOnHost = true;
  constexpr float rateInitialValue = 3;
  constexpr std::array<size_t, 2> dims = {3, 4};
  const Type partialsType = FLOAT;
  const Type outType = FLOAT;

  // Create a graph.
  Target target = Target::createIPUTarget(2, 2, "IPU-POD16");
  Graph graph(target, replication_factor(2u));
  popops::addCodelets(graph);

  size_t numElements = !dims.empty();
  for (size_t dim : dims)
    numElements *= dim;

  Tensor in = graph.addVariable(partialsType, dims, "in");
  graph.setTileMapping(in, 0);

  Tensor rate = graph.addVariable(FLOAT, {});
  graph.setTileMapping(rate, 0);
  graph.setInitialValue(rate, rateInitialValue);

  DataStream h2d = graph.addHostToDeviceFIFO("h2d", partialsType, numElements);
  DataStream d2h = graph.addDeviceToHostFIFO("d2h", outType, dims[0]);

  // Create a control program.
  Sequence prog;
  prog.add(Copy(h2d, in, rearrangeOnHost));
  Tensor out = popops::reduce(graph, in, outType, {1},
                              {popops::Operation::ADD, false, rate}, prog);
  prog.add(Copy(out, d2h, rearrangeOnHost));

  // Serialise and dump the executable.
  Executable exe = compileGraph(graph, {prog});
  std::ofstream exeFileStream(exeFilePath);
  exe.serialize(exeFileStream);
  std::cout << "Wrote: " << exeFilePath << std::endl;
  return EXIT_SUCCESS;
} catch (std::runtime_error const &ex) {
  std::cerr << ex.what() << "\n";
  return EXIT_FAILURE;
}
