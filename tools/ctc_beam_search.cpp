// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/MatrixTransforms.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/CTCInference.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <fstream>
#include <iomanip>
#include <random>

namespace po = boost::program_options;

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test;
using namespace poplibs_test::util;
using namespace poplibs_support;
using namespace poputil;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.04
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-3
#define HALF_ABS_TOL 1e-2

void beamSearchIPU(std::size_t maxTime, std::size_t batchSize,
                   unsigned blankClass, std::size_t numClasses,
                   unsigned beamwidth, unsigned topPaths, Type inType,
                   Type outType, const DeviceType &deviceType,
                   boost::optional<unsigned> tiles) {

  auto device = createTestDevice(deviceType, 1, tiles);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);

  // Create the inputs to the beam search function
  const popnn::ctc::Plan plan = popnn::ctc_infer::plan(
      graph, inType, batchSize, maxTime, numClasses, beamwidth);

  auto data = popnn::ctc_infer::createDataInput(
      graph, inType, batchSize, maxTime, numClasses, plan, "DataInput");

  auto dataLengths = graph.addVariable(UNSIGNED_INT, {batchSize});
  graph.setTileMapping(dataLengths, 0);

  // Call both beam search functions as a placeholder test that they exist and
  // execute
  Sequence prog;
  popnn::ctc_infer::beamSearchDecoderLogits(graph, data, dataLengths, prog,
                                            blankClass, beamwidth, topPaths,
                                            plan, "BeamSearchLogits");
  popnn::ctc_infer::beamSearchDecoderLogProbabilities(
      graph, data, dataLengths, prog, blankClass, beamwidth, topPaths, plan,
      "BeamSearchLogProbabilities");
  Engine engine(graph, prog);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();
  });

  return;
}

int main(int argc, char **argv) {
  // Default input parameters.
  DeviceType deviceType = DeviceType::IpuModel2;
  unsigned maxTime = 15;
  unsigned blankClass = 0;
  unsigned numClasses = 4;
  unsigned batchSize = 1;
  unsigned beamwidth = 3;
  unsigned topPaths = 2;
  boost::optional<unsigned> tiles = boost::none;
  Type inType = FLOAT;
  Type outType = FLOAT;

  beamSearchIPU(maxTime, batchSize, blankClass, numClasses, beamwidth, topPaths,
                inType, outType, deviceType, tiles);

  return false;
}
