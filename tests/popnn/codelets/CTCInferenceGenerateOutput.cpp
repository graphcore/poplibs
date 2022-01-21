// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceCodeletTestConnection.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/Util.hpp>
#include <poputil/VertexTemplates.hpp>

#include <poplibs_test/TempDir.hpp>

#include <boost/multi_array.hpp>

#include <limits>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::ctc;
using namespace poplibs_test;
using namespace poplibs_test::util;
using namespace poplibs_support;
using namespace poputil;

namespace poplibs_test {
namespace ctc {

std::vector<unsigned> runGenerateOutputCodelet(
    Graph &graph, TestDevice &device, DeviceType deviceType, unsigned timestep,
    const BeamHistory &beamHistory, unsigned beamOutLength, unsigned outputBeam,
    unsigned numClassesIncBlank, bool profile) {
  const auto target = graph.getTarget();

  const auto beamwidth = beamHistory.symbols.size();
  const auto maxT = beamHistory.symbols[0].size();

  auto beamOutput = graph.addVariable(UNSIGNED_INT, {maxT}, "beamOutput");
  auto outputLength = graph.addVariable(UNSIGNED_INT, {}, "outputLength");

  auto beamAddend =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamAddend");
  auto beamParent =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamParent");
  const std::vector<unsigned> length(2 * beamwidth, beamOutLength);
  auto beamLength = graph.addConstant(UNSIGNED_INT, {2 * beamwidth},
                                      ArrayRef(length), "beamLength");

  // TODO - this is the length already formed, what will the timestep be
  // when we run for real?
  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);

  graph.setTileMapping(beamOutput, 0);
  graph.setTileMapping(outputLength, 0);
  graph.setTileMapping(beamAddend, 0);
  graph.setTileMapping(beamParent, 0);
  graph.setTileMapping(beamLength, 0);

  graph.setTileMapping(currentTimestep, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex = graph.addVertex(
      cs, templateVertex("popnn::CTCGenerateOutput", UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["beamOutput"], beamOutput);
  graph.connect(vertex["outputLength"], outputLength);
  graph.connect(vertex["beamAddend"], beamAddend.flatten());
  graph.connect(vertex["beamParent"], beamParent.flatten());
  graph.connect(vertex["dataLength"], currentTimestep);
  graph.connect(vertex["beamLength"], beamLength);

  graph.setInitialValue(vertex["beam"], outputBeam);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  // Inputs
  std::unique_ptr<char[]> rawBeamAddend, rawBeamParent;

  rawBeamAddend = allocateHostMemoryForTensor(beamAddend, "beamAddend", graph,
                                              uploadProg, downloadProg, tmap);
  rawBeamParent = allocateHostMemoryForTensor(beamParent, "beamParent", graph,
                                              uploadProg, downloadProg, tmap);

  std::vector<unsigned> beamAddendIn{};
  std::vector<unsigned> beamParentIn{};
  for (unsigned t = 0; t < maxT; t++) {
    for (unsigned b = 0; b < beamwidth; b++) {
      beamAddendIn.push_back(beamHistory.symbols[b][t]);
      if (beamHistory.parents[b][t]) {
        beamParentIn.push_back(*beamHistory.parents[b][t]);
      } else {
        // TODO need this column in real impl?
        beamParentIn.push_back(std::numeric_limits<unsigned>::max());
      }
    }
  }

  copy(target, beamAddendIn, UNSIGNED_INT, rawBeamAddend.get());
  copy(target, beamParentIn, UNSIGNED_INT, rawBeamParent.get());

  // Outputs
  std::unique_ptr<char[]> rawBeamOutput, rawOutputLength;

  rawBeamOutput = allocateHostMemoryForTensor(beamOutput, "beamOutput", graph,
                                              uploadProg, downloadProg, tmap);
  rawOutputLength = allocateHostMemoryForTensor(
      outputLength, "outputLength", graph, uploadProg, downloadProg, tmap);

  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (profile) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }
  Sequence prog;
  prog.add(Execute(cs));
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);
  attachStreams(engine, tmap);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();
  });

  std::vector<unsigned> outputLengthResult(1);
  std::vector<unsigned> beamOutputResult(maxT);
  copy(target, UNSIGNED_INT, rawOutputLength.get(), outputLengthResult);
  copy(target, UNSIGNED_INT, rawBeamOutput.get(), beamOutputResult);

  beamOutputResult.resize(outputLengthResult[0]);

  if (profile && deviceType != DeviceType::Cpu) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  return beamOutputResult;
}

} // namespace ctc
} // namespace poplibs_test
