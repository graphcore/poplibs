// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceCodeletTestConnection.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/VertexTemplates.hpp>

#include <boost/multi_array.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::ctc;
using namespace poplibs_test;
using namespace poplibs_test::util;
using namespace poplibs_support;
using namespace poputil;

namespace poplibs_test {
namespace ctc {
template <typename PartialsType>
std::vector<Candidate<PartialsType>> runRankCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &unsortedCandidates,
    unsigned beamwidth, unsigned timestep, bool profile) {
  const auto target = graph.getTarget();

  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);
  auto dataLength = graph.addConstant(UNSIGNED_INT, {}, timestep);

  graph.setTileMapping(currentTimestep, 0);
  graph.setTileMapping(dataLength, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex = graph.addVertex(cs, templateVertex("popnn::CTCRankCandidates",
                                                   partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["currentTimestep"], currentTimestep);
  graph.connect(vertex["dataLength"], currentTimestep);

  const auto totalCandidates = unsortedCandidates.size();
  graph.setInitialValue(vertex["totalCandidates"], totalCandidates);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);
  graph.setInitialValue(vertex["firstCandidateToRank"], 0);
  graph.setInitialValue(vertex["lastCandidateToRank"], totalCandidates);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  auto rawCandidates = createAndConnectCandidates(
      graph, vertex, "candidate", partialsType, {totalCandidates}, uploadProg,
      downloadProg, tmap);

  std::vector<unsigned> candidateParentIn{};
  std::vector<unsigned> candidateAddendIn{};
  std::vector<float> candidateBeamProbNonBlankIn{};
  std::vector<float> candidateBeamProbBlankIn{};
  std::vector<float> candidateBeamProbTotalIn{};

  for (unsigned c = 0; c < totalCandidates; c++) {
    candidateParentIn.push_back(unsortedCandidates[c].beam);
    candidateAddendIn.push_back(unsortedCandidates[c].addend);
    candidateBeamProbNonBlankIn.push_back(unsortedCandidates[c].pnb);
    candidateBeamProbBlankIn.push_back(unsortedCandidates[c].pb);
    candidateBeamProbTotalIn.push_back(unsortedCandidates[c].pTotal);
  }

  copy(target, candidateParentIn, UNSIGNED_INT, rawCandidates.parent.get());
  copy(target, candidateAddendIn, UNSIGNED_INT, rawCandidates.addend.get());
  copy(target, candidateBeamProbNonBlankIn, partialsType,
       rawCandidates.probNonBlank.get());
  copy(target, candidateBeamProbBlankIn, partialsType,
       rawCandidates.probBlank.get());
  copy(target, candidateBeamProbTotalIn, partialsType,
       rawCandidates.probTotal.get().get());

  // Outputs
  auto rawSortedCandidates =
      createAndConnectCandidates(graph, vertex, "sortedCandidate", partialsType,
                                 {beamwidth}, uploadProg, downloadProg, tmap);

  OptionFlags engineOptions;
  if (profile) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Sequence prog;
  prog.add(Execute(cs));
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);
  attachStreams(engine, tmap);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();
  });

  std::vector<unsigned> candidateParentOut(beamwidth);
  std::vector<unsigned> candidateAddendOut(beamwidth);

  // TODO partialsType == float
  std::vector<float> candidateBeamProbBlankOut(beamwidth);
  std::vector<float> candidateBeamProbNonBlankOut(beamwidth);
  std::vector<float> candidateBeamProbTotalOut(beamwidth);

  copy(target, UNSIGNED_INT, rawSortedCandidates.parent.get(),
       candidateParentOut);
  copy(target, UNSIGNED_INT, rawSortedCandidates.addend.get(),
       candidateAddendOut);
  copy(target, partialsType, rawSortedCandidates.probNonBlank.get(),
       candidateBeamProbNonBlankOut);
  copy(target, partialsType, rawSortedCandidates.probBlank.get(),
       candidateBeamProbBlankOut);
  copy(target, partialsType, rawSortedCandidates.probTotal.get().get(),
       candidateBeamProbTotalOut);
  if (profile && deviceType != DeviceType::Cpu) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  std::vector<Candidate<float>> selectedCandidates;
  for (unsigned i = 0; i < beamwidth; i++) {
    selectedCandidates.push_back({candidateParentOut[i], candidateAddendOut[i],
                                  candidateBeamProbNonBlankOut[i],
                                  candidateBeamProbBlankOut[i],
                                  candidateBeamProbTotalOut[i]});
  }
  return selectedCandidates;
}

template std::vector<Candidate<float>> runRankCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<float>> &candidates, unsigned beamwidth,
    unsigned timestep, bool profile);

} // namespace ctc
} // namespace poplibs_test
