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
// TODO - This test is disabled at present.  The vertex is likely to get
// modified further so leave changing this (as it is not trivial) until the
// proper solution is found.
template <typename PartialsType>
std::vector<Candidate<PartialsType>> runSimpleSortCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &unsortedCandidates,
    unsigned beamwidth, unsigned timestep, bool profile) {
  const auto target = graph.getTarget();

  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);
  auto dataLength = graph.addConstant(UNSIGNED_INT, {}, timestep);

  auto cs = graph.addComputeSet("cs");
  auto vertex =
      graph.addVertex(cs, templateVertex("popnn::CTCSimpleSortCandidates",
                                         partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  const auto totalCandidates = unsortedCandidates.size();
  graph.setInitialValue(vertex["totalCandidates"], totalCandidates);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);

  graph.connect(vertex["currentTimestep"], currentTimestep);
  graph.connect(vertex["dataLength"], dataLength);

  graph.setTileMapping(currentTimestep, 0);
  graph.setTileMapping(dataLength, 0);

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

  std::vector<unsigned> candidateParentOut(totalCandidates);
  std::vector<unsigned> candidateAddendOut(totalCandidates);

  // TODO partialsType == float
  std::vector<float> candidateBeamProbBlankOut(totalCandidates);
  std::vector<float> candidateBeamProbNonBlankOut(totalCandidates);
  std::vector<float> candidateBeamProbTotalOut(totalCandidates);

  copy(target, UNSIGNED_INT, rawCandidates.parent.get(), candidateParentOut);
  copy(target, UNSIGNED_INT, rawCandidates.addend.get(), candidateAddendOut);
  copy(target, partialsType, rawCandidates.probNonBlank.get(),
       candidateBeamProbNonBlankOut);
  copy(target, partialsType, rawCandidates.probBlank.get(),
       candidateBeamProbBlankOut);
  copy(target, partialsType, rawCandidates.probTotal.get().get(),
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

template std::vector<Candidate<float>> runSimpleSortCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<float>> &candidates, unsigned beamwidth,
    unsigned timestep, bool profile);

} // namespace ctc
} // namespace poplibs_test
