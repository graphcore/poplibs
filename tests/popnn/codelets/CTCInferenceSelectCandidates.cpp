// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceSelectCandidates.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/TopK.hpp>
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
std::vector<Candidate<PartialsType>> runSelectCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<PartialsType>> &unsortedCandidates,
    unsigned beamwidth, bool profile) {
  const auto target = graph.getTarget();

  const auto totalCandidates = unsortedCandidates.size();

  auto candidateParent =
      graph.addVariable(UNSIGNED_INT, {totalCandidates}, "candidateParent");
  auto candidateAddend =
      graph.addVariable(UNSIGNED_INT, {totalCandidates}, "candidateAddend");
  auto candidateBeamProbNonBlank = graph.addVariable(
      partialsType, {totalCandidates}, "candidateBeamProbNonBlank");
  auto candidateBeamProbBlank = graph.addVariable(
      partialsType, {totalCandidates}, "candidateBeamProbBlank");
  auto candidateProbTotalScratch = graph.addVariable(
      partialsType, {totalCandidates}, "candidateProbTotalScratch");

  graph.setTileMapping(candidateParent, 0);
  graph.setTileMapping(candidateAddend, 0);
  graph.setTileMapping(candidateBeamProbNonBlank, 0);
  graph.setTileMapping(candidateBeamProbBlank, 0);
  graph.setTileMapping(candidateProbTotalScratch, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex = graph.addVertex(cs, templateVertex("popnn::CTCSelectCandidates",
                                                   partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["candidateParent"], candidateParent);
  graph.connect(vertex["candidateAddend"], candidateAddend);
  graph.connect(vertex["candidateBeamProbNonBlank"], candidateBeamProbNonBlank);
  graph.connect(vertex["candidateBeamProbBlank"], candidateBeamProbBlank);
  graph.connect(vertex["candidateProbTotalScratch"], candidateProbTotalScratch);

  graph.setInitialValue(vertex["totalCandidates"], totalCandidates);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  // Inputs
  std::unique_ptr<char[]> rawCandidateParent, rawCandidateAddend,
      rawCandidateBeamProbNonBlank, rawCandidateBeamProbBlank;

  rawCandidateParent =
      allocateHostMemoryForTensor(candidateParent, "candidateParent", graph,
                                  uploadProg, downloadProg, tmap);
  rawCandidateAddend =
      allocateHostMemoryForTensor(candidateAddend, "candidateAddend", graph,
                                  uploadProg, downloadProg, tmap);
  rawCandidateBeamProbNonBlank = allocateHostMemoryForTensor(
      candidateBeamProbNonBlank, "candidateBeamProbNonBlank", graph, uploadProg,
      downloadProg, tmap);
  rawCandidateBeamProbBlank = allocateHostMemoryForTensor(
      candidateBeamProbBlank, "candidateBeamProbBlank", graph, uploadProg,
      downloadProg, tmap);

  std::vector<unsigned> candidateParentIn{};
  std::vector<unsigned> candidateAddendIn{};
  std::vector<float> candidateBeamProbNonBlankIn{};
  std::vector<float> candidateBeamProbBlankIn{};

  for (unsigned c = 0; c < totalCandidates; c++) {
    candidateParentIn.push_back(unsortedCandidates[c].beam);
    candidateAddendIn.push_back(unsortedCandidates[c].addend);
    candidateBeamProbNonBlankIn.push_back(unsortedCandidates[c].pnb);
    candidateBeamProbBlankIn.push_back(unsortedCandidates[c].pb);
  }

  copy(target, candidateParentIn, UNSIGNED_INT, rawCandidateParent.get());
  copy(target, candidateAddendIn, UNSIGNED_INT, rawCandidateAddend.get());
  copy(target, candidateBeamProbNonBlankIn, partialsType,
       rawCandidateBeamProbNonBlank.get());
  copy(target, candidateBeamProbBlankIn, partialsType,
       rawCandidateBeamProbBlank.get());

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

  copy(target, UNSIGNED_INT, rawCandidateParent.get(), candidateParentOut);
  copy(target, UNSIGNED_INT, rawCandidateAddend.get(), candidateAddendOut);
  copy(target, partialsType, rawCandidateBeamProbNonBlank.get(),
       candidateBeamProbNonBlankOut);
  copy(target, partialsType, rawCandidateBeamProbBlank.get(),
       candidateBeamProbBlankOut);
  if (profile && deviceType != DeviceType::Cpu) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  std::vector<Candidate<float>> selectedCandidates;
  for (unsigned i = 0; i < beamwidth; i++) {
    selectedCandidates.push_back({candidateParentOut[i], candidateAddendOut[i],
                                  candidateBeamProbNonBlankOut[i],
                                  candidateBeamProbBlankOut[i]});
  }
  return selectedCandidates;
}

template std::vector<Candidate<float>> runSelectCandidatesCodelet(
    poplar::Graph &graph, poplibs_support::TestDevice &device,
    poplibs_support::DeviceType deviceType, poplar::Type partialsType,
    const std::vector<Candidate<float>> &candidates, unsigned beamwidth,
    bool profile);

} // namespace ctc
} // namespace poplibs_test
