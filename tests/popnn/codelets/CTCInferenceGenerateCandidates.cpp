// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceCodeletTestConnection.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/Util.hpp>
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

template <typename InputType, typename PartialsType>
std::vector<Candidate<PartialsType>> runGenerateCandidatesCodelet(
    Graph &graph, TestDevice &device, DeviceType deviceType, Type inType,
    Type partialsType, const boost::multi_array<InputType, 2> &logProbsIn,
    unsigned timestep,
    const std::vector<BeamProbability<PartialsType>> &beamProbs,
    const BeamHistory &beamHistory, unsigned classToMakeAddend, unsigned beam,
    unsigned blankClass, bool testGenerateCopyVertex, bool profile) {
  const auto target = graph.getTarget();

  const auto maxT = logProbsIn.size();
  const auto numClassesIncBlank = logProbsIn[0].size();
  const auto beamwidth = beamProbs.size();

  auto logProbs =
      graph.addVariable(inType, {maxT, numClassesIncBlank}, "logProbs");

  // Codelet tests don't add an extra timestep = 0 initial state like the
  // main implementation does. The effect of this is that we need to provide
  // timestep + 1 here to compensate
  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep + 1);
  auto complete = graph.addConstant(UNSIGNED_INT, {}, 0u);

  graph.setTileMapping(logProbs, 0);
  graph.setTileMapping(currentTimestep, 0);
  graph.setTileMapping(complete, 0);

  auto cs = graph.addComputeSet("cs");
  const auto vertexName = testGenerateCopyVertex
                              ? "popnn::CTCGenerateCopyCandidates"
                              : "popnn::CTCGenerateExtendCandidates";
  auto vertex = graph.addVertex(
      cs, templateVertex(vertexName, inType, partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["logProbs"], logProbs.flatten());
  graph.connect(vertex["currentTimestep"], currentTimestep);
  graph.connect(vertex["complete"], complete);

  graph.setInitialValue(vertex["numClassesIncBlank"], numClassesIncBlank);

  if (testGenerateCopyVertex) {
    graph.setInitialValue(vertex["beamIdx"], beam);
    graph.setInitialValue(vertex["blankClass"], blankClass);
  } else {
    graph.setInitialValue(vertex["startBeam"], 0);
    graph.setInitialValue(vertex["endBeam"], beamwidth);
    graph.setInitialValue(vertex["addendSymbol"], classToMakeAddend);
  }
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  CandidateHandles rawCandidates;
  if (testGenerateCopyVertex) {
    rawCandidates =
        createAndConnectCandidates(graph, vertex, "candidate", partialsType, {},
                                   uploadProg, downloadProg, tmap);

  } else {
    rawCandidates = createAndConnectCandidates(graph, vertex, "extendCandidate",
                                               partialsType, {beamwidth},
                                               uploadProg, downloadProg, tmap);
  }

  // Inputs
  std::unique_ptr<char[]> rawLogProbs;
  rawLogProbs = allocateHostMemoryForTensor(logProbs, "logProbs", graph,
                                            uploadProg, downloadProg, tmap);

  auto rawBeamProbs = createAndConnectBeamProbs(
      graph, vertex, partialsType, {beamwidth},
      (testGenerateCopyVertex ? BeamScalars::NON_BLANK : BeamScalars::BLANK),
      uploadProg, downloadProg, tmap);

  // Initialise inputs
  std::vector<unsigned> lastBeamOutputsIn{};
  std::vector<double> beamProbNonBlankIn{};
  std::vector<double> beamProbBlankIn{};
  std::vector<double> beamProbTotalIn{};
  for (unsigned i = 0; i < beamwidth; i++) {
    lastBeamOutputsIn.push_back(beamHistory.getLastOutput(i));
    beamProbNonBlankIn.push_back(beamProbs[i].pnb);
    beamProbBlankIn.push_back(beamProbs[i].pb);
    beamProbTotalIn.push_back(log::add(beamProbs[i].pb, beamProbs[i].pnb));
  }

  copy(target, logProbsIn, inType, rawLogProbs.get());
  copy(target, lastBeamOutputsIn, UNSIGNED_INT, rawBeamProbs.lastOutput.get());
  copy(target, beamProbNonBlankIn, partialsType, rawBeamProbs.pnb.get());
  copy(target, beamProbBlankIn, partialsType, rawBeamProbs.pb.get());
  copy(target, beamProbTotalIn, partialsType, rawBeamProbs.pTotal.get());

  // TODO Need to initialise outputs?

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

  const auto totalCandidates = testGenerateCopyVertex ? 1 : beamwidth;
  std::vector<unsigned> candidateParentOut(totalCandidates);
  std::vector<unsigned> candidateAddendOut(totalCandidates);
  std::vector<unsigned> validCandidatesOut(1);

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

  std::vector<Candidate<float>> candidates;
  for (unsigned i = 0; i < totalCandidates; i++) {
    candidates.push_back({candidateParentOut[i], candidateAddendOut[i],
                          candidateBeamProbNonBlankOut[i],
                          candidateBeamProbBlankOut[i],
                          candidateBeamProbTotalOut[i]});
  }
  return candidates;
}

template std::vector<Candidate<float>> runGenerateCandidatesCodelet(
    Graph &graph, TestDevice &device, DeviceType deviceType, Type inType,
    Type partialsType, const boost::multi_array<float, 2> &logProbsIn,
    unsigned timestep, const std::vector<BeamProbability<float>> &beamProbs,
    const BeamHistory &beamHistory, unsigned classToMakeAddend, unsigned beam,
    unsigned blankClass, bool testGenerateCopyVertex, bool profile);

} // namespace ctc
} // namespace poplibs_test
