// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceGenerateCandidates.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

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
    const BeamHistory &beamHistory, unsigned blankClass, bool profile) {
  const auto target = graph.getTarget();

  const auto maxT = logProbsIn.size();
  const auto numClassesIncBlank = logProbsIn[0].size();
  const auto beamwidth = beamProbs.size();

  auto logProbs =
      graph.addVariable(inType, {maxT, numClassesIncBlank}, "logProbs");
  auto lastBeamOutputs =
      graph.addVariable(UNSIGNED_INT, {beamwidth}, "lastBeamOutputs");
  auto beamProbNonBlank =
      graph.addVariable(partialsType, {beamwidth}, "beamProbNonBlank");
  auto beamProbBlank =
      graph.addVariable(partialsType, {beamwidth}, "beamProbBlank");

  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);

  const auto totalCandidates = beamwidth * numClassesIncBlank;
  auto candidateParent =
      graph.addVariable(UNSIGNED_INT, {totalCandidates}, "candidateParent");
  auto candidateAddend =
      graph.addVariable(UNSIGNED_INT, {totalCandidates}, "candidateAddend");
  auto candidateBeamProbNonBlank = graph.addVariable(
      partialsType, {totalCandidates}, "candidateBeamProbNonBlank");
  auto candidateBeamProbBlank = graph.addVariable(
      partialsType, {totalCandidates}, "candidateBeamProbBlank");

  graph.setTileMapping(logProbs, 0);
  graph.setTileMapping(lastBeamOutputs, 0);
  graph.setTileMapping(beamProbNonBlank, 0);
  graph.setTileMapping(beamProbBlank, 0);

  graph.setTileMapping(currentTimestep, 0);

  graph.setTileMapping(candidateParent, 0);
  graph.setTileMapping(candidateAddend, 0);
  graph.setTileMapping(candidateBeamProbNonBlank, 0);
  graph.setTileMapping(candidateBeamProbBlank, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex =
      graph.addVertex(cs, templateVertex("popnn::GenerateCandidates", inType,
                                         partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["logProbs"], logProbs.flatten());
  graph.connect(vertex["lastBeamOutputs"], lastBeamOutputs);
  graph.connect(vertex["beamProbNonBlank"], beamProbNonBlank);
  graph.connect(vertex["beamProbBlank"], beamProbBlank);

  graph.connect(vertex["currentTimestep"], currentTimestep);

  graph.connect(vertex["candidateParent"], candidateParent);
  graph.connect(vertex["candidateAddend"], candidateAddend);
  graph.connect(vertex["candidateBeamProbNonBlank"], candidateBeamProbNonBlank);
  graph.connect(vertex["candidateBeamProbBlank"], candidateBeamProbBlank);

  graph.setInitialValue(vertex["numClassesIncBlank"], numClassesIncBlank);
  graph.setInitialValue(vertex["blankClass"], blankClass);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  // Inputs
  std::unique_ptr<char[]> rawLogProbs, rawLastBeamOutputs, rawBeamProbNonBlank,
      rawBeamProbBlank;

  rawLogProbs = allocateHostMemoryForTensor(logProbs, "logProbs", graph,
                                            uploadProg, downloadProg, tmap);
  rawLastBeamOutputs =
      allocateHostMemoryForTensor(lastBeamOutputs, "lastBeamOutputs", graph,
                                  uploadProg, downloadProg, tmap);
  rawBeamProbNonBlank =
      allocateHostMemoryForTensor(beamProbNonBlank, "beamProbNonBlank", graph,
                                  uploadProg, downloadProg, tmap);
  rawBeamProbBlank = allocateHostMemoryForTensor(
      beamProbBlank, "beamProbBlank", graph, uploadProg, downloadProg, tmap);

  // Initialise inputs

  std::vector<unsigned> lastBeamOutputsIn{};
  std::vector<double> beamProbNonBlankIn{};
  std::vector<double> beamProbBlankIn{};
  for (unsigned i = 0; i < beamwidth; i++) {
    lastBeamOutputsIn.push_back(beamHistory.getLastOutput(i));
    beamProbNonBlankIn.push_back(beamProbs[i].pnb);
    beamProbBlankIn.push_back(beamProbs[i].pb);
  }

  copy(target, logProbsIn, inType, rawLogProbs.get());
  copy(target, lastBeamOutputsIn, UNSIGNED_INT, rawLastBeamOutputs.get());
  copy(target, beamProbNonBlankIn, partialsType, rawBeamProbNonBlank.get());
  copy(target, beamProbBlankIn, partialsType, rawBeamProbBlank.get());

  // Outputs
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

  // TODO Need to initialise outputs?

  OptionFlags engineOptions;
  if (profile) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Sequence prog;
  prog.add(Execute(cs));
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);
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

  std::vector<Candidate<float>> candidates;
  for (unsigned i = 0; i < totalCandidates; i++) {
    candidates.push_back({candidateParentOut[i], candidateAddendOut[i],
                          candidateBeamProbNonBlankOut[i],
                          candidateBeamProbBlankOut[i]});
  }
  return candidates;
}

template std::vector<Candidate<float>> runGenerateCandidatesCodelet(
    Graph &graph, TestDevice &device, DeviceType deviceType, Type inType,
    Type partialsType, const boost::multi_array<float, 2> &logProbsIn,
    unsigned timestep, const std::vector<BeamProbability<float>> &beamProbs,
    const BeamHistory &beamHistory, unsigned blankClass, bool profile);

} // namespace ctc
} // namespace poplibs_test
