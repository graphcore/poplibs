// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceUpdate.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Type.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/CTCInference.hpp>
#include <poplibs_test/Util.hpp>
#include <poputil/VertexTemplates.hpp>

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

template <typename PartialsType>
std::pair<BeamHistory, std::vector<BeamProbability<PartialsType>>>
runUpdateCodelet(Graph &graph, TestDevice &device, DeviceType deviceType,
                 Type inType, Type partialsType,
                 const std::vector<Candidate<PartialsType>> &candidates,

                 unsigned timestep, const BeamHistory &beamHistory,
                 const std::vector<BeamProbability<PartialsType>> &beamProbs,
                 unsigned blankClass, bool profile) {
  const auto target = graph.getTarget();

  const auto totalCandidates = candidates.size();
  const auto beamwidth = beamHistory.symbols.size();
  const auto maxT = beamHistory.symbols[0].size();

  auto candidateParent =
      graph.addVariable(UNSIGNED_INT, {totalCandidates}, "candidateParent");
  auto candidateAddend =
      graph.addVariable(UNSIGNED_INT, {totalCandidates}, "candidateAddend");
  auto candidateBeamProbNonBlank = graph.addVariable(
      partialsType, {totalCandidates}, "candidateBeamProbNonBlank");
  auto candidateBeamProbBlank = graph.addVariable(
      partialsType, {totalCandidates}, "candidateBeamProbBlank");

  auto beamAddend =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamAddend");
  auto beamParent =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamParent");
  auto beamProbNonBlank =
      graph.addVariable(partialsType, {beamwidth}, "beamProbNonBlank");
  auto beamProbBlank =
      graph.addVariable(partialsType, {beamwidth}, "beamProbBlank");

  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);

  graph.setTileMapping(candidateParent, 0);
  graph.setTileMapping(candidateAddend, 0);
  graph.setTileMapping(candidateBeamProbNonBlank, 0);
  graph.setTileMapping(candidateBeamProbBlank, 0);

  graph.setTileMapping(beamAddend, 0);
  graph.setTileMapping(beamParent, 0);
  graph.setTileMapping(beamProbNonBlank, 0);
  graph.setTileMapping(beamProbBlank, 0);

  graph.setTileMapping(currentTimestep, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex = graph.addVertex(
      cs, templateVertex("popnn::CTCUpdate", partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["candidateParent"], candidateParent);
  graph.connect(vertex["candidateAddend"], candidateAddend);
  graph.connect(vertex["candidateBeamProbNonBlank"], candidateBeamProbNonBlank);
  graph.connect(vertex["candidateBeamProbBlank"], candidateBeamProbBlank);

  graph.connect(vertex["beamAddend"], beamAddend.flatten());
  graph.connect(vertex["beamParent"], beamParent.flatten());
  graph.connect(vertex["beamProbNonBlank"], beamProbNonBlank);
  graph.connect(vertex["beamProbBlank"], beamProbBlank);

  graph.connect(vertex["currentTimestep"], currentTimestep);

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
    candidateParentIn.push_back(candidates[c].beam);
    candidateAddendIn.push_back(candidates[c].addend);
    candidateBeamProbNonBlankIn.push_back(candidates[c].pnb);
    candidateBeamProbBlankIn.push_back(candidates[c].pb);
  }

  copy(target, candidateParentIn, UNSIGNED_INT, rawCandidateParent.get());
  copy(target, candidateAddendIn, UNSIGNED_INT, rawCandidateAddend.get());
  copy(target, candidateBeamProbNonBlankIn, partialsType,
       rawCandidateBeamProbNonBlank.get());
  copy(target, candidateBeamProbBlankIn, partialsType,
       rawCandidateBeamProbBlank.get());

  // InOut
  std::unique_ptr<char[]> rawBeamAddend, rawBeamParent, rawBeamProbNonBlank,
      rawBeamProbBlank;

  rawBeamAddend = allocateHostMemoryForTensor(beamAddend, "beamAddend", graph,
                                              uploadProg, downloadProg, tmap);
  rawBeamParent = allocateHostMemoryForTensor(beamParent, "beamParent", graph,
                                              uploadProg, downloadProg, tmap);
  rawBeamProbNonBlank =
      allocateHostMemoryForTensor(beamProbNonBlank, "beamProbNonBlank", graph,
                                  uploadProg, downloadProg, tmap);
  rawBeamProbBlank = allocateHostMemoryForTensor(
      beamProbBlank, "beamProbBlank", graph, uploadProg, downloadProg, tmap);

  std::vector<unsigned> beamAddendIn{};
  std::vector<unsigned> beamParentIn{};
  std::vector<double> beamProbNonBlankIn{};
  std::vector<double> beamProbBlankIn{};

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

  for (unsigned b = 0; b < beamwidth; b++) {
    beamProbNonBlankIn.push_back(beamProbs[b].pnb);
    beamProbBlankIn.push_back(beamProbs[b].pb);
  }

  copy(target, beamAddendIn, UNSIGNED_INT, rawBeamAddend.get());
  copy(target, beamParentIn, UNSIGNED_INT, rawBeamParent.get());
  copy(target, beamProbNonBlankIn, partialsType, rawBeamProbNonBlank.get());
  copy(target, beamProbBlankIn, partialsType, rawBeamProbBlank.get());

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

  std::vector<unsigned> beamAddendOut(maxT * beamwidth);
  std::vector<unsigned> beamParentOut(maxT * beamwidth);
  // TODO partialsType == float
  std::vector<float> beamProbNonBlankOut(beamwidth);
  std::vector<float> beamProbBlankOut(beamwidth);

  copy(target, UNSIGNED_INT, rawBeamAddend.get(), beamAddendOut);
  copy(target, UNSIGNED_INT, rawBeamParent.get(), beamParentOut);
  copy(target, partialsType, rawBeamProbNonBlank.get(), beamProbNonBlankOut);
  copy(target, partialsType, rawBeamProbBlank.get(), beamProbBlankOut);

  if (profile && deviceType != DeviceType::Cpu) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }

  std::vector<BeamProbability<float>> beamProbOut;
  BeamHistory beamHistoryOut(beamwidth, maxT);
  beamHistoryOut.nextIndexToAssign = timestep + 1;

  for (unsigned b = 0; b < beamwidth; b++) {
    beamProbOut.push_back({beamProbNonBlankOut[b], beamProbBlankOut[b]});
    for (unsigned t = 0; t < maxT; t++) {
      beamHistoryOut.symbols[b][t] = beamAddendOut[b + beamwidth * t];
      beamHistoryOut.parents[b][t] = beamParentOut[b + beamwidth * t];
    }
  }
  return std::make_pair(beamHistoryOut, beamProbOut);
}

template std::pair<BeamHistory, std::vector<BeamProbability<float>>>
runUpdateCodelet(Graph &graph, TestDevice &device, DeviceType deviceType,
                 Type inType, Type partialsType,
                 const std::vector<Candidate<float>> &candidates,

                 unsigned timestep, const BeamHistory &beamHistory,
                 const std::vector<BeamProbability<float>> &beamProbs,
                 unsigned blankClass, bool profile);

} // namespace ctc
} // namespace poplibs_test
