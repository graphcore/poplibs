// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceMergeCandidates.hpp"

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
std::vector<Candidate<PartialsType>> runMergeCandidatesCodelet(
    Graph &graph, TestDevice &device, DeviceType deviceType, Type inType,
    Type partialsType, const std::vector<Candidate<PartialsType>> &candidates,
    unsigned timestep, const BeamHistory &beamHistory, unsigned blankClass,
    bool profile) {
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

  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);

  graph.setTileMapping(candidateParent, 0);
  graph.setTileMapping(candidateAddend, 0);
  graph.setTileMapping(candidateBeamProbNonBlank, 0);
  graph.setTileMapping(candidateBeamProbBlank, 0);

  graph.setTileMapping(beamAddend, 0);
  graph.setTileMapping(beamParent, 0);

  graph.setTileMapping(currentTimestep, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex =
      graph.addVertex(cs, templateVertex("popnn::MergeCandidates", inType,
                                         partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["candidateParent"], candidateParent);
  graph.connect(vertex["candidateAddend"], candidateAddend);
  graph.connect(vertex["candidateBeamProbNonBlank"], candidateBeamProbNonBlank);
  graph.connect(vertex["candidateBeamProbBlank"], candidateBeamProbBlank);

  graph.connect(vertex["beamAddend"], beamAddend.flatten());
  graph.connect(vertex["beamParent"], beamParent.flatten());

  graph.connect(vertex["currentTimestep"], currentTimestep);

  graph.setInitialValue(vertex["totalCandidates"], totalCandidates);
  graph.setInitialValue(vertex["blankClass"], blankClass);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  // InOut
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

  std::vector<Candidate<float>> mergedCandidates;
  for (unsigned i = 0; i < totalCandidates; i++) {
    mergedCandidates.push_back({candidateParentOut[i], candidateAddendOut[i],
                                candidateBeamProbNonBlankOut[i],
                                candidateBeamProbBlankOut[i]});
  }
  return mergedCandidates;
}

template std::vector<Candidate<float>>
runMergeCandidatesCodelet(Graph &graph, TestDevice &device,
                          DeviceType deviceType, Type inType, Type partialsType,
                          const std::vector<Candidate<float>> &candidates,
                          unsigned timestep, const BeamHistory &beamHistory,
                          unsigned blankClass, bool profile);

} // namespace ctc
} // namespace poplibs_test
