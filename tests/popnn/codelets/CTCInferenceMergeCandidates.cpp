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
    Type partialsType,
    const std::vector<Candidate<PartialsType>> &extendCandidates,
    const Candidate<PartialsType> &copyCandidate, unsigned timestep,
    const BeamHistory &beamHistory, unsigned blankClass, bool profile) {
  const auto target = graph.getTarget();
  const auto beamwidth = beamHistory.symbols.size();
  const auto numExtendCandidates = beamwidth;
  const auto maxT = beamHistory.symbols[0].size();

  auto extendCandidateParent = graph.addVariable(
      UNSIGNED_INT, {numExtendCandidates}, "extendCandidateParent");
  auto extendCandidateAddend = graph.addVariable(
      UNSIGNED_INT, {numExtendCandidates}, "extendCandidateAddend");
  auto extendCandidateBeamProbNonBlank = graph.addVariable(
      partialsType, {numExtendCandidates}, "extendCandidateBeamProbNonBlank");
  auto extendCandidateBeamProbBlank = graph.addVariable(
      partialsType, {numExtendCandidates}, "extendCandidateBeamProbBlank");

  auto copyCandidateParent =
      graph.addVariable(UNSIGNED_INT, {}, "copyCandidateParent");
  auto copyCandidateAddend =
      graph.addVariable(UNSIGNED_INT, {}, "copyCandidateAddend");
  auto copyCandidateBeamProbNonBlank =
      graph.addVariable(partialsType, {}, "copyCandidateBeamProbNonBlank");
  auto copyCandidateBeamProbBlank =
      graph.addVariable(partialsType, {}, "copyCandidateBeamProbBlank");

  auto beamAddend =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamAddend");
  auto beamParent =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamParent");

  auto invalidCandidate =
      graph.addVariable(UNSIGNED_INT, {}, "invalidCandidate");

  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);

  graph.setTileMapping(extendCandidateParent, 0);
  graph.setTileMapping(extendCandidateAddend, 0);
  graph.setTileMapping(extendCandidateBeamProbNonBlank, 0);
  graph.setTileMapping(extendCandidateBeamProbBlank, 0);

  graph.setTileMapping(copyCandidateParent, 0);
  graph.setTileMapping(copyCandidateAddend, 0);
  graph.setTileMapping(copyCandidateBeamProbNonBlank, 0);
  graph.setTileMapping(copyCandidateBeamProbBlank, 0);

  graph.setTileMapping(beamAddend, 0);
  graph.setTileMapping(beamParent, 0);

  graph.setTileMapping(invalidCandidate, 0);
  graph.setTileMapping(currentTimestep, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex =
      graph.addVertex(cs, templateVertex("popnn::CTCMergeCandidates", inType,
                                         partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["extendCandidateParent"], extendCandidateParent);
  graph.connect(vertex["extendCandidateAddend"], extendCandidateAddend);
  graph.connect(vertex["extendCandidateBeamProbNonBlank"],
                extendCandidateBeamProbNonBlank);
  graph.connect(vertex["extendCandidateBeamProbBlank"],
                extendCandidateBeamProbBlank);

  graph.connect(vertex["copyCandidateParent"], copyCandidateParent);
  graph.connect(vertex["copyCandidateAddend"], copyCandidateAddend);
  graph.connect(vertex["copyCandidateBeamProbNonBlank"],
                copyCandidateBeamProbNonBlank);
  graph.connect(vertex["copyCandidateBeamProbBlank"],
                copyCandidateBeamProbBlank);

  graph.connect(vertex["beamAddend"], beamAddend.flatten());
  graph.connect(vertex["beamParent"], beamParent.flatten());

  graph.connect(vertex["invalidCandidate"], invalidCandidate);
  graph.connect(vertex["currentTimestep"], currentTimestep);

  graph.setInitialValue(vertex["extendCandidates"], numExtendCandidates);
  graph.setInitialValue(vertex["blankClass"], blankClass);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  // InOut
  std::unique_ptr<char[]> rawCopyCandidateBeamProbNonBlank,
      rawCopyCandidateBeamProbBlank;

  rawCopyCandidateBeamProbNonBlank = allocateHostMemoryForTensor(
      copyCandidateBeamProbNonBlank, "copyCandidateBeamProbNonBlank", graph,
      uploadProg, downloadProg, tmap);
  rawCopyCandidateBeamProbBlank = allocateHostMemoryForTensor(
      copyCandidateBeamProbBlank, "copyCandidateBeamProbBlank", graph,
      uploadProg, downloadProg, tmap);

  // Inputs
  std::unique_ptr<char[]> rawExtendCandidateParent, rawExtendCandidateAddend,
      rawCopyCandidateParent, rawCopyCandidateAddend,
      rawExtendCandidateBeamProbNonBlank, rawExtendCandidateBeamProbBlank;

  rawCopyCandidateParent =
      allocateHostMemoryForTensor(copyCandidateParent, "copyCandidateParent",
                                  graph, uploadProg, downloadProg, tmap);
  rawCopyCandidateAddend =
      allocateHostMemoryForTensor(copyCandidateAddend, "copyCandidateAddend",
                                  graph, uploadProg, downloadProg, tmap);
  rawExtendCandidateParent = allocateHostMemoryForTensor(
      extendCandidateParent, "extendCandidateParent", graph, uploadProg,
      downloadProg, tmap);
  rawExtendCandidateAddend = allocateHostMemoryForTensor(
      extendCandidateAddend, "extendCandidateAddend", graph, uploadProg,
      downloadProg, tmap);
  rawExtendCandidateBeamProbNonBlank = allocateHostMemoryForTensor(
      extendCandidateBeamProbNonBlank, "extendCandidateBeamProbNonBlank", graph,
      uploadProg, downloadProg, tmap);
  rawExtendCandidateBeamProbBlank = allocateHostMemoryForTensor(
      extendCandidateBeamProbBlank, "extendCandidateBeamProbBlank", graph,
      uploadProg, downloadProg, tmap);

  // Extend candidate inputs
  std::vector<unsigned> extendCandidateParentIn{};
  std::vector<unsigned> extendCandidateAddendIn{};
  std::vector<float> extendCandidateBeamProbNonBlankIn{};
  std::vector<float> extendCandidateBeamProbBlankIn{};

  for (unsigned c = 0; c < numExtendCandidates; c++) {
    extendCandidateParentIn.push_back(extendCandidates[c].beam);
    extendCandidateAddendIn.push_back(extendCandidates[c].addend);
    extendCandidateBeamProbNonBlankIn.push_back(extendCandidates[c].pnb);
    extendCandidateBeamProbBlankIn.push_back(extendCandidates[c].pb);
  }

  copy(target, extendCandidateParentIn, UNSIGNED_INT,
       rawExtendCandidateParent.get());
  copy(target, extendCandidateAddendIn, UNSIGNED_INT,
       rawExtendCandidateAddend.get());
  copy(target, extendCandidateBeamProbNonBlankIn, partialsType,
       rawExtendCandidateBeamProbNonBlank.get());
  copy(target, extendCandidateBeamProbBlankIn, partialsType,
       rawExtendCandidateBeamProbBlank.get());

  // Copy candidate inputs (A single candidate)
  std::vector<unsigned> copyCandidateParentIn = {copyCandidate.beam};
  std::vector<unsigned> copyCandidateAddendIn = {copyCandidate.addend};
  std::vector<float> copyCandidateBeamProbNonBlankIn = {copyCandidate.pnb};
  std::vector<float> copyCandidateBeamProbBlankIn = {copyCandidate.pb};

  copy(target, copyCandidateParentIn, UNSIGNED_INT,
       rawCopyCandidateParent.get());
  copy(target, copyCandidateAddendIn, UNSIGNED_INT,
       rawCopyCandidateAddend.get());
  copy(target, copyCandidateBeamProbNonBlankIn, partialsType,
       rawCopyCandidateBeamProbNonBlank.get());
  copy(target, copyCandidateBeamProbBlankIn, partialsType,
       rawCopyCandidateBeamProbBlank.get());

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

  // Out only
  std::unique_ptr<char[]> rawInvalidCandidateOut;
  rawInvalidCandidateOut =
      allocateHostMemoryForTensor(invalidCandidate, "invalidCandidate", graph,
                                  uploadProg, downloadProg, tmap);

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

  std::vector<unsigned> invalidCandidateOut(1);

  // TODO partialsType == float
  std::vector<float> copyCandidateBeamProbBlankOut(1);
  std::vector<float> copyCandidateBeamProbNonBlankOut(1);

  // This is all that should have changed - either a merge happened or it didn't
  // and if so the copy candidate is updated with probabilities.
  // The vector of extend beams can be shared between multiple vertices so is
  // not changed and `invalidCandidate` indicates which was merged into the
  // copy.
  copy(target, UNSIGNED_INT, rawInvalidCandidateOut.get(), invalidCandidateOut);
  copy(target, partialsType, rawCopyCandidateBeamProbNonBlank.get(),
       copyCandidateBeamProbNonBlankOut);
  copy(target, partialsType, rawCopyCandidateBeamProbBlank.get(),
       copyCandidateBeamProbBlankOut);

  if (profile && deviceType != DeviceType::Cpu) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  // Return a vector of candidates for comparison: The copy candidate first
  // and the the extend candidates, omitting the one that was merged if a merge
  // happened
  std::vector<Candidate<float>> mergedCandidates;
  mergedCandidates.push_back(
      {copyCandidateParentIn[0], copyCandidateAddendIn[0],
       copyCandidateBeamProbNonBlankOut[0], copyCandidateBeamProbBlankOut[0]});
  for (unsigned i = 0; i < numExtendCandidates; i++) {
    if (i != invalidCandidateOut[0]) {
      mergedCandidates.push_back({extendCandidateParentIn[i],
                                  extendCandidateAddendIn[i],
                                  extendCandidateBeamProbNonBlankIn[i],
                                  extendCandidateBeamProbBlankIn[i]});
    }
  }
  return mergedCandidates;
}

template std::vector<Candidate<float>> runMergeCandidatesCodelet(
    Graph &graph, TestDevice &device, DeviceType deviceType, Type inType,
    Type partialsType, const std::vector<Candidate<float>> &extendCandidates,
    const Candidate<float> &copyCandidate, unsigned timestep,
    const BeamHistory &beamHistory, unsigned blankClass, bool profile);

} // namespace ctc
} // namespace poplibs_test
