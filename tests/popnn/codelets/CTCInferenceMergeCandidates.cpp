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
#include <optional>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::ctc;
using namespace poplibs_test;
using namespace poplibs_test::util;
using namespace poplar_test;
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
    unsigned blankClass, const BeamHistory &beamHistory,
    const ArrayRef<unsigned> &outputLengths, unsigned lastBeamOutputSym,
    unsigned numClasses, bool profile) {

  const auto target = graph.getTarget();
  const auto beamwidth = beamHistory.symbols.size();
  const auto numExtendCandidates = extendCandidates.size();
  const auto maxT = beamHistory.symbols[0].size() - 1;

  auto beamAddend =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamAddend");
  auto beamParent =
      graph.addVariable(UNSIGNED_INT, {maxT, beamwidth}, "beamParent");
  auto lastBeamOutput =
      graph.addConstant(UNSIGNED_INT, {}, lastBeamOutputSym, "lastBeamOutput");

  auto beamLength = graph.addConstant(UNSIGNED_INT, {2 * beamwidth},
                                      outputLengths, "beamLength");

  auto currentTimestep = graph.addConstant(UNSIGNED_INT, {}, timestep);
  auto complete = graph.addConstant(UNSIGNED_INT, {}, 0u);

  graph.setTileMapping(beamAddend, 0);
  graph.setTileMapping(beamParent, 0);
  graph.setTileMapping(lastBeamOutput, 0);
  graph.setTileMapping(beamLength, 0);

  graph.setTileMapping(currentTimestep, 0);
  graph.setTileMapping(complete, 0);

  auto cs = graph.addComputeSet("cs");
  auto vertex = graph.addVertex(cs, templateVertex("popnn::CTCMergeCandidates",
                                                   partialsType, UNSIGNED_INT));
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["beamAddend"], beamAddend.flatten());
  graph.connect(vertex["beamParent"], beamParent.flatten());
  graph.connect(vertex["lastBeamOutput"], lastBeamOutput);
  graph.connect(vertex["beamLength"], beamLength);

  graph.connect(vertex["currentTimestep"], currentTimestep);
  graph.connect(vertex["complete"], complete);

  graph.setInitialValue(vertex["beamwidth"], beamwidth);
  graph.setInitialValue(vertex["blankClass"], blankClass);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  // InOut
  auto rawCopyCandidates =
      createAndConnectCandidates(graph, vertex, "copyCandidate", partialsType,
                                 {}, uploadProg, downloadProg, tmap);

  // Inputs
  auto rawExtendCandidates = createAndConnectCandidates(
      graph, vertex, "extendCandidate", partialsType, {numExtendCandidates},
      uploadProg, downloadProg, tmap, false);

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
       rawExtendCandidates.parent.get());
  copy(target, extendCandidateAddendIn, UNSIGNED_INT,
       rawExtendCandidates.addend.get());
  copy(target, extendCandidateBeamProbNonBlankIn, partialsType,
       rawExtendCandidates.probNonBlank.get());

  // Copy candidate inputs (A single candidate)
  std::vector<unsigned> copyCandidateParentIn = {copyCandidate.beam};
  std::vector<unsigned> copyCandidateAddendIn = {copyCandidate.addend};
  std::vector<float> copyCandidateBeamProbNonBlankIn = {copyCandidate.pnb};
  std::vector<float> copyCandidateBeamProbBlankIn = {copyCandidate.pb};
  std::vector<float> copyCandidateBeamProbTotalIn = {copyCandidate.pTotal};

  copy(target, copyCandidateParentIn, UNSIGNED_INT,
       rawCopyCandidates.parent.get());
  copy(target, copyCandidateAddendIn, UNSIGNED_INT,
       rawCopyCandidates.addend.get());
  copy(target, copyCandidateBeamProbNonBlankIn, partialsType,
       rawCopyCandidates.probNonBlank.get());
  copy(target, copyCandidateBeamProbBlankIn, partialsType,
       rawCopyCandidates.probBlank.get());
  copy(target, copyCandidateBeamProbTotalIn, partialsType,
       rawCopyCandidates.probTotal.get().get());

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

  // TODO partialsType == float
  std::vector<float> copyCandidateBeamProbBlankOut(1);
  std::vector<float> copyCandidateBeamProbNonBlankOut(1);
  std::vector<float> copyCandidateBeamProbTotalOut(1);
  std::vector<unsigned> copyCandidateAddendOut(1);

  // This is all that should have changed - either a merge happened or it didn't
  // and if so the copy candidate is updated with probabilities.
  // The vector of extend beams can be shared between multiple vertices so is
  // not changed and the copy candidate addend indicates which was merged into
  // the copy.
  copy(target, partialsType, rawCopyCandidates.probNonBlank.get(),
       copyCandidateBeamProbNonBlankOut);
  copy(target, partialsType, rawCopyCandidates.probBlank.get(),
       copyCandidateBeamProbBlankOut);
  copy(target, partialsType, rawCopyCandidates.probTotal.get().get(),
       copyCandidateBeamProbTotalOut);
  copy(target, UNSIGNED_INT, rawCopyCandidates.addend.get(),
       copyCandidateAddendOut);

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
       copyCandidateBeamProbNonBlankOut[0], copyCandidateBeamProbBlankOut[0],
       copyCandidateBeamProbTotalOut[0]});
  for (unsigned i = 0; i < numExtendCandidates; i++) {
    if (extendCandidateAddendIn[i] != copyCandidateAddendOut[0]) {
      mergedCandidates.push_back(
          {extendCandidateParentIn[i], extendCandidateAddendIn[i],
           extendCandidateBeamProbNonBlankIn[i],
           extendCandidateBeamProbBlankIn[i], extendCandidates[i].pTotal});
    }
  }
  return mergedCandidates;
}

template std::vector<Candidate<float>> runMergeCandidatesCodelet(
    Graph &graph, TestDevice &device, DeviceType deviceType, Type inType,
    Type partialsType, const std::vector<Candidate<float>> &extendCandidates,
    const Candidate<float> &copyCandidate, unsigned timestep,
    unsigned blankClass, const BeamHistory &beamHistory,
    const ArrayRef<unsigned> &outputLengths, unsigned lastBeamOutputSym,
    unsigned numClasses, bool profile);

} // namespace ctc
} // namespace poplibs_test
