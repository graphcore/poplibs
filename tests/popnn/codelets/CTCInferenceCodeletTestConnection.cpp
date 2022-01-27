// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceCodeletTestConnection.hpp"

#include <poplibs_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;

namespace poplibs_test {
namespace ctc {

CandidateHandles
createAndConnectCandidates(Graph &graph, const VertexRef &vertex,
                           const std::string &prefix, const Type &partialsType,
                           const ArrayRef<std::size_t> &shape,
                           Sequence &uploadProg, Sequence &downloadProg,
                           std::vector<std::pair<std::string, char *>> &tmap,
                           bool includeTotalAndBlank) {

  auto candidateParent =
      graph.addVariable(UNSIGNED_INT, shape, "candidateParent");
  auto candidateAddend =
      graph.addVariable(UNSIGNED_INT, shape, "candidateAddend");
  auto candidateBeamProbNonBlank =
      graph.addVariable(partialsType, shape, "candidateBeamProbNonBlank");

  graph.setTileMapping(candidateParent, 0);
  graph.setTileMapping(candidateAddend, 0);
  graph.setTileMapping(candidateBeamProbNonBlank, 0);

  graph.connect(vertex[prefix + "Parent"], candidateParent);
  graph.connect(vertex[prefix + "Addend"], candidateAddend);
  graph.connect(vertex[prefix + "BeamProbNonBlank"], candidateBeamProbNonBlank);

  CandidateHandles handles;
  handles.parent =
      allocateHostMemoryForTensor(candidateParent, prefix + "Parent", graph,
                                  uploadProg, downloadProg, tmap);
  handles.addend =
      allocateHostMemoryForTensor(candidateAddend, prefix + "Addend", graph,
                                  uploadProg, downloadProg, tmap);
  handles.probNonBlank = allocateHostMemoryForTensor(
      candidateBeamProbNonBlank, prefix + "ProbNonBlank", graph, uploadProg,
      downloadProg, tmap);

  if (includeTotalAndBlank) {
    auto candidateBeamProbBlank =
        graph.addVariable(partialsType, shape, "candidateBeamProbBlank");
    graph.setTileMapping(candidateBeamProbBlank, 0);
    graph.connect(vertex[prefix + "BeamProbBlank"], candidateBeamProbBlank);
    handles.probBlank = allocateHostMemoryForTensor(
        candidateBeamProbBlank, prefix + "ProbBlank", graph, uploadProg,
        downloadProg, tmap);

    auto candidateBeamProbTotal =
        graph.addVariable(partialsType, shape, "candidateBeamProbTotal");
    graph.setTileMapping(candidateBeamProbTotal, 0);
    graph.connect(vertex[prefix + "BeamProbTotal"], candidateBeamProbTotal);
    handles.probTotal = allocateHostMemoryForTensor(
        candidateBeamProbTotal, prefix + "BeamProbTotal", graph, uploadProg,
        downloadProg, tmap);
  }
  return handles;
}

BeamHandles createAndConnectBeamProbs(
    Graph &graph, const VertexRef &vertex, const Type &probsType,
    const ArrayRef<std::size_t> &shape, BeamScalars selectBlank,
    Sequence &uploadProg, Sequence &downloadProg,
    std::vector<std::pair<std::string, char *>> &tmap) {
  auto lastBeamOutputs =
      graph.addVariable(UNSIGNED_INT, shape, "lastBeamOutputs");
  auto beamProbNonBlank =
      graph.addVariable(probsType, shape, "beamProbNonBlank");
  auto beamProbBlank = graph.addVariable(probsType, shape, "beamProbBlank");
  auto beamProbTotal = graph.addVariable(probsType, shape, "beamProbTotal");

  graph.setTileMapping(lastBeamOutputs, 0);
  graph.setTileMapping(beamProbNonBlank, 0);
  graph.setTileMapping(beamProbBlank, 0);
  graph.setTileMapping(beamProbTotal, 0);

  graph.connect(vertex["lastBeamOutputs"], lastBeamOutputs);
  if (selectBlank == BeamScalars::NON_BLANK ||
      selectBlank == BeamScalars::BLANK_AND_NON_BLANK) {
    graph.connect(vertex["beamProbNonBlank"], beamProbNonBlank);
  }
  if (selectBlank == BeamScalars::BLANK ||
      selectBlank == BeamScalars::BLANK_AND_NON_BLANK) {
    graph.connect(vertex["beamProbBlank"], beamProbBlank);
  }
  graph.connect(vertex["beamProbTotal"], beamProbTotal);

  BeamHandles handles;
  handles.lastOutput =
      allocateHostMemoryForTensor(lastBeamOutputs, "lastBeamOutputs", graph,
                                  uploadProg, downloadProg, tmap);
  handles.pnb =
      allocateHostMemoryForTensor(beamProbNonBlank, "beamProbNonBlank", graph,
                                  uploadProg, downloadProg, tmap);
  handles.pb = allocateHostMemoryForTensor(
      beamProbBlank, "beamProbBlank", graph, uploadProg, downloadProg, tmap);
  handles.pTotal = allocateHostMemoryForTensor(
      beamProbTotal, "beamProbTotal", graph, uploadProg, downloadProg, tmap);

  return handles;
}

} // namespace ctc
} // namespace poplibs_test
