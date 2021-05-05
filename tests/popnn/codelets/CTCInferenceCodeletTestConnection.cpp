// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CTCInferenceCodeletTestConnection.hpp"

#include <poplibs_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

namespace poplibs_test {
namespace ctc {

CandidateHandles createAndConnectCandidates(
    poplar::Graph &graph, const poplar::VertexRef &vertex,
    const std::string &prefix, const Type &partialsType,
    const ArrayRef<std::size_t> &shape, Sequence &uploadProg,
    Sequence &downloadProg, std::vector<std::pair<std::string, char *>> &tmap,
    bool includeTotal) {

  auto candidateParent =
      graph.addVariable(UNSIGNED_INT, shape, "candidateParent");
  auto candidateAddend =
      graph.addVariable(UNSIGNED_INT, shape, "candidateAddend");
  auto candidateBeamProbNonBlank =
      graph.addVariable(partialsType, shape, "candidateBeamProbNonBlank");
  auto candidateBeamProbBlank =
      graph.addVariable(partialsType, shape, "candidateBeamProbBlank");

  graph.setTileMapping(candidateParent, 0);
  graph.setTileMapping(candidateAddend, 0);
  graph.setTileMapping(candidateBeamProbNonBlank, 0);
  graph.setTileMapping(candidateBeamProbBlank, 0);

  graph.connect(vertex[prefix + "Parent"], candidateParent);
  graph.connect(vertex[prefix + "Addend"], candidateAddend);
  graph.connect(vertex[prefix + "BeamProbNonBlank"], candidateBeamProbNonBlank);
  graph.connect(vertex[prefix + "BeamProbBlank"], candidateBeamProbBlank);

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
  handles.probBlank =
      allocateHostMemoryForTensor(candidateBeamProbBlank, prefix + "ProbBlank",
                                  graph, uploadProg, downloadProg, tmap);

  if (includeTotal) {
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

} // namespace ctc
} // namespace poplibs_test
