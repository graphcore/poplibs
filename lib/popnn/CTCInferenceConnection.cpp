// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCInferenceConnection.hpp"

#include "CTCInferencePlan.hpp"
#include "CTCPlanInternal.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/Tracepoint.hpp>
#include <poplibs_support/logging.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <boost/optional.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poputil;

template <unsigned size> using Slice = std::array<std::size_t, size>;

namespace {

using TempTensors = popnn::ctc_infer::TempTensors;
using BeamTensors = popnn::ctc_infer::BeamTensors;

void attachBeamScalars(Graph &graph, const BeamTensors &beams, unsigned batch,
                       unsigned partition, unsigned beamwidth,
                       const VertexRef &vertex) {

  Slice<4> begin = {batch, partition, 0, 0};
  Slice<4> end = {batch + 1, partition + 1, beamwidth, 1};
  graph.connect(vertex["beamProbBlank"], beams.pb.slice(begin, end).flatten());
  graph.connect(vertex["beamProbNonBlank"],
                beams.pnb.slice(begin, end).flatten());
  graph.connect(vertex["lastBeamOutputs"],
                beams.lastOutput.slice(begin, end).flatten());
}

void attachBeamHistory(Graph &graph, const BeamTensors &beams,
                       const Interval &time, unsigned batch, unsigned partition,
                       unsigned beamwidth, const VertexRef &vertex) {

  Slice<4> begin = {batch, partition, time.begin(), 0};
  Slice<4> end = {batch + 1, partition + 1, time.end(), beamwidth};
  graph.connect(vertex["beamAddend"], beams.addend.slice(begin, end).flatten());
  graph.connect(vertex["beamParent"], beams.parent.slice(begin, end).flatten());
}

void attachExtendCandidates(Graph &graph, const TempTensors &tempTensors,
                            unsigned batch, unsigned partition,
                            unsigned beamwidth, const VertexRef &vertex) {

  Slice<3> begin = {batch, partition, 0};
  Slice<3> end = {batch + 1, partition + 1, beamwidth};
  graph.connect(vertex["extendCandidateParent"],
                tempTensors.extendCandidatesParent.slice(begin, end).flatten());
  graph.connect(vertex["extendCandidateAddend"],
                tempTensors.extendCandidatesAddend.slice(begin, end).flatten());
  graph.connect(vertex["extendCandidateBeamProbNonBlank"],
                tempTensors.extendCandidatesPnb.slice(begin, end).flatten());
  graph.connect(vertex["extendCandidateBeamProbBlank"],
                tempTensors.extendCandidatesPb.slice(begin, end).flatten());
}

void attachData(Graph &graph, const Tensor &data, unsigned batch,
                unsigned partition, unsigned numClasses, const Interval &time,
                const VertexRef &vertex) {
  Slice<4> begin = {batch, partition, time.begin(), 0};
  Slice<4> end = {batch + 1, partition + 1, time.end(), numClasses};
  graph.connect(vertex["logProbs"], data.slice(begin, end).flatten());
}

void attachTimeAndLength(Graph &graph, const TempTensors &tempTensors,
                         unsigned batch, unsigned partition,
                         const VertexRef &vertex) {
  graph.connect(vertex["currentTimestep"],
                tempTensors.currentTimestep[batch][partition][0]);
  graph.connect(vertex["dataLength"],
                tempTensors.dataLengths[batch][partition][0]);
}

std::vector<Tensor> gatherMergeSlices(const std::vector<Tensor> &input,
                                      unsigned batch) {
  // Gather slices of the merge candidate tensors to attach to the Select and
  // Update vertices.  The order is important for the Select vertex, and only
  // the specific beamwidth elements are needed for the Update vertex
  // This step gathers the copy candidates in the order:
  // copy[addend0,parent0]
  // copy[addend0,parent1]...
  // copy[addend1,parent0]
  // copy[addend1,parent1]...
  // ... all addends
  // The extend candidates are concatenated with these later as required
  //
  // TODO - this should become simpler, or need to change when simplifying the
  // way that the Select vertex figures out which candidates were merged
  std::vector<Tensor> output(input.size() * input[0].dim(1));
  for (unsigned i = 0; i < input[0].dim(1); i++) {
    Slice<3> candidateBegin = {batch, i, 0};
    Slice<3> candidateEnd = {batch + 1, i + 1, 1};
    for (unsigned j = 0; j < input.size(); j++) {
      output[i * input.size() + j] =
          input[j].slice(candidateBegin, candidateEnd).flatten();
    }
  }
  return output;
}

std::vector<Tensor> gatherBeamsSlices(const std::vector<Tensor> &input,
                                      unsigned batch, unsigned size,
                                      unsigned offset) {
  // TODO - This only needs the selected beamwidth candidates, but the
  // ordering of slices is that of the select vertex.  Using the same function
  // to extract the slices makes sense but having to resize less so.
  // Expect to review ordering when looking at a better way to extract the
  // candidates to sort, so TODO - later.
  auto result = gatherMergeSlices(input, batch);
  std::vector<Tensor> resultSliced(result.begin() + offset,
                                   result.begin() + offset + size);
  return resultSliced;
}

} // namespace
namespace popnn {
namespace ctc_infer {

void generateExtendCandidateVertex(
    Graph &graph, const Tensor &data, const BeamTensors &beams,
    const TempTensors &tempTensors, ComputeSet &cs, unsigned batch,
    const Interval &time, unsigned addendPartition, unsigned blankClass,
    unsigned beamwidth, unsigned addendClass, unsigned tile) {

  const auto partialsType = beams.pb.elementType();
  const auto vertexName =
      templateVertex("popnn::CTCGenerateExtendCandidates", data.elementType(),
                     partialsType, UNSIGNED_INT);
  const auto vertex = graph.addVertex(cs, vertexName);
  logging::popnn::trace("Making {} vertex for symbol {} on tile {}", vertexName,
                        addendClass, tile);
  graph.setTileMapping(vertex, tile);

  // Data connection
  const auto numClasses = data.dim(3);
  attachData(graph, data, batch, addendPartition, numClasses, time, vertex);
  // Beam connection
  attachBeamScalars(graph, beams, batch, addendPartition, beamwidth, vertex);
  // Timestep, data length connection
  attachTimeAndLength(graph, tempTensors, batch, addendPartition, vertex);
  // Extend candidate connection
  attachExtendCandidates(graph, tempTensors, batch, addendPartition, beamwidth,
                         vertex);
  // Constants
  graph.setInitialValue(vertex["numClassesIncBlank"], numClasses);
  graph.setInitialValue(vertex["blankClass"], blankClass);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);
  graph.setInitialValue(vertex["addendSymbol"], addendClass);
}

void generateCopyCandidateVertex(Graph &graph, const Tensor &data,
                                 const BeamTensors &beams,
                                 const TempTensors &tempTensors, ComputeSet &cs,
                                 unsigned batch, const Interval &time,
                                 unsigned beamPartition, unsigned blankClass,
                                 unsigned beamwidth, unsigned tile) {

  const auto partialsType = beams.pb.elementType();
  const auto vertexName =
      templateVertex("popnn::CTCGenerateCopyCandidates", data.elementType(),
                     partialsType, UNSIGNED_INT);
  const auto vertex = graph.addVertex(cs, vertexName);
  logging::popnn::trace("Making {} vertex on tile {}", vertexName, tile);
  graph.setTileMapping(vertex, tile);

  // Data connection
  const auto numClasses = data.dim(3);
  attachData(graph, data, batch, beamPartition, numClasses, time, vertex);
  // Beam connection
  attachBeamScalars(graph, beams, batch, beamPartition, beamwidth, vertex);
  // Timestep, data length connection
  attachTimeAndLength(graph, tempTensors, batch, beamPartition, vertex);
  // Copy candidate connection
  auto transform = [=](const Tensor &in) {
    Slice<3> copyBegin = {batch, beamPartition, 0};
    Slice<3> copyEnd = {batch + 1, beamPartition + 1, 1};
    return in.slice(copyBegin, copyEnd).reshape({});
  };
  graph.connect(vertex["candidateParent"],
                transform(tempTensors.copyCandidatesParent));
  graph.connect(vertex["candidateAddend"],
                transform(tempTensors.copyCandidatesAddend));
  graph.connect(vertex["candidateBeamProbNonBlank"],
                transform(tempTensors.copyCandidatesPnb));
  graph.connect(vertex["candidateBeamProbBlank"],
                transform(tempTensors.copyCandidatesPb));

  // Constants
  graph.setInitialValue(vertex["numClassesIncBlank"], numClasses);
  graph.setInitialValue(vertex["blankClass"], blankClass);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);
  graph.setInitialValue(vertex["beamIdx"], beamPartition);
}

void mergeCandidateVertex(Graph &graph, const BeamTensors &beams,
                          const TempTensors &tempTensors, ComputeSet &cs,
                          unsigned batch, const Interval &time,
                          unsigned addendPartition, unsigned beamPartition,
                          unsigned beamwidth, unsigned tile) {

  const auto partialsType = beams.pb.elementType();

  const auto vertexName =
      templateVertex("popnn::CTCMergeCandidates", partialsType, UNSIGNED_INT);
  const auto vertex = graph.addVertex(cs, vertexName);
  logging::popnn::trace("Making {} vertex for copy {}, addend {} on tile {}",
                        vertexName, beamPartition, addendPartition, tile);
  graph.setTileMapping(vertex, tile);

  // Extend candidate connection
  attachExtendCandidates(graph, tempTensors, batch, addendPartition, beamwidth,
                         vertex);

  // Merge candidate connection (broadcast copy candidates)
  auto transform = [=](const std::vector<Tensor> &in) {
    Slice<3> mergeBegin = {batch, addendPartition, 0};
    Slice<3> mergeEnd = {batch + 1, addendPartition + 1, 1};
    return in[beamPartition].slice(mergeBegin, mergeEnd).reshape({});
  };
  graph.connect(vertex["copyCandidateParent"],
                transform(tempTensors.mergeCandidatesParent));
  graph.connect(vertex["copyCandidateAddend"],
                transform(tempTensors.mergeCandidatesAddend));
  graph.connect(vertex["copyCandidateBeamProbNonBlank"],
                transform(tempTensors.mergeCandidatesPnb));
  graph.connect(vertex["copyCandidateBeamProbBlank"],
                transform(tempTensors.mergeCandidatesPb));

  // Beam history connection
  attachBeamHistory(graph, beams, time, batch, addendPartition, beamwidth,
                    vertex);
  // Timestep, data length connection
  attachTimeAndLength(graph, tempTensors, batch, addendPartition, vertex);
  // Invalid (merged) candidate indication connection
  Tensor mergedCandidateIndicator =
      tempTensors.mergedCandidateIndicator[beamPartition];
  graph.connect(
      vertex["invalidCandidate"],
      mergedCandidateIndicator[batch][addendPartition][0].reshape({}));

  // Constants
  const auto extendCandidates = tempTensors.extendCandidatesParent.dim(2);
  graph.setInitialValue(vertex["extendCandidates"], extendCandidates);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);
}

void selectCandidatesVertex(Graph &graph, const Tensor &scratch,
                            const TempTensors &tempTensors, ComputeSet &cs,
                            unsigned batch, unsigned partition,
                            unsigned candidatesPerMerge,
                            unsigned candidatesToCompare, unsigned beamwidth,
                            unsigned tile) {

  const auto partialsType = tempTensors.mergeCandidatesPb[0].elementType();
  const auto vertexName =
      templateVertex("popnn::CTCSelectCandidates", partialsType, UNSIGNED_INT);
  const auto vertex = graph.addVertex(cs, vertexName);
  logging::popnn::trace("Making {} vertex on tile {}", vertexName, tile);
  graph.setTileMapping(vertex, tile);

  // Connect candidates, the vertex needs correctly ordered slices of the
  // merged (broadcast copy candidates) candidates...
  auto parents = gatherMergeSlices(tempTensors.mergeCandidatesParent, batch);
  auto addends = gatherMergeSlices(tempTensors.mergeCandidatesAddend, batch);
  auto pnb = gatherMergeSlices(tempTensors.mergeCandidatesPnb, batch);
  auto pb = gatherMergeSlices(tempTensors.mergeCandidatesPb, batch);
  // ... followed by the extend candidates
  parents.push_back(tempTensors.extendCandidatesParent[batch].flatten());
  addends.push_back(tempTensors.extendCandidatesAddend[batch].flatten());
  pnb.push_back(tempTensors.extendCandidatesPnb[batch].flatten());
  pb.push_back(tempTensors.extendCandidatesPb[batch].flatten());
  graph.connect(vertex["candidateParent"], concat(parents));
  graph.connect(vertex["candidateAddend"], concat(addends));
  graph.connect(vertex["candidateBeamProbNonBlank"], concat(pnb));
  graph.connect(vertex["candidateBeamProbBlank"], concat(pb));

  // Invalid (merged) candidate indication connection for each copy comparison
  const auto mergeIndicator =
      gatherMergeSlices(tempTensors.mergedCandidateIndicator, batch);
  graph.connect(vertex["mergedCandidateIndicator"], concat(mergeIndicator));
  // Timestep, data length connection (Only for early end)
  attachTimeAndLength(graph, tempTensors, batch, partition, vertex);
  // Scratch
  graph.connect(vertex["candidateProbTotalScratch"], scratch);

  // Constants
  graph.setInitialValue(vertex["beamwidth"], beamwidth);
  graph.setInitialValue(vertex["totalCandidates"], candidatesToCompare);
  graph.setInitialValue(vertex["extendCandidateGroups"], candidatesPerMerge);
}

void updateVertex(Graph &graph, const Tensor &scratch, const BeamTensors &beams,
                  const TempTensors &tempTensors, ComputeSet &cs,
                  unsigned batch, const Interval &time, unsigned beamPartition,
                  unsigned sortedResultOffset, unsigned beamwidth,
                  unsigned tile) {

  const auto partialsType = tempTensors.mergeCandidatesPb[0].elementType();
  const auto vertexName =
      templateVertex("popnn::CTCUpdate", partialsType, UNSIGNED_INT);
  const auto vertex = graph.addVertex(cs, vertexName);
  logging::popnn::trace("Making {} vertex on tile {}", vertexName, tile);
  graph.setTileMapping(vertex, tile);

  // Beam connection
  attachBeamScalars(graph, beams, batch, beamPartition, beamwidth, vertex);
  attachBeamHistory(graph, beams, time, batch, beamPartition, beamwidth,
                    vertex);
  // Scratch
  graph.connect(vertex["lastBeamOutputsScratch"], scratch);
  // Timestep, data length connections
  attachTimeAndLength(graph, tempTensors, batch, beamPartition, vertex);

  // Candidate connection
  const auto parents = gatherBeamsSlices(tempTensors.mergeCandidatesParent,
                                         batch, beamwidth, sortedResultOffset);
  const auto addends = gatherBeamsSlices(tempTensors.mergeCandidatesAddend,
                                         batch, beamwidth, sortedResultOffset);
  const auto pnb = gatherBeamsSlices(tempTensors.mergeCandidatesPnb, batch,
                                     beamwidth, sortedResultOffset);
  const auto pb = gatherBeamsSlices(tempTensors.mergeCandidatesPb, batch,
                                    beamwidth, sortedResultOffset);

  graph.connect(vertex["candidateParent"], concat(parents));
  graph.connect(vertex["candidateAddend"], concat(addends));
  graph.connect(vertex["candidateBeamProbNonBlank"], concat(pnb));
  graph.connect(vertex["candidateBeamProbBlank"], concat(pb));

  // Constants
  graph.setInitialValue(vertex["beamwidth"], beamwidth);
}

void generateOutputVertex(Graph &graph, const BeamTensors &beams,
                          const TempTensors &tempTensors, const Tensor &labels,
                          const Tensor &labelLengths, ComputeSet &cs,
                          unsigned batch, unsigned beamwidth,
                          unsigned partition, unsigned path, unsigned tile) {

  const auto maxT = beams.parent.dim(2);

  const auto vertexName =
      templateVertex("popnn::CTCGenerateOutput", UNSIGNED_INT);
  const auto vertex = graph.addVertex(cs, vertexName);
  logging::popnn::trace("Making {} vertex on tile {}", vertexName, tile);
  graph.setTileMapping(vertex, tile);

  graph.connect(
      vertex["beamOutput"],
      labels.slice({batch, path, 0}, {batch + 1, path + 1, maxT}).flatten());
  graph.connect(
      vertex["outputLength"],
      labelLengths.slice({batch, path}, {batch + 1, path + 1}).reshape({}));

  attachBeamHistory(graph, beams, {0, maxT}, batch, partition, beamwidth,
                    vertex);

  // Timestep connection
  graph.connect(vertex["currentTimestep"],
                tempTensors.currentTimestep[batch][partition][0]);

  // Constants
  graph.setInitialValue(vertex["beam"], path);
  graph.setInitialValue(vertex["maxT"], maxT);
  graph.setInitialValue(vertex["beamwidth"], beamwidth);
}

} // namespace ctc_infer
} // end namespace popnn
