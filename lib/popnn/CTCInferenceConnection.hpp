// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popnn_CTCInferenceConnection_hpp
#define popnn_CTCInferenceConnection_hpp

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <boost/optional.hpp>

namespace popnn {
namespace ctc_infer {

struct TempTensors {
  // dataLengths broadcast, for vertices to access
  // [batchSize][batchEntryPartitions][1]
  poplar::Tensor dataLengths;

  // Timestep counter to count loop passes with a counter on each tile
  // [batchSize][batchEntryPartitions][1]
  poplar::Tensor currentTimestep;
  // Extend candidates
  // [batchSize][numClasses-1][beamwidth]
  poplar::Tensor extendCandidatesPb;
  poplar::Tensor extendCandidatesPnb;
  poplar::Tensor extendCandidatesPTotal;
  poplar::Tensor extendCandidatesParent;
  poplar::Tensor extendCandidatesAddend;
  // Copy candidates - used for both the original generation of copy candidates
  // and the result from the `CTCSelectCopyCandidates` vertices which are the
  // end merged candidates before Select
  // [batchSize][beamwidth][1]
  poplar::Tensor copyCandidatesPb;
  poplar::Tensor copyCandidatesPnb;
  poplar::Tensor copyCandidatesPTotal;
  poplar::Tensor copyCandidatesParent;
  poplar::Tensor copyCandidatesAddend;
  // Merge candidates (The broadcast of the copy candidates which can be
  // modified)
  // To avoid rearrangements into a single contiguous variable
  // each is a vector of tensors.  Vector size is beamwidth
  // [batchSize][plan.parallel.merge][1]
  std::vector<poplar::Tensor> mergeCandidatesPb;
  std::vector<poplar::Tensor> mergeCandidatesPnb;
  std::vector<poplar::Tensor> mergeCandidatesPTotal;
  std::vector<poplar::Tensor> mergeCandidatesParent;
  std::vector<poplar::Tensor> mergeCandidatesAddend;

  // The sorted results from the "Rank" sort method
  // [batchSize][plan.parallel.sortRanking][beamWidth]
  poplar::Tensor sortedCandidatesPb;
  poplar::Tensor sortedCandidatesPnb;
  poplar::Tensor sortedCandidatesPTotal;
  poplar::Tensor sortedCandidatesParent;
  poplar::Tensor sortedCandidatesAddend;
};

struct BeamTensors {
  // Beam history
  // 0th timestep is an initial state
  // [batchSize][batchEntryPartitions][maxT+1][beamWidth]
  poplar::Tensor parent;
  poplar::Tensor addend;
  // Beam probabilities, and last output from each beam
  // [batchSize][batchEntryPartitions][beamWidth][1]
  poplar::Tensor pb;
  poplar::Tensor pnb;
  poplar::Tensor lastOutput;
  poplar::Tensor length;
  // A copy of lastOutput and length, made to use in the update process
  poplar::Tensor previousLastOutput;
  poplar::Tensor previousLength;
};

void generateExtendCandidateVertex(
    poplar::Graph &graph, const poplar::Tensor &data, const BeamTensors &beams,
    const TempTensors &TempTensors, poplar::ComputeSet &cs, unsigned batch,
    const poplar::Interval &time, unsigned addendPartition,
    unsigned dataPartition, unsigned blankClass, unsigned beamwidth,
    const poplar::Interval &beamPartition, unsigned addendClass, unsigned tile);

void generateCopyCandidateVertex(
    poplar::Graph &graph, const poplar::Tensor &data, const BeamTensors &beams,
    const TempTensors &TempTensors, poplar::ComputeSet &cs, unsigned batch,
    const poplar::Interval &time, unsigned beamPartition,
    unsigned dataPartition, unsigned blankClass, unsigned beamwidth,
    unsigned tile);

void mergeCandidateVertex(poplar::Graph &graph, const BeamTensors &beams,
                          const TempTensors &tempTensors,
                          poplar::ComputeSet &cs, unsigned batch,
                          const poplar::Interval &time,
                          unsigned extendPartition, unsigned beamPartition,
                          unsigned blankClass, unsigned beamwidth,
                          unsigned tile);

void selectCopyCandidateVertex(poplar::Graph &graph,
                               const TempTensors &tempTensors,
                               poplar::ComputeSet &cs, unsigned batch,
                               unsigned copyPartition,
                               unsigned numCopyCandidates, unsigned tile);

void selectExtendCandidateVertex(poplar::Graph &graph,
                                 const TempTensors &tempTensors,
                                 poplar::ComputeSet &cs, unsigned batch,
                                 unsigned extendPartition,
                                 unsigned numCopyCandidates,
                                 unsigned blankClass, unsigned tile);

void selectCandidatesVertex(poplar::Graph &graph,
                            const TempTensors &tempTensors,
                            poplar::ComputeSet &cs, unsigned batch,
                            unsigned partition, unsigned candidatesToCompare,
                            unsigned beamwidth, unsigned tile);

void rankCandidatesVertex(poplar::Graph &graph, const TempTensors &tempTensors,
                          poplar::ComputeSet &cs, unsigned batch,
                          unsigned partition, unsigned candidatesToCompare,
                          const poplar::Interval &rangeToRank,
                          unsigned beamwidth, unsigned tile);

void reduceCandidatesVertex(poplar::Graph &graph,
                            const TempTensors &tempTensors,
                            poplar::ComputeSet &cs, unsigned batch,
                            unsigned partition, unsigned candidatesToReduce,
                            unsigned tile);

void updateVertex(poplar::Graph &graph, const BeamTensors &beams,
                  const TempTensors &tempTensors, poplar::ComputeSet &cs,
                  unsigned batch, const poplar::Interval &time,
                  unsigned beamPartition, unsigned sortedResultOffset,
                  unsigned beamwidth, unsigned tile);

void generateOutputVertex(poplar::Graph &graph, const BeamTensors &beams,
                          const TempTensors &tempTensors,
                          const poplar::Tensor &labels,
                          const poplar::Tensor &labelLengths,
                          poplar::ComputeSet &cs, unsigned batch,
                          unsigned beamwidth, unsigned numClassesIncBlank,
                          unsigned partition, unsigned path, unsigned tile);

} // namespace ctc_infer
} // end namespace popnn
#endif // popnn_CTCInferenceConnection_hpp
