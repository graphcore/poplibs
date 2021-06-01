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
  // Flag to indicate that processing is complete - for each individual
  // batch entry, when times are < maxT
  poplar::Tensor complete;

  // Timestep counter used by the loop program to count loop passes, and end
  // when enough passes have been taken.
  // (Scalar)
  poplar::Tensor loopTimestep;
  // Per tile copy of `loopTimestep` which is broadcast to all tiles
  // at the start of each loop and used where vertices need the loop
  // count.  This avoids repeated exchange of `loopTimestep`
  // [batchSize][batchEntryPartitions][1]
  poplar::Tensor currentTimestep;
  // Loop count limit - the largest time for all entries in the current batch
  // (Scalar)
  poplar::Tensor maxTimeInBatch;

  // Extend candidates
  // [batchSize][numClasses-1][beamwidth]
  poplar::Tensor extendCandidatesPb;
  poplar::Tensor extendCandidatesPnb;
  poplar::Tensor extendCandidatesPTotal;
  poplar::Tensor extendCandidatesParent;
  poplar::Tensor extendCandidatesAddend;

  // A tensor mapped to the correct tiles for the Select stage,
  // containing a copy of the extendCandidatesPTotal and addend data
  // [batchSize][numClasses-1][beamwidth]
  poplar::Tensor selectExtendCandidatesPTotal;
  poplar::Tensor selectExtendCandidatesAddend;

  // Copy candidates - used for:
  // The original generation of copy candidates
  // The result from the `CTCSelectCopyCandidates` vertices which are the
  //     end merged candidates before sorting
  // The sorted results
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
};

struct SortTensors {
  // Input candidates to a sort stage
  // [batchSize][candidates to sort][1]
  // For the first stage the shape will be the extend and copy candidates
  // concatenated together:
  // [batchSize][numClasses * beamwidth][1]
  poplar::Tensor inCandidatesPb;
  poplar::Tensor inCandidatesPnb;
  poplar::Tensor inCandidatesPTotal;
  poplar::Tensor inCandidatesParent;
  poplar::Tensor inCandidatesAddend;

  // The results from the "Rank" vertex, before the "Reduce" vertex
  // when using the Rank sort method
  // [batchSize][rankingPartitions * groups][beamWidth]
  poplar::Tensor rankedCandidatesPb;
  poplar::Tensor rankedCandidatesPnb;
  poplar::Tensor rankedCandidatesPTotal;
  poplar::Tensor rankedCandidatesParent;
  poplar::Tensor rankedCandidatesAddend;

  // Output candidates from a sort stage
  // [batchSize][groups * beamwidth][1]
  // The last stage will have groups=1 and so shape
  // [batchSize][beamwidth][1]
  poplar::Tensor outCandidatesPb;
  poplar::Tensor outCandidatesPnb;
  poplar::Tensor outCandidatesPTotal;
  poplar::Tensor outCandidatesParent;
  poplar::Tensor outCandidatesAddend;
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
  poplar::Tensor pTotal;
  poplar::Tensor lastOutput;
  // Length of each beam output, use a dimension of 2 * beamWidth to achieve
  // a ping-pong length/previousLength while being updated
  // [batchSize][batchEntryPartitions][2 * beamWidth][1]
  poplar::Tensor length;
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
                          unsigned extendPartition, unsigned copyPartition,
                          unsigned beamPartition, unsigned blankClass,
                          unsigned beamwidth, unsigned numClasses,
                          unsigned tile);

void selectCopyCandidateVertex(poplar::Graph &graph,
                               const TempTensors &tempTensors,
                               poplar::ComputeSet &cs, unsigned batch,
                               unsigned copyPartition, unsigned beamPartition,
                               unsigned numCopyCandidates, unsigned tile);

void selectExtendCandidateVertex(poplar::Graph &graph,
                                 const TempTensors &tempTensors,
                                 poplar::ComputeSet &cs, unsigned batch,
                                 unsigned extendPartition,
                                 unsigned beamPartition,
                                 unsigned numCopyCandidates,
                                 unsigned blankClass, unsigned tile);

void simpleSortCandidatesVertex(poplar::Graph &graph,
                                const TempTensors &tempTensors,
                                poplar::ComputeSet &cs, unsigned batch,
                                unsigned partition,
                                unsigned candidatesToCompare,
                                unsigned beamwidth, unsigned tile);

void rankCandidatesVertex(poplar::Graph &graph, const TempTensors &tempTensors,
                          const SortTensors &sortTensors,
                          poplar::ComputeSet &cs, unsigned batch,
                          unsigned partition, unsigned beamPartition,
                          const poplar::Interval &candidatesToCompare,
                          const poplar::Interval &rangeToRank,
                          unsigned beamwidth, unsigned tile);

void reduceCandidatesVertex(
    poplar::Graph &graph, const TempTensors &tempTensors,
    const SortTensors &sortTensors, poplar::ComputeSet &cs, unsigned batch,
    unsigned group, unsigned partition, unsigned beamPartition,
    const poplar::Interval &candidatesToReduce, unsigned tile);

void updateVertex(poplar::Graph &graph, const BeamTensors &beams,
                  const TempTensors &tempTensors, poplar::ComputeSet &cs,
                  unsigned batch, const poplar::Interval &time,
                  unsigned beamPartition, unsigned beamwidth, unsigned tile);

void generateOutputVertex(poplar::Graph &graph, const BeamTensors &beams,
                          const TempTensors &tempTensors,
                          const poplar::Tensor &labels,
                          const poplar::Tensor &labelLengths,
                          poplar::ComputeSet &cs, unsigned batch, unsigned path,
                          unsigned partition, unsigned beamwidth,
                          unsigned numClassesIncBlank, unsigned tile);

} // namespace ctc_infer
} // end namespace popnn
#endif // popnn_CTCInferenceConnection_hpp
