// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_ctc_inference_hpp
#define poplibs_test_ctc_inference_hpp

#include "CTCUtil.hpp"
#include <poplibs_support/LogArithmetic.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>

#include <vector>

namespace poplibs_test {
namespace ctc {

// placeholder to represent no change to output sequence
inline constexpr unsigned voidSymbol = std::numeric_limits<unsigned>::max();

// Probabilities representing the final symbol of a beam
template <typename FPType> struct BeamProbability {
  FPType pnb; // non blank
  FPType pb;  // blank
};

template <typename FPType> struct Candidate {
  unsigned beam;   // Parent output sequence
  unsigned addend; // Symbol to append onto beam

  FPType pnb; // beam and non blank
  FPType pb;  // beam and blank
};

struct BeamHistory {
  // symbols array contains the output sequences for each beam, with an entry at
  // every timestep (so potentially "oversampled"). Traversing through this
  // array builds up the actual output sequence by either:
  //  - voidSymbol -> no change in output sequence
  //  - otherwise -> appended to output sequence
  // so traversing `a` -> `a` in the table represents the output sequence `aa`,
  // not `a`!
  boost::multi_array<unsigned, 2> symbols;
  boost::multi_array<boost::optional<unsigned>, 2>
      parents; // If unset, no parent for convenience

  // This represents the position which is one beyond the end of the assigned
  // beams (assessed with candidates)
  unsigned nextIndexToAssign = 0;

  BeamHistory(unsigned beamwidth, unsigned t);

  std::vector<unsigned> getOutputSequence(unsigned beamIndex) const;
  template <typename FPType>
  std::vector<unsigned>
  getOutputSequence(const Candidate<FPType> &candidate) const {
    auto output = getOutputSequence(candidate.beam);
    if (candidate.addend != voidSymbol) {
      output.push_back(candidate.addend);
    }
    return output;
  }

  // Find the last symbol in the beam output sequence
  unsigned getLastOutput(unsigned beamIndex) const;
  void assignParent(unsigned beamIndex, unsigned parentBeamIndex);
  void assignSymbol(unsigned beamIndex, unsigned addend);
  void incrementIndex();
};

void print(const BeamHistory &beamHistory);

template <typename FPType>
std::vector<Candidate<FPType>> generateCandidates(
    const boost::multi_array<FPType, 2> &input, unsigned t,
    const std::vector<BeamProbability<FPType>> &beamProbabilities,
    const BeamHistory &beamHistory, unsigned blankSymbol, bool useLog);

template <typename FPType>
std::vector<Candidate<FPType>>
mergeEquivalentCandidates(const std::vector<Candidate<FPType>> &candidates,
                          const BeamHistory &beamHistory, bool useLog);

template <typename FPType>
std::vector<std::pair<unsigned, unsigned>>
listMergeableCandidates(const std::vector<Candidate<FPType>> &candidates,
                        const BeamHistory &beamHistory);
template <typename FPType>
std::vector<Candidate<FPType>>
sortCandidates(const std::vector<Candidate<FPType>> &candidates, bool useLog);

template <typename FPType>
std::vector<Candidate<FPType>>
pruneCandidates(const std::vector<Candidate<FPType>> &candidates, size_t max,
                bool useLog);

template <typename FPType>
void applyCandidates(BeamHistory &beamHistory,
                     std::vector<BeamProbability<FPType>> &beamProbabilities,
                     const std::vector<Candidate<FPType>> &candidates,
                     bool useLog);

template <typename FPType>
std::tuple<FPType, std::vector<unsigned>>
infer(const boost::multi_array<FPType, 2> &input, unsigned blankSymbol,
      unsigned beamwidth, bool useLog, bool verbose = false);

//------------------------------------------------------------------------------
// Exhaustive path inference functions. Coded to look simple and be divided
// into lots of individually verifiable steps and help with debug.
boost::multi_array<unsigned, 2> findAllInputPaths(unsigned timeSteps,
                                                  unsigned sequenceLength);

template <typename FPType>
boost::multi_array<FPType, 1>
findAllInputPathProbabilities(const boost::multi_array<FPType, 2> &sequence,
                              const boost::multi_array<unsigned, 2> &inputPaths,
                              bool isLog);

boost::multi_array<unsigned, 2>
inputToOutputPath(const boost::multi_array<unsigned, 2> &inputPath,
                  unsigned blankSymbol);

template <typename FPType>
std::tuple<std::vector<unsigned>, std::vector<FPType>,
           std::vector<std::vector<unsigned>>>
mergePaths(boost::multi_array<unsigned, 2> &outPaths,
           boost::multi_array<FPType, 1> &probabilities, bool isLog);

} // namespace ctc
} // namespace poplibs_test

#endif // poplibs_test_ctc_inference_hpp
