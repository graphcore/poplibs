// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplibs_test/CTCInference.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>

using namespace poplibs_support;

namespace poplibs_test {
namespace ctc {

BeamHistory::BeamHistory(unsigned beamwidth, unsigned t)
    : symbols(boost::extents[beamwidth][t]),
      parents(boost::extents[beamwidth][t]) {}

// Returning tuple<symbol, beamIndex, timeIndex>
std::tuple<unsigned, unsigned, int>
BeamHistory::getNextSymbol(unsigned beamIndex, int timeIndex) const {
  unsigned result = popnn::ctc_infer::voidSymbol;
  while (timeIndex >= 0 && result == popnn::ctc_infer::voidSymbol) {
    if (symbols[beamIndex][timeIndex] != popnn::ctc_infer::voidSymbol) {
      result = symbols[beamIndex][timeIndex];
    }
    if (parents[beamIndex][timeIndex]) {
      beamIndex = *parents[beamIndex][timeIndex];
    }
    timeIndex--;
  }
  return {result, beamIndex, timeIndex};
}

// For speed, and use when comparing/merging return the sequence in reverse
std::vector<unsigned>
BeamHistory::getReversedOutputSequence(unsigned beamIndex,
                                       unsigned addend) const {
  std::vector<unsigned> reversedSequence;
  // Start with the addend if there was one - the last symbol
  if (addend != popnn::ctc_infer::voidSymbol) {
    reversedSequence.push_back(addend);
  }
  int timeIndex = nextIndexToAssign - 1;
  while (timeIndex >= 0) {
    unsigned symbol;
    std::tie(symbol, beamIndex, timeIndex) =
        getNextSymbol(beamIndex, timeIndex);
    if (symbol != popnn::ctc_infer::voidSymbol) {
      reversedSequence.push_back(symbol);
    }
  }
  return reversedSequence;
}

// For speed, compare a reversed output sequence to another beams history
// symbol by symbol
bool BeamHistory::compareReversedOutputSequence(
    unsigned beamIndex, unsigned addend,
    const std::vector<unsigned> &reversedSequence) const {
  unsigned reversedSequenceIndex = 0;
  // First the addend if non void
  if (addend != popnn::ctc_infer::voidSymbol) {
    if (reversedSequence.size() == 0) {
      return false;
    }
    if (addend != reversedSequence[reversedSequenceIndex]) {
      return false;
    }
    reversedSequenceIndex++;
  }
  int timeIndex = nextIndexToAssign - 1;
  while (timeIndex >= 0) {
    unsigned symbol;
    std::tie(symbol, beamIndex, timeIndex) =
        getNextSymbol(beamIndex, timeIndex);
    if (symbol != popnn::ctc_infer::voidSymbol) {
      if (reversedSequenceIndex == reversedSequence.size()) {
        return false;
      }
      if (reversedSequence[reversedSequenceIndex] != symbol) {
        return false;
      }
      reversedSequenceIndex++;
    }
  }
  return (reversedSequenceIndex == reversedSequence.size());
}

// Find the last symbol in the beam output sequence
unsigned BeamHistory::getLastOutput(unsigned beamIndex) const {
  auto o = getOutputSequence(beamIndex);
  if (o.empty()) {
    return popnn::ctc_infer::voidSymbol;
  } else {
    return o.back();
  }
}

void BeamHistory::assignParent(unsigned beamIndex, unsigned parentBeamIndex) {
  parents[beamIndex][nextIndexToAssign] = parentBeamIndex;
}

void BeamHistory::assignSymbol(unsigned beamIndex, unsigned addend) {
  symbols[beamIndex][nextIndexToAssign] = addend;
}

void BeamHistory::incrementIndex() { nextIndexToAssign++; }

template <typename FPType>
void print(const std::vector<Candidate<FPType>> &candidates,
           const BeamHistory &beamHistory) {
  for (const auto &candidate : candidates) {
    std::cout << "(Beam=" << candidate.beam;
    std::cout << ", addend: ";
    if (candidate.addend == popnn::ctc_infer::voidSymbol) {
      std::cout << " ";
    } else {
      std::cout << candidate.addend;
    }
    std::cout << std::fixed << std::setprecision(4)
              << " [pnb: " << candidate.pnb << ", pb: " << candidate.pb
              << "]) ";
    print(beamHistory.getOutputSequence(candidate));
  }
}

void print(const BeamHistory &beamHistory) {
  for (size_t b = 0; b < beamHistory.symbols.size(); b++) {
    for (size_t t = 0; t < beamHistory.nextIndexToAssign; t++) {
      const std::string parent =
          beamHistory.parents[b][t] ? std::to_string(*beamHistory.parents[b][t])
                                    : std::string{" "};
      const std::string symbol =
          beamHistory.symbols[b][t] != popnn::ctc_infer::voidSymbol
              ? std::to_string(beamHistory.symbols[b][t])
              : std::string{" "};
      std::cout << "(" << parent << ", " << symbol << ") ";
    }
    std::cout << std::endl;
  }
}

template <typename FPType>
std::vector<Candidate<FPType>> generateCandidates(
    const boost::multi_array<FPType, 2> &input, unsigned t,
    const std::vector<BeamProbability<FPType>> &beamProbabilities,
    const BeamHistory &beamHistory, unsigned blankSymbol, bool useLog) {
  // Each beam can be considered to end with both a non-blank and blank
  // symbol, which is represented by `BeamProbability`. Candidates are
  // notionally generated by appending symbols (all non-blank and the blank
  // symbol) at the next timestep to the beams (considering also that the beam
  // ends in both a non-blank and blank symbol).
  // However we need to combine beams with the same output. We can do some of
  // this immediately by considering copying (same output as the parent beam)
  // and extending (different output to the parent beam) beams. Examples of the
  // combining cases are given inline below for a beam output sequence of "a".

  std::vector<Candidate<FPType>> candidates;
  const auto numClassesIncBlank = input.size();
  const FPType zero = useLog ? log::probabilityZero : 0;

  unsigned beamIdx = 0;
  for (const auto &beam : beamProbabilities) {
    auto prevSymbol = beamHistory.getLastOutput(beamIdx);

    // Copy beams ---
    // Where we maintain the same beam output sequence

    // By appending a blank to beam ending in a blank
    // e.g. beam: "a-", addend: "-" -> output: "a"
    const auto blankProb = input[blankSymbol][t];
    const auto prevBlankProb =
        useLog ? log::mul(beam.pb, blankProb) : beam.pb * blankProb;
    // By appending a blank to a beam ending in a non blank
    // e.g. beam: "a", addend: "-" -> output: "a"
    const auto prevNonBlankProb =
        useLog ? log::mul(beam.pnb, blankProb) : beam.pnb * blankProb;
    const auto prob = useLog ? log::add(prevBlankProb, prevNonBlankProb)
                             : prevBlankProb + prevNonBlankProb;
    candidates.push_back({beamIdx, popnn::ctc_infer::voidSymbol, zero, prob});

    // By appending the same symbol as at the end of the beam
    // e.g. beam: "a", addend: "a" -> output: "a"
    if (prevSymbol != popnn::ctc_infer::voidSymbol) {
      const auto addendProb = input[prevSymbol][t];
      const auto nonBlankProb =
          useLog ? log::mul(beam.pnb, addendProb) : beam.pnb * addendProb;
      // Note: We don't need to create a new candidate as this will have the
      // same output sequence as the previous copy beam candidate which appended
      // a blank
      candidates.back().pnb = nonBlankProb;
    }
    candidates.back().pTotal =
        useLog ? log::add(candidates.back().pb, candidates.back().pnb)
               : candidates.back().pb + candidates.back().pnb;
    // Extend beams ---
    // Where we extend a beam by adding a symbol
    for (unsigned s = 0; s < numClassesIncBlank; s++) {
      if (s == blankSymbol) {
        continue;
      }
      // Extending a beam ending in a blank with a non-blank symbol
      // e.g. beam: "a-", addend: "a" -> output: "aa" (extended by the
      // same symbol)
      // or beam: "a-", addend: "b" -> output: "ab" (extended by a
      // different symbol)
      // The second of these cases is referenced below
      const auto addendProb = input[s][t];
      const auto blankProb =
          useLog ? log::mul(beam.pb, addendProb) : beam.pb * addendProb;
      candidates.push_back({beamIdx, s, blankProb, zero});

      // Extending a beam ending in a non-blank with a different
      // non-blank symbol
      // e.g. beam: "a", addend: "b" -> output: "ab"
      if (prevSymbol != s) {
        const auto nonBlankProb =
            useLog ? log::mul(beam.pnb, addendProb) : beam.pnb * addendProb;
        // Note: We don't need to create a new candidate as this will have the
        // same output sequence as the previous extend beam candidate
        // "(extended by a different symbol)". Here we append the new
        // symbol which is different to the symbol the beam ended with to the
        // non-blank beam.
        candidates.back().pnb =
            useLog ? log::add(candidates.back().pnb, nonBlankProb)
                   : candidates.back().pnb + nonBlankProb;
      }
      candidates.back().pTotal =
          useLog ? log::add(candidates.back().pb, candidates.back().pnb)
                 : candidates.back().pb + candidates.back().pnb;
    }
    beamIdx++;
  }

  return candidates;
}

template <typename FPType>
std::vector<Candidate<FPType>>
mergeEquivalentCandidates(const std::vector<Candidate<FPType>> &candidates,
                          const BeamHistory &beamHistory, bool useLog) {
  std::vector<Candidate<FPType>> mergedCandidates = candidates;

  for (size_t j = 0; j < mergedCandidates.size(); j++) {
    // Get the whole of this sequence once, as we compare it to many other
    // sequences
    auto &lhs = mergedCandidates[j];
    const auto lhsSequence = beamHistory.getReversedOutputSequence(lhs);
    for (size_t i = j + 1; i < mergedCandidates.size(); i++) {
      const auto &rhs = mergedCandidates[i];
      // The only way for candidates to become mergeable is if one is a copy
      // beam (same output sequence from parent beam), and the other extension
      // (of a different beam). This is from; if both candidates are copy, the
      // output sequence of parent beams will be unchanged so not made
      // equivalent. Or alternatively, both are extension, they will need to be
      // from the same output sequence which means they cannot be extending by
      // the same symbol and so not equivalent.
      if (lhs.beam == rhs.beam) {
        continue;
      }
      if (beamHistory.compareReversedOutputSequence(rhs.beam, rhs.addend,
                                                    lhsSequence)) {
        lhs.pnb = useLog ? log::add(lhs.pnb, rhs.pnb) : lhs.pnb + rhs.pnb;
        lhs.pb = useLog ? log::add(lhs.pb, rhs.pb) : lhs.pb + rhs.pb;
        lhs.pTotal = useLog ? log::add(lhs.pb, lhs.pnb) : lhs.pb + lhs.pnb;
        // It doesn't matter which we remove as they result in the same output
        // sequence since they are equivalent
        mergedCandidates.erase(mergedCandidates.begin() + i);
        i--;
      }
    }
  }

  return mergedCandidates;
}

// Return - a vector containing pairs of indicies indicating which (if any)
// candidates are mergeable.  One of each pair will always be a copy candidate
// which is the first of the pair returned.
template <typename FPType>
std::vector<std::pair<unsigned, unsigned>>
listMergeableCandidates(const std::vector<Candidate<FPType>> &candidates,
                        const BeamHistory &beamHistory) {
  std::vector<std::pair<unsigned, unsigned>> mergeablePairs;

  for (size_t j = 0; j < candidates.size(); j++) {
    for (size_t i = j + 1; i < candidates.size(); i++) {
      auto &lhs = candidates[j];
      const auto &rhs = candidates[i];
      if (lhs.beam == rhs.beam) {
        continue;
      }
      if ((lhs.addend == popnn::ctc_infer::voidSymbol &&
           rhs.addend == popnn::ctc_infer::voidSymbol) ||
          (lhs.addend != popnn::ctc_infer::voidSymbol &&
           rhs.addend != popnn::ctc_infer::voidSymbol)) {
        continue;
      }
      if (beamHistory.getOutputSequence(lhs) ==
          beamHistory.getOutputSequence(rhs)) {
        if (lhs.addend == popnn::ctc_infer::voidSymbol) {
          // lhs, indexed by j is a copy candidate
          mergeablePairs.push_back(std::make_pair(j, i));
        } else {
          // rhs, indexed by i is a copy candidate
          mergeablePairs.push_back(std::make_pair(i, j));
        }
      }
    }
  }
  return mergeablePairs;
}

template <typename FPType>
std::vector<Candidate<FPType>>
sortCandidates(const std::vector<Candidate<FPType>> &candidates, bool useLog) {
  std::vector<Candidate<FPType>> out = candidates;
  std::sort(out.begin(), out.end(),
            [&](const Candidate<FPType> &lhs, const Candidate<FPType> &rhs) {
              auto lhsSum =
                  useLog ? log::add(lhs.pnb, lhs.pb) : lhs.pnb + lhs.pb;
              auto rhsSum =
                  useLog ? log::add(rhs.pnb, rhs.pb) : rhs.pnb + rhs.pb;
              return lhsSum > rhsSum;
            });
  return out;
}

template <typename FPType>
std::vector<Candidate<FPType>>
pruneCandidates(const std::vector<Candidate<FPType>> &candidates, size_t max,
                bool useLog) {
  std::vector<Candidate<FPType>> out = candidates;
  out.resize(max);
  for (unsigned i = candidates.size(); i < out.size(); i++) {
    out[i].pnb = useLog ? log::probabilityZero : 0;
    out[i].pb = useLog ? log::probabilityZero : 0;
  }
  return out;
}

template <typename FPType>
void applyCandidates(BeamHistory &beamHistory,
                     std::vector<BeamProbability<FPType>> &beamProbabilities,
                     const std::vector<Candidate<FPType>> &candidates,
                     bool useLog) {
  // Ideally we would maintain consistency for easier debugging (stop changing
  // beam parents where possible), but for now let's reorder beams in terms of
  // probability (implicit assumption candidates is sorted here)
  auto idx = 0;
  for (auto candidate : candidates) {
    beamHistory.assignParent(idx, candidate.beam);
    beamHistory.assignSymbol(idx, candidate.addend);
    beamProbabilities.at(idx).pnb = candidate.pnb;
    beamProbabilities.at(idx).pb = candidate.pb;
    idx++;
  }
  beamHistory.incrementIndex();
}

template <typename FPType>
std::vector<std::pair<std::vector<unsigned>, FPType>>
infer(const boost::multi_array<FPType, 2> &input, unsigned blankSymbol,
      unsigned beamwidth, unsigned topBeams, bool useLog, bool verbose) {

  const FPType maxProb = useLog ? log::probabilityOne : 1;
  const FPType minProb = useLog ? log::probabilityZero : 0;
  std::vector<BeamProbability<FPType>> beamProbabilities{};
  beamProbabilities.push_back({maxProb, minProb}); // Only one origin to begin
  for (size_t i = 1; i < beamwidth; i++) {
    beamProbabilities.push_back({minProb, minProb}); // Ignore other beams
  }

  BeamHistory beamHistory{beamwidth, static_cast<unsigned>(input[0].size())};

  for (size_t t = 0; t < input[0].size(); t++) {
    auto candidates = generateCandidates(input, t, beamProbabilities,
                                         beamHistory, blankSymbol, useLog);

    if (verbose) {
      std::cout << "Candidates:" << std::endl;
      print(candidates, beamHistory);
      std::cout << std::endl;
    }
    candidates = mergeEquivalentCandidates(candidates, beamHistory, useLog);

    if (verbose) {
      std::cout << "Merged:" << std::endl;
      print(candidates, beamHistory);
      std::cout << std::endl;
    }

    candidates = sortCandidates(candidates, useLog);

    if (verbose) {
      std::cout << "Sorted:" << std::endl;
      print(candidates, beamHistory);
      std::cout << std::endl;
    }

    const auto selectedCandidates =
        pruneCandidates(candidates, beamwidth, useLog);

    if (verbose) {
      std::cout << "Pruned:" << std::endl;
      print(selectedCandidates, beamHistory);
      std::cout << std::endl;
    }

    applyCandidates(beamHistory, beamProbabilities, selectedCandidates, useLog);

    if (verbose) {
      std::cout << "============== State after time step:" << t << std::endl;
      std::cout << std::endl;

      std::cout << "Beam history: (Parent Beam reference, current symbol)"
                << std::endl;
      print(beamHistory);
      std::cout << std::endl;

      std::cout << "Current beam outputs:" << std::endl;
      for (size_t i = 0; i < beamwidth; i++) {
        std::cout << "[pnb: " << beamProbabilities[i].pnb
                  << ", pb: " << beamProbabilities[i].pb << ", pnb + pb: "
                  << (useLog
                          ? log::add(beamProbabilities[i].pnb,
                                     beamProbabilities[i].pb)
                          : beamProbabilities[i].pnb + beamProbabilities[i].pb)
                  << "] ";
        print(beamHistory.getOutputSequence(i));
      }
      std::cout << std::endl;
      std::cout << "==============" << std::endl;
      std::cout << std::endl;
    }
  }

  // We don't need to sort since beam probabilities already kept sorted
  // (implicit)
  std::vector<std::pair<std::vector<unsigned>, FPType>> outputs;
  for (unsigned i = 0; i < topBeams; i++) {
    const auto sequence = [&]() {
      auto seq = beamHistory.getOutputSequence(i);
      if (!seq.empty() && seq.back() == blankSymbol) {
        seq.resize(0);
      }
      return seq;
    }();

    auto prob =
        useLog
            ? log::add(beamProbabilities.at(i).pnb, beamProbabilities.at(i).pb)
            : beamProbabilities.at(i).pnb + beamProbabilities.at(i).pb;
    // Always return logProb
    if (!useLog) {
      prob = std::log(prob);
    }
    outputs.push_back({sequence, prob});
  }
  return outputs;
}

template std::vector<Candidate<double>> generateCandidates(
    const boost::multi_array<double, 2> &input, unsigned t,
    const std::vector<BeamProbability<double>> &beamProbabilities,
    const BeamHistory &beamHistory, unsigned blankSymbol, bool useLog);
template std::vector<Candidate<float>>
generateCandidates(const boost::multi_array<float, 2> &input, unsigned t,
                   const std::vector<BeamProbability<float>> &beamProbabilities,
                   const BeamHistory &beamHistory, unsigned blankSymbol,
                   bool useLog);

template std::vector<Candidate<double>>
mergeEquivalentCandidates(const std::vector<Candidate<double>> &candidates,
                          const BeamHistory &beamHistory, bool useLog);
template std::vector<Candidate<float>>
mergeEquivalentCandidates(const std::vector<Candidate<float>> &candidates,
                          const BeamHistory &beamHistory, bool useLog);

template std::vector<std::pair<unsigned, unsigned>>
listMergeableCandidates(const std::vector<Candidate<double>> &candidates,
                        const BeamHistory &beamHistory);
template std::vector<std::pair<unsigned, unsigned>>
listMergeableCandidates(const std::vector<Candidate<float>> &candidates,
                        const BeamHistory &beamHistory);

template std::vector<Candidate<double>>
sortCandidates(const std::vector<Candidate<double>> &candidates, bool useLog);
template std::vector<Candidate<float>>
sortCandidates(const std::vector<Candidate<float>> &candidates, bool useLog);

template std::vector<Candidate<double>>
pruneCandidates(const std::vector<Candidate<double>> &candidates, size_t max,
                bool useLog);
template std::vector<Candidate<float>>
pruneCandidates(const std::vector<Candidate<float>> &candidates, size_t max,
                bool useLog);
template void
applyCandidates(BeamHistory &beamHistory,
                std::vector<BeamProbability<double>> &beamProbabilities,
                const std::vector<Candidate<double>> &candidates, bool useLog);
template void
applyCandidates(BeamHistory &beamHistory,
                std::vector<BeamProbability<float>> &beamProbabilities,
                const std::vector<Candidate<float>> &candidates, bool useLog);

template std::vector<std::pair<std::vector<unsigned>, double>>
infer(const boost::multi_array<double, 2> &input, unsigned blankSymbol,
      unsigned beamwidth, unsigned topBeams, bool useLog, bool verbose);
template std::vector<std::pair<std::vector<unsigned>, float>>
infer(const boost::multi_array<float, 2> &input, unsigned blankSymbol,
      unsigned beamwidth, unsigned topBeams, bool useLog, bool verbose);

/// ====================================================================
/// ====================================================================
/// ====================================================================
/// ====================================================================
/// ====================================================================

// Exhaustive path inference functions.  Coded to look simple and be divided
// into lots of individually verifiable steps and help with debug.

// For inference, build an array of the indices of all possible input paths
// given no constraint whatsoever.  This can blow up huge so be careful!
boost::multi_array<unsigned, 2> findAllInputPaths(unsigned timeSteps,
                                                  unsigned sequenceLength) {

  auto paths = std::pow(sequenceLength, timeSteps);
  boost::multi_array<unsigned, 2> result(boost::extents[paths][timeSteps]);
  unsigned beamSize = sequenceLength;
  for (unsigned t = 0; t < timeSteps; t++) {
    for (unsigned s = 0; s < beamSize; s++) {
      for (unsigned p = 0; p < paths / beamSize; p++)
        result[s * (paths / beamSize) + p][t] = s % sequenceLength;
    }
    beamSize *= sequenceLength;
  }
  return result;
}

// Just follow each inputPath we are given, and multiply probabilities
template <typename FPType>
boost::multi_array<FPType, 1>
findAllInputPathProbabilities(const boost::multi_array<FPType, 2> &sequence,
                              const boost::multi_array<unsigned, 2> &inputPaths,
                              bool isLog) {

  auto paths = inputPaths.shape()[0];
  auto timeSteps = inputPaths.shape()[1];

  boost::multi_array<FPType, 1> result(boost::extents[paths]);
  std::fill(result.data(), result.data() + result.num_elements(),
            isLog ? 0 : 1);

  for (unsigned t = 0; t < timeSteps; t++) {
    for (unsigned p = 0; p < paths; p++) {
      if (isLog) {
        result[p] = log::mul(result[p], sequence[inputPaths[p][t]][t]);
      } else {
        result[p] *= sequence[inputPaths[p][t]][t];
      }
    }
  }
  return result;
}

// Convert input paths into output paths (padded with -1)
boost::multi_array<unsigned, 2>
inputToOutputPath(const boost::multi_array<unsigned, 2> &inputPath,
                  unsigned blankSymbol) {
  auto paths = inputPath.shape()[0];
  auto timeSteps = inputPath.shape()[1];
  boost::multi_array<unsigned, 2> result(boost::extents[paths][timeSteps]);
  // Fill with void symbol to represent null
  std::fill(result.data(), result.data() + result.num_elements(),
            popnn::ctc_infer::voidSymbol);

  for (unsigned p = 0; p < paths; p++) {
    auto lastOne = blankSymbol;
    unsigned length = 0;
    for (unsigned t = 0; t < timeSteps; t++) {

      if (lastOne != inputPath[p][t] && inputPath[p][t] != blankSymbol) {
        // Insert a new symbol
        result[p][length] = inputPath[p][t];
        length++;
      }
      lastOne = inputPath[p][t];
    }
  }
  return result;
}

// Merge identical output paths by generating a reference to the first instance
// of a path, and summing the probabilities of all identical paths.
template <typename FPType>
std::tuple<std::vector<unsigned>, std::vector<FPType>,
           std::vector<std::vector<unsigned>>>
mergePaths(boost::multi_array<unsigned, 2> &outPaths,
           boost::multi_array<FPType, 1> &probabilities, bool isLog) {
  std::vector<unsigned> pathRefs;
  std::vector<FPType> pathProbs;
  std::vector<std::vector<unsigned>> instances;

  for (unsigned outPath = 0; outPath < outPaths.size(); outPath++) {
    unsigned matchIndex = 0;
    bool matchFound = false;
    for (unsigned ref = 0; ref < pathRefs.size(); ref++) {
      if (outPaths[outPath] == outPaths[pathRefs[ref]]) {
        matchIndex = ref;
        matchFound = true;
        break;
      }
    }
    if (matchFound) {
      instances[matchIndex].push_back(outPath);
      if (isLog) {
        pathProbs[matchIndex] =
            log::add(pathProbs[matchIndex], probabilities[outPath]);
      } else {
        pathProbs[matchIndex] += probabilities[outPath];
      }
    } else {
      pathRefs.push_back(outPath);
      pathProbs.push_back(probabilities[outPath]);
      instances.resize(instances.size() + 1);
      instances.back().push_back(outPath);
    }
  }
  return {pathRefs, pathProbs, instances};
}

template std::tuple<std::vector<unsigned>, std::vector<double>,
                    std::vector<std::vector<unsigned>>>
mergePaths(boost::multi_array<unsigned, 2> &outPaths,
           boost::multi_array<double, 1> &probabilities, bool isLog);

template std::tuple<std::vector<unsigned>, std::vector<float>,
                    std::vector<std::vector<unsigned>>>
mergePaths(boost::multi_array<unsigned, 2> &outPaths,
           boost::multi_array<float, 1> &probabilities, bool isLog);

template boost::multi_array<double, 1>
findAllInputPathProbabilities(const boost::multi_array<double, 2> &sequence,
                              const boost::multi_array<unsigned, 2> &inputPaths,
                              bool isLog);

template boost::multi_array<float, 1>
findAllInputPathProbabilities(const boost::multi_array<float, 2> &sequence,
                              const boost::multi_array<unsigned, 2> &inputPaths,
                              bool isLog);

} // namespace ctc
} // namespace poplibs_test
