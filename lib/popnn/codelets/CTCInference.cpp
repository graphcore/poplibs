// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "LogOps.hpp"

#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include <poplibs_support/CTCInferenceDefs.hpp>
#include <poplibs_support/ExternalCodelet.hpp>
#include <poplibs_support/LogArithmetic.hpp>

#include <cassert>
#include <cmath>
#include <print.h>
#include <tuple>
#include <type_traits>

using namespace poplar;
using namespace poplibs_support;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto voidSymbol = popnn::ctc_infer::voidSymbol;

inline constexpr auto invalidParent = std::numeric_limits<unsigned>::max();

namespace popnn {

template <typename InType, typename PartialsType, typename SymbolType>
class CTCGenerateCopyCandidates : public Vertex {

public:
  CTCGenerateCopyCandidates();
  Input<Vector<InType, ONE_PTR>> logProbs; // [maxT, numClassIncBlank]
  Input<Vector<SymbolType, ONE_PTR>> lastBeamOutputs;    // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbNonBlank; // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbBlank;    // [beamwidth]

  // Index into logProbs for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;

  // Outputs are for a single candidate only
  Output<unsigned> candidateParent;
  Output<SymbolType> candidateAddend;
  Output<PartialsType> candidateBeamProbNonBlank;
  Output<PartialsType> candidateBeamProbBlank;
  Output<PartialsType> candidateBeamProbTotal;

  const unsigned numClassesIncBlank;
  const SymbolType blankClass;
  const unsigned beamwidth;
  const unsigned beamIdx;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (currentTimestep >= dataLength) {
      return true;
    }
    const unsigned baseOffset = (*currentTimestep) * numClassesIncBlank;
    const auto blankProb =
        static_cast<PartialsType>(logProbs[baseOffset + blankClass]);

    const auto prevSymbol = lastBeamOutputs[beamIdx];
    // Create the copy candidate from the specific beam

    // Copy beams are where we maintain the same beam output sequence
    // By appending a blank to beam ending in a blank
    // e.g. beam: "a-", addend: "-" -> output: "a"
    const auto prevBlankProb =
        logMul<PartialsType>(beamProbBlank[beamIdx], blankProb);
    // By appending a blank to a beam ending in a non blank
    // e.g. beam: "a", addend: "-" -> output: "a"
    const auto prevNonBlankProb =
        logMul<PartialsType>(beamProbNonBlank[beamIdx], blankProb);
    const auto prob = logAdd<PartialsType>(prevBlankProb, prevNonBlankProb);
    *candidateParent = beamIdx;
    *candidateAddend = voidSymbol;
    *candidateBeamProbNonBlank = log::probabilityZero;
    *candidateBeamProbBlank = prob;
    *candidateBeamProbTotal = prob;

    // By appending the same symbol as at the end of the beam
    // e.g. beam: "a", addend: "a" -> output: "a"
    if (prevSymbol != voidSymbol) {
      const auto addendProb =
          static_cast<PartialsType>(logProbs[baseOffset + prevSymbol]);
      const auto nonBlankProb =
          logMul<PartialsType>(beamProbNonBlank[beamIdx], addendProb);
      // Note: We don't need to create a new candidate as this will have the
      // same output sequence as the previous copy beam candidate which
      // appended a blank
      *candidateBeamProbNonBlank = nonBlankProb;
      *candidateBeamProbTotal = logAdd<PartialsType>(prob, nonBlankProb);
    }
    return true;
  }
};

template class CTCGenerateCopyCandidates<float, float, unsigned>;
template class CTCGenerateCopyCandidates<half, float, unsigned>;
template class CTCGenerateCopyCandidates<half, half, unsigned>;

template <typename InType, typename PartialsType, typename SymbolType>
class CTCGenerateExtendCandidates : public Vertex {

public:
  CTCGenerateExtendCandidates();
  Input<Vector<InType, ONE_PTR>> logProbs; // [maxT, numClassIncBlank]
  Input<Vector<SymbolType, ONE_PTR>> lastBeamOutputs;    // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbNonBlank; // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbBlank;    // [beamwidth]

  // Index into logProbs for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;

  // Outputs have size[endBeam-startBeam],
  // the vertex generates endBeam-startBeam] candidates
  Output<Vector<unsigned, ONE_PTR>> extendCandidateParent;
  Output<Vector<SymbolType, ONE_PTR>> extendCandidateAddend;
  Output<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbNonBlank;
  Output<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbBlank;
  Output<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbTotal;

  const unsigned numClassesIncBlank;
  const SymbolType blankClass;
  // Extending beams in this range
  const unsigned startBeam;
  const unsigned endBeam;
  // Creating extend beams with this symbol
  const unsigned addendSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (currentTimestep >= dataLength) {
      return true;
    }
    const unsigned baseOffset = (*currentTimestep) * numClassesIncBlank;
    const auto blankProb =
        static_cast<PartialsType>(logProbs[baseOffset + blankClass]);
    unsigned outIdx = 0;
    for (unsigned beamIdx = startBeam; beamIdx < endBeam; ++beamIdx) {
      // Extend beams ---
      // Extend the beam using addendSymbol
      // Extending a beam ending in a blank with a non-blank symbol
      // e.g. beam: "a-", addend: "a" -> output: "aa" (extended by the
      // same symbol)
      extendCandidateParent[outIdx] = beamIdx;
      extendCandidateAddend[outIdx] = addendSymbol;
      extendCandidateBeamProbBlank[outIdx] = log::probabilityZero;

      const auto addendProb =
          static_cast<PartialsType>(logProbs[baseOffset + addendSymbol]);
      auto extendingBlankProb =
          logMul<PartialsType>(beamProbBlank[beamIdx], addendProb);

      // Extending a beam ending in a non-blank with a different
      // non-blank symbol
      // e.g. beam: "a", addend: "b" -> output: "ab"
      const auto prevSymbol = lastBeamOutputs[beamIdx];
      if (prevSymbol != addendSymbol) {
        const auto extendingNonBlankProb =
            logMul<PartialsType>(beamProbNonBlank[beamIdx], addendProb);
        // Note: We don't need to create a new candidate as this will have the
        // same output sequence as the previous extend beam candidate
        // "(extended by a different symbol)". Here we append the new
        // symbol which is different to the symbol the beam ended with to the
        // non-blank beam.

        extendingBlankProb =
            logAdd<PartialsType>(extendingBlankProb, extendingNonBlankProb);
      }
      extendCandidateBeamProbNonBlank[outIdx] = extendingBlankProb;
      extendCandidateBeamProbTotal[outIdx] = extendingBlankProb;
      outIdx++;
    }
    return true;
  }
};

template class CTCGenerateExtendCandidates<float, float, unsigned>;
template class CTCGenerateExtendCandidates<half, float, unsigned>;
template class CTCGenerateExtendCandidates<half, half, unsigned>;

template <typename SymbolType>
// return {Symbol, t, beam}
inline std::tuple<SymbolType, unsigned, unsigned>
getNextSymbol(SymbolType *beamAddend, unsigned *beamParent, unsigned beamwidth,
              unsigned t, unsigned beam) {
  while (t > 0) {
    t--;
    auto symbol = beamAddend[beamwidth * t + beam];
    beam = beamParent[beamwidth * t + beam];
    if (symbol != voidSymbol) {
      return {symbol, t, beam};
    }
  }
  return {voidSymbol, 0, beam};
}

template <typename SymbolType>
inline bool equivalentOutputSequence(SymbolType *beamAddend,
                                     unsigned *beamParent, unsigned t,
                                     unsigned beamwidth, unsigned parentLhs,
                                     SymbolType addendLhs, unsigned parentRhs,
                                     SymbolType addendRhs) {

  // Assumption that addendLhs != addendRhs, by design we shouldn't be
  // comparing two addends with classes which are both not voidSymbol
  auto tLhs = t;
  auto tRhs = t;
  auto beamLhs = parentLhs;
  auto beamRhs = parentRhs;
  unsigned nextSymbolLhs, nextSymbolRhs;

  if (addendLhs == voidSymbol) {
    std::tie(nextSymbolLhs, tLhs, beamLhs) =
        getNextSymbol(beamAddend, beamParent, beamwidth, tLhs, beamLhs);
  } else {
    nextSymbolLhs = addendLhs;
  }

  if (addendRhs == voidSymbol) {
    std::tie(nextSymbolRhs, tRhs, beamRhs) =
        getNextSymbol(beamAddend, beamParent, beamwidth, tRhs, beamRhs);
  } else {
    nextSymbolRhs = addendRhs;
  }

  while (true) {
    if (nextSymbolLhs != nextSymbolRhs) {
      return false;
    }
    if (tLhs == tRhs && beamLhs == beamRhs) {
      return true;
    }
    std::tie(nextSymbolLhs, tLhs, beamLhs) =
        getNextSymbol(beamAddend, beamParent, beamwidth, tLhs, beamLhs);
    std::tie(nextSymbolRhs, tRhs, beamRhs) =
        getNextSymbol(beamAddend, beamParent, beamwidth, tRhs, beamRhs);
  }
}

template <typename PartialsType, typename SymbolType>
class CTCMergeCandidates : public Vertex {
public:
  CTCMergeCandidates();
  // Shape/size of Inputs is [extendCandidates]
  Input<Vector<unsigned, ONE_PTR>> extendCandidateParent;
  Input<Vector<SymbolType, ONE_PTR>> extendCandidateAddend;
  Input<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbNonBlank;
  Input<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbBlank;
  // Note - total probability not required for extend candidates, because:
  // 1. They are never modified.
  // 2. When merging, the resulting copy candidate needs a updated pb and pnb
  //    pb = pbCopy + pbExtend and pnb = pnbCopy + pnbExtend
  //    and so the pTotal of the merged result can be calculated from pb, pnb

  InOut<unsigned> copyCandidateParent;
  InOut<SymbolType> copyCandidateAddend;
  InOut<PartialsType> copyCandidateBeamProbNonBlank;
  InOut<PartialsType> copyCandidateBeamProbBlank;
  InOut<PartialsType> copyCandidateBeamProbTotal;

  Input<Vector<SymbolType, ONE_PTR>> beamAddend; // [maxT, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamParent;   // [maxT, beamwidth]

  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;
  // An output indicating the index of the single extend candidate (if any)
  // which was merged with the copy candidate.  beamwidth signifies no merged
  Output<unsigned> invalidCandidate;

  const unsigned extendCandidates;
  const unsigned beamwidth;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // Consider a single copy candidate and a number of extend candidates.
    // The copy candidate is compared to each of the extend candidates in turn,
    // we expect only 1 match (at most) so stop if a match is found.
    // Comparison involves tracking backward through the beam history of both
    // candidates in order to generate the output symbol by symbol.  Outputs are
    // compared one by one and conclusions reached:
    // If the output symbols differ - there is no match (and exit)
    // If the parent beam is the same (and all symbols so far match) - the
    // outputs match.
    // If the beginning of one beam history is reached and not the other then
    // the lengths are difference so there is no match
    if (currentTimestep >= dataLength) {
      return true;
    }
    // TODO - This allows detecting copy candidates with close to zero
    // probability.  The definition of this is a bit arbitrary
    const auto closeToZeroProbability =
        log::probabilityZero + static_cast<PartialsType>(1000);
    // Flag as all valid until we merge
    *invalidCandidate = beamwidth;

    const unsigned parentLhs = *copyCandidateParent;
    const unsigned addendLhs = *copyCandidateAddend;
    // No candidates are mergeable at t = 0
    // Don't merge candidates with zero probability as the results introduce
    // beams with outputs and zero probability
    if (currentTimestep == 0 ||
        (*copyCandidateBeamProbNonBlank <= closeToZeroProbability &&
         *copyCandidateBeamProbBlank <= closeToZeroProbability)) {
      return true;
    }

    // The only way for candidates to be mergeable is if one is a copy
    // beam (same output sequence from parent beam), and the other extension
    // (of a different beam). This is from; if both candidates are copy, the
    // output sequence of parent beams will be unchanged so not made
    // equivalent. Or alternatively, both are extension, they will need to
    // be from the same output sequence which means they cannot be extending
    // by the same symbol and so not equivalent.
    //
    // Here we compare a single copy beam to multiple extend beams.  There is a
    // possibility of a merge if the beam parents are different, and only 1
    // merge is possible
    for (unsigned i = 0; i < extendCandidates; i++) {

      const auto parentRhs = extendCandidateParent[i];
      const auto addendRhs = extendCandidateAddend[i];
      if (parentLhs == parentRhs) {
        // From the same beam
        continue;
      }

      // TODO: improve efficiency by simplifting traceback mechanism, possibly
      // with no voidSymbol, and therefore different length paths for each beam
      if (equivalentOutputSequence(&(beamAddend[0]), &(beamParent[0]),
                                   *currentTimestep, beamwidth, parentLhs,
                                   addendLhs, parentRhs, addendRhs)) {

        *copyCandidateBeamProbNonBlank = logAdd(
            *copyCandidateBeamProbNonBlank, extendCandidateBeamProbNonBlank[i]);
        *copyCandidateBeamProbBlank = logAdd(*copyCandidateBeamProbBlank,
                                             extendCandidateBeamProbBlank[i]);
        *copyCandidateBeamProbTotal =
            logAdd(*copyCandidateBeamProbBlank, *copyCandidateBeamProbNonBlank);
        // Preserve the addend and parent of the extend candidate
        *copyCandidateParent = parentRhs;
        *copyCandidateAddend = addendRhs;
        *invalidCandidate = i;
        // Only 1 can match
        break;
      }
    }
    return true;
  }
};

template class CTCMergeCandidates<float, unsigned>;
template class CTCMergeCandidates<half, unsigned>;

template <typename PartialsType, typename SymbolType>
class CTCSelectCandidates : public Vertex {

public:
  CTCSelectCandidates();
  // The input is a flat list of candidates:
  // Total broadcast copy candidates followed by the extend candidates
  // Suppose Beamwidth = 3, and numSymbols = 3 so numSymbolsM1 = 2 (no blank)
  // numGroups = numSymbolsM1, numParents = beamwidth
  //----------------------------------------------------------------------------
  // copy[group0,parent0]  // Was compared to group 0
  // copy[group0,parent1]  //        (But could have been updated if merged)
  // copy[group0,parent2]
  // copy[group1,parent0]  // A copy of copy[g0,p0], was compared to group1
  // copy[group1,parent1]  //        (But could have been updated if merged)
  // copy[group1,parent2]
  // extend[group0,parent0]  // Group 0 of extend candidates (same addend)
  // extend[group0,parent1]
  // extend[group0,parent2]
  // extend[group1,parent0]  // Group 1 of extend candidates (same addend)
  // extend[group1,parent1]
  // extend[group1,parent2]
  //
  // Following merge we know that 1 or none of copy[groupX,parent0] was merged
  // with an extend candidate, likewise for copy[groupX,parent1] and
  // copy[groupX,parent2]
  // First we will determine which merges happened and overwrite last group of
  // copy candidates with the merged candidates (if any) and null the
  // probability of the extend candidate that was merged

  InOut<Vector<unsigned, ONE_PTR>> candidateParent;
  InOut<Vector<SymbolType, ONE_PTR>> candidateAddend;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbTotal;

  // TODO - this appears redundant, although this code could be subject to
  // change.  Leave it in place for now, as it may prove useful later!
  Input<Vector<unsigned>> mergedCandidateIndicator;

  // Only use of current timestep and dataLength is to end early
  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;

  const unsigned beamwidth; // beamwidth indicates the number of copy candidates
  const unsigned totalCandidates;
  const unsigned extendCandidateGroups;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (currentTimestep >= dataLength) {
      return true;
    }
    // Precondition - candidates to be padded by previous codelets or memory
    // suitably initialized (probability zero)
    assert(beamwidth <= totalCandidates);
    const auto numCopyCandidatesBroadcast = beamwidth * extendCandidateGroups;

    // TODO - This preparation stage could maybe be parellelised by using a
    // dedicated codelet

    // We already broadcast the copy candidates - so only keep one of each
    // broadcast group, but select the one that was part of a merge
    // (if there was a merge)
    for (unsigned i = 0; i < beamwidth; i++) {
      // Loop over each copy candidate (Before the broadcast)
      auto mergeFound = false;
      unsigned mergedIndex;
      unsigned groupOffset = 0;
      // Loop over all but the last one in a broadcast group.
      for (unsigned j = 0; j < extendCandidateGroups - 1; j++) {
        const auto idx = i + groupOffset;
        // Any copy candidate that was merged will have its addend (Which was
        // the voidSymbol) overwritten with that of the extend candidate it was
        // merged with
        if (candidateAddend[idx] != voidSymbol) {
          // There was a merge among the group of broadcast copy candidates
          mergeFound = true;
          mergedIndex = idx;
          // Null out the probability of the corresponding extend candidate.
          // When merged the copy candidate's parent was overwritten with that
          // of the extend candidate it merged with  This allows us to find and
          // modify that extend candidate
          candidateBeamProbTotal[numCopyCandidatesBroadcast + // Start of extend
                                                              // candidates
                                 groupOffset +           // group0, group1 etc
                                 candidateParent[idx]] = // parent within group
              log::probabilityZero;
          groupOffset = beamwidth * (extendCandidateGroups - 1);
          break;
        }
        groupOffset += beamwidth;
      }
      // Keep or update the last one depending on if a merge was already found
      const auto idx = i + groupOffset;
      if (mergeFound) {
        // There was already a merge among the copy candidates so overwrite the
        // last one
        candidateParent[idx] = candidateParent[mergedIndex];
        candidateAddend[idx] = candidateAddend[mergedIndex];
        candidateBeamProbNonBlank[idx] = candidateBeamProbNonBlank[mergedIndex];
        candidateBeamProbBlank[idx] = candidateBeamProbBlank[mergedIndex];
        candidateBeamProbTotal[idx] = candidateBeamProbTotal[mergedIndex];
      } else {
        // No merge found yet, so use the last one, checking if it was a merge
        if (candidateAddend[idx] != voidSymbol) {
          candidateBeamProbTotal[numCopyCandidatesBroadcast + groupOffset +
                                 candidateParent[idx]] = log::probabilityZero;
        }
      }
    }
    // The actual select - operate on the last group of copy candidates and
    // the extend candidates only.
    // The result is in the position of hte last group of copy candidates
    const unsigned offset = numCopyCandidatesBroadcast - beamwidth;
    for (unsigned b = 0; b < beamwidth; b++) {
      const auto beamOutIdx = offset + b;
      unsigned maxIdx = beamOutIdx;
      PartialsType max = candidateBeamProbTotal[beamOutIdx];
      for (unsigned i = beamOutIdx; i < totalCandidates; i++) {
        const auto cmp = candidateBeamProbTotal[i];
        if (cmp > max) {
          maxIdx = i;
          max = cmp;
        }
      }

      unsigned tmpParent = candidateParent[beamOutIdx];
      candidateParent[beamOutIdx] = candidateParent[maxIdx];
      candidateParent[maxIdx] = tmpParent;

      SymbolType tmpAddend = candidateAddend[beamOutIdx];
      candidateAddend[beamOutIdx] = candidateAddend[maxIdx];
      candidateAddend[maxIdx] = tmpAddend;

      PartialsType tmpBeamProbNonBlank = candidateBeamProbNonBlank[beamOutIdx];
      candidateBeamProbNonBlank[beamOutIdx] = candidateBeamProbNonBlank[maxIdx];
      candidateBeamProbNonBlank[maxIdx] = tmpBeamProbNonBlank;

      PartialsType tmpBeamProbBlank = candidateBeamProbBlank[beamOutIdx];
      candidateBeamProbBlank[beamOutIdx] = candidateBeamProbBlank[maxIdx];
      candidateBeamProbBlank[maxIdx] = tmpBeamProbBlank;

      PartialsType tmpProbTotalScratch = candidateBeamProbTotal[beamOutIdx];
      candidateBeamProbTotal[beamOutIdx] = candidateBeamProbTotal[maxIdx];
      candidateBeamProbTotal[maxIdx] = tmpProbTotalScratch;
    }

    return true;
  }
};

template class CTCSelectCandidates<float, unsigned>;
template class CTCSelectCandidates<half, unsigned>;

// TODO - Consider splitting this up - different field can be done in parallel
template <typename PartialsType, typename SymbolType>
class CTCUpdate : public Vertex {
public:
  CTCUpdate();
  // Candidates
  // [beamWidth] in size, or larger but only beamWidth entries are used
  Input<Vector<unsigned, ONE_PTR>> candidateParent;
  Input<Vector<SymbolType, ONE_PTR>> candidateAddend;
  Input<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  Input<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;

  // Beams
  InOut<Vector<PartialsType, ONE_PTR>> beamProbNonBlank; // [beamwidth]
  InOut<Vector<PartialsType, ONE_PTR>> beamProbBlank;    // [beamwidth]
  InOut<Vector<unsigned, ONE_PTR>> lastBeamOutputs;      // [beamwidth]
  InOut<Vector<unsigned, ONE_PTR>> beamAddend;           // [maxT, beamwidth]
  InOut<Vector<unsigned, ONE_PTR>> beamParent;           // [maxT, beamwidth]

  Output<Vector<unsigned, ONE_PTR>> lastBeamOutputsScratch; //[beamWidth]

  InOut<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;
  const unsigned beamwidth;

  IS_EXTERNAL_CODELET(false);
  bool compute() {
    if (currentTimestep >= dataLength) {
      // Early exit here avoids updating beams, probabilities and the count
      // and so nothing will change regardless
      return true;
    }
    // Preserve the last beam outputs which may be needed after they are
    // overwritten
    for (unsigned i = 0; i < beamwidth; i++) {
      lastBeamOutputsScratch[i] = lastBeamOutputs[i];
    }
    const unsigned baseOffset = (*currentTimestep) * beamwidth;
    const auto parent = &beamParent[baseOffset];
    const auto addend = &beamAddend[baseOffset];
    for (unsigned i = 0; i < beamwidth; i++) {
      // TODO - this special case shouldn't be needed - we are dealing with
      // candidates which are from invalid beams (At t=0 only beam zero exists)
      parent[i] = currentTimestep == 0 ? 0 : candidateParent[i];
      addend[i] = candidateAddend[i];
      beamProbNonBlank[i] = candidateBeamProbNonBlank[i];
      beamProbBlank[i] = candidateBeamProbBlank[i];

      // Keep the output from the parent beam, which can be a new parent
      lastBeamOutputs[i] = candidateAddend[i] == voidSymbol
                               ? lastBeamOutputsScratch[candidateParent[i]]
                               : candidateAddend[i];
    }
    // Increment time, which is used for all the codelets
    *currentTimestep = *currentTimestep + 1;
    // TODO - here we could compare to a max count and end the whole poplar
    // loop early by outputting a "loop end" flag.
    // Only for 1 of the vertices in the batch though
    return true;
  }
};

template class CTCUpdate<float, unsigned>;
template class CTCUpdate<half, unsigned>;

template <typename SymbolType> class CTCGenerateOutput : public Vertex {
public:
  CTCGenerateOutput();

  Input<Vector<unsigned, ONE_PTR>> beamAddend; // [maxT, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamParent; // [maxT, beamwidth]
  Input<unsigned> currentTimestep;
  // The actual number of valid symbols found in the beamOutput
  Output<unsigned> outputLength;
  // The valid symbol sequence
  Output<Vector<SymbolType>> beamOutput; // [maxT]
  const unsigned beamwidth;
  const unsigned maxT;
  const unsigned beam;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    auto traceBackBeam = beam;
    auto traceBackTime = *currentTimestep;

    for (unsigned i = 0; i < currentTimestep; i++) {
      SymbolType symbol;
      std::tie(symbol, traceBackTime, traceBackBeam) =
          getNextSymbol(&beamAddend[0], &beamParent[0], beamwidth,
                        traceBackTime, traceBackBeam);
      if (symbol == voidSymbol) {
        // Beam end reached
        break;
      }
      // Maintain the output length now we have another symbol
      *outputLength = i + 1;
      // Store the symbol sequence starting at the end of the output and
      // tracking backwards - so in the correct order but offset as we don't
      // know how long it is until we've decoded it all
      beamOutput[maxT - 1 - i] = symbol;
    }
    // Shuffle back to the start
    if (*outputLength != maxT) {
      for (unsigned i = 0; i < *outputLength; i++) {
        beamOutput[i] = beamOutput[maxT - *outputLength + i];
      }
    }
    return true;
  }
};

template class CTCGenerateOutput<unsigned>;

} // namespace popnn
