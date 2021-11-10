// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "LogOps.hpp"

#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include <poplibs_support/CTCInferenceDefs.hpp>
#include <poplibs_support/ExternalCodelet.hpp>
#include <poplibs_support/LogArithmetic.hpp>

#include <cmath>
#include <print.h>
#include <tuple>
#include <type_traits>

using namespace poplar;
using namespace poplibs_support;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto voidSymbol = popnn::ctc_infer::voidSymbol;
static constexpr auto invalidSymbol = popnn::ctc_infer::invalidSymbol;

namespace popnn {

template <typename InType, typename PartialsType, typename SymbolType>
class CTCGenerateCopyCandidates : public Vertex {

public:
  CTCGenerateCopyCandidates();
  Input<Vector<InType, ONE_PTR>> logProbs; // [maxT, numClassIncBlank]
  Input<Vector<SymbolType, ONE_PTR>> lastBeamOutputs;    // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbNonBlank; // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbTotal;    // [beamwidth]

  // Index into logProbs for HEAD position
  Input<unsigned> currentTimestep;
  // Complete or not, when this batch entry is finished
  Input<unsigned> complete;

  // Outputs are for a single candidate only
  Output<unsigned> candidateParent;
  Output<SymbolType> candidateAddend;
  Output<PartialsType> candidateBeamProbNonBlank;
  Output<PartialsType> candidateBeamProbBlank;
  Output<PartialsType> candidateBeamProbTotal;

  const unsigned numClassesIncBlank;
  const SymbolType blankClass;
  const unsigned beamIdx;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (complete) {
      // Guard against overprocessing an input that is smaller than maxT for
      // this batch entry.
      return true;
    }
    const auto logProbsBase =
        &logProbs[(*currentTimestep - 1) * numClassesIncBlank];
    const auto blankProb = static_cast<PartialsType>(logProbsBase[blankClass]);

    const auto prevSymbol = lastBeamOutputs[beamIdx];
    // Create the copy candidate from the specific beam

    // Copy beams are where we maintain the same beam output sequence
    // By appending a blank to beam ending in a blank
    // e.g. beam: "a-", addend: "-" -> output: "a"
    // Or by appending a blank to a beam ending in a non blank
    // e.g. beam: "a", addend: "-" -> output: "a"
    const auto prob = logMul<PartialsType>(beamProbTotal[beamIdx], blankProb);
    *candidateParent = beamIdx;
    *candidateAddend = voidSymbol;
    *candidateBeamProbNonBlank = log::probabilityZero;
    *candidateBeamProbBlank = prob;
    *candidateBeamProbTotal = prob;

    // By appending the same symbol as at the end of the beam
    // e.g. beam: "a", addend: "a" -> output: "a"
    if (prevSymbol < invalidSymbol) {
      const auto addendProb =
          static_cast<PartialsType>(logProbsBase[prevSymbol]);
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
  Input<Vector<SymbolType, ONE_PTR>> lastBeamOutputs; // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbBlank; // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbTotal; // [beamwidth]

  // Index into logProbs for HEAD position
  Input<unsigned> currentTimestep;
  // Complete or not, when this batch entry is finished
  Input<unsigned> complete;

  // Outputs have size[endBeam-startBeam],
  // the vertex generates endBeam-startBeam] candidates
  Output<Vector<unsigned, ONE_PTR>> extendCandidateParent;
  Output<Vector<SymbolType, ONE_PTR>> extendCandidateAddend;
  Output<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbNonBlank;
  Output<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbBlank;
  Output<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbTotal;

  const unsigned numClassesIncBlank;
  // Extending beams in this range
  const unsigned startBeam;
  const unsigned endBeam;
  // Creating extend beams with this symbol
  const unsigned addendSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (complete) {
      // Guard against overprocessing an input that is smaller than maxT for
      // this batch entry.
      return true;
    }
    const auto logProbsBase =
        &logProbs[(*currentTimestep - 1) * numClassesIncBlank];
    unsigned outIdx = 0;
    for (unsigned beamIdx = startBeam; beamIdx < endBeam; ++beamIdx) {
      // Extend beams ---
      // Extend the beam using addendSymbol
      // Extending a beam ending in a blank with a non-blank symbol
      // e.g. beam: "a-", addend: "a" -> output: "aa" (extended by the
      // same symbol)
      // Extending a beam ending in a non-blank with a different
      // non-blank symbol
      // e.g. beam: "a", addend: "b" -> output: "ab"
      extendCandidateParent[outIdx] = beamIdx;
      extendCandidateAddend[outIdx] = addendSymbol;
      extendCandidateBeamProbBlank[outIdx] = log::probabilityZero;

      const auto addendProb =
          static_cast<PartialsType>(logProbsBase[addendSymbol]);

      const auto prevSymbol = lastBeamOutputs[beamIdx];

      // If prevSymbol == addendSymbol we extend only the beam ending in blank,
      // and so use beamProbBlank.
      // If prevSymbol != addendSymbol we extend both the beam ending in blank
      // and in non-blank, and so use beamProbTotal.
      const auto extendingBlankProb =
          (prevSymbol == addendSymbol)
              ? logMul<PartialsType>(beamProbBlank[beamIdx], addendProb)
              : logMul<PartialsType>(beamProbTotal[beamIdx], addendProb);

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
// return {symbol, beamIndex}
inline std::tuple<SymbolType, unsigned>
getNextSymbol(const SymbolType *beamAddend, const unsigned *beamParent,
              unsigned beamIndex) {
  auto symbol = beamAddend[beamIndex];
  beamIndex = beamParent[beamIndex];
  return {symbol, beamIndex};
}

template <typename SymbolType>
inline bool equivalentOutputSequence(const SymbolType *beamAddend,
                                     const unsigned *beamParent, unsigned t,
                                     unsigned beamwidth, unsigned parentLhs,
                                     SymbolType addendLhs, unsigned parentRhs,
                                     SymbolType addendRhs) {

  // Assumption that addendLhs != addendRhs, by design we shouldn't be
  // comparing two addends with classes which are both not voidSymbol
  auto beamIndexLhs = parentLhs + (t - 1) * beamwidth;
  auto beamIndexRhs = parentRhs + (t - 1) * beamwidth;
  unsigned nextSymbolLhs, nextSymbolRhs;

  // Assumption that LHS is a copy candidate so we need to get the
  // last beam output to begin the comparison
  std::tie(nextSymbolLhs, beamIndexLhs) =
      getNextSymbol(beamAddend, beamParent, beamIndexLhs);
  // Assumption that RHS is not a copy candidate so the addend is needed to
  // begin the comparison
  nextSymbolRhs = addendRhs;

  while (true) {
    if (nextSymbolLhs != nextSymbolRhs) {
      return false;
    }
    if (beamIndexLhs == beamIndexRhs) {
      return true;
    }
    std::tie(nextSymbolLhs, beamIndexLhs) =
        getNextSymbol(beamAddend, beamParent, beamIndexLhs);
    std::tie(nextSymbolRhs, beamIndexRhs) =
        getNextSymbol(beamAddend, beamParent, beamIndexRhs);
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
  // Note - total probability not required for extend candidates, because:
  // 1. They are never modified.
  // 2. When merging, the resulting copy candidate needs a updated pb and pnb
  //    pb = pbCopy + pbExtend and pnb = pnbCopy + pnbExtend
  //    and so the pTotal of the merged result can be calculated from pb, pnb

  // Note that the copy candidate parent and addend fields are updated with
  // those of the extend candidate it was merged with.  This is so that the
  // beam history can be updated but also allows us to identify the extend
  // candidate that was merged later in the process.
  InOut<unsigned> copyCandidateParent;
  InOut<SymbolType> copyCandidateAddend;
  InOut<PartialsType> copyCandidateBeamProbNonBlank;
  InOut<PartialsType> copyCandidateBeamProbBlank;
  InOut<PartialsType> copyCandidateBeamProbTotal;

  Input<Vector<SymbolType, ONE_PTR>> beamAddend; // [maxT+1, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamParent;   // [maxT+1, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamLength;   // [beamwidth]
  // The last output from the beam that the copy candidate came from
  Input<SymbolType> lastBeamOutput;

  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  // Complete or not, when this batch entry is finished
  Input<unsigned> complete;

  const unsigned beamwidth;
  const SymbolType blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // Comparison involves tracking backward through the beam history of both
    // candidates in order to generate the output symbol by symbol.  Outputs are
    // compared one by one and conclusions reached:
    // If the output symbols differ - there is no match (and exit)
    // If the parent beam is the same (and all symbols so far match) - the
    // outputs match.
    // If the beginning of one beam history is reached and not the other then
    // the lengths are difference so there is no match
    if (complete) {
      // Guard against processing an input that is smaller than maxT for
      // this batch entry. This vertex could take the longest of all batch
      // merge vertices and therefore slow the process down even when it has
      // finished the work it needs to do.
      return true;
    }

    const unsigned parentLhs = *copyCandidateParent;
    const unsigned addendLhs = *copyCandidateAddend;

    // Consider a single copy candidate and a number of extend candidates.
    // The extend candidates are all from a single beam and are listed:
    // extend[addendA]
    // extend[addendB]
    // extend[addendC]
    // ...
    // In other words in order of addend, with the blankClass being omitted.
    // The last symbol of the copy candidate is the last symbol of its parent
    // beam.  The extend candidate with the same addend is the only one that
    // need be compared.
    if (lastBeamOutput >= invalidSymbol) {
      return true;
    }
    const auto i =
        lastBeamOutput > blankClass ? lastBeamOutput - 1 : lastBeamOutput;

    const auto parentRhs = extendCandidateParent[i];
    const auto addendRhs = extendCandidateAddend[i];
    // To merge the parent of the copy candidate must have 1 less symbol than
    // the parent of the extend candidate

    // Based on the current timestep, extract the current beamLength
    // from the `beamLength` ping-pong buffer
    const auto lengthOffset = beamwidth * ((currentTimestep - 1) & 1);
    const auto currentBeamLength = &beamLength[lengthOffset];
    if (currentBeamLength[parentLhs] != currentBeamLength[parentRhs] + 1) {
      return true;
    }

    if (equivalentOutputSequence(&(beamAddend[0]), &(beamParent[0]),
                                 *currentTimestep, beamwidth, parentLhs,
                                 addendLhs, parentRhs, addendRhs)) {

      *copyCandidateBeamProbNonBlank = logAdd(
          *copyCandidateBeamProbNonBlank, extendCandidateBeamProbNonBlank[i]);
      // extendCandidateBeamProbBlank[i] is always log::probabilityZero so
      // we don't need to add it here
      *copyCandidateBeamProbTotal =
          logAdd(*copyCandidateBeamProbBlank, *copyCandidateBeamProbNonBlank);
      // Preserve the addend and parent of the extend candidate
      *copyCandidateParent = parentRhs;
      *copyCandidateAddend = addendRhs;
    }
    return true;
  }
};

template class CTCMergeCandidates<float, unsigned>;
template class CTCMergeCandidates<half, unsigned>;

template <typename PartialsType, typename SymbolType>
class CTCSelectCopyCandidates : public Vertex {

public:
  CTCSelectCopyCandidates();
  // Shape and size of inputs is [numCandidates]
  Input<Vector<unsigned, ONE_PTR>> copyCandidateParent;
  Input<Vector<SymbolType, ONE_PTR>> copyCandidateAddend;
  Input<Vector<PartialsType, ONE_PTR>> copyCandidateBeamProbNonBlank;
  Input<Vector<PartialsType, ONE_PTR>> copyCandidateBeamProbBlank;
  Input<Vector<PartialsType, ONE_PTR>> copyCandidateBeamProbTotal;

  // A single result candidate
  Output<unsigned> candidateParent;
  Output<SymbolType> candidateAddend;
  Output<PartialsType> candidateBeamProbNonBlank;
  Output<PartialsType> candidateBeamProbBlank;
  Output<PartialsType> candidateBeamProbTotal;

  // Complete or not, when this batch entry is finished
  Input<unsigned> complete;
  unsigned numCandidates;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (complete) {
      return true;
    }
    // Select a single copy candidate from those attached, the one that was
    // merged if there is one
    unsigned i;
    for (i = 0; i < numCandidates - 1; i++) {
      // Loop over each copy candidate
      if (copyCandidateAddend[i] != voidSymbol) {
        // There was a merge among the group of broadcast copy candidates so
        // select the merged one: i will indicate the merged one
        break;
      }
    }
    // Either `i` indicates the merged candidate or if no merge was found -
    // there was no merge, or the last one was merged.  In both of the
    // last 2 cases, we want the last one
    *candidateParent = copyCandidateParent[i];
    *candidateAddend = copyCandidateAddend[i];
    *candidateBeamProbNonBlank = copyCandidateBeamProbNonBlank[i];
    *candidateBeamProbBlank = copyCandidateBeamProbBlank[i];
    *candidateBeamProbTotal = copyCandidateBeamProbTotal[i];
    return true;
  }
};

template class CTCSelectCopyCandidates<float, unsigned>;
template class CTCSelectCopyCandidates<half, unsigned>;

template <typename PartialsType, typename SymbolType>
class CTCSelectExtendCandidates : public Vertex {

public:
  CTCSelectExtendCandidates();
  // The parent from each copy candidate that was compared to the extend
  // candidates.
  Input<Vector<SymbolType, ONE_PTR>> copyCandidateAddend; // [numCopyCandidates]
  // The extend candidates, which can have their total probability zeroed if
  // a merge took place
  // [numSymbols-1]
  InOut<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbTotal;
  InOut<Vector<SymbolType, ONE_PTR>> extendCandidateAddend;

  // Complete or not, when this batch entry is finished
  Input<unsigned> complete;
  // The number of copy candidates
  const unsigned numCopyCandidates;
  const SymbolType blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (complete) {
      return true;
    }
    // Where a copy candidate indicates a merge, zero the total probability
    // of the corresponding extend candidate which it was merged with
    for (unsigned i = 0; i < numCopyCandidates; i++) {
      // Loop over each copy candidate
      if (copyCandidateAddend[i] != voidSymbol) {
        // There was a merge among the group of broadcast copy candidates
        // so 'discard' the corresponding extend candidate by zeroing its
        // sum probability and setting the addend to invalidSymbol
        const auto symbol = copyCandidateAddend[i];
        const auto idx = symbol > blankClass ? symbol - 1 : symbol;
        extendCandidateBeamProbTotal[idx] = log::probabilityZero;
        extendCandidateAddend[idx] = invalidSymbol;
        // Don't exit, there could be several merges to pick up
      }
    }
    return true;
  }
};

template class CTCSelectExtendCandidates<float, unsigned>;
template class CTCSelectExtendCandidates<half, unsigned>;

template <typename PartialsType, typename SymbolType>
class CTCRankCandidates
    : public SupervisorVertexIf<std::is_same<PartialsType, float>::value &&
                                std::is_same<SymbolType, unsigned>::value &&
                                ASM_CODELETS_ENABLED> {
  static const bool needsAlignWorkers = false;

public:
  CTCRankCandidates();
  constexpr static bool isExternal() {
    return (std::is_same<PartialsType, float>::value &&
            std::is_same<SymbolType, unsigned>::value);
  }

  // Inputs have size [totalCandidates]
  Input<Vector<unsigned, ONE_PTR>> candidateParent;
  Input<Vector<SymbolType, ONE_PTR>> candidateAddend;
  Input<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  Input<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;
  // Total requires alignment to 64 bits, due to 64 bit reads in the
  // assembler vertex
  Input<Vector<PartialsType, ONE_PTR, 8>> candidateBeamProbTotal;

  // Outputs have size [beamwidth]
  // They must have 32 bit data types to avoid sub-word writes
  Output<Vector<unsigned, ONE_PTR>> rankedCandidateParent;
  Output<Vector<unsigned, ONE_PTR>> rankedCandidateAddend;
  Output<Vector<float, ONE_PTR>> rankedCandidateBeamProbNonBlank;
  Output<Vector<float, ONE_PTR>> rankedCandidateBeamProbBlank;
  Output<Vector<float, ONE_PTR>> rankedCandidateBeamProbTotal;

  const unsigned totalCandidates;
  // Complete or not, when this batch entry is finished
  Input<unsigned> complete;

  const unsigned beamwidth; // The number of result candidates = beamwidth
  const unsigned firstCandidateToRank;
  const unsigned lastCandidateToRank;

  IS_EXTERNAL_CODELET(isExternal());

  bool compute() {
    if (complete) {
      return true;
    }
    // Zero the initial content
    for (unsigned i = 0; i < beamwidth; i++) {
      rankedCandidateParent[i] = 0;
      rankedCandidateAddend[i] = 0;
      rankedCandidateBeamProbNonBlank[i] = 0.0f;
      rankedCandidateBeamProbBlank[i] = 0.0f;
      rankedCandidateBeamProbTotal[i] = 0.0f;
    }
    for (unsigned i = firstCandidateToRank; i < lastCandidateToRank; i++) {
      const auto toRankProbTotal = candidateBeamProbTotal[i];
      unsigned rankCount = 0;
      for (unsigned j = 0; j < i; j++) {
        rankCount +=
            static_cast<unsigned>(toRankProbTotal <= candidateBeamProbTotal[j]);
      }
      // Use `<` instead of `<=` in the comparison of those ordered after the
      // one we compare because:
      // For example, given {1, 3, 4, 3, 2}
      // We expect: {4, 3, 3, 2, 1}.
      // However if we use `<` throughout we get {4, 3, ?, 2, 1} as both 3's
      // are equally ranked (There is only the 4 that is < either of them)
      // If we use `<=` throughout we get {4, ?, 3, 2, 1} as both 3's
      // are equally ranked (The other 3 and the 4 is <= either of them)
      // Using the index of the current candidate as a pivot to change `<=` to
      // `<` ensures each ranking is unique
      for (unsigned j = i + 1; j < totalCandidates; j++) {
        rankCount +=
            static_cast<unsigned>(toRankProbTotal < candidateBeamProbTotal[j]);
      }
      // Ranking of this candidate is low enough (ProbTotal is large) to store
      if (rankCount < beamwidth) {
        // Total probablility is only needed as an output for multi stage sort.
        rankedCandidateParent[rankCount] = candidateParent[i];
        rankedCandidateAddend[rankCount] =
            static_cast<unsigned>(candidateAddend[i]);
        rankedCandidateBeamProbNonBlank[rankCount] =
            static_cast<float>(candidateBeamProbNonBlank[i]);
        rankedCandidateBeamProbBlank[rankCount] =
            static_cast<float>(candidateBeamProbBlank[i]);
        rankedCandidateBeamProbTotal[rankCount] =
            static_cast<float>(candidateBeamProbTotal[i]);
      }
    }

    return true;
  }
};

template class CTCRankCandidates<float, unsigned>;
template class CTCRankCandidates<half, unsigned>;

template <typename PartialsType, typename SymbolType>
class CTCReduceCandidates : public MultiVertex {
public:
  CTCReduceCandidates();
  constexpr static bool isExternal() {
    return (std::is_same<PartialsType, float>::value &&
            std::is_same<SymbolType, unsigned>::value);
  }
  // Inputs have size [totalCandidates]
  Input<Vector<unsigned, ONE_PTR, 8>> candidateParent;
  Input<Vector<unsigned, ONE_PTR, 8>> candidateAddend;
  Input<Vector<float, ONE_PTR, 8>> candidateBeamProbNonBlank;
  Input<Vector<float, ONE_PTR, 8>> candidateBeamProbBlank;
  Input<Vector<float, ONE_PTR, 8>> candidateBeamProbTotal;

  // Outputs have size [1]
  Output<unsigned> reducedCandidateParent;
  Output<SymbolType> reducedCandidateAddend;
  Output<PartialsType> reducedCandidateBeamProbNonBlank;
  Output<PartialsType> reducedCandidateBeamProbBlank;
  Output<PartialsType> reducedCandidateBeamProbTotal;

  const unsigned totalCandidates;

  IS_EXTERNAL_CODELET(isExternal());

  bool compute(unsigned wid) {
    if (wid == 0) {
      // No complete flag check - There is no downside to running this vertex
      // every time.  It won't be slower than any vertex doing the same job
      // for another batch entries data.

      // For each variable we assume only 1 of the `totalCandidates` inputs is
      // non-zero.  Adding them all together reduces this to a single result.
      auto parent = candidateParent[0];
      auto addend = candidateAddend[0];
      auto probNonBlank = candidateBeamProbNonBlank[0];
      auto probBlank = candidateBeamProbBlank[0];
      auto probTotal = candidateBeamProbTotal[0];

      for (unsigned i = 1; i < totalCandidates; i++) {
        parent += candidateParent[i];
        addend += candidateAddend[i];
        probNonBlank += candidateBeamProbNonBlank[i];
        probBlank += candidateBeamProbBlank[i];
        probTotal += candidateBeamProbTotal[i];
      }

      *reducedCandidateParent = parent;
      *reducedCandidateAddend = static_cast<SymbolType>(addend);
      *reducedCandidateBeamProbNonBlank =
          static_cast<PartialsType>(probNonBlank);
      *reducedCandidateBeamProbBlank = static_cast<PartialsType>(probBlank);
      *reducedCandidateBeamProbTotal = static_cast<PartialsType>(probTotal);
    }
    return true;
  }
};

template class CTCReduceCandidates<float, unsigned>;
template class CTCReduceCandidates<half, unsigned>;

// Create the parent and addend data.
// Data format explanation:
// Parent indices are an index into the flattened beam history, valid for both
// the parent and addend arrays. index = (beamwidth * timestep) + beam
// For example with beamwidth = 3, then beam 2 at timestep 4 has index 14
// (using 3 * 4 + 2)
// If we insert a voidSymbol then the addend is the last non void addend in
// that beam history and parent references back to the symbol before.
// Therefore when tracing back through any path we should only see a voidSymbol
// when we reach the beginning of the beam history.
// In addition there is a dummy 1st timestep inserted before the codelets are
// run.  This makes an origin point for beam 0 (the only initial valid beam) and
// a dummy symbol for each other beam.
// Example:
// P = parent Index, A = addend, - = blank, x = dummy symbol
// T*bw is the parent index for beam 0 at the specific timestep which helps in
// understanding traceback
// beamwidth = 5
//       T=0      T=1     T=2     T=3
//       T*bw=0   T*bw=5  T*bw=10 T*bw=15
// beam  P  A     P  A    P  A    P  A
// -----------------------------------
// 0     0  -     0  4    0  4    0  4
// 1     0  x     0  1    5  3    11 4
// 2     0  x     0  -    6  4    6  4
// 3     0  x     0  2    5  2    10 3
// 4     0  x     0  3    8  4    13 4
//
// Traceback:
// Beam0.  At (T=3,beam=0) Addend=4, Parent = 0, which references the origin so
//         we are done. Output = 4
// Beam1.  At (T=3,beam=1) Addend = 4, Parent = 11 (T=2,beam=1)
//         At (T=2,beam=1) Addend = 3, Parent = 5  (T=1,beam=0)
//         At (T=1,beam=0) Addend = 4, Parent = 0, which references the origin
//         so we are done. Output = 4,3,4
// Beam2.  At (T=3,beam=2) Addend = 4, Parent = 6  (T=1, beam=1)
//         At (T=1,beam=1) Addend = 1, Parent = 0, which references the origin
//         so we are done. Output = 1,4
// Tracing back for beam 3,4
// Beam 3: Output 4,3
// Beam 4: Output 4,2,4

template <typename PartialsType, typename SymbolType>
class CTCUpdate
    : public SupervisorVertexIf<std::is_same<PartialsType, float>::value &&
                                std::is_same<SymbolType, unsigned>::value &&
                                ASM_CODELETS_ENABLED> {
public:
  CTCUpdate();
  constexpr static bool isExternal() {
    return (std::is_same<PartialsType, float>::value &&
            std::is_same<SymbolType, unsigned>::value);
  }
  // Candidates
  // [beamWidth] in size, or larger but only beamWidth entries are used
  Input<Vector<unsigned, ONE_PTR>> candidateParent;
  Input<Vector<SymbolType, ONE_PTR>> candidateAddend;
  Input<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  Input<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;
  Input<Vector<PartialsType, ONE_PTR>> candidateBeamProbTotal;

  // Beams
  InOut<Vector<PartialsType, ONE_PTR>> beamProbNonBlank; // [beamwidth]
  InOut<Vector<PartialsType, ONE_PTR>> beamProbBlank;    // [beamwidth]
  InOut<Vector<PartialsType, ONE_PTR>> beamProbTotal;    // [beamwidth]
  InOut<Vector<unsigned, ONE_PTR>> beamAddend;           // [maxT+1, beamwidth]
  InOut<Vector<unsigned, ONE_PTR>> beamParent;           // [maxT+1, beamwidth]

  InOut<Vector<unsigned, ONE_PTR>> beamLength;       // [2*beamwidth]
  Output<Vector<unsigned, ONE_PTR>> lastBeamOutputs; // [beamwidth]

  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;
  // Complete or not, when this batch entry is finished, set by this vertex for
  // all the others to use
  InOut<unsigned> complete;
  const unsigned beamwidth;

  IS_EXTERNAL_CODELET(isExternal());
  bool compute() {
    if (complete) {
      // Early exit here avoids updating beams, probabilities and the count
      // and so nothing will change regardless
      return true;
    }
    *complete = currentTimestep == dataLength;
    const unsigned baseOffset = (*currentTimestep) * beamwidth;
    const auto parent = &beamParent[baseOffset];
    const auto addend = &beamAddend[baseOffset];

    const unsigned previousBaseOffset = baseOffset - beamwidth;
    const auto previousParent = &beamParent[previousBaseOffset];
    const auto previousAddend = &beamAddend[previousBaseOffset];

    // Based on the current timestep, reference the previous and new
    // beamLengths from the `beamLength` ping-pong buffer
    const auto newLengthOffset = beamwidth * (currentTimestep & 1);
    const auto newBeamLength = &beamLength[newLengthOffset];
    const auto previousBeamLength = &beamLength[newLengthOffset ^ beamwidth];

    for (unsigned i = 0; i < beamwidth; i++) {
      if (candidateAddend[i] == voidSymbol) {
        // If the candidate addend is voidSymbol - add nothing to the beam,
        // maintaining parent and last symbol at the current timestep
        newBeamLength[i] = previousBeamLength[candidateParent[i]];
        parent[i] = previousParent[candidateParent[i]];
        lastBeamOutputs[i] = previousAddend[candidateParent[i]];
      } else {
        newBeamLength[i] = previousBeamLength[candidateParent[i]] + 1;
        parent[i] = candidateParent[i] + previousBaseOffset;
        lastBeamOutputs[i] = candidateAddend[i];
      }
      addend[i] = lastBeamOutputs[i];
      if (addend[i] == invalidSymbol) {
        // Where a candidate was merged it was marked as invalidSymbol and
        // its probTotal set to log::probabilityZero.  Complete the task of
        // nulling it by setting ProbBlank, ProgNonBlank to log::probabilityZero
        // for the beam it is used to update. (Only the case in early timesteps,
        // where the beamwidth is large compared to the number of candidates
        // generated)
        beamProbNonBlank[i] = log::probabilityZero;
        beamProbBlank[i] = log::probabilityZero;
        beamProbTotal[i] = log::probabilityZero;
      } else {
        beamProbNonBlank[i] = candidateBeamProbNonBlank[i];
        beamProbBlank[i] = candidateBeamProbBlank[i];
        beamProbTotal[i] = candidateBeamProbTotal[i];
      }
    }
    return true;
  }
};

template class CTCUpdate<float, unsigned>;
template class CTCUpdate<half, unsigned>;

template <typename SymbolType> class CTCGenerateOutput : public Vertex {
public:
  CTCGenerateOutput();

  Input<Vector<unsigned, ONE_PTR>> beamAddend; // [maxT+1, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamParent; // [maxT+1, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamLength; // [2 * beamwidth]
  Input<unsigned> dataLength;
  // The actual number of valid symbols found in the beamOutput
  Output<unsigned> outputLength;
  // The valid symbol sequence
  Output<Vector<SymbolType, ONE_PTR>> beamOutput; // [maxT]
  const unsigned beamwidth;
  const unsigned beam;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    auto traceBackBeamIndex = beam + dataLength * beamwidth;

    // Based on the total timesteps processed, extract the current beamLength
    // from the `beamLength` ping-pong buffer
    *outputLength = beamLength[beam + (dataLength & 1) * beamwidth];
    auto outIdx = *outputLength - 1;
    while (true) {
      SymbolType symbol;
      std::tie(symbol, traceBackBeamIndex) =
          getNextSymbol(&beamAddend[0], &beamParent[0], traceBackBeamIndex);
      if (symbol >= invalidSymbol) {
        // Detected a voidSymbol (Beam 0 initial value), or one of the dummy
        // symbols assigned to the the other beams so:
        // Beam end reached, not always at t=0 as beams can exist with an
        // empty output for several timesteps
        if (symbol != voidSymbol) {
          *outputLength = 0;
        }
        break;
      }
      // Store the symbol sequence starting at the end of the output (given the
      // output length) and tracking backwards
      beamOutput[outIdx] = symbol;
      outIdx--;
    }
    return true;
  }
};

template class CTCGenerateOutput<unsigned>;

} // namespace popnn
