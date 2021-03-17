// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "LogOps.hpp"

#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/LogArithmetic.hpp"

#include <cassert>
#include <cmath>
#include <print.h>
#include <tuple>
#include <type_traits>

using namespace poplar;
using namespace poplibs_support;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

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

  // Outputs have size[beamwidth], the vertex generates 0 to beamwidth
  // candidates, the number of which is indicated in validCandidates
  Output<Vector<unsigned, ONE_PTR>> candidateParent;
  Output<Vector<SymbolType, ONE_PTR>> candidateAddend;
  Output<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  Output<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;
  Output<unsigned> validCandidates;

  const unsigned numClassesIncBlank;
  const SymbolType blankClass;
  const unsigned beamwidth;
  const unsigned addendSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // TODO Stop duplicating this defn
    const auto voidSymbol = std::numeric_limits<SymbolType>::max();

    const unsigned baseOffset = (*currentTimestep) * numClassesIncBlank;
    unsigned candidateIdx = 0;
    const auto blankProb =
        static_cast<PartialsType>(logProbs[baseOffset + blankClass]);

    for (unsigned beamIdx = 0; beamIdx < beamwidth; ++beamIdx) {
      const auto prevSymbol = lastBeamOutputs[beamIdx];
      // Only create copy beams which will end in the designated symbol
      if (prevSymbol != addendSymbol) {
        continue;
      }
      // Copy beams ---
      // Where we maintain the same beam output sequence
      // By appending a blank to beam ending in a blank
      // e.g. beam: "a-", addend: "-" -> output: "a"
      const auto prevBlankProb =
          logMul<PartialsType>(beamProbBlank[beamIdx], blankProb);
      // By appending a blank to a beam ending in a non blank
      // e.g. beam: "a", addend: "-" -> output: "a"
      const auto prevNonBlankProb =
          logMul<PartialsType>(beamProbNonBlank[beamIdx], blankProb);
      const auto prob = logAdd<PartialsType>(prevBlankProb, prevNonBlankProb);
      candidateParent[candidateIdx] = beamIdx;
      candidateAddend[candidateIdx] = voidSymbol;
      candidateBeamProbNonBlank[candidateIdx] = log::probabilityZero;
      candidateBeamProbBlank[candidateIdx] = prob;

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
        candidateBeamProbNonBlank[candidateIdx] = nonBlankProb;
      }
      candidateIdx++;
    }

    *validCandidates = candidateIdx;
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

  // Outputs have size[beamwidth], the vertex generates beamwidth candidates
  Output<Vector<unsigned, ONE_PTR>> candidateParent;
  Output<Vector<SymbolType, ONE_PTR>> candidateAddend;
  Output<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  Output<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;

  const unsigned numClassesIncBlank;
  const SymbolType blankClass;
  const unsigned beamwidth;
  const unsigned addendSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // TODO Stop duplicating this defn
    const auto voidSymbol = std::numeric_limits<SymbolType>::max();

    const unsigned baseOffset = (*currentTimestep) * numClassesIncBlank;
    unsigned candidateIdx = 0;
    const auto blankProb =
        static_cast<PartialsType>(logProbs[baseOffset + blankClass]);

    for (unsigned beamIdx = 0; beamIdx < beamwidth; ++beamIdx) {
      const auto prevSymbol = lastBeamOutputs[beamIdx];

      // Extend beams ---
      // Extend the beam using addendSymbol
      // Extending a beam ending in a blank with a non-blank symbol
      // e.g. beam: "a-", addend: "a" -> output: "aa" (extended by the
      // same symbol)
      const auto addendProb =
          static_cast<PartialsType>(logProbs[baseOffset + addendSymbol]);
      const auto extendingBlankProb =
          logMul<PartialsType>(beamProbBlank[beamIdx], addendProb);

      candidateParent[candidateIdx] = beamIdx;
      candidateAddend[candidateIdx] = addendSymbol;
      candidateBeamProbNonBlank[candidateIdx] = extendingBlankProb;
      candidateBeamProbBlank[candidateIdx] = log::probabilityZero;

      // Extending a beam ending in a non-blank with a different
      // non-blank symbol
      // e.g. beam: "a", addend: "b" -> output: "ab"
      if (prevSymbol != addendSymbol) {
        const auto extendingNonBlankProb =
            logMul<PartialsType>(beamProbNonBlank[beamIdx], addendProb);
        // Note: We don't need to create a new candidate as this will have the
        // same output sequence as the previous extend beam candidate
        // "(extended by a different symbol)". Here we append the new
        // symbol which is different to the symbol the beam ended with to the
        // non-blank beam.

        candidateBeamProbNonBlank[candidateIdx] = logAdd<PartialsType>(
            candidateBeamProbNonBlank[candidateIdx], extendingNonBlankProb);
      }
      candidateIdx++;
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
  const auto voidSymbol = std::numeric_limits<SymbolType>::max();
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
  const auto voidSymbol = std::numeric_limits<SymbolType>::max();
  assert(addendLhs != addendRhs);

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

template <typename InType, typename PartialsType, typename SymbolType>
class CTCMergeCandidates : public Vertex {

public:
  CTCMergeCandidates();
  // Shape/size of Inputs is [extendCandidates]
  Input<Vector<unsigned, ONE_PTR>> extendCandidateParent;
  Input<Vector<SymbolType, ONE_PTR>> extendCandidateAddend;
  Input<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbNonBlank;
  Input<Vector<PartialsType, ONE_PTR>> extendCandidateBeamProbBlank;

  Input<unsigned> copyCandidateParent;
  Input<SymbolType> copyCandidateAddend;
  InOut<PartialsType> copyCandidateBeamProbNonBlank;
  InOut<PartialsType> copyCandidateBeamProbBlank;

  Input<Vector<SymbolType, ONE_PTR>> beamAddend; // [maxT, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamParent;   // [maxT, beamwidth]

  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  Output<unsigned> invalidCandidate;

  const unsigned extendCandidates;
  const SymbolType blankClass;
  const unsigned beamwidth;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // TODO Stop duplicating this defn
    const auto voidSymbol = std::numeric_limits<SymbolType>::max();

    // Flag as all valid until we merge
    *invalidCandidate = extendCandidates;
    if (currentTimestep == 0) { // No beams are mergeable at t = 0
      return true;
    }
    const unsigned parentLhs = *copyCandidateParent;
    const unsigned addendLhs = *copyCandidateAddend;

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
        *invalidCandidate = i;
        // Only 1 can match
        break;
      }
    }
    return true;
  }
};

template class CTCMergeCandidates<float, float, unsigned>;
template class CTCMergeCandidates<half, float, unsigned>;
template class CTCMergeCandidates<half, half, unsigned>;

template <typename PartialsType, typename SymbolType>
class SelectCandidates : public Vertex {

public:
  SelectCandidates();

  // [beamwidth * (1 + numClasses)] -> per beam (copy + extend)
  InOut<Vector<unsigned, ONE_PTR>> candidateParent;
  InOut<Vector<SymbolType, ONE_PTR>> candidateAddend;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;

  // Scratch space to store Pt
  // TODO Consider doing this ahead of time if sorting is not very parallel
  Output<Vector<PartialsType, ONE_PTR>> candidateProbTotalScratch;

  const unsigned beamwidth;
  const unsigned totalCandidates;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // Precondition - candidates to be padded by previous codelets or memory
    // suitably initialized (probability zero)
    assert(beamwidth <= totalCandidates);

    for (unsigned i = 0; i < totalCandidates; i++) {
      candidateProbTotalScratch[i] =
          logAdd(candidateBeamProbNonBlank[i], candidateBeamProbBlank[i]);
    }

    for (unsigned b = 0; b < beamwidth; b++) {
      unsigned maxIdx = b;
      PartialsType max = candidateProbTotalScratch[b];
      for (unsigned i = b; i < totalCandidates; i++) {
        const auto cmp = candidateProbTotalScratch[i];
        if (cmp > max) {
          maxIdx = i;
          max = cmp;
        }
      }

      unsigned tmpParent = candidateParent[b];
      SymbolType tmpAddend = candidateAddend[b];
      PartialsType tmpBeamProbNonBlank = candidateBeamProbNonBlank[b];
      PartialsType tmpBeamProbBlank = candidateBeamProbBlank[b];
      PartialsType tmpProbTotalScratch = candidateProbTotalScratch[b];

      candidateParent[b] = candidateParent[maxIdx];
      candidateAddend[b] = candidateAddend[maxIdx];
      candidateBeamProbNonBlank[b] = candidateBeamProbNonBlank[maxIdx];
      candidateBeamProbBlank[b] = candidateBeamProbBlank[maxIdx];
      candidateProbTotalScratch[b] = candidateProbTotalScratch[maxIdx];

      candidateParent[maxIdx] = tmpParent;
      candidateAddend[maxIdx] = tmpAddend;
      candidateBeamProbNonBlank[maxIdx] = tmpBeamProbNonBlank;
      candidateBeamProbBlank[maxIdx] = tmpBeamProbBlank;
      candidateProbTotalScratch[maxIdx] = tmpProbTotalScratch;
    }

    return true;
  }
};

template class SelectCandidates<float, unsigned>;
template class SelectCandidates<half, unsigned>;

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
  InOut<Vector<unsigned, ONE_PTR>> beamAddend;           // [maxT, beamwidth]
  InOut<Vector<unsigned, ONE_PTR>> beamParent;           // [maxT, beamwidth]

  Input<unsigned> currentTimestep;
  const unsigned beamwidth;

  IS_EXTERNAL_CODELET(false);
  bool compute() {
    const unsigned baseOffset = (*currentTimestep) * beamwidth;
    for (unsigned i = 0; i < beamwidth; i++) {
      beamParent[baseOffset + i] = candidateParent[i];
      beamAddend[baseOffset + i] = candidateAddend[i];
      beamProbNonBlank[i] = candidateBeamProbNonBlank[i];
      beamProbBlank[i] = candidateBeamProbBlank[i];
    }
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
    // TODO Stop duplicating this defn
    const auto voidSymbol = std::numeric_limits<unsigned>::max();

    auto traceBackBeam = beam;
    auto traceBackTime = *currentTimestep;

    for (unsigned i = 0; i < currentTimestep + 1; i++) {
      SymbolType symbol;
      std::tie(symbol, traceBackTime, traceBackBeam) =
          getNextSymbol(&beamAddend[0], &beamParent[0], beamwidth,
                        traceBackTime, traceBackBeam);
      if (symbol == voidSymbol) {
        // Beam end reached, so capture the length
        *outputLength = i;
        break;
      }
      // Store the symbol sequence starting at the end of the output and
      // tracking backwards - so in the correct order but offset as we don't
      // know how long it is until we've decoded it all
      beamOutput[maxT - 1 - i] = symbol;
    }
    // Shuffle back to the start
    for (unsigned i = 0; i < *outputLength; i++) {
      beamOutput[i] = beamOutput[maxT - *outputLength + i];
    }
    return true;
  }
};

template class CTCGenerateOutput<unsigned>;

} // namespace popnn
