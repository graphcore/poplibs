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
    if (currentTimestep > dataLength) {
      return true;
    }
    const unsigned baseOffset = (*currentTimestep - 1) * numClassesIncBlank;
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
    if (currentTimestep > dataLength) {
      return true;
    }
    const unsigned baseOffset = (*currentTimestep - 1) * numClassesIncBlank;
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
  // The last output from the beam that the copy candidate came from
  Input<SymbolType> lastBeamOutput;

  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;

  const unsigned extendCandidates;
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
    if (currentTimestep > dataLength) {
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
    const auto copyCandidateLastOut = lastBeamOutput;
    if (copyCandidateLastOut == voidSymbol) {
      return true;
    }
    const auto i = copyCandidateLastOut > blankClass ? copyCandidateLastOut - 1
                                                     : copyCandidateLastOut;

    const auto parentRhs = extendCandidateParent[i];
    const auto addendRhs = extendCandidateAddend[i];

    // TODO: improve efficiency by simplifting traceback mechanism, possibly
    // with no voidSymbol, and therefore different length paths for each beam
    if (equivalentOutputSequence(&(beamAddend[0]), &(beamParent[0]),
                                 *currentTimestep, beamwidth, parentLhs,
                                 addendLhs, parentRhs, addendRhs)) {

      *copyCandidateBeamProbNonBlank = logAdd(
          *copyCandidateBeamProbNonBlank, extendCandidateBeamProbNonBlank[i]);
      *copyCandidateBeamProbBlank =
          logAdd(*copyCandidateBeamProbBlank, extendCandidateBeamProbBlank[i]);
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
  Input<Vector<unsigned>> copyCandidateParent;
  Input<Vector<SymbolType>> copyCandidateAddend;
  Input<Vector<PartialsType>> copyCandidateBeamProbNonBlank;
  Input<Vector<PartialsType>> copyCandidateBeamProbBlank;
  Input<Vector<PartialsType>> copyCandidateBeamProbTotal;

  // A single result candidate
  Output<unsigned> candidateParent;
  Output<SymbolType> candidateAddend;
  Output<PartialsType> candidateBeamProbNonBlank;
  Output<PartialsType> candidateBeamProbBlank;
  Output<PartialsType> candidateBeamProbTotal;

  // Only use of current timestep and dataLength is to end early
  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;
  unsigned numCandidates;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (currentTimestep > dataLength) {
      return true;
    }
    // Select a single copy candidate from those attached, the one that was
    // merged if there is one
    for (unsigned i = 0; i < numCandidates - 1; i++) {
      // Loop over each copy candidate
      if (copyCandidateAddend[i] != voidSymbol) {
        // There was a merge among the group of broadcast copy candidates so
        // select the merged one
        *candidateParent = copyCandidateParent[i];
        *candidateAddend = copyCandidateAddend[i];
        *candidateBeamProbNonBlank = copyCandidateBeamProbNonBlank[i];
        *candidateBeamProbBlank = copyCandidateBeamProbBlank[i];
        *candidateBeamProbTotal = copyCandidateBeamProbTotal[i];
        return true;
      }
    }
    // No merge, or the last one was merged.  Either way, use the last one
    const auto i = numCandidates - 1;
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
  Input<Vector<SymbolType>> copyCandidateAddend; // [numCopyCandidates]
  // The extend candidates, which can have their total probability zeroed if
  // a merge took place
  InOut<Vector<PartialsType>> extendCandidateBeamProbTotal; // [numSymbols-1]

  // Only use of current timestep and dataLength is to end early
  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;
  // The number of copy candidates
  unsigned numCopyCandidates;
  const SymbolType blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (currentTimestep > dataLength) {
      return true;
    }
    // Where a copy candidate indicates a merge, zero the total probability
    // of the corresponding extend candidate which it was merged with
    for (unsigned i = 0; i < numCopyCandidates; i++) {
      // Loop over each copy candidate
      if (copyCandidateAddend[i] != voidSymbol) {
        // There was a merge among the group of broadcast copy candidates
        // so 'discard' the corresponding extend candidate by zeroing its
        // sum probability
        const auto symbol = copyCandidateAddend[i];
        const auto idx = symbol > blankClass ? symbol - 1 : symbol;
        extendCandidateBeamProbTotal[idx] = log::probabilityZero;
        // Don't exit, there could be several merges to pick up
      }
    }
    return true;
  }
};

template class CTCSelectExtendCandidates<float, unsigned>;
template class CTCSelectExtendCandidates<half, unsigned>;

template <typename PartialsType, typename SymbolType>
class CTCSelectCandidates : public Vertex {

public:
  CTCSelectCandidates();
  // Inputs have size [totalCandidates]
  InOut<Vector<unsigned, ONE_PTR>> candidateParent;
  InOut<Vector<SymbolType, ONE_PTR>> candidateAddend;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbTotal;

  // Only use of current timestep and dataLength is to end early
  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;

  const unsigned beamwidth; // The number of result candidates = beamwidth
  const unsigned totalCandidates;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    if (currentTimestep > dataLength) {
      return true;
    }
    // Precondition - candidates to be padded by previous codelets or memory
    // suitably initialized (probability zero)
    assert(beamwidth <= totalCandidates);
    // The input is a flat list of candidates:
    // Copy candidates followed by the extend candidates. Although order isn't
    // important for the input sorting process, the 1st beamwidth candidates
    // (Equal to the number of copy candidates) will be overwritten with the
    // most probable candidates
    for (unsigned b = 0; b < beamwidth; b++) {
      unsigned maxIdx = b;
      PartialsType max = candidateBeamProbTotal[b];
      for (unsigned i = b; i < totalCandidates; i++) {
        const auto cmp = candidateBeamProbTotal[i];
        if (cmp > max) {
          maxIdx = i;
          max = cmp;
        }
      }
      // Total probablility is not needed as an output, but the displaced
      // result total probability is needed
      candidateBeamProbTotal[maxIdx] = candidateBeamProbTotal[b];

      unsigned tmpParent = candidateParent[b];
      candidateParent[b] = candidateParent[maxIdx];
      candidateParent[maxIdx] = tmpParent;

      SymbolType tmpAddend = candidateAddend[b];
      candidateAddend[b] = candidateAddend[maxIdx];
      candidateAddend[maxIdx] = tmpAddend;

      PartialsType tmpBeamProbNonBlank = candidateBeamProbNonBlank[b];
      candidateBeamProbNonBlank[b] = candidateBeamProbNonBlank[maxIdx];
      candidateBeamProbNonBlank[maxIdx] = tmpBeamProbNonBlank;

      PartialsType tmpBeamProbBlank = candidateBeamProbBlank[b];
      candidateBeamProbBlank[b] = candidateBeamProbBlank[maxIdx];
      candidateBeamProbBlank[maxIdx] = tmpBeamProbBlank;
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
  InOut<Vector<unsigned, ONE_PTR>> beamAddend;           // [maxT+1, beamwidth]
  InOut<Vector<unsigned, ONE_PTR>> beamParent;           // [maxT+1, beamwidth]

  Output<Vector<unsigned, ONE_PTR>> lastBeamOutputsScratch; //[beamWidth]

  InOut<unsigned> currentTimestep;
  // The length of the data input (Valid for this specific input)
  Input<unsigned> dataLength;
  const unsigned beamwidth;

  IS_EXTERNAL_CODELET(false);
  bool compute() {
    if (currentTimestep > dataLength) {
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
      parent[i] = candidateParent[i];
      addend[i] = candidateAddend[i];
      // Keep the output from the parent beam, which can be a new parent
      lastBeamOutputs[i] = candidateAddend[i] == voidSymbol
                               ? lastBeamOutputsScratch[candidateParent[i]]
                               : candidateAddend[i];
      beamProbNonBlank[i] = candidateBeamProbNonBlank[i];
      beamProbBlank[i] = candidateBeamProbBlank[i];
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

  Input<Vector<unsigned, ONE_PTR>> beamAddend; // [maxT+1, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamParent; // [maxT+1, beamwidth]
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
      *outputLength = i;
      // Store the symbol sequence starting at the end of the output and
      // tracking backwards - so in the correct order but offset as we don't
      // know how long it is until we've decoded it all
      beamOutput[maxT - 1 - i] = symbol;
    }
    // Shuffle back to the start
    *outputLength = *outputLength + 1;
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
