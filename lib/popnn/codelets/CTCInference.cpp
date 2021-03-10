// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "LogOps.hpp"

#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/LogArithmetic.hpp"
#include "poplibs_support/TileConstants.hpp"

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
class GenerateCandidates : public Vertex {

public:
  GenerateCandidates();
  Input<Vector<InType, ONE_PTR>> logProbs; // [maxT, numClassIncBlank]
  Input<Vector<SymbolType, ONE_PTR>> lastBeamOutputs;    // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbNonBlank; // [beamwidth]
  Input<Vector<PartialsType, ONE_PTR>> beamProbBlank;    // [beamwidth]

  // Index into logProbs for HEAD position
  Input<unsigned> currentTimestep;

  // [beamwidth * (1 + numClasses)] -> per beam (copy + extend)
  Output<Vector<unsigned, ONE_PTR>> candidateParent;
  Output<Vector<SymbolType, ONE_PTR>> candidateAddend;
  Output<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  Output<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;

  const unsigned numClassesIncBlank;
  const SymbolType blankClass;
  const unsigned beamwidth;

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

      // Extend beams ---
      // Where we extend a beam by adding a symbol
      for (SymbolType s = 0; s < numClassesIncBlank; s++) {
        if (s == blankClass) {
          continue;
        }
        // Extending a beam ending in a blank with a non-blank symbol
        // e.g. beam: "a-", addend: "a" -> output: "aa" (extended by the
        // same symbol)
        const auto addendProb =
            static_cast<PartialsType>(logProbs[baseOffset + s]);
        const auto extendingBlankProb =
            logMul<PartialsType>(beamProbBlank[beamIdx], addendProb);

        candidateParent[candidateIdx] = beamIdx;
        candidateAddend[candidateIdx] = s;
        candidateBeamProbNonBlank[candidateIdx] = extendingBlankProb;
        candidateBeamProbBlank[candidateIdx] = log::probabilityZero;

        // Extending a beam ending in a non-blank with a different
        // non-blank symbol
        // e.g. beam: "a", addend: "b" -> output: "ab"
        if (prevSymbol != s) {
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
    }

    return true;
  }
};

template class GenerateCandidates<float, float, unsigned>;
template class GenerateCandidates<half, float, unsigned>;
template class GenerateCandidates<half, half, unsigned>;

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
class MergeCandidates : public Vertex {

public:
  MergeCandidates();
  // [beamwidth * (1 + numClasses)]
  InOut<Vector<unsigned, ONE_PTR>> candidateParent;
  InOut<Vector<SymbolType, ONE_PTR>> candidateAddend;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbNonBlank;
  InOut<Vector<PartialsType, ONE_PTR>> candidateBeamProbBlank;

  Input<Vector<SymbolType, ONE_PTR>> beamAddend; // [maxT, beamwidth]
  Input<Vector<unsigned, ONE_PTR>> beamParent;   // [maxT, beamwidth]

  // Index into beamAddend/Parent for HEAD position
  Input<unsigned> currentTimestep;

  const unsigned totalCandidates;
  const SymbolType blankClass;
  const unsigned beamwidth;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // TODO Stop duplicating this defn
    const auto voidSymbol = std::numeric_limits<SymbolType>::max();

    if (currentTimestep == 0) { // No beams are mergeable at t = 0
      return true;
    }
    for (unsigned j = 0; j < totalCandidates; j++) {
      for (unsigned i = j + 1; i < totalCandidates; i++) {
        // The only way for candidates to become mergeable is if one is a copy
        // beam (same output sequence from parent beam), and the other extension
        // (of a different beam). This is from; if both candidates are copy, the
        // output sequence of parent beams will be unchanged so not made
        // equivalent. Or alternatively, both are extension, they will need to
        // be from the same output sequence which means they cannot be extending
        // by the same symbol and so not equivalent.

        const auto parentLhs = candidateParent[j];
        const auto parentRhs = candidateParent[i];
        const auto addendLhs = candidateAddend[j];
        const auto addendRhs = candidateAddend[i];

        if ((parentLhs == parentRhs) || // From the same beam
            (addendLhs == voidSymbol && addendRhs == voidSymbol) || // Both copy
            (addendLhs != voidSymbol &&
             addendRhs != voidSymbol)) { // Both extend
          continue;
        }

        // TODO: improve efficiency
        if (equivalentOutputSequence(&(beamAddend[0]), &(beamParent[0]),
                                     *currentTimestep, beamwidth, parentLhs,
                                     addendLhs, parentRhs, addendRhs)) {

          candidateBeamProbNonBlank[j] = logAdd(candidateBeamProbNonBlank[j],
                                                candidateBeamProbNonBlank[i]);
          candidateBeamProbBlank[j] =
              logAdd(candidateBeamProbBlank[j], candidateBeamProbBlank[i]);

          candidateBeamProbNonBlank[i] = log::probabilityZero;
          candidateBeamProbBlank[i] = log::probabilityZero;
        }
      }
    }
    return true;
  }
};

template class MergeCandidates<float, float, unsigned>;
template class MergeCandidates<half, float, unsigned>;
template class MergeCandidates<half, half, unsigned>;

} // namespace popnn
