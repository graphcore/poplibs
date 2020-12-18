// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace {

// Given log values, perform an equivalent `linear mul` operation
template <typename FPType>
inline FPType logMul(const FPType a, const FPType b) {
  return a + b;
}
// Given log values, perform an equivalent `linear add` operation
template <typename FPType>
FPType inline logAdd(const FPType a_, const FPType b_) {
  FPType a = a_ < b_ ? b_ : a_;
  FPType b = a_ < b_ ? a_ : b_;
  // Casting required as exp<half>() undefined
  return static_cast<FPType>(static_cast<float>(a) +
                             std::log(1 + std::exp(static_cast<float>(b - a))));
}

}; // namespace
namespace popnn {

template <typename InType, typename OutType, typename SymbolType>
class CTCAlpha : public Vertex {

public:
  CTCAlpha();
  // Symbols, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> symbols;
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,nSymbols]
  Output<Vector<OutType, ONE_PTR>> alphas;      // [maxT][extendedSymbols]
  Input<Vector<OutType, ONE_PTR>> alphaTemp;    // [extendedSymbols]
  const unsigned short maxT;
  const unsigned short nSymbols;
  const unsigned short blankSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedSymbols = symbols.size() * 2 + 1;
    // First time round reference the "previous alpha" which could be carried
    // from another vertex, or if starting up [0,-inf,-inf,...]
    auto alphaM1 = &alphaTemp[0];
    for (unsigned t = 0; t < maxT; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[nSymbols * t];
      auto probabilityM1 = &probabilities[nSymbols * (t - 1)];
      auto alpha = &alphas[extendedSymbols * t];
      if (t != 0) {
        alphaM1 = &alphas[extendedSymbols * (t - 1)];
      }
      // 1st symbol has fewer possible parents as it's on the top row
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankSymbol]);
      alpha[0] = logMul(alphaM1[0], blank);
      alpha[1] = logMul(logAdd(alphaM1[0], alphaM1[1]),
                        static_cast<OutType>(probability[symbols[0]]));

      for (unsigned symbol = 1; symbol < symbols.size(); symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the symbols[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        alpha[idx] = logMul(logAdd(alphaM1[idx - 1], alphaM1[idx]), blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        auto sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        if (symbols[symbol] != symbols[symbol - 1]) {
          sum = logAdd(sum, alphaM1[idx - 2]);
        }
        alpha[idx] =
            logMul(sum, static_cast<OutType>(probability[symbols[symbol]]));
      }
      // The final blank entry, therefore preceded by a symbol which cannot be
      // skipped. So always combine only 2 probabilities. Last blank in a
      // sequence
      const auto idx = 2 * symbols.size();
      alpha[idx] = logMul(logAdd(alphaM1[idx - 1], alphaM1[idx]), blank);
    }
    return true;
  }
};

template class CTCAlpha<float, float, unsigned short>;
template class CTCAlpha<half, half, unsigned short>;
template class CTCAlpha<half, float, unsigned short>;

template <typename InType, typename OutType, typename SymbolType>
class CTCBeta : public Vertex {

public:
  CTCBeta();
  // Symbols, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> symbols;
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,nSymbols]
  Output<Vector<OutType, ONE_PTR>> betas;       // [maxT,extendedSymbols]
  Input<Vector<OutType, ONE_PTR>> betaTemp;     // [extendedSymbols]
  const unsigned short maxT;
  const unsigned short nSymbols;
  const unsigned short blankSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedSymbols = symbols.size() * 2 + 1;
    // First time round reference the "previous beta" which could be carried
    // from another vertex, or if starting up [-inf,-inf,....0]
    auto betaP1 = &betaTemp[0];
    for (unsigned t = maxT; t != 0; t--) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[nSymbols * (t - 1)];
      auto probabilityP1 = &probabilities[nSymbols * t];
      auto beta = &betas[extendedSymbols * (t - 1)];
      if (t != maxT) {
        betaP1 = &betas[extendedSymbols * t];
      }
      // last symbol has fewer possible parents as it's on the bottom row
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankSymbol]);
      const auto symbolIdx = extendedSymbols - 1;
      beta[symbolIdx] = logMul(betaP1[symbolIdx], blank);
      beta[symbolIdx - 1] = logMul(
          logAdd(betaP1[symbolIdx], betaP1[symbolIdx - 1]),
          static_cast<OutType>(probability[symbols[symbols.size() - 1]]));

      for (unsigned symbol = 0; symbol < symbols.size() - 1; symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the symbols[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        beta[idx] = logMul(logAdd(betaP1[idx + 1], betaP1[idx]), blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        auto sum = logAdd(betaP1[idx + 1], betaP1[idx]);
        if (symbols[symbol] != symbols[symbol + 1]) {
          sum = logAdd(sum, betaP1[idx + 2]);
        }
        beta[idx] =
            logMul(sum, static_cast<OutType>(probability[symbols[symbol]]));
      }
      // The remaining blank entry, therefore preceded by a symbol which cannot
      // be skipped. So always combine only 2 probabilities. Not the last but
      // in a sequence: - a - a - b - c -
      //                            ^ This one
      const auto idx = 2 * symbols.size() - 2;
      beta[idx] = logMul(logAdd(betaP1[idx + 1], betaP1[idx]), blank);
    }
    return true;
  }
};

template class CTCBeta<float, float, unsigned short>;
template class CTCBeta<half, half, unsigned short>;
template class CTCBeta<half, float, unsigned short>;

template <typename InType, typename OutType, typename SymbolType>
class CTCGradGivenAlpha : public Vertex {

public:
  CTCGradGivenAlpha();
  // Symbols, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> symbols;
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,nSymbols]
  Input<Vector<OutType, ONE_PTR>> alphas;       // [maxT,extendedSymbols]
  InOut<Vector<OutType, ONE_PTR>> betaTemp;     // [2,extendedSymbols]
  InOut<Vector<OutType, ONE_PTR>> grads;        // [maxT,nSymbols]
  const unsigned short maxT;
  const unsigned short nSymbols;
  const unsigned short blankSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedSymbols = symbols.size() * 2 + 1;
    // First time round reference the "previous beta" in betaTemp[0] which
    // could be carried from another vertex, or if starting up [-inf,-inf,... 0]

    unsigned oldIdx = 0;
    for (unsigned t = maxT; t != 0; t--) {
      // References to each row, next row of the input and output
      auto probability = &probabilities[nSymbols * (t - 1)];
      auto probabilityP1 = &probabilities[nSymbols * t];
      auto alpha = &alphas[extendedSymbols * (t - 1)];
      auto grad = &grads[nSymbols * (t - 1)];
      auto beta = &betaTemp[oldIdx ^ extendedSymbols];
      auto betaP1 = &betaTemp[oldIdx];
      // last symbol has fewer possible parents as it's at the end
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankSymbol]);
      const auto symbolIdx = extendedSymbols - 1;
      grad[blankSymbol] = logAdd(logMul(betaP1[symbolIdx], alpha[symbolIdx]),
                                 grad[blankSymbol]);
      beta[symbolIdx] = logMul(betaP1[symbolIdx], blank);
      auto sum = logAdd(betaP1[symbolIdx], betaP1[symbolIdx - 1]);
      grad[symbols[symbols.size() - 1]] = logAdd(
          logMul(sum, alpha[symbolIdx - 1]), grad[symbols[symbols.size() - 1]]);
      beta[symbolIdx - 1] = logMul(
          sum, static_cast<OutType>(probability[symbols[symbols.size() - 1]]));

      for (unsigned symbol = 0; symbol < symbols.size() - 1; symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the symbols[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        auto sum = logAdd(betaP1[idx], betaP1[idx + 1]);
        grad[blankSymbol] = logAdd(logMul(sum, alpha[idx]), grad[blankSymbol]);
        beta[idx] = logMul(sum, blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        sum = logAdd(betaP1[idx + 1], betaP1[idx]);
        if (symbols[symbol] != symbols[symbol + 1]) {
          sum = logAdd(sum, betaP1[idx + 2]);
        }
        grad[symbols[symbol]] =
            logAdd(logMul(sum, alpha[idx]), grad[symbols[symbol]]);
        beta[idx] =
            logMul(sum, static_cast<OutType>(probability[symbols[symbol]]));
      }
      // The remaining blank entry, therefore preceded by a symbol which cannot
      // be skipped. So always combine only 2 probabilities. Not the last but
      // in a sequence: - a - a - b - c -
      //                            ^ This one
      const auto idx = 2 * symbols.size() - 2;
      sum = logAdd(betaP1[idx + 1], betaP1[idx]);
      beta[idx] = logMul(sum, blank);
      grad[blankSymbol] = logAdd(logMul(sum, alpha[idx]), grad[blankSymbol]);

      // Swap new <-> old in the alphaTemp buffer
      oldIdx = oldIdx ^ extendedSymbols;
    }
    return true;
  }
};

template class CTCGradGivenAlpha<float, float, unsigned short>;
template class CTCGradGivenAlpha<half, half, unsigned short>;
template class CTCGradGivenAlpha<half, float, unsigned short>;

template <typename InType, typename OutType, typename SymbolType>
class CTCGradGivenBeta : public Vertex {

public:
  CTCGradGivenBeta();
  // Symbols, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> symbols;
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,nSymbols]
  Input<Vector<OutType, ONE_PTR>> betas;        // [maxT,extendedSymbols]
  InOut<Vector<OutType, ONE_PTR>> alphaTemp;    // [2,extendedSymbols]
  InOut<Vector<OutType, ONE_PTR>> grads;        // [maxT,nSymbols]
  const unsigned short maxT;
  const unsigned short nSymbols;
  const unsigned short blankSymbol;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedSymbols = symbols.size() * 2 + 1;
    // First time round reference the "previous alpha" in alphaTemp[0] which
    // could be carried from another vertex, or if starting up [0,-inf,-inf,...]
    unsigned oldIdx = 0;
    for (unsigned t = 0; t < maxT; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[nSymbols * t];
      auto probabilityM1 = &probabilities[nSymbols * (t - 1)];
      auto beta = &betas[extendedSymbols * t];
      auto grad = &grads[nSymbols * t];
      auto alpha = &alphaTemp[oldIdx ^ extendedSymbols];
      auto alphaM1 = &alphaTemp[oldIdx];
      // 1st symbol has fewer possible parents as it's on the top row
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankSymbol]);
      grad[blankSymbol] =
          logAdd(logMul(alphaM1[0], beta[0]), grad[blankSymbol]);
      alpha[0] = logMul(alphaM1[0], blank);
      auto sum = logAdd(alphaM1[0], alphaM1[1]);
      grad[symbols[0]] = logAdd(logMul(sum, beta[1]), grad[symbols[0]]);
      alpha[1] = logMul(sum, static_cast<OutType>(probability[symbols[0]]));

      for (unsigned symbol = 1; symbol < symbols.size(); symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the symbols[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        auto sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        grad[blankSymbol] = logAdd(logMul(sum, beta[idx]), grad[blankSymbol]);
        alpha[idx] = logMul(sum, blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        if (symbols[symbol] != symbols[symbol - 1]) {
          sum = logAdd(sum, alphaM1[idx - 2]);
        }
        grad[symbols[symbol]] =
            logAdd(logMul(sum, beta[idx]), grad[symbols[symbol]]);
        alpha[idx] =
            logMul(sum, static_cast<OutType>(probability[symbols[symbol]]));
      }
      // The final blank entry, therefore preceded by a symbol which cannot be
      // skipped. So always combine only 2 probabilities. Last blank in a
      // sequence
      const auto idx = 2 * symbols.size();
      sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
      alpha[idx] = logMul(sum, blank);
      grad[blankSymbol] = logAdd(logMul(sum, beta[idx]), grad[blankSymbol]);

      // Swap new <-> old in the alphaTemp buffer
      oldIdx = oldIdx ^ extendedSymbols;
    }
    return true;
  }
};

template class CTCGradGivenBeta<float, float, unsigned short>;
template class CTCGradGivenBeta<half, half, unsigned short>;
template class CTCGradGivenBeta<half, float, unsigned short>;

} // namespace popnn
