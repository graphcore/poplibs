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
inline FPType logAdd(const FPType a_, const FPType b_) {
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
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> labels;             // [maxLabels]
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,numClasses]
  Output<Vector<OutType, ONE_PTR>> alphas;      // [maxT][extendedLabels]
  Input<Vector<OutType, ONE_PTR>> alphaTemp;    // [extendedLabels]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed
  Input<unsigned short> validLabels;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedLabels = labels.size() * 2 + 1;
    // First time round reference the "previous alpha" which could be carried
    // from another vertex, or if starting up [0,-inf,-inf,...]
    auto alphaM1 = &alphaTemp[0];
    for (unsigned t = 0; t < maxT; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * t];
      auto probabilityM1 = &probabilities[numClasses * (t - 1)];
      auto alpha = &alphas[extendedLabels * t];
      if (t != 0) {
        alphaM1 = &alphas[extendedLabels * (t - 1)];
      }
      // 1st symbol has fewer possible parents as it's on the top row
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankClass]);
      alpha[0] = logMul(alphaM1[0], blank);
      alpha[1] = logMul(logAdd(alphaM1[0], alphaM1[1]),
                        static_cast<OutType>(probability[labels[0]]));

      for (unsigned symbol = 1; symbol < validLabels; symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the labels[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        alpha[idx] = logMul(logAdd(alphaM1[idx - 1], alphaM1[idx]), blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        auto sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        if (labels[symbol] != labels[symbol - 1]) {
          sum = logAdd(sum, alphaM1[idx - 2]);
        }
        alpha[idx] =
            logMul(sum, static_cast<OutType>(probability[labels[symbol]]));
      }
      // The final blank entry, therefore preceded by a symbol which cannot be
      // skipped. So always combine only 2 probabilities. Last blank in a
      // sequence
      const auto idx = 2 * validLabels;
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
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> labels;             // [maxLabels]
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,numClasses]
  Output<Vector<OutType, ONE_PTR>> betas;       // [maxT,extendedLabels]
  Input<Vector<OutType, ONE_PTR>> betaTemp;     // [extendedLabels]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed
  Input<unsigned short> validLabels;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedLabels = labels.size() * 2 + 1;
    // First time round reference the "previous beta" which could be carried
    // from another vertex, or if starting up [-inf,-inf,....0]
    // Reference this with an offset so that the last valid symbol aligns with
    // the zero when starting up.
    const auto betaOffset = 2 * (labels.size() - validLabels);
    auto betaP1 = &betaTemp[betaOffset];
    for (unsigned t = maxT; t != 0; t--) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * (t - 1)];
      auto probabilityP1 = &probabilities[numClasses * t];
      // Write using an offset into the beta array so we can carry data from
      // vertex to vertex when there are < maxLabels in the sequence
      auto beta = &betas[betaOffset + extendedLabels * (t - 1)];
      if (t != maxT) {
        betaP1 = &betas[betaOffset + extendedLabels * t];
      }
      // last symbol has fewer possible parents as it's on the bottom row
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankClass]);
      const auto symbolIdx = 2 * validLabels;
      beta[symbolIdx] = logMul(betaP1[symbolIdx], blank);
      beta[symbolIdx - 1] =
          logMul(logAdd(betaP1[symbolIdx], betaP1[symbolIdx - 1]),
                 static_cast<OutType>(probability[labels[validLabels - 1]]));

      for (unsigned symbol = 0; symbol < validLabels - 1; symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the labels[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        beta[idx] = logMul(logAdd(betaP1[idx + 1], betaP1[idx]), blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        auto sum = logAdd(betaP1[idx + 1], betaP1[idx]);
        if (labels[symbol] != labels[symbol + 1]) {
          sum = logAdd(sum, betaP1[idx + 2]);
        }
        beta[idx] =
            logMul(sum, static_cast<OutType>(probability[labels[symbol]]));
      }
      // The remaining blank entry, therefore preceded by a symbol which cannot
      // be skipped. So always combine only 2 probabilities. Not the last but
      // in a sequence: - a - a - b - c -
      //                            ^ This one
      const auto idx = 2 * validLabels - 2;
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
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> labels;             // [maxLabels]
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> alphas;       // [maxT,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> betaTemp;     // [2,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> grads;        // [maxT,numClasses]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed
  Input<unsigned short> validLabels;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedLabels = labels.size() * 2 + 1;
    // First time round reference the "previous beta" in betaTemp[0] which
    // could be carried from another vertex, or if starting up [-inf,-inf,... 0]
    // Reference this with an offset so that the last valid symbol aligns with
    // the zero when starting up.
    const auto betaTempOffset = 2 * (labels.size() - validLabels);

    unsigned oldIdx = 0;
    for (unsigned t = maxT; t != 0; t--) {
      // References to each row, next row of the input and output
      auto probability = &probabilities[numClasses * (t - 1)];
      auto probabilityP1 = &probabilities[numClasses * t];
      auto alpha = &alphas[extendedLabels * (t - 1)];
      auto grad = &grads[numClasses * (t - 1)];
      auto beta = &betaTemp[betaTempOffset + (oldIdx ^ extendedLabels)];
      auto betaP1 = &betaTemp[betaTempOffset + oldIdx];
      // last symbol has fewer possible parents as it's at the end
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankClass]);
      const auto symbolIdx = 2 * validLabels;
      grad[blankClass] =
          logAdd(logMul(betaP1[symbolIdx], alpha[symbolIdx]), grad[blankClass]);
      beta[symbolIdx] = logMul(betaP1[symbolIdx], blank);
      auto sum = logAdd(betaP1[symbolIdx], betaP1[symbolIdx - 1]);
      const auto lastSymbol = labels[validLabels - 1];
      grad[lastSymbol] =
          logAdd(logMul(sum, alpha[symbolIdx - 1]), grad[lastSymbol]);
      beta[symbolIdx - 1] =
          logMul(sum, static_cast<OutType>(probability[lastSymbol]));

      for (unsigned symbol = 0; symbol < validLabels - 1; symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the labels[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        auto sum = logAdd(betaP1[idx], betaP1[idx + 1]);
        grad[blankClass] = logAdd(logMul(sum, alpha[idx]), grad[blankClass]);
        beta[idx] = logMul(sum, blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        sum = logAdd(betaP1[idx + 1], betaP1[idx]);
        if (labels[symbol] != labels[symbol + 1]) {
          sum = logAdd(sum, betaP1[idx + 2]);
        }
        grad[labels[symbol]] =
            logAdd(logMul(sum, alpha[idx]), grad[labels[symbol]]);
        beta[idx] =
            logMul(sum, static_cast<OutType>(probability[labels[symbol]]));
      }
      // The remaining blank entry, therefore preceded by a symbol which cannot
      // be skipped. So always combine only 2 probabilities. Not the last but
      // in a sequence: - a - a - b - c -
      //                            ^ This one
      const auto idx = 2 * validLabels - 2;
      sum = logAdd(betaP1[idx + 1], betaP1[idx]);
      beta[idx] = logMul(sum, blank);
      grad[blankClass] = logAdd(logMul(sum, alpha[idx]), grad[blankClass]);

      // Swap new <-> old in the alphaTemp buffer
      oldIdx = oldIdx ^ extendedLabels;
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
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> labels;             // [maxLabels]
  Input<Vector<InType, ONE_PTR>> probabilities; // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> betas;        // [maxT,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> alphaTemp;    // [2,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> grads;        // [maxT,numClasses]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed
  Input<unsigned short> validLabels;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    const auto extendedLabels = labels.size() * 2 + 1;
    // First time round reference the "previous alpha" in alphaTemp[0] which
    // could be carried from another vertex, or if starting up [0,-inf,-inf,...]
    unsigned oldIdx = 0;

    // Beta (calculated by a previous vertex) is stored so that any results
    // related to non-existent labels lie at the start of the tensor.  This
    // happens when this batch validLabels < maximum planned for
    const auto betaOffset = 2 * (labels.size() - validLabels);
    for (unsigned t = 0; t < maxT; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * t];
      auto probabilityM1 = &probabilities[numClasses * (t - 1)];
      auto beta = &betas[betaOffset + extendedLabels * t];
      auto grad = &grads[numClasses * t];
      auto alpha = &alphaTemp[oldIdx ^ extendedLabels];
      auto alphaM1 = &alphaTemp[oldIdx];
      // 1st symbol has fewer possible parents as it's on the top row
      // Process preceding blank and the symbol
      const auto blank = static_cast<OutType>(probability[blankClass]);
      grad[blankClass] = logAdd(logMul(alphaM1[0], beta[0]), grad[blankClass]);
      alpha[0] = logMul(alphaM1[0], blank);
      auto sum = logAdd(alphaM1[0], alphaM1[1]);
      grad[labels[0]] = logAdd(logMul(sum, beta[1]), grad[labels[0]]);
      alpha[1] = logMul(sum, static_cast<OutType>(probability[labels[0]]));

      for (unsigned symbol = 1; symbol < validLabels; symbol++) {
        // Each loop outputs the result for the symbol and a blank, yet consumes
        // only 1 index from the labels[] input
        auto idx = 2 * symbol;
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        auto sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        grad[blankClass] = logAdd(logMul(sum, beta[idx]), grad[blankClass]);
        alpha[idx] = logMul(sum, blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        if (labels[symbol] != labels[symbol - 1]) {
          sum = logAdd(sum, alphaM1[idx - 2]);
        }
        grad[labels[symbol]] =
            logAdd(logMul(sum, beta[idx]), grad[labels[symbol]]);
        alpha[idx] =
            logMul(sum, static_cast<OutType>(probability[labels[symbol]]));
      }
      // The final blank entry, therefore preceded by a symbol which cannot be
      // skipped. So always combine only 2 probabilities. Last blank in a
      // sequence
      const auto idx = 2 * validLabels;
      sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
      alpha[idx] = logMul(sum, blank);
      grad[blankClass] = logAdd(logMul(sum, beta[idx]), grad[blankClass]);

      // Swap new <-> old in the alphaTemp buffer
      oldIdx = oldIdx ^ extendedLabels;
    }
    return true;
  }
};

template class CTCGradGivenBeta<float, float, unsigned short>;
template class CTCGradGivenBeta<half, half, unsigned short>;
template class CTCGradGivenBeta<half, float, unsigned short>;

} // namespace popnn
