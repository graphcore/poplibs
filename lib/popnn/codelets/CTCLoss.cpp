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

template <typename InType, typename OutType, typename SymbolType,
          bool isLastLabel>
class CTCAlpha : public Vertex {

public:
  CTCAlpha();
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  // Although the last `-` is only processed if `isLastLabel`
  Input<Vector<SymbolType>> labels; // [maxLabels]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;   // [maxT,numClasses]
  Output<Vector<OutType, ONE_PTR>> alphas;        // [maxT][extendedLabels]
  Input<Vector<OutType, ONE_PTR>> alphaPrevTime;  // [extendedLabels]
  Input<Vector<OutType, ONE_PTR>> alphaPrevLabel; // [maxT]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed in the whole data input.
  Input<unsigned short> validLabels;
  // This vertex processes the slice of the labels starting at this offset
  // and so may have nothing to do for this whole data input
  const unsigned short labelOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    // Eg: Maximum planned 10 labels, partitioned into 3: Partition sizes:4,4,2
    // But this data entry has validLabels = 6.
    // Partition 0, labelOffset=0 processes 4 items
    // Partition 1, labelOffset=4 processes 2 items
    // Partition 2, labelOffset=8 processes 0 items
    if (validLabels <= labelOffset) {
      return true;
    }
    const auto labelLength = validLabels - labelOffset > labels.size()
                                 ? labels.size()
                                 : validLabels - labelOffset;
    const auto extendedLabels = labels.size() * 2 + isLastLabel;
    // First time round reference the "previous alpha" which could be carried
    // from another vertex, or if starting up [0,-inf,-inf,...]
    auto alphaM1 = &alphaPrevTime[0];
    for (unsigned t = 0; t < maxT; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * t];
      auto probabilityM1 = &probabilities[numClasses * (t - 1)];
      auto alpha = &alphas[extendedLabels * t];
      if (t != 0) {
        alphaM1 = &alphas[extendedLabels * (t - 1)];
      }
      // 1st symbol takes its parents from the alphaPrevLabel input.
      const auto blank = static_cast<OutType>(probability[blankClass]);
      // 1st blank.  Can't skip the symbol before it so combine 2 probabilities
      alpha[0] = logMul(logAdd(alphaM1[0], alphaPrevLabel[t]), blank);
      // First symbol. Can skip the blank before it if the previous symbol is
      // different
      auto sum = logAdd(alphaM1[0], alphaM1[1]);
      if (labels[0] != prevSymbol) {
        sum = logAdd(sum, alphaPrevLabel[t]);
      }
      alpha[1] = logMul(sum, static_cast<OutType>(probability[labels[0]]));
      for (unsigned symbol = 1; symbol < labelLength; symbol++) {
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
      if constexpr (isLastLabel) {
        // The final blank entry, therefore preceded by a symbol which cannot be
        // skipped. So always combine only 2 probabilities. Last blank in a
        // sequence
        const auto idx = 2 * labelLength;
        alpha[idx] = logMul(logAdd(alphaM1[idx - 1], alphaM1[idx]), blank);
      }
    }
    return true;
  }
};

template class CTCAlpha<float, float, unsigned short, true>;
template class CTCAlpha<half, half, unsigned short, true>;
template class CTCAlpha<half, float, unsigned short, true>;

template class CTCAlpha<float, float, unsigned short, false>;
template class CTCAlpha<half, half, unsigned short, false>;
template class CTCAlpha<half, float, unsigned short, false>;

template <typename InType, typename OutType, typename SymbolType,
          bool isFirstLabel>
class CTCBeta : public Vertex {

public:
  CTCBeta();
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> labels; // [maxLabels]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;  // [maxT,numClasses]
  Output<Vector<OutType, ONE_PTR>> betas;        // [maxT,extendedLabels]
  Input<Vector<OutType, ONE_PTR>> betaPrevTime;  // [extendedLabels]
  Input<Vector<OutType, ONE_PTR>> betaPrevLabel; // [2 * maxT]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed in the whole data input.
  Input<unsigned short> validLabels;
  // This vertex processes the slice of the labels starting at this offset
  // and so may have nothing to do for this whole data input
  const unsigned short labelOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    if (validLabels <= labelOffset) {
      return true;
    }
    const auto labelLength = validLabels - labelOffset > labels.size()
                                 ? labels.size()
                                 : validLabels - labelOffset;
    const auto extendedLabels = labels.size() * 2 + isFirstLabel;
    // First time round reference the "previous beta" which could be carried
    // from another vertex, or if starting up [-inf,-inf,....0]
    // Reference this with an offset so that the last valid symbol aligns with
    // the zero when starting up.
    const auto betaOffset = 2 * (labels.size() - labelLength);
    auto betaP1 = &betaPrevTime[betaOffset];
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
      const auto blank = static_cast<OutType>(probability[blankClass]);

      for (unsigned symbol = 0; symbol < labelLength - 1; symbol++) {
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
      // Suppose we have a sequence: - a - a - b - c - .
      // The remaining part is the result for - c - . Call these X Y Z below
      // Process `X` (always needed).  As X is a blank then it is preceded
      // by a symbol which cannot be skipped and we need only 2 probabilities
      auto idx = 2 * labelLength - 2;
      beta[idx] = logMul(logAdd(betaP1[idx + 1], betaP1[idx]), blank);
      idx++;
      const auto lastSymbol = labels[labelLength - 1];
      const auto prob = static_cast<OutType>(probability[lastSymbol]);
      if constexpr (isFirstLabel) {
        // As this is a vertex to process the last Symbol, write `Y` and `Z`
        // But as it is the last symbol's vertex there is no previous label's
        // beta to use.
        beta[idx] = logMul(logAdd(betaP1[idx], betaP1[idx + 1]), prob); //`Y`
        idx++;
        beta[idx] = logMul(betaP1[idx], blank); //`Z`
      } else {
        // As this isn't a vertex to process the last Symbol, there is no `Z`
        // So just write `Y`, which uses the probabilty of the previous blank,
        // and conditionally that of the previous symbol.
        auto sum = logAdd(betaP1[idx], betaPrevLabel[t - 1]);
        if (lastSymbol != prevSymbol) {
          sum = logAdd(sum, betaPrevLabel[maxT + t - 1]);
        }
        beta[idx] = logMul(sum, prob);
      }
    }
    return true;
  }
};

template class CTCBeta<float, float, unsigned short, true>;
template class CTCBeta<half, half, unsigned short, true>;
template class CTCBeta<half, float, unsigned short, true>;

template class CTCBeta<float, float, unsigned short, false>;
template class CTCBeta<half, half, unsigned short, false>;
template class CTCBeta<half, float, unsigned short, false>;

template <typename InType, typename OutType, typename SymbolType,
          bool isFirstLabel>
class CTCGradGivenAlpha : public Vertex {

public:
  CTCGradGivenAlpha();
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> labels; // [maxLabels]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;  // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> alphas;        // [maxT,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> betaPrevTime;  // [2,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> betaPrevLabel; // [2*maxT]
  InOut<Vector<OutType, ONE_PTR>> grads;         // [maxT,numClasses]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed in the whole data input.
  Input<unsigned short> validLabels;
  // This vertex processes the slice of the labels starting at this offset
  // and so may have nothing to do for this whole data input
  const unsigned short labelOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    if (validLabels <= labelOffset) {
      return true;
    }
    const auto labelLength = validLabels - labelOffset > labels.size()
                                 ? labels.size()
                                 : validLabels - labelOffset;
    const auto extendedLabels = labels.size() * 2 + isFirstLabel;
    // First time round reference the "previous beta" in betaPrevTime[0] which
    // could be carried from another vertex, or if starting up [-inf,-inf,... 0]
    // Reference this with an offset so that the last valid symbol aligns with
    // the zero when starting up.
    const auto betaTempOffset = 2 * (labels.size() - labelLength);

    unsigned oldIdx = 0;
    for (unsigned t = maxT; t != 0; t--) {
      // References to each row, next row of the input and output
      auto probability = &probabilities[numClasses * (t - 1)];
      auto probabilityP1 = &probabilities[numClasses * t];
      auto alpha = &alphas[extendedLabels * (t - 1)];
      auto grad = &grads[numClasses * (t - 1)];
      auto beta = &betaPrevTime[betaTempOffset + (oldIdx ^ extendedLabels)];
      auto betaP1 = &betaPrevTime[betaTempOffset + oldIdx];

      const auto blank = static_cast<OutType>(probability[blankClass]);
      for (unsigned symbol = 0; symbol < labelLength - 1; symbol++) {
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
      // Suppose we have a sequence: - a - a - b - c - .
      // The remaining part is the result for - c - . Call these X Y Z below
      // Process `X` (always needed).  As X is a blank then it is preceded
      // by a symbol which cannot be skipped and we need only 2 probabilities
      auto idx = 2 * labelLength - 2;
      auto sum = logAdd(betaP1[idx + 1], betaP1[idx]);
      beta[idx] = logMul(sum, blank);
      grad[blankClass] = logAdd(logMul(sum, alpha[idx]), grad[blankClass]);
      idx++;
      const auto lastSymbol = labels[labelLength - 1];
      const auto prob = static_cast<OutType>(probability[lastSymbol]);

      if constexpr (isFirstLabel) {
        // As this is a vertex to process the last Symbol, write `Y` and `Z`
        // But as it is the last symbol's vertex there is no previous label's
        // beta to use.

        // Writing `Y`
        auto sum = logAdd(betaP1[idx], betaP1[idx + 1]);
        grad[lastSymbol] = logAdd(logMul(sum, alpha[idx]), grad[lastSymbol]);
        beta[idx] = logMul(sum, prob);
        idx++;
        // Writing `Z`
        grad[blankClass] =
            logAdd(logMul(betaP1[idx], alpha[idx]), grad[blankClass]);
        beta[idx] = logMul(betaP1[idx], blank);
      } else {
        // As this isn't a vertex to process the last Symbol, there is no `Z`
        // So just write `Y`, which uses the probabilty of the previous blank,
        // and conditionally that of the previous symbol.
        auto sum = logAdd(betaP1[idx], betaPrevLabel[t - 1]);
        if (lastSymbol != prevSymbol) {
          sum = logAdd(sum, betaPrevLabel[maxT + t - 1]);
        }
        beta[idx] = logMul(sum, prob);
        grad[lastSymbol] = logAdd(logMul(sum, alpha[idx]), grad[lastSymbol]);
      }
      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex
      betaPrevLabel[t - 1] = beta[0];
      betaPrevLabel[maxT + t - 1] = beta[1];
      // Swap new <-> old in the alphaTemp buffer
      oldIdx = oldIdx ^ extendedLabels;
    }
    return true;
  }
};

template class CTCGradGivenAlpha<float, float, unsigned short, true>;
template class CTCGradGivenAlpha<half, half, unsigned short, true>;
template class CTCGradGivenAlpha<half, float, unsigned short, true>;

template class CTCGradGivenAlpha<float, float, unsigned short, false>;
template class CTCGradGivenAlpha<half, half, unsigned short, false>;
template class CTCGradGivenAlpha<half, float, unsigned short, false>;

template <typename InType, typename OutType, typename SymbolType,
          bool isLastLabel>
class CTCGradGivenBeta : public Vertex {

public:
  CTCGradGivenBeta();
  // Labels, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> labels; // [maxLabels]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;   // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> betas;          // [maxT,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> alphaPrevTime;  // [2,extendedLabels]
  InOut<Vector<OutType, ONE_PTR>> alphaPrevLabel; // [maxT]
  InOut<Vector<OutType, ONE_PTR>> grads;          // [maxT,numClasses]
  // labels is always of size [maxLabels] - only validLabels are to be
  // processed in the whole data input.
  Input<unsigned short> validLabels;
  // This vertex processes the slice of the labels starting at this offset
  // and so may have nothing to do for this whole data input
  const unsigned short labelOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    if (validLabels <= labelOffset) {
      return true;
    }
    const auto labelLength = validLabels - labelOffset > labels.size()
                                 ? labels.size()
                                 : validLabels - labelOffset;
    const auto extendedLabels = labels.size() * 2 + isLastLabel;
    // First time round reference the "previous alpha" in alphaPrevTime[0] which
    // could be carried from another vertex, or if starting up [0,-inf,-inf,...]
    unsigned oldIdx = 0;

    // Beta (calculated by a previous vertex) is stored so that any results
    // related to non-existent labels lie at the start of the tensor.  This
    // happens when this batch validLabels < maximum planned for
    const auto betaOffset = 2 * (labels.size() - labelLength);
    for (unsigned t = 0; t < maxT; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * t];
      auto probabilityM1 = &probabilities[numClasses * (t - 1)];
      auto beta = &betas[betaOffset + extendedLabels * t];
      auto grad = &grads[numClasses * t];
      auto alpha = &alphaPrevTime[oldIdx ^ extendedLabels];
      auto alphaM1 = &alphaPrevTime[oldIdx];
      // 1st symbol takes its parents from the alphaPrevLabel input.
      const auto blank = static_cast<OutType>(probability[blankClass]);
      // 1st blank.  Can't skip the symbol before it so combine 2 probabilities
      auto sum = logAdd(alphaM1[0], alphaPrevLabel[t]);
      alpha[0] = logMul(sum, blank);
      grad[blankClass] = logAdd(logMul(sum, beta[0]), grad[blankClass]);
      // First symbol. Can skip the blank before it if the previous symbol is
      // different
      sum = logAdd(alphaM1[0], alphaM1[1]);
      if (labels[0] != prevSymbol) {
        sum = logAdd(sum, alphaPrevLabel[t]);
      }
      alpha[1] = logMul(sum, static_cast<OutType>(probability[labels[0]]));
      grad[labels[0]] = logAdd(logMul(sum, beta[1]), grad[labels[0]]);

      for (unsigned symbol = 1; symbol < labelLength; symbol++) {
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
      if constexpr (isLastLabel) {
        const auto idx = 2 * labelLength;
        sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        alpha[idx] = logMul(sum, blank);
        grad[blankClass] = logAdd(logMul(sum, beta[idx]), grad[blankClass]);
      }
      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex
      alphaPrevLabel[t] = alpha[2 * labelLength - 1];
      // Swap new <-> old in the alphaPrevTime buffer
      oldIdx = oldIdx ^ extendedLabels;
    }
    return true;
  }
};

template class CTCGradGivenBeta<float, float, unsigned short, true>;
template class CTCGradGivenBeta<half, half, unsigned short, true>;
template class CTCGradGivenBeta<half, float, unsigned short, true>;

template class CTCGradGivenBeta<float, float, unsigned short, false>;
template class CTCGradGivenBeta<half, half, unsigned short, false>;
template class CTCGradGivenBeta<half, float, unsigned short, false>;

} // namespace popnn
