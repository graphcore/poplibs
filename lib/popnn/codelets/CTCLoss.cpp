// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/LogArithmetic.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;
using namespace poplibs_support;

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

// Limit to the desired maximum
inline unsigned clamp(unsigned value, unsigned offset, unsigned limit) {
  assert(!(value < offset));
  return (value - offset > limit) ? limit : value - offset;
}
}; // namespace
namespace popnn {

template <typename InType, typename OutType, typename SymbolType,
          bool isLastLabel>
class CTCAlpha : public Vertex {

public:
  CTCAlpha();
  // Label, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  // Although the last `-` is only processed in one vertex when `label` is split
  // between vertices
  Input<Vector<SymbolType>> label; // [maxLabel]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;   // [maxT,numClasses]
  Output<Vector<OutType, ONE_PTR>> alphas;        // [maxT][extendedLabel]
  Input<Vector<OutType, ONE_PTR>> alphaPrevTime;  // [extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> alphaPrevLabel; // [maxT]
  // This vertex processes a labelSlice with size[label.size()] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned short> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned short> validTime;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    // Eg: Maximum planned 10 labelLength, partitioned into 3:
    // Partition sizes:4,4,2
    // But this data entry has validLabel = 6.
    // Partition 0, labelOffset=0 processes 4 items
    // Partition 1, labelOffset=4 processes 2 items
    // Partition 2, labelOffset=8 processes 0 items
    if (validLabel < labelOffset || validTime <= timeOffset) {
      return true;
    }
    const auto labelLength = clamp(validLabel, labelOffset, label.size());
    const auto timeSteps = clamp(validTime, timeOffset, maxT);
    bool doLastBlank = labelLength != label.size() || isLastLabel;
    const auto extendedLabel = label.size() * 2 + isLastLabel;
    // First time round reference the "previous alpha" which could be carried
    // from another vertex, or if starting up [0,-inf,-inf,...]
    auto alphaM1 = &alphaPrevTime[0];
    for (unsigned t = 0; t < timeSteps; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * t];
      auto alpha = &alphas[extendedLabel * t];
      if (t != 0) {
        alphaM1 = &alphas[extendedLabel * (t - 1)];
      }
      // 1st symbol takes its parents from the alphaPrevLabel input.
      const auto blank = static_cast<OutType>(probability[blankClass]);
      // 1st blank.  Can't skip the symbol before it so combine 2 probabilities
      alpha[0] = logMul(logAdd(alphaM1[0], alphaPrevLabel[t]), blank);
      if (!labelLength) {
        // Although we need to process the blank above this (as it could be the
        // "last blank" there is no part of the label to process so continue
        // (not break)
        continue;
      }
      // First symbol. Can skip the blank before it if the previous symbol is
      // different
      auto sum = logAdd(alphaM1[0], alphaM1[1]);
      if (label[0] != prevSymbol) {
        sum = logAdd(sum, alphaPrevLabel[t]);
      }
      alpha[1] = logMul(sum, static_cast<OutType>(probability[label[0]]));
      // Each loop outputs the result for the symbol and a blank, and consumes 1
      // index from the label[] input.
      unsigned idx = 2;
      for (unsigned symbol = 1; symbol < labelLength; symbol++) {
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        alpha[idx] = logMul(logAdd(alphaM1[idx - 1], alphaM1[idx]), blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        auto sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        if (label[symbol] != label[symbol - 1]) {
          sum = logAdd(sum, alphaM1[idx - 2]);
        }
        alpha[idx] =
            logMul(sum, static_cast<OutType>(probability[label[symbol]]));
        idx++;
      }
      if (doLastBlank) {
        // The final blank entry, therefore preceded by a symbol which cannot
        // be skipped. So always combine only 2 probabilities. Last blank in a
        // sequence
        alpha[idx] = logMul(logAdd(alphaM1[idx - 1], alphaM1[idx]), blank);
      }
      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex
      alphaPrevLabel[t] = alpha[2 * labelLength - 1];
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
  // Label, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> label; // [maxLabel]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;  // [maxT,numClasses]
  Output<Vector<OutType, ONE_PTR>> betas;        // [maxT,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> betaPrevTime;  // [extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> betaPrevLabel; // [maxT,2]
  // This vertex processes a labelSlice with size[label.size()] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned short> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned short> validTime;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    if (validLabel < labelOffset) {
      return true;
    }
    const auto extendedLabel = label.size() * 2 + isFirstLabel;
    if (validTime <= timeOffset) {
      // There is nothing to do, but propogate beta for the next vertex call
      // (Write the 1st beta timeslice)
      for (unsigned i = 0; i < extendedLabel; i++) {
        betas[i] = betaPrevTime[i];
      }
      return true;
    }
    const auto labelLength = clamp(validLabel, labelOffset, label.size());
    const auto timeSteps = clamp(validTime, timeOffset, maxT);
    bool doLastBlank = labelLength != label.size() || isFirstLabel;
    bool isLastLabel = (labelLength + labelOffset) == validLabel;
    bool doLastTimeStep = (timeSteps + timeOffset) == validTime;
    if (doLastTimeStep) {
      // If the last timestep we can insert an initial 0 (probability=1)
      if (doLastBlank) {
        // This partition is responsible for the last blank so insert prob=1
        // in the previous timeslice input to initiate the calculation
        betaPrevTime[2 * labelLength] = log::probabilityOne;
      } else if (isLastLabel) {
        // This partition is responsible for the isLastLabel but not the last
        // blank so insert the 0 in the previous timeslice input to initiate
        // the calculation
        betaPrevLabel[2 * (timeSteps - 1)] = log::probabilityOne;
      }
    }
    // First time round reference the "previous beta" which could be carried
    // from another vertex, or if starting up [-inf,-inf,....-inf] with the
    // 0 inserted in the correct place above
    auto betaP1 = &betaPrevTime[0];
    for (unsigned t = timeSteps; t != 0; t--) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * (t - 1)];
      // Write using an offset into the beta array so we can carry data from
      // vertex to vertex when there are < maxLabel in the sequence
      auto beta = &betas[0 + extendedLabel * (t - 1)];
      if (t != timeSteps) {
        betaP1 = &betas[0 + extendedLabel * t];
      }
      const auto blank = static_cast<OutType>(probability[blankClass]);

      // Suppose we have a sequence: - a - a - b - c - .
      // The non loop part is the result for - c - . Call these X Y Z below
      if (!labelLength) {
        // Just the `Z` on this tile, nothing else
        beta[0] = logMul(betaP1[0], blank); //`Z`
        // For use when data is split by label, output this timestep, last label
        // result for use by the next vertex. Include prob=0 to avoid special
        // cases when this is used later
        betaPrevLabel[2 * (t - 1)] = beta[0];
        betaPrevLabel[1 + 2 * (t - 1)] = log::probabilityZero;
        continue;
      }
      // Each loop outputs the result for the symbol and a blank, and consumes 1
      // index from the label[] input.
      unsigned idx = 0;
      for (unsigned symbol = 0; symbol < labelLength - 1; symbol++) {
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        beta[idx] = logMul(logAdd(betaP1[idx + 1], betaP1[idx]), blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        auto sum = logAdd(betaP1[idx + 1], betaP1[idx]);
        if (label[symbol] != label[symbol + 1]) {
          sum = logAdd(sum, betaP1[idx + 2]);
        }
        beta[idx] =
            logMul(sum, static_cast<OutType>(probability[label[symbol]]));
        idx++;
      }
      // Process `X` (always needed).  As X is a blank then it is preceded
      // by a symbol which cannot be skipped and we need only 2 probabilities
      beta[idx] = logMul(logAdd(betaP1[idx + 1], betaP1[idx]), blank);
      idx++;
      const auto lastSymbol = label[labelLength - 1];
      const auto prob = static_cast<OutType>(probability[lastSymbol]);
      if (doLastBlank) {
        // Write `Y` and `Z`. There is no previous label's beta to use.
        // Writing `Y`
        beta[idx] = logMul(logAdd(betaP1[idx], betaP1[idx + 1]), prob);
        idx++;
        beta[idx] = logMul(betaP1[idx], blank); //`Z`
      } else {
        // We are not writing the last blank (`Z`) So just write `Y`, which
        // uses the probabilty of the previous blank, and conditionally that of
        // the previous symbol.
        auto sum = logAdd(betaP1[idx], betaPrevLabel[2 * (t - 1)]);
        beta[idx] = logMul(sum, prob);
        if (lastSymbol != prevSymbol) {
          sum = logAdd(sum, betaPrevLabel[1 + 2 * (t - 1)]);
        }
        beta[idx] = logMul(sum, prob);
      }
      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex
      betaPrevLabel[2 * (t - 1)] = beta[0];
      betaPrevLabel[1 + 2 * (t - 1)] = beta[1];
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
  // Label, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> label; // [maxLabel]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;  // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> alphas;        // [maxT,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> betaPrevTime;  // [2,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> betaPrevLabel; // [maxT,2]
  InOut<Vector<OutType, ONE_PTR>> grads;         // [maxT,numClasses]
  // This vertex processes a labelSlice with size[label.size()] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned short> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned short> validTime;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    if (validLabel < labelOffset || validTime <= timeOffset) {
      return true;
    }
    const auto extendedLabel = label.size() * 2 + isFirstLabel;
    const auto labelLength = clamp(validLabel, labelOffset, label.size());
    const auto timeSteps = clamp(validTime, timeOffset, maxT);
    bool doLastBlank = labelLength != label.size() || isFirstLabel;
    bool isLastLabel = (labelLength + labelOffset) == validLabel;
    bool doLastTimeStep = (timeSteps + timeOffset) == validTime;
    if (doLastTimeStep) {
      // If the last timestep we can insert an initial 0 (probability=1)
      if (doLastBlank) {
        // This partition is responsible for the last blank so insert prob=1
        // in the previous timeslice input to initiate the calculation.
        // Inserting in both timeslices as we may need to start processing from
        // the other one
        betaPrevTime[2 * labelLength] = log::probabilityOne;
        betaPrevTime[2 * labelLength + extendedLabel] = log::probabilityOne;
      } else if (isLastLabel) {
        // This partition is responsible for the isLastLabel but not the last
        // blank so insert the 0 in the previous timeslice input to initiate
        // the calculation
        betaPrevLabel[2 * (timeSteps - 1)] = log::probabilityOne;
      }
    }
    // If maxT is odd and timeSteps also odd, the ping pong effect in beta[0:1]
    // is as we expect (output ends in beta[1]). However if timeSteps is even,
    // the output ends in the incorrect column (beta[0]). To ammend for this,
    // we start processing from the other column (beta[1], than usual beta[0])
    unsigned oldIdx = 0;
    if ((timeSteps & 1) ^ (maxT & 1)) {
      // Offset a flat[2][extendedLabel] array to reference beta[1][0]
      oldIdx = extendedLabel;
    }
    for (unsigned t = timeSteps; t != 0; t--) {
      // References to each row, next row of the input and output
      auto probability = &probabilities[numClasses * (t - 1)];
      auto alpha = &alphas[extendedLabel * (t - 1)];
      auto grad = &grads[numClasses * (t - 1)];
      auto beta = &betaPrevTime[oldIdx ^ extendedLabel];
      auto betaP1 = &betaPrevTime[oldIdx];

      const auto blank = static_cast<OutType>(probability[blankClass]);
      // Suppose we have a sequence: - a - a - b - c - .
      // The non loop part is the result for - c - . Call these X Y Z below
      if (!labelLength) {
        // Writing `Z`
        grad[blankClass] =
            logAdd(logMul(betaP1[0], alpha[0]), grad[blankClass]);
        beta[0] = logMul(betaP1[0], blank);

        // For use when data is split by label, output this timestep, last label
        // result for use by the next vertex. Include prob=0 to avoid special
        // cases when this is used later
        betaPrevLabel[2 * (t - 1)] = beta[0];
        betaPrevLabel[1 + 2 * (t - 1)] = log::probabilityZero;
        // Swap new <-> old in the alphaTemp buffer
        oldIdx = oldIdx ^ extendedLabel;
        continue;
      }
      // Each loop outputs the result for the symbol and a blank, and consumes 1
      // index from the label[] input.
      unsigned idx = 0;
      for (unsigned symbol = 0; symbol < labelLength - 1; symbol++) {

        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        auto sum = logAdd(betaP1[idx], betaP1[idx + 1]);
        grad[blankClass] = logAdd(logMul(sum, alpha[idx]), grad[blankClass]);
        beta[idx] = logMul(sum, blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        sum = logAdd(betaP1[idx + 1], betaP1[idx]);
        if (label[symbol] != label[symbol + 1]) {
          sum = logAdd(sum, betaP1[idx + 2]);
        }
        grad[label[symbol]] =
            logAdd(logMul(sum, alpha[idx]), grad[label[symbol]]);
        beta[idx] =
            logMul(sum, static_cast<OutType>(probability[label[symbol]]));
        idx++;
      }
      // Process `X` (always needed).  As X is a blank then it is preceded
      // by a symbol which cannot be skipped and we need only 2 probabilities
      auto sum = logAdd(betaP1[idx + 1], betaP1[idx]);
      beta[idx] = logMul(sum, blank);
      grad[blankClass] = logAdd(logMul(sum, alpha[idx]), grad[blankClass]);
      idx++;
      const auto lastSymbol = label[labelLength - 1];
      const auto prob = static_cast<OutType>(probability[lastSymbol]);

      if (doLastBlank) {
        // Write `Y` and `Z`. There is no previous label's beta to use.
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
        // We are not writing the last blank (`Z`) So just write `Y`, which
        // uses the probabilty of the previous blank, and conditionally that of
        // the previous symbol.
        auto sum = logAdd(betaP1[idx], betaPrevLabel[2 * (t - 1)]);
        if (lastSymbol != prevSymbol) {
          sum = logAdd(sum, betaPrevLabel[1 + 2 * (t - 1)]);
        }
        beta[idx] = logMul(sum, prob);
        grad[lastSymbol] = logAdd(logMul(sum, alpha[idx]), grad[lastSymbol]);
      }
      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex
      betaPrevLabel[2 * (t - 1)] = beta[0];
      betaPrevLabel[1 + 2 * (t - 1)] = beta[1];
      // Swap new <-> old in the alphaTemp buffer
      oldIdx = oldIdx ^ extendedLabel;
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
  // Label, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> label; // [maxLabel]
  Input<SymbolType> prevSymbol;
  Input<Vector<InType, ONE_PTR>> probabilities;   // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> betas;          // [maxT,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> alphaPrevTime;  // [2,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> alphaPrevLabel; // [maxT]
  InOut<Vector<OutType, ONE_PTR>> grads;          // [maxT,numClasses]
  // This vertex processes a labelSlice with size[label.size()] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned short> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned short> validTime;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // How much does this vertex need to process ?
    if (validLabel < labelOffset || validTime <= timeOffset) {
      return true;
    }
    const auto labelLength = clamp(validLabel, labelOffset, label.size());
    const auto timeSteps = clamp(validTime, timeOffset, maxT);
    const auto extendedLabel = label.size() * 2 + isLastLabel;
    bool doLastBlank = labelLength != label.size() || isLastLabel;
    unsigned oldIdx = 0;

    for (unsigned t = 0; t < timeSteps; t++) {
      // References to each row, previous row of the input and output
      auto probability = &probabilities[numClasses * t];
      auto beta = &betas[extendedLabel * t];
      auto grad = &grads[numClasses * t];
      auto alpha = &alphaPrevTime[oldIdx ^ extendedLabel];
      auto alphaM1 = &alphaPrevTime[oldIdx];
      // 1st symbol takes its parents from the alphaPrevLabel input.
      const auto blank = static_cast<OutType>(probability[blankClass]);
      // 1st blank.  Can't skip the symbol before it so combine 2 probabilities
      auto sum = logAdd(alphaM1[0], alphaPrevLabel[t]);
      alpha[0] = logMul(sum, blank);
      grad[blankClass] = logAdd(logMul(sum, beta[0]), grad[blankClass]);
      if (!labelLength) {
        // Although we need to process the blank above this (as it could be the
        // "last blank" there is no part of label to process so continue
        // (not break)
        // Swap new <-> old in the alphaPrevTime buffer
        oldIdx = oldIdx ^ extendedLabel;
        continue;
      }
      // First symbol. Can skip the blank before it if the previous symbol is
      // different
      sum = logAdd(alphaM1[0], alphaM1[1]);
      if (label[0] != prevSymbol) {
        sum = logAdd(sum, alphaPrevLabel[t]);
      }
      alpha[1] = logMul(sum, static_cast<OutType>(probability[label[0]]));
      grad[label[0]] = logAdd(logMul(sum, beta[1]), grad[label[0]]);

      // Each loop outputs the result for the symbol and a blank, and consumes 1
      // index from the labels[] input.
      unsigned idx = 2;
      for (unsigned symbol = 1; symbol < labelLength; symbol++) {
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        auto sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        grad[blankClass] = logAdd(logMul(sum, beta[idx]), grad[blankClass]);
        alpha[idx] = logMul(sum, blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        idx++;
        sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        if (label[symbol] != label[symbol - 1]) {
          sum = logAdd(sum, alphaM1[idx - 2]);
        }
        grad[label[symbol]] =
            logAdd(logMul(sum, beta[idx]), grad[label[symbol]]);
        alpha[idx] =
            logMul(sum, static_cast<OutType>(probability[label[symbol]]));
        idx++;
      }
      // The final blank entry, therefore preceded by a symbol which cannot be
      // skipped. So always combine only 2 probabilities. Last blank in a
      // sequence
      if (doLastBlank) {
        sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
        alpha[idx] = logMul(sum, blank);
        grad[blankClass] = logAdd(logMul(sum, beta[idx]), grad[blankClass]);
      }
      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex
      alphaPrevLabel[t] = alpha[2 * labelLength - 1];

      // Swap new <-> old in the alphaPrevTime buffer
      oldIdx = oldIdx ^ extendedLabel;
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
