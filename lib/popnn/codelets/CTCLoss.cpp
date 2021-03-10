// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "LogOps.hpp"

#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>
#include <cmath>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/LogArithmetic.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;
using namespace poplibs_support;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace {
// Limit to the desired maximum
inline unsigned clamp(unsigned value, unsigned offset, unsigned limit) {
  assert(!(value < offset));
  return (value - offset > limit) ? limit : value - offset;
}

inline bool timestepOutOfRange(unsigned begin, unsigned end, unsigned validTime,
                               unsigned currentTimestep) {
  return validTime <= begin || currentTimestep < begin ||
         currentTimestep >= end || currentTimestep >= validTime;
}
} // namespace

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
  Input<Vector<SymbolType>> label;                    // [previous1 + maxLabel]
  Input<Vector<InType, ONE_PTR>> probabilities;       // [maxT,numClasses]
  InOut<Vector<OutType, ONE_PTR>> alphas;             // [maxT][extendedLabel]
  Input<Vector<OutType, ONE_PTR>> alphaPrevTime;      // [extendedLabel]
  Input<Vector<OutType, ONE_PTR>> alphaPrevLabel;     // [1]
  Output<Vector<OutType, ONE_PTR>> alphaPrevLabelOut; // [1]
  Input<Vector<OutType, ONE_PTR>> alphaPrevLabelTime; // [1]
  InOut<OutType> loss;
  // This vertex processes a labelSlice with size[label.size()-1] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned> validTime;
  InOut<unsigned> count;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // This vertex may have nothing to process.
    if (validLabel < labelOffset ||
        timestepOutOfRange(timeOffset, timeOffset + maxT, validTime, count)) {
      count++;
      return true;
    }
    auto t = *count - timeOffset;
    count++;
    const auto maxLabel = label.size() - 1;
    const auto labelLength = clamp(validLabel, labelOffset, maxLabel);
    const auto extendedLabel = maxLabel * 2 + isLastLabel;
    bool doLastBlank = labelLength != maxLabel || isLastLabel;
    bool doLastTimeStep = t + timeOffset == validTime - 1;
    // First time round reference the "previous alpha" which could be carried
    // from another vertex, or if starting up [0,-inf,-inf,...]
    auto alphaM1 =
        t == 0 ? &alphaPrevTime[0] : &alphas[extendedLabel * (t - 1)];

    // References to each row, previous row of the input and output
    auto probability = &probabilities[numClasses * t];
    auto alpha = &alphas[extendedLabel * t];
    // 1st symbol takes its parents from the alphaPrevLabel input.
    auto alphaPrevLabelValue =
        t == 0 ? alphaPrevLabelTime[0] : alphaPrevLabel[0];
    const auto blank = static_cast<OutType>(probability[blankClass]);
    if (!labelLength) {
      // A blank. Can't skip the symbol before it so combine 2 probabilities
      alpha[0] = logMul(logAdd(alphaM1[0], alphaPrevLabelValue), blank);
      // That was the "last blank", there is no part of the label to process

      // Loss is the sum of the last two alpha values computed (symbol and
      // blank). Each tile can contribute to these, all tiles results are
      // reduced to get the result. In this case only the result for the
      // last blank is on this tile
      if (doLastTimeStep) {
        *loss = alpha[0];
      }
      // so we are done
      return true;
    }
    // Each loop outputs the result for the symbol and a blank, and consumes 1
    // index from the label[] input.
    unsigned idx = 0;
    for (unsigned symbol = 1; symbol < labelLength + 1; symbol++) {
      // If idx references a symbol's output
      // The blank entry before the symbol (at idx-1) has a result:
      // pBlank * (alphaM1[idx-1] + alphaM1[idx-2])
      // The symbol result is
      // pSym * (alphaM1[idx] + alphaM1[idx-1] + sameSym ? 0: alphaM1[idx-2])
      // So we only compute commonSum = (alphaM1[idx-1] + alphaM1[idx-2]) once
      const auto commonSum = logAdd(alphaPrevLabelValue, alphaM1[idx]);
      // The blank entry, therefore preceded by a symbol which cannot be
      // skipped So always combine only 2 probabilities
      alpha[idx] = logMul(commonSum, blank);
      // Next the non-blank entry, therefore preceded by a blank which can
      // be skipped if the symbol before it is different to this one
      auto sum =
          (label[symbol] == label[symbol - 1]) ? alphaM1[idx] : commonSum;
      idx++;
      sum = logAdd(sum, alphaM1[idx]);
      alpha[idx] =
          logMul(sum, static_cast<OutType>(probability[label[symbol]]));
      alphaPrevLabelValue = alphaM1[idx]; // For the next loop pass
      idx++;
    }
    if (doLastBlank) {
      // The final blank entry, therefore preceded by a symbol which cannot
      // be skipped. So always combine only 2 probabilities. Last blank in a
      // sequence
      alpha[idx] = logMul(logAdd(alphaM1[idx - 1], alphaM1[idx]), blank);
    }
    if (doLastTimeStep) {
      // Loss is the sum of the last two alpha values computed (symbol and
      // blank). Each tile can contribute to these, all tiles results are
      // reduced to get the result.
      bool nextPartitionOnlyBlank = labelOffset + maxLabel == validLabel;
      if (doLastBlank) {
        // In this case both the result for the last symbol and last blank is on
        // this tile
        *loss = logAdd(alpha[idx - 1], alpha[idx]);
      } else if (nextPartitionOnlyBlank) {
        // In this case only the result for the last symbol is on this tile
        *loss = alpha[idx - 1];
      }
    }
    // For use when data is split by label, output this timestep, last label
    // result for use by the next vertex
    alphaPrevLabelOut[0] = alpha[2 * labelLength - 1];

    return true;
  }
};

template class CTCAlpha<float, float, unsigned, true>;
template class CTCAlpha<half, half, unsigned, true>;
template class CTCAlpha<half, float, unsigned, true>;

template class CTCAlpha<float, float, unsigned, false>;
template class CTCAlpha<half, half, unsigned, false>;
template class CTCAlpha<half, float, unsigned, false>;

template <typename InType, typename OutType, typename SymbolType,
          bool isFirstLabel>
class CTCBeta : public Vertex {

public:
  CTCBeta();
  // Label, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> label;                   // [maxLabel, next1]
  Input<Vector<InType, ONE_PTR>> probabilities;      // [maxT,numClasses]
  InOut<Vector<OutType, ONE_PTR>> betas;             // [maxT,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> betaPrevTime;      // [extendedLabel]
  Input<Vector<OutType, ONE_PTR>> betaPrevLabel;     // [2]
  Output<Vector<OutType, ONE_PTR>> betaPrevLabelOut; // [2]
  Input<Vector<OutType, ONE_PTR>> betaPrevLabelTime; // [2]
  // This vertex processes a labelSlice with size[label.size()-1] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned> validTime;
  InOut<unsigned> count;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // This vertex may have nothing to process.
    if (validLabel < labelOffset ||
        timestepOutOfRange(timeOffset, timeOffset + maxT, validTime, count)) {
      count--;
      return true;
    }
    const auto t = *count - timeOffset;
    count--;
    const auto maxLabel = label.size() - 1;
    const auto extendedLabel = maxLabel * 2 + isFirstLabel;
    const auto labelLength = clamp(validLabel, labelOffset, maxLabel);
    bool doLastBlank = labelLength != maxLabel || isFirstLabel;
    bool isLastLabel = (labelLength + labelOffset) == validLabel;
    bool doLastTimeStep = t + timeOffset == validTime - 1;

    auto betaPrev = t == maxT - 1 ? betaPrevLabelTime : betaPrevLabel;
    auto betaPrev0 = betaPrev[0];
    auto betaPrev1 = betaPrev[1];
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
        betaPrev0 = log::probabilityOne;
      }
    }
    // References to each row, previous row of the input and output
    auto probability = &probabilities[numClasses * t];
    auto beta = &betas[extendedLabel * t];
    // First time round reference the "previous beta" which could be carried
    // from another vertex, or if starting up [-inf,-inf,....-inf] with the
    // 0 inserted in the correct place above
    auto betaP1 = (t == maxT - 1 || doLastTimeStep)
                      ? &betaPrevTime[0]
                      : &betas[extendedLabel * (t + 1)];
    const auto blank = static_cast<OutType>(probability[blankClass]);

    // Suppose we have a sequence: - a - a - b - c - .
    // The non loop part is the result for - c - . Call these X Y Z below
    if (!labelLength) {
      // Just the `Z` on this tile, nothing else
      beta[0] = logMul(betaP1[0], blank); //`Z`
      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex. Include prob=0 to avoid special
      // cases when this is used later
      betaPrevLabelOut[0] = beta[0];
      betaPrevLabelOut[1] = log::probabilityZero;
      return true;
    }
    unsigned idx = 2 * (labelLength - 1) + 1 + doLastBlank;
    const auto lastSymbol = label[labelLength - 1];
    const auto prob = static_cast<OutType>(probability[lastSymbol]);
    if (doLastBlank) {
      // Write `Y` and `Z`. There is no previous label's beta to use.
      beta[idx] = logMul(betaP1[idx], blank); //`Z`
      idx--;
      beta[idx] = logMul(logAdd(betaP1[idx], betaP1[idx + 1]), prob); // `Y`
    } else {
      // We are not writing the last blank (`Z`) So just write `Y`, which
      // uses the probabilty of the previous blank, and conditionally that of
      // the previous symbol.
      auto sum = logAdd(betaP1[idx], betaPrev0);
      if (lastSymbol != label[maxLabel]) {
        sum = logAdd(sum, betaPrev1);
      }
      beta[idx] = logMul(sum, prob);
    }
    idx--;
    // If idx references a symbol's output
    // The blank entry before the symbol (at idx+1) has a result:
    // pBlank * (betaP1[idx+1] + betaP1[idx+2])
    // The symbol result is
    // pSym * (betaP1[idx] + betaP1[idx+1] + sameSym ? 0: betaP1[idx+2])
    // So we only compute commonSum = (betaP1[idx+1] + betaP1[idx+2]) once

    // Each loop outputs the result for the symbol and a blank, and consumes 1
    // index from the label[] input.
    // The first blank processed in the loop is `X`
    for (unsigned s = 0; s < labelLength - 1; s++) {
      unsigned symbol = labelLength - 2 - s;
      // The blank entry, therefore preceded by a symbol which cannot be
      // skipped So always combine only 2 probabilities
      auto commonSum = logAdd(betaP1[idx + 1], betaP1[idx]);
      beta[idx] = logMul(commonSum, blank);
      idx--;
      // The non-blank entry, therefore preceded by a blank which can
      // be skipped if the symbol before it is different to this one
      auto sum =
          label[symbol] == label[symbol + 1] ? betaP1[idx + 1] : commonSum;
      sum = logAdd(sum, betaP1[idx]);
      beta[idx] = logMul(sum, static_cast<OutType>(probability[label[symbol]]));
      idx--;
    }
    // Remaining blank entry
    beta[idx] = logMul(logAdd(betaP1[idx + 1], betaP1[idx]), blank);
    // For use when data is split by label, output this timestep, last label
    // result for use by the next vertex
    betaPrevLabelOut[0] = beta[0];
    betaPrevLabelOut[1] = beta[1];
    return true;
  }
};

template class CTCBeta<float, float, unsigned, true>;
template class CTCBeta<half, half, unsigned, true>;
template class CTCBeta<half, float, unsigned, true>;

template class CTCBeta<float, float, unsigned, false>;
template class CTCBeta<half, half, unsigned, false>;
template class CTCBeta<half, float, unsigned, false>;

template <typename InType, typename OutType, typename SymbolType,
          bool isFirstLabel>
class CTCGradGivenAlpha : public Vertex {

public:
  CTCGradGivenAlpha();
  // Label, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> label;                   // [maxLabel, next1]
  Input<Vector<InType, ONE_PTR>> probabilities;      // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> alphas;            // [maxT,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> betaPrevTime;      // [2,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> betaPrevPartition; // [2,extendedLabel]
  Input<Vector<OutType, ONE_PTR>> betaPrevLabel;     // [2]
  Output<Vector<OutType, ONE_PTR>> betaPrevLabelOut; // [2]
  Input<Vector<OutType, ONE_PTR>> betaPrevLabelTime; // [2]
  InOut<Vector<OutType, ONE_PTR>> grads;             // [maxT,numClasses]
  // This vertex processes a labelSlice with size[label.size()-1] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned> validTime;
  InOut<unsigned> count;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // This vertex may have nothing to process.
    if (validLabel < labelOffset ||
        timestepOutOfRange(timeOffset, timeOffset + maxT, validTime, count)) {
      count--;
      return true;
    }
    const auto t = *count - timeOffset;
    count--;
    const auto maxLabel = label.size() - 1;
    const auto extendedLabel = maxLabel * 2 + isFirstLabel;
    const auto labelLength = clamp(validLabel, labelOffset, maxLabel);
    bool doLastBlank = labelLength != maxLabel || isFirstLabel;
    bool isLastLabel = (labelLength + labelOffset) == validLabel;
    bool doLastTimeStep = t + timeOffset == validTime - 1;

    auto betaPrev = t == maxT - 1 ? betaPrevLabelTime : betaPrevLabel;
    auto betaPrev0 = betaPrev[0];
    auto betaPrev1 = betaPrev[1];
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
        betaPrev0 = log::probabilityOne;
      }
    }
    // Select an index to ping-pong between the 1st/2nd timslices in the
    // temporary alpha input
    unsigned oldIdx = (count & 1) ? extendedLabel : 0;

    // References to each row, next row of the input and output
    auto probability = &probabilities[numClasses * t];
    auto alpha = &alphas[extendedLabel * t];
    auto grad = &grads[numClasses * t];
    auto beta = &betaPrevTime[oldIdx ^ extendedLabel];
    auto betaP1 = (t == maxT - 1 && !doLastTimeStep)
                      ? &betaPrevPartition[oldIdx]
                      : &betaPrevTime[oldIdx];

    const auto blank = static_cast<OutType>(probability[blankClass]);
    // Suppose we have a sequence: - a - a - b - c - .
    // The non loop part is the result for - c - . Call these X Y Z below
    // We are writing gradient - the first write of gradient for blank and the
    // first symbol need not be added to a previous result.
    if (!labelLength) {
      // Writing `Z`
      grad[blankClass] = logMul(betaP1[0], alpha[0]);
      beta[0] = logMul(betaP1[0], blank);

      // For use when data is split by label, output this timestep, last label
      // result for use by the next vertex. Include prob=0 to avoid special
      // cases when this is used later
      betaPrevLabelOut[0] = beta[0];
      betaPrevLabelOut[1] = log::probabilityZero;
      return true;
    }
    unsigned idx = 2 * (labelLength - 1) + 1 + doLastBlank;
    const auto lastSymbol = label[labelLength - 1];
    const auto prob = static_cast<OutType>(probability[lastSymbol]);

    if (doLastBlank) {
      // Write `Y` and `Z`. There is no previous label's beta to use.
      // Writing `Z`
      grad[blankClass] = logMul(betaP1[idx], alpha[idx]);
      beta[idx] = logMul(betaP1[idx], blank);
      idx--;
      // Writing `Y`
      auto sum = logAdd(betaP1[idx], betaP1[idx + 1]);
      grad[lastSymbol] = logMul(sum, alpha[idx]);
      beta[idx] = logMul(sum, prob);
    } else {
      // We are not writing the last blank (`Z`) So just write `Y`, which
      // uses the probabilty of the previous blank, and conditionally that of
      // the previous symbol.
      auto sum = logAdd(betaP1[idx], betaPrev0);
      if (lastSymbol != label[maxLabel]) {
        sum = logAdd(sum, betaPrev1);
      }
      beta[idx] = logMul(sum, prob);
      grad[lastSymbol] = logMul(sum, alpha[idx]);
    }
    idx--;
    // If idx references a symbol's output
    // The blank entry before the symbol (at idx+1) has a result:
    // pBlank * (betaP1[idx+1] + betaP1[idx+2])
    // The symbol result is
    // pSym * (betaP1[idx] + betaP1[idx+1] + sameSym ? 0: betaP1[idx+2])
    // So we only compute commonSum = (betaP1[idx+1] + betaP1[idx+2]) once

    // Each loop outputs the result for the symbol and a blank, and consumes 1
    // index from the label[] input.
    // The first loop pass processes `X`
    // When writing gradient, assume that the value in question has already been
    // written and add to it
    for (unsigned s = 0; s < labelLength - 1; s++) {
      unsigned symbol = labelLength - 2 - s;
      // The blank entry, therefore preceded by a symbol which cannot be
      // skipped So always combine only 2 probabilities
      auto commonSum = logAdd(betaP1[idx + 1], betaP1[idx]);
      beta[idx] = logMul(commonSum, blank);
      grad[blankClass] =
          logAdd(logMul(commonSum, alpha[idx]), grad[blankClass]);
      idx--;
      // The non-blank entry, therefore preceded by a blank which can
      // be skipped if the symbol before it is different to this one
      auto sum =
          label[symbol] == label[symbol + 1] ? betaP1[idx + 1] : commonSum;
      sum = logAdd(sum, betaP1[idx]);
      grad[label[symbol]] =
          logAdd(logMul(sum, alpha[idx]), grad[label[symbol]]);
      beta[idx] = logMul(sum, static_cast<OutType>(probability[label[symbol]]));
      idx--;
    }
    // The last blank entry
    // When writing gradient, with label length 1 (common use case) the value in
    // question has not been written to so we need not use logAdd
    auto commonSum = logAdd(betaP1[idx], betaP1[idx + 1]);
    grad[blankClass] =
        (labelLength == 1 && !doLastBlank)
            ? logMul(commonSum, alpha[idx])
            : logAdd(logMul(commonSum, alpha[idx]), grad[blankClass]);
    beta[idx] = logMul(commonSum, blank);
    // For use when data is split by label, output this timestep, last label
    // result for use by the next vertex
    betaPrevLabelOut[0] = beta[0];
    betaPrevLabelOut[1] = beta[1];
    return true;
  }
};

template class CTCGradGivenAlpha<float, float, unsigned, true>;
template class CTCGradGivenAlpha<half, half, unsigned, true>;
template class CTCGradGivenAlpha<half, float, unsigned, true>;

template class CTCGradGivenAlpha<float, float, unsigned, false>;
template class CTCGradGivenAlpha<half, half, unsigned, false>;
template class CTCGradGivenAlpha<half, float, unsigned, false>;

template <typename InType, typename OutType, typename SymbolType,
          bool isLastLabel>
class CTCGradGivenBeta : public Vertex {

public:
  CTCGradGivenBeta();
  // Label, not padded with blanks, this is done implicitly in the code.
  // Eg this contains abcbb  and the code processes -a-b-c-b-b-
  Input<Vector<SymbolType>> label;                    // [previous1 + maxLabel]
  Input<Vector<InType, ONE_PTR>> probabilities;       // [maxT,numClasses]
  Input<Vector<OutType, ONE_PTR>> betas;              // [maxT,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> alphaPrevTime;      // [2,extendedLabel]
  InOut<Vector<OutType, ONE_PTR>> alphaPrevPartition; // [2,extendedLabel]
  Input<Vector<OutType, ONE_PTR>> alphaPrevLabel;     // [1]
  Output<Vector<OutType, ONE_PTR>> alphaPrevLabelOut; // [1]
  Input<Vector<OutType, ONE_PTR>> alphaPrevLabelTime; // [1]
  InOut<Vector<OutType, ONE_PTR>> grads;              // [maxT,numClasses]
  InOut<OutType> loss;
  // This vertex processes a labelSlice with size[label.size()-1] starting at
  // labelOffset within the whole input.  Only validLabel (of the whole input)
  // is to be processed). This may mean it has nothing to do
  Input<unsigned> validLabel;
  // This vertex processes a timeSlice with size[maxT] starting at timeOffset
  // within the whole input.  Only validTime (of the whole input) is to be
  // processed. This may mean it has nothing to do
  Input<unsigned> validTime;
  InOut<unsigned> count;
  const unsigned short labelOffset;
  const unsigned short timeOffset;
  const unsigned short maxT;
  const unsigned short numClasses;
  const unsigned short blankClass;

  IS_EXTERNAL_CODELET(false);

  bool compute() {
    // This vertex may have nothing to process.
    if (validLabel < labelOffset ||
        timestepOutOfRange(timeOffset, timeOffset + maxT, validTime, count)) {
      count++;
      return true;
    }
    const auto t = *count - timeOffset;
    count++;
    const auto maxLabel = label.size() - 1;
    const auto labelLength = clamp(validLabel, labelOffset, maxLabel);
    const auto extendedLabel = maxLabel * 2 + isLastLabel;
    bool doLastBlank = labelLength != maxLabel || isLastLabel;
    bool doLastTimeStep = t + timeOffset == validTime - 1;

    // Select an index to ping-pong between the 1st/2nd timslices in the
    // temporary alpha input
    unsigned oldIdx = ((count - 1) & 1) ? extendedLabel : 0;
    // References to each row, previous row of the input and output
    auto probability = &probabilities[numClasses * t];
    auto beta = &betas[extendedLabel * t];
    auto grad = &grads[numClasses * t];
    auto alpha = &alphaPrevTime[oldIdx ^ extendedLabel];
    auto alphaM1 =
        t == 0 ? &alphaPrevPartition[oldIdx] : &alphaPrevTime[oldIdx];

    // We are writing gradient - the first write of gradient for blank and the
    // first symbol need not be added to a previous result.
    // 1st symbol takes its parents from the alphaPrevLabel input.
    auto alphaPrevLabelValue =
        t == 0 ? alphaPrevLabelTime[0] : alphaPrevLabel[0];
    const auto blank = static_cast<OutType>(probability[blankClass]);
    if (!labelLength) {
      // A blank. Can't skip the symbol before it so combine 2 probabilities
      auto sum = logAdd(alphaM1[0], alphaPrevLabelValue);
      alpha[0] = logMul(sum, blank);
      // Loss is the sum of the last two alpha values computed (symbol and
      // blank). Each tile can contribute to these, all tiles results are
      // reduced to get the result. In this case only the result for the
      // last blank is on this tile
      if (doLastTimeStep) {
        *loss = alpha[0];
      }
      grad[blankClass] = logMul(sum, beta[0]);
      // That was the "last blank", there is no part of the label to process
      // so we are done
      return true;
    }
    // Each loop outputs the result for the symbol and a blank, and consumes 1
    // index from the labels[] input.
    // When writing any gradient result for the first time we need not use
    // a logAdd.  Structure the code so this happens cleanly in the common
    // labelLength=1 case.
    unsigned idx = 0;
    if (labelLength == 1) {
      const unsigned symbol = 1;
      auto commonSum = logAdd(alphaPrevLabelValue, alphaM1[idx]);
      // The blank entry, therefore preceded by a symbol which cannot be
      // skipped So always combine only 2 probabilities
      grad[blankClass] = logMul(commonSum, beta[idx]);
      alpha[idx] = logMul(commonSum, blank);
      // Next the non-blank entry, therefore preceded by a blank which can
      // be skipped if the symbol before it is different to this one
      auto sum =
          (label[symbol] == label[symbol - 1]) ? alphaM1[idx] : commonSum;
      idx++;
      sum = logAdd(sum, alphaM1[idx]);
      grad[label[symbol]] = logMul(sum, beta[idx]);
      alpha[idx] =
          logMul(sum, static_cast<OutType>(probability[label[symbol]]));
      idx++;
    } else {
      for (unsigned symbol = 1; symbol < labelLength + 1; symbol++) {
        // If idx references a symbol's output
        // The blank entry before the symbol (at idx-1) has a result:
        // pBlank * (alphaM1[idx-1] + alphaM1[idx-2])
        // The symbol result is
        // pSym * (alphaM1[idx] + alphaM1[idx-1] + sameSym ? 0: alphaM1[idx-2])
        // So we only compute commonSum = (alphaM1[idx-1] + alphaM1[idx-2]) once
        auto commonSum = logAdd(alphaPrevLabelValue, alphaM1[idx]);
        // The blank entry, therefore preceded by a symbol which cannot be
        // skipped So always combine only 2 probabilities
        grad[blankClass] =
            logAdd(logMul(commonSum, beta[idx]), grad[blankClass]);
        alpha[idx] = logMul(commonSum, blank);
        // Next the non-blank entry, therefore preceded by a blank which can
        // be skipped if the symbol before it is different to this one
        auto sum =
            (label[symbol] == label[symbol - 1]) ? alphaM1[idx] : commonSum;
        idx++;
        sum = logAdd(sum, alphaM1[idx]);
        grad[label[symbol]] =
            logAdd(logMul(sum, beta[idx]), grad[label[symbol]]);
        alpha[idx] =
            logMul(sum, static_cast<OutType>(probability[label[symbol]]));
        alphaPrevLabelValue = alphaM1[idx]; // For the next loop pass
        idx++;
      }
    }
    // The final blank entry, therefore preceded by a symbol which cannot be
    // skipped. So always combine only 2 probabilities. Last blank in a
    // sequence
    if (doLastBlank) {
      auto sum = logAdd(alphaM1[idx - 1], alphaM1[idx]);
      alpha[idx] = logMul(sum, blank);
      grad[blankClass] = logAdd(logMul(sum, beta[idx]), grad[blankClass]);
    }
    if (doLastTimeStep) {
      // Loss is the sum of the last two alpha values computed (symbol and
      // blank). Each tile can contribute to these, all tiles results are
      // reduced to get the result.
      bool nextPartitionOnlyBlank = labelOffset + maxLabel == validLabel;
      if (doLastBlank) {
        // In this case both the result for the last symbol and last blank is on
        // this tile
        *loss = logAdd(alpha[idx - 1], alpha[idx]);
      } else if (nextPartitionOnlyBlank) {
        // In this case only the result for the last symbol is on this tile
        *loss = alpha[idx - 1];
      }
    }
    // For use when data is split by label, output this timestep, last label
    // result for use by the next vertex
    alphaPrevLabelOut[0] = alpha[2 * labelLength - 1];

    return true;
  }
};

template class CTCGradGivenBeta<float, float, unsigned, true>;
template class CTCGradGivenBeta<half, half, unsigned, true>;
template class CTCGradGivenBeta<half, float, unsigned, true>;

template class CTCGradGivenBeta<float, float, unsigned, false>;
template class CTCGradGivenBeta<half, half, unsigned, false>;
template class CTCGradGivenBeta<half, float, unsigned, false>;

} // namespace popnn
