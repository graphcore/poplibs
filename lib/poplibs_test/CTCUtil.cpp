// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplibs_test/CTCUtil.hpp>

#include <poputil/exceptions.hpp>

#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_test/MatrixTransforms.hpp>

#include <boost/optional/optional.hpp>

#include <iomanip>
#include <iostream>

namespace poplibs_test {
namespace ctc {

void print(const std::vector<unsigned> &sequence) {
  for (const auto symbol : sequence) {
    std::cout << symbol << " ";
  }
  std::cout << std::endl;
}

void print(const std::vector<unsigned> &sequence, unsigned blank) {
  for (const auto symbol : sequence) {
    if (symbol == blank) {
      std::cout << "- ";
    } else {
      std::cout << symbol << " ";
    }
  }
  std::cout << std::endl;
}

template <typename FPType>
void printInput(const boost::multi_array<FPType, 2> &in, unsigned blank) {
  for (unsigned i = 0; i < in.size(); i++) {
    std::cout << std::setw(11) << (std::string{"t"} + std::to_string(i));
  }

  for (unsigned i = 0; i < in[0].size(); i++) {
    if (i == blank) {
      std::cout << "\nIndex:-  ";
    } else {
      std::cout << "\nIndex:" << i << "  ";
    }
    for (unsigned j = 0; j < in.size(); j++) {
      std::cout << std::setw(10) << std::setprecision(4) << in[j][i] << ",";
    }
  }
  std::cout << std::endl;
}

template void printInput(const boost::multi_array<float, 2> &in,
                         unsigned blank);
template void printInput(const boost::multi_array<double, 2> &in,
                         unsigned blank);

template <typename FPType>
void printBeams(
    const std::vector<std::pair<std::vector<unsigned>, FPType>> &beams,
    unsigned blank) {
  auto i = 0;
  for (const auto &beam : beams) {
    std::cout << "Beam " << i << " ";
    std::cout << "(LogProb = " << std::setw(8) << std::setprecision(4)
              << beam.second << ")";
    std::cout << ": ";
    for (const auto symbol : beam.first) {
      if (symbol == blank) {
        std::cout << "- ";
      } else {
        std::cout << symbol << " ";
      }
    }
    i++;
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template void
printBeams(const std::vector<std::pair<std::vector<unsigned>, float>> &beams,
           unsigned blank);
template void
printBeams(const std::vector<std::pair<std::vector<unsigned>, double>> &beams,
           unsigned blank);

std::vector<unsigned> extendedLabels(const std::vector<unsigned> &input,
                                     unsigned blank, bool stripDuplicates) {
  std::vector<unsigned> output;
  output.push_back(blank);
  unsigned prev = blank;
  for (const auto c : input) {
    if (c == prev && stripDuplicates) {
      continue;
    }
    if (prev != blank && c != blank) {
      output.push_back(blank);
    }
    output.push_back(c);
    prev = c;
  }
  if (output.back() != blank) {
    output.push_back(blank);
  }

  return output;
}

void validateBounds(const std::string &ref,
                    const boost::optional<unsigned> &min,
                    const boost::optional<unsigned> &fixed, unsigned max) {
  if (min && fixed) {
    throw poputil::poplibs_error(std::string{"Cannot specify both `"} + ref +
                                 std::string{"` and `min-"} + ref +
                                 std::string{"`"});
  }
  if (min) {
    if (*min > max) {
      throw poputil::poplibs_error(
          std::string{"`min-"} + ref +
          std::string{"` cannot be greater than `max-"} + ref +
          std::string{"`"});
    }
  } else if (fixed) {
    if (*fixed > max) {
      throw poputil::poplibs_error(
          std::string{"`"} + ref +
          std::string{"` cannot be greater than `max-"} + ref +
          std::string{"`"});
    }
  } else {
    throw poputil::poplibs_error(std::string{"Neither `"} + ref +
                                 std::string{"`, nor `min-"} + ref +
                                 std::string{"` specified"});
  }
}

void validateTimeAndLabelBounds(
    const boost::optional<unsigned> &minRandomTime,
    const boost::optional<unsigned> &fixedTime, unsigned maxTime,
    const boost::optional<unsigned> &minRandomLabelLength,
    const boost::optional<unsigned> &fixedLabelLength,
    unsigned maxLabelLength) {
  validateBounds("time", minRandomTime, fixedTime, maxTime);
  validateBounds("label-length", minRandomLabelLength, fixedLabelLength,
                 maxLabelLength);

  auto maxTimestepsGenerated = fixedTime ? *fixedTime : maxTime;
  auto minLabelLengthGenerated =
      minRandomLabelLength ? *minRandomLabelLength : *fixedLabelLength;

  if (maxTimestepsGenerated < minLabelLengthGenerated) {
    throw poputil::poplibs_error(
        "Combination of time and label-length cannot create valid sequences. "
        "Either increase `max-time`/`time` or decrease "
        "`min-label-length`/`label-length`");
  }
}

// {T, LabelLength}
std::pair<unsigned, unsigned>
getRandomSize(const boost::optional<unsigned> &minT,
              const boost::optional<unsigned> &fixedT, unsigned maxT,
              const boost::optional<unsigned> &minLabelLength,
              const boost::optional<unsigned> &fixedLabelLength,
              unsigned maxLabelLength, bool disableAlwaysSatisfiableError,
              RandomUtil &rand) {
  auto checkSatisfiable = [&](unsigned t, unsigned labelLength) -> void {
    if (t < labelLength) {
      throw poputil::poplibs_error(
          std::string{"Length of t ("} + std::to_string(t) +
          std::string{") is too short to be able to represent a label "
                      "(of length "} +
          std::to_string(labelLength) + std::string{")"});
    }
    if (!disableAlwaysSatisfiableError) {
      if (t < labelLength * 2 - 1) {
        throw poputil::poplibs_error(
            std::string{"Length of t ("} + std::to_string(t) +
            std::string{") is too short to always be able to represent a label "
                        "(of length "} +
            std::to_string(labelLength) +
            std::string{"). This is an overly cautious error, which considers "
                        "the worst case of all duplicate classes (requiring t "
                        ">= labelLength * 2 - 1). This error can be disabled "
                        "with --disable-always-satisfiable-error"});
      }
    }
  };

  if (fixedT && fixedLabelLength) {
    auto t = *fixedT;
    auto labelLength = *fixedLabelLength;
    checkSatisfiable(t, labelLength);
    return {t, labelLength};
  } else if (fixedT || fixedLabelLength) {
    if (fixedT) {
      auto t = *fixedT;
      auto maxLabelLengthForT = t;
      auto upperBound = std::min(maxLabelLengthForT, maxLabelLength);
      auto labelLength = rand.range(*minLabelLength, upperBound);
      checkSatisfiable(t, labelLength);
      return {t, labelLength};
    } else {
      auto labelLength = *fixedLabelLength;
      auto minTForLabelLength = labelLength;
      auto lowerBound = std::max(minTForLabelLength, *minT);
      auto t = rand.range(lowerBound, maxT);
      checkSatisfiable(t, labelLength);
      return {t, labelLength};
    }
  } else { // Generate both randomly
    // Prune upper bound of label
    auto minTForMinLabelLength = *minLabelLength * 2 - 1;
    auto TLowerBound = std::max(minTForMinLabelLength, *minT);

    auto t = rand.range(TLowerBound, maxT);
    // Prune upper bound of label for given T
    auto maxLabelLengthForT = (t + 1) / 2;
    auto labelLengthUpperBound = std::min(maxLabelLengthForT, maxLabelLength);

    auto labelLength = rand.range(*minLabelLength, labelLengthUpperBound);

    checkSatisfiable(t, labelLength);
    return {t, labelLength};
  }
}

// Create a random data input for all timesteps and symbols. Apply an increased
// probability to a sequence using a random label (Intended to be shorter than
// a label that would occupy all timesteps) and padding it to represent a
// probable sequence
template <typename FPType>
std::pair<boost::multi_array<FPType, 2>, std::vector<unsigned>>
provideInputWithPath(unsigned labelLength, unsigned timeSteps, unsigned maxT,
                     unsigned numClasses, unsigned blankClass,
                     RandomUtil &rand) {
  auto randClass = rand.generator<unsigned>(0, numClasses - 2);
  auto randInput = rand.generator<double>(0.0, 10.0);

  std::vector<unsigned> label(labelLength);

  // Random label sequence of the right length
  for (size_t i = 0; i < labelLength; i++) {
    const unsigned random = randClass();
    label[i] = static_cast<unsigned>(random >= blankClass) + random;
  }

  // Input sequence of max length
  boost::multi_array<FPType, 2> input(boost::extents[maxT][numClasses]);
  for (size_t i = 0; i < numClasses; i++) {
    for (size_t j = 0; j < maxT; j++) {
      input[j][i] = randInput();
    }
  }
  std::vector<unsigned> expandedLabel;
  expandedLabel.reserve(timeSteps);
  // The input sequence, padding with blanks as needed for repeat
  // symbols. This is the shortest path that could represent the input sequence
  // Eg:
  // Input sequence a b c c d d d e
  // Pad with blanks: a b c - c d - d - d e
  for (unsigned i = 0; i < label.size(); i++) {
    if (i != 0 && label[i] == label[i - 1]) {
      expandedLabel.push_back(blankClass);
    }
    expandedLabel.push_back(label[i]);
  }
  if (expandedLabel.size() > timeSteps) {
    // The expanded input sequence is already bigger than timeSteps so we
    // can't increase the probability at all points matching the sequence
    std::cerr << "\n\nMinimum timesteps for this sequence (blank padded from "
              << label.size() << ") is " << expandedLabel.size()
              << " but the test has " << timeSteps << " timesteps."
              << " Expect -inf loss and likely comparison errors.\n";
    return std::make_pair(input, label);
  }
  // We are able to add this many random symbols to make a expanded sequence
  // with the same number of timeSteps as the input.  This is a random path
  // of length timeSteps that has correlation with the input
  auto padSymbols = timeSteps - expandedLabel.size();

  // Add symbols at random points to duplicate the symbol found at that
  // point. Eg
  // Pad with blanks gave us: a b c - c d - d - d e
  // Pad with 4 random symbols at the points marked ^
  // a a b c - - - c d - d - d d e
  //   ^       ^ ^             ^

  // Note and maintain the start index of each of the original symbols in the
  // expandedlabel.  (If we just insert randomly into the expandedLabel as it
  // grows, symbols which are already duplicated are more likely to be
  // duplicated again)
  std::vector<unsigned> indices(expandedLabel.size());
  std::iota(indices.begin(), indices.end(), 0);
  for (unsigned i = 0; i < padSymbols; i++) {
    // Pick an index (and so an original symbol to duplicate)
    auto insertIndex = rand.range<unsigned>(0, indices.size() - 1);
    auto insertPoint = indices[insertIndex];
    for (unsigned j = insertIndex; j < indices.size(); j++) {
      indices[j]++;
    }
    auto insertValue = expandedLabel[insertPoint];
    expandedLabel.insert(expandedLabel.begin() + insertPoint, insertValue);
  }

  // Now increase the probability of the points in the path generated to provide
  // a more realistic input with a reasonable loss
  for (unsigned i = 0; i < timeSteps; i++) {
    input[i][expandedLabel[i]] += 10.0;
  }
  return std::make_pair(input, label);
}

template <typename FPType>
std::pair<boost::multi_array<FPType, 2>, std::vector<unsigned>>
getRandomTestInput(unsigned timesteps, unsigned maxT, unsigned labelLength,
                   unsigned numClassesIncBlank, unsigned blankClass,
                   bool isLogits, RandomUtil &rand) {
  auto [input, label] = provideInputWithPath<FPType>(
      labelLength, timesteps, maxT, numClassesIncBlank, blankClass, rand);

  if (!isLogits) { // Convert to log probs
    input = poplibs_support::log::log(matrix::transpose(
        poplibs_support::log::softMax(matrix::transpose(input))));
  }

  return {input, label};
}

template std::pair<boost::multi_array<float, 2>, std::vector<unsigned>>
getRandomTestInput(unsigned timesteps, unsigned maxT, unsigned labelLength,
                   unsigned numClassesIncBlank, unsigned blankClass,
                   bool isLogits, RandomUtil &rand);
template std::pair<boost::multi_array<double, 2>, std::vector<unsigned>>
getRandomTestInput(unsigned timesteps, unsigned maxT, unsigned labelLength,
                   unsigned numClassesIncBlank, unsigned blankClass,
                   bool isLogits, RandomUtil &rand);

} // namespace ctc
} // namespace poplibs_test
