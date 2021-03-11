// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <random>

#include <poplibs_test/CTCUtil.hpp>
#include <poputil/exceptions.hpp>

#include <boost/optional/optional.hpp>

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

// Return a Random Generator for the given the input range
static std::function<unsigned()>
getInputGen(unsigned min, unsigned max,
            const std::function<unsigned(unsigned, unsigned)> &randRange) {
  return [=]() { return randRange(min, max); };
}

// Create a random data input for all timesteps and symbols. Apply an increased
// probability to a sequence using a random label (Intended to be shorter than
// a label that would occupy all timesteps) and padding it to represent a
// probable sequence
template <typename FPType>
std::pair<boost::multi_array<FPType, 2>, std::vector<unsigned>>
provideInputWithPath(
    unsigned labelLength, unsigned timeSteps, unsigned maxT,
    unsigned numClasses, unsigned blankClass,
    const std::function<unsigned(unsigned, unsigned)> &randRange) {
  auto randClass =
      getInputGen(0U, static_cast<unsigned>(numClasses - 2), randRange);
  auto randInput = getInputGen(0U, 10U, randRange);

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
    auto insertIndex = randRange(0, indices.size() - 1);
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
provideInputWithPath(unsigned labelLength, unsigned timesteps, unsigned maxT,
                     unsigned numClasses, unsigned blankClass, unsigned seed) {
  std::mt19937 gen;
  gen.seed(seed);
  const auto randRange = [&](unsigned min, unsigned max) -> unsigned {
    if (max < min) {
      poputil::poplibs_error(
          "max must be greater than min when specifying random range");
    }
    std::uniform_int_distribution<> range(min, max);
    return range(gen);
  };

  return provideInputWithPath<FPType>(labelLength, timesteps, maxT, numClasses,
                                      blankClass, randRange);
}

template std::pair<boost::multi_array<double, 2>, std::vector<unsigned>>
provideInputWithPath(
    unsigned labelLength, unsigned timeSteps, unsigned maxT,
    unsigned numClasses, unsigned blankClass,
    const std::function<unsigned(unsigned, unsigned)> &randRange);

template std::pair<boost::multi_array<float, 2>, std::vector<unsigned>>
provideInputWithPath(
    unsigned labelLength, unsigned timeSteps, unsigned maxT,
    unsigned numClasses, unsigned blankClass,
    const std::function<unsigned(unsigned, unsigned)> &randRange);

template std::pair<boost::multi_array<double, 2>, std::vector<unsigned>>
provideInputWithPath(unsigned labelLength, unsigned timeSteps, unsigned maxT,
                     unsigned numClasses, unsigned blankClass, unsigned seed);

template std::pair<boost::multi_array<float, 2>, std::vector<unsigned>>
provideInputWithPath(unsigned labelLength, unsigned timeSteps, unsigned maxT,
                     unsigned numClasses, unsigned blankClass, unsigned seed);
} // namespace ctc
} // namespace poplibs_test
