// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplibs_test/CTCUtil.hpp>

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

} // namespace ctc
} // namespace poplibs_test
