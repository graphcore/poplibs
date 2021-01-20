// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_ctc_util_hpp
#define poplibs_test_ctc_util_hpp

#include <vector>

namespace poplibs_test {
namespace ctc {

// Converts input sequence to ctc loss extendedLabels version, e.g.
//   `a` -> `- a -`
//   `aabb` -> `- a - b -`
// Optionally remove duplicates then insert blanks
std::vector<unsigned> extendedLabels(const std::vector<unsigned> &input,
                                     unsigned blankSymbol,
                                     bool stripDuplicates = false);

void print(const std::vector<unsigned> &sequence);
void print(const std::vector<unsigned> &sequence, unsigned blank);

} // namespace ctc
} // namespace poplibs_test

#endif // poplibs_test_ctc_util_hpp
